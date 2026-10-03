"""Synthetic CPU guard fixtures, never empirical model outputs."""
from __future__ import annotations
import copy
import csv
import json
import tempfile
from unittest import mock
import hashlib
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts import analyze_acl_paired_prompt_scores as a


def score(logits):
    maximum=max(logits)
    weights=[math.exp(x-maximum) for x in logits]
    probs=[x/sum(weights) for x in weights]
    ties=[c for c,x in zip(a.OPTIONS,logits) if x==maximum]
    return {'raw_option_logits':dict(zip(a.OPTIONS,logits)),
            'conditional_option_probabilities':dict(zip(a.OPTIONS,probs)),
            'top_option_id':ties[0],'tied_top_option_ids':ties,
            'logit_margin':sorted(logits,reverse=True)[0]-sorted(logits,reverse=True)[1],
            'probability_margin':sorted(probs,reverse=True)[0]-sorted(probs,reverse=True)[1]}


def fixture(qcount=2):
    jobs=[];rows=[]
    for q in range(qcount):
        for condition in a.MENU_CONDITIONS:
            prompt='Choose a menu. '+a.ORIGINAL_SUFFIX
            job=dict(job_id=f'q{q}-{condition}',qid=f'q{q}',group_id=f'g{q}',split='test',
                     condition=condition,menu_id='fixed_1',prompt=prompt,
                     prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                     options=[{'id':c,'text':f'answer {c}'} for c in a.OPTIONS])
            index=len(jobs);jobs.append(job)
            for variant in a.PROMPT_CONDITIONS:
                text=a.derive_prompt(job,variant)
                row={k:job[k] for k in ('job_id','qid','group_id','split','condition','menu_id')}
                row.update(schema_version='acl-paired-prompt-scores-v1',score_index=len(rows),
                           score_id=f"{job['job_id']}:{variant}",job_index=index,prompt_condition=variant,
                           original_prompt_sha256=job['prompt_sha256'],
                           prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           **score([1.,2.,3.,4.] if variant=='original' else [2.,3.,4.,5.]))
                rows.append(row)
    return jobs,rows


def numeric_fixture(root):
    jobs,rows=fixture(qcount=4)
    option_ids=dict(zip(a.OPTIONS,range(32,36)))
    for i,row in enumerate(rows):
        tokens=[20]*(2+i//2)
        row.update(input_token_ids=tokens,scored_input_token_ids=tokens+[9,9],
                   option_token_ids=option_ids,rendered_prompt_sha256='a'*64,
                   scored_context_sha256='b'*64,batch_index=i//32)
    metadata={'global_padded_width':max(len(r['scored_input_token_ids']) for r in rows),
              'option_token_ids':option_ids}
    selection={**a.expected_diagnostic_selection(jobs,rows),'selection_uses_gold':False}
    (root/'diagnostic_selection.json').write_text(json.dumps(selection))
    metadata['diagnostic_selection_sha256']=a.sha256(root/'diagnostic_selection.json')
    indices=selection['score_indices'];n=len(indices)
    contexts=[{k:rows[i][k] for k in a.CONTEXT_FIELDS} for i in indices]
    batched=[[rows[i]['raw_option_logits'][c] for c in a.OPTIONS] for i in indices]
    single=[[v+.0001 for v in row] for row in batched]
    bf=[[v+.1*(j==0) for j,v in enumerate(row)] for row in batched]
    identity={'score_indices':indices,'score_ids':[rows[i]['score_id'] for i in indices],
              'contexts':contexts,'batch_size':32,'global_padded_width':metadata['global_padded_width']}
    (root/'diagnostics_bf16.json').write_text(json.dumps({**identity,'batched_logits':bf,
        'single_logits':bf,'diagnostics':a.matrix_diagnostics(bf,bf),'gate':False}))
    (root/'diagnostics_fp32.json').write_text(json.dumps({**identity,'batched_logits':batched,
        'single_logits':single,'replay_logits':batched,'permutation_indices':list(reversed(range(n))),
        'permuted_logits':list(reversed(batched)),
        'bf16_vs_fp32_batched':a.matrix_diagnostics(bf,batched),'bf16_vs_fp32_single':a.matrix_diagnostics(bf,single)}))
    promotion={'schema_version':'acl-paired-dtype-promotion-v1','all_checks_passed':True,
        'sampled_values_preserved_exactly':True,'all_floating_tensors_fp32':True,
        'original':{'weight':{'dtype':'torch.bfloat16','shape':[1],'sample':[1.]},
                    'rope':{'dtype':'torch.float32','shape':[1],'sample':[.1]}},
        'promoted_dtypes':{'weight':'torch.float32','rope':'torch.float32'}}
    (root/'dtype_promotion.json').write_text(json.dumps(promotion))
    gate_extra={'numeric_gate_passed':True,'absolute_tolerance':1e-3,'relative_tolerance':1e-5,'exact_top_equality_required':False}
    gates={'validation_method':a.VALIDATION_METHOD,'all_gates_passed':True,'exact_replay_passed':True,
        'fp32_atol':1e-3,'fp32_rtol':1e-5,
        'batch_vs_unpadded_single':{**a.matrix_diagnostics(batched,single),**gate_extra},
        'permutation':{**a.matrix_diagnostics(batched,batched),**gate_extra},
        'diagnostic_bf16_sha256':a.sha256(root/'diagnostics_bf16.json'),
        'diagnostic_fp32_sha256':a.sha256(root/'diagnostics_fp32.json'),
        'dtype_promotion_sha256':a.sha256(root/'dtype_promotion.json')}
    (root/'diagnostics_validation.json').write_text(json.dumps(gates))
    receipt={'numeric_evidence_sha256':{name:a.sha256(root/name) for name in a.NUMERIC_FILES}}
    return jobs,rows,metadata,receipt


def rehash_numeric(root,receipt):
    gates=json.loads((root/'diagnostics_validation.json').read_text())
    for key,name in [('diagnostic_fp32_sha256','diagnostics_fp32.json'),('dtype_promotion_sha256','dtype_promotion.json')]:
        gates[key]=a.sha256(root/name)
    (root/'diagnostics_validation.json').write_text(json.dumps(gates))
    receipt['numeric_evidence_sha256']={name:a.sha256(root/name) for name in a.NUMERIC_FILES}


class PairedAnalysisTests(unittest.TestCase):
    def test_common_logit_offset_has_zero_conditional_effect(self):
        effect=a.pair_effect(score([1.,2.,3.,4.]),score([9.,10.,11.,12.]))
        self.assertAlmostEqual(effect['total_variation'],0.)
        self.assertAlmostEqual(effect['centered_logit_shift_rms'],0.)
        self.assertAlmostEqual(effect['max_abs_pairwise_log_odds_shift'],0.)
        self.assertFalse(effect['top_option_changed'])

    def test_pairwise_log_odds_shift_and_tv_are_exactly_defined(self):
        before=score([0.,0.,0.,0.]);after=score([math.log(3),0.,0.,0.])
        effect=a.pair_effect(before,after)
        self.assertAlmostEqual(effect['total_variation'],.25)
        self.assertAlmostEqual(effect['probability_delta_A'],.25)
        self.assertAlmostEqual(effect['pairwise_log_odds_shift_A_B'],math.log(3))
        self.assertAlmostEqual(effect['max_abs_pairwise_log_odds_shift'],math.log(3))
        self.assertTrue(effect['top_option_set_changed'])
        self.assertFalse(effect['top_option_changed'])
        self.assertFalse(effect['top_option_sets_disjoint'])

    def test_coverage_and_prompt_validation_reject_corruption(self):
        jobs,rows=fixture()
        pairs=a.validate_rows(rows,jobs)
        self.assertEqual(len(pairs),len(jobs))
        for bad in (rows[:-1],rows[:-1]+[copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError):a.validate_rows(bad,jobs)
        for changes in ({'prompt_sha256':'wrong'},{'original_prompt_sha256':'wrong'},
                        {'top_option_id':'A'},{'score_index':2},
                        {'conditional_option_probabilities':dict.fromkeys(a.OPTIONS,.25)}):
            bad=copy.deepcopy(rows);bad[0].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):a.validate_rows(bad,jobs)

    def test_prompt_edit_preserves_original_and_rejects_ambiguous_suffix(self):
        jobs,_=fixture();job=jobs[0]
        self.assertEqual(a.derive_prompt(job,'original'),job['prompt'])
        self.assertEqual(a.derive_prompt(job,'forced'),'Choose a menu. '+a.FORCED_SUFFIX)
        with self.assertRaises(ValueError):
            a.derive_prompt({**job,'prompt':job['prompt']+' trailing'},'forced')
        with self.assertRaises(ValueError):a.derive_prompt(job,'unknown')

    def test_pooled_bootstrap_keeps_two_menus_in_question_cluster(self):
        jobs,rows=fixture();pairs=a.validate_rows(rows,jobs)
        # Opposite within-question changes average to exactly .5 for each question.
        for index,pair in enumerate(pairs):pair['total_variation']=float(index%2)
        result=a.summarize_pairs(pairs,samples=200,seed=1)
        self.assertEqual(result['n_questions'],2)
        self.assertEqual(result['n_menus'],4)
        self.assertEqual(result['mean_total_variation'],.5)
        self.assertEqual(result['question_bootstrap_95_intervals']['mean_total_variation'],[.5,.5])
        self.assertEqual(result,a.summarize_pairs(pairs,samples=200,seed=1))

    def test_numeric_artifacts_validate_and_reject_hash_replay_and_tolerance_corruption(self):
        for corruption in ('hash','replay','tolerance','promotion'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);jobs,rows,metadata,receipt=numeric_fixture(root)
                a.validate_contexts(rows,metadata)
                self.assertTrue(a.validate_numeric_evidence(root,metadata,receipt,jobs,rows)['all_numeric_gates_independently_passed'])
                fp_path=root/'diagnostics_fp32.json';fp=json.loads(fp_path.read_text())
                if corruption=='hash':
                    receipt['numeric_evidence_sha256']['diagnostics_fp32.json']='wrong'
                elif corruption=='replay':
                    fp['replay_logits'][0][0]+=.1;fp_path.write_text(json.dumps(fp));rehash_numeric(root,receipt)
                elif corruption=='tolerance':
                    fp['single_logits'][0][0]+=.1;fp_path.write_text(json.dumps(fp));rehash_numeric(root,receipt)
                else:
                    path=root/'dtype_promotion.json';data=json.loads(path.read_text())
                    data['promoted_dtypes']['weight']='torch.bfloat16';path.write_text(json.dumps(data));rehash_numeric(root,receipt)
                with self.assertRaises(ValueError):a.validate_numeric_evidence(root,metadata,receipt,jobs,rows)

    def test_diagnostics_must_match_production_scores_and_contexts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobs,rows,metadata,receipt=numeric_fixture(root)
            rows[0]['raw_option_logits']['A']+=.1
            with self.assertRaisesRegex(ValueError,'numeric tolerance'):
                a.validate_numeric_evidence(root,metadata,receipt,jobs,rows)
            rows[0]['input_token_ids']=[999]
            with self.assertRaisesRegex(ValueError,'context mismatch'):
                a.validate_numeric_evidence(root,metadata,receipt,jobs,rows)

    def test_model_cache_binding_rejects_different_weight_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model,revision=a.MODELS['qwen3b']
            path=root/'cache_prepare_receipt.json'
            path.write_text(json.dumps({'model_receipts':{'qwen3b':{'model':model,'revision':revision,
                'model_files_sha256':{'weights':'known'},'original_trace_sha256':'trace'}}}))
            with mock.patch.object(a,'CACHE_PREPARE_SHA256',a.sha256(path)):
                a.validate_model_cache_binding(root/'qwen3b','qwen3b',{'model_files_sha256':{'weights':'known'}})
                with self.assertRaisesRegex(ValueError,'differ from frozen'):
                    a.validate_model_cache_binding(root/'qwen3b','qwen3b',{'model_files_sha256':{'weights':'altered'}})
            with self.assertRaisesRegex(ValueError,'preparation receipt hash'):
                a.validate_model_cache_binding(root/'qwen3b','qwen3b',{})

    def test_source_identity_rejects_modified_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'scripts').mkdir()
            for name in a.SOURCE_FILES:(root/name).write_text('synthetic source fixture')
            metadata={'source_files_sha256':{name:a.sha256(root/name) for name in a.SOURCE_FILES}}
            a.validate_source_identity(metadata,root)
            (root/a.SOURCE_FILES[0]).write_text('changed')
            with self.assertRaisesRegex(ValueError,'source hash mismatch'):a.validate_source_identity(metadata,root)

    def test_pair_csv_preserves_exact_text_both_distributions_and_quoting(self):
        jobs,rows=fixture();pairs=a.validate_rows(rows,jobs)
        jobs[0]['options'][-1]['text']='Name, "quoted"\nnext line'
        record=a.csv_pair_row('qwen3b',pairs[0],jobs[0])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'output.csv'
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=a.CSV_FIELDS);writer.writeheader();writer.writerow(record)
            with path.open(newline='') as stream:read=list(csv.DictReader(stream))[0]
            self.assertEqual(read['original_top_option_text'],jobs[0]['options'][-1]['text'])
            self.assertEqual(read['forced_top_option_text'],read['original_top_option_text'])
            self.assertAlmostEqual(float(read['original_probability_D']),float(read['forced_probability_D']))
            self.assertEqual(set(read),set(a.CSV_FIELDS))

if __name__=='__main__':unittest.main()
