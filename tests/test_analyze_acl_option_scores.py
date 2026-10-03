"""Synthetic unit fixtures; these are not model scores or empirical findings."""
from __future__ import annotations
import copy
import math
import json
import tempfile
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_acl_option_scores as a


def fixtures():
    jobs = []
    gold = {}
    rows = []
    for i, (answer, selected) in enumerate([('A', 'A'), ('B', 'A'), ('C', 'C'), ('D', 'D')]):
        for condition in a.CONDITIONS:
            ident = f'q{i}-{condition}'
            job = dict(job_id=ident, qid=f'q{i}', group_id=f'g{i}', split='test',
                       condition=condition, menu_id='fixed_1', prompt_sha256='abc')
            jobs.append(job)
            gold[ident] = answer
            logits = {x: float(x == selected) for x in a.OPTIONS}
            denom = sum(math.exp(x) for x in logits.values())
            row = dict(job, schema_version='acl-option-scores-v1', job_index=len(rows),
                       raw_option_logits=logits,
                       conditional_option_probabilities={x: math.exp(v)/denom for x,v in logits.items()},
                       top_option_id=selected, tied_top_option_ids=[selected])
            rows.append(row)
    return jobs, gold, rows


class ScoreAnalysisTests(unittest.TestCase):
    def test_validate_scores_and_known_accuracy(self):
        jobs, gold, rows = fixtures()
        validated = a.validate_rows(rows, jobs, gold)
        report = a.summarize_rows(validated[:4])
        self.assertEqual(report['n'], 4)
        self.assertEqual(report['n_correct'], 2)
        self.assertEqual(report['accuracy'], .5)
        self.assertEqual(report['gold_option_counts'], {'A': 2, 'B': 2, 'C': 0, 'D': 0})
        self.assertEqual(report['selected_option_counts'], {'A': 4, 'B': 0, 'C': 0, 'D': 0})

    def test_duplicate_or_missing_rows_rejected(self):
        jobs, gold, rows = fixtures()
        for bad in [rows[:-1], rows[:-1]+[copy.deepcopy(rows[0])]]:
            with self.assertRaises(ValueError):
                a.validate_rows(bad, jobs, gold)

    def test_bad_identity_top_probability_and_nonfinite_rejected(self):
        jobs, gold, rows = fixtures()
        changes = [{'qid': 'wrong'}, {'top_option_id': 'D'},
                   {'conditional_option_probabilities': dict(A=.25,B=.25,C=.25,D=.25)},
                   {'raw_option_logits': dict(A=math.nan,B=0,C=0,D=0)},
                   {'conditional_option_probabilities': dict(A=1.1,B=-.1,C=0.,D=0.)}]
        for change in changes:
            bad = copy.deepcopy(rows)
            bad[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                a.validate_rows(bad, jobs, gold)

    def test_exact_ties_use_declared_deterministic_rule_and_report_sensitivity(self):
        jobs, gold, rows = fixtures()
        rows[0].update(raw_option_logits=dict.fromkeys(a.OPTIONS, 0.),
                       conditional_option_probabilities=dict.fromkeys(a.OPTIONS, .25),
                       top_option_id='A', tied_top_option_ids=list(a.OPTIONS))
        report = a.summarize_rows(a.validate_rows(rows, jobs, gold)[:1])
        self.assertEqual(report['n_exact_top_ties'], 1)
        self.assertEqual(report['accuracy'], 1.)
        self.assertEqual(report['uniform_exact_tie_expected_accuracy'], .25)
        self.assertEqual(report['exact_tie_accuracy_bounds'], [0., 1.])

    def test_near_ties_do_not_silently_change_selected_answer(self):
        jobs, gold, rows = fixtures()
        logits = dict(A=1., B=1.+5e-7, C=0., D=0.)
        den = sum(math.exp(v) for v in logits.values())
        rows[0].update(raw_option_logits=logits,
                       conditional_option_probabilities={k:math.exp(v)/den for k,v in logits.items()},
                       top_option_id='B', tied_top_option_ids=['B'])
        report = a.summarize_rows(a.validate_rows(rows,jobs,gold)[:1])
        self.assertEqual(report['n_correct'], 0)
        self.assertEqual(report['n_near_top_ties'], 1)
        self.assertEqual(report['n_exact_top_ties'], 0)

    def test_wilson_and_exact_binomial_known_cases(self):
        self.assertEqual(a.wilson_interval(0,0), None)
        lo, hi = a.wilson_interval(0,100)
        self.assertAlmostEqual(lo,0.)
        self.assertAlmostEqual(hi,.0369934982,places=8)
        self.assertAlmostEqual(a.binomial_upper_tail(4,4,.25),.25**4)
        self.assertEqual(a.binomial_upper_tail(0,100,.25),1.)

    def test_paired_bootstrap_preserves_question_pairing_and_is_reproducible(self):
        jobs, gold, rows = fixtures()
        valid = a.validate_rows(rows,jobs,gold)
        first = a.paired_menu_report(valid, samples=100, seed=1)
        self.assertEqual(first, a.paired_menu_report(valid,samples=100,seed=1))
        self.assertEqual(first['accuracy_delta_independent_minus_same_category'],0.)
        self.assertEqual(first['paired_question_bootstrap_95_interval'],[0.,0.])
        with self.assertRaises(ValueError):
            a.paired_menu_report(valid[:-1],samples=100,seed=1)

    def test_receipt_hash_and_model_provenance_rejected_on_mismatch(self):
        jobs, gold, rows = fixtures()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            score_file = root/'scores.jsonl'
            score_file.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            model, revision = a.MODELS['qwen3b']
            metadata = dict(model_tag='qwen3b',model=model,revision=revision,
                            public_file_sha256=a.PUBLIC_SHA256,n_jobs=len(jobs),
                            assistant_prefix=a.ASSISTANT_PREFIX,protocol=a.PROTOCOL,
                            model_files_sha256={'weights':'fixture'},chat_template_sha256='fixture',
                            source_sha256='fixture')
            meta_file=root/'metadata.json'
            meta_file.write_text(json.dumps(metadata))
            receipt=dict(status='complete',completed_rows=len(jobs),expected_rows=len(jobs),
                         scores_sha256=a.sha256(score_file),metadata_sha256=a.sha256(meta_file))
            receipt_file=root/'receipt.json'
            receipt_file.write_text(json.dumps(receipt))
            self.assertEqual(len(a.load_model_scores(root,'qwen3b',jobs,gold)[0]),len(jobs))
            receipt['scores_sha256']='wrong'
            receipt_file.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'scores_sha256 mismatch'):
                a.load_model_scores(root,'qwen3b',jobs,gold)
            receipt['scores_sha256']=a.sha256(score_file)
            metadata['revision']='wrong'
            meta_file.write_text(json.dumps(metadata))
            receipt['metadata_sha256']=a.sha256(meta_file)
            receipt_file.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'metadata revision mismatch'):
                a.load_model_scores(root,'qwen3b',jobs,gold)

    def test_frozen_inputs_reject_modified_hash_before_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'frozen public or gold file hash mismatch'):
                a.load_frozen_inputs(path,path)

    def test_holm_adjustment_is_monotonic_and_capped(self):
        self.assertEqual(a.holm_adjust([.01,.03,.04,.8]),[.04,.09,.09,.8])

if __name__ == '__main__':
    unittest.main()
