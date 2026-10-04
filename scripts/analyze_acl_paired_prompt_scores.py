#!/usr/bin/env python3
"""Audit paired instruction effects on prefix-conditional A--D option scores.

This public-input-only analysis neither generates answers nor estimates
abstention probabilities, full-response frequencies, or an IIA parameter.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))
from scripts.analyze_acl_option_scores import MODELS, PUBLIC_SHA256, sha256, load_json, write_json, _unique

OPTIONS=('A','B','C','D')
MENU_CONDITIONS=('independent_pool','same_category_pool')
PROMPT_CONDITIONS=('original','forced')
OPTION_PAIRS=tuple(combinations(OPTIONS,2))
PROTOCOL='paired_menu_prompt_fp32_option_softmax_v1'
ASSISTANT_PREFIX='{"answer":"'
ORIGINAL_SUFFIX='If you decline to answer, use null for answer and confidence, and "abstain" for status. Do not add markdown or explanation.'
FORCED_SUFFIX='You must choose one of A, B, C, or D. Do not add markdown or explanation.'


def derive_prompt(job, variant):
    """Reconstruct the exact declared intervention independently of the scorer."""
    prompt=job['prompt']
    if not prompt.endswith(ORIGINAL_SUFFIX) or prompt.count(ORIGINAL_SUFFIX)!=1:
        raise ValueError('original prompt does not have one exact terminal abstention suffix')
    if variant=='original':return prompt
    if variant=='forced':return prompt[:-len(ORIGINAL_SUFFIX)]+FORCED_SUFFIX
    raise ValueError('unknown prompt condition')


def validate_score(row):
    logits=row.get('raw_option_logits',{});probs=row.get('conditional_option_probabilities',{})
    if set(logits)!=set(OPTIONS) or set(probs)!=set(OPTIONS):
        raise ValueError('four A--D logits and probabilities required')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (*logits.values(),*probs.values())):
        raise ValueError('nonfinite/non-numeric option scores')
    if any(v<0 or v>1 for v in probs.values()) or abs(math.fsum(probs.values())-1)>2e-6:
        raise ValueError('invalid probability range or normalization')
    top=max(logits.values());weights={c:math.exp(logits[c]-top) for c in OPTIONS}
    total=math.fsum(weights.values())
    if any(abs(probs[c]-weights[c]/total)>2e-6 for c in OPTIONS):
        raise ValueError('conditional probabilities differ from A--D softmax')
    ties=[c for c in OPTIONS if logits[c]==top]
    if row.get('tied_top_option_ids')!=ties or row.get('top_option_id')!=ties[0]:
        raise ValueError('top option/tie evidence mismatch')
    ordered_logits=sorted(logits.values(),reverse=True);ordered_probs=sorted(probs.values(),reverse=True)
    for name,wanted in (('logit_margin',ordered_logits[0]-ordered_logits[1]),
                        ('probability_margin',ordered_probs[0]-ordered_probs[1])):
        actual=row.get(name)
        if type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(actual,wanted,rel_tol=1e-6,abs_tol=2e-6):
            raise ValueError(f'{name} mismatch')


def pair_effect(original, forced):
    """Calculate offset-invariant effects, with forced minus original signs."""
    p=np.asarray([original['conditional_option_probabilities'][c] for c in OPTIONS],dtype=float)
    q=np.asarray([forced['conditional_option_probabilities'][c] for c in OPTIONS],dtype=float)
    delta=np.asarray([forced['raw_option_logits'][c]-original['raw_option_logits'][c] for c in OPTIONS],dtype=float)
    centered=delta-delta.mean()
    effects={'total_variation':float(.5*np.abs(q-p).sum()),
             'max_abs_probability_shift':float(np.max(np.abs(q-p))),
             'centered_logit_shift_rms':float(np.sqrt(np.mean(centered**2))),
             'max_abs_centered_logit_shift':float(np.max(np.abs(centered))),
             'max_abs_pairwise_log_odds_shift':float(delta.max()-delta.min()),
             'top_option_changed':original['top_option_id']!=forced['top_option_id'],
             'top_option_set_changed':original['tied_top_option_ids']!=forced['tied_top_option_ids'],
             'top_option_sets_disjoint':not set(original['tied_top_option_ids'])&set(forced['tied_top_option_ids']),
             'top_changed_both_margins_gt_0_01':original['top_option_id']!=forced['top_option_id'] and min(original['logit_margin'],forced['logit_margin'])>.01,
             'original_exact_top_tie':len(original['tied_top_option_ids'])>1,
             'forced_exact_top_tie':len(forced['tied_top_option_ids'])>1,
             'original_top_logit_margin':original['logit_margin'],
             'forced_top_logit_margin':forced['logit_margin']}
    for i,c in enumerate(OPTIONS):
        effects[f'probability_delta_{c}']=float(q[i]-p[i])
        effects[f'centered_logit_delta_{c}']=float(centered[i])
    for a,b in OPTION_PAIRS:
        effects[f'pairwise_log_odds_shift_{a}_{b}']=float(delta[OPTIONS.index(a)]-delta[OPTIONS.index(b)])
    return effects


def validate_rows(rows,jobs):
    """Require complete ordered original/forced pairs with exact prompt hashes."""
    if len({job['job_id'] for job in jobs})!=len(jobs) or len(rows)!=2*len(jobs):
        raise ValueError('duplicate jobs or incomplete paired score coverage')
    seen=set();pairs=[]
    for index,row in enumerate(rows):
        job_index=index//2;job=jobs[job_index];variant=PROMPT_CONDITIONS[index%2]
        expected={'schema_version':'acl-paired-prompt-scores-v1','score_index':index,
                  'score_id':f"{job['job_id']}:{variant}",'job_index':job_index,
                  'prompt_condition':variant,'original_prompt_sha256':job['prompt_sha256'],
                  'prompt_sha256':hashlib.sha256(derive_prompt(job,variant).encode()).hexdigest(),
                  **{key:job[key] for key in ('job_id','qid','group_id','split','condition','menu_id')}}
        for key,value in expected.items():
            if row.get(key)!=value:raise ValueError(f'paired row {index}: {key} mismatch')
        if row['score_id'] in seen:raise ValueError('duplicate score ID')
        seen.add(row['score_id']);validate_score(row)
        if variant=='forced':
            original=rows[index-1]
            pairs.append({**{key:job[key] for key in ('job_id','qid','group_id','split','condition','menu_id')},
                          'original':original,'forced':row,**pair_effect(original,row)})
    return pairs


def load_public(path):
    if sha256(path)!=PUBLIC_SHA256:raise ValueError('frozen public input hash mismatch')
    package=load_json(path);jobs=package.get('jobs',[])
    if package.get('schema_version')!='jane-choice-controls-v1' or package.get('evidence_scope')!='scientific' or len(jobs)!=10000:
        raise ValueError('frozen public schema/scope/count mismatch')
    questions=defaultdict(list)
    for job in jobs:
        if hashlib.sha256(job['prompt'].encode()).hexdigest()!=job['prompt_sha256']:
            raise ValueError('frozen per-prompt hash mismatch')
        derive_prompt(job,'forced')
        if [x['id'] for x in job['options']]!=list(OPTIONS):raise ValueError('option IDs/order mismatch')
        questions[job['qid']].append(job)
    if len(questions)!=5000 or len({j['job_id'] for j in jobs})!=10000:raise ValueError('frozen identity counts mismatch')
    if Counter(v[0]['split'] for v in questions.values())!=Counter(calibration=1000,selection=1000,test=3000):
        raise ValueError('frozen question split mismatch')
    for pair in questions.values():
        if len(pair)!=2 or {j['condition'] for j in pair}!=set(MENU_CONDITIONS):raise ValueError('paired menu coverage mismatch')
        if len({j['split'] for j in pair})!=1 or len({j['group_id'] for j in pair})!=1:raise ValueError('question group/split mismatch')
    if len({pair[0]['group_id'] for pair in questions.values()})!=5000:raise ValueError('expected one question per group')
    return jobs


SUMMARY_METRICS={'mean_total_variation':'total_variation',
    'mean_max_abs_probability_shift':'max_abs_probability_shift',
    'mean_centered_logit_shift_rms':'centered_logit_shift_rms',
    'mean_max_abs_pairwise_log_odds_shift':'max_abs_pairwise_log_odds_shift',
    'top_option_change_fraction':'top_option_changed',
    'top_option_set_change_fraction':'top_option_set_changed',
    'disjoint_top_option_sets_fraction':'top_option_sets_disjoint',
    'top_changed_both_margins_gt_0_01_fraction':'top_changed_both_margins_gt_0_01',
    'original_exact_top_tie_fraction':'original_exact_top_tie',
    'forced_exact_top_tie_fraction':'forced_exact_top_tie',
    **{f'mean_probability_delta_{c}':f'probability_delta_{c}' for c in OPTIONS},
    **{f'mean_centered_logit_delta_{c}':f'centered_logit_delta_{c}' for c in OPTIONS},
    **{f'mean_pairwise_log_odds_shift_{a}_{b}':f'pairwise_log_odds_shift_{a}_{b}' for a,b in OPTION_PAIRS}}


def summarize_pairs(pairs,samples=2000,seed=1):
    """Bootstrap question means, retaining each question's paired menus."""
    questions=defaultdict(list)
    for pair in pairs:questions[pair['qid']].append(pair)
    if not questions:raise ValueError('empty paired summary')
    if len({len(v) for v in questions.values()})!=1:raise ValueError('unequal menus per question in summary')
    for group in questions.values():
        if len({r['condition'] for r in group})!=len(group):raise ValueError('duplicate question/menu in summary')
    metric_names=list(SUMMARY_METRICS)
    values=np.asarray([[np.mean([float(r[key]) for r in questions[qid]]) for key in SUMMARY_METRICS.values()]
                       for qid in sorted(questions)],dtype=float)
    means=values.mean(axis=0);rng=np.random.default_rng(seed);boot=np.empty((samples,len(metric_names)))
    for start in range(0,samples,32):
        count=min(32,samples-start)
        indices=rng.integers(0,len(values),size=(count,len(values)))
        boot[start:start+count]=values[indices].mean(axis=1)
    bounds=np.quantile(boot,[.025,.975],axis=0)
    tv=np.asarray([r['total_variation'] for r in pairs])
    return {'n_questions':len(questions),'n_menus':len(pairs),
            **{name:float(value) for name,value in zip(metric_names,means)},
            'question_bootstrap_95_intervals':{name:[float(bounds[0,i]),float(bounds[1,i])] for i,name in enumerate(metric_names)},
            'total_variation_quantiles':{str(q):float(np.quantile(tv,q)) for q in (0,.25,.5,.75,.9,.95,.99,1)},
            'bootstrap_seed':seed,'bootstrap_samples':samples,'bootstrap_unit':'question; average its menus before resampling',
            'interval_scope':'Descriptive marginal intervals conditional on frozen menus and scoring protocol, without multiplicity correction; all/test summaries overlap.'}


VALIDATION_METHOD='fp32_fixed_shape_single_replay_permutation_v1'
CACHE_VALIDATION_METHOD='fp32_shared_prefix_cache_equivalence_v1'
NUMERIC_FILES=('diagnostics_bf16.json','diagnostics_fp32.json','diagnostics_validation.json','dtype_promotion.json')
CACHE_PREPARE_SHA256='95cc6dcd2e99e74597c95c1bf4580457dd843a01053267cf3888b21461a7a1e0'
SOURCE_FILES=('scripts/acl_paired_prompt_scoring.py','scripts/acl_option_scoring.py',
              'scripts/jane_gpu_backend.py','scripts/jane_qwen_backend.py','scripts/jane_output_constraints.py')
CONTEXT_FIELDS=('rendered_prompt_sha256','scored_context_sha256','input_token_ids',
                'scored_input_token_ids','option_token_ids')


def execution_settings(metadata):
    cached=metadata.get('shared_prefix_cache',False)
    if type(cached) is not bool:raise ValueError('shared-prefix cache mode must be explicit boolean')
    batch=128 if cached else 32
    method=CACHE_VALIDATION_METHOD if cached else VALIDATION_METHOD
    if metadata.get('batch_size')!=batch or metadata.get('validation_method')!=method:
        raise ValueError('batch size/validation method disagrees with declared cache mode')
    if cached and (metadata.get('reference_batch_size')!=32 or metadata.get('use_cache') is not True
                   or metadata.get('cache_mode')!='paired_exact_token_lcp_dynamic_cache'):
        raise ValueError('cached execution metadata mismatch')
    return {'cached':cached,'batch_size':batch,'reference_batch_size':32,'validation_method':method,
            'numeric_files':NUMERIC_FILES+(('diagnostics_cached.json',) if cached else ())}


def cache_geometry(rows):
    prefix_lengths=[];suffix_lengths=[]
    for i in range(0,len(rows),2):
        left=rows[i]['scored_input_token_ids'];right=rows[i+1]['scored_input_token_ids']
        prefix=0
        while prefix<min(len(left),len(right)) and left[prefix]==right[prefix]:prefix+=1
        if prefix<=0 or prefix>=min(len(left),len(right)):
            raise ValueError('cache requires nonempty exact common prefix and two nonempty suffixes')
        prefix_lengths.append(prefix);suffix_lengths.extend((len(left)-prefix,len(right)-prefix))
    return {'prefix_width':max(prefix_lengths),'suffix_width':max(suffix_lengths),
            'prefix_tokens':sum(prefix_lengths),'suffix_tokens':sum(suffix_lengths),
            'n_pairs':len(prefix_lengths)}


def expected_cache_plan(rows):
    geometry=cache_geometry(rows)
    p,s=geometry['prefix_width'],geometry['suffix_width']
    full=max(len(row['scored_input_token_ids']) for row in rows)
    return {'schema_version':'acl-shared-prefix-cache-plan-v1','n_pairs':geometry['n_pairs'],
            'prefix_width':p,'suffix_width':s,'full_width':full,
            'total_full_input_tokens':sum(len(row['scored_input_token_ids']) for row in rows),
            'total_shared_prefix_tokens':geometry['prefix_tokens'],'total_suffix_tokens':geometry['suffix_tokens'],
            'padded_cached_token_positions_per_pair':p+2*s,
            'padded_uncached_token_positions_per_pair':2*full,
            'padded_token_position_reduction_fraction':1-(p+2*s)/(2*full)}


def _matrix(values,n):
    array=np.asarray(values,dtype=float)
    if array.shape!=(n,4) or not np.all(np.isfinite(array)):
        raise ValueError('diagnostics require finite n-by-four matrices')
    return array


def matrix_diagnostics(left,right):
    left=_matrix(left,len(left));right=_matrix(right,len(left))
    def probs(values):
        weights=np.exp(values-values.max(axis=1,keepdims=True))
        return weights/weights.sum(axis=1,keepdims=True)
    lt=left==left.max(axis=1,keepdims=True);rt=right==right.max(axis=1,keepdims=True)
    return {'rows':len(left),'maximum_absolute_logit_difference':float(np.abs(left-right).max()),
        'maximum_centered_logit_difference':float(np.abs((left-left.mean(axis=1,keepdims=True))-(right-right.mean(axis=1,keepdims=True))).max()),
        'maximum_option_probability_difference':float(np.abs(probs(left)-probs(right)).max()),
        'changed_top_option_rows':np.flatnonzero(left.argmax(axis=1)!=right.argmax(axis=1)).tolist(),
        'changed_top_option_set_rows':np.flatnonzero(np.any(lt!=rt,axis=1)).tolist()}


def verify_diagnostics(recorded,actual):
    for key,wanted in actual.items():
        value=recorded.get(key)
        if isinstance(wanted,float):
            if type(value) not in (float,int) or not math.isfinite(value) or not math.isclose(value,wanted,rel_tol=1e-8,abs_tol=1e-10):
                raise ValueError(f'recorded numeric diagnostic mismatch: {key}')
        elif value!=wanted:raise ValueError(f'recorded numeric diagnostic mismatch: {key}')


def numeric_gate(left,right):
    left=_matrix(left,len(left));right=_matrix(right,len(left))
    if np.any(np.abs(left-right)>1e-3+1e-5*np.abs(right)):
        raise ValueError('independent FP32 numeric tolerance failed')
    return matrix_diagnostics(left,right)


def _score_from_vector(values):
    values=_matrix([values],1)[0]
    weights=np.exp(values-values.max());probs=weights/weights.sum()
    ties=[c for c,v in zip(OPTIONS,values) if v==values.max()]
    return {'raw_option_logits':dict(zip(OPTIONS,values.tolist())),
        'conditional_option_probabilities':dict(zip(OPTIONS,probs.tolist())),
        'top_option_id':ties[0],'tied_top_option_ids':ties,
        'logit_margin':float(np.sort(values)[-1]-np.sort(values)[-2]),
        'probability_margin':float(np.sort(probs)[-1]-np.sort(probs)[-2])}


def validate_contexts(rows,metadata):
    execution=execution_settings(metadata)
    option_ids=metadata.get('option_token_ids',{})
    if set(option_ids)!=set(OPTIONS) or len(set(option_ids.values()))!=4 or any(type(v) is not int or v<0 for v in option_ids.values()):
        raise ValueError('invalid metadata option token IDs')
    for row in rows:
        original=row.get('input_token_ids',[]);scored=row.get('scored_input_token_ids',[])
        if not original or len(scored)<=len(original) or scored[:len(original)]!=original:
            raise ValueError('invalid prompt/prefix token boundary evidence')
        if any(type(v) is not int or v<0 for v in original+scored):raise ValueError('invalid token ID')
        if row.get('option_token_ids')!=option_ids:raise ValueError('row option token IDs mismatch')
        for field in ('rendered_prompt_sha256','scored_context_sha256'):
            value=row.get(field)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid context hash evidence')
        if row.get('batch_index')!=row['score_index']//execution['batch_size']:raise ValueError('production batch assignment mismatch')
    width=max(len(row['scored_input_token_ids']) for row in rows)
    if metadata.get('global_padded_width')!=width or width>2048:raise ValueError('global padded width mismatch')
    if execution['cached']:
        geometry=cache_geometry(rows)
        if metadata.get('cache_prefix_width')!=geometry['prefix_width'] or metadata.get('cache_suffix_width')!=geometry['suffix_width']:
            raise ValueError('cache global prefix/suffix width mismatch')


def expected_diagnostic_selection(jobs,rows):
    selected=set();strata={}
    for condition in sorted(MENU_CONDITIONS):
        indices=[i for i,job in enumerate(jobs) if job['condition']==condition]
        if len(indices)<4:raise ValueError('too few menus for diagnostic selection')
        indices.sort(key=lambda i:(max(len(rows[2*i+v]['scored_input_token_ids']) for v in (0,1)),i))
        picks=indices[:2]+indices[-2:];selected.update(picks)
        strata[condition]={'shortest_menu_indices':indices[:2],'longest_menu_indices':indices[-2:]}
    menus=sorted(selected)
    return {'menu_indices':menus,'score_indices':[2*i+v for i in menus for v in (0,1)],'strata':strata}


def validate_numeric_evidence(directory,metadata,receipt,jobs,rows):
    execution=execution_settings(metadata)
    hashes=receipt.get('numeric_evidence_sha256',{})
    if set(hashes)!=set(execution['numeric_files']):raise ValueError('complete numeric artifact hash map required')
    for name,digest in hashes.items():
        if sha256(directory/name)!=digest:raise ValueError(f'numeric artifact hash mismatch: {name}')
    expected=expected_diagnostic_selection(jobs,rows)
    selection=load_json(directory/'diagnostic_selection.json')
    if sha256(directory/'diagnostic_selection.json')!=metadata.get('diagnostic_selection_sha256'):
        raise ValueError('diagnostic selection hash mismatch')
    if any(selection.get(key)!=value for key,value in expected.items()) or selection.get('selection_uses_gold') is not False:
        raise ValueError('diagnostic selection differs from declared public length rule')
    bf=load_json(directory/'diagnostics_bf16.json');fp=load_json(directory/'diagnostics_fp32.json')
    gates=load_json(directory/'diagnostics_validation.json');promotion=load_json(directory/'dtype_promotion.json')
    indices=expected['score_indices'];n=len(indices)
    expected_ids=[rows[i]['score_id'] for i in indices]
    contexts=[{k:rows[i][k] for k in CONTEXT_FIELDS} for i in indices]
    for item in (bf,fp):
        if item.get('score_indices')!=indices or item.get('score_ids')!=expected_ids or item.get('contexts')!=contexts:
            raise ValueError('numeric subset identity/context mismatch')
        if item.get('batch_size')!=execution['reference_batch_size'] or item.get('global_padded_width')!=metadata['global_padded_width']:
            raise ValueError('numeric diagnostic shape mismatch')
    if bf.get('gate') is not False:raise ValueError('BF16 sensitivity must not be an FP32 pass/fail gate')
    for key,wanted in {'validation_method':execution['validation_method'],'all_gates_passed':True,
                      'exact_replay_passed':True,'fp32_atol':1e-3,'fp32_rtol':1e-5}.items():
        if gates.get(key)!=wanted:raise ValueError(f'numerical gate setting mismatch: {key}')
    for key,name in (('diagnostic_bf16_sha256','diagnostics_bf16.json'),('diagnostic_fp32_sha256','diagnostics_fp32.json'),
                     ('dtype_promotion_sha256','dtype_promotion.json')):
        if gates.get(key)!=hashes[name]:raise ValueError('numeric gate artifact binding mismatch')
    batched=_matrix(fp['batched_logits'],n);single=_matrix(fp['single_logits'],n)
    replay=_matrix(fp['replay_logits'],n);permuted=_matrix(fp['permuted_logits'],n)
    permutation=fp['permutation_indices']
    if permutation!=list(reversed(range(n))):raise ValueError('diagnostic permutation mismatch')
    aligned=permuted[np.argsort(permutation)]
    if not np.array_equal(batched,replay):raise ValueError('exact FP32 diagnostic replay mismatch')
    comparisons={'batch_vs_unpadded_single':numeric_gate(batched,single),'permutation':numeric_gate(batched,aligned)}
    for key,actual in comparisons.items():
        declared=gates.get(key,{})
        for field,wanted in {'numeric_gate_passed':True,'absolute_tolerance':1e-3,'relative_tolerance':1e-5,'exact_top_equality_required':False}.items():
            if declared.get(field)!=wanted:raise ValueError('declared FP32 gate report mismatch')
        verify_diagnostics(declared,actual)
    bf_batched=_matrix(bf['batched_logits'],n);bf_single=_matrix(bf['single_logits'],n)
    bf_comparison=matrix_diagnostics(bf_batched,bf_single);verify_diagnostics(bf.get('diagnostics',{}),bf_comparison)
    precision_batched=matrix_diagnostics(bf_batched,batched);precision_single=matrix_diagnostics(bf_single,single)
    verify_diagnostics(fp.get('bf16_vs_fp32_batched',{}),precision_batched)
    verify_diagnostics(fp.get('bf16_vs_fp32_single',{}),precision_single)
    production=np.asarray([[rows[i]['raw_option_logits'][c] for c in OPTIONS] for i in indices])
    cached_report=None
    primary_diagnostic=batched
    if execution['cached']:
        cached_report=validate_cached_evidence(directory,metadata,gates,hashes,indices,expected_ids,contexts,batched,single,rows)
        primary_diagnostic=_matrix(load_json(directory/'diagnostics_cached.json')['cached_logits'],n)
    production_agreement=numeric_gate(production,primary_diagnostic)
    if (promotion.get('schema_version')!='acl-paired-dtype-promotion-v1' or promotion.get('all_checks_passed') is not True
            or promotion.get('sampled_values_preserved_exactly') is not True or promotion.get('all_floating_tensors_fp32') is not True):
        raise ValueError('dtype promotion did not pass')
    original=promotion.get('original',{});promoted=promotion.get('promoted_dtypes',{})
    if not original or set(original)!=set(promoted):raise ValueError('dtype promotion tensor identity mismatch')
    floating={'torch.bfloat16','torch.float16','torch.float32','torch.float64'}
    original_dtypes=Counter()
    for name,state in original.items():
        dtype=state.get('dtype');original_dtypes[dtype]+=1
        if not isinstance(state.get('shape'),list) or not isinstance(state.get('sample'),list):raise ValueError('missing tensor shape/sample evidence')
        expected_dtype='torch.float32' if dtype in floating else dtype
        if promoted[name]!=expected_dtype:raise ValueError('tensor promotion dtype mismatch')
    if not original_dtypes['torch.bfloat16'] or not original_dtypes['torch.float32']:
        raise ValueError('expected original BF16 weights and FP32 buffers')
    effect_rows=[]
    for offset,menu_index in enumerate(expected['menu_indices']):
        index=offset*2;primary=pair_effect(rows[2*menu_index],rows[2*menu_index+1])
        fp_effect=pair_effect(_score_from_vector(batched[index]),_score_from_vector(batched[index+1]))
        bf_effect=pair_effect(_score_from_vector(bf_batched[index]),_score_from_vector(bf_batched[index+1]))
        single_effect=pair_effect(_score_from_vector(single[index]),_score_from_vector(single[index+1]))
        p0=_score_from_vector(production[index]);p1=_score_from_vector(production[index+1])
        b0=_score_from_vector(bf_batched[index]);b1=_score_from_vector(bf_batched[index+1])
        record={k:jobs[menu_index][k] for k in ('job_id','qid','condition','split')}
        effect_rows.append({**record,'primary_fp32_total_variation':primary['total_variation'],
            'diagnostic_fp32_total_variation':fp_effect['total_variation'],
            'diagnostic_cached_fp32_total_variation':pair_effect(_score_from_vector(primary_diagnostic[index]),_score_from_vector(primary_diagnostic[index+1]))['total_variation'] if execution['cached'] else None,
            'diagnostic_fp32_single_total_variation':single_effect['total_variation'],
            'bf16_total_variation':bf_effect['total_variation'],
            'abs_bf16_minus_primary_fp32_effect_tv':abs(bf_effect['total_variation']-primary['total_variation']),
            'abs_fp32_single_minus_batch_effect_tv':abs(single_effect['total_variation']-fp_effect['total_variation']),
            'primary_fp32_top_changed':primary['top_option_changed'],'bf16_top_changed':bf_effect['top_option_changed'],
            'original_bf16_vs_primary_fp32_tv':pair_effect(p0,b0)['total_variation'],
            'forced_bf16_vs_primary_fp32_tv':pair_effect(p1,b1)['total_variation'],
            'original_bf16_vs_primary_fp32_top_changed':p0['top_option_id']!=b0['top_option_id'],
            'forced_bf16_vs_primary_fp32_top_changed':p1['top_option_id']!=b1['top_option_id']})
    return {'all_numeric_gates_independently_passed':True,'validation_method':execution['validation_method'],
        'shared_prefix_cache':execution['cached'],'cache_validation':cached_report,
        'fp32_comparisons':comparisons,'production_vs_diagnostic_fp32':production_agreement,
        'bf16_batch_vs_single':bf_comparison,'bf16_vs_fp32_batched':precision_batched,
        'bf16_vs_fp32_single':precision_single,'precision_subset_effects':effect_rows,
        'dtype_promotion':{'n_tensors':len(original),'original_dtypes':dict(original_dtypes),'sampled_values_preserved_assertion':True},
        'numeric_artifact_sha256':hashes,'selection':selection,
        'scope':'Eight menus selected as two shortest and two longest per menu condition; diagnostic effects are descriptive and are not a representative precision study. Tensor preservation is checked through stored dtypes and sampled-value assertions, not a full tensor-value hash.'}


def validate_cached_evidence(directory,metadata,gates,hashes,indices,score_ids,contexts,uncached,uncached_single,rows):
    path=directory/'diagnostics_cached.json'
    if gates.get('diagnostic_cached_sha256')!=hashes.get(path.name):
        raise ValueError('cached diagnostic hash binding mismatch')
    cached=load_json(path);n=len(indices);geometry=cache_geometry(rows)
    expected={'score_indices':indices,'score_ids':score_ids,'contexts':contexts,'batch_size':128,
              'prefix_width':geometry['prefix_width'],'suffix_width':geometry['suffix_width']}
    for key,value in expected.items():
        if cached.get(key)!=value:raise ValueError(f'cached diagnostic identity/shape mismatch: {key}')
    reference=_matrix(cached.get('uncached_logits'),n)
    if not np.array_equal(reference,uncached):raise ValueError('cached gate reference differs from retained uncached FP32 vectors')
    single_reference=_matrix(cached.get('single_logits'),n)
    if not np.array_equal(single_reference,uncached_single):
        raise ValueError('cached single reference differs from retained unpadded FP32 vectors')
    values=_matrix(cached.get('cached_logits'),n);replay=_matrix(cached.get('cached_replay_logits'),n)
    if not np.array_equal(values,replay) or gates.get('exact_cached_replay_passed') is not True:
        raise ValueError('exact cached FP32 diagnostic replay mismatch')
    expected_permutation=[i for pair in reversed(range(n//2)) for i in (2*pair,2*pair+1)]
    permutation=cached.get('permutation_indices')
    if permutation!=expected_permutation:raise ValueError('cached diagnostic pair permutation mismatch')
    permuted=_matrix(cached.get('permuted_cached_logits'),n)
    comparisons={'cached_vs_uncached':numeric_gate(values,uncached),
                 'cached_vs_unpadded_single':numeric_gate(values,uncached_single),
                 'cached_permutation':numeric_gate(values,permuted[np.argsort(permutation)])}
    for key,actual in comparisons.items():
        declared=gates.get(key,{})
        for field,wanted in {'numeric_gate_passed':True,'absolute_tolerance':1e-3,'relative_tolerance':1e-5,'exact_top_equality_required':False}.items():
            if declared.get(field)!=wanted:raise ValueError('declared cached FP32 gate report mismatch')
        verify_diagnostics(declared,actual)
    plan_path=directory/'cache_plan.json'
    if sha256(plan_path)!=metadata.get('cache_plan_sha256'):raise ValueError('cache plan hash mismatch')
    plan=load_json(plan_path)
    # Recompute shape and token accounting from every actual scored token pair.
    for key,wanted in expected_cache_plan(rows).items():
        actual=plan.get(key)
        if isinstance(wanted,float):
            if type(actual) not in (float,int) or not math.isfinite(actual) or not math.isclose(actual,wanted,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError(f'cache plan mismatch: {key}')
        elif actual!=wanted:raise ValueError(f'cache plan mismatch: {key}')
    return {'all_cached_gates_independently_passed':True,'comparisons':comparisons,
            'cache_geometry_from_all_rows':geometry,'cache_plan':plan,
            'cache_plan_sha256':sha256(plan_path),
            'scope':'Exact token common-prefix reuse changes execution only. Cached/uncached numerical equivalence is checked on the retained 16-score subset at the original FP32 tolerances, not proven for every context.'}


def validate_source_identity(metadata,source_root):
    hashes=metadata.get('source_files_sha256',{})
    if set(hashes)!=set(SOURCE_FILES):raise ValueError('complete scorer source identity required')
    for relative,digest in hashes.items():
        if sha256(source_root/relative)!=digest:raise ValueError(f'scorer source hash mismatch: {relative}')


def validate_prompt_manifest(directory,metadata,jobs):
    path=directory/'prompts.json'
    if sha256(path)!=metadata.get('prompt_manifest_sha256'):raise ValueError('prompt manifest hash mismatch')
    manifest=load_json(path)
    for key,wanted in {'schema_version':'acl-paired-prompt-manifest-v1','protocol':PROTOCOL,
                      'replaced_terminal_clause':ORIGINAL_SUFFIX,'replacement_terminal_clause':FORCED_SUFFIX,
                      'assistant_prefix':ASSISTANT_PREFIX}.items():
        if manifest.get(key)!=wanted:raise ValueError(f'prompt manifest {key} mismatch')
    templates=manifest.get('templates',{});hashes=manifest.get('template_sha256',{})
    if set(templates)!=set(PROMPT_CONDITIONS) or set(hashes)!=set(PROMPT_CONDITIONS):raise ValueError('prompt template conditions mismatch')
    for variant,template in templates.items():
        if template.count('{menu}')!=1 or hashlib.sha256(template.encode()).hexdigest()!=hashes[variant]:raise ValueError('prompt template hash/slot mismatch')
        for job in jobs:
            menu='\n'.join(f"{option['id']}. {option['text']}" for option in job['options'])
            if template.replace('{menu}',menu)!=derive_prompt(job,variant):raise ValueError('template does not reproduce frozen/derived prompt')
    return manifest


def validate_model_cache_binding(directory,tag,metadata):
    path=directory.parent/'cache_prepare_receipt.json'
    if sha256(path)!=CACHE_PREPARE_SHA256:raise ValueError('original cache preparation receipt hash mismatch')
    cached=load_json(path).get('model_receipts',{}).get(tag,{})
    model,revision=MODELS[tag]
    if cached.get('model')!=model or cached.get('revision')!=revision:
        raise ValueError('original cache model identity mismatch')
    if not cached.get('model_files_sha256') or metadata.get('model_files_sha256')!=cached['model_files_sha256']:
        raise ValueError('model weight/tokenizer files differ from frozen original cache')
    return {'cache_prepare_receipt_sha256':CACHE_PREPARE_SHA256,
            'original_trace_sha256':cached.get('original_trace_sha256'),
            'model_files_sha256':cached['model_files_sha256']}


def load_model(directory,tag,jobs,source_root):
    receipt=load_json(directory/'receipt.json');metadata=load_json(directory/'metadata.json')
    if receipt.get('status')!='complete' or receipt.get('completed_rows')!=2*len(jobs) or receipt.get('expected_rows')!=2*len(jobs):
        raise ValueError('complete paired scoring receipt required')
    for name,key in (('scores.jsonl','scores_sha256'),('metadata.json','metadata_sha256')):
        if sha256(directory/name)!=receipt.get(key):raise ValueError(f'{name} receipt hash mismatch')
    execution=execution_settings(metadata)
    model,revision=MODELS[tag]
    required={'schema_version':'acl-paired-prompt-scoring-metadata-v1','model_tag':tag,'model':model,'revision':revision,
        'protocol':PROTOCOL,'public_file_sha256':PUBLIC_SHA256,'n_jobs':len(jobs),'n_score_rows':2*len(jobs),
        'prompt_conditions':list(PROMPT_CONDITIONS),'assistant_prefix':ASSISTANT_PREFIX,'dtype':'float32',
        'weight_load_dtype':'bfloat16','model_hashes_bound_to_original_run':True,'batch_size':execution['batch_size'],
        'padding_side':'left',
        'filler_policy':'duplicate final complete original/forced pair; discard filler outputs' if execution['cached'] else 'duplicate final real context to fixed batch size; discard filler outputs',
        'attention_implementation':'eager','tf32':False,'float32_matmul_precision':'highest',
        'deterministic_algorithms':True,'logits_to_keep':1,'use_cache':execution['cached'],'generation':False,'sampling':False,
        'quantization':False,'validation_method':execution['validation_method'],
        'fp32_numeric_gate':{'atol':1e-3,'rtol':1e-5,'exact_top_equality_required':False,'exact_replay_required':True}}
    for key,wanted in required.items():
        if metadata.get(key)!=wanted:raise ValueError(f'metadata {key} mismatch')
    for key in ('model_files_sha256','expected_model_hashes_receipt_sha256','chat_template_sha256'):
        if not metadata.get(key):raise ValueError(f'missing model provenance: {key}')
    cache_binding=validate_model_cache_binding(directory,tag,metadata)
    validate_source_identity(metadata,source_root)
    manifest=validate_prompt_manifest(directory,metadata,jobs)
    rows=[]
    with (directory/'scores.jsonl').open() as stream:
        for line in stream:
            if not line.strip():raise ValueError('blank score row')
            rows.append(json.loads(line,object_pairs_hook=_unique))
    pairs=validate_rows(rows,jobs);validate_contexts(rows,metadata)
    numeric=validate_numeric_evidence(directory,metadata,receipt,jobs,rows)
    files=('receipt.json','metadata.json','scores.jsonl','prompts.json','diagnostic_selection.json',*execution['numeric_files'])+(('cache_plan.json',) if execution['cached'] else ())
    return pairs,{'metadata':metadata,'receipt':receipt,'prompt_manifest':manifest,'original_cache_binding':cache_binding,
                  'numeric_validation':numeric,'files_sha256':{name:sha256(directory/name) for name in files}}


CSV_FIELDS=['model_tag','job_id','qid','group_id','split','condition','menu_id']+[f'option_text_{c}' for c in OPTIONS]+[
    f'{variant}_{field}' for variant in PROMPT_CONDITIONS for field in
    ('top_option_id','top_option_text','tied_top_option_ids','logit_margin','probability_margin')]+[
    f'{variant}_{kind}_{c}' for variant in PROMPT_CONDITIONS for kind in ('probability','logit') for c in OPTIONS]+[
    'total_variation','max_abs_probability_shift','top_option_changed','top_option_set_changed','top_option_sets_disjoint',
    'centered_logit_shift_rms','max_abs_pairwise_log_odds_shift']+[
    f'probability_delta_{c}' for c in OPTIONS]+[f'centered_logit_delta_{c}' for c in OPTIONS]+[
    f'pairwise_log_odds_shift_{left}_{right}' for left,right in OPTION_PAIRS]


def csv_pair_row(tag,pair,job):
    if pair['job_id']!=job['job_id']:raise ValueError('CSV pair/menu identity mismatch')
    options={item['id']:item['text'] for item in job['options']}
    row={'model_tag':tag,**{key:pair[key] for key in ('job_id','qid','group_id','split','condition','menu_id')}}
    row.update({f'option_text_{c}':options[c] for c in OPTIONS})
    for variant in PROMPT_CONDITIONS:
        record=pair[variant]
        row.update({f'{variant}_{key}':record[key] for key in ('top_option_id','logit_margin','probability_margin')})
        row[f'{variant}_top_option_text']=options[record['top_option_id']]
        row[f'{variant}_tied_top_option_ids']=json.dumps(record['tied_top_option_ids'],separators=(',',':'))
        for c in OPTIONS:
            row[f'{variant}_probability_{c}']=record['conditional_option_probabilities'][c]
            row[f'{variant}_logit_{c}']=record['raw_option_logits'][c]
    for field in CSV_FIELDS:
        if field not in row:row[field]=pair[field]
    return row


def analyze(public_path,scores_root,out,source_root=REPO,samples=2000,seed=1):
    jobs=load_public(public_path);job_map={j['job_id']:j for j in jobs}
    if out.exists():raise ValueError('output directory exists; use a new analysis directory')
    summaries=[];provenance={};export=[]
    for tag in MODELS:
        pairs,evidence=load_model(scores_root/tag,tag,jobs,source_root)
        provenance[tag]=evidence
        export.extend(csv_pair_row(tag,pair,job_map[pair['job_id']]) for pair in pairs)
        for subset in ('all','test'):
            subset_pairs=[p for p in pairs if subset=='all' or p['split']=='test']
            for condition in (*MENU_CONDITIONS,'pooled_menus'):
                selected=[p for p in subset_pairs if condition=='pooled_menus' or p['condition']==condition]
                summaries.append({'model_tag':tag,'subset':subset,'menu_condition':condition,
                                  **summarize_pairs(selected,samples,seed)})
    limits=[
        'Primary estimand: change in the conditional A--D next-token distribution after the fixed assistant prefix, with only the terminal abstention instruction replaced.',
        'This intervention changes the prompt text. It is not removal of an abstention option by masking an otherwise fixed probability vector, and does not test classical independence of irrelevant alternatives.',
        'These are forward-pass option scores, not generated full-response frequencies, abstention probabilities, calibrated correctness probabilities, or an estimate of willingness to answer.',
        'Primary scores use FP32 arithmetic on the original stored weights loaded as BF16 and promoted; original model IDs and weight-file hashes are retained.',
        'When explicitly declared, primary inference reuses the exact token common-prefix KV cache and validates cached logits against the uncached FP32 reference, cached replay, pair permutation, and production outputs at unchanged tolerances.',
        'FP32 numerical gates cover a deliberately selected 16-score diagnostic subset. They do not mathematically bound numerical errors on every scored context.',
        'BF16 comparisons cover eight menus selected by extreme token lengths, and are descriptive rather than representative of all menus.',
        'All/test summaries overlap. Bootstrap intervals resample questions, preserving their menus; intervals are marginal with no multiplicity adjustment and condition on these frozen menus.',
        'Source and artifact hashes validate identity and consistency. Tensor-promotion records include dtype and sampled-value checks, and do not independently attest every tensor value or the remote hardware.',
    ]
    report={'schema_version':'acl-paired-prompt-analysis-v1','protocol':PROTOCOL,
            'public_sha256':sha256(public_path),'primary_dtype':'float32','assistant_prefix':ASSISTANT_PREFIX,
            'prompt_conditions':list(PROMPT_CONDITIONS),'signed_effect_direction':'forced minus original',
            'total_variation_definition':'0.5 * sum_A_to_D(abs(p_forced - p_original))',
            'centered_logit_definition':'(logits_forced - logits_original) minus its mean across A--D',
            'pairwise_log_odds_definition':'(logit_forced[a]-logit_forced[b]) - (logit_original[a]-logit_original[b]); natural-log units',
            'top_tie_policy':'first A/B/C/D among exact maxima; top-set changes and disjoint top sets reported separately',
            'summaries':summaries,'provenance':provenance,'interpretation_limits':limits}
    out.mkdir(parents=True);write_json(out/'report.json',report)
    with (out/'per_menu_paired_scores.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_FIELDS);writer.writeheader();writer.writerows(export)
    fields=['model_tag','subset','menu_condition','n_questions','n_menus',*SUMMARY_METRICS]
    with (out/'summary.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in summaries)
    numeric_export=[{'model_tag':tag,**row} for tag,evidence in provenance.items()
                    for row in evidence['numeric_validation']['precision_subset_effects']]
    with (out/'precision_subset_effects.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(numeric_export[0]));writer.writeheader();writer.writerows(numeric_export)
    with (out/'findings.txt').open('x',encoding='utf-8') as stream:
        stream.write(limits[0]+'\n\n')
        for row in summaries:
            lo,hi=row['question_bootstrap_95_intervals']['mean_total_variation']
            stream.write(f"{row['model_tag']} {row['subset']} {row['menu_condition']}: {row['n_menus']} paired menus / {row['n_questions']} questions; mean TV {row['mean_total_variation']:.6f} (question-bootstrap 95% {lo:.6f}, {hi:.6f}); top option changes {row['top_option_change_fraction']:.3%}.\n")
        stream.write('\nInterpretation limits:\n'+'\n'.join(limits)+'\n')
    output_names=('report.json','per_menu_paired_scores.csv','summary.csv','precision_subset_effects.csv','findings.txt')
    write_json(out/'analysis_receipt.json',{'status':'complete','inference_run':False,'gold_labels_used':False,
        'analysis_source_sha256':sha256(Path(__file__)),
        'dependency_source_sha256':{'scripts/analyze_acl_option_scores.py':sha256(REPO/'scripts/analyze_acl_option_scores.py')},
        'paired_menus':len(export),'score_rows':2*len(export),
        'outputs_sha256':{name:sha256(out/name) for name in output_names}})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public',type=Path,required=True)
    parser.add_argument('--scores-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=REPO)
    parser.add_argument('--bootstrap-samples',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=1)
    args=parser.parse_args()
    if args.bootstrap_samples<100:parser.error('at least 100 bootstrap samples required')
    report=analyze(args.public,args.scores_root,args.out,args.source_root,args.bootstrap_samples,args.seed)
    print(json.dumps({'status':'complete','paired_menus':sum(r['n_menus'] for r in report['summaries']
        if r['subset']=='all' and r['menu_condition']=='pooled_menus'),'output_directory':str(args.out)},indent=2))


if __name__=='__main__':main()
