#!/usr/bin/env python3
"""Validate and analyze frozen menu-only option scores; never runs inference.

The estimand is option preference conditional on a fixed assistant answer
prefix. Conditional A--D probabilities are not probabilities of correctness.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

OPTIONS = ('A', 'B', 'C', 'D')
CONDITIONS = ('independent_pool', 'same_category_pool')
NEAR_TIE_LOGIT_ATOL = 1e-6
PROTOCOL = 'conditional_next_token_option_softmax_v1'
ASSISTANT_PREFIX = '{"answer":"'
PUBLIC_SHA256 = '9db13301d928cd31dc54c97f0c5cfd88b9bc25774ceb56027c032c52e4afb043'
GOLD_SHA256 = '93a4eec6e9792e432f96b55e88a527765b812635f3c11a70f94e8361095a8162'
MODELS = {
    'qwen3b': ('Qwen/Qwen2.5-3B-Instruct', 'aa8e72537993ba99e69dfaafa59ed015b17504d1'),
    'qwen7b': ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def wilson_interval(k, n, confidence=.95):
    """Return a binomial Wilson interval; no interval for an empty sample."""
    if not n:
        return None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def binomial_upper_tail(k, n, p=.25):
    """Exact upper-tail binomial reference, evaluated with log-sum-exp."""
    if k == 0:
        return 1.
    values = [math.lgamma(n+1) - math.lgamma(i+1) - math.lgamma(n-i+1)
              + i*math.log(p) + (n-i)*math.log1p(-p) for i in range(k, n+1)]
    maximum = max(values)
    return min(1., math.exp(maximum) * math.fsum(math.exp(x-maximum) for x in values))


def holm_adjust(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.] * len(pvalues)
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(pvalues)-rank)*pvalues[index]))
        adjusted[index] = previous
    return adjusted


def validate_rows(rows, jobs, gold):
    """Reject incomplete, duplicate, inconsistent or nonfinite score records."""
    expected = {job['job_id']: job for job in jobs}
    if len(expected) != len(jobs) or set(gold) != set(expected):
        raise ValueError('public/gold identity mismatch or duplicate public job IDs')
    seen = set()
    validated = []
    for row in rows:
        ident = row.get('job_id')
        if ident not in expected or ident in seen:
            raise ValueError(f'unknown/duplicate score job ID: {ident}')
        seen.add(ident)
        job = expected[ident]
        for field in ('qid', 'group_id', 'split', 'condition', 'menu_id', 'prompt_sha256'):
            if row.get(field) != job[field]:
                raise ValueError(f'score/public {field} mismatch: {ident}')
        if row.get('schema_version') != 'acl-option-scores-v1':
            raise ValueError('wrong score schema')
        logits = row.get('raw_option_logits', {})
        probs = row.get('conditional_option_probabilities', {})
        if set(logits) != set(OPTIONS) or set(probs) != set(OPTIONS):
            raise ValueError('exactly A--D logits and probabilities required')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (*logits.values(),*probs.values())):
            raise ValueError('nonfinite/non-numeric score')
        if any(v < 0 or v > 1 for v in probs.values()) or abs(math.fsum(probs.values())-1) > 2e-6:
            raise ValueError('invalid probability range or normalization')
        top = max(logits.values())
        exact_ties = [x for x in OPTIONS if logits[x] == top]
        weights = {x: math.exp(logits[x]-top) for x in OPTIONS}
        denom = math.fsum(weights.values())
        if any(abs(probs[x]-weights[x]/denom) > 2e-6 for x in OPTIONS):
            raise ValueError('probabilities do not match logit softmax')
        if row.get('top_option_id') != exact_ties[0] or row.get('tied_top_option_ids') != exact_ties:
            raise ValueError('top option or exact ties mismatch')
        if gold[ident] not in OPTIONS:
            raise ValueError('invalid gold option')
        validated.append({**row, 'gold_option_id': gold[ident],
                          'correct': row['top_option_id'] == gold[ident],
                          'near_top_option_ids': [x for x in OPTIONS if top-logits[x] <= NEAR_TIE_LOGIT_ATOL]})
    if seen != set(expected):
        raise ValueError(f'incomplete score coverage: {len(seen)}/{len(expected)}')
    return validated


def summarize_rows(rows):
    """Summarize one model, menu condition, and question subset."""
    n = len(rows)
    k = sum(r['correct'] for r in rows)
    if not n:
        raise ValueError('cannot summarize an empty subset')
    gold_counts = {x: sum(r['gold_option_id']==x for r in rows) for x in OPTIONS}
    selected_counts = {x: sum(r['top_option_id']==x for r in rows) for x in OPTIONS}
    majority = max(gold_counts.values())
    uniform_tie_sum = math.fsum((1 / len(r['tied_top_option_ids']))
                              if r['gold_option_id'] in r['tied_top_option_ids'] else 0. for r in rows)
    lower = sum(r['gold_option_id'] in r['tied_top_option_ids'] and len(r['tied_top_option_ids'])==1 for r in rows)
    upper = sum(r['gold_option_id'] in r['tied_top_option_ids'] for r in rows)
    return {
        'n': n, 'n_correct': k, 'accuracy': k/n, 'wilson_95_interval': wilson_interval(k,n),
        'uniform_guessing_reference': .25, 'accuracy_minus_uniform_reference': k/n-.25,
        'binomial_one_sided_p_above_25_percent': binomial_upper_tail(k,n),
        'gold_option_counts': gold_counts, 'selected_option_counts': selected_counts,
        'evaluated_set_majority_position_baseline_accuracy': majority/n,
        'evaluated_set_majority_position_options': [x for x in OPTIONS if gold_counts[x]==majority],
        'evaluated_set_majority_baseline_note': 'Oracle descriptive baseline using this evaluated subset; not a learned policy.',
        'n_exact_top_ties': sum(len(r['tied_top_option_ids'])>1 for r in rows),
        'n_near_top_ties': sum(len(r['near_top_option_ids'])>1 for r in rows),
        'near_tie_absolute_logit_tolerance': NEAR_TIE_LOGIT_ATOL,
        'uniform_exact_tie_expected_accuracy': uniform_tie_sum/n,
        'exact_tie_accuracy_bounds': [lower/n,upper/n],
        'mean_gold_conditional_probability': math.fsum(r['conditional_option_probabilities'][r['gold_option_id']] for r in rows)/n,
        'mean_top_conditional_probability': math.fsum(max(r['conditional_option_probabilities'].values()) for r in rows)/n,
    }


def paired_menu_report(rows, samples=5000, seed=1):
    """Paired question bootstrap for the accuracy difference between menus."""
    paired = defaultdict(dict)
    for row in rows:
        if row['condition'] in paired[row['qid']]:
            raise ValueError('duplicate question/condition in paired comparison')
        paired[row['qid']][row['condition']] = row
    if not paired or any(set(pair)!=set(CONDITIONS) for pair in paired.values()):
        raise ValueError('paired comparison requires both menus for every question')
    values = np.asarray([int(pair[CONDITIONS[0]]['correct']) - int(pair[CONDITIONS[1]]['correct'])
                         for _,pair in sorted(paired.items())],dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for start in range(0,samples,100):
        count = min(100,samples-start)
        boot[start:start+count] = values[rng.integers(0,len(values),size=(count,len(values)))].mean(axis=1)
    agreements = sum(pair[CONDITIONS[0]]['top_option_id']==pair[CONDITIONS[1]]['top_option_id'] for pair in paired.values())
    return {'n_questions':len(values),
            'accuracy_delta_independent_minus_same_category':float(values.mean()),
            'paired_question_bootstrap_95_interval':[float(v) for v in np.quantile(boot,[.025,.975])],
            'bootstrap_samples':samples, 'bootstrap_seed':seed,
            'bootstrap_unit':'question, with its two menus kept together',
            'independent_only_correct':int(np.sum(values==1)),
            'same_category_only_correct':int(np.sum(values==-1)),
            'selected_option_id_agreements':agreements,
            'interpretation':'Descriptive paired interval, conditional on the frozen menus and scoring protocol; no multiplicity adjustment for menu differences.'}


def load_frozen_inputs(public_path, gold_path):
    if sha256(public_path) != PUBLIC_SHA256 or sha256(gold_path) != GOLD_SHA256:
        raise ValueError('frozen public or gold file hash mismatch')
    package, gold = load_json(public_path), load_json(gold_path)
    jobs = package['jobs']
    if package.get('schema_version') != 'jane-choice-controls-v1' or package.get('evidence_scope') != 'scientific':
        raise ValueError('wrong public schema/scope')
    by_qid = defaultdict(list)
    for index,job in enumerate(jobs):
        if hashlib.sha256(job['prompt'].encode()).hexdigest() != job['prompt_sha256']:
            raise ValueError('per-job prompt hash mismatch')
        if [x['id'] for x in job['options']] != list(OPTIONS):
            raise ValueError('expected A--D option order')
        by_qid[job['qid']].append(job)
    if len(jobs)!=10000 or len(by_qid)!=5000 or len(gold)!=10000:
        raise ValueError('expected full frozen 5000-question, 10000-menu dataset')
    if Counter(v[0]['split'] for v in by_qid.values()) != Counter(calibration=1000,selection=1000,test=3000):
        raise ValueError('wrong frozen split counts')
    for pair in by_qid.values():
        if len(pair)!=2 or {j['condition'] for j in pair} != set(CONDITIONS):
            raise ValueError('wrong paired menu coverage')
        if len({j['group_id'] for j in pair})!=1 or len({j['split'] for j in pair})!=1:
            raise ValueError('inconsistent question group/split')
    if len({v[0]['group_id'] for v in by_qid.values()}) != 5000:
        raise ValueError('question bootstrap requires these frozen one-question groups')
    return jobs,gold


def load_model_scores(directory, tag, jobs, gold):
    receipt = load_json(directory/'receipt.json')
    metadata = load_json(directory/'metadata.json')
    scores_path = directory/'scores.jsonl'
    if receipt.get('status') != 'complete' or receipt.get('completed_rows') != len(jobs) or receipt.get('expected_rows') != len(jobs):
        raise ValueError(f'{tag}: full complete receipt required')
    for path,key in ((scores_path,'scores_sha256'),(directory/'metadata.json','metadata_sha256')):
        if sha256(path)!=receipt.get(key):
            raise ValueError(f'{tag}: {key} mismatch')
    model,revision=MODELS[tag]
    for key,wanted in {'model_tag':tag,'model':model,'revision':revision,'public_file_sha256':PUBLIC_SHA256,
                       'n_jobs':len(jobs),'assistant_prefix':ASSISTANT_PREFIX,'protocol':PROTOCOL}.items():
        if metadata.get(key)!=wanted:
            raise ValueError(f'{tag}: metadata {key} mismatch')
    for key in ('model_files_sha256','chat_template_sha256','source_sha256'):
        if not metadata.get(key):
            raise ValueError(f'{tag}: missing provenance {key}')
    rows=[]
    with scores_path.open() as stream:
        for line in stream:
            if not line.strip():
                raise ValueError('blank score row')
            rows.append(json.loads(line,object_pairs_hook=_unique))
    if [row.get('job_index') for row in rows] != list(range(len(jobs))):
        raise ValueError('score job_index order/coverage mismatch')
    if [row.get('job_id') for row in rows] != [job['job_id'] for job in jobs]:
        raise ValueError('score/public job order mismatch')
    return validate_rows(rows,jobs,gold), metadata, receipt


def analyze(public_path, gold_path, scores_root, out, samples=5000, seed=1):
    jobs,gold=load_frozen_inputs(public_path,gold_path)
    if out.exists():
        raise ValueError('output directory exists; use a new analysis output directory')
    summaries=[]
    paired=[]
    provenance={}
    for tag in MODELS:
        directory=scores_root/tag
        rows,metadata,receipt=load_model_scores(directory,tag,jobs,gold)
        provenance[tag]={'metadata':metadata,'receipt':receipt,
                         'files_sha256':{name:sha256(directory/name) for name in ('scores.jsonl','receipt.json','metadata.json')}}
        for subset in ('all','test'):
            subset_rows=[r for r in rows if subset=='all' or r['split']=='test']
            for condition in CONDITIONS:
                group=[r for r in subset_rows if r['condition']==condition]
                summary={'model_tag':tag,'subset':subset,'condition':condition,**summarize_rows(group)}
                development=[r for r in rows if r['split']=='calibration' and r['condition']==condition]
                counts=Counter(r['gold_option_id'] for r in development)
                baseline=max(OPTIONS,key=lambda x:counts[x])
                summary['calibration_selected_constant_option']=baseline
                summary['calibration_selected_position_baseline_accuracy']=sum(r['gold_option_id']==baseline for r in group)/len(group)
                summaries.append(summary)
            paired.append({'model_tag':tag,'subset':subset,**paired_menu_report(subset_rows,samples,seed)})
    tests=[r for r in summaries if r['subset']=='test']
    for row,pvalue in zip(tests,holm_adjust([r['binomial_one_sided_p_above_25_percent'] for r in tests])):
        row['holm_p_above_25_percent_four_heldout_comparisons']=pvalue
    report={'schema_version':'acl-option-scores-analysis-v1',
            'estimand':'Accuracy of the top A--D next-token logit conditional on the unchanged menu-only prompt plus fixed assistant prefix '+repr(ASSISTANT_PREFIX),
            'inputs_sha256':{'public':sha256(public_path),'gold':sha256(gold_path)},
            'protocol':PROTOCOL,'tie_rule':'Exact argmax; ties break in A,B,C,D order. Near ties are flagged without changing decisions.',
            'summaries':summaries,'paired_menu_comparisons':paired,'provenance':provenance,
            'interpretation_limits':[
                'Diagnostic follow-up defined after observing universal abstention; not a preregistered confirmatory outcome.',
                'The original prompt still allows abstention; the fixed assistant prefix conditions evaluation on emitting an option.',
                'A--D softmax scores are conditional relative preferences, not calibrated correctness probabilities.',
                'Uniform 25% guessing is a reference; gold and selected position frequencies and constant-position baselines are also reported.',
                'Wilson intervals and exact binomial tests use an independent-question Bernoulli reference; frozen questions are not a random sample of all quiz bowl.',
                'All-set and held-out summaries overlap. Holm correction covers only the four model-by-condition held-out tests; all-set tests are descriptive.',
                'Above-reference accuracy can motivate menu construction or positional-bias investigation; it does not prove shortcut use when question clues are present.',
                'Model/source metadata are recorded provenance assertions; file-hash checks do not independently attest remote hardware execution.',
            ]}
    out.mkdir(parents=True)
    write_json(out/'report.json',report)
    fields=['model_tag','subset','condition','n','n_correct','accuracy','wilson_lower','wilson_upper',
            'accuracy_minus_uniform_reference','evaluated_set_majority_position_baseline_accuracy',
            'calibration_selected_constant_option','calibration_selected_position_baseline_accuracy',
            'binomial_one_sided_p_above_25_percent','holm_p_above_25_percent_four_heldout_comparisons',
            'n_exact_top_ties','n_near_top_ties']
    with (out/'summary.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in summaries:
            lo,hi=row['wilson_95_interval']
            writer.writerow({key:({**row,'wilson_lower':lo,'wilson_upper':hi}).get(key,'') for key in fields})
    with (out/'findings.txt').open('x') as stream:
        stream.write(report['estimand']+'\n\n')
        for row in summaries:
            lo,hi=row['wilson_95_interval']
            stream.write(f"{row['model_tag']} {row['subset']} {row['condition']}: {row['n_correct']}/{row['n']} = {row['accuracy']:.3%}; Wilson 95% [{lo:.3%}, {hi:.3%}]; gold-position majority {row['evaluated_set_majority_position_baseline_accuracy']:.3%}; exact ties {row['n_exact_top_ties']}.\n")
        stream.write('\nInterpretation limits:\n'+'\n'.join(report['interpretation_limits'])+'\n')
    write_json(out/'analysis_receipt.json',{'status':'complete','inference_run':False,
        'analysis_source_sha256':sha256(Path(__file__)),
        'outputs_sha256':{name:sha256(out/name) for name in ('report.json','summary.csv','findings.txt')}})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public',type=Path,required=True)
    parser.add_argument('--gold',type=Path,required=True)
    parser.add_argument('--scores-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bootstrap-samples',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=1)
    args=parser.parse_args()
    if args.bootstrap_samples<100:
        parser.error('at least 100 bootstrap samples required')
    report=analyze(args.public,args.gold,args.scores_root,args.out,args.bootstrap_samples,args.seed)
    print(json.dumps({'status':'complete','n_score_rows':sum(r['n'] for r in report['summaries'] if r['subset']=='all'),
                      'output_directory':str(args.out)},indent=2))


if __name__=='__main__':
    main()
