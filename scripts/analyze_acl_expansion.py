#!/usr/bin/env python3
"""Postprocess actual expanded-model traces; never generates model responses.

Unmatched OE answers stay unresolved in primary outputs. Optional exact-match
proxy analysis changes the estimand explicitly and is never semantic grading.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import statistics
import random
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_CONTRACT = REPO / 'configs/acl_expansion_analysis.json'
UNRESOLVED = {'needs_review', 'clarification_required'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()

def load(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f'duplicate JSON key: {k}')
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)

def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

def verify_contract(frozen_dir, contract_path=DEFAULT_CONTRACT):
    contract = load(contract_path)
    for base, mapping in ((frozen_dir, contract['frozen_inputs_sha256']),
                          (REPO, contract['code_inputs_sha256'])):
        for relative, digest in mapping.items():
            if sha(base / relative) != digest:
                raise ValueError(f'frozen analysis input changed: {relative}')
    from qb_data.jane_paired import validate_dataset
    dataset = load(frozen_dir / 'evaluator/main_dataset.json')
    validate_dataset(dataset)
    splits = dict(Counter(q['split'] for q in dataset['questions']))
    if splits != contract['question_splits']:
        raise ValueError(f'wrong question splits: {splits}')
    if any(len(q['prefixes']) != 10 or len(q['menus']) != 2 for q in dataset['questions']):
        raise ValueError('expected exactly ten prefixes and two MC menus per question')
    return contract, dataset

def accuracy_bounds(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['qid']].append(row)
    lower, upper = [], []
    for group in grouped.values():
        lower.append(sum(r['grade'] == 'accepted' for r in group) / len(group))
        upper.append(sum(r['grade'] == 'accepted' or r['grade'] in UNRESOLVED for r in group) / len(group))
    return {'n_questions': len(grouped), 'n_prefixes': len(rows),
            'grade_counts': dict(Counter(r['grade'] for r in rows)),
            'accuracy_identification_interval': [statistics.mean(lower), statistics.mean(upper)] if grouped else None,
            'interpretation': 'conditional on deterministic accepts/rejects; unresolved grading endpoints, not a sampling interval'}

def fixed_policy_bounds(trajectories, calibrator, threshold):
    from evaluation.jane_paired import first_crossing
    selected = [r for tr in trajectories.values() if (r := first_crossing(tr, calibrator, threshold)) is not None]
    n = len(selected)
    errors = sum(r['grade'] in {'rejected', 'abstain', 'invalid'} for r in selected)
    unknown = sum(r['grade'] in UNRESOLVED for r in selected)
    return {'n_questions': len(trajectories), 'n_committed': n,
            'n_known_correct_commits': sum(r['grade'] == 'accepted' for r in selected),
            'n_known_incorrect_commits': errors, 'n_unresolved_commits': unknown,
            'coverage': n / len(trajectories),
            'mean_commitment_fraction': statistics.mean(r['fraction'] for r in selected) if n else None,
            'risk_identification_interval': [errors/n, (errors+unknown)/n] if n else None,
            'interpretation': 'policy was fixed without target correctness; risk endpoints condition on deterministic accepted/rejected labels; coverage and positions do not require OE correctness',
            'commitments': [{k: r[k] for k in ('qid', 'job_id', 'prefix_id', 'fraction', 'grade', 'confidence')} for r in selected]}

def primary_report(rows, contract):
    from evaluation.jane_paired import arm_id, fit_calibrator, select_threshold, _GroupBootstrap, _policy_report
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        data[arm_id(r)][r['split']][r['qid']].append(r)
    stats, native, transfers = {}, {}, {}
    settings = contract['analysis']
    boot = _GroupBootstrap({q: tr[0]['group_id'] for q, tr in data['oe']['test'].items()}, settings['bootstrap_samples'], settings['seed'])
    for arm, splits in data.items():
        stats[arm] = {}
        for split, trajectories in splits.items():
            rr = [r for tr in trajectories.values() for r in tr]
            stats[arm][split] = {'all_prefixes': accuracy_bounds(rr),
                'full_question': accuracy_bounds([r for r in rr if r['fraction'] == 1]),
                'early_fraction_le_0_2': accuracy_bounds([r for r in rr if r['fraction'] <= .2]),
                'fraction_bins': [{'lower': i/10, 'upper': (i+1)/10,
                                  **accuracy_bounds([r for r in rr if i/10 < r['fraction'] <= (i+1)/10])} for i in range(10)]}
        if arm == 'oe':
            continue
        try:
            c = fit_calibrator(splits['calibration'].values())
        except ValueError as exc:
            if 'no resolved answer confidences' not in str(exc):
                raise
            native[arm] = {'available': False, 'reason': str(exc)}
            transfers[arm] = {'available': False, 'reason': 'MC calibration unavailable'}
            continue
        selection = select_threshold(splits['selection'], c, settings['risk_budget'])
        native[arm] = {'available': True, 'calibrator': c, 'selection': selection,
                      'test': _policy_report(splits['test'], c, selection['threshold'], boot, settings['risk_budget'])}
        transfers[arm] = {'available': True, 'source_arm': arm, 'target_arm': 'oe',
                         'n_oe_development_labeled_questions': 0, 'threshold': selection['threshold'],
                         'test': fixed_policy_bounds(data['oe']['test'], c, selection['threshold'])}
    return {'schema_version': 'acl-expanded-primary-analysis-v1', 'accuracy': stats,
            'native_mc_policies': native, 'direct_oe_transfers_with_unresolved_risk': transfers,
            'oe_recalibration_status': 'not estimated in this primary automatic report; use resolved semantic adjudications or explicitly labeled exact-match proxy scenario',
            'config': settings, 'warnings': contract['interpretation_limits']}

def run(args):
    frozen_dir = args.frozen_dir.resolve()
    contract, dataset = verify_contract(frozen_dir, args.contract)
    if args.command == 'preflight':
        print(json.dumps({'status': 'preflight_passed_no_inference',
                          'dataset_sha256': sha(frozen_dir/'evaluator/main_dataset.json'),
                          'question_splits': contract['question_splits'],
                          'expected_main_predictions_per_model': 150000,
                          'expected_control_predictions_per_model': 10000}, indent=2))
        return
    if args.command == 'controls':
        trace = load(args.trace)
        marker = trace.get('metadata', {}).get('execution', '')
        if not isinstance(marker, str) or not marker.startswith('actual_') or 'generation' not in marker:
            raise ValueError('controls trace lacks actual-generation marker')
        if len(trace.get('predictions', [])) != 10000:
            raise ValueError('complete 10,000-response controls trace required')
        analyze_controls(argparse.Namespace(repo=REPO, dataset=frozen_dir/'evaluator/main_dataset.json',
            jobs=frozen_dir/'public/main_choices_only.json', trace=args.trace, out=args.out,
            graded=args.graded, bootstrap_samples=contract['analysis']['bootstrap_samples'],
            seed=contract['analysis']['seed']))
        verify_contract(frozen_dir, args.contract)
        return
    from qb_data.jane_paired import build_jobs, grade_predictions
    from evaluation.jane_paired import analyze
    if args.out.exists():
        raise ValueError('output directory already exists; use a new run-specific directory')
    inputs = {str(args.trace.resolve()): sha(args.trace)}
    trace = load(args.trace)
    metadata = trace.get('metadata', {})
    marker = metadata.get('execution', '')
    if not isinstance(marker, str) or not marker.startswith('actual_') or 'generation' not in marker:
        raise ValueError('trace lacks an explicit actual model-generation execution marker')
    if len(trace.get('predictions', [])) != 150000:
        raise ValueError('complete 150,000-response main trajectory trace required; partial completion is not an experiment result')
    if metadata.get('evidence_scope') != 'scientific':
        raise ValueError('trace evidence_scope must match frozen dataset: scientific')
    decisions = None
    if args.adjudications:
        decisions = load(args.adjudications)
        inputs[str(args.adjudications.resolve())] = sha(args.adjudications)
    rows = grade_predictions(dataset, build_jobs(dataset), trace)
    args.out.mkdir(parents=True)
    write(args.out/'automatic_graded_rows.json', rows)
    write(args.out/'automatic_report.json', primary_report(rows, contract))
    unresolved = Counter(r['grade'] for r in rows if r['grade'] in UNRESOLVED)
    status = {'automatic_analysis': 'completed', 'unresolved_automatic_rows': dict(unresolved),
              'semantic_analysis': 'unavailable_pending_adjudication', 'exact_match_proxy': 'not_requested'}
    if decisions is not None or not unresolved:
        reviewed = grade_predictions(dataset, build_jobs(dataset), trace, adjudications=decisions) if decisions is not None else rows
        write(args.out/'resolved_graded_rows.json', reviewed)
        remaining = sum(r['grade'] in UNRESOLVED for r in reviewed)
        status['unresolved_after_adjudication'] = remaining
        if remaining == 0:
            settings = contract['analysis']
            try:
                report = analyze(reviewed, **settings)
            except ValueError as exc:
                if 'no resolved answer confidences' not in str(exc):
                    raise
                status['semantic_analysis'] = f'unavailable: {exc}'
            else:
                report['grading_status'] = 'provided adjudications; human validation is not inferred' if decisions is not None else 'all deterministic rules resolved; independent semantic validation not inferred'
                write(args.out/'resolved_report.json', report)
                status['semantic_analysis'] = 'completed_conditional_on_provided_judgments'
        else:
            status['semantic_analysis'] = 'unavailable_incomplete_adjudication'
    if args.exact_match_proxy:
        proxy = copy.deepcopy(rows)
        for r in proxy:
            if r['grade'] in UNRESOLVED:
                r.update(grade='rejected', correct=False, proxy_assumption='nonmatch counted as failure of deterministic-match target; not semantic rejection')
        try:
            report = analyze(proxy, **contract['analysis'])
        except ValueError as exc:
            if 'no resolved answer confidences' not in str(exc):
                raise
            status['exact_match_proxy'] = f'unavailable: {exc}'
        else:
            report['estimand'] = 'probability of satisfying deterministic answer-string matching, not semantic answer correctness'
            report['unresolved_semantic_grades_retained_in'] = 'automatic_graded_rows.json'
            report['warnings'].append('Every unmatched OE answer counts as proxy failure only; do not report these risks as semantic correctness or compare them directly to human-adjudicated pilot risk.')
            write(args.out/'exact_match_proxy_report.json', report)
            status['exact_match_proxy'] = 'completed_different_estimand'
    if args.make_review_packet:
        make_review_packet(argparse.Namespace(dataset=frozen_dir/'evaluator/main_dataset.json', trace=[args.trace], out=args.out/'blinded_oe_review'))
        status['review_packet'] = 'created_from_actual_answered_oe_outputs'
    for p, digest in inputs.items():
        if sha(p) != digest:
            raise ValueError(f'input mutated during analysis: {p}')
    verify_contract(frozen_dir, args.contract)
    write(args.out/'completion_receipt.json', {'schema_version': 'acl-expanded-analysis-receipt-v1',
        'status': status, 'trace_metadata': metadata, 'inputs_sha256': inputs,
        'analysis_contract_sha256': sha(args.contract), 'runner_sha256': sha(__file__),
        'outputs_sha256': {str(p.relative_to(args.out)): sha(p) for p in args.out.rglob('*') if p.is_file()}})
    print(json.dumps(status, indent=2))

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_review_packet(args):
    from qb_data.jane_paired import build_jobs, grade_predictions
    dataset = load(args.dataset)
    jobs = build_jobs(dataset)
    question_by_id = {q['qid']: q for q in dataset['questions']}
    cases, mapping, summaries, inputs = {}, defaultdict(list), {}, {str(args.dataset.resolve()): sha(args.dataset)}
    for trace_path in args.trace:
        trace_path = trace_path.resolve()
        digest = sha(trace_path)
        trace = load(trace_path)
        rows = grade_predictions(dataset, jobs, trace)
        summaries[str(trace_path)] = {'sha256': digest, 'n_rows': len(rows),
            'grade_counts': dict(Counter(row['grade'] for row in rows)),
            'n_questions': len(dataset['questions'])}
        inputs[str(trace_path)] = digest
        for row in rows:
            if row['format'] != 'oe' or row['status'] != 'answer':
                continue
            # Include exact accepts as an answerline audit. Never disclose model,
            # split, calibrated/raw confidence, automatic grade, or other arms.
            q = question_by_id[row['qid']]
            prefix = next(p for p in q['prefixes'] if p['prefix_id'] == row['prefix_id'])
            key = identity({'qid': row['qid'], 'answer': row['answer']})
            prefix_key = identity({'qid': row['qid'], 'prefix_text': prefix['text']})
            if key not in cases:
                cases[key] = {'case_id': key, 'answer': row['answer'],
                    'full_question': q['question'], 'full_answerline': q['answer']['raw'],
                    'prefixes': {}}
            cases[key]['prefixes'][prefix_key] = {'prefix_key': prefix_key, 'text': prefix['text']}
            mapping[key].append({'trace_path': str(trace_path), 'trace_sha256': digest,
                'job_id': row['job_id'], 'prompt_sha256': row['prompt_sha256'],
                'answer': row['answer'], 'prefix_key': prefix_key})
    ordered = []
    for key in sorted(cases):
        case = cases[key]
        case['prefixes'] = sorted(case['prefixes'].values(), key=lambda p: len(p['text']))
        ordered.append(case)
    args.out.mkdir(parents=True, exist_ok=False)
    review = {'schema': 'jane-blinded-oe-review-v1', 'instructions': [
        'Judge only semantic answer correctness against the full official answerline.',
        'Accept exact aliases or clearly equivalent names; reject different referents and insufficiently specific answers.',
        'Do not assess whether the clues justify confidence or whether a model ought to know the answer.',
        'If a response requires prompting/clarification, reject under this frozen single-response protocol, and mark the rationale.',
        'Record ambiguity explicitly; do not silently force an uncertain binary grade.',
        'Return case_id -> {grade: accepted|rejected, reviewer, rationale}. A common grade applies to all displayed prefixes.',
        'If an answerline timing condition changes the grade, supply prefix_decisions keyed by prefix_key instead of a common grade.',
        'For sensitivity analysis, preserve a separately justified alternative decision file; do not change frozen prompts or policies.'
    ], 'masked_fields': ['model', 'split', 'confidence', 'automatic_grade'], 'cases': ordered}
    write(args.out / 'review_packet.json', review)
    write(args.out / 'private_mapping.json', {'inputs': inputs, 'mapping': mapping, 'summaries': summaries,
        'review_packet_sha256': sha(args.out / 'review_packet.json')})
    print(json.dumps({'n_unique_question_answer_cases': len(cases), 'summaries': summaries}, indent=2))



def control_interval(values, *, samples, seed):
    # Each control arm contains one row per question; resample whole groups.
    groups = defaultdict(list)
    for row in values:
        groups[row['group_id']].append(row['value'])
    if len(groups) < 2 or not samples:
        return None
    clusters = [groups[key] for key in sorted(groups)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample = [value for group in rng.choices(clusters, k=len(clusters)) for value in group]
        estimates.append(statistics.mean(sample))
    estimates.sort()
    def percentile(p):
        index = p * (len(estimates) - 1)
        lo = int(index)
        hi = min(lo + 1, len(estimates) - 1)
        return estimates[lo] + (index - lo) * (estimates[hi] - estimates[lo])
    return [percentile(.025), percentile(.975)]


def control_summary(rows, args):
    n = len(rows)
    return {'n_questions': n, 'n_groups': len({row['group_id'] for row in rows}),
        'n_correct': sum(row['correct'] for row in rows),
        'accuracy': sum(row['correct'] for row in rows) / n if n else None,
        'ci95': control_interval([{'group_id': row['group_id'], 'value': int(row['correct'])} for row in rows],
                         samples=args.bootstrap_samples, seed=args.seed),
        'status_counts': dict(Counter(row['status'] for row in rows)),
        'answer_id_counts': dict(Counter(row['answer'] for row in rows if row['status'] == 'answer')),
        'gold_id_counts': dict(Counter(row['gold'] for row in rows)),
        'raw_confidence_gt_0_99': {
            'n_answers': sum(row['status'] == 'answer' and row['confidence'] > .99 for row in rows),
            'n_errors': sum(row['status'] == 'answer' and row['confidence'] > .99 and not row['correct'] for row in rows)}}


def analyze_controls(args):
    sys.path.insert(0, str(args.repo.resolve()))
    from qb_data.jane_paired import build_jobs, validate_dataset
    from scripts.jane_gpu_backend import validate_package
    from scripts.run_jane_paired import _validate_backend_predictions
    dataset, package, trace = load(args.dataset), load(args.jobs), load(args.trace)
    validate_dataset(dataset)
    if package['schema_version'] != 'jane-choice-controls-v1':
        raise ValueError('expected choices-only package')
    if package['evidence_scope'] != dataset['evidence_scope']:
        raise ValueError('public control and dataset scopes disagree')
    jobs = validate_package(package, max_jobs=10000)
    if trace.get('schema_version') != 'jane-choice-control-traces-v1':
        raise ValueError('expected jane-choice-control-traces-v1; main trajectory traces are not control traces')
    if trace['metadata'].get('evidence_scope') != dataset['evidence_scope']:
        raise ValueError('evidence scopes disagree')
    _validate_backend_predictions(jobs, trace['predictions'])
    predictions = {prediction['job_id']: prediction for prediction in trace['predictions']}
    questions = {question['qid']: question for question in dataset['questions']}
    expected = {(q['qid'], m['condition'], m['menu_id']) for q in dataset['questions'] for m in q['menus']}
    observed, rows = set(), []
    for job in jobs:
        key = (job['qid'], job['condition'], job['menu_id'])
        if key not in expected or key in observed:
            raise ValueError('unexpected or duplicated control')
        observed.add(key)
        question = questions[job['qid']]
        menu = next(menu for menu in question['menus'] if (menu['condition'], menu['menu_id']) == key[1:])
        if job['options'] != menu['options'] or job['split'] != question['split'] or job['group_id'] != question['group_id']:
            raise ValueError('control metadata differs from frozen dataset')
        p = predictions[job['job_id']]
        legal_answer = p['answer'] in {option['id'] for option in menu['options']}
        status = p['status'] if p['status'] != 'answer' or legal_answer else 'invalid'
        rows.append({'job_id': job['job_id'], 'qid': job['qid'], 'group_id': job['group_id'],
            'split': job['split'], 'arm': f"mc:{job['condition']}:{job['menu_id']}",
            'status': status, 'answer': p['answer'], 'confidence': p['confidence'],
            'gold': menu['gold_option_id'], 'correct': status == 'answer' and p['answer'] == menu['gold_option_id']})
    if observed != expected:
        raise ValueError('missing frozen choices-only jobs')
    strata = {}
    for split in ('calibration', 'selection', 'test', 'all'):
        selected = [row for row in rows if split == 'all' or row['split'] == split]
        strata[split] = {arm: control_summary([row for row in selected if row['arm'] == arm], args)
                         for arm in sorted({row['arm'] for row in selected})}
    comparisons = []
    if args.graded is not None:
        graded = load(args.graded)
        indexed = defaultdict(list)
        canonical_mc = {job['job_id']: job for job in build_jobs(dataset) if job['format'] == 'mc'}
        seen_mc = set()
        for row in graded:
            if row['format'] == 'mc':
                if row['job_id'] not in canonical_mc or row['job_id'] in seen_mc:
                    raise ValueError('unknown or duplicate graded MC trajectory job')
                seen_mc.add(row['job_id'])
                frozen = canonical_mc[row['job_id']]
                if any(row[key] != frozen[key] for key in ('qid', 'group_id', 'split', 'format',
                    'condition', 'menu_id', 'prefix_id', 'fraction', 'prompt_sha256')):
                    raise ValueError('graded MC comparator differs from frozen public job')
                menu = next(menu for menu in questions[row['qid']]['menus']
                    if (menu['condition'], menu['menu_id']) == (row['condition'], row['menu_id']))
                expected_correct = row['status'] == 'answer' and row['answer'] == menu['gold_option_id']
                if type(row['correct']) is not bool or row['correct'] != expected_correct:
                    raise ValueError('graded MC comparator correctness differs from frozen menu')
                indexed[(row['qid'], f"mc:{row['condition']}:{row['menu_id']}")].append(row)
        if seen_mc != set(canonical_mc):
            raise ValueError('graded MC comparators do not cover complete frozen trajectories')
        for arm in sorted(strata['test']):
            arm_rows = [row for row in rows if row['split'] == 'test' and row['arm'] == arm]
            for label, pick in (('first_available_prefix', min), ('full_question', max)):
                differences = []
                fractions = []
                for row in arm_rows:
                    trajectory = indexed[(row['qid'], arm)]
                    if not trajectory:
                        raise ValueError('missing trajectory for paired control comparison')
                    point = pick(trajectory, key=lambda r: r['fraction'])
                    if label == 'full_question' and point['fraction'] != 1:
                        raise ValueError('full-question comparator missing final prefix')
                    if point['grade'] not in ('accepted', 'rejected', 'invalid', 'abstain'):
                        raise ValueError('unresolved paired MC comparator')
                    differences.append({'group_id': row['group_id'], 'value': int(point['correct']) - int(row['correct'])})
                    fractions.append(point['fraction'])
                comparisons.append({'arm': arm, 'split': 'test', 'contrast': label + ' minus choices_only',
                    'n_questions': len(differences), 'accuracy_difference': statistics.mean(v['value'] for v in differences),
                    'ci95': control_interval(differences, samples=args.bootstrap_samples, seed=args.seed),
                    'prefix_fraction_min': min(fractions), 'prefix_fraction_max': max(fractions),
                    'prefix_fraction_mean': statistics.mean(fractions)})
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = {str(p.resolve()): sha(p) for p in (args.dataset, args.jobs, args.trace, args.graded) if p is not None}
    report = {'schema': 'jane-choices-only-analysis-v1', 'trace_metadata': trace['metadata'],
        'inputs_sha256': inputs, 'script_sha256': sha(__file__), 'bootstrap': {'samples': args.bootstrap_samples,
        'seed': args.seed, 'method': 'whole-group percentile bootstrap; identical seeded draws per matched question/group set'},
        'uniform_random_guess_reference': .25, 'strata': strata, 'paired_trajectory_comparisons': comparisons,
        'limitations': ['Choices-only is a separate prompt and contains no question clues.',
            'A 25% reference describes uniform guessing over four IDs; the model is not assumed to guess uniformly.',
            'An above-chance score can reflect menu construction or answer-prior biases; no causal mechanism is established.',
            'Unreviewed menu equivalence and question filtering limit interpretation.',
            'No interval adjusts for multiple models, arms, or comparisons.']}
    write(args.out / 'graded_controls.json', rows)
    write(args.out / 'report.json', report)
    write(args.out / 'completion_receipt.json', {'status': 'complete', 'inputs_sha256': inputs,
        'outputs_sha256': {name: sha(args.out / name) for name in ('graded_controls.json', 'report.json')}})
    print(json.dumps({'test': strata['test'], 'out': str(args.out)}, indent=2))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-dir', type=Path, required=True, help='Extracted final frozen dataset directory; never modified')
    parser.add_argument('--contract', type=Path, default=DEFAULT_CONTRACT)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight')
    c = sub.add_parser('controls')
    c.add_argument('--trace', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--graded', type=Path)
    r = sub.add_parser('run')
    r.add_argument('--trace', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--adjudications', type=Path)
    r.add_argument('--exact-match-proxy', action='store_true')
    r.add_argument('--make-review-packet', action='store_true')
    run(parser.parse_args())

if __name__ == '__main__':
    main()
