#!/usr/bin/env python3
"""Independently replay crossed settings from raw scores, without main-evaluator imports.

Uses the earlier independent raw-score reader, retains each menu's original
calibration map, and changes only threshold and fixed round. All uncertainty
resamples the same 100 question IDs after averaging their four rotations.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from scripts import audit_imcqa_fixed_abstention as independent

PLAN_SHA = 'd678b4bc4a28bb9570d77e1ad79f388d5ad7c7b9080a2b3d1b801a8b5a23ebd1'
MENUS = independent.MENUS
POLICIES = independent.POLICIES
REWARDS = independent.REWARD
OUTCOME_FIELDS = ('round', 'committed', 'correct', 'wrong', 'terminal_pass',
                  'canonical_choice', 'reward', 'observed_round')


def interval(values, quantiles=(.025, .975)):
    finite = values[np.isfinite(values)]
    return np.quantile(finite, quantiles).tolist() if len(finite) else None


def stats(values, indices):
    """Compute question means and pooled conditional error per paired resample."""
    result = {}
    boot = {}
    for metric, field in (('mean_reward', 'reward'), ('coverage', 'committed'),
                          ('mean_observed_round', 'observed_round')):
        boot[metric] = values[field][indices].mean(axis=1)
        result[metric] = {'mean': float(values[field].mean()), 'ci95': interval(boot[metric]),
                          'defined_resamples': len(indices), 'total_resamples': len(indices)}
    denominator = values['committed'][indices].sum(axis=1)
    boot['conditional_error'] = np.divide(values['wrong'][indices].sum(axis=1), denominator,
        out=np.full(len(indices), np.nan), where=denominator > 0)
    result['conditional_error'] = {
        'mean': float(values['wrong'].sum() / values['committed'].sum()) if values['committed'].sum() else None,
        'ci95': interval(boot['conditional_error']),
        'defined_resamples': int(np.isfinite(boot['conditional_error']).sum()),
        'total_resamples': len(indices)}
    return result, boot


def reconstruct(root, prior_plan, prior_analysis, plan_path):
    """Recompute candidates/calibration independently before crossing the settings."""
    independent.ensure(independent.sha(plan_path) == PLAN_SHA, 'crossed plan changed')
    plan = independent.read(plan_path)
    old, old_episodes, _, _, _, trajectories = independent.reconstruct(root, prior_plan)
    settings = plan['settings']
    independent.ensure(len(settings) == 4 and {(s['threshold'], s['fixed_round']) for s in settings}
                       == set(itertools.product((.6, .85), (1, 2))), 'complete theta/round crossing')
    qids = sorted({key[0] for key in trajectories})
    independent.ensure(len(qids) == 100, '100 independent questions required')
    episodes, grouped = {}, collections.defaultdict(list)
    for (qid, menu, rotation), trajectory in sorted(trajectories.items()):
        probs = [trajectory[r]['calibrated_probability'] for r in range(1, 6)]
        for setting in settings:
            for policy in POLICIES:
                stop = independent.choice_round(probs, setting['threshold'], setting['fixed_round'], policy)
                committed = stop is not None
                correct = bool(committed and trajectory[stop]['correct'])
                row = {'qid': qid, 'condition': menu, 'rotation': rotation,
                    'setting_id': setting['setting_id'], 'threshold': setting['threshold'],
                    'fixed_round': setting['fixed_round'], 'policy': policy, 'round': stop,
                    'committed': committed, 'correct': correct, 'wrong': committed and not correct,
                    'terminal_pass': not committed, 'canonical_choice': trajectory[stop]['choice'] if committed else None,
                    'reward': (REWARDS[stop-1] if correct else -1.) if committed else 0.,
                    'observed_round': stop if committed else 5}
                key = (qid, menu, setting['setting_id'], rotation, policy)
                independent.ensure(key not in episodes, 'duplicate crossed episode')
                episodes[key] = row
                grouped[qid, menu, setting['setting_id'], policy].append(row)
    independent.ensure(len(episodes) == 12800, 'crossed episode count')
    original_settings = {'independent_pool': 't060_r1', 'same_category_pool': 't085_r2'}
    matched = 0
    for key, prior in old_episodes.items():
        qid, menu, rotation, policy = key
        current = episodes[qid, menu, original_settings[menu], rotation, policy]
        independent.ensure(all(current[k] == prior[k] for k in OUTCOME_FIELDS), 'old independent episode changed')
        matched += 1
    prior_csv = list(csv.DictReader((prior_analysis/'episodes.csv').open()))
    independent.ensure(len(prior_csv) == 3200, 'prior main episode count')
    seen = set()
    for row in prior_csv:
        key = (row['qid'], row['condition'], original_settings[row['condition']], int(row['rotation']), row['policy'])
        independent.ensure(key not in seen, 'duplicate prior episode')
        seen.add(key)
        for field in OUTCOME_FIELDS:
            value = episodes[key][field]
            independent.ensure(row[field] == ('' if value is None else str(value)), 'prior published episode changed '+field)
    adaptive_checks = forced_checks = nesting_checks = 0
    for qid, menu, rotation in itertools.product(qids, MENUS, range(4)):
        for threshold in ('060', '085'):
            for policy in ('adaptive_forced', 'adaptive_selective'):
                left = episodes[qid, menu, f't{threshold}_r1', rotation, policy]
                right = episodes[qid, menu, f't{threshold}_r2', rotation, policy]
                independent.ensure(all(left[k] == right[k] for k in OUTCOME_FIELDS), 'adaptive fixed-round invariance')
                adaptive_checks += 1
        for fixed in (1, 2):
            left = episodes[qid, menu, f't060_r{fixed}', rotation, 'fixed_forced']
            right = episodes[qid, menu, f't085_r{fixed}', rotation, 'fixed_forced']
            independent.ensure(all(left[k] == right[k] for k in OUTCOME_FIELDS), 'fixed-forced threshold invariance')
            forced_checks += 1
            low = episodes[qid, menu, f't060_r{fixed}', rotation, 'fixed_selective']
            high = episodes[qid, menu, f't085_r{fixed}', rotation, 'fixed_selective']
            independent.ensure(not high['committed'] or low['committed'], 'fixed-selective threshold nesting')
            nesting_checks += 1
    means = {}
    for key, rows in grouped.items():
        independent.ensure(sorted(r['rotation'] for r in rows) == [0, 1, 2, 3], 'four rotations per question')
        means[key] = {field: float(np.mean([r[field] for r in rows])) for field in
            ('reward', 'committed', 'correct', 'wrong', 'terminal_pass', 'observed_round')}
    indices = np.random.default_rng(1).integers(0, 100, (20000, 100))
    arrays, summaries, boots = {}, [], {}
    for menu, setting, policy in itertools.product(MENUS, settings, POLICIES):
        sid = setting['setting_id']
        key = (menu, sid, policy)
        values = {field: np.array([means[qid, menu, sid, policy][field] for qid in qids]) for field in
            ('reward', 'committed', 'correct', 'wrong', 'terminal_pass', 'observed_round')}
        arrays[key] = values
        estimates, boot = stats(values, indices)
        boots[key] = boot
        summaries.append({'condition': menu, 'setting_id': sid, 'threshold': setting['threshold'],
            'fixed_round': setting['fixed_round'], 'policy': policy, 'n_questions': 100, 'n_episodes': 400,
            **{field+'_count': int(round(values[field].sum()*4)) for field in ('committed', 'correct', 'wrong', 'terminal_pass')},
            **estimates})
    focal, gains, menu_differences, menu_metrics = [], [], [], []
    gain_values = {}
    for menu, setting in itertools.product(MENUS, settings):
        sid = setting['setting_id']
        values = arrays[menu, sid, 'adaptive_selective']['reward']-arrays[menu, sid, 'fixed_selective']['reward']
        gain_values[menu, sid] = values
        resampled = values[indices].mean(axis=1)
        focal.append({'condition': menu, 'setting_id': sid, 'left': 'adaptive_selective', 'right': 'fixed_selective',
            'mean_delta': float(values.mean()), 'ci95': interval(resampled),
            'ci99_375': interval(resampled, (.003125, .996875)), 'family_size': 8})
    for setting in settings:
        sid = setting['setting_id']
        values = gain_values['same_category_pool', sid] - gain_values['independent_pool', sid]
        gains.append({'setting_id': sid, 'mean_delta': float(values.mean()), 'ci95': interval(values[indices].mean(axis=1))})
        for policy in POLICIES:
            values = arrays['same_category_pool', sid, policy]['reward'] - arrays['independent_pool', sid, policy]['reward']
            menu_differences.append({'setting_id': sid, 'policy': policy,
                'mean_delta': float(values.mean()), 'ci95': interval(values[indices].mean(axis=1))})
            for metric, field in (('coverage', 'committed'), ('conditional_error', None),
                                  ('mean_observed_round', 'observed_round')):
                lhs = arrays['same_category_pool', sid, policy]
                rhs = arrays['independent_pool', sid, policy]
                if field is None:
                    left = lhs['wrong'].sum()/lhs['committed'].sum() if lhs['committed'].sum() else None
                    right = rhs['wrong'].sum()/rhs['committed'].sum() if rhs['committed'].sum() else None
                    point = float(left-right) if left is not None and right is not None else None
                else:
                    point = float((lhs[field]-rhs[field]).mean())
                resampled = boots['same_category_pool', sid, policy][metric]-boots['independent_pool', sid, policy][metric]
                menu_metrics.append({'setting_id': sid, 'policy': policy, 'metric': metric,
                    'mean_delta': point, 'ci95': interval(resampled),
                    'defined_resamples': int(np.isfinite(resampled).sum()), 'total_resamples': len(indices)})
    result = {'schema_version': 'imcqa-crossed-settings-independent-audit-v1', 'status': 'passed',
        'plan_sha256': PLAN_SHA, 'fitting_performed': False, 'model_inference_performed': False,
        'imports_main_evaluator': False, 'manifest_members_verified': old['manifest_members_verified'],
        'plain_states_reconstructed': 4000, 'episodes_reconstructed': len(episodes),
        'question_cells_reconstructed': len(means), 'n_questions': 100, 'original_episodes_unchanged': matched,
        'old_published_episodes_checked': len(prior_csv), 'adaptive_invariance_checks': adaptive_checks,
        'fixed_forced_invariance_checks': forced_checks, 'fixed_selective_nesting_checks': nesting_checks,
        'edge_checks_reused': old['edge_tests_passed'], 'bootstrap_samples': 20000, 'bootstrap_seed': 1,
        'policy_summaries': summaries, 'primary_contrasts': focal,
        'menu_gain_differences': gains, 'menu_policy_differences': menu_differences,
        'menu_metric_differences': menu_metrics,
        'upstream_independent_auditor_sha256': independent.sha(Path(independent.__file__))}
    return result, episodes, means, arrays, boots, indices


def nearly_equal(left, right, name):
    if left is None or right is None:
        independent.ensure(left is None and right is None, name)
    else:
        independent.ensure(np.max(np.abs(np.asarray(left)-np.asarray(right))) < 1e-12, name)


def compare(result, episodes, means, analysis):
    """Compare raw-derived outcomes and question-bootstrap estimates to main output."""
    summary = independent.read(analysis/'summary.json')
    rows = list(csv.DictReader((analysis/'episodes.csv').open()))
    seen = set()
    for row in rows:
        key = (row['qid'], row['condition'], row['setting_id'], int(row['rotation']), row['policy'])
        independent.ensure(key in episodes and key not in seen, 'main episode coverage')
        seen.add(key)
        for field in OUTCOME_FIELDS:
            value = episodes[key][field]
            independent.ensure(row[field] == ('' if value is None else str(value)), 'main episode '+field)
    independent.ensure(seen == set(episodes), 'missing main episodes')
    pq = list(csv.DictReader((analysis/'per_question.csv').open()))
    seen = set()
    for row in pq:
        key = (row['qid'], row['condition'], row['setting_id'], row['policy'])
        independent.ensure(key in means and key not in seen, 'main per-question coverage')
        seen.add(key)
        for field, value in means[key].items():
            nearly_equal(float(row[field]), value, 'main per-question '+field)
    independent.ensure(seen == set(means), 'missing main question cells')
    own = {(r['condition'], r['setting_id'], r['policy']): r for r in result['policy_summaries']}
    independent.ensure(len(summary['policy_summaries']) == 32, 'main 32 policy cells')
    independent.ensure({(r['condition'], r['setting_id'], r['policy']) for r in summary['policy_summaries']} == set(own),
                       'main summary complete unique cell coverage')
    for row in summary['policy_summaries']:
        expected = own[row['condition'], row['setting_id'], row['policy']]
        for metric in ('mean_reward', 'coverage', 'conditional_error', 'mean_observed_round'):
            nearly_equal(row[metric]['mean'], expected[metric]['mean'], 'main metric point '+metric)
            nearly_equal(row[metric]['ci95'], expected[metric]['ci95'], 'main metric interval '+metric)
            independent.ensure(row[metric]['defined_resamples'] == expected[metric]['defined_resamples'], 'main defined resamples')
            independent.ensure(row[metric]['total_resamples'] == expected[metric]['total_resamples'], 'main total resamples')
        for field in ('committed_count', 'correct_count', 'wrong_count', 'terminal_pass_count',
                      'n_questions', 'n_episodes', 'threshold', 'fixed_round'):
            independent.ensure(row[field] == expected[field], 'main summary '+field)
    for field, keys, intervals, count in (
        ('primary_contrasts', ('condition', 'setting_id'), ('ci95', 'ci99_375'), 8),
        ('menu_gain_differences', ('setting_id',), ('ci95',), 4),
        ('menu_policy_differences', ('setting_id', 'policy'), ('ci95',), 16),
        ('menu_metric_differences', ('setting_id', 'policy', 'metric'), ('ci95',), 48)):
        expected = {tuple(r[k] for k in keys): r for r in result[field]}
        independent.ensure(len(summary[field]) == count, 'main contrast count '+field)
        independent.ensure({tuple(r[k] for k in keys) for r in summary[field]} == set(expected),
                           'main contrast complete unique coverage '+field)
        for row in summary[field]:
            ownrow = expected[tuple(row[k] for k in keys)]
            nearly_equal(row['mean_delta'], ownrow['mean_delta'], 'main contrast point '+field)
            for ci in intervals:
                nearly_equal(row[ci], ownrow[ci], 'main contrast interval '+field)
            independent.ensure(row['total_resamples'] == 20000, 'main contrast total resamples')
            independent.ensure(row['defined_resamples'] == ownrow.get('defined_resamples', 20000),
                               'main contrast defined resamples')
            independent.ensure(row['n_questions'] == 100, 'main contrast question count')
            if field == 'primary_contrasts':
                independent.ensure(row['family_size'] == 8 and row['left'] == 'adaptive_selective'
                    and row['right'] == 'fixed_selective', 'main focal contrast definition')
    result['main_comparison'] = {'status': 'passed', 'episode_rows_checked': len(rows),
        'question_rows_checked': len(pq), 'policy_metric_cells_checked': 32*4,
        'focal_contrasts_checked': 8, 'paired_menu_gain_contrasts_checked': 4,
        'paired_menu_policy_contrasts_checked': 16,
        'paired_menu_metric_contrasts_checked': 48,
        'summary_sha256': independent.sha(analysis/'summary.json'),
        'episodes_sha256': independent.sha(analysis/'episodes.csv'),
        'per_question_sha256': independent.sha(analysis/'per_question.csv')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'prior-plan', 'prior-analysis', 'plan', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--analysis', type=Path)
    args = parser.parse_args()
    result, episodes, means, arrays, boots, indices = reconstruct(args.root, args.prior_plan, args.prior_analysis, args.plan)
    if args.analysis:
        compare(result, episodes, means, args.analysis)
    result['auditor_sha256'] = independent.sha(__file__)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('policy_summaries',)}, indent=2))


if __name__ == '__main__':
    main()
