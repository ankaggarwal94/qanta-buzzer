"""Independent semantic and statistical checks for the locked fresh evaluator."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import analyze_imcqa_tuned as analyzer


def policy_lock():
    policies = []
    for menu, fs_round, fs_threshold, as_threshold in (
            (analyzer.MENUS[0], 1, .6, .8), (analyzer.MENUS[1], 4, .85, .6)):
        policies += [
            {"condition": menu, "family": "fixed_selective", "always_pass": False,
             "fixed_round": fs_round, "threshold": fs_threshold},
            {"condition": menu, "family": "adaptive_selective", "always_pass": False,
             "fixed_round": None, "threshold": as_threshold}]
    return {"schema_version": "imcqa-independently-tuned-policies-v1", "status": "frozen_for_fresh_evaluation",
        "policies": policies, "selection_qids": ["development"], "calibration_qids": ["calibration"],
        "selection_group_ids": ["development-group"], "calibration_group_ids": ["calibration-group"],
        "calibrators": {menu: {"intercept": 0., "slope": 1., "feature_clip": [1e-6, 1 - 1e-6],
            # Deliberately different historical choices must not affect new rules.
            "selected_threshold": .95, "selected_fixed_policy": "fixed_5"} for menu in analyzer.MENUS}}


def candidate_views(n=4):
    result = []
    for q in range(n):
        for menu in analyzer.MENUS:
            for rotation in range(4):
                for rnd, confidence in enumerate([.65, .45, .8, .9, .95], 1):
                    correct = (q + rotation + rnd) % 3 == 0
                    result.append({"qid": f"q{q}", "group_id": f"group{q}", "condition": menu,
                        "rotation": rotation, "round": rnd, "arm": "plain",
                        "canonical_answer_probabilities": {"A": confidence,
                            **{letter: (1 - confidence) / 3 for letter in "BCD"}},
                        "candidate_choice": "A", "candidate_correct": correct,
                        "score_id": f"q{q}|{menu}|{rotation}|{rnd}"})
    return result


def test_four_policies_use_independent_selected_thresholds():
    episodes = analyzer.build_episodes(candidate_views(), policy_lock())
    assert len(episodes) == 4 * 2 * 4 * 4
    expected = {analyzer.MENUS[0]: {"fixed_forced": 1, "fixed_selective": 1,
                                  "adaptive_forced": 3, "adaptive_selective": 3},
                analyzer.MENUS[1]: {"fixed_forced": 4, "fixed_selective": 4,
                                  "adaptive_forced": 1, "adaptive_selective": 1}}
    for row in episodes:
        assert row["round"] == expected[row["condition"]][row["policy"]]
        assert row["answer_score_id"].endswith("|" + str(row["round"]))
        expected_reward = float(analyzer.selection.REWARDS[row["round"] - 1]) if row["correct"] else -1.
        assert row["reward"] == expected_reward


def test_fixed_refusal_does_not_wait_for_later_crossing():
    lock = policy_lock()
    fs = next(p for p in lock["policies"] if p["condition"] == analyzer.MENUS[0] and p["family"] == "fixed_selective")
    fs["fixed_round"] = 2
    episodes = analyzer.build_episodes(candidate_views(), lock)
    rows = [r for r in episodes if r["condition"] == analyzer.MENUS[0]]
    assert all(r["round"] is None and r["reward"] == 0. and r["observed_round"] == 5
               for r in rows if r["policy"] == "fixed_selective")
    assert all(r["round"] == 3 for r in rows if r["policy"] == "adaptive_selective")
    assert all(r["round"] == 2 for r in rows if r["policy"] == "fixed_forced")


def test_explicit_pass_keeps_selective_pass_and_forced_final_controls():
    lock = policy_lock()
    for policy in lock["policies"]:
        policy.update(always_pass=True, threshold=None, fixed_round=None)
    episodes = analyzer.build_episodes(candidate_views(), lock)
    for row in episodes:
        if row["policy"].endswith("selective"):
            assert row["round"] is None and row["canonical_choice"] is None
            assert not row["committed"] and row["reward"] == 0.
        else:
            assert row["round"] == 5 and row["committed"]


def test_no_adaptive_crossing_uses_final_round_only_for_forced_control():
    lock = policy_lock()
    for policy in lock["policies"]:
        if policy["family"] == "adaptive_selective":
            policy["threshold"] = 1.
    episodes = analyzer.build_episodes(candidate_views(), lock)
    assert all(r["round"] is None for r in episodes if r["policy"] == "adaptive_selective")
    assert all(r["round"] == 5 for r in episodes if r["policy"] == "adaptive_forced")


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "bad_pass", "bad_threshold", "bad_round"])
def test_invalid_or_duplicate_policy_lock_rejected(mutation):
    lock = policy_lock()
    if mutation == "duplicate":
        lock["policies"].append(deepcopy(lock["policies"][0]))
    elif mutation == "missing":
        lock["policies"].pop()
    elif mutation == "bad_pass":
        lock["policies"][0]["always_pass"] = True
    elif mutation == "bad_threshold":
        lock["policies"][0]["threshold"] = .613
    else:
        lock["policies"][0]["fixed_round"] = 6
    with pytest.raises(ValueError):
        analyzer.policy_map(lock)


@pytest.mark.parametrize("role", ["calibration", "selection"])
def test_calibration_or_development_question_rejected(role):
    lock = policy_lock()
    lock[f"{role}_qids"].append("q0")
    with pytest.raises(ValueError, match="development question"):
        analyzer.build_episodes(candidate_views(), lock)


@pytest.mark.parametrize("role", ["calibration", "selection"])
def test_new_qid_cannot_reuse_calibration_or_development_group(role):
    lock = policy_lock()
    lock[f"{role}_group_ids"].append("group0")
    with pytest.raises(ValueError, match="group in fresh evaluation"):
        analyzer.build_episodes(candidate_views(), lock)


@pytest.mark.parametrize("mutation", ["missing_round", "duplicate_round", "shared_group", "changing_group", "nonplain"])
def test_incomplete_or_dependent_trajectory_rejected(mutation):
    views = candidate_views()
    if mutation == "missing_round":
        views.pop()
    elif mutation == "duplicate_round":
        views.append(deepcopy(views[0]))
    elif mutation == "shared_group":
        for row in views:
            if row["qid"] == "q1":
                row["group_id"] = "group0"
    elif mutation == "changing_group":
        views[0]["group_id"] = "different"
    else:
        views[0]["arm"] = "wait"
    with pytest.raises(ValueError):
        analyzer.build_episodes(views, policy_lock())


def test_paired_summary_intervals_average_rotations_before_resampling():
    episodes = analyzer.build_episodes(candidate_views(n=9), policy_lock())
    actual, per_question = analyzer.summarize(episodes)
    assert actual["n_questions"] == 9 and len(per_question) == 9 * 2 * 4
    ids = np.random.default_rng(1).integers(0, 9, (20000, 9))
    for menu in analyzer.MENUS:
        by_policy = {}
        for policy in analyzer.POLICIES:
            by_policy[policy] = np.array([np.mean([r["reward"] for r in episodes if
                r["qid"] == f"q{q}" and r["condition"] == menu and r["policy"] == policy]) for q in range(9)])
            expected = np.quantile(by_policy[policy][ids].mean(axis=1), [.025, .975])
            summary = next(r for r in actual["policy_summaries"] if r["condition"] == menu and r["policy"] == policy)
            assert summary["mean_reward"]["mean"] == pytest.approx(by_policy[policy].mean())
            assert summary["mean_reward"]["ci95"] == pytest.approx(expected)
        delta = by_policy["adaptive_selective"] - by_policy["fixed_selective"]
        draws = delta[ids].mean(axis=1)
        contrast = next(r for r in actual["primary_contrasts"] if r["condition"] == menu)
        assert contrast["mean_delta"] == pytest.approx(delta.mean())
        assert contrast["ci95"] == pytest.approx(np.quantile(draws, [.025, .975]))
        assert contrast["ci97_5"] == pytest.approx(np.quantile(draws, [.0125, .9875]))
        assert contrast["family_size"] == 2
        assert contrast["positive_difference_supported"] == (contrast["ci97_5"][0] > 0)
        assert contrast["worthwhile_difference_supported"] == (contrast["ci97_5"][0] > .05)
