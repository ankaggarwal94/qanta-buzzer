"""Counterexamples for fixed-plan CPU answer/stopping factorization."""
import copy
import json
from pathlib import Path

import pytest

from scripts import analyze_imcqa_factorized_cpu as a


SETTINGS = {"feature_clip": [1e-6, 1-1e-6], "l2": .01, "maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8}


def states(source="plain", *, qid="q", split="calibration", correct=True, confidence=.7, rotations=(0, 1, 2, 3)):
    result = []
    for rotation in rotations:
        for round_number in range(1, 6):
            candidate = "A" if correct else "B"
            p = {label: (confidence if label == candidate else (1-confidence)/3) for label in "ABCD"}
            result.append({"qid": qid, "group_id": qid, "split": split, "condition": "independent_pool", "round": round_number,
                "fraction": round_number/5, "rotation": rotation, "answer_source": source, "candidate_choice": candidate,
                "candidate_correct": correct, "canonical_gold_option_id": "A", "candidate_confidence": confidence,
                "candidate_probabilities": p, "native_commit": source == "wait" and round_number >= 3})
    return result


def simple_fit(source="plain", probability=.5):
    from scipy.special import logit
    return {"answer_source": source, "condition": "independent_pool", "intercept": float(logit(probability)), "slope": 0.,
            "feature_clip": [1e-6, 1-1e-6], "selected_fixed_policy": "fixed_2", "selected_threshold": .5}


def test_fit_rejects_selection_and_unbalanced_orders():
    with pytest.raises(ValueError, match="calibration data only"):
        a.fit_calibrator(states(split="selection"), SETTINGS)
    with pytest.raises(ValueError, match="balance"):
        a.fit_calibrator(states()[:-1], SETTINGS)


def test_degenerate_calibration_uses_smoothed_intercept_and_zero_slope():
    fitted = a.fit_calibrator(states(correct=True), SETTINGS)
    assert fitted["slope"] == 0
    assert a.calibrated_probability(states()[0], fitted) == pytest.approx(20.5/21)
    ensemble = a.fit_calibrator(states(source=a.ENSEMBLE, rotations=(-1,)), SETTINGS)
    assert ensemble["n_fit_states"] == 5
    assert a.calibrated_probability(states()[0], ensemble) == pytest.approx(5.5/6)


def test_monotone_fit_cannot_reverse_confidence_order():
    rows = states(qid="low", correct=True, confidence=.3) + states(qid="high", correct=False, confidence=.95)
    fit = a.fit_calibrator(rows, SETTINGS)
    assert fit["slope"] >= 0
    assert a.calibrated_probability(rows[0], fit) <= a.calibrated_probability(rows[-1], fit)


def test_selection_ties_prefer_pass_then_higher_threshold_and_earlier_fixed_round():
    wrong = list(a.trajectories(states(correct=False)).values())
    selected = a.select_calibration_policies(wrong, simple_fit(), [0., .5, 1.])
    assert selected["selected_fixed_policy"] == "pass"
    assert selected["selected_threshold"] is None
    right = list(a.trajectories(states(correct=True)).values())
    selected = a.select_calibration_policies(right, simple_fit(), [0., .5, 1.])
    assert selected["selected_fixed_policy"] == "fixed_1"
    assert selected["selected_threshold"] == .5
    right[0][0]["split"] = "selection"
    with pytest.raises(ValueError, match="calibration trajectories only"):
        a.select_calibration_policies(right, simple_fit(), [0., .5, 1.])


def test_myopic_threshold_is_strict_and_terminal_threshold_is_not_one_half():
    trajectory = states(rotations=(0,))
    assert a.stopping_round(trajectory, "myopic", simple_fit(probability=.5)) is None
    assert a.stopping_round(trajectory, "threshold", simple_fit(probability=.5), threshold=.5) == 1
    assert a.stopping_round(trajectory, "myopic", simple_fit(probability=.51)) == 1
    # A confidence feature that only becomes .8 at the final round still does
    # not clear the terminal +.2/-1 threshold of 5/6.
    fit = {**simple_fit(), "intercept": 0., "slope": 1.}
    for row in trajectory:
        row["candidate_confidence"] = .8 if row["round"] == 5 else .3
    assert a.stopping_round(trajectory, "myopic", fit) is None
    trajectory[-1]["candidate_confidence"] = .9
    assert a.stopping_round(trajectory, "myopic", fit) == 5


def test_cyclic_ensemble_averages_canonical_probabilities_and_is_reindexing_invariant():
    rows = states()
    # Every display rotation already maps to the same canonical answer.
    ensemble = a.cyclic_ensemble(rows)
    assert len(ensemble) == 5
    assert all(row["candidate_choice"] == "A" and row["rotation"] == -1 for row in ensemble)
    permuted = copy.deepcopy(rows)
    for row in permuted:
        row["rotation"] = (row["rotation"] + 1) % 4
    assert a.cyclic_ensemble(list(reversed(permuted))) == ensemble
    with pytest.raises(ValueError, match="four distinct rotations"):
        a.cyclic_ensemble(rows[:-1])


def test_ensemble_uses_probability_mean_not_majority_vote():
    rows = states()
    for row in rows:
        row["candidate_probabilities"] = {"A": .34, "B": .33, "C": .17, "D": .16} if row["rotation"] != 3 else {"A": .01, "B": .97, "C": .01, "D": .01}
    assert all(row["candidate_choice"] == "B" for row in a.cyclic_ensemble(rows))


def test_crossed_policies_hold_native_stopping_round_constant():
    rows = states("plain", correct=True) + states("forced", correct=False) + states("wait", correct=False)
    fits = {(source, "independent_pool"): simple_fit(source) for source in a.ARMS}
    records, _ = a.evaluate_policies(rows, fits)
    native = [row for row in records if row["stop_method"] == "native"]
    assert {row["round"] for row in native} == {3}
    assert {row["reward"] for row in native if row["answer_source"] == "plain"} == {.6}
    assert {row["reward"] for row in native if row["answer_source"] == "wait"} == {-1.}


def test_ensemble_one_episode_vs_four_orders_is_paired_at_question_level():
    left, right = [], []
    for qid, is_correct in (("q", True), ("r", False)):
        left.append(a.outcome(states(a.ENSEMBLE, qid=qid, correct=is_correct, rotations=(-1,)), 5, "fixed", "fixed_5"))
        for trajectory in a.trajectories(states(qid=qid, correct=is_correct)).values():
            right.append(a.outcome(trajectory, 5, "fixed", "fixed_5"))
    compared = a.paired_summary(left, right, 100, 1)
    assert compared["n_questions"] == 2
    assert compared["n_left_episodes"] == 2 and compared["n_right_episodes"] == 8
    assert compared["differences"]["mean_reward"]["mean"] == 0
    assert compared["differences"]["mean_reward"]["ci95"] == [0, 0]
    assert compared["differences"]["risk"]["mean"] == 0
    left[0]["qid"] = "other"
    with pytest.raises(ValueError, match="identical questions"):
        a.paired_summary(left, right, 100, 1)


def test_full_factorization_freezes_choices_independently_of_selection_outcomes():
    plan = json.loads((Path(__file__).resolve().parents[1] / "configs/imcqa_factorized_cpu_analysis.json").read_text())
    plan["fit_questions"] = 1
    plan["evaluation"]["bootstrap_samples"] = 5
    rows = []
    for split in ("calibration", "selection"):
        for condition in a.protocol.MENUS:
            for source in a.ARMS:
                added = states(source, qid=split, split=split, correct=True)
                for row in added:
                    row["condition"] = condition
                rows.extend(added)
    rows += a.cyclic_ensemble(rows)
    fits = a.freeze_fits(rows, plan)
    altered = copy.deepcopy(rows)
    for row in altered:
        if row["split"] == "selection":
            row["candidate_correct"] = False
            row["candidate_confidence"] = .999
    assert a.freeze_fits(altered, plan) == fits
    records, _ = a.evaluate_policies(rows, fits)
    report = a.summarize(records, rows, fits, plan)
    primary = [row for row in report["paired_contrasts"] if row["tier"] == "primary"]
    assert len(primary) == 12
    assert all(row["identical_stopping_schedule"] for row in primary)
    assert all(row["differences"]["mean_reward"]["mean"] == 0 for row in primary)
    ensemble = [row for row in report["paired_contrasts"] if row["tier"] == "ensemble_complete_pipeline"]
    assert all(row["n_left_episodes"] == 1 and row["n_right_episodes"] == 4 and row["n_questions"] == 1 for row in ensemble)
