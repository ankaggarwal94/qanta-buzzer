"""Synthetic fixtures for retrospective analysis, never model-run evidence."""
import copy

import numpy as np
import pytest

from scripts import analyze_imcqa_retrospective as a


def trajectory(qid="q", split="test", statuses=None, correct=None, confidence=None):
    statuses = statuses or ["answer"] * 10
    correct = correct or [True] * 10
    confidence = confidence or [.8] * 10
    return [dict(qid=qid, group_id=qid, split=split, round=i + 1,
                 fraction=(i + 1) / 10, status=statuses[i], correct=correct[i] if statuses[i] == "answer" else False,
                 confidence=confidence[i] if statuses[i] == "answer" else None,
                 job_id=f"{qid}-{i}", condition="independent_pool") for i in range(10)]


def test_linear_reward_endpoints_and_abstention():
    rows = trajectory()
    rec = a.policy_records({"q": rows}, {"kind": "first_answer"}, None)[0]
    assert a.reward_values([rec], "early_wrong1").tolist() == [1.0]
    rec = a.policy_records({"q": rows}, {"kind": "fixed_round", "round": 10}, None)[0]
    assert a.reward_values([rec], "early_wrong1").tolist() == [.1]
    rec["correct"] = False
    assert a.reward_values([rec], "early_wrong025").tolist() == [-.25]
    rec["committed"] = False
    assert a.reward_values([rec], "early_wrong1").tolist() == [0.0]


def test_first_answer_stops_on_wrong_answer_and_does_not_peek():
    rows = trajectory(statuses=["abstain", "answer"] + ["answer"] * 8,
                      correct=[False, False] + [True] * 8)
    rec = a.policy_records({"q": rows}, {"kind": "first_answer"}, None)[0]
    assert rec["round"] == 2
    assert rec["correct"] is False
    assert a.reward_values([rec], "early_wrong1")[0] == -1


def test_fixed_round_does_not_advance_after_abstention():
    rows = trajectory(statuses=["abstain"] + ["answer"] * 9)
    rec = a.policy_records({"q": rows}, {"kind": "fixed_round", "round": 1}, None)[0]
    assert rec["committed"] is False


def test_threshold_uses_calibration_and_observation_not_future_correctness():
    rows = trajectory(correct=[False] + [True] * 9)
    cal = {"x": [0, 1], "y": [0, 1]}
    rec = a.policy_records({"q": rows}, {"kind": "threshold", "value": .7}, cal)[0]
    assert rec["round"] == 1 and rec["correct"] is False


def test_never_keeps_risk_and_position_undefined_in_bootstrap():
    recs = a.policy_records({"q": trajectory()}, {"kind": "never"}, None)
    result = a.summarize_policy(recs, np.zeros((10, 1), dtype=int))
    assert result["coverage"] == 0
    assert result["risk"] is None
    assert result["mean_round_when_committed"] is None
    assert result["bootstrap"]["risk"]["ci95"] is None
    assert result["bootstrap"]["risk"]["defined_resamples"] == 0
    assert result["risk_clopper_pearson95"] is None


def test_zero_error_small_sample_interval_does_not_degenerate():
    recs = a.policy_records({str(i): trajectory(str(i)) for i in range(7)}, {"kind": "first_answer"}, None)
    result = a.summarize_policy(recs)
    assert result["risk"] == 0
    lower, upper = result["risk_clopper_pearson95"]
    assert lower == 0 and upper == pytest.approx(1 - .025**(1/7))


def test_historical_policy_uses_its_original_frozen_calibrator():
    policy = {"kind": "threshold", "value": .9,
              "frozen_historical_calibrator": {"x": [0, 1], "y": [1, 1]}}
    rec = a.policy_records({"q": trajectory()}, policy, {"x": [0, 1], "y": [0, 0]})[0]
    assert rec["committed"] is True


def test_selection_refuses_test_trajectories():
    with pytest.raises(ValueError, match="selection"):
        a.select_policies({"q": trajectory()}, {"x": [0, 1], "y": [0, 1]})


def test_selection_tie_can_choose_never_and_cannot_win_using_future_answer():
    rows = trajectory(split="selection", correct=[False] + [True] * 9,
                      confidence=[.8] * 10)
    selected = a.select_policies({"q": rows}, {"x": [0, 1], "y": [0, 1]})
    assert selected["confidence_reward_selected"]["kind"] == "never"
    assert selected["fixed_round_selected"]["round"] == 2


def test_reversal_distinguishes_wrong_answers_and_abstentions():
    rows = trajectory(statuses=["answer", "abstain", "answer"] + ["answer"] * 7,
                      correct=[True, False, False] + [True] * 7)
    result = a.trajectory_diagnostics({"q": rows})
    assert result["questions_correct_then_later_wrong_answer"] == 1
    assert result["adjacent_correct_to_wrong_answer"] == 0
    assert result["adjacent_correct_to_noncorrect"] == 1


def test_trajectory_validation_rejects_missing_rounds_and_repeated_group():
    with pytest.raises(ValueError, match="round"):
        a.validate_trajectories({"q": trajectory()[:-1]})
    rows = trajectory("r")
    for row in rows:
        row["group_id"] = "q"
    with pytest.raises(ValueError, match="group"):
        a.validate_trajectories({"q": trajectory(), "r": rows})


def test_paired_bootstrap_zero_for_identical_records():
    recs = a.policy_records({"q": trajectory(), "r": trajectory("r")}, {"kind": "first_answer"}, None)
    idx = np.array([[0, 1], [1, 1], [0, 0]])
    diff = a.paired_difference(recs, copy.deepcopy(recs), idx)
    assert diff["reward_early_wrong1_difference"]["mean"] == 0
    assert diff["reward_early_wrong1_difference"]["ci95"] == [0, 0]
    bad = copy.deepcopy(recs)
    bad[0]["qid"] = "x"
    with pytest.raises(ValueError, match="paired"):
        a.paired_difference(recs, bad, idx)
