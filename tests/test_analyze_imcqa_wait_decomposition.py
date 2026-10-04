"""Synthetic counterexamples for same-candidate stopping diagnosis."""
import copy

import pytest

from scripts import analyze_imcqa_wait_decomposition as d


def trajectory(candidates="BAAAA", actions="BEEEE", split="selection"):
    result = []
    for i, (candidate, action) in enumerate(zip(candidates, actions)):
        logits = {label: float(label == candidate) * 2 for label in "ABCD"}
        logits["E"] = 3.0 if action == "E" else -1.0
        result.append({"qid": "q", "group_id": "g", "split": split, "condition": "independent_pool", "variant": "wait",
                       "round": i + 1, "raw_action_logits": logits, "gold_option_id": "A", "top_option_id": action,
                       "canonical_choice": action if action in "ABCD" else None, "correct": action == "A", "actual_fraction": (i + 1) / 5})
    return result


def test_wrong_early_later_correct_is_counterfactual_and_additive():
    r, states, policies = d.decompose_trajectory(trajectory())
    assert r["native_outcome"] == "wrong_early"
    assert r["wrong_early_with_later_correct"]
    assert not r["visited_correct_candidate_skipped"]
    assert sum(s["visited_by_native_policy"] for s in states) == 1
    assert r["hindsight_reward"] == .8
    assert r["same_wait_regret"] == pytest.approx(1.8)
    assert r["wrong_answer_penalty_component"] == 1
    assert r["forgone_correct_reward_component"] == .8
    assert next(p for p in policies if p["policy"] == "same_wait_fixed_round_2")["correct"]


def test_correct_candidate_skipped_is_observed_and_delay_cost():
    r, _, _ = d.decompose_trajectory(trajectory("AAAAA", "EEAEE"))
    assert r["native_outcome"] == "correct_early"
    assert r["n_visited_correct_candidate_skipped"] == 2
    assert r["correct_answer_delay_component"] == pytest.approx(.4)
    assert r["wrong_answer_penalty_component"] == r["forgone_correct_reward_component"] == 0


def test_terminal_pass_all_wrong_oracle_also_passes():
    r, states, policies = d.decompose_trajectory(trajectory("BBBBB", "EEEEE"))
    assert r["native_outcome"] == "terminal_pass"
    assert r["same_wait_regret"] == 0
    assert not r["post_termination_correct_candidate"]
    assert all(s["visited_by_native_policy"] for s in states)
    assert next(p for p in policies if p["policy"] == "hindsight_same_wait_or_pass")["terminal_pass"]
    assert next(p for p in policies if p["policy"] == "same_wait_fixed_round_5")["wrong"]


def test_terminal_wrong_after_earlier_correct_counts_skipped_opportunity():
    r, _, _ = d.decompose_trajectory(trajectory("AAAAB", "EEEEB"))
    assert r["native_outcome"] == "wrong_terminal"
    assert r["visited_correct_candidate_skipped"]
    assert r["same_wait_regret"] == 2
    assert not r["wrong_early_with_later_correct"]


def test_hindsight_bounds_every_synthetic_answer_sequence():
    import itertools
    for candidates in itertools.product("AB", repeat=5):
        for stop in range(6):
            actions = ["E"] * 5
            if stop < 5:
                actions[stop] = candidates[stop]
            r, _, _ = d.decompose_trajectory(trajectory(candidates, actions))
            assert r["hindsight_reward"] >= r["native_reward"]


def test_rejects_wrong_prompt_and_unordered_trajectory():
    rows = trajectory()
    rows[0]["variant"] = "forced"
    with pytest.raises(ValueError, match="WAIT-prompt"):
        d.candidate_states(rows)
    with pytest.raises(ValueError, match="ordered"):
        d.candidate_states(list(reversed(trajectory())))


def test_calibrator_never_accepts_selection_labels():
    rows = d.decompose_trajectory(trajectory())[1]
    with pytest.raises(ValueError, match="calibration"):
        d.fit_calibrator(rows)


def test_calibrator_is_monotone_and_handles_degenerate_labels():
    rows = d.decompose_trajectory(trajectory("AAAAA", "EEEEE", "calibration"))[1]
    fit = d.fit_calibrator(rows)
    assert fit["slope"] == 0
    assert 0 < d.calibrated_probability(fit, .7) < 1
    rows[0]["candidate_correct"] = False
    fit = d.fit_calibrator(rows)
    assert fit["slope"] >= 0
    assert d.calibrated_probability(fit, .9) >= d.calibrated_probability(fit, .3)


def test_myopic_pass_threshold_uses_final_reward():
    rows = trajectory("AAAAA", "EEEEE")
    fit = {"intercept": 0., "slope": 0.}
    assert d.myopic_record(rows, fit)["terminal_pass"]  # p=.5: first EV zero, not positive.
    fit["intercept"] = 0.1
    assert d.myopic_record(rows, fit)["round"] == 1


def test_calibration_bins_include_probability_one_once():
    rows = [{"qid": "q", "candidate_correct": True, "p": 1.0}]
    result = d.calibration_metrics(rows, "p")
    assert result["brier"] == 0
    assert sum(bin_["n_states"] for bin_ in result["bins"]) == 1


def test_historical_match_rejects_prompt_hash_before_grade():
    job = {"variant": "wait", "round": 5, "source_job_id": "x", "source_prompt_sha256": "good", "gold_option_id": "A",
           "qid": "q", "group_id": "g", "split": "selection", "condition": "independent_pool", "menu_id": "m", "prefix_id": "p10", "fraction": 1.0}
    row = {**job, "job_id": "x", "prompt_sha256": "bad", "format": "mc"}
    with pytest.raises(ValueError, match="prompt hash"):
        d.match_historical([job], [row], [])


def test_historical_match_rejects_duplicate_identity():
    job = {"variant": "wait", "round": 5, "source_job_id": "x"}
    with pytest.raises(ValueError, match="duplicate"):
        d.match_historical([job], [{"job_id": "x"}, {"job_id": "x"}], [])
