"""Counterexamples for protocol diagnosis; synthetic fixtures are not run data."""
import copy
import json

import pytest

from scripts import analyze_imcqa_protocol_pilot as a


def state(*, rotation=0, wait_label="E", arm="wait", round_number=1, logits=None):
    mapping = a.canonical_mapping(rotation, wait_label)
    allowed = "ABCDE" if arm == "wait" else "ABCD"
    raw = logits or {label: float(label == "A") for label in "ABCDE"}
    top = max(allowed, key=raw.get)
    job = {"qid": "q", "group_id": "g", "split": "selection", "condition": "independent_pool", "round": round_number,
           "rotation": rotation, "wait_label": wait_label if arm == "wait" else None,
           "arm": arm, "block": "factorial", "option_source_ids": mapping, "allowed_actions": allowed,
           "canonical_gold_option_id": "A", "fraction": round_number / 5}
    row = {"raw_action_logits": raw, "action_probabilities": a.old.softmax(raw, tuple(allowed)),
           "conditional_answer_probabilities": a.old.softmax(raw, tuple(label for label in "ABCDE" if label in mapping)),
           "chosen_action": top, "tied_top_actions": [label for label in allowed if raw[label] == raw[top]],
           "legal_action_vocabulary_mass": .9}
    return a.score_view(row, job)


def test_wait_a_keeps_e_candidate_in_conditional_answer_distribution():
    row = state(wait_label="A", logits={"A": 4., "B": 0., "C": 0., "D": 0., "E": 3.})
    assert row["native_semantic_action"] == "WAIT"
    assert row["candidate_choice"] == "A"
    assert row["candidate_correct"] is True
    assert row["canonical_answer_probabilities"]["A"] > .8
    assert set(row["conditional_answer_probabilities"]) == set("BCDE")


def test_equal_semantics_under_rotation_have_zero_effect():
    base = state(logits={"A": 2., "B": 0., "C": -1., "D": -2., "E": 1.})
    rotated = state(rotation=1, logits={"A": -2., "B": 2., "C": 0., "D": -1., "E": 1.})
    result = a.semantic_effect(base, rotated)
    assert result["candidate_changed"] is False
    assert result["semantic_action_changed"] is False
    assert result["semantic_total_variation"] == 0


def test_equal_semantics_under_wait_label_swap_have_zero_effect():
    base = state(logits={"A": 2., "B": 0., "C": -1., "D": -2., "E": 3.})
    swapped = state(wait_label="A", logits={"A": 3., "B": 0., "C": -1., "D": -2., "E": 2.})
    assert a.semantic_effect(base, swapped)["semantic_total_variation"] == 0
    assert a.semantic_effect(base, swapped)["wait_decision_changed"] is False


def test_raw_softmax_tampering_and_wrong_argmax_fail():
    row = state()
    job = {key: row[key] for key in ("option_source_ids", "allowed_actions", "wait_label", "round", "canonical_gold_option_id")}
    row["conditional_answer_probabilities"]["A"] = .25
    with pytest.raises(ValueError, match="softmax"):
        a.score_view(row, job)
    row = state()
    row["chosen_action"] = "B"
    with pytest.raises(ValueError, match="argmax"):
        a.score_view(row, job)


def test_native_terminal_pass_and_own_candidate_hindsight_are_distinct():
    trajectory = [state(round_number=r, logits={"A": 1., "B": 0., "C": 0., "D": 0., "E": 2.}) for r in range(1, 6)]
    native = a.policy_record(trajectory, "native_wait")
    oracle = a.policy_record(trajectory, "candidate_hindsight_or_pass")
    assert native["reward"] == 0
    assert native["terminal_pass"] is True
    assert oracle["reward"] == 1
    assert oracle["round"] == 1
    assert a.policy_record(trajectory, "candidate_fixed_5")["reward"] == .2


def test_wrong_native_commit_terminates_before_later_correct_candidate():
    trajectory = [state(round_number=r, logits={"A": float(r > 1) * 3, "B": 2., "C": 0., "D": 0., "E": 1.}) for r in range(1, 6)]
    assert a.policy_record(trajectory, "native_wait")["reward"] == -1
    assert a.policy_record(trajectory, "candidate_hindsight_or_pass")["reward"] == .8


def test_question_bootstrap_does_not_count_rotations_as_independent_questions():
    rows = [{"qid": qid, "correct": value} for qid, value in (("q", 1), ("r", 0)) for _ in range(8)]
    summary = a.question_summary(rows, ("correct",), samples=100, seed=1)
    assert summary["n_questions"] == 2
    assert summary["n_states"] == 16
    assert summary["correct"]["mean"] == .5
    assert summary["correct"]["ci95"] == [0, 1]
    with pytest.raises(ValueError, match="unbalanced"):
        a.question_summary(rows[:-1], ("correct",), samples=100, seed=1)


def test_pairing_rejects_different_question_or_rotation():
    left, right = state(arm="plain"), state(arm="forced")
    assert a.paired_candidate_effect(left, right)["candidate_accuracy_difference"] == 0
    right["rotation"] = 1
    with pytest.raises(ValueError, match="paired"):
        a.paired_candidate_effect(left, right)


def test_full_synthetic_factorial_uses_all_rotations_and_split_strata():
    rows = []
    for split in a.SPLITS:
        for condition in a.MENUS:
            for round_number in range(1, 6):
                for rotation in range(4):
                    for arm in a.ARMS:
                        row = state(rotation=rotation, arm=arm, round_number=round_number)
                        row.update(qid=split, group_id=split, split=split, condition=condition)
                        rows.append(row)
                swapped = state(wait_label="A", round_number=round_number)
                swapped.update(qid=split, group_id=split, split=split, condition=condition, block="label_swap")
                rows.append(swapped)
    report = a.analyze_rows({"fixture": rows}, samples=10)
    assert len(report["round_summaries"]) == 2 * 2 * 3 * 5
    assert all(row["n_questions"] == 1 and row["n_states"] == 4 for row in report["round_summaries"])
    assert all(row["candidate_inconsistent"]["mean"] == 1 for row in report["rotation_consistency"])
    assert {row["split"] for row in report["policy_summaries"]} == set(a.SPLITS)
    broken = copy.deepcopy(rows[:-2])
    with pytest.raises((ValueError, KeyError)):
        a.analyze_rows({"fixture": broken}, samples=10)


def test_candidate_argmax_flip_is_rejected_even_when_wait_remains_top():
    config = {"raw_logit_atol": .001, "raw_logit_rtol": .00001, "probability_atol": .001}
    left = [{"logits": [1.00001, 1., 0., 0., 3.]}]
    right = [{"logits": [1., 1.00001, 0., 0., 3.]}]
    job = {"allowed_actions": "ABCDE", "option_source_ids": a.canonical_mapping(0)}
    with pytest.raises(ValueError, match="candidate numerical"):
        a.compare_numeric(left, right, [job], config)


def test_candidate_numeric_gate_includes_e_when_a_is_wait():
    config = {"raw_logit_atol": .001, "raw_logit_rtol": .00001, "probability_atol": .001}
    left = [{"logits": [3., 1., 0., 0., 1.00001]}]
    right = [{"logits": [3., 1.00001, 0., 0., 1.]}]
    job = {"allowed_actions": "ABCDE", "option_source_ids": a.canonical_mapping(0, "A")}
    with pytest.raises(ValueError, match="candidate numerical"):
        a.compare_numeric(left, right, [job], config)


def test_synthetic_uniform_cases_use_all_orders_and_two_wait_labels():
    cases = a.synthetic_specs()
    assert len(cases) == 32
    assert sum(case["expected_semantic_action"] == "WAIT" for case in cases.values()) == 8
    assert sum(case["expected_semantic_action"] == "PASS" for case in cases.values()) == 8
    for case in cases.values():
        mapping = a.canonical_mapping(case["rotation"], case["wait_label"])
        assert set(mapping.values()) == set("ABCD")
        assert case["wait_label"] not in mapping


def test_plain_prompt_has_no_round_reward_or_wait_metadata():
    options = [{"id": label, "text": "candidate " + label} for label in "ABCD"]
    p1 = a.expected_prompt("a clue", options, 1, "plain", "E")
    p5 = a.expected_prompt("a clue", options, 5, "plain", "E")
    assert p1 == p5
    assert "WAIT" not in p1 and "round" not in p1 and "points" not in p1
    terminal = a.expected_prompt("a clue", options, 5, "wait", "A")
    assert "A means PASS, ending with 0 points." in terminal
    assert "no further question text or sixth round is available" in terminal


def test_new_inference_must_have_completed_all_new_rows(tmp_path):
    directory = tmp_path / "new"
    directory.mkdir()
    (directory / "receipt.json").write_text(json.dumps({"status": "complete", "completed_rows": 4031}))
    with pytest.raises(ValueError, match="completion receipt"):
        a.validate_new_model(directory, "qwen3b", {}, [], {}, "hash", tmp_path / "old", [])
