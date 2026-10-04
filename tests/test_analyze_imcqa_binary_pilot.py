"""Counterexamples for binary protocol semantics and question-level analysis."""
import copy
import json

import pytest

from scripts import analyze_imcqa_binary_pilot as a


def job(mapping="submit_x", round_number=1):
    submit, defer = a.labels_for_mapping(mapping)
    return {"mapping": mapping, "submit_label": submit, "defer_label": defer, "round": round_number,
            "proposal_id": "A", "canonical_gold_option_id": "A"}


def score(mapping="submit_x", logits=None, round_number=1):
    j = job(mapping, round_number)
    logits = logits or {"X": 2., "Y": 1.}
    chosen, semantic, ties = a.semantic_argmax(logits, j)
    p = a.old.softmax(logits, a.LABELS)
    return {"raw_action_logits": logits, "action_probabilities": p,
            "semantic_probabilities": {"SUBMIT": p[j["submit_label"]], "DEFER": p[j["defer_label"]]},
            "chosen_action": chosen, "chosen_semantic_action": semantic, "tied_top_actions": ties, "exact_tie": len(ties) == 2}


def trajectory(mapping="submit_x", candidates="BAAAA", actions="SDDDD", split="selection", qid="q", condition="independent_pool"):
    result = []
    for i, (candidate, action) in enumerate(zip(candidates, actions)):
        j = {**job(mapping, i + 1), "qid": qid, "group_id": qid + "-group", "split": split, "condition": condition,
             "fraction": (i + 1) / 5, "proposal_id": candidate, "source_plain_score_sha256": str(i), "block": "real"}
        raw = {label: 2. if label == (j["submit_label"] if action == "S" else j["defer_label"]) else 1. for label in "XY"}
        result.append({**a.score_view(score(mapping, raw, i + 1), j), "legal_action_vocabulary_mass": .9})
    return result


def test_exact_tie_always_defers_under_both_encodings():
    for mapping in a.MAPPINGS:
        result = a.score_view(score(mapping, {"X": 1., "Y": 1.}), job(mapping))
        assert result["exact_tie"]
        assert result["semantic_action"] == "DEFER"
        assert result["chosen_action"] == job(mapping)["defer_label"]


def test_terminal_deferral_is_pass_not_sixth_round_wait():
    j = job("submit_y", 5)
    result = a.score_view(score("submit_y", {"X": 2., "Y": 1.}, 5), j)
    assert result["semantic_action"] == "DEFER"
    assert result["game_action"] == "PASS"


def test_reversed_labels_preserve_semantics_when_logits_reverse():
    first = a.score_view(score("submit_x"), job("submit_x"))
    second = a.score_view(score("submit_y", {"X": 1., "Y": 2.}), job("submit_y"))
    assert first["semantic_action"] == second["semantic_action"] == "SUBMIT"
    assert first["chosen_action"] != second["chosen_action"]
    assert first["semantic_probabilities"] == second["semantic_probabilities"]


def test_incorrect_reported_tie_choice_fails():
    row = score("submit_x", {"X": 1., "Y": 1.})
    row["chosen_action"] = "X"
    with pytest.raises(ValueError, match="tie evidence"):
        a.score_view(row, job())


def test_malformed_or_nonfinite_probabilities_fail():
    row = score()
    row["semantic_probabilities"]["SUBMIT"] = float("nan")
    with pytest.raises(ValueError, match="probabilities"):
        a.score_view(row, job())


def test_native_and_hybrid_use_same_plain_proposals():
    rows = trajectory()
    native = a.policy_record(rows, "binary_native")
    hybrid = a.policy_record(rows, "old_wait_timing_plain_proposal", 2)
    assert native["reward"] == -1
    assert hybrid["canonical_choice"] == "A"
    assert hybrid["reward"] == .8
    assert a.policy_record(rows, "plain_hindsight_or_pass")["reward"] >= native["reward"]


def test_hybrid_pass_does_not_force_terminal_answer():
    result = a.policy_record(trajectory(), "old_wait_timing_plain_proposal", None)
    assert result["terminal_pass"] and result["reward"] == 0


def test_hindsight_with_no_correct_proposal_passes():
    result = a.policy_record(trajectory(candidates="BBBBB"), "plain_hindsight_or_pass")
    assert result["terminal_pass"] and result["reward"] == 0


def test_missing_round_is_not_silently_ignored():
    with pytest.raises(ValueError, match="five ordered"):
        a.policy_record(trajectory()[:-1], "binary_native")
    with pytest.raises(ValueError, match="incomplete"):
        a.trajectories(trajectory()[:-1])


def test_numerical_gate_detects_action_flip_inside_logit_tolerance():
    config = {"raw_logit_atol": .001, "raw_logit_rtol": .00001, "probability_atol": .001}
    with pytest.raises(ValueError, match="agreement"):
        a.compare_numeric([{"logits": [1., 1.00001]}], [{"logits": [1.00001, 1.]}], [job()], config)


def test_numerical_gate_detects_raw_logit_failure_despite_same_choice():
    config = {"raw_logit_atol": .001, "raw_logit_rtol": .00001, "probability_atol": .001}
    with pytest.raises(ValueError, match="raw-logit"):
        a.compare_numeric([{"logits": [100., 0.]}], [{"logits": [100.01, 0.]}], [job()], config)


def test_synthetic_expected_actions_follow_exact_payoffs():
    specs = a.synthetic_specs()
    assert len(specs) == 16
    assert sum(spec["expected_semantic_action"] == "SUBMIT" for spec in specs) == 8
    assert sum(spec["expected_semantic_action"] == "WAIT" for spec in specs) == 4
    assert sum(spec["expected_semantic_action"] == "PASS" for spec in specs) == 4
    for spec in specs:
        assert (spec["submit_expected_reward"] > spec["defer_value"]) == (spec["expected_semantic_action"] == "SUBMIT")


def test_expected_prompt_keeps_candidate_order_and_changes_action_mapping_only():
    options = [{"id": x, "text": x + " answer"} for x in "ABCD"]
    prompts = [a.expected_prompt("question", options, options[0], 1, mapping) for mapping in a.MAPPINGS]
    payloads = [json.loads(prompt.split("\n\n")[1]) for prompt in prompts]
    assert payloads[0] == payloads[1]
    assert payloads[0]["options"] == options
    assert prompts[0] != prompts[1]


def test_mapping_average_retains_question_not_state_bootstrap_units():
    rows, old_rows = [], []
    for split in a.old.SPLITS:
        for menu in a.old.MENUS:
            for mapping in a.MAPPINGS:
                rows += trajectory(mapping, split=split, qid="q-" + split, condition=menu)
            old_rows.extend({"qid": "q-" + split, "condition": menu, "block": "factorial", "arm": "wait", "rotation": 0,
                             "round": r, "chosen_action": "E" if r == 1 else "A", "wait_label": "E"} for r in range(1, 6))
    report = a.analyze_rows(rows, old_rows, samples=10)
    averages = [r for r in report["policy_summaries"] if r["mapping"] == "mapping_average"]
    assert all(r["n_questions"] == 1 and r["n_question_menu_records"] == 2 for r in averages)
    effects = report["mapping_summaries"]
    assert all(row["n_questions"] == 1 and row["n_states"] == 5 for row in effects)
    assert all(row["semantic_action_changed"]["mean"] == 0 for row in effects)


def test_attempt_glob_does_not_confuse_raw_with_gated(tmp_path):
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / "000_diagnostics.json").write_text('{"kind":"gated"}')
    (attempts / "000_diagnostics_raw.json").write_text('{"kind":"raw"}')
    (attempts / "000_production_diagnostics.json").write_text('{"kind":"production"}')
    assert a.one_attempt(tmp_path, "diagnostics")["kind"] == "gated"
    assert a.one_attempt(tmp_path, "diagnostics_raw")["kind"] == "raw"


def test_active_raw_must_match_gated_record_and_sequential_identity(tmp_path):
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    rows = [{"score_id": "s", "live": {"logits": [1., 2.]}}]
    path = attempts / "000_active_000_raw.json"
    path.write_text(json.dumps(rows[0]))
    a.validate_active_raw(tmp_path, rows)
    path.write_text(json.dumps({"score_id": "other", "live": rows[0]["live"]}))
    with pytest.raises(ValueError, match="differs"):
        a.validate_active_raw(tmp_path, rows)
