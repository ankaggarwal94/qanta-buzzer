"""Partial results must preserve the frozen failed gate and two-model scope."""
import json

import pytest

from scripts import analyze_imcqa_protocol_partial as p


def test_partial_scope_keeps_planned_and_validated_counts_distinct():
    scope = p.partial_scope(["qwen3b", "qwen7b"], ["qwen7b"])
    assert scope["status"] == "partial"
    assert scope["schema_version"] == "imcqa-protocol-partial-analysis-v1"
    assert scope["expected_n_score_rows"] == 10464
    assert scope["n_score_rows"] == 5232
    assert scope["n_new_score_rows"] == 4032
    assert scope["n_reused_score_rows"] == 1200
    assert scope["failed_models"] == ["qwen3b"]


@pytest.mark.parametrize("models", [[], ["qwen3b", "qwen7b"], ["qwen7b", "qwen7b"], ["other"]])
def test_partial_mode_cannot_mislabel_empty_complete_duplicate_or_unknown_models(models):
    with pytest.raises(ValueError, match="proper subset"):
        p.partial_scope(["qwen3b", "qwen7b"], models)


def test_diagnostic_summary_keeps_failed_raw_gate_when_actions_are_stable():
    cached, single = {"logits": [10., 0., -1., -2., -3.]}, {"logits": [10.002, .002, -.998, -1.998, -2.998]}
    diagnostic = {"score_ids": ["j"], "cached": [cached], "uncached": [cached], "singles": [single],
                  "replay": [cached], "permuted_aligned": [cached]}
    job = {"score_id": "j", "allowed_actions": "ABCD", "option_source_ids": dict(zip("ABCD", "ABCD")),
           "arm": "plain", "block": "factorial", "round": 1, "rotation": 0, "wait_label": "E"}
    config = {"raw_logit_atol": .001, "raw_logit_rtol": .00001, "probability_atol": .001}
    result = p.numerical_failure_summary(diagnostic, [job], config)
    failed = [row for row in result if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["comparison"] == "cached_vs_singles"
    assert failed[0]["raw_logit_failed_rows"] == 1
    assert failed[0]["action_argmax_changes"] == failed[0]["candidate_argmax_changes"] == 0
    assert failed[0]["max_candidate_probability_difference"] < 1e-12


def test_partial_failure_path_rejects_complete_receipt(tmp_path):
    (tmp_path / "receipt.json").write_text(json.dumps({"status": "complete"}))
    with pytest.raises(ValueError, match="failed receipt"):
        p.validate_failure(tmp_path, "qwen3b", "hash", [], {}, tmp_path)
