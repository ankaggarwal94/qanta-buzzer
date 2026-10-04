"""Honest additive recovery provenance, actual production pairs, and hard budget."""
from decimal import Decimal
import json
from pathlib import Path

import pytest

from scripts import imcqa_3b_protocol_recovery as recovery
from scripts import modal_imcqa_3b_protocol_recovery as runner


def test_diagnostic_contexts_include_actual_production_companions():
    jobs = [{"score_id": str(i)} for i in range(40)]
    ids = [str(i) for i in range(0, 20, 2)]
    indices = recovery.production_pair_diagnostic_indices(jobs, ids)
    assert indices == list(range(20))
    assert len(indices) == 20
    assert [jobs[i]["score_id"] for i in indices[::2]] == ids


def test_original_complete_pairs_remain_exact_original_ten_contexts():
    jobs = [{"score_id": str(i)} for i in range(40)]
    ids = [str(i) for i in [2, 3, 10, 11, 16, 17, 26, 27, 38, 39]]
    indices = recovery.production_pair_diagnostic_indices(jobs, ids)
    assert [jobs[i]["score_id"] for i in indices] == ids


@pytest.mark.parametrize("ids", [["0"]*10, [str(i) for i in range(9)], [str(i) for i in range(100, 110)]])
def test_missing_changed_or_duplicate_diagnostic_ids_fail(ids):
    with pytest.raises(ValueError):
        recovery.production_pair_diagnostic_indices([{"score_id": str(i)} for i in range(40)], ids)


def test_batch_two_permutation_really_swaps_each_pair_and_restores_outputs(monkeypatch):
    observed = []
    def fake_forward(torch, model, tokenizer, contexts, *, cached, batch_size):
        observed.append(([item["id"] for item in contexts], cached, batch_size))
        return [{"value": item["id"]} for item in contexts]
    monkeypatch.setattr(recovery, "forward", fake_forward)
    contexts = [{"id": i} for i in range(6)]
    result = recovery.forward_pairs(None, None, None, contexts, cached=True, reverse_within_pairs=True)
    assert observed == [([1, 0], True, 2), ([3, 2], True, 2), ([5, 4], True, 2)]
    assert result == [{"value": i} for i in range(6)]


def test_singleton_is_not_silently_padded_into_an_unplanned_pair():
    with pytest.raises(ValueError, match="complete production pairs"):
        recovery.forward_pairs(None, None, None, [{}], cached=True)


def test_failing_live_gate_preserves_the_offending_forward_before_raising(tmp_path):
    job = {"score_id": "example", "qid": "q1", "condition": "independent_pool", "block": "factorial",
           "rotation": 2, "round": 3, "allowed_actions": "ABCDE", "option_source_ids": dict(zip("ABCD", "CDAB"))}
    live = {"logits": [0, 0, 0, 0, 1.1]}
    production = {"logits": [0, 0, 0, 0, 1]}
    with pytest.raises(ValueError, match="logits"):
        recovery.record_active_comparison(tmp_path, 0, 0, job, live, production)
    retained = json.loads((tmp_path/"000_active_000_raw.json").read_text())
    assert retained["live"] == live and retained["production"] == production
    assert retained["score_id"] == job["score_id"] and retained["sequence_index"] == 0


def test_budget_covers_allocation_startup_staging_and_contingency():
    plan = runner.budget_plan()
    reserved = Decimal(".00063924")*(1800+90+2)+Decimal(".01723232")+Decimal(".15")
    assert reserved == Decimal("1.37667440") == Decimal(plan["reserved_estimate_usd"])
    assert reserved < Decimal(plan["ceiling_usd"]) == Decimal("1.40")
    assert plan["gpu_calls"] == 1 and plan["automatic_retries"] == 0
    assert plan["internal_deadline_seconds"] == 1650


def test_recovery_records_distinct_execution_source_identity_and_original_scientific_input():
    assert recovery.EXECUTION_PROTOCOL != recovery.PROTOCOL
    assert recovery.SOURCE_FILES[-2:] == ("scripts/imcqa_3b_protocol_recovery.py", "configs/imcqa_3b_protocol_recovery.json")
    assert runner.RUN_ID not in {runner.PROTOCOL_RUN, runner.PRIOR_RUN, runner.NUMERICAL_RUN, runner.CACHE_RUN}
    assert recovery.BATCH_SIZE == 2
    assert recovery.PUBLIC_SHA256 == "3c84125a4891f276565f127436b1e2fa7aa3e30ffbab75c55549e742c145fdbe"


def test_source_config_and_budget_are_checked_before_allocation():
    root = Path(__file__).resolve().parents[1]
    control = {"budget": runner.budget_plan(), "run_id": runner.RUN_ID, "cache_run": runner.CACHE_RUN,
               "protocol_run": runner.PROTOCOL_RUN, "prior_run": runner.PRIOR_RUN, "numerical_run": runner.NUMERICAL_RUN,
               "public_input_sha256": recovery.PUBLIC_SHA256,
               "diagnostic_receipt_sha256": recovery.DIAGNOSTIC_RECEIPT_SHA256,
               "source_files_sha256": {name: runner.digest(root/name) for name in runner.SOURCES}}
    runner.verify_sources(root, control)
    control["budget"]["automatic_retries"] = 1
    with pytest.raises(ValueError, match="budget"):
        runner.verify_sources(root, control)


def test_config_and_worker_keep_limits_and_predecessor_binding():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"configs/imcqa_3b_protocol_recovery.json").read_text())
    assert config["new_contexts"] == 4032 and config["reused_contexts"] == 1200
    assert config["original_plan_sha256"] == recovery.ORIGINAL_PLAN_SHA256
    assert config["predecessor_receipt_sha256"] == recovery.PREDECESSOR_RECEIPT_SHA256
    assert config["scientific_inputs_changed"] is False
    assert config["numerical_checks"]["raw_logit_atol"] == .001
    assert config["numerical_checks"]["raw_logit_rtol"] == .00001
    assert config["numerical_checks"]["active_only_episodes"] == 16


def test_only_reviewed_model_deadline_and_source_commit_are_accepted(tmp_path):
    with pytest.raises(ValueError, match="invalid pinned model"):
        recovery.run_recovery("qwen7b", tmp_path/"missing", recovery.PUBLIC_SHA256, tmp_path, tmp_path/"out",
            prior_root=tmp_path, protocol_root=tmp_path, numerical_root=tmp_path, source_commit="0"*40, max_seconds=1650)
    with pytest.raises(ValueError, match="invalid pinned model"):
        recovery.run_recovery("qwen3b", tmp_path/"missing", recovery.PUBLIC_SHA256, tmp_path, tmp_path/"out",
            prior_root=tmp_path, protocol_root=tmp_path, numerical_root=tmp_path, source_commit="uncommitted", max_seconds=1650)
