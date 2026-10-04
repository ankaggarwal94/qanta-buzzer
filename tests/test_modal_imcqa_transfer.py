"""Frozen transfer source, budget, and allocation gates without provider calls."""
from decimal import Decimal
import json
from pathlib import Path

import pytest
from scripts import modal_imcqa_transfer as runner


def control(root):
    return {"run_id": runner.RUN_ID, "cache_run": runner.CACHE_RUN,
        "prepare_receipt_sha256": runner.PREPARE_SHA256, "budget": runner.budget_plan(),
        "source_files_sha256": {name: runner.digest(root/name) for name in runner.SOURCES}}


def test_reservation_bounds_allocation_startup_idle_staging_and_contingency():
    plan = runner.budget_plan()
    computed = Decimal(plan["allocation_rate_usd_per_second"])*(3000+90+2)+Decimal(".10")+Decimal(".01723232")
    assert computed == Decimal("2.09376240") == Decimal(plan["reserved_estimate_usd"])
    assert computed < Decimal(plan["ceiling_usd"]) == Decimal("4.00")
    assert plan["gpu_calls"] == 1 and plan["automatic_retries"] == 0
    assert plan["model_limits"] == {"qwen7b": {"timeout": 3000, "deadline": 2880}}
    runner.validate_plan(plan)


@pytest.mark.parametrize("field,value", [("ceiling_usd", "5"), ("automatic_retries", 1), ("gpu_calls", 2), ("startup_timeout_seconds", 0)])
def test_budget_mutation_rejected(field, value):
    plan = runner.budget_plan()
    plan[field] = value
    with pytest.raises(ValueError):
        runner.validate_plan(plan)


def test_only_new_create_once_run_and_original_model_cache():
    assert runner.RUN_ID == "imcqa-frozen-transfer-20261004"
    assert runner.CACHE_RUN == "acl5000-option-scores-20261003"
    assert "scripts/imcqa_transfer_design.py" in runner.SOURCES
    assert "scripts/imcqa_protocol_scoring.py" in runner.SOURCES
    assert "configs/imcqa_frozen_transfer.json" in runner.SOURCES


def test_hash_mismatch_fails_before_parse_or_provider():
    with pytest.raises(ValueError, match="hash differs"):
        runner.verify_public(b"{}", "0"*64)


def test_frozen_sources_and_semantic_config_are_bound():
    root = Path(__file__).resolve().parents[1]
    bound = control(root)
    runner.verify_sources(root, bound)
    bound["source_files_sha256"]["scripts/imcqa_transfer_scoring.py"] = "0"*64
    with pytest.raises(ValueError, match="sources differ"):
        runner.verify_sources(root, bound)


@pytest.mark.parametrize("section,key,value", [
    ("execution", "batch_size", 32), ("execution", "new_contexts_per_model", 7999),
    ("execution", "internal_deadline_seconds", 2999), ("numerical", "raw_logit_atol", .01),
    ("budget", "reserved_estimate_usd", "1.00"),
])
def test_truthful_source_hash_does_not_license_mismatched_execution(tmp_path, section, key, value):
    root = Path(__file__).resolve().parents[1]
    for name in runner.SOURCES:
        destination = tmp_path/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root/name).read_bytes())
    config_path = tmp_path/"configs/imcqa_frozen_transfer.json"
    config = json.loads(config_path.read_text())
    config[section][key] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="semantic configuration differs"):
        runner.verify_sources(tmp_path, control(tmp_path))
