"""No provider calls: new allocation identity and exact four-dollar reservation."""
from decimal import Decimal

import pytest

from scripts import modal_imcqa_protocol_pilot as runner


def test_distinct_create_once_run_reuses_old_models_and_prior_scores():
    assert runner.RUN_ID == "imcqa-protocol-dev-20261004"
    assert runner.PRIOR_RUN == "imcqa-wait-dev-20261004"
    assert runner.RUN_ID != runner.PRIOR_RUN != runner.CACHE_RUN
    assert "scripts/imcqa_protocol_design.py" in runner.SOURCES
    assert "scripts/imcqa_wait_scoring.py" in runner.SOURCES


def test_timeout_reservation_includes_each_startup_cpu_and_contingency():
    plan = runner.budget_plan()
    computed = Decimal(plan["allocation_rate_usd_per_second"])*sum(
        value["timeout"]+92 for value in plan["model_limits"].values())+Decimal(".40")+Decimal(".01723232")
    assert computed == Decimal(plan["reserved_estimate_usd"]) == Decimal("3.98674848")
    assert computed < Decimal(plan["ceiling_usd"])
    assert plan["automatic_retries"] == 0 and plan["gpu_calls"] == 2
    runner.validate_plan(plan)


@pytest.mark.parametrize("field,value", [("ceiling_usd", "5"), ("automatic_retries", 1), ("gpu_calls", 3)])
def test_budget_mutation_rejected(field, value):
    plan = runner.budget_plan()
    plan[field] = value
    with pytest.raises(ValueError):
        runner.validate_plan(plan)


def test_public_hash_is_checked_before_parsing_or_provider_access():
    with pytest.raises(ValueError, match="hash differs"):
        runner.verify_public(b"{}", "0"*64)
