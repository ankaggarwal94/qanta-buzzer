"""Budget and identity guards without provider access or model allocation."""
from copy import deepcopy
from decimal import Decimal
import pytest

from scripts.modal_acl_paired_prompt_scores import budget_plan, validate_plan, verify_prepare


def test_combined_reservation_covers_both_models_and_stays_under_two_dollars():
    plan = budget_plan()
    seconds = sum(v["timeout"] + plan["startup_timeout_seconds"] + plan["scaledown_seconds"]
                  for v in plan["model_limits"].values())
    expected = Decimal(plan["allocation_rate_usd_per_second"]) * seconds + Decimal("0.40")
    assert expected == Decimal(plan["reserved_estimate_usd"]) == Decimal("1.98787216")
    assert expected < Decimal(plan["ceiling_usd"])
    assert plan["automatic_retries"] == plan["cpu_calls"] == 0
    assert plan["gpu_calls"] == 2
    assert all(v["deadline"] < v["timeout"] for v in plan["model_limits"].values())
    validate_plan(plan)


@pytest.mark.parametrize("field,value", [("ceiling_usd", "3"), ("automatic_retries", 1),
                                        ("reserved_estimate_usd", "0")])
def test_budget_tampering_is_rejected(field, value):
    plan = budget_plan()
    plan[field] = value
    with pytest.raises(ValueError, match="two-dollar"):
        validate_plan(plan)


def test_model_limit_is_not_mutable_through_returned_plan():
    plan = deepcopy(budget_plan())
    plan["model_limits"]["qwen3b"]["timeout"] += 1
    with pytest.raises(ValueError):
        validate_plan(plan)
    assert budget_plan()["model_limits"]["qwen3b"]["timeout"] == 850


def test_cache_receipt_cannot_be_substituted():
    with pytest.raises(ValueError, match="receipt changed"):
        verify_prepare(b'{}')
