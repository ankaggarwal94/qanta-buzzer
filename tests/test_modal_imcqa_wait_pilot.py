"""No provider calls: budget ceilings, exact package boundary, source identities."""
from decimal import Decimal
from copy import deepcopy
import pytest

from scripts import modal_imcqa_wait_pilot as runner


def test_timeout_reservation_includes_startups_cpu_allowance_and_contingency():
    plan=runner.budget_plan()
    cost=Decimal(plan["allocation_rate_usd_per_second"])*sum(v["timeout"]+92 for v in plan["model_limits"].values())+Decimal(".40")+Decimal(".01723232")
    assert cost==Decimal("3.98674848")==Decimal(plan["reserved_estimate_usd"])
    assert cost<Decimal(plan["ceiling_usd"])
    assert plan["gpu_calls"]==2 and plan["cpu_calls"]==plan["automatic_retries"]==0
    assert all(v["deadline"]<v["timeout"] for v in plan["model_limits"].values())
    runner.validate_plan(plan)


@pytest.mark.parametrize("field,value",[("ceiling_usd","5"),("automatic_retries",1),("gpu_calls",3),("reserved_estimate_usd","1")])
def test_budget_modification_fails_closed(field,value):
    plan=runner.budget_plan();plan[field]=value
    with pytest.raises(ValueError,match="four-dollar"):runner.validate_plan(plan)


def test_no_mutable_plan_alias_or_model_receipt_substitution():
    plan=runner.budget_plan();plan["model_limits"]["qwen7b"]["timeout"]+=1
    with pytest.raises(ValueError):runner.validate_plan(plan)
    assert runner.budget_plan()["model_limits"]["qwen7b"]["timeout"]==3000
    with pytest.raises(ValueError,match="receipt changed"):runner.verify_prepare(b'{}')


def test_public_input_hash_checked_before_parsing_or_provider_access():
    with pytest.raises(ValueError,match="hash differs"):runner.verify_public(b'{}',"0"*64)
