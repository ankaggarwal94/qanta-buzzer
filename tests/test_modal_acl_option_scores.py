"""Control-plane tests requiring no Modal account or model weights."""
from decimal import Decimal
import pytest
from scripts.modal_acl_option_scores import budget_plan, validate_plan, safe_output_path


def test_total_reserved_cost_includes_startup_and_cpu():
    plan = budget_plan()
    assert Decimal(plan['reserved_estimate_usd']) <= Decimal('3')
    assert plan['gpu_function_timeout_seconds'] == 1800
    assert plan['scorer_deadline_seconds'] < plan['gpu_function_timeout_seconds']
    assert plan['automatic_retries'] == 0
    validate_plan(plan)


def test_budget_tampering_fails():
    plan = budget_plan()
    plan['gpu_function_timeout_seconds'] = 7200
    with pytest.raises(ValueError):
        validate_plan(plan)


@pytest.mark.parametrize('path', ['models/a.bin', '../secret', 'output/../../secret', '/output/a.json'])
def test_collection_is_output_only(path):
    with pytest.raises(ValueError):
        safe_output_path(path)


def test_valid_output_path():
    assert safe_output_path('output/qwen3b/scores.jsonl') == 'output/qwen3b/scores.jsonl'
