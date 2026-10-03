"""Control-plane tests requiring no Modal account or model weights."""
from decimal import Decimal
import pytest
from scripts.modal_acl_option_scores import (budget_plan, validate_plan, safe_output_path,
                                            recovery_budget_plan, verify_prior_receipts)


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


def test_recovery_reserves_both_attempts_within_original_cap():
    plan = recovery_budget_plan()
    assert Decimal(plan['reserved_estimate_usd']) < Decimal('3')
    assert Decimal(plan['prior_attempt_reserved_estimate_usd']) > Decimal('0.16')
    assert plan['cpu_calls'] == 0
    assert plan['gpu_calls'] == 2
    assert plan['scorer_deadline_seconds'] < plan['gpu_function_timeout_seconds']
    validate_plan(plan)
    plan['prior_gpu_observed_seconds'] = '0'
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_recovery_rejects_unknown_prior_attempt():
    with pytest.raises(ValueError, match='prior receipt mismatch'):
        verify_prior_receipts(lambda name: b'{}')


@pytest.mark.parametrize('path', ['models/a.bin', '../secret', 'output/../../secret', '/output/a.json'])
def test_collection_is_output_only(path):
    with pytest.raises(ValueError):
        safe_output_path(path)


def test_valid_output_path():
    assert safe_output_path('output/qwen3b/scores.jsonl') == 'output/qwen3b/scores.jsonl'
