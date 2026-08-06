"""Focused calibration-contract checks used by the standalone checker."""
from __future__ import annotations

import math
from typing import Any


def _finite_number(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return (
        math.isfinite(number)
        and (minimum is None or number >= minimum)
        and (maximum is None or number <= maximum)
    )


def platt_phase_errors(block: Any, *, phase: str) -> list[str]:
    """Return producer-contract errors for one staged Platt phase."""
    prefix = f"adapter calibration {phase}"
    if not isinstance(block, dict):
        return [f"{prefix} parameters are noncanonical"]

    legacy_fields = {"platt_coef", "platt_intercept"}
    compatibility_fields = legacy_fields | {
        "platt_model_type",
        "platt_constant_probability",
    }
    producer_fields = legacy_fields | {
        "ece",
        "n_samples",
        "platt_model_type",
        "platt_fallback_reason",
        "platt_constant_probability",
    }
    fields = set(block)
    if fields == legacy_fields:
        if not all(_finite_number(block.get(name)) for name in legacy_fields):
            return [f"{prefix} logistic parameters are invalid"]
        return []
    if fields == compatibility_fields:
        model_type = block.get("platt_model_type")
        coefficient = block.get("platt_coef")
        intercept = block.get("platt_intercept")
        probability = block.get("platt_constant_probability")
        if model_type == "logistic":
            if (
                not _finite_number(coefficient)
                or not _finite_number(intercept)
                or probability is not None
            ):
                return [f"{prefix} logistic parameters are invalid"]
            return []
        if model_type == "constant":
            if (
                coefficient is not None
                or intercept is not None
                or not _finite_number(probability, minimum=0.0, maximum=1.0)
            ):
                return [f"{prefix} constant parameters are invalid"]
            return []
        return [f"{prefix} platt_model_type is invalid"]
    if fields != producer_fields:
        return [f"{prefix} parameters are noncanonical"]

    ece = block.get("ece")
    n_samples = block.get("n_samples")
    if not _finite_number(ece, minimum=0.0, maximum=1.0):
        return [f"{prefix} ece is invalid"]
    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, int)
        or n_samples < 0
    ):
        return [f"{prefix} n_samples is invalid"]

    model_type = block.get("platt_model_type")
    coefficient = block.get("platt_coef")
    intercept = block.get("platt_intercept")
    fallback_reason = block.get("platt_fallback_reason")
    probability = block.get("platt_constant_probability")
    if model_type == "logistic":
        if (
            not _finite_number(coefficient)
            or not _finite_number(intercept)
            or probability is not None
            or fallback_reason is not None
            or n_samples == 0
        ):
            return [f"{prefix} logistic parameters are invalid"]
        return []
    if model_type == "constant":
        probability_valid = _finite_number(
            probability,
            minimum=0.0,
            maximum=1.0,
        )
        fallback_valid = (
            fallback_reason == "empty_validation_bucket"
            and probability_valid
            and float(probability) == 0.0
        ) or (
            fallback_reason == "single_class_validation_bucket"
            and probability_valid
            and float(probability) in {0.0, 1.0}
        )
        if (
            coefficient is not None
            or intercept is not None
            or not probability_valid
            or not fallback_valid
        ):
            return [f"{prefix} constant parameters are invalid"]
        return []
    return [f"{prefix} platt_model_type is invalid"]
