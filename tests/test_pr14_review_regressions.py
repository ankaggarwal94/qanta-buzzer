"""Regression tests for active PR #14 review findings."""

from __future__ import annotations

import sys

import numpy as np

sys.modules.pop("evaluation.controls", None)
sys.modules.pop("scripts.compute_csli", None)

from scripts import compute_csli, compute_prefix_calibration
from scripts.compute_stopdff import check_threshold_reachability

_filter_test_mc_questions = getattr(compute_csli, "_filter_test_mc_questions")
_fit_bucket_calibrator = getattr(
    compute_prefix_calibration,
    "_fit_bucket_calibrator",
)
_calibrate_bucket_scores = getattr(
    compute_prefix_calibration,
    "_calibrate_bucket_scores",
)


def test_csli_coverage_uses_unique_matched_qids_not_row_count() -> None:
    """Duplicate MC rows for one qid must not hide another missing qid."""
    mc_questions = [
        {"qid": "q1", "question": "first copy"},
        {"qid": "q1", "question": "duplicate copy"},
        {"qid": "outside-test", "question": "not in test"},
    ]

    questions, coverage = _filter_test_mc_questions(mc_questions, {"q1", "q2"})

    assert len(questions) == 2
    assert coverage["matched_test_mc_questions"] == 2
    assert coverage["matched_test_mc_qids"] == 1
    assert coverage["missing_qids"] == {"q2"}
    assert coverage["coverage_rate"] == 0.5


def test_prefix_calibration_uses_constant_model_for_empty_val_bucket() -> None:
    """Empty validation buckets should not read labels[0] or fit Platt."""
    model, metadata = _fit_bucket_calibrator(
        "early",
        np.array([], dtype=float),
        np.array([], dtype=int),
    )

    calibrated = _calibrate_bucket_scores(model, np.array([0.1, 0.2]))

    assert metadata["platt_model_type"] == "constant"
    assert metadata["platt_fallback_reason"] == "empty_validation_bucket"
    assert calibrated.tolist() == [0.0, 0.0]


def test_prefix_calibration_uses_constant_model_for_single_class_bucket() -> None:
    """Single-class validation buckets should not call LogisticRegression.fit."""
    model, metadata = _fit_bucket_calibrator(
        "mid",
        np.array([0.1, 0.2, 0.3], dtype=float),
        np.array([1, 1, 1], dtype=int),
    )

    calibrated = _calibrate_bucket_scores(model, np.array([0.4, 0.5]))

    assert metadata["platt_model_type"] == "constant"
    assert metadata["platt_fallback_reason"] == "single_class_validation_bucket"
    assert calibrated.tolist() == [1.0, 1.0]


def test_prefix_calibration_empty_test_bucket_returns_empty_scores() -> None:
    """Empty test buckets should skip predict_proba and produce no samples."""
    model, metadata = _fit_bucket_calibrator(
        "late",
        np.array([0.1, 0.2, 0.3], dtype=float),
        np.array([0, 1, 1], dtype=int),
    )

    calibrated = _calibrate_bucket_scores(model, np.array([], dtype=float))

    assert metadata["platt_model_type"] == "logistic"
    assert calibrated.shape == (0,)


def test_stopdff_reachability_uses_negative_raw_bound_for_negative_coef() -> None:
    """For negative Platt coefficients, max probability occurs at cosine=-1."""
    reachability = check_threshold_reachability(
        {"early": (-2.0, 0.0)},
        threshold=0.7,
    )

    early = reachability["early"]
    assert early["max_calibrated_raw_score"] == -1.0
    assert early["max_calibrated_probability"] > 0.7
    assert early["threshold_reachable"] is True
    assert -1.0 <= early["required_raw_score"] <= 1.0
