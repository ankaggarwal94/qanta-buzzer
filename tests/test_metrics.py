"""Unit tests for evaluation metrics.

Tests edge cases for system_score (S_q), calibration metrics (ECE, Brier),
and per-category accuracy grouping.
"""

import pytest

from evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    per_category_accuracy,
    summarize_buzz_metrics,
    system_score,
)


# ---------------------------------------------------------------------------
# system_score (S_q) edge cases
# ---------------------------------------------------------------------------


def test_system_score_empty_trace():
    """S_q should return 0.0 for empty traces."""
    assert system_score([], []) == 0.0


def test_system_score_all_zero_confidence():
    """S_q should return 0.0 when agent never considers buzzing."""
    c_trace = [0.0, 0.0, 0.0]
    g_trace = [1.0, 1.0, 1.0]  # All correct but agent doesn't buzz
    assert system_score(c_trace, g_trace) == 0.0


def test_system_score_all_correct_immediate_buzz():
    """S_q should equal first g_trace value when agent buzzes immediately."""
    c_trace = [1.0, 0.0, 0.0]  # Buzz on step 0
    g_trace = [1.0, 1.0, 1.0]
    expected = 1.0 * 1.0  # b_0 = c_0 * 1.0 = 1.0, survival after = 0
    assert abs(system_score(c_trace, g_trace) - expected) < 1e-9


def test_system_score_gradual_confidence():
    """S_q should accumulate survival-weighted correctness."""
    c_trace = [0.3, 0.5, 1.0]
    g_trace = [0.0, 0.0, 1.0]  # Only correct at final step
    # b_0 = 0.3 * 1.0 = 0.3, survival = 0.7
    # b_1 = 0.5 * 0.7 = 0.35, survival = 0.7 * 0.5 = 0.35
    # b_2 = 1.0 * 0.35 = 0.35
    # S_q = 0.3*0 + 0.35*0 + 0.35*1 = 0.35
    expected = 0.35
    assert abs(system_score(c_trace, g_trace) - expected) < 1e-9


def test_system_score_single_step():
    """S_q should work for single-step episodes."""
    c_trace = [1.0]
    g_trace = [1.0]
    assert abs(system_score(c_trace, g_trace) - 1.0) < 1e-9

    c_trace = [0.5]
    g_trace = [1.0]
    assert abs(system_score(c_trace, g_trace) - 0.5) < 1e-9


def test_system_score_never_correct():
    """S_q should return 0.0 when g_trace is all zeros."""
    c_trace = [0.5, 0.5, 0.5]
    g_trace = [0.0, 0.0, 0.0]
    assert system_score(c_trace, g_trace) == 0.0


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------


def test_expected_calibration_error_perfect():
    """ECE should be near 0.0 for perfectly calibrated predictions."""
    # 70% confidence with 70% accuracy
    confidences = [0.7] * 10
    outcomes = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    ece = expected_calibration_error(confidences, outcomes, n_bins=10)
    assert ece < 0.01  # Near zero for perfect calibration


def test_expected_calibration_error_empty():
    """ECE should return 0.0 for empty inputs."""
    assert expected_calibration_error([], []) == 0.0


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------


def test_brier_score_perfect():
    """Brier score should be 0.0 for perfect predictions."""
    confidences = [1.0, 1.0, 0.0, 0.0]
    outcomes = [1, 1, 0, 0]
    bs = brier_score(confidences, outcomes)
    assert bs == 0.0


def test_brier_score_worst():
    """Brier score should be 1.0 for worst-case predictions."""
    confidences = [0.0, 0.0, 1.0, 1.0]
    outcomes = [1, 1, 0, 0]
    bs = brier_score(confidences, outcomes)
    assert abs(bs - 1.0) < 1e-9


def test_brier_score_empty():
    """Brier score should return 0.0 for empty inputs."""
    assert brier_score([], []) == 0.0


# ---------------------------------------------------------------------------
# summarize_buzz_metrics
# ---------------------------------------------------------------------------


def test_summarize_buzz_metrics_empty():
    """summarize_buzz_metrics should handle empty results."""
    result = summarize_buzz_metrics([])
    assert result["n"] == 0.0
    assert result["buzz_accuracy"] == 0.0


def test_summarize_buzz_metrics_basic():
    """summarize_buzz_metrics should compute correct aggregates."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 2,
            "c_trace": [0.0, 0.0, 1.0],
            "g_trace": [0.0, 0.0, 1.0],
            "reward_like": 0.8,
        },
        {
            "qid": "q2",
            "correct": False,
            "buzz_step": 1,
            "c_trace": [0.0, 1.0],
            "g_trace": [0.0, 0.0],
            "reward_like": -0.1,
        },
    ]
    summary = summarize_buzz_metrics(results)
    assert summary["n"] == 2.0
    assert abs(summary["buzz_accuracy"] - 0.5) < 1e-9
    assert abs(summary["mean_buzz_step"] - 1.5) < 1e-9


# ---------------------------------------------------------------------------
# per_category_accuracy
# ---------------------------------------------------------------------------


def test_per_category_accuracy_basic():
    """per_category_accuracy should group results by question category."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 2,
            "c_trace": [0.0, 0.0, 1.0],
            "g_trace": [0.0, 0.0, 1.0],
            "reward_like": 0.8,
        },
        {
            "qid": "q2",
            "correct": False,
            "buzz_step": 1,
            "c_trace": [0.0, 1.0],
            "g_trace": [0.0, 0.0],
            "reward_like": -0.1,
        },
        {
            "qid": "q3",
            "correct": True,
            "buzz_step": 3,
            "c_trace": [0.0, 0.0, 0.0, 1.0],
            "g_trace": [0.0, 0.0, 0.0, 1.0],
            "reward_like": 0.7,
        },
    ]
    questions = [
        {"qid": "q1", "category": "History"},
        {"qid": "q2", "category": "Science"},
        {"qid": "q3", "category": "History"},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "History" in cat_metrics
    assert "Science" in cat_metrics
    assert cat_metrics["History"]["n"] == 2.0
    assert cat_metrics["History"]["buzz_accuracy"] == 1.0
    assert cat_metrics["Science"]["n"] == 1.0
    assert cat_metrics["Science"]["buzz_accuracy"] == 0.0


def test_per_category_accuracy_missing_category():
    """per_category_accuracy should default missing categories to 'unknown'."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [1.0],
            "g_trace": [1.0],
            "reward_like": 1.0,
        },
    ]
    questions = [
        {"qid": "q1", "category": ""},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "unknown" in cat_metrics
    assert cat_metrics["unknown"]["n"] == 1.0


def test_per_category_accuracy_none_category():
    """per_category_accuracy should handle None category."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [1.0],
            "g_trace": [1.0],
            "reward_like": 1.0,
        },
    ]
    questions = [
        {"qid": "q1", "category": None},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "unknown" in cat_metrics


def test_per_category_accuracy_unmatched_qid():
    """Results with qids not in questions should group to 'unknown'."""
    results = [
        {
            "qid": "q_orphan",
            "correct": False,
            "buzz_step": 0,
            "c_trace": [1.0],
            "g_trace": [0.0],
            "reward_like": -0.1,
        },
    ]
    questions = [
        {"qid": "q1", "category": "History"},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "unknown" in cat_metrics
    assert cat_metrics["unknown"]["n"] == 1.0
