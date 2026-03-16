"""Unit tests for evaluation metrics.

Tests edge cases for system_score (S_q), calibration metrics (ECE, Brier),
and per-category accuracy grouping.
"""

import pytest

from evaluation.metrics import (
    brier_score,
    calibration_at_buzz,
    expected_calibration_error,
    expected_wins_score,
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


# ---------------------------------------------------------------------------
# calibration_at_buzz — uses top_p_trace, not g_trace
# ---------------------------------------------------------------------------


def test_calibration_at_buzz_uses_top_p_trace():
    """calibration_at_buzz must use top_p_trace (belief prob), not g_trace (binary)."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 2,
            "c_trace": [0.1, 0.3, 0.9],
            "g_trace": [0.0, 0.0, 1.0],
            "top_p_trace": [0.3, 0.5, 0.8],
        },
        {
            "qid": "q2",
            "correct": False,
            "buzz_step": 1,
            "c_trace": [0.2, 0.7],
            "g_trace": [0.0, 0.0],
            "top_p_trace": [0.4, 0.6],
        },
    ]
    cal = calibration_at_buzz(results)
    assert cal["n_calibration"] == 2.0
    # Confidence from top_p_trace at buzz_step:
    # q1: top_p_trace[2] = 0.8, q2: top_p_trace[1] = 0.6
    # Brier = ((0.8-1)^2 + (0.6-0)^2)/2 = (0.04+0.36)/2 = 0.2
    assert abs(cal["brier"] - 0.2) < 1e-9


def test_calibration_at_buzz_falls_back_to_c_trace():
    """When top_p_trace is absent, calibration should fall back to c_trace."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [0.7],
            "g_trace": [1.0],
        },
    ]
    cal = calibration_at_buzz(results)
    assert cal["n_calibration"] == 1.0
    assert abs(cal["brier"] - (0.7 - 1.0) ** 2) < 1e-9


def test_calibration_at_buzz_empty():
    """calibration_at_buzz should return zeros for empty input."""
    cal = calibration_at_buzz([])
    assert cal["ece"] == 0.0
    assert cal["brier"] == 0.0
    assert cal["n_calibration"] == 0.0


def test_calibration_at_buzz_binary_g_trace_not_used():
    """Regression: binary g_trace must NOT be used as confidence.

    If g_trace (binary 0/1) were used, Brier for a correct episode with
    g_trace=[1.0] would be 0.0 regardless of actual confidence.  With
    top_p_trace=[0.5] and correct=True, Brier = (0.5-1)^2 = 0.25.
    """
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [0.9],
            "g_trace": [1.0],
            "top_p_trace": [0.5],
        },
    ]
    cal = calibration_at_buzz(results)
    assert abs(cal["brier"] - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# expected_wins_score
# ---------------------------------------------------------------------------


def test_expected_wins_score_binary_g_trace():
    """Hand-worked EW with baseline-style binary g_trace.

    Agent buzzes immediately (c=[1.0]), correct (g=[1.0]),
    opponent survival=0.8 → EW = 1.0 * [0.8*10 + 0.2*0] = 8.0
    """
    ew = expected_wins_score(
        c_trace=[1.0],
        g_trace=[1.0],
        opponent_survival_trace=[0.8],
        reward_correct=10.0,
        reward_incorrect=-5.0,
        opponent_expected_value=0.0,
    )
    assert abs(ew - 8.0) < 1e-9


def test_expected_wins_score_fractional_g_trace():
    """Hand-worked EW with PPO-style fractional g_trace.

    c=[1.0], g=[0.6], S=[0.8]
    V_self = 0.6*10 + 0.4*(-5) = 4.0
    V = 0.8*4.0 + 0.2*0 = 3.2
    EW = 1.0 * 3.2 = 3.2
    """
    ew = expected_wins_score(
        c_trace=[1.0],
        g_trace=[0.6],
        opponent_survival_trace=[0.8],
        reward_correct=10.0,
        reward_incorrect=-5.0,
        opponent_expected_value=0.0,
    )
    assert abs(ew - 3.2) < 1e-9


def test_expected_wins_score_empty():
    assert expected_wins_score([], [], []) == 0.0


def test_expected_wins_does_not_regress_system_score():
    """system_score must remain unchanged by EW addition."""
    c = [0.3, 0.5, 1.0]
    g = [0.0, 0.0, 1.0]
    expected = 0.35
    assert abs(system_score(c, g) - expected) < 1e-9
