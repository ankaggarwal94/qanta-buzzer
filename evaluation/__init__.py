"""
Evaluation Package

Metrics computation for quiz bowl buzzer agents, including S_q scoring,
calibration analysis (ECE, Brier score), and buzz timing statistics.

Ported from qb-rl reference implementation with adaptations for
qanta-buzzer's EpisodeResult / SoftmaxEpisodeResult / PPOEpisodeTrace
dataclass structures.
"""

from evaluation.metrics import (
    calibration_at_buzz,
    expected_calibration_error,
    expected_wins_score,
    per_category_accuracy,
    summarize_buzz_metrics,
    system_score,
)

__all__ = [
    "system_score",
    "expected_wins_score",
    "summarize_buzz_metrics",
    "calibration_at_buzz",
    "expected_calibration_error",
    "per_category_accuracy",
]
