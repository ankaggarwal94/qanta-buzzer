"""
Evaluation Metrics for Quiz Bowl Buzzer Agents

Computes buzz accuracy, S_q scoring, calibration metrics (ECE, Brier score),
and buzz timing statistics from episode trace data.

Ported from qb-rl reference implementation (evaluation/metrics.py).
Accepts both raw dicts and dataclass instances (EpisodeResult,
SoftmaxEpisodeResult, PPOEpisodeTrace) via the _to_dict helper.

Functions
---------
system_score(c_trace, g_trace)
    Compute S_q = sum_t b_t * g_t where b_t = c_t * prod_{i<t} (1 - c_i).
expected_calibration_error(confidences, outcomes, n_bins)
    Binned ECE over confidence-outcome pairs.
brier_score(confidences, outcomes)
    Mean squared error between confidence and binary outcome.
summarize_buzz_metrics(results)
    Aggregate accuracy, buzz step, S_q, and reward across episodes.
calibration_at_buzz(results)
    Extract buzz-time top_p confidence and compute ECE + Brier score.
expected_wins_score(c_trace, g_trace, opponent_survival_trace, ...)
    Offline Expected Wins scoring over an episode.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def _to_dict(item: Any) -> dict[str, Any]:
    """Convert dataclass or object to dict for uniform access.

    Parameters
    ----------
    item : Any
        A dict, dataclass instance, or object with __dict__.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the item.
    """
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    return item.__dict__


def system_score(c_trace: list[float], g_trace: list[float]) -> float:
    """Compute S_q scoring metric for a single episode.

    S_q = sum_t b_t * g_t, where b_t = c_t * prod_{i<t} (1 - c_i).
    This is the expected correctness under the agent's buzz policy,
    accounting for the survival probability of not having buzzed earlier.

    Parameters
    ----------
    c_trace : list[float]
        Buzz probability at each time step (confidence proxy).
    g_trace : list[float]
        Correctness indicator at each time step (1.0 if top answer is
        correct, 0.0 otherwise).

    Returns
    -------
    float
        S_q score for the episode, in [0, 1].
    """
    c = np.array(c_trace, dtype=np.float64)
    g = np.array(g_trace, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    b = np.zeros_like(c)
    survival = 1.0
    for t in range(len(c)):
        b[t] = c[t] * survival
        survival *= (1.0 - c[t])
    return float(np.sum(b * g))


def expected_wins_score(
    c_trace: list[float],
    g_trace: list[float],
    opponent_survival_trace: list[float],
    reward_correct: float = 10.0,
    reward_incorrect: float = -5.0,
    opponent_expected_value: float = 0.0,
) -> float:
    """Compute offline Expected Wins score for a single episode.

    Uses the continuous V_self formulation::

        V_self_t = g_t * reward_correct + (1 - g_t) * reward_incorrect

    NOT a binary branch on ``g_t``.

    The full formula is::

        EW = sum_t  b_t * [S_t * V_self_t + (1 - S_t) * V_opp]

    where ``b_t = c_t * prod_{i<t}(1 - c_i)`` is the agent's buzz
    probability mass at step *t*, and ``S_t`` is opponent survival.

    Parameters
    ----------
    c_trace : list[float]
        Per-step buzz probability from the agent.
    g_trace : list[float]
        Per-step correctness probability (P(gold) / P(buzz) for PPO,
        binary 0/1 for baseline agents).
    opponent_survival_trace : list[float]
        Per-step P(opponent has not buzzed before step t).
    reward_correct : float
        Points for buzzing correctly before the opponent.
    reward_incorrect : float
        Points for buzzing incorrectly before the opponent.
    opponent_expected_value : float
        Expected score when the opponent buzzes first.

    Returns
    -------
    float
        Expected Wins score for the episode.
    """
    c = np.array(c_trace, dtype=np.float64)
    g = np.array(g_trace, dtype=np.float64)
    s = np.array(opponent_survival_trace, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    n = min(len(c), len(g), len(s))
    c, g, s = c[:n], g[:n], s[:n]

    b = np.zeros(n, dtype=np.float64)
    survival = 1.0
    for t in range(n):
        b[t] = c[t] * survival
        survival *= 1.0 - c[t]

    v_self = g * reward_correct + (1.0 - g) * reward_incorrect
    v = s * v_self + (1.0 - s) * opponent_expected_value
    return float(np.sum(b * v))


def expected_calibration_error(
    confidences: list[float], outcomes: list[int], n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) with uniform binning.

    ECE measures the gap between predicted confidence and actual accuracy
    across confidence bins. Lower ECE indicates better-calibrated predictions.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).
    n_bins : int
        Number of uniform bins for confidence bucketing.

    Returns
    -------
    float
        Expected calibration error in [0, 1]. Returns 0.0 if no data.
    """
    if not confidences:
        return 0.0
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        bin_acc = y[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    """Compute Brier score (mean squared calibration error).

    Brier score measures the mean squared difference between predicted
    confidence and binary outcome. Lower is better; 0 is perfect.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).

    Returns
    -------
    float
        Brier score in [0, 1]. Returns 0.0 if no data.
    """
    if not confidences:
        return 0.0
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    return float(np.mean((conf - y) ** 2))


def summarize_buzz_metrics(results: list[Any]) -> dict[str, float]:
    """Aggregate buzz metrics across a list of episode results.

    Computes accuracy, mean buzz step, mean S_q score, and mean reward
    from episode trace data. Accepts dicts or dataclass instances.

    Parameters
    ----------
    results : list[Any]
        List of episode results (dicts, EpisodeResult, SoftmaxEpisodeResult,
        or PPOEpisodeTrace instances). Each must have: correct, buzz_step,
        c_trace, g_trace. Optionally: reward_like or episode_reward.

    Returns
    -------
    dict[str, float]
        Summary metrics: n, buzz_accuracy, mean_buzz_step, mean_sq,
        mean_reward_like.
    """
    rows = [_to_dict(r) for r in results]
    if not rows:
        return {
            "n": 0.0,
            "buzz_accuracy": 0.0,
            "mean_buzz_step": 0.0,
            "mean_sq": 0.0,
            "mean_reward_like": 0.0,
        }

    correct = np.array(
        [1 if bool(r.get("correct", False)) else 0 for r in rows],
        dtype=np.float64,
    )
    buzz_steps = np.array(
        [int(r.get("buzz_step", 0)) for r in rows], dtype=np.float64
    )
    sq_scores = np.array(
        [
            system_score(
                list(r.get("c_trace", [])),
                list(r.get("g_trace", [])),
            )
            for r in rows
        ],
        dtype=np.float64,
    )
    reward_like = np.array(
        [
            float(r.get("reward_like", r.get("episode_reward", 0.0)))
            for r in rows
        ],
        dtype=np.float64,
    )

    return {
        "n": float(len(rows)),
        "buzz_accuracy": float(correct.mean()),
        "mean_buzz_step": float(buzz_steps.mean()),
        "mean_sq": float(sq_scores.mean()),
        "mean_reward_like": float(reward_like.mean()),
    }


def per_category_accuracy(
    results: list[Any],
    questions: list[Any],
) -> dict[str, dict[str, float]]:
    """Compute accuracy and S_q metrics grouped by question category.

    Joins results with questions to extract category field, then groups
    and computes summarize_buzz_metrics per category.

    Parameters
    ----------
    results : list[Any]
        Episode results from agent evaluation (dicts or dataclasses).
        Must have qid field for joining.
    questions : list[Any]
        Original questions with category field (MCQuestion or similar).

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from category name to metrics dict with keys:
        n, buzz_accuracy, mean_buzz_step, mean_sq, mean_reward_like.
    """
    from collections import defaultdict

    # Build qid -> category lookup, default to "unknown" for missing
    qid_to_category: dict[str, str] = {}
    for q in questions:
        q_dict = _to_dict(q)
        cat = q_dict.get("category", "") or ""
        qid = q_dict.get("qid", "")
        qid_to_category[qid] = cat if cat else "unknown"

    # Group results by category
    by_category: dict[str, list[Any]] = defaultdict(list)
    for r in results:
        r_dict = _to_dict(r)
        qid = r_dict.get("qid", "")
        category = qid_to_category.get(qid, "unknown")
        by_category[category].append(r)

    # Compute metrics per category
    return {
        cat: summarize_buzz_metrics(rows)
        for cat, rows in sorted(by_category.items())
    }


def calibration_at_buzz(results: list[Any]) -> dict[str, float]:
    """Compute calibration metrics at the buzz decision point.

    Uses the belief model's top-answer probability (``top_p_trace``) at
    buzz time as the confidence proxy.  This measures whether the belief
    distribution is well-calibrated: when the model assigns 0.8
    probability to its top answer, that answer should be correct ~80% of
    the time.

    Falls back to ``c_trace`` (sigmoid confidence) when ``top_p_trace``
    is unavailable (e.g. PPO episode traces that lack per-step belief
    breakdowns).

    Parameters
    ----------
    results : list[Any]
        List of episode results (dicts or dataclass instances). Each must
        have: buzz_step, correct, and at least one of top_p_trace or
        c_trace.

    Returns
    -------
    dict[str, float]
        Calibration metrics: ece, brier, n_calibration.
    """
    rows = [_to_dict(r) for r in results]
    confidences: list[float] = []
    outcomes: list[int] = []
    for row in rows:
        top_p_trace = list(row.get("top_p_trace", []))
        c_trace = list(row.get("c_trace", []))
        conf_trace = top_p_trace if top_p_trace else c_trace
        if not conf_trace:
            continue
        buzz_step = int(row.get("buzz_step", max(0, len(conf_trace) - 1)))
        idx = min(max(0, buzz_step), len(conf_trace) - 1)
        confidences.append(float(conf_trace[idx]))
        outcomes.append(1 if bool(row.get("correct", False)) else 0)

    return {
        "ece": expected_calibration_error(confidences, outcomes),
        "brier": brier_score(confidences, outcomes),
        "n_calibration": float(len(confidences)),
    }
