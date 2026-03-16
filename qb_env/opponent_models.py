"""Opponent buzz-position models for Expected Wins reward computation.

Provides pluggable opponent models that estimate the probability an
opponent has buzzed before a given step.  Used by the ``expected_wins``
reward mode in :class:`TossupMCEnv`.

Three built-in models:

* :class:`EmpiricalHistogramOpponentModel` — derives CDF from
  ``MCQuestion.human_buzz_positions`` data.
* :class:`LogisticOpponentModel` — parametric sigmoid CDF for
  questions that lack empirical data.
* :func:`build_opponent_model_from_config` — factory with fallback
  hierarchy: question-level empirical → global empirical → logistic.

The ``expected_wins`` reward mode is disabled by default.  To enable,
set ``environment.reward_mode: expected_wins`` and optionally configure
``environment.opponent_buzz_model`` in the YAML config.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import numpy as np

from qb_data.mc_builder import MCQuestion


@runtime_checkable
class OpponentBuzzModel(Protocol):
    """Protocol for opponent buzz-position models."""

    def prob_buzzed_before_step(self, question: MCQuestion, step_idx: int) -> float:
        """Cumulative probability that the opponent has buzzed before *step_idx*.

        Parameters
        ----------
        question : MCQuestion
            Current question (may carry ``human_buzz_positions``).
        step_idx : int
            0-based clue step.

        Returns
        -------
        float
            P(opponent buzzed before step_idx), in [0, 1].
        """
        ...

    def prob_survive_to_step(self, question: MCQuestion, step_idx: int) -> float:
        """Probability that the opponent has NOT buzzed by *step_idx*.

        Complement of :meth:`prob_buzzed_before_step`.
        """
        ...


class LogisticOpponentModel:
    """Parametric logistic CDF opponent model.

    Models the opponent's cumulative buzz probability at step *t* as::

        P(buzzed before t) = 1 / (1 + exp(-steepness * (t/total - midpoint)))

    Parameters
    ----------
    midpoint : float
        Fraction of total steps at which the CDF reaches 0.5.
    steepness : float
        Controls how sharply the probability increases around the
        midpoint.  Higher values → sharper transition.
    """

    def __init__(self, midpoint: float = 0.6, steepness: float = 6.0) -> None:
        self.midpoint = midpoint
        self.steepness = steepness

    def prob_buzzed_before_step(self, question: MCQuestion, step_idx: int) -> float:
        total = len(question.cumulative_prefixes)
        if total <= 1:
            return 0.0
        frac = step_idx / total
        x = self.steepness * (frac - self.midpoint)
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        z = math.exp(x)
        return z / (1.0 + z)

    def prob_survive_to_step(self, question: MCQuestion, step_idx: int) -> float:
        return 1.0 - self.prob_buzzed_before_step(question, step_idx)


class EmpiricalHistogramOpponentModel:
    """Opponent model derived from empirical human buzz-position data.

    Builds a per-step CDF from the ``human_buzz_positions`` field on
    each question.  Falls back to a :class:`LogisticOpponentModel`
    when a question has no empirical data.

    Parameters
    ----------
    fallback : LogisticOpponentModel or None
        Model to use when a question lacks empirical data.
    global_positions : list of (int, int) or None
        Pooled (position, count) pairs from the entire dataset.
        Used when a question has no per-question data but a global
        distribution is available.
    """

    def __init__(
        self,
        fallback: LogisticOpponentModel | None = None,
        global_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        self.fallback = fallback or LogisticOpponentModel()
        self._global_cdf: np.ndarray | None = None
        if global_positions:
            self._global_cdf = self._build_cdf(global_positions)

    @staticmethod
    def _build_cdf(positions: list[tuple[int, int]]) -> np.ndarray:
        """Build a CDF array from (position, count) pairs.

        Returns an array where ``cdf[i]`` is the cumulative probability
        that a buzz has occurred at or before position *i*.
        """
        if not positions:
            return np.array([], dtype=np.float64)
        max_pos = max(p for p, _ in positions)
        counts = np.zeros(max_pos + 1, dtype=np.float64)
        for pos, count in positions:
            counts[pos] += count
        total = counts.sum()
        if total <= 0:
            return np.zeros(max_pos + 1, dtype=np.float64)
        return np.cumsum(counts) / total

    def _cdf_at_step(
        self, cdf: np.ndarray, question: MCQuestion, step_idx: int
    ) -> float:
        """Look up cumulative probability at a token position."""
        if cdf.size == 0:
            return 0.0
        if not question.run_indices:
            token_pos = step_idx
        elif step_idx < len(question.run_indices):
            token_pos = question.run_indices[step_idx]
        else:
            token_pos = question.run_indices[-1] if question.run_indices else step_idx
        idx = min(token_pos, len(cdf) - 1)
        return float(cdf[idx])

    def prob_buzzed_before_step(self, question: MCQuestion, step_idx: int) -> float:
        if question.human_buzz_positions:
            cdf = self._build_cdf(question.human_buzz_positions)
            return self._cdf_at_step(cdf, question, step_idx)
        if self._global_cdf is not None and self._global_cdf.size > 0:
            return self._cdf_at_step(self._global_cdf, question, step_idx)
        return self.fallback.prob_buzzed_before_step(question, step_idx)

    def prob_survive_to_step(self, question: MCQuestion, step_idx: int) -> float:
        return 1.0 - self.prob_buzzed_before_step(question, step_idx)


def build_opponent_model_from_config(
    questions: list[MCQuestion] | None = None,
    config: dict[str, Any] | None = None,
) -> OpponentBuzzModel | None:
    """Build an opponent model from YAML configuration.

    Returns ``None`` when the opponent model is disabled (the default).

    Parameters
    ----------
    questions : list[MCQuestion] or None
        Dataset questions for building global empirical distribution.
    config : dict or None
        Full YAML config dict.

    Returns
    -------
    OpponentBuzzModel or None
    """
    if config is None:
        return None
    env_cfg = config.get("environment", {})
    opp_cfg = env_cfg.get("opponent_buzz_model", {})
    if not opp_cfg or opp_cfg.get("type", "none") == "none":
        return None

    model_type = opp_cfg.get("type", "logistic")

    if model_type == "logistic":
        return LogisticOpponentModel(
            midpoint=float(opp_cfg.get("midpoint", 0.6)),
            steepness=float(opp_cfg.get("steepness", 6.0)),
        )

    if model_type == "empirical":
        global_positions: list[tuple[int, int]] = []
        if questions:
            for q in questions:
                if q.human_buzz_positions:
                    global_positions.extend(q.human_buzz_positions)
        fallback = LogisticOpponentModel(
            midpoint=float(opp_cfg.get("midpoint", 0.6)),
            steepness=float(opp_cfg.get("steepness", 6.0)),
        )
        return EmpiricalHistogramOpponentModel(
            fallback=fallback,
            global_positions=global_positions if global_positions else None,
        )

    raise ValueError(f"Unknown opponent_buzz_model type: {model_type}")
