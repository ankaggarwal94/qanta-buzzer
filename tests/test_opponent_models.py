"""Tests for qb_env/opponent_models.py."""

from __future__ import annotations

import pytest

from qb_data.mc_builder import MCQuestion
from qb_env.opponent_models import (
    EmpiricalHistogramOpponentModel,
    LogisticOpponentModel,
    build_opponent_model_from_config,
)


def _make_question(
    human_buzz_positions=None,
    num_steps: int = 6,
) -> MCQuestion:
    tokens = [f"t{i}" for i in range(num_steps * 2)]
    run_indices = list(range(0, num_steps * 2, 2))
    prefixes = [" ".join(tokens[: ri + 1]) for ri in run_indices]
    return MCQuestion(
        qid="q_test",
        question=" ".join(tokens),
        tokens=tokens,
        answer_primary="Answer A",
        clean_answers=["Answer A"],
        run_indices=run_indices,
        human_buzz_positions=human_buzz_positions or [],
        category="Test",
        cumulative_prefixes=prefixes,
        options=["Answer A", "Answer B", "Answer C", "Answer D"],
        gold_index=0,
        option_profiles=["prof_a", "prof_b", "prof_c", "prof_d"],
        option_answer_primary=["Answer A", "Answer B", "Answer C", "Answer D"],
        distractor_strategy="test",
    )


class TestLogisticOpponentModel:
    def test_monotonicity(self) -> None:
        model = LogisticOpponentModel(midpoint=0.5, steepness=6.0)
        q = _make_question(num_steps=10)
        probs = [model.prob_buzzed_before_step(q, t) for t in range(10)]
        for i in range(1, len(probs)):
            assert probs[i] >= probs[i - 1] - 1e-12

    def test_range_01(self) -> None:
        model = LogisticOpponentModel()
        q = _make_question(num_steps=20)
        for t in range(20):
            p = model.prob_buzzed_before_step(q, t)
            assert 0.0 <= p <= 1.0

    def test_survive_complement(self) -> None:
        model = LogisticOpponentModel()
        q = _make_question(num_steps=10)
        for t in range(10):
            assert abs(
                model.prob_buzzed_before_step(q, t)
                + model.prob_survive_to_step(q, t)
                - 1.0
            ) < 1e-12

    def test_step_zero_near_zero(self) -> None:
        model = LogisticOpponentModel(midpoint=0.6, steepness=6.0)
        q = _make_question(num_steps=10)
        assert model.prob_buzzed_before_step(q, 0) < 0.1


class TestEmpiricalHistogramOpponentModel:
    def test_cumulative_from_positions(self) -> None:
        q = _make_question(human_buzz_positions=[(2, 3), (4, 7)], num_steps=6)
        model = EmpiricalHistogramOpponentModel()
        p_at_3 = model.prob_buzzed_before_step(q, 1)
        p_at_5 = model.prob_buzzed_before_step(q, 2)
        assert p_at_5 >= p_at_3

    def test_fallback_when_no_data(self) -> None:
        q = _make_question(human_buzz_positions=[], num_steps=10)
        model = EmpiricalHistogramOpponentModel()
        p = model.prob_buzzed_before_step(q, 5)
        assert 0.0 <= p <= 1.0

    def test_global_fallback(self) -> None:
        q = _make_question(human_buzz_positions=[], num_steps=6)
        model = EmpiricalHistogramOpponentModel(
            global_positions=[(2, 5), (4, 5)]
        )
        p = model.prob_buzzed_before_step(q, 2)
        assert p > 0.0

    def test_cache_key_includes_buzz_positions(self) -> None:
        """Two questions with the same qid but different human_buzz_positions
        must NOT collide in the per-question CDF cache. Keying by qid alone
        would silently return the first question's CDF for the second."""
        model = EmpiricalHistogramOpponentModel()

        q1 = _make_question(
            human_buzz_positions=[(2, 100)],  # all buzzes early
            num_steps=6,
        )
        q1_p_late = model.prob_buzzed_before_step(q1, 5)
        assert q1_p_late >= 0.99  # CDF should be saturated by step 5

        q2 = _make_question(
            human_buzz_positions=[(10, 100)],  # all buzzes late
            num_steps=6,
        )
        # q2 shares qid="q_test" with q1 by construction; buzz histogram
        # differs. Without the fingerprint in the cache key, we'd reuse q1's
        # CDF here and report ~1.0 (wrong).
        q2_p_step_2 = model.prob_buzzed_before_step(q2, 2)
        assert q2_p_step_2 < q1_p_late, (
            "CDF cache returned q1's saturated value for q2 with different "
            "human_buzz_positions; cache key must include a positions "
            "fingerprint, not just qid."
        )

    def test_cache_hit_on_repeat_call(self) -> None:
        """Repeat calls on the same question reuse the cached CDF."""
        model = EmpiricalHistogramOpponentModel()
        q = _make_question(human_buzz_positions=[(2, 5), (4, 5)], num_steps=6)

        # First call populates the cache.
        model.prob_buzzed_before_step(q, 1)
        assert len(model._per_question_cdf_cache) == 1
        # Second call must reuse it (cache size unchanged).
        model.prob_buzzed_before_step(q, 4)
        assert len(model._per_question_cdf_cache) == 1


class TestBuildOpponentModelFromConfig:
    def test_none_when_disabled(self) -> None:
        cfg = {"environment": {"opponent_buzz_model": {"type": "none"}}}
        assert build_opponent_model_from_config(config=cfg) is None

    def test_none_when_missing(self) -> None:
        assert build_opponent_model_from_config(config={}) is None
        assert build_opponent_model_from_config(config=None) is None

    def test_logistic(self) -> None:
        cfg = {"environment": {"opponent_buzz_model": {"type": "logistic", "midpoint": 0.4}}}
        model = build_opponent_model_from_config(config=cfg)
        assert isinstance(model, LogisticOpponentModel)
        assert model.midpoint == 0.4

    def test_empirical(self) -> None:
        q = _make_question(human_buzz_positions=[(2, 5)])
        cfg = {"environment": {"opponent_buzz_model": {"type": "empirical"}}}
        model = build_opponent_model_from_config(questions=[q], config=cfg)
        assert isinstance(model, EmpiricalHistogramOpponentModel)
