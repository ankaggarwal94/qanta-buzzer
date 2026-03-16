"""Integration test exercising a mixed-K pipeline path."""

from __future__ import annotations

import numpy as np
import pytest

from qb_data.data_loader import TossupQuestion
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.mc_builder import MCBuilder
from models.likelihoods import TfIdfLikelihood
from qb_env.tossup_env import TossupMCEnv
from qb_env.text_wrapper import TextObservationWrapper
from agents.threshold_buzzer import ThresholdBuzzer


def _make_questions(n: int = 30) -> list[TossupQuestion]:
    questions = []
    for i in range(n):
        tokens = [f"word{i}_{j}" for j in range(8)]
        questions.append(
            TossupQuestion(
                qid=f"q{i:03d}",
                question=" ".join(tokens),
                tokens=tokens,
                answer_primary=f"Answer_{i}",
                clean_answers=[f"Answer_{i}"],
                run_indices=[1, 3, 7],
                human_buzz_positions=[],
                category=["History", "Science"][i % 2],
                cumulative_prefixes=[
                    " ".join(tokens[:2]),
                    " ".join(tokens[:4]),
                    " ".join(tokens),
                ],
            )
        )
    return questions


def test_mixed_k_build_env_baseline() -> None:
    """Build mixed-K dataset, construct env, run a baseline agent."""
    questions = _make_questions(30)
    builder = MCBuilder(
        K=5, strategy="category_random", random_seed=42,
        variable_K=True, min_K=2, max_K=5,
    )
    profile = AnswerProfileBuilder()
    mc = builder.build(questions, profile)
    assert len(mc) > 0

    option_counts = {len(q.options) for q in mc}
    assert len(option_counts) > 1, f"Expected mixed K, got {option_counts}"

    corpus = [q.question for q in mc] + [p for q in mc for p in q.option_profiles]
    lm = TfIdfLikelihood(corpus_texts=corpus)

    max_k = max(len(q.options) for q in mc)
    env = TossupMCEnv(
        questions=mc, likelihood_model=lm,
        K=max_k, variable_K=True, max_K=max_k,
        reward_mode="simple", belief_mode="from_scratch",
    )

    obs, info = env.reset(seed=42, options={"question_idx": 0})
    assert obs.shape == (max_k + 6,)

    mask = env.action_masks()
    k_actual = len(mc[0].options)
    assert mask[0]
    assert all(mask[1: k_actual + 1])

    buzzer = ThresholdBuzzer(
        likelihood_model=lm, threshold=0.5, beta=5.0, alpha=10.0,
    )
    result = buzzer.run_episode(mc[0])
    assert 0 <= result.buzz_index < len(mc[0].options)


def test_mixed_k_text_wrapper_formats_correctly() -> None:
    """TextObservationWrapper formats per-question K dynamically."""
    questions = _make_questions(30)
    builder = MCBuilder(
        K=4, strategy="category_random", random_seed=42,
        variable_K=True, min_K=2, max_K=4,
    )
    profile = AnswerProfileBuilder()
    mc = builder.build(questions, profile)
    assert len(mc) > 0

    corpus = [q.question for q in mc] + [p for q in mc for p in q.option_profiles]
    lm = TfIdfLikelihood(corpus_texts=corpus)

    max_k = max(len(q.options) for q in mc)
    env = TossupMCEnv(
        questions=mc, likelihood_model=lm,
        K=max_k, variable_K=True, max_K=max_k,
        reward_mode="simple", belief_mode="from_scratch",
    )
    wrapped = TextObservationWrapper(env)

    for idx in range(min(5, len(mc))):
        obs, _ = wrapped.reset(seed=42, options={"question_idx": idx})
        n_opts = len(mc[idx].options)
        assert f"({n_opts})" in obs
        if n_opts < max_k:
            assert f"({n_opts + 1})" not in obs
