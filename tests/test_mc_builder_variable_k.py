"""Tests for variable-K MC question construction."""

from __future__ import annotations

import pytest

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.mc_builder import MCBuilder


def _make_questions(n: int = 20, n_unique_answers: int | None = None) -> list[TossupQuestion]:
    n_ans = n_unique_answers if n_unique_answers is not None else n
    questions = []
    for i in range(n):
        tokens = [f"word{i}_{j}" for j in range(10)]
        questions.append(
            TossupQuestion(
                qid=f"q{i:03d}",
                question=" ".join(tokens),
                tokens=tokens,
                answer_primary=f"Answer_{i % n_ans}",
                clean_answers=[f"Answer_{i % n_ans}"],
                run_indices=[2, 5, 9],
                human_buzz_positions=[],
                category=["History", "Science"][i % 2],
                cumulative_prefixes=[
                    " ".join(tokens[:3]),
                    " ".join(tokens[:6]),
                    " ".join(tokens),
                ],
            )
        )
    return questions


class TestFixedKUnchanged:
    def test_fixed_k_default(self) -> None:
        qs = _make_questions(20)
        builder = MCBuilder(K=4, strategy="category_random", random_seed=42)
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        for q in mc:
            assert len(q.options) == 4

    def test_variable_k_false_is_fixed(self) -> None:
        qs = _make_questions(20)
        builder = MCBuilder(K=4, strategy="category_random", random_seed=42, variable_K=False)
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        for q in mc:
            assert len(q.options) == 4


class TestVariableK:
    def test_variable_k_yields_mixed(self) -> None:
        qs = _make_questions(40)
        builder = MCBuilder(
            K=6, strategy="category_random", random_seed=42,
            variable_K=True, min_K=2, max_K=6,
        )
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        option_counts = {len(q.options) for q in mc}
        assert len(option_counts) > 1, f"Expected mixed K, got only {option_counts}"
        for q in mc:
            assert 2 <= len(q.options) <= 6

    def test_gold_index_valid(self) -> None:
        qs = _make_questions(30)
        builder = MCBuilder(
            K=5, strategy="category_random", random_seed=42,
            variable_K=True, min_K=2, max_K=5,
        )
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        for q in mc:
            assert 0 <= q.gold_index < len(q.options)
            assert q.options[q.gold_index] in q.clean_answers or \
                q.option_answer_primary[q.gold_index] == q.answer_primary

    def test_profiles_match_options(self) -> None:
        qs = _make_questions(20)
        builder = MCBuilder(
            K=5, strategy="category_random", random_seed=42,
            variable_K=True, min_K=3, max_K=5,
        )
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        for q in mc:
            assert len(q.option_profiles) == len(q.options)
            assert len(q.option_answer_primary) == len(q.options)
