"""Tests for scripts/optimize_dspy.py — offline DSPy compilation."""

from __future__ import annotations

import pytest

from scripts.optimize_dspy import build_dspy_trainset


def _make_mc_question():
    from qb_data.mc_builder import MCQuestion

    return MCQuestion(
        qid="q1",
        question="Who was the first president?",
        tokens=["Who", "was", "the", "first", "president"],
        answer_primary="George Washington",
        clean_answers=["George Washington"],
        run_indices=[1, 3, 4],
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=["Who was", "Who was the first", "Who was the first president"],
        options=["George Washington", "Thomas Jefferson"],
        gold_index=0,
        option_profiles=["Washington profile", "Jefferson profile"],
        option_answer_primary=["George Washington", "Thomas Jefferson"],
        distractor_strategy="test",
    )


class TestBuildDspyTrainset:
    def test_trainset_structure(self) -> None:
        mc = [_make_mc_question()]
        trainset = build_dspy_trainset(mc, max_examples=10)
        assert len(trainset) == 1
        ex = trainset[0]
        assert "clue_prefix" in ex
        assert "option_profiles" in ex
        assert "gold_index" in ex

    def test_trainset_caps_at_max(self) -> None:
        mc = [_make_mc_question()] * 100
        trainset = build_dspy_trainset(mc, max_examples=5)
        assert len(trainset) == 5

    def test_trainset_empty(self) -> None:
        assert build_dspy_trainset([], max_examples=10) == []


class TestCompileDspyScorer:
    def test_compile_requires_dspy(self) -> None:
        dspy = pytest.importorskip("dspy", reason="dspy not installed")
        from scripts.optimize_dspy import compile_dspy_scorer

        mc = [_make_mc_question()]
        trainset = build_dspy_trainset(mc, max_examples=1)
        # This would require a real LM backend, so just check it's callable
        assert callable(compile_dspy_scorer)
