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
        pytest.importorskip("dspy", reason="dspy not installed")
        from scripts.optimize_dspy import compile_dspy_scorer
        assert callable(compile_dspy_scorer)

    def test_score_metric_logic(self) -> None:
        """The _score_metric used by compile_dspy_scorer is argmax-based."""
        import json
        from unittest.mock import MagicMock

        # Simulate a prediction that matches the gold argmax
        example = MagicMock()
        example.scores = json.dumps([0.0, 1.0, 0.0])
        pred_correct = MagicMock()
        pred_correct.scores = json.dumps([0.1, 0.8, 0.1])
        pred_wrong = MagicMock()
        pred_wrong.scores = json.dumps([0.9, 0.05, 0.05])
        pred_malformed = MagicMock()
        pred_malformed.scores = "not json"

        # Import the metric from the module scope
        # The metric is defined inside compile_dspy_scorer, so we test
        # equivalent logic directly
        def _score_metric(ex, prediction, _trace=None):
            try:
                pred_scores = json.loads(prediction.scores)
                target_scores = json.loads(ex.scores)
            except (json.JSONDecodeError, AttributeError):
                return 0.0
            if not pred_scores or not target_scores:
                return 0.0
            return 1.0 if (
                max(range(len(pred_scores)), key=lambda i: pred_scores[i])
                == max(range(len(target_scores)), key=lambda i: target_scores[i])
            ) else 0.0

        assert _score_metric(example, pred_correct) == 1.0
        assert _score_metric(example, pred_wrong) == 0.0
        assert _score_metric(example, pred_malformed) == 0.0

    def test_trainset_uses_mid_prefix(self) -> None:
        """build_dspy_trainset picks a mid-point cumulative prefix."""
        mc = [_make_mc_question()]
        trainset = build_dspy_trainset(mc, max_examples=1)
        ex = trainset[0]
        # The question has 3 prefixes; mid = 3//2 = 1
        assert ex["clue_prefix"] == "Who was the first"
