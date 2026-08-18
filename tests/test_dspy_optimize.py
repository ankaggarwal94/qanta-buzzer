"""Tests for scripts/optimize_dspy.py — offline DSPy compilation."""

from __future__ import annotations

import pytest

from scripts.optimize_dspy import build_dspy_trainset, _score_metric


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

        example = MagicMock()
        example.scores = json.dumps([0.0, 1.0, 0.0])
        pred_correct = MagicMock()
        pred_correct.scores = json.dumps([0.1, 0.8, 0.1])
        pred_wrong = MagicMock()
        pred_wrong.scores = json.dumps([0.9, 0.05, 0.05])
        pred_malformed = MagicMock()
        pred_malformed.scores = "not json"

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


class TestScoreMetricValidation:
    """PR #31 external review: _score_metric must reject malformed predictions
    (previously a short/long/non-numeric prediction could argmax-match
    independently and score 1.0; a JSON object raised an uncaught KeyError)."""

    class _P:
        def __init__(self, scores: str) -> None:
            self.scores = scores

    def test_length_mismatch_scores_zero(self) -> None:
        assert _score_metric(self._P("[0.9, 0.1, 0.0]"), self._P("[0.9]")) == 0.0
        assert _score_metric(self._P("[1.0]"), self._P("[0.1, 0.2, 0.3]")) == 0.0

    def test_json_object_scores_zero_not_keyerror(self) -> None:
        assert _score_metric(self._P('{"a": 1}'), self._P("[1.0, 0.0]")) == 0.0
        assert _score_metric(self._P("[1.0, 0.0]"), self._P('{"a": 1}')) == 0.0

    def test_null_or_nonfinite_entries_score_zero(self) -> None:
        assert _score_metric(self._P("[1.0, 0.0]"), self._P("[null, 0.0]")) == 0.0
        assert _score_metric(self._P("[Infinity, 0.0]"), self._P("[1.0, 0.0]")) == 0.0

    def test_valid_match_and_mismatch_unchanged(self) -> None:
        assert _score_metric(self._P("[0.1, 0.9]"), self._P("[0.2, 0.8]")) == 1.0
        assert _score_metric(self._P("[0.1, 0.9]"), self._P("[0.8, 0.2]")) == 0.0


class TestScoreMetricOverflowSafety:
    """PR #31 external review R3: an un-float-able big int (e.g. 400-digit LM
    score) must score 0.0, not crash the metric via math.isfinite OverflowError."""

    class _P:
        def __init__(self, scores: str) -> None:
            self.scores = scores

    def test_oversized_integer_scores_zero_not_overflow(self) -> None:
        huge = "1" + "0" * 400
        assert _score_metric(self._P(f"[{huge}, 0]"), self._P("[1, 0]")) == 0.0
        assert _score_metric(self._P("[1, 0]"), self._P(f"[{huge}, 0]")) == 0.0
