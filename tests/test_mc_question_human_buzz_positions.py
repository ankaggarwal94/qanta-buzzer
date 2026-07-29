"""Regression tests for MCQuestion human-buzz-position deserialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from qb_data.mc_builder import MCQuestion
from scripts._common import load_mc_questions, mc_question_from_dict, save_json


def _question(
    human_buzz_positions: list[tuple[int, int]] | None,
) -> MCQuestion:
    """Return a schema-complete MCQuestion with configurable buzz metadata."""
    return MCQuestion(
        qid="q-human-buzz",
        question="alpha beta",
        tokens=["alpha", "beta"],
        answer_primary="Alpha",
        clean_answers=["Alpha"],
        run_indices=[0, 1],
        human_buzz_positions=human_buzz_positions,
        category="Test",
        cumulative_prefixes=["alpha", "alpha beta"],
        options=["Alpha", "Beta"],
        gold_index=0,
        option_profiles=["alpha profile", "beta profile"],
        option_answer_primary=["Alpha", "Beta"],
        distractor_strategy="test",
    )


@pytest.mark.parametrize("payload_shape", ["plain", "wrapped"])
@pytest.mark.parametrize(
    "human_buzz_positions",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty"),
        pytest.param([(5, 1)], id="one-pair"),
        pytest.param([(5, 1), (12, 2)], id="multiple-pairs"),
    ],
)
def test_human_buzz_positions_json_roundtrip_preserves_schema(
    tmp_path: Path,
    payload_shape: str,
    human_buzz_positions: list[tuple[int, int]] | None,
) -> None:
    """Both supported payload shapes must round-trip the full dataclass."""
    original = _question(human_buzz_positions)
    payload: object
    if payload_shape == "plain":
        payload = [original]
    else:
        payload = {"metadata": {"producer": "test"}, "questions": [original]}
    path = tmp_path / f"{payload_shape}.json"
    save_json(path, payload)

    loaded = load_mc_questions(path)

    assert loaded == [original]
    if loaded[0].human_buzz_positions is not None:
        assert all(
            type(pair) is tuple for pair in loaded[0].human_buzz_positions
        )


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param([[1, 2, 3]], id="wrong-arity"),
        pytest.param([7], id="scalar-entry"),
        pytest.param([["not-an-integer", 2]], id="non-integer-component"),
    ],
)
def test_malformed_human_buzz_positions_fail_with_index_context(
    malformed: object,
) -> None:
    """Malformed JSON must fail at the exact field entry, not drift in type."""
    row = {
        "qid": "q-human-buzz",
        "question": "alpha beta",
        "tokens": ["alpha", "beta"],
        "answer_primary": "Alpha",
        "clean_answers": ["Alpha"],
        "run_indices": [0, 1],
        "human_buzz_positions": malformed,
        "category": "Test",
        "cumulative_prefixes": ["alpha", "alpha beta"],
        "options": ["Alpha", "Beta"],
        "gold_index": 0,
        "option_profiles": ["alpha profile", "beta profile"],
        "option_answer_primary": ["Alpha", "Beta"],
        "distractor_strategy": "test",
    }

    with pytest.raises(ValueError, match=r"human_buzz_positions\[0\]"):
        mc_question_from_dict(row)
