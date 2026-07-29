"""Regressions for explicit empty-reference MC construction."""

from __future__ import annotations

import pytest

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.mc_builder import MCBuilder


def _question(qid: str, answer: str, text: str) -> TossupQuestion:
    tokens = text.split()
    return TossupQuestion(
        qid=qid,
        question=text,
        tokens=tokens,
        answer_primary=answer,
        clean_answers=[answer],
        run_indices=list(range(len(tokens))),
        human_buzz_positions=None,
        category="History",
        cumulative_prefixes=[
            " ".join(tokens[: index + 1]) for index in range(len(tokens))
        ],
    )


@pytest.mark.parametrize(
    "strategy",
    ["tfidf_profile", "sbert_profile", "openai_profile", "category_random"],
)
@pytest.mark.parametrize("variable_k", [False, True])
def test_nonempty_targets_with_empty_references_drop_consistently(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    variable_k: bool,
) -> None:
    targets = [
        _question("t0", "Alpha", "first unrelated target clue"),
        _question("t1", "Beta", "second unrelated target clue"),
    ]
    builder = MCBuilder(
        K=3,
        strategy=strategy,
        variable_K=variable_k,
        min_K=2,
        max_K=3,
        random_seed=13,
    )
    expected_rng = MCBuilder(
        K=3,
        strategy=strategy,
        variable_K=variable_k,
        min_K=2,
        max_K=3,
        random_seed=13,
    )

    def fail_if_rankings_are_computed(*_args: object) -> object:
        raise AssertionError(
            "an explicitly empty reference corpus must not invoke a "
            "ranking strategy or embedding backend"
        )

    monkeypatch.setattr(
        builder,
        "_compute_rankings",
        fail_if_rankings_are_computed,
    )
    built = builder.build(
        targets,
        AnswerProfileBuilder(),
        reference_questions=[],
    )
    for _target in targets:
        expected_rng._repair_rng.getrandbits(256)
        expected_rng._target_k()

    assert built == []
    assert builder.last_build_stats == {
        "target_questions": 2,
        "reference_questions": 0,
        "reference_answer_count": 0,
        "built_questions": 0,
        "dropped_questions": 2,
        "retention_rate": 0.0,
        "drop_reasons": {"unseen_gold_answer": 2},
        "repair": {
            "attempted_questions": 0,
            "succeeded_questions": 0,
            "ranked_successes": 0,
            "fallback_successes": 0,
            "budget_exhausted_questions": 0,
            "candidate_attempts": 0,
            "candidate_scans": 0,
            "length_ratio_triggers": 0,
            "question_overlap_triggers": 0,
            "simultaneous_guard_triggers": 0,
            "failed_questions": 0,
            "exhaustive_no_solution_questions": 0,
            "unrecoverable_gold_overlap_questions": 0,
        },
    }
    assert builder.rng.getstate() == expected_rng.rng.getstate()
    assert builder._repair_rng.getstate() == expected_rng._repair_rng.getstate()


def test_empty_reference_call_preserves_subsequent_variable_k_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_targets = [
        _question("empty0", "Missing", "first empty reference target"),
        _question("empty1", "Absent", "second empty reference target"),
    ]
    references = [
        _question("r0", "Alpha", "meridian quartz"),
        _question("r1", "Beta", "nebula cobalt"),
        _question("r2", "Gamma", "tundra saffron"),
    ]
    valid_targets = [
        _question("t0", "Alpha", "obscure historical prompt"),
        _question("t1", "Beta", "another historical prompt"),
    ]
    after_empty = MCBuilder(
        K=3,
        strategy="tfidf_profile",
        variable_K=True,
        min_K=2,
        max_K=3,
        random_seed=13,
    )
    advanced_control = MCBuilder(
        K=3,
        strategy="tfidf_profile",
        variable_K=True,
        min_K=2,
        max_K=3,
        random_seed=13,
    )

    real_compute_rankings = after_empty._compute_rankings

    def fail_if_rankings_are_computed(*_args: object) -> object:
        raise AssertionError(
            "an explicitly empty reference corpus must not invoke a "
            "ranking strategy or embedding backend"
        )

    monkeypatch.setattr(
        after_empty,
        "_compute_rankings",
        fail_if_rankings_are_computed,
    )
    after_empty.build(
        empty_targets,
        AnswerProfileBuilder(),
        reference_questions=[],
    )
    monkeypatch.setattr(
        after_empty,
        "_compute_rankings",
        real_compute_rankings,
    )
    for _target in empty_targets:
        advanced_control._repair_rng.getrandbits(256)
        advanced_control._target_k()

    actual = after_empty.build(
        valid_targets,
        AnswerProfileBuilder().fit(references),
        reference_questions=references,
    )
    expected = advanced_control.build(
        valid_targets,
        AnswerProfileBuilder().fit(references),
        reference_questions=references,
    )

    assert [
        (question.qid, question.options, question.gold_index)
        for question in actual
    ] == [
        (question.qid, question.options, question.gold_index)
        for question in expected
    ]
    assert after_empty.rng.getstate() == advanced_control.rng.getstate()
    assert (
        after_empty._repair_rng.getstate()
        == advanced_control._repair_rng.getstate()
    )
