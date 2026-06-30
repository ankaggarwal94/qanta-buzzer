"""Regression tests for MCBuilder guard-preserving option repair."""

from __future__ import annotations

import random
from collections.abc import Iterable

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.mc_builder import MCBuilder


def _make_question(qid: str, answer: str, question: str | None = None) -> TossupQuestion:
    text = question or f"training clue for {answer}"
    tokens = text.split()
    run_indices = list(range(len(tokens)))
    return TossupQuestion(
        qid=qid,
        question=text,
        tokens=tokens,
        answer_primary=answer,
        clean_answers=[answer],
        run_indices=run_indices,
        human_buzz_positions=None,
        category="History",
        cumulative_prefixes=[" ".join(tokens[: i + 1]) for i in run_indices],
    )


def _reference(answers: Iterable[str]) -> list[TossupQuestion]:
    return [
        _make_question(f"ref-{idx}", answer)
        for idx, answer in enumerate(answers)
    ]


def _build_with_rankings(
    target: list[TossupQuestion],
    answers: list[str],
    rankings: dict[str, list[str]],
    *,
    seed: int = 13,
) -> tuple[list, MCBuilder]:
    reference = _reference(answers)
    profile_builder = AnswerProfileBuilder(min_questions_per_answer=1).fit(reference)
    builder = MCBuilder(K=4, strategy="tfidf_profile", random_seed=seed)
    builder._compute_rankings = lambda _answers, _profiles, _categories: rankings
    return builder.build(target, profile_builder, reference_questions=reference), builder


def test_length_ratio_guard_searches_later_ranked_replacements() -> None:
    """A first ranked option set that fails length guard should be repaired."""
    answers = [
        "Belgium",
        "Night of the Long Knives",
        "Patrice Lumumba",
        "Charles XII of Sweden",
        "Austria",
        "Adolf Hitler",
    ]
    rankings = {
        "Belgium": [
            "Night of the Long Knives",
            "Patrice Lumumba",
            "Charles XII of Sweden",
            "Austria",
            "Adolf Hitler",
        ]
    }
    target = [
        _make_question(
            "target-belgium",
            "Belgium",
            "This clue asks about a European country without naming any option.",
        )
    ]

    built, builder = _build_with_rankings(target, answers, rankings)

    assert len(built) == 1
    assert builder.last_build_stats["drop_reasons"] == {}
    assert set(built[0].option_answer_primary) == {
        "Belgium",
        "Patrice Lumumba",
        "Austria",
        "Adolf Hitler",
    }


def test_question_overlap_guard_searches_later_ranked_replacements() -> None:
    """A distractor appearing in the clue should be replaced when possible."""
    answers = [
        "Mao Zedong",
        "White Lotus Rebellion",
        "Chiang Kai-shek",
        "Pol Pot",
        "Deng Xiaoping",
    ]
    rankings = {
        "Mao Zedong": [
            "White Lotus Rebellion",
            "Chiang Kai-shek",
            "Pol Pot",
            "Deng Xiaoping",
        ]
    }
    target = [
        _make_question(
            "target-mao",
            "Mao Zedong",
            "This leader's forces fought Chiang Kai-shek during a civil war.",
        )
    ]

    built, builder = _build_with_rankings(target, answers, rankings)

    assert len(built) == 1
    assert builder.last_build_stats["drop_reasons"] == {}
    assert set(built[0].option_answer_primary) == {
        "Mao Zedong",
        "White Lotus Rebellion",
        "Pol Pot",
        "Deng Xiaoping",
    }


def test_gold_overlap_remains_unrecoverable() -> None:
    """Repair must not bypass question overlap when the gold itself leaks."""
    answers = ["Rome", "Carthage", "Athens", "Sparta", "Thebes"]
    rankings = {"Rome": ["Carthage", "Athens", "Sparta", "Thebes"]}
    target = [
        _make_question(
            "target-rome",
            "Rome",
            "This clue explicitly names Rome before the answer line.",
        )
    ]

    built, builder = _build_with_rankings(target, answers, rankings)

    assert built == []
    assert builder.last_build_stats["drop_reasons"] == {
        "question_overlap_guard": 1
    }


def test_repair_uses_cloned_fallback_order_without_perturbing_rng() -> None:
    """Would-be fallback search must not consume the live builder RNG."""
    answers = [
        "Belgium",
        "Night of the Long Knives",
        "Charles XII of Sweden",
        "War of the Spanish Succession",
        "Austria",
        "Adolf Hitler",
        "France",
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
    ]
    rankings = {
        "Belgium": [
            "Night of the Long Knives",
            "Charles XII of Sweden",
            "War of the Spanish Succession",
        ],
        "Alpha": ["Beta", "Gamma", "Delta"],
    }
    targets = [
        _make_question(
            "target-belgium",
            "Belgium",
            "This clue asks about a European country without naming any option.",
        ),
        _make_question(
            "target-alpha",
            "Alpha",
            "This second question checks deterministic option order.",
        ),
    ]
    expected_rng = random.Random(13)
    first_failed_options = [
        "Belgium",
        "Night of the Long Knives",
        "Charles XII of Sweden",
        "War of the Spanish Succession",
    ]
    expected_rng.shuffle(first_failed_options)
    expected_second_options = ["Alpha", "Beta", "Gamma", "Delta"]
    expected_rng.shuffle(expected_second_options)

    built, builder = _build_with_rankings(targets, answers, rankings)

    assert len(built) == 2
    assert builder.last_build_stats["drop_reasons"] == {}
    assert built[1].qid == "target-alpha"
    assert built[1].options == expected_second_options


def test_repair_handles_option_set_that_already_used_fallback() -> None:
    """Repair must work when the failed set includes a fallback distractor."""
    answers = [
        "Belgium",
        "Patrice Lumumba",
        "Austria",
        "Night of the Long Knives",
        "Adolf Hitler",
        "France",
        "Spain",
    ]
    rankings = {
        "Belgium": [
            "Patrice Lumumba",
            "Austria",
        ]
    }
    selected_after_ranked = ["Patrice Lumumba", "Austria"]
    fallback = sorted(
        answer
        for answer in answers
        if answer not in selected_after_ranked and answer != "Belgium"
    )
    random.Random(13).shuffle(fallback)
    first_fallback = fallback[0]
    target = [
        _make_question(
            "target-belgium-fallback",
            "Belgium",
            f"This clue names {first_fallback} so the first fallback set fails.",
        )
    ]

    built, builder = _build_with_rankings(target, answers, rankings)

    assert len(built) == 1
    assert builder.last_build_stats["drop_reasons"] == {}
    assert first_fallback not in built[0].option_answer_primary
    assert set(built[0].option_answer_primary) >= {
        "Belgium",
        "Patrice Lumumba",
        "Austria",
    }
