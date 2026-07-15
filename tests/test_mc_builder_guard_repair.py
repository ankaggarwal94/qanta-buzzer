"""Regression tests for MCBuilder guard-preserving option repair."""

from __future__ import annotations

import random
import sys
from collections.abc import Iterable

import pytest

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.mc_builder import MCBuilder, build_mc_questions


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
    K: int = 4,
    max_repair_attempts: int = 10_000,
) -> tuple[list, MCBuilder]:
    reference = _reference(answers)
    profile_builder = AnswerProfileBuilder(min_questions_per_answer=1).fit(reference)
    builder = MCBuilder(
        K=K,
        strategy="tfidf_profile",
        random_seed=seed,
        max_repair_attempts=max_repair_attempts,
    )
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
    expected = ["Adolf Hitler", "Belgium", "Patrice Lumumba", "Austria"]
    assert built[0].options == expected
    assert built[0].option_answer_primary == expected
    assert built[0].gold_index == 1
    assert built[0].options[built[0].gold_index] == "Belgium"
    assert built[0].option_profiles == [
        f"training clue for {answer}" for answer in expected
    ]
    repair = builder.last_build_stats["repair"]
    assert repair["attempted_questions"] == 1
    assert repair["succeeded_questions"] == 1
    assert repair["ranked_successes"] == 1
    assert repair["fallback_successes"] == 0
    assert repair["length_ratio_triggers"] == 1


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
    repair = builder.last_build_stats["repair"]
    assert repair["attempted_questions"] == 1
    assert repair["failed_questions"] == 1
    assert repair["unrecoverable_gold_overlap_questions"] == 1
    assert repair["candidate_attempts"] == 0


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

    reference = _reference(answers)
    profile_builder = AnswerProfileBuilder(min_questions_per_answer=1).fit(reference)
    builder = MCBuilder(K=4, strategy="tfidf_profile", random_seed=13)
    builder._compute_rankings = lambda _answers, _profiles, _categories: rankings
    original_repair = builder._repair_options_after_guard_failure
    repair_rng_states: list[tuple[object, object]] = []

    def repair_with_rng_probe(*args, **kwargs):
        before = builder.rng.getstate()
        result = original_repair(*args, **kwargs)
        repair_rng_states.append((before, builder.rng.getstate()))
        return result

    builder._repair_options_after_guard_failure = repair_with_rng_probe
    built = builder.build(targets, profile_builder, reference_questions=reference)

    assert len(built) == 2
    assert builder.last_build_stats["drop_reasons"] == {}
    assert built[1].qid == "target-alpha"
    assert built[1].options == expected_second_options
    assert repair_rng_states
    assert all(before == after for before, after in repair_rng_states)
    assert builder.last_build_stats["repair"]["fallback_successes"] == 1


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


def test_repair_budget_is_shared_across_ranked_and_fallback_searches() -> None:
    """Fallback must not receive a fresh budget after ranked exhaustion.

    The ranked-only A/A/A/B pool has no compatible triple and takes exactly
    five candidate-extension attempts to exhaust.  Adding fallback-only C
    makes A/B/C succeed on attempt five if the second pass incorrectly resets
    the budget.  With one shared budget, the fallback pass fails closed before
    evaluating C.
    """
    gold = "gold answer has six unique tokens"
    poison = "poison phrase appears right inside clue"
    a_family = [
        "alpha shared first second third a0",
        "alpha shared first second third a1",
        "alpha shared first second third a2",
    ]
    b_only = "bravo cluster fourth fifth sixth b0"
    fallback_only = "charlie group seventh eighth ninth c0"
    answers = [gold, poison, *a_family, b_only, fallback_only]
    rankings = {gold: [poison, *a_family, b_only]}
    target = [
        _make_question(
            "target-budget",
            gold,
            f"This clue explicitly contains {poison} verbatim.",
        )
    ]

    built, builder = _build_with_rankings(
        target,
        answers,
        rankings,
        max_repair_attempts=5,
    )

    assert built == []
    assert builder.last_build_stats["drop_reasons"] == {
        "repair_budget_exhausted": 1
    }
    assert builder.last_build_stats["repair"] == {
        "attempted_questions": 1,
        "succeeded_questions": 0,
        "ranked_successes": 0,
        "fallback_successes": 0,
        "budget_exhausted_questions": 1,
        "candidate_attempts": 5,
        "length_ratio_triggers": 0,
        "question_overlap_triggers": 1,
        "simultaneous_guard_triggers": 0,
        "failed_questions": 1,
        "exhaustive_no_solution_questions": 0,
        "unrecoverable_gold_overlap_questions": 0,
    }


def test_solution_on_final_budgeted_attempt_succeeds() -> None:
    """Attempt B is allowed; only a request for attempt B+1 exhausts B."""
    gold = "Rome"
    poison = "Carthage"
    too_long = "Very Long Distractor Name"
    valid = "Athens"
    answers = [gold, poison, too_long, valid]
    rankings = {gold: [poison, too_long, valid]}
    target = [
        _make_question(
            "target-exact-budget",
            gold,
            "This clue explicitly names Carthage.",
        )
    ]

    built, builder = _build_with_rankings(
        target,
        answers,
        rankings,
        K=2,
        max_repair_attempts=2,
    )

    assert len(built) == 1
    assert set(built[0].options) == {gold, valid}
    repair = builder.last_build_stats["repair"]
    assert repair["candidate_attempts"] == 2
    assert repair["succeeded_questions"] == 1
    assert repair["budget_exhausted_questions"] == 0


def test_category_random_skips_redundant_fallback_search() -> None:
    """A full-universe category ranking must not be searched twice."""
    gold = "Zeta"
    distractors = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    answers = [gold, *distractors]
    reference = _reference(answers)
    profile_builder = AnswerProfileBuilder(min_questions_per_answer=1).fit(reference)
    builder = MCBuilder(
        K=4,
        strategy="category_random",
        random_seed=13,
        max_repair_attempts=7,
    )
    original_search = builder._search_repaired_options
    search_calls = 0

    def counted_search(*args, **kwargs):
        nonlocal search_calls
        search_calls += 1
        return original_search(*args, **kwargs)

    builder._search_repaired_options = counted_search
    target = [
        _make_question(
            "target-category-random",
            gold,
            "This clue names Alpha, Beta, Gamma, Delta, and Epsilon.",
        )
    ]

    built = builder.build(target, profile_builder, reference_questions=reference)

    assert built == []
    assert search_calls == 1
    assert builder.last_build_stats["drop_reasons"] == {
        "guard_repair_failed": 1
    }
    repair = builder.last_build_stats["repair"]
    assert repair["candidate_attempts"] == 0
    assert repair["exhaustive_no_solution_questions"] == 1


def test_large_legal_k_fails_closed_without_recursion_error() -> None:
    """Repair safety must not depend on Python's recursion limit."""
    target_k = sys.getrecursionlimit() + 50
    gold = "Gold"
    poison = "Poison"
    viable = [f"candidate-{idx}" for idx in range(target_k - 1)]
    answers = [gold, poison, *viable]
    rankings = {gold: [poison, *viable]}
    target = [
        _make_question(
            "target-large-k",
            gold,
            "This clue explicitly names Poison.",
        )
    ]

    built, builder = _build_with_rankings(
        target,
        answers,
        rankings,
        K=target_k,
        max_repair_attempts=32,
    )

    assert built == []
    assert builder.last_build_stats["drop_reasons"] == {
        "repair_budget_exhausted": 1
    }
    repair = builder.last_build_stats["repair"]
    assert repair["candidate_attempts"] == 32
    assert repair["budget_exhausted_questions"] == 1


def test_simultaneous_guard_failure_has_joint_diagnostics() -> None:
    """A joint failure must record both triggers without false attribution."""
    long_distractor = "Very Long Distractor Name"
    answers = ["Rome", long_distractor]
    rankings = {"Rome": [long_distractor]}
    target = [
        _make_question(
            "target-simultaneous",
            "Rome",
            f"This clue explicitly names {long_distractor}.",
        )
    ]

    built, builder = _build_with_rankings(target, answers, rankings, K=2)

    assert built == []
    assert builder.last_build_stats["drop_reasons"] == {
        "guard_repair_failed": 1
    }
    assert builder.last_build_stats["repair"] == {
        "attempted_questions": 1,
        "succeeded_questions": 0,
        "ranked_successes": 0,
        "fallback_successes": 0,
        "budget_exhausted_questions": 0,
        "candidate_attempts": 0,
        "length_ratio_triggers": 1,
        "question_overlap_triggers": 1,
        "simultaneous_guard_triggers": 1,
        "failed_questions": 1,
        "exhaustive_no_solution_questions": 1,
        "unrecoverable_gold_overlap_questions": 0,
    }


def test_joint_search_failure_is_not_misattributed_to_initial_guard() -> None:
    """No joint solution is not evidence that the initial guard was causal."""
    long_distractor = "Very Long Distractor Name"
    answers = ["Rome", long_distractor, "Athens"]
    rankings = {"Rome": [long_distractor, "Athens"]}
    target = [
        _make_question(
            "target-ambiguous",
            "Rome",
            "This polis rivaled Athens.",
        )
    ]

    built, builder = _build_with_rankings(target, answers, rankings, K=2)

    assert built == []
    assert builder.last_build_stats["drop_reasons"] == {
        "guard_repair_failed": 1
    }
    repair = builder.last_build_stats["repair"]
    assert repair["length_ratio_triggers"] == 1
    assert repair["question_overlap_triggers"] == 0
    assert repair["simultaneous_guard_triggers"] == 0
    assert repair["candidate_attempts"] == 1
    assert repair["exhaustive_no_solution_questions"] == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"max_repair_attempts": 0},
            "max_repair_attempts must be a positive integer",
        ),
        (
            {"max_repair_attempts": float("inf")},
            "max_repair_attempts must be a positive integer",
        ),
        (
            {"max_repair_attempts": float("nan")},
            "max_repair_attempts must be a positive integer",
        ),
        (
            {"max_repair_attempts": 1.5},
            "max_repair_attempts must be a positive integer",
        ),
        (
            {"max_repair_attempts": True},
            "max_repair_attempts must be a positive integer",
        ),
        (
            {"variable_K": True, "min_K": 5, "max_K": 4},
            "max_K must be >= min_K",
        ),
    ],
)
def test_invalid_repair_and_variable_k_configuration_fails_early(
    kwargs: dict,
    message: str,
) -> None:
    """Invalid bounds should fail at construction, not during a build."""
    with pytest.raises(ValueError, match=message):
        MCBuilder(K=4, **kwargs)


def test_factory_does_not_coerce_an_invalid_repair_budget() -> None:
    """The guards mapping must preserve constructor validation semantics."""
    with pytest.raises(
        ValueError,
        match="max_repair_attempts must be a positive integer",
    ):
        build_mc_questions(
            questions=[],
            K=4,
            strategy="tfidf_profile",
            profile_builder=AnswerProfileBuilder(),
            guards={"max_repair_attempts": 1.5},
        )
