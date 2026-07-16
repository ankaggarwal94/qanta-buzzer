"""Regression tests for MCBuilder guard-preserving option repair."""

from __future__ import annotations

import random
import sys
from bisect import bisect_right
from collections.abc import Iterable

import pytest

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.mc_builder import MCBuilder, build_mc_questions
from qb_data.text_utils import normalize_answer


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


def _repair_directly(
    builder: MCBuilder,
    *,
    gold: str,
    selected: list[str],
    ranked: list[str],
    answers: list[str],
    target_k: int,
    question: str = "an unrelated clue",
    fallback_selected: list[str] | None = None,
    fallback_order: list[str] | None = None,
):
    """Call the private repair seam with a stable identity permutation."""
    ordered_options = [gold] + selected[: target_k - 1]
    return builder._repair_options_after_guard_failure(
        question=question,
        gold=gold,
        selected=selected,
        ranked=ranked,
        answers=answers,
        gold_aliases=[gold],
        gold_norms={str(normalize_answer(gold))},
        target_k=target_k,
        shuffled_options=ordered_options,
        fallback_rng_state=builder.rng.getstate(),
        fallback_selected=fallback_selected or [],
        fallback_order=fallback_order,
    )


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


def test_repair_caps_raw_preprocessing_before_expensive_candidate_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 20k ranking may trigger at most B alias/normalization scans."""
    scan_limit = 7
    gold = "Gold"
    question = "blocked"
    ranked = [f"candidate-{idx:05d}" for idx in range(20_000)]
    builder = MCBuilder(K=2, max_repair_attempts=scan_limit)
    real_alias_check = builder._aliases_collide
    alias_checks: list[str] = []
    candidate_normalizations: list[str] = []

    def normalize_spy(value: str):
        if value == question:
            return "blocked"
        if value == gold:
            return "gold"
        if value.startswith("candidate-"):
            candidate_normalizations.append(value)
            return "blocked"
        return normalize_answer(value)

    def alias_spy(candidate, gold_aliases, _gold_norms=None):
        alias_checks.append(candidate)
        return real_alias_check(candidate, gold_aliases, _gold_norms)

    monkeypatch.setattr("qb_data.mc_builder.normalize_answer", normalize_spy)
    monkeypatch.setattr(builder, "_aliases_collide", alias_spy)

    result = _repair_directly(
        builder,
        gold=gold,
        selected=[ranked[0]],
        ranked=ranked,
        answers=[gold, *ranked],
        target_k=2,
        question=question,
        fallback_order=[],
    )

    assert result.options is None
    assert result.candidate_scans == scan_limit
    assert result.candidate_attempts == 0
    assert result.budget_exhausted is True
    assert alias_checks == ranked[:scan_limit]
    # Once in the alias guard and once in the question-overlap filter.
    assert len(candidate_normalizations) == 2 * scan_limit


def test_hypothetical_fallback_is_bounded_deterministic_and_rng_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback samples only remaining capacity using a cloned RNG."""
    scan_limit = 5
    gold = "Gold"
    universe = [f"candidate-{idx:05d}" for idx in range(20_000)]
    answers = [gold, *universe]
    ranked = [universe[0]]
    builder = MCBuilder(K=2, random_seed=13, max_repair_attempts=scan_limit)
    monkeypatch.setattr(builder, "_aliases_collide", lambda *_args, **_kwargs: True)
    original_bounded_fallback = builder._bounded_fallback_order_from_state
    fallback_calls: list[tuple[int, int, list[str], bool]] = []

    def bounded_fallback_spy(
        fallback_answers,
        selected,
        fallback_gold,
        rng_state,
        limit,
    ):
        sampled, exhausted = original_bounded_fallback(
            fallback_answers,
            selected,
            fallback_gold,
            rng_state,
            limit,
        )
        fallback_calls.append(
            (len(fallback_answers), limit, sampled[:], exhausted)
        )
        return sampled, exhausted

    def forbidden_shuffle(*_args, **_kwargs):
        raise AssertionError("repair fallback must not shuffle the full universe")

    monkeypatch.setattr(
        builder,
        "_bounded_fallback_order_from_state",
        bounded_fallback_spy,
    )
    monkeypatch.setattr(random.Random, "shuffle", forbidden_shuffle)
    before = builder.rng.getstate()

    first = _repair_directly(
        builder,
        gold=gold,
        selected=ranked,
        ranked=ranked,
        answers=answers,
        target_k=2,
        fallback_selected=ranked,
        fallback_order=None,
    )
    second = _repair_directly(
        builder,
        gold=gold,
        selected=ranked,
        ranked=ranked,
        answers=answers,
        target_k=2,
        fallback_selected=ranked,
        fallback_order=None,
    )

    assert builder.rng.getstate() == before
    assert first == second
    assert first.candidate_scans == scan_limit
    assert first.candidate_attempts == 0
    assert first.budget_exhausted is True
    assert len(fallback_calls) == 2
    assert all(call[:2] == (len(answers), scan_limit - 1) for call in fallback_calls)
    assert all(len(call[2]) == scan_limit - 1 for call in fallback_calls)
    assert all(call[3] is False for call in fallback_calls)
    assert fallback_calls[0][2] == fallback_calls[1][2]


def test_bounded_fallback_binary_search_preserves_sample_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many exclusions use one logarithmic rank lookup per sampled answer."""
    exclusion_count = 1_000
    answers = [f"answer-{idx:05d}" for idx in range(2 * exclusion_count)]
    selected = answers[:exclusion_count]
    eligible = answers[exclusion_count:]
    builder = MCBuilder(K=2, random_seed=13)
    rng_state = builder.rng.getstate()

    expected_rng = random.Random()
    expected_rng.setstate(rng_state)
    expected_ranks = expected_rng.sample(range(len(eligible)), len(eligible))
    expected = [eligible[rank] for rank in expected_ranks]

    lookup_calls = 0

    def counting_bisect_right(values, rank):
        nonlocal lookup_calls
        lookup_calls += 1
        return bisect_right(values, rank)

    monkeypatch.setattr(
        "qb_data.mc_builder.bisect_right",
        counting_bisect_right,
    )
    before = builder.rng.getstate()

    sampled, exhaustive = builder._bounded_fallback_order_from_state(
        answers,
        selected,
        selected[0],
        rng_state,
        len(eligible),
    )

    assert sampled == expected
    assert exhaustive is True
    assert builder.rng.getstate() == before
    assert lookup_calls == len(eligible)


def test_fallback_does_not_replay_ranked_work_at_budget_two() -> None:
    """The second distinct K=2 candidate must fit in a budget of two."""
    gold = "Rome"
    too_long = "Very Long Distractor Name"
    valid = "Athens"
    answers = [gold, too_long, valid]
    rankings = {gold: [too_long]}
    target = [
        _make_question(
            "target-no-replay",
            gold,
            "This clue does not name an option.",
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
    assert repair["candidate_scans"] == 2
    assert repair["candidate_attempts"] == 2
    assert repair["fallback_successes"] == 1
    assert repair["budget_exhausted_questions"] == 0


def test_bounded_fallback_excludes_every_ranked_candidate_already_scanned() -> None:
    """Remaining scan capacity must be spent only on unseen fallback entries."""
    gold = "Rome"
    first = "Very Long Ranked Distractor One"
    second = "Very Long Ranked Distractor Two"
    valid = "Athens"
    builder = MCBuilder(K=2, random_seed=13, max_repair_attempts=3)

    result = _repair_directly(
        builder,
        gold=gold,
        selected=[first],
        ranked=[first, second],
        answers=sorted([gold, first, second, valid]),
        target_k=2,
        fallback_selected=[first],
        fallback_order=None,
    )

    assert result.options == [gold, valid]
    assert result.source == "fallback"
    assert result.candidate_scans == 3
    assert result.candidate_attempts == 3
    assert result.budget_exhausted is False


def test_materialized_fallback_does_not_charge_seen_ranked_candidates() -> None:
    """An existing fallback order must spend its scan slot on a new entry."""
    gold = "Rome"
    first = "Very Long Ranked Distractor One"
    second = "Very Long Ranked Distractor Two"
    valid = "Athens"
    builder = MCBuilder(K=2, max_repair_attempts=3)

    result = _repair_directly(
        builder,
        gold=gold,
        selected=[first],
        ranked=[first, second],
        answers=sorted([gold, first, second, valid]),
        target_k=2,
        fallback_selected=[first],
        fallback_order=[second, valid],
    )

    assert result.options == [gold, valid]
    assert result.source == "fallback"
    assert result.candidate_scans == 3
    assert result.candidate_attempts == 3
    assert result.budget_exhausted is False


def test_transition_memo_is_prefix_sensitive() -> None:
    """A candidate rejected under A may still be valid at the root."""
    gold = "gold answer"
    first = "alpha beta"
    second = "alpha beta red"
    fallback = "alpha beta blue"
    builder = MCBuilder(K=3, max_repair_attempts=5)

    result = _repair_directly(
        builder,
        gold=gold,
        selected=[first, fallback],
        ranked=[first, second],
        answers=[gold, first, second, fallback],
        target_k=3,
        fallback_selected=[first],
        fallback_order=[fallback],
    )

    assert result.options == [gold, second, fallback]
    assert result.source == "fallback"
    assert result.candidate_scans == 3
    assert result.candidate_attempts == 5
    assert result.budget_exhausted is False


def test_ranked_only_solution_remains_preferred_to_earlier_mixed_solution() -> None:
    """Avoiding replay must not change the ranked-first preference contract."""
    gold = "gold answer"
    first = "alpha beta"
    second = "alpha beta red"
    third = "alpha beta blue"
    fallback = "charlie delta"
    builder = MCBuilder(K=3, max_repair_attempts=10)

    result = _repair_directly(
        builder,
        gold=gold,
        selected=[first, fallback],
        ranked=[first, second, third],
        answers=[gold, first, second, third, fallback],
        target_k=3,
        fallback_selected=[first],
        fallback_order=[fallback],
    )

    assert result.options == [gold, second, third]
    assert result.source == "ranked"
    assert result.candidate_scans == 3
    assert result.candidate_attempts == 5


def test_solution_on_final_candidate_scan_succeeds_even_if_source_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate B may succeed; only needing candidate B+1 is truncation."""
    scan_limit = 4
    gold = "Rome"
    blocked = [f"blocked-{idx}" for idx in range(scan_limit - 1)]
    valid = "Athens"
    ranked = [*blocked, valid, "unscanned-tail"]
    builder = MCBuilder(K=2, max_repair_attempts=scan_limit)
    monkeypatch.setattr(
        builder,
        "_aliases_collide",
        lambda candidate, *_args, **_kwargs: candidate != valid,
    )

    result = _repair_directly(
        builder,
        gold=gold,
        selected=[blocked[0]],
        ranked=ranked,
        answers=[gold, *ranked],
        target_k=2,
        fallback_order=[],
    )

    assert result.options == [gold, valid]
    assert result.source == "ranked"
    assert result.candidate_scans == scan_limit
    assert result.candidate_attempts == 1
    assert result.budget_exhausted is False


@pytest.mark.parametrize(
    ("has_unseen_fallback", "expected_exhausted"),
    [(False, False), (True, True)],
)
def test_full_ranked_scan_distinguishes_exhaustive_from_truncated_fallback(
    monkeypatch: pytest.MonkeyPatch,
    has_unseen_fallback: bool,
    expected_exhausted: bool,
) -> None:
    """At scan B, only an unexamined candidate universe is truncation."""
    scan_limit = 4
    gold = "Gold"
    ranked = [f"ranked-{idx}" for idx in range(scan_limit)]
    answers = sorted(
        [gold, *ranked]
        + (["unseen-fallback"] if has_unseen_fallback else [])
    )
    builder = MCBuilder(K=2, max_repair_attempts=scan_limit)
    monkeypatch.setattr(builder, "_aliases_collide", lambda *_args, **_kwargs: True)

    result = _repair_directly(
        builder,
        gold=gold,
        selected=[ranked[0]],
        ranked=ranked,
        answers=answers,
        target_k=2,
        fallback_selected=[ranked[0]],
        fallback_order=None,
    )

    assert result.options is None
    assert result.candidate_scans == scan_limit
    assert result.candidate_attempts == 0
    assert result.budget_exhausted is expected_exhausted


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
        "candidate_scans": 5,
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
    too_long = "Very Long Distractor Name"
    valid = "Athens"
    answers = [gold, too_long, valid]
    rankings = {gold: [too_long, valid]}
    target = [
        _make_question(
            "target-exact-budget",
            gold,
            "This clue does not name either option.",
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
    assert repair["candidate_scans"] == 2
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
    assert repair["candidate_attempts"] == 0
    assert repair["candidate_scans"] == 0
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
        "candidate_scans": 1,
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("K", True, "K must be an integer"),
        ("K", 4.0, "K must be an integer"),
        ("K", float("inf"), "K must be an integer"),
        ("K", float("nan"), "K must be an integer"),
        ("min_K", True, "min_K must be an integer"),
        ("min_K", 2.0, "min_K must be an integer"),
        ("min_K", float("inf"), "min_K must be an integer"),
        ("min_K", float("nan"), "min_K must be an integer"),
        ("max_K", True, "max_K must be an integer or None"),
        ("max_K", 4.0, "max_K must be an integer or None"),
        ("max_K", float("inf"), "max_K must be an integer or None"),
        ("max_K", float("nan"), "max_K must be an integer or None"),
    ],
)
def test_choice_count_bounds_require_strict_integers(
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject booleans and numeric lookalikes before any comparisons."""
    kwargs = {field: value}
    if field != "K":
        kwargs["K"] = 4
    with pytest.raises(ValueError, match=message):
        MCBuilder(**kwargs)


def test_choice_count_integer_compatibility_is_preserved() -> None:
    """Keep min-K clamping, max-K defaulting, and fixed-K behavior."""
    defaulted = MCBuilder(K=4, min_K=1, max_K=None)
    assert defaulted.min_K == 2
    assert defaulted.max_K == 4
    assert defaulted._target_k() == 4

    fixed = MCBuilder(K=4, variable_K=False, min_K=5, max_K=3)
    assert fixed._target_k() == 4

    variable = MCBuilder(
        K=4,
        variable_K=True,
        min_K=2,
        max_K=4,
        random_seed=13,
    )
    assert 2 <= variable._target_k() <= 4


@pytest.mark.parametrize("value", ["false", "true", "yes", 0, 1, None])
def test_variable_k_requires_a_boolean(value: object) -> None:
    """Reject truthy and falsy lookalikes at the behavior-owning boundary."""
    with pytest.raises(ValueError, match="variable_K must be a boolean"):
        MCBuilder(K=4, variable_K=value)


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
