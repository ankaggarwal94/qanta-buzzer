"""Regression tests for split-safe dataset construction."""

from __future__ import annotations

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.mc_builder import MCBuilder


def _make_question(
    qid: str,
    answer: str,
    question: str,
    category: str = "Science",
) -> TossupQuestion:
    """Construct a minimal TossupQuestion for MC builder tests."""
    tokens = question.split()
    run_indices = list(range(len(tokens)))
    return TossupQuestion(
        qid=qid,
        question=question,
        tokens=tokens,
        answer_primary=answer,
        clean_answers=[answer],
        run_indices=run_indices,
        human_buzz_positions=[],
        category=category,
        cumulative_prefixes=[" ".join(tokens[: i + 1]) for i in run_indices],
    )


def test_train_only_profiles_prevent_held_out_token_leakage() -> None:
    """Val/test option profiles must not absorb tokens from held-out questions."""
    raw_train = [
        _make_question("q1", "Hydrogen", "light gas on periodic table"),
        _make_question("q2", "Helium", "noble gas used in balloons"),
        _make_question("q3", "Lithium", "soft metal used in batteries"),
        _make_question("q4", "Carbon", "element central to organic chemistry"),
    ]
    heldout = _make_question(
        "q5",
        "Hydrogen",
        "LEAKTOKEN clue about the lightest element",
    )

    profile_builder = AnswerProfileBuilder(min_questions_per_answer=1).fit(raw_train)
    builder = MCBuilder(K=4, strategy="tfidf_profile", random_seed=7)

    mc_questions = builder.build(
        [heldout],
        profile_builder,
        reference_questions=raw_train,
    )

    assert len(mc_questions) == 1
    assert builder.last_build_stats["reference_questions"] == len(raw_train)
    assert builder.last_build_stats["reference_answer_count"] == 4
    assert all(
        "leaktoken" not in profile.lower()
        for profile in mc_questions[0].option_profiles
    )


def test_category_random_uses_train_only_answer_universe() -> None:
    """Held-out answers unseen in train are dropped instead of leaking in."""
    raw_train = [
        _make_question("q1", "Alpha", "first training clue", category="History"),
        _make_question("q2", "Beta", "second training clue", category="History"),
        _make_question("q3", "Gamma", "third training clue", category="History"),
        _make_question("q4", "Delta", "fourth training clue", category="History"),
    ]
    unseen = _make_question(
        "q5",
        "Epsilon",
        "held out answer unseen in training",
        category="History",
    )

    profile_builder = AnswerProfileBuilder(min_questions_per_answer=1).fit(raw_train)
    builder = MCBuilder(K=4, strategy="category_random", random_seed=11)

    mc_questions = builder.build(
        [unseen],
        profile_builder,
        reference_questions=raw_train,
    )

    assert mc_questions == []
    assert builder.last_build_stats["dropped_questions"] == 1
    assert builder.last_build_stats["drop_reasons"]["unseen_gold_answer"] == 1


def test_explicit_reference_questions_refit_stale_profile_builder() -> None:
    """Providing a new reference corpus should refit even if the builder is warm."""

    stale_reference = [
        _make_question("s1", "Mercury", "stale planet clue"),
        _make_question("s2", "Venus", "stale second planet clue"),
        _make_question("s3", "Earth", "stale third planet clue"),
        _make_question("s4", "Mars", "stale fourth planet clue"),
    ]
    raw_train = [
        _make_question("q1", "Hydrogen", "light gas on periodic table"),
        _make_question("q2", "Helium", "noble gas used in balloons"),
        _make_question("q3", "Lithium", "soft metal used in batteries"),
        _make_question("q4", "Carbon", "element central to organic chemistry"),
    ]
    heldout = _make_question(
        "q5",
        "Hydrogen",
        "fresh heldout clue about the lightest element",
    )

    profile_builder = AnswerProfileBuilder(min_questions_per_answer=1).fit(
        stale_reference
    )
    builder = MCBuilder(K=4, strategy="tfidf_profile", random_seed=13)

    mc_questions = builder.build(
        [heldout],
        profile_builder,
        reference_questions=raw_train,
    )

    assert len(mc_questions) == 1
    assert builder.last_build_stats["reference_questions"] == len(raw_train)
    assert "Hydrogen" in mc_questions[0].option_answer_primary
