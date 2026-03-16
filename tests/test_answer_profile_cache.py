"""Tests for AnswerProfileBuilder._cache memoization.

Verifies that:
1. Distractor profiles (exclude_qid=None) are cached and return identical results
2. Leave-one-out profiles (answer, qid) are cached and return identical results
3. Cache is invalidated on fit() with new data
4. Cached distractor profile is byte-identical to freshly computed profile
5. Cached leave-one-out profile is byte-identical to freshly computed profile
6. Cache reduces actual computation (single entry per unique key)
"""

from __future__ import annotations

import pytest

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion


def _make_question(
    qid: str,
    answer: str,
    text: str,
    category: str = "History",
) -> TossupQuestion:
    """Create a minimal TossupQuestion for cache testing."""
    tokens = text.split()
    return TossupQuestion(
        qid=qid,
        question=text,
        tokens=tokens,
        answer_primary=answer,
        clean_answers=[answer],
        run_indices=[len(tokens) - 1],
        human_buzz_positions=[],
        category=category,
        cumulative_prefixes=[text],
    )


@pytest.fixture
def sample_questions() -> list[TossupQuestion]:
    """Five questions with 3 shared answers for exercising cache hits."""
    return [
        _make_question("q1", "Washington", "first president commander in chief"),
        _make_question("q2", "Washington", "led the continental army to victory"),
        _make_question("q3", "Jefferson", "wrote the declaration of independence"),
        _make_question("q4", "Jefferson", "third president and diplomat to France"),
        _make_question("q5", "Lincoln", "preserved the union during civil war"),
    ]


@pytest.fixture
def builder(sample_questions: list[TossupQuestion]) -> AnswerProfileBuilder:
    """Return a fitted AnswerProfileBuilder."""
    b = AnswerProfileBuilder(max_tokens_per_profile=2000, min_questions_per_answer=1)
    b.fit(sample_questions)
    return b


class TestProfileCacheHits:
    """Repeated calls with the same args return the same cached result."""

    def test_distractor_profile_cached(
        self, builder: AnswerProfileBuilder
    ) -> None:
        """profile_for_answer returns identical string on repeated (answer, None)."""
        first = builder.profile_for_answer("Washington", exclude_qid=None)
        second = builder.profile_for_answer("Washington", exclude_qid=None)
        assert first is second  # same object, not just equal

    def test_leave_one_out_profile_cached(
        self, builder: AnswerProfileBuilder
    ) -> None:
        """profile_for_answer returns identical string on repeated (answer, qid)."""
        first = builder.profile_for_answer("Washington", exclude_qid="q1")
        second = builder.profile_for_answer("Washington", exclude_qid="q1")
        assert first is second  # same object from cache


class TestCacheInvalidation:
    """fit() with new data clears the cache."""

    def test_fit_clears_cache(
        self, builder: AnswerProfileBuilder, sample_questions: list[TossupQuestion]
    ) -> None:
        """After fit() with new data, cache is empty and profiles reflect new data."""
        # Populate cache
        builder.profile_for_answer("Washington", exclude_qid=None)
        assert len(builder._cache) > 0

        # Re-fit with different data
        new_questions = [
            _make_question("q99", "Washington", "completely different text about cherry trees"),
        ]
        builder.fit(new_questions)
        assert len(builder._cache) == 0

        # New profile should reflect new data
        profile = builder.profile_for_answer("Washington", exclude_qid=None)
        assert "cherry" in profile


class TestCacheEquivalence:
    """Cached profiles are byte-identical to freshly computed profiles."""

    def test_distractor_cache_equivalence(
        self, sample_questions: list[TossupQuestion]
    ) -> None:
        """Cached (answer, None) profile is byte-identical to a fresh computation."""
        # Build fresh (uncached) profile
        fresh_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        fresh_builder.fit(sample_questions)
        fresh_profile = fresh_builder._profile_text("Jefferson", exclude_qid=None)

        # Build cached profile
        cached_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        cached_builder.fit(sample_questions)
        _ = cached_builder._profile_text("Jefferson", exclude_qid=None)  # populate cache
        cached_profile = cached_builder._profile_text("Jefferson", exclude_qid=None)  # from cache

        assert fresh_profile == cached_profile

    def test_leave_one_out_cache_equivalence(
        self, sample_questions: list[TossupQuestion]
    ) -> None:
        """Cached (answer, qid) profile is byte-identical to a fresh computation."""
        fresh_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        fresh_builder.fit(sample_questions)
        fresh_profile = fresh_builder._profile_text("Washington", exclude_qid="q1")

        cached_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        cached_builder.fit(sample_questions)
        _ = cached_builder._profile_text("Washington", exclude_qid="q1")
        cached_profile = cached_builder._profile_text("Washington", exclude_qid="q1")

        assert fresh_profile == cached_profile


class TestCacheEfficiency:
    """Cache reduces computation to one real call per unique key."""

    def test_cache_stores_one_entry_per_unique_key(
        self, builder: AnswerProfileBuilder
    ) -> None:
        """Calling _profile_text N times with same args results in 1 cache entry."""
        for _ in range(10):
            builder.profile_for_answer("Lincoln", exclude_qid=None)

        # Only one cache entry for (Lincoln, None)
        assert ("Lincoln", None) in builder._cache
        assert len([k for k in builder._cache if k[0] == "Lincoln"]) == 1
