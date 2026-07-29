"""Regression tests for top-M distractor ranking in MCBuilder._compute_rankings.

Validates that the argpartition-based top-M retrieval produces the same top
distractors as a full argsort, truncates ranking lists correctly, degrades
gracefully when N is small, and leaves category_random strategy unchanged.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qb_data.mc_builder import MCBuilder


@functools.lru_cache(maxsize=1)
def _load_smoke_splits_cached() -> tuple | None:
    """Memoised load of the in-tree ``questions.csv`` smoke fixture.

    The CSV (~14MB, ~20k rows) is checked into the repo for the
    category_random parity / cache-reuse regression tests. Without this
    cache each test method that calls ``_smoke_questions()`` re-parses
    the entire CSV, which is wasteful (the parse dominates each test
    method's runtime) even though absolute time stays sub-second. Returns
    ``None`` when the CSV is absent so the caller can pytest.skip cleanly.
    """
    from pathlib import Path

    if not Path("questions.csv").exists():  # pragma: no cover - tied to local CSV
        return None

    from qb_data.data_loader import QANTADatasetLoader
    from qb_data.dataset_splits import create_stratified_splits

    qs = QANTADatasetLoader.load_from_csv("questions.csv")[:80]
    return create_stratified_splits(qs, seed=42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_answers(n: int) -> tuple[list[str], dict[str, str]]:
    """Create *n* synthetic answers with distinct TF-IDF profiles.

    Each answer is a short phrase and its profile is a sentence containing
    unique vocabulary so TF-IDF can discriminate between them.
    """
    topics = [
        ("George Washington", "first president commander revolutionary war continental army"),
        ("Thomas Jefferson", "third president declaration independence Virginia Monticello"),
        ("John Adams", "second president Massachusetts diplomat federalist"),
        ("Benjamin Franklin", "inventor diplomat Philadelphia printing press electricity"),
        ("Abraham Lincoln", "sixteenth president civil war emancipation slavery"),
        ("Alexander Hamilton", "treasury secretary banking system federalist papers"),
        ("James Madison", "bill rights constitution fourth president Virginia"),
        ("Andrew Jackson", "military hero populist president battle New Orleans"),
        ("Theodore Roosevelt", "progressive trust buster national parks rough riders"),
        ("Ulysses Grant", "civil war general eighteenth president reconstruction"),
        ("Woodrow Wilson", "world war one league nations progressive president"),
        ("Franklin Roosevelt", "new deal world war two great depression fireside"),
        ("Harry Truman", "atomic bomb cold war Korean conflict fair deal"),
        ("Dwight Eisenhower", "supreme commander NATO interstate highway system"),
        ("John Kennedy", "space race Cuban missile crisis new frontier"),
        ("Lyndon Johnson", "great society civil rights Vietnam escalation"),
        ("Richard Nixon", "detente China opening Watergate resignation"),
        ("Ronald Reagan", "cold war end conservative revolution economic growth"),
        ("Barack Obama", "affordable care act first African American president"),
        ("Jimmy Carter", "Camp David accords energy crisis human rights"),
    ]
    answers = [t[0] for t in topics[:n]]
    profiles = {t[0]: t[1] for t in topics[:n]}
    return answers, profiles


def _full_sort_rankings(
    answers: list[str], profiles: dict[str, str]
) -> dict[str, list[str]]:
    """Compute full rankings with canonical-index tie-breaking."""
    docs = [profiles[a] for a in answers]
    answer_idx = {a: i for i, a in enumerate(answers)}
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    sim = cosine_similarity(matrix, matrix)
    rankings: dict[str, list[str]] = {}
    for answer in answers:
        idx = answer_idx[answer]
        order = np.lexsort(
            (np.arange(len(answers)), -sim[idx])
        ).tolist()
        rankings[answer] = [answers[i] for i in order if answers[i] != answer]
    return rankings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTopMRanking:
    """Tests for top-M argpartition ranking in MCBuilder._compute_rankings."""

    def test_top_m_truncation(self) -> None:
        """Rankings should have length <= min(M, N-1)."""
        answers, profiles = _make_synthetic_answers(20)
        builder = MCBuilder(K=4, strategy="tfidf_profile")
        categories: dict[str, str] = {}

        rankings = builder._compute_rankings(answers, profiles, categories)

        M = min(max(5 * 4, 30), len(answers) - 1)  # min(30, 19) = 19
        for answer, ranked in rankings.items():
            assert len(ranked) <= min(M, len(answers) - 1), (
                f"Answer '{answer}' has {len(ranked)} distractors, "
                f"expected <= {min(M, len(answers) - 1)}"
            )

    def test_order_preservation(self) -> None:
        """Top-3 distractors must match the full-sort reference."""
        answers, profiles = _make_synthetic_answers(20)
        builder = MCBuilder(K=4, strategy="tfidf_profile")
        categories: dict[str, str] = {}

        rankings = builder._compute_rankings(answers, profiles, categories)
        reference = _full_sort_rankings(answers, profiles)

        for answer in answers:
            actual_top3 = rankings[answer][:3]
            expected_top3 = reference[answer][:3]
            assert actual_top3 == expected_top3, (
                f"Answer '{answer}': top-3 mismatch.\n"
                f"  actual:   {actual_top3}\n"
                f"  expected: {expected_top3}"
            )

    def test_small_n_graceful(self) -> None:
        """With N=5, rankings should have length N-1=4 without error."""
        answers, profiles = _make_synthetic_answers(5)
        builder = MCBuilder(K=4, strategy="tfidf_profile")
        categories: dict[str, str] = {}

        rankings = builder._compute_rankings(answers, profiles, categories)

        for answer, ranked in rankings.items():
            assert len(ranked) == 4, (
                f"Answer '{answer}' has {len(ranked)} distractors, expected 4"
            )

    def test_category_random_unaffected(self) -> None:
        """category_random strategy should not use argpartition path."""
        answers, profiles = _make_synthetic_answers(10)
        categories = {a: "History" for a in answers}
        builder = MCBuilder(K=4, strategy="category_random")

        rankings = builder._compute_rankings(answers, profiles, categories)

        for answer, ranked in rankings.items():
            # All same-category peers (minus self) should be present
            assert set(ranked) == set(a for a in answers if a != answer), (
                f"Answer '{answer}': category_random should include all peers"
            )


class TestRefCacheRngParity:
    """The per-reference-corpus ranking cache must not change rng-dependent
    output. For ``category_random`` the cache is intentionally disabled
    because ``_compute_rankings`` consumes ``self.rng.shuffle`` once per
    answer; cache hits would skip those draws and leave the per-question
    option-shuffle loop in a different rng state than the legacy
    fresh-MCBuilder-per-split flow.
    """

    def _smoke_questions(self):
        result = _load_smoke_splits_cached()
        if result is None:  # pragma: no cover - tied to local CSV
            pytest.skip("questions.csv not present in repo root")
        return result

    def test_category_random_split_loop_matches_fresh_per_split(self) -> None:
        """A shared MCBuilder with rng-reset-per-split must produce the
        same per-question options as building a fresh MCBuilder per split.
        Regression for the cache-hit / rng-skip divergence on val/test."""
        import random
        from qb_data.answer_profiles import AnswerProfileBuilder
        from qb_data.mc_builder import MCBuilder

        raw_train, raw_val, raw_test = self._smoke_questions()

        # Path A: legacy — fresh MCBuilder(random_seed=13) per split.
        pb_a = AnswerProfileBuilder()
        pb_a.fit(raw_train)
        legacy: dict[str, list] = {}
        for name, target in (
            ("train", raw_train),
            ("val", raw_val),
            ("test", raw_test),
        ):
            b = MCBuilder(K=4, strategy="category_random", random_seed=13)
            legacy[name] = b.build(target, pb_a, reference_questions=raw_train)

        # Path B: shared MCBuilder with rng reset between splits (the
        # current ``build_mc_dataset.py`` flow).
        pb_b = AnswerProfileBuilder()
        pb_b.fit(raw_train)
        shared = MCBuilder(K=4, strategy="category_random", random_seed=13)
        replayed: dict[str, list] = {}
        for name, target in (
            ("train", raw_train),
            ("val", raw_val),
            ("test", raw_test),
        ):
            shared.rng = random.Random(13)
            replayed[name] = shared.build(target, pb_b, reference_questions=raw_train)

        # For category_random the shared builder must NOT populate
        # ``_ref_cache``; otherwise val/test would diverge.
        assert shared._ref_cache is None

        for split in ("train", "val", "test"):
            legacy_opts = [(q.qid, tuple(q.options)) for q in legacy[split]]
            replayed_opts = [(q.qid, tuple(q.options)) for q in replayed[split]]
            assert legacy_opts == replayed_opts, (
                f"{split} differs between fresh-per-split and shared-rng-reset; "
                f"the ranking cache silently skipped rng draws."
            )

    def test_profile_strategy_cache_is_populated_and_reused(self) -> None:
        """Profile-based strategies are rng-deterministic in
        ``_compute_rankings``, so the cache stays enabled and gives the
        full perf win for SBERT / TF-IDF / OpenAI."""
        from qb_data.answer_profiles import AnswerProfileBuilder
        from qb_data.mc_builder import MCBuilder

        raw_train, raw_val, raw_test = self._smoke_questions()
        pb = AnswerProfileBuilder()
        pb.fit(raw_train)

        b = MCBuilder(K=4, strategy="tfidf_profile", random_seed=13)
        assert b._ref_cache is None
        b.build(raw_train, pb, reference_questions=raw_train)
        assert b._ref_cache is not None
        rankings_id = id(b._ref_cache["rankings"])
        b.build(raw_val, pb, reference_questions=raw_train)
        assert id(b._ref_cache["rankings"]) == rankings_id, (
            "tfidf_profile cache should be reused across splits"
        )
        b.build(raw_test, pb, reference_questions=raw_train)
        assert id(b._ref_cache["rankings"]) == rankings_id

    def test_category_random_never_populates_cache(self) -> None:
        """Direct micro-check independent of the smoke CSV."""
        from qb_data.answer_profiles import AnswerProfileBuilder
        from qb_data.mc_builder import MCBuilder
        from qb_data.data_loader import TossupQuestion

        qs = [
            TossupQuestion(
                qid=str(i),
                question=f"clue {i} text",
                tokens=["clue", str(i), "text"],
                answer_primary=f"answer_{i % 5}",
                clean_answers=[f"answer_{i % 5}"],
                run_indices=[0, 1, 2],
                human_buzz_positions=None,
                category="History",
                cumulative_prefixes=["clue", f"clue {i}", f"clue {i} text"],
            )
            for i in range(10)
        ]
        pb = AnswerProfileBuilder()
        pb.fit(qs)
        b = MCBuilder(K=4, strategy="category_random", random_seed=13)
        b.build(qs, pb, reference_questions=qs)
        assert b._ref_cache is None, (
            "category_random must not populate _ref_cache; doing so would "
            "skip rng-consuming shuffles on subsequent build() calls."
        )

    def test_content_mutation_invalidates_ref_cache(self) -> None:
        """Mutating ``q.question`` between build() calls must miss the
        cache. Pre-fix the cache was keyed on qid only, so rewriting
        the question text (or answer_primary / category) silently
        returned stale rankings/profiles from the prior corpus."""
        from qb_data.answer_profiles import AnswerProfileBuilder
        from qb_data.mc_builder import MCBuilder
        from qb_data.data_loader import TossupQuestion

        qs = [
            TossupQuestion(
                qid=str(i),
                question=f"original clue {i} text",
                tokens=["original", "clue", str(i), "text"],
                answer_primary=f"answer_{i % 3}",
                clean_answers=[f"answer_{i % 3}"],
                run_indices=[0, 1, 2, 3],
                human_buzz_positions=None,
                category="History",
                cumulative_prefixes=[
                    "original",
                    "original clue",
                    f"original clue {i}",
                    f"original clue {i} text",
                ],
            )
            for i in range(8)
        ]
        pb = AnswerProfileBuilder()
        pb.fit(qs)
        b = MCBuilder(K=4, strategy="tfidf_profile", random_seed=13)
        b.build(qs, pb, reference_questions=qs)
        assert b._ref_cache is not None
        first_rankings_id = id(b._ref_cache["rankings"])
        first_key = b._ref_cache_key

        # Mutate one question's content in place; the qid set is
        # unchanged so a qid-only cache key would silently re-use the
        # stale rankings derived from "original ..." text.
        qs[0] = TossupQuestion(
            qid=qs[0].qid,
            question="completely different content for question zero",
            tokens=["completely", "different", "content"],
            answer_primary=qs[0].answer_primary,
            clean_answers=qs[0].clean_answers,
            run_indices=qs[0].run_indices,
            human_buzz_positions=qs[0].human_buzz_positions,
            category=qs[0].category,
            cumulative_prefixes=qs[0].cumulative_prefixes,
        )
        pb2 = AnswerProfileBuilder()
        pb2.fit(qs)
        b.build(qs, pb2, reference_questions=qs)
        assert id(b._ref_cache["rankings"]) != first_rankings_id, (
            "content-aware cache key must invalidate when q.question "
            "changes; otherwise mutated reference content silently "
            "returns stale rankings."
        )
        assert b._ref_cache_key != first_key

    def test_alias_mutation_invalidates_ref_cache(self) -> None:
        """Mutating ``q.clean_answers`` between build() calls must also
        miss the cache. Pre-fix the cache key omitted aliases, so an
        alias-normalisation refresh that rewrote ``clean_answers``
        while leaving question text / answer_primary / category
        unchanged silently reused stale ``answer_to_aliases``."""
        from qb_data.answer_profiles import AnswerProfileBuilder
        from qb_data.mc_builder import MCBuilder
        from qb_data.data_loader import TossupQuestion

        qs = [
            TossupQuestion(
                qid=str(i),
                question=f"clue {i} text",
                tokens=["clue", str(i), "text"],
                answer_primary=f"answer_{i % 3}",
                clean_answers=[f"answer_{i % 3}"],
                run_indices=[0, 1, 2],
                human_buzz_positions=None,
                category="History",
                cumulative_prefixes=["clue", f"clue {i}", f"clue {i} text"],
            )
            for i in range(6)
        ]
        pb = AnswerProfileBuilder()
        pb.fit(qs)
        b = MCBuilder(K=4, strategy="tfidf_profile", random_seed=13)
        b.build(qs, pb, reference_questions=qs)
        assert b._ref_cache is not None
        first_key = b._ref_cache_key

        # Refresh aliases on one question; everything else is identical.
        qs[0] = TossupQuestion(
            qid=qs[0].qid,
            question=qs[0].question,
            tokens=qs[0].tokens,
            answer_primary=qs[0].answer_primary,
            clean_answers=[qs[0].answer_primary, "added alias"],
            run_indices=qs[0].run_indices,
            human_buzz_positions=qs[0].human_buzz_positions,
            category=qs[0].category,
            cumulative_prefixes=qs[0].cumulative_prefixes,
        )
        pb2 = AnswerProfileBuilder()
        pb2.fit(qs)
        b.build(qs, pb2, reference_questions=qs)
        assert b._ref_cache_key != first_key, (
            "alias-aware cache key must invalidate when q.clean_answers "
            "changes; otherwise an alias-normalisation refresh silently "
            "reuses stale answer_to_aliases for downstream guard checks."
        )

    def test_profile_builder_config_change_invalidates_ref_cache(self) -> None:
        """Reusing an MCBuilder with a different AnswerProfileBuilder
        config (``max_tokens_per_profile`` / ``min_questions_per_answer``)
        must miss the cache. Pre-fix the key only fingerprinted reference
        questions, so the second build silently returned distractors
        derived from the first profile_builder's settings."""
        from qb_data.answer_profiles import AnswerProfileBuilder
        from qb_data.mc_builder import MCBuilder
        from qb_data.data_loader import TossupQuestion

        qs = [
            TossupQuestion(
                qid=str(i),
                question=f"clue {i} text body content",
                tokens=["clue", str(i), "text", "body", "content"],
                answer_primary=f"answer_{i % 3}",
                clean_answers=[f"answer_{i % 3}"],
                run_indices=[0, 1, 2, 3, 4],
                human_buzz_positions=None,
                category="History",
                cumulative_prefixes=[
                    "clue",
                    f"clue {i}",
                    f"clue {i} text",
                    f"clue {i} text body",
                    f"clue {i} text body content",
                ],
            )
            for i in range(6)
        ]
        pb_wide = AnswerProfileBuilder(max_tokens_per_profile=2000)
        pb_wide.fit(qs)
        b = MCBuilder(K=4, strategy="tfidf_profile", random_seed=13)
        b.build(qs, pb_wide, reference_questions=qs)
        assert b._ref_cache is not None
        first_key = b._ref_cache_key

        # Same questions, but a different profile_builder config -- the
        # cache should miss because the cached ``answer_profiles`` were
        # produced under the old config.
        pb_narrow = AnswerProfileBuilder(max_tokens_per_profile=10)
        pb_narrow.fit(qs)
        b.build(qs, pb_narrow, reference_questions=qs)
        assert b._ref_cache_key != first_key, (
            "config-aware cache key must invalidate when the caller "
            "passes a profile_builder with different settings; otherwise "
            "comparing profile-builder variants in one process silently "
            "reuses stale answer_profiles."
        )
