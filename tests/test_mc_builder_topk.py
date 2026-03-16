"""Regression tests for top-M distractor ranking in MCBuilder._compute_rankings.

Validates that the argpartition-based top-M retrieval produces the same top
distractors as a full argsort, truncates ranking lists correctly, degrades
gracefully when N is small, and leaves category_random strategy unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qb_data.mc_builder import MCBuilder


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
    """Compute rankings via full argsort (reference implementation)."""
    docs = [profiles[a] for a in answers]
    answer_idx = {a: i for i, a in enumerate(answers)}
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    sim = cosine_similarity(matrix, matrix)
    rankings: dict[str, list[str]] = {}
    for answer in answers:
        idx = answer_idx[answer]
        order = np.argsort(-sim[idx]).tolist()
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
