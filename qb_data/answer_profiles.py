"""Answer profile builder with leave-one-out exclusion for quiz bowl questions."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain, islice

from qb_data.data_loader import TossupQuestion


_WORD = re.compile(r"\S+")


def _profile_tokens(texts: Iterable[str], limit: int) -> list[str]:
    """Split complete short documents natively and stream only a potentially oversized one."""
    words: list[str] = []
    for text in texts:
        remaining = limit - len(words)
        # N whitespace-delimited words need at least 2*N - 1 characters.
        if limit <= 0 or len(text) <= 2 * remaining:
            words.extend(text.split())
        else:
            words.extend(match[0] for match in islice(_WORD.finditer(text), remaining))
        if limit > 0 and len(words) == limit:
            break
    return words


class AnswerProfileBuilder:
    """Builds profiles for answers by aggregating question texts.

    The profile for an answer is created by concatenating all question texts
    that have that answer. When building profiles for distractors, we use
    all questions. For the gold answer, we exclude the current question to
    prevent information leakage (leave-one-out).

    Attributes:
        max_tokens_per_profile: Maximum number of tokens to keep in each profile.
        min_questions_per_answer: Minimum questions needed to build a profile.
        _grouped: Dictionary mapping answer_primary to list of (qid, question_text) tuples.
    """

    def __init__(
        self,
        max_tokens_per_profile: int = 2000,
        min_questions_per_answer: int = 1,
    ) -> None:
        """Initialize the answer profile builder.

        Args:
            max_tokens_per_profile: Maximum tokens to keep in each profile.
            min_questions_per_answer: Minimum questions needed to build a profile.
        """
        self.max_tokens_per_profile = max_tokens_per_profile
        self.min_questions_per_answer = min_questions_per_answer
        self._grouped: dict[str, list[tuple[str, str]]] = {}
        self._cache: dict[tuple[str, str | None], str] = {}
        self._fit_signature: tuple[tuple[str, str, str], ...] | None = None

    def fit(self, questions: list[TossupQuestion]) -> AnswerProfileBuilder:
        """Fit the builder on a set of questions.

        Groups questions by their primary answer for efficient profile building.
        Skips the refit when the builder is already fit on the same ordered
        question content, avoiding redundant work when the same reference
        corpus is passed across multiple ``MCBuilder.build()`` calls.  Order
        and multiplicity are significant because profiles concatenate text
        before truncation.

        Args:
            questions: List of tossup questions to group by answer.

        Returns:
            Self for method chaining.
        """
        fit_signature = tuple(
            (q.qid, q.answer_primary, q.question) for q in questions
        )
        if fit_signature == self._fit_signature:
            return self
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for q in questions:
            grouped[q.answer_primary].append((q.qid, q.question))
        self._grouped = dict(grouped)
        self._cache = {}
        self._fit_signature = fit_signature
        return self

    def _profile_text(
        self,
        answer_primary: str,
        exclude_qid: str | None = None,
    ) -> str:
        """Build profile text for an answer with optional exclusion.

        Args:
            answer_primary: The answer to build a profile for.
            exclude_qid: Optional question ID to exclude (leave-one-out).

        Returns:
            Profile text truncated to max_tokens_per_profile.
        """
        key = (answer_primary, exclude_qid)
        if key in self._cache:
            return self._cache[key]

        items = self._grouped.get(answer_primary, ())
        texts = (
            text for qid, text in items
            if exclude_qid is None or qid != exclude_qid
        )
        # Count the minimum number of eligible questions, including blank ones,
        # before stopping tokenization at the requested prefix.
        first = list(islice(texts, min(len(items), max(0, self.min_questions_per_answer))))
        result = answer_primary
        if len(first) >= self.min_questions_per_answer:
            words = _profile_tokens(chain(first, texts), self.max_tokens_per_profile)
            result = " ".join(words) or answer_primary
        self._cache[key] = result
        return result

    def profile_for_answer(
        self,
        answer_primary: str,
        exclude_qid: str | None = None,
    ) -> str:
        """Get the profile for a specific answer.

        Args:
            answer_primary: The answer to get a profile for.
            exclude_qid: Optional question ID to exclude (for gold answer).

        Returns:
            Profile text for the answer.
        """
        return self._profile_text(
            answer_primary=answer_primary,
            exclude_qid=exclude_qid
        )

    def build_profiles(
        self,
        questions: list[TossupQuestion],
        exclude_qid: str | None = None,
    ) -> dict[str, str]:
        """Build profiles for all answers in the dataset.

        Args:
            questions: List of questions (used to fit if not already fitted).
            exclude_qid: Optional question ID to exclude from all profiles.

        Returns:
            Dictionary mapping answer_primary to profile text.
        """
        if not self._grouped:
            self.fit(questions)

        return {
            answer: self._profile_text(answer, exclude_qid=exclude_qid)
            for answer in self._grouped.keys()
        }
