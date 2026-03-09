"""Answer profile builder with leave-one-out exclusion for quiz bowl questions."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from qb_data.data_loader import TossupQuestion


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
        min_questions_per_answer: int = 1
    ):
        """Initialize the answer profile builder.

        Args:
            max_tokens_per_profile: Maximum tokens to keep in each profile.
            min_questions_per_answer: Minimum questions needed to build a profile.
        """
        self.max_tokens_per_profile = max_tokens_per_profile
        self.min_questions_per_answer = min_questions_per_answer
        self._grouped: Dict[str, List[Tuple[str, str]]] = {}

    def fit(self, questions: List[TossupQuestion]) -> "AnswerProfileBuilder":
        """Fit the builder on a set of questions.

        Groups questions by their primary answer for efficient profile building.

        Args:
            questions: List of tossup questions to group by answer.

        Returns:
            Self for method chaining.
        """
        grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for q in questions:
            # Store qid and full question text for each answer
            grouped[q.answer_primary].append((q.qid, q.question))
        self._grouped = dict(grouped)
        return self

    def _profile_text(
        self,
        answer_primary: str,
        exclude_qid: Optional[str] = None
    ) -> str:
        """Build profile text for an answer with optional exclusion.

        Args:
            answer_primary: The answer to build a profile for.
            exclude_qid: Optional question ID to exclude (leave-one-out).

        Returns:
            Profile text truncated to max_tokens_per_profile.
        """
        items = self._grouped.get(answer_primary, [])
        texts: List[str] = []

        # Collect all question texts except the excluded one
        for qid, qtext in items:
            if exclude_qid is not None and qid == exclude_qid:
                continue
            texts.append(qtext)

        # If not enough questions after exclusion, fall back to answer text
        if len(texts) < self.min_questions_per_answer:
            return answer_primary

        # Merge all texts and split into tokens
        merged = " ".join(texts).split()

        # Truncate to max tokens if specified
        if self.max_tokens_per_profile > 0:
            merged = merged[:self.max_tokens_per_profile]

        return " ".join(merged) if merged else answer_primary

    def profile_for_answer(
        self,
        answer_primary: str,
        exclude_qid: Optional[str] = None
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
        questions: List[TossupQuestion],
        exclude_qid: Optional[str] = None,
    ) -> Dict[str, str]:
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