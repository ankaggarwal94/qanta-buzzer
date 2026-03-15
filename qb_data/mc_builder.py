"""Multiple-choice question builder with anti-artifact guards."""

from __future__ import annotations

import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.text_utils import normalize_answer


@dataclass
class MCQuestion(TossupQuestion):
    """A tossup question with multiple-choice options.

    Extends TossupQuestion with fields for multiple-choice presentation
    and tracking of distractor generation strategy.
    """
    options: List[str]
    gold_index: int
    option_profiles: List[str]
    option_answer_primary: List[str]
    distractor_strategy: str


def _normalized_edit_distance(a: str, b: str) -> float:
    """Compute normalized edit distance between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Distance between 0 (identical) and 1 (completely different).
    """
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _token_overlap(a: str, b: str) -> float:
    """Compute token overlap between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Fraction of overlapping tokens (0 to 1).
    """
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


class MCBuilder:
    """Builder for multiple-choice questions with anti-artifact guards.

    This class implements four layers of guards to prevent spurious patterns
    that agents could exploit:
    1. Alias collision guard: Prevents distractors that are aliases of the gold answer
    2. Duplicate guard: Prevents distractors with high token overlap
    3. Length ratio guard: Prevents distractors much longer/shorter than others
    4. Question overlap guard: Prevents answers that appear in the question text
    """

    def __init__(
        self,
        K: int = 4,
        strategy: str = "sbert_profile",
        alias_edit_distance_threshold: float = 0.2,
        duplicate_token_overlap_threshold: float = 0.8,
        max_length_ratio: float = 3.0,
        random_seed: int = 13,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_model: str = "text-embedding-3-small",
        variable_K: bool = False,
        min_K: int = 2,
        max_K: int | None = None,
    ):
        """Initialize the MC builder.

        Args:
            K: Default number of answer choices (must be >= 2).
            strategy: Distractor selection strategy.
            alias_edit_distance_threshold: Max edit distance for alias detection.
            duplicate_token_overlap_threshold: Max token overlap between options.
            max_length_ratio: Max ratio between longest and shortest option.
            random_seed: Random seed for reproducibility.
            embedding_model: SentenceTransformer model name for ``sbert_profile``.
            openai_model: OpenAI embedding model for ``openai_profile``.
            variable_K: If True, sample target K per question from
                ``[min_K, max_K or K]``.
            min_K: Minimum K when ``variable_K`` is True.
            max_K: Maximum K when ``variable_K`` is True.  Defaults to ``K``.
        """
        if K < 2:
            raise ValueError("K must be >= 2")
        self.K = K
        self.variable_K = variable_K
        self.min_K = max(2, min_K)
        self.max_K = max_K if max_K is not None else K
        self.strategy = strategy
        self.alias_edit_distance_threshold = alias_edit_distance_threshold
        self.duplicate_token_overlap_threshold = duplicate_token_overlap_threshold
        self.max_length_ratio = max_length_ratio
        self.rng = random.Random(random_seed)
        self.embedding_model = embedding_model
        self.openai_model = openai_model

    def _prepare_lookup(
        self, questions: List[TossupQuestion]
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], List[str]]:
        """Prepare lookup structures for answer processing.

        Args:
            questions: List of tossup questions.

        Returns:
            Tuple of (answer_to_aliases, answer_to_category, answer_to_norm, answers).
        """
        answer_to_aliases: Dict[str, Set[str]] = {}
        answer_to_category: Dict[str, str] = {}

        for q in questions:
            # Collect all aliases for each answer
            aliases = answer_to_aliases.setdefault(q.answer_primary, set())
            aliases.update(str(alias) for alias in q.clean_answers)
            aliases.add(q.answer_primary)

            # Track category for category-based distractor selection
            if q.category and q.answer_primary not in answer_to_category:
                answer_to_category[q.answer_primary] = q.category

        # Convert to sorted lists for consistency
        answer_to_aliases_list = {k: sorted(v) for k, v in answer_to_aliases.items()}
        answers = sorted(answer_to_aliases_list.keys())
        answer_to_norm = {a: str(normalize_answer(a)) for a in answers}

        return answer_to_aliases_list, answer_to_category, answer_to_norm, answers

    def _rank_by_similarity(
        self,
        sim: np.ndarray,
        answers: List[str],
        answer_idx: Dict[str, int],
        M: int,
    ) -> Dict[str, List[str]]:
        """Rank distractors for each answer using a similarity matrix.

        Uses ``np.argpartition`` for top-M retrieval when M < N-1,
        reducing per-answer work from O(N log N) to O(N + M log M).

        Parameters
        ----------
        sim : np.ndarray
            Pairwise similarity matrix of shape (N, N).
        answers : list[str]
            Ordered answer strings corresponding to matrix rows/cols.
        answer_idx : dict[str, int]
            Mapping from answer string to its index in *sim*.
        M : int
            Number of top candidates to retain per answer.

        Returns
        -------
        dict[str, list[str]]
            Each answer mapped to its ranked distractor list (length <= M).
        """
        N = len(answers)
        rankings: Dict[str, List[str]] = {}
        for answer in answers:
            idx = answer_idx[answer]
            row = sim[idx]
            if M >= N - 1:
                # Small N: full sort (no benefit from partition)
                order = np.argsort(-row).tolist()
            else:
                # Top-M retrieval: O(N) partition + O(M log M) sort
                top_m_idx = np.argpartition(-row, M)[:M]
                top_m_idx = top_m_idx[np.argsort(-row[top_m_idx])]
                order = top_m_idx.tolist()
            rankings[answer] = [answers[i] for i in order if answers[i] != answer]
        return rankings

    def _compute_rankings(
        self,
        answers: List[str],
        answer_profiles: Dict[str, str],
        answer_to_category: Dict[str, str],
    ) -> Dict[str, List[str]]:
        """Compute distractor rankings for each answer.

        For profile-based strategies, uses top-M retrieval via
        ``np.argpartition`` instead of full ``np.argsort`` to reduce
        per-answer complexity from O(N log N) to O(N + M log M) and
        total memory from O(N^2) to O(N*M), where M = max(5*K, 30).

        Args:
            answers: List of all unique answers.
            answer_profiles: Dictionary mapping answers to their profiles.
            answer_to_category: Dictionary mapping answers to categories.

        Returns:
            Dictionary mapping each answer to a ranked list of distractors.
        """
        if self.strategy == "category_random":
            # Random selection within the same category
            rankings: Dict[str, List[str]] = {}
            for answer in answers:
                category = answer_to_category.get(answer, "")
                # First try same category, then fall back to all answers
                candidates = [
                    a for a in answers
                    if a != answer and answer_to_category.get(a, "") == category
                ]
                if len(candidates) < self.K - 1:
                    candidates = [a for a in answers if a != answer]
                self.rng.shuffle(candidates)
                rankings[answer] = candidates
            return rankings

        # Profile-based ranking strategies
        docs = [answer_profiles[a] for a in answers]
        answer_idx = {a: i for i, a in enumerate(answers)}
        M = min(max(5 * self.K, 30), len(answers) - 1)

        if self.strategy == "tfidf_profile":
            # TF-IDF based similarity
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(docs)
            sim = cosine_similarity(matrix, matrix)
            return self._rank_by_similarity(sim, answers, answer_idx, M)

        if self.strategy in {"sbert_profile", "openai_profile"}:
            if self.strategy == "sbert_profile":
                # One-shot SBERT encoding for distractor ranking.
                # This is separate from the SBERTLikelihood runtime cache
                # because it runs only during MC dataset construction.
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(self.embedding_model)
                embeddings = encoder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
                sim = embeddings @ embeddings.T
            else:
                from models.likelihoods import OpenAILikelihood

                likelihood = OpenAILikelihood(model=self.openai_model)
                embeddings = likelihood.embed_and_cache(docs)
                sim = embeddings @ embeddings.T

            return self._rank_by_similarity(sim, answers, answer_idx, M)

        raise ValueError(f"Unknown distractor strategy: {self.strategy}")

    def _aliases_collide(self, candidate: str, gold_aliases: List[str]) -> bool:
        """Check if a candidate is too similar to any gold answer alias.

        Args:
            candidate: Candidate distractor.
            gold_aliases: List of aliases for the gold answer.

        Returns:
            True if the candidate collides with a gold alias.
        """
        candidate_norm = str(normalize_answer(candidate))
        gold_norms = [str(normalize_answer(alias)) for alias in gold_aliases]

        # Check exact match
        if candidate_norm in set(gold_norms):
            return True

        # Check edit distance
        for gold_norm in gold_norms:
            if _normalized_edit_distance(candidate_norm, gold_norm) < self.alias_edit_distance_threshold:
                return True

        return False

    def _violates_duplicate_guard(self, candidate: str, selected: List[str]) -> bool:
        """Check if candidate has too much token overlap with already selected options.

        Args:
            candidate: Candidate distractor.
            selected: List of already selected distractors.

        Returns:
            True if the candidate has too much overlap.
        """
        for chosen in selected:
            if _token_overlap(candidate, chosen) > self.duplicate_token_overlap_threshold:
                return True
        return False

    def _violates_length_ratio_guard(self, options: List[str]) -> bool:
        """Check if options have too different lengths.

        Args:
            options: List of all options.

        Returns:
            True if the length ratio is too high.
        """
        lengths = [max(1, len(o.split())) for o in options]
        return (max(lengths) / min(lengths)) > self.max_length_ratio

    def _violates_question_overlap_guard(self, question: str, options: List[str]) -> bool:
        """Check if any option appears in the question text.

        Args:
            question: Question text.
            options: List of answer options.

        Returns:
            True if any option appears in the question.
        """
        q_norm = str(normalize_answer(question))
        for option in options:
            o_norm = str(normalize_answer(option))
            if o_norm and o_norm in q_norm:
                return True
        return False

    def _target_k(self) -> int:
        """Return the target K for the next question.

        When ``variable_K`` is False, always returns ``self.K``.
        When True, samples uniformly from ``[min_K, max_K]``.
        """
        if not self.variable_K:
            return self.K
        return self.rng.randint(self.min_K, self.max_K)

    def build(
        self,
        questions: List[TossupQuestion],
        profile_builder: AnswerProfileBuilder,
    ) -> List[MCQuestion]:
        """Build multiple-choice questions with anti-artifact guards.

        Args:
            questions: List of tossup questions.
            profile_builder: Profile builder for answer representations.

        Returns:
            List of MCQuestion objects that passed all guards.
        """
        if not questions:
            return []

        # Build answer profiles
        profile_builder.fit(questions)
        answer_profiles = profile_builder.build_profiles(questions)

        # Prepare lookup structures
        answer_to_aliases, answer_to_category, _answer_to_norm, answers = self._prepare_lookup(questions)

        # Compute distractor rankings
        rankings = self._compute_rankings(answers, answer_profiles, answer_to_category)

        mc_questions: List[MCQuestion] = []

        for q in questions:
            target_k = self._target_k()
            gold = q.answer_primary
            gold_aliases = answer_to_aliases.get(gold, [gold])
            ranked = rankings.get(gold, [a for a in answers if a != gold])
            selected: List[str] = []

            # Select distractors from ranked list
            for candidate in ranked:
                if candidate == gold:
                    continue
                if self._aliases_collide(candidate, gold_aliases):
                    continue
                if self._violates_duplicate_guard(candidate, selected):
                    continue
                selected.append(candidate)
                if len(selected) >= target_k - 1:
                    break

            # If not enough distractors from ranking, try random fallback
            if len(selected) < target_k - 1:
                fallback = [a for a in answers if a not in selected and a != gold]
                self.rng.shuffle(fallback)
                for candidate in fallback:
                    if self._aliases_collide(candidate, gold_aliases):
                        continue
                    if self._violates_duplicate_guard(candidate, selected):
                        continue
                    selected.append(candidate)
                    if len(selected) >= target_k - 1:
                        break

            # Skip question if we can't find enough valid distractors
            if len(selected) < target_k - 1:
                continue

            # Create options and shuffle
            option_answer_primary = [gold] + selected[:target_k - 1]
            self.rng.shuffle(option_answer_primary)
            gold_index = option_answer_primary.index(gold)
            options = option_answer_primary[:]

            # Apply guard 3: Check length ratio
            if self._violates_length_ratio_guard(options):
                continue

            # Apply guard 4: Check question overlap
            if self._violates_question_overlap_guard(q.question, options):
                continue

            # Build option profiles with leave-one-out for gold
            option_profiles: List[str] = []
            for answer in option_answer_primary:
                exclude_qid = q.qid if answer == gold else None
                option_profiles.append(
                    profile_builder.profile_for_answer(answer, exclude_qid=exclude_qid)
                )

            # Create MCQuestion
            mc_questions.append(
                MCQuestion(
                    qid=q.qid,
                    question=q.question,
                    tokens=q.tokens,
                    answer_primary=q.answer_primary,
                    clean_answers=q.clean_answers,
                    run_indices=q.run_indices,
                    human_buzz_positions=q.human_buzz_positions,
                    category=q.category,
                    cumulative_prefixes=q.cumulative_prefixes,
                    options=options,
                    gold_index=gold_index,
                    option_profiles=option_profiles,
                    option_answer_primary=option_answer_primary,
                    distractor_strategy=self.strategy,
                )
            )

        return mc_questions


def build_mc_questions(
    questions: List[TossupQuestion],
    K: int,
    strategy: str,
    profile_builder: AnswerProfileBuilder,
    guards: Optional[Dict[str, Any]] = None,
    random_seed: int = 13,
) -> List[MCQuestion]:
    """Factory function to build multiple-choice questions.

    Args:
        questions: List of tossup questions.
        K: Number of answer choices.
        strategy: Distractor selection strategy.
        profile_builder: Profile builder for answer representations.
        guards: Optional dictionary of guard thresholds.
        random_seed: Random seed for reproducibility.

    Returns:
        List of MCQuestion objects that passed all guards.
    """
    guards = guards or {}
    builder = MCBuilder(
        K=K,
        strategy=strategy,
        alias_edit_distance_threshold=float(guards.get("alias_edit_distance_threshold", 0.2)),
        duplicate_token_overlap_threshold=float(guards.get("duplicate_token_overlap_threshold", 0.8)),
        max_length_ratio=float(guards.get("max_length_ratio", 3.0)),
        random_seed=random_seed,
    )
    return builder.build(questions=questions, profile_builder=profile_builder)
