"""Multiple-choice question builder with anti-artifact guards."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
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


@dataclass
class _RepairSearchBudget:
    """Deterministic scan/search ceilings shared by one repair."""

    limit: int
    attempts: int = 0
    exhausted: bool = False
    candidate_scans: int = 0
    candidate_scan_truncated: bool = False

    def consume(self) -> bool:
        """Consume one candidate-extension attempt, or mark exhaustion."""
        if self.attempts >= self.limit:
            self.exhausted = True
            return False
        self.attempts += 1
        return True

    def consume_candidate_scan(self) -> bool:
        """Consume one raw-candidate scan, or mark the source truncated."""
        if self.candidate_scans >= self.limit:
            self.candidate_scan_truncated = True
            return False
        self.candidate_scans += 1
        return True


@dataclass
class _RepairResult:
    """Outcome and provenance for one guard-repair attempt."""

    options: Optional[List[str]]
    source: Optional[str]
    candidate_attempts: int
    candidate_scans: int
    budget_exhausted: bool


def _new_repair_stats() -> Dict[str, int]:
    """Return a stable zero-valued repair telemetry schema."""
    return {
        "attempted_questions": 0,
        "succeeded_questions": 0,
        "ranked_successes": 0,
        "fallback_successes": 0,
        "budget_exhausted_questions": 0,
        "candidate_attempts": 0,
        "candidate_scans": 0,
        "length_ratio_triggers": 0,
        "question_overlap_triggers": 0,
        "simultaneous_guard_triggers": 0,
        "failed_questions": 0,
        "exhaustive_no_solution_questions": 0,
        "unrecoverable_gold_overlap_questions": 0,
    }


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
        max_repair_attempts: int = 10_000,
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
            max_repair_attempts: Deterministic per-repair ceiling applied
                independently to raw candidate scans and unique search
                extensions, shared by ranked and fallback phases.
        """
        if isinstance(K, bool) or not isinstance(K, int):
            raise ValueError("K must be an integer")
        if K < 2:
            raise ValueError("K must be >= 2")
        if isinstance(min_K, bool) or not isinstance(min_K, int):
            raise ValueError("min_K must be an integer")
        if max_K is not None and (
            isinstance(max_K, bool) or not isinstance(max_K, int)
        ):
            raise ValueError("max_K must be an integer or None")
        self.K = K
        self.variable_K = variable_K
        self.min_K = max(2, min_K)
        self.max_K = max_K if max_K is not None else K
        if self.variable_K and self.max_K < self.min_K:
            raise ValueError("max_K must be >= min_K when variable_K is enabled")
        if (
            isinstance(max_repair_attempts, bool)
            or not isinstance(max_repair_attempts, int)
            or max_repair_attempts < 1
        ):
            raise ValueError("max_repair_attempts must be a positive integer")
        self.max_repair_attempts = max_repair_attempts
        self.strategy = strategy
        self.alias_edit_distance_threshold = alias_edit_distance_threshold
        self.duplicate_token_overlap_threshold = duplicate_token_overlap_threshold
        self.max_length_ratio = max_length_ratio
        self._random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.embedding_model = embedding_model
        self.openai_model = openai_model
        self.last_build_stats: Dict[str, Any] = {}
        # Cache of (answer_to_aliases, answer_to_category, answers,
        # answer_profiles, rankings) keyed by the frozenset of reference
        # qids. The expensive ``_compute_rankings`` call (TF-IDF / SBERT /
        # OpenAI encoding over ~20k answer profiles) is shared across
        # split invocations of ``build()`` when ``reference_questions``
        # is identical. Only profile-based strategies are cached because
        # the ``category_random`` branch of ``_compute_rankings`` consumes
        # ``self.rng`` (one ``shuffle`` per answer); skipping it on cache
        # hits would put the rng into a different state than fresh-per-
        # split callers expect, silently changing per-question option
        # ordering on val/test.
        # Cache key shape:
        #   ((max_tokens_per_profile, min_questions_per_answer),
        #    frozenset[(qid, answer_primary, category,
        #               hash(question), hash(sorted clean_answers))])
        # Sentinel ``None`` means "no cache populated yet"; the type is
        # tuple-or-None so equality checks against a real cache miss for
        # the first build() call without spuriously matching an empty
        # frozenset.
        self._ref_cache_key: Optional[
            tuple[tuple[Any, Any], frozenset[tuple[str, str, str, int, int]]]
        ] = None
        self._ref_cache: Optional[Dict[str, Any]] = None

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the per-builder RNG to the constructor seed (or override).

        Used by ``scripts/build_mc_dataset.py`` to keep a single
        ``MCBuilder`` instance across train/val/test builds (so the
        ``_ref_cache`` survives) while still drawing each split from a
        fresh RNG state — matching the legacy fresh-per-split MCBuilder
        contract for byte-equivalent option ordering.

        Parameters
        ----------
        seed : int | None
            Override the constructor's ``random_seed``. ``None`` resets
            to the value passed to ``__init__`` (the documented default
            for callers that just want a deterministic reset).
        """
        self.rng = random.Random(self._random_seed if seed is None else seed)

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

    def _aliases_collide(
        self,
        candidate: str,
        gold_aliases: List[str],
        _gold_norms: set[str] | None = None,
    ) -> bool:
        """Check if a candidate is too similar to any gold answer alias.

        Args:
            candidate: Candidate distractor.
            gold_aliases: List of aliases for the gold answer.
            _gold_norms: Pre-computed normalized alias set (avoids recomputing per candidate).

        Returns:
            True if the candidate collides with a gold alias.
        """
        candidate_norm = str(normalize_answer(candidate))
        if _gold_norms is None:
            _gold_norms = {str(normalize_answer(alias)) for alias in gold_aliases}

        if candidate_norm in _gold_norms:
            return True

        for gold_norm in _gold_norms:
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

    def _bounded_fallback_order_from_state(
        self,
        answers: List[str],
        selected: List[str],
        gold: str,
        rng_state: object,
        limit: int,
    ) -> Tuple[List[str], bool]:
        """Return a bounded random fallback prefix without advancing ``self.rng``.

        A prefix of a full ``random.shuffle`` cannot be reproduced without
        processing the full input.  ``sample`` instead gives a deterministic
        ordered sample of at most ``limit`` raw answers using a cloned RNG.

        ``answers`` is the sorted unique universe produced by
        ``_prepare_lookup``.  Known exclusions are located by binary search,
        avoiding a scan merely to construct the eligible population.

        Returns the sampled prefix and whether it covers the eligible source.
        """
        excluded_indices: List[int] = []
        for answer in {*selected, gold}:
            idx = bisect_left(answers, answer)
            if idx < len(answers) and answers[idx] == answer:
                excluded_indices.append(idx)
        excluded_indices.sort()

        eligible_count = len(answers) - len(excluded_indices)
        fallback_rng = random.Random()
        fallback_rng.setstate(rng_state)
        sample_size = min(limit, eligible_count)
        eligible_ranks = fallback_rng.sample(range(eligible_count), sample_size)

        sampled: List[str] = []
        for rank in eligible_ranks:
            answer_idx = rank
            for excluded_idx in excluded_indices:
                if excluded_idx > answer_idx:
                    break
                answer_idx += 1
            sampled.append(answers[answer_idx])
        return (
            sampled,
            sample_size == eligible_count,
        )

    def _append_repair_candidates(
        self,
        raw_candidates: List[str],
        question_norm: str,
        gold: str,
        gold_aliases: List[str],
        gold_norms: set[str],
        candidates: List[str],
        seen: Set[str],
        budget: _RepairSearchBudget,
        excluded: Optional[Set[str]] = None,
    ) -> bool:
        """Append bounded, fully filtered candidates in raw source order.

        Every raw entry consumes the scan budget before alias normalization or
        question-overlap work.  Returns True only when the source was fully
        scanned.
        """
        excluded = excluded or set()
        for idx in range(len(raw_candidates)):
            if not budget.consume_candidate_scan():
                return False
            candidate = raw_candidates[idx]
            if candidate == gold or candidate in excluded or candidate in seen:
                continue
            seen.add(candidate)
            if self._aliases_collide(candidate, gold_aliases, _gold_norms=gold_norms):
                continue
            candidate_norm = str(normalize_answer(candidate))
            if candidate_norm and candidate_norm in question_norm:
                continue
            candidates.append(candidate)
        return True

    @staticmethod
    def _bounded_unseen_prefix(
        raw_candidates: List[str],
        excluded: Set[str],
        limit: int,
    ) -> Tuple[List[str], bool]:
        """Return up to ``limit`` unseen entries from a unique ordered source.

        Materialized fallback orders are unique.  Therefore at most
        ``limit + len(excluded)`` membership checks can either reach the end
        or prove that another unseen entry remains.  Skipped known entries do
        not consume the expensive candidate scan/normalization budget.
        """
        prefix: List[str] = []
        membership_limit = min(
            len(raw_candidates),
            limit + len(excluded),
        )
        for idx in range(membership_limit):
            candidate = raw_candidates[idx]
            if candidate in excluded:
                continue
            if len(prefix) >= limit:
                return prefix, False
            prefix.append(candidate)
        return prefix, membership_limit == len(raw_candidates)

    def _search_repaired_options(
        self,
        gold: str,
        candidates: List[str],
        target_k: int,
        permutation: List[int],
        budget: _RepairSearchBudget,
        transition_memo: Dict[Tuple[int, str], int],
        next_node_id: List[int],
    ) -> Optional[List[str]]:
        """Find a guard-passing option set within a deterministic work budget."""
        needed = target_k - 1
        if needed <= 0 or len(candidates) < needed:
            return None

        # Preserve the recursive implementation's depth-first candidate order
        # and monotone pruning without tying supported K to Python's recursion
        # limit.
        chosen: List[str] = []
        chosen_indices: List[int] = []
        chosen_node_ids: List[int] = []
        next_idx = 0

        while True:
            remaining = needed - len(chosen)
            last_start = len(candidates) - remaining

            if next_idx > last_start:
                if not chosen:
                    return None
                next_idx = chosen_indices.pop() + 1
                chosen.pop()
                chosen_node_ids.pop()
                continue

            idx = next_idx
            next_idx += 1
            candidate = candidates[idx]
            parent_node_id = chosen_node_ids[-1] if chosen_node_ids else 0
            transition_key = (parent_node_id, candidate)
            child_node_id = transition_memo.get(transition_key)
            if child_node_id is None:
                if not budget.consume():
                    return None
                accepted = not self._violates_duplicate_guard(candidate, chosen)
                if accepted:
                    partial = [gold] + chosen + [candidate]
                    accepted = not self._violates_length_ratio_guard(partial)
                if accepted:
                    child_node_id = next_node_id[0]
                    next_node_id[0] += 1
                else:
                    child_node_id = -1
                transition_memo[transition_key] = child_node_id
            if child_node_id < 0:
                continue

            chosen.append(candidate)
            chosen_indices.append(idx)
            chosen_node_ids.append(child_node_id)
            if len(chosen) == needed:
                ordered = [gold] + chosen
                return [ordered[i] for i in permutation]
            next_idx = idx + 1

    def _repair_options_after_guard_failure(
        self,
        question: str,
        gold: str,
        selected: List[str],
        ranked: List[str],
        answers: List[str],
        gold_aliases: List[str],
        gold_norms: set[str],
        target_k: int,
        shuffled_options: List[str],
        fallback_rng_state: object,
        fallback_selected: List[str],
        fallback_order: Optional[List[str]],
    ) -> _RepairResult:
        """Try later ranked/fallback distractors without weakening guards."""
        budget = _RepairSearchBudget(self.max_repair_attempts)
        needed = target_k - 1
        question_norm = str(normalize_answer(question))
        gold_norm = str(normalize_answer(gold))
        if gold_norm and gold_norm in question_norm:
            return _RepairResult(None, None, 0, 0, False)
        if needed <= 0:
            return _RepairResult(None, None, 0, 0, False)
        if needed > budget.limit:
            return _RepairResult(None, None, 0, 0, True)

        ordered_options = [gold] + selected[: target_k - 1]
        permutation = [ordered_options.index(option) for option in shuffled_options]
        ranked_candidates: List[str] = []
        seen: Set[str] = set()
        ranked_exhausted = self._append_repair_candidates(
            ranked,
            question_norm,
            gold,
            gold_aliases,
            gold_norms,
            ranked_candidates,
            seen,
            budget,
        )
        transition_memo: Dict[Tuple[int, str], int] = {}
        next_node_id = [1]
        repaired = self._search_repaired_options(
            gold,
            ranked_candidates,
            target_k,
            permutation,
            budget,
            transition_memo,
            next_node_id,
        )
        if repaired is not None:
            return _RepairResult(
                repaired,
                "ranked",
                budget.attempts,
                budget.candidate_scans,
                False,
            )

        if budget.exhausted or not ranked_exhausted:
            return _RepairResult(
                None,
                None,
                budget.attempts,
                budget.candidate_scans,
                True,
            )

        # A complete, unique N-1 ranking already covers every possible
        # non-gold answer, so a fallback pass cannot add a candidate.
        ranked_covers_answers = (
            len(ranked) == max(0, len(answers) - 1)
            and len(seen) == len(ranked)
        )
        if ranked_covers_answers:
            return _RepairResult(
                None,
                None,
                budget.attempts,
                budget.candidate_scans,
                False,
            )

        if fallback_order is None:
            remaining_scans = budget.limit - budget.candidate_scans
            fallback_order, fallback_source_exhausted = (
                self._bounded_fallback_order_from_state(
                    answers,
                    sorted(seen | set(fallback_selected)),
                    gold,
                    fallback_rng_state,
                    remaining_scans,
                )
            )
        else:
            remaining_scans = budget.limit - budget.candidate_scans
            fallback_order, fallback_source_exhausted = (
                self._bounded_unseen_prefix(
                    fallback_order,
                    seen | set(fallback_selected) | {gold},
                    remaining_scans,
                )
            )

        candidates = ranked_candidates[:]
        fallback_exhausted = self._append_repair_candidates(
            fallback_order,
            question_norm,
            gold,
            gold_aliases,
            gold_norms,
            candidates,
            seen,
            budget,
            excluded=set(fallback_selected),
        )
        if not fallback_source_exhausted:
            budget.candidate_scan_truncated = True
        fallback_exhausted = fallback_exhausted and fallback_source_exhausted

        if candidates == ranked_candidates:
            return _RepairResult(
                None,
                None,
                budget.attempts,
                budget.candidate_scans,
                not fallback_exhausted,
            )

        repaired = self._search_repaired_options(
            gold,
            candidates,
            target_k,
            permutation,
            budget,
            transition_memo,
            next_node_id,
        )
        return _RepairResult(
            repaired,
            "fallback" if repaired is not None else None,
            budget.attempts,
            budget.candidate_scans,
            False if repaired is not None else (
                budget.exhausted or not fallback_exhausted
            ),
        )

    def build(
        self,
        questions: List[TossupQuestion],
        profile_builder: AnswerProfileBuilder,
        reference_questions: Optional[List[TossupQuestion]] = None,
    ) -> List[MCQuestion]:
        """Build multiple-choice questions with anti-artifact guards.

        Args:
            questions: List of tossup questions.
            profile_builder: Profile builder for answer representations.
            reference_questions: Reference corpus used to build answer
                profiles and distractor rankings. Defaults to ``questions``
                to preserve legacy behavior.

        Returns:
            List of MCQuestion objects that passed all guards.
        """
        if not questions:
            self.last_build_stats = {
                "target_questions": 0,
                "reference_questions": 0,
                "reference_answer_count": 0,
                "built_questions": 0,
                "dropped_questions": 0,
                "retention_rate": 0.0,
                "drop_reasons": {},
                "repair": _new_repair_stats(),
            }
            return []

        ref_questions = questions if reference_questions is None else list(reference_questions)

        # ref_questions defaults to the target questions for convenience.
        # ``AnswerProfileBuilder.fit`` is a no-op when ``qid_set`` already
        # matches the cached fit, so we can call it on every build to keep
        # the builder's invariant local and the cache check explicit.
        profile_builder.fit(ref_questions)

        # Cache the per-reference-corpus heavy work so back-to-back
        # ``build()`` calls with identical ``reference_questions`` (the
        # common case in ``scripts/build_mc_dataset.py`` which iterates
        # train/val/test against the same raw_train) do not redo the
        # SBERT / TF-IDF encoder pass.
        #
        # ``category_random`` is excluded from the cache because its
        # ``_compute_rankings`` branch calls ``self.rng.shuffle`` once
        # per answer. Reusing a cached ``rankings`` dict would skip
        # those draws and leave ``self.rng`` in a different state than
        # fresh-per-split callers (the legacy behaviour) — silently
        # changing val/test option ordering and the distractor-fallback
        # path. Encoder-bound strategies are deterministic w.r.t. rng,
        # so caching them is safe and gives the full perf win.
        # Cache key includes content fingerprints so an in-place mutation
        # of ``q.question`` / ``q.answer_primary`` / ``q.category`` /
        # ``q.clean_answers`` between builds invalidates the cache. Two
        # distinct corpora with the same qid set but different content
        # (e.g. tournament-duplicate qids with edited answer text, or an
        # alias-normalisation refresh that rewrites ``clean_answers``)
        # also miss the cache, defending the invariant "same key ⇒ same
        # answer_profiles + answer_to_aliases ⇒ same rankings". Aliases
        # specifically feed ``_prepare_lookup``'s ``answer_to_aliases``,
        # so reusing the cache after an alias refresh would otherwise
        # leak stale aliases into downstream guard checks and distractor
        # filtering.
        #
        # We also fold the ``profile_builder`` config knobs into the key
        # (``max_tokens_per_profile`` / ``min_questions_per_answer``)
        # because the cache stores ``answer_profiles`` -- the output of
        # ``profile_builder.build_profiles(ref_questions)``. Reusing an
        # ``MCBuilder`` across calls that pass differently-configured
        # ``AnswerProfileBuilder`` instances would otherwise silently
        # return distractors derived from the first call's profile
        # settings, corrupting any in-process comparison of profile
        # variants.
        profile_cfg = (
            getattr(profile_builder, "max_tokens_per_profile", None),
            getattr(profile_builder, "min_questions_per_answer", None),
        )
        ref_key = (
            profile_cfg,
            frozenset(
                (
                    q.qid,
                    q.answer_primary,
                    q.category,
                    hash(q.question),
                    hash(tuple(sorted(q.clean_answers))),
                )
                for q in ref_questions
            ),
        )
        cacheable = self.strategy != "category_random"
        if (
            not cacheable
            or self._ref_cache is None
            or self._ref_cache_key != ref_key
        ):
            answer_profiles = profile_builder.build_profiles(ref_questions)
            answer_to_aliases, answer_to_category, _answer_to_norm, answers = (
                self._prepare_lookup(ref_questions)
            )
            rankings = self._compute_rankings(
                answers, answer_profiles, answer_to_category
            )
            if cacheable:
                self._ref_cache = {
                    "answer_to_aliases": answer_to_aliases,
                    "answer_to_category": answer_to_category,
                    "answers": answers,
                    "answer_profiles": answer_profiles,
                    "rankings": rankings,
                }
                self._ref_cache_key = ref_key
        else:
            answer_to_aliases = self._ref_cache["answer_to_aliases"]
            answer_to_category = self._ref_cache["answer_to_category"]
            answers = self._ref_cache["answers"]
            answer_profiles = self._ref_cache["answer_profiles"]
            rankings = self._ref_cache["rankings"]

        mc_questions: List[MCQuestion] = []
        drop_reasons: Dict[str, int] = defaultdict(int)
        repair_stats = _new_repair_stats()

        for q in questions:
            target_k = self._target_k()
            gold = q.answer_primary
            if gold not in answer_to_aliases:
                drop_reasons["unseen_gold_answer"] += 1
                continue
            gold_aliases = answer_to_aliases.get(gold, [gold])
            gold_norms = {str(normalize_answer(alias)) for alias in gold_aliases}
            ranked = rankings.get(gold, [a for a in answers if a != gold])
            selected: List[str] = []

            # Select distractors from ranked list
            for candidate in ranked:
                if candidate == gold:
                    continue
                if self._aliases_collide(candidate, gold_aliases, _gold_norms=gold_norms):
                    continue
                if self._violates_duplicate_guard(candidate, selected):
                    continue
                selected.append(candidate)
                if len(selected) >= target_k - 1:
                    break

            # If not enough distractors from ranking, try random fallback
            selected_after_ranked = selected[:]
            fallback_rng_state = self.rng.getstate()
            fallback_order: Optional[List[str]] = None
            if len(selected) < target_k - 1:
                fallback = [a for a in answers if a not in selected and a != gold]
                self.rng.shuffle(fallback)
                fallback_order = fallback[:]
                for candidate in fallback:
                    if self._aliases_collide(candidate, gold_aliases, _gold_norms=gold_norms):
                        continue
                    if self._violates_duplicate_guard(candidate, selected):
                        continue
                    selected.append(candidate)
                    if len(selected) >= target_k - 1:
                        break

            # Skip question if we can't find enough valid distractors
            if len(selected) < target_k - 1:
                drop_reasons["insufficient_distractors"] += 1
                continue

            # Create options and shuffle
            option_answer_primary = [gold] + selected[:target_k - 1]
            self.rng.shuffle(option_answer_primary)
            gold_index = option_answer_primary.index(gold)
            options = option_answer_primary[:]

            length_guard_failed = self._violates_length_ratio_guard(options)
            question_guard_failed = self._violates_question_overlap_guard(
                q.question,
                options,
            )
            if length_guard_failed or question_guard_failed:
                if length_guard_failed:
                    repair_stats["length_ratio_triggers"] += 1
                if question_guard_failed:
                    repair_stats["question_overlap_triggers"] += 1
                if length_guard_failed and question_guard_failed:
                    repair_stats["simultaneous_guard_triggers"] += 1

                repair_stats["attempted_questions"] += 1

                # Replacing distractors cannot repair a leaked gold answer.
                if self._violates_question_overlap_guard(q.question, [gold]):
                    repair_stats["failed_questions"] += 1
                    repair_stats["unrecoverable_gold_overlap_questions"] += 1
                    drop_reasons["question_overlap_guard"] += 1
                    continue

                repair = self._repair_options_after_guard_failure(
                    q.question,
                    gold,
                    selected[: target_k - 1],
                    ranked,
                    answers,
                    gold_aliases,
                    gold_norms,
                    target_k,
                    option_answer_primary,
                    fallback_rng_state,
                    selected_after_ranked,
                    fallback_order,
                )
                repair_stats["candidate_attempts"] += repair.candidate_attempts
                repair_stats["candidate_scans"] += repair.candidate_scans
                if repair.options is None:
                    repair_stats["failed_questions"] += 1
                    if repair.budget_exhausted:
                        repair_stats["budget_exhausted_questions"] += 1
                        drop_reasons["repair_budget_exhausted"] += 1
                    else:
                        repair_stats["exhaustive_no_solution_questions"] += 1
                        drop_reasons["guard_repair_failed"] += 1
                    continue

                repair_stats["succeeded_questions"] += 1
                if repair.source == "ranked":
                    repair_stats["ranked_successes"] += 1
                else:
                    repair_stats["fallback_successes"] += 1
                option_answer_primary = repair.options
                gold_index = option_answer_primary.index(gold)
                options = option_answer_primary[:]

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

        self.last_build_stats = {
            "target_questions": len(questions),
            "reference_questions": len(ref_questions),
            "reference_answer_count": len(answers),
            "built_questions": len(mc_questions),
            "dropped_questions": len(questions) - len(mc_questions),
            "retention_rate": (
                len(mc_questions) / len(questions) if questions else 0.0
            ),
            "drop_reasons": dict(drop_reasons),
            "repair": repair_stats,
        }
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
        max_repair_attempts=guards.get("max_repair_attempts", 10_000),
        random_seed=random_seed,
    )
    return builder.build(questions=questions, profile_builder=profile_builder)
