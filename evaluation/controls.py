"""
Control Experiments for Quiz Bowl Buzzer Evaluation

Implements three control experiments to validate that the buzzer agent
genuinely uses question clues rather than exploiting surface-form artifacts:

1. **Choices-only control**: Strips all clues, trains a logistic regression
   on option surface features (char n-grams, length, capitalization). Expected
   accuracy ~25% (1/K) if options have no exploitable artifacts.

2. **Shuffle control**: Randomizes option ordering to verify the agent has
   no position bias. Performance should be unchanged.

3. **Alias substitution control**: Swaps answer text with aliases to verify
   robustness to surface-form changes.

Ported from qb-rl reference implementation (evaluation/controls.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from qb_data.mc_builder import MCQuestion

if TYPE_CHECKING:
    from agents.threshold_buzzer import _PrecomputedQuestion


def _option_scalar_features(option: str) -> list[float]:
    """Extract scalar surface features from a single option string.

    Parameters
    ----------
    option : str
        Answer option text.

    Returns
    -------
    list[float]
        Six scalar features: char length, token count, has_parens,
        has_comma, is_title, is_lower.
    """
    tokens = option.split()
    has_parens = 1.0 if "(" in option or ")" in option else 0.0
    has_comma = 1.0 if "," in option else 0.0
    is_title = 1.0 if option.istitle() else 0.0
    is_lower = 1.0 if option.islower() else 0.0
    return [
        float(len(option)),
        float(len(tokens)),
        has_parens,
        has_comma,
        is_title,
        is_lower,
    ]


def _cross_option_features(options: list[str]) -> list[float]:
    """Extract cross-option comparative features.

    Parameters
    ----------
    options : list[str]
        All answer options for a question.

    Returns
    -------
    list[float]
        Three features: max/min length ratio, length std, number of
        distinct capitalization patterns.
    """
    lengths = np.array(
        [max(1, len(o.split())) for o in options], dtype=np.float32
    )
    cap_patterns = len(
        set(
            ("title" if o.istitle() else "lower" if o.islower() else "mixed")
            for o in options
        )
    )
    return [
        float(lengths.max() / lengths.min()),
        float(lengths.std()),
        float(cap_patterns),
    ]


def run_choices_only_control(
    questions: list[MCQuestion],
    random_seed: int = 13,
    test_fraction: float = 0.25,
) -> dict[str, float]:
    """Run choices-only control: predict answer from surface features only.

    Strips all question clues and trains a logistic regression on option
    surface features (char n-grams, length, capitalization patterns).
    Expected accuracy ~25% (1/K) if options are well-constructed.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    random_seed : int
        Seed for reproducible train/test split.
    test_fraction : float
        Fraction of questions held out for testing.

    Returns
    -------
    dict[str, float]
        Control results: accuracy, chance baseline, and test set size.
    """
    if not questions:
        return {"accuracy": 0.0, "chance": 0.0, "n_test": 0.0}

    rng = random.Random(random_seed)
    shuffled = questions[:]
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - test_fraction)))
    train_q = shuffled[:split_idx]
    test_q = shuffled[split_idx:]
    if not test_q:
        test_q = train_q

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
    vec.fit([opt for q in train_q for opt in q.options])

    def build_matrix(
        rows: list[MCQuestion],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        X = []
        y = []
        group_sizes: list[int] = []
        for q in rows:
            cross = _cross_option_features(q.options)
            group_sizes.append(len(q.options))
            tfidf = vec.transform(q.options).toarray()
            for i, option in enumerate(q.options):
                feat = np.array(
                    _option_scalar_features(option) + cross, dtype=np.float32
                )
                row = np.concatenate([feat, tfidf[i]], axis=0)
                X.append(row)
                y.append(1 if i == q.gold_index else 0)
        return np.array(X), np.array(y), group_sizes

    X_train, y_train, _ = build_matrix(train_q)
    X_test, y_test, test_group_sizes = build_matrix(test_q)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    offset = 0
    correct = 0
    total = 0
    for q, group_size in zip(test_q, test_group_sizes):
        group_probs = probs[offset : offset + group_size]
        pred_idx = int(np.argmax(group_probs))
        if pred_idx == q.gold_index:
            correct += 1
        total += 1
        offset += group_size

    accuracy = correct / max(1, total)
    chance = 1.0 / max(1, len(questions[0].options))
    return {
        "accuracy": float(accuracy),
        "chance": float(chance),
        "n_test": float(total),
    }


def shuffled_option_copy(
    question: MCQuestion, rng: random.Random
) -> MCQuestion:
    """Create a copy of an MCQuestion with shuffled option ordering.

    Parameters
    ----------
    question : MCQuestion
        Original question.
    rng : random.Random
        Random number generator for shuffling.

    Returns
    -------
    MCQuestion
        Copy with permuted options, profiles, answer_primary, and
        updated gold_index.
    """
    perm = list(range(len(question.options)))
    rng.shuffle(perm)
    new_options = [question.options[i] for i in perm]
    new_profiles = [question.option_profiles[i] for i in perm]
    new_answer_primary = [question.option_answer_primary[i] for i in perm]
    new_gold = perm.index(question.gold_index)
    return replace(
        question,
        options=new_options,
        option_profiles=new_profiles,
        option_answer_primary=new_answer_primary,
        gold_index=new_gold,
    )


def run_shuffle_control(
    questions: list[MCQuestion],
    evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run shuffle control: randomize option ordering and evaluate.

    Permutes the answer options for each question and runs the evaluator.
    If the agent has no position bias, performance should be unchanged.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    evaluator : callable
        Function that takes a list of MCQuestion and returns a metrics dict.
    random_seed : int
        Seed for reproducible shuffling.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics on shuffled questions.
    """
    rng = random.Random(random_seed)
    shuffled = [shuffled_option_copy(q, rng) for q in questions]
    return evaluator(shuffled)


def alias_substitution_copy(
    question: MCQuestion,
    alias_lookup: dict[str, list[str]],
    rng: random.Random,
) -> MCQuestion:
    """Create a copy of an MCQuestion with alias-substituted options.

    Parameters
    ----------
    question : MCQuestion
        Original question.
    alias_lookup : dict[str, list[str]]
        Mapping from canonical answer to list of known aliases.
    rng : random.Random
        Random number generator for alias selection.

    Returns
    -------
    MCQuestion
        Copy with alias-substituted option text and profiles.
    """
    new_options = []
    new_profiles = list(question.option_profiles)
    for i, (option_text, answer_primary) in enumerate(
        zip(question.options, question.option_answer_primary)
    ):
        aliases = [
            a
            for a in alias_lookup.get(answer_primary, [])
            if a and a != option_text
        ]
        if aliases:
            alias = rng.choice(aliases)
            new_options.append(alias)
            if new_profiles[i].strip() == answer_primary.strip():
                new_profiles[i] = alias
        else:
            new_options.append(option_text)
    return replace(question, options=new_options, option_profiles=new_profiles)


def run_alias_substitution_control(
    questions: list[MCQuestion],
    alias_lookup: dict[str, list[str]],
    evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run alias substitution control: swap answer text with aliases.

    Replaces option text with known aliases to verify the agent is robust
    to surface-form changes. Performance should be similar to full eval.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    alias_lookup : dict[str, list[str]]
        Mapping from canonical answer to list of known aliases.
    evaluator : callable
        Function that takes a list of MCQuestion and returns a metrics dict.
    random_seed : int
        Seed for reproducible alias selection.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics on alias-substituted questions.
    """
    rng = random.Random(random_seed)
    swapped = [
        alias_substitution_copy(q, alias_lookup=alias_lookup, rng=rng)
        for q in questions
    ]
    return evaluator(swapped)


def run_shuffle_control_precomputed(
    precomputed: list["_PrecomputedQuestion"],
    threshold: float,
    alpha: float,
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run shuffle control by permuting precomputed belief vectors.

    Produces numerically identical results to ``run_shuffle_control`` with
    a live ``SoftmaxProfileBuzzer`` evaluator, but makes zero
    ``likelihood_model.score()`` calls.  Instead, the belief vectors
    stored in each ``_PrecomputedQuestion`` are reordered according to
    the same random permutation that ``shuffled_option_copy`` would apply.

    Parameters
    ----------
    precomputed : list[_PrecomputedQuestion]
        Pre-computed belief distributions (one per question).
    threshold : float
        Buzz threshold for the softmax profile buzzer.
    alpha : float
        Sigmoid steepness for the confidence proxy.
    random_seed : int
        Seed for reproducible shuffling (must match the seed used in
        ``run_shuffle_control`` for equivalence).

    Returns
    -------
    dict[str, Any]
        Summary metrics with ``"runs"`` key containing per-question dicts.
    """
    from dataclasses import asdict

    from agents.threshold_buzzer import (
        _PrecomputedQuestion,
        _softmax_episode_from_precomputed,
    )
    from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics

    rng = random.Random(random_seed)
    runs: list[dict[str, Any]] = []
    for pq in precomputed:
        perm = list(range(pq.num_options))
        rng.shuffle(perm)
        new_gold = perm.index(pq.gold_index)
        shuffled_beliefs = [b[perm] for b in pq.beliefs]
        shuffled_pq = _PrecomputedQuestion(
            qid=pq.qid,
            gold_index=new_gold,
            num_options=pq.num_options,
            beliefs=shuffled_beliefs,
        )
        result = _softmax_episode_from_precomputed(shuffled_pq, threshold, alpha)
        runs.append(asdict(result))
    summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
    summary["runs"] = runs
    return summary


def bootstrap_ci(
    values: list[float],
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : list[float]
        Observed values.
    n_samples : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (0.05 = 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    samples = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(arr), size=len(arr))
        samples.append(float(arr[idx].mean()))
    lo = np.quantile(samples, alpha / 2.0)
    hi = np.quantile(samples, 1.0 - alpha / 2.0)
    return float(lo), float(hi)
