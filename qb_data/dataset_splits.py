"""
Stratified dataset splitting utilities for quiz bowl data.

This module provides functions to create train/val/test splits that maintain
category distribution across all splits.
"""

import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

from qb_data.data_loader import TossupQuestion
from qb_data.text_utils import normalize_answer


def normalize_question_text(text: str) -> str:
    """Return the shared split-integrity key for a question.

    Parameters
    ----------
    text
        Raw question text.

    Returns
    -------
    str
        NFKC-normalized, case-folded text with collapsed whitespace.

    Raises
    ------
    TypeError
        If ``text`` is not a string.

    Notes
    -----
    Compatibility normalization is deliberately conservative: Unicode NFKC,
    case-folding, and whitespace collapse. Punctuation remains significant.
    """
    if not isinstance(text, str):
        raise TypeError("question text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def normalize_split_answer(answer: str) -> str:
    """Normalize an answer for split-integrity comparisons."""
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    return normalize_answer(unicodedata.normalize("NFKC", answer).casefold())


def _target_counts(total: int, ratios: List[float]) -> list[int]:
    """Hamilton-apportion ``total`` with train→val→test as the tie order."""
    quotas = [total * ratio for ratio in ratios]
    targets = [math.floor(quota) for quota in quotas]
    remaining = total - sum(targets)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(quotas[index] - targets[index]), index),
    )
    for index in order[:remaining]:
        targets[index] += 1
    return targets


def create_stratified_splits(
    questions: List[TossupQuestion],
    ratios: List[float] = [0.7, 0.15, 0.15],
    seed: int = 42
) -> Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]:
    """
    Create stratified train/val/test splits maintaining category distribution.

    Parameters
    ----------
    questions : List[TossupQuestion]
        List of questions to split
    ratios : List[float]
        Three finite, nonnegative train/val/test ratios that sum to 1.0.
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]
        Train, validation, and test splits

    Raises
    ------
    ValueError
        If ratios are invalid or questions list is empty
    """
    # Validate inputs
    if not questions:
        raise ValueError("Cannot split empty question list")

    if not isinstance(ratios, (list, tuple)) or len(ratios) != 3:
        raise ValueError("ratios must contain exactly three values")
    normalized_inputs: list[float] = []
    for value in ratios:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("ratios must be finite, nonnegative numeric values")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "ratios must be finite, nonnegative numeric values"
            ) from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("ratios must be finite, nonnegative numeric values")
        normalized_inputs.append(numeric)
    try:
        ratio_total = math.fsum(normalized_inputs)
    except OverflowError as exc:
        raise ValueError("ratios must sum to 1.0") from exc
    if not math.isfinite(ratio_total) or abs(ratio_total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratio_total}")
    # Normalize the accepted tolerance window so apportionment always allocates
    # exactly the observed total, even for very large datasets.
    ratios = [value / ratio_total for value in normalized_inputs]

    # Group globally by normalized question text before considering categories.
    # This prevents paraphrase-equivalent rows from leaking across splits.
    text_groups: dict[str, list[TossupQuestion]] = defaultdict(list)
    qid_text: dict[str, str] = {}
    for q in questions:
        text_key = normalize_question_text(q.question)
        if not text_key:
            raise ValueError(f"question {q.qid!r} has empty normalized text")
        qid = str(q.qid)
        if qid in qid_text:
            raise ValueError(f"duplicate question ID {qid!r}")
        qid_text[qid] = text_key
        text_groups[text_key].append(q)

    for text_key, group in text_groups.items():
        normalized_answers = {
            normalize_split_answer(question.answer_primary)
            for question in group
        }
        if "" in normalized_answers:
            qids = sorted(str(question.qid) for question in group)
            raise ValueError(
                "normalized question group has an empty normalized answer: "
                f"text={text_key!r}, qids={qids}"
            )
        if len(normalized_answers) > 1:
            qids = sorted(str(question.qid) for question in group)
            raise ValueError(
                "normalized question group has conflicting normalized answers: "
                f"text={text_key!r}, qids={qids}, answers={sorted(normalized_answers)}"
            )

    # Preserve category proportions as closely as atomic text groups allow. The
    # greedy objective compares category and global count errors against the same
    # integer targets the legacy row-wise splitter used.
    category_groups: dict[str, list[TossupQuestion]] = defaultdict(list)
    for q in questions:
        category_groups[str(q.category)].append(q)

    split_count = 3
    category_targets = {
        category: _target_counts(len(category_questions), ratios)
        for category, category_questions in category_groups.items()
    }
    global_targets = _target_counts(len(questions), ratios)
    category_counts = {
        category: [0] * split_count
        for category in category_groups
    }
    global_counts = [0] * split_count

    def _seeded_group_key(item: tuple[str, list[TossupQuestion]]) -> tuple[int, str]:
        text_key, group = item
        digest = hashlib.sha256(f"{seed}\0{text_key}".encode("utf-8")).hexdigest()
        return (-len(group), digest)

    ordered_groups = sorted(text_groups.items(), key=_seeded_group_key)
    split_questions: list[list[TossupQuestion]] = [[], [], []]

    for _text_key, group in ordered_groups:
        group_category_counts: dict[str, int] = defaultdict(int)
        for question in group:
            group_category_counts[str(question.category)] += 1

        def _assignment_penalty(split_idx: int) -> tuple[float, int]:
            penalty = 0.0
            for category, added in group_category_counts.items():
                targets = category_targets[category]
                for idx in range(split_count):
                    count = category_counts[category][idx]
                    if idx == split_idx:
                        count += added
                    target = targets[idx]
                    scale = max(1, target)
                    penalty += ((count - target) / scale) ** 2

            for idx in range(split_count):
                count = global_counts[idx]
                if idx == split_idx:
                    count += len(group)
                target = global_targets[idx]
                scale = max(1, target)
                penalty += ((count - target) / scale) ** 2

            # Prefer the split with more remaining capacity on an exact tie.
            remaining = global_targets[split_idx] - global_counts[split_idx]
            return penalty, -remaining

        selected = min(range(split_count), key=_assignment_penalty)
        ordered_members = sorted(group, key=lambda question: str(question.qid))
        split_questions[selected].extend(ordered_members)
        global_counts[selected] += len(group)
        for category, added in group_category_counts.items():
            category_counts[category][selected] += added

    train_questions, val_questions, test_questions = split_questions

    # Verify all questions assigned exactly once
    total_original = len(questions)
    total_split = len(train_questions) + len(val_questions) + len(test_questions)

    if total_original != total_split:
        raise RuntimeError(f"Split mismatch: {total_original} original vs {total_split} split")

    qid_sets = [
        {str(question.qid) for question in split}
        for split in split_questions
    ]
    text_sets = [
        {normalize_question_text(question.question) for question in split}
        for split in split_questions
    ]
    for left in range(split_count):
        for right in range(left + 1, split_count):
            if qid_sets[left] & qid_sets[right]:
                raise RuntimeError("question ID overlap across generated splits")
            if text_sets[left] & text_sets[right]:
                raise RuntimeError("normalized question-text overlap across generated splits")

    # Log category distribution statistics
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_questions)} questions ({len(train_questions)/total_original:.1%})")
    print(f"  Val:   {len(val_questions)} questions ({len(val_questions)/total_original:.1%})")
    print(f"  Test:  {len(test_questions)} questions ({len(test_questions)/total_original:.1%})")

    # Category distribution analysis
    train_categories = defaultdict(int)
    val_categories = defaultdict(int)
    test_categories = defaultdict(int)

    for q in train_questions:
        train_categories[q.category] += 1
    for q in val_questions:
        val_categories[q.category] += 1
    for q in test_questions:
        test_categories[q.category] += 1

    all_categories = set(train_categories.keys()) | set(val_categories.keys()) | set(test_categories.keys())
    print(f"\nCategory distribution ({len(all_categories)} categories):")

    for category in sorted(all_categories)[:5]:  # Show first 5 categories
        orig_count = len(category_groups[category])
        train_count = train_categories.get(category, 0)
        val_count = val_categories.get(category, 0)
        test_count = test_categories.get(category, 0)
        print(f"  {category}: {train_count}/{val_count}/{test_count} (orig: {orig_count})")

    if len(all_categories) > 5:
        print(f"  ... and {len(all_categories) - 5} more categories")

    return train_questions, val_questions, test_questions


def save_splits(
    train: List[TossupQuestion],
    val: List[TossupQuestion],
    test: List[TossupQuestion],
    output_dir: str = "data"
) -> None:
    """
    Save dataset splits to JSON files with metadata.

    Parameters
    ----------
    train : List[TossupQuestion]
        Training split
    val : List[TossupQuestion]
        Validation split
    test : List[TossupQuestion]
        Test split
    output_dir : str
        Directory to save split files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Helper to convert TossupQuestion to dict
    def questions_to_dict(questions: List[TossupQuestion]) -> List[Dict[str, Any]]:
        return [
            {
                "qid": q.qid,
                "question": q.question,
                "tokens": q.tokens,
                "answer_primary": q.answer_primary,
                "clean_answers": q.clean_answers,
                "run_indices": q.run_indices,
                "human_buzz_positions": q.human_buzz_positions,
                "category": q.category,
                "cumulative_prefixes": q.cumulative_prefixes
            }
            for q in questions
        ]

    # Calculate category distributions for metadata
    def get_category_distribution(questions: List[TossupQuestion]) -> Dict[str, int]:
        dist = defaultdict(int)
        for q in questions:
            dist[q.category] += 1
        return dict(dist)

    # Save each split with metadata
    splits = [
        ("train_dataset.json", train),
        ("val_dataset.json", val),
        ("test_dataset.json", test)
    ]

    for filename, questions in splits:
        filepath = output_path / filename

        data = {
            "metadata": {
                "total_questions": len(questions),
                "categories": len(set(q.category for q in questions)),
                "category_distribution": get_category_distribution(questions),
                "split_type": filename.replace("_dataset.json", "")
            },
            "questions": questions_to_dict(questions)
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(questions)} questions to {filepath}")

    # Save combined metadata file
    metadata_path = output_path / "split_metadata.json"
    metadata = {
        "train": {
            "count": len(train),
            "categories": get_category_distribution(train)
        },
        "val": {
            "count": len(val),
            "categories": get_category_distribution(val)
        },
        "test": {
            "count": len(test),
            "categories": get_category_distribution(test)
        },
        "total_questions": len(train) + len(val) + len(test),
        "split_ratios": [
            len(train) / (len(train) + len(val) + len(test)),
            len(val) / (len(train) + len(val) + len(test)),
            len(test) / (len(train) + len(val) + len(test))
        ]
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved split metadata to {metadata_path}")
