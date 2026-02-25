"""
Stratified dataset splitting utilities for quiz bowl data.

This module provides functions to create train/val/test splits that maintain
category distribution across all splits.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from qb_data import TossupQuestion


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
        Train/val/test split ratios (must sum to 1.0)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]
        Train, validation, and test splits

    Raises
    ------
    ValueError
        If ratios don't sum to 1.0 or questions list is empty
    """
    # Validate inputs
    if not questions:
        raise ValueError("Cannot split empty question list")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    # Initialize random generator for reproducibility
    rng = random.Random(seed)

    # Group questions by category
    category_groups = defaultdict(list)
    for q in questions:
        category_groups[q.category].append(q)

    # Initialize output lists
    train_questions = []
    val_questions = []
    test_questions = []

    # Split each category maintaining ratios
    for category, category_questions in category_groups.items():
        # Sort for deterministic splits
        sorted_questions = sorted(category_questions, key=lambda q: q.qid)

        # Shuffle with fixed seed for this category
        category_seed = seed + hash(category) % 1000000
        category_rng = random.Random(category_seed)
        shuffled = sorted_questions.copy()
        category_rng.shuffle(shuffled)

        n = len(shuffled)

        # Calculate split indices
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        # Handle small categories - ensure at least 1 in train if possible
        if n == 1:
            train_questions.extend(shuffled)
        elif n == 2:
            train_questions.extend(shuffled[:1])
            val_questions.extend(shuffled[1:])
        else:
            # Standard split
            train_questions.extend(shuffled[:train_end])
            val_questions.extend(shuffled[train_end:val_end])
            test_questions.extend(shuffled[val_end:])

    # Verify all questions assigned exactly once
    total_original = len(questions)
    total_split = len(train_questions) + len(val_questions) + len(test_questions)

    if total_original != total_split:
        raise RuntimeError(f"Split mismatch: {total_original} original vs {total_split} split")

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
                "answer": q.answer,
                "category": q.category,
                "tokenized_question": q.tokenized_question,
                "tokenized_answer": q.tokenized_answer,
                "run_indices": q.run_indices,
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