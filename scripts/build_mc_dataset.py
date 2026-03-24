#!/usr/bin/env python3
"""
Build multiple-choice dataset from QANTA quiz bowl questions.

This script orchestrates the complete data pipeline:
1. Load questions from CSV or HuggingFace
2. Create stratified raw train/val/test splits
3. Build answer profiles from the raw training split only
4. Generate MC questions for train/val/test using train-only reference data
5. Save processed datasets as JSON

Usage:
    python scripts/build_mc_dataset.py
    python scripts/build_mc_dataset.py --smoke  # Quick test with 50 questions in artifacts/smoke
    python scripts/build_mc_dataset.py --config configs/custom.yaml
    python scripts/build_mc_dataset.py --data.K=5 --data.distractor_strategy=tfidf_profile
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qb_data import TossupQuestion
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.config import load_config, merge_overrides, resolve_data_loading_options
from qb_data.data_loader import QANTADatasetLoader
from qb_data.dataset_splits import create_stratified_splits
from qb_data.huggingface_loader import load_from_huggingface
from qb_data.mc_builder import MCBuilder, MCQuestion
from scripts._common import parse_overrides

DEFAULT_OUTPUT_DIR = Path("data/processed")
SMOKE_OUTPUT_DIR = Path("artifacts/smoke")


def resolve_output_dir(output_dir: Optional[str], smoke: bool) -> Path:
    """Resolve the dataset output directory from CLI inputs."""
    if output_dir is not None:
        return Path(output_dir)
    return SMOKE_OUTPUT_DIR if smoke else DEFAULT_OUTPUT_DIR


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for dataset construction."""
    parser = argparse.ArgumentParser(
        description="Build multiple-choice dataset from QANTA questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=(
            "Path to YAML configuration file. Defaults to configs/default.yaml, "
            "or the smoke config path selected by load_config() when --smoke is set."
        ),
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Use smoke test settings (50 questions, quick run, outputs to artifacts/smoke by default).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save processed datasets. Defaults to data/processed, or artifacts/smoke when --smoke is set.',
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides in format: data.K=5 data.distractor_strategy=tfidf_profile',
    )

    return parser.parse_args(argv)


def save_json(path: Path, data: Any) -> None:
    """
    Save data to JSON file (lists of dataclasses or plain dicts).

    Parameters
    ----------
    path : Path
        Output file path
    data : Any
        Data to serialize (list of dataclass objects, dict, etc.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_serializable(item: Any) -> Any:
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, list):
            return [to_serializable(v) for v in item]
        if isinstance(item, dict):
            return {k: to_serializable(v) for k, v in item.items()}
        return item

    json_data = to_serializable(data)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    item_count = len(data) if isinstance(data, list) else 1
    print(f"Saved {item_count} items to {path}")


def make_profile_builder(config: dict[str, Any]) -> AnswerProfileBuilder:
    """Create the canonical answer profile builder from config."""
    return AnswerProfileBuilder(
        max_tokens_per_profile=config['answer_profiles']['max_tokens_per_profile'],
        min_questions_per_answer=config['answer_profiles']['min_questions_per_answer'],
    )


def make_mc_builder(config: dict[str, Any]) -> MCBuilder:
    """Create the canonical MC builder from config."""
    data_cfg = config['data']
    return MCBuilder(
        K=data_cfg['K'],
        strategy=data_cfg['distractor_strategy'],
        embedding_model=config['likelihood'].get(
            'sbert_name',
            config['likelihood'].get('embedding_model', 'all-MiniLM-L6-v2'),
        ),
        openai_model=config['likelihood'].get('openai_model', 'text-embedding-3-small'),
        variable_K=bool(data_cfg.get('variable_K', False)),
        min_K=int(data_cfg.get('min_K', 2)),
        max_K=int(data_cfg['max_K']) if data_cfg.get('max_K') is not None else None,
        **config['mc_guards'],
    )


def build_metadata_entry(raw_questions: List[TossupQuestion], mc_questions: List[MCQuestion], stats: dict[str, Any]) -> dict[str, Any]:
    """Build split metadata for ``build_metadata.json``."""
    retained = len(mc_questions)
    raw_count = len(raw_questions)
    return {
        "raw_count": raw_count,
        "retained_count": retained,
        "dropped_count": max(0, raw_count - retained),
        "retention_rate": retained / raw_count if raw_count else 0.0,
        "reference_answer_count": int(stats.get("reference_answer_count", 0)),
        "drop_reasons": dict(stats.get("drop_reasons", {})),
    }


def warn_on_low_retention(split_name: str, metadata: dict[str, Any], threshold: float) -> None:
    """Warn when retained questions for a split drop below the threshold."""
    retention = float(metadata.get("retention_rate", 0.0))
    if retention < threshold:
        print(
            f"Warning: {split_name} retention {retention:.1%} is below the "
            f"configured threshold of {threshold:.0%}"
        )


def print_statistics(
    train: List[MCQuestion],
    val: List[MCQuestion],
    test: List[MCQuestion],
    profile_builder: Optional[AnswerProfileBuilder] = None,
    mc_builder: Optional[MCBuilder] = None
) -> None:
    """
    Print dataset statistics.

    Parameters
    ----------
    train : List[MCQuestion]
        Training split
    val : List[MCQuestion]
        Validation split
    test : List[MCQuestion]
        Test split
    profile_builder : Optional[AnswerProfileBuilder]
        Answer profile builder for profile stats
    mc_builder : Optional[MCBuilder]
        MC builder for guard rejection stats
    """
    print("\n" + "="*60)
    print("Dataset Construction Complete")
    print("="*60)

    # Split statistics
    total = len(train) + len(val) + len(test)
    print(f"\nTotal MC questions: {total}")
    if total == 0:
        print("  Warning: all splits are empty after guard filtering.")
        return
    print(f"  Train: {len(train)} ({100*len(train)/total:.1f}%)")
    print(f"  Val:   {len(val)} ({100*len(val)/total:.1f}%)")
    print(f"  Test:  {len(test)} ({100*len(test)/total:.1f}%)")

    # Category distribution
    def get_categories(questions):
        return set(q.category for q in questions if q.category)

    all_categories = get_categories(train) | get_categories(val) | get_categories(test)
    print(f"\nCategories: {len(all_categories)}")

    # Sample categories
    sample_cats = sorted(all_categories)[:5]
    print("Sample categories:", ", ".join(sample_cats))

    # Answer profile statistics
    if profile_builder and hasattr(profile_builder, '_grouped'):
        print(f"\nAnswer profiles: {len(profile_builder._grouped)}")
        # Get average questions per answer
        avg_questions = sum(len(items) for items in profile_builder._grouped.values()) / len(profile_builder._grouped)
        print(f"Average questions per answer: {avg_questions:.1f}")

    # Guard rejection statistics (from last_build_stats)
    if mc_builder and hasattr(mc_builder, 'last_build_stats'):
        drop_reasons = mc_builder.last_build_stats.get("drop_reasons", {})
        if drop_reasons:
            print("\nGuard rejection statistics:")
            for reason, count in drop_reasons.items():
                print(f"  {reason}: {count} rejections")

    # Sample MC question
    if train:
        sample = train[0]
        print(f"\nSample MC question:")
        # Get first sentence from the question
        first_sentence = sample.question[:100] + "..." if len(sample.question) > 100 else sample.question
        print(f"  Question: {first_sentence}")
        print(f"  Correct answer: {sample.answer_primary}")
        print(f"  Options: {', '.join(sample.options[:3])}...")
        print(f"  Category: {sample.category}")


def main(argv: Optional[list[str]] = None):
    """Main entry point for dataset construction."""
    args = parse_args(argv)

    # Start timing
    start_time = time.time()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config, smoke=args.smoke)

    # Apply overrides
    overrides = parse_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = merge_overrides(config, overrides)

    # Create output directory
    output_dir = resolve_output_dir(args.output_dir, smoke=args.smoke)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    print("\nLoading questions...")
    questions = None
    data_opts = resolve_data_loading_options(config, smoke=args.smoke)

    # Try CSV first
    csv_path = data_opts.get('csv_path')
    if csv_path and Path(csv_path).exists():
        print(f"Loading from CSV: {csv_path}")
        loader = QANTADatasetLoader()
        questions = loader.load_from_csv(csv_path)
        print(f"Loaded {len(questions)} questions from CSV")

    # Fallback to HuggingFace if configured
    if questions is None and data_opts.get('use_huggingface'):
        print("CSV not found, falling back to HuggingFace")
        dataset_name = data_opts.get('dataset') or 'qanta-challenge/acf-co24-tossups'
        questions = load_from_huggingface(
            dataset_name,
            config_name=data_opts.get('dataset_config'),
            split=data_opts.get('split', 'eval'),
        )
        print(f"Loaded {len(questions)} questions from HuggingFace")

    if questions is None:
        raise FileNotFoundError(f"Could not load questions from {csv_path} and HuggingFace fallback not enabled")

    # Apply configured limit after loading
    max_questions = data_opts.get('max_questions')
    if max_questions is not None and len(questions) > int(max_questions):
        print(f"Limiting dataset to {int(max_questions)} questions")
        questions = questions[: int(max_questions)]

    # Create raw stratified splits before any answer-profile fitting.
    print("\nCreating raw stratified splits...")
    data_cfg = config['data']
    ratios = [
        data_cfg['train_ratio'],
        data_cfg['val_ratio'],
        data_cfg['test_ratio'],
    ]
    split_seed = int(data_cfg.get('shuffle_seed', 42))
    raw_train, raw_val, raw_test = create_stratified_splits(
        questions,
        ratios=ratios,
        seed=split_seed,
    )

    # Initialize answer profile builder; fitting will be done inside MCBuilder
    # using the raw train split as the reference corpus.
    print("\nInitializing answer profile builder (will be fit on raw train during MC construction)...")
    profile_builder = make_profile_builder(config)

    # Construct MC questions for each split against the train reference corpus.
    print("\nConstructing split-safe MC questions...")
    split_targets = {
        "train": raw_train,
        "val": raw_val,
        "test": raw_test,
    }
    built_splits: dict[str, list[MCQuestion]] = {}
    build_stats: dict[str, dict[str, Any]] = {}
    builders: dict[str, MCBuilder] = {}
    for split_name, target_questions in split_targets.items():
        builder = make_mc_builder(config)
        built = builder.build(
            target_questions,
            profile_builder,
            reference_questions=raw_train,
        )
        built_splits[split_name] = built
        build_stats[split_name] = builder.last_build_stats
        builders[split_name] = builder
        print(
            f"  {split_name}: kept {len(built)}/{len(target_questions)} "
            f"({builder.last_build_stats.get('retention_rate', 0.0):.1%})"
        )

    train = built_splits["train"]
    val = built_splits["val"]
    test = built_splits["test"]
    mc_questions = train + val + test

    smoke_threshold = 0.50
    full_threshold = 0.80
    retention_threshold = smoke_threshold if args.smoke else full_threshold
    val_metadata = build_metadata_entry(raw_val, val, build_stats["val"])
    test_metadata = build_metadata_entry(raw_test, test, build_stats["test"])
    warn_on_low_retention("val", val_metadata, retention_threshold)
    warn_on_low_retention("test", test_metadata, retention_threshold)

    # Save datasets
    print("\nSaving datasets...")
    save_json(output_dir / "mc_dataset.json", mc_questions)
    save_json(output_dir / "train_dataset.json", train)
    save_json(output_dir / "val_dataset.json", val)
    save_json(output_dir / "test_dataset.json", test)

    build_metadata = {
        "split_reference": "train_dataset.json is the canonical reference corpus for all splits",
        "retention_thresholds": {
            "smoke": smoke_threshold,
            "full": full_threshold,
        },
        "splits": {
            "train": build_metadata_entry(raw_train, train, build_stats["train"]),
            "val": val_metadata,
            "test": test_metadata,
        },
        "combined_mc_dataset_count": len(mc_questions),
    }
    save_json(output_dir / "build_metadata.json", build_metadata)

    # Save answer profiles for debugging
    if profile_builder._grouped:
        profiles_dict = {
            answer: {
                'question_count': len(items),
                'sample_qids': [qid for qid, _ in items[:5]]  # First 5 question IDs
            }
            for answer, items in profile_builder._grouped.items()
        }
        with open(output_dir / "answer_profiles.json", 'w') as f:
            json.dump(profiles_dict, f, indent=2)
        print(f"Saved answer profiles to {output_dir / 'answer_profiles.json'}")

    # Print statistics
    print_statistics(train, val, test, profile_builder, builders["train"])

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")

    if args.smoke:
        # Print sample MC questions for verification
        print("\n" + "="*60)
        print("Sample MC Questions (Smoke Test)")
        print("="*60)

        for i, q in enumerate(train[:3], 1):
            print(f"\nQuestion {i}:")
            # Get first clue from cumulative_prefixes if available
            if q.cumulative_prefixes:
                first_clue = q.cumulative_prefixes[0][:100] + "..." if len(q.cumulative_prefixes[0]) > 100 else q.cumulative_prefixes[0]
            else:
                first_clue = q.question[:100] + "..." if len(q.question) > 100 else q.question
            print(f"  First clue: {first_clue}")
            print(f"  Category: {q.category}")
            print(f"  Correct: {q.answer_primary}")
            print(f"  Options: {', '.join(q.options[:3])}...")

    print("\nDataset construction complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
