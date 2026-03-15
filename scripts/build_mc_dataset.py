#!/usr/bin/env python3
"""
Build multiple-choice dataset from QANTA quiz bowl questions.

This script orchestrates the complete data pipeline:
1. Load questions from CSV or HuggingFace
2. Build answer profiles from training data
3. Generate MC questions with anti-artifact guards
4. Create stratified train/val/test splits
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qb_data import TossupQuestion
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.config import load_config, merge_overrides, resolve_data_loading_options
from qb_data.data_loader import QANTADatasetLoader
from qb_data.dataset_splits import create_stratified_splits
from qb_data.huggingface_loader import load_from_huggingface
from qb_data.mc_builder import MCBuilder, MCQuestion

DEFAULT_OUTPUT_DIR = Path("data/processed")
SMOKE_OUTPUT_DIR = Path("artifacts/smoke")


def _parse_value(value: str) -> Any:
    """Parse a CLI override value string into a typed Python value.

    Tries JSON first, then bool/int/float, and falls back to str.
    """
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lstrip("-").isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def parse_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Parse CLI override arguments into flat dotted-key overrides.

    Returns a dict with dotted keys (e.g. ``{"data.K": 5}``) that
    ``merge_overrides`` can apply leaf-by-leaf without clobbering
    sibling config entries.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.  Positional ``overrides`` are
        ``key=value`` strings where *key* uses dot-notation
        (e.g. ``data.K=5``).

    Returns
    -------
    Dict[str, Any]
        Flat dotted-key overrides ready for ``merge_overrides()``.
    """
    overrides: Dict[str, Any] = {}
    if hasattr(args, "overrides") and args.overrides:
        for token in args.overrides:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            overrides[key] = _parse_value(value)
    return overrides


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


def save_json(path: Path, data: List[Any]) -> None:
    """
    Save dataclass objects to JSON file.

    Parameters
    ----------
    path : Path
        Output file path
    data : List[Any]
        List of dataclass objects (TossupQuestion or MCQuestion)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dictionaries
    if data and hasattr(data[0], '__dataclass_fields__'):
        # It's a dataclass, use asdict
        from dataclasses import asdict
        json_data = [asdict(item) for item in data]
    else:
        json_data = data

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} items to {path}")


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

    # Guard rejection statistics
    if mc_builder and hasattr(mc_builder, 'guard_stats'):
        stats = mc_builder.guard_stats
        if stats:
            print("\nGuard rejection statistics:")
            for guard_name, count in stats.items():
                print(f"  {guard_name}: {count} rejections")

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

    # Build answer profiles
    print("\nBuilding answer profiles...")
    profile_builder = AnswerProfileBuilder(
        max_tokens_per_profile=config['answer_profiles']['max_tokens_per_profile'],
        min_questions_per_answer=config['answer_profiles']['min_questions_per_answer']
    )
    profile_builder.fit(questions)
    print(f"Built {len(profile_builder._grouped)} answer profiles")

    # Construct MC questions with guards
    print("\nConstructing MC questions...")
    mc_builder = MCBuilder(
        K=config['data']['K'],
        strategy=config['data']['distractor_strategy'],
        embedding_model=config['likelihood'].get(
            'sbert_name',
            config['likelihood'].get('embedding_model', 'all-MiniLM-L6-v2'),
        ),
        openai_model=config['likelihood'].get('openai_model', 'text-embedding-3-small'),
        **config['mc_guards']
    )

    # Track guard statistics
    mc_builder.guard_stats = {}

    mc_questions = mc_builder.build(questions, profile_builder)
    print(f"Generated {len(mc_questions)} MC questions")

    if len(mc_questions) < len(questions):
        print(f"Note: {len(questions) - len(mc_questions)} questions filtered by guards")

    # Create stratified splits
    print("\nCreating stratified splits...")
    ratios = [
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        config['data']['test_ratio']
    ]

    train, val, test = create_stratified_splits(mc_questions, ratios=ratios)

    # Save datasets
    print("\nSaving datasets...")
    save_json(output_dir / "mc_dataset.json", mc_questions)
    save_json(output_dir / "train_dataset.json", train)
    save_json(output_dir / "val_dataset.json", val)
    save_json(output_dir / "test_dataset.json", test)

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
    print_statistics(train, val, test, profile_builder, mc_builder)

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
