#!/usr/bin/env python
"""Test script to verify MC construction with anti-artifact guards."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qb_data.data_loader import QANTADatasetLoader
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.mc_builder import MCBuilder
from qb_data.config import load_config


def main():
    """Test MC question construction with guards."""
    print("Testing MC Builder with Anti-Artifact Guards")
    print("=" * 50)

    # Load configuration
    config = load_config("configs/default.yaml")

    # Load test questions
    data_path = "data/test_questions.csv"
    if not os.path.exists(data_path):
        print(f"Error: Test data not found at {data_path}")
        print("Please ensure test_questions.csv exists")
        return 1

    # Load questions
    questions = QANTADatasetLoader.load_from_csv(data_path)
    print(f"\nLoaded {len(questions)} test questions")

    # Create answer profile builder
    profile_builder = AnswerProfileBuilder(
        max_tokens_per_profile=config["answer_profiles"]["max_tokens_per_profile"],
        min_questions_per_answer=config["answer_profiles"]["min_questions_per_answer"]
    )
    profile_builder.fit(questions)
    print(f"Built profiles for {len(profile_builder._grouped)} unique answers")

    # Create MC builder with guards from config
    mc_builder = MCBuilder(
        K=config["data"]["K"],
        strategy="tfidf_profile",  # Use TF-IDF since it doesn't require embeddings
        alias_edit_distance_threshold=config["mc_guards"]["alias_edit_distance_threshold"],
        duplicate_token_overlap_threshold=config["mc_guards"]["duplicate_token_overlap_threshold"],
        max_length_ratio=config["mc_guards"]["max_length_ratio"],
        random_seed=config["data"]["shuffle_seed"]
    )

    # Build MC questions
    print(f"\nBuilding MC questions with K={config['data']['K']} options...")
    mc_questions = mc_builder.build(questions, profile_builder)
    print(f"Created {len(mc_questions)} MC questions (from {len(questions)} originals)")

    # Calculate rejection rate
    rejection_rate = 1.0 - (len(mc_questions) / len(questions))
    print(f"Rejection rate: {rejection_rate:.1%} (due to guard violations)")

    # Print sample MC questions
    print("\n" + "=" * 50)
    print("Sample MC Questions:")
    print("=" * 50)

    for i, mc_q in enumerate(mc_questions[:3]):  # Show first 3
        print(f"\n[Question {i+1}]")
        print(f"Category: {mc_q.category or 'Unknown'}")
        print(f"Question ID: {mc_q.qid}")

        # Show first clue (truncated)
        first_clue = mc_q.tokens[0] if mc_q.tokens else mc_q.question[:100]
        print(f"First clue: {first_clue[:150]}...")

        print("\nOptions:")
        for j, option in enumerate(mc_q.options):
            marker = " [CORRECT]" if j == mc_q.gold_index else ""
            print(f"  {j+1}. {option}{marker}")

        print(f"\nDistractor strategy: {mc_q.distractor_strategy}")

        # Check guards for this question
        print("\nGuard checks:")

        # Check alias collision
        gold_aliases = [mc_q.answer_primary] + list(mc_q.clean_answers)
        alias_violations = []
        for j, option in enumerate(mc_q.options):
            if j != mc_q.gold_index:
                for alias in gold_aliases:
                    from difflib import SequenceMatcher
                    dist = 1.0 - SequenceMatcher(None, option.lower(), alias.lower()).ratio()
                    if dist < 0.2:
                        alias_violations.append((option, alias, dist))

        if alias_violations:
            print(f"  ✗ Alias collision detected: {alias_violations}")
        else:
            print("  ✓ No alias collisions")

        # Check token overlap between options
        from qb_data.mc_builder import _token_overlap
        high_overlaps = []
        for j in range(len(mc_q.options)):
            for k in range(j+1, len(mc_q.options)):
                overlap = _token_overlap(mc_q.options[j], mc_q.options[k])
                if overlap > 0.8:
                    high_overlaps.append((mc_q.options[j], mc_q.options[k], overlap))

        if high_overlaps:
            print(f"  ✗ High token overlap: {high_overlaps}")
        else:
            print("  ✓ No high token overlaps")

        # Check length ratio
        lengths = [len(o.split()) for o in mc_q.options]
        ratio = max(lengths) / max(1, min(lengths))
        if ratio > 3.0:
            print(f"  ✗ Length ratio violation: {ratio:.2f} (max: {max(lengths)}, min: {min(lengths)})")
        else:
            print(f"  ✓ Length ratio OK: {ratio:.2f}")

        # Check question overlap
        from qb_data.text_utils import normalize_answer
        q_norm = normalize_answer(mc_q.question).lower()
        overlaps = []
        for option in mc_q.options:
            o_norm = normalize_answer(option).lower()
            if o_norm and o_norm in q_norm:
                overlaps.append(option)

        if overlaps:
            print(f"  ✗ Options appear in question: {overlaps}")
        else:
            print("  ✓ No options in question text")

    # Print statistics
    print("\n" + "=" * 50)
    print("Statistics:")
    print("=" * 50)
    print(f"Total questions processed: {len(questions)}")
    print(f"MC questions built: {len(mc_questions)}")
    print(f"Questions rejected by guards: {len(questions) - len(mc_questions)}")

    # Analyze rejection reasons (would need to track in MCBuilder for full details)
    if len(mc_questions) < len(questions):
        print("\nNote: Some questions were rejected due to guard violations.")
        print("Common reasons include:")
        print("  - Not enough valid distractors after alias/duplicate filtering")
        print("  - Length ratio violations between options")
        print("  - Answer text appearing in question")

    print("\n✓ MC questions built successfully with guards active")
    return 0


if __name__ == "__main__":
    exit(main())
