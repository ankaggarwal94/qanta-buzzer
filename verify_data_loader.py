#!/usr/bin/env python
"""
Verification script for data loader functionality.
"""

from qb_data.data_loader import QANTADatasetLoader
from qb_data.text_utils import normalize_answer


def main():
    """Test the data loader with the test CSV file."""
    print("=" * 60)
    print("Testing QANTADatasetLoader")
    print("=" * 60)

    # Load test questions
    loader = QANTADatasetLoader()
    questions = loader.load_from_csv('data/test_questions.csv')

    print(f"\nLoaded {len(questions)} questions from test CSV")
    print("-" * 60)

    # Display first few questions
    for i, q in enumerate(questions[:3], 1):
        print(f"\nQuestion {i}:")
        print(f"  QID: {q.qid}")
        print(f"  Category: {q.category}")
        print(f"  Answer: {q.answer_primary}")
        print(f"  Clean answers: {q.clean_answers}")
        print(f"  Number of tokens: {len(q.tokens)}")
        print(f"  Number of clues: {len(q.run_indices)}")
        print(f"  Run indices: {q.run_indices}")

        # Show cumulative prefixes (first 50 chars of each)
        print(f"  Cumulative prefixes:")
        for j, prefix in enumerate(q.cumulative_prefixes, 1):
            preview = prefix[:50] + "..." if len(prefix) > 50 else prefix
            print(f"    Clue {j}: {preview}")

    print("\n" + "=" * 60)
    print("Testing normalize_answer function")
    print("=" * 60)

    test_cases = [
        ("The Great Gatsby", "great gatsby"),
        ("A Tale of Two Cities", "tale of two cities"),
        ("An Example!!!", "example"),
        ("  Ludwig   van   Beethoven  ", "ludwig van beethoven"),
        ("", ""),
        ("The", ""),
    ]

    for input_text, expected in test_cases:
        result = normalize_answer(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} normalize_answer({input_text!r}) = {result!r} (expected: {expected!r})")

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()