"""Tests for stratified dataset splitting reproducibility.

Verifies that splits are deterministic across invocations and do not
depend on Python's hash randomization (PYTHONHASHSEED).
"""

import subprocess
import sys

import pytest

from qb_data.data_loader import TossupQuestion
from qb_data.dataset_splits import create_stratified_splits


def _make_questions(n: int, categories: list[str]) -> list[TossupQuestion]:
    """Create n dummy TossupQuestion instances cycling through categories."""
    questions = []
    for i in range(n):
        cat = categories[i % len(categories)]
        questions.append(
            TossupQuestion(
                qid=f"q{i:04d}",
                question=f"Question {i}",
                tokens=[f"token{i}"],
                answer_primary=f"Answer {i}",
                clean_answers=[f"Answer {i}"],
                run_indices=[0],
                human_buzz_positions=[],
                category=cat,
                cumulative_prefixes=[f"token{i}"],
            )
        )
    return questions


def _question(
    qid: str,
    text: str,
    answer: str,
    category: str = "History",
) -> TossupQuestion:
    """Build a minimally valid question for group-integrity tests."""
    tokens = text.split()
    return TossupQuestion(
        qid=qid,
        question=text,
        tokens=tokens,
        answer_primary=answer,
        clean_answers=[answer],
        run_indices=[max(0, len(tokens) - 1)],
        human_buzz_positions=[],
        category=category,
        cumulative_prefixes=[text],
    )


def test_splits_deterministic_same_process():
    """Same seed produces identical splits within one process."""
    questions = _make_questions(60, ["History", "Science", "Literature"])
    train1, val1, test1 = create_stratified_splits(questions, seed=42)
    train2, val2, test2 = create_stratified_splits(questions, seed=42)
    assert [q.qid for q in train1] == [q.qid for q in train2]
    assert [q.qid for q in val1] == [q.qid for q in val2]
    assert [q.qid for q in test1] == [q.qid for q in test2]


def test_splits_deterministic_across_processes():
    """Splits must be identical even with different PYTHONHASHSEED values.

    Runs the split in two subprocesses with different PYTHONHASHSEED and
    checks that they produce identical qid orderings.
    """
    script = (
        "import json, sys, io; sys.path.insert(0, '.'); "
        "sys.stdout = io.StringIO(); "
        "from qb_data.data_loader import TossupQuestion; "
        "from qb_data.dataset_splits import create_stratified_splits; "
        "qs = [TossupQuestion(qid=f'q{i:04d}', question=f'Q{i}', tokens=[f't{i}'], "
        "answer_primary=f'A{i}', clean_answers=[f'A{i}'], run_indices=[0], "
        "human_buzz_positions=[], category=['History','Science','Lit'][i%3], "
        "cumulative_prefixes=[f't{i}']) for i in range(60)]; "
        "tr,va,te = create_stratified_splits(qs, seed=42); "
        "sys.stdout = sys.__stdout__; "
        "print(json.dumps([q.qid for q in tr]))"
    )
    import json
    import os

    base_env = {k: v for k, v in os.environ.items()}
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    results = []
    for hashseed in ["0", "12345"]:
        env = {**base_env, "PYTHONHASHSEED": hashseed}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root,
            timeout=30,
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        results.append(json.loads(proc.stdout.strip()))
    assert results[0] == results[1], (
        "Splits differ across PYTHONHASHSEED values — hash(category) is not deterministic"
    )


def test_splits_different_seeds_differ():
    """Different seeds should produce different splits."""
    questions = _make_questions(60, ["History", "Science", "Literature"])
    train1, _, _ = create_stratified_splits(questions, seed=42)
    train2, _, _ = create_stratified_splits(questions, seed=99)
    assert [q.qid for q in train1] != [q.qid for q in train2]


def test_splits_all_questions_assigned():
    """Every question must appear in exactly one split."""
    questions = _make_questions(100, ["A", "B", "C", "D"])
    train, val, test = create_stratified_splits(questions, seed=1)
    all_qids = {q.qid for q in train} | {q.qid for q in val} | {q.qid for q in test}
    assert len(all_qids) == 100
    assert len(train) + len(val) + len(test) == 100


def test_normalized_duplicate_questions_are_grouped_atomically():
    """Unicode/case/whitespace variants of one question cannot cross splits."""
    questions = [
        _question("q0", "Unique question zero?", "Zero"),
        _question("q1", "Unique question one?", "One"),
        _question("q2", "café history?", "Paris"),
        _question("q3", "Ｃａｆé History?", "Paris"),
        _question("q4", "  Café   History?  ", "Paris"),
        _question("q5", "Unique question five?", "Five"),
    ]

    train, val, test = create_stratified_splits(
        questions,
        ratios=[0.5, 0.25, 0.25],
        seed=789685,
    )
    membership = {}
    for split_name, split_questions in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        for question in split_questions:
            membership[question.qid] = split_name

    assert len({membership[qid] for qid in ("q2", "q3", "q4")}) == 1


def test_normalized_question_group_with_conflicting_answers_fails_closed():
    """A duplicated question with contradictory answers is not split arbitrarily."""
    questions = [
        _question("q0", "  Who wrote Hamlet? ", "William Shakespeare"),
        _question("q1", "ｗｈｏ wrote HAMLET?", "Christopher Marlowe"),
        _question("q2", "A unique question?", "Unique"),
    ]

    with pytest.raises(ValueError, match="conflicting normalized answers"):
        create_stratified_splits(questions, seed=42)


def test_duplicate_question_id_fails_closed_even_when_text_matches():
    questions = [
        _question("same", "Who wrote Hamlet?", "William Shakespeare"),
        _question("same", "Who wrote Hamlet?", "William Shakespeare"),
    ]

    with pytest.raises(ValueError, match="duplicate question ID"):
        create_stratified_splits(questions, seed=42)


def test_empty_normalized_answer_fails_closed():
    questions = [
        _question("q0", "A question with no usable answer?", " !!! "),
        _question("q1", "A unique question?", "Unique"),
    ]

    with pytest.raises(ValueError, match="empty normalized answer"):
        create_stratified_splits(questions, seed=42)


def test_answer_compatibility_variants_are_not_false_conflicts():
    questions = [
        _question("q0", "Where is the Eiffel Tower?", "ＰＡＲＩＳ"),
        _question("q1", "WHERE is the Eiffel Tower?", "Paris"),
        _question("q2", "A unique question?", "Unique"),
    ]

    train, val, test = create_stratified_splits(questions, seed=42)
    membership = {
        question.qid: split
        for split, split_questions in (
            ("train", train),
            ("val", val),
            ("test", test),
        )
        for question in split_questions
    }
    assert membership["q0"] == membership["q1"]


def test_grouped_split_is_input_order_invariant():
    questions = [
        _question("q0", "Shared text?", "Same", "History"),
        _question("q1", "ＳＨＡＲＥＤ   TEXT?", "Same", "Science"),
        _question("q2", "Unique two?", "Two", "History"),
        _question("q3", "Unique three?", "Three", "Science"),
        _question("q4", "Unique four?", "Four", "Literature"),
        _question("q5", "Unique five?", "Five", "Literature"),
    ]

    first = create_stratified_splits(questions, seed=7)
    second = create_stratified_splits(list(reversed(questions)), seed=7)
    assert [
        [question.qid for question in split]
        for split in first
    ] == [
        [question.qid for question in split]
        for split in second
    ]


@pytest.mark.parametrize(
    "ratios",
    [
        [],
        [1.0],
        [0.5, 0.5],
        [0.25, 0.25, 0.25, 0.25],
        [float("nan"), 0.5, 0.5],
        [float("inf"), 0.0, 0.0],
        [float("-inf"), 1.0, 1.0],
        [-0.1, 0.5, 0.6],
        [0.0, 0.0, 0.0],
        [True, 0.5, 0.5],
        ["0.5", 0.25, 0.25],
        [10**1000, 1, 1],
        [1e308, 1e308, 1.0],
        [2.0, 1.0, 1.0],
    ],
)
def test_split_ratios_fail_closed_unless_three_finite_nonnegative_weights(ratios):
    questions = _make_questions(6, ["History", "Science"])

    with pytest.raises(ValueError, match="ratios"):
        create_stratified_splits(questions, ratios=ratios, seed=42)
