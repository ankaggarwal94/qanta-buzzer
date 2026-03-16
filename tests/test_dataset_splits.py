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
