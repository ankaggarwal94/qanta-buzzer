"""Tests for split resolution in scripts/train_t5_policy.py."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import scripts.train_t5_policy as train_t5_policy


def test_load_question_splits_prefers_sibling_persisted_splits(
    tmp_path,
    monkeypatch,
) -> None:
    """When sibling split artifacts exist, do not fall back to random split."""
    for filename in (
        "mc_dataset.json",
        "train_dataset.json",
        "val_dataset.json",
        "test_dataset.json",
    ):
        (tmp_path / filename).write_text("[]", encoding="utf-8")

    loaded_paths: list[str] = []

    def fake_load_mc_questions(path: str | Path):
        loaded_paths.append(Path(path).name)
        return [Path(path).stem]

    def fail_load_questions(*_args, **_kwargs):
        raise AssertionError("combined-dataset fallback should not run")

    monkeypatch.setattr(train_t5_policy, "load_mc_questions", fake_load_mc_questions)
    monkeypatch.setattr(train_t5_policy, "load_questions", fail_load_questions)

    args = Namespace(
        mc_path=str(tmp_path / "mc_dataset.json"),
        smoke=False,
    )

    train_questions, val_questions, test_questions = (
        train_t5_policy.load_question_splits(args, {"data": {}})
    )

    assert loaded_paths == [
        "train_dataset.json",
        "val_dataset.json",
        "test_dataset.json",
    ]
    assert train_questions == ["train_dataset"]
    assert val_questions == ["val_dataset"]
    assert test_questions == ["test_dataset"]


def test_load_question_splits_falls_back_to_random_split(monkeypatch) -> None:
    """Missing persisted splits should reuse the legacy combined-dataset path."""
    args = Namespace(
        mc_path="/tmp/nonexistent/mc_dataset.json",
        smoke=False,
    )

    monkeypatch.setattr(
        train_t5_policy,
        "load_questions",
        lambda _args, _config: ["combined"],
    )
    monkeypatch.setattr(
        train_t5_policy,
        "split_questions",
        lambda questions, _config: (
            questions + ["train"],
            ["val"],
            ["test"],
        ),
    )

    train_questions, val_questions, test_questions = (
        train_t5_policy.load_question_splits(args, {"data": {}})
    )

    assert train_questions == ["combined", "train"]
    assert val_questions == ["val"]
    assert test_questions == ["test"]
