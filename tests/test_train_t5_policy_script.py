"""Tests for split resolution in scripts/train_t5_policy.py."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

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


def test_load_question_splits_with_metadata_records_persisted_paths(
    tmp_path,
    monkeypatch,
) -> None:
    """Persisted split resolution should return a manifest with exact paths/qids."""
    for filename in ("train_dataset.json", "val_dataset.json", "test_dataset.json"):
        (tmp_path / filename).write_text("[]", encoding="utf-8")

    def fake_load_mc_questions(path: str | Path):
        stem = Path(path).stem
        return [type("Q", (), {"qid": f"{stem}_qid"})()]

    monkeypatch.setattr(train_t5_policy, "load_mc_questions", fake_load_mc_questions)

    args = Namespace(mc_path=str(tmp_path / "mc_dataset.json"), smoke=False)
    train_qs, val_qs, test_qs, manifest = (
        train_t5_policy.load_question_splits_with_metadata(args, {"data": {}})
    )

    assert manifest["source"] == "persisted_artifacts"
    assert manifest["train_path"].endswith("train_dataset.json")
    assert manifest["train_qids"] == ["train_dataset_qid"]
    assert manifest["test_qids"] == ["test_dataset_qid"]
    assert train_qs[0].qid == "train_dataset_qid"


def test_load_question_splits_falls_back_to_random_split(monkeypatch) -> None:
    """Missing persisted splits should reuse the legacy combined-dataset path."""
    args = Namespace(
        mc_path="/tmp/nonexistent/mc_dataset.json",
        smoke=False,
    )

    monkeypatch.setattr(
        train_t5_policy,
        "load_questions",
        lambda _args, _config, return_path=False: (
            (["combined"], Path("/tmp/nonexistent/mc_dataset.json"))
            if return_path
            else ["combined"]
        ),
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


def test_main_writes_t5_config_and_split_manifest(
    tmp_path, monkeypatch
) -> None:
    """train_t5_policy main should persist checkpoint-sidecar provenance."""
    import sys
    import training.train_ppo_t5 as ppo_t5_mod

    fake_question = type("Q", (), {"qid": "q1"})()
    fake_manifest = {
        "source": "persisted_artifacts",
        "mc_path": None,
        "train_path": "/tmp/train_dataset.json",
        "val_path": "/tmp/val_dataset.json",
        "test_path": "/tmp/test_dataset.json",
        "train_qids": ["q1"],
        "val_qids": ["q1"],
        "test_qids": ["q1"],
    }

    checkpoint_dir = tmp_path / "ppo_t5"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class FakeTrainer:
        def __init__(self, checkpoint_dir):
            self.checkpoint_dir = checkpoint_dir

    monkeypatch.setattr(
        train_t5_policy,
        "load_question_splits_with_metadata",
        lambda _args, _config: ([fake_question], [fake_question], [fake_question], fake_manifest),
    )
    monkeypatch.setattr(
        ppo_t5_mod,
        "run_ppo_training",
        lambda **_kwargs: (object(), FakeTrainer(checkpoint_dir)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_t5_policy.py",
            "--config",
            str(train_t5_policy.PROJECT_ROOT / "configs" / "t5_policy.yaml"),
            "--skip-supervised",
            "--model-path",
            str(tmp_path / "pretrained"),
        ],
    )

    train_t5_policy.main()

    assert (checkpoint_dir / "config_used.json").exists()
    assert (checkpoint_dir / "split_manifest.json").exists()


def test_ma017_skip_test_eval_flag_threads_none_to_ppo(
    tmp_path, monkeypatch
) -> None:
    """MA-017: ``--skip-test-eval`` parses (default False) and threads
    ``test_questions=None`` into ``run_ppo_training`` so the unconditional
    test-eval tail is skipped for discarded PPO phases."""
    import sys
    import training.train_ppo_t5 as ppo_t5_mod

    assert train_t5_policy.parse_args([]).skip_test_eval is False
    assert train_t5_policy.parse_args(["--skip-test-eval"]).skip_test_eval is True

    fake_question = type("Q", (), {"qid": "q1"})()
    fake_manifest = {
        "source": "persisted_artifacts",
        "mc_path": None,
        "train_path": "/tmp/train_dataset.json",
        "val_path": "/tmp/val_dataset.json",
        "test_path": "/tmp/test_dataset.json",
        "train_qids": ["q1"],
        "val_qids": ["q1"],
        "test_qids": ["q1"],
    }
    checkpoint_dir = tmp_path / "ppo_t5"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class FakeTrainer:
        def __init__(self, checkpoint_dir):
            self.checkpoint_dir = checkpoint_dir

    captured_kwargs: list[dict] = []

    def fake_run_ppo_training(**kwargs):
        captured_kwargs.append(dict(kwargs))
        return object(), FakeTrainer(checkpoint_dir)

    monkeypatch.setattr(
        train_t5_policy,
        "load_question_splits_with_metadata",
        lambda _args, _config: (
            [fake_question], [fake_question], [fake_question], fake_manifest
        ),
    )
    monkeypatch.setattr(ppo_t5_mod, "run_ppo_training", fake_run_ppo_training)

    base_argv = [
        "train_t5_policy.py",
        "--config",
        str(train_t5_policy.PROJECT_ROOT / "configs" / "t5_policy.yaml"),
        "--skip-supervised",
        "--model-path",
        str(tmp_path / "pretrained"),
    ]
    monkeypatch.setattr(sys, "argv", base_argv + ["--skip-test-eval"])
    train_t5_policy.main()
    assert captured_kwargs[-1]["test_questions"] is None, (
        "--skip-test-eval must thread test_questions=None into PPO"
    )

    # Control: without the flag the test questions pass through unchanged.
    monkeypatch.setattr(sys, "argv", list(base_argv))
    train_t5_policy.main()
    assert captured_kwargs[-1]["test_questions"] == [fake_question]


def test_apply_max_questions_global_raises_when_max_q_below_num_splits():
    """Global mode must refuse to silently empty held-out splits.

    Regression for the unresolved P2 review thread on ``_apply_max_questions``.
    """
    train = ["a", "b", "c"]
    val = ["d", "e"]
    test = ["f"]
    with pytest.raises(ValueError, match="non-empty splits"):
        train_t5_policy._apply_max_questions(
            train, val, test, max_q=2, scope="global"
        )


def test_apply_max_questions_per_split_legacy_still_truncates():
    """``per_split`` keeps the legacy independent-truncation semantics."""
    out = train_t5_policy._apply_max_questions(
        ["a", "b"], ["c", "d"], ["e", "f"], max_q=1, scope="per_split"
    )
    assert out == (["a"], ["c"], ["e"])


def test_apply_max_questions_global_distributes_proportionally():
    """When max_q can cover at least one item per non-empty split, the
    function distributes the rest proportionally."""
    out = train_t5_policy._apply_max_questions(
        ["a"] * 10, ["b"] * 4, ["c"] * 4, max_q=6, scope="global"
    )
    counts = tuple(len(x) for x in out)
    # Each split keeps at least one item; total respects cap; train (the
    # largest source at 10/18 ≈ 56%) keeps at least as many as val or test
    # (each 4/18 ≈ 22%) so the distribution is actually proportional, not
    # just a uniform 2/2/2 split that would also satisfy sum==6.
    assert all(c >= 1 for c in counts)
    assert sum(counts) == 6
    assert counts[0] >= counts[1] and counts[0] >= counts[2]


def test_load_question_splits_raises_when_global_cap_empties_held_out(
    tmp_path, monkeypatch
):
    """Even when no cap is applied, the loader should refuse to return
    a manifest with empty val or test under ``scope='global'`` and the
    error message must distinguish 'upstream empty' from 'cap emptied'."""
    for filename in (
        "mc_dataset.json",
        "train_dataset.json",
        "val_dataset.json",
        "test_dataset.json",
    ):
        (tmp_path / filename).write_text("[]", encoding="utf-8")

    def fake_load_mc_questions(path: str | Path):
        name = Path(path).name
        if name == "train_dataset.json":
            return ["a", "b", "c"]
        if name == "val_dataset.json":
            return []  # val empty after some upstream filter
        return ["e", "f"]

    monkeypatch.setattr(train_t5_policy, "load_mc_questions", fake_load_mc_questions)
    monkeypatch.setattr(
        train_t5_policy,
        "load_questions",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not fall through")),
    )

    args = Namespace(mc_path=str(tmp_path / "mc_dataset.json"), smoke=False)
    config = {"data": {"max_questions": 0, "max_questions_scope": "global"}}
    with pytest.raises(ValueError, match="empty upstream"):
        train_t5_policy.load_question_splits_with_metadata(args, config)


def test_load_question_splits_rejects_typo_in_max_questions_scope(
    tmp_path, monkeypatch
):
    """A typo in ``data.max_questions_scope`` (e.g. ``'globall'``) must
    raise immediately rather than silently falling through to
    ``_apply_max_questions``'s global branch and then bypassing the
    post-cap empty-split guard via strict-equality check."""
    args = Namespace(mc_path=None, smoke=False)
    config = {"data": {"max_questions": 5, "max_questions_scope": "globall"}}
    with pytest.raises(ValueError, match="max_questions_scope"):
        train_t5_policy.load_question_splits_with_metadata(args, config)
