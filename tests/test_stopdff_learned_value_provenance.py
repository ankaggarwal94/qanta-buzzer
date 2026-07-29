"""Provenance regressions for learned-value training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch


def _training_provenance(**overrides: object) -> dict[str, object]:
    provenance: dict[str, object] = {
        "script_path": "scripts/train_stopdff_value_model.py",
        "script_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "commit_script_sha256": "a" * 64,
        "commit_contains_exact_script": True,
    }
    provenance.update(overrides)
    return provenance


def _checkpoint(
    path: Path,
    *,
    producer_provenance: dict[str, object] | None = None,
    git_sha: str = "b" * 40,
    git_dirty: bool = False,
) -> None:
    from models.stopdff_value_model import StopDFFValueModel

    model = StopDFFValueModel(input_dim=1, hidden_sizes=(2,), dropout=0.0)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.to_config(),
            "feature_schema": {
                "feature_names": ["prefix_idx"],
                "numeric_columns": ["prefix_idx"],
                "categorical_columns": [],
                "categorical_levels": {},
            },
            "scaler": {
                "feature_names": ["prefix_idx"],
                "mean": [0.0],
                "std": [1.0],
            },
            "reward_schedule": "power_mark",
            "seeds": [1],
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "producer_provenance": (
                _training_provenance()
                if producer_provenance is None
                else producer_provenance
            ),
        },
        path,
    )


def test_ensemble_metadata_binds_exact_checkpoint_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import compute_stopdff_learned_value as learned
    from scripts._common import sha256_file

    monkeypatch.setattr(
        learned,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )
    monkeypatch.setattr(learned, "_current_git_commit", lambda: "b" * 40)
    checkpoint = tmp_path / "best.ckpt"
    _checkpoint(checkpoint)

    _models, _schema, _scaler, metadata = learned._load_ensemble(
        checkpoint,
        requested_device="cpu",
        reward_override=None,
    )

    assert metadata["checkpoint_sha256s"] == {
        str(checkpoint): sha256_file(checkpoint)
    }
    assert metadata["checkpoint_producer_commits"] == ["b" * 40]


@pytest.mark.parametrize(
    ("producer", "message"),
    [
        (_training_provenance(git_dirty=True), "dirty training producer"),
        (
            _training_provenance(
                script_path="scripts/compute_stopdff_learned_value.py"
            ),
            "canonical training producer path",
        ),
        (_training_provenance(git_commit="c" * 40), "different source commit"),
        (
            _training_provenance(commit_contains_exact_script=False),
            "does not bind its exact training producer",
        ),
    ],
)
def test_ensemble_rejects_dirty_or_unbound_training_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    producer: dict[str, object],
    message: str,
) -> None:
    from scripts import compute_stopdff_learned_value as learned

    monkeypatch.setattr(
        learned,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )
    monkeypatch.setattr(learned, "_current_git_commit", lambda: "b" * 40)
    checkpoint = tmp_path / "best.ckpt"
    _checkpoint(checkpoint, producer_provenance=producer)

    with pytest.raises(ValueError, match=message):
        learned._load_ensemble(
            checkpoint,
            requested_device="cpu",
            reward_override=None,
        )


def test_ensemble_rejects_dirty_or_mismatched_checkpoint_save_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import compute_stopdff_learned_value as learned

    monkeypatch.setattr(
        learned,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )
    monkeypatch.setattr(learned, "_current_git_commit", lambda: "b" * 40)

    dirty = tmp_path / "dirty" / "best.ckpt"
    dirty.parent.mkdir()
    _checkpoint(dirty, git_dirty=True)
    with pytest.raises(ValueError, match="dirty checkpoint save state"):
        learned._load_ensemble(
            dirty,
            requested_device="cpu",
            reward_override=None,
        )

    moved = tmp_path / "moved" / "best.ckpt"
    moved.parent.mkdir()
    _checkpoint(moved, git_sha="c" * 40)
    with pytest.raises(ValueError, match="checkpoint save commit"):
        learned._load_ensemble(
            moved,
            requested_device="cpu",
            reward_override=None,
        )


def test_learned_value_generation_fails_closed_on_provenance_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import compute_stopdff_learned_value as learned
    from scripts import _common

    monkeypatch.setattr(
        _common,
        "build_generation_provenance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    args = argparse.Namespace(data_dir=str(tmp_path), eval_split="test")

    with pytest.raises(RuntimeError, match="boom"):
        learned._generation_provenance(
            args=args,
            out_json=tmp_path / "out.json",
            checkpoint_metadata={
                "checkpoint_paths": [],
                "checkpoint_sha256s": {},
            },
            calibration_path=None,
        )


def test_learned_value_generation_rejects_dirty_parent_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import compute_stopdff_learned_value as learned
    from scripts import _common

    monkeypatch.setattr(
        _common,
        "build_generation_provenance",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "script_path": "scripts/compute_stopdff_learned_value.py",
            "script_sha256": "a" * 64,
            "git_commit": "b" * 40,
            "git_dirty": True,
        },
    )
    monkeypatch.setattr(
        learned,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )
    args = argparse.Namespace(data_dir=str(tmp_path), eval_split="test")

    with pytest.raises(RuntimeError, match="uncommitted producer"):
        learned._generation_provenance(
            args=args,
            out_json=tmp_path / "out.json",
            checkpoint_metadata={
                "checkpoint_paths": [],
                "checkpoint_sha256s": {},
            },
            calibration_path=None,
        )


def test_learned_value_generation_rejects_redirected_producer_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import compute_stopdff_learned_value as learned

    generation = {
        "schema_version": 1,
        "script_path": "scripts/_common.py",
        "script_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
    }
    monkeypatch.setattr(
        learned,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )

    with pytest.raises(RuntimeError, match="canonical producer script path"):
        learned._require_exact_producer_binding(generation)


def test_learned_value_generation_hashes_data_and_calibration_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import _common
    from scripts import compute_stopdff_learned_value as learned

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_paths = [
        data_dir / "mc_dataset.json",
        data_dir / "test_dataset.json",
        data_dir / "build_metadata.json",
    ]
    for index, path in enumerate(data_paths):
        path.write_text(f'{{"input": {index}}}\n', encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"calibration": 1}\n', encoding="utf-8")

    def generation():
        return {
            "schema_version": 1,
            "script_path": "scripts/compute_stopdff_learned_value.py",
            "script_sha256": "a" * 64,
            "git_commit": "b" * 40,
            "git_dirty": False,
        }

    monkeypatch.setattr(
        _common,
        "build_generation_provenance",
        lambda *_args, **_kwargs: generation(),
    )
    monkeypatch.setattr(
        learned,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )
    args = argparse.Namespace(data_dir=str(data_dir), eval_split="test")
    checkpoint_metadata = {
        "checkpoint_paths": [],
        "checkpoint_sha256s": {},
    }

    original = learned._generation_provenance(
        args=args,
        out_json=tmp_path / "out.json",
        checkpoint_metadata=checkpoint_metadata,
        calibration_path=calibration,
    )
    data_paths[0].write_text('{"input": "mutated"}\n', encoding="utf-8")
    mutated = learned._generation_provenance(
        args=args,
        out_json=tmp_path / "out.json",
        checkpoint_metadata=checkpoint_metadata,
        calibration_path=calibration,
    )

    key = str(data_paths[0])
    assert original["input_sha256s"][key] != mutated["input_sha256s"][key]
    assert original["input_sha256s"][str(calibration)] == mutated["input_sha256s"][
        str(calibration)
    ]


def test_training_checkpoint_provenance_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import train_stopdff_value_model as trainer

    monkeypatch.setattr(trainer, "_git_sha", lambda: "b" * 40)
    monkeypatch.setattr(trainer, "_git_dirty", lambda: False)
    monkeypatch.setattr(
        trainer,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )
    monkeypatch.setattr(trainer, "sha256_file", lambda _path: "a" * 64)
    provenance = trainer._training_provenance()
    assert provenance["commit_contains_exact_script"] is True
    assert trainer._checkpoint_save_identity(provenance) == "b" * 40

    monkeypatch.setattr(trainer, "_git_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="dirty training producer"):
        trainer._training_provenance()
    with pytest.raises(RuntimeError, match="dirty checkpoint save state"):
        trainer._checkpoint_save_identity(provenance)


def test_trainer_uses_host_injected_git_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import train_stopdff_value_model as trainer

    commit = "c" * 40
    monkeypatch.setenv("MODAL_HOST_GIT_COMMIT", commit)
    monkeypatch.setenv("MODAL_HOST_GIT_STATUS", "")
    assert trainer._git_sha() == commit
    assert trainer._git_dirty() is False

    monkeypatch.setenv("MODAL_HOST_GIT_STATUS", " M scripts/train_stopdff_value_model.py")
    assert trainer._git_dirty() is True
