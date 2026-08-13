"""Direct coverage for ``verify_volume_artifact``'s sentinel contract.

``verify_volume_artifact`` (scripts/modal_stopdff_v5_runner.py) is the
volume-side gate the control plane consumes first.  Control-plane tests fake
it wholesale, so its real body is exercised here under the fake-modal seam
from ``tests.test_stopdff_v5_control_plane``: the exception -> ``{"ok": False}``
sentinel mapping, the unknown-kind and missing-``stopdff.json`` branches, and
the ``myopic_artifact_sha256`` identity-binding extraction for raw bundles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.stopdff_v5.identity import sha256_file
from scripts.stopdff_v5.manifests import RAW_INPUT_ROLES
from tests.test_stopdff_v5_control_plane import (
    _load_modal_runner,
    _write_raw_manifest,
)

RAW_ID = "b" * 64


def _staged_raw_dir(runner, tmp_path: Path) -> tuple[Path, str]:
    """Stage one fully valid raw-input bundle under the fake volume root."""
    runner.MNT = str(tmp_path)
    rel_dir = f"inputs/raw_{RAW_ID}"
    staged = tmp_path / "inputs" / f"raw_{RAW_ID}"
    staged.mkdir(parents=True)
    return staged, rel_dir


def test_verify_raw_artifact_success_binds_stopdff_sha(tmp_path, monkeypatch):
    """The happy path reports identity fields and plucks the right role sha."""
    runner = _load_modal_runner(monkeypatch)
    staged, rel_dir = _staged_raw_dir(runner, tmp_path)
    manifest = _write_raw_manifest(staged, passed=True)

    result = runner.verify_volume_artifact(rel_dir, "raw")

    # Recomputed from the staged bytes, independently of the manifest: a
    # wrong-role or wrong-key pluck would return a different digest because
    # every staged role file has distinct content.
    stopdff_sha = sha256_file(staged / "stopdff.json")
    assert result == {
        "ok": True,
        "id": manifest["id"],
        "mismatches": [],
        "n_files": len(RAW_INPUT_ROLES),
        "myopic_artifact_sha256": stopdff_sha,
    }
    stopdff_entry = next(
        entry
        for entry in manifest["identity"]["files"]
        if entry["role"] == "stopdff.json"
    )
    assert stopdff_sha == stopdff_entry["sha256"]


def test_verify_raw_artifact_maps_validation_failure_to_sentinel(
    tmp_path,
    monkeypatch,
):
    """A real byte-tamper surfaces as the soft sentinel, not an exception."""
    runner = _load_modal_runner(monkeypatch)
    staged, rel_dir = _staged_raw_dir(runner, tmp_path)
    _write_raw_manifest(staged, passed=True)
    (staged / "stopdff.json").write_bytes(b"tampered-after-manifest\n")

    result = runner.verify_volume_artifact(rel_dir, "raw")

    assert result["ok"] is False
    assert set(result) == {"ok", "error"}
    # The embedded error names the mismatched file for the control plane.
    assert "stopdff.json" in result["error"]


@pytest.mark.parametrize(
    "exc_class",
    [OSError, UnicodeError, TypeError, ValueError],
)
def test_verify_artifact_maps_each_declared_exception_to_sentinel(
    tmp_path,
    monkeypatch,
    exc_class,
):
    """Every exception class in the dispatcher's tuple maps to ``ok: False``."""
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)

    def raising(*_args, **_kwargs):
        raise exc_class("staged artifact rejected")

    monkeypatch.setattr(runner, "_verified_raw_input_manifest", raising)

    result = runner.verify_volume_artifact(f"inputs/raw_{RAW_ID}", "raw")

    assert result == {"ok": False, "error": "staged artifact rejected"}


def test_verify_artifact_rejects_unknown_kind_without_dispatch(
    tmp_path,
    monkeypatch,
):
    """An unknown kind is refused before any validator touches the volume."""
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)

    def unexpected(*_args, **_kwargs):
        pytest.fail("unknown kind must not dispatch a validator")

    monkeypatch.setattr(runner, "_verified_raw_input_manifest", unexpected)
    monkeypatch.setattr(runner, "_verified_content_manifest", unexpected)

    result = runner.verify_volume_artifact(f"inputs/raw_{RAW_ID}", "bogus")

    assert result == {"ok": False, "error": "unknown kind bogus"}


def test_verify_raw_artifact_requires_stopdff_role_entry(tmp_path, monkeypatch):
    """A raw manifest without a stopdff.json role fails closed.

    The real validator enforces the closed ``RAW_INPUT_ROLES`` set, so this
    defense-in-depth branch is reachable only if that invariant regresses --
    which is exactly what it exists to catch.
    """
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    manifest = {
        "id": "d" * 64,
        "identity": {
            "files": [
                {"role": "mc_dataset.json", "size": 1, "sha256": "e" * 64},
            ],
        },
    }
    monkeypatch.setattr(
        runner,
        "_verified_raw_input_manifest",
        lambda *_args, **_kwargs: manifest,
    )

    result = runner.verify_volume_artifact(f"inputs/raw_{RAW_ID}", "raw")

    assert result == {
        "ok": False,
        "error": "raw-input manifest lacks stopdff.json",
    }


@pytest.mark.parametrize(
    ("kind", "manifest_name", "content_subdir", "expected_kind"),
    [
        ("source", "source_manifest.json", "source", "source_snapshot"),
        ("model", "model_snapshot_manifest.json", "snapshot", "model_snapshot"),
    ],
)
def test_verify_artifact_dispatches_source_and_model_without_myopic_binding(
    tmp_path,
    monkeypatch,
    kind,
    manifest_name,
    content_subdir,
    expected_kind,
):
    """Non-raw kinds bind the right validator arguments and never emit the
    raw-only ``myopic_artifact_sha256`` identity key."""
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    manifest = {
        "id": "f" * 64,
        "identity": {
            "files": [{"path": "a", "size": 1, "sha256": "0" * 64}],
        },
    }
    calls: list[tuple[Path, dict]] = []

    def capture(base, **kwargs):
        calls.append((Path(base), kwargs))
        return manifest

    monkeypatch.setattr(runner, "_verified_content_manifest", capture)

    rel_dir = f"inputs/{kind}_{RAW_ID}"
    result = runner.verify_volume_artifact(rel_dir, kind)

    assert result == {
        "ok": True,
        "id": "f" * 64,
        "mismatches": [],
        "n_files": 1,
    }
    assert "myopic_artifact_sha256" not in result
    ((base, kwargs),) = calls
    assert base == tmp_path / "inputs" / f"{kind}_{RAW_ID}"
    assert kwargs["manifest_name"] == manifest_name
    assert kwargs["content_subdir"] == content_subdir
    assert kwargs["expected_kind"] == expected_kind
    assert kwargs["expected_id"] is None
