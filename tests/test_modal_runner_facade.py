"""Coverage for the PR30 modal-runner structure cluster.

Covers the three-way module split (runner facade over
``scripts/stopdff_v5_control_plane.py`` and
``scripts/stopdff_v5_assurance_stages.py``), the gated
``STOPDFF_V5_APP_NAME`` override, the stage-then-rename cache publication
with staging reclaim, the fileio adoption of the control-byte publisher,
and the apt-free v5 image.
"""
from __future__ import annotations

import json
import os
import time
import types
from pathlib import Path

import pytest

from scripts import stopdff_v5_assurance_stages, stopdff_v5_control_plane
from scripts.stopdff_v5 import checker
from tests.harness_control_plane import (
    _load_modal_runner,
    _write_model_manifest,
)


def _backdate_past_reap_age(runner, path: Path) -> None:
    """Age one staging directory past the reclaim gate (a crash relic)."""
    stale = time.time() - runner._STAGING_REAP_AGE_S - 3600
    os.utime(path, (stale, stale))


# --- app-name override gate (adv#L-2) --------------------------------------


def test_app_name_override_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("STOPDFF_V5_APP_NAME", "cs321m-stopdff-v5-experiment")
    monkeypatch.delenv("STOPDFF_V5_ALLOW_APP_OVERRIDE", raising=False)

    with pytest.raises(RuntimeError, match="STOPDFF_V5_ALLOW_APP_OVERRIDE"):
        _load_modal_runner(monkeypatch)


def test_app_name_override_with_opt_in_warns_loudly(monkeypatch, capsys):
    app_name = "cs321m-stopdff-v5-assurance-deadbeef"
    monkeypatch.setenv("STOPDFF_V5_APP_NAME", app_name)
    monkeypatch.setenv("STOPDFF_V5_ALLOW_APP_OVERRIDE", "1")

    runner = _load_modal_runner(monkeypatch)

    assert runner.APP_NAME == app_name
    stderr = capsys.readouterr().err
    assert "WARNING" in stderr
    assert app_name in stderr
    assert "max_containers=1" in stderr


def test_default_app_name_loads_without_opt_in(monkeypatch, capsys):
    monkeypatch.delenv("STOPDFF_V5_APP_NAME", raising=False)
    monkeypatch.delenv("STOPDFF_V5_ALLOW_APP_OVERRIDE", raising=False)

    runner = _load_modal_runner(monkeypatch)

    assert runner.APP_NAME == "cs321m-stopdff-v5"
    assert "WARNING" not in capsys.readouterr().err


# --- image spec (learnings#L-1, torch floor) --------------------------------


def test_v5_image_installs_no_apt_packages(monkeypatch):
    runner = _load_modal_runner(monkeypatch)
    assert runner.modal.apt_installs == []


def test_v5_image_pins_torch_at_cve_floor(monkeypatch):
    runner = _load_modal_runner(monkeypatch)
    torch_pins = [spec for spec in runner._PIP if spec.startswith("torch")]
    assert torch_pins == ["torch>=2.6"]


# --- facade over the split modules (maint#H-1) -------------------------------


def test_runner_facade_reexports_moved_control_plane_names(monkeypatch):
    runner = _load_modal_runner(monkeypatch)
    for name in (
        "_record_control_event",
        "_run_control_stage",
        "_reconcile_control_journal",
        "_load_control_json",
        "_validate_control_plan",
        "_adapter_attempt_subdirs",
        "_canonical_adapter_subdir",
        "_retry_adapter_subdir",
        "_validate_promotion_result",
        "_validate_adapter_result",
        "_atomic_create_control_bytes",
        "_atomic_replace_control_bytes",
        "_close_interrupted_control_attempt",
    ):
        assert getattr(runner, name) is getattr(
            stopdff_v5_control_plane, name
        ), name
    # The driver facade binds runner-owned globals (image source id, source
    # dir, default stage api), so it is a wrapper, not the moved impl.
    assert callable(runner.run_control_plane)
    assert runner.run_control_plane is not stopdff_v5_control_plane.run_control_plane
    assert runner._assurance_stages is stopdff_v5_assurance_stages


def test_run_control_plane_wrapper_binds_patched_image_source(
    tmp_path, monkeypatch
):
    runner = _load_modal_runner(monkeypatch)
    assert runner.IMAGE_SOURCE_MANIFEST_ID == "1" * 64
    plan = {
        "source_id": "2" * 64,
        "raw_id": "3" * 64,
        "adapter_subdirs": ["build_a", "build_b"],
    }

    with pytest.raises(ValueError, match="control plan source_id"):
        runner.run_control_plane(
            plan,
            tmp_path / "control.json",
            resume=False,
            stage_api={},
        )


# --- fileio adoption (R5c) ---------------------------------------------------


def test_atomic_replace_control_bytes_delegates_to_fileio(
    tmp_path, monkeypatch
):
    calls: list[Path] = []
    real = stopdff_v5_control_plane.publish_bytes

    def recording(path, data):
        calls.append(Path(path))
        real(path, data)

    monkeypatch.setattr(stopdff_v5_control_plane, "publish_bytes", recording)
    target = tmp_path / "control.json"

    stopdff_v5_control_plane._atomic_replace_control_bytes(target, b"{}\n")

    assert calls == [target]
    assert target.read_bytes() == b"{}\n"


def test_atomic_replace_control_bytes_still_rejects_noncanonical_dest(
    tmp_path,
):
    target = tmp_path / "control.json"
    target.mkdir()

    with pytest.raises(ValueError, match="noncanonical"):
        stopdff_v5_control_plane._atomic_replace_control_bytes(target, b"{}\n")


# --- staging + reclaim cache publication (adv#M-2) ---------------------------


def test_freeze_model_crash_cannot_brick_the_singleton_slot(
    tmp_path, monkeypatch
):
    """A freeze interrupted mid-download leaves no partial live slot, a
    crash-persisted staging leftover is reclaimed, and the retry succeeds —
    the permanent FileExistsError brick from partial cache persistence is
    gone."""
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    root = tmp_path / "inputs" / "model"
    model_id = "3" * 64
    from scripts.stopdff_v5 import adapter_build

    def interrupted_download(destination):
        (Path(destination) / "snapshot").mkdir(parents=True)
        (Path(destination) / "snapshot" / "partial.bin").write_text(
            "partial bytes", encoding="utf-8"
        )
        raise RuntimeError("interrupted download")

    monkeypatch.setattr(
        adapter_build, "freeze_model_snapshot", interrupted_download
    )
    with pytest.raises(RuntimeError, match="interrupted download"):
        runner.freeze_model()

    # The live singleton slot never held partial content.
    assert not root.exists()

    # Simulate a stale crash-persisted staging leftover (background commit
    # case); reclaim is age-gated, so only a leftover past the reap age is
    # provably not a live peer's in-flight staging.
    leftover = tmp_path / "inputs" / ".staging_deadbeef"
    leftover.mkdir(parents=True)
    (leftover / "junk.bin").write_text("orphan", encoding="utf-8")
    _backdate_past_reap_age(runner, leftover)

    def freeze_valid_model(destination):
        manifest = _write_model_manifest(
            Path(destination),
            kind="model_snapshot",
            model_id=model_id,
        )
        manifest["identity"]["model_revision"] = "a" * 40
        (Path(destination) / "model_snapshot_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return manifest

    monkeypatch.setattr(
        adapter_build, "freeze_model_snapshot", freeze_valid_model
    )
    monkeypatch.setattr(
        runner,
        "_verified_content_manifest",
        lambda *_args, **_kwargs: {"id": model_id},
    )

    result = runner.freeze_model()

    assert result == {
        "model_id": model_id,
        "revision": "a" * 40,
        "cached": False,
    }
    assert (root / "model_snapshot_manifest.json").is_file()
    assert not leftover.exists()
    assert not list((tmp_path / "inputs").glob(".staging_*"))


def test_publish_helpers_fail_closed_and_reclaim_only_staging(
    tmp_path, monkeypatch
):
    runner = _load_modal_runner(monkeypatch)

    # Non-empty destination fails closed and is never overwritten.
    staging = tmp_path / ".staging_a"
    staging.mkdir()
    (staging / "new").write_text("new", encoding="utf-8")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        runner._publish_staged_dir(staging, occupied)
    assert (occupied / "old").read_text(encoding="utf-8") == "old"

    # Symlinked destination fails closed before any rename (the mkdir-claim
    # never follows the link into its target).
    link_dest = tmp_path / "linked"
    link_dest.symlink_to(occupied, target_is_directory=True)
    with pytest.raises(FileExistsError, match="already exists"):
        runner._publish_staged_dir(staging, link_dest)

    # A pre-existing EMPTY destination also fails closed (create-once): a bare
    # os.rename would silently replace it, so this is exactly the mutation the
    # os.mkdir claim closes.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        runner._publish_staged_dir(staging, empty)
    assert list(empty.iterdir()) == []  # never populated
    assert (staging / "new").read_text(encoding="utf-8") == "new"  # staging intact

    # Reclaim removes only stale .staging_* directories: never live entries,
    # never files, and never a young staging dir that could still belong to a
    # live peer container (the age gate).
    parent = tmp_path / "cache"
    (parent / ".staging_orphan").mkdir(parents=True)
    _backdate_past_reap_age(runner, parent / ".staging_orphan")
    (parent / ".staging_young").mkdir()
    (parent / "live_slot").mkdir()
    (parent / ".staging_file").write_text("file, not dir", encoding="utf-8")
    assert runner._reclaim_staging_dirs(parent) == 1
    assert not (parent / ".staging_orphan").exists()
    assert (parent / ".staging_young").is_dir()
    assert (parent / "live_slot").is_dir()
    assert (parent / ".staging_file").is_file()


def test_publish_staged_dir_fails_closed_on_pre_existing_empty_dest(
    tmp_path, monkeypatch
):
    """Create-once publish: an empty destination a racing peer created fails
    closed instead of being silently replaced -- the pre-fix os.rename hole
    (model/FVI/bootstrap/promote all publish through this helper). Asserts the
    raise, so it is red against the pre-fix os.rename code.
    """
    runner = _load_modal_runner(monkeypatch)
    staging = tmp_path / ".staging_x"
    staging.mkdir()
    (staging / "payload").write_text("staged", encoding="utf-8")
    empty_dest = tmp_path / "slot"
    empty_dest.mkdir()  # a racing peer's empty slot
    with pytest.raises(FileExistsError, match="already exists"):
        runner._publish_staged_dir(staging, empty_dest)
    assert list(empty_dest.iterdir()) == []  # never populated
    assert (staging / "payload").read_text(encoding="utf-8") == "staged"


def test_bootstrap_plan_publishes_by_rename_and_reclaims_staging(
    tmp_path, monkeypatch
):
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    adapter_id = "a" * 64
    (tmp_path / "adapters" / f"canonical_{adapter_id}").mkdir(parents=True)
    rows = [
        {"item_id": "q1", "split": "test", "format": "MC"},
        {"item_id": "q1", "split": "test", "format": "QA"},
        {"item_id": "q2", "split": "test", "format": "MC"},
        {"item_id": "q2", "split": "test", "format": "QA"},
    ]
    monkeypatch.setattr(
        checker,
        "validate_adapter",
        lambda _path: types.SimpleNamespace(
            passed=True,
            errors=[],
            recomputed={"adapter_bundle_id": adapter_id},
        ),
    )
    monkeypatch.setattr(checker, "load_adapter_rows", lambda _path: rows)
    leftover = tmp_path / "bootstrap" / ".staging_leftover"
    leftover.mkdir(parents=True)
    (leftover / "bootstrap_plan.json").write_text("partial", encoding="utf-8")
    _backdate_past_reap_age(runner, leftover)

    result = runner.bootstrap_plan(adapter_id, 3)

    assert result["cached"] is False
    out = tmp_path / "bootstrap" / result["bootstrap_plan_id"]
    assert (out / "bootstrap_plan.json").is_file()
    assert not leftover.exists()
    assert not list((tmp_path / "bootstrap").glob(".staging_*"))

    # The published slot is canonical: a repeat call is a pure cache hit.
    again = runner.bootstrap_plan(adapter_id, 3)
    assert again == {**result, "cached": True}
