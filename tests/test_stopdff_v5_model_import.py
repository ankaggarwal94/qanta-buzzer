"""Regression tests for offline, identity-bound archived model imports."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import run_stopdff_v5_local as runner
from scripts.stopdff_v5.identity import build_manifest, sha256_bytes
from scripts.stopdff_v5.manifests import ADAPTER_SCORING_SPEC, model_snapshot_identity


def _model(root: Path, *, payload: bytes = b"model") -> dict:
    files = {"model.bin": payload, ".cache/huggingface/download/model.metadata": b"historic-cache"}
    for name, data in files.items():
        path = root / "snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = build_manifest(model_snapshot_identity(
        model_id=ADAPTER_SCORING_SPEC["model_id"], revision="a" * 40,
        files=[{"path": name, "size": len(data), "sha256": sha256_bytes(data)}
               for name, data in sorted(files.items())],
        sentence_transformers_version="historic", transformers_version="historic",
    ))
    manifest["snapshot_dir"] = "/original/producer/model/snapshot"
    (root / "model_snapshot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _argv(out: Path, source: Path, model_id: str | None = None) -> list[str]:
    result = ["--out-dir", str(out), "--model-snapshot-dir", str(source)]
    if model_id:
        result += ["--expected-model-snapshot-id", model_id]
    return result


@pytest.fixture
def offline(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("archived model import must not download or freeze a model")
    monkeypatch.setattr(runner.adapter_build, "freeze_model_snapshot", forbidden)


def test_import_preserves_manifest_and_cache_bytes_offline(tmp_path, offline):
    source, out = tmp_path / "archive", tmp_path / "out"
    manifest = _model(source)
    out.mkdir()
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert runner._stage_model_snapshot(
        out, model_snapshot_dir=source, expected_model_snapshot_id=manifest["id"],
    ) == manifest["id"]
    assert {p.relative_to(out / "model"): p.read_bytes()
            for p in (out / "model").rglob("*") if p.is_file()} == before
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("corruption", ["body", "extra", "manifest", "extra-root", "symlink", "directory-symlink", "path-traversal", "duplicate-key", "fifo"])
def test_invalid_import_rejected_before_output_or_heavy_work(tmp_path, monkeypatch, offline, corruption):
    source, out = tmp_path / "archive", tmp_path / "out"
    manifest = _model(source)
    if corruption == "body":
        (source / "snapshot/model.bin").write_bytes(b"wrong")
    elif corruption == "extra":
        (source / "snapshot/extra").write_text("extra")
    elif corruption == "manifest":
        manifest["id"] = "f" * 64
        (source / "model_snapshot_manifest.json").write_text(json.dumps(manifest))
    elif corruption == "extra-root":
        (source / "unlisted-root-file").write_text("extra")
    elif corruption == "symlink":
        (source / "snapshot/model.bin").unlink()
        (source / "snapshot/model.bin").symlink_to(tmp_path / "absent")
    elif corruption == "directory-symlink":
        cache = source / "snapshot/.cache"
        cache.rename(tmp_path / "cache")
        cache.symlink_to(tmp_path / "cache", target_is_directory=True)
    elif corruption == "path-traversal":
        manifest["identity"]["files"][0]["path"] = "../escape"
        manifest = build_manifest(manifest["identity"])
        (source / "model_snapshot_manifest.json").write_text(json.dumps(manifest))
    elif corruption == "duplicate-key":
        (source / "model_snapshot_manifest.json").write_text('{"id":"x","id":"y"}')
    else:
        import os
        os.mkfifo(source / "snapshot/pipe")
    monkeypatch.setattr(runner, "_preflight_clean_worktree", lambda _args: (out, "a" * 40))
    monkeypatch.setattr(runner, "_stage_source_snapshot", lambda **_kwargs: pytest.fail("heavy work before import validation"))
    with pytest.raises(ValueError):
        runner.main(_argv(out, source))
    assert not out.exists()


@pytest.mark.parametrize("layout", ["source-in-out", "out-in-source", "same", "symlink-ancestor"])
def test_noncanonical_or_overlapping_import_paths_rejected(tmp_path, monkeypatch, layout):
    source, out = tmp_path / "archive", tmp_path / "out"
    if layout == "source-in-out":
        source = out / "archive"
    elif layout == "out-in-source":
        out = source / "out"
    elif layout == "same":
        out = source
    _model(source)
    if layout == "symlink-ancestor":
        link = tmp_path / "alias"
        link.symlink_to(tmp_path, target_is_directory=True)
        source = link / "archive"
    monkeypatch.setattr(runner, "_preflight_clean_worktree", lambda _args: (out, "a" * 40))
    with pytest.raises(ValueError, match="overlap|symlink|canonical"):
        runner.main(_argv(out, source))


def test_expected_id_mismatch_is_rejected_before_output(tmp_path, monkeypatch):
    source, out = tmp_path / "archive", tmp_path / "out"
    _model(source)
    monkeypatch.setattr(runner, "_preflight_clean_worktree", lambda _args: (out, "a" * 40))
    with pytest.raises(ValueError, match="expected|match"):
        runner.main(_argv(out, source, "f" * 64))
    assert not out.exists()


def test_expected_id_requires_external_snapshot():
    with pytest.raises(SystemExit) as exc:
        runner._parse_args(["--out-dir", "out", "--expected-model-snapshot-id", "a" * 64])
    assert exc.value.code == 2


def test_import_starts_fresh_and_resume_binds_identity_not_external_path(tmp_path, monkeypatch, offline):
    source, out = tmp_path / "archive", tmp_path / "out"
    manifest = _model(source)
    monkeypatch.setattr(runner, "_preflight_clean_worktree", lambda _args: (out, "a" * 40))
    class StopBeforeHeavyWork(Exception):
        pass
    def stop(**_kwargs):
        assert (out / "model/snapshot/model.bin").read_bytes() == b"model"
        raise StopBeforeHeavyWork
    monkeypatch.setattr(runner, "_stage_source_snapshot", stop)
    with pytest.raises(StopBeforeHeavyWork):
        runner.main(_argv(out, source, manifest["id"]))
    state = json.loads((out / "local_lifecycle.json").read_text())
    assert state["imported_model_snapshot_id"] == manifest["id"]
    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    with pytest.raises(StopBeforeHeavyWork):
        runner.main(_argv(out, relocated, manifest["id"]) + ["--resume"])
    # Reusing the output without resume is still forbidden.
    with pytest.raises(FileExistsError):
        runner.main(_argv(out, source, manifest["id"]))
    # Omitted or replaced imported identity cannot silently select a new model.
    with pytest.raises(ValueError, match="lifecycle checkpoint does not match"):
        runner.main(["--out-dir", str(out), "--resume"])
    other = tmp_path / "different"
    other_manifest = _model(other, payload=b"other")
    with pytest.raises(ValueError, match="lifecycle checkpoint does not match"):
        runner.main(_argv(out, other, other_manifest["id"]) + ["--resume"])


def test_preseeded_directory_is_not_a_resume_checkpoint(tmp_path, monkeypatch):
    source, out = tmp_path / "archive", tmp_path / "out"
    manifest = _model(source)
    shutil.copytree(source, out / "model")
    monkeypatch.setattr(runner, "_preflight_clean_worktree", lambda _args: (out, "a" * 40))
    with pytest.raises(ValueError, match="requires local_lifecycle.json"):
        runner.main(_argv(out, source, manifest["id"]) + ["--resume"])


def test_copy_race_revalidates_before_publishing(tmp_path, monkeypatch):
    source, out = tmp_path / "archive", tmp_path / "out"
    manifest = _model(source)
    out.mkdir()
    original_copytree = shutil.copytree
    def corrupting_copytree(src, dst, *args, **kwargs):
        result = original_copytree(src, dst, *args, **kwargs)
        if Path(src) == source:
            (Path(dst) / "snapshot/model.bin").write_bytes(b"wrong")
        return result
    monkeypatch.setattr(shutil, "copytree", corrupting_copytree)
    with pytest.raises(ValueError, match="mismatch"):
        runner._stage_model_snapshot(out, model_snapshot_dir=source,
                                     expected_model_snapshot_id=manifest["id"])
    assert not (out / "model").exists()
    assert not list(out.glob(".model-*"))


def test_existing_staged_model_must_match_requested_import(tmp_path, offline):
    source, out = tmp_path / "archive", tmp_path / "out"
    manifest = _model(source)
    _model(out / "model", payload=b"different")
    with pytest.raises(ValueError, match="expected|match"):
        runner._stage_model_snapshot(out, model_snapshot_dir=source,
                                     expected_model_snapshot_id=manifest["id"])


def test_legacy_lifecycle_without_model_import_remains_compatible(tmp_path):
    args = runner._parse_args(["--out-dir", str(tmp_path)])
    original = runner._load_or_create_lifecycle(out=tmp_path, args=args, run_sha="a" * 40, resume=False)
    original.pop("imported_model_snapshot_id", None)
    (tmp_path / "local_lifecycle.json").write_text(json.dumps(original))
    assert runner._load_or_create_lifecycle(out=tmp_path, args=args, run_sha="a" * 40, resume=True) == original


def test_sweep_resume_also_checks_imported_model_binding(tmp_path, monkeypatch):
    source, out = tmp_path / "archive", tmp_path / "out"
    manifest = _model(source)
    out.mkdir()
    args = runner._parse_args(_argv(out, source, manifest["id"]) + ["--resume"])
    args.imported_model_snapshot_id = manifest["id"]
    staged = {
        "source_snapshot": {"identity": {"git_sha": "a" * 40}},
        "raw_inputs": {"identity": {}},
        "model": {"id": "f" * 64},
    }
    monkeypatch.setattr(runner, "_load_bound_content_manifest", lambda path, **_kwargs: staged[path.name])
    monkeypatch.setattr(runner, "_verified_local_source_execution", lambda *_args: None)
    with pytest.raises(ValueError, match="resume model snapshot does not match"):
        runner._resume_local_run(args=args, out=out, run_sha="a" * 40)


def test_import_checkpoint_is_checked_before_sweep_resume_dispatch(tmp_path, monkeypatch):
    source, out = tmp_path / "archive", tmp_path / "out"
    manifest = _model(source)
    (out / "runs/smoke_local_example").mkdir(parents=True)
    args = runner._parse_args(_argv(out, source, manifest["id"]))
    args.imported_model_snapshot_id = manifest["id"]
    runner._load_or_create_lifecycle(out=out, args=args, run_sha="a" * 40, resume=False)
    monkeypatch.setattr(runner, "_preflight_clean_worktree", lambda _args: (out, "a" * 40))
    dispatched = []
    monkeypatch.setattr(runner, "_resume_local_run", lambda **kwargs: dispatched.append(kwargs) or 0)
    assert runner.main(_argv(out, source, manifest["id"]) + ["--resume"]) == 0
    assert dispatched[0]["args"].imported_model_snapshot_id == manifest["id"]
    with pytest.raises(ValueError, match="lifecycle checkpoint does not match"):
        runner.main(["--out-dir", str(out), "--resume"])
    assert len(dispatched) == 1
