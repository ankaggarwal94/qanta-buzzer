"""Packaging must preserve the source manifest's executable distinction."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from scripts.stopdff_v5 import fileio, writers
from scripts.stopdff_v5.content_manifest import validate_bound_content_manifest
from scripts.stopdff_v5.identity import build_manifest, sha256_file
from scripts.stopdff_v5.manifests import (
    ADAPTER_SCORING_SPEC,
    RAW_INPUT_ROLES,
    model_snapshot_identity,
    raw_input_identity,
    source_manifest_identity,
)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "runs" / "smoke"
    root.mkdir(parents=True)
    stage = tmp_path / "inputs"
    source = stage / "source"
    source.mkdir(parents=True)
    payloads = {"pyproject.toml": b"project\n", "uv.lock": b"lock\n", "bin/run.sh": b"#!/bin/sh\nexit 0\n"}
    files = []
    for name, data in payloads.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        mode = "100755" if name.endswith(".sh") else "100644"
        path.chmod(0o755 if mode == "100755" else 0o644)
        files.append({"path": name, "mode": mode, "sha256": sha256_file(path), "size": len(data)})
    hashes = {entry["path"]: entry["sha256"] for entry in files}
    manifests = {"source_manifest": build_manifest(source_manifest_identity(
        git_sha="a" * 40, files=files,
        pyproject_sha256=hashes["pyproject.toml"], uv_lock_sha256=hashes["uv.lock"],
    ))}
    raw = stage / "raw"
    raw.mkdir()
    raw_files = []
    for role in RAW_INPUT_ROLES:
        path = raw / role
        path.write_bytes(role.encode())
        raw_files.append({"role": role, "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifests["raw_input_manifest"] = build_manifest(raw_input_identity(files=raw_files, semantic_checks={"all_semantic_checks_pass": True, "question_trajectory_binding_id": "b" * 64}))
    model = stage / "snapshot"
    model.mkdir()
    (model / "model.bin").write_bytes(b"weights")
    manifests["model_snapshot_manifest"] = build_manifest(model_snapshot_identity(
        model_id=ADAPTER_SCORING_SPEC["model_id"], revision="c" * 40,
        files=[{"path": "model.bin", "size": 7, "sha256": sha256_file(model / "model.bin")}],
        sentence_transformers_version="fixture", transformers_version="fixture",
    ))
    manifests["fvi_study"] = build_manifest({"kind": "fvi_study_fixed"})
    manifests["environment_contract"] = build_manifest({"kind": "environment_contract"})
    artifacts, evidence = [], {}
    for role, manifest in manifests.items():
        path = stage / f"{role}.json"
        path.write_text(json.dumps(manifest))
        artifacts.append({"role": role, "content_id": manifest["id"], "sha256": sha256_file(path), "byte_size": path.stat().st_size, "retrieval_path": str(path)})
        if role in {"fvi_study", "environment_contract"}:
            evidence[f"evidence/{role}.json"] = path.read_bytes()
    spec = build_manifest({"profile_variant": "smoke", "evidence_roots": {"prerequisite_receipts": {}}})
    (root / "run_spec.json").write_text(json.dumps(spec))
    # Rendering and scientific inference have separate suites. These real
    # content manifests exercise the complete source packaging path.
    monkeypatch.setattr(writers, "write_figures", lambda *_args, **_kwargs: [])
    aggregate = {"profile_variant": "smoke", "cells": {}, "family": {}, "fvi_selected": {"tolerance": "1e-8", "max_iterations": 100}, "gate_overrides": {}}
    def package():
        writers.package_run(root, aggregate, resource_summary={"backend": "local"}, external_artifacts=artifacts, evidence_files=evidence)
    return root, manifests["source_manifest"], package


def _validate(root: Path, manifest: dict):
    return validate_bound_content_manifest(root / "evidence", manifest_name="source_manifest.json", expected_id=manifest["id"], expected_kind="source_snapshot", file_key="files", name_key="path", content_subdir="source_snapshot/source")


@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX executable-bit semantics")
def test_new_package_preserves_source_modes_and_cached_publication(tmp_path, monkeypatch):
    root, manifest, package = _fixture(tmp_path, monkeypatch)
    package()
    assert _validate(root, manifest) == manifest
    source = root / "evidence/source_snapshot/source"
    assert stat.S_IMODE((source / "bin/run.sh").stat().st_mode) & 0o111
    assert not stat.S_IMODE((source / "uv.lock").stat().st_mode) & 0o111
    before = {str(p): (p.stat().st_mtime_ns, p.stat().st_mode, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    package()
    assert before == {str(p): (p.stat().st_mtime_ns, p.stat().st_mode, p.read_bytes()) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["missing_exec", "extra_exec", "bytes", "symlink", "dangling_symlink"])
@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX executable-bit semantics")
def test_bad_cached_source_rejected_before_publishing_missing_candidates(tmp_path, monkeypatch, fault):
    root, _manifest, package = _fixture(tmp_path, monkeypatch)
    package()
    source = root / "evidence/source_snapshot/source"
    target = source / ("uv.lock" if fault == "extra_exec" else "bin/run.sh")
    if fault == "missing_exec":
        target.chmod(0o644)
    elif fault == "extra_exec":
        target.chmod(0o755)
    elif fault == "bytes":
        target.write_bytes(b"changed")
    else:
        target.unlink()
        target.symlink_to(source / ("uv.lock" if fault == "symlink" else "absent"))
    report = root / "reports/report.md"
    report.unlink()
    with pytest.raises(ValueError, match="package.*mismatch|noncanonical|symlink"):
        package()
    assert not report.exists()


@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX executable-bit semantics")
def test_create_once_file_mode_is_set_before_publication(tmp_path, monkeypatch):
    target = tmp_path / "script.sh"
    original_link = os.link
    observed = []
    def link(source, destination):
        observed.append(stat.S_IMODE(Path(source).stat().st_mode))
        assert not target.exists()
        return original_link(source, destination)
    monkeypatch.setattr(fileio.os, "link", link)
    fileio.create_once_bytes(target, b"data", file_mode=0o755)
    assert observed == [0o755]
    assert target.read_bytes() == b"data"
    monkeypatch.setattr(fileio.os, "link", original_link)
    with pytest.raises(FileExistsError):
        fileio.create_once_bytes(target, b"replacement", file_mode=0o644)
    assert target.read_bytes() == b"data"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


@pytest.mark.parametrize("mode", [True, -1, 0o4755, "755", [], 0o777])
def test_invalid_publication_modes_do_not_create_files(tmp_path, mode):
    target = tmp_path / "invalid"
    with pytest.raises(ValueError, match="file_mode"):
        fileio.create_once_bytes(target, b"data", file_mode=mode)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_unsupported_executable_mode_fails_before_publication(tmp_path, monkeypatch):
    target = tmp_path / "script.sh"
    monkeypatch.setattr(fileio.os, "fchmod", lambda *_args: None, raising=False)
    with pytest.raises(ValueError, match="cannot preserve source executable mode"):
        fileio.create_once_bytes(target, b"data", file_mode=0o755)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX executable-bit semantics")
def test_source_publication_race_does_not_overwrite_competing_file(tmp_path, monkeypatch):
    root, _manifest, package = _fixture(tmp_path, monkeypatch)
    target = root / "evidence/source_snapshot/source/bin/run.sh"
    publish = writers.create_once_bytes
    def competing_publish(path, data, **kwargs):
        if path == target:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"concurrent writer")
        return publish(path, data, **kwargs)
    monkeypatch.setattr(writers, "create_once_bytes", competing_publish)
    with pytest.raises(FileExistsError):
        package()
    assert target.read_bytes() == b"concurrent writer"
    assert not (root / "SHA256SUMS").exists()
