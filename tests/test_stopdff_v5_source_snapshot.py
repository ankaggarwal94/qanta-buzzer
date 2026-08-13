"""Producer-level regressions for the content-addressed source snapshot."""

from __future__ import annotations

import json
import stat
import subprocess
import warnings
from pathlib import Path

import pytest

from scripts.stopdff_v5.identity import compute_id, sha256_bytes
from scripts.stopdff_v5.producers import build_source_snapshot


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _initialize_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Source Snapshot Test")
    _git(repo, "config", "user.email", "snapshot-test@example.invalid")


def test_build_source_snapshot_uses_committed_archive_bytes_and_inventory(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    committed_bytes = {
        "pyproject.toml": b"[project]\nname = \"snapshot-fixture\"\n",
        "uv.lock": b"version = 1\nrevision = 3\n",
        "bin/run.sh": b"#!/bin/sh\nprintf 'from-commit\\n'\n",
        "nested/payload.bin": b"\x00snapshot\xffpayload\n",
    }
    for relative_path, data in committed_bytes.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (repo / "bin" / "run.sh").chmod(0o755)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "snapshot fixture")
    run_sha = _git(repo, "rev-parse", "HEAD")

    # The producer must read RUN_SHA through git archive, not the dirty tree.
    (repo / "pyproject.toml").write_bytes(b"dirty worktree bytes\n")
    (repo / "untracked.txt").write_bytes(b"must not be archived\n")

    output = tmp_path / "snapshot"
    manifest = build_source_snapshot(repo, run_sha, output)
    source = output / "source"

    assert not (output / "source.tar").exists()
    assert not (source / "untracked.txt").exists()
    for relative_path, expected_bytes in committed_bytes.items():
        assert (source / relative_path).read_bytes() == expected_bytes

    inventory = manifest["identity"]["files"]
    assert [entry["path"] for entry in inventory] == sorted(committed_bytes)
    assert {
        entry["path"]: (entry["mode"], entry["size"], entry["sha256"])
        for entry in inventory
    } == {
        relative_path: (
            "100755" if relative_path == "bin/run.sh" else "100644",
            len(data),
            sha256_bytes(data),
        )
        for relative_path, data in committed_bytes.items()
    }
    assert stat.S_IMODE((source / "bin" / "run.sh").stat().st_mode) & 0o111
    assert not (
        stat.S_IMODE((source / "pyproject.toml").stat().st_mode) & 0o111
    )

    identity = manifest["identity"]
    assert identity["kind"] == "source_snapshot"
    assert identity["git_sha"] == run_sha
    assert identity["pyproject_toml_sha256"] == sha256_bytes(
        committed_bytes["pyproject.toml"]
    )
    assert identity["uv_lock_sha256"] == sha256_bytes(
        committed_bytes["uv.lock"]
    )
    assert manifest["id"] == compute_id(identity)
    assert manifest["file_count"] == len(committed_bytes)
    assert json.loads(
        (output / "source_manifest.json").read_text(encoding="utf-8")
    ) == manifest
    assert json.loads(
        (output / "source_build_record.json").read_text(encoding="utf-8")
    ) == {
        "git_sha": run_sha,
        "repo_dir": str(repo),
        "file_count": len(committed_bytes),
    }


def test_build_source_snapshot_extraction_pins_tar_filter_semantics(
    tmp_path: Path,
) -> None:
    """Extraction passes filter='data' explicitly: no tarfile deprecation on
    Python 3.12+, and bytes plus the git-tracked exec bit stay unchanged."""
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    (repo / "tool.sh").write_bytes(b"#!/bin/sh\nexit 0\n")
    (repo / "tool.sh").chmod(0o755)
    (repo / "data.txt").write_bytes(b"plain\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "filter fixture")
    run_sha = _git(repo, "rev-parse", "HEAD")

    output = tmp_path / "snapshot"
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        manifest = build_source_snapshot(repo, run_sha, output)

    source = output / "source"
    assert (source / "data.txt").read_bytes() == b"plain\n"
    assert stat.S_IMODE((source / "tool.sh").stat().st_mode) & 0o100
    assert not (stat.S_IMODE((source / "data.txt").stat().st_mode) & 0o111)
    assert [entry["path"] for entry in manifest["identity"]["files"]] == [
        "data.txt",
        "tool.sh",
    ]


def test_build_source_snapshot_inventories_c_quoted_paths(tmp_path: Path) -> None:
    """Names git C-quotes in porcelain listings (unicode, embedded quotes)
    must resolve to their extracted files instead of silently dropping out."""
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    # Pin quoting so the fixture C-quotes deterministically on every machine.
    _git(repo, "config", "core.quotepath", "true")
    committed_bytes = {
        "unicode-源.txt": b"unicode payload\n",
        'quo"ted.txt': b"quoted payload\n",
    }
    for relative_path, data in committed_bytes.items():
        (repo / relative_path).write_bytes(data)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "c-quoted fixture")
    run_sha = _git(repo, "rev-parse", "HEAD")

    # Precondition: without -z these paths really are C-quoted by git.
    assert '"' in _git(repo, "ls-tree", "-r", "-l", run_sha)

    output = tmp_path / "snapshot"
    manifest = build_source_snapshot(repo, run_sha, output)
    source = output / "source"

    inventory = {entry["path"]: entry for entry in manifest["identity"]["files"]}
    assert sorted(inventory) == sorted(committed_bytes)
    assert manifest["file_count"] == len(committed_bytes)
    for relative_path, data in committed_bytes.items():
        assert (source / relative_path).read_bytes() == data
        assert inventory[relative_path]["mode"] == "100644"
        assert inventory[relative_path]["size"] == len(data)
        assert inventory[relative_path]["sha256"] == sha256_bytes(data)


def test_build_source_snapshot_rejects_unextracted_tree_entries(
    tmp_path: Path,
) -> None:
    """A listed tree entry with no extracted counterpart (a gitlink, which
    git archive never extracts) must fail closed, not silently vanish."""
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    (repo / "real.txt").write_bytes(b"real\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")
    # Stage a gitlink (submodule entry) without materializing a submodule.
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{base_sha},vendored")
    _git(repo, "commit", "-q", "-m", "gitlink fixture")
    run_sha = _git(repo, "rev-parse", "HEAD")

    output = tmp_path / "snapshot"
    with pytest.raises(
        ValueError, match=r"not extracted: 'vendored' \(type commit\)"
    ):
        build_source_snapshot(repo, run_sha, output)
    assert not (output / "source_manifest.json").exists()


def test_build_source_snapshot_rejects_git_link_members(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    (repo / "target.txt").write_bytes(b"target\n")
    (repo / "linked.txt").symlink_to("target.txt")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "link fixture")
    run_sha = _git(repo, "rev-parse", "HEAD")

    output = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="rejects link member: linked.txt"):
        build_source_snapshot(repo, run_sha, output)
    assert not (output / "source_manifest.json").exists()
