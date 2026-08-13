"""Regressions for canonical content-manifest roots on PR #30."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.stopdff_v5.content_manifest import validate_bound_content_manifest
from scripts.stopdff_v5.identity import build_manifest, sha256_file
from scripts.stopdff_v5.manifests import source_manifest_identity


def _write_source_snapshot(base: Path) -> dict:
    source = base / "source"
    source.mkdir(parents=True)
    payloads = {
        "pyproject.toml": b"[project]\nname = \"manifest-root-fixture\"\n",
        "uv.lock": b"version = 1\nrevision = 3\n",
    }
    files = []
    digests = {}
    for relative_path, payload in payloads.items():
        path = source / relative_path
        path.write_bytes(payload)
        digest = sha256_file(path)
        digests[relative_path] = digest
        files.append(
            {
                "path": relative_path,
                "mode": "100644",
                "size": len(payload),
                "sha256": digest,
            }
        )

    manifest = build_manifest(
        source_manifest_identity(
            git_sha="a" * 40,
            files=files,
            pyproject_sha256=digests["pyproject.toml"],
            uv_lock_sha256=digests["uv.lock"],
        )
    )
    (base / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_bound_content_manifest_rejects_symlinked_stage_root(
    tmp_path: Path,
) -> None:
    actual_root = tmp_path / "actual-source-stage"
    manifest = _write_source_snapshot(actual_root)
    validation_args = {
        "manifest_name": "source_manifest.json",
        "expected_id": manifest["id"],
        "file_key": "files",
        "name_key": "path",
        "content_subdir": "source",
        "expected_kind": "source_snapshot",
    }

    assert validate_bound_content_manifest(
        actual_root,
        **validation_args,
    ) == manifest

    linked_root = tmp_path / "linked-source-stage"
    linked_root.symlink_to(actual_root, target_is_directory=True)
    with pytest.raises(ValueError, match="manifest root is noncanonical"):
        validate_bound_content_manifest(
            linked_root,
            **validation_args,
        )
