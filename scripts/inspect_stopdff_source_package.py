#!/usr/bin/env python3
"""Inspect source extraction permissions, optionally repair a new package copy.

The source manifest ID must come from the trusted reproduction instructions.
This checks package checksums and source identity/bytes; it does not replace the
full scientific package validator. Use a stable, private extraction directory
with no concurrent writers. Never run against an actively produced run.

Repair refuses all non-mode source failures, links, special files, existing
outputs, and receipts inside either package. Only Git's executable distinction
is restored; other permission bits and all bytes are preserved. If copying or
post-copy verification fails, any incomplete destination is left for inspection
and no success receipt is emitted. The original is never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stopdff_v5.checker_package import check_complete_checksums
from scripts.stopdff_v5.content_manifest import (
    git_mode_for_path,
    validate_bound_content_manifest,
    validate_content_manifest_document,
)
from scripts.stopdff_v5.identity import (
    is_sha256_hex,
    loads_no_duplicate_keys,
    sha256_file,
)

_SOURCE_SUBDIR = "evidence/source_snapshot/source"


def _unlinked_path(path: Path) -> Path:
    """Reject symlink traversal without resolving it away first."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symlink path is unsupported: {part}")
    return path


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        raise ValueError(f"package is not a directory: {root}")
    files: dict[str, dict[str, Any]] = {}
    # Inspect types for the entire tree before opening any payload. rglob does
    # not descend through directory symlinks; every symlink itself is rejected.
    paths = sorted(root.rglob("*"))
    for path in paths:
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"package contains a link or special file: {path}")
    for path in paths:
        if path.is_file():
            metadata = path.stat()
            files[path.relative_to(root).as_posix()] = {
                "sha256": sha256_file(path),
                "size": metadata.st_size,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
    return files


def _inspect(
    root: Path, expected_source_id: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not is_sha256_hex(expected_source_id):
        raise ValueError("expected source ID must be a canonical SHA256")
    inventory = _inventory(root)
    manifest_path = root / "evidence/source_manifest.json"
    if manifest_path.relative_to(root).as_posix() not in inventory:
        raise ValueError("missing regular evidence/source_manifest.json")
    manifest = loads_no_duplicate_keys(manifest_path.read_text(encoding="utf-8"))
    validate_content_manifest_document(
        manifest,
        manifest_name="source_manifest.json",
        expected_id=expected_source_id,
        expected_kind="source_snapshot",
    )
    source = root / _SOURCE_SUBDIR
    if not source.is_dir():
        raise ValueError("missing source snapshot directory")
    entries = manifest["identity"]["files"]
    declared = {entry["path"] for entry in entries}
    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*") if path.is_file()
    }
    if actual != declared:
        raise ValueError(
            f"source inventory mismatch: unlisted={sorted(actual - declared)}, "
            f"missing={sorted(declared - actual)}"
        )
    declared_dirs = {
        parent.as_posix()
        for name in declared for parent in Path(name).parents
        if parent != Path(".")
    }
    actual_dirs = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*") if path.is_dir()
    }
    if actual_dirs != declared_dirs:
        raise ValueError("source directory inventory mismatch")
    modes = []
    for entry in entries:
        name = entry["path"]
        record = inventory[f"{_SOURCE_SUBDIR}/{name}"]
        if record["size"] != entry["size"]:
            raise ValueError(f"source size mismatch: {name}")
        if record["sha256"] != entry["sha256"]:
            raise ValueError(f"source sha256 mismatch: {name}")
        actual_mode = git_mode_for_path(source / name)
        if actual_mode != entry["mode"]:
            modes.append({"path": name, "expected": entry["mode"], "actual": actual_mode})
    checksum_errors: list[str] = []
    check_complete_checksums(root, checksum_errors)
    if checksum_errors:
        raise ValueError("package checksum validation failed: " + "; ".join(checksum_errors))
    report = {
        "source_manifest_id": manifest["id"],
        "git_sha": manifest["identity"]["git_sha"],
        "source_files_verified": len(entries),
        "package_files_verified": len(inventory),
        "checksum_entries_verified": len(inventory) - 1,
        "package_checksums_valid": True,
        "mode_mismatches": modes,
    }
    return report, inventory


def inspect_package(root: Path, *, expected_source_id: str) -> dict[str, Any]:
    """Check a complete run root and list every source executable mismatch."""
    report, _ = _inspect(_unlinked_path(root), expected_source_id)
    return report


def repair_package(
    root: Path, destination: Path, receipt_path: Path, *, expected_source_id: str
) -> dict[str, Any]:
    """Restore source executable bits only in a verified, newly created copy."""
    root = _unlinked_path(root)
    destination = _unlinked_path(destination)
    receipt_path = _unlinked_path(receipt_path)
    if root == destination or root in destination.parents or destination in root.parents:
        raise ValueError("source and destination packages must be disjoint")
    if destination.exists():
        raise ValueError("repair destination already exists")
    if receipt_path.exists():
        raise ValueError("receipt already exists")
    if any(receipt_path == p or p in receipt_path.parents for p in (root, destination)):
        raise ValueError("receipt must be outside both checksum-bound package roots")
    if not destination.parent.is_dir() or not receipt_path.parent.is_dir():
        raise ValueError("destination and receipt parent directories must exist")
    report, before = _inspect(root, expected_source_id)
    # Preserve links if a concurrent writer inserts one: post-copy inspection
    # rejects it, rather than copytree following it outside the source tree.
    shutil.copytree(root, destination, symlinks=True)
    copied = _inventory(destination)
    if copied != before:
        raise ValueError("copy bytes or file modes differ before repair")
    for mismatch in report["mode_mismatches"]:
        target = destination / _SOURCE_SUBDIR / mismatch["path"]
        mode = stat.S_IMODE(target.stat().st_mode)
        restored = (mode | 0o111) if mismatch["expected"] == "100755" else (mode & ~0o111)
        target.chmod(restored)
    after_report, after = _inspect(destination, expected_source_id)
    if after_report["mode_mismatches"]:
        raise ValueError("copied source executable modes could not be restored")
    expected_after = {name: dict(record) for name, record in before.items()}
    for mismatch in report["mode_mismatches"]:
        record = expected_after[f"{_SOURCE_SUBDIR}/{mismatch['path']}"]
        record["mode"] = (record["mode"] | 0o111) if mismatch["expected"] == "100755" else (record["mode"] & ~0o111)
    if after != expected_after:
        raise ValueError("repair changed bytes or unexpected file modes")
    validate_bound_content_manifest(
        destination / "evidence",
        manifest_name="source_manifest.json",
        expected_id=expected_source_id,
        expected_kind="source_snapshot",
        file_key="files",
        name_key="path",
        content_subdir="source_snapshot/source",
    )
    if _inventory(root) != before:
        raise ValueError("source changed during inspection or copying; discard this copy")
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_package": str(root),
        "repaired_package": str(destination),
        **report,
        "source_files_changed": len(report["mode_mismatches"]),
        "byte_changes": 0,
        "repair": "Restored only source-manifest executable bits in a new copy; original unchanged; all package bytes verified before and after.",
    }
    # Exclusive create prevents accidental replacement of an existing receipt.
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="extracted verified_export directory")
    parser.add_argument("--expected-source-id", required=True)
    parser.add_argument("--repair-copy", type=Path, help="nonexistent destination package directory")
    parser.add_argument("--receipt", type=Path, help="new JSON file outside both package roots")
    args = parser.parse_args(argv)
    if bool(args.repair_copy) != bool(args.receipt):
        parser.error("--repair-copy and --receipt must be supplied together")
    try:
        if args.repair_copy:
            report = repair_package(args.run_root, args.repair_copy, args.receipt, expected_source_id=args.expected_source_id)
        else:
            report = inspect_package(args.run_root, expected_source_id=args.expected_source_id)
    except (OSError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    # Inspection distinguishes intact extraction (0), repairable mode mismatch
    # (2), and validation failure (1). A successful repair always returns 0.
    return 0 if args.repair_copy or not report["mode_mismatches"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
