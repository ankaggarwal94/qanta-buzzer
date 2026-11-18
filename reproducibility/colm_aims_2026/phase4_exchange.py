"""Publish and pull settled Phase 4 evidence trees using a v2 file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import os
import platform
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from scripts.stopdff_v5 import fileio

from . import schema

_REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "RETURN_MANIFEST.json"
DEFAULT_MANIFEST_TYPE = "phase4_exchange_return"
MANIFEST_SCHEMA_VERSION = 2


class ExchangeError(schema.ColmAimsError):
    """The bundle could not be published or verified."""


def _portable_component(name: str) -> bool:
    """Recognize one literal filename shared by Windows and POSIX."""
    return (
        schema.is_path_component(name)
        and not name.endswith((".", " "))
        and not any(c in '<>:"/\\|?*' or ord(c) < 32 for c in name)
        and not (
            ntpath.isreserved(name)
            if hasattr(ntpath, "isreserved")
            else PureWindowsPath(name).is_reserved()
        )
    )


def _check_member_names(names: Iterable[str]) -> None:
    # Validate directory prefixes too: otherwise Data/a and data/b would
    # silently share a different directory on the receiving filesystem.
    seen: dict[str, str] = {}
    for name in names:
        parts = name.split("/")
        if not all(_portable_component(part) for part in parts):
            raise ExchangeError(f"path {name!r} escapes the bundle or is not portable")
        if parts[0].casefold() == MANIFEST_NAME.casefold():
            raise ExchangeError(
                f"{name!r} uses the reserved manifest name {MANIFEST_NAME}"
            )
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            prior = seen.setdefault(
                unicodedata.normalize("NFC", prefix).casefold(), prefix
            )
            if prior != prefix:
                raise ExchangeError(f"case collision between {prior!r} and {prefix!r}")


def _members(root: Path) -> set[str]:
    """Enumerate regular files and propagate directory traversal failures."""
    if not root.is_dir():
        raise ExchangeError(f"missing directory for run or exchange: {root}")
    if schema.is_filesystem_link(root):
        raise ExchangeError(f"symlink or reparse link refused: {root}")
    members = set()
    for current, dirs, files in os.walk(root, onerror=_raise_walk_error):
        for name in [*dirs, *files]:
            path = Path(current, name)
            if schema.is_filesystem_link(path):
                raise ExchangeError(f"symlink or reparse link refused: {path}")
        for name in files:
            path = Path(current, name)
            if not path.is_file():
                raise ExchangeError(f"expected a regular file: {path}")
            members.add(path.relative_to(root).as_posix())
    return members


def _raise_walk_error(error: OSError) -> None:
    raise ExchangeError(f"could not enumerate bundle: {error}") from error


def _copy_file_hashed(src: Path, dest: Path) -> tuple[int, str]:
    """Hash the same stream that is written, with bounded memory."""
    chain = schema.stable_directory_chain(src.parent, src.parent)
    before = src.lstat()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(src, flags), "rb") as reader:
        info = os.fstat(reader.fileno())
        if not stat.S_ISREG(info.st_mode) or not os.path.samestat(info, before):
            raise ExchangeError(f"expected a stable regular file: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        digest = hashlib.sha256()
        with dest.open("xb") as writer:
            while chunk := reader.read(1 << 20):
                writer.write(chunk)
                size += len(chunk)
                digest.update(chunk)
        after = src.lstat()
        if (
            not os.path.samestat(info, after)
            or (info.st_size, info.st_mtime_ns) != (size, after.st_mtime_ns)
            or schema.stable_directory_chain(src.parent, src.parent) != chain
        ):
            raise ExchangeError(f"source changed during copy: {src}")
    return size, digest.hexdigest()


def _outside_repo(path: Path) -> None:
    if schema.resolves_inside(path, _REPO_ROOT):
        raise ExchangeError(f"destination is inside the repository: {path}")


def _require_new(path: Path) -> None:
    if os.path.lexists(path):
        raise ExchangeError(f"destination already exists: {path}")


def _publish_dir(staged: Path, dest: Path) -> None:
    # The shared primitive claims an empty directory on POSIX. Only remove
    # that claim after an ordinary failure if this call acquired it.
    claimed = False

    def own_claim() -> None:
        nonlocal claimed
        claimed = True

    try:
        fileio.publish_dir_create_once(staged, dest, claim_created=own_claim)
    except fileio.DirectoryPublicationCommittedError as exc:
        # The completed tree is visible; a retry using this name must refuse.
        raise ExchangeError(
            f"published at {dest}, but parent sync failed: {exc.cause}"
        ) from exc
    except OSError as exc:
        if claimed:
            dest.rmdir()
        raise ExchangeError(f"could not publish {dest}: {exc}") from exc


def publish_run_results(
    run_root: Path,
    exchange_root: Path,
    *,
    name: str,
    provenance: dict[str, Any],
    now: Callable[[], str] | None = None,
    manifest_type: str = DEFAULT_MANIFEST_TYPE,
) -> Path:
    """Package the regular files of a settled run under a fresh exchange name.

    Parameters
    ----------
    run_root, exchange_root
        Disjoint source and exchange directories. The exchange lives outside the repo.
    name, provenance
        Portable bundle name and caller-supplied JSON metadata.
    now, manifest_type
        Optional timestamp supplier and label for the v2 manifest.

    Returns
    -------
    Path
        Published directory containing the payload and its manifest.
    """
    run_root, exchange_root = Path(run_root), Path(exchange_root)
    _outside_repo(exchange_root)
    if not isinstance(name, str) or not _portable_component(name):
        raise ExchangeError(f"expected a portable bundle name, got {name!r}")
    if (
        not isinstance(provenance, dict)
        or not isinstance(manifest_type, str)
        or not manifest_type
    ):
        raise ExchangeError("expected a provenance object and a nonempty manifest_type")
    if schema.resolves_inside(exchange_root, run_root) or schema.resolves_inside(
        run_root, exchange_root
    ):
        raise ExchangeError("the run and exchange roots cannot lie inside one another")
    dest = exchange_root / name
    _require_new(dest)
    files = _members(run_root)
    if not files:
        raise ExchangeError(f"empty run root: {run_root}")
    _check_member_names(files)

    exchange_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=exchange_root, prefix=f"{name}.partial-"
    ) as tmp:
        staged = Path(tmp)
        entries = []
        for path in sorted(files):
            size, digest = _copy_file_hashed(run_root / path, staged / path)
            entries.append({"path": path, "size": size, "sha256": digest})
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "manifest_type": manifest_type,
            "entries": entries,
            "file_count": len(entries),
            "total_bytes": sum(entry["size"] for entry in entries),
            "provenance": provenance,
            "generated_at": now() if now is not None else datetime.now(UTC).isoformat(),
            "platform": platform.platform() or "unknown-platform",
            "node": platform.node() or "unknown-node",
        }
        data = schema.encode_json(manifest)
        if len(data) > schema.MAX_ARTIFACT_BYTES:
            raise ExchangeError(f"{MANIFEST_NAME} exceeds the manifest size limit")
        (staged / MANIFEST_NAME).write_bytes(data)
        _publish_dir(staged, dest)
    return dest


def _manifest_entries(manifest: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ExchangeError(f"{MANIFEST_NAME} must contain an object")
    version = manifest.get("schema_version")
    if type(version) is not int or version != MANIFEST_SCHEMA_VERSION:
        raise ExchangeError(f"unexpected manifest schema_version: {version!r}")
    if (
        not isinstance(manifest.get("manifest_type"), str)
        or not manifest["manifest_type"]
    ):
        raise ExchangeError("expected a nonempty manifest_type")
    if not isinstance(manifest.get("provenance"), dict):
        raise ExchangeError("expected a provenance object")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ExchangeError("expected nonempty manifest entries")
    by_path = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ExchangeError("expected a file entry with a path")
        path = entry["path"]
        if path in by_path:
            raise ExchangeError(f"duplicate manifest entry: {path!r}")
        if (
            type(entry.get("size")) is not int
            or entry["size"] < 0
            or not schema.is_sha256_hex(entry.get("sha256"))
        ):
            raise ExchangeError(f"invalid size or sha256 for {path!r}")
        by_path[path] = entry
    _check_member_names(by_path)
    for key, count in (
        ("file_count", len(entries)),
        ("total_bytes", sum(e["size"] for e in entries)),
    ):
        if type(manifest.get(key)) is not int or manifest[key] != count:
            raise ExchangeError(f"{key} disagrees with the manifest entries")
    return by_path


def pull_verified(exchange_dir: Path, local_dest: Path) -> dict[str, Any]:
    """Copy a complete manifest-listed tree to a fresh local directory.

    Parameters
    ----------
    exchange_dir, local_dest
        Published bundle and a fresh, out-of-repository intake destination.

    Returns
    -------
    dict
        Verified file and byte counts, plus the publisher's metadata and type.
    """
    exchange_dir, local_dest = Path(exchange_dir), Path(local_dest)
    _outside_repo(local_dest)
    if schema.resolves_inside(local_dest, exchange_dir):
        raise ExchangeError("local destination is inside the published bundle")
    _require_new(local_dest)
    manifest_path = exchange_dir / MANIFEST_NAME
    try:
        data = schema.read_regular_file_bytes(manifest_path, tree_root=exchange_dir)
        manifest = schema.parse_json_bytes_strict(data)
    except (schema.ColmAimsError, ValueError) as exc:
        raise ExchangeError(f"cannot read {MANIFEST_NAME}: {exc}") from exc
    entries = _manifest_entries(manifest)
    files = _members(exchange_dir) - {MANIFEST_NAME}
    if missing := sorted(entries.keys() - files):
        raise ExchangeError(f"missing manifest file: {missing[0]}")
    if extra := sorted(files - entries.keys()):
        raise ExchangeError(f"extra file outside manifest: {extra[0]}")

    local_dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=local_dest.parent, prefix=f"{local_dest.name}.pulling-"
    ) as tmp:
        staged = Path(tmp)
        for path, entry in entries.items():
            size, digest = _copy_file_hashed(exchange_dir / path, staged / path)
            if size != entry["size"]:
                raise ExchangeError(
                    f"{path}: expected size {entry['size']}, observed {size}"
                )
            if digest != entry["sha256"]:
                raise ExchangeError(
                    f"{path}: expected sha256 {entry['sha256']}, observed {digest}"
                )
        if _members(exchange_dir) != files | {MANIFEST_NAME}:
            raise ExchangeError("bundle file list changed during pull")
        try:
            latest = schema.read_regular_file_bytes(
                manifest_path, tree_root=exchange_dir
            )
        except schema.ColmAimsError as exc:
            raise ExchangeError(f"cannot recheck {MANIFEST_NAME}: {exc}") from exc
        if latest != data:
            raise ExchangeError(f"{MANIFEST_NAME} changed during pull")
        _publish_dir(staged, local_dest)
    return {
        "verdict": "PULLED",
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "provenance": manifest["provenance"],
        "manifest_type": manifest["manifest_type"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish", help="package a settled directory")
    publish.add_argument("run_root", type=Path)
    publish.add_argument("exchange_root", type=Path)
    publish.add_argument("--name", required=True)
    publish.add_argument("--provenance", type=Path, help="JSON metadata file")
    publish.add_argument("--manifest-type", default=DEFAULT_MANIFEST_TYPE)
    pull = commands.add_parser("pull", help="verify and copy a published bundle")
    pull.add_argument("exchange_dir", type=Path)
    pull.add_argument("local_dest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "publish":
            metadata = {}
            if args.provenance is not None:
                metadata = schema.parse_json_bytes_strict(
                    schema.read_regular_file_bytes(args.provenance)
                )
            dest = publish_run_results(
                args.run_root,
                args.exchange_root,
                name=args.name,
                provenance=metadata,
                manifest_type=args.manifest_type,
            )
            print(dest)
        else:
            print(
                json.dumps(
                    pull_verified(args.exchange_dir, args.local_dest), sort_keys=True
                )
            )
    except (OSError, ValueError, TypeError, schema.ColmAimsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
