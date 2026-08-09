"""Fail-closed validation for staged source, raw, and model content manifests."""
from __future__ import annotations

import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from . import PROTOCOL_VERSION
from .identity import compute_id, loads_no_duplicate_keys, sha256_file
from .manifests import ADAPTER_SCORING_SPEC, RAW_INPUT_ROLES

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_FIELDS = {
    "kind",
    "protocol_version",
    "git_sha",
    "files",
    "pyproject_toml_sha256",
    "uv_lock_sha256",
}
_RAW_FIELDS = {"kind", "files", "semantic_checks"}
_MODEL_FIELDS = {
    "kind",
    "model_id",
    "model_revision",
    "trust_remote_code",
    "files",
    "sentence_transformers_version",
    "transformers_version",
}
_CONTENT_LAYOUTS = {
    "source_snapshot": (
        "files",
        "path",
        {"path", "mode", "size", "sha256"},
    ),
    "raw_input_bundle": (
        "files",
        "role",
        {"role", "size", "sha256"},
    ),
    "model_snapshot": (
        "files",
        "path",
        {"path", "size", "sha256"},
    ),
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def git_mode_for_path(path: Path) -> str:
    """Return the only executable-bit distinction tracked by Git."""
    return "100755" if stat.S_IMODE(Path(path).stat().st_mode) & 0o111 else "100644"


def _canonical_entry(
    entry: Any,
    *,
    expected_fields: set[str],
    name_key: str,
    manifest_name: str,
) -> tuple[str, int, str]:
    """Return the canonical path, size, and digest for one manifest entry."""
    if not isinstance(entry, dict) or set(entry) != expected_fields:
        raise ValueError(f"{manifest_name} file entry fields mismatch")
    name = entry.get(name_key)
    parsed = PurePosixPath(name) if isinstance(name, str) else None
    size = entry.get("size")
    digest = entry.get("sha256")
    if (
        not isinstance(name, str)
        or not name
        or parsed is None
        or parsed.is_absolute()
        or parsed.as_posix() != name
        or ".." in parsed.parts
        or "\\" in name
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not _is_sha256(digest)
    ):
        raise ValueError(f"{manifest_name} contains a noncanonical file entry")
    return name, size, digest


def _validate_identity_schema(
    identity: dict[str, Any],
    *,
    kind: str,
    names: list[str],
    entries_by_name: dict[str, dict[str, Any]],
    manifest_name: str,
    require_semantic_pass: bool,
) -> None:
    """Validate the exact identity envelope and producer-defined closed sets."""
    if kind == "source_snapshot":
        if set(identity) != _SOURCE_FIELDS:
            raise ValueError(f"{manifest_name} identity fields mismatch")
        if identity.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"{manifest_name} protocol version mismatch")
        if not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("git_sha", ""))):
            raise ValueError(f"{manifest_name} git_sha is not canonical")
        if not names:
            raise ValueError(f"{manifest_name} source inventory is empty")
        for field, path in (
            ("pyproject_toml_sha256", "pyproject.toml"),
            ("uv_lock_sha256", "uv.lock"),
        ):
            expected = entries_by_name.get(path, {}).get("sha256", "")
            if identity.get(field) != expected:
                raise ValueError(f"{manifest_name} {field} mismatch")
    elif kind == "raw_input_bundle":
        if set(identity) != _RAW_FIELDS:
            raise ValueError(f"{manifest_name} identity fields mismatch")
        if tuple(names) != tuple(sorted(RAW_INPUT_ROLES)):
            raise ValueError(f"{manifest_name} raw-input roles mismatch")
        semantic_checks = identity.get("semantic_checks")
        if (
            not isinstance(semantic_checks, dict)
            or not _is_sha256(
                semantic_checks.get("question_trajectory_binding_id")
            )
        ):
            raise ValueError(
                f"{manifest_name} lacks a canonical question trajectory binding"
            )
        if require_semantic_pass and (
            semantic_checks.get("all_semantic_checks_pass") is not True
        ):
            raise ValueError(
                f"{manifest_name} does not record passing semantic checks"
            )
    elif kind == "model_snapshot":
        if set(identity) != _MODEL_FIELDS:
            raise ValueError(f"{manifest_name} identity fields mismatch")
        if identity.get("model_id") != ADAPTER_SCORING_SPEC["model_id"]:
            raise ValueError(f"{manifest_name} model_id mismatch")
        if not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("model_revision", ""))):
            raise ValueError(f"{manifest_name} model revision is not canonical")
        if identity.get("trust_remote_code") is not False:
            raise ValueError(f"{manifest_name} enables remote model code")
        if not names:
            raise ValueError(f"{manifest_name} model inventory is empty")
        for field in ("sentence_transformers_version", "transformers_version"):
            if not isinstance(identity.get(field), str) or not identity[field]:
                raise ValueError(f"{manifest_name} {field} is empty")
    else:
        raise ValueError(f"{manifest_name} has unsupported kind {kind!r}")


def validate_content_manifest_document(
    manifest: Any,
    *,
    manifest_name: str,
    expected_id: str | None,
    expected_kind: str | None = None,
    require_semantic_pass: bool = False,
) -> dict[str, Any]:
    """Validate one source/raw/model manifest without trusting external bytes.

    This document-level gate is shared by staging and package validation so a
    self-consistently rehashed manifest cannot weaken the canonical identity
    envelope.  ``validate_bound_content_manifest`` additionally proves that
    the declared inventory is exhaustive for a supplied content root.
    """
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if not isinstance(identity, dict) or compute_id(identity) != manifest.get("id"):
        raise ValueError(f"{manifest_name} id mismatch")
    if expected_id is not None and manifest["id"] != expected_id:
        raise ValueError(
            f"{manifest_name} id {manifest['id']} != expected {expected_id}"
        )
    kind = identity.get("kind")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(
            f"{manifest_name} kind {kind!r} != expected {expected_kind!r}"
        )
    layout = _CONTENT_LAYOUTS.get(kind)
    if layout is None:
        raise ValueError(f"{manifest_name} has unsupported kind {kind!r}")
    file_key, name_key, entry_fields = layout
    entries = identity.get(file_key)
    if not isinstance(entries, list):
        raise ValueError(f"{manifest_name} lacks {file_key}")

    seen: set[str] = set()
    names: list[str] = []
    entries_by_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name, _size, _digest = _canonical_entry(
            entry,
            expected_fields=entry_fields,
            name_key=name_key,
            manifest_name=manifest_name,
        )
        if name in seen:
            raise ValueError(f"{manifest_name} contains a duplicate path")
        seen.add(name)
        names.append(name)
        entries_by_name[name] = entry
        if kind == "source_snapshot" and entry.get("mode") not in {
            "100644",
            "100755",
        }:
            raise ValueError(f"{manifest_name} source mode is noncanonical: {name}")
    if names != sorted(names):
        raise ValueError(f"{manifest_name} file entries are not canonically ordered")
    _validate_identity_schema(
        identity,
        kind=str(kind),
        names=names,
        entries_by_name=entries_by_name,
        manifest_name=manifest_name,
        require_semantic_pass=require_semantic_pass,
    )
    return manifest


def validate_bound_content_manifest(
    base: Path,
    *,
    manifest_name: str,
    expected_id: str | None,
    file_key: str,
    name_key: str,
    content_subdir: str = "",
    expected_kind: str | None = None,
    require_semantic_pass: bool = False,
) -> dict[str, Any]:
    """Validate identity schema, declared bytes, and the exhaustive inventory."""
    base = Path(base)
    manifest_path = base / manifest_name
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"{manifest_name} is missing or not a regular file")
    manifest = loads_no_duplicate_keys(manifest_path.read_text(encoding="utf-8"))
    validate_content_manifest_document(
        manifest,
        manifest_name=manifest_name,
        expected_id=expected_id,
        expected_kind=expected_kind,
        require_semantic_pass=require_semantic_pass,
    )
    identity = manifest["identity"]
    kind = identity["kind"]
    canonical_file_key, canonical_name_key, _entry_fields = _CONTENT_LAYOUTS[kind]
    if file_key != canonical_file_key or name_key != canonical_name_key:
        raise ValueError(f"{manifest_name} content layout mismatch")
    entries = identity[file_key]

    content_root = base / content_subdir if content_subdir else base
    if content_root.is_symlink() or not content_root.is_dir():
        raise ValueError(f"{manifest_name} content root is noncanonical")
    seen: set[str] = set()
    for entry in entries:
        name = entry[name_key]
        size = entry["size"]
        digest = entry["sha256"]
        seen.add(name)
        target = content_root / name
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != size
            or sha256_file(target) != digest
            or (
                kind == "source_snapshot"
                and git_mode_for_path(target) != entry["mode"]
            )
        ):
            raise ValueError(f"{manifest_name} file mismatch: {name}")
    actual: set[str] = set()
    for path in content_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{manifest_name} content contains a symlink")
        if path.is_file():
            relative = path.relative_to(content_root).as_posix()
            if content_root == base and relative == manifest_name:
                continue
            actual.add(relative)
        elif not path.is_dir():
            raise ValueError(f"{manifest_name} content contains a special file")
    if actual != seen:
        raise ValueError(
            f"{manifest_name} inventory mismatch: "
            f"unlisted={sorted(actual - seen)}, missing={sorted(seen - actual)}"
        )
    return manifest
