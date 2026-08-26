"""QA-012 ``format:"QA"`` detector + inventory manifest (R-072).

Closure requires the sign-off §8 inventory procedure executed over the
declared corpora: for every ``.json``/``.jsonl`` file record path, size, and
SHA-256; strictly parse; recursively detect any key named ``format`` whose
value is exactly ``"QA"``; emit the matching RFC-6901 JSON pointer for every
hit; store a hashed zero-hit/match manifest.
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import schema

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures" / "qa012_item10"
_FIXTURE_BINDINGS_BASENAME = "bindings.json"


def _escape_pointer_token(token: str) -> str:
    """RFC 6901 escaping: ``~`` -> ``~0`` then ``/`` -> ``~1``."""
    return token.replace("~", "~0").replace("/", "~1")


def detect_format_qa(document: Any, _prefix: str = "") -> list[str]:
    """Recursively detect keys named exactly ``format`` whose value is the
    exact string ``"QA"``; return the RFC-6901 JSON pointer of every hit."""
    pointers: list[str] = []
    if isinstance(document, dict):
        for key, value in document.items():
            pointer = f"{_prefix}/{_escape_pointer_token(str(key))}"
            if key == "format" and isinstance(value, str) and value == "QA":
                pointers.append(pointer)
            pointers.extend(detect_format_qa(value, pointer))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            pointers.extend(detect_format_qa(value, f"{_prefix}/{index}"))
    return pointers


def _scan_file(path: Path, root: Path) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    data = schema.read_regular_file_bytes(path, tree_root=root)
    hits: list[str] = []
    if path.suffix == ".jsonl":
        try:
            text = data.decode("utf-8", errors="strict") if data else ""
        except UnicodeDecodeError as exc:
            raise schema.TypedIngressError(
                f"{rel}: invalid UTF-8 bytes at byte offset {exc.start}"
                " (R-020)"
            ) from exc
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            obj = schema.parse_json_text_strict(line, f"{rel}: line {lineno}")
            hits.extend(detect_format_qa(obj))
    else:
        try:
            obj = schema.parse_json_bytes_strict(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise schema.TypedIngressError(
                f"{rel}: malformed JSON: {exc} (R-020)"
            ) from exc
        hits.extend(detect_format_qa(obj))
    entry = {
        "path": rel,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "hits": sorted(hits),
    }
    if path.suffix == ".jsonl":
        first_records = b"".join(data.splitlines(keepends=True)[:2])
        entry["first_two_records_sha256"] = hashlib.sha256(
            first_records
        ).hexdigest()
    return entry


def hit_fixtures_verified(
    manifest: dict[str, Any], fixtures_root: Path | None = None
) -> bool:
    """Verify that every hit file is covered by the committed R-078 fixtures."""
    root = Path(fixtures_root or _DEFAULT_FIXTURES_ROOT)
    try:
        bindings_raw = schema.read_regular_file_bytes(
            root / _FIXTURE_BINDINGS_BASENAME, tree_root=root
        )
        bindings = schema.parse_json_bytes_strict(bindings_raw)
        if not isinstance(bindings, dict):
            return False
        schema.check_schema_version(bindings, "QA-012 fixture bindings")
        bound_files = bindings.get("files")
        if not isinstance(bound_files, dict) or not bound_files:
            return False

        hit_entries = [
            entry
            for entry in manifest.get("files", [])
            if isinstance(entry, dict) and entry.get("hits")
        ]
        by_basename: dict[str, dict[str, Any]] = {}
        for entry in hit_entries:
            basename = Path(str(entry.get("path", ""))).name
            if not basename or basename in by_basename:
                return False
            by_basename[basename] = entry
        if set(by_basename) != set(bound_files):
            return False

        for basename, binding_value in bound_files.items():
            if not isinstance(binding_value, dict):
                return False
            excerpt_name = binding_value.get("excerpt_fixture")
            if not isinstance(excerpt_name, str) or not excerpt_name:
                return False
            excerpt = schema.read_regular_file_bytes(
                root / excerpt_name, tree_root=root
            )
            excerpt_sha256 = hashlib.sha256(excerpt).hexdigest()
            if excerpt_sha256 != binding_value.get("excerpt_sha256"):
                return False
            if len(excerpt.splitlines()) != binding_value.get("excerpt_lines"):
                return False

            entry = by_basename[basename]
            if entry.get("sha256") != binding_value.get("full_file_sha256"):
                return False
            if entry.get("size") != binding_value.get("full_file_size"):
                return False
            if entry.get("first_two_records_sha256") != excerpt_sha256:
                return False
    except (OSError, KeyError, TypeError, schema.ColmAimsError):
        return False
    return True


def build_inventory_manifest(roots: list[Path]) -> dict[str, Any]:
    """Execute the QA-012 inventory over the given corpus roots (R-072).

    Strict parse failures are typed ingress errors (the inventory never
    silently skips an unreadable file). The manifest hash is deterministic
    over the sorted per-file entries.
    """
    if not roots:
        raise schema.TypedIngressError(
            "QA-012 requires at least one declared corpus root"
        )
    files: list[dict[str, Any]] = []
    root_file_counts: list[int] = []
    for root_index, root in enumerate(roots):
        root = Path(root)
        if not root.exists() or not root.is_dir() or root.is_symlink():
            raise schema.TypedIngressError(
                f"QA-012 corpus root {root_index} is missing or not a"
                " regular directory"
            )
        candidates = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in (".json", ".jsonl")
        ]
        if not candidates:
            raise schema.TypedIngressError(
                f"QA-012 corpus root {root_index} contains no JSON/JSONL"
                " files; a vacuous scope cannot satisfy closure"
            )
        root_file_counts.append(len(candidates))
        for path in candidates:
            if not path.is_file():
                continue
            entry = _scan_file(path, root)
            entry["root_index"] = root_index
            files.append(entry)
    files.sort(key=lambda entry: entry["path"])
    inventory_payload = json.dumps(
        {"root_file_counts": root_file_counts, "files": files},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    any_hits = any(entry["hits"] for entry in files)
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "result": "hits" if any_hits else "zero_hit",
        "root_file_counts": root_file_counts,
        "files": files,
        "inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
    }
