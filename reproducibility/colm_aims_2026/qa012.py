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
    return {
        "path": rel,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "hits": sorted(hits),
    }


def build_inventory_manifest(roots: list[Path]) -> dict[str, Any]:
    """Execute the QA-012 inventory over the given corpus roots (R-072).

    Strict parse failures are typed ingress errors (the inventory never
    silently skips an unreadable file). The manifest hash is deterministic
    over the sorted per-file entries.
    """
    files: list[dict[str, Any]] = []
    for root in roots:
        root = Path(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in (".json", ".jsonl"):
                continue
            files.append(_scan_file(path, root))
    files.sort(key=lambda entry: entry["path"])
    inventory_payload = json.dumps(
        files, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    any_hits = any(entry["hits"] for entry in files)
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "result": "hits" if any_hits else "zero_hit",
        "files": files,
        "inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
    }
