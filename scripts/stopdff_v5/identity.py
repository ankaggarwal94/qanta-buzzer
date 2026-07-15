"""Canonical JSON identity bytes and content-addressed IDs.

Implements IDENTITY_AND_ARTIFACT_CONTRACT.md section 1:

1. recursively reject duplicate keys;
2. normalize all strings to Unicode NFC;
3. permit only objects, arrays, strings, integers, booleans, and null;
4. encode scientific decimal quantities as strings (e.g. "0.05", "1e-8");
5. sort object keys lexicographically by Unicode code point;
6. preserve array order;
7. encode UTF-8 with ensure_ascii=false;
8. use separators "," and ":" with no extra whitespace;
9. write no trailing newline.

    identifier = sha256(canonical_json_bytes(identity))

Floats are rejected on purpose: scientific decimals MUST be pre-encoded as strings so
that two conforming implementations produce byte-identical identity bytes. ``bool`` is
permitted (it is a JSON boolean) even though it subclasses ``int`` in Python.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


class IdentityError(ValueError):
    """Raised when an object cannot be canonicalized under the identity contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise IdentityError(f"non-string JSON key: {key!r}")
        if key in seen:
            raise IdentityError(f"duplicate JSON key: {key!r}")
        seen.add(key)
        out[key] = value
    return out


def loads_no_duplicate_keys(text: str) -> Any:
    """Parse JSON text, rejecting duplicate object keys anywhere in the tree."""
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def _canonicalize(value: Any, *, path: str = "$") -> Any:
    """Return an NFC-normalized, type-validated copy suitable for canonical dumping."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise IdentityError(
            f"float not permitted in identity at {path}: {value!r}. "
            "Encode scientific decimal quantities as strings."
        )
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        seen: set[str] = set()
        for key, sub in value.items():
            if not isinstance(key, str):
                raise IdentityError(f"non-string key at {path}: {key!r}")
            norm_key = unicodedata.normalize("NFC", key)
            if norm_key in seen:
                raise IdentityError(f"duplicate (NFC-folded) key at {path}: {norm_key!r}")
            seen.add(norm_key)
            out[norm_key] = _canonicalize(sub, path=f"{path}.{norm_key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item, path=f"{path}[{i}]") for i, item in enumerate(value)]
    raise IdentityError(f"unsupported type at {path}: {type(value).__name__}")


def canonical_bytes(identity: Any) -> bytes:
    """Return the canonical UTF-8 identity bytes for ``identity``."""
    normalized = _canonicalize(identity)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def compute_id(identity: Any) -> str:
    """Return sha256(canonical_json_bytes(identity)) as a hex digest."""
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Stream a file through sha256 (raw bytes; no newline normalization)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(identity: dict[str, Any], **volatile: Any) -> dict[str, Any]:
    """Assemble a content-addressed manifest: {"id", "identity", **volatile}.

    The ``id`` is recomputed from the canonical identity bytes. Volatile fields
    (paths, timestamps, wall times, display metadata) are stored alongside but are
    excluded from the identity and therefore from the ID.
    """
    manifest: dict[str, Any] = {"id": compute_id(identity), "identity": identity}
    for key, val in volatile.items():
        if key in ("id", "identity"):
            raise IdentityError(f"volatile field {key!r} collides with a reserved manifest field")
        manifest[key] = val
    return manifest


def verify_manifest_id(manifest: dict[str, Any]) -> str:
    """Recompute the manifest ID from its identity block and reject disagreement.

    Returns the verified ID. Raises IdentityError on mismatch or missing fields.
    """
    if "identity" not in manifest:
        raise IdentityError("manifest is missing its 'identity' block")
    if "id" not in manifest:
        raise IdentityError("manifest is missing its 'id'")
    recomputed = compute_id(manifest["identity"])
    if recomputed != manifest["id"]:
        raise IdentityError(
            f"manifest ID mismatch: stored={manifest['id']}, recomputed={recomputed}"
        )
    return recomputed
