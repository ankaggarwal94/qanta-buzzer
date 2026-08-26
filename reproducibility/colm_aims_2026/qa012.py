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
import io
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from . import schema

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures" / "qa012_item10"
_FIXTURE_BINDINGS_BASENAME = "bindings.json"
_FIXTURE_BINDINGS_SHA256 = (
    "c928ff4cbde99e3107c789d0e19cbcf88bdbe86b1dea39ab18b29f3c9d7dd3b0"
)
CANONICAL_AUTHORITY_RELPATH = "qa012_inventory_2026-08-22_rev3.json"
CANONICAL_AUTHORITY_PATH = _REPO_ROOT / CANONICAL_AUTHORITY_RELPATH
CANONICAL_AUTHORITY_SHA256 = (
    "bb692446ad07bea63b5fc6799d4c0b6474cc084076c87b2db7c2c2a9b7334303"
)
_AUTHORITY_KEYS = frozenset(
    {
        "manifest_type",
        "revision",
        "supersession_chain",
        "procedure",
        "conventions",
        "generated_at",
        "files_scanned",
        "parse_failures",
        "total_format_qa_hits",
        "hits_by_file",
        "verdict",
        "entries",
    }
)
AUTHORITY_SOURCE_PRONGS = {
    "d6_bundle": "d6_checksum_closure",
    "sibling_paper_exports": "historical_paper_exports",
    "modal_repo_paper_exports": "historical_paper_exports",
    "modal_staging_paper_exports": "historical_paper_exports",
    "repo_paper_exports": "historical_paper_exports",
    "item10_bundle": "source_export_bundles",
    "data_processed": "source_export_bundles",
    "git": "external_sidecars",
    "phase4_successor_artifact": "successor_suite_transcripts",
}
_AUTHORITY_SOURCE_COUNTS = {
    "d6_bundle": 4,
    "sibling_paper_exports": 12,
    "modal_repo_paper_exports": 11,
    "modal_staging_paper_exports": 11,
    "repo_paper_exports": 7,
    "item10_bundle": 12,
    "data_processed": 7,
    "git": 1,
    "phase4_successor_artifact": 2,
}
_AUTHORITY_SOURCE_RELATIVE_PREFIXES = {
    "item10_bundle": "CS321M/final_project/",
}
DROPBOX_CONTENT_BLOCK_BYTES = 4 * 1024 * 1024
REQUIRED_SCOPE_PRONGS = (
    "d6_checksum_closure",
    "historical_paper_exports",
    "source_export_bundles",
    "external_sidecars",
    "successor_suite_transcripts",
)
_SCANNED = "LOCATED_SCANNED"
_UNLOCATABLE = "UNLOCATABLE_ESCALATE"
_EMPTY = "LOCATED_EMPTY_ESCALATE"
_MANIFEST_KEYS = frozenset(
    {"schema_version", "result", "scope_prongs", "files", "inventory_sha256"}
)
_PRONG_KEYS = frozenset({"name", "status", "root_basename", "file_count"})
_FILE_KEYS = frozenset(
    {"scope_prong", "path", "size", "content_hash", "sha256", "hits"}
)
_HIT_KEYS = frozenset({"line", "pointer"})
MAX_QA_DIRECTORIES = 10_000
MAX_QA_FILES = 100_000
# The frozen 67-file authority contains two JSON documents larger than the
# package-wide 64-MiB artifact default and totals roughly 634 MiB.  QA-012 is a
# bounded corpus inventory, so it owns explicit per-file and aggregate limits
# large enough for that designated scope while remaining fail-closed.
MAX_QA_FILE_BYTES = 512 * 1024 * 1024
MAX_QA_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_QA_JSONL_ROWS = 100_000
MAX_QA_HITS = 100_000
# Traversal and emitted-identity limits are separate from the input byte cap.
# In particular, a compact JSON document can fan out into many Python objects,
# while long object keys can make its RFC-6901 identities much larger than the
# number of hits alone suggests.
MAX_QA_TRAVERSAL_NODES = 10_000_000
MAX_QA_POINTER_BYTES = 64 * 1024
MAX_QA_TOTAL_POINTER_BYTES = 64 * 1024 * 1024


def _fixture_excerpt_is_rejected(excerpt: bytes, excerpt_name: str) -> bool:
    """Require the exact historical incompatibility represented by a fixture."""
    try:
        loaded = schema.load_records_bytes(excerpt, excerpt_name)
        for record in loaded["records"]:
            schema.validate_record(record)
    except schema.RecordValidationError as exc:
        return "record missing opaque item_key (R-031)" in str(exc)
    except schema.ColmAimsError:
        return False
    return False


def _full_fixture_is_incompatible(data: bytes, fixture_name: str) -> bool:
    """Require every record in an exact hit-file fixture to fail R-031."""
    try:
        loaded = schema.load_records_bytes(data, fixture_name)
    except schema.ColmAimsError:
        return False
    records = loaded.get("records")
    if not isinstance(records, list) or not records:
        return False
    for record in records:
        try:
            schema.validate_record(record)
        except schema.RecordValidationError as exc:
            if "record missing opaque item_key (R-031)" not in str(exc):
                return False
        except schema.ColmAimsError:
            return False
        else:
            return False
    return True


def _authority_source(path: Any) -> str:
    """Return the portable source identifier from a rev3 inventory path."""
    if not isinstance(path, str) or not path or "\\" in path:
        raise schema.SchemaValidationError(
            "QA-012 authority entry path is not a portable source identity"
        )
    source, separator, relative = path.partition(":")
    if (
        not separator
        or source not in AUTHORITY_SOURCE_PRONGS
        or not relative.strip()
        or relative.lstrip().startswith("/")
        or bool(re.search(r"[A-Za-z]:[/\\]", path))
    ):
        raise schema.SchemaValidationError(
            "QA-012 authority entry names an unknown or absolute source"
        )
    return source


def _authority_prong_relative_path(path: str) -> tuple[str, str]:
    """Map one authority path to its frozen prong and exact portable relative path."""
    source = _authority_source(path)
    relative = path.partition(":")[2].strip()
    prefix = _AUTHORITY_SOURCE_RELATIVE_PREFIXES.get(source, "")
    if prefix:
        if not relative.startswith(prefix):
            raise schema.SchemaValidationError(
                "QA-012 authority path does not carry its frozen source prefix"
            )
        relative = relative[len(prefix) :]
    return AUTHORITY_SOURCE_PRONGS[source], relative


def load_authority_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the exact tracked rev3 scope authority, independent of location.

    The runtime path is intentionally not returned or serialized.  Only the
    raw-byte SHA-256 is durable authority, so a relocated byte-identical copy
    remains admissible while rev1 or any mutated/self-authored file fails.
    """
    authority_path = Path(path or CANONICAL_AUTHORITY_PATH).absolute()
    raw = schema.read_regular_file_bytes(authority_path)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CANONICAL_AUTHORITY_SHA256:
        raise schema.SchemaValidationError(
            "QA-012 authority bytes do not match the canonical rev3 SHA-256"
        )
    authority = schema.parse_json_bytes_strict(raw)
    if not isinstance(authority, dict) or set(authority) != _AUTHORITY_KEYS:
        raise schema.SchemaValidationError(
            "QA-012 rev3 authority has a non-closed top-level shape"
        )
    if (
        authority.get("manifest_type") != "qa012_format_qa_inventory"
        or authority.get("revision") != 3
        or authority.get("files_scanned") != 67
        or authority.get("parse_failures") != []
        or authority.get("total_format_qa_hits") != 4556
        or authority.get("verdict") != "HITS_PRESENT_NOT_VACUOUS"
        or not isinstance(authority.get("entries"), list)
        or len(authority["entries"]) != 67
    ):
        raise schema.SchemaValidationError(
            "QA-012 rev3 authority metadata does not encode the adjudicated scope"
        )
    source_counts = {source: 0 for source in AUTHORITY_SOURCE_PRONGS}
    total_hits = 0
    total_bytes = 0
    seen_paths: set[str] = set()
    for entry in authority["entries"]:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "size",
            "sha256",
            "dropbox_content_hash",
            "parse_ok",
            "format_qa_hits",
        }:
            raise schema.SchemaValidationError(
                "QA-012 rev3 authority entry has a non-closed shape"
            )
        source = _authority_source(entry["path"])
        if entry["path"] in seen_paths:
            raise schema.SchemaValidationError(
                "QA-012 rev3 authority contains a duplicate source identity"
            )
        seen_paths.add(entry["path"])
        source_counts[source] += 1
        if (
            not schema.is_real_int(entry["size"])
            or entry["size"] < 0
            or entry["size"] > MAX_QA_FILE_BYTES
            or not schema.is_sha256_hex(entry["sha256"])
            or not schema.is_sha256_hex(entry["dropbox_content_hash"])
            or entry["parse_ok"] is not True
            or not isinstance(entry["format_qa_hits"], list)
            or not all(isinstance(hit, str) and hit for hit in entry["format_qa_hits"])
        ):
            raise schema.SchemaValidationError(
                "QA-012 rev3 authority entry has invalid hashes or hit evidence"
            )
        total_bytes += entry["size"]
        if total_bytes > MAX_QA_TOTAL_BYTES:
            raise schema.SchemaValidationError(
                "QA-012 rev3 authority exceeds the aggregate byte limit"
            )
        total_hits += len(entry["format_qa_hits"])
    if source_counts != _AUTHORITY_SOURCE_COUNTS or total_hits != 4556:
        raise schema.SchemaValidationError(
            "QA-012 rev3 authority does not cover every frozen source/prong"
        )
    if set(AUTHORITY_SOURCE_PRONGS.values()) != set(REQUIRED_SCOPE_PRONGS):
        raise schema.SchemaValidationError(
            "QA-012 authority source map does not cover every required prong"
        )
    return authority


def dropbox_content_hash(data: bytes) -> str:
    """Dropbox 4-MiB-block SHA-256-of-SHA-256s content hash."""
    block_hashes = b"".join(
        hashlib.sha256(data[offset : offset + DROPBOX_CONTENT_BLOCK_BYTES]).digest()
        for offset in range(0, len(data), DROPBOX_CONTENT_BLOCK_BYTES)
    )
    return hashlib.sha256(block_hashes).hexdigest()


def _inventory_digest(scope_prongs: list[Any], files: list[Any]) -> str:
    payload = json.dumps(
        {"scope_prongs": scope_prongs, "files": files},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_inventory_manifest(manifest: Any) -> None:
    """Validate a closed, internally coherent QA-012 inventory manifest."""
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise schema.SchemaValidationError(
            "QA-012 manifest must have the exact closed top-level shape (R-072)"
        )
    schema.check_schema_version(manifest, "QA-012 manifest")
    scope_prongs = manifest["scope_prongs"]
    files = manifest["files"]
    if not isinstance(scope_prongs, list) or not isinstance(files, list):
        raise schema.SchemaValidationError(
            "QA-012 scope_prongs/files must be arrays (R-072)"
        )
    if len(files) > MAX_QA_FILES:
        raise schema.SchemaValidationError(
            "QA-012 manifest exceeds the aggregate file-count limit"
        )
    if len(scope_prongs) != len(REQUIRED_SCOPE_PRONGS):
        raise schema.SchemaValidationError(
            "QA-012 manifest must carry every frozen scope prong (R-072)"
        )

    prong_counts: dict[str, int] = {}
    for expected_name, prong in zip(REQUIRED_SCOPE_PRONGS, scope_prongs):
        if not isinstance(prong, dict) or set(prong) != _PRONG_KEYS:
            raise schema.SchemaValidationError(
                "QA-012 scope prong has a non-closed shape (R-072)"
            )
        if prong["name"] != expected_name:
            raise schema.SchemaValidationError(
                "QA-012 scope prongs are missing, reordered, or renamed (R-072)"
            )
        if prong["status"] not in {_SCANNED, _UNLOCATABLE, _EMPTY}:
            raise schema.SchemaValidationError(
                "QA-012 scope prong status is outside the closed vocabulary"
            )
        if not isinstance(prong["root_basename"], str) or not prong[
            "root_basename"
        ]:
            raise schema.SchemaValidationError(
                "QA-012 scope root_basename must be a non-empty string"
            )
        if (
            not schema.is_real_int(prong["file_count"])
            or prong["file_count"] < 0
        ):
            raise schema.SchemaValidationError(
                "QA-012 scope file_count must be a nonnegative real integer"
            )
        prong_counts[expected_name] = 0

    any_hits = False
    file_identities: list[tuple[str, str]] = []
    seen_file_identities: set[tuple[str, str]] = set()
    total_bytes = 0
    total_hits = 0
    total_pointer_bytes = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise schema.SchemaValidationError(
                "QA-012 file inventory entries must be objects (R-072)"
            )
        allowed = _FILE_KEYS | {"first_two_records_sha256"}
        if not _FILE_KEYS <= set(entry) or not set(entry) <= allowed:
            raise schema.SchemaValidationError(
                "QA-012 file inventory entry has a non-closed shape (R-072)"
            )
        prong = entry["scope_prong"]
        if prong not in prong_counts:
            raise schema.SchemaValidationError(
                "QA-012 file names an unknown scope prong (R-072)"
            )
        path = entry["path"]
        posix_path = PurePosixPath(path) if isinstance(path, str) else None
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or bool(re.match(r"^[A-Za-z]:", path))
            or posix_path is None
            or posix_path.is_absolute()
            or path != posix_path.as_posix()
            or any(part in {"", ".", ".."} for part in posix_path.parts)
            or posix_path.suffix not in {".json", ".jsonl"}
        ):
            raise schema.SchemaValidationError(
                "QA-012 file path must be a safe JSON/JSONL relative path"
            )
        identity = (prong, path)
        if identity in seen_file_identities:
            raise schema.SchemaValidationError(
                "QA-012 file inventory contains a duplicate prong/path"
            )
        seen_file_identities.add(identity)
        file_identities.append(identity)
        if not schema.is_real_int(entry["size"]) or entry["size"] < 0:
            raise schema.SchemaValidationError(
                "QA-012 file size must be a nonnegative real integer"
            )
        if entry["size"] > MAX_QA_FILE_BYTES:
            raise schema.SchemaValidationError(
                "QA-012 file exceeds the per-file byte limit"
            )
        total_bytes += entry["size"]
        if total_bytes > MAX_QA_TOTAL_BYTES:
            raise schema.SchemaValidationError(
                "QA-012 manifest exceeds the aggregate byte limit"
            )
        for digest_name in ("content_hash", "sha256"):
            if not schema.is_sha256_hex(entry[digest_name]):
                raise schema.SchemaValidationError(
                    f"QA-012 file {digest_name} must be lowercase SHA-256"
                )
        excerpt_digest = entry.get("first_two_records_sha256")
        if excerpt_digest is not None and not schema.is_sha256_hex(
            excerpt_digest
        ):
            raise schema.SchemaValidationError(
                "QA-012 first_two_records_sha256 must be lowercase SHA-256"
            )
        hits = entry["hits"]
        if not isinstance(hits, list):
            raise schema.SchemaValidationError("QA-012 hits must be an array")
        total_hits += len(hits)
        if total_hits > MAX_QA_HITS:
            raise schema.SchemaValidationError(
                "QA-012 manifest exceeds the aggregate hit limit"
            )
        for hit in hits:
            if not isinstance(hit, dict) or set(hit) != _HIT_KEYS:
                raise schema.SchemaValidationError(
                    "QA-012 hit identities must have exact line/pointer fields"
                )
            line = hit["line"]
            if Path(path).suffix == ".jsonl":
                if not schema.is_real_int(line) or line < 1:
                    raise schema.SchemaValidationError(
                        "QA-012 JSONL hit line identities are 1-based"
                    )
            elif line is not None:
                raise schema.SchemaValidationError(
                    "QA-012 JSON document hits must carry line=null"
                )
            pointer = hit["pointer"]
            if not isinstance(pointer, str) or not pointer.endswith("/format"):
                raise schema.SchemaValidationError(
                    "QA-012 hit pointer must identify a format field"
                )
            pointer_bytes = len(pointer.encode("utf-8"))
            if pointer_bytes > MAX_QA_POINTER_BYTES:
                raise schema.SchemaValidationError(
                    "QA-012 hit pointer exceeds the per-pointer byte limit"
                )
            total_pointer_bytes += pointer_bytes
            if total_pointer_bytes > MAX_QA_TOTAL_POINTER_BYTES:
                raise schema.SchemaValidationError(
                    "QA-012 manifest exceeds the aggregate pointer byte limit"
                )
        if hits != sorted(
            hits,
            key=lambda hit: (
                -1 if hit["line"] is None else hit["line"],
                hit["pointer"],
            ),
        ):
            raise schema.SchemaValidationError(
                "QA-012 hit identities must be in canonical line/pointer order"
            )
        any_hits = any_hits or bool(hits)
        prong_counts[prong] += 1

    if file_identities != sorted(file_identities):
        raise schema.SchemaValidationError(
            "QA-012 files must be in canonical prong/path order"
        )

    for prong in scope_prongs:
        observed_count = prong_counts[prong["name"]]
        if prong["file_count"] != observed_count:
            raise schema.SchemaValidationError(
                "QA-012 scope file_count does not match file inventory"
            )
        if prong["status"] == _SCANNED and observed_count == 0:
            raise schema.SchemaValidationError(
                "QA-012 scanned scope prong cannot be vacuous"
            )
        if prong["status"] != _SCANNED and observed_count != 0:
            raise schema.SchemaValidationError(
                "QA-012 unscanned scope prong cannot carry file evidence"
            )

    complete = all(prong["status"] == _SCANNED for prong in scope_prongs)
    expected_result = (
        "incomplete_scope" if not complete else "hits" if any_hits else "zero_hit"
    )
    if manifest["result"] != expected_result:
        raise schema.SchemaValidationError(
            "QA-012 result disagrees with its prong/file evidence (R-072)"
        )
    if manifest["inventory_sha256"] != _inventory_digest(scope_prongs, files):
        raise schema.SchemaValidationError(
            "QA-012 inventory_sha256 does not bind its prong/file evidence"
        )


def _escape_pointer_token(token: str) -> str:
    """RFC 6901 escaping: ``~`` -> ``~0`` then ``/`` -> ``~1``."""
    return token.replace("~", "~0").replace("/", "~1")


def detect_format_qa(
    document: Any,
    _prefix: str = "",
    *,
    max_hits: int | None = None,
    max_nodes: int | None = None,
    max_pointer_bytes: int | None = None,
    max_total_pointer_bytes: int | None = None,
) -> list[str]:
    """Detect exact ``format: "QA"`` keys with bounded RFC-6901 output.

    Traversal stores only one iterator and one raw path token per nesting
    level.  It never materializes a container's complete child list or builds
    pointer strings for non-hits, so a broad object cannot induce a second
    fanout-sized allocation merely from walking it.
    """
    max_hits = MAX_QA_HITS if max_hits is None else max_hits
    max_nodes = MAX_QA_TRAVERSAL_NODES if max_nodes is None else max_nodes
    max_pointer_bytes = (
        MAX_QA_POINTER_BYTES if max_pointer_bytes is None else max_pointer_bytes
    )
    max_total_pointer_bytes = (
        MAX_QA_TOTAL_POINTER_BYTES
        if max_total_pointer_bytes is None
        else max_total_pointer_bytes
    )
    limits = {
        "hit": max_hits,
        "node": max_nodes,
        "pointer": max_pointer_bytes,
        "aggregate pointer": max_total_pointer_bytes,
    }
    for label, limit in limits.items():
        if not schema.is_real_int(limit) or limit < 0:
            raise schema.ConfigSurfaceError(
                f"QA-012 {label} limit must be a nonnegative real integer"
            )
    if not isinstance(_prefix, str):
        raise schema.ConfigSurfaceError("QA-012 pointer prefix must be a string")
    if len(_prefix.encode("utf-8")) > max_pointer_bytes:
        raise schema.TypedIngressError(
            "QA-012 pointer prefix exceeds the admissible per-pointer byte"
            f" limit {max_pointer_bytes} (R-072/R-020)"
        )

    def children(value: Any):
        if isinstance(value, dict):
            return iter(value.items())
        if isinstance(value, list):
            return enumerate(value)
        return None

    pointers: list[str] = []
    emitted_pointer_bytes = 0
    visited_nodes = 1
    if visited_nodes > max_nodes:
        raise schema.TypedIngressError(
            "QA-012 document exceeds the admissible traversal-node limit"
            f" {max_nodes} (R-072/R-020)"
        )
    root_children = children(document)
    if root_children is None:
        return pointers

    # Each frame is (child iterator, owns one token in path_tokens).  Dict keys
    # remain references to the already-parsed document until a hit requires
    # escaping; list indexes are small integers.  This keeps traversal memory
    # proportional to depth, not fanout.
    stack: list[tuple[Any, bool]] = [(root_children, False)]
    path_tokens: list[str | int] = []
    while stack:
        child_iterator, owns_token = stack[-1]
        try:
            raw_token, child = next(child_iterator)
        except StopIteration:
            stack.pop()
            if owns_token:
                path_tokens.pop()
            continue

        visited_nodes += 1
        if visited_nodes > max_nodes:
            raise schema.TypedIngressError(
                "QA-012 document exceeds the admissible traversal-node limit"
                f" {max_nodes} (R-072/R-020)"
            )
        path_tokens.append(raw_token)
        if (
            isinstance(raw_token, str)
            and raw_token == "format"
            and isinstance(child, str)
            and child == "QA"
        ):
            suffix = "".join(
                f"/{_escape_pointer_token(str(token))}" for token in path_tokens
            )
            pointer = _prefix + suffix
            pointer_bytes = len(pointer.encode("utf-8"))
            if pointer_bytes > max_pointer_bytes:
                raise schema.TypedIngressError(
                    "QA-012 hit pointer exceeds the admissible per-pointer"
                    f" byte limit {max_pointer_bytes} (R-072/R-020)"
                )
            if emitted_pointer_bytes + pointer_bytes > max_total_pointer_bytes:
                raise schema.TypedIngressError(
                    "QA-012 document exceeds the admissible aggregate pointer"
                    f" byte limit {max_total_pointer_bytes} (R-072/R-020)"
                )
            pointers.append(pointer)
            emitted_pointer_bytes += pointer_bytes
            if len(pointers) > max_hits:
                raise schema.TypedIngressError(
                    "QA-012 document exceeds the admissible hit limit"
                    f" {max_hits} (R-072/R-020)"
                )

        nested_children = children(child)
        if nested_children is None:
            path_tokens.pop()
        else:
            stack.append((nested_children, True))
    return pointers


def _scan_file(path: Path, root: Path, scope_prong: str) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    data = schema.read_regular_file_bytes(
        path,
        tree_root=root,
        max_bytes=MAX_QA_FILE_BYTES,
    )
    hits: list[dict[str, Any]] = []
    emitted_pointer_bytes = 0
    if path.suffix == ".jsonl":
        try:
            text = data.decode("utf-8", errors="strict") if data else ""
        except UnicodeDecodeError as exc:
            raise schema.TypedIngressError(
                f"{rel}: invalid UTF-8 bytes at byte offset {exc.start}"
                " (R-020)"
            ) from exc
        for lineno, line in enumerate(io.StringIO(text), start=1):
            if lineno > MAX_QA_JSONL_ROWS:
                raise schema.TypedIngressError(
                    f"{rel}: exceeds QA-012 JSONL row limit"
                    f" {MAX_QA_JSONL_ROWS} (R-072/R-020)"
                )
            if not line.strip():
                continue
            obj = schema.parse_json_text_strict(line, f"{rel}: line {lineno}")
            pointers = detect_format_qa(
                obj,
                max_hits=MAX_QA_HITS - len(hits),
                max_total_pointer_bytes=(
                    MAX_QA_TOTAL_POINTER_BYTES - emitted_pointer_bytes
                ),
            )
            emitted_pointer_bytes += sum(
                len(pointer.encode("utf-8")) for pointer in pointers
            )
            hits.extend(
                {"line": lineno, "pointer": pointer} for pointer in pointers
            )
            if len(hits) > MAX_QA_HITS:
                raise schema.TypedIngressError(
                    f"{rel}: exceeds QA-012 hit limit {MAX_QA_HITS}"
                    " (R-072/R-020)"
                )
    else:
        try:
            obj = schema.parse_json_bytes_strict(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise schema.TypedIngressError(
                f"{rel}: malformed JSON: {exc} (R-020)"
            ) from exc
        hits.extend(
            {"line": None, "pointer": pointer}
            for pointer in detect_format_qa(obj, max_hits=MAX_QA_HITS)
        )
        if len(hits) > MAX_QA_HITS:
            raise schema.TypedIngressError(
                f"{rel}: exceeds QA-012 hit limit {MAX_QA_HITS}"
                " (R-072/R-020)"
            )
    entry = {
        "scope_prong": scope_prong,
        "path": rel,
        "size": len(data),
        "content_hash": dropbox_content_hash(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "hits": sorted(
            hits,
            key=lambda hit: (
                -1 if hit["line"] is None else hit["line"],
                hit["pointer"],
            ),
        ),
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
        validate_inventory_manifest(manifest)
        bindings_raw = schema.read_regular_file_bytes(
            root / _FIXTURE_BINDINGS_BASENAME, tree_root=root
        )
        if hashlib.sha256(bindings_raw).hexdigest() != _FIXTURE_BINDINGS_SHA256:
            return False
        bindings = schema.parse_json_bytes_strict(bindings_raw)
        if not isinstance(bindings, dict):
            return False
        schema.check_schema_version(bindings, "QA-012 fixture bindings")
        bound_files = bindings.get("files")
        if not isinstance(bound_files, dict) or not bound_files:
            return False

        scope_prongs = manifest.get("scope_prongs")
        if (
            manifest.get("result") != "hits"
            or not isinstance(scope_prongs, list)
            or [prong.get("name") for prong in scope_prongs if isinstance(prong, dict)]
            != list(REQUIRED_SCOPE_PRONGS)
            or any(
                not isinstance(prong, dict) or prong.get("status") != _SCANNED
                for prong in scope_prongs
            )
        ):
            return False
        hit_entries = [
            entry
            for entry in manifest.get("files", [])
            if isinstance(entry, dict) and entry.get("hits")
        ]
        by_identity: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in hit_entries:
            identity = (entry.get("scope_prong"), entry.get("path"))
            if not all(isinstance(value, str) and value for value in identity):
                return False
            if identity in by_identity:
                return False
            by_identity[identity] = entry
        bound_identities = {
            (value.get("scope_prong"), value.get("relative_path"))
            for value in bound_files.values()
            if isinstance(value, dict)
        }
        if set(by_identity) != bound_identities:
            return False

        for basename, binding_value in bound_files.items():
            if not isinstance(binding_value, dict):
                return False
            full_name = binding_value.get("full_fixture")
            excerpt_name = binding_value.get("excerpt_fixture")
            if (
                not isinstance(full_name, str)
                or not full_name
                or not isinstance(excerpt_name, str)
                or not excerpt_name
            ):
                return False
            full_fixture = schema.read_regular_file_bytes(
                root / full_name, tree_root=root
            )
            excerpt = schema.read_regular_file_bytes(
                root / excerpt_name, tree_root=root
            )
            excerpt_sha256 = hashlib.sha256(excerpt).hexdigest()
            if excerpt_sha256 != binding_value.get("excerpt_sha256"):
                return False
            if len(excerpt.splitlines()) != binding_value.get("excerpt_lines"):
                return False
            if b"\n".join(full_fixture.splitlines()[:2]) + b"\n" != excerpt:
                return False
            if (
                hashlib.sha256(full_fixture).hexdigest()
                != binding_value.get("full_file_sha256")
                or len(full_fixture) != binding_value.get("full_file_size")
                or dropbox_content_hash(full_fixture)
                != binding_value.get("dropbox_content_hash")
                or not _full_fixture_is_incompatible(full_fixture, full_name)
            ):
                return False
            if not _fixture_excerpt_is_rejected(excerpt, excerpt_name):
                # The historical ``format: QA`` rows are compatibility
                # fixtures only.  If they ever become admissible v2 records,
                # they could substitute for the required semantic block.
                return False

            identity = (
                binding_value.get("scope_prong"),
                binding_value.get("relative_path"),
            )
            entry = by_identity.get(identity)
            if entry is None:
                return False
            if entry.get("sha256") != binding_value.get("full_file_sha256"):
                return False
            if entry.get("size") != binding_value.get("full_file_size"):
                return False
            if entry.get("content_hash") != binding_value.get(
                "dropbox_content_hash"
            ):
                return False
            hits = entry.get("hits")
            if not isinstance(hits, list) or len(hits) != binding_value.get(
                "hit_count"
            ):
                return False
            hits_sha256 = hashlib.sha256(
                json.dumps(
                    hits, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if hits_sha256 != binding_value.get("hits_sha256"):
                return False
            if entry.get("first_two_records_sha256") != excerpt_sha256:
                return False
    except (OSError, KeyError, TypeError, schema.ColmAimsError):
        return False
    return True


def authority_hit_fixtures_verified(
    authority: dict[str, Any], fixtures_root: Path | None = None
) -> bool:
    """Bind every canonical rev3 hit file to the committed compatibility fixtures."""
    root = Path(fixtures_root or _DEFAULT_FIXTURES_ROOT)
    try:
        # The caller may supply a relocated copy, but never a caller-authored
        # parsed object: reload the tracked authority and require exact equality.
        if authority != load_authority_manifest():
            return False
        bindings_raw = schema.read_regular_file_bytes(
            root / _FIXTURE_BINDINGS_BASENAME, tree_root=root
        )
        if hashlib.sha256(bindings_raw).hexdigest() != _FIXTURE_BINDINGS_SHA256:
            return False
        bindings = schema.parse_json_bytes_strict(bindings_raw)
        if not isinstance(bindings, dict):
            return False
        schema.check_schema_version(bindings, "QA-012 fixture bindings")
        bound_files = bindings.get("files")
        if not isinstance(bound_files, dict) or not bound_files:
            return False
        hit_entries = [entry for entry in authority["entries"] if entry["format_qa_hits"]]
        if len(hit_entries) != len(bound_files):
            return False
        matched_paths: set[str] = set()
        for binding in bound_files.values():
            if not isinstance(binding, dict):
                return False
            relative_path = binding.get("relative_path")
            scope_prong = binding.get("scope_prong")
            matches = [
                entry
                for entry in hit_entries
                if isinstance(relative_path, str)
                and isinstance(scope_prong, str)
                and _authority_prong_relative_path(entry["path"])
                == (scope_prong, relative_path)
            ]
            if len(matches) != 1:
                return False
            entry = matches[0]
            matched_paths.add(entry["path"])
            full_name = binding.get("full_fixture")
            excerpt_name = binding.get("excerpt_fixture")
            if (
                not isinstance(full_name, str)
                or not full_name
                or not isinstance(excerpt_name, str)
                or not excerpt_name
            ):
                return False
            full_fixture = schema.read_regular_file_bytes(
                root / full_name, tree_root=root
            )
            excerpt = schema.read_regular_file_bytes(
                root / excerpt_name, tree_root=root
            )
            if (
                hashlib.sha256(excerpt).hexdigest() != binding.get("excerpt_sha256")
                or len(excerpt.splitlines()) != binding.get("excerpt_lines")
                or b"\n".join(full_fixture.splitlines()[:2]) + b"\n" != excerpt
                or hashlib.sha256(full_fixture).hexdigest()
                != binding.get("full_file_sha256")
                or len(full_fixture) != binding.get("full_file_size")
                or dropbox_content_hash(full_fixture)
                != binding.get("dropbox_content_hash")
                or entry["sha256"] != binding.get("full_file_sha256")
                or entry["size"] != binding.get("full_file_size")
                or entry["dropbox_content_hash"]
                != binding.get("dropbox_content_hash")
                or len(entry["format_qa_hits"]) != binding.get("hit_count")
            ):
                return False
            normalized_hits: list[dict[str, Any]] = []
            for hit in entry["format_qa_hits"]:
                match = re.fullmatch(r"line ([1-9][0-9]*): (/.*/?format|/format)", hit)
                if match is None:
                    return False
                normalized_hits.append(
                    {"line": int(match.group(1)), "pointer": match.group(2)}
                )
            hits_sha256 = hashlib.sha256(
                json.dumps(
                    normalized_hits, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if hits_sha256 != binding.get("hits_sha256"):
                return False
            if not _full_fixture_is_incompatible(full_fixture, full_name):
                return False
            if not _fixture_excerpt_is_rejected(excerpt, excerpt_name):
                return False
        return matched_paths == {entry["path"] for entry in hit_entries}
    except (OSError, KeyError, TypeError, schema.ColmAimsError):
        return False


def _candidate_files(root: Path) -> list[Path]:
    """Enumerate JSON inputs without following directory aliases."""
    schema.stable_directory_chain(root, root)
    candidates: list[Path] = []
    directory_count = 0
    pending = [root]
    while pending:
        current = pending.pop()
        directory_count += 1
        if directory_count > MAX_QA_DIRECTORIES:
            raise schema.TypedIngressError(
                "QA-012 corpus exceeds the aggregate directory-count limit"
                f" {MAX_QA_DIRECTORIES} (R-072/R-020)"
            )
        schema.stable_directory_chain(current, root)
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_symlink() or schema.is_filesystem_link(path):
                        raise schema.TypedIngressError(
                            "QA-012 corpus contains a symlink or reparse"
                            " point (R-072/R-013)"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False) and path.suffix in (
                        ".json",
                        ".jsonl",
                    ):
                        candidates.append(path)
                    if len(candidates) > MAX_QA_FILES:
                        raise schema.TypedIngressError(
                            "QA-012 corpus exceeds the aggregate file-count"
                            f" limit {MAX_QA_FILES} (R-072/R-020)"
                        )
                    if len(pending) + directory_count > MAX_QA_DIRECTORIES:
                        raise schema.TypedIngressError(
                            "QA-012 corpus exceeds the aggregate directory-count"
                            f" limit {MAX_QA_DIRECTORIES} (R-072/R-020)"
                        )
        except schema.ColmAimsError:
            raise
        except OSError as exc:
            raise schema.TypedIngressError(
                "QA-012 corpus traversal failed"
                f" ({exc.__class__.__name__}) (R-072/R-020)"
            ) from exc
    return sorted(candidates, key=lambda path: path.relative_to(root).as_posix())


def _candidate_file_info(path: Path, root: Path) -> os.stat_result:
    """Return no-follow metadata for one ordinary bounded candidate."""
    rel = path.relative_to(root).as_posix()
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise schema.TypedIngressError(
            f"{rel}: unreadable QA-012 candidate"
            f" ({exc.__class__.__name__}) (R-072/R-020)"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise schema.TypedIngressError(
            f"{rel}: QA-012 candidate is a symlink or reparse point"
            " (R-072/R-013)"
        )
    if not stat.S_ISREG(info.st_mode):
        raise schema.TypedIngressError(
            f"{rel}: QA-012 candidate is not a regular file (R-072/R-020)"
        )
    if int(info.st_size) > MAX_QA_FILE_BYTES:
        raise schema.TypedIngressError(
            f"{rel}: exceeds QA-012 per-file byte limit"
            f" {MAX_QA_FILE_BYTES} (R-072/R-020)"
        )
    return info


def _candidate_identity(info: os.stat_result) -> tuple[int, int, int]:
    """Return the same stable-enough no-follow identity used at ingress."""
    return (int(info.st_dev), int(info.st_ino), int(info.st_mode))


def _candidate_file_size(path: Path, root: Path) -> int:
    """Return one candidate size without reading or following an alias."""
    return int(_candidate_file_info(path, root).st_size)


def _verify_candidate_unchanged(
    path: Path,
    root: Path,
    expected_identity: tuple[int, int, int],
    entry: dict[str, Any],
) -> None:
    """Stream one candidate again and reject replacement or content drift."""
    rel = path.relative_to(root).as_posix()
    parent_chain = schema.stable_directory_chain(path.parent, root)
    before = _candidate_file_info(path, root)
    if _candidate_identity(before) != expected_identity:
        raise schema.TypedIngressError(
            f"{rel}: QA-012 candidate identity changed during inventory"
            " (R-072/R-020)"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise schema.TypedIngressError(
            f"{rel}: unreadable QA-012 candidate"
            f" ({exc.__class__.__name__}) (R-072/R-020)"
        ) from exc
    sha256 = hashlib.sha256()
    dropbox_outer = hashlib.sha256()
    dropbox_block = bytearray()
    observed_size = 0
    try:
        try:
            opened = os.fstat(fd)
            if _candidate_identity(opened) != expected_identity:
                raise schema.TypedIngressError(
                    f"{rel}: QA-012 candidate identity changed during inventory"
                    " (R-072/R-020)"
                )
            if hasattr(os, "O_NONBLOCK") and hasattr(os, "get_blocking"):
                os.set_blocking(fd, True)
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                observed_size += len(chunk)
                if observed_size > MAX_QA_FILE_BYTES:
                    raise schema.TypedIngressError(
                        f"{rel}: exceeds QA-012 per-file byte limit"
                        f" {MAX_QA_FILE_BYTES} (R-072/R-020)"
                    )
                sha256.update(chunk)
                dropbox_block.extend(chunk)
                while len(dropbox_block) >= DROPBOX_CONTENT_BLOCK_BYTES:
                    block = bytes(dropbox_block[:DROPBOX_CONTENT_BLOCK_BYTES])
                    del dropbox_block[:DROPBOX_CONTENT_BLOCK_BYTES]
                    dropbox_outer.update(hashlib.sha256(block).digest())
            if dropbox_block:
                dropbox_outer.update(hashlib.sha256(dropbox_block).digest())
        except schema.ColmAimsError:
            raise
        except OSError as exc:
            raise schema.TypedIngressError(
                f"{rel}: QA-012 candidate read failed"
                f" ({exc.__class__.__name__}) (R-072/R-020)"
            ) from exc
    finally:
        os.close(fd)
    after = _candidate_file_info(path, root)
    after_chain = schema.stable_directory_chain(path.parent, root)
    if (
        _candidate_identity(after) != expected_identity
        or after_chain != parent_chain
    ):
        raise schema.TypedIngressError(
            f"{rel}: QA-012 candidate identity changed during inventory"
            " (R-072/R-020)"
        )
    if (
        observed_size != entry["size"]
        or sha256.hexdigest() != entry["sha256"]
        or dropbox_outer.hexdigest() != entry["content_hash"]
    ):
        raise schema.TypedIngressError(
            f"{rel}: QA-012 candidate content changed during inventory"
            " (R-072/R-020)"
        )


def build_inventory_manifest(roots: dict[str, Path]) -> dict[str, Any]:
    """Execute the QA-012 inventory over the given corpus roots (R-072).

    Strict parse failures are typed ingress errors (the inventory never
    silently skips an unreadable file). The manifest hash is deterministic
    over the sorted per-file entries.
    """
    if not isinstance(roots, dict) or set(roots) != set(REQUIRED_SCOPE_PRONGS):
        observed = sorted(roots) if isinstance(roots, dict) else []
        raise schema.ConfigSurfaceError(
            "QA-012 scope must declare exactly the frozen prongs"
            f" {list(REQUIRED_SCOPE_PRONGS)!r}; observed={observed!r} (R-072)"
        )
    lexical_roots: dict[str, Path] = {}
    resolved_roots: dict[str, Path] = {}
    for scope_prong in REQUIRED_SCOPE_PRONGS:
        lexical_root = Path(roots[scope_prong]).absolute()
        if lexical_root.exists():
            schema.stable_directory_chain(lexical_root, lexical_root)
        root = lexical_root.resolve(strict=False)
        for other_prong, other_root in resolved_roots.items():
            if (
                root == other_root
                or root in other_root.parents
                or other_root in root.parents
            ):
                raise schema.ConfigSurfaceError(
                    "QA-012 scope roots must be distinct and non-overlapping;"
                    f" {scope_prong!r} overlaps {other_prong!r} (R-072)"
                )
        resolved_roots[scope_prong] = root
        lexical_roots[scope_prong] = lexical_root

    files: list[dict[str, Any]] = []
    scope_prongs: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    preflight_total_bytes = 0
    total_hits = 0
    candidate_sets: dict[str, list[Path]] = {}
    candidate_identities: dict[tuple[str, str], tuple[int, int, int]] = {}
    entries_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for scope_prong in REQUIRED_SCOPE_PRONGS:
        root = lexical_roots[scope_prong]
        if (
            not root.exists()
            or not root.is_dir()
            or schema.is_filesystem_link(root)
        ):
            scope_prongs.append(
                {
                    "name": scope_prong,
                    "status": _UNLOCATABLE,
                    "root_basename": root.name or "<root>",
                    "file_count": 0,
                }
            )
            continue
        candidates = _candidate_files(root)
        if not candidates:
            scope_prongs.append(
                {
                    "name": scope_prong,
                    "status": _EMPTY,
                    "root_basename": root.name or "<root>",
                    "file_count": 0,
                }
            )
            continue
        for path in candidates:
            preflight_total_bytes += _candidate_file_size(path, root)
            rel = path.relative_to(root).as_posix()
            candidate_identities[(scope_prong, rel)] = _candidate_identity(
                _candidate_file_info(path, root)
            )
            if preflight_total_bytes > MAX_QA_TOTAL_BYTES:
                raise schema.TypedIngressError(
                    "QA-012 inventory exceeds aggregate byte limit"
                    f" {MAX_QA_TOTAL_BYTES} (R-072/R-020)"
                )
        candidate_sets[scope_prong] = candidates
        scope_prongs.append(
            {
                "name": scope_prong,
                "status": _SCANNED,
                "root_basename": root.name or "<root>",
                "file_count": len(candidates),
            }
        )
        for path in candidates:
            entry = _scan_file(path, root, scope_prong)
            files.append(entry)
            entries_by_identity[(scope_prong, entry["path"])] = entry
            total_files += 1
            total_bytes += entry["size"]
            total_hits += len(entry["hits"])
            if total_files > MAX_QA_FILES:
                raise schema.TypedIngressError(
                    "QA-012 inventory exceeds aggregate file-count limit"
                    f" {MAX_QA_FILES} (R-072/R-020)"
                )
            if total_bytes > MAX_QA_TOTAL_BYTES:
                raise schema.TypedIngressError(
                    "QA-012 inventory exceeds aggregate byte limit"
                    f" {MAX_QA_TOTAL_BYTES} (R-072/R-020)"
                )
            if total_hits > MAX_QA_HITS:
                raise schema.TypedIngressError(
                    "QA-012 inventory exceeds aggregate hit limit"
                    f" {MAX_QA_HITS} (R-072/R-020)"
                )
    # Re-enumerate and re-read every scanned prong only after all scans finish.
    # A same-name replacement or in-place edit is therefore not hidden by the
    # membership-only comparison, including drift in an earlier prong while a
    # later prong is being scanned.
    for scope_prong, candidates in candidate_sets.items():
        root = lexical_roots[scope_prong]
        after = _candidate_files(root)
        candidate_rels = [path.relative_to(root).as_posix() for path in candidates]
        after_rels = [path.relative_to(root).as_posix() for path in after]
        if after_rels != candidate_rels:
            raise schema.TypedIngressError(
                "QA-012 corpus membership changed during inventory"
                " (R-072/R-020)"
            )
        for path, rel in zip(after, after_rels):
            identity = (scope_prong, rel)
            _verify_candidate_unchanged(
                path,
                root,
                candidate_identities[identity],
                entries_by_identity[identity],
            )
    files.sort(key=lambda entry: (entry["scope_prong"], entry["path"]))
    any_hits = any(entry["hits"] for entry in files)
    complete_scope = all(prong["status"] == _SCANNED for prong in scope_prongs)
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "result": (
            "incomplete_scope"
            if not complete_scope
            else "hits" if any_hits else "zero_hit"
        ),
        "scope_prongs": scope_prongs,
        "files": files,
        "inventory_sha256": _inventory_digest(scope_prongs, files),
    }
