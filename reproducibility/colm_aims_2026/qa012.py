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
from pathlib import Path, PurePosixPath
from typing import Any

from . import schema

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures" / "qa012_item10"
_FIXTURE_BINDINGS_BASENAME = "bindings.json"
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
MAX_QA_BYTES = 512 * 1024 * 1024
MAX_QA_JSONL_ROWS = 100_000
MAX_QA_HITS = 100_000


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
    total_bytes = 0
    total_hits = 0
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
        if identity in file_identities:
            raise schema.SchemaValidationError(
                "QA-012 file inventory contains a duplicate prong/path"
            )
        file_identities.append(identity)
        if not schema.is_real_int(entry["size"]) or entry["size"] < 0:
            raise schema.SchemaValidationError(
                "QA-012 file size must be a nonnegative real integer"
            )
        total_bytes += entry["size"]
        if total_bytes > MAX_QA_BYTES:
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
    document: Any, _prefix: str = "", *, max_hits: int = MAX_QA_HITS
) -> list[str]:
    """Detect exact ``format: "QA"`` keys with bounded RFC-6901 output."""
    pointers: list[str] = []
    stack: list[tuple[Any, str]] = [(document, _prefix)]
    while stack:
        value, prefix = stack.pop()
        if isinstance(value, dict):
            children = []
            for key, child in value.items():
                pointer = f"{prefix}/{_escape_pointer_token(str(key))}"
                if key == "format" and isinstance(child, str) and child == "QA":
                    pointers.append(pointer)
                    if len(pointers) > max_hits:
                        raise schema.TypedIngressError(
                            "QA-012 document exceeds the admissible hit limit"
                            f" {max_hits} (R-072/R-020)"
                        )
                children.append((child, pointer))
            stack.extend(reversed(children))
        elif isinstance(value, list):
            stack.extend(
                (child, f"{prefix}/{index}")
                for index, child in reversed(list(enumerate(value)))
            )
    return pointers


def _scan_file(path: Path, root: Path, scope_prong: str) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    data = schema.read_regular_file_bytes(path, tree_root=root)
    hits: list[dict[str, Any]] = []
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
            hits.extend(
                {"line": lineno, "pointer": pointer}
                for pointer in detect_format_qa(
                    obj, max_hits=MAX_QA_HITS - len(hits)
                )
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
            try:
                loaded = schema.load_records_bytes(excerpt, excerpt_name)
                for record in loaded["records"]:
                    schema.validate_record(record)
            except schema.ColmAimsError:
                pass
            else:
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
            if root == other_root or root in other_root.parents or other_root in root.parents:
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
    total_hits = 0
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
            total_files += 1
            total_bytes += entry["size"]
            total_hits += len(entry["hits"])
            if total_files > MAX_QA_FILES or total_bytes > MAX_QA_BYTES:
                raise schema.TypedIngressError(
                    "QA-012 inventory exceeds aggregate file/byte limits"
                    f" ({MAX_QA_FILES} files, {MAX_QA_BYTES} bytes; R-072/R-020)"
                )
            if total_hits > MAX_QA_HITS:
                raise schema.TypedIngressError(
                    "QA-012 inventory exceeds aggregate hit limit"
                    f" {MAX_QA_HITS} (R-072/R-020)"
                )
        after = _candidate_files(root)
        if [path.relative_to(root) for path in after] != [
            path.relative_to(root) for path in candidates
        ]:
            raise schema.TypedIngressError(
                "QA-012 corpus membership changed during inventory"
                " (R-072/R-020)"
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
