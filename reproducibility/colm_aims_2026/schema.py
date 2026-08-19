"""Strict schema-versioned constructed-reference profile: types, ingress, writer.

Spec rules owned here: R-001..R-004, R-016 (writer side), R-020, R-029,
R-031, R-032, R-039 (publish side).
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from scripts.stopdff_v5 import fileio


class ColmAimsError(Exception):
    """Base class for every typed error raised by this namespace."""


class SchemaValidationError(ColmAimsError):
    """Strict-profile semantic/schema violation (R-001..R-003, R-029, R-032)."""


class RecordValidationError(SchemaValidationError):
    """Per-item record violates the non-reversible record contract (R-031)."""


class TypedIngressError(ColmAimsError):
    """Artifact bytes failed typed validation at the load boundary (R-020)."""


class EmptyEvaluationError(ColmAimsError):
    """Explicitly empty evaluation population refused (R-006, R-012)."""


# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
# Inclusive supported schema_version range for typed ingress (R-020).
SUPPORTED_SCHEMA_VERSION_MIN = 1
SUPPORTED_SCHEMA_VERSION_MAX = 1
VERIFIER_REVISION = "reproducibility.colm_aims_2026 r1"

STRICT_PROFILE_ID = "colm_aims_constructed_reference_v1"
# R-002: reserved identifier for genuinely observed future studies. The
# constructed-reference validator never accepts it; no code path converts
# one profile into the other.
RESERVED_OBSERVED_PROFILE_ID = "colm_aims_observed_paired_v1"

# R-032: pinned maximum admissible numerical tolerance. DECISION: 1e-3 —
# joint-class rates must sum to 1; anything looser is meaningless.
MAX_ADMISSIBLE_TOLERANCE = 1e-3

# R-001: the pinned semantic block, verbatim from handoff §8 (AP-031).
SEMANTIC_BLOCK: dict[str, Any] = {
    "trajectory_source": "constructed_reference",
    "observed_open_ended": False,
    "observed_open_ended_answers": False,
    "observed_open_ended_stopping_actions": False,
    "pairing_unit": "matched_item_prefix_grid",
    "pairing_is_observed_sessions": False,
    "supports": "reference_sensitivity_diagnostic",
    "does_not_support": "actual_decision_preservation_or_format_effect",
}

# R-008: pinned item-key derivation, re-derivable by third parties.
ITEM_KEY_DERIVATION: dict[str, Any] = {
    "hash": "sha256",
    "text_normalization": "NFC",
    "prefix": "itm-",
    "hex_digits": 16,
}

# R-029: contribution axes; `none` is an explicit value, never absent.
LLM_INVOLVEMENT_AXES = (
    "reference_construction",
    "data_plot_creation",
    "evaluation",
)

# R-003: per-arm identification fields.
ARM_REQUIRED_FIELDS = (
    "construction",
    "cardinality",
    "selector",
    "scorer",
    "candidate_pool_role",
    "correctness_assignment",
    "calibration_role",
    "continuation_role",
    "seed_contract",
    "reporting_eligibility",
)
ARM_CARDINALITIES = ("scalar", "k_way")

# R-007/R-008: enumerated exclusion reasons; missing reasons are recorded
# UNKNOWN_NOT_INFERRED, never guessed.
EXCLUSION_REASONS = frozenset(
    {
        "MALFORMED_STOP",
        "MISSING_STOP",
        "GRID_MISMATCH",
        "UNKNOWN_NOT_INFERRED",
    }
)

PROFILE_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "profile_id",
        "semantic",
        "llm_involvement",
        "numerical_tolerance",
        "item_key_derivation",
        "arms",
        "provenance",
        "cells",
    }
)

CELL_REQUIRED_KEYS = (
    "cell_id",
    "estimand",
    "estimand_digest",
    "counts",
    "rates",
    "timing_summary_finite_only",
    "timing_summary_sentinel_coded_historical",
    "complete_pair_keys",
    "excluded_keys",
    "pairing_population_keyset_sha256",
)

# R-015: the recorded interval identity (everything but the interval itself);
# a cell that carries an interval must also carry its `ci`.
INTERVAL_IDENTITY_KEYS = (
    "procedure",
    "draw_count",
    "resampling_seeds",
    "statistic",
)
INTERVAL_REQUIRED_KEYS = INTERVAL_IDENTITY_KEYS + ("ci",)

# R-031: the enumerated identifier allowlist — the ONLY string-valued field
# a per-item record may carry is its opaque item key; every other allowed
# field is numeric, boolean, or an enumerated categorical.
RECORD_IDENTIFIER_FIELDS = frozenset({"item_key"})
RECORD_NUMERIC_FIELDS = frozenset(
    {
        "trajectory_horizon",
        "mc_trajectory_horizon",
        "ref_trajectory_horizon",
        "mc_stop_step",
        "ref_stop_step",
    }
)
RECORD_BOOL_FIELDS = frozenset({"excluded"})
RECORD_CATEGORICAL_FIELDS = frozenset({"exclusion_reason"})
RECORD_CATEGORICAL_LIST_FIELDS = frozenset({"secondary_diagnostics"})
RECORD_ALLOWED_FIELDS = (
    RECORD_IDENTIFIER_FIELDS
    | RECORD_NUMERIC_FIELDS
    | RECORD_BOOL_FIELDS
    | RECORD_CATEGORICAL_FIELDS
    | RECORD_CATEGORICAL_LIST_FIELDS
)


# ---------------------------------------------------------------------------
# Shared value predicates (used across this namespace's validators)
# ---------------------------------------------------------------------------

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def is_number(value: Any) -> bool:
    """True for a real int/float; ``bool`` never counts as a number here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_sha256_hex(value: Any) -> bool:
    """True for a full-length lowercase sha256 hex digest."""
    return isinstance(value, str) and _SHA256_HEX_RE.fullmatch(value) is not None


def is_commit_sha(value: Any) -> bool:
    """True for a full-length 40-hex commit SHA.

    Short hashes, tags, and branch names never qualify — they are
    reassignable and cannot pin an immutable source identity (R-012/R-013).
    """
    return isinstance(value, str) and _COMMIT_SHA_RE.fullmatch(value) is not None


def is_admissible_tolerance(value: Any) -> bool:
    """R-032: a declared tolerance is admissible when it is a real number in
    ``(0, MAX_ADMISSIBLE_TOLERANCE]``; non-finite values never qualify."""
    return is_number(value) and 0 < float(value) <= MAX_ADMISSIBLE_TOLERANCE


def is_path_component(value: Any) -> bool:
    """True for a non-empty single path component (no separators/traversal)."""
    return isinstance(value, str) and bool(value) and Path(value).name == value


def resolves_inside(path: Path, root: Path) -> bool:
    """True when ``path`` resolves to ``root`` itself or anything beneath it.

    Containment decisions use fully resolved, symlink-free paths
    (R-013 expectations containment, R-036 receipt placement).
    """
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    return resolved == root_resolved or root_resolved in resolved.parents


def canonical_estimand_digest(estimand: dict[str, Any]) -> str:
    """R-011 pinned digest: sha256 over canonical compact JSON of the block."""
    payload = json.dumps(estimand, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_tolerance(tolerance: Any, where: str) -> None:
    """R-032: declared tolerance must be finite, positive, and at most the
    pinned MAX_ADMISSIBLE_TOLERANCE."""
    if not is_number(tolerance) or not math.isfinite(float(tolerance)):
        raise SchemaValidationError(
            f"{where}: declared numerical_tolerance must be a finite number"
        )
    if not is_admissible_tolerance(tolerance):
        raise SchemaValidationError(
            f"{where}: declared numerical_tolerance {tolerance!r} outside the"
            f" admissible range (0, {MAX_ADMISSIBLE_TOLERANCE}] —"
            " oversized tolerance fails validation (R-032)"
        )


def _validate_semantic_block(semantic: Any) -> None:
    """R-001: exact pinned key set and values; unknown/renamed/altered fail."""
    if not isinstance(semantic, dict):
        raise SchemaValidationError("semantic block must be an object (R-001)")
    expected_keys = set(SEMANTIC_BLOCK)
    got_keys = set(semantic)
    missing = sorted(expected_keys - got_keys)
    if missing:
        raise SchemaValidationError(
            f"semantic block missing pinned field(s): {missing} (R-001)"
        )
    unknown = sorted(got_keys - expected_keys)
    if unknown:
        raise SchemaValidationError(
            f"semantic block carries unknown field(s): {unknown} (R-001)"
        )
    for key, pinned in SEMANTIC_BLOCK.items():
        if semantic[key] != pinned:
            raise SchemaValidationError(
                f"semantic field {key!r} must be {pinned!r}, found"
                f" {semantic[key]!r} — a constructed-reference artifact may"
                " not alter the pinned semantic layer (R-001/R-002)"
            )


def _validate_llm_involvement(block: Any) -> None:
    """R-029: per-axis declaration; non-none axes require a tool note."""
    if not isinstance(block, dict):
        raise SchemaValidationError(
            "llm_involvement block missing or malformed (R-029)"
        )
    for axis in LLM_INVOLVEMENT_AXES:
        if axis not in block:
            raise SchemaValidationError(
                f"llm_involvement missing axis {axis!r} — `none` is an"
                " explicit value, never an absent field (R-029)"
            )
        if not isinstance(block[axis], str) or not block[axis]:
            raise SchemaValidationError(
                f"llm_involvement axis {axis!r} must be a non-empty string"
            )
    unknown = sorted(set(block) - set(LLM_INVOLVEMENT_AXES) - {"tool_version_note"})
    if unknown:
        raise SchemaValidationError(
            f"llm_involvement carries unknown field(s): {unknown} (R-029)"
        )
    non_none = [axis for axis in LLM_INVOLVEMENT_AXES if block[axis] != "none"]
    if non_none:
        note = block.get("tool_version_note")
        if not isinstance(note, str) or not note.strip():
            raise SchemaValidationError(
                "llm_involvement declares non-none axis(es)"
                f" {non_none} but carries no tool_version_note (R-029)"
            )


def _validate_arm(arm: Any, index: int) -> None:
    """R-003: full per-arm identification; idealized arms are scalar."""
    if not isinstance(arm, dict):
        raise SchemaValidationError(f"arms[{index}] must be an object (R-003)")
    if not isinstance(arm.get("arm_id"), str) or not arm.get("arm_id"):
        raise SchemaValidationError(f"arms[{index}] missing arm_id (R-003)")
    for field in ARM_REQUIRED_FIELDS:
        if field not in arm:
            raise SchemaValidationError(
                f"arm {arm['arm_id']!r} missing identification field"
                f" {field!r} (R-003)"
            )
    if arm["cardinality"] not in ARM_CARDINALITIES:
        raise SchemaValidationError(
            f"arm {arm['arm_id']!r} cardinality must be one of"
            f" {list(ARM_CARDINALITIES)} (R-003)"
        )
    if arm["construction"] == "idealized" and arm["cardinality"] != "scalar":
        raise SchemaValidationError(
            f"arm {arm['arm_id']!r} is idealized (scalar prefix-to-gold cosine"
            " with oracle-assigned correctness) but declares"
            f" cardinality={arm['cardinality']!r}; idealized arms are scalar"
            " (R-003)"
        )


def _validate_cell_shape(cell: Any, index: int, seen_ids: set[str]) -> None:
    """Structural cell checks that need no per-item records (R-011/R-032)."""
    if not isinstance(cell, dict):
        raise SchemaValidationError(f"cells[{index}] must be an object")
    for key in CELL_REQUIRED_KEYS:
        if key not in cell:
            raise SchemaValidationError(
                f"cells[{index}] missing required field {key!r}"
            )
    cell_id = cell["cell_id"]
    if not isinstance(cell_id, str) or not cell_id:
        raise SchemaValidationError(f"cells[{index}] cell_id must be a string")
    if cell_id in seen_ids:
        raise SchemaValidationError(
            f"duplicate cell identifier {cell_id!r} — cell identifiers are"
            " unique per artifact and duplicates fail closed (R-011)"
        )
    seen_ids.add(cell_id)
    estimand = cell["estimand"]
    if not isinstance(estimand, dict):
        raise SchemaValidationError(f"cell {cell_id!r} estimand must be an object")
    _check_tolerance(
        estimand.get("numerical_tolerance"), f"cell {cell_id!r} estimand"
    )
    recomputed = canonical_estimand_digest(estimand)
    if cell["estimand_digest"] != recomputed:
        raise SchemaValidationError(
            f"cell {cell_id!r} recorded estimand_digest does not match the"
            " digest recomputed over all estimand-defining fields (R-011)"
        )
    interval = cell.get("interval")
    if interval is not None:
        if not isinstance(interval, dict):
            raise SchemaValidationError(
                f"cell {cell_id!r} interval must be an object (R-015)"
            )
        for key in INTERVAL_REQUIRED_KEYS:
            if key not in interval:
                raise SchemaValidationError(
                    f"cell {cell_id!r} interval missing recorded identity"
                    f" field {key!r} — missing interval identity leaves the"
                    " interval non-certifying (R-015)"
                )


def validate_profile(profile: dict[str, Any]) -> None:
    """Validate a strict constructed-reference profile dict; raise on defect."""
    if not isinstance(profile, dict):
        raise SchemaValidationError("profile must be an object")
    missing = sorted(PROFILE_TOP_LEVEL_KEYS - set(profile))
    if missing:
        raise SchemaValidationError(
            f"profile missing required field(s): {missing}"
        )
    unknown = sorted(set(profile) - PROFILE_TOP_LEVEL_KEYS)
    if unknown:
        raise SchemaValidationError(
            f"profile carries unknown top-level field(s): {unknown} —"
            " historical identifiers never substitute for the semantic"
            " layer (R-001)"
        )
    profile_id = profile["profile_id"]
    if profile_id == RESERVED_OBSERVED_PROFILE_ID:
        raise SchemaValidationError(
            f"profile_id {profile_id!r} is the reserved observed-study"
            " identifier; the constructed-reference validator never accepts"
            " it (R-002)"
        )
    if profile_id != STRICT_PROFILE_ID:
        raise SchemaValidationError(
            f"profile_id {profile_id!r} is not the strict constructed-"
            f"reference identifier {STRICT_PROFILE_ID!r} (R-001)"
        )
    _validate_semantic_block(profile["semantic"])
    _validate_llm_involvement(profile["llm_involvement"])
    _check_tolerance(profile["numerical_tolerance"], "profile")
    if profile["item_key_derivation"] != ITEM_KEY_DERIVATION:
        raise SchemaValidationError(
            "item_key_derivation must pin exactly the re-derivable scheme"
            f" {ITEM_KEY_DERIVATION} (R-008)"
        )
    arms = profile["arms"]
    if not isinstance(arms, list) or not arms:
        raise SchemaValidationError("arms must be a non-empty list (R-003)")
    for index, arm in enumerate(arms):
        _validate_arm(arm, index)
    if not isinstance(profile["provenance"], dict):
        raise SchemaValidationError("provenance must be an object (R-012)")
    cells = profile["cells"]
    if not isinstance(cells, list) or not cells:
        raise SchemaValidationError("cells must be a non-empty list")
    seen_ids: set[str] = set()
    for index, cell in enumerate(cells):
        _validate_cell_shape(cell, index, seen_ids)


def validate_record(record: dict[str, Any]) -> None:
    """Validate one per-item record against the non-reversible contract (R-031)."""
    if not isinstance(record, dict):
        raise RecordValidationError("record must be an object (R-031)")
    key = record.get("item_key")
    if not isinstance(key, str) or not key:
        raise RecordValidationError("record missing opaque item_key (R-031)")
    for field, value in record.items():
        if field not in RECORD_ALLOWED_FIELDS:
            raise RecordValidationError(
                f"record field {field!r} is outside the enumerated record"
                " field set — per-item records are non-reversible (R-031)"
            )
        if field in RECORD_IDENTIFIER_FIELDS:
            continue
        if field in RECORD_BOOL_FIELDS:
            if not isinstance(value, bool):
                raise RecordValidationError(
                    f"record field {field!r} must be a boolean (R-031)"
                )
        elif field in RECORD_CATEGORICAL_FIELDS:
            if value not in EXCLUSION_REASONS:
                raise RecordValidationError(
                    f"record field {field!r} must be an enumerated"
                    " categorical, not free text (R-031)"
                )
        elif field in RECORD_CATEGORICAL_LIST_FIELDS:
            if not isinstance(value, list) or any(
                v not in EXCLUSION_REASONS for v in value
            ):
                raise RecordValidationError(
                    f"record field {field!r} must be a list of enumerated"
                    " categoricals (R-031)"
                )
        else:  # numeric fields
            if value is not None and not is_number(value):
                raise RecordValidationError(
                    f"record field {field!r} must be numeric or null —"
                    " string values outside the identifier allowlist are"
                    " rejected (R-031)"
                )


def encode_json(payload: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, 2-space indent, trailing newline.

    ``allow_nan=False`` rejects non-finite floats at encode time, before any
    filesystem effect (R-004).
    """
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def encode_profile(profile: dict[str, Any]) -> bytes:
    """Encode a profile to canonical bytes (allow_nan=False semantics, R-004)."""
    return encode_json(profile)


def _reject_nonfinite_constant(token: str) -> Any:
    raise TypedIngressError(
        f"non-finite JSON constant {token!r} rejected (allow_nan=False, R-004)"
    )


def _parse_json_bytes(data: bytes) -> Any:
    """utf-8 + JSON parse with non-finite constants rejected (R-004).

    Raises ``UnicodeDecodeError``/``json.JSONDecodeError``; each caller wraps
    them in the typed error carrying its own artifact identification.
    """
    return json.loads(
        data.decode("utf-8"), parse_constant=_reject_nonfinite_constant
    )


def decode_profile(data: bytes) -> dict[str, Any]:
    """Decode canonical profile bytes back to an equal value (R-004)."""
    try:
        return _parse_json_bytes(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedIngressError(f"malformed profile bytes: {exc}") from exc


def _relative_name(path: Path, tree_root: Path | None) -> str:
    """Repo-/tree-relative identification for error messages (R-020/R-026)."""
    if tree_root is not None:
        try:
            return (
                Path(path).resolve().relative_to(Path(tree_root).resolve())
            ).as_posix()
        except ValueError:
            pass
    return Path(path).name


def _check_schema_version(obj: dict[str, Any], rel: str) -> None:
    """R-020: schema_version is validated before any other check."""
    if "schema_version" not in obj:
        raise TypedIngressError(
            f"{rel}: missing required field 'schema_version' (R-020)"
        )
    version = obj["schema_version"]
    supported = (
        isinstance(version, int)
        and not isinstance(version, bool)
        and SUPPORTED_SCHEMA_VERSION_MIN
        <= version
        <= SUPPORTED_SCHEMA_VERSION_MAX
    )
    if not supported:
        raise TypedIngressError(
            f"{rel}: unsupported schema_version {version!r}; supported range"
            f" {SUPPORTED_SCHEMA_VERSION_MIN}..{SUPPORTED_SCHEMA_VERSION_MAX};"
            f" verifier revision {VERIFIER_REVISION} (R-020)"
        )


def _load_records(data: bytes, rel: str) -> dict[str, Any]:
    """Typed ingress for a ``*.jsonl`` record file (R-020)."""
    records: list[dict[str, Any]] = []
    text = data.decode("utf-8", errors="strict") if data else ""
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line, parse_constant=_reject_nonfinite_constant)
        except json.JSONDecodeError as exc:
            raise TypedIngressError(
                f"{rel}: line {lineno}: malformed JSON record: {exc} (R-020)"
            ) from exc
        if not isinstance(obj, dict):
            raise TypedIngressError(
                f"{rel}: line {lineno}: record must be an object (R-020)"
            )
        records.append(obj)
    return {"kind": "records", "records": records}


def load_artifact(path: Path, *, tree_root: Path | None = None) -> dict[str, Any]:
    """Typed ingress: validate artifact bytes into typed records (R-020).

    ``*.jsonl`` files load as ``{"kind": "records", "records": [...]}``;
    other files load as strict-profile-shaped objects. Errors are typed and
    identify files by tree-relative path only.
    """
    path = Path(path)
    rel = _relative_name(path, tree_root)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TypedIngressError(
            f"{rel}: unreadable artifact ({exc.__class__.__name__})"
        ) from exc

    if path.suffix == ".jsonl":
        return _load_records(data, rel)

    try:
        obj = _parse_json_bytes(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedIngressError(f"{rel}: malformed JSON: {exc} (R-020)") from exc
    if not isinstance(obj, dict):
        raise TypedIngressError(f"{rel}: artifact must be a JSON object (R-020)")
    _check_schema_version(obj, rel)
    unknown = sorted(set(obj) - PROFILE_TOP_LEVEL_KEYS)
    if unknown:
        raise TypedIngressError(
            f"{rel}: unknown top-level field(s) {unknown} — no silent"
            " key-dropping (R-020)"
        )
    return obj


def write_profile(path: Path, profile: dict[str, Any]) -> None:
    """Create-once strict-profile writer (R-001, R-004, R-016).

    Encodes first (``allow_nan=False`` rejects non-finite floats before any
    filesystem effect), validates the strict profile, then publishes via the
    ``scripts/stopdff_v5/fileio`` create-once primitive (module-attribute
    call so routing stays interceptable/auditable).
    """
    data = encode_profile(profile)
    validate_profile(profile)
    fileio.create_once_bytes(
        Path(path), data, exists_label="strict profile artifact"
    )


def publish_evidence_package(staged: Path, runs_root: Path, run_id: str) -> Path:
    """Publish a staged evidence package into a run-scoped create-once dir (R-039)."""
    staged = Path(staged)
    runs_root = Path(runs_root)
    if not is_path_component(run_id):
        raise ColmAimsError(
            f"run_id {run_id!r} must be a non-empty single path component"
            " (R-039)"
        )
    if not staged.is_dir():
        raise ColmAimsError("staged evidence package must be a directory")
    dest = runs_root / run_id
    fileio.publish_dir_create_once(
        staged, dest, exists_label="evidence package run slot"
    )
    return dest
