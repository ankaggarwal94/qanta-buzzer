"""Strict schema-versioned v2 constructed-reference profile: types, ingress.

Spec rules owned here: R-001..R-004, R-016 (writer side), R-020, R-029,
R-031, R-032, R-039 (publish side), R-045..R-047 (record contract), R-058,
R-059 (single bool-safe version checker), R-061 (record-level integer
domains), R-062/R-067 (hardened parse hooks), R-063 (closed trusted maps).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

from scripts.stopdff_v5 import fileio


class ColmAimsError(Exception):
    """Base class for every typed error raised by this namespace."""


class SchemaValidationError(ColmAimsError):
    """Strict-profile semantic/schema violation (R-001..R-003, R-029, R-032)."""


class RecordValidationError(SchemaValidationError):
    """Per-item record violates the non-reversible record contract
    (R-031/R-045/R-061)."""


class TypedIngressError(ColmAimsError):
    """Artifact bytes failed typed validation at the load boundary (R-020)."""


class SchemaVersionError(TypedIngressError):
    """Unsupported/missing/mistyped ``schema_version`` on a versioned surface
    (R-059). The ONE bool-safe version-defect type raised by the single
    shared checker on every versioned surface."""


class ConfigSurfaceError(ColmAimsError):
    """Unknown key/flag on the config surface — usage error, never a no-op
    (R-022/R-037/R-063)."""


class EmptyEvaluationError(ColmAimsError):
    """Explicitly empty evaluation population refused (R-006, R-012).

    Reserved strictly for ``n_pairing_population == 0`` — the one condition
    the rule text names as a typed error.
    """


# ---------------------------------------------------------------------------
# Pinned constants (R-058: ONE canonical revision constant set)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSION_MIN = 2
SUPPORTED_SCHEMA_VERSION_MAX = 2
VERIFIER_REVISION = "reproducibility.colm_aims_2026:r2"

STRICT_PROFILE_ID = "colm_aims_constructed_reference_v2"
# R-002: reserved identifier for genuinely observed future studies. The
# constructed-reference validator never accepts it; no code path converts
# one profile into the other.
RESERVED_OBSERVED_PROFILE_ID = "colm_aims_observed_paired_v1"

# Handoff sanctioned output when the intended headline becomes actual
# preservation/change (R-027).
OBSERVED_PAIRED_CLAIM_OUTPUT = (
    "observed_paired_claim=OBSERVED_PAIRED_STUDY_REQUIRED"
)

# R-032: pinned maximum admissible numerical tolerance.
MAX_ADMISSIBLE_TOLERANCE = 1e-3

# Operational allocation safeguards (R-061: safeguards, never construct
# definitions).
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_BOOTSTRAP_DRAWS = 200_000
MAX_BOOTSTRAP_CELLS = 200_000_000

# R-062 (Track A' R5, D8): NON-SEMANTIC token-length crash guard applied
# lexically BEFORE int() conversion. This cap must stay well below ~300
# digits so every ingress-parsed int stays below float-max and unguarded
# float() coercions at gate predicates can never overflow.
MAX_JSON_INT_TOKEN_DIGITS = 100

# R-001: the pinned semantic block, verbatim from handoff §8.
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

# R-003: per-arm identification fields (v2 adds arm_id/family/stop_semantics
# to the carried v1 set).
ARM_REQUIRED_FIELDS = (
    "arm_id",
    "family",
    "stop_semantics",
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

# R-003: closed per-family stop-semantics vocabularies — no overloaded
# global stop integer across families.
FAMILY_STOP_VOCAB: dict[str, str] = {
    "constructed_reference": "reference_threshold_crossing",
    "fixed_threshold": "fixed_threshold_crossing",
    "myopic": "myopic_one_step_stop",
    "learned_continuation": "learned_value_stop",
}

# R-045: closed canonical event vocabulary + terminal-imputation enum.
EVENT_FINITE = "FINITE_STOP"
EVENT_NEVER = "NEVER_STOPPED"
EVENT_STATUSES = frozenset({EVENT_FINITE, EVENT_NEVER})
IMPUTATION_NONE = "NONE"
IMPUTATION_FINAL_PREFIX = "FINAL_PREFIX_IF_NEVER"
TERMINAL_IMPUTATIONS = frozenset({IMPUTATION_NONE, IMPUTATION_FINAL_PREFIX})

# R-047: spec-pinned new exclusion-reason enum member joins the carried set.
AMBIGUOUS_TERMINAL_SENTINEL = "AMBIGUOUS_TERMINAL_SENTINEL"
# R-074 (PRE-1): pre-package exclusion reason for items whose trajectory has
# fewer than two cumulative prefixes (the frozen eligibility artifact's nine
# excluded qids all carry this reason).
SINGLE_PREFIX_TRAJECTORY = "SINGLE_PREFIX_TRAJECTORY"
EXCLUSION_REASONS = frozenset(
    {
        "MALFORMED_STOP",
        "MISSING_STOP",
        "GRID_MISMATCH",
        "UNKNOWN_NOT_INFERRED",
        AMBIGUOUS_TERMINAL_SENTINEL,
        SINGLE_PREFIX_TRAJECTORY,
    }
)

# R-046/R-010: the preserved fair-QA producer's derived-scalar convention.
SENTINEL_CONVENTION = "timeout_coded_as_horizon"

# R-057: exact analysis-provenance discriminator token for D7(b) outputs.
ANALYSIS_PROVENANCE_D7B = "d7b_regenerated_2026"

# R-054: closed population enum for estimand/interval identities.
POPULATION_ALL = "all_complete_pairs_terminal_imputed"
POPULATION_FINITE = "both_finite_only"
POPULATIONS = frozenset({POPULATION_ALL, POPULATION_FINITE})

# R-048/R-049: closed estimand labels for the two named estimands.
HEADLINE_ESTIMAND_LABEL = (
    "mean_signed_shift_mc_minus_ref_all_complete_pairs_terminal_imputed"
)
FINITE_ONLY_ESTIMAND_LABEL = "mean_signed_shift_mc_minus_ref_both_finite_only"
ESTIMAND_LABELS = frozenset(
    {HEADLINE_ESTIMAND_LABEL, FINITE_ONLY_ESTIMAND_LABEL}
)

# Frozen 5x2 grid identity (R-040/R-061: cardinalities 10/5/2).
REFERENCE_IDS = ("idealized", "kdisjoint", "khard", "klex", "krandom")
CALIBRATION_IDS = ("format_specific", "shared")
CELL_IDS = tuple(
    sorted(f"{r}__{c}" for r in REFERENCE_IDS for c in CALIBRATION_IDS)
)
EXPECTED_COMPLETE_PAIRS = 2249  # R-042: exactly 2,249 complete pairs per cell
BOOTSTRAP_DRAW_COUNT = 1000  # R-051/R-061: exactly B=1000 for this profile

# R-052: the recorded seed-derivation string (exact).
SEED_DERIVATION_STRING = (
    'sha256(b"colm_aims_2026/v2/bootstrap_holm\\0"'
    " + bytes.fromhex(pairing_population_keyset_sha256)).digest()[:8]"
    " big-endian unsigned"
)

# R-051: pinned resample-plan tokens.
GENERATOR_CONSTRUCTION = "numpy.random.Generator(numpy.random.PCG64(seed))"
RESAMPLING_UNIT = "item_tossup_clustered_all_prefixes_both_arms"
INTERVAL_PROCEDURE = "d7b_shared_percentile_bootstrap"
INTERVAL_STATISTIC = "mean_signed_shift"
QUANTILE_METHOD = "linear"

# R-011: closed legal-value vocabularies for the 7B/M3 surface.
PAIRING_DEFINITIONS = frozenset({"matched_item_prefix_grid"})
TIMEOUT_RULES = frozenset({"zero_indexed_stop_ge_horizon_is_timeout"})
# The rule vocabulary each pairing definition reconciles with (R-011).
PAIRING_RULE_RECONCILIATION: dict[str, frozenset[str]] = {
    "matched_item_prefix_grid": frozenset(
        {"zero_indexed_stop_ge_horizon_is_timeout"}
    ),
}

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
        "grid",
        "inference",
        "cells",
    }
)

GRID_KEYS = frozenset(
    {
        "reference_ids",
        "calibration_ids",
        "cell_ids",
        "record_files",
        "item_keys_sha256",
        "held_fixed",
    }
)
HELD_FIXED_KEYS = frozenset({"mc_trajectory_identity", "horizon_identity"})


def horizon_map_sha256(mapping: dict[str, int]) -> str:
    """R-073 canonical per-item horizon-map digest.

    Serialization is pinned: a JSON object mapping item key -> positive
    integer horizon, keys sorted ascending by UTF-8 byte order, compact
    separators, UTF-8 encoded; digest is lowercase-hex SHA-256 of those
    bytes. This one function is the shared source of truth for the
    producer-side freeze artifact, the verifier's recompute legs, and the
    held-fixed ``horizon_identity`` (R-043) — reimplementations are
    forbidden so the three surfaces cannot drift.

    DECISION-4 fail-closed domain guards (Phase-4 PRE, 2026-08-22): the
    digest domain is EXACTLY ``str -> positive real int``. Non-string keys
    are never stringified, bools are never digested as 0/1, floats
    (including integer-valued ones like 2.0) are never truncated or
    coerced, non-positive horizons are refused, and the empty map is a
    defect, never a valid digestible identity (vacuously-empty
    authoritative sets — seed catalog).
    """
    if not isinstance(mapping, dict):
        raise SchemaValidationError(
            "horizon map must be a JSON object mapping item keys to"
            " positive integer horizons (R-073)"
        )
    if not mapping:
        raise SchemaValidationError(
            "horizon map is empty — a vacuously-empty horizon map is a"
            " defect, never a valid digestible identity (R-073)"
        )
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise SchemaValidationError(
                "horizon map key is not a non-empty string — item keys are"
                " never silently stringified into the digest domain (R-073)"
            )
        if not is_real_int(value):
            raise SchemaValidationError(
                f"horizon map value for a key is {value!r} — the digest"
                " domain is positive real int; bools and floats (including"
                " integer-valued floats) are never coerced (R-073/R-061)"
            )
        if value <= 0:
            raise SchemaValidationError(
                f"horizon map value {value!r} is not a positive integer"
                " (R-073)"
            )
    canonical = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

INFERENCE_KEYS = frozenset(
    {
        "analysis_provenance",
        "numpy_version",
        "bit_generator",
        "generator_construction",
        "draw_count",
        "sample_size",
        "resampling_unit",
        "with_replacement",
        "dtype",
        "endpoint",
        "seed",
        "seed_derivation",
        "pairing_population_keyset_sha256",
        "canonical_item_order_digest",
        "resample_matrix_digest",
        "familywise_alpha",
        "family_size",
        "ordered_family",
        "rejected_cell_ids",
    }
)
MATRIX_DIGEST_KEYS = frozenset(
    {"sha256", "dtype", "shape", "byte_order", "canonical_item_order_digest"}
)

CELL_REQUIRED_KEYS = (
    "cell_id",
    "reference_id",
    "calibration_id",
    "estimand",
    "estimand_digest",
    "records_file",
    "counts",
    "rates",
    "headline_summary",
    "finite_only_summary",
    "interval",
    "raw_p_value",
    "holm_rank",
    "holm_adjusted_p_value",
    "holm_rejected",
    "excluded_keys",
    "pairing_population_keyset_sha256",
)

ESTIMAND_KEYS = frozenset(
    {
        "arm_mc",
        "arm_ref",
        "reference_id",
        "calibration_id",
        "pairing_definition",
        "timeout_parameters",
        "event_representation",
        "population",
        "denominator_policy",
        "numerical_tolerance",
        "calibration_identity",
        "continuation_identity",
        "random_k_draw_id",
    }
)
# R-073 (PRE-2): the scalar ``trajectory_horizon`` member is RETIRED — the
# declaration is the canonical per-item horizon-map digest.
TIMEOUT_PARAMETER_KEYS = frozenset({"horizon_map_sha256", "rule"})
EVENT_REPRESENTATION_KEYS = frozenset(
    {
        "index_base",
        "horizon_identity",
        "mc_trajectory_identity",
        "historical_sentinel_convention",
        "terminal_imputation_policy",
        "producer_profile_identity",
    }
)

# R-015: recorded interval identity (everything but the interval itself).
INTERVAL_IDENTITY_KEYS = (
    "procedure",
    "draw_count",
    "seed",
    "seed_derivation",
    "statistic",
    "population",
)
INTERVAL_REQUIRED_KEYS = INTERVAL_IDENTITY_KEYS + ("quantile_method", "ci")

# D1: calibration identity is a MAP with exactly one entry per calibration.
CALIBRATION_IDENTITY_KEYS = frozenset({"format_specific", "shared"})

# R-031 (v2): the enumerated record field set — the ONLY string-valued field
# a per-item record may carry outside enumerated categoricals is its opaque
# item key.
RECORD_IDENTIFIER_FIELDS = frozenset({"item_key"})
RECORD_EVENT_STATUS_FIELDS = frozenset({"mc_event_status", "ref_event_status"})
RECORD_STOP_FIELDS = frozenset({"mc_stop_step", "ref_stop_step"})
RECORD_IMPUTATION_FIELDS = frozenset(
    {"mc_terminal_imputation", "ref_terminal_imputation"}
)
RECORD_NUMERIC_FIELDS = frozenset(
    {
        "trajectory_horizon",
        "mc_original_encoded_stop",
        "ref_original_encoded_stop",
    }
)
RECORD_BOOL_FIELDS = frozenset(
    {"excluded", "mc_crossing_indicator", "ref_crossing_indicator"}
)
RECORD_CATEGORICAL_FIELDS = frozenset({"exclusion_reason"})
RECORD_CATEGORICAL_LIST_FIELDS = frozenset({"secondary_diagnostics"})
RECORD_ALLOWED_FIELDS = (
    RECORD_IDENTIFIER_FIELDS
    | RECORD_EVENT_STATUS_FIELDS
    | RECORD_STOP_FIELDS
    | RECORD_IMPUTATION_FIELDS
    | RECORD_NUMERIC_FIELDS
    | RECORD_BOOL_FIELDS
    | RECORD_CATEGORICAL_FIELDS
    | RECORD_CATEGORICAL_LIST_FIELDS
)
RECORD_REQUIRED_FIELDS = (
    "item_key",
    "trajectory_horizon",
    "mc_event_status",
    "mc_stop_step",
    "mc_terminal_imputation",
    "ref_event_status",
    "ref_stop_step",
    "ref_terminal_imputation",
)


# ---------------------------------------------------------------------------
# Shared value predicates
# ---------------------------------------------------------------------------

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_OBJECT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def is_number(value: Any) -> bool:
    """True for a real int/float; ``bool`` never counts as a number here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_real_int(value: Any) -> bool:
    """True for a real int; ``bool`` never satisfies an integer domain
    (R-061)."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_sha256_hex(value: Any) -> bool:
    """True for a full-length lowercase sha256 hex digest."""
    return isinstance(value, str) and _SHA256_HEX_RE.fullmatch(value) is not None


def is_commit_sha(value: Any) -> bool:
    """True for a full-length 40-hex commit SHA.

    This narrower predicate remains available for external model revision
    domains that explicitly require SHA-1.
    """
    return isinstance(value, str) and _COMMIT_SHA_RE.fullmatch(value) is not None


def is_git_object_id(value: Any) -> bool:
    """True for a native SHA-1 or SHA-256 Git object identifier.

    Repository identities may be 40 or 64 lowercase hexadecimal characters.
    Short hashes, tags, and branch names never qualify (R-012/R-013).
    """
    return (
        isinstance(value, str)
        and _GIT_OBJECT_ID_RE.fullmatch(value) is not None
    )


def is_uint64(value: Any) -> bool:
    """True for a real int in the unsigned 64-bit domain (R-052/R-061)."""
    return is_real_int(value) and 0 <= value < 2**64


def is_admissible_tolerance(value: Any) -> bool:
    """R-032: a declared tolerance is admissible when it is a real number in
    ``(0, MAX_ADMISSIBLE_TOLERANCE]``; non-finite values never qualify."""
    if not is_number(value):
        return False
    as_float = float(value)
    return math.isfinite(as_float) and 0 < as_float <= MAX_ADMISSIBLE_TOLERANCE


def is_native_finite_number(value: Any) -> bool:
    """Native finite JSON number, gated BEFORE any float() coercion (R-067).

    str/bool are rejected outright. Parsed integers are bounded by the
    R-062 token-length guard (100 digits, well under float-max), so the
    float() below can never overflow on ingress-parsed values.
    """
    if not is_number(value):
        return False
    return math.isfinite(float(value))


def is_path_component(value: Any) -> bool:
    """True for a non-empty single path component (no separators/traversal).

    Bare ``.`` and ``..`` are excluded explicitly — ``..`` satisfies
    ``Path(value).name == value`` yet traverses out of any joined root.
    """
    return (
        isinstance(value, str)
        and bool(value)
        and value not in (".", "..")
        and Path(value).name == value
    )


def resolves_inside(path: Path, root: Path) -> bool:
    """True when ``path`` resolves to ``root`` itself or anything beneath it.

    Containment decisions use fully resolved, symlink-free paths (R-013).
    """
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    return resolved == root_resolved or root_resolved in resolved.parents


def canonical_estimand_digest(estimand: dict[str, Any]) -> str:
    """R-011 pinned digest: sha256 over canonical compact JSON of the block."""
    payload = json.dumps(estimand, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# R-059: the ONE bool-safe version checker for EVERY versioned surface
# ---------------------------------------------------------------------------


def check_schema_version(obj: dict[str, Any], rel: str) -> None:
    """Single shared bool-safe schema-version gate (R-058/R-059).

    Admissibility is exactly ``type(version) is int and
    min_supported <= version <= max_supported`` — a JSON Boolean is not an
    integer version. Every version error names the observed version, the
    supported range, and the canonical verifier revision.
    """
    supported_range = (
        f"{SUPPORTED_SCHEMA_VERSION_MIN}..{SUPPORTED_SCHEMA_VERSION_MAX}"
    )
    if "schema_version" not in obj:
        raise SchemaVersionError(
            f"{rel}: missing required field 'schema_version'; supported"
            f" range {supported_range}; verifier revision"
            f" {VERIFIER_REVISION} (R-059)"
        )
    version = obj["schema_version"]
    admissible = (
        type(version) is int
        and SUPPORTED_SCHEMA_VERSION_MIN
        <= version
        <= SUPPORTED_SCHEMA_VERSION_MAX
    )
    if not admissible:
        raise SchemaVersionError(
            f"{rel}: unsupported schema_version {version!r}; supported"
            f" range {supported_range}; verifier revision"
            f" {VERIFIER_REVISION} (R-059)"
        )


# ---------------------------------------------------------------------------
# Hardened JSON ingress (R-062/R-067: protective hooks everywhere)
# ---------------------------------------------------------------------------


def _reject_nonfinite_constant(token: str) -> Any:
    raise TypedIngressError(
        f"non-finite JSON constant {token!r} rejected (R-067)"
    )


def _reject_nonfinite_float(token: str) -> float:
    """``parse_constant`` never sees a huge-exponent numeric literal
    (``1e999`` parses straight to float infinity), so every parsed float is
    finiteness-checked at the token level too (R-067)."""
    value = float(token)
    if not math.isfinite(value):
        raise TypedIngressError(
            f"non-finite JSON number {token!r} rejected (R-067)"
        )
    return value


def _length_guarded_parse_int(token: str) -> int:
    """Length-bounded integer token parse (R-062, D8).

    A NON-SEMANTIC crash guard, not a domain rule: tokens longer than
    ``MAX_JSON_INT_TOKEN_DIGITS`` digits (sign excluded) are refused with a
    typed error BEFORE ``int()`` runs (CPython's int-str conversion limit
    would otherwise raise a bare ValueError). The global ±2^53 parser
    ceiling is REMOVED and never revived (D8/Track A' R4): legitimate
    beyond-float-exact integers (uint64 seeds, ns timestamps) parse.
    """
    digits = len(token) - (1 if token.startswith("-") else 0)
    if digits > MAX_JSON_INT_TOKEN_DIGITS:
        shown = token if len(token) <= 32 else token[:32] + "…"
        raise TypedIngressError(
            f"JSON integer token {shown!r} exceeds the maximum admissible"
            f" token length of {MAX_JSON_INT_TOKEN_DIGITS} digits"
            " (non-semantic crash guard, R-062)"
        )
    return int(token)


def _reject_duplicate_object_members(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build one JSON object while rejecting repeated member names.

    CPython's default decoder silently keeps the final value for a duplicate
    name. Identity, configuration, and record objects are security-relevant
    inputs here, so ambiguity is a typed ingress refusal before any schema or
    semantic interpretation occurs.
    """
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise TypedIngressError(
                f"duplicate JSON object member {name!r} rejected (R-067)"
            )
        result[name] = value
    return result


# NOTE: both parse entrypoints below spell the protective hooks out as
# literal keyword arguments. This duplication is REQUIRED, not drift: the
# R-067/R-062 suite AST-walks this module and asserts every ``json.loads``
# call site names object_pairs_hook/parse_constant/parse_float/parse_int
# directly, so a shared ``**hooks`` mapping would defeat the check.
def _parse_json_bytes(data: bytes) -> Any:
    """utf-8 + JSON parse with every non-finite form rejected (R-067) and
    overlong integer tokens refused pre-conversion (R-062).

    Raises ``UnicodeDecodeError``/``json.JSONDecodeError``; each caller wraps
    them in the typed error carrying its own artifact identification.
    ``TypedIngressError`` from the hooks propagates as-is.
    """
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_members,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_reject_nonfinite_float,
            parse_int=_length_guarded_parse_int,
        )
    except RecursionError:
        # QA2-001: deeply-nested hostile JSON must be a MALFORMED-INPUT
        # refusal (typed, file-named by each call site's existing wrapper),
        # never a run-aborting internal error. Sibling of the overlong-int
        # token guard: a non-semantic crash bound at the parse boundary.
        raise json.JSONDecodeError(
            "JSON nesting depth exceeds the admissible bound"
            " (non-semantic crash guard, R-062)",
            "",
            0,
        ) from None


def parse_json_bytes_strict(data: bytes) -> Any:
    """Public strict ingress parse shared across this namespace (R-067).

    EVERY JSON parse site in reproducibility/colm_aims_2026/ routes through
    this module's hooked loaders — sibling modules never call raw
    ``json.loads``.
    """
    return _parse_json_bytes(data)


def parse_json_text_strict(text: str, rel: str) -> Any:
    """Hooked parse of one already-decoded JSON text (record lines, R-067)."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_members,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_reject_nonfinite_float,
            parse_int=_length_guarded_parse_int,
        )
    except json.JSONDecodeError as exc:
        raise TypedIngressError(
            f"{rel}: malformed JSON: {exc} (R-020)"
        ) from exc
    except RecursionError:
        # QA2-001 twin for the text-line parser (records lines).
        raise TypedIngressError(
            f"{rel}: malformed JSON: nesting depth exceeds the admissible"
            " bound (non-semantic crash guard, R-062/R-020)"
        ) from None


# ---------------------------------------------------------------------------
# Canonical encode/decode (R-004)
# ---------------------------------------------------------------------------


def encode_json(payload: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, 2-space indent, trailing newline.

    ``allow_nan=False`` rejects non-finite floats at encode time, before any
    filesystem effect (R-004).
    """
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def encode_profile(profile: dict[str, Any]) -> bytes:
    """Encode a profile to canonical bytes (allow_nan=False semantics)."""
    return encode_json(profile)


def decode_profile(data: bytes) -> dict[str, Any]:
    """Decode canonical profile bytes back to an equal value (R-004)."""
    try:
        return _parse_json_bytes(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedIngressError(f"malformed profile bytes: {exc}") from exc


# ---------------------------------------------------------------------------
# Bounded, symlink-free untrusted reads (R-020)
# ---------------------------------------------------------------------------


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


def read_regular_file_bytes(
    path: Path,
    *,
    tree_root: Path | None = None,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> bytes:
    """Bounded, symlink-free read of an untrusted path (R-020).

    Opens with ``O_NOFOLLOW`` (a symlink at ``path`` fails, never followed),
    fstats the open descriptor and refuses anything that is not a regular
    file, and caps the read at ``max_bytes``. Errors are typed and identify
    the file by basename/tree-relative path only (never a local absolute
    path, R-026).
    """
    path = Path(path)
    rel = _relative_name(path, tree_root)
    if path.is_symlink():
        raise TypedIngressError(
            f"{rel}: refusing to read a symlink (R-020/R-013)"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        # Windows otherwise opens the descriptor in CRT text mode: CRLF is
        # translated and Ctrl-Z truncates the stream.  Every digest and
        # strict parser above this boundary requires the literal file bytes.
        | getattr(os, "O_BINARY", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise TypedIngressError(
                f"{rel}: refusing to read a symlink (R-020/R-013)"
            ) from exc
        raise TypedIngressError(
            f"{rel}: unreadable artifact ({exc.__class__.__name__}) (R-020)"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise TypedIngressError(
                f"{rel}: not a regular file — refusing to read a FIFO,"
                " device, or socket (R-020)"
            )
        if hasattr(os, "O_NONBLOCK") and hasattr(os, "get_blocking"):
            os.set_blocking(fd, True)
        if st.st_size > max_bytes:
            raise TypedIngressError(
                f"{rel}: artifact size {st.st_size} exceeds the maximum"
                f" admissible {max_bytes} bytes (R-020)"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise TypedIngressError(
                f"{rel}: artifact exceeds the maximum admissible"
                f" {max_bytes} bytes (R-020)"
            )
        return data
    finally:
        os.close(fd)


def load_records_bytes(data: bytes, rel: str) -> dict[str, Any]:
    """Typed ingress for a ``*.jsonl`` per-cell record file (R-020).

    Record lines carry NO envelope version (OQ-V2-003 decision); a smuggled
    ``schema_version`` key is an unknown record field caught downstream.
    """
    records: list[dict[str, Any]] = []
    line_numbers: list[int] = []
    try:
        text = data.decode("utf-8", errors="strict") if data else ""
    except UnicodeDecodeError as exc:
        raise TypedIngressError(
            f"{rel}: invalid UTF-8 bytes at byte offset {exc.start} (R-020)"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        obj = parse_json_text_strict(line, f"{rel}: line {lineno}")
        if not isinstance(obj, dict):
            raise TypedIngressError(
                f"{rel}: line {lineno}: record must be an object (R-020)"
            )
        records.append(obj)
        line_numbers.append(lineno)
    return {"kind": "records", "records": records, "line_numbers": line_numbers}


def load_artifact_bytes(data: bytes, rel: str) -> dict[str, Any]:
    """Typed ingress over already-snapshotted bytes (R-020).

    Validation order: container shape → schema_version (R-059, the shared
    bool-safe checker) → all other key checks. ``*.jsonl`` names load as
    record sets.
    """
    if rel.endswith(".jsonl"):
        return load_records_bytes(data, rel)
    try:
        obj = _parse_json_bytes(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedIngressError(f"{rel}: malformed JSON: {exc} (R-020)") from exc
    if not isinstance(obj, dict):
        raise TypedIngressError(
            f"{rel}: artifact must be a JSON object (R-020)"
        )
    check_schema_version(obj, rel)
    unknown = sorted(set(obj) - PROFILE_TOP_LEVEL_KEYS)
    if unknown:
        raise TypedIngressError(
            f"{rel}: unknown top-level field(s) {unknown} — no silent"
            " key-dropping (R-020)"
        )
    return obj


def load_artifact(path: Path, *, tree_root: Path | None = None) -> dict[str, Any]:
    """Typed ingress: validate artifact bytes into typed records (R-020)."""
    path = Path(path)
    rel = _relative_name(path, tree_root)
    data = read_regular_file_bytes(path, tree_root=tree_root)
    return load_artifact_bytes(data, rel)


# ---------------------------------------------------------------------------
# Per-item record validation (R-031/R-045/R-046/R-061)
# ---------------------------------------------------------------------------


def _validate_arm_event(record: dict[str, Any], prefix: str, horizon: Any) -> None:
    status = record[f"{prefix}_event_status"]
    if status not in EVENT_STATUSES:
        raise RecordValidationError(
            f"record field {prefix}_event_status is outside the closed event"
            f" vocabulary {sorted(EVENT_STATUSES)} (R-045)"
        )
    stop = record[f"{prefix}_stop_step"]
    imputation = record[f"{prefix}_terminal_imputation"]
    if imputation not in TERMINAL_IMPUTATIONS:
        raise RecordValidationError(
            f"record field {prefix}_terminal_imputation is outside the"
            f" closed enum {sorted(TERMINAL_IMPUTATIONS)} (R-045)"
        )
    crossing = record.get(f"{prefix}_crossing_indicator")
    if status == EVENT_FINITE:
        if not is_real_int(stop):
            raise RecordValidationError(
                f"{prefix}_event_status FINITE_STOP requires a real finite"
                f" integer {prefix}_stop_step; null/bool/missing values are"
                " rejected (R-045/R-061)"
            )
        if not is_real_int(horizon) or horizon <= 0:
            raise RecordValidationError(
                "record trajectory_horizon must be a positive real integer"
                " (R-061)"
            )
        if not 0 <= stop < horizon:
            raise RecordValidationError(
                f"{prefix}_stop_step is outside the finite-stop domain"
                " 0 <= stop_step < horizon — a finite stop AT the horizon is"
                " the old sentinel coding, illegal in the canonical"
                " representation (R-061)"
            )
        if imputation != IMPUTATION_NONE:
            raise RecordValidationError(
                f"FINITE_STOP {prefix} arm must carry terminal_imputation"
                f" {IMPUTATION_NONE!r} — an imputation marker on a finite"
                " stop is rejected (R-045)"
            )
    else:  # NEVER_STOPPED
        if stop is not None:
            raise RecordValidationError(
                f"NEVER_STOPPED {prefix} arm must carry stop_step exactly"
                " null — a numeric stop on NEVER_STOPPED is the"
                " derived-scalar overwrite signature (R-045/R-046)"
            )
        if imputation != IMPUTATION_FINAL_PREFIX:
            raise RecordValidationError(
                f"NEVER_STOPPED {prefix} arm must carry terminal_imputation"
                f" {IMPUTATION_FINAL_PREFIX!r} (R-045)"
            )
        if crossing is True:
            raise RecordValidationError(
                f"{prefix} arm declares an explicit crossing indicator with"
                " NEVER_STOPPED — the crossing happened; a genuine"
                " final-prefix crossing is FINITE_STOP (R-046)"
            )


def validate_record(record: dict[str, Any]) -> None:
    """Validate one per-item record against the non-reversible v2 contract.

    Error messages reference records by opaque key/field name and never echo
    string values (R-026 sentinel-leak discipline).
    """
    if not isinstance(record, dict):
        raise RecordValidationError("record must be an object (R-031)")
    key = record.get("item_key")
    if not isinstance(key, str) or not key:
        raise RecordValidationError("record missing opaque item_key (R-031)")
    for field in record:
        if field not in RECORD_ALLOWED_FIELDS:
            raise RecordValidationError(
                f"record field {field!r} is outside the enumerated record"
                " field set — per-item records are non-reversible and carry"
                " no free text (R-031)"
            )
    for field in RECORD_REQUIRED_FIELDS:
        if field not in record:
            raise RecordValidationError(
                f"record missing required canonical-event field {field!r}"
                " (R-045)"
            )
    horizon = record["trajectory_horizon"]
    if not is_real_int(horizon) or horizon <= 0:
        raise RecordValidationError(
            "record trajectory_horizon must be a positive real integer —"
            " bools never satisfy an integer domain (R-061)"
        )
    for field, value in record.items():
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
        elif field in RECORD_NUMERIC_FIELDS:
            if value is not None and not is_number(value):
                raise RecordValidationError(
                    f"record field {field!r} must be numeric or null —"
                    " string values outside the identifier allowlist are"
                    " rejected (R-031)"
                )
        elif field in RECORD_STOP_FIELDS:
            if value is not None and not is_real_int(value):
                raise RecordValidationError(
                    f"record field {field!r} must be a real integer or"
                    " exactly null — bools never satisfy an integer domain"
                    " (R-061)"
                )
        elif field in RECORD_EVENT_STATUS_FIELDS | RECORD_IMPUTATION_FIELDS:
            if not isinstance(value, str):
                raise RecordValidationError(
                    f"record field {field!r} must be an enumerated"
                    " categorical string (R-045)"
                )
    _validate_arm_event(record, "mc", horizon)
    _validate_arm_event(record, "ref", horizon)


# ---------------------------------------------------------------------------
# Strict v2 profile validation (R-001..R-003, R-029, R-031, R-032, R-063)
# ---------------------------------------------------------------------------


def _check_tolerance(tolerance: Any, where: str) -> None:
    if not is_native_finite_number(tolerance):
        raise SchemaValidationError(
            f"{where}: declared numerical_tolerance must be a finite number"
        )
    if not is_admissible_tolerance(tolerance):
        raise SchemaValidationError(
            f"{where}: declared numerical_tolerance {tolerance!r} outside"
            f" the admissible range (0, {MAX_ADMISSIBLE_TOLERANCE}] (R-032)"
        )


def _validate_semantic_block(semantic: Any) -> None:
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
    unknown = sorted(
        set(block) - set(LLM_INVOLVEMENT_AXES) - {"tool_version_note"}
    )
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


def _validate_arm(arm: Any, index: int, seen_arm_ids: set[str]) -> None:
    if not isinstance(arm, dict):
        raise SchemaValidationError(f"arms[{index}] must be an object (R-003)")
    for field in ARM_REQUIRED_FIELDS:
        if field not in arm:
            raise SchemaValidationError(
                f"arms[{index}] missing identification field {field!r}"
                " (R-003)"
            )
    arm_id = arm["arm_id"]
    if not isinstance(arm_id, str) or not arm_id:
        raise SchemaValidationError(f"arms[{index}] arm_id must be a string")
    if arm_id in seen_arm_ids:
        raise SchemaValidationError(
            f"duplicate arm identifier {arm_id!r} (R-003)"
        )
    seen_arm_ids.add(arm_id)
    family = arm["family"]
    if family not in FAMILY_STOP_VOCAB:
        raise SchemaValidationError(
            f"arm {arm_id!r} family {family!r} is outside the closed family"
            f" set {sorted(FAMILY_STOP_VOCAB)} (R-003)"
        )
    if arm["stop_semantics"] != FAMILY_STOP_VOCAB[family]:
        raise SchemaValidationError(
            f"arm {arm_id!r} stop_semantics {arm['stop_semantics']!r} is not"
            f" the {family!r} family's qualified closed vocabulary"
            f" {FAMILY_STOP_VOCAB[family]!r} — no overloaded global stop"
            " integer across families (R-003)"
        )
    if arm["cardinality"] not in ARM_CARDINALITIES:
        raise SchemaValidationError(
            f"arm {arm_id!r} cardinality must be one of"
            f" {list(ARM_CARDINALITIES)} (R-003)"
        )
    if arm["construction"] == "idealized":
        if arm["cardinality"] != "scalar":
            raise SchemaValidationError(
                f"arm {arm_id!r} is idealized (scalar prefix-to-gold cosine)"
                f" but declares cardinality={arm['cardinality']!r};"
                " idealized arms are scalar (R-003)"
            )
        if arm["correctness_assignment"] != "oracle_gold":
            raise SchemaValidationError(
                f"arm {arm_id!r} is idealized but declares"
                f" correctness_assignment={arm['correctness_assignment']!r};"
                " idealized correctness is oracle-assigned (R-003)"
            )


def _validate_closed_map(
    block: Any, allowed: frozenset[str], required: frozenset[str], where: str
) -> None:
    """R-063: closed trusted block — unknown nested keys are typed errors."""
    if not isinstance(block, dict):
        raise SchemaValidationError(f"{where} must be an object (R-063)")
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise SchemaValidationError(
            f"{where} carries unknown key(s) {unknown} — trusted"
            " configuration objects use closed key sets (R-063)"
        )
    missing = sorted(required - set(block))
    if missing:
        raise SchemaValidationError(
            f"{where} missing required key(s) {missing} (R-063)"
        )


def _validate_provenance(prov: Any) -> None:
    if not isinstance(prov, dict):
        raise SchemaValidationError("provenance must be an object (R-012)")
    calibration = prov.get("calibration_identity")
    if not isinstance(calibration, dict):
        raise SchemaValidationError(
            "provenance calibration_identity must be a MAP with one entry"
            " per calibration ID — the v1 scalar shape is rejected (R-001/D1)"
        )
    got = set(calibration)
    if got != CALIBRATION_IDENTITY_KEYS:
        raise SchemaValidationError(
            "provenance calibration_identity map keys"
            f" {sorted(got)} must be exactly"
            f" {sorted(CALIBRATION_IDENTITY_KEYS)} (R-001/D1)"
        )
    for cal_id, value in calibration.items():
        if not isinstance(value, str) or not value:
            raise SchemaValidationError(
                f"provenance calibration_identity[{cal_id!r}] must be a"
                " non-empty identity string (R-001)"
            )
    retention = prov.get("pre_package_retention")
    if not isinstance(retention, dict):
        raise SchemaValidationError(
            "provenance missing pre_package_retention documentation — the"
            " upstream-unpaired items are pre-package retention"
            " documentation, never in-package excluded_keys (R-052)"
        )
    retained = retention.get("retained_count")
    paired = retention.get("paired_count")
    unpaired = retention.get("upstream_unpaired_count")
    for name, value in (
        ("retained_count", retained),
        ("paired_count", paired),
        ("upstream_unpaired_count", unpaired),
    ):
        if not is_real_int(value) or value < 0:
            raise SchemaValidationError(
                f"pre_package_retention.{name} must be a nonnegative real"
                " integer (R-052/R-061)"
            )
    if retained - paired != unpaired:
        raise SchemaValidationError(
            f"pre_package_retention arithmetic does not hold:"
            f" retained_count {retained} - paired_count {paired} !="
            f" upstream_unpaired_count {unpaired} (R-052)"
        )


def _validate_grid_block(grid: Any) -> None:
    _validate_closed_map(grid, GRID_KEYS, GRID_KEYS, "grid block")
    for axis in ("reference_ids", "calibration_ids", "cell_ids"):
        values = grid[axis]
        if not isinstance(values, list) or not all(
            isinstance(v, str) and v for v in values
        ):
            raise SchemaValidationError(
                f"grid.{axis} must be a list of non-empty strings (R-040)"
            )
    record_files = grid["record_files"]
    if not isinstance(record_files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v
        for k, v in record_files.items()
    ):
        raise SchemaValidationError(
            "grid.record_files must map cell ids to record-file paths"
            " (R-041)"
        )
    if not is_sha256_hex(grid["item_keys_sha256"]):
        raise SchemaValidationError(
            "grid.item_keys_sha256 must be a sha256 digest (R-042)"
        )
    held_fixed = grid["held_fixed"]
    _validate_closed_map(
        held_fixed, HELD_FIXED_KEYS, HELD_FIXED_KEYS, "grid.held_fixed"
    )
    for key in HELD_FIXED_KEYS:
        value = held_fixed[key]
        if not isinstance(value, str) or not value:
            raise SchemaValidationError(
                f"grid.held_fixed.{key} must be a non-empty identity string"
                " (R-043)"
            )


def _validate_inference_block(inference: Any) -> None:
    _validate_closed_map(
        inference, INFERENCE_KEYS, INFERENCE_KEYS, "inference block"
    )
    if inference["analysis_provenance"] != ANALYSIS_PROVENANCE_D7B:
        raise SchemaValidationError(
            "inference.analysis_provenance"
            f" {inference['analysis_provenance']!r} is not the pinned new-"
            f"analysis discriminator {ANALYSIS_PROVENANCE_D7B!r} — the D7(b)"
            " outputs are a NEW analysis (R-057)"
        )
    seed = inference["seed"]
    if not is_uint64(seed):
        raise SchemaValidationError(
            f"inference.seed must be exactly one real integer in the"
            " unsigned 64-bit domain [0, 2**64); bools are rejected"
            " (R-052/R-061)"
        )
    seed_derivation = inference["seed_derivation"]
    if not isinstance(seed_derivation, str) or not seed_derivation:
        raise SchemaValidationError(
            "inference.seed_derivation must record the derivation string"
            " beside the derived integer seed (R-052)"
        )
    if not is_sha256_hex(inference["pairing_population_keyset_sha256"]):
        raise SchemaValidationError(
            "inference.pairing_population_keyset_sha256 must be a sha256"
            " digest (R-052)"
        )
    if not is_sha256_hex(inference["canonical_item_order_digest"]):
        raise SchemaValidationError(
            "inference.canonical_item_order_digest must be a sha256 digest"
            " (R-050)"
        )
    _validate_closed_map(
        inference["resample_matrix_digest"],
        MATRIX_DIGEST_KEYS,
        MATRIX_DIGEST_KEYS,
        "inference.resample_matrix_digest",
    )
    for name in ("draw_count", "sample_size", "family_size"):
        if not is_real_int(inference[name]):
            raise SchemaValidationError(
                f"inference.{name} must be a real integer (R-061)"
            )
    for name in ("with_replacement", "endpoint"):
        if not isinstance(inference[name], bool):
            raise SchemaValidationError(
                f"inference.{name} must be a boolean (R-051)"
            )
    for name in ("ordered_family", "rejected_cell_ids"):
        if not isinstance(inference[name], list) or not all(
            isinstance(v, str) for v in inference[name]
        ):
            raise SchemaValidationError(
                f"inference.{name} must be a list of cell ids (R-056)"
            )


def _validate_estimand(cell_id: str, estimand: Any) -> None:
    _validate_closed_map(
        estimand, ESTIMAND_KEYS, ESTIMAND_KEYS, f"cell {cell_id!r} estimand"
    )
    _check_tolerance(
        estimand["numerical_tolerance"], f"cell {cell_id!r} estimand"
    )
    _validate_closed_map(
        estimand["timeout_parameters"],
        TIMEOUT_PARAMETER_KEYS,
        TIMEOUT_PARAMETER_KEYS,
        f"cell {cell_id!r} estimand.timeout_parameters",
    )
    _validate_closed_map(
        estimand["event_representation"],
        EVENT_REPRESENTATION_KEYS,
        EVENT_REPRESENTATION_KEYS,
        f"cell {cell_id!r} estimand.event_representation",
    )


def _validate_cell_shape(cell: Any, index: int, seen_ids: set[str]) -> None:
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
    _validate_estimand(cell_id, estimand)
    recomputed = canonical_estimand_digest(estimand)
    if cell["estimand_digest"] != recomputed:
        raise SchemaValidationError(
            f"cell {cell_id!r} recorded estimand_digest does not match the"
            " digest recomputed over all estimand-defining fields (R-011)"
        )
    counts = cell["counts"]
    if not isinstance(counts, dict):
        raise SchemaValidationError(f"cell {cell_id!r} counts must be an object")
    for name, value in counts.items():
        if name == "exclusion_reason_counts":
            if not isinstance(value, dict):
                raise SchemaValidationError(
                    f"cell {cell_id!r} counts.exclusion_reason_counts must"
                    " be an object (R-008)"
                )
            continue
        if not is_real_int(value) or value < 0:
            raise SchemaValidationError(
                f"cell {cell_id!r} count {name!r} must be a nonnegative real"
                " integer — bools never satisfy an integer domain (R-061)"
            )
    if not isinstance(cell["rates"], dict):
        raise SchemaValidationError(f"cell {cell_id!r} rates must be an object")
    excluded_keys = cell["excluded_keys"]
    if not isinstance(excluded_keys, list):
        raise SchemaValidationError(
            f"cell {cell_id!r} excluded_keys must be a list (R-008)"
        )
    if excluded_keys:
        raise SchemaValidationError(
            f"cell {cell_id!r} carries nonempty in-package excluded_keys —"
            " the frozen v2 package has ZERO in-package exclusions; the"
            " upstream-unpaired items are pre-package retention"
            " documentation in provenance (R-052)"
        )
    if not is_sha256_hex(cell["pairing_population_keyset_sha256"]):
        raise SchemaValidationError(
            f"cell {cell_id!r} pairing_population_keyset_sha256 must be a"
            " sha256 digest (R-052)"
        )
    if not isinstance(cell["records_file"], str) or not cell["records_file"]:
        raise SchemaValidationError(
            f"cell {cell_id!r} records_file must be a path string (R-041)"
        )
    for summary_key in ("headline_summary", "finite_only_summary"):
        summary = cell[summary_key]
        if not isinstance(summary, dict):
            raise SchemaValidationError(
                f"cell {cell_id!r} {summary_key} must be an object (R-048)"
            )
        for field in ("estimand_label", "population", "n"):
            if field not in summary:
                raise SchemaValidationError(
                    f"cell {cell_id!r} {summary_key} missing required field"
                    f" {field!r} (R-048/R-049)"
                )
    interval = cell["interval"]
    if not isinstance(interval, dict):
        raise SchemaValidationError(
            f"cell {cell_id!r} interval must be an object (R-015)"
        )
    for key in INTERVAL_REQUIRED_KEYS:
        if key not in interval:
            raise SchemaValidationError(
                f"cell {cell_id!r} interval missing recorded identity field"
                f" {key!r} — missing interval identity leaves the interval"
                " non-certifying (R-015)"
            )
    ci = interval["ci"]
    # R-067: exactly 2 NATIVE finite numbers, ordered, gated BEFORE float().
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or not all(is_native_finite_number(v) for v in ci)
    ):
        raise SchemaValidationError(
            f"cell {cell_id!r} interval ci must be exactly two native finite"
            " numbers — string/bool/null/non-finite endpoints are rejected"
            " before any float conversion (R-067)"
        )
    if float(ci[0]) > float(ci[1]):
        raise SchemaValidationError(
            f"cell {cell_id!r} interval ci is not ordered (lo <= hi) (R-067)"
        )
    seed = interval["seed"]
    if not is_uint64(seed):
        raise SchemaValidationError(
            f"cell {cell_id!r} interval seed must be a real integer in the"
            " unsigned 64-bit domain (R-052/R-061)"
        )


def validate_profile(profile: dict[str, Any]) -> None:
    """Validate a strict v2 constructed-reference profile dict.

    Validation order: container shape → schema_version (shared bool-safe
    checker; R-059) → every other key and semantic check.
    """
    if not isinstance(profile, dict):
        raise SchemaValidationError("profile must be an object")
    check_schema_version(profile, "profile")
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
            f" {ITEM_KEY_DERIVATION} (R-008/R-063)"
        )
    arms = profile["arms"]
    if not isinstance(arms, list) or not arms:
        raise SchemaValidationError("arms must be a non-empty list (R-003)")
    seen_arm_ids: set[str] = set()
    for index, arm in enumerate(arms):
        _validate_arm(arm, index, seen_arm_ids)
    _validate_provenance(profile["provenance"])
    _validate_grid_block(profile["grid"])
    _validate_inference_block(profile["inference"])
    cells = profile["cells"]
    if not isinstance(cells, list) or not cells:
        raise SchemaValidationError("cells must be a non-empty list")
    seen_ids: set[str] = set()
    for index, cell in enumerate(cells):
        _validate_cell_shape(cell, index, seen_ids)


# ---------------------------------------------------------------------------
# Create-once publish (R-016/R-039: primitives consumed, not forked)
# ---------------------------------------------------------------------------


def write_profile(path: Path, profile: dict[str, Any]) -> None:
    """Create-once strict-profile writer (R-001, R-004, R-016)."""
    data = encode_profile(profile)
    validate_profile(profile)
    fileio.create_once_bytes(
        Path(path), data, exists_label="strict profile artifact"
    )


def publish_evidence_package(
    staged: Path,
    runs_root: Path,
    run_id: str,
    *,
    reclaim_crashed_relic: bool = False,
) -> Path:
    """Publish a staged evidence package into a run-scoped create-once dir
    (R-016/R-039).

    A crash between the destination's ``mkdir`` claim and the filling
    ``rename`` leaves an EMPTY run-slot relic that fails closed on every
    retry; ``reclaim_crashed_relic=True`` is the explicit recovery path.
    Every failure surfaces as a typed ``ColmAimsError`` (a pre-existing slot
    included), never a bare ``FileExistsError``.
    """
    staged = Path(staged)
    runs_root = Path(runs_root)
    if not is_path_component(run_id):
        raise ColmAimsError(
            f"run_id {run_id!r} must be a non-empty single path component"
            " (R-039)"
        )
    if not staged.is_dir():
        raise ColmAimsError("staged evidence package must be a directory")
    runs_root.mkdir(parents=True, exist_ok=True)
    dest = runs_root / run_id
    try:
        if os.stat(staged).st_dev != os.stat(runs_root).st_dev:
            raise ColmAimsError(
                "staged evidence package must reside on the same filesystem"
                " as the runs root so the create-once publish is atomic"
                " (R-016/R-039)"
            )
    except OSError as exc:
        raise ColmAimsError(
            f"cannot stat staged/runs-root for the same-filesystem check:"
            f" {exc.__class__.__name__} (R-016)"
        ) from exc
    if reclaim_crashed_relic:
        fileio.reclaim_empty_relic(dest)
    try:
        fileio.publish_dir_create_once(
            staged, dest, exists_label="evidence package run slot"
        )
    except FileExistsError as exc:
        raise ColmAimsError(
            f"evidence package run slot already exists: run_id {run_id!r} —"
            " second publish to an existing path fails closed (R-016/R-039)"
        ) from exc
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.EXDEV:
            fileio.reclaim_empty_relic(dest)
            raise ColmAimsError(
                "cross-device publish (EXDEV) — staged package is not on the"
                " runs-root filesystem; reclaimed the empty slot"
                " (R-016/R-039)"
            ) from exc
        raise
    return dest
