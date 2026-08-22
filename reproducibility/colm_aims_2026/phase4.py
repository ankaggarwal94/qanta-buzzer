"""Phase-4 PRE-run gates: eligibility, staged inputs, snapshots, parity, cert.

Spec rules owned here: R-074 (frozen pairing eligibility loader), R-075
(role-keyed model snapshot manifests + snapshot-directory verification),
R-076 (staged-input hash gates), R-077 (materialized parity comparator),
R-079 (PRE_RUN_READY certificate assembly/generation).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
("Phase-4 PRE-run repairs"); intent:
phase4_pre_run_reconciliation_2026-08-22.md sections 4-5.

Error taxonomy: loaders and gates raise ``schema.TypedIngressError``
subclasses; snapshot-directory refusals raise a ``schema.ColmAimsError``
subclass; ``compare_parity`` NEVER raises on missing regenerated fields
(failure rows, guarded builder); ``assemble_certificate`` never raises and
never emits a partial pass.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from . import pairing, schema


class EligibilityArtifactError(schema.TypedIngressError):
    """Frozen pairing-eligibility artifact failed typed validation (R-074)."""


class StagedInputError(schema.TypedIngressError):
    """A staged fit/eval input failed its fail-closed hash gate (R-076)."""


class StagedCoverageError(schema.TypedIngressError):
    """The consumed-input coverage plan could not be resolved: an uncovered
    input, an operator digest contradicting a frozen pin, or an operator
    entry outside the consumed set (F-1, R-076)."""


class SnapshotManifestError(schema.TypedIngressError):
    """Model snapshot manifest bytes failed typed validation (R-075)."""


class SnapshotMismatchError(schema.ColmAimsError):
    """A local snapshot directory deviates from its pinned manifest (R-075)."""


class ParityAnchorError(schema.TypedIngressError):
    """The committed parity anchor is malformed — fail closed, never a
    vacuous comparison over a truncated allowlist (R-077)."""


# ---------------------------------------------------------------------------
# Pinned constants (R-074/R-075/R-079)
# ---------------------------------------------------------------------------

ELIGIBILITY_ARTIFACT_TYPE = "pairing_eligibility"
ELIGIBILITY_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "derived_from",
        "eligible_count",
        "eligible_keys",
        "excluded",
        "excluded_count",
        "horizon_map",
        "horizon_map_sha256",
        "pairing_population_keyset_sha256",
    }
)
DERIVED_FROM_KEYS = frozenset(
    {
        "derivation",
        "test_dataset_basename",
        "test_dataset_sha256",
        "two_party_pin",
    }
)
EXCLUDED_ENTRY_KEYS = frozenset({"item_key", "reason"})
# R-074: the frozen artifact's cardinalities are spec-pinned (2,249 eligible
# sorted keys; 9 SINGLE_PREFIX_TRAJECTORY exclusions).
EXPECTED_ELIGIBLE_COUNT = schema.EXPECTED_COMPLETE_PAIRS
EXPECTED_EXCLUDED_COUNT = 9
# DECISION: horizon 1 contradicts the SINGLE_PREFIX_TRAJECTORY exclusion rule
# that produced the artifact — eligible horizons start at 2.
MIN_ELIGIBLE_HORIZON = 2

MANIFEST_ARTIFACT_TYPE = "model_snapshot_manifests"
MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "note",
        "offline_flags_required",
        "roles",
        "tfidf_config",
    }
)
SNAPSHOT_ROLES = frozenset({"primary_scorer", "disjoint_selector"})
ROLE_ENTRY_KEYS = frozenset({"model_name", "hf_revision", "file_count", "files"})
FILE_ENTRY_KEYS = frozenset({"sha256", "size"})
REQUIRED_OFFLINE_FLAGS = ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1")
TFIDF_CONFIG_KEYS = frozenset({"analyzer", "ngram_range", "fit_corpus"})

STAGED_ENTRY_KEYS = frozenset({"path", "expected_sha256", "label"})

PARITY_ANCHOR_ARTIFACT_TYPE = "parity_anchor"
IDENTITY_FIELDS = ("n_eval", "n_fit")
# Amended R-077 (F-3): the frozen anchor allowlist cardinalities are pinned —
# a truncated anchor must refuse, never produce a vacuous sub-194 PASS.
PARITY_ANCHOR_CARDINALITIES = (
    ("nonrandom_cells", 8),
    ("policies", 2),
    ("point_fields", 10),
    ("ci_fields", 2),
)
# 8 cells x 2 policies x (10 point + 2 CI) fields + 2 identity fields.
EXPECTED_PARITY_CHECKED = 8 * 2 * (10 + 2) + len(IDENTITY_FIELDS)

CERT_SCHEMA_VERSION = 2
CERT_COMPONENT_KEYS = (
    "repo",
    "content_hashes",
    "eligibility",
    "snapshots",
    "offline_flags",
    "staged_inputs",
    "suite_receipts",
    "parity",
    "qa012",
    "environment",
)
CONTENT_HASH_KEYS = ("producer_sha256", "verifier_sha256", "spec_sha256")
SUITE_RECEIPT_NAMES = ("focused", "full")
# R-070: every suite receipt must carry the full machine-readable binding —
# a receipt missing any of these is a failing suite_receipts component.
R070_RECEIPT_FIELDS = (
    "exit_code",
    "command",
    "environment_lock_sha256",
    "workflow_sha256",
    "interpreter_realpath",
    "counts",
    "skip_identities",
)
CERT_ENVIRONMENT_KEYS = (
    "interpreter_realpath",
    "os",
    "arch",
    "cpu",
    "blas",
    "thread_settings",
    "environment_lock_sha256",
    "command",
    "seeds",
    "pythonhashseed",
    "archived_rng_pinned",
    "fresh_rng_pinned",
)

_MISSING = object()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_json_object(
    path: Path, error_cls: type[schema.TypedIngressError]
) -> tuple[dict[str, Any], str]:
    """Bounded strict-parse of one frozen JSON artifact, version-first."""
    path = Path(path)
    rel = path.name
    data = schema.read_regular_file_bytes(path)
    try:
        obj = schema.parse_json_bytes_strict(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error_cls(f"{rel}: malformed JSON: {exc} (R-020)") from exc
    if not isinstance(obj, dict):
        raise error_cls(f"{rel}: artifact must be a JSON object (R-020)")
    # Version-first (R-059): the shared bool-safe checker runs before any
    # other key validation so mixed-invalid artifacts surface the VERSION
    # error. SchemaVersionError is itself a TypedIngressError subclass.
    schema.check_schema_version(obj, rel)
    return obj, rel


def _check_closed_keys(
    obj: dict[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
    where: str,
    error_cls: type[schema.TypedIngressError],
) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise error_cls(
            f"{where}: unknown field(s) {unknown} — no silent key-dropping"
            " (R-020/R-063)"
        )
    missing = sorted(required - set(obj))
    if missing:
        raise error_cls(f"{where}: missing required field(s) {missing}")


def _sha256_regular_file(
    path: Path, *, error_cls: type[schema.ColmAimsError], label: str
) -> str:
    """Streaming SHA-256 of one regular file, symlink-free and FIFO-safe.

    DECISION: staged inputs include multi-hundred-MB data files
    (mc_dataset.json is ~330 MB), so this deliberately does NOT reuse
    ``schema.read_regular_file_bytes`` and its 64 MB parse-artifact cap —
    the gate only hashes, never parses. The O_NOFOLLOW + S_ISREG
    discipline is identical.
    """
    path = Path(path)
    name = path.name
    if path.is_symlink():
        raise error_cls(f"{label} ({name}): refusing to hash a symlink (R-020)")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise error_cls(
                f"{label} ({name}): refusing to hash a symlink (R-020)"
            ) from exc
        raise error_cls(
            f"{label} ({name}): missing or unreadable"
            f" ({exc.__class__.__name__}) (R-076)"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise error_cls(
                f"{label} ({name}): not a regular file — refusing to hash a"
                " FIFO, device, or socket (R-020)"
            )
        if hasattr(os, "O_NONBLOCK") and hasattr(os, "get_blocking"):
            os.set_blocking(fd, True)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# R-074: frozen pairing eligibility loader
# ---------------------------------------------------------------------------


def load_pairing_eligibility(path: Path) -> dict[str, Any]:
    """Strict, closed-key, digest-recomputing load of the frozen pairing
    eligibility artifact (R-074).

    Recomputes BOTH digests — ``pairing.keyset_sha256(eligible_keys)`` and
    ``schema.horizon_map_sha256(horizon_map)`` — and compares them to the
    declared values (recompute-from-source, never mirror-equality). ANY
    mismatch or malformation raises ``EligibilityArtifactError``.
    """
    obj, rel = _load_json_object(path, EligibilityArtifactError)
    _check_closed_keys(
        obj, ELIGIBILITY_KEYS, ELIGIBILITY_KEYS, rel, EligibilityArtifactError
    )
    if obj["artifact_type"] != ELIGIBILITY_ARTIFACT_TYPE:
        raise EligibilityArtifactError(
            f"{rel}: artifact_type {obj['artifact_type']!r} is not"
            f" {ELIGIBILITY_ARTIFACT_TYPE!r} (R-074)"
        )

    keys = obj["eligible_keys"]
    if not isinstance(keys, list) or not all(
        isinstance(k, str) and k for k in keys
    ):
        raise EligibilityArtifactError(
            f"{rel}: eligible_keys must be a list of non-empty strings"
            " (R-074)"
        )
    if len(set(keys)) != len(keys):
        raise EligibilityArtifactError(
            f"{rel}: eligible_keys carries duplicate item keys (R-074)"
        )
    # keyset_sha256 sorts internally, so an out-of-order artifact would still
    # digest-match — sortedness needs its OWN check (RED contract).
    if keys != pairing.canonical_item_order(keys):
        raise EligibilityArtifactError(
            f"{rel}: eligible_keys are not sorted ascending by UTF-8 byte"
            " order (R-074)"
        )
    count = obj["eligible_count"]
    if not schema.is_real_int(count) or count != len(keys):
        raise EligibilityArtifactError(
            f"{rel}: eligible_count {count!r} does not equal the actual"
            f" eligible key count {len(keys)} — count drift (R-074)"
        )
    if count != EXPECTED_ELIGIBLE_COUNT:
        raise EligibilityArtifactError(
            f"{rel}: eligible_count {count!r} is not the frozen pinned"
            f" population {EXPECTED_ELIGIBLE_COUNT} (R-074/R-042)"
        )

    excluded = obj["excluded"]
    if not isinstance(excluded, list):
        raise EligibilityArtifactError(
            f"{rel}: excluded must be a list of exclusion entries (R-074)"
        )
    eligible_set = set(keys)
    seen_excluded: set[str] = set()
    for index, entry in enumerate(excluded):
        where = f"{rel}: excluded[{index}]"
        if not isinstance(entry, dict):
            raise EligibilityArtifactError(f"{where}: must be an object")
        _check_closed_keys(
            entry,
            EXCLUDED_ENTRY_KEYS,
            EXCLUDED_ENTRY_KEYS,
            where,
            EligibilityArtifactError,
        )
        item_key = entry["item_key"]
        if not isinstance(item_key, str) or not item_key:
            raise EligibilityArtifactError(
                f"{where}: item_key must be a non-empty string (R-074)"
            )
        if item_key in eligible_set:
            raise EligibilityArtifactError(
                f"{where}: excluded item also appears in eligible_keys"
                " (R-074/R-008)"
            )
        if item_key in seen_excluded:
            raise EligibilityArtifactError(
                f"{where}: duplicate excluded item key (R-074/R-008)"
            )
        seen_excluded.add(item_key)
        reason = entry["reason"]
        if (
            reason not in schema.EXCLUSION_REASONS
            or reason != schema.SINGLE_PREFIX_TRAJECTORY
        ):
            raise EligibilityArtifactError(
                f"{where}: reason {reason!r} is not the enumerated frozen"
                f" derivation reason {schema.SINGLE_PREFIX_TRAJECTORY!r}"
                " (R-074)"
            )
    excluded_count = obj["excluded_count"]
    if not schema.is_real_int(excluded_count) or excluded_count != len(excluded):
        raise EligibilityArtifactError(
            f"{rel}: excluded_count {excluded_count!r} does not equal the"
            f" actual excluded entry count {len(excluded)} — count drift"
            " (R-074)"
        )
    if excluded_count != EXPECTED_EXCLUDED_COUNT:
        raise EligibilityArtifactError(
            f"{rel}: excluded_count {excluded_count!r} is not the frozen"
            f" pinned exclusion count {EXPECTED_EXCLUDED_COUNT} (R-074)"
        )

    horizon_map = obj["horizon_map"]
    if not isinstance(horizon_map, dict):
        raise EligibilityArtifactError(
            f"{rel}: horizon_map must be an object (R-073/R-074)"
        )
    if set(horizon_map) != eligible_set:
        raise EligibilityArtifactError(
            f"{rel}: horizon_map keys do not cover exactly the eligible"
            " keys (R-073/R-074)"
        )
    for value in horizon_map.values():
        if not schema.is_real_int(value):
            raise EligibilityArtifactError(
                f"{rel}: horizon_map value {value!r} is outside the"
                " positive-int domain — bools never satisfy an integer"
                " domain (R-061/R-073)"
            )
        if value < MIN_ELIGIBLE_HORIZON:
            raise EligibilityArtifactError(
                f"{rel}: horizon_map value {value!r} is below the minimum"
                f" eligible horizon {MIN_ELIGIBLE_HORIZON} — a sub-2 horizon"
                " contradicts the SINGLE_PREFIX_TRAJECTORY exclusion rule"
                " that produced this artifact (R-074)"
            )

    declared_horizon_digest = obj["horizon_map_sha256"]
    if not schema.is_sha256_hex(declared_horizon_digest):
        raise EligibilityArtifactError(
            f"{rel}: horizon_map_sha256 is not a lowercase sha256 hex digest"
            " (R-073)"
        )
    try:
        recomputed_horizon = schema.horizon_map_sha256(horizon_map)
    except schema.ColmAimsError as exc:
        raise EligibilityArtifactError(f"{rel}: {exc}") from exc
    if recomputed_horizon != declared_horizon_digest:
        raise EligibilityArtifactError(
            f"{rel}: recomputed horizon-map digest {recomputed_horizon} !="
            f" declared horizon_map_sha256 {declared_horizon_digest}"
            " (R-073/R-074)"
        )

    declared_keyset_digest = obj["pairing_population_keyset_sha256"]
    if not schema.is_sha256_hex(declared_keyset_digest):
        raise EligibilityArtifactError(
            f"{rel}: pairing_population_keyset_sha256 is not a lowercase"
            " sha256 hex digest (R-074)"
        )
    recomputed_keyset = pairing.keyset_sha256(keys)
    if recomputed_keyset != declared_keyset_digest:
        raise EligibilityArtifactError(
            f"{rel}: recomputed eligible-keyset digest {recomputed_keyset}"
            f" != declared pairing_population_keyset_sha256"
            f" {declared_keyset_digest} (R-074/R-052)"
        )

    derived = obj["derived_from"]
    if not isinstance(derived, dict):
        raise EligibilityArtifactError(
            f"{rel}: derived_from must be an object (R-074)"
        )
    _check_closed_keys(
        derived,
        DERIVED_FROM_KEYS,
        DERIVED_FROM_KEYS,
        f"{rel}: derived_from",
        EligibilityArtifactError,
    )
    if not schema.is_sha256_hex(derived["test_dataset_sha256"]):
        raise EligibilityArtifactError(
            f"{rel}: derived_from.test_dataset_sha256 is not a sha256 digest"
            " (R-074)"
        )
    for field in ("derivation", "test_dataset_basename", "two_party_pin"):
        if not isinstance(derived[field], str) or not derived[field]:
            raise EligibilityArtifactError(
                f"{rel}: derived_from.{field} must be a non-empty string"
                " (R-074)"
            )
    return obj


# ---------------------------------------------------------------------------
# R-076: staged-input hash gate (fail-closed, before any loader)
# ---------------------------------------------------------------------------


def staged_input_gate(staged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hash-verify EVERY staged fit/eval input fail-closed (R-076).

    Raises ``StagedInputError`` on the FIRST (list-order) missing file,
    hash mismatch, or malformed entry, naming the file plus the expected
    and observed digests. An EMPTY staged list raises — a gate over zero
    inputs is a vacuously-empty authoritative set, not a trivially-passing
    gate. Returns fresh entries carrying ``observed_sha256``.
    """
    if not isinstance(staged, list):
        raise StagedInputError(
            "staged inputs must be a list of {path, expected_sha256, label}"
            " entries (R-076)"
        )
    if not staged:
        raise StagedInputError(
            "staged-input gate invoked over ZERO inputs — an empty"
            " authoritative input set is a defect, never a trivially"
            " passing gate (R-076)"
        )
    verified: list[dict[str, Any]] = []
    for index, entry in enumerate(staged):
        where = f"staged input [{index}]"
        if not isinstance(entry, dict):
            raise StagedInputError(f"{where}: entry must be an object (R-076)")
        _check_closed_keys(
            entry, STAGED_ENTRY_KEYS, STAGED_ENTRY_KEYS, where, StagedInputError
        )
        label = entry["label"]
        if not isinstance(label, str) or not label:
            raise StagedInputError(
                f"{where}: label must be a non-empty string (R-076)"
            )
        expected = entry["expected_sha256"]
        path = Path(entry["path"])
        if not schema.is_sha256_hex(expected):
            raise StagedInputError(
                f"staged input {label!r} ({path.name}): expected_sha256"
                f" {expected!r} is not a lowercase 64-hex sha256 digest"
                " (R-076)"
            )
        observed = _sha256_regular_file(
            path, error_cls=StagedInputError, label=f"staged input {label!r}"
        )
        if observed != expected:
            raise StagedInputError(
                f"staged input {label!r} ({path.name}): observed sha256"
                f" {observed} != expected {expected} — refusing to proceed"
                " to any loader or model construction (R-076)"
            )
        verified.append(
            {
                "label": label,
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
            }
        )
    return verified


def required_staged_coverage(
    consumed: list[dict[str, Any]], staged_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve the producer's consumed-input enumeration into the full
    staged-gate plan (F-1, R-076).

    ``consumed`` is the ordered enumeration of every fit/eval input as
    ``{"label", "path", "frozen_sha256": str | None}``; ``staged_entries``
    are the operator's ``--staged-input`` triples
    ``{"label", "path", "expected_sha256"}``. Returns one plan entry per
    consumed input, IN CONSUMED ORDER, each ``{"label", "path",
    "expected_sha256"}`` with the expected digest filled from the frozen
    pin when present (frozen-pin-wins), else from the operator entry
    covering the same path.

    Typed refusals (``StagedCoverageError``): (a) a consumed input with
    neither a frozen pin nor an operator digest (uncovered — named); (b) an
    operator digest CONTRADICTING a frozen pin (names the file and BOTH
    digests); (c) an operator entry naming a path outside the consumed set
    (unknown staged input — silently-ignored coverage is a defect); plus
    malformed shapes, duplicate contradictory operator entries, and the
    vacuously-empty consumed enumeration.
    """
    if not isinstance(consumed, list) or not consumed:
        raise StagedCoverageError(
            "consumed-input enumeration must be a non-empty list — a gate"
            " plan over ZERO consumed inputs is a vacuously-empty"
            " authoritative set (F-1/R-076)"
        )
    if not isinstance(staged_entries, list):
        raise StagedCoverageError(
            "staged entries must be a list of {label, path, expected_sha256}"
            " triples (F-1/R-076)"
        )

    # Operator entries keyed by resolved path; duplicate entries for one
    # path must agree (duplicate-key last-wins is a laundering vector).
    operator_by_path: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(staged_entries):
        where = f"staged entry [{index}]"
        if not isinstance(entry, dict):
            raise StagedCoverageError(f"{where}: entry must be an object (F-1)")
        _check_closed_keys(
            entry,
            STAGED_ENTRY_KEYS,
            STAGED_ENTRY_KEYS,
            where,
            StagedCoverageError,
        )
        label = entry["label"]
        if not isinstance(label, str) or not label:
            raise StagedCoverageError(
                f"{where}: label must be a non-empty string (F-1)"
            )
        digest = entry["expected_sha256"]
        if not schema.is_sha256_hex(digest):
            raise StagedCoverageError(
                f"staged entry {label!r}: expected_sha256 {digest!r} is not"
                " a lowercase 64-hex sha256 digest (F-1/R-076)"
            )
        resolved = str(Path(entry["path"]).resolve())
        prior = operator_by_path.get(resolved)
        if prior is not None and prior["expected_sha256"] != digest:
            raise StagedCoverageError(
                f"staged entries {prior['label']!r} and {label!r} both cover"
                f" {Path(resolved).name} with contradictory digests"
                f" {prior['expected_sha256']} != {digest} (F-1/R-076)"
            )
        operator_by_path[resolved] = {
            "label": label,
            "expected_sha256": digest,
        }

    plan: list[dict[str, Any]] = []
    consumed_paths: set[str] = set()
    for index, entry in enumerate(consumed):
        where = f"consumed input [{index}]"
        if not isinstance(entry, dict):
            raise StagedCoverageError(f"{where}: entry must be an object (F-1)")
        unknown = sorted(set(entry) - {"label", "path", "frozen_sha256"})
        if unknown:
            raise StagedCoverageError(
                f"{where}: unknown field(s) {unknown} — consumed entries are"
                " exactly {{label, path, frozen_sha256}} (F-1)"
            )
        missing = sorted({"label", "path", "frozen_sha256"} - set(entry))
        if missing:
            raise StagedCoverageError(
                f"{where}: missing required field(s) {missing} (F-1)"
            )
        label = entry["label"]
        if not isinstance(label, str) or not label:
            raise StagedCoverageError(
                f"{where}: label must be a non-empty string (F-1)"
            )
        path = Path(entry["path"])
        resolved = str(path.resolve())
        if resolved in consumed_paths:
            raise StagedCoverageError(
                f"consumed input {label!r} ({path.name}): duplicate consumed"
                " path — one gate entry per input (F-1)"
            )
        consumed_paths.add(resolved)
        frozen = entry["frozen_sha256"]
        if frozen is not None and not schema.is_sha256_hex(frozen):
            raise StagedCoverageError(
                f"consumed input {label!r} ({path.name}): frozen_sha256"
                f" {frozen!r} is neither None nor a lowercase 64-hex sha256"
                " digest (F-1)"
            )
        operator = operator_by_path.get(resolved)
        if frozen is not None:
            # Frozen-pin-wins: an agreeing operator entry is redundant; a
            # contradicting one is a laundering attempt and refuses loudly.
            if operator is not None and operator["expected_sha256"] != frozen:
                raise StagedCoverageError(
                    f"consumed input {label!r} ({path.name}): operator digest"
                    f" {operator['expected_sha256']} CONTRADICTS the frozen"
                    f" pin {frozen} — frozen pins are two-party and never"
                    " overridable (F-1/R-076)"
                )
            expected = frozen
        elif operator is not None:
            expected = operator["expected_sha256"]
        else:
            raise StagedCoverageError(
                f"consumed input {label!r} ({path.name}): UNCOVERED — no"
                " frozen pin and no --staged-input digest; every consumed"
                " fit/eval input must be hash-gated (F-1/R-076)"
            )
        plan.append(
            {"label": label, "path": path, "expected_sha256": expected}
        )

    uncovered_operators = sorted(
        set(operator_by_path) - consumed_paths
    )
    if uncovered_operators:
        names = [Path(p).name for p in uncovered_operators]
        raise StagedCoverageError(
            f"staged entr{'y' if len(names) == 1 else 'ies'} for"
            f" {names} name path(s) OUTSIDE the consumed-input set —"
            " unknown staged inputs are a defect, never silently-ignored"
            " coverage (F-1/R-076)"
        )
    return plan


# ---------------------------------------------------------------------------
# R-075: role-keyed model snapshot manifests + snapshot verification
# ---------------------------------------------------------------------------


def _is_hf_revision(value: Any) -> bool:
    """HF snapshot revisions are 40-hex git SHAs; 64-hex content digests are
    also admissible pins."""
    return schema.is_commit_sha(value) or schema.is_sha256_hex(value)


def _validate_role_entry(
    entry: Any,
    where: str,
    error_cls: type[schema.ColmAimsError],
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise error_cls(f"{where}: role entry must be an object (R-075)")
    unknown = sorted(set(entry) - ROLE_ENTRY_KEYS)
    if unknown:
        raise error_cls(
            f"{where}: unknown role-entry field(s) {unknown} (R-075)"
        )
    missing = sorted(ROLE_ENTRY_KEYS - set(entry))
    if missing:
        raise error_cls(
            f"{where}: missing role-entry field(s) {missing} (R-075)"
        )
    if not isinstance(entry["model_name"], str) or not entry["model_name"]:
        raise error_cls(f"{where}: model_name must be a non-empty string")
    if not _is_hf_revision(entry["hf_revision"]):
        raise error_cls(
            f"{where}: hf_revision is not an immutable 40-hex git SHA or"
            " 64-hex content digest (R-075)"
        )
    files = entry["files"]
    if not isinstance(files, dict) or not files:
        raise error_cls(
            f"{where}: files must be a NON-EMPTY per-file manifest — an"
            " empty manifest is a vacuously-empty authoritative set (R-075)"
        )
    for rel_name, meta in files.items():
        file_where = f"{where}: files[{rel_name!r}]"
        if not isinstance(rel_name, str) or not rel_name:
            raise error_cls(f"{file_where}: file name must be a non-empty string")
        parts = rel_name.split("/")
        if not all(schema.is_path_component(part) for part in parts):
            raise error_cls(
                f"{file_where}: file name must be a relative path of plain"
                " components (no traversal, no absolute paths) (R-020)"
            )
        if not isinstance(meta, dict):
            raise error_cls(f"{file_where}: entry must be an object")
        unknown_meta = sorted(set(meta) - FILE_ENTRY_KEYS)
        missing_meta = sorted(FILE_ENTRY_KEYS - set(meta))
        if unknown_meta or missing_meta:
            raise error_cls(
                f"{file_where}: entry keys must be exactly"
                f" {sorted(FILE_ENTRY_KEYS)}"
            )
        if not schema.is_sha256_hex(meta["sha256"]):
            raise error_cls(f"{file_where}: sha256 is not a sha256 hex digest")
        if not schema.is_real_int(meta["size"]) or meta["size"] <= 0:
            raise error_cls(
                f"{file_where}: size must be a positive real integer —"
                " bools never satisfy an integer domain (R-061)"
            )
    file_count = entry["file_count"]
    if not schema.is_real_int(file_count) or file_count != len(files):
        raise error_cls(
            f"{where}: file_count {file_count!r} does not equal the actual"
            f" per-file manifest size {len(files)} (R-075)"
        )
    return entry


def load_model_snapshot_manifest(path: Path) -> dict[str, Any]:
    """Strict load of the frozen role-keyed model snapshot manifest (R-075)."""
    obj, rel = _load_json_object(path, SnapshotManifestError)
    _check_closed_keys(
        obj, MANIFEST_KEYS, MANIFEST_KEYS, rel, SnapshotManifestError
    )
    if obj["artifact_type"] != MANIFEST_ARTIFACT_TYPE:
        raise SnapshotManifestError(
            f"{rel}: artifact_type {obj['artifact_type']!r} is not"
            f" {MANIFEST_ARTIFACT_TYPE!r} (R-075)"
        )
    if not isinstance(obj["note"], str):
        raise SnapshotManifestError(f"{rel}: note must be a string (R-075)")
    roles = obj["roles"]
    if not isinstance(roles, dict) or set(roles) != SNAPSHOT_ROLES:
        raise SnapshotManifestError(
            f"{rel}: roles must be exactly {sorted(SNAPSHOT_ROLES)} — a"
            " missing or extra role fails closed (R-075)"
        )
    for role in sorted(SNAPSHOT_ROLES):
        _validate_role_entry(
            roles[role], f"{rel}: roles[{role!r}]", SnapshotManifestError
        )
    if obj["offline_flags_required"] != list(REQUIRED_OFFLINE_FLAGS):
        raise SnapshotManifestError(
            f"{rel}: offline_flags_required must be exactly"
            f" {list(REQUIRED_OFFLINE_FLAGS)} (R-075)"
        )
    tfidf = obj["tfidf_config"]
    if not isinstance(tfidf, dict) or set(tfidf) != TFIDF_CONFIG_KEYS:
        raise SnapshotManifestError(
            f"{rel}: tfidf_config keys must be exactly"
            f" {sorted(TFIDF_CONFIG_KEYS)} (R-075)"
        )
    if not isinstance(tfidf["analyzer"], str) or not tfidf["analyzer"]:
        raise SnapshotManifestError(
            f"{rel}: tfidf_config.analyzer must be a non-empty string"
        )
    ngram = tfidf["ngram_range"]
    if (
        not isinstance(ngram, list)
        or len(ngram) != 2
        or not all(schema.is_real_int(v) and v > 0 for v in ngram)
    ):
        raise SnapshotManifestError(
            f"{rel}: tfidf_config.ngram_range must be a 2-list of positive"
            " integers (R-075)"
        )
    if not isinstance(tfidf["fit_corpus"], str) or not tfidf["fit_corpus"]:
        raise SnapshotManifestError(
            f"{rel}: tfidf_config.fit_corpus must be a non-empty string"
        )
    return obj


def verify_snapshot_dir(
    manifest_role_entry: dict[str, Any], snapshot_dir: Path
) -> None:
    """Verify one local snapshot directory byte-for-byte against its pinned
    manifest entry (R-075): per-file sha256 AND size, no extra files, no
    missing files, file_count consistency. Raises ``SnapshotMismatchError``
    naming the offending relative path on any deviation.
    """
    entry = _validate_role_entry(
        manifest_role_entry, "snapshot manifest entry", SnapshotManifestError
    )
    snapshot_dir = Path(snapshot_dir)
    if snapshot_dir.is_symlink():
        raise SnapshotMismatchError(
            f"snapshot directory {snapshot_dir.name!r} is a symlink —"
            " refusing (R-075/R-013)"
        )
    if not snapshot_dir.is_dir():
        raise SnapshotMismatchError(
            f"snapshot directory {snapshot_dir.name!r} does not exist or is"
            " not a directory (R-075)"
        )
    declared: dict[str, dict[str, Any]] = entry["files"]
    observed: dict[str, Path] = {}
    for member in sorted(snapshot_dir.rglob("*")):
        rel_name = member.relative_to(snapshot_dir).as_posix()
        if member.is_symlink():
            # Symlinked tree members hash bytes from OUTSIDE the tree —
            # refuse (seed catalog: DoS/containment trio).
            raise SnapshotMismatchError(
                f"snapshot member {rel_name!r} is a symlink — refusing"
                " (R-075/R-013)"
            )
        if member.is_dir():
            continue
        if not member.is_file():
            raise SnapshotMismatchError(
                f"snapshot member {rel_name!r} is not a regular file (R-020)"
            )
        observed[rel_name] = member
    extra = sorted(set(observed) - set(declared))
    if extra:
        raise SnapshotMismatchError(
            f"snapshot carries undeclared file {extra[0]!r} — an extra file"
            " is a mismatch even when every declared file checks (R-075)"
        )
    missing = sorted(set(declared) - set(observed))
    if missing:
        raise SnapshotMismatchError(
            f"snapshot is missing declared file {missing[0]!r} (R-075)"
        )
    for rel_name in sorted(declared):
        meta = declared[rel_name]
        member = observed[rel_name]
        actual_size = os.lstat(member).st_size
        if actual_size != meta["size"]:
            # Size is a REAL check, independently of the content hash — a
            # manifest with the correct sha but wrong declared size fails.
            raise SnapshotMismatchError(
                f"snapshot file {rel_name!r} size {actual_size} != declared"
                f" size {meta['size']} (R-075)"
            )
        actual_sha = _sha256_regular_file(
            member,
            error_cls=SnapshotMismatchError,
            label=f"snapshot file {rel_name!r}",
        )
        if actual_sha != meta["sha256"]:
            raise SnapshotMismatchError(
                f"snapshot file {rel_name!r} sha256 {actual_sha} != declared"
                f" {meta['sha256']} (R-075)"
            )
    return None


# ---------------------------------------------------------------------------
# R-077: materialized parity comparator
# ---------------------------------------------------------------------------


def _values_equal(expected: Any, observed: Any) -> bool:
    """Exact parsed-JSON-value equality AT THE SAME JSON TYPE (amended R-077).

    ``True == 1`` / ``False == 0`` are Python-equal — a bool on either side
    matches ONLY a bool of the same value (seed catalog: bool laundering).
    ``2249 == 2249.0`` is Python-equal — an int drifting to float (or vice
    versa) is a serialization-identity change and MUST fail: no cross-type
    numeric laundering. Non-finite floats never compare equal. Lists compare
    element-wise under the same rules.
    """
    if isinstance(expected, bool) or isinstance(observed, bool):
        return (
            isinstance(expected, bool)
            and isinstance(observed, bool)
            and expected == observed
        )
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if type(expected) is not type(observed):
            return False
        if isinstance(expected, float) and (
            not math.isfinite(expected) or not math.isfinite(observed)
        ):
            return False
        return expected == observed
    if isinstance(expected, list) or isinstance(observed, list):
        if not (isinstance(expected, list) and isinstance(observed, list)):
            return False
        if len(expected) != len(observed):
            return False
        return all(_values_equal(e, o) for e, o in zip(expected, observed))
    return type(expected) is type(observed) and expected == observed


def _validate_parity_anchor(anchor: Any) -> dict[str, Any]:
    """Fail closed on a malformed ANCHOR: a truncated allowlist must never
    produce a vacuous comparison. (The REGENERATED side, by contrast, never
    raises — absences become failure rows.)"""
    anchor = _as_dict(anchor)
    required = (
        "nonrandom_cells",
        "policies",
        "point_fields",
        "ci_fields",
        "expected",
        "identity_fields",
        "random_k",
    )
    missing = sorted(k for k in required if k not in anchor)
    if missing:
        raise ParityAnchorError(
            f"parity anchor missing required field(s) {missing} (R-077)"
        )
    for list_field in ("nonrandom_cells", "policies", "point_fields", "ci_fields"):
        value = anchor[list_field]
        if not isinstance(value, list) or not value or not all(
            isinstance(v, str) and v for v in value
        ):
            raise ParityAnchorError(
                f"parity anchor {list_field} must be a non-empty list of"
                " strings — an empty allowlist axis is a vacuously-empty"
                " authoritative set (R-077)"
            )
    # Amended R-077 (F-3): the allowlist cardinalities are HARD-PINNED —
    # exactly 8 nonrandom cells x 2 policies x 10 point fields x 2 CI
    # fields. A truncated (or padded, or duplicate-carrying) anchor must
    # refuse: a comparison over fewer than the full 194-field allowlist can
    # never produce a vacuous PASS.
    for field_name, expected_count in PARITY_ANCHOR_CARDINALITIES:
        entries = anchor[field_name]
        if len(entries) != expected_count:
            raise ParityAnchorError(
                f"parity anchor {field_name} has {len(entries)} entries —"
                f" the frozen allowlist pins exactly {expected_count};"
                " a truncated anchor never yields a vacuous PASS"
                " (amended R-077/F-3)"
            )
        if len(set(entries)) != len(entries):
            raise ParityAnchorError(
                f"parity anchor {field_name} carries duplicate entries —"
                " duplicates shrink the effective allowlist under a"
                " full-looking count (amended R-077/F-3)"
            )
    if not isinstance(anchor["expected"], dict):
        raise ParityAnchorError("parity anchor expected block must be an object")
    identity = anchor["identity_fields"]
    if not isinstance(identity, dict) or any(
        f not in identity for f in IDENTITY_FIELDS
    ):
        raise ParityAnchorError(
            f"parity anchor identity_fields must carry {list(IDENTITY_FIELDS)}"
            " (R-077)"
        )
    rk = anchor["random_k"]
    if not isinstance(rk, dict) or not isinstance(rk.get("cells"), list):
        raise ParityAnchorError(
            "parity anchor random_k block must carry the informational cell"
            " list (R-077)"
        )
    return anchor


def compare_parity(
    anchor: dict[str, Any], regenerated_export: dict[str, Any]
) -> dict[str, Any]:
    """R-077 materialized parity comparison: anchor allowlist vs a
    producer-payload-shaped regenerated export.

    Checks all 160 nonrandom point fields + all 32 nonrandom CI arrays
    (every element) + the 2 population identity fields (n_eval, n_fit) —
    ``checked == 194`` against the frozen anchor. ANY mismatch, including
    any CI-array element, is a blocking FAIL. Missing cells/policies/fields
    become failure rows, never exceptions (guarded builder). The two
    Random-K cells are exempt from historical parity and reported
    informationally.
    """
    anchor = _validate_parity_anchor(anchor)
    regenerated = _as_dict(regenerated_export)
    metadata = _as_dict(regenerated.get("metadata"))
    results = _as_dict(regenerated.get("results"))

    failures: list[dict[str, Any]] = []
    checked = 0

    # Identity fields (failure rows carry cell=None, policy=None).
    identity = _as_dict(anchor["identity_fields"])
    for field in IDENTITY_FIELDS:
        checked += 1
        expected_value = identity.get(field)
        observed_value = metadata.get(field, _MISSING)
        if observed_value is _MISSING or not _values_equal(
            expected_value, observed_value
        ):
            failures.append(
                {
                    "cell": None,
                    "policy": None,
                    "field": field,
                    "expected": expected_value,
                    "observed": (
                        None if observed_value is _MISSING else observed_value
                    ),
                }
            )

    expected_block = _as_dict(anchor["expected"])
    point_fields = list(anchor["point_fields"])
    ci_fields = list(anchor["ci_fields"])
    for cell in anchor["nonrandom_cells"]:
        expected_policies = _as_dict(expected_block.get(cell))
        observed_policies = _as_dict(results.get(cell))
        for policy in anchor["policies"]:
            expected_values = _as_dict(expected_policies.get(policy))
            observed_values = _as_dict(observed_policies.get(policy))
            for field in point_fields + ci_fields:
                checked += 1
                expected_value = expected_values.get(field, _MISSING)
                if expected_value is _MISSING:
                    raise ParityAnchorError(
                        f"parity anchor expected[{cell!r}][{policy!r}] is"
                        f" missing allowlisted field {field!r} — the anchor"
                        " allowlist must be complete (R-077)"
                    )
                observed_value = observed_values.get(field, _MISSING)
                if observed_value is _MISSING or not _values_equal(
                    expected_value, observed_value
                ):
                    failures.append(
                        {
                            "cell": cell,
                            "policy": policy,
                            "field": field,
                            "expected": expected_value,
                            "observed": (
                                None
                                if observed_value is _MISSING
                                else observed_value
                            ),
                        }
                    )

    # Random-K cells: NEVER blocking; informational report only (R-077).
    rk = _as_dict(anchor["random_k"])
    archived = _as_dict(rk.get("informational_archived_values"))
    divergences: list[dict[str, Any]] = []
    rk_compared = 0
    for cell in rk.get("cells", []):
        archived_policies = _as_dict(archived.get(cell))
        observed_policies = _as_dict(results.get(cell))
        for policy in anchor["policies"]:
            archived_values = _as_dict(archived_policies.get(policy))
            observed_values = _as_dict(observed_policies.get(policy))
            for field in point_fields + ci_fields:
                if field not in archived_values:
                    continue
                rk_compared += 1
                archived_value = archived_values.get(field)
                observed_value = observed_values.get(field, _MISSING)
                if observed_value is _MISSING or not _values_equal(
                    archived_value, observed_value
                ):
                    divergences.append(
                        {
                            "cell": cell,
                            "policy": policy,
                            "field": field,
                            "archived": archived_value,
                            "regenerated": (
                                None
                                if observed_value is _MISSING
                                else observed_value
                            ),
                        }
                    )
    random_k_informational = {
        "cells": [str(c) for c in rk.get("cells", [])],
        "exempt_from_historical_parity": True,
        "archived_rng_pinned": rk.get("archived_rng_pinned"),
        "fresh_rng_pinned": rk.get("fresh_rng_pinned"),
        "compared": rk_compared,
        "divergences": divergences,
    }

    # Amended R-077 (F-3): PASS additionally requires the full 194-field
    # allowlist to have been checked — belt-and-braces behind the anchor
    # cardinality pins; a sub-allowlist comparison can never PASS.
    verdict = (
        "PASS"
        if not failures and checked == EXPECTED_PARITY_CHECKED
        else "FAIL"
    )
    return {
        "verdict": verdict,
        "checked": checked,
        "failures": failures,
        "random_k_informational": random_k_informational,
    }


# ---------------------------------------------------------------------------
# R-079: PRE_RUN_READY certificate (pure core + thin generator)
# ---------------------------------------------------------------------------


def _is_git_object_id(value: Any) -> bool:
    """R-079 (SPEC_ISSUE-1 adjudication 2026-08-22): repo commit and tree
    bind the repository's NATIVE git object ids — 40-hex SHA-1 and 64-hex
    SHA-256 object formats are both admissible, lowercase hex, fixed
    length."""
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(char in "0123456789abcdef" for char in value)
    )


def _check_repo(repo: Any, fail: Any) -> None:
    if not isinstance(repo, dict):
        fail("repo: component must be an object")
        return
    if repo.get("dirty") is not False:
        fail(
            "repo: dirty must be exactly False (clean-state proof); found"
            f" {repo.get('dirty')!r}"
        )
    if not _is_git_object_id(repo.get("commit")):
        fail(
            "repo: commit is not a native git object id (40- or 64-hex"
            " lowercase)"
        )
    if not _is_git_object_id(repo.get("tree_sha256")):
        fail(
            "repo: tree_sha256 is not a native git object id (40- or 64-hex"
            " lowercase)"
        )


def _check_content_hashes(hashes: Any, fail: Any) -> None:
    if not isinstance(hashes, dict):
        fail("content_hashes: component must be an object")
        return
    for key in CONTENT_HASH_KEYS:
        if not schema.is_sha256_hex(hashes.get(key)):
            fail(f"content_hashes: {key} is not a sha256 hex digest")


def _check_eligibility(eligibility: Any, fail: Any) -> None:
    if not isinstance(eligibility, dict):
        fail("eligibility: component must be an object")
        return
    if not schema.is_sha256_hex(eligibility.get("digest")):
        fail("eligibility: digest is not a sha256 hex digest")
    if not schema.is_sha256_hex(eligibility.get("horizon_map_sha256")):
        fail("eligibility: horizon_map_sha256 is not a sha256 hex digest")


def _check_snapshots(snapshots: Any, fail: Any) -> None:
    if not isinstance(snapshots, dict):
        fail("snapshots: component must be an object")
        return
    if set(snapshots) != SNAPSHOT_ROLES:
        fail(
            f"snapshots: roles must be exactly {sorted(SNAPSHOT_ROLES)};"
            f" found {sorted(map(str, snapshots))}"
        )
    for role in sorted(SNAPSHOT_ROLES):
        entry = snapshots.get(role)
        if not isinstance(entry, dict):
            fail(f"snapshots: {role} snapshot entry missing or malformed")
            continue
        if entry.get("verified") is not True:
            fail(
                f"snapshots: {role} snapshot verified must be exactly True;"
                f" found {entry.get('verified')!r}"
            )
        if not isinstance(entry.get("model_name"), str) or not entry.get(
            "model_name"
        ):
            fail(f"snapshots: {role} snapshot model_name missing")
        if not _is_hf_revision(entry.get("hf_revision")):
            fail(f"snapshots: {role} snapshot hf_revision is not a valid pin")


def _check_offline_flags(flags: Any, fail: Any) -> None:
    if flags != list(REQUIRED_OFFLINE_FLAGS):
        fail(
            "offline_flags: must be exactly"
            f" {list(REQUIRED_OFFLINE_FLAGS)}; found {flags!r}"
        )


def _check_staged_inputs(staged: Any, fail: Any) -> None:
    if not isinstance(staged, list):
        fail("staged_inputs: component must be a list")
        return
    if not staged:
        fail(
            "staged_inputs: empty staged-input set — a vacuously-empty"
            " authoritative set is a defect"
        )
        return
    for index, entry in enumerate(staged):
        if not isinstance(entry, dict):
            fail(f"staged_inputs: entry [{index}] must be an object")
            continue
        label = entry.get("label")
        name = label if isinstance(label, str) and label else f"[{index}]"
        expected = entry.get("expected_sha256")
        observed = entry.get("observed_sha256")
        if not schema.is_sha256_hex(expected):
            fail(f"staged_inputs: {name}: expected_sha256 is not a sha256 digest")
        if not schema.is_sha256_hex(observed):
            fail(
                f"staged_inputs: {name}: observed_sha256 missing or not a"
                " sha256 digest — a missing observation is never a pass"
            )
        elif schema.is_sha256_hex(expected) and observed != expected:
            fail(
                f"staged_inputs: {name}: observed sha256 {observed} !="
                f" expected {expected}"
            )
        if not isinstance(entry.get("path"), str) or not entry.get("path"):
            fail(f"staged_inputs: {name}: path missing")


def _check_suite_receipts(receipts: Any, fail: Any) -> None:
    if not isinstance(receipts, dict):
        fail("suite_receipts: component must be an object")
        return
    for name in SUITE_RECEIPT_NAMES:
        receipt = receipts.get(name)
        if not isinstance(receipt, dict):
            fail(f"suite_receipts: {name} receipt missing or malformed")
            continue
        # R-070: the full machine-readable receipt binding is REQUIRED —
        # a receipt missing any field is a failing suite_receipts component.
        for field in R070_RECEIPT_FIELDS:
            if field not in receipt:
                fail(
                    f"suite_receipts: {name} receipt is missing the R-070"
                    f" field {field!r}"
                )
        exit_code = receipt.get("exit_code")
        # Bool-guard: False == 0 in Python; only the exact int 0 is success.
        if type(exit_code) is not int or exit_code != 0:
            fail(
                f"suite_receipts: {name} exit_code must be exactly int 0;"
                f" found {exit_code!r}"
            )
        command = receipt.get("command")
        if not command or not isinstance(command, (str, list)):
            fail(f"suite_receipts: {name} command missing")
        if "environment_lock_sha256" in receipt and not schema.is_sha256_hex(
            receipt["environment_lock_sha256"]
        ):
            fail(
                f"suite_receipts: {name} environment_lock_sha256 is not a"
                " sha256 hex digest (R-070: a HASH, not a metadata object)"
            )
        if "workflow_sha256" in receipt and not schema.is_sha256_hex(
            receipt["workflow_sha256"]
        ):
            fail(
                f"suite_receipts: {name} workflow_sha256 is not a sha256"
                " hex digest (R-070)"
            )
        if "interpreter_realpath" in receipt and (
            not isinstance(receipt["interpreter_realpath"], str)
            or not receipt["interpreter_realpath"]
        ):
            fail(
                f"suite_receipts: {name} interpreter_realpath must be a"
                " non-empty string (R-070)"
            )
        if "counts" in receipt and not isinstance(receipt["counts"], dict):
            fail(
                f"suite_receipts: {name} counts must be a machine-readable"
                " object (R-070)"
            )
        if "skip_identities" in receipt and not isinstance(
            receipt["skip_identities"], list
        ):
            fail(
                f"suite_receipts: {name} skip_identities must be a list"
                " (R-070)"
            )


def _check_parity(parity: Any, fail: Any) -> None:
    if not isinstance(parity, dict):
        fail("parity: component must be an object")
        return
    identity = parity.get("comparator_identity")
    if not isinstance(identity, str) or not identity:
        fail("parity: comparator_identity must be a non-empty string")
    if not schema.is_sha256_hex(parity.get("anchor_sha256")):
        fail("parity: anchor_sha256 is not a sha256 hex digest")


def _check_qa012(qa012: Any, fail: Any) -> None:
    if not isinstance(qa012, dict):
        fail("qa012: component must be an object")
        return
    if not schema.is_sha256_hex(qa012.get("rev2_manifest_sha256")):
        fail("qa012: rev2_manifest_sha256 is not a sha256 hex digest")


def _check_environment(env: Any, fail: Any) -> None:
    if not isinstance(env, dict):
        fail("environment: component must be an object")
        return
    for key in CERT_ENVIRONMENT_KEYS:
        if key not in env:
            fail(f"environment: required field {key!r} missing")
    for key in ("interpreter_realpath", "os", "arch", "cpu", "blas"):
        if key in env and (not isinstance(env[key], str) or not env[key]):
            fail(f"environment: {key} must be a non-empty string")
    if "thread_settings" in env and not isinstance(env["thread_settings"], dict):
        fail("environment: thread_settings must be an object")
    if "environment_lock_sha256" in env and not schema.is_sha256_hex(
        env["environment_lock_sha256"]
    ):
        fail("environment: environment_lock_sha256 is not a sha256 digest")
    if "command" in env and (
        not isinstance(env["command"], list)
        or not env["command"]
        or not all(isinstance(part, str) for part in env["command"])
    ):
        fail("environment: command must be a non-empty list of strings")
    if "seeds" in env and (
        not isinstance(env["seeds"], list)
        or not env["seeds"]
        or not all(schema.is_real_int(seed) for seed in env["seeds"])
    ):
        fail("environment: seeds must be a non-empty list of real integers")
    if "pythonhashseed" in env and (
        not isinstance(env["pythonhashseed"], str) or not env["pythonhashseed"]
    ):
        fail("environment: pythonhashseed must be a non-empty string")
    if "archived_rng_pinned" in env and env["archived_rng_pinned"] is not False:
        fail(
            "environment: archived_rng_pinned must be exactly False"
            " (R-077 flags)"
        )
    if "fresh_rng_pinned" in env and env["fresh_rng_pinned"] is not True:
        fail("environment: fresh_rng_pinned must be exactly True (R-077 flags)")


_COMPONENT_CHECKERS = {
    "repo": _check_repo,
    "content_hashes": _check_content_hashes,
    "eligibility": _check_eligibility,
    "snapshots": _check_snapshots,
    "offline_flags": _check_offline_flags,
    "staged_inputs": _check_staged_inputs,
    "suite_receipts": _check_suite_receipts,
    "parity": _check_parity,
    "qa012": _check_qa012,
    "environment": _check_environment,
}


def assemble_certificate(components: dict[str, Any]) -> dict[str, Any]:
    """Pure core of the PRE_RUN_READY generator (R-079).

    ``ready`` is True ONLY when every check passes; ANY defect yields
    ``ready: False`` with EVERY failing component named in
    ``failing_checks`` — never a partial pass, never an exception.
    """
    failing_checks: list[str] = []
    fail = failing_checks.append
    if not isinstance(components, dict):
        return {
            "schema_version": CERT_SCHEMA_VERSION,
            "ready": False,
            "failing_checks": [
                f"{key}: required component missing"
                for key in CERT_COMPONENT_KEYS
            ],
            "components": {},
        }
    for key in CERT_COMPONENT_KEYS:
        if key not in components:
            fail(f"{key}: required component missing")
            continue
        checker = _COMPONENT_CHECKERS[key]
        try:
            checker(components[key], fail)
        except Exception as exc:  # noqa: BLE001 - the never-raise contract
            fail(
                f"{key}: check evaluation failed"
                f" ({exc.__class__.__name__}) — fail closed"
            )
    return {
        "schema_version": CERT_SCHEMA_VERSION,
        "ready": not failing_checks,
        "failing_checks": failing_checks,
        "components": components,
    }


PARITY_COMPARATOR_IDENTITY = (
    "reproducibility.colm_aims_2026.phase4.compare_parity"
)
CERT_CONFIG_KEYS = (
    "repo_root",
    "eligibility_path",
    "snapshot_manifest_path",
    "snapshot_dirs",
    "parity_anchor_path",
    "qa012_manifest_path",
    "staged_plan",
    "suite_receipt_paths",
    "content_hash_paths",
    "environment",
    "offline_flags",
)


def _default_command_runner(repo_root: Path):
    """Subprocess-backed ``run(cmd) -> stdout`` for production gathering."""

    def run(cmd: list[str]) -> str:
        completed = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout

    return run


def _recompute_file_sha256(path: Any) -> str | None:
    """Record-not-raise recompute: None (a failing component under
    ``assemble_certificate``) when the file is missing/irregular."""
    try:
        return _sha256_regular_file(
            Path(path), error_cls=schema.ColmAimsError, label="certificate input"
        )
    except schema.ColmAimsError:
        return None


def gather_certificate_components(
    config: dict[str, Any], run: Any = None
) -> dict[str, Any]:
    """Gather every PRE_RUN_READY component for ``assemble_certificate``
    (F-4, R-079).

    ``run`` is an injectable command-runner ``run(cmd: list[str]) -> str``
    (stdout; defaults to subprocess in ``config["repo_root"]``). Repo
    identity is RUNNER-SOURCED: dirty from ``git status --porcelain``
    (empty == clean), commit from ``git rev-parse HEAD``, tree from
    ``git rev-parse HEAD^{tree}`` — never a caller assertion. Every staged
    input is REHASHED from file bytes (observed is never copied from the
    expectation); content/anchor/qa012 hashes are recomputed from their
    files; the eligibility digests come through the typed loader; snapshot
    verification failures are RECORDED as ``verified: False`` (assemble
    decides); suite receipts are ingested from the receipt FILES.
    """
    if not isinstance(config, dict):
        raise schema.ColmAimsError(
            "gather_certificate_components config must be an object (F-4)"
        )
    missing = sorted(k for k in CERT_CONFIG_KEYS if k not in config)
    if missing:
        raise schema.ColmAimsError(
            f"gather_certificate_components config missing key(s) {missing}"
            " (F-4)"
        )
    repo_root = Path(config["repo_root"])
    if run is None:
        run = _default_command_runner(repo_root)

    # Repo identity: runner-sourced, verbatim native git object ids
    # (40-hex SHA-1 or 64-hex SHA-256 — SPEC_ISSUE-1 adjudication).
    # Clean-state proof is the TRACKED tree (untracked evidence artifacts —
    # the certificate itself, suite receipts, staged inputs, shuttle
    # documents — are unavoidable by construction and are DISCLOSED by
    # list instead; adjudicated amendment 2026-08-22). Code identity is
    # already bound by commit+tree, which untracked files cannot alter.
    tracked_status = run(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    )
    dirty = bool(str(tracked_status).strip())
    full_status = run(["git", "status", "--porcelain"])
    untracked = sorted(
        line[3:]
        for line in str(full_status).splitlines()
        if line.startswith("??")
    )
    commit = str(run(["git", "rev-parse", "HEAD"])).strip()
    tree = str(run(["git", "rev-parse", "HEAD^{tree}"])).strip()
    repo = {
        "commit": commit,
        "tree_sha256": tree,
        "dirty": dirty,
        "untracked_disclosure": untracked,
    }

    content_hashes = {
        str(key): _recompute_file_sha256(path)
        for key, path in dict(config["content_hash_paths"]).items()
    }

    try:
        art = load_pairing_eligibility(Path(config["eligibility_path"]))
        eligibility = {
            "digest": art["pairing_population_keyset_sha256"],
            "horizon_map_sha256": art["horizon_map_sha256"],
        }
    except schema.ColmAimsError as exc:
        eligibility = {
            "digest": None,
            "horizon_map_sha256": None,
            "error": str(exc),
        }

    snapshots: dict[str, dict[str, Any]] = {}
    manifest_roles: dict[str, Any] | None
    manifest_error = None
    try:
        manifest = load_model_snapshot_manifest(
            Path(config["snapshot_manifest_path"])
        )
        manifest_roles = manifest["roles"]
    except schema.ColmAimsError as exc:
        manifest_roles = None
        manifest_error = str(exc)
    snapshot_dirs = dict(config["snapshot_dirs"])
    for role in sorted(SNAPSHOT_ROLES):
        if manifest_roles is None:
            snapshots[role] = {
                "verified": False,
                "model_name": None,
                "hf_revision": None,
                "error": manifest_error,
            }
            continue
        role_entry = manifest_roles[role]
        record: dict[str, Any] = {
            "verified": False,
            "model_name": role_entry["model_name"],
            "hf_revision": role_entry["hf_revision"],
        }
        snap_dir = snapshot_dirs.get(role)
        if snap_dir is None:
            record["error"] = "no snapshot directory configured for this role"
        else:
            try:
                verify_snapshot_dir(role_entry, Path(snap_dir))
                record["verified"] = True
            except schema.ColmAimsError as exc:
                record["error"] = str(exc)
        snapshots[role] = record

    staged_inputs: list[dict[str, Any]] = []
    for entry in list(config["staged_plan"]):
        entry = _as_dict(entry)
        label = entry.get("label")
        path = Path(entry.get("path", ""))
        try:
            observed = _sha256_regular_file(
                path,
                error_cls=StagedInputError,
                label=f"staged input {label!r}",
            )
        except schema.ColmAimsError:
            observed = None
        staged_inputs.append(
            {
                "label": label,
                "path": str(path),
                "expected_sha256": entry.get("expected_sha256"),
                "observed_sha256": observed,
            }
        )

    suite_receipts: dict[str, Any] = {}
    receipt_paths = dict(config["suite_receipt_paths"])
    for name in SUITE_RECEIPT_NAMES:
        path = receipt_paths.get(name)
        try:
            data = schema.read_regular_file_bytes(Path(path))
            receipt = schema.parse_json_bytes_strict(data)
        except (schema.ColmAimsError, OSError, TypeError, ValueError) as exc:
            suite_receipts[name] = {"ingest_error": str(exc)}
            continue
        if not isinstance(receipt, dict):
            suite_receipts[name] = {
                "ingest_error": "receipt file is not a JSON object"
            }
            continue
        suite_receipts[name] = receipt

    return {
        "repo": repo,
        "content_hashes": content_hashes,
        "eligibility": eligibility,
        "snapshots": snapshots,
        "offline_flags": list(config["offline_flags"]),
        "staged_inputs": staged_inputs,
        "suite_receipts": suite_receipts,
        "parity": {
            "comparator_identity": PARITY_COMPARATOR_IDENTITY,
            "anchor_sha256": _recompute_file_sha256(
                config["parity_anchor_path"]
            ),
        },
        "qa012": {
            "rev2_manifest_sha256": _recompute_file_sha256(
                config["qa012_manifest_path"]
            ),
        },
        "environment": dict(_as_dict(config["environment"])),
    }


def generate_pre_run_ready(
    components: dict[str, Any], out_path: Path
) -> dict[str, Any]:
    """Assemble the PRE_RUN_READY certificate and write it as canonical JSON.

    DECISION: component GATHERING (git state, content hashes, suite
    receipts, environment capture) is the orchestrating runner's job — this
    generator stays a thin, deterministic assemble-serialize-bind step so
    the pure core (``assemble_certificate``) carries every check. Returns
    the certificate plus the written file's SHA-256; the author's
    single-run exception activation references that digest (R-079).
    """
    certificate = assemble_certificate(components)
    payload = schema.encode_json(certificate)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    return {
        "certificate": certificate,
        "ready": certificate["ready"],
        "path": str(out_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
