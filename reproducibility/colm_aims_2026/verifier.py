"""Two-mode fail-closed verifier (source-contract / release).

Spec rules owned here: R-012..R-015, R-017, R-019 (as verified surface),
R-021 (as CLI backend), R-033, R-035, R-036 (emission call), R-039
(canonical selection).
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import ledger as ledger_mod
from . import receipt as receipt_mod
from . import pairing, schema
from .schema import ColmAimsError

# MA-CC-5: the object-existence check binds to THIS repository explicitly —
# never ambient cwd/.git — so no cwd move or GIT_* env var can flip the gate.
_SOURCE_REPO = Path(__file__).resolve().parents[2]
# Git env vars that redirect which repository/worktree/objects git consults.
_GIT_ENV_DENYLIST = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_INDEX_FILE",
        "GIT_CEILING_DIRECTORIES",
        "GIT_NAMESPACE",
    }
)


class VacuousInputError(ColmAimsError):
    """Zero candidate artifacts / empty ledger / empty manifest (R-033)."""


class ContainmentError(ColmAimsError):
    """Expectations file not outside the verified artifact tree (R-013)."""


# R-017: closed source-mode verdict enum; strongest member PASS_SOURCE_ONLY.
VERDICT_SOURCE_PASS = "PASS_SOURCE_ONLY"
VERDICT_RELEASE_PASS = "PASS_RELEASE"
VERDICT_FAIL = "FAIL"
SOURCE_MODE_VERDICTS = frozenset({VERDICT_SOURCE_PASS, VERDICT_FAIL})
RELEASE_MODE_VERDICTS = frozenset({VERDICT_RELEASE_PASS, VERDICT_FAIL})

CERTIFIABLE = "CERTIFIABLE"
HISTORICAL_NONCERTIFYING = "HISTORICAL_NONCERTIFYING"

# R-012: the release binding legs, one per independently anchored binding.
BINDING_KEYS = (
    "schema_profile",
    "producer",
    "semantic_command",
    "seeds",
    "dirty_state",
    "splits",
    "calibration_identity",
    "continuation_identity",
    "input_hashes",
    "split_metadata_sha256",
    "mc_build",
    "model",
    "runtime_packages",
)

_EXPECTATIONS_KEYS = frozenset(
    {"schema_version", "anchor", "rights_inventory", "tree_files", "bindings"}
)
_MANIFEST_KEYS = frozenset(
    {"schema_version", "artifacts", "allowlist_undeclared"}
)

# R-012/R-014: the estimand-defining dependency-closure identities.
_CLOSURE_IDENTITY_KEYS = ("calibration_identity", "continuation_identity")

_EXPECTED_LAYOUT = (
    "profile.json (strict constructed-reference profile), records.jsonl"
    " (retained per-item records), presentation_manifest.json"
    " (presentation manifest)"
)

# Row-status strength order for the R-012 stale-status recomputation gate.
_STATUS_STRENGTH = {"FAIL": 0, "UNVERIFIED": 1, "PASS": 2}


@dataclass
class VerificationReport:
    """Structured result of one verifier run (type definition for tests)."""

    mode: str
    verdict: str
    legs: list[dict[str, Any]] = field(default_factory=list)
    validated_artifacts: list[str] = field(default_factory=list)
    receipt_path: Path | None = None
    classifications: dict[str, str] = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _keyset_sha256(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def _tree_file_map(tree: Path) -> dict[str, Path]:
    return {
        p.relative_to(tree).as_posix(): p
        for p in Path(tree).rglob("*")
        if p.is_file()
    }


def _read_tree_snapshot(tree: Path) -> dict[str, bytes]:
    """Read every tree member's bytes ONCE, symlink-free (MA-HI-002/MA-HI-004).

    The single content-addressed load that drives content-validation, every
    binding/tree-file hash, and the receipt's ``input_tree_sha256`` — so the
    receipt provably attests exactly the bytes the gates saw (no TOCTOU
    desync). A symlink member (or a member resolving outside the tree) is
    refused with a typed containment error rather than followed, read, and
    hashed into the receipt. Each read is bounded and regular-file-only
    (MA-HI-001).
    """
    tree = Path(tree)
    snapshot: dict[str, bytes] = {}
    for p in sorted(tree.rglob("*")):
        if p.is_symlink():
            rel = p.relative_to(tree).as_posix()
            raise ContainmentError(
                f"tree member {rel!r} is a symlink — refusing to follow, read,"
                " or hash bytes outside the verified tree (R-036/R-013)"
            )
        if p.is_dir():
            continue
        rel = p.relative_to(tree).as_posix()
        if not schema.resolves_inside(p, tree):
            raise ContainmentError(
                f"tree member {rel!r} resolves outside the verified tree"
                " (R-036/R-013)"
            )
        snapshot[rel] = schema.read_regular_file_bytes(p, tree_root=tree)
    return snapshot


def _sha_map(snapshot: dict[str, bytes]) -> dict[str, str]:
    return {rel: hashlib.sha256(data).hexdigest() for rel, data in snapshot.items()}


def _digest_over_lines(lines: list[str]) -> str:
    """Pinned digest shape (R-036): sha256 over newline-joined
    ``<posix relpath>:<sha256>`` lines with a trailing newline."""
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _tree_digest_from_shas(sha_by_rel: dict[str, str]) -> str:
    """Pinned input-tree digest (R-036) over the one-shot snapshot hashes."""
    return _digest_over_lines(
        [f"{rel}:{sha}" for rel, sha in sorted(sha_by_rel.items())]
    )


def _code_digest() -> str:
    """Pinned verifier-code digest (R-036) over the namespace's .py files."""
    namespace = Path(__file__).resolve().parent
    return _digest_over_lines(
        [
            f"{p.relative_to(namespace).as_posix()}:{_sha256_file(p)}"
            for p in sorted(namespace.glob("**/*.py"))
        ]
    )


def _pass(leg_id: str) -> dict[str, Any]:
    return {"leg_id": leg_id, "outcome": "PASS"}


def _fail(
    leg_id: str,
    *,
    expected: Any,
    observed: Any,
    remediation: str = "ARTIFACT_DEFECT",
) -> dict[str, Any]:
    return {
        "leg_id": leg_id,
        "outcome": "FAIL",
        "expected": expected,
        "observed": observed,
        "remediation_class": remediation,
    }


def _skipped(leg_id: str, *, reason: str) -> dict[str, Any]:
    """A leg that could not run because a required capability is unavailable
    (MA-CC-5). SKIPPED never fails the verdict but is rendered and receipted
    so the gap is on the record."""
    return {"leg_id": leg_id, "outcome": "SKIPPED", "reason": reason}


def _record_leg(
    legs: list[dict[str, Any]],
    leg_id: str,
    passed: bool,
    *,
    expected: Any,
    observed: Any,
    remediation: str = "ARTIFACT_DEFECT",
) -> None:
    """Append one PASS leg, or one FAIL leg carrying expected/observed."""
    if passed:
        legs.append(_pass(leg_id))
    else:
        legs.append(
            _fail(
                leg_id,
                expected=expected,
                observed=observed,
                remediation=remediation,
            )
        )


def _is_resolved_identity(value: Any) -> bool:
    """A closure identity is resolved when it is a non-empty string other than
    the explicit ``UNRESOLVED`` marker (R-012/R-014)."""
    return isinstance(value, str) and bool(value) and value != "UNRESOLVED"


# ---------------------------------------------------------------------------
# Binding admissibility (QA-001): a binding leg carries TWO obligations —
# the observed value must be admissible in its own right AND must match the
# anchored expectation. Mirror-equality against an author-controlled proxy
# is never enough on its own.
# ---------------------------------------------------------------------------


def _admissible_value(value: Any, where: str) -> str | None:
    """Recursive admissibility: rejects None, empty/whitespace strings, the
    ``UNRESOLVED`` marker, and empty containers at any depth (QA-001)."""
    if value is None:
        return f"{where} is null"
    if isinstance(value, str):
        if not value.strip():
            return f"{where} is empty"
        if value == "UNRESOLVED":
            return f"{where} is UNRESOLVED"
        return None
    if isinstance(value, dict):
        if not value:
            return f"{where} is an empty object"
        for key, sub in value.items():
            problem = _admissible_value(sub, f"{where}.{key}")
            if problem is not None:
                return problem
        return None
    if isinstance(value, list):
        if not value:
            return f"{where} is an empty list"
        for index, sub in enumerate(value):
            problem = _admissible_value(sub, f"{where}[{index}]")
            if problem is not None:
                return problem
        return None
    return None  # numbers and booleans are admissible leaves


def _valid_schema_profile(value: Any) -> str | None:
    problem = _admissible_value(value, "schema_profile")
    if problem is not None:
        return problem
    if not isinstance(value, dict):
        return "schema_profile must be an object"
    if not isinstance(value.get("profile_id"), str):
        return "schema_profile.profile_id must be a string"
    version = value.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        return "schema_profile.schema_version must be an integer"
    if not schema.is_sha256_hex(value.get("profile_sha256")):
        return "schema_profile.profile_sha256 is not a sha256 digest"
    return None


def _valid_producer(value: Any) -> str | None:
    # QA-017: field-wise admissibility — helper_sha256s may be an honest
    # EMPTY map (a helperless producer); its entries, when present, must be
    # sha256 digests. Everything else stays recursively admissible.
    if not isinstance(value, dict) or not value:
        return "producer must be a non-empty object"
    problem = _admissible_value(
        value.get("entrypoint"), "producer.entrypoint"
    )
    if problem is not None:
        return problem
    if not isinstance(value.get("entrypoint"), str):
        return "producer.entrypoint must be a string"
    if not schema.is_sha256_hex(value.get("sha256")):
        return "producer.sha256 is not a sha256 digest"
    helpers = value.get("helper_sha256s")
    if not isinstance(helpers, dict):
        return "producer.helper_sha256s must be an object"
    for name, sha in helpers.items():
        if not isinstance(name, str) or not name:
            return "producer.helper_sha256s keys must be non-empty strings"
        if not schema.is_sha256_hex(sha):
            return f"producer.helper_sha256s[{name!r}] is not a sha256 digest"
    return None


def _valid_semantic_command(value: Any) -> str | None:
    problem = _admissible_value(value, "semantic_command")
    if problem is not None:
        return problem
    if not isinstance(value, list) or not all(
        isinstance(part, str) for part in value
    ):
        return "semantic_command must be a list of strings"
    return None


def _valid_seeds(value: Any) -> str | None:
    problem = _admissible_value(value, "seeds")
    if problem is not None:
        return problem
    if not isinstance(value, list) or not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in value
    ):
        return "seeds must be a list of integers"
    return None


def _valid_dirty_state(value: Any) -> str | None:
    problem = _admissible_value(value, "dirty_state")
    if problem is not None:
        return problem
    if not isinstance(value, dict):
        return "dirty_state must be an object"
    if not isinstance(value.get("git_dirty"), bool):
        return "dirty_state.git_dirty must be a boolean"
    if not schema.is_commit_sha(value.get("source_commit")):
        return "dirty_state.source_commit is not a full-length commit SHA"
    return None


def _valid_splits(value: Any) -> str | None:
    problem = _admissible_value(value, "splits")
    if problem is not None:
        return problem
    if not isinstance(value, dict):
        return "splits must be an object"
    for split_name in ("fit", "eval"):
        if not isinstance(value.get(split_name), dict):
            return f"splits.{split_name} must be an object"
    if not isinstance(value.get("zero_overlap"), bool):
        return "splits.zero_overlap must be a boolean"
    return None


def _valid_resolved_identity(value: Any) -> str | None:
    if not _is_resolved_identity(value):
        return f"identity {value!r} is missing, empty, or UNRESOLVED"
    return None


def _valid_hash_map(value: Any) -> str | None:
    problem = _admissible_value(value, "input_hashes")
    if problem is not None:
        return problem
    if not isinstance(value, dict):
        return "input_hashes must be an object"
    for name, sha in value.items():
        if not isinstance(name, str) or not name:
            return "input_hashes keys must be non-empty strings"
        if not schema.is_sha256_hex(sha):
            return f"input_hashes[{name!r}] is not a sha256 digest"
    return None


def _valid_sha256_scalar(value: Any) -> str | None:
    if not schema.is_sha256_hex(value):
        return f"{value!r} is not a sha256 digest"
    return None


def _valid_mc_build(value: Any) -> str | None:
    problem = _admissible_value(value, "mc_build")
    if problem is not None:
        return problem
    if not isinstance(value, dict):
        return "mc_build must be an object"
    if not isinstance(value.get("built_after_split"), bool):
        return "mc_build.built_after_split must be a boolean"
    return None


def _valid_model(value: Any) -> str | None:
    problem = _admissible_value(value, "model")
    if problem is not None:
        return problem
    if not isinstance(value, dict):
        return "model must be an object"
    if not isinstance(value.get("repository_namespace"), str):
        return "model.repository_namespace must be a string"
    revision = value.get("revision")
    digest_manifest = value.get("byte_digest_manifest")
    if revision is not None:
        if not schema.is_commit_sha(revision):
            return (
                f"model.revision {revision!r} is not an immutable full-length"
                " 40-hex commit SHA"
            )
    elif not (
        isinstance(digest_manifest, dict)
        and digest_manifest
        and all(schema.is_sha256_hex(v) for v in digest_manifest.values())
    ):
        return (
            "model carries neither an immutable revision nor a complete"
            " canonical byte-digest manifest"
        )
    if not schema.is_sha256_hex(value.get("weights_sha256")):
        return "model.weights_sha256 is not a sha256 digest"
    return None


def _valid_runtime_packages(value: Any) -> str | None:
    problem = _admissible_value(value, "runtime_packages")
    if problem is not None:
        return problem
    if not isinstance(value, dict) or not all(
        isinstance(v, str) for v in value.values()
    ):
        return "runtime_packages must map package names to version strings"
    return None


# QA-001 class fix: every BINDING_KEYS entry has a registered validity
# predicate — the meta-test asserts registry/key-set equality.
BINDING_VALIDATORS: dict[str, Callable[[Any], str | None]] = {
    "schema_profile": _valid_schema_profile,
    "producer": _valid_producer,
    "semantic_command": _valid_semantic_command,
    "seeds": _valid_seeds,
    "dirty_state": _valid_dirty_state,
    "splits": _valid_splits,
    "calibration_identity": _valid_resolved_identity,
    "continuation_identity": _valid_resolved_identity,
    "input_hashes": _valid_hash_map,
    "split_metadata_sha256": _valid_sha256_scalar,
    "mc_build": _valid_mc_build,
    "model": _valid_model,
    "runtime_packages": _valid_runtime_packages,
}


# ---------------------------------------------------------------------------
# Declarative provenance-noun table (QA-002): transcribed one-to-one from
# R-012's enumeration; the meta-test asserts full noun coverage.
# ---------------------------------------------------------------------------

_FIELD_PREDICATES: dict[str, Callable[[Any], bool]] = {
    "nonempty_str": lambda v: isinstance(v, str)
    and bool(v.strip())
    and v != "UNRESOLVED",
    "resolved_identity": _is_resolved_identity,
    "sha256_hex": schema.is_sha256_hex,
    "commit_sha": schema.is_commit_sha,
    "is_false": lambda v: v is False,
    "is_true": lambda v: v is True,
    "positive_int": lambda v: isinstance(v, int)
    and not isinstance(v, bool)
    and v > 0,
    "nonneg_int": lambda v: isinstance(v, int)
    and not isinstance(v, bool)
    and v >= 0,
    "finite_number": lambda v: schema.is_number(v) and math.isfinite(float(v)),
    "nonempty_str_list": lambda v: isinstance(v, list)
    and bool(v)
    and all(isinstance(x, str) and x for x in v),
    "nonempty_int_list": lambda v: isinstance(v, list)
    and bool(v)
    and all(isinstance(x, int) and not isinstance(x, bool) for x in v),
    "sha256_map": lambda v: isinstance(v, dict)
    and bool(v)
    and all(
        isinstance(k, str) and k and schema.is_sha256_hex(x)
        for k, x in v.items()
    ),
    # QA-017: an honest helperless producer records an EMPTY helper map;
    # entries, when present, must still be sha256 digests.
    "sha256_map_allow_empty": lambda v: isinstance(v, dict)
    and all(
        isinstance(k, str) and k and schema.is_sha256_hex(x)
        for k, x in v.items()
    ),
    # QA-018: a coverage rate is a proportion — the tightest predicate the
    # noun admits is the closed unit interval.
    "unit_interval": lambda v: schema.is_number(v)
    and math.isfinite(float(v))
    and 0.0 <= float(v) <= 1.0,
    "nonempty_str_map": lambda v: isinstance(v, dict)
    and bool(v)
    and all(
        isinstance(k, str) and k and isinstance(x, str) and x
        for k, x in v.items()
    ),
    "nonempty_dict": lambda v: isinstance(v, dict) and bool(v),
}

# (dotted_path, predicate name, leg_id, remediation class) — one row per
# R-012 noun: producer/helper hashes; semantic command; seeds; dirty-state
# identity; fit/eval split names, counts, key-set hashes, zero-overlap;
# calibration/continuation identities; input + split_metadata.json hashes;
# MC-build freshness, coverage, retention; model repository namespace,
# weights/tokenizer content hashes, dtype/device class/numerical settings;
# runtime package versions. (The immutable model revision has its own
# either-form gate, model_revision_immutability.)
REQUIRED_PROVENANCE_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("producer_entrypoint", "nonempty_str", "producer_recorded", "ARTIFACT_DEFECT"),
    ("producer_sha256", "sha256_hex", "producer_recorded", "ARTIFACT_DEFECT"),
    ("helper_sha256s", "sha256_map_allow_empty", "producer_recorded", "ARTIFACT_DEFECT"),
    ("semantic_command", "nonempty_str_list", "semantic_command_recorded", "ARTIFACT_DEFECT"),
    ("seeds", "nonempty_int_list", "seeds_recorded", "ARTIFACT_DEFECT"),
    ("dirty_state.git_dirty", "is_false", "dirty_state_clean", "ARTIFACT_DEFECT"),
    ("dirty_state.source_commit", "commit_sha", "dirty_state_identity", "ARTIFACT_DEFECT"),
    ("splits.fit.name", "nonempty_str", "splits_fit_recorded", "ARTIFACT_DEFECT"),
    ("splits.fit.count", "positive_int", "splits_fit_recorded", "ARTIFACT_DEFECT"),
    ("splits.fit.keyset_sha256", "sha256_hex", "splits_fit_recorded", "ARTIFACT_DEFECT"),
    ("splits.eval.name", "nonempty_str", "splits_eval_recorded", "ARTIFACT_DEFECT"),
    ("splits.eval.count", "positive_int", "splits_eval_recorded", "ARTIFACT_DEFECT"),
    ("splits.eval.keyset_sha256", "sha256_hex", "splits_eval_recorded", "ARTIFACT_DEFECT"),
    ("splits.zero_overlap", "is_true", "splits_zero_overlap", "ARTIFACT_DEFECT"),
    ("calibration_identity", "resolved_identity", "binding_calibration_identity_resolved", "AUTHOR_DECISION_REQUIRED"),
    ("continuation_identity", "resolved_identity", "binding_continuation_identity_resolved", "AUTHOR_DECISION_REQUIRED"),
    ("input_sha256", "sha256_map", "input_hashes_recorded", "ARTIFACT_DEFECT"),
    ("split_metadata_sha256", "sha256_hex", "split_metadata_recorded", "ARTIFACT_DEFECT"),
    ("mc_build.built_after_split", "is_true", "mc_build_freshness", "ARTIFACT_DEFECT"),
    ("mc_build.coverage_rate", "unit_interval", "mc_build_coverage_retention", "ARTIFACT_DEFECT"),
    ("mc_build.retention_policy", "nonempty_str", "mc_build_coverage_retention", "ARTIFACT_DEFECT"),
    ("mc_build.retained_count", "nonneg_int", "mc_build_coverage_retention", "ARTIFACT_DEFECT"),
    ("model.repository_namespace", "nonempty_str", "model_identity_completeness", "ARTIFACT_DEFECT"),
    ("model.weights_sha256", "sha256_hex", "model_weights_hash", "ARTIFACT_DEFECT"),
    ("model.tokenizer_config_sha256", "sha256_hex", "model_identity_completeness", "ARTIFACT_DEFECT"),
    ("model.dtype", "nonempty_str", "model_identity_completeness", "ARTIFACT_DEFECT"),
    ("model.device_class", "nonempty_str", "model_identity_completeness", "ARTIFACT_DEFECT"),
    ("model.numerical_settings", "nonempty_dict", "model_identity_completeness", "ARTIFACT_DEFECT"),
    ("runtime_packages", "nonempty_str_map", "runtime_packages_recorded", "ARTIFACT_DEFECT"),
)


def _resolve_dotted(prov: dict[str, Any], dotted: str) -> tuple[bool, Any]:
    node: Any = prov
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _provenance_table_legs(
    legs: list[dict[str, Any]], prov: dict[str, Any]
) -> None:
    """Table-driven R-012 noun legs (QA-002): every enumerated provenance
    field must be present and admissible in its own right — mirrored
    expectations cannot substitute for a recorded value."""
    order: list[str] = []
    problems_by_leg: dict[str, list[str]] = {}
    remediation_by_leg: dict[str, str] = {}
    for dotted, predicate_name, leg_id, remediation in (
        REQUIRED_PROVENANCE_FIELDS
    ):
        if leg_id not in problems_by_leg:
            problems_by_leg[leg_id] = []
            remediation_by_leg[leg_id] = remediation
            order.append(leg_id)
        found, value = _resolve_dotted(prov, dotted)
        if not found:
            problems_by_leg[leg_id].append(f"{dotted}: absent")
        elif not _FIELD_PREDICATES[predicate_name](value):
            problems_by_leg[leg_id].append(
                f"{dotted}: {value!r} fails {predicate_name}"
            )
    for leg_id in order:
        problems = problems_by_leg[leg_id]
        _record_leg(
            legs,
            leg_id,
            not problems,
            expected="R-012 provenance noun(s) recorded and admissible",
            observed="; ".join(problems),
            remediation=remediation_by_leg[leg_id],
        )


def classify_certifiability(profile: dict[str, Any]) -> str:
    """CERTIFIABLE vs HISTORICAL_NONCERTIFYING closure classification (R-014).

    Only a producer/closure change invalidates an artifact: superseded,
    dirty, or unresolved estimand-defining dependency closures classify
    HISTORICAL_NONCERTIFYING; non-closure metadata differences do not.
    """
    if not isinstance(profile, dict):
        raise ColmAimsError("profile must be an object (R-014)")
    prov = profile.get("provenance") or {}
    dirty_state = prov.get("dirty_state") or {}
    if dirty_state.get("git_dirty") is not False:
        return HISTORICAL_NONCERTIFYING
    if "superseded_by_producer_sha256" in prov:
        return HISTORICAL_NONCERTIFYING
    for name in _CLOSURE_IDENTITY_KEYS:
        if not _is_resolved_identity(prov.get(name)):
            return HISTORICAL_NONCERTIFYING
    return CERTIFIABLE


def parse_legacy_profile(data: bytes) -> dict[str, Any]:
    """Parse a known legacy profile family from captured bytes (R-014).

    Families: the ``paper_exports/csli.json``, ``paper_exports/
    calibration.json``, and ``paper_exports/audit_card.json`` aggregate
    formats. Legacy artifacts are refused only on a demonstrably missing
    named invariant — never merely for predating the strict schema.
    """
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise schema.TypedIngressError(
            f"legacy artifact bytes are not valid JSON: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise schema.TypedIngressError("legacy artifact must be a JSON object")
    if "panel_csli" in obj:
        family = "csli"
    elif "per_bucket" in obj and "max_ece" in obj:
        family = "calibration"
    elif "metrics" in obj and "overall_verdict" in obj:
        family = "audit_card"
    else:
        raise schema.SchemaValidationError(
            "unknown legacy artifact family — not one of the enumerated"
            " csli/calibration/audit_card aggregate formats (R-014)"
        )
    if "metadata" not in obj:
        raise schema.SchemaValidationError(
            f"legacy {family} artifact is missing the named invariant"
            " 'metadata' (generation/provenance block) — refusal requires a"
            " demonstrably missing named invariant (R-014)"
        )
    return {"legacy_family": family, "aggregate_only": True, "payload": obj}


def legacy_certifies(legacy: dict[str, Any], claim_kind: str) -> bool:
    """Whether a legacy artifact can certify a claim kind (R-014).

    Aggregate-only files cannot certify per-item paired claims.
    """
    if legacy.get("aggregate_only", True):
        return claim_kind == "aggregate"
    return True


def resolve_canonical_package(runs_root: Path, ledger: dict[str, Any]) -> Path:
    """Canonical run selection strictly via the ledger pointer (R-039)."""
    run_id = ledger.get("canonical_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ColmAimsError(
            "ledger declares no canonical_run_id pointer — canonical"
            " selection happens only via the ledger/expectations pointer,"
            " never newest-wins (R-039)"
        )
    # CX-4 (R-039): the pointer is a NAME inside the runs root, never a path —
    # `../escape` or an absolute value must not select a directory outside
    # the runs root.
    if not schema.is_path_component(run_id):
        raise ColmAimsError(
            f"canonical run pointer {run_id!r} must be a single path"
            " component inside the runs root — path traversal and absolute"
            " pointers are refused (R-039)"
        )
    path = Path(runs_root) / run_id
    if not path.is_dir():
        raise ColmAimsError(
            f"canonical run pointer {run_id!r} does not resolve to a"
            " published run directory under the runs root — dangling"
            " pointers never fall back (R-039)"
        )
    if not any(path.iterdir()):
        # QA-008: an empty run directory is a mid-publish crash relic, not a
        # published evidence package — the pointer is dangling.
        raise ColmAimsError(
            f"canonical run pointer {run_id!r} resolves to an EMPTY run"
            " directory (a crashed-publish relic) — not a published evidence"
            " package (R-039/R-016)"
        )
    return path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_expectations(path: Path) -> tuple[dict[str, Any], bytes]:
    """Typed, fail-closed load of the anchored expectations file (R-022).

    Uses the bounded, symlink-free, regular-file-only reader (MA-HI-001) so a
    FIFO / device / oversized file at ``--expectations`` is rejected fast
    instead of hanging or OOM-ing the run.
    """
    name = Path(path).name
    data = schema.read_regular_file_bytes(path)
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise schema.TypedIngressError(
            f"{name}: malformed JSON: {exc} (R-020)"
        ) from exc
    if not isinstance(obj, dict):
        raise schema.TypedIngressError(f"{name}: expectations must be an object")
    if "schema_version" not in obj:
        raise schema.TypedIngressError(
            f"{name}: missing required field 'schema_version' (R-020)"
        )
    version = obj["schema_version"]
    if version != schema.SCHEMA_VERSION:
        raise schema.TypedIngressError(
            f"{name}: unsupported schema_version {version!r}; supported range"
            f" {schema.SUPPORTED_SCHEMA_VERSION_MIN}.."
            f"{schema.SUPPORTED_SCHEMA_VERSION_MAX}; verifier revision"
            f" {schema.VERIFIER_REVISION} (R-020)"
        )
    unknown = sorted(set(obj) - _EXPECTATIONS_KEYS)
    if unknown:
        # QA-009: unknown config-surface keys are a USAGE/CONFIG error
        # (exit 2, matching the documented contract), not an ingress error.
        raise schema.ConfigSurfaceError(
            f"{name}: unknown expectations key(s) {unknown} — the config"
            " surface fails closed; no key disables a release gate"
            " (R-022/R-037)"
        )
    return obj, data


def _load_json_bytes_lenient(
    data: bytes, name: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Collect-don't-halt parse of already-read bytes: (object, error)."""
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{name}: malformed JSON: {exc}"
    if not isinstance(obj, dict):
        return None, f"{name}: must be a JSON object"
    return obj, None


def _git_object_exists(commit: str) -> bool | None:
    """Anchor object-existence check bound to THIS repository (MA-CC-5, R-013).

    Returns ``True``/``False`` when the source repository is available and git
    answers, ``None`` when the object-existence check cannot run (no ``.git``,
    git missing, or a timeout). Never consults ambient ``cwd``/``.git`` and
    scrubs every ``GIT_*`` redirection variable from the child environment, so
    no cwd move or environment door (R-022) can flip the gate. The subprocess
    is bounded (``timeout``), non-interactive (``GIT_TERMINAL_PROMPT=0``,
    ``stdin=DEVNULL``) so a stalled ``.git`` cannot hang the run (MA-RB-002).
    """
    repo = _SOURCE_REPO
    if not (repo / ".git").exists():
        return None
    env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV_DENYLIST}
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=str(repo),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# run_verifier
# ---------------------------------------------------------------------------


def _classify_legacy_artifacts(
    snapshot: dict[str, bytes]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Classify known historical artifact families from the tree snapshot
    (R-014, MA-HI-004).

    Returns the parsed legacy artifacts keyed by tree-relative path, plus
    their human-readable classifications. A file that is not a known legacy
    family is left to the other gates — never refused merely for predating
    the strict schema.
    """
    parsed_by_rel: dict[str, dict[str, Any]] = {}
    classifications: dict[str, str] = {}
    for rel in sorted(snapshot):
        if rel in ("profile.json", "presentation_manifest.json"):
            continue
        if not rel.endswith(".json"):
            continue
        try:
            parsed = parse_legacy_profile(snapshot[rel])
        except ColmAimsError:
            continue  # not a known legacy family; other gates govern it
        parsed_by_rel[rel] = parsed
        classifications[rel] = (
            f"legacy_{parsed['legacy_family']}_aggregate"
            " (historical, aggregate-only)"
        )
    return parsed_by_rel, classifications


def _reject_empty_evaluation(cells: Any) -> None:
    """R-006/R-012: an explicitly empty evaluation errors before any report."""
    if not isinstance(cells, list):
        return
    for cell in cells:
        if (
            isinstance(cell, dict)
            and isinstance(cell.get("counts"), dict)
            and cell["counts"].get("n_pairing_population") == 0
        ):
            raise schema.EmptyEvaluationError(
                f"cell {cell.get('cell_id')!r} declares an explicitly empty"
                " evaluation (n_pairing_population == 0); refused before any"
                " report is emitted (R-006/R-012)"
            )


def run_verifier(
    tree: Path,
    *,
    mode: str,
    receipts_dir: Path,
    expectations: Path | None = None,
) -> VerificationReport:
    """Run one verifier pass over an artifact tree; never mutate inputs."""
    if mode not in ("source", "release"):
        raise ColmAimsError(
            f"unknown verifier mode {mode!r}; expected 'source' or 'release'"
        )
    tree = Path(tree)
    receipts_dir = Path(receipts_dir)

    # MA-HI-004/MA-HI-002: one symlink-free, bounded snapshot of every tree
    # member, read ONCE. Every gate hash and the receipt digest derive from
    # it, so nothing can desync gate-validated bytes from receipt-attested
    # bytes. sha_by_rel is the single hash source of truth.
    snapshot = _read_tree_snapshot(tree) if tree.is_dir() else {}
    if not snapshot or "profile.json" not in snapshot:
        # QA-013: name the tree path exactly as supplied on the command line
        # (R-033 requires naming the path; R-026 forbids amplifying it into a
        # resolved local absolute form the caller never supplied).
        raise VacuousInputError(
            f"zero candidate artifacts under {tree}; expected"
            f" layout: {_EXPECTED_LAYOUT} (R-033)"
        )
    sha_by_rel = _sha_map(snapshot)

    expectations_obj: dict[str, Any] | None = None
    expectations_bytes: bytes | None = None
    expectations_path: Path | None = None
    if mode == "release":
        if expectations is None:
            raise ColmAimsError(
                "release mode requires an independently anchored expectations"
                " file located outside the verified artifact tree (R-013)"
            )
        expectations_path = Path(expectations)
        if schema.resolves_inside(expectations_path, tree):
            raise ContainmentError(
                "expectations file resolves inside the verified artifact"
                " tree — self-attestation is refused; containment decisions"
                " use fully resolved, symlink-free paths (R-013)"
            )
        expectations_obj, expectations_bytes = _load_expectations(
            expectations_path
        )

    # Typed ingress (R-020) — from the snapshot bytes, not a re-read (MA-HI-004).
    profile = schema.load_artifact_bytes(snapshot["profile.json"], "profile.json")
    records: list[dict[str, Any]] | None = None
    record_lines: list[int] = []
    if "records.jsonl" in snapshot:
        loaded_records = schema.load_artifact_bytes(
            snapshot["records.jsonl"], "records.jsonl"
        )
        records = loaded_records["records"]
        record_lines = loaded_records.get("line_numbers") or []

    cells = profile.get("cells")
    _reject_empty_evaluation(cells)

    legs: list[dict[str, Any]] = []
    validated: list[str] = []

    # ---- shared (source minimum positive set) --------------------------
    legs.append(_pass("typed_ingress"))

    profile_valid = True
    try:
        schema.validate_profile(profile)
        legs.append(_pass("profile_validation"))
    except schema.SchemaValidationError as exc:
        profile_valid = False
        legs.append(
            _fail(
                "profile_validation",
                expected="valid strict constructed-reference profile"
                " (R-001..R-003, R-029, R-031, R-032)",
                observed=str(exc),
            )
        )

    records_valid = False
    if records is None:
        legs.append(
            _fail(
                "records_present",
                expected="records.jsonl with retained per-item records —"
                " absent records are non-certifying (R-015)",
                observed="records.jsonl absent from the artifact tree",
            )
        )
    else:
        record_errors: list[str] = []
        for index, record in enumerate(records):
            try:
                schema.validate_record(record)
            except schema.RecordValidationError as exc:
                # QA-014: name the file and source line, tree-relatively.
                lineno = (
                    record_lines[index]
                    if index < len(record_lines)
                    else "?"
                )
                record_errors.append(
                    f"records.jsonl: line {lineno}: {exc}"
                )
        if record_errors:
            legs.append(
                _fail(
                    "records_validation",
                    expected="non-reversible per-item records (R-031)",
                    observed="; ".join(record_errors[:5]),
                )
            )
        else:
            legs.append(_pass("records_validation"))
            records_valid = True

    cells_valid = True
    if records is None:
        cells_valid = False
    elif isinstance(cells, list):
        for cell in cells:
            cell_id = (
                cell.get("cell_id", "unnamed")
                if isinstance(cell, dict)
                else "unnamed"
            )
            leg_id = f"cell_{cell_id}_validation"
            try:
                pairing.validate_cell(cell, records)
                legs.append(_pass(leg_id))
            except schema.EmptyEvaluationError:
                raise
            except (ColmAimsError, KeyError, TypeError, ValueError) as exc:
                cells_valid = False
                legs.append(
                    _fail(
                        leg_id,
                        expected="cell count/rate/key/summary/interval"
                        " recomputation identities hold"
                        " (R-005..R-011, R-015)",
                        observed=str(exc),
                    )
                )

    legacy_parsed, classifications = _classify_legacy_artifacts(snapshot)

    try:
        closure = classify_certifiability(profile)
    except ColmAimsError:
        closure = HISTORICAL_NONCERTIFYING
    classifications["profile.json"] = closure

    artifacts_valid = profile_valid and records_valid and cells_valid
    if profile_valid and cells_valid:
        validated.append("profile.json")
    if records_valid:
        validated.append("records.jsonl")

    # ---- release-only gates --------------------------------------------
    if mode == "release":
        assert expectations_obj is not None and expectations_path is not None
        _release_legs(
            legs,
            expectations_obj,
            expectations_path,
            tree,
            snapshot,
            sha_by_rel,
            profile,
            records,
            legacy_parsed,
            closure,
            artifacts_valid,
        )

    failing = [leg for leg in legs if leg["outcome"] == "FAIL"]
    if failing:
        verdict = VERDICT_FAIL
    elif mode == "source":
        verdict = VERDICT_SOURCE_PASS
    else:
        verdict = VERDICT_RELEASE_PASS

    report = VerificationReport(
        mode=mode,
        verdict=verdict,
        legs=legs,
        validated_artifacts=validated,
        classifications=classifications,
    )

    payload = {
        "schema_version": receipt_mod.RECEIPT_SCHEMA_VERSION,
        "mode": mode,
        "verdict": verdict,
        "legs": legs,
        "validated_artifacts": validated,
        "classifications": classifications,
        "input_tree_sha256": _tree_digest_from_shas(sha_by_rel),
        "expectations_anchor_sha256": (
            hashlib.sha256(expectations_bytes).hexdigest()
            if expectations_bytes is not None
            else None
        ),
        "verifier_code_sha256": _code_digest(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    report.receipt_path = receipt_mod.emit_receipt(
        payload, receipts_dir=receipts_dir, verified_tree=tree
    )
    return report


# ---------------------------------------------------------------------------
# Release legs
# ---------------------------------------------------------------------------


def _release_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    expectations_path: Path,
    tree: Path,
    snapshot: dict[str, bytes],
    sha_by_rel: dict[str, str],
    profile: dict[str, Any],
    records: list[dict[str, Any]] | None,
    legacy_parsed: dict[str, dict[str, Any]],
    closure: str,
    artifacts_valid: bool,
) -> None:
    """Append every release-only leg, in the pinned order.

    Each section owns one gate family; a section that cannot even reach its
    inputs records the failure and returns rather than skipping silently.
    ``snapshot``/``sha_by_rel`` are the one-shot tree bytes/hashes (MA-HI-004).
    """
    prov = profile.get("provenance") or {}
    base = expectations_path.parent

    ledger_doc, external_ids = _anchor_legs(legs, exp, base, tree, prov)
    _tree_file_legs(legs, exp, sha_by_rel)
    _binding_legs(legs, exp, sha_by_rel, profile, prov)
    # QA-002: table-driven per-noun admissibility legs (replaces the former
    # scattered inline gates; leg ids preserved).
    _provenance_table_legs(legs, prov)
    _mc_build_consistency_leg(legs, prov)
    _model_revision_leg(legs, prov)
    _splits_recompute_leg(legs, prov, records)
    # MA-HAI-001: reconcile each cell estimand identity field against its
    # authoritative source (never a self-recomputed unanchored digest).
    _estimand_reconciliation_leg(legs, profile, prov, records, ledger_doc)
    _record_leg(
        legs,
        "closure_certifiability",
        closure == CERTIFIABLE,
        expected=CERTIFIABLE,
        observed=closure,
    )

    _rights_legs(legs, exp, base, tree, snapshot)
    _manifest_legs(legs, snapshot)
    # QA-003: the profile estimand set is the source of truth a row's
    # declared estimand must re-derive against.
    profile_estimands = {
        (cell.get("timing_summary_finite_only") or {}).get("estimand")
        for cell in (profile.get("cells") or [])
        if isinstance(cell, dict)
    }
    profile_estimands.discard(None)
    _ledger_legs(
        legs,
        ledger_doc,
        snapshot,
        legacy_parsed,
        artifacts_valid,
        prov,
        profile_estimands,
        external_ids,
    )


def _contained_reference(base: Path, rel: Any, tree: Path) -> Path | None:
    """Resolve an anchor/expectations-referenced sidecar path safely (MA-PI-001).

    Returns the joined path only when ``rel`` is a plain relative path that
    resolves (symlink-free) UNDER ``base`` and does NOT collapse INTO the
    verified ``tree`` — the containment that keeps the frozen ledger and
    rights inventory genuine out-of-tree anchors. Any absolute path, ``..``
    escape, symlink redirect out of base, or in-tree collapse returns None.
    """
    if not isinstance(rel, str) or not rel:
        return None
    candidate = Path(rel)
    if candidate.is_absolute():
        return None
    joined = base / candidate
    try:
        resolved = joined.resolve()
    except OSError:
        return None
    base_resolved = base.resolve()
    if not (resolved == base_resolved or base_resolved in resolved.parents):
        return None  # escapes base (via .. or a symlink redirect)
    if schema.resolves_inside(joined, tree):
        return None  # collapsed into the verified tree (self-attestation)
    return joined


def _anchor_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    base: Path,
    tree: Path,
    prov: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str] | None]:
    """Anchor cross-check before any expectation is consumed (R-013).

    Returns ``(ledger_doc, external_claim_ids)``: the frozen claim ledger
    document when reachable and parseable, and the anchored EXTERNAL claim-id
    list (QA-005) when declared.
    """
    anchor = exp.get("anchor")
    if not isinstance(anchor, dict):
        legs.append(
            _fail(
                "anchor",
                expected="anchor block binding a reviewed source commit and"
                " the frozen claim ledger (R-013)",
                observed="absent",
                remediation="MISSING_EXPECTATION",
            )
        )
        return None, None

    ledger_doc: dict[str, Any] | None = None
    ledger_rel = anchor.get("ledger_path", "ledger.json")
    anchor_ledger_sha = anchor.get("ledger_sha256")
    # MA-PI-001 (R-013): the frozen-ledger reference must be a plain relative
    # sibling under the expectations base — never an absolute path, a `..`
    # escape, a symlink redirect, or a path that collapses INTO the verified
    # tree. A contained-and-out-of-tree anchor is what makes third-party
    # re-verification meaningful.
    ledger_path = _contained_reference(base, ledger_rel, tree)
    if ledger_path is None:
        legs.append(
            _fail(
                "anchor_ledger",
                expected="frozen ledger at a relative sibling path under the"
                " expectations base, outside the verified tree (R-013)",
                observed=f"non-contained ledger_path {ledger_rel!r}",
                remediation="ARTIFACT_DEFECT",
            )
        )
    elif not ledger_path.is_file():
        legs.append(
            _fail(
                "anchor_ledger",
                expected=anchor_ledger_sha,
                observed=f"frozen ledger {ledger_rel!r} absent",
                remediation="MISSING_EXPECTATION",
            )
        )
    else:
        # CX-6: ONE bounded, regular-file-only read feeds both the anchor
        # hash and the parse — an oversized/irregular ledger is a typed leg
        # refusal, never an unbounded read_bytes() memory exhaustion, and
        # hash/parse can never desync (no double read).
        try:
            ledger_bytes = schema.read_regular_file_bytes(ledger_path)
        except schema.TypedIngressError as exc:
            legs.append(
                _fail(
                    "anchor_ledger",
                    expected=anchor_ledger_sha,
                    observed=str(exc),
                )
            )
        else:
            actual_ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
            _record_leg(
                legs,
                "anchor_ledger",
                actual_ledger_sha == anchor_ledger_sha,
                expected=anchor_ledger_sha,
                observed=actual_ledger_sha,
            )
            ledger_doc, ledger_err = _load_json_bytes_lenient(
                ledger_bytes, ledger_path.name
            )
            if ledger_err is not None:
                legs.append(
                    _fail(
                        "ledger_parse",
                        expected="parseable frozen claim ledger",
                        observed=ledger_err,
                    )
                )

    anchor_commit = anchor.get("source_commit")
    observed_commit = (prov.get("dirty_state") or {}).get("source_commit")
    if not schema.is_commit_sha(anchor_commit):
        legs.append(
            _fail(
                "anchor_source_commit",
                expected="full-length reviewed source commit SHA (R-013)",
                observed=anchor_commit,
                remediation="MISSING_EXPECTATION",
            )
        )
    elif anchor_commit != observed_commit:
        # String-exact identity comparison; works without a git checkout.
        legs.append(
            _fail(
                "anchor_source_commit",
                expected=anchor_commit,
                observed=observed_commit,
            )
        )
    else:
        legs.append(_pass("anchor_source_commit"))
        # MA-CC-5: the object-existence check is a SEPARATE leg bound to this
        # repository. False (repo available, object missing) FAILs; None (no
        # git available) is SKIPPED-with-reason — never silently passed.
        exists = _git_object_exists(anchor_commit)
        if exists is False:
            legs.append(
                _fail(
                    "anchor_source_commit_object",
                    expected=f"commit {anchor_commit} present in the"
                    " source repository (R-013)",
                    observed="object not found",
                )
            )
        elif exists is None:
            legs.append(
                _skipped(
                    "anchor_source_commit_object",
                    reason="source git repository unavailable for the"
                    " object-existence check (string-exact anchor still"
                    " enforced) (R-013)",
                )
            )
        else:
            legs.append(_pass("anchor_source_commit_object"))

    # QA-005: the EXTERNAL predicate is anchored membership — a list the
    # ledger editor cannot reach from inside the ledger document.
    external_ids = anchor.get("external_claim_ids")
    if isinstance(external_ids, list) and all(
        isinstance(cid, str) for cid in external_ids
    ):
        legs.append(_pass("anchor_external_claim_ids"))
        return ledger_doc, list(external_ids)
    legs.append(
        _fail(
            "anchor_external_claim_ids",
            expected="anchored external_claim_ids list in the expectations"
            " anchor block (R-024/QA-005)",
            observed="absent or malformed",
            remediation="MISSING_EXPECTATION",
        )
    )
    return ledger_doc, None


def _tree_file_legs(
    legs: list[dict[str, Any]], exp: dict[str, Any], sha_by_rel: dict[str, str]
) -> None:
    """Artifact-tree byte identity against the anchored hash map (R-014).

    Hashes come from the one-shot snapshot (MA-HI-004), so this leg and the
    receipt digest attest identical bytes.
    """
    declared_tree = exp.get("tree_files")
    if not isinstance(declared_tree, dict):
        legs.append(
            _fail(
                "tree_files",
                expected="tree byte-hash map in the expectations file",
                observed="absent",
                remediation="MISSING_EXPECTATION",
            )
        )
        return

    problems: list[str] = []
    actual = dict(sha_by_rel)
    for rel, sha in sorted(declared_tree.items()):
        if rel not in actual:
            problems.append(f"declared-but-absent {rel!r}")
        elif actual[rel] != sha:
            problems.append(f"byte-hash mismatch {rel!r}")
    for rel in sorted(set(actual) - set(declared_tree)):
        problems.append(f"present-but-unanchored {rel!r}")
    _record_leg(
        legs,
        "tree_files",
        not problems,
        expected="artifact tree byte-identical to the anchored hash map"
        " (R-014)",
        observed="; ".join(problems),
    )


def _binding_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    sha_by_rel: dict[str, str],
    profile: dict[str, Any],
    prov: dict[str, Any],
) -> None:
    """The thirteen independently anchored binding legs (R-012).

    File hashes come from the one-shot snapshot (MA-HI-004).
    """
    observed_bindings: dict[str, Any] = {
        "schema_profile": {
            "profile_id": profile.get("profile_id"),
            "schema_version": profile.get("schema_version"),
            "profile_sha256": sha_by_rel.get("profile.json"),
        },
        "producer": {
            "entrypoint": prov.get("producer_entrypoint"),
            "sha256": prov.get("producer_sha256"),
            "helper_sha256s": prov.get("helper_sha256s"),
        },
        "semantic_command": prov.get("semantic_command"),
        "seeds": prov.get("seeds"),
        "dirty_state": prov.get("dirty_state"),
        "splits": prov.get("splits"),
        "calibration_identity": prov.get("calibration_identity"),
        "continuation_identity": prov.get("continuation_identity"),
        "input_hashes": prov.get("input_sha256"),
        "split_metadata_sha256": prov.get("split_metadata_sha256"),
        "mc_build": prov.get("mc_build"),
        "model": prov.get("model"),
        "runtime_packages": prov.get("runtime_packages"),
    }
    bindings = exp.get("bindings")
    if not isinstance(bindings, dict):
        legs.append(
            _fail(
                "bindings",
                expected="per-leg bindings block in the expectations file",
                observed="absent",
                remediation="MISSING_EXPECTATION",
            )
        )
        bindings = {}
    for key in BINDING_KEYS:
        _check_binding(legs, key, observed_bindings[key], bindings, sha_by_rel)


def _check_binding(
    legs: list[dict[str, Any]],
    key: str,
    observed: Any,
    bindings: dict[str, Any],
    sha_by_rel: dict[str, str],
) -> None:
    """One binding leg = two obligations (QA-001): the observed value must be
    admissible in its own right AND must match the anchored expectation.

    Admissibility fires first and is a distinct ARTIFACT_DEFECT — a defect
    mirrored into the expectations (author-controlled proxy) still fails.
    """
    leg_id = f"binding_{key}"
    problem = BINDING_VALIDATORS[key](observed)
    if problem is not None:
        legs.append(
            _fail(
                leg_id,
                expected=f"admissible {key} binding value —"
                " never null/empty/UNRESOLVED (R-012/QA-001);"
                f" defect: {problem}",
                observed=observed,
            )
        )
        return
    if key not in bindings:
        legs.append(
            _fail(
                leg_id,
                expected="<missing anchored expectation>",
                observed=observed,
                remediation="MISSING_EXPECTATION",
            )
        )
        return
    expected = bindings[key]
    if expected != observed:
        legs.append(_fail(leg_id, expected=expected, observed=observed))
        return
    if key == "input_hashes" and isinstance(expected, dict):
        mismatched: dict[str, Any] = {}
        for fname, sha in expected.items():
            actual_sha = sha_by_rel.get(fname, "absent")
            if actual_sha != sha:
                mismatched[fname] = actual_sha
        if mismatched:
            legs.append(_fail(leg_id, expected=expected, observed=mismatched))
            return
    legs.append(_pass(leg_id))


def _mc_build_consistency_leg(
    legs: list[dict[str, Any]], prov: dict[str, Any]
) -> None:
    """QA-018 (R-012): mc_build values must be internally consistent, not
    merely recorded — retained_count > 0 iff coverage_rate > 0, and the
    retained count squares with the eval-split size under the declared
    retention policy."""
    mc_build = prov.get("mc_build") or {}
    coverage = mc_build.get("coverage_rate")
    retained = mc_build.get("retained_count")
    policy = mc_build.get("retention_policy")
    eval_count = ((prov.get("splits") or {}).get("eval") or {}).get("count")

    problems: list[str] = []
    if schema.is_number(coverage) and isinstance(retained, int) and not (
        isinstance(retained, bool)
    ):
        if (retained > 0) != (float(coverage) > 0.0):
            problems.append(
                f"retained_count {retained!r} inconsistent with"
                f" coverage_rate {coverage!r} (retained_count > 0 iff"
                " coverage_rate > 0)"
            )
        if (
            isinstance(eval_count, int)
            and not isinstance(eval_count, bool)
            and eval_count > 0
        ):
            if retained > eval_count:
                problems.append(
                    f"retained_count {retained!r} exceeds splits.eval.count"
                    f" {eval_count!r}"
                )
            if (
                policy == "retain_all"
                and float(coverage) == 1.0
                and retained != eval_count
            ):
                problems.append(
                    f"retention_policy 'retain_all' with full coverage must"
                    f" retain the whole eval split ({eval_count!r}), recorded"
                    f" {retained!r}"
                )
    _record_leg(
        legs,
        "mc_build_internal_consistency",
        not problems,
        expected="mc_build coverage/retention internally consistent with"
        " the eval split (R-012/QA-018)",
        observed="; ".join(problems),
    )


def _model_revision_leg(
    legs: list[dict[str, Any]], prov: dict[str, Any]
) -> None:
    """Immutable model revision identity (either-form gate, R-012)."""
    model = prov.get("model") or {}
    revision = model.get("revision")
    digest_manifest = model.get("byte_digest_manifest")
    if revision is not None:
        # Short hashes, tags, branch names, and bare repo ids are rejected —
        # repo ids are reassignable (R-012).
        _record_leg(
            legs,
            "model_revision_immutability",
            schema.is_commit_sha(revision),
            expected="immutable full-length 40-hex commit SHA, or a"
            " complete canonical byte-digest manifest",
            observed={"revision": revision},
        )
        return
    complete_manifest = (
        isinstance(digest_manifest, dict)
        and bool(digest_manifest)
        and all(schema.is_sha256_hex(v) for v in digest_manifest.values())
    )
    _record_leg(
        legs,
        "model_revision_immutability",
        complete_manifest,
        expected="immutable full-length commit SHA or complete"
        " canonical byte-digest manifest",
        observed={
            "revision": None,
            "byte_digest_manifest": digest_manifest,
        },
    )


def _splits_recompute_leg(
    legs: list[dict[str, Any]],
    prov: dict[str, Any],
    records: list[dict[str, Any]] | None,
) -> None:
    """Eval-split key-set recomputation against retained records (R-012)."""
    if records is None:
        return
    splits = prov.get("splits") or {}
    eval_split = splits.get("eval") or {}
    keys = [
        r.get("item_key") for r in records if isinstance(r.get("item_key"), str)
    ]
    recomputed_hash = _keyset_sha256(keys)
    declared_hash = eval_split.get("keyset_sha256")
    declared_count = eval_split.get("count")
    _record_leg(
        legs,
        "splits_eval_recompute",
        declared_hash == recomputed_hash and declared_count == len(keys),
        expected={"keyset_sha256": declared_hash, "count": declared_count},
        observed={"keyset_sha256": recomputed_hash, "count": len(keys)},
    )


def _authorized_random_k_draws(ledger_doc: dict[str, Any] | None) -> set[str]:
    """Draw ids the frozen ledger authorizes via a sanctioned Random-K
    disposition (R-025). Any other draw id in a cell estimand is unanchored
    and must be refused (MA-HAI-001)."""
    authorized: set[str] = {"draw-none"}  # the explicit no-draw sentinel
    if not isinstance(ledger_doc, dict):
        return authorized
    for row in ledger_doc.get("rows") or []:
        if not isinstance(row, dict):
            continue
        if row.get("artifact_family") != "random_k":
            continue
        if row.get("author_decision") not in ledger_mod.RANDOM_K_DISPOSITIONS:
            continue
        draw = row.get("random_k_draw_id")
        if isinstance(draw, str) and draw:
            authorized.add(draw)
    return authorized


def _estimand_reconciliation_leg(
    legs: list[dict[str, Any]],
    profile: dict[str, Any],
    prov: dict[str, Any],
    records: list[dict[str, Any]] | None,
    ledger_doc: dict[str, Any] | None,
) -> None:
    """Reconcile every cell estimand identity field against its authoritative
    source (MA-HAI-001, R-011/R-025).

    The estimand digest is self-recomputed from the estimand block, so a
    FABRICATED estimand (a mismatched calibration/continuation identity, a
    wrong timeout horizon, a non-``n_complete`` denominator, an arm id absent
    from the profile, or a substituted favorable Random-K draw) yields a
    self-consistent digest and would otherwise reach PASS_RELEASE. This leg
    binds each estimand-defining field to an independently verified value.
    """
    arm_ids = {
        arm.get("arm_id")
        for arm in (profile.get("arms") or [])
        if isinstance(arm, dict)
    }
    # The horizon actually applied to the records (the authoritative source
    # for timeout_parameters.trajectory_horizon). CX-3: records may carry the
    # horizon as a shared ``trajectory_horizon`` OR as (equal) per-arm
    # ``mc_trajectory_horizon``/``ref_trajectory_horizon`` fields — both
    # forms feed the authoritative set, so a per-arm-horizon package cannot
    # produce an EMPTY set and skip the comparison. Distinct per-arm values
    # widen the set and (correctly) fail an estimand claiming one horizon.
    record_horizons: set[Any] = set()
    for r in records or []:
        if not isinstance(r, dict):
            continue
        shared = r.get("trajectory_horizon")
        for h in (
            r.get("mc_trajectory_horizon", shared),
            r.get("ref_trajectory_horizon", shared),
        ):
            if schema.is_number(h):
                record_horizons.add(h)
    authorized_draws = _authorized_random_k_draws(ledger_doc)

    problems: list[str] = []
    for cell in profile.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        cid = cell.get("cell_id", "unnamed")
        est = cell.get("estimand") or {}
        if est.get("calibration_identity") != prov.get("calibration_identity"):
            problems.append(
                f"{cid}: estimand.calibration_identity"
                f" {est.get('calibration_identity')!r} != provenance"
                f" {prov.get('calibration_identity')!r}"
            )
        if est.get("continuation_identity") != prov.get("continuation_identity"):
            problems.append(
                f"{cid}: estimand.continuation_identity"
                f" {est.get('continuation_identity')!r} != provenance"
                f" {prov.get('continuation_identity')!r}"
            )
        if est.get("denominator_policy") != "n_complete":
            problems.append(
                f"{cid}: estimand.denominator_policy"
                f" {est.get('denominator_policy')!r} != 'n_complete'"
            )
        est_horizon = (est.get("timeout_parameters") or {}).get(
            "trajectory_horizon"
        )
        if record_horizons and {est_horizon} != record_horizons:
            problems.append(
                f"{cid}: estimand timeout horizon {est_horizon!r} !="
                f" the horizon applied to records {sorted(record_horizons)}"
            )
        for role in ("arm_mc", "arm_ref"):
            if est.get(role) not in arm_ids:
                problems.append(
                    f"{cid}: estimand.{role} {est.get(role)!r} not among the"
                    f" profile arms {sorted(a for a in arm_ids if a)}"
                )
        draw = est.get("random_k_draw_id")
        if draw not in authorized_draws:
            problems.append(
                f"{cid}: estimand.random_k_draw_id {draw!r} is not authorized"
                " by a sanctioned Random-K disposition in the frozen ledger"
                " (a substituted favorable draw is refused)"
            )
    _record_leg(
        legs,
        "estimand_reconciliation",
        not problems,
        expected="every cell estimand identity field reconciles with its"
        " authoritative source (provenance/records/arms/ledger)"
        " (R-011/R-025/MA-HAI-001)",
        observed="; ".join(problems),
    )


def _rights_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    base: Path,
    tree: Path,
    snapshot: dict[str, bytes],
) -> None:
    """Rights inventory binding + release clearance (R-026/R-035)."""
    rights_decl = exp.get("rights_inventory")
    if not isinstance(rights_decl, dict):
        legs.append(
            _fail(
                "rights_inventory",
                expected="rights inventory binding (path + sha256) in the"
                " expectations file (R-026)",
                observed="absent",
                remediation="MISSING_EXPECTATION",
            )
        )
        return

    # MA-PI-001 (R-013): the rights inventory reference is contained the same
    # way the frozen ledger is — relative, under base, out of the tree.
    rights_rel = rights_decl.get("path", "rights.json")
    rights_path = _contained_reference(base, rights_rel, tree)
    if rights_path is None:
        legs.append(
            _fail(
                "rights_inventory",
                expected="rights inventory at a relative sibling path under"
                " the expectations base, outside the verified tree (R-013)",
                observed=f"non-contained rights path {rights_rel!r}",
            )
        )
        return
    # CX-6 (sibling site): one bounded read feeds the rights hash + parse.
    try:
        rights_bytes = schema.read_regular_file_bytes(rights_path)
    except schema.TypedIngressError as exc:
        legs.append(
            _fail(
                "rights_inventory",
                expected="readable, bounded, regular-file rights inventory",
                observed=str(exc),
            )
        )
        return
    rights_obj, rights_err = _load_json_bytes_lenient(
        rights_bytes, rights_path.name
    )
    if rights_err is not None:
        legs.append(
            _fail(
                "rights_inventory",
                expected="parseable rights inventory",
                observed=rights_err,
            )
        )
        return

    actual_sha = hashlib.sha256(rights_bytes).hexdigest()
    _record_leg(
        legs,
        "rights_inventory_hash",
        actual_sha == rights_decl.get("sha256"),
        expected=rights_decl.get("sha256"),
        observed=actual_sha,
    )
    try:
        # R-035: rights cover every file FOUND, not merely declared.
        ledger_mod.check_rights_release(rights_obj, sorted(snapshot))
        legs.append(_pass("rights_release"))
    except ledger_mod.RightsError as exc:
        legs.append(
            _fail(
                "rights_release",
                expected="every included path VERIFIED_ALLOWED and"
                " inventoried (R-026)",
                observed=str(exc),
                remediation="AUTHOR_DECISION_REQUIRED",
            )
        )


def _manifest_legs(
    legs: list[dict[str, Any]], snapshot: dict[str, bytes]
) -> None:
    """Presentation manifest reconciliation (R-033/R-035).

    The manifest is a TREE member, so it is read from the one-shot snapshot
    (MA-HI-004) — never re-read from disk.
    """
    files = snapshot
    if "presentation_manifest.json" not in snapshot:
        legs.append(
            _fail(
                "presentation_manifest_present",
                expected="presentation_manifest.json in the artifact tree",
                observed="absent",
            )
        )
        return

    manifest_obj, manifest_err = _load_json_bytes_lenient(
        snapshot["presentation_manifest.json"], "presentation_manifest.json"
    )
    if manifest_err is None:
        unknown_keys = sorted(set(manifest_obj) - _MANIFEST_KEYS)
        if unknown_keys:
            manifest_err = f"unknown manifest key(s) {unknown_keys}"
    if manifest_err is not None:
        legs.append(
            _fail(
                "presentation_manifest_parse",
                expected="typed presentation manifest",
                observed=manifest_err,
            )
        )
        return

    declared = [
        a.get("path")
        for a in manifest_obj.get("artifacts", [])
        if isinstance(a, dict) and isinstance(a.get("path"), str)
    ]
    allowlist = [
        p
        for p in manifest_obj.get("allowlist_undeclared", [])
        if isinstance(p, str)
    ]
    _record_leg(
        legs,
        "manifest_nonempty",
        bool(declared),
        expected=">=1 manifest-declared artifact (R-033)",
        observed="0 declared artifacts",
    )
    ghosts = sorted(p for p in declared if p not in files)
    _record_leg(
        legs,
        "manifest_declared_absent",
        not ghosts,
        expected="every manifest-declared artifact present",
        observed=ghosts,
    )
    undeclared = sorted(
        rel
        for rel in files
        if rel not in declared
        and rel not in allowlist
        and rel != "presentation_manifest.json"
    )
    _record_leg(
        legs,
        "manifest_undeclared_present",
        not undeclared,
        expected="no present-but-undeclared file without an explicit"
        " per-file allowlist entry (R-035)",
        observed=undeclared,
    )


def _ledger_legs(
    legs: list[dict[str, Any]],
    ledger_doc: dict[str, Any] | None,
    snapshot: dict[str, bytes],
    legacy_parsed: dict[str, dict[str, Any]],
    artifacts_valid: bool,
    prov: dict[str, Any],
    profile_estimands: set[str],
    external_ids: list[str] | None,
) -> None:
    """Claim-ledger validation + per-row status recomputation (R-012/R-033).

    QA-005: the EXTERNAL predicate is membership in the anchored
    ``external_claim_ids`` list — never the row's own hand-editable fields.
    """
    if ledger_doc is None:
        legs.append(
            _fail(
                "ledger_present",
                expected="frozen claim ledger reachable via the anchor",
                observed="absent or unparseable",
                remediation="MISSING_EXPECTATION",
            )
        )
        return
    try:
        # QA-015: the anchored EXTERNAL predicate reaches the rule-named
        # surface too — validate_ledger enforces membership itself.
        ledger_mod.validate_ledger(
            ledger_doc, external_claim_ids=external_ids
        )
        legs.append(_pass("ledger_validation"))
    except ledger_mod.LedgerValidationError as exc:
        legs.append(
            _fail(
                "ledger_validation",
                expected="structurally valid claim ledger (R-023..R-025,"
                " R-030)",
                observed=str(exc),
            )
        )
    rows = ledger_doc.get("rows") or []
    _record_leg(
        legs,
        "ledger_nonempty",
        bool(rows),
        expected=">=1 retained claim-ledger row (R-033)",
        observed="empty ledger rows",
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        claim_id = row.get("claim_id", "unnamed")
        if external_ids is not None and claim_id in external_ids:
            # Anchored-EXTERNAL row: immune to repo recompute (R-024), but a
            # transition away from EXTERNAL demands human attribution.
            attribution = row.get("human_attribution")
            attributed = (
                isinstance(attribution, dict)
                and bool(attribution.get("attributed_to"))
                and bool(attribution.get("date"))
            )
            _record_leg(
                legs,
                f"ledger_row_{claim_id}_external_immunity",
                status == "EXTERNAL" or attributed,
                expected="anchored-EXTERNAL row recorded EXTERNAL, or a"
                " human-attributed transition (R-024/QA-005)",
                observed={
                    "claim_id": claim_id,
                    "recorded": status,
                    "human_attribution": attributed,
                },
            )
            continue
        if status == "EXTERNAL":
            # QA-005 (reverse laundering): a row cannot grant ITSELF
            # recompute immunity by relabeling its status EXTERNAL.
            outcome_ok = external_ids is None  # fail-closed leg already fired
            _record_leg(
                legs,
                f"ledger_row_{claim_id}_external_immunity",
                outcome_ok,
                expected="EXTERNAL status only on rows anchored in the"
                " expectations external_claim_ids list (R-024/QA-005)",
                observed={
                    "claim_id": claim_id,
                    "recorded": "EXTERNAL",
                    "anchored_external": False,
                },
            )
            continue
        recomputed = _recompute_row_status(
            row, snapshot, legacy_parsed, artifacts_valid, prov,
            profile_estimands,
        )
        _record_leg(
            legs,
            f"ledger_row_{claim_id}_recompute",
            _STATUS_STRENGTH.get(status, 0) <= _STATUS_STRENGTH[recomputed],
            expected="recorded status no stronger than the recomputed"
            f" status {recomputed!r} (R-012)",
            observed={
                "claim_id": claim_id,
                "recorded": status,
                "recomputed": recomputed,
            },
        )


def _provenance_identity_closure(prov: dict[str, Any]) -> set[str]:
    """Every content-hash identity the verified provenance records (QA-003).

    A row's ``input_identity`` must re-derive to a member of this closure —
    an identity matching nothing in the verified provenance is unverifiable.
    """
    closure: set[Any] = set()
    closure.update((prov.get("input_sha256") or {}).values())
    closure.update((prov.get("helper_sha256s") or {}).values())
    closure.add(prov.get("split_metadata_sha256"))
    closure.add(prov.get("producer_sha256"))
    model = prov.get("model") or {}
    closure.add(model.get("weights_sha256"))
    closure.add(model.get("tokenizer_config_sha256"))
    closure.update((model.get("byte_digest_manifest") or {}).values())
    closure.discard(None)
    return {value for value in closure if isinstance(value, str)}


def _recompute_row_status(
    row: dict[str, Any],
    snapshot: dict[str, bytes],
    legacy_parsed: dict[str, dict[str, Any]],
    artifacts_valid: bool,
    prov: dict[str, Any],
    profile_estimands: set[str],
) -> str:
    """Recompute a non-EXTERNAL claim row's status from current verification
    (R-012). Recompute means RE-DERIVE from the verified source of truth —
    the row's identity fields are cross-checked against the verified profile
    provenance, never proxy-checked against themselves (QA-003).
    """
    if row.get("rights_status") != "VERIFIED_ALLOWED":
        return "UNVERIFIED"
    artifact = row.get("artifact_id")
    if not isinstance(artifact, str) or artifact not in snapshot:
        return "UNVERIFIED"
    # QA-003: identity cross-checks against the verified provenance.
    if row.get("producer_entrypoint") != prov.get("producer_entrypoint"):
        return "UNVERIFIED"
    if row.get("calibration_identity") != prov.get("calibration_identity"):
        return "UNVERIFIED"
    splits = prov.get("splits") or {}
    split_names = {
        (splits.get("fit") or {}).get("name"),
        (splits.get("eval") or {}).get("name"),
    }
    split_names.discard(None)
    if row.get("split_identity") not in split_names:
        return "UNVERIFIED"
    # QA-016 (R-012): no partial-string identity comparisons — the row's
    # model identity is decomposed into namespace@revision and EVERY
    # component is cross-checked against the verified provenance. Short
    # hashes, tags, branch names, and bare repo ids are exactly the mutable
    # forms R-012 rejects.
    model = prov.get("model") or {}
    namespace = model.get("repository_namespace")
    model_identity = row.get("model_identity")
    if not isinstance(model_identity, str) or "@" not in model_identity:
        return "UNVERIFIED"  # bare repo id: reassignable, never an identity
    row_namespace, row_revision = model_identity.split("@", 1)
    if not isinstance(namespace, str) or row_namespace != namespace:
        return "UNVERIFIED"
    if not row_revision:
        return "UNVERIFIED"
    prov_revision = model.get("revision")
    if prov_revision is not None:
        # The row's claimed revision must be an immutable full-length commit
        # SHA AND equal the verified provenance's revision.
        if not schema.is_commit_sha(row_revision):
            return "UNVERIFIED"
        if row_revision != prov_revision:
            return "UNVERIFIED"
    else:
        # Byte-digest alternative: the row's revision part must name one of
        # the anchored canonical byte digests.
        digest_values = set(
            (model.get("byte_digest_manifest") or {}).values()
        )
        if row_revision not in digest_values:
            return "UNVERIFIED"
    # DECISION: input_identity must be a member of the verified provenance's
    # content-hash identity closure.
    if row.get("input_identity") not in _provenance_identity_closure(prov):
        return "UNVERIFIED"
    # QA-003: the claimed estimand must exist in the verified profile.
    if row.get("estimand") not in profile_estimands:
        return "UNVERIFIED"
    # QA-004: the claim-kind discriminant is a validated closed enum; a
    # missing/renamed discriminant is rejected, never routed permissively.
    claim_kind = row.get("claim_kind")
    if claim_kind not in ("aggregate", "per_item_paired"):
        return "UNVERIFIED"
    if artifact in legacy_parsed:
        if not legacy_certifies(legacy_parsed[artifact], claim_kind):
            return "UNVERIFIED"
        return "PASS"
    # QA-003: an arbitrary tree file certifies nothing — PASS requires the
    # strictly validated evidence artifacts (or a legacy family above).
    if artifact not in ("profile.json", "records.jsonl"):
        return "UNVERIFIED"
    if not artifacts_valid:
        return "UNVERIFIED"
    return "PASS"
