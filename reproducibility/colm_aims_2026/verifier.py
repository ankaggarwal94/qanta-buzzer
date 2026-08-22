"""Two-mode fail-closed verifier (source-contract / release), v2.

Spec rules owned here: R-012..R-015, R-017, R-021 (as CLI backend), R-033,
R-035, R-036 (emission call), R-040..R-056 (grid/event/inference legs),
R-064 (sidecar boundary), R-065/R-066 (anchor equality + git object),
R-068 (estimand label binding), R-069/R-039 (canonical selection wired into
the actual release path).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
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

import numpy as np

from . import ledger as ledger_mod
from . import pairing, receipt as receipt_mod, schema
from .schema import ColmAimsError

# The object-existence check binds to THIS repository explicitly — never
# ambient cwd/.git — so no cwd move or GIT_* env var can flip the gate.
_SOURCE_REPO = Path(__file__).resolve().parents[2]
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
    """Expectations/tree containment violation (R-013)."""


# R-017: closed verdict vocabularies; strongest source member is
# PASS_SOURCE_ONLY, and no source-mode code path emits a release token.
VERDICT_SOURCE_PASS = "PASS_SOURCE_ONLY"
VERDICT_RELEASE_PASS = "PASS_RELEASE"
VERDICT_FAIL = "FAIL"
SOURCE_MODE_VERDICTS = frozenset({VERDICT_SOURCE_PASS, VERDICT_FAIL})
RELEASE_MODE_VERDICTS = frozenset({VERDICT_RELEASE_PASS, VERDICT_FAIL})

CERTIFIABLE = "CERTIFIABLE"
HISTORICAL_NONCERTIFYING = "HISTORICAL_NONCERTIFYING"

# Exact v2 leg-id vocabulary (helpers DECISION).
LEG_TYPED_INGRESS = "typed_ingress"
LEG_PROFILE_VALIDATION = "profile_validation"
LEG_GRID_COMPLETENESS = "grid_completeness"
LEG_RECORD_FILE_BIJECTION = "grid_record_file_bijection"
LEG_ITEM_KEY_SET = "grid_item_key_set_equality"
LEG_HELD_FIXED = "grid_held_fixed_identities"
LEG_MC_STOP_WITHIN_CAL = "grid_mc_stop_within_calibration"
LEG_EVENT_REPRESENTATION = "event_representation"
LEG_COUNTS = "counts_identities"
LEG_RATES = "rates"
LEG_ESTIMAND_LABELS = "estimand_label_binding"
LEG_CELL_COMPARABILITY = "cell_comparability"
LEG_INFERENCE_SEED = "inference_seed_derivation"
LEG_INFERENCE_MATRIX = "inference_resample_matrix_digest"
LEG_INFERENCE_RECOMPUTE = "inference_recompute"
LEG_INFERENCE_HOLM = "inference_holm_family"
LEG_LEDGER_ANCHOR_EQ = "ledger_anchor_commit_equality"
LEG_GIT_OBJECT = "anchor_source_commit_object"
LEG_ANCHOR_COMMIT = "anchor_source_commit"
LEG_CANONICAL_SELECTION = "canonical_selection"
SIDECAR_LEG_PREFIX = "sidecar_ingress:"
ANCHORED_GRID_PREFIX = "anchored_grid_"
ANCHORED_INFERENCE_PREFIX = "anchored_inference_"

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
# R-063 (Track A' R1): the anchor block is CLOSED with every key REQUIRED —
# a dict.get default engaging on a typo'd anchor key must never silently
# disable a gate.
_ANCHOR_KEYS = frozenset(
    {"source_commit", "ledger_path", "ledger_sha256", "external_claim_ids"}
)
_RIGHTS_DECL_KEYS = frozenset({"path", "sha256"})
_MANIFEST_KEYS = frozenset(
    {"schema_version", "artifacts", "allowlist_undeclared"}
)

# R-044: the anchored grid pins compared field-by-field at release.
ANCHORED_GRID_FIELDS = (
    "reference_ids",
    "calibration_ids",
    "cell_ids",
    "record_files",
    "item_keys_sha256",
    "held_fixed",
)
# R-052(b)/R-053: the anchored inference pins.
ANCHORED_INFERENCE_FIELDS = (
    "seed",
    "seed_derivation",
    "pairing_population_keyset_sha256",
    "canonical_item_order_digest",
    "resample_matrix_digest",
    "draw_count",
    "numpy_version",
    "bit_generator",
)

# Fields that legitimately vary across the ten cells' estimands; every OTHER
# estimand field must be identical for the cells to pool into one Holm
# family (R-011 comparability via ``pairing.check_comparable``).
_ESTIMAND_AXIS_FIELDS = frozenset(
    {
        "reference_id",
        "calibration_id",
        "arm_ref",
        "calibration_identity",
        "random_k_draw_id",
    }
)

# The ledger estimand identity of the D7(b) Holm family row.
INFERENCE_FAMILY_ESTIMAND = "d7b_holm_family_m10"

_CLOSURE_IDENTITY_KEYS = ("continuation_identity",)

_EXPECTED_LAYOUT = (
    "profile.json (strict v2 ten-cell profile), records/<cell_id>.jsonl"
    " (per-cell retained records), presentation_manifest.json"
    " (presentation manifest)"
)

_STATUS_STRENGTH = {"FAIL": 0, "UNVERIFIED": 1, "PASS": 2}


@dataclass
class VerificationReport:
    """Structured result of one verifier run."""

    mode: str
    verdict: str
    legs: list[dict[str, Any]] = field(default_factory=list)
    validated_artifacts: list[str] = field(default_factory=list)
    receipt_path: Path | None = None
    classifications: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Leg plumbing
# ---------------------------------------------------------------------------


def _pass(leg_id: str) -> dict[str, Any]:
    return {"leg_id": leg_id, "status": "PASS"}


def _fail(
    leg_id: str,
    *,
    expected: Any,
    observed: Any,
    remediation: str = "ARTIFACT_DEFECT",
) -> dict[str, Any]:
    return {
        "leg_id": leg_id,
        "status": "FAIL",
        "expected": expected,
        "observed": observed,
        "remediation": remediation,
    }


def _record_leg(
    legs: list[dict[str, Any]],
    leg_id: str,
    passed: bool,
    *,
    expected: Any,
    observed: Any,
    remediation: str = "ARTIFACT_DEFECT",
) -> None:
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


# Gate arithmetic must never abort a run: typed errors plus the classic
# untyped escapes (including OverflowError, FD-001 lineage) fail the LEG.
_LEG_CATCH = (ColmAimsError, KeyError, TypeError, ValueError, OverflowError)


def _as_dict(value: Any) -> dict[str, Any]:
    """Shape coercion for artifact-controlled blocks (MA2-002a): {} for ANY
    non-dict — including truthy lists, which the ``value or {}`` idiom let
    through into ``.get``/``.items`` crashes."""
    return value if isinstance(value, dict) else {}


def _str_list(value: Any) -> tuple[list[str], list[str]]:
    """Split an artifact-controlled list into its string entries and the
    reprs of any non-string entries (MA2-002a): mixed-type lists must fail
    the owning LEG, never crash ``sorted`` with a TypeError."""
    if not isinstance(value, list):
        return [], ([] if value is None else [f"<{type(value).__name__}>"])
    strings = [v for v in value if isinstance(v, str)]
    bad = [f"<{type(v).__name__}>" for v in value if not isinstance(v, str)]
    return strings, bad


def _guarded(
    legs: list[dict[str, Any]],
    leg_id: str,
    builder: Callable[..., Any],
    *args: Any,
    default: Any = None,
) -> Any:
    """Run one leg builder; an UNEXPECTED exception inside it becomes THAT
    leg's FAIL — the run still reaches a verdict and a receipt, never an
    internal abort (MA2-002b class fix).

    Typed control flow keeps its semantics: every ``ColmAimsError``-family
    exception (typed ingress raises, ``EmptyEvaluationError``'s deliberate
    pre-report abort, containment/config errors) is re-raised unchanged.
    The failure message names the exception class and stays free of
    absolute paths and artifact content.
    """
    try:
        return builder(*args)
    except ColmAimsError:
        raise
    except Exception as exc:  # noqa: BLE001 - the class fix by definition
        detail = str(exc)
        if len(detail) > 300:
            detail = detail[:300] + "…"
        legs.append(
            _fail(
                leg_id,
                expected="leg evaluation completes without an internal"
                " error (shape defects fail the leg, never the run)"
                " (MA2-002)",
                observed=(
                    f"unexpected {exc.__class__.__name__} while evaluating"
                    f" this leg: {detail}"
                ),
            )
        )
        return default


# ---------------------------------------------------------------------------
# Hash/tree helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_tree_snapshot(tree: Path) -> dict[str, bytes]:
    """Read every tree member's bytes ONCE, symlink-free.

    The single content-addressed load that drives content validation, every
    binding/tree-file hash, and the receipt's ``input_tree_sha256`` — the
    receipt provably attests exactly the bytes the gates saw. A symlink
    member (or a member resolving outside the tree) is refused with a typed
    containment error rather than followed.
    """
    tree = Path(tree)
    snapshot: dict[str, bytes] = {}
    for p in sorted(tree.rglob("*")):
        if p.is_symlink():
            rel = p.relative_to(tree).as_posix()
            raise ContainmentError(
                f"tree member {rel!r} is a symlink — refusing to follow,"
                " read, or hash bytes outside the verified tree"
                " (R-036/R-013)"
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
    return {
        rel: hashlib.sha256(data).hexdigest()
        for rel, data in snapshot.items()
    }


def _digest_over_lines(lines: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(lines) + "\n").encode("utf-8")
    ).hexdigest()


def _tree_digest_from_shas(sha_by_rel: dict[str, str]) -> str:
    return _digest_over_lines(
        [f"{rel}:{sha}" for rel, sha in sorted(sha_by_rel.items())]
    )


def _code_digest() -> str:
    namespace = Path(__file__).resolve().parent
    return _digest_over_lines(
        [
            f"{p.relative_to(namespace).as_posix()}:{_sha256_file(p)}"
            for p in sorted(namespace.glob("**/*.py"))
        ]
    )


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


# ---------------------------------------------------------------------------
# Certifiability + legacy sidecars
# ---------------------------------------------------------------------------


def _is_resolved_identity(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value != "UNRESOLVED"


def classify_certifiability(profile: dict[str, Any]) -> str:
    """CERTIFIABLE vs HISTORICAL_NONCERTIFYING closure classification
    (R-014). Only a producer/closure change invalidates an artifact."""
    if not isinstance(profile, dict):
        raise ColmAimsError("profile must be an object (R-014)")
    prov = _as_dict(profile.get("provenance"))
    dirty_state = _as_dict(prov.get("dirty_state"))
    if dirty_state.get("git_dirty") is not False:
        return HISTORICAL_NONCERTIFYING
    if "superseded_by_producer_sha256" in prov:
        return HISTORICAL_NONCERTIFYING
    calibration = prov.get("calibration_identity")
    if not isinstance(calibration, dict) or not calibration:
        return HISTORICAL_NONCERTIFYING
    for value in calibration.values():
        if not _is_resolved_identity(value):
            return HISTORICAL_NONCERTIFYING
    for name in _CLOSURE_IDENTITY_KEYS:
        if not _is_resolved_identity(prov.get(name)):
            return HISTORICAL_NONCERTIFYING
    return CERTIFIABLE


def _classify_tree_sidecars(
    snapshot: dict[str, bytes],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """Classify tree ``*.json`` sidecars per the R-064 pinned boundary.

    Returns ``(legacy_by_rel, classifications, ingress_defects)``. A
    well-formed unknown-family OBJECT stays a tolerated historical sidecar;
    a top-level array/string/number/Boolean/null is ingress-DEFECTIVE, as
    are malformed JSON, invalid UTF-8, non-finite tokens, and overlong
    integer tokens.
    """
    from . import legacy as legacy_mod

    legacy_by_rel: dict[str, dict[str, Any]] = {}
    classifications: dict[str, str] = {}
    ingress_defects: dict[str, str] = {}
    for rel in sorted(snapshot):
        if rel in ("profile.json", "presentation_manifest.json"):
            continue
        if rel.startswith("records/") and rel.endswith(".jsonl"):
            continue
        if not rel.endswith(".json"):
            continue
        try:
            obj = schema.parse_json_bytes_strict(snapshot[rel])
        except schema.TypedIngressError as exc:
            ingress_defects[rel] = str(exc)
            continue
        except UnicodeDecodeError as exc:
            ingress_defects[rel] = (
                f"invalid UTF-8 bytes at byte offset {exc.start} (R-020)"
            )
            continue
        except json.JSONDecodeError as exc:
            ingress_defects[rel] = f"malformed JSON: {exc} (R-020)"
            continue
        if not isinstance(obj, dict):
            type_name = _json_type_name(obj)
            ingress_defects[rel] = (
                f"top-level JSON {type_name} is ingress-defective for this"
                " evidence-tree namespace — only a well-formed JSON object"
                " may be a tolerated historical sidecar (R-064)"
            )
            continue
        family = legacy_mod.classify_legacy_family(obj)
        if family is not None:
            legacy_by_rel[rel] = {
                "legacy_family": family,
                "aggregate_only": True,
                "certifying": False,
                "payload": obj,
            }
            classifications[rel] = (
                f"legacy_{family}_aggregate (historical, aggregate-only)"
            )
        else:
            classifications[rel] = "unknown_historical_sidecar (tolerated)"
    return legacy_by_rel, classifications, ingress_defects


# ---------------------------------------------------------------------------
# Canonical selection (R-039/R-069)
# ---------------------------------------------------------------------------


def resolve_canonical_package(
    runs_root: Path, ledger: dict[str, Any]
) -> Path:
    """Canonical run selection strictly via the ledger pointer (R-039/R-069).

    Rejects: a missing pointer, path traversal / absolute pointers, symlink
    run directories (even in-root targets), out-of-root resolution, empty
    crash relics, and dangling pointers. NEVER selects newest-wins; NEVER
    falls back after an invalid pointer.
    """
    run_id = ledger.get("canonical_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ColmAimsError(
            "ledger declares no canonical_run_id pointer — canonical"
            " selection happens only via the ledger pointer, never"
            " newest-wins (R-069)"
        )
    if not schema.is_path_component(run_id):
        raise ColmAimsError(
            f"canonical run pointer {run_id!r} must be a single path"
            " component inside the runs root — path traversal and absolute"
            " pointers are refused (R-069)"
        )
    path = Path(runs_root) / run_id
    if path.is_symlink():
        raise ColmAimsError(
            f"canonical run pointer {run_id!r} names a symlink — the pointer"
            " must name a real run directory, never an alias (R-069)"
        )
    if not path.is_dir():
        raise ColmAimsError(
            f"canonical run pointer {run_id!r} does not resolve to a"
            " published run directory under the runs root — dangling"
            " pointers never fall back to any other package (R-069)"
        )
    if not schema.resolves_inside(path, runs_root):
        raise ColmAimsError(
            f"canonical run pointer {run_id!r} resolves outside the runs"
            " root — out-of-root evidence is refused (R-069)"
        )
    if not any(path.iterdir()):
        raise ColmAimsError(
            f"canonical run pointer {run_id!r} resolves to an EMPTY run"
            " directory (a crashed-publish relic) — not a published evidence"
            " package (R-069/R-016)"
        )
    return path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_expectations(path: Path) -> tuple[dict[str, Any], bytes]:
    """Typed, fail-closed load of the anchored expectations file.

    Order: bounded read → hooked parse → container shape → schema_version
    via the ONE shared checker (R-059) → closed key sets (R-063).
    """
    name = Path(path).name
    data = schema.read_regular_file_bytes(path)
    try:
        obj = schema.parse_json_bytes_strict(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise schema.TypedIngressError(
            f"{name}: malformed JSON: {exc} (R-020)"
        ) from exc
    if not isinstance(obj, dict):
        raise schema.TypedIngressError(
            f"{name}: expectations must be an object (R-020)"
        )
    schema.check_schema_version(obj, name)
    unknown = sorted(set(obj) - _EXPECTATIONS_KEYS)
    if unknown:
        raise schema.ConfigSurfaceError(
            f"{name}: unknown expectations key(s) {unknown} — the config"
            " surface fails closed; no key disables a release gate"
            " (R-022/R-063)"
        )
    anchor = obj.get("anchor")
    if not isinstance(anchor, dict):
        raise schema.ConfigSurfaceError(
            f"{name}: expectations anchor block missing or malformed — the"
            " anchor is a closed trusted block (R-013/R-063)"
        )
    unknown_anchor = sorted(set(anchor) - _ANCHOR_KEYS)
    if unknown_anchor:
        raise schema.ConfigSurfaceError(
            f"{name}: unknown anchor key(s) {unknown_anchor} — the anchor"
            " block is CLOSED; a typo'd key is a typed error, never a"
            " silent default (R-063)"
        )
    missing_anchor = sorted(_ANCHOR_KEYS - set(anchor))
    if missing_anchor:
        raise schema.ConfigSurfaceError(
            f"{name}: anchor block missing required key(s) {missing_anchor}"
            " (R-013/R-063)"
        )
    rights_decl = obj.get("rights_inventory")
    if not isinstance(rights_decl, dict):
        raise schema.ConfigSurfaceError(
            f"{name}: expectations rights_inventory block missing or"
            " malformed (R-026/R-063)"
        )
    unknown_rights = sorted(set(rights_decl) - _RIGHTS_DECL_KEYS)
    if unknown_rights:
        raise schema.ConfigSurfaceError(
            f"{name}: unknown rights_inventory key(s) {unknown_rights}"
            " (R-063)"
        )
    missing_rights = sorted(_RIGHTS_DECL_KEYS - set(rights_decl))
    if missing_rights:
        raise schema.ConfigSurfaceError(
            f"{name}: rights_inventory block missing required key(s)"
            f" {missing_rights} (R-063)"
        )
    return obj, data


def _load_json_object_strict(data: bytes, name: str) -> dict[str, Any]:
    """Strict shape+version load for an out-of-tree anchored document."""
    try:
        obj = schema.parse_json_bytes_strict(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise schema.TypedIngressError(
            f"{name}: malformed JSON: {exc} (R-020)"
        ) from exc
    if not isinstance(obj, dict):
        raise schema.TypedIngressError(
            f"{name}: artifact must be a JSON object (R-020)"
        )
    schema.check_schema_version(obj, name)
    return obj


def _load_manifest(data: bytes) -> dict[str, Any]:
    """Strict presentation-manifest load: shape → version (shared checker,
    R-059) → closed keys."""
    name = "presentation_manifest.json"
    obj = _load_json_object_strict(data, name)
    unknown = sorted(set(obj) - _MANIFEST_KEYS)
    if unknown:
        raise schema.TypedIngressError(
            f"{name}: unknown manifest key(s) {unknown} — no silent"
            " key-dropping (R-020)"
        )
    return obj


def _git_object_exists(commit: str) -> bool | None:
    """Anchor object-existence check bound to THIS repository (R-066).

    ``True``/``False`` when git answers; ``None`` when the check cannot run
    (no ``.git``, git missing, or a timeout) — the caller converts ``None``
    into a FAILING release leg: ``PASS_RELEASE`` cannot be obtained by
    making ``git`` disappear.
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
# In-package legs (source-mode minimum positive set + shared release input)
# ---------------------------------------------------------------------------


def _reject_empty_evaluation(cells: Any) -> None:
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


def _complete_key_map(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Complete records keyed by item key (excluded/malformed skipped)."""
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        outcome = pairing.classify_record(record)
        if outcome["status"] == "complete":
            out[record["item_key"]] = record
    return out


def _grid_completeness_leg(
    legs: list[dict[str, Any]],
    grid: dict[str, Any],
    cells: list[dict[str, Any]],
) -> None:
    problems: list[str] = []
    frozen_refs = sorted(schema.REFERENCE_IDS)
    frozen_cals = sorted(schema.CALIBRATION_IDS)
    frozen_cells = sorted(schema.CELL_IDS)
    # MA2-002a: coerce artifact-controlled axis lists to their string
    # entries before sorting — non-string members are a leg FAIL naming the
    # shape defect, never a mixed-type ``sorted`` TypeError crash.
    declared_refs, bad_refs = _str_list(grid.get("reference_ids"))
    declared_cals, bad_cals = _str_list(grid.get("calibration_ids"))
    declared_cells, bad_cells = _str_list(grid.get("cell_ids"))
    for axis, bad in (
        ("reference_ids", bad_refs),
        ("calibration_ids", bad_cals),
        ("cell_ids", bad_cells),
    ):
        if bad:
            problems.append(
                f"grid.{axis} carries non-string entr(y/ies) {bad} — shape"
                " defect fails the leg, never the run"
            )
    if sorted(declared_refs) != frozen_refs:
        problems.append(
            f"grid.reference_ids {declared_refs!r} != the five approved"
            f" constructed references {frozen_refs} (exact set, never subset)"
        )
    if sorted(declared_cals) != frozen_cals:
        problems.append(
            f"grid.calibration_ids {declared_cals!r} != exactly"
            f" {frozen_cals}"
        )
    if sorted(declared_cells) != frozen_cells:
        problems.append(
            f"grid.cell_ids != the ten derived Cartesian cells"
            f" (cardinality {len(frozen_cells)})"
        )
    raw_cell_ids = [cell.get("cell_id") for cell in cells]
    cell_ids, bad_cell_ids = _str_list(raw_cell_ids)
    if bad_cell_ids:
        problems.append(
            f"profile cells carry non-string cell_id entr(y/ies)"
            f" {bad_cell_ids}"
        )
    if sorted(cell_ids) != frozen_cells or len(raw_cell_ids) != len(
        frozen_cells
    ):
        problems.append(
            f"profile cells carry ids {sorted(cell_ids)} != the ten frozen"
            " grid cells"
        )
    for cell in cells:
        rid = cell.get("reference_id")
        cid = cell.get("calibration_id")
        if cell.get("cell_id") != f"{rid}__{cid}":
            problems.append(
                f"cell {cell.get('cell_id')!r} does not compose"
                " reference_id__calibration_id"
            )
    _record_leg(
        legs,
        LEG_GRID_COMPLETENESS,
        not problems,
        expected="exactly the 5x2 Cartesian grid: 5 references x 2"
        " calibrations = 10 cells (R-040)",
        observed="; ".join(problems),
    )


def _record_file_bijection_leg(
    legs: list[dict[str, Any]],
    grid: dict[str, Any],
    cells: list[dict[str, Any]],
    snapshot: dict[str, bytes],
) -> None:
    problems: list[str] = []
    mapping = _as_dict(grid.get("record_files"))
    declared_cells, _bad_cells = _str_list(grid.get("cell_ids"))
    if sorted(mapping) != sorted(declared_cells):
        problems.append(
            "grid.record_files keys do not equal the declared cell ids"
            " (orphaned or missing mapping entries)"
        )
    seen_targets: set[str] = set()
    mapped_targets: set[str] = set()
    for cell_id, rel in sorted(mapping.items()):
        # MA2-002a: a non-string mapping value is a shape defect failing
        # THIS leg — never an unhashable-membership TypeError crash.
        if not isinstance(rel, str):
            problems.append(
                f"record file mapping value for cell {cell_id!r} is not a"
                f" string (shape defect: {type(rel).__name__})"
            )
            continue
        expected_rel = f"records/{cell_id}.jsonl"
        if rel != expected_rel:
            problems.append(
                f"cell {cell_id!r} maps to {rel!r}, not its own"
                f" {expected_rel!r} (exactly one records/<cell_id>.jsonl per"
                " cell)"
            )
        if rel in seen_targets:
            problems.append(f"record file {rel!r} mapped by multiple cells")
        seen_targets.add(rel)
        mapped_targets.add(rel)
        if rel not in snapshot:
            problems.append(f"declared-but-absent record file {rel!r}")
    present = {
        rel
        for rel in snapshot
        if rel.startswith("records/") and rel.endswith(".jsonl")
    }
    for rel in sorted(present - mapped_targets):
        problems.append(f"present-but-undeclared record file {rel!r}")
    for cell in cells:
        cell_id = cell.get("cell_id")
        if cell_id in mapping and cell.get("records_file") != mapping[cell_id]:
            problems.append(
                f"cell {cell_id!r} records_file disagrees with the grid"
                " mapping"
            )
    _record_leg(
        legs,
        LEG_RECORD_FILE_BIJECTION,
        not problems,
        expected="cell<->record-file bijection: exactly one"
        " records/<cell_id>.jsonl per declared cell, no missing, duplicate,"
        " undeclared, or orphaned record file (R-041/R-035)",
        observed="; ".join(problems),
    )


def _item_key_set_leg(
    legs: list[dict[str, Any]],
    grid: dict[str, Any],
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[str] | None, str | None]:
    """Returns ``(shared_sorted_keys, shared_keyset_digest)`` when the
    byte-exact cross-cell equality holds."""
    problems: list[str] = []
    reference_set: set[str] | None = None
    reference_cell: str | None = None
    for cell_id in sorted(complete_by_cell):
        keys = set(complete_by_cell[cell_id])
        if len(keys) != schema.EXPECTED_COMPLETE_PAIRS:
            problems.append(
                f"cell {cell_id!r} carries {len(keys)} complete paired item"
                f" keys, not exactly {schema.EXPECTED_COMPLETE_PAIRS}"
                " (no partial-population escape hatch)"
            )
        if reference_set is None:
            reference_set = keys
            reference_cell = cell_id
        elif keys != reference_set:
            drift = len(keys.symmetric_difference(reference_set))
            problems.append(
                f"cell {cell_id!r} item-key SET differs from cell"
                f" {reference_cell!r} ({drift} differing key(s)) — the set"
                " must be byte-exact identical across all ten cells"
            )
    shared_keys: list[str] | None = None
    shared_digest: str | None = None
    if reference_set is not None and not problems:
        shared_keys = pairing.canonical_item_order(sorted(reference_set))
        shared_digest = pairing.keyset_sha256(shared_keys)
        declared = grid.get("item_keys_sha256")
        if declared != shared_digest:
            problems.append(
                f"grid.item_keys_sha256 {declared!r} != the digest of the"
                f" shared complete-key set {shared_digest!r}"
            )
            shared_keys, shared_digest = None, None
    if not complete_by_cell:
        problems.append("no per-cell record files available to compare")
    _record_leg(
        legs,
        LEG_ITEM_KEY_SET,
        not problems,
        expected=f"exactly {schema.EXPECTED_COMPLETE_PAIRS} complete paired"
        " item keys per cell, byte-exact identical across all ten cells"
        " (R-042/R-008)",
        observed="; ".join(problems),
    )
    return shared_keys, shared_digest


def _held_fixed_leg(
    legs: list[dict[str, Any]],
    grid: dict[str, Any],
    cells: list[dict[str, Any]],
) -> None:
    problems: list[str] = []
    held = _as_dict(grid.get("held_fixed"))
    expected_mc = held.get("mc_trajectory_identity")
    expected_hz = held.get("horizon_identity")
    if not _is_resolved_identity(expected_mc) or not _is_resolved_identity(
        expected_hz
    ):
        problems.append("grid.held_fixed identities missing or unresolved")
    for cell in cells:
        cell_id = cell.get("cell_id", "unnamed")
        er = _as_dict(_as_dict(cell.get("estimand")).get("event_representation"))
        if er.get("mc_trajectory_identity") != expected_mc:
            problems.append(
                f"cell {cell_id!r} mc_trajectory_identity"
                f" {er.get('mc_trajectory_identity')!r} != held-fixed"
                f" {expected_mc!r}"
            )
        if er.get("horizon_identity") != expected_hz:
            problems.append(
                f"cell {cell_id!r} horizon_identity"
                f" {er.get('horizon_identity')!r} != held-fixed"
                f" {expected_hz!r}"
            )
    _record_leg(
        legs,
        LEG_HELD_FIXED,
        not problems,
        expected="the same raw MC trajectory identity and horizon identity"
        " wherever the contract declares them held fixed (R-043)",
        observed="; ".join(problems),
    )


def _mc_stop_within_calibration_leg(
    legs: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """R-043: ``mc_stop_step`` equality across the five references WITHIN
    each calibration ID only. There is NO cross-calibration requirement —
    those conditions use separately fitted MC calibrators, and a verifier
    that enforces cross-calibration equality is itself defective."""
    problems: list[str] = []
    by_calibration: dict[str, list[str]] = {}
    for cell in cells:
        cell_id = cell.get("cell_id")
        cal_id = cell.get("calibration_id")
        if isinstance(cell_id, str) and isinstance(cal_id, str):
            by_calibration.setdefault(cal_id, []).append(cell_id)
    for cal_id in sorted(by_calibration):
        group = sorted(by_calibration[cal_id])
        base_cell: str | None = None
        base_map: dict[str, tuple[Any, Any]] | None = None
        for cell_id in group:
            records = complete_by_cell.get(cell_id)
            if records is None:
                continue
            mc_map = {
                key: (rec.get("mc_event_status"), rec.get("mc_stop_step"))
                for key, rec in records.items()
            }
            if base_map is None:
                base_cell, base_map = cell_id, mc_map
            elif mc_map != base_map:
                differing = sum(
                    1
                    for key in base_map
                    if mc_map.get(key) != base_map[key]
                )
                problems.append(
                    f"calibration {cal_id!r}: cell {cell_id!r} MC stops"
                    f" differ from cell {base_cell!r} on {differing} item(s)"
                    " — the five references within a calibration share the"
                    " same raw MC trajectory stops"
                )
    _record_leg(
        legs,
        LEG_MC_STOP_WITHIN_CAL,
        not problems,
        expected="mc_stop_step equality across the five references WITHIN"
        " each calibration ID only (no cross-calibration requirement)"
        " (R-043)",
        observed="; ".join(problems),
    )


def _event_representation_leg(
    legs: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> None:
    problems: list[str] = []
    for cell in cells:
        cell_id = cell.get("cell_id", "unnamed")
        est = _as_dict(cell.get("estimand"))
        er = _as_dict(est.get("event_representation"))
        for key in schema.EVENT_REPRESENTATION_KEYS:
            if key not in er:
                problems.append(
                    f"cell {cell_id!r} event_representation missing bound"
                    f" identity {key!r}"
                )
        if er.get("index_base") != 0 or not schema.is_real_int(
            er.get("index_base")
        ):
            problems.append(
                f"cell {cell_id!r} index_base must be exactly 0 (R-061)"
            )
        if (
            er.get("historical_sentinel_convention")
            != schema.SENTINEL_CONVENTION
        ):
            problems.append(
                f"cell {cell_id!r} historical_sentinel_convention"
                f" {er.get('historical_sentinel_convention')!r} is not the"
                f" preserved producer convention"
                f" {schema.SENTINEL_CONVENTION!r} (R-046)"
            )
        if er.get("terminal_imputation_policy") != schema.IMPUTATION_FINAL_PREFIX:
            problems.append(
                f"cell {cell_id!r} terminal_imputation_policy"
                f" {er.get('terminal_imputation_policy')!r} is not"
                f" {schema.IMPUTATION_FINAL_PREFIX!r} (R-045)"
            )
        if not _is_resolved_identity(er.get("producer_profile_identity")):
            problems.append(
                f"cell {cell_id!r} producer_profile_identity missing (R-045)"
            )
        pairing_definition = est.get("pairing_definition")
        if pairing_definition not in schema.PAIRING_DEFINITIONS:
            problems.append(
                f"cell {cell_id!r} pairing_definition"
                f" {pairing_definition!r} outside the closed vocabulary"
                " (R-011)"
            )
        timeout = _as_dict(est.get("timeout_parameters"))
        rule = timeout.get("rule")
        if rule not in schema.TIMEOUT_RULES:
            problems.append(
                f"cell {cell_id!r} timeout rule {rule!r} outside the closed"
                " vocabulary (R-011)"
            )
        elif pairing_definition in schema.PAIRING_RULE_RECONCILIATION and (
            rule
            not in schema.PAIRING_RULE_RECONCILIATION[pairing_definition]
        ):
            problems.append(
                f"cell {cell_id!r} timeout rule {rule!r} does not reconcile"
                f" with pairing definition {pairing_definition!r} (R-011)"
            )
        if est.get("population") != schema.POPULATION_ALL:
            problems.append(
                f"cell {cell_id!r} headline estimand population"
                f" {est.get('population')!r} is not"
                f" {schema.POPULATION_ALL!r} (R-054)"
            )
        if est.get("denominator_policy") != "n_complete":
            problems.append(
                f"cell {cell_id!r} denominator_policy"
                f" {est.get('denominator_policy')!r} != 'n_complete' (R-048)"
            )
        if est.get("reference_id") != cell.get("reference_id") or est.get(
            "calibration_id"
        ) != cell.get("calibration_id"):
            problems.append(
                f"cell {cell_id!r} estimand axes disagree with the cell's"
                " declared axes"
            )
        records = complete_by_cell.get(cell_id)
        if records:
            horizons = {
                rec.get("trajectory_horizon") for rec in records.values()
            }
            declared_horizon = timeout.get("trajectory_horizon")
            if horizons != {declared_horizon}:
                problems.append(
                    f"cell {cell_id!r} estimand timeout horizon"
                    f" {declared_horizon!r} != the horizon applied to"
                    f" records {sorted(map(str, horizons))}"
                )
    _record_leg(
        legs,
        LEG_EVENT_REPRESENTATION,
        not problems,
        expected="canonical event-representation bindings: index_base 0,"
        " pinned sentinel convention, FINAL_PREFIX_IF_NEVER imputation"
        " policy, reconciled closed pairing/timeout vocabularies"
        " (R-045/R-046/R-011)",
        observed="; ".join(problems),
    )


def _counts_and_rates_legs(
    legs: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    records_by_cell: dict[str, list[dict[str, Any]]],
) -> None:
    count_problems: list[str] = []
    rate_problems: list[str] = []
    for cell in cells:
        cell_id = cell.get("cell_id", "unnamed")
        records = records_by_cell.get(cell_id)
        if records is None:
            count_problems.append(
                f"cell {cell_id!r}: record file unavailable for recompute"
            )
            continue
        try:
            pairing.check_count_identities(
                _as_dict(cell.get("counts")), records
            )
        except _LEG_CATCH as exc:
            count_problems.append(f"cell {cell_id!r}: {exc}")
        try:
            _check_recorded_rates(cell)
        except _LEG_CATCH as exc:
            rate_problems.append(f"cell {cell_id!r}: {exc}")
    _record_leg(
        legs,
        LEG_COUNTS,
        not count_problems,
        expected="per-cell counts recompute exactly from"
        " records/<cell_id>.jsonl (R-005)",
        observed="; ".join(count_problems),
    )
    _record_leg(
        legs,
        LEG_RATES,
        not rate_problems,
        expected="joint-class rates over n_complete, null at zero, summing"
        " to 1 within the declared tolerance (R-006)",
        observed="; ".join(rate_problems),
    )


def _declared_tolerance(cell: dict[str, Any]) -> float:
    estimand = _as_dict(cell.get("estimand"))
    tolerance = estimand.get("numerical_tolerance")
    if not schema.is_admissible_tolerance(tolerance):
        raise schema.SchemaValidationError(
            "cell declares no admissible numerical_tolerance (R-032)"
        )
    return float(tolerance)


def _check_recorded_rates(cell: dict[str, Any]) -> None:
    counts = cell["counts"]
    recorded = cell["rates"]
    tolerance = _declared_tolerance(cell)
    expected = pairing.compute_rates(counts)
    missing = sorted(set(pairing.RATE_KEYS) - set(recorded))
    if missing:
        raise pairing.RateError(f"rates block missing {missing} (R-006)")
    if counts["n_complete"] == 0:
        bad = [key for key in pairing.RATE_KEYS if recorded[key] is not None]
        if bad:
            raise pairing.RateError(
                f"rates must be null when n_complete is zero; found {bad}"
                " (R-006)"
            )
        return
    for key in pairing.RATE_KEYS:
        value = recorded[key]
        if not schema.is_native_finite_number(value):
            raise pairing.RateError(f"rate {key!r} must be numeric (R-006)")
        if abs(float(value) - expected[key]) > tolerance:
            raise pairing.RateError(
                f"rate {key!r} recorded {value!r} != recomputed"
                f" {expected[key]!r} within declared tolerance {tolerance!r}"
                " (R-006)"
            )
    total = sum(float(recorded[key]) for key in pairing.RATE_KEYS)
    if abs(total - 1.0) > tolerance:
        raise pairing.RateError(
            f"joint-class rates sum to {total!r}, not 1 within declared"
            f" tolerance {tolerance!r} (R-006)"
        )


def _estimand_label_leg(
    legs: list[dict[str, Any]], cells: list[dict[str, Any]]
) -> None:
    """R-048/R-049/R-068: recorded labels/populations validate against the
    CANONICAL recompute identities — never against the recorded strings
    themselves (the F7A trusted-label regression)."""
    problems: list[str] = []
    for cell in cells:
        cell_id = cell.get("cell_id", "unnamed")
        headline = _as_dict(cell.get("headline_summary"))
        finite = _as_dict(cell.get("finite_only_summary"))
        if headline.get("estimand_label") != schema.HEADLINE_ESTIMAND_LABEL:
            problems.append(
                f"cell {cell_id!r} headline estimand_label"
                f" {headline.get('estimand_label')!r} != the canonical"
                " sentinel-coded all-pair recompute identity (R-048/R-068)"
            )
        if headline.get("population") != schema.POPULATION_ALL:
            problems.append(
                f"cell {cell_id!r} headline population"
                f" {headline.get('population')!r} !="
                f" {schema.POPULATION_ALL!r} — a both-finite population"
                " under the headline label FAILS (R-048/R-049)"
            )
        if finite.get("estimand_label") != schema.FINITE_ONLY_ESTIMAND_LABEL:
            problems.append(
                f"cell {cell_id!r} finite-only estimand_label"
                f" {finite.get('estimand_label')!r} != the canonical"
                " finite-only recompute identity (R-049/R-068)"
            )
        if finite.get("population") != schema.POPULATION_FINITE:
            problems.append(
                f"cell {cell_id!r} finite-only population"
                f" {finite.get('population')!r} must declare its conditional"
                f" population {schema.POPULATION_FINITE!r} (R-049)"
            )
    _record_leg(
        legs,
        LEG_ESTIMAND_LABELS,
        not problems,
        expected="estimand labels and populations bound to the canonical"
        " recompute identities (R-048/R-049/R-068)",
        observed="; ".join(problems),
    )


def _cell_comparability_leg(
    legs: list[dict[str, Any]], cells: list[dict[str, Any]]
) -> None:
    """R-011: pooling the ten cells into one Holm family requires their
    axis-stripped residual estimands to be comparable — every production
    pooling site routes through ``pairing.check_comparable``."""
    problems: list[str] = []

    def residual(cell: dict[str, Any]) -> dict[str, Any]:
        est = {
            key: value
            for key, value in _as_dict(cell.get("estimand")).items()
            if key not in _ESTIMAND_AXIS_FIELDS
        }
        return {
            "cell_id": cell.get("cell_id"),
            "estimand": est,
            "estimand_digest": pairing.estimand_digest(est),
        }

    if len(cells) >= 2:
        try:
            first = residual(cells[0])
            for other in cells[1:]:
                pairing.check_comparable(first, residual(other))
        except _LEG_CATCH as exc:
            problems.append(str(exc))
    _record_leg(
        legs,
        LEG_CELL_COMPARABILITY,
        not problems,
        expected="the ten cells' axis-stripped estimand identities are"
        " comparable — pooling across differing estimand digests is refused"
        " (R-011)",
        observed="; ".join(problems),
    )


# ---------------------------------------------------------------------------
# In-package D7(b) inference legs (R-050..R-056)
# ---------------------------------------------------------------------------


def _inference_legs(
    legs: list[dict[str, Any]],
    profile: dict[str, Any],
    cells: list[dict[str, Any]],
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
    shared_keys: list[str] | None,
    shared_digest: str | None,
) -> None:
    inference = _as_dict(profile.get("inference"))

    # ---- inference_seed_derivation (R-052 triple bind) -----------------
    seed_problems: list[str] = []
    recorded_seed = inference.get("seed")
    if not schema.is_uint64(recorded_seed):
        seed_problems.append(
            f"inference.seed {recorded_seed!r} is not exactly one real"
            " integer in the unsigned 64-bit (uint64) domain (R-052/R-061)"
        )
    if inference.get("seed_derivation") != schema.SEED_DERIVATION_STRING:
        seed_problems.append(
            "inference.seed_derivation does not record the pinned"
            " derivation string (R-052)"
        )
    if shared_digest is None:
        seed_problems.append(
            "the shared complete-key set is unavailable, so the seed's"
            " key-set binding cannot be verified (R-052)"
        )
    else:
        if inference.get("pairing_population_keyset_sha256") != shared_digest:
            seed_problems.append(
                "inference.pairing_population_keyset_sha256"
                f" {inference.get('pairing_population_keyset_sha256')!r} !="
                f" the digest of the shared complete-key set {shared_digest!r}"
                " (R-052 triple bind)"
            )
        derived_seed = pairing.d7b_seed(shared_digest)
        if schema.is_uint64(recorded_seed) and recorded_seed != derived_seed:
            seed_problems.append(
                f"recorded seed {recorded_seed!r} != the seed"
                f" {derived_seed!r} derived from the pairing-population"
                " key-set digest (R-052)"
            )
        for cell in cells:
            cell_id = cell.get("cell_id", "unnamed")
            if cell.get("pairing_population_keyset_sha256") != shared_digest:
                seed_problems.append(
                    f"cell {cell_id!r} pairing_population_keyset_sha256"
                    " disagrees with the shared complete-key set digest"
                )
            interval = _as_dict(cell.get("interval"))
            if interval.get("seed") != recorded_seed:
                seed_problems.append(
                    f"cell {cell_id!r} interval seed disagrees with the ONE"
                    " recorded collection-level seed"
                )
            if interval.get("seed_derivation") != schema.SEED_DERIVATION_STRING:
                seed_problems.append(
                    f"cell {cell_id!r} interval seed_derivation is not the"
                    " pinned derivation string"
                )
    _record_leg(
        legs,
        LEG_INFERENCE_SEED,
        not seed_problems,
        expected="exactly ONE uint64 seed, derived from the"
        " pairing-population key-set digest, triple-bound across inference"
        " block, cells, and records (R-052)",
        observed="; ".join(seed_problems),
    )

    # ---- inference_resample_matrix_digest (R-051/R-053) ----------------
    matrix_problems: list[str] = []
    plan_expected = {
        "numpy_version": "2.4.6",
        "bit_generator": "PCG64",
        "generator_construction": schema.GENERATOR_CONSTRUCTION,
        "draw_count": schema.BOOTSTRAP_DRAW_COUNT,
        "sample_size": schema.EXPECTED_COMPLETE_PAIRS,
        "resampling_unit": schema.RESAMPLING_UNIT,
        "dtype": "int64",
    }
    for name, expected_value in plan_expected.items():
        if inference.get(name) != expected_value:
            matrix_problems.append(
                f"inference.{name} {inference.get(name)!r} !="
                f" {expected_value!r} (R-051 exact-match plan)"
            )
    if inference.get("with_replacement") is not True:
        matrix_problems.append(
            "inference.with_replacement must be exactly true (R-051)"
        )
    if inference.get("endpoint") is not False:
        matrix_problems.append(
            "inference.endpoint must be exactly false (endpoint=True is a"
            " DIFFERENT plan) (R-051)"
        )
    if np.__version__ != "2.4.6":
        matrix_problems.append(
            f"runtime NumPy {np.__version__} != the pinned 2.4.6 (D5/R-051)"
        )
    matrix: np.ndarray | None = None
    order_digest: str | None = None
    if shared_keys is not None:
        order_digest = pairing.item_order_sha256(shared_keys)
        if inference.get("canonical_item_order_digest") != order_digest:
            matrix_problems.append(
                "inference.canonical_item_order_digest"
                f" {inference.get('canonical_item_order_digest')!r} != the"
                " ascending-UTF-8 canonical item order digest"
                f" {order_digest!r} (R-050)"
            )
    if schema.is_uint64(recorded_seed) and not matrix_problems:
        try:
            matrix = pairing.d7b_resample_matrix(recorded_seed)
        except _LEG_CATCH as exc:
            matrix_problems.append(f"matrix regeneration failed: {exc}")
    if matrix is not None and order_digest is not None:
        computed = pairing.d7b_matrix_digest_record(matrix, order_digest)
        recorded_digest = _as_dict(inference.get("resample_matrix_digest"))
        for field_name in sorted(schema.MATRIX_DIGEST_KEYS):
            if field_name not in recorded_digest:
                matrix_problems.append(
                    f"resample_matrix_digest missing covering field"
                    f" {field_name!r} (R-053)"
                )
            elif recorded_digest[field_name] != computed[field_name]:
                matrix_problems.append(
                    f"resample_matrix_digest.{field_name}"
                    f" {recorded_digest[field_name]!r} != regenerated"
                    f" {computed[field_name]!r} (R-053)"
                )
    elif matrix is None and not matrix_problems:
        matrix_problems.append(
            "resample matrix could not be regenerated from the recorded"
            " seed/plan (R-053)"
        )
    _record_leg(
        legs,
        LEG_INFERENCE_MATRIX,
        not matrix_problems,
        expected="ONE shared collection-level resample matrix regenerated"
        " from the recorded seed; digest + dtype/shape/byte-order/item-order"
        " covering fields match (R-051/R-053)",
        observed="; ".join(matrix_problems),
    )

    # ---- inference_recompute (R-050/R-054/R-055/R-015) -----------------
    recompute_problems: list[str] = []
    raw_p_by_cell: dict[str, float] = {}
    can_recompute = (
        matrix is not None and shared_keys is not None and not seed_problems
    )
    for cell in cells:
        cell_id = cell.get("cell_id", "unnamed")
        try:
            tolerance = _declared_tolerance(cell)
        except _LEG_CATCH as exc:
            recompute_problems.append(f"cell {cell_id!r}: {exc}")
            continue
        interval = _as_dict(cell.get("interval"))
        identity_expected = {
            "procedure": schema.INTERVAL_PROCEDURE,
            "draw_count": schema.BOOTSTRAP_DRAW_COUNT,
            "statistic": schema.INTERVAL_STATISTIC,
            "quantile_method": schema.QUANTILE_METHOD,
        }
        for name, expected_value in identity_expected.items():
            if interval.get(name) != expected_value:
                recompute_problems.append(
                    f"cell {cell_id!r} interval {name}"
                    f" {interval.get(name)!r} != {expected_value!r}"
                    " (R-054/R-015)"
                )
        population = interval.get("population")
        if population not in schema.POPULATIONS:
            recompute_problems.append(
                f"cell {cell_id!r} interval population {population!r}"
                " outside the closed population enum (R-054)"
            )
        elif population != schema.POPULATION_ALL:
            recompute_problems.append(
                f"cell {cell_id!r} headline interval declares"
                f" {population!r} — the both-finite conditioning is"
                " RETRACTED for headline cells; they bind the sentinel-coded"
                " all-pair population (R-054/R-015)"
            )
        headline = _as_dict(cell.get("headline_summary"))
        finite = _as_dict(cell.get("finite_only_summary"))
        records_map = complete_by_cell.get(cell_id)
        if records_map is None or not can_recompute:
            recompute_problems.append(
                f"cell {cell_id!r}: recompute inputs unavailable"
            )
            continue
        try:
            ordered = pairing.canonical_item_order(sorted(records_map))
            d = pairing.sentinel_coded_shift_vector(records_map, ordered)
            mu_hat = float(np.mean(d))
            if headline.get("n") != len(ordered) or headline.get("n") != (
                schema.EXPECTED_COMPLETE_PAIRS
            ):
                recompute_problems.append(
                    f"cell {cell_id!r} headline n {headline.get('n')!r} !="
                    f" the {schema.EXPECTED_COMPLETE_PAIRS} complete pairs"
                    " (R-048)"
                )
            recorded_mean = headline.get("mean_signed_shift")
            if not schema.is_native_finite_number(recorded_mean) or abs(
                float(recorded_mean) - mu_hat
            ) > tolerance:
                recompute_problems.append(
                    f"cell {cell_id!r} headline mean_signed_shift"
                    f" {recorded_mean!r} does not recompute from records"
                    f" (expected {mu_hat!r}) (R-048/R-015)"
                )
            if headline.get("convention") != schema.SENTINEL_CONVENTION:
                recompute_problems.append(
                    f"cell {cell_id!r} headline convention is not the"
                    " preserved producer convention (R-046)"
                )
            expected_finite = pairing.finite_only_timing_summary(
                list(records_map.values())
            )
            if finite.get("n") != expected_finite["n"]:
                recompute_problems.append(
                    f"cell {cell_id!r} finite-only n {finite.get('n')!r} !="
                    f" recomputed n_both_finite {expected_finite['n']!r}"
                    " (R-049)"
                )
            for stat in pairing.FINITE_ONLY_STATISTICS:
                expected_value = expected_finite[stat]
                recorded_value = finite.get(stat)
                if expected_value is None:
                    if recorded_value is not None:
                        recompute_problems.append(
                            f"cell {cell_id!r} finite-only {stat} must be"
                            " null over zero both-finite pairs (R-006)"
                        )
                    continue
                if not schema.is_native_finite_number(recorded_value) or abs(
                    float(recorded_value) - expected_value
                ) > tolerance:
                    recompute_problems.append(
                        f"cell {cell_id!r} finite-only {stat}"
                        f" {recorded_value!r} does not recompute (expected"
                        f" {expected_value!r}) (R-015)"
                    )
            lo, hi = pairing.d7b_interval(d, matrix)
            recorded_ci = interval.get("ci")
            if (
                not isinstance(recorded_ci, list)
                or len(recorded_ci) != 2
                or not all(
                    schema.is_native_finite_number(v) for v in recorded_ci
                )
            ):
                recompute_problems.append(
                    f"cell {cell_id!r} interval ci must be two native finite"
                    " numbers (R-067)"
                )
            elif (
                abs(float(recorded_ci[0]) - lo) > tolerance
                or abs(float(recorded_ci[1]) - hi) > tolerance
            ):
                recompute_problems.append(
                    f"cell {cell_id!r} interval ci {recorded_ci!r} does not"
                    f" reproduce the D7(b) recompute [{lo!r}, {hi!r}] —"
                    " UNROUNDED endpoints are stored in the package; display"
                    " rounding happens only in the renderer (R-054)"
                )
            p = pairing.d7b_p_value(d, matrix)
            raw_p_by_cell[str(cell_id)] = p
            recorded_p = cell.get("raw_p_value")
            if not schema.is_native_finite_number(recorded_p) or abs(
                float(recorded_p) - p
            ) > tolerance:
                recompute_problems.append(
                    f"cell {cell_id!r} raw_p_value {recorded_p!r} does not"
                    f" recompute (expected {p!r}; the +1/1001"
                    " finite-resample correction is mandatory) (R-055)"
                )
        except _LEG_CATCH as exc:
            recompute_problems.append(f"cell {cell_id!r}: {exc}")
    _record_leg(
        legs,
        LEG_INFERENCE_RECOMPUTE,
        not recompute_problems,
        expected="per-cell means, finite-only summaries, unrounded"
        " percentile intervals, and +1/1001-corrected p-values recompute"
        " deterministically from the recorded D7(b) procedure"
        " (R-050/R-054/R-055/R-015)",
        observed="; ".join(recompute_problems),
    )

    # ---- inference_holm_family (R-056) ----------------------------------
    holm_problems: list[str] = []
    if len(raw_p_by_cell) != 10:
        holm_problems.append(
            f"recomputed raw p-values cover {len(raw_p_by_cell)} cells, not"
            " the exact ten-cell 5x2 family (m=10)"
        )
    else:
        try:
            family = pairing.d7b_holm(raw_p_by_cell)
        except _LEG_CATCH as exc:
            family = None
            holm_problems.append(str(exc))
        if family is not None:
            if inference.get("ordered_family") != family["ordered_family"]:
                holm_problems.append(
                    "inference.ordered_family disagrees with the recomputed"
                    " ascending-raw-p family (ties by ascending UTF-8 byte"
                    " order of cell_id)"
                )
            if inference.get("rejected_cell_ids") != family[
                "rejected_cell_ids"
            ]:
                holm_problems.append(
                    "inference.rejected_cell_ids"
                    f" {inference.get('rejected_cell_ids')!r} != recomputed"
                    f" {family['rejected_cell_ids']!r}"
                )
            if inference.get("familywise_alpha") != 0.05:
                holm_problems.append(
                    "inference.familywise_alpha must be exactly 0.05"
                )
            if inference.get("family_size") != 10:
                holm_problems.append("inference.family_size must be exactly 10")
            for cell in cells:
                cell_id = str(cell.get("cell_id"))
                per = family["per_cell"].get(cell_id)
                if per is None:
                    holm_problems.append(
                        f"cell {cell_id!r} absent from the recomputed family"
                    )
                    continue
                if cell.get("holm_rank") != per["holm_rank"]:
                    holm_problems.append(
                        f"cell {cell_id!r} holm_rank"
                        f" {cell.get('holm_rank')!r} != recomputed"
                        f" {per['holm_rank']!r}"
                    )
                recorded_adj = cell.get("holm_adjusted_p_value")
                if not schema.is_native_finite_number(recorded_adj) or abs(
                    float(recorded_adj) - per["holm_adjusted_p_value"]
                ) > 1e-12:
                    holm_problems.append(
                        f"cell {cell_id!r} holm_adjusted_p_value"
                        f" {recorded_adj!r} != recomputed"
                        f" {per['holm_adjusted_p_value']!r}"
                    )
                if cell.get("holm_rejected") is not per["holm_rejected"]:
                    holm_problems.append(
                        f"cell {cell_id!r} holm_rejected"
                        f" {cell.get('holm_rejected')!r} != recomputed"
                        f" {per['holm_rejected']!r}"
                    )
    _record_leg(
        legs,
        LEG_INFERENCE_HOLM,
        not holm_problems,
        expected="Holm step-down over exactly the ten-cell family (m=10,"
        " alpha 0.05, UTF-8 tie order), all stored fields recompute; no"
        " selective omission of non-headline cells (R-056)",
        observed="; ".join(holm_problems),
    )


# ---------------------------------------------------------------------------
# run_verifier
# ---------------------------------------------------------------------------


def run_verifier(
    tree: Path,
    *,
    mode: str,
    receipts_dir: Path,
    expectations: Path | None = None,
    pre_legs: list[dict[str, Any]] | None = None,
) -> VerificationReport:
    """Run one verifier pass over an artifact tree; never mutate inputs."""
    if mode not in ("source", "release"):
        raise ColmAimsError(
            f"unknown verifier mode {mode!r}; expected 'source' or 'release'"
        )
    tree = Path(tree)
    receipts_dir = Path(receipts_dir)

    snapshot = _read_tree_snapshot(tree) if tree.is_dir() else {}
    if not snapshot or "profile.json" not in snapshot:
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
                "release mode requires an independently anchored"
                " expectations file located outside the verified artifact"
                " tree (R-013)"
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

    # Typed ingress (R-020) — from the snapshot bytes, version-first on
    # every versioned surface (R-059).
    profile = schema.load_artifact_bytes(
        snapshot["profile.json"], "profile.json"
    )
    manifest_obj: dict[str, Any] | None = None
    if "presentation_manifest.json" in snapshot:
        manifest_obj = _load_manifest(snapshot["presentation_manifest.json"])

    records_by_rel: dict[str, list[dict[str, Any]]] = {}
    record_lines_by_rel: dict[str, list[int]] = {}
    for rel in sorted(snapshot):
        if rel.startswith("records/") and rel.endswith(".jsonl"):
            loaded = schema.load_records_bytes(snapshot[rel], rel)
            records_by_rel[rel] = loaded["records"]
            record_lines_by_rel[rel] = loaded["line_numbers"]

    legacy_by_rel, classifications, sidecar_defects = _classify_tree_sidecars(
        snapshot
    )

    cells_raw = profile.get("cells")
    _reject_empty_evaluation(cells_raw)

    if sidecar_defects and mode == "source":
        details = "; ".join(
            f"{rel}: {err}" for rel, err in sorted(sidecar_defects.items())
        )
        raise schema.TypedIngressError(
            f"tree artifact(s) failed typed ingress: {details} (R-064)"
        )

    legs: list[dict[str, Any]] = list(pre_legs or [])
    validated: list[str] = []

    legs.append(_pass(LEG_TYPED_INGRESS))
    for rel in sorted(sidecar_defects):
        legs.append(
            _fail(
                f"{SIDECAR_LEG_PREFIX}{rel}",
                expected="tree .json sidecar bytes pass typed ingress; only"
                " well-formed JSON objects are tolerated historical"
                " sidecars (R-064)",
                observed=f"{rel}: {sidecar_defects[rel]}",
            )
        )

    profile_valid = True
    try:
        schema.validate_profile(profile)
        legs.append(_pass(LEG_PROFILE_VALIDATION))
    except schema.SchemaValidationError as exc:
        profile_valid = False
        legs.append(
            _fail(
                LEG_PROFILE_VALIDATION,
                expected="valid strict v2 ten-cell constructed-reference"
                " profile (R-001..R-011)",
                observed=str(exc),
            )
        )

    # Per-record validation (R-031/R-045), file+line named.
    record_errors: list[str] = []
    for rel in sorted(records_by_rel):
        lines = record_lines_by_rel[rel]
        for index, record in enumerate(records_by_rel[rel]):
            try:
                schema.validate_record(record)
            except schema.RecordValidationError as exc:
                lineno = lines[index] if index < len(lines) else "?"
                record_errors.append(f"{rel}: line {lineno}: {exc}")
    records_valid = not record_errors
    _record_leg(
        legs,
        "record_validation",
        records_valid,
        expected="non-reversible canonical-event per-item records"
        " (R-031/R-045)",
        observed="; ".join(record_errors[:5]),
    )

    grid = _as_dict(profile.get("grid"))
    cells = [c for c in (cells_raw or []) if isinstance(c, dict)]

    # Records keyed per declared cell (mapping first, naming fallback).
    mapping = _as_dict(grid.get("record_files"))
    records_by_cell: dict[str, list[dict[str, Any]]] = {}
    complete_by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in cells:
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str):
            continue
        rel = mapping.get(cell_id)
        if not isinstance(rel, str):
            rel = f"records/{cell_id}.jsonl"
        if rel in records_by_rel:
            records_by_cell[cell_id] = records_by_rel[rel]
            try:
                complete_by_cell[cell_id] = _complete_key_map(
                    records_by_rel[rel]
                )
            except _LEG_CATCH:
                pass

    # MA2-002b: every leg builder runs under the class-fix guard — an
    # unexpected exception inside a builder becomes THAT leg's FAIL and the
    # run still reaches a verdict + receipt (exit 1, never an internal
    # abort). Typed ColmAimsError-family control flow passes through.
    _guarded(legs, LEG_GRID_COMPLETENESS, _grid_completeness_leg, legs, grid, cells)
    _guarded(
        legs,
        LEG_RECORD_FILE_BIJECTION,
        _record_file_bijection_leg,
        legs,
        grid,
        cells,
        snapshot,
    )
    shared_keys, shared_digest = _guarded(
        legs,
        LEG_ITEM_KEY_SET,
        _item_key_set_leg,
        legs,
        grid,
        complete_by_cell,
        default=(None, None),
    )
    _guarded(legs, LEG_HELD_FIXED, _held_fixed_leg, legs, grid, cells)
    _guarded(
        legs,
        LEG_MC_STOP_WITHIN_CAL,
        _mc_stop_within_calibration_leg,
        legs,
        cells,
        complete_by_cell,
    )
    _guarded(
        legs,
        LEG_EVENT_REPRESENTATION,
        _event_representation_leg,
        legs,
        cells,
        complete_by_cell,
    )
    _guarded(legs, LEG_COUNTS, _counts_and_rates_legs, legs, cells, records_by_cell)
    _guarded(legs, LEG_ESTIMAND_LABELS, _estimand_label_leg, legs, cells)
    _guarded(legs, LEG_CELL_COMPARABILITY, _cell_comparability_leg, legs, cells)
    _guarded(
        legs,
        LEG_INFERENCE_RECOMPUTE,
        _inference_legs,
        legs,
        profile,
        cells,
        complete_by_cell,
        shared_keys,
        shared_digest,
    )

    try:
        closure = classify_certifiability(profile)
    except Exception:  # noqa: BLE001 - any classification defect fails closed
        closure = HISTORICAL_NONCERTIFYING
    classifications["profile.json"] = closure

    cells_valid = not any(
        leg["status"] == "FAIL"
        for leg in legs
        if str(leg.get("leg_id", "")).startswith(("grid_", "counts", "rates"))
    )
    artifacts_valid = profile_valid and records_valid and cells_valid
    if profile_valid and cells_valid:
        validated.append("profile.json")
    if records_valid:
        validated.extend(sorted(records_by_rel))

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
            manifest_obj,
            legacy_by_rel,
            complete_by_cell,
            shared_keys,
            shared_digest,
            closure,
            artifacts_valid,
        )

    failing = [leg for leg in legs if leg["status"] == "FAIL"]
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
        "schema_version": schema.SCHEMA_VERSION,
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    report.receipt_path = receipt_mod.emit_receipt(
        payload, receipts_dir=receipts_dir, verified_tree=tree
    )
    return report


# ---------------------------------------------------------------------------
# R-069: canonical selection wired into the ACTUAL release path
# ---------------------------------------------------------------------------


def run_release_over_runs_root(
    runs_root: Path,
    *,
    expectations: Path,
    receipts_dir: Path,
) -> VerificationReport:
    """The release entry over a runs site: resolve the canonical package
    exclusively through the ledger pointer, then verify exactly that
    package. Every refusal class FAILS the release run itself (a failing
    ``canonical_selection`` leg + receipt), never falls back, never selects
    newest-wins (R-069/R-039)."""
    runs_root = Path(runs_root)
    receipts_dir = Path(receipts_dir)
    expectations_path = Path(expectations)
    exp_obj, _ = _load_expectations(expectations_path)
    anchor = exp_obj["anchor"]
    base = expectations_path.parent
    ledger_rel = anchor["ledger_path"]
    ledger_candidate = base / ledger_rel if isinstance(ledger_rel, str) else None
    try:
        if ledger_candidate is None or Path(ledger_rel).is_absolute():
            raise ColmAimsError(
                f"anchor ledger_path {ledger_rel!r} must be a plain relative"
                " sibling path under the expectations base (R-013)"
            )
        ledger_bytes = schema.read_regular_file_bytes(ledger_candidate)
        ledger_doc = _load_json_object_strict(
            ledger_bytes, Path(str(ledger_rel)).name
        )
        run_dir = resolve_canonical_package(runs_root, ledger_doc)
    except schema.TypedIngressError:
        raise
    except ColmAimsError as exc:
        legs = [
            _fail(
                LEG_CANONICAL_SELECTION,
                expected="canonical package selected exclusively through"
                " the ledger pointer: a real, non-empty, non-symlink run"
                " directory resolved under the runs root (R-069)",
                observed=str(exc),
            )
        ]
        report = VerificationReport(
            mode="release",
            verdict=VERDICT_FAIL,
            legs=legs,
            validated_artifacts=[],
        )
        payload = {
            "schema_version": schema.SCHEMA_VERSION,
            "mode": "release",
            "verdict": VERDICT_FAIL,
            "legs": legs,
            "validated_artifacts": [],
            "classifications": {},
            "input_tree_sha256": None,
            "expectations_anchor_sha256": None,
            "verifier_code_sha256": _code_digest(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        report.receipt_path = receipt_mod.emit_receipt(
            payload, receipts_dir=receipts_dir, verified_tree=runs_root
        )
        return report
    return run_verifier(
        run_dir / "tree",
        mode="release",
        receipts_dir=receipts_dir,
        expectations=expectations_path,
        pre_legs=[_pass(LEG_CANONICAL_SELECTION)],
    )


# ---------------------------------------------------------------------------
# Release legs
# ---------------------------------------------------------------------------


def _contained_reference(base: Path, rel: Any, tree: Path) -> Path | None:
    """Resolve an anchor-referenced sidecar path safely: plain relative,
    resolves (symlink-free) UNDER ``base``, does NOT collapse INTO the
    verified ``tree``."""
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
        return None
    if schema.resolves_inside(joined, tree):
        return None
    return joined


def _release_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    expectations_path: Path,
    tree: Path,
    snapshot: dict[str, bytes],
    sha_by_rel: dict[str, str],
    profile: dict[str, Any],
    manifest_obj: dict[str, Any] | None,
    legacy_by_rel: dict[str, dict[str, Any]],
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
    shared_keys: list[str] | None,
    shared_digest: str | None,
    closure: str,
    artifacts_valid: bool,
) -> None:
    prov = _as_dict(profile.get("provenance"))
    base = expectations_path.parent

    # MA2-002b: release leg builders run under the same class-fix guard.
    ledger_doc, external_ids = _guarded(
        legs,
        LEG_ANCHOR_COMMIT,
        _anchor_legs,
        legs,
        exp,
        base,
        tree,
        prov,
        default=(None, None),
    )
    _guarded(legs, "tree_files", _tree_file_legs, legs, exp, sha_by_rel)
    _guarded(
        legs, "bindings", _binding_legs, legs, exp, sha_by_rel, profile, prov
    )
    _guarded(
        legs, f"{ANCHORED_GRID_PREFIX}pins", _anchored_grid_legs, legs, exp,
        profile,
    )
    _guarded(
        legs,
        f"{ANCHORED_INFERENCE_PREFIX}pins",
        _anchored_inference_legs,
        legs,
        exp,
        profile,
    )
    _guarded(legs, "provenance_nouns", _provenance_table_legs, legs, prov)
    _guarded(
        legs,
        "mc_build_internal_consistency",
        _mc_build_consistency_leg,
        legs,
        prov,
    )
    _guarded(
        legs, "model_revision_immutability", _model_revision_leg, legs, prov
    )
    _guarded(
        legs,
        "splits_eval_recompute",
        _splits_recompute_leg,
        legs,
        prov,
        shared_keys,
        shared_digest,
    )
    _guarded(
        legs,
        "estimand_reconciliation",
        _estimand_reconciliation_leg,
        legs,
        profile,
        prov,
        ledger_doc,
    )
    _record_leg(
        legs,
        "closure_certifiability",
        closure == CERTIFIABLE,
        expected=CERTIFIABLE,
        observed=closure,
    )
    _guarded(legs, "rights_release", _rights_legs, legs, exp, base, tree, snapshot)
    _guarded(
        legs, "manifest_reconciliation", _manifest_legs, legs, manifest_obj,
        snapshot,
    )
    _guarded(
        legs,
        "ledger_validation",
        _ledger_legs,
        legs,
        ledger_doc,
        snapshot,
        legacy_by_rel,
        artifacts_valid,
        prov,
        profile,
        external_ids,
    )


def _anchor_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    base: Path,
    tree: Path,
    prov: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str] | None]:
    """Anchor cross-check before any expectation is consumed (R-013)."""
    anchor = exp["anchor"]

    ledger_doc: dict[str, Any] | None = None
    ledger_rel = anchor["ledger_path"]
    anchor_ledger_sha = anchor["ledger_sha256"]
    ledger_path = _contained_reference(base, ledger_rel, tree)
    if ledger_path is None:
        legs.append(
            _fail(
                "anchor_ledger",
                expected="frozen ledger at a relative sibling path under"
                " the expectations base, outside the verified tree (R-013)",
                observed=f"non-contained ledger_path {ledger_rel!r}",
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
        # ONE bounded read feeds both the anchor hash and the strict parse.
        ledger_bytes = schema.read_regular_file_bytes(ledger_path)
        actual_ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
        _record_leg(
            legs,
            "anchor_ledger",
            actual_ledger_sha == anchor_ledger_sha,
            expected=anchor_ledger_sha,
            observed=actual_ledger_sha,
        )
        ledger_doc = _load_json_object_strict(
            ledger_bytes, Path(str(ledger_rel)).name
        )

    anchor_commit = anchor["source_commit"]
    observed_commit = _as_dict(prov.get("dirty_state")).get("source_commit")
    if not schema.is_commit_sha(anchor_commit):
        legs.append(
            _fail(
                LEG_ANCHOR_COMMIT,
                expected="full-length reviewed source commit SHA (R-013)",
                observed=anchor_commit,
                remediation="MISSING_EXPECTATION",
            )
        )
    elif anchor_commit != observed_commit:
        legs.append(
            _fail(
                LEG_ANCHOR_COMMIT,
                expected=anchor_commit,
                observed=observed_commit,
            )
        )
    else:
        legs.append(_pass(LEG_ANCHOR_COMMIT))

    if schema.is_commit_sha(anchor_commit):
        # R-066: a SEPARATE release leg bound to the source repository.
        # False (repo available, object missing) FAILs; None (git
        # unavailable) is ALSO a FAILING leg — PASS_RELEASE cannot be
        # obtained by making git disappear. The string-exact anchor leg
        # above stays independent and passes without a checkout.
        exists = _git_object_exists(anchor_commit)
        if exists is True:
            legs.append(_pass(LEG_GIT_OBJECT))
        elif exists is False:
            legs.append(
                _fail(
                    LEG_GIT_OBJECT,
                    expected=f"commit {anchor_commit} present in the source"
                    " repository (R-066)",
                    observed="object not found",
                )
            )
        else:
            legs.append(
                _fail(
                    LEG_GIT_OBJECT,
                    expected=f"commit {anchor_commit} object-existence check"
                    " executed against the source repository (R-066)",
                    observed="source git repository unavailable — the"
                    " object-existence check could not run; release mode"
                    " fails closed",
                )
            )

    # R-065: cross-document commit equality — the frozen ledger must itself
    # be anchored to the SAME reviewed source commit; the failing leg names
    # both commits.
    if ledger_doc is not None:
        ledger_commit = ledger_doc.get("anchored_source_commit")
        _record_leg(
            legs,
            LEG_LEDGER_ANCHOR_EQ,
            ledger_commit == anchor_commit,
            expected=anchor_commit,
            observed=ledger_commit,
        )

    external_ids = anchor["external_claim_ids"]
    if isinstance(external_ids, list) and all(
        isinstance(cid, str) for cid in external_ids
    ):
        legs.append(_pass("anchor_external_claim_ids"))
        return ledger_doc, list(external_ids)
    legs.append(
        _fail(
            "anchor_external_claim_ids",
            expected="anchored external_claim_ids list in the expectations"
            " anchor block (R-024)",
            observed="absent or malformed",
            remediation="MISSING_EXPECTATION",
        )
    )
    return ledger_doc, None


def _tree_file_legs(
    legs: list[dict[str, Any]], exp: dict[str, Any], sha_by_rel: dict[str, str]
) -> None:
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


# --- binding admissibility (value predicates fire before mirror equality) --


def _admissible_value(value: Any, where: str) -> str | None:
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
    return None


def _valid_schema_profile(value: Any) -> str | None:
    problem = _admissible_value(value, "schema_profile")
    if problem is not None:
        return problem
    if not isinstance(value, dict):
        return "schema_profile must be an object"
    if not isinstance(value.get("profile_id"), str):
        return "schema_profile.profile_id must be a string"
    version = value.get("schema_version")
    if not schema.is_real_int(version):
        return "schema_profile.schema_version must be an integer"
    if not schema.is_sha256_hex(value.get("profile_sha256")):
        return "schema_profile.profile_sha256 is not a sha256 digest"
    return None


def _valid_producer(value: Any) -> str | None:
    if not isinstance(value, dict) or not value:
        return "producer must be a non-empty object"
    problem = _admissible_value(value.get("entrypoint"), "producer.entrypoint")
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
        schema.is_real_int(seed) for seed in value
    ):
        return "seeds must be a list of integers"
    return None


def _valid_dirty_state(value: Any) -> str | None:
    problem = _admissible_value(value, "dirty_state")
    if problem is not None:
        return problem
    if not isinstance(value, dict):
        return "dirty_state must be an object"
    if value.get("git_dirty") is not False:
        return "dirty_state.git_dirty must be exactly false for release"
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
    if value.get("zero_overlap") is not True:
        return "splits.zero_overlap must be exactly true"
    return None


def _valid_calibration_identity_map(value: Any) -> str | None:
    if not isinstance(value, dict):
        return (
            "calibration_identity must be a map with one entry per"
            " calibration ID (the v1 scalar shape is rejected)"
        )
    got = set(value)
    if got != schema.CALIBRATION_IDENTITY_KEYS:
        return (
            f"calibration_identity keys {sorted(got)} must be exactly"
            f" {sorted(schema.CALIBRATION_IDENTITY_KEYS)}"
        )
    for cal_id, identity in value.items():
        if not _is_resolved_identity(identity):
            return (
                f"calibration_identity[{cal_id!r}] is missing, empty, or"
                " UNRESOLVED"
            )
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
    if value.get("built_after_split") is not True:
        return "mc_build.built_after_split must be exactly true"
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
                f"model.revision {revision!r} is not an immutable"
                " full-length 40-hex commit SHA (short hashes, tags, branch"
                " names, and bare repo ids are rejected)"
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


BINDING_VALIDATORS: dict[str, Callable[[Any], str | None]] = {
    "schema_profile": _valid_schema_profile,
    "producer": _valid_producer,
    "semantic_command": _valid_semantic_command,
    "seeds": _valid_seeds,
    "dirty_state": _valid_dirty_state,
    "splits": _valid_splits,
    "calibration_identity": _valid_calibration_identity_map,
    "continuation_identity": _valid_resolved_identity,
    "input_hashes": _valid_hash_map,
    "split_metadata_sha256": _valid_sha256_scalar,
    "mc_build": _valid_mc_build,
    "model": _valid_model,
    "runtime_packages": _valid_runtime_packages,
}


def _binding_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    sha_by_rel: dict[str, str],
    profile: dict[str, Any],
    prov: dict[str, Any],
) -> None:
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
    """One binding leg = two obligations: the observed value must be
    admissible in its own right AND must match the anchored expectation —
    mirror-equality against an author-controlled proxy is never enough."""
    leg_id = f"binding_{key}"
    problem = BINDING_VALIDATORS[key](observed)
    if problem is not None:
        legs.append(
            _fail(
                leg_id,
                expected=f"admissible {key} binding value (R-012);"
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


def _anchored_grid_legs(
    legs: list[dict[str, Any]], exp: dict[str, Any], profile: dict[str, Any]
) -> None:
    """R-044: the out-of-tree expectations pin the grid identity
    SEMANTICALLY, field by field — the in-package grid block is never its
    own release oracle."""
    bindings = _as_dict(exp.get("bindings"))
    grid_pins = bindings.get("grid")
    package_grid = _as_dict(profile.get("grid"))
    if not isinstance(grid_pins, dict):
        for field_name in ANCHORED_GRID_FIELDS:
            legs.append(
                _fail(
                    f"{ANCHORED_GRID_PREFIX}{field_name}",
                    expected="<missing anchored grid pin>",
                    observed=package_grid.get(field_name),
                    remediation="MISSING_EXPECTATION",
                )
            )
        return
    for field_name in ANCHORED_GRID_FIELDS:
        if field_name not in grid_pins:
            legs.append(
                _fail(
                    f"{ANCHORED_GRID_PREFIX}{field_name}",
                    expected="<missing anchored grid pin>",
                    observed=package_grid.get(field_name),
                    remediation="MISSING_EXPECTATION",
                )
            )
            continue
        _record_leg(
            legs,
            f"{ANCHORED_GRID_PREFIX}{field_name}",
            grid_pins[field_name] == package_grid.get(field_name),
            expected=grid_pins[field_name],
            observed=package_grid.get(field_name),
        )


def _anchored_inference_legs(
    legs: list[dict[str, Any]], exp: dict[str, Any], profile: dict[str, Any]
) -> None:
    """R-052(b)/R-053: expectations pin the inference identities; the
    release additionally re-derives the seed from the expectations-pinned
    key-set digest and requires equality with the recorded integer."""
    bindings = _as_dict(exp.get("bindings"))
    inference_pins = bindings.get("inference")
    package_inference = _as_dict(profile.get("inference"))
    if not isinstance(inference_pins, dict):
        for field_name in ANCHORED_INFERENCE_FIELDS:
            legs.append(
                _fail(
                    f"{ANCHORED_INFERENCE_PREFIX}{field_name}",
                    expected="<missing anchored inference pin>",
                    observed=package_inference.get(field_name),
                    remediation="MISSING_EXPECTATION",
                )
            )
        return
    for field_name in ANCHORED_INFERENCE_FIELDS:
        if field_name not in inference_pins:
            legs.append(
                _fail(
                    f"{ANCHORED_INFERENCE_PREFIX}{field_name}",
                    expected="<missing anchored inference pin>",
                    observed=package_inference.get(field_name),
                    remediation="MISSING_EXPECTATION",
                )
            )
            continue
        pin = inference_pins[field_name]
        observed = package_inference.get(field_name)
        passed = pin == observed
        if field_name == "seed" and passed:
            pinned_digest = inference_pins.get(
                "pairing_population_keyset_sha256"
            )
            if schema.is_sha256_hex(pinned_digest):
                try:
                    passed = pairing.d7b_seed(pinned_digest) == pin
                except _LEG_CATCH:
                    passed = False
        _record_leg(
            legs,
            f"{ANCHORED_INFERENCE_PREFIX}{field_name}",
            passed,
            expected=pin,
            observed=observed,
        )


_FIELD_PREDICATES: dict[str, Callable[[Any], bool]] = {
    "nonempty_str": lambda v: isinstance(v, str)
    and bool(v.strip())
    and v != "UNRESOLVED",
    "resolved_identity": _is_resolved_identity,
    "sha256_hex": schema.is_sha256_hex,
    "commit_sha": schema.is_commit_sha,
    "is_false": lambda v: v is False,
    "is_true": lambda v: v is True,
    "positive_int": lambda v: schema.is_real_int(v) and v > 0,
    "nonneg_int": lambda v: schema.is_real_int(v) and v >= 0,
    "nonempty_str_list": lambda v: isinstance(v, list)
    and bool(v)
    and all(isinstance(x, str) and x for x in v),
    "nonempty_int_list": lambda v: isinstance(v, list)
    and bool(v)
    and all(schema.is_real_int(x) for x in v),
    "sha256_map": lambda v: isinstance(v, dict)
    and bool(v)
    and all(
        isinstance(k, str) and k and schema.is_sha256_hex(x)
        for k, x in v.items()
    ),
    "sha256_map_allow_empty": lambda v: isinstance(v, dict)
    and all(
        isinstance(k, str) and k and schema.is_sha256_hex(x)
        for k, x in v.items()
    ),
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
    "calibration_map": lambda v: _valid_calibration_identity_map(v) is None,
}

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
    ("calibration_identity", "calibration_map", "binding_calibration_identity_resolved", "AUTHOR_DECISION_REQUIRED"),
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
                f"{dotted}: fails {predicate_name}"
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


def _mc_build_consistency_leg(
    legs: list[dict[str, Any]], prov: dict[str, Any]
) -> None:
    mc_build = _as_dict(prov.get("mc_build"))
    coverage = mc_build.get("coverage_rate")
    retained = mc_build.get("retained_count")
    policy = mc_build.get("retention_policy")
    eval_count = _as_dict(
        _as_dict(prov.get("splits")).get("eval")
    ).get("count")
    problems: list[str] = []
    if schema.is_number(coverage) and schema.is_real_int(retained):
        if (retained > 0) != (float(coverage) > 0.0):
            problems.append(
                f"retained_count {retained!r} inconsistent with"
                f" coverage_rate {coverage!r}"
            )
        if schema.is_real_int(eval_count) and eval_count > 0:
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
                    "retention_policy 'retain_all' with full coverage must"
                    f" retain the whole eval split ({eval_count!r}),"
                    f" recorded {retained!r}"
                )
    _record_leg(
        legs,
        "mc_build_internal_consistency",
        not problems,
        expected="mc_build coverage/retention internally consistent with"
        " the eval split (R-012)",
        observed="; ".join(problems),
    )


def _model_revision_leg(
    legs: list[dict[str, Any]], prov: dict[str, Any]
) -> None:
    model = _as_dict(prov.get("model"))
    revision = model.get("revision")
    digest_manifest = model.get("byte_digest_manifest")
    if revision is not None:
        _record_leg(
            legs,
            "model_revision_immutability",
            schema.is_commit_sha(revision),
            expected="immutable full-length 40-hex commit SHA, or a complete"
            " canonical byte-digest manifest (short hashes, tags, branch"
            " names, and bare repo ids are rejected — repo ids are"
            " reassignable) (R-012)",
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
        expected="immutable full-length commit SHA or complete canonical"
        " byte-digest manifest (R-012)",
        observed={"revision": None, "byte_digest_manifest": digest_manifest},
    )


def _splits_recompute_leg(
    legs: list[dict[str, Any]],
    prov: dict[str, Any],
    shared_keys: list[str] | None,
    shared_digest: str | None,
) -> None:
    if shared_keys is None or shared_digest is None:
        legs.append(
            _fail(
                "splits_eval_recompute",
                expected="eval-split key set recomputable from the shared"
                " complete-key set (R-012)",
                observed="shared complete-key set unavailable",
            )
        )
        return
    splits = _as_dict(prov.get("splits"))
    eval_split = _as_dict(splits.get("eval"))
    declared_hash = eval_split.get("keyset_sha256")
    declared_count = eval_split.get("count")
    _record_leg(
        legs,
        "splits_eval_recompute",
        declared_hash == shared_digest and declared_count == len(shared_keys),
        expected={"keyset_sha256": declared_hash, "count": declared_count},
        observed={"keyset_sha256": shared_digest, "count": len(shared_keys)},
    )


def _authorized_random_k_draws(
    ledger_doc: dict[str, Any] | None,
) -> set[str]:
    """Draw ids the frozen ledger authorizes (R-025): the explicit no-draw
    sentinel plus the ARCHIVED draw identity of sanctioned Random-K rows —
    a differing fresh draw is never treated as confirmation."""
    authorized: set[str] = {"draw-none"}
    if not isinstance(ledger_doc, dict):
        return authorized
    for row in ledger_doc.get("rows") or []:
        if not isinstance(row, dict):
            continue
        if row.get("artifact_family") != "random_k":
            continue
        if row.get("author_decision") not in ledger_mod.RANDOM_K_DISPOSITIONS:
            continue
        archived = row.get("archived_draw_id")
        if isinstance(archived, str) and archived:
            authorized.add(archived)
    return authorized


def _estimand_reconciliation_leg(
    legs: list[dict[str, Any]],
    profile: dict[str, Any],
    prov: dict[str, Any],
    ledger_doc: dict[str, Any] | None,
) -> None:
    """Reconcile every cell estimand identity field against its authoritative
    source — never a self-recomputed unanchored digest (R-011/R-025)."""
    arm_ids = {
        arm.get("arm_id")
        for arm in (profile.get("arms") or [])
        if isinstance(arm, dict)
    }
    calibration_map = _as_dict(prov.get("calibration_identity"))
    authorized_draws = _authorized_random_k_draws(ledger_doc)
    problems: list[str] = []
    for cell in profile.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        cid = cell.get("cell_id", "unnamed")
        est = _as_dict(cell.get("estimand"))
        cal_id = cell.get("calibration_id")
        expected_cal = calibration_map.get(cal_id)
        if est.get("calibration_identity") != expected_cal or not (
            _is_resolved_identity(expected_cal)
        ):
            problems.append(
                f"{cid}: estimand.calibration_identity"
                f" {est.get('calibration_identity')!r} != provenance map"
                f" entry {expected_cal!r}"
            )
        if est.get("continuation_identity") != prov.get(
            "continuation_identity"
        ):
            problems.append(
                f"{cid}: estimand.continuation_identity"
                f" {est.get('continuation_identity')!r} != provenance"
                f" {prov.get('continuation_identity')!r}"
            )
        for role in ("arm_mc", "arm_ref"):
            if est.get(role) not in arm_ids:
                problems.append(
                    f"{cid}: estimand.{role} {est.get(role)!r} not among the"
                    " profile arms"
                )
        draw = est.get("random_k_draw_id")
        if draw not in authorized_draws:
            problems.append(
                f"{cid}: estimand.random_k_draw_id {draw!r} is not"
                " authorized by a sanctioned Random-K disposition in the"
                " frozen ledger (a substituted favorable draw is refused)"
            )
    _record_leg(
        legs,
        "estimand_reconciliation",
        not problems,
        expected="every cell estimand identity field reconciles with its"
        " authoritative source (provenance/arms/ledger) (R-011/R-025)",
        observed="; ".join(problems),
    )


def _rights_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    base: Path,
    tree: Path,
    snapshot: dict[str, bytes],
) -> None:
    rights_decl = exp["rights_inventory"]
    rights_rel = rights_decl["path"]
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
    rights_bytes = schema.read_regular_file_bytes(rights_path)
    rights_obj = _load_json_object_strict(
        rights_bytes, Path(str(rights_rel)).name
    )
    actual_sha = hashlib.sha256(rights_bytes).hexdigest()
    _record_leg(
        legs,
        "rights_inventory_hash",
        actual_sha == rights_decl["sha256"],
        expected=rights_decl["sha256"],
        observed=actual_sha,
    )
    # R-035/R-026: rights cover every file FOUND (the manifest itself is the
    # packaging metadata surface, not an included evidence path).
    included = sorted(
        rel for rel in snapshot if rel != "presentation_manifest.json"
    )
    try:
        ledger_mod.check_rights_release(rights_obj, included)
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
    legs: list[dict[str, Any]],
    manifest_obj: dict[str, Any] | None,
    snapshot: dict[str, bytes],
) -> None:
    if manifest_obj is None:
        legs.append(
            _fail(
                "presentation_manifest_present",
                expected="presentation_manifest.json in the artifact tree",
                observed="absent",
            )
        )
        return
    declared = [
        a.get("path")
        for a in manifest_obj.get("artifacts", [])
        if isinstance(a, dict) and isinstance(a.get("path"), str)
    ]
    allowlist = []
    for entry in manifest_obj.get("allowlist_undeclared", []):
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            allowlist.append(entry["path"])
        elif isinstance(entry, str):
            allowlist.append(entry)
    _record_leg(
        legs,
        "manifest_nonempty",
        bool(declared),
        expected=">=1 manifest-declared artifact (R-033)",
        observed="0 declared artifacts",
    )
    ghosts = sorted(p for p in declared if p not in snapshot)
    _record_leg(
        legs,
        "manifest_declared_absent",
        not ghosts,
        expected="every manifest-declared artifact present",
        observed=ghosts,
    )
    undeclared = sorted(
        rel
        for rel in snapshot
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


def _profile_estimand_set(profile: dict[str, Any]) -> set[str]:
    """The estimand identities a ledger row may claim: the two canonical
    labels (validated fields, never recorded strings) and the D7(b) Holm
    family identity."""
    return {
        schema.HEADLINE_ESTIMAND_LABEL,
        schema.FINITE_ONLY_ESTIMAND_LABEL,
        INFERENCE_FAMILY_ESTIMAND,
    }


def _provenance_identity_closure(prov: dict[str, Any]) -> set[str]:
    # MA2-002a: only string identities enter the set — an unhashable
    # artifact-controlled value must not crash set membership.
    closure: set[str] = set()

    def _add_all(values: Any) -> None:
        for value in values:
            if isinstance(value, str):
                closure.add(value)

    _add_all(_as_dict(prov.get("input_sha256")).values())
    _add_all(_as_dict(prov.get("helper_sha256s")).values())
    model = _as_dict(prov.get("model"))
    _add_all(
        (
            prov.get("split_metadata_sha256"),
            prov.get("producer_sha256"),
            model.get("weights_sha256"),
            model.get("tokenizer_config_sha256"),
        )
    )
    _add_all(_as_dict(model.get("byte_digest_manifest")).values())
    return closure


def _ledger_legs(
    legs: list[dict[str, Any]],
    ledger_doc: dict[str, Any] | None,
    snapshot: dict[str, bytes],
    legacy_by_rel: dict[str, dict[str, Any]],
    artifacts_valid: bool,
    prov: dict[str, Any],
    profile: dict[str, Any],
    external_ids: list[str] | None,
) -> None:
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
        ledger_mod.validate_ledger(ledger_doc, external_claim_ids=external_ids)
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
    package_rejected = _as_dict(profile.get("inference")).get(
        "rejected_cell_ids"
    )
    profile_estimands = _profile_estimand_set(profile)
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        claim_id = row.get("claim_id", "unnamed")
        if external_ids is not None and claim_id in external_ids:
            attribution = row.get("human_attribution")
            attributed = (
                isinstance(attribution, dict)
                and bool(attribution.get("attributed_to"))
                and bool(attribution.get("as_of"))
            )
            _record_leg(
                legs,
                f"ledger_row_{claim_id}_external_immunity",
                status == "EXTERNAL" or attributed,
                expected="anchored-EXTERNAL row recorded EXTERNAL, or a"
                " human-attributed transition (R-024)",
                observed={"claim_id": claim_id, "recorded": status},
            )
            continue
        if status == "EXTERNAL":
            # Reverse laundering: a row cannot grant ITSELF recompute
            # immunity by relabeling its status EXTERNAL.
            _record_leg(
                legs,
                f"ledger_row_{claim_id}_external_immunity",
                external_ids is None,
                expected="EXTERNAL status only on rows anchored in the"
                " expectations external_claim_ids list (R-024)",
                observed={"claim_id": claim_id, "anchored_external": False},
            )
            continue
        if row.get("artifact_family") == "inference_block":
            _record_leg(
                legs,
                f"ledger_row_{claim_id}_rejected_ids",
                row.get("rejected_cell_ids") == package_rejected,
                expected=package_rejected,
                observed=row.get("rejected_cell_ids"),
            )
        recomputed = _recompute_row_status(
            row, snapshot, legacy_by_rel, artifacts_valid, prov,
            profile_estimands,
        )
        # MA2-002a: a non-string recorded status is a shape defect scored at
        # the weakest strength — never an unhashable-lookup TypeError crash.
        recorded_strength = (
            _STATUS_STRENGTH.get(status, 0) if isinstance(status, str) else 0
        )
        _record_leg(
            legs,
            f"ledger_row_{claim_id}_recompute",
            recorded_strength <= _STATUS_STRENGTH[recomputed],
            expected="recorded status no stronger than the recomputed"
            f" status {recomputed!r} (R-012)",
            observed={
                "claim_id": claim_id,
                "recorded": status,
                "recomputed": recomputed,
            },
        )


def _recompute_row_status(
    row: dict[str, Any],
    snapshot: dict[str, bytes],
    legacy_by_rel: dict[str, dict[str, Any]],
    artifacts_valid: bool,
    prov: dict[str, Any],
    profile_estimands: set[str],
) -> str:
    """Recompute a non-EXTERNAL claim row's status from current verification
    (R-012): RE-DERIVE from the verified source of truth; identity fields
    cross-check against the verified provenance, never against themselves."""
    if row.get("rights_status") != "VERIFIED_ALLOWED":
        return "UNVERIFIED"
    artifact = row.get("artifact_id")
    if not isinstance(artifact, str) or artifact not in snapshot:
        return "UNVERIFIED"
    if row.get("producer_entrypoint") != prov.get("producer_entrypoint"):
        return "UNVERIFIED"
    calibration_values = {
        v
        for v in _as_dict(prov.get("calibration_identity")).values()
        if isinstance(v, str)
    }
    row_calibration = row.get("calibration_identity")
    if not isinstance(row_calibration, str) or (
        row_calibration not in calibration_values
    ):
        return "UNVERIFIED"
    splits = _as_dict(prov.get("splits"))
    split_names = {
        _as_dict(splits.get("fit")).get("name"),
        _as_dict(splits.get("eval")).get("name"),
    }
    split_names.discard(None)
    row_split = row.get("split_identity")
    if not isinstance(row_split, str) or row_split not in split_names:
        return "UNVERIFIED"
    model = _as_dict(prov.get("model"))
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
        if not schema.is_commit_sha(row_revision):
            return "UNVERIFIED"
        if row_revision != prov_revision:
            return "UNVERIFIED"
    else:
        digest_values = set(_as_dict(model.get("byte_digest_manifest")).values())
        if row_revision not in digest_values:
            return "UNVERIFIED"
    row_input = row.get("input_identity")
    if not isinstance(row_input, str) or row_input not in (
        _provenance_identity_closure(prov)
    ):
        return "UNVERIFIED"
    row_estimand = row.get("estimand")
    if not isinstance(row_estimand, str) or row_estimand not in (
        profile_estimands
    ):
        return "UNVERIFIED"
    claim_kind = row.get("claim_kind")
    if claim_kind not in ("aggregate", "per_item_paired"):
        return "UNVERIFIED"
    if artifact in legacy_by_rel:
        from . import legacy as legacy_mod

        # MA2-003: the per-family certify table gates this grant — only the
        # three captured paper_exports aggregate families may back an
        # aggregate row; a row backed by a `v1_profile` sidecar recomputes
        # to UNVERIFIED (so a recorded PASS on it FAILs the row leg). The
        # sidecar file itself stays tolerated (_classify_tree_sidecars).
        if not legacy_mod.legacy_certifies(legacy_by_rel[artifact], claim_kind):
            return "UNVERIFIED"
        return "PASS"
    if artifact != "profile.json" and not (
        artifact.startswith("records/") and artifact.endswith(".jsonl")
    ):
        return "UNVERIFIED"
    if not artifacts_valid:
        return "UNVERIFIED"
    return "PASS"
