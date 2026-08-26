"""Phase-4 D7(b) real-input driver (source-only; loads no models).

The FIRST production reader that constructs the caller-supplied identity
blocks the ``phase4_assemble_d7b`` orchestrator needs from the REAL frozen
inputs + an authenticated launch receipt + a records root + the pinned D6
baseline + a freshly scanned five-prong QA-012 manifest,
then calls ``build_evidence_package`` to emit the real ``profile.json`` +
evidence package + closure inventory. Before this module the identity blocks
existed only as ``tests/_colm_aims_v2_helpers.make_*`` synthesizers.

Identity-block provenance (constants vs. input-derived):

  * arms / llm_involvement / grid.held_fixed identities / estimand study-design
    fields  -> FIXED study-design constants. The per-family stop vocabulary,
    populations, sentinel/imputation conventions, and the 5x2 grid axes are
    sourced from ``schema`` (never re-hardcoded); the remaining study-design
    scalars (arm selector/scorer, MC-trajectory / calibration / continuation /
    producer-profile identities) are documented constants below, pinned by the
    independent oracle in ``tests/_colm_aims_v2_helpers`` at test time.
  * grid.item_keys_sha256 / grid.held_fixed.horizon_identity / estimand horizon
    digests  -> RECORD-DERIVED (recomputed from ``records_root`` fail-closed;
    they must equal the assembler's own recompute or the verifier refuses).
  * provenance.model  -> derived from ``frozen/model_snapshot_manifests.json``
    (the ``primary_scorer`` role). provenance.pre_package_retention /
    splits.eval  -> derived from ``frozen/pairing_eligibility_v2.json`` + the
    record keyset. provenance.input_sha256 + dirty_state.source_commit are
    bound by ``build_evidence_package`` from the staged record bytes.
  * d6_baseline  -> PINNED in ``closure``: the two-party ``main.tex``/
    ``main.pdf`` hashes and the COMPLETE FINAL_CHECKSUMS closure are source
    constants, not caller-selected authority.
  * qa012  -> the executed detector manifest (``qa012.build_inventory_manifest``
    over exactly five labelled ``--qa012-root PRONG=PATH`` arguments).

Exit codes mirror ``verify.py`` / ``phase4_assemble_d7b`` (0 pass, 2 usage,
3 ingress, 4 internal). Spec rules: R-001..R-011 (profile shape), R-040/R-043
(grid/held-fixed), R-052 (retention), R-071/R-072 (closure), R-074/R-075
(frozen loaders), R-016/R-039 (create-once publish).
The launch receipt authenticates the exact ten record hashes and activation
digest before any evidence is assembled. Spec:
.correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproducibility.colm_aims_2026 import (  # noqa: E402
    closure,
    pairing,
    phase4,
    phase4_assemble_d7b as assembler,
    qa012,
    schema,
)

# Exit-code contract (mirrors ``verify.py`` / the assembler).
EXIT_PASS = 0
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3
EXIT_INTERNAL_ERROR = 4

# ---------------------------------------------------------------------------
# Study-design identity constants (documented; ceremony-overridable).
#
# These are the fixed study-design scalars that have no dedicated ``schema``
# constant. Each is pinned by the independent ``make_*`` oracle in
# ``tests/_colm_aims_v2_helpers`` (the driver's builders are asserted equal to
# that oracle), so a drift here is caught by the contract suite rather than by
# re-hardcoding a schema-owned value.
# ---------------------------------------------------------------------------
MC_ARM_ID = "mc_trajectory"
MC_TRAJECTORY_IDENTITY = "traj-mc-v2-0001"
CONTINUATION_IDENTITY = "cont-0001"
CALIBRATION_IDENTITY: dict[str, str] = {
    "shared": "cal-shared-0001",
    "format_specific": "cal-fmt-0001",
}
PRODUCER_PROFILE_IDENTITY = f"{schema.STRICT_PROFILE_ID}:producer-0001"
PAIRING_DEFINITION = "matched_item_prefix_grid"
DENOMINATOR_POLICY = "n_complete"
TIMEOUT_RULE = "zero_indexed_stop_ge_horizon_is_timeout"
RANDOM_K_REFERENCE = "krandom"
RANDOM_K_DRAW_ID = "draw-archived-0001"
NO_DRAW_ID = "draw-none"
PRODUCER_ENTRYPOINT = "reproducibility/colm_aims_2026/phase4_driver_d7b.py"
LAUNCH_RECEIPT_NAME = "LAUNCH_RECEIPT.json"
LAUNCH_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "activation_digest",
        "ledger_sha256",
        "producer_exit_code",
        "comparator_verdict",
        "comparator_checked",
        "export_basename",
        "export_sha256",
        "records_sha256",
    }
)

# The frozen model manifest role/file map bound into provenance.model.
_PRIMARY_SCORER_ROLE = "primary_scorer"
_MODEL_WEIGHTS_FILE = "model.safetensors"
_MODEL_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_MODEL_DTYPE = "float32"
_MODEL_DEVICE_CLASS = "cpu"

# dirty_state.source_commit placeholder for a standalone provenance build;
# ``build_evidence_package`` overwrites it with the bound ``source_commit``.
_UNBOUND_SOURCE_COMMIT = "0" * 40

_FROZEN_ELIGIBILITY_BASENAME = "pairing_eligibility_v2.json"
_FROZEN_MANIFEST_BASENAME = "model_snapshot_manifests.json"

_SHA256SUM_LINE = re.compile(r"^([0-9a-f]{64})[ \t]+(.+)$")


# ---------------------------------------------------------------------------
# Record-derived identities (fail-closed; recomputed from records_root)
# ---------------------------------------------------------------------------


def record_keyset_digest(
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> str:
    """Recompute the shared pairing-population keyset digest from records.

    All ten cells share ONE item key set (R-050); a cell that disagrees is a
    fail-closed defect, never a silently-taken first cell.
    """
    reference: frozenset[str] | None = None
    for cell_id in schema.CELL_IDS:
        keys = frozenset(complete_by_cell[cell_id])
        if reference is None:
            reference = keys
        elif keys != reference:
            raise schema.ColmAimsError(
                f"cell {cell_id!r} item key set differs from the shared"
                " pairing population; all ten cells share ONE key set (R-050)"
            )
    assert reference is not None
    ordered = pairing.canonical_item_order(list(reference))
    return pairing.keyset_sha256(ordered)


def record_horizon_identity(
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> str:
    """Recompute the held-fixed horizon identity (the per-item horizon-map
    digest, R-073/R-043) from records, fail-closed across cells.

    Each cell's ``{item_key: trajectory_horizon}`` map is digested with the
    shared ``schema.horizon_map_sha256``; a cell whose horizon map digests
    differently breaks the held-fixed invariant and is refused.
    """
    per_cell: dict[str, str] = {}
    for cell_id in schema.CELL_IDS:
        records = complete_by_cell[cell_id]
        horizon_map = {
            item_key: record["trajectory_horizon"]
            for item_key, record in records.items()
        }
        per_cell[cell_id] = schema.horizon_map_sha256(horizon_map)
    distinct = set(per_cell.values())
    if len(distinct) != 1:
        raise schema.ColmAimsError(
            "per-cell horizon-map digests disagree across the ten cells —"
            " the held-fixed horizon identity is not shared (R-043/R-073)"
        )
    return next(iter(distinct))


# ---------------------------------------------------------------------------
# Fixed study-design identity blocks (arms / llm_involvement / grid / estimands)
# ---------------------------------------------------------------------------


def _build_arm(
    arm_id: str,
    *,
    family: str = "constructed_reference",
    cardinality: str = "k_way",
    construction: str = "mc_grid",
    reporting_eligibility: str = "headline_eligible",
) -> dict[str, Any]:
    """One arm identity block (R-003); stop_semantics sourced from schema."""
    return {
        "arm_id": arm_id,
        "family": family,
        "stop_semantics": schema.FAMILY_STOP_VOCAB[family],
        "construction": construction,
        "cardinality": cardinality,
        "selector": "argmax_calibrated_score",
        "scorer": "tiny-scorer",
        "candidate_pool_role": (
            "distractor_pool" if cardinality == "k_way" else "none"
        ),
        "correctness_assignment": (
            "oracle_gold" if construction == "idealized" else "option_match"
        ),
        "calibration_role": "calibrated",
        "continuation_role": "dp_continuation",
        "seed_contract": {"seeds": [1, 2, 3]},
        "reporting_eligibility": reporting_eligibility,
    }


def _build_idealized_arm(arm_id: str = "idealized") -> dict[str, Any]:
    arm = _build_arm(arm_id, cardinality="scalar", construction="idealized")
    arm["scorer"] = "prefix_to_gold_cosine"
    arm["selector"] = "threshold_on_scalar"
    return arm


def build_arms() -> list[dict[str, Any]]:
    """The frozen six-arm study design (R-003): the MC trajectory arm, the
    idealized scalar reference, three k-way constructed references, and the
    non-headline random-K disclosure arm."""
    arms = [
        _build_arm("mc_trajectory", family="learned_continuation"),
        _build_idealized_arm("idealized"),
    ]
    for reference_id in ("kdisjoint", "khard", "klex"):
        arms.append(_build_arm(reference_id))
    arms.append(
        _build_arm(
            "krandom", reporting_eligibility="non_headline_disclosure_only"
        )
    )
    return arms


def build_llm_involvement() -> dict[str, Any]:
    """The all-``none`` LLM-involvement disclosure block (no LLM in the loop)."""
    return {
        "reference_construction": "none",
        "data_plot_creation": "none",
        "evaluation": "none",
    }


def build_grid_block(
    *,
    keyset_digest: str,
    horizon_identity: str,
    mc_trajectory_identity: str = MC_TRAJECTORY_IDENTITY,
) -> dict[str, Any]:
    """The in-profile grid identity block (GRID_KEYS, R-040/R-044).

    Mirrors ``assemble_grid_block`` but is built from the record-derived
    ``keyset_digest``/``horizon_identity`` (rather than an ``InferenceResult``)
    so the driver can construct the grid before the assembler recomputes the
    inference; the two must agree or the verifier's grid legs refuse.
    """
    return {
        "reference_ids": sorted(schema.REFERENCE_IDS),
        "calibration_ids": sorted(schema.CALIBRATION_IDS),
        "cell_ids": list(schema.CELL_IDS),
        "record_files": {
            cell_id: f"records/{cell_id}.jsonl" for cell_id in schema.CELL_IDS
        },
        "item_keys_sha256": keyset_digest,
        "held_fixed": {
            "mc_trajectory_identity": mc_trajectory_identity,
            "horizon_identity": horizon_identity,
        },
    }


def _build_estimand(
    cell_id: str, *, horizon_identity: str, numerical_tolerance: float = 1e-9
) -> dict[str, Any]:
    reference_id, calibration_id = cell_id.split("__", 1)
    return {
        "arm_mc": MC_ARM_ID,
        "arm_ref": reference_id,
        "reference_id": reference_id,
        "calibration_id": calibration_id,
        "pairing_definition": PAIRING_DEFINITION,
        "timeout_parameters": {
            "horizon_map_sha256": horizon_identity,
            "rule": TIMEOUT_RULE,
        },
        "event_representation": {
            "index_base": 0,
            "horizon_identity": horizon_identity,
            "mc_trajectory_identity": MC_TRAJECTORY_IDENTITY,
            "historical_sentinel_convention": schema.SENTINEL_CONVENTION,
            "terminal_imputation_policy": schema.IMPUTATION_FINAL_PREFIX,
            "producer_profile_identity": PRODUCER_PROFILE_IDENTITY,
        },
        "population": schema.POPULATION_ALL,
        "denominator_policy": DENOMINATOR_POLICY,
        "numerical_tolerance": numerical_tolerance,
        "calibration_identity": CALIBRATION_IDENTITY[calibration_id],
        "continuation_identity": CONTINUATION_IDENTITY,
        "random_k_draw_id": (
            RANDOM_K_DRAW_ID if reference_id == RANDOM_K_REFERENCE else NO_DRAW_ID
        ),
    }


def build_estimands(
    *, horizon_identity: str, numerical_tolerance: float = 1e-9
) -> dict[str, dict[str, Any]]:
    """The per-cell estimand identity blocks (ESTIMAND_KEYS, R-005/R-073).

    Every estimand's ``timeout_parameters``/``event_representation`` horizon is
    the record-derived ``horizon_identity`` so the assembler's estimand digest
    and the verifier's horizon legs agree.
    """
    return {
        cell_id: _build_estimand(
            cell_id,
            horizon_identity=horizon_identity,
            numerical_tolerance=numerical_tolerance,
        )
        for cell_id in schema.CELL_IDS
    }


# ---------------------------------------------------------------------------
# provenance (frozen model manifest + eligibility retention + record keyset)
# ---------------------------------------------------------------------------


def _runtime_packages() -> dict[str, str]:
    import numpy as np

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
    }


def build_provenance_from_frozen(
    frozen_dir: Path,
    *,
    keyset_digest: str,
    horizon_identity: str,
    source_commit: str | None = None,
    git_dirty: bool = False,
) -> dict[str, Any]:
    """Assemble the provenance identity block from the frozen inputs.

    ``model`` is bound from the ``primary_scorer`` role of the frozen snapshot
    manifest; ``pre_package_retention`` and the eval split count come from the
    frozen pairing-eligibility artifact; the eval keyset is the record-derived
    ``keyset_digest``. The record-derived keyset and horizon identities must
    equal the eligibility artifact's frozen pins. Both frozen artifacts are
    consumed through the strict, digest-recomputing ``phase4`` loaders
    (R-074/R-075), so a malformed, missing, or mismatched frozen input is a
    typed ingress refusal. ``input_sha256`` and ``dirty_state.source_commit``
    are (re)bound by ``build_evidence_package``.
    """
    frozen_dir = Path(frozen_dir)
    eligibility = phase4.load_pairing_eligibility(
        frozen_dir / _FROZEN_ELIGIBILITY_BASENAME
    )
    manifest = phase4.load_model_snapshot_manifest(
        frozen_dir / _FROZEN_MANIFEST_BASENAME
    )

    frozen_keyset = eligibility["pairing_population_keyset_sha256"]
    if keyset_digest != frozen_keyset:
        raise schema.TypedIngressError(
            "record-derived keyset digest does not match the frozen pairing"
            f" population: {keyset_digest} != {frozen_keyset} (R-052/R-074)"
        )
    frozen_horizon = eligibility["horizon_map_sha256"]
    if horizon_identity != frozen_horizon:
        raise schema.TypedIngressError(
            "record-derived horizon identity does not match the frozen"
            f" eligibility horizon map: {horizon_identity} !="
            f" {frozen_horizon} (R-043/R-073/R-074)"
        )

    role = manifest["roles"][_PRIMARY_SCORER_ROLE]
    files = role["files"]
    model = {
        "repository_namespace": role["model_name"],
        "revision": role["hf_revision"],
        "weights_sha256": files[_MODEL_WEIGHTS_FILE]["sha256"],
        "tokenizer_config_sha256": files[_MODEL_TOKENIZER_CONFIG_FILE]["sha256"],
        "dtype": _MODEL_DTYPE,
        "device_class": _MODEL_DEVICE_CLASS,
        "numerical_settings": {"deterministic": True},
    }

    eligible_count = eligibility["eligible_count"]
    excluded_count = eligibility["excluded_count"]

    return {
        "producer_entrypoint": PRODUCER_ENTRYPOINT,
        "dirty_state": {
            "git_dirty": bool(git_dirty),
            "source_commit": source_commit or _UNBOUND_SOURCE_COMMIT,
        },
        "calibration_identity": dict(CALIBRATION_IDENTITY),
        "continuation_identity": CONTINUATION_IDENTITY,
        # R-052(a): the upstream-unpaired items are pre-package retention
        # documentation here, never in-package excluded_keys.
        "pre_package_retention": {
            "retained_count": eligible_count + excluded_count,
            "paired_count": eligible_count,
            "upstream_unpaired_count": excluded_count,
        },
        "splits": {
            "eval": {
                "name": "eval-v2",
                "count": eligible_count,
                "keyset_sha256": keyset_digest,
            }
        },
        "model": model,
        "runtime_packages": _runtime_packages(),
        "input_sha256": {},
    }


# ---------------------------------------------------------------------------
# D6 baseline (parameterized FINAL_CHECKSUMS closure) + QA-012 block
# ---------------------------------------------------------------------------


def _parse_final_checksums(raw: bytes, rel: str) -> dict[str, str]:
    """Parse a FINAL_CHECKSUMS manifest into a ``{relpath: sha256}`` map.

    Accepts a JSON object (a flat ``path -> hash`` map, a ``path -> {sha256}``
    map, or one nested under ``entries``/``files``) and falls back to the
    ``sha256sum`` text form (``<64-hex>  <relpath>`` per line).
    """

    def _normalize_path(value: str) -> str:
        normalized = value[2:] if value.startswith("./") else value
        if (
            not normalized
            or "\\" in normalized
            or normalized.startswith("/")
            or any(part in ("", ".", "..") for part in normalized.split("/"))
        ):
            raise schema.ConfigSurfaceError(
                f"{rel}: FINAL_CHECKSUMS carries unsafe path {value!r}"
            )
        return normalized

    def _insert(entries: dict[str, str], path: str, digest: str) -> None:
        normalized = _normalize_path(path)
        if normalized in entries:
            raise schema.ConfigSurfaceError(
                f"{rel}: FINAL_CHECKSUMS duplicates normalized path"
                f" {normalized!r}"
            )
        entries[normalized] = digest

    def _coerce_map(obj: dict[str, Any]) -> dict[str, str]:
        entries: dict[str, str] = {}
        for key, value in obj.items():
            if isinstance(value, str):
                _insert(entries, str(key), value)
            elif isinstance(value, dict) and isinstance(
                value.get("sha256"), str
            ):
                _insert(entries, str(key), value["sha256"])
        return entries

    try:
        # R-067: route through the hardened loader, never raw ``json.loads``.
        obj = schema.parse_json_bytes_strict(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, schema.TypedIngressError):
        obj = None
    if isinstance(obj, dict):
        for nested_key in ("entries", "files"):
            nested = obj.get(nested_key)
            if isinstance(nested, dict):
                entries = _coerce_map(nested)
                if entries:
                    return entries
        entries = _coerce_map(obj)
        if entries:
            return entries

    entries = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _SHA256SUM_LINE.match(line)
        if match is None:
            continue
        _insert(entries, match.group(2).strip(), match.group(1))
    if not entries:
        raise schema.ConfigSurfaceError(
            f"{rel}: FINAL_CHECKSUMS carries no parseable path->sha256"
            " entries (JSON map or sha256sum text)"
        )
    return entries


def build_d6_baseline(
    *,
    checksums_path: Path | None = None,
    main_tex_sha256: str | None = None,
    main_pdf_sha256: str | None = None,
    final_checksums_sha256: str | None = None,
    final_checksums_entries: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a candidate D6 manuscript baseline block.

    The COMPLETE FINAL_CHECKSUMS closure may be loaded from ``checksums_path``
    for comparison. No caller value is authority: ``evaluate_closure`` later
    requires the exact source-pinned main hashes, raw-manifest digest, and full
    entry map. Missing or different values therefore fail closed.
    """
    baseline: dict[str, Any] = {
        "main_tex_sha256": main_tex_sha256 or closure.D6_MAIN_TEX_SHA256,
        "main_pdf_sha256": main_pdf_sha256 or closure.D6_MAIN_PDF_SHA256,
    }
    if checksums_path is not None:
        path = Path(checksums_path)
        rel = path.name
        try:
            raw = schema.read_regular_file_bytes(path)
        except (OSError, schema.ColmAimsError) as exc:
            raise schema.TypedIngressError(
                f"{rel}: FINAL_CHECKSUMS file is missing or unreadable"
                f" ({exc.__class__.__name__})"
            ) from exc
        baseline["final_checksums_sha256"] = hashlib.sha256(raw).hexdigest()
        baseline["final_checksums_entries"] = _parse_final_checksums(raw, rel)
    if final_checksums_sha256 is not None:
        baseline["final_checksums_sha256"] = final_checksums_sha256
    if final_checksums_entries is not None:
        baseline["final_checksums_entries"] = dict(final_checksums_entries)
    entries = baseline.get("final_checksums_entries")
    if isinstance(entries, dict) and entries:
        baseline["final_checksums_entries_sha256"] = (
            closure.checksum_entries_sha256(entries)
        )
    return baseline


def build_qa012_block(
    *,
    roots: dict[str, Path] | None = None,
    status: str | None = None,
    inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Build QA-012 evidence from a scan or explicit blocking status (R-072).

    ``roots`` runs ``qa012.build_inventory_manifest`` over the given corpora and
    returns the full manifest so the publication boundary can validate and
    derive any satisfying closure status itself. An explicit unsatisfied
    ``status``+``inventory_sha256`` may short-circuit the scan. One of the two
    inputs is required — a satisfying QA-012 block fabricated from nothing is
    refused.
    """
    if status is not None or inventory_sha256 is not None:
        if status is None or not schema.is_sha256_hex(inventory_sha256):
            raise schema.ConfigSurfaceError(
                "explicit QA-012 requires both a status and a sha256"
                " inventory hash (R-072)"
            )
        if status in {"VERIFIED_VACUOUS", "VERIFIED_WITH_FIXTURES"}:
            raise schema.ConfigSurfaceError(
                "a closure-satisfying QA-012 status must be derived from a"
                " locally validated non-vacuous scan and, for hits, verified"
                " committed fixtures"
            )
        return {"status": status, "inventory_sha256": inventory_sha256}
    if roots:
        manifest = qa012.build_inventory_manifest(
            {name: Path(root) for name, root in roots.items()}
        )
        return manifest
    raise schema.ConfigSurfaceError(
        "QA-012 block requires either a corpus root to scan or an explicit"
        " status + inventory sha256 (R-072)"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def validate_launch_receipt(
    records_root: Path,
    receipt_path: Path,
    ledger_path: Path,
    expected_activation_digest: str,
) -> dict[str, bytes]:
    """Authenticate the ledger, promoted export, and record transaction."""
    records_root = Path(records_root).absolute()
    receipt_path = Path(receipt_path).absolute()
    if (
        receipt_path.name != LAUNCH_RECEIPT_NAME
        or receipt_path.parent != records_root.parent
        or not schema.is_sha256_hex(expected_activation_digest)
    ):
        raise schema.ConfigSurfaceError(
            "records require the sibling LAUNCH_RECEIPT.json and an exact"
            " activation digest"
        )
    raw = schema.read_regular_file_bytes(
        receipt_path, tree_root=records_root.parent
    )
    doc = schema.parse_json_bytes_strict(raw)
    if not isinstance(doc, dict) or set(doc) != LAUNCH_RECEIPT_KEYS:
        raise schema.TypedIngressError("launch receipt has a non-closed shape")
    schema.check_schema_version(doc, "launch receipt")
    if (
        doc["receipt_type"] != "phase4_launch"
        or doc["activation_digest"] != expected_activation_digest
        or not schema.is_sha256_hex(doc["ledger_sha256"])
        or type(doc["producer_exit_code"]) is not int
        or doc["producer_exit_code"] != 0
        or doc["comparator_verdict"] != "PASS"
        or doc["comparator_checked"] != 194
        or not schema.is_path_component(doc["export_basename"])
        or not schema.is_sha256_hex(doc["export_sha256"])
    ):
        raise schema.TypedIngressError(
            "launch receipt does not bind an accepted Phase-4 launch"
        )
    ledger_bytes = schema.read_regular_file_bytes(Path(ledger_path).absolute())
    if hashlib.sha256(ledger_bytes).hexdigest() != doc["ledger_sha256"]:
        raise schema.TypedIngressError(
            "launch receipt does not authenticate the supplied exception ledger"
        )
    ledger = schema.parse_json_bytes_strict(ledger_bytes)
    ledger_keys = {
        "activation_digest",
        "certificate_path",
        "certificate_commit",
        "certificate_tree",
        "argv",
        "consumed_at",
    }
    if (
        not isinstance(ledger, dict)
        or set(ledger) != ledger_keys
        or ledger.get("activation_digest") != expected_activation_digest
        or not isinstance(ledger.get("certificate_path"), str)
        or not ledger["certificate_path"]
        or not schema.is_git_object_id(ledger.get("certificate_commit"))
        or not schema.is_git_object_id(ledger.get("certificate_tree"))
        or not isinstance(ledger.get("argv"), list)
        or not ledger["argv"]
        or not all(isinstance(token, str) and token for token in ledger["argv"])
        or not isinstance(ledger.get("consumed_at"), str)
        or not ledger["consumed_at"]
    ):
        raise schema.TypedIngressError(
            "exception ledger does not bind the accepted launch transaction"
        )
    export_bytes = schema.read_regular_file_bytes(
        records_root.parent / doc["export_basename"],
        tree_root=records_root.parent,
    )
    if hashlib.sha256(export_bytes).hexdigest() != doc["export_sha256"]:
        raise schema.TypedIngressError(
            "launch receipt does not authenticate the promoted export"
        )
    bytes_by_rel = {
        f"records/{cell_id}.jsonl": schema.read_regular_file_bytes(
            records_root / f"{cell_id}.jsonl", tree_root=records_root
        )
        for cell_id in schema.CELL_IDS
    }
    observed = {
        cell_id: hashlib.sha256(
            bytes_by_rel[f"records/{cell_id}.jsonl"]
        ).hexdigest()
        for cell_id in schema.CELL_IDS
    }
    if doc["records_sha256"] != observed:
        raise schema.TypedIngressError(
            "launch receipt record hashes do not match the supplied records"
        )
    return bytes_by_rel


def run_driver(
    records_root: Path,
    out_dir: Path,
    frozen_dir: Path,
    *,
    source_commit: str,
    launch_receipt: Path,
    launch_ledger: Path,
    activation_digest: str,
    d6_checksums: Path | None = None,
    d6_main_tex_sha256: str | None = None,
    d6_main_pdf_sha256: str | None = None,
    qa012_roots: dict[str, Path] | None = None,
    run_id: str = "run-0001",
    reclaim_crashed_relic: bool = False,
) -> assembler.BuildResult:
    """Read the frozen inputs + records, construct the real identity blocks,
    and build/create-once-publish the D7(b) evidence package.

    Ordering fixes the exit-code taxonomy: record ingress first (typed ingress
    refusals -> exit 3), then the frozen loaders (ingress -> 3), then the QA-012
    surface (config -> 2), then the create-once publish.
    """
    records_root = Path(records_root)
    record_bytes = validate_launch_receipt(
        records_root, launch_receipt, launch_ledger, activation_digest
    )
    record_snapshot = assembler._record_snapshot_from_bytes(
        records_root.absolute(), record_bytes
    )
    complete_by_cell = record_snapshot.complete_by_cell
    horizon_identity = record_horizon_identity(complete_by_cell)
    keyset_digest = record_keyset_digest(complete_by_cell)

    arms = build_arms()
    llm_involvement = build_llm_involvement()
    grid = build_grid_block(
        keyset_digest=keyset_digest, horizon_identity=horizon_identity
    )
    estimands = build_estimands(horizon_identity=horizon_identity)
    provenance = build_provenance_from_frozen(
        frozen_dir,
        keyset_digest=keyset_digest,
        horizon_identity=horizon_identity,
        source_commit=source_commit,
    )
    d6_baseline = build_d6_baseline(
        checksums_path=d6_checksums,
        main_tex_sha256=d6_main_tex_sha256,
        main_pdf_sha256=d6_main_pdf_sha256,
    )
    if qa012_roots is None:
        raise schema.ConfigSurfaceError(
            "publication requires all five QA-012 corpus roots (R-072)"
        )

    return assembler.build_evidence_package(
        records_root,
        out_dir,
        source_commit=source_commit,
        arms=arms,
        provenance=provenance,
        grid=grid,
        llm_involvement=llm_involvement,
        estimands=estimands,
        d6_baseline=d6_baseline,
        qa012_roots=qa012_roots,
        item_key_derivation=schema.PHASE4_ITEM_KEY_DERIVATION,
        run_id=run_id,
        reclaim_crashed_relic=reclaim_crashed_relic,
        record_snapshot=record_snapshot,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reproducibility.colm_aims_2026.phase4_driver_d7b",
        description=(
            "Construct the real D7(b) identity blocks from the frozen inputs"
            " + a records root and build/publish the evidence package"
            " (source-only)."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--launch-receipt", required=True)
    parser.add_argument("--launch-ledger", required=True)
    parser.add_argument("--activation-digest", required=True)
    parser.add_argument("--d6-checksums", default=None)
    parser.add_argument("--d6-main-tex-sha256", default=None)
    parser.add_argument("--d6-main-pdf-sha256", default=None)
    parser.add_argument(
        "--qa012-root",
        action="append",
        default=None,
        dest="qa012_roots",
        metavar="PRONG=PATH",
    )
    parser.add_argument("--run-id", default="run-0001")
    parser.add_argument(
        "--reclaim-crashed-relic", action="store_true", default=False
    )
    return parser


def _parse_qa012_roots(values: list[str] | None) -> dict[str, Path] | None:
    """Parse the exact repeatable QA-012 ``PRONG=PATH`` CLI surface."""
    if values is None:
        return None
    parsed: dict[str, Path] = {}
    for raw in values:
        name, separator, path = str(raw).partition("=")
        if not separator or not name or not path:
            raise schema.ConfigSurfaceError(
                "--qa012-root must be PRONG=PATH (R-072)"
            )
        if name in parsed:
            raise schema.ConfigSurfaceError(
                f"--qa012-root carries duplicate prong {name!r} (R-072)"
            )
        parsed[name] = Path(path)
    if set(parsed) != set(qa012.REQUIRED_SCOPE_PRONGS):
        raise schema.ConfigSurfaceError(
            "--qa012-root must declare exactly the frozen QA-012 prongs"
            f" {list(qa012.REQUIRED_SCOPE_PRONGS)!r} (R-072)"
        )
    return parsed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the pinned exit code (0/2/3/4)."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse usage error -> pinned exit 2
        code = exc.code
        return code if isinstance(code, int) else EXIT_USAGE_ERROR

    try:
        qa012_roots = _parse_qa012_roots(args.qa012_roots)
        result = run_driver(
            Path(args.records_root),
            Path(args.out_dir),
            Path(args.frozen_dir),
            source_commit=args.source_commit,
            launch_receipt=Path(args.launch_receipt),
            launch_ledger=Path(args.launch_ledger),
            activation_digest=args.activation_digest,
            d6_checksums=(
                Path(args.d6_checksums) if args.d6_checksums else None
            ),
            d6_main_tex_sha256=args.d6_main_tex_sha256,
            d6_main_pdf_sha256=args.d6_main_pdf_sha256,
            qa012_roots=qa012_roots,
            run_id=args.run_id,
            reclaim_crashed_relic=args.reclaim_crashed_relic,
        )
    except (schema.TypedIngressError, schema.EmptyEvaluationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INGRESS_ERROR
    except (schema.ConfigSurfaceError, schema.ColmAimsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except Exception as exc:  # noqa: BLE001 - pinned internal-error code
        print(
            f"error: unexpected {exc.__class__.__name__} during driver run",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR

    print(
        f"[driver] published {result.published_tree.name}: closure"
        " SATISFIED (prepublication profile and QA-012 bytes verified)"
    )
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
