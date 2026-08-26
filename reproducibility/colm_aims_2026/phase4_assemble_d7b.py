"""Phase-4 D7(b) evidence-package assembler (source-only; loads no models).

The FIRST production code that assembles the regenerated D7(b) ten-cell
evidence package from per-cell record files:

  * ``profile.json`` — the strict v2 ten-cell constructed-reference profile
    (grid + inference in-profile), with per-cell shared-percentile-bootstrap
    intervals and the ten-cell Holm family (m=10, alpha 0.05).
  * the published evidence package (create-once, content-addressed).
  * the satisfied ``CAMERA_READY_CLOSURE`` inventory.

Responsibilities are split into PURE builders (``compute_inference`` and the
``assemble_*`` shapers) and a create-once orchestrator/CLI
(``build_evidence_package`` / ``main``). The D7(b) arithmetic is delegated to
``pairing`` (never reimplemented here); artifact writes go through the
``schema`` create-once primitives (``write_profile`` /
``publish_evidence_package``), which consume the ``scripts/stopdff_v5`` fileio
primitives rather than forking them (R-016/R-018/R-039).

The pure shapers are the import target for ``tests/_colm_aims_v2_helpers``:
the existing contract suite drives production ``assemble_*`` through the test
identity builders, so a shape drift is caught by the full suite.

Spec rules exercised: R-050..R-058 (D7(b) inference), R-001..R-011 (profile
shape), R-071/R-072 (closure), R-016/R-039 (create-once publish).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproducibility.colm_aims_2026 import (  # noqa: E402
    closure,
    pairing,
    qa012,
    schema,
    verifier,
)

# Exit-code contract (mirrors ``verify.py``): 0 pass, 2 usage, 3 ingress,
# 4 internal.
EXIT_PASS = 0
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3
EXIT_INTERNAL_ERROR = 4

# QA-012 build result -> closure status (closure.py:35-37; qa012.py:97).
_QA012_STATUS_BY_RESULT = {
    "zero_hit": "VERIFIED_VACUOUS",
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellInference:
    """The recomputed D7(b) inference for one cell (R-005/R-006/R-048..R-055)."""

    counts: dict[str, Any]
    rates: dict[str, Any]
    headline_summary: dict[str, Any]
    finite_only_summary: dict[str, Any]
    ci: tuple[float, float]
    raw_p_value: float
    holm_rank: int
    holm_adjusted_p_value: float
    holm_rejected: bool


@dataclass(frozen=True)
class InferenceResult:
    """The collection-level D7(b) inference over all ten cells (R-050..R-056)."""

    keyset_digest: str
    order_digest: str
    seed: int
    matrix_digest: dict[str, Any]
    per_cell: dict[str, CellInference]
    holm: dict[str, Any]


@dataclass
class BuildResult:
    """Outputs of a single create-once ``build_evidence_package`` call."""

    published_tree: Path
    profile_path: Path
    closure_inventory_path: Path
    profile: dict[str, Any]
    closure_inventory: dict[str, Any]
    inference: InferenceResult
    input_sha256: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordSnapshot:
    """One validated byte snapshot of all ten record files."""

    records_root: Path
    complete_by_cell: dict[str, dict[str, dict[str, Any]]]
    bytes_by_rel: dict[str, bytes]


# ---------------------------------------------------------------------------
# Record ingress
# ---------------------------------------------------------------------------


def _record_snapshot_from_bytes(
    records_root: Path, bytes_by_rel: dict[str, bytes]
) -> RecordSnapshot:
    """Validate retained bytes and derive their complete-pair map."""
    expected_rels = {
        f"records/{cell_id}.jsonl" for cell_id in schema.CELL_IDS
    }
    if set(bytes_by_rel) != expected_rels:
        raise schema.TypedIngressError(
            "record snapshot must carry exactly the ten canonical record"
            " paths"
        )

    complete: dict[str, dict[str, dict[str, Any]]] = {}
    captured_bytes: dict[str, bytes] = {}
    for cell_id in schema.CELL_IDS:
        rel = f"records/{cell_id}.jsonl"
        raw = bytes_by_rel[rel]
        if type(raw) is not bytes:
            raise schema.TypedIngressError(
                f"{rel}: retained record snapshot must contain bytes"
            )
        loaded = schema.load_records_bytes(raw, rel)
        by_key: dict[str, dict[str, Any]] = {}
        first_line_by_key: dict[str, int] = {}
        for record, lineno in zip(
            loaded["records"], loaded["line_numbers"], strict=True
        ):
            try:
                schema.validate_record(record)
            except schema.SchemaValidationError as exc:
                raise schema.TypedIngressError(
                    f"{rel}: line {lineno}: invalid record: {exc}"
                ) from exc
            if pairing.classify_record(record)["status"] != "complete":
                raise schema.TypedIngressError(
                    f"{rel}: line {lineno}: excluded or incomplete record"
                    " is not admissible in the D7(b) complete-pair corpus"
                    " (R-041/R-042)"
                )
            item_key = record["item_key"]
            if item_key in by_key:
                raise schema.TypedIngressError(
                    f"{rel}: line {lineno}: duplicate item_key; first seen"
                    f" on line {first_line_by_key[item_key]} (R-041)"
                )
            by_key[item_key] = record
            first_line_by_key[item_key] = lineno
        if len(loaded["records"]) != schema.EXPECTED_COMPLETE_PAIRS:
            raise schema.TypedIngressError(
                f"{rel}: carries {len(loaded['records'])} physical records;"
                f" exactly {schema.EXPECTED_COMPLETE_PAIRS} required (R-042)"
            )
        if len(by_key) != schema.EXPECTED_COMPLETE_PAIRS:
            raise schema.TypedIngressError(
                f"{rel}: carries {len(by_key)} unique complete pairs;"
                f" exactly {schema.EXPECTED_COMPLETE_PAIRS} required (R-042)"
            )
        complete[cell_id] = by_key
        captured_bytes[rel] = raw
    return RecordSnapshot(
        records_root=records_root,
        complete_by_cell=complete,
        bytes_by_rel=captured_bytes,
    )


def load_record_snapshot(records_root: Path) -> RecordSnapshot:
    """Read, validate, and retain every physical record exactly once.

    Duplicate item keys are rejected before dictionary insertion. The same
    captured bytes drive validation, inference, hashing, and publication, so a
    later pathname substitution cannot change the assembled package.
    """
    records_root = Path(records_root).absolute()
    bytes_by_rel: dict[str, bytes] = {}
    for cell_id in schema.CELL_IDS:
        rel = f"records/{cell_id}.jsonl"
        path = records_root / f"{cell_id}.jsonl"
        bytes_by_rel[rel] = schema.read_regular_file_bytes(
            path, tree_root=records_root
        )
    return _record_snapshot_from_bytes(records_root, bytes_by_rel)


def _revalidate_record_snapshot(
    record_snapshot: RecordSnapshot, records_root: Path
) -> RecordSnapshot:
    """Copy and revalidate a caller-retained snapshot before publication."""
    if not isinstance(record_snapshot, RecordSnapshot):
        raise schema.ConfigSurfaceError("record_snapshot has the wrong type")
    if record_snapshot.records_root != records_root:
        raise schema.ConfigSurfaceError(
            "record snapshot root does not match records_root"
        )
    validated = _record_snapshot_from_bytes(
        records_root, dict(record_snapshot.bytes_by_rel)
    )
    if record_snapshot.complete_by_cell != validated.complete_by_cell:
        raise schema.TypedIngressError(
            "retained record snapshot bytes do not match its parsed records"
        )
    return validated


def load_complete_by_cell(
    records_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load the validated one-read record snapshot into the cell/key map."""
    return load_record_snapshot(records_root).complete_by_cell


# ---------------------------------------------------------------------------
# Inference (delegates all arithmetic to ``pairing``)
# ---------------------------------------------------------------------------


def compute_inference(
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> InferenceResult:
    """Recompute the frozen D7(b) inference from the ten complete-pair cells.

    The shared item key set (identical across all ten cells, R-050) fixes the
    seed and the one resample matrix; each cell's interval/p-value/summaries
    are recomputed over the canonical item order, then the ten raw p-values
    feed the m=10 Holm family (R-056).
    """
    present = set(complete_by_cell)
    expected = set(schema.CELL_IDS)
    if present != expected:
        raise schema.ColmAimsError(
            "complete_by_cell must carry exactly the ten-cell 5x2 grid;"
            f" missing {sorted(expected - present)} /"
            f" unexpected {sorted(present - expected)} (R-040)"
        )

    reference_keys: frozenset[str] | None = None
    for cell_id in schema.CELL_IDS:
        keys = frozenset(complete_by_cell[cell_id])
        if len(keys) != schema.EXPECTED_COMPLETE_PAIRS:
            raise schema.ColmAimsError(
                f"cell {cell_id!r} carries {len(keys)} complete pairs;"
                f" exactly {schema.EXPECTED_COMPLETE_PAIRS} required (R-042)"
            )
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise schema.ColmAimsError(
                f"cell {cell_id!r} item key set differs from the shared"
                " pairing population; all ten cells share ONE key set (R-050)"
            )
    assert reference_keys is not None  # ten cells guaranteed above

    # R-043: reference construction may change the REF arm only. The raw MC
    # trajectory stop is held fixed across all five references within each
    # calibration condition. Enforce this before any inferential arithmetic.
    for calibration_id in schema.CALIBRATION_IDS:
        baseline_cell = f"{schema.REFERENCE_IDS[0]}__{calibration_id}"
        baseline = {
            key: (record.get("mc_event_status"), record.get("mc_stop_step"))
            for key, record in complete_by_cell[baseline_cell].items()
        }
        for reference_id in schema.REFERENCE_IDS[1:]:
            cell_id = f"{reference_id}__{calibration_id}"
            observed = {
                key: (
                    record.get("mc_event_status"),
                    record.get("mc_stop_step"),
                )
                for key, record in complete_by_cell[cell_id].items()
            }
            if observed != baseline:
                differing = sum(
                    observed.get(key) != baseline[key] for key in baseline
                )
                raise schema.ColmAimsError(
                    f"cell {cell_id!r} MC stops differ from"
                    f" {baseline_cell!r} on {differing} item(s); the five"
                    " references within each calibration must hold raw MC"
                    " trajectory stops fixed (R-043)"
                )

    ordered_keys = pairing.canonical_item_order(list(reference_keys))
    keyset_digest = pairing.keyset_sha256(ordered_keys)
    order_digest = pairing.item_order_sha256(ordered_keys)
    seed = pairing.d7b_seed(keyset_digest)
    matrix = pairing.d7b_resample_matrix(seed)
    matrix_digest = pairing.d7b_matrix_digest_record(matrix, order_digest)

    per_cell: dict[str, CellInference] = {}
    raw_p_by_cell: dict[str, float] = {}
    for cell_id in schema.CELL_IDS:
        records_map = complete_by_cell[cell_id]
        ordered = pairing.canonical_item_order(list(records_map))
        ordered_records = [records_map[key] for key in ordered]
        # ``sentinel_coded_shift_vector`` already returns float64 over the
        # canonical item order — the exact vector the interval/p-value legs
        # consume (R-050/R-054/R-055).
        d = pairing.sentinel_coded_shift_vector(records_map, ordered)
        lo, hi = pairing.d7b_interval(d, matrix)
        p = pairing.d7b_p_value(d, matrix)
        raw_p_by_cell[cell_id] = p
        counts = pairing.recompute_counts(ordered_records)
        per_cell[cell_id] = CellInference(
            counts=counts,
            rates=pairing.compute_rates(counts),
            headline_summary=pairing.sentinel_coded_headline_summary(
                ordered_records
            ),
            finite_only_summary=pairing.finite_only_timing_summary(
                ordered_records
            ),
            ci=(lo, hi),
            raw_p_value=p,
            holm_rank=0,
            holm_adjusted_p_value=0.0,
            holm_rejected=False,
        )

    holm = pairing.d7b_holm(raw_p_by_cell)
    for cell_id in schema.CELL_IDS:
        cell_holm = holm["per_cell"][cell_id]
        base = per_cell[cell_id]
        per_cell[cell_id] = CellInference(
            counts=base.counts,
            rates=base.rates,
            headline_summary=base.headline_summary,
            finite_only_summary=base.finite_only_summary,
            ci=base.ci,
            raw_p_value=base.raw_p_value,
            holm_rank=cell_holm["holm_rank"],
            holm_adjusted_p_value=cell_holm["holm_adjusted_p_value"],
            holm_rejected=cell_holm["holm_rejected"],
        )

    return InferenceResult(
        keyset_digest=keyset_digest,
        order_digest=order_digest,
        seed=seed,
        matrix_digest=matrix_digest,
        per_cell=per_cell,
        holm=holm,
    )


# ---------------------------------------------------------------------------
# Pure profile shapers
# ---------------------------------------------------------------------------


def assemble_inference_block(result: InferenceResult) -> dict[str, Any]:
    """The in-profile inference identity block (INFERENCE_KEYS, R-052/R-057).

    ``numpy_version`` is pulled from the live interpreter so the recorded
    plan token can never drift from the pinned runtime (R-051).
    """
    return {
        "analysis_provenance": schema.ANALYSIS_PROVENANCE_D7B,
        "numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "generator_construction": schema.GENERATOR_CONSTRUCTION,
        "draw_count": schema.BOOTSTRAP_DRAW_COUNT,
        "sample_size": schema.EXPECTED_COMPLETE_PAIRS,
        "resampling_unit": schema.RESAMPLING_UNIT,
        "with_replacement": True,
        "dtype": "int64",
        "endpoint": False,
        "seed": result.seed,
        "seed_derivation": schema.SEED_DERIVATION_STRING,
        "pairing_population_keyset_sha256": result.keyset_digest,
        "canonical_item_order_digest": result.order_digest,
        "resample_matrix_digest": dict(result.matrix_digest),
        "familywise_alpha": 0.05,
        "family_size": 10,
        "ordered_family": list(result.holm["ordered_family"]),
        "rejected_cell_ids": list(result.holm["rejected_cell_ids"]),
    }


def assemble_cell(
    cell_id: str, result: InferenceResult, estimand: dict[str, Any]
) -> dict[str, Any]:
    """One assembled cell block (CELL_REQUIRED_KEYS, R-005/R-006/R-015)."""
    cell = result.per_cell[cell_id]
    reference_id, calibration_id = cell_id.split("__", 1)
    return {
        "cell_id": cell_id,
        "reference_id": reference_id,
        "calibration_id": calibration_id,
        "estimand": estimand,
        "estimand_digest": pairing.estimand_digest(estimand),
        "records_file": f"records/{cell_id}.jsonl",
        "counts": dict(cell.counts),
        "rates": dict(cell.rates),
        "headline_summary": dict(cell.headline_summary),
        "finite_only_summary": dict(cell.finite_only_summary),
        "interval": {
            "procedure": schema.INTERVAL_PROCEDURE,
            "draw_count": schema.BOOTSTRAP_DRAW_COUNT,
            "seed": result.seed,
            "seed_derivation": schema.SEED_DERIVATION_STRING,
            "statistic": schema.INTERVAL_STATISTIC,
            "population": schema.POPULATION_ALL,
            "quantile_method": schema.QUANTILE_METHOD,
            "ci": [cell.ci[0], cell.ci[1]],
        },
        "raw_p_value": cell.raw_p_value,
        "holm_rank": cell.holm_rank,
        "holm_adjusted_p_value": cell.holm_adjusted_p_value,
        "holm_rejected": cell.holm_rejected,
        "excluded_keys": [],
        "pairing_population_keyset_sha256": result.keyset_digest,
    }


def assemble_grid_block(
    result: InferenceResult, *, held_fixed: dict[str, Any]
) -> dict[str, Any]:
    """The in-profile grid identity block (GRID_KEYS, R-040/R-044)."""
    return {
        "reference_ids": sorted(schema.REFERENCE_IDS),
        "calibration_ids": sorted(schema.CALIBRATION_IDS),
        "cell_ids": list(schema.CELL_IDS),
        "record_files": {
            cell_id: f"records/{cell_id}.jsonl" for cell_id in schema.CELL_IDS
        },
        "item_keys_sha256": result.keyset_digest,
        "held_fixed": dict(held_fixed),
    }


def assemble_profile(
    result: InferenceResult,
    *,
    arms: list[dict[str, Any]],
    provenance: dict[str, Any],
    grid: dict[str, Any],
    llm_involvement: dict[str, Any],
    estimands: dict[str, dict[str, Any]],
    item_key_derivation: dict[str, Any] | None = None,
    numerical_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Combine the fixed profile identities with the recomputed D7(b)
    inference into a strict v2 ten-cell profile (PROFILE_TOP_LEVEL_KEYS).

    The caller supplies the identity blocks (``arms``/``provenance``/``grid``/
    ``llm_involvement``/``estimands``). The fixed profile identities come from
    ``schema``; ``item_key_derivation`` selects one of schema's two exact
    schemes and defaults to the generic opaque-hash scheme.
    """
    if item_key_derivation is None:
        item_key_derivation = schema.ITEM_KEY_DERIVATION
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "profile_id": schema.STRICT_PROFILE_ID,
        "semantic": dict(schema.SEMANTIC_BLOCK),
        "llm_involvement": dict(llm_involvement),
        "numerical_tolerance": numerical_tolerance,
        "item_key_derivation": dict(item_key_derivation),
        "arms": arms,
        "provenance": provenance,
        "grid": grid,
        "inference": assemble_inference_block(result),
        "cells": [
            assemble_cell(cell_id, result, estimands[cell_id])
            for cell_id in schema.CELL_IDS
        ],
    }


# ---------------------------------------------------------------------------
# Closure inventory
# ---------------------------------------------------------------------------


def qa012_status_block(manifest: dict[str, Any]) -> dict[str, Any]:
    """Map a ``qa012.build_inventory_manifest`` result to the closure
    ``qa012`` block (closure.py:35-37/110-120)."""
    qa012.validate_inventory_manifest(manifest)
    result = manifest["result"]
    if result == "hits":
        status = (
            "VERIFIED_WITH_FIXTURES"
            if qa012.hit_fixtures_verified(manifest)
            else "HITS_PRESENT"
        )
    elif result in _QA012_STATUS_BY_RESULT:
        status = _QA012_STATUS_BY_RESULT[result]
    elif result == "incomplete_scope":
        status = "UNLOCATABLE_ESCALATE"
    else:
        raise schema.ColmAimsError(
            f"unrecognised QA-012 inventory result {result!r};"
            " expected 'zero_hit', 'hits', or 'incomplete_scope' (R-072)"
        )
    return {
        "status": status,
        "inventory_sha256": manifest["inventory_sha256"],
        "manifest": manifest,
    }


def _qa012_closure_block(evidence: dict[str, Any]) -> dict[str, Any]:
    """Derive a closure block; satisfying states require a full inventory."""
    if isinstance(evidence, dict) and "result" in evidence:
        return qa012_status_block(evidence)
    if not isinstance(evidence, dict) or set(evidence) != {
        "status",
        "inventory_sha256",
    }:
        raise schema.ConfigSurfaceError(
            "QA-012 evidence must be a full inventory manifest or an exact"
            " unsatisfied status/hash block (R-072)"
        )
    if evidence.get("status") in {
        "VERIFIED_VACUOUS",
        "VERIFIED_WITH_FIXTURES",
    }:
        raise schema.ConfigSurfaceError(
            "closure-satisfying QA-012 status must be derived from a full"
            " validated five-prong inventory manifest (R-072)"
        )
    if not isinstance(evidence.get("status"), str) or not schema.is_sha256_hex(
        evidence.get("inventory_sha256")
    ):
        raise schema.ConfigSurfaceError(
            "unsatisfied QA-012 evidence requires a status and SHA-256"
        )
    return dict(evidence)


def assemble_closure_inventory(
    *,
    d6_baseline: dict[str, Any],
    qa012: dict[str, Any],
    profile_sha256: str | None = None,
    analysis_provenance: str | None = None,
) -> dict[str, Any]:
    """The satisfied ``CAMERA_READY_CLOSURE`` inventory (R-071/R-072).

    ``d6_baseline`` (the two-party-verified manuscript baseline + complete
    FINAL_CHECKSUMS closure) stays PARAMETERIZED so the real run binds the
    final manuscript hashes; the Holm/inference row is satisfiable only by the
    D7(b) regenerated outputs (closure.py:100-108).
    """
    bound_d6_baseline = dict(d6_baseline)
    entries = bound_d6_baseline.get("final_checksums_entries")
    if isinstance(entries, dict) and entries:
        bound_d6_baseline.setdefault(
            "final_checksums_entries_sha256",
            closure.checksum_entries_sha256(entries),
        )
    profile_verified = schema.is_sha256_hex(profile_sha256) and (
        analysis_provenance == schema.ANALYSIS_PROVENANCE_D7B
    )
    d6_verified = (
        bound_d6_baseline.get("main_tex_sha256")
        == closure.D6_MAIN_TEX_SHA256
        and bound_d6_baseline.get("main_pdf_sha256")
        == closure.D6_MAIN_PDF_SHA256
        and bound_d6_baseline.get("final_checksums_sha256")
        == closure.D6_FINAL_CHECKSUMS_SHA256
        and bound_d6_baseline.get("final_checksums_entries_sha256")
        == closure.D6_FINAL_CHECKSUMS_ENTRIES_SHA256
    )
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "d6_baseline": bound_d6_baseline,
        "rows": [
            {
                "item": "table-1-headline-shifts",
                "status": "SATISFIED" if profile_verified else "UNSATISFIED",
                "evidence": (
                    f"profile.json sha256:{profile_sha256}"
                    if profile_verified
                    else None
                ),
            },
            {
                "item": "manuscript-identity",
                "status": "EXTERNAL" if d6_verified else "UNSATISFIED",
                "evidence": (
                    "pinned D6 FINAL_CHECKSUMS raw+entry-map authority"
                    if d6_verified
                    else None
                ),
            },
        ],
        "holm_row": {
            "satisfied_by": analysis_provenance if profile_verified else None
        },
        "qa012": dict(qa012),
    }


# ---------------------------------------------------------------------------
# Create-once orchestrator
# ---------------------------------------------------------------------------


def _presentation_manifest() -> dict[str, Any]:
    artifacts: list[dict[str, str]] = [
        {"path": "profile.json", "role": "strict_profile"}
    ]
    for rel in sorted(f"records/{cell_id}.jsonl" for cell_id in schema.CELL_IDS):
        artifacts.append({"path": rel, "role": "per_item_records"})
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "artifacts": artifacts,
        "allowlist_undeclared": [],
    }


def _dump_json_bytes(obj: dict[str, Any]) -> bytes:
    return (
        json.dumps(obj, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_evidence_package(
    records_root: Path,
    out_dir: Path,
    *,
    source_commit: str,
    arms: list[dict[str, Any]],
    provenance: dict[str, Any],
    grid: dict[str, Any],
    llm_involvement: dict[str, Any],
    estimands: dict[str, dict[str, Any]],
    d6_baseline: dict[str, Any],
    qa012_roots: dict[str, Path],
    item_key_derivation: dict[str, Any] | None = None,
    run_id: str = "run-0001",
    reclaim_crashed_relic: bool = False,
    record_snapshot: RecordSnapshot | None = None,
) -> BuildResult:
    """Assemble and create-once-publish the D7(b) evidence package.

    Records are staged byte-for-byte from one validated snapshot and their
    hashes are bound into ``provenance.input_sha256``. One complete run
    envelope is published at ``out_dir/runs/<run_id>``: artifact bytes live
    under ``tree/`` and their closure evidence under ``closure/``. Nothing is
    written to the run after the create-once publish (R-016/R-039).
    """
    records_root = Path(records_root).absolute()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = out_dir / "runs"

    if record_snapshot is None:
        record_snapshot = load_record_snapshot(records_root)
    else:
        record_snapshot = _revalidate_record_snapshot(
            record_snapshot, records_root
        )
    result = compute_inference(record_snapshot.complete_by_cell)
    input_sha256 = {
        rel: hashlib.sha256(blob).hexdigest()
        for rel, blob in record_snapshot.bytes_by_rel.items()
    }
    bound_provenance = dict(provenance)
    bound_provenance["input_sha256"] = dict(input_sha256)
    dirty_state = (
        dict(bound_provenance["dirty_state"])
        if "dirty_state" in bound_provenance
        else {}
    )
    dirty_state["source_commit"] = source_commit
    bound_provenance["dirty_state"] = dirty_state
    profile = assemble_profile(
        result,
        arms=arms,
        provenance=bound_provenance,
        grid=grid,
        llm_involvement=llm_involvement,
        estimands=estimands,
        item_key_derivation=item_key_derivation,
    )
    profile_bytes = schema.encode_profile(profile)
    qa012_manifest = qa012.build_inventory_manifest(qa012_roots)
    inventory = assemble_closure_inventory(
        d6_baseline=d6_baseline,
        qa012=qa012_status_block(qa012_manifest),
        profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        analysis_provenance=profile["inference"]["analysis_provenance"],
    )
    # Fail before staging or create-once publication. A structurally valid but
    # semantically unsatisfied closure must never consume an immutable run ID.
    closure_result = closure.evaluate_closure(
        inventory,
        profile_bytes=profile_bytes,
        qa012_roots=qa012_roots,
    )
    if not closure_result["satisfied"]:
        failing_rows = ", ".join(closure_result["failing_rows"])
        raise schema.ConfigSurfaceError(
            "closure is unsatisfied; refusing immutable publication; failing"
            f" rows: {failing_rows}"
        )

    # Build the complete run envelope before its single atomic publication.
    staged_run = Path(tempfile.mkdtemp(prefix="staged-", dir=out_dir))
    staged_tree = staged_run / "tree"
    staged_closure = staged_run / "closure"
    closure_bytes = _dump_json_bytes(inventory)

    try:
        (staged_tree / "records").mkdir(parents=True)
        staged_closure.mkdir()
        for cell_id in schema.CELL_IDS:
            rel = f"records/{cell_id}.jsonl"
            blob = record_snapshot.bytes_by_rel[rel]
            (staged_tree / rel).write_bytes(blob)
        (staged_tree / "profile.json").write_bytes(profile_bytes)
        (staged_tree / "presentation_manifest.json").write_bytes(
            _dump_json_bytes(_presentation_manifest())
        )
        (staged_closure / "closure_inventory.json").write_bytes(closure_bytes)
        envelope_snapshot = verifier._read_tree_snapshot(staged_run)

        # Structural assembly alone is not publication authority.  Exercise
        # the complete source-mode semantic verifier over the exact staged
        # bytes before the create-once rename.  Its transient receipt is kept
        # outside the candidate tree so it cannot become self-attesting input.
        with tempfile.TemporaryDirectory(
            prefix="prepublish-receipts-", dir=out_dir
        ) as receipts_dir:
            prepublish = verifier.run_verifier(
                staged_tree,
                mode="source",
                receipts_dir=Path(receipts_dir),
            )
        if prepublish.verdict != verifier.VERDICT_SOURCE_PASS:
            failing_legs = ", ".join(
                leg["leg_id"]
                for leg in prepublish.legs
                if leg.get("status") != "PASS"
            )
            raise schema.ConfigSurfaceError(
                "staged evidence failed full source semantic verification;"
                f" refusing immutable publication; failing legs: {failing_legs}"
            )

        if verifier._read_tree_snapshot(staged_run) != envelope_snapshot:
            raise schema.ConfigSurfaceError(
                "staged envelope changed during semantic verification;"
                " refusing immutable publication"
            )

        published_run = schema.publish_evidence_package(
            staged_run,
            runs_root,
            run_id,
            reclaim_crashed_relic=reclaim_crashed_relic,
        )
    except BaseException:
        # ``staged_run`` is private temporary state owned by this invocation.
        # Never leave a failed candidate that could be mistaken for a run.
        shutil.rmtree(staged_run, ignore_errors=True)
        raise
    published_tree = published_run / "tree"
    closure_path = published_run / "closure" / "closure_inventory.json"

    return BuildResult(
        published_tree=published_tree,
        profile_path=published_tree / "profile.json",
        closure_inventory_path=closure_path,
        profile=profile,
        closure_inventory=inventory,
        inference=result,
        input_sha256=input_sha256,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reproducibility.colm_aims_2026.phase4_assemble_d7b",
        description=(
            "Recompute and summarise the D7(b) ten-cell inference from"
            " per-cell record files (source-only)."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser


def _inference_summary(
    result: InferenceResult, *, source_commit: str, records_root: Path
) -> dict[str, Any]:
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "analysis_provenance": schema.ANALYSIS_PROVENANCE_D7B,
        "source_commit": source_commit,
        "records_root_basename": Path(records_root).name,
        "seed": result.seed,
        "seed_derivation": schema.SEED_DERIVATION_STRING,
        "pairing_population_keyset_sha256": result.keyset_digest,
        "canonical_item_order_digest": result.order_digest,
        "resample_matrix_digest": dict(result.matrix_digest),
        "rejected_cell_ids": list(result.holm["rejected_cell_ids"]),
        "ordered_family": list(result.holm["ordered_family"]),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the pinned exit code (0/2/3/4).

    Computes the D7(b) inference from ``--records-root`` and writes an
    inference summary to ``--out-dir/inference_summary.json``. Full-profile
    assembly + create-once publish is exposed as the ``build_evidence_package``
    Python API (the producer-side identity blocks are supplied by the caller).
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse usage error -> pinned exit 2
        code = exc.code
        return code if isinstance(code, int) else EXIT_USAGE_ERROR

    try:
        records_root = Path(args.records_root)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        complete_by_cell = load_complete_by_cell(records_root)
        result = compute_inference(complete_by_cell)
        summary = _inference_summary(
            result,
            source_commit=args.source_commit,
            records_root=records_root,
        )
        (out_dir / "inference_summary.json").write_bytes(
            _dump_json_bytes(summary)
        )
    except (
        schema.TypedIngressError,
        schema.EmptyEvaluationError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INGRESS_ERROR
    except (schema.ConfigSurfaceError, schema.ColmAimsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except Exception as exc:  # noqa: BLE001 - pinned internal-error code
        print(
            f"error: unexpected {exc.__class__.__name__} during assembly",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR

    print(
        f"[assemble] D7(b) inference computed: seed={result.seed}"
        f" keyset={result.keyset_digest[:12]}..."
    )
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
