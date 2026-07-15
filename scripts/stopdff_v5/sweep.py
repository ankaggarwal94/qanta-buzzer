"""Sweep orchestration (IMPLEMENTATION_PLAN.md Task 4.6 + SCIENTIFIC_CONTRACT 11-13).

Consumes adapter rows, a run spec (with selected FVI params + bootstrap replicate count),
and a common bootstrap plan; produces per-cell JSON (atomic + committed), an aggregate with
the family maximum-statistic, verdicts, release validity, backend manifest, attempt record,
and reports. It does NOT recompute embeddings or select scientific parameters.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import PROFILE_NAME
from .bootstrap import BootstrapPlan, cell_bootstrap_stats, family_statistic
from .cellcompute import compute_cell
from .identity import build_manifest, canonical_bytes, compute_id, sha256_file
from .manifests import cell_fingerprint_identity
from .profile import cell_key_str, full_grid
from .verdicts import (
    ceiling_flags,
    cell_verdict,
    coverage_clean,
    family_verdict,
    release_validity,
)

CommitFn = Callable[[], None]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8"))


@dataclass
class SweepContext:
    rows: Sequence[dict]
    calibration_json: dict | None
    run_spec: dict
    run_spec_id: str
    bootstrap_plan: BootstrapPlan
    output_dir: Path
    fvi_tolerance: str
    fvi_max_iterations: int
    backend: str  # "modal" | "local"
    profile_variant: str  # "final" | "smoke"
    adapter_fit_rows_sha256: str
    adapter_eval_rows_sha256: str
    myopic_artifact_sha256: str
    producer_hashes: dict[str, str]
    gate_overrides: dict[str, bool] = field(default_factory=lambda: {
        "allow_low_mc_retention": False, "allow_incomplete_mc_coverage": False,
    })
    cells: list[dict[str, str]] | None = None
    commit_fn: CommitFn | None = None
    environment: dict = field(default_factory=dict)
    resource_summary: dict = field(default_factory=dict)
    attempt: dict = field(default_factory=dict)


def _commit(ctx: SweepContext) -> None:
    if ctx.commit_fn is not None:
        ctx.commit_fn()


def _cell_record(ctx: SweepContext, cell: dict[str, str]) -> dict[str, Any]:
    tol = float(ctx.fvi_tolerance)
    result = compute_cell(
        rows=ctx.rows, cell=cell, calibration_json=ctx.calibration_json,
        tolerance=tol, max_iterations=ctx.fvi_max_iterations, tolerance_label=ctx.fvi_tolerance,
        metric_split="test",
    )
    fvi_settings = {
        "damping": "0.5", "tolerance": ctx.fvi_tolerance,
        "max_iterations": ctx.fvi_max_iterations, "numeric_dtype": "float64",
    }
    fp_ident = cell_fingerprint_identity(
        run_spec_id=ctx.run_spec_id, cell=cell,
        adapter_fit_rows_sha256=ctx.adapter_fit_rows_sha256,
        adapter_eval_rows_sha256=ctx.adapter_eval_rows_sha256,
        calibrator_parameters=result.calibrator_parameters,
        fvi_settings=fvi_settings, producer_hashes=ctx.producer_hashes,
        myopic_artifact_sha256=ctx.myopic_artifact_sha256,
    )
    fp_id = compute_id(fp_ident)

    record: dict[str, Any] = {
        "cell": {k: cell[k] for k in cell},
        "cell_key": cell_key_str(cell),
        "fingerprint_id": fp_id,
        "fingerprint_identity": fp_ident,
        "status": result.status,
        "run_spec_id": ctx.run_spec_id,
        "calibrator_parameters": result.calibrator_parameters,
    }
    if result.status != "completed":
        record["reason"] = result.reason
        if result.fvi is not None:
            record["fvi"] = {
                "status": result.fvi.status, "converged": result.fvi.converged,
                "iterations": result.fvi.iterations, "final_delta": result.fvi.final_delta,
            }
        return record

    stats = cell_bootstrap_stats(result.index_shift_by_item, ctx.bootstrap_plan)
    flags = ceiling_flags(result.mc_stops, result.qa_stops, list(result.index_shift_by_item.values()))
    ceiling_any = any(flags.values())
    cov_clean = coverage_clean(
        result.coverage["fallback_fraction"], result.coverage["missing_fraction"]
    )
    mc_overridden = bool(ctx.gate_overrides.get("allow_low_mc_retention") or
                         ctx.gate_overrides.get("allow_incomplete_mc_coverage"))
    verdict = cell_verdict(
        abs_median_ci=stats["abs_median_ci"], coverage_is_clean=cov_clean,
        ceiling_any=ceiling_any, mc_gate_overridden=mc_overridden,
    )
    record.update({
        "fvi": {
            "status": result.fvi.status, "converged": result.fvi.converged,
            "iterations": result.fvi.iterations, "final_delta": result.fvi.final_delta,
            "tolerance": ctx.fvi_tolerance, "max_iterations": ctx.fvi_max_iterations,
        },
        "coverage": {**result.coverage, "clean": cov_clean},
        "ceiling_flags": flags,
        "index_shift_by_item": result.index_shift_by_item,
        "bootstrap": {"point": stats["point"], "ci": stats["ci"]},
        "descriptive": result.descriptive,
        "mc_gate_overridden": mc_overridden,
        "verdict": verdict,
    })
    return record


def run_sweep(ctx: SweepContext) -> dict[str, Any]:
    cells = ctx.cells if ctx.cells is not None else full_grid()
    expected_keys = {cell_key_str(c) for c in cells}
    cells_dir = ctx.output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    per_cell_summary: dict[str, dict[str, Any]] = {}
    abs_median_reps: dict[str, np.ndarray] = {}
    abs_median_point: dict[str, float] = {}
    completed: set[str] = set()
    failed: set[str] = set()
    all_calibrators_fitted = True
    all_fvi_converged = True

    for cell in cells:
        key = cell_key_str(cell)
        record = _cell_record(ctx, cell)
        atomic_write_json(cells_dir / f"{key}.json", record)
        _commit(ctx)  # explicit Volume commit after each cell

        if record["status"] != "completed":
            failed.add(key)
            if record["status"] == "calibrator_failed":
                all_calibrators_fitted = False
            if record["status"] == "fvi_failed":
                all_fvi_converged = False
            per_cell_summary[key] = {"status": record["status"], "verdict": "INVALID"}
            continue

        completed.add(key)
        stats = cell_bootstrap_stats(record["index_shift_by_item"], ctx.bootstrap_plan)
        abs_median_reps[key] = stats["abs_median_replicates"]
        abs_median_point[key] = stats["abs_median_point"]
        per_cell_summary[key] = {
            "status": "completed",
            "verdict": record["verdict"],
            "abs_median_point": record["bootstrap"]["point"]["absolute_index_median"],
            "abs_median_ci": record["bootstrap"]["ci"]["absolute_index_median"],
            "coverage_clean": record["coverage"]["clean"],
            "ceiling_any": any(record["ceiling_flags"].values()),
        }

    # Family statistic across completed cells.
    family = None
    if abs_median_reps:
        fam = family_statistic(abs_median_reps, abs_median_point)
        all_cells_pass = all(
            v.get("verdict") == "PASS" for v in per_cell_summary.values()
        ) and completed == expected_keys
        mc_override_active = bool(ctx.gate_overrides.get("allow_low_mc_retention") or
                                 ctx.gate_overrides.get("allow_incomplete_mc_coverage"))
        fam_verdict = family_verdict(
            family_ci=fam["ci"], all_cells_pass=all_cells_pass, mc_override_active=mc_override_active,
        )
        family = {"M": fam["M"], "ci": fam["ci"], "verdict": fam_verdict}

    release = release_validity(
        expected_cell_keys=expected_keys,
        present_cell_keys=sorted(completed | failed),
        completed_keys=completed,
        failed_keys=failed,
        skipped_keys=set(),
        all_calibrators_fitted=all_calibrators_fitted,
        all_fvi_converged=all_fvi_converged,
        manifests_valid=True,
        cache_matches_aggregate=True,
        bootstrap_valid=(family is not None),
        family_valid=(family is not None),
        backend_manifest_valid=True,
        attempt_history_valid=True,
    )

    aggregate = {
        "profile_name": PROFILE_NAME,
        "profile_variant": ctx.profile_variant,
        "backend": ctx.backend,
        "run_spec_id": ctx.run_spec_id,
        "bootstrap_plan_id": compute_id(_plan_identity_for(ctx.bootstrap_plan)),
        "requested": len(expected_keys),
        "completed": len(completed),
        "skipped": 0,
        "failed": len(failed),
        "expected_cell_keys": sorted(expected_keys),
        "fvi_selected": {"tolerance": ctx.fvi_tolerance, "max_iterations": ctx.fvi_max_iterations},
        "cells": per_cell_summary,
        "family": family,
        "gate_overrides": ctx.gate_overrides,
        "release_status": "VALID" if release.valid else "INVALID",
        "release_reasons": release.reasons,
    }
    atomic_write_json(ctx.output_dir / "aggregate.json", aggregate)

    # Backend manifest exclusivity (IDENTITY 11).
    if ctx.backend == "modal":
        run_manifest = build_manifest(
            {"kind": "run_manifest", "backend": "modal", "run_spec_id": ctx.run_spec_id},
            environment=ctx.environment, resource_summary=ctx.resource_summary,
        )
        atomic_write_json(ctx.output_dir / "run_manifest.json", run_manifest)
    else:
        cmd_manifest = build_manifest(
            {"kind": "command_manifest", "backend": "local", "run_spec_id": ctx.run_spec_id},
            environment=ctx.environment, resource_summary=ctx.resource_summary,
        )
        atomic_write_json(ctx.output_dir / "command_manifest.json", cmd_manifest)

    # Self-contained spec + bootstrap plan manifests for the standalone checker.
    atomic_write_json(
        ctx.output_dir / "run_spec.json", {"id": ctx.run_spec_id, "identity": ctx.run_spec}
    )
    from .bootstrap import plan_identity
    plan_ident = plan_identity(ctx.bootstrap_plan)
    atomic_write_json(
        ctx.output_dir / "bootstrap_plan.json",
        {"id": compute_id(plan_ident), "identity": plan_ident,
         "item_ids": ctx.bootstrap_plan.item_ids},
    )

    if ctx.environment:
        atomic_write_json(ctx.output_dir / "environment.json", ctx.environment)
    if ctx.resource_summary:
        atomic_write_json(ctx.output_dir / "resource_summary.json", ctx.resource_summary)
    if ctx.attempt:
        _append_attempt(ctx.output_dir / "attempts.jsonl", ctx.attempt)
    _commit(ctx)

    return aggregate


def _plan_identity_for(plan: BootstrapPlan) -> dict:
    from .bootstrap import plan_identity
    return plan_identity(plan)


def _append_attempt(path: Path, attempt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(attempt, sort_keys=True) + "\n")
