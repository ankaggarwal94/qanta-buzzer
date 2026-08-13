"""FVI candidate study + deterministic selector (indexed by SCIENTIFIC_CONTRACT.md).

Candidate grid: tolerance in {1e-6,1e-8,1e-10} x max_iterations in {50,100,200}; damping 0.5.
Representative set = 24 cells. Strict reference = tolerance 1e-10, max_iterations 200.

A candidate is eligible when all 24 representative cells:
  - converge;
  - satisfy their configured residual;
  - have median index metrics exactly equal to the strict reference;
  - have signed and absolute mean index metrics within 0.001 of the strict reference;
  - have fit-policy action disagreement rate at most 0.001 vs the strict reference.

Selection order among eligible: (1) smallest total iterations across 24 cells;
(2) larger tolerance; (3) smaller max-iteration cap. The selected candidate is then run on
all 96 fit-only cells; on any failure the next eligible candidate is chosen.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .cellcompute import (
    CellInputs,
    CellResult,
    compute_cell,
    prepare_cell_inputs,
)
from .profile import (
    FVI_MAX_ITERATIONS,
    FVI_STRICT_REFERENCE,
    FVI_TOLERANCES,
    cell_key_str,
    full_grid,
    representative_24,
)

MEAN_TOL = 0.001
ACTION_DISAGREEMENT_MAX = 0.001


def _cell_metrics(result: CellResult) -> dict[str, Any]:
    shifts = list(result.index_shift_by_item.values())
    if not shifts and result.status == "completed":
        # Fail closed instead of substituting a default: a completed cell with
        # no paired MC/QA items would otherwise fabricate median/mean shifts of
        # exactly 0.0, indistinguishable from a genuinely perfect cell on both
        # the candidate and reference sides of the eligibility comparison
        # (docs/solutions/logic-errors/scientific-metric-edge-case-guards.md).
        raise ValueError("FVI study cell has no paired MC/QA index shifts")
    arr = np.array(shifts, dtype=np.float64) if shifts else np.array([0.0])
    return {
        "status": result.status,
        "converged": bool(result.fvi.converged) if result.fvi else False,
        "iterations": int(result.fvi.iterations) if result.fvi else 0,
        "residual": float(result.fvi.final_delta) if result.fvi else float("inf"),
        "median_index": float(np.median(np.abs(arr))),
        "signed_mean": float(np.mean(arr)),
        "abs_mean": float(np.mean(np.abs(arr))),
        "actions": dict(result.stop_index_by_key),
    }


def run_candidate_on_cells(
    *,
    rows: Sequence[dict],
    cells: list[dict[str, str]],
    calibration_json: dict | None,
    tolerance_label: str,
    max_iterations: int,
    prepared: CellInputs | None = None,
) -> dict[str, Any]:
    tol = float(tolerance_label)
    per_cell: dict[str, dict[str, Any]] = {}
    total_iters = 0
    all_conv = True
    for cell in cells:
        res = compute_cell(
            rows=rows, cell=cell, calibration_json=calibration_json,
            tolerance=tol, max_iterations=max_iterations, tolerance_label=tolerance_label,
            metric_split="val",  # FVI study is fit-only
            prepared=prepared,
        )
        m = _cell_metrics(res)
        per_cell[cell_key_str(cell)] = m
        total_iters += m["iterations"]
        if not m["converged"]:
            all_conv = False
    return {
        "tolerance": tolerance_label,
        "max_iterations": int(max_iterations),
        "total_iterations": total_iters,
        "all_converged": all_conv,
        "cells": per_cell,
    }


def _action_disagreement(cand_actions: dict[str, int], ref_actions: dict[str, int]) -> float:
    keys = set(cand_actions) | set(ref_actions)
    if not keys:
        return 0.0
    diff = sum(1 for k in keys if cand_actions.get(k) != ref_actions.get(k))
    return diff / len(keys)


def candidate_is_eligible(candidate: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if reference.get("all_converged") is not True:
        reasons.append("strict reference did not converge")
    if not candidate["all_converged"]:
        reasons.append("not all representative cells converged")
    for key, ref_cell in reference["cells"].items():
        cand_cell = candidate["cells"].get(key)
        if cand_cell is None:
            reasons.append(f"missing cell {key}")
            continue
        if not cand_cell["converged"]:
            reasons.append(f"{key}: not converged")
            continue
        if cand_cell["median_index"] != ref_cell["median_index"]:
            reasons.append(f"{key}: median != reference")
        if abs(cand_cell["signed_mean"] - ref_cell["signed_mean"]) > MEAN_TOL:
            reasons.append(f"{key}: signed mean drift")
        if abs(cand_cell["abs_mean"] - ref_cell["abs_mean"]) > MEAN_TOL:
            reasons.append(f"{key}: abs mean drift")
        if _action_disagreement(cand_cell["actions"], ref_cell["actions"]) > ACTION_DISAGREEMENT_MAX:
            reasons.append(f"{key}: action disagreement")
    return (not reasons), reasons


def order_eligible(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Selection order: total iterations asc, tolerance desc, max_iterations asc."""
    return sorted(
        candidates,
        key=lambda c: (c["total_iterations"], -float(c["tolerance"]), c["max_iterations"]),
    )


def run_fvi_study(
    *,
    rows: Sequence[dict],
    calibration_json: dict | None,
) -> dict[str, Any]:
    """Full data-dependent FVI study driver. Returns a study record + selected params."""
    rep_cells = representative_24()
    prepared = prepare_cell_inputs(rows, calibration_json)

    strict = run_candidate_on_cells(
        rows=rows, cells=rep_cells, calibration_json=calibration_json,
        tolerance_label=FVI_STRICT_REFERENCE["tolerance"],
        max_iterations=int(FVI_STRICT_REFERENCE["max_iterations"]),
        prepared=prepared,
    )
    if strict.get("all_converged") is not True:
        raise ValueError("strict FVI reference did not converge")

    candidate_records: list[dict[str, Any]] = []
    for tol in FVI_TOLERANCES:
        for max_iter in FVI_MAX_ITERATIONS:
            if (
                tol == FVI_STRICT_REFERENCE["tolerance"]
                and max_iter == int(FVI_STRICT_REFERENCE["max_iterations"])
            ):
                rec = dict(strict)
            else:
                rec = run_candidate_on_cells(
                    rows=rows,
                    cells=rep_cells,
                    calibration_json=calibration_json,
                    tolerance_label=tol,
                    max_iterations=max_iter,
                    prepared=prepared,
                )
            eligible, reasons = candidate_is_eligible(rec, strict)
            rec["eligible"] = eligible
            rec["ineligibility_reasons"] = reasons
            candidate_records.append(rec)

    eligible = [c for c in candidate_records if c["eligible"]]
    ordered = order_eligible(eligible)

    selected = None
    all96 = None
    all_grid = full_grid()
    for cand in ordered:
        val = run_candidate_on_cells(
            rows=rows, cells=all_grid, calibration_json=calibration_json,
            tolerance_label=cand["tolerance"], max_iterations=cand["max_iterations"],
            prepared=prepared,
        )
        if val["all_converged"]:
            selected = {"tolerance": cand["tolerance"], "max_iterations": cand["max_iterations"]}
            all96 = {
                "tolerance": cand["tolerance"],
                "max_iterations": cand["max_iterations"],
                "all_converged": True,
                "total_iterations": val["total_iterations"],
            }
            break

    return {
        "candidate_grid": {"tolerance": list(FVI_TOLERANCES), "max_iterations": list(FVI_MAX_ITERATIONS)},
        "representative_cell_generator": "representative_24_parity",
        "strict_reference": {
            "tolerance": strict["tolerance"], "max_iterations": strict["max_iterations"],
            "total_iterations": strict["total_iterations"], "all_converged": strict["all_converged"],
        },
        "candidate_convergence_results": [
            {
                "tolerance": c["tolerance"], "max_iterations": c["max_iterations"],
                "total_iterations": c["total_iterations"], "all_converged": c["all_converged"],
                "eligible": c["eligible"], "ineligibility_reasons": c["ineligibility_reasons"],
            }
            for c in candidate_records
        ],
        "selector_rule": "min_total_iterations__then_larger_tolerance__then_smaller_max_iter",
        "selected_parameters": selected,
        "all96_fit_only_validation": all96,
    }
