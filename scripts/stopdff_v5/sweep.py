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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import PROFILE_NAME
from .bootstrap import BootstrapPlan, cell_bootstrap_stats, family_statistic
from .cellcompute import compute_cell
from .identity import build_manifest, compute_id, loads_no_duplicate_keys
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
_INTERRUPTED_REASON = "terminal_result_missing_at_resume"


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


def _write_bound_json(path: Path, obj: Any, *, resume: bool) -> None:
    """Create evidence once; on resume accept only byte-identical existing data."""
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if not resume:
            raise FileExistsError(f"fresh run would overwrite {path}")
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"resume evidence mismatch at {path}")
        return
    atomic_write_bytes(path, data)


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
    adapter_bundle_id: str
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
    resume: bool = False


def _commit(ctx: SweepContext) -> None:
    if ctx.commit_fn is not None:
        ctx.commit_fn()


def _context_identity(ctx: SweepContext) -> tuple[dict[str, Any], str]:
    """Validate producer inputs before writing any run evidence."""
    if compute_id(ctx.run_spec) != ctx.run_spec_id:
        raise ValueError("run_spec_id does not match run_spec identity")
    spec_ids = ctx.run_spec.get("identity")
    if not isinstance(spec_ids, dict):
        raise ValueError("run spec is missing its identity graph")
    if spec_ids.get("adapter_bundle_id") != ctx.adapter_bundle_id:
        raise ValueError("run spec adapter_bundle_id does not match sweep context")
    if ctx.run_spec.get("profile_variant") != ctx.profile_variant:
        raise ValueError("run spec profile_variant does not match sweep context")
    evidence_roots = ctx.run_spec.get("evidence_roots")
    if not isinstance(evidence_roots, dict):
        raise ValueError("run spec is missing evidence_roots")
    if (
        evidence_roots.get("myopic_artifact_sha256")
        != ctx.myopic_artifact_sha256
    ):
        raise ValueError(
            "run spec myopic artifact does not match sweep context"
        )
    if evidence_roots.get("producer_hashes") != ctx.producer_hashes:
        raise ValueError(
            "run spec producer hashes do not match sweep context"
        )

    selected = ctx.run_spec.get("fvi_selected", {})
    if str(selected.get("tolerance")) != str(ctx.fvi_tolerance):
        raise ValueError("run spec FVI tolerance does not match sweep context")
    if int(selected.get("max_iterations", -1)) != int(ctx.fvi_max_iterations):
        raise ValueError("run spec FVI max_iterations does not match sweep context")
    bound_overrides = {
        key: bool(ctx.run_spec.get("gate", {}).get(key, False))
        for key in (
            "allow_low_mc_retention",
            "allow_incomplete_mc_coverage",
        )
    }
    if bound_overrides != ctx.gate_overrides:
        raise ValueError("run spec gate overrides do not match sweep context")

    plan_id = compute_id(_plan_identity_for(ctx.bootstrap_plan))
    if spec_ids.get("bootstrap_plan_id") != plan_id:
        raise ValueError("run spec bootstrap_plan_id does not match sweep context")
    return spec_ids, plan_id


def _prepare_attempt(
    ctx: SweepContext,
    *,
    bootstrap_plan_id: str,
) -> dict[str, Any]:
    """Validate attempt history and bind the next record before any evidence write."""
    if not ctx.attempt:
        raise ValueError("sweep context requires an attempt record")
    attempt = dict(ctx.attempt)
    expected_attempt_fields = {
        "run_spec_id": ctx.run_spec_id,
        "adapter_id": ctx.adapter_bundle_id,
        "bootstrap_plan_id": bootstrap_plan_id,
    }
    for key, expected in expected_attempt_fields.items():
        if key in attempt and attempt[key] != expected:
            raise ValueError(f"attempt {key} does not match sweep context")
        attempt[key] = expected

    attempt_number = attempt.get("attempt")
    if (
        not isinstance(attempt_number, int)
        or isinstance(attempt_number, bool)
        or attempt_number < 1
    ):
        raise ValueError("attempt number must be a positive integer")
    mode = attempt.get("mode")
    command = attempt.get("command")
    if not isinstance(command, list) or not all(
        isinstance(part, str)
        for part in command
    ):
        raise ValueError("attempt command must be a string list")
    if "--overwrite" in command:
        raise ValueError("evidence attempts cannot use --overwrite")

    attempts_path = ctx.output_dir / "attempts.jsonl"
    if ctx.resume:
        if mode != "resume" or command.count("--resume") != 1:
            raise ValueError("resume attempt requires one bare --resume")
        if not attempts_path.is_file():
            raise ValueError("resume requires an existing attempts.jsonl")
        existing = [
            loads_no_duplicate_keys(line)
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not existing:
            raise ValueError("resume requires a prior attempt")
        previous_number = 0
        for index, prior in enumerate(existing, start=1):
            if not isinstance(prior, dict):
                raise ValueError("resume attempt history contains a non-object")
            prior_number = prior.get("attempt")
            if (
                not isinstance(prior_number, int)
                or isinstance(prior_number, bool)
                or prior_number != index
            ):
                raise ValueError(
                    "resume attempt history contains a nonconsecutive number"
                )
            prior_command = prior.get("command")
            if (
                prior_number <= previous_number
                or prior.get("state") != "started"
                or not isinstance(prior_command, list)
                or not all(isinstance(part, str) for part in prior_command)
                or "--overwrite" in prior_command
                or prior.get("run_spec_id") != ctx.run_spec_id
                or prior.get("adapter_id") != ctx.adapter_bundle_id
                or prior.get("bootstrap_plan_id") != bootstrap_plan_id
            ):
                raise ValueError("resume attempt history invariant mismatch")
            if prior_number == 1:
                if prior.get("mode") != "fresh" or "--resume" in prior_command:
                    raise ValueError("initial attempt history invariant mismatch")
            elif (
                prior.get("mode") != "resume"
                or prior_command.count("--resume") != 1
            ):
                raise ValueError("resume attempt history mode mismatch")
            previous_number = prior_number
        last_number = int(existing[-1].get("attempt", 0))
        if attempt_number != last_number + 1:
            raise ValueError(
                "resume attempt number must immediately follow attempt history"
            )
    else:
        if mode != "fresh" or "--resume" in command:
            raise ValueError("fresh attempt must omit --resume")
        if attempt_number != 1:
            raise ValueError("fresh attempt number must be 1")
        if attempts_path.exists():
            raise FileExistsError("fresh run already has attempt history")
    return attempt


def _cell_record(ctx: SweepContext, cell: dict[str, str]) -> dict[str, Any]:
    tol = float(ctx.fvi_tolerance)
    result = compute_cell(
        rows=ctx.rows, cell=cell, calibration_json=ctx.calibration_json,
        tolerance=tol, max_iterations=ctx.fvi_max_iterations, tolerance_label=ctx.fvi_tolerance,
        metric_split="test",
    )
    fvi_settings = {
        "damping": "0.5", "tolerance": ctx.fvi_tolerance,
        "max_iterations": ctx.fvi_max_iterations,
        "required_consecutive_converged_iterations": 2,
        "numeric_dtype": "float64",
    }
    fp_ident = cell_fingerprint_identity(
        run_spec_id=ctx.run_spec_id,
        adapter_bundle_id=ctx.adapter_bundle_id,
        bootstrap_plan_id=compute_id(_plan_identity_for(ctx.bootstrap_plan)),
        cell=cell,
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
        "adapter_bundle_id": ctx.adapter_bundle_id,
        "bootstrap_plan_id": compute_id(_plan_identity_for(ctx.bootstrap_plan)),
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
        "bootstrap": {
            "point": stats["point"],
            "ci": stats["ci"],
            # The family statistic consumes the same deterministic resamples.
            # Persisting them avoids a second 1000-by-N indexed bootstrap for
            # every cell while keeping resume evidence byte-verifiable.
            "abs_median_replicates": stats["abs_median_replicates"].tolist(),
        },
        "descriptive": result.descriptive,
        "mc_gate_overridden": mc_overridden,
        "verdict": verdict,
    })
    return record


def _run_sweep_body(
    ctx: SweepContext,
    *,
    spec_ids: dict[str, Any],
    bootstrap_plan_id: str,
    precomputed_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
        record = (
            precomputed_records.get(key)
            if precomputed_records is not None
            else None
        )
        if record is None:
            record = _cell_record(ctx, cell)
        _write_bound_json(
            cells_dir / f"{key}.json",
            record,
            resume=ctx.resume,
        )
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
        abs_median_reps[key] = np.asarray(
            record["bootstrap"]["abs_median_replicates"],
            dtype=np.float64,
        )
        abs_median_point[key] = float(
            record["bootstrap"]["point"]["absolute_index_median"]
        )
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
        "adapter_bundle_id": ctx.adapter_bundle_id,
        "bootstrap_plan_id": bootstrap_plan_id,
        "fvi_study_id": spec_ids["fvi_study_id"],
        "adapter_fit_rows_sha256": ctx.adapter_fit_rows_sha256,
        "adapter_eval_rows_sha256": ctx.adapter_eval_rows_sha256,
        "myopic_artifact_sha256": ctx.myopic_artifact_sha256,
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
    _write_bound_json(
        ctx.output_dir / "aggregate.json",
        aggregate,
        resume=ctx.resume,
    )

    # Backend manifest exclusivity (IDENTITY 11).
    if ctx.backend == "modal":
        run_manifest = build_manifest(
            {
                "kind": "run_manifest",
                "backend": "modal",
                "run_spec_id": ctx.run_spec_id,
                "adapter_bundle_id": ctx.adapter_bundle_id,
                "bootstrap_plan_id": bootstrap_plan_id,
                "adapter_fit_rows_sha256": ctx.adapter_fit_rows_sha256,
                "adapter_eval_rows_sha256": ctx.adapter_eval_rows_sha256,
            },
            environment=ctx.environment, resource_summary=ctx.resource_summary,
        )
        _write_bound_json(
            ctx.output_dir / "run_manifest.json",
            run_manifest,
            resume=ctx.resume,
        )
    else:
        cmd_manifest = build_manifest(
            {
                "kind": "command_manifest",
                "backend": "local",
                "run_spec_id": ctx.run_spec_id,
                "adapter_bundle_id": ctx.adapter_bundle_id,
                "bootstrap_plan_id": bootstrap_plan_id,
                "adapter_fit_rows_sha256": ctx.adapter_fit_rows_sha256,
                "adapter_eval_rows_sha256": ctx.adapter_eval_rows_sha256,
            },
            environment=ctx.environment, resource_summary=ctx.resource_summary,
        )
        _write_bound_json(
            ctx.output_dir / "command_manifest.json",
            cmd_manifest,
            resume=ctx.resume,
        )

    # Self-contained spec + bootstrap plan manifests for the standalone checker.
    _write_bound_json(
        ctx.output_dir / "run_spec.json",
        {"id": ctx.run_spec_id, "identity": ctx.run_spec},
        resume=ctx.resume,
    )
    from .bootstrap import plan_identity
    plan_ident = plan_identity(ctx.bootstrap_plan)
    _write_bound_json(
        ctx.output_dir / "bootstrap_plan.json",
        {"id": compute_id(plan_ident), "identity": plan_ident,
         "item_ids": ctx.bootstrap_plan.item_ids},
        resume=ctx.resume,
    )

    if ctx.environment:
        _write_bound_json(
            ctx.output_dir / "environment.json",
            ctx.environment,
            resume=ctx.resume,
        )
    if ctx.resource_summary:
        _write_bound_json(
            ctx.output_dir / "resource_summary.json",
            ctx.resource_summary,
            resume=ctx.resume,
        )
    return aggregate


def _plan_identity_for(plan: BootstrapPlan) -> dict:
    from .bootstrap import plan_identity
    return plan_identity(plan)


def _append_attempt(path: Path, attempt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(attempt, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_attempt_result(
    result: Any,
    *,
    attempt_number: int,
    run_spec_id: str,
) -> str | None:
    """Return the first violation of the canonical terminal-result schema."""
    if not isinstance(result, dict):
        return "result is not an object"
    state = result.get("state")
    common = {"attempt", "state", "run_spec_id"}
    state_fields = {
        "completed": {"completed", "failed"},
        "failed": {"error_type", "error_message"},
        "interrupted": {"reason"},
    }
    if not isinstance(state, str) or state not in state_fields:
        return "state is not completed, failed, or interrupted"
    if set(result) != common | state_fields[state]:
        return f"{state} result fields do not match the canonical schema"
    if result.get("attempt") != attempt_number:
        return "attempt number does not match its filename"
    if result.get("run_spec_id") != run_spec_id:
        return "run_spec_id does not match the sweep context"
    if state == "completed":
        for field_name in ("completed", "failed"):
            value = result.get(field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                return f"{field_name} must be a nonnegative integer"
    elif state == "failed":
        if (
            not isinstance(result.get("error_type"), str)
            or not result["error_type"]
            or not isinstance(result.get("error_message"), str)
        ):
            return "failed result requires an error type and message"
    elif result.get("reason") != _INTERRUPTED_REASON:
        return "interrupted result has a noncanonical reason"
    return None


def _resume_preflight(
    ctx: SweepContext,
    *,
    cells: list[dict[str, str]],
    spec_ids: dict[str, Any],
    bootstrap_plan_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Validate all existing run evidence before the resume attempt writes."""
    cells_by_key = {cell_key_str(cell): cell for cell in cells}
    expected_keys = set(cells_by_key)
    cells_dir = ctx.output_dir / "cells"
    actual_keys = (
        {path.stem for path in cells_dir.glob("*.json")}
        if cells_dir.is_dir()
        else set()
    )
    if not actual_keys <= expected_keys:
        raise ValueError("resume cell set contains unexpected evidence")
    # Recompute only evidence that actually exists. Missing cells are computed
    # later by _run_sweep_body after every cached byte has passed preflight.
    expected_records = {
        key: _cell_record(ctx, cells_by_key[key])
        for key in sorted(actual_keys)
    }
    for key in sorted(actual_keys):
        path = cells_dir / f"{key}.json"
        data = (
            json.dumps(expected_records[key], indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"resume evidence mismatch at {path}")

    # Reconstruct the complete expected non-attempt evidence in an isolated
    # directory, then compare every existing run-level file before any actual
    # path is created or repaired.
    run_level_names = (
        "aggregate.json",
        "run_manifest.json",
        "command_manifest.json",
        "run_spec.json",
        "bootstrap_plan.json",
        "environment.json",
        "resource_summary.json",
    )
    existing_run_level = [
        name for name in run_level_names if (ctx.output_dir / name).exists()
    ]
    if existing_run_level:
        with tempfile.TemporaryDirectory(
            prefix="stopdff_v5_resume_preflight_"
        ) as td:
            probe = replace(
                ctx,
                output_dir=Path(td) / "run",
                commit_fn=None,
                resume=False,
            )
            _run_sweep_body(
                probe,
                spec_ids=spec_ids,
                bootstrap_plan_id=bootstrap_plan_id,
                precomputed_records=expected_records,
            )
            for name in existing_run_level:
                actual = ctx.output_dir / name
                expected = probe.output_dir / name
                if (
                    actual.is_symlink()
                    or not actual.is_file()
                    or not expected.is_file()
                    or actual.read_bytes() != expected.read_bytes()
                ):
                    raise ValueError(f"resume evidence mismatch at {actual}")

    attempt_records = [
        loads_no_duplicate_keys(line)
        for line in (ctx.output_dir / "attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    known_attempts = [record["attempt"] for record in attempt_records]
    results_dir = ctx.output_dir / "attempt_results"
    result_numbers: set[int] = set()
    if results_dir.exists():
        if results_dir.is_symlink() or not results_dir.is_dir():
            raise ValueError("resume attempt_results is not a directory")
        for path in results_dir.iterdir():
            try:
                number = int(path.stem)
                if (
                    number < 1
                    or path.name != f"{number}.json"
                    or number in result_numbers
                    or path.is_symlink()
                    or not path.is_file()
                ):
                    raise ValueError("noncanonical attempt result path")
                result = loads_no_duplicate_keys(
                    path.read_text(encoding="utf-8")
                )
            except (
                OSError,
                UnicodeError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError(
                    f"resume attempt result is invalid: {path}"
                ) from exc
            violation = _validate_attempt_result(
                result,
                attempt_number=number,
                run_spec_id=ctx.run_spec_id,
            )
            if number not in known_attempts or violation is not None:
                detail = f": {violation}" if violation is not None else ""
                raise ValueError(
                    f"resume attempt result mismatch at {path}{detail}"
                )
            result_numbers.add(number)

    missing_results = set(known_attempts) - result_numbers
    if not missing_results:
        return expected_records, None
    if missing_results != {known_attempts[-1]}:
        raise ValueError("resume unterminated attempt history is ambiguous")
    interrupted_number = known_attempts[-1]
    return expected_records, {
        "attempt": interrupted_number,
        "state": "interrupted",
        "run_spec_id": ctx.run_spec_id,
        "reason": _INTERRUPTED_REASON,
    }


def run_sweep(ctx: SweepContext) -> dict[str, Any]:
    """Run one durable attempt, preserving an auditable interrupted state."""
    spec_ids, bootstrap_plan_id = _context_identity(ctx)
    bound_attempt = _prepare_attempt(
        ctx,
        bootstrap_plan_id=bootstrap_plan_id,
    )
    cells = list(ctx.cells if ctx.cells is not None else full_grid())
    precomputed = None
    interrupted_result = None
    if ctx.resume:
        precomputed, interrupted_result = _resume_preflight(
            ctx,
            cells=cells,
            spec_ids=spec_ids,
            bootstrap_plan_id=bootstrap_plan_id,
        )
    if interrupted_result is not None:
        interrupted_number = int(interrupted_result["attempt"])
        _write_bound_json(
            ctx.output_dir / "attempt_results" / f"{interrupted_number}.json",
            interrupted_result,
            resume=False,
        )
        _commit(ctx)

    started = {**bound_attempt, "state": "started"}
    _append_attempt(ctx.output_dir / "attempts.jsonl", started)
    _commit(ctx)
    attempt_number = int(bound_attempt["attempt"])
    result_path = (
        ctx.output_dir / "attempt_results" / f"{attempt_number}.json"
    )
    try:
        aggregate = _run_sweep_body(
            ctx,
            spec_ids=spec_ids,
            bootstrap_plan_id=bootstrap_plan_id,
            precomputed_records=precomputed,
        )
    except BaseException as exc:
        _write_bound_json(
            result_path,
            {
                "attempt": attempt_number,
                "state": "failed",
                "run_spec_id": ctx.run_spec_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            resume=False,
        )
        _commit(ctx)
        raise

    _write_bound_json(
        result_path,
        {
            "attempt": attempt_number,
            "state": "completed",
            "run_spec_id": ctx.run_spec_id,
            "completed": aggregate["completed"],
            "failed": aggregate["failed"],
        },
        resume=False,
    )
    _commit(ctx)
    return aggregate
