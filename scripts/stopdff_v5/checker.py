"""Standalone checker (ACCEPTANCE_CONTRACT.md).

Independently recomputes cell/family/release statistics from the adapter rows and the
run package, never trusting serialized verdict fields. Also validates package structure:
backend-manifest exclusivity, attempt history, bootstrap plan, safe checksums,
external_artifacts, report semantics, and PNG validity.

No standalone validation requires another backend or comparison policy.
"""
from __future__ import annotations

import gzip
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bootstrap import build_bootstrap_plan, cell_bootstrap_stats, family_statistic
from .cellcompute import compute_cell
from .identity import compute_id, loads_no_duplicate_keys, sha256_file
from .profile import (
    EXPECTED_CELLS,
    FINAL_REPLICATES,
    SMOKE_REPLICATES,
    cell_key_str,
    full_grid,
    smoke_cells,
)
from .verdicts import (
    ceiling_flags,
    cell_verdict,
    coverage_clean,
    family_verdict,
    release_validity,
)

_FLOAT_TOL = 1e-9


@dataclass
class CheckResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    recomputed: dict[str, Any] = field(default_factory=dict)


def _err(errors: list[str], cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def load_json(path: Path) -> Any:
    return loads_no_duplicate_keys(Path(path).read_text(encoding="utf-8"))


def load_jsonl_gz(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_adapter_rows(bundle_dir: Path) -> list[dict]:
    rows = load_jsonl_gz(bundle_dir / "fit_rows.jsonl.gz")
    rows += load_jsonl_gz(bundle_dir / "eval_rows.jsonl.gz")
    return rows


# --- validate-spec ----------------------------------------------------------------


def validate_spec(spec_path: Path, *, require_final_profile: bool) -> CheckResult:
    errors: list[str] = []
    spec = load_json(spec_path)
    ident = spec.get("identity", spec)
    _err(errors, compute_id(ident) == spec.get("id", compute_id(ident)) if "id" in spec else True,
         "run-spec id does not match its identity")
    body = ident if "kind" in ident else spec
    _err(errors, body.get("profile_name") == "stopdff_bucketed_dp_paired_v2",
         f"unexpected profile_name {body.get('profile_name')!r}")
    # No unresolved template placeholders anywhere.
    raw = spec_path.read_text(encoding="utf-8")
    for placeholder in ("<SOURCE_ID>", "<RAW_INPUT_ID>", "<MODEL_SNAPSHOT_ID>", "<ADAPTER_ID>",
                        "<FVI_STUDY_ID>", "<BOOTSTRAP_PLAN_ID>", "<SELECTED_FROM_FVI_STUDY>",
                        "<ENVIRONMENT_CONTRACT_ID>"):
        _err(errors, placeholder not in raw, f"unresolved template placeholder {placeholder}")
    if require_final_profile:
        boot = body.get("bootstrap", {})
        _err(errors, int(boot.get("replicate_count", 0)) == FINAL_REPLICATES,
             "final profile must use 1000 bootstrap replicates")
        ids = body.get("identity", {})
        for key in ("source_manifest_id", "raw_input_bundle_id", "model_snapshot_id",
                    "adapter_bundle_id", "fvi_study_id", "bootstrap_plan_id"):
            _err(errors, bool(ids.get(key)), f"final spec missing {key}")
    return CheckResult(passed=not errors, errors=errors)


# --- validate-adapter -------------------------------------------------------------


def validate_adapter(bundle_dir: Path) -> CheckResult:
    errors: list[str] = []
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    _err(errors, manifest_path.exists(), "adapter manifest.json missing")
    if not manifest_path.exists():
        return CheckResult(passed=False, errors=errors)
    manifest = load_json(manifest_path)
    ident = manifest.get("identity", {})
    _err(errors, compute_id(ident) == manifest.get("id"), "adapter manifest id mismatch")
    for name in ("fit_rows.jsonl.gz", "eval_rows.jsonl.gz"):
        p = bundle_dir / name
        _err(errors, p.exists(), f"adapter bundle missing {name}")
    if (bundle_dir / "fit_rows.jsonl.gz").exists():
        _err(errors, sha256_file(bundle_dir / "fit_rows.jsonl.gz") == ident.get("fit_rows_sha256"),
             "adapter fit_rows sha mismatch")
    if (bundle_dir / "eval_rows.jsonl.gz").exists():
        _err(errors, sha256_file(bundle_dir / "eval_rows.jsonl.gz") == ident.get("eval_rows_sha256"),
             "adapter eval_rows sha mismatch")
    return CheckResult(passed=not errors, errors=errors)


# --- validate (run) ---------------------------------------------------------------


def _check_png(path: Path, errors: list[str]) -> None:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        errors.append(f"invalid PNG signature: {path.name}")
        return
    # IHDR chunk: length(4) 'IHDR'(4) width(4) height(4)
    if len(data) < 24 or data[12:16] != b"IHDR":
        errors.append(f"invalid PNG IHDR: {path.name}")
        return
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        errors.append(f"PNG has non-positive dimensions: {path.name}")


def _check_checksums(run_root: Path, errors: list[str]) -> None:
    sums_path = run_root / "SHA256SUMS"
    if not sums_path.exists():
        errors.append("missing SHA256SUMS")
        return
    listed: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            errors.append(f"malformed SHA256SUMS line: {line!r}")
            continue
        digest, name = parts[0], parts[1]
        if name.startswith("/") or ".." in Path(name).parts:
            errors.append(f"unsafe checksum path: {name!r}")
        if name in listed:
            errors.append(f"duplicate checksum entry: {name!r}")
        listed[name] = digest
    # Every listed file must exist, be a regular file (no symlink), and match.
    for name, digest in listed.items():
        p = run_root / name
        if p.is_symlink():
            errors.append(f"symlink in checksums: {name!r}")
            continue
        if not p.is_file():
            errors.append(f"checksum target missing: {name!r}")
            continue
        if sha256_file(p) != digest:
            errors.append(f"checksum mismatch: {name!r}")


def _check_attempts(run_root: Path, errors: list[str]) -> None:
    path = run_root / "attempts.jsonl"
    if not path.exists():
        return  # attempts optional for some flows; presence validated elsewhere
    attempts = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    last_num = 0
    for a in attempts:
        num = int(a.get("attempt", 0))
        mode = a.get("mode")
        cmd = a.get("command", [])
        if num <= last_num:
            errors.append("attempt numbers not monotonic")
        last_num = num
        if "--overwrite" in cmd:
            errors.append("--overwrite present in an evidence attempt")
        resume_count = list(cmd).count("--resume")
        if num == 1 and mode == "fresh":
            if resume_count != 0:
                errors.append("fresh attempt must omit --resume")
        elif mode == "resume":
            if resume_count != 1:
                errors.append("resume attempt must contain exactly one bare --resume")


def _check_reports(run_root: Path, errors: list[str]) -> None:
    md = run_root / "reports" / "report.md"
    tex = run_root / "reports" / "report.tex"
    required_md = [
        "profile", "paired", "reward", "calibrat", "continuation", "fvi",
        "family", "verdict", "override", "resource",
    ]
    if not md.exists():
        errors.append("missing reports/report.md")
    else:
        text = md.read_text(encoding="utf-8").lower()
        for token in required_md:
            if token not in text:
                errors.append(f"report.md missing required content: {token!r}")
    if not tex.exists():
        errors.append("missing reports/report.tex")
    figs = run_root / "figures"
    if figs.exists():
        for png in sorted(figs.glob("*.png")):
            _check_png(png, errors)


def validate_run(
    run_root: Path,
    *,
    backend: str,
    adapter_bundle: Path,
    require_final_profile: bool = False,
    require_package: bool = False,
) -> CheckResult:
    run_root = Path(run_root)
    adapter_bundle = Path(adapter_bundle)
    errors: list[str] = []

    aggregate = load_json(run_root / "aggregate.json")
    _err(errors, aggregate.get("backend") == backend,
         f"aggregate backend {aggregate.get('backend')!r} != requested {backend!r}")

    # Backend manifest exclusivity.
    run_manifest = run_root / "run_manifest.json"
    cmd_manifest = run_root / "command_manifest.json"
    if backend == "modal":
        _err(errors, run_manifest.exists(), "modal backend requires run_manifest.json")
        _err(errors, not cmd_manifest.exists(), "modal backend forbids command_manifest.json")
    else:
        _err(errors, cmd_manifest.exists(), "local backend requires command_manifest.json")
        _err(errors, not run_manifest.exists(), "local backend forbids run_manifest.json")

    # Run spec + bootstrap plan manifests (id recompute).
    spec_manifest = load_json(run_root / "run_spec.json")
    _err(errors, compute_id(spec_manifest["identity"]) == spec_manifest["id"], "run_spec id mismatch")
    plan_manifest = load_json(run_root / "bootstrap_plan.json")
    _err(errors, compute_id(plan_manifest["identity"]) == plan_manifest["id"], "bootstrap_plan id mismatch")

    # Rebuild bootstrap plan from (item_ids, seed, replicates) and verify hashes.
    plan_ident = plan_manifest["identity"]
    item_ids = plan_manifest["item_ids"]
    seed = int(plan_ident["seed"])
    replicates = int(plan_ident["replicate_count"])
    _err(errors, seed == 1, f"bootstrap seed must be 1 (got {seed})")
    variant = aggregate.get("profile_variant")
    expected_reps = FINAL_REPLICATES if variant == "final" else SMOKE_REPLICATES
    _err(errors, replicates == expected_reps,
         f"{variant} profile must use {expected_reps} replicates (got {replicates})")
    plan = build_bootstrap_plan(item_ids, replicates=replicates, seed=seed)
    _err(errors, plan.resample_index_sha256 == plan_ident["resample_index_sha256"],
         "bootstrap resample-index hash mismatch")
    _err(errors, plan.item_id_list_sha256 == plan_ident["evaluation_item_id_list_sha256"],
         "bootstrap item-id-list hash mismatch")

    # Cells to check.
    if require_final_profile:
        cells = full_grid()
        _err(errors, len(cells) == EXPECTED_CELLS, "final profile must have 96 cells")
    elif variant == "smoke":
        cells = smoke_cells()
    else:
        cells = full_grid()
    expected_keys = {cell_key_str(c) for c in cells}

    # Adapter rows + calibration for independent recomputation.
    rows = load_adapter_rows(adapter_bundle)
    cal_path = adapter_bundle / "calibration.json"
    calibration = load_json(cal_path) if cal_path.exists() else None

    fvi_sel = aggregate.get("fvi_selected", {})
    tol_label = str(fvi_sel.get("tolerance"))
    max_iter = int(fvi_sel.get("max_iterations"))
    gate_overrides = aggregate.get("gate_overrides", {})
    mc_overridden = bool(gate_overrides.get("allow_low_mc_retention") or
                         gate_overrides.get("allow_incomplete_mc_coverage"))

    abs_median_reps: dict[str, Any] = {}
    abs_median_point: dict[str, float] = {}
    recomputed_verdicts: dict[str, str] = {}
    completed: set[str] = set()
    failed: set[str] = set()

    for cell in cells:
        key = cell_key_str(cell)
        cell_path = run_root / "cells" / f"{key}.json"
        if not cell_path.exists():
            errors.append(f"missing cell file: {key}")
            failed.add(key)
            continue
        stored = load_json(cell_path)
        # fingerprint recompute
        if "fingerprint_identity" in stored:
            _err(errors, compute_id(stored["fingerprint_identity"]) == stored.get("fingerprint_id"),
                 f"{key}: fingerprint id mismatch")

        res = compute_cell(
            rows=rows, cell=cell, calibration_json=calibration,
            tolerance=float(tol_label), max_iterations=max_iter, tolerance_label=tol_label,
            metric_split="test",
        )
        if res.status != "completed":
            failed.add(key)
            _err(errors, stored.get("status") == res.status,
                 f"{key}: stored status {stored.get('status')!r} != recomputed {res.status!r}")
            # A non-completed cell must not be serialized as completed with a verdict.
            _err(errors, stored.get("status") != "completed",
                 f"{key}: non-converged/failed cell serialized as completed")
            continue

        completed.add(key)
        # index shifts must match exactly (no trusting serialized).
        stored_shifts = {str(k): int(v) for k, v in stored.get("index_shift_by_item", {}).items()}
        _err(errors, stored_shifts == res.index_shift_by_item,
             f"{key}: index_shift_by_item mismatch (cache stale or tampered)")

        stats = cell_bootstrap_stats(res.index_shift_by_item, plan)
        flags = ceiling_flags(res.mc_stops, res.qa_stops, list(res.index_shift_by_item.values()))
        ceiling_any = any(flags.values())
        cov_clean = coverage_clean(res.coverage["fallback_fraction"], res.coverage["missing_fraction"])
        verdict = cell_verdict(
            abs_median_ci=stats["abs_median_ci"], coverage_is_clean=cov_clean,
            ceiling_any=ceiling_any, mc_gate_overridden=mc_overridden,
        )
        recomputed_verdicts[key] = verdict
        _err(errors, stored.get("verdict") == verdict,
             f"{key}: stored verdict {stored.get('verdict')!r} != recomputed {verdict!r}")
        # coverage/ceiling serialized must match recomputed (no hiding a WARN).
        stored_cov = stored.get("coverage", {})
        _err(errors, bool(stored_cov.get("clean")) == cov_clean,
             f"{key}: coverage clean flag mismatch")
        _err(errors, stored.get("ceiling_flags") == flags, f"{key}: ceiling flags mismatch")
        # bootstrap CI recompute matches serialized within tolerance
        stored_ci = stored.get("bootstrap", {}).get("ci", {}).get("absolute_index_median", [None, None])
        _err(errors,
             abs(float(stored_ci[0]) - stats["abs_median_ci"][0]) < _FLOAT_TOL
             and abs(float(stored_ci[1]) - stats["abs_median_ci"][1]) < _FLOAT_TOL,
             f"{key}: abs-median CI mismatch")

        abs_median_reps[key] = stats["abs_median_replicates"]
        abs_median_point[key] = stats["abs_median_point"]

    # Family recompute.
    family_valid = bool(abs_median_reps)
    if family_valid:
        fam = family_statistic(abs_median_reps, abs_median_point)
        all_cells_pass = (completed == expected_keys) and all(
            recomputed_verdicts.get(k) == "PASS" for k in expected_keys
        )
        fam_verdict = family_verdict(
            family_ci=fam["ci"], all_cells_pass=all_cells_pass, mc_override_active=mc_overridden
        )
        stored_family = aggregate.get("family") or {}
        _err(errors, stored_family.get("verdict") == fam_verdict,
             f"family verdict mismatch: stored {stored_family.get('verdict')!r} != recomputed {fam_verdict!r}")
        if "ci" in stored_family:
            _err(errors,
                 abs(float(stored_family["ci"][0]) - fam["ci"][0]) < _FLOAT_TOL
                 and abs(float(stored_family["ci"][1]) - fam["ci"][1]) < _FLOAT_TOL,
                 "family CI mismatch")

    # Counts.
    _err(errors, aggregate.get("requested") == len(expected_keys), "requested count mismatch")
    _err(errors, aggregate.get("completed") == len(completed), "completed count mismatch")
    _err(errors, aggregate.get("failed") == len(failed), "failed count mismatch")
    _err(errors, int(aggregate.get("skipped", 0)) == 0, "skipped must be 0")

    # Release validity recompute.
    release = release_validity(
        expected_cell_keys=expected_keys, present_cell_keys=sorted(completed | failed),
        completed_keys=completed, failed_keys=failed, skipped_keys=set(),
        all_calibrators_fitted=not any("calibrator_failed" == load_json(run_root / "cells" / f"{k}.json").get("status") for k in failed),
        all_fvi_converged=not any("fvi_failed" == load_json(run_root / "cells" / f"{k}.json").get("status") for k in failed),
        manifests_valid=True, cache_matches_aggregate=True,
        bootstrap_valid=family_valid, family_valid=family_valid,
        backend_manifest_valid=True, attempt_history_valid=True,
    )
    recomputed_status = "VALID" if release.valid else "INVALID"
    _err(errors, aggregate.get("release_status") == recomputed_status,
         f"release_status mismatch: stored {aggregate.get('release_status')!r} != recomputed {recomputed_status!r}")

    _check_attempts(run_root, errors)
    if require_package:
        _check_checksums(run_root, errors)
        _err(errors, (run_root / "external_artifacts.json").exists(), "missing external_artifacts.json")
        _check_reports(run_root, errors)

    return CheckResult(
        passed=not errors, errors=errors,
        recomputed={"release_status": recomputed_status,
                    "family": aggregate.get("family"),
                    "completed": len(completed), "failed": len(failed)},
    )
