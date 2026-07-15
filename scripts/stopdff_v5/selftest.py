"""Negative mutation suite + valid-package builder (ACCEPTANCE_CONTRACT.md section 6).

build_valid_package() creates a self-contained synthetic package (adapter bundle + run
package) that the checker accepts. run_self_test() applies a battery of mutations and
asserts the checker REJECTS every one. Synthetic fixtures only; real-data mutation gate
runs the same logic on Modal against the real package.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import checker
from .bootstrap import build_bootstrap_plan, plan_identity
from .identity import build_manifest, compute_id, sha256_file
from .manifests import adapter_identity, run_spec_identity
from .profile import cell_key_str, smoke_cells
from .rowio import write_jsonl_gz
from .sweep import SweepContext, run_sweep
from .writers import package_run

CATEGORIES = ["history", "science", "arts"]
PREFIX_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def _synth_rows(n_items: int = 44, seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_items):
        qid = f"q{i:03d}"
        split = "val" if i < n_items // 2 else "test"
        cat = CATEGORIES[i % len(CATEGORIES)]
        off = rng.uniform(-0.15, 0.15)
        for t, frac in enumerate(PREFIX_FRACS):
            mc = float(np.clip(0.25 + 0.55 * frac + off + rng.uniform(-0.05, 0.05), 0.0, 1.0))
            qa = float(np.clip(0.20 + 0.60 * frac + off + rng.uniform(-0.05, 0.05), 0.0, 1.0))
            correct = int(mc + rng.uniform(-0.15, 0.15) > 0.55)
            base = {"item_id": qid, "prefix_idx": t, "prefix_fraction": frac, "split": split,
                    "category": cat, "K": 4, "option_set_id": f"{qid}:K4",
                    "distractor_strategy": "unknown"}
            rows.append({**base, "format": "MC", "raw_similarity": mc, "correct": correct,
                         "p_second_best": max(0.0, mc - 0.1), "top2_margin": 0.1})
            rows.append({**base, "format": "QA", "raw_similarity": qa, "correct": 1,
                         "p_second_best": 0.0, "top2_margin": 0.0})
    return rows


def _calibration_json() -> dict:
    block = {"platt_coef": 5.0, "platt_intercept": -2.5}
    return {"per_bucket": {"early": dict(block), "mid": dict(block), "late": dict(block)},
            "fit_split": "val"}


def _hex(n: str) -> str:
    return (n * 64)[:64]


def build_valid_package(base_dir: Path) -> dict[str, Any]:
    base_dir = Path(base_dir)
    bundle = base_dir / "adapter_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    rows = _synth_rows()
    fit_rows = [r for r in rows if r["split"] == "val"]
    eval_rows = [r for r in rows if r["split"] == "test"]
    write_jsonl_gz(bundle / "fit_rows.jsonl.gz", fit_rows)
    write_jsonl_gz(bundle / "eval_rows.jsonl.gz", eval_rows)
    (bundle / "calibration.json").write_text(json.dumps(_calibration_json(), sort_keys=True), encoding="utf-8")
    fit_sha = sha256_file(bundle / "fit_rows.jsonl.gz")
    eval_sha = sha256_file(bundle / "eval_rows.jsonl.gz")

    source_id, raw_id, model_id = _hex("1"), _hex("2"), _hex("3")
    adapter_ident = adapter_identity(
        source_manifest_id=source_id, raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        scoring_spec={"model": "sentence-transformers/all-MiniLM-L6-v2"}, fit_split="val", eval_split="test",
        schema_columns=list(rows[0].keys()), fit_row_count=len(fit_rows), eval_row_count=len(eval_rows),
        fit_rows_sha256=fit_sha, eval_rows_sha256=eval_sha,
        mc_coverage={"paired": True}, mc_retention={"ok": True}, producer_hashes={"build": _hex("4")},
    )
    adapter_man = build_manifest(adapter_ident)
    (bundle / "manifest.json").write_text(json.dumps(adapter_man, indent=2, sort_keys=True), encoding="utf-8")
    adapter_id = adapter_man["id"]

    test_items = sorted({r["item_id"] for r in eval_rows if r["format"] == "MC"} &
                        {r["item_id"] for r in eval_rows if r["format"] == "QA"})
    plan = build_bootstrap_plan(test_items, replicates=100, seed=1)
    bootstrap_id = compute_id(plan_identity(plan))

    fvi_study_id, env_id = _hex("5"), _hex("6")
    run_spec = run_spec_identity(
        source_manifest_id=source_id, raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        adapter_bundle_id=adapter_id, fvi_study_id=fvi_study_id, bootstrap_plan_id=bootstrap_id,
        environment_contract_id=env_id, fvi_selected={"tolerance": "1e-8", "max_iterations": 100},
        replicate_count=100, profile_variant="smoke",
    )
    run_spec_id = compute_id(run_spec)

    run_root = base_dir / "run"
    ctx = SweepContext(
        rows=rows, calibration_json=_calibration_json(), run_spec=run_spec, run_spec_id=run_spec_id,
        bootstrap_plan=plan, output_dir=run_root, fvi_tolerance="1e-8", fvi_max_iterations=100,
        backend="modal", profile_variant="smoke", adapter_fit_rows_sha256=fit_sha,
        adapter_eval_rows_sha256=eval_sha, myopic_artifact_sha256=_hex("7"),
        producer_hashes={"sweep": _hex("8")}, cells=smoke_cells(),
        environment={"python_version": "3.11"}, resource_summary={"backend": "modal", "usd": 0},
        attempt={"attempt": 1, "mode": "fresh", "command": ["dp_sweep"], "run_spec_id": run_spec_id,
                 "adapter_id": adapter_id, "completed": 2, "skipped": 0, "failed": 0},
    )
    aggregate = run_sweep(ctx)
    package_run(
        run_root, aggregate, resource_summary={"backend": "modal", "usd": 0},
        external_artifacts=[{"role": "adapter_fit_rows", "content_id": fit_sha, "sha256": fit_sha,
                             "byte_size": (bundle / "fit_rows.jsonl.gz").stat().st_size,
                             "retrieval": "/stopdff/adapters/<id>/fit_rows.jsonl.gz",
                             "retention": "keep"}],
    )
    return {"run_root": run_root, "adapter_bundle": bundle, "aggregate": aggregate,
            "run_spec_id": run_spec_id}


# --- mutations --------------------------------------------------------------------


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _save(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _first_cell(run_root: Path) -> Path:
    return sorted((run_root / "cells").glob("*.json"))[0]


def _mut_stale_cache(rr, b):
    p = _first_cell(rr); rec = _load(p)
    k = next(iter(rec["index_shift_by_item"]))
    rec["index_shift_by_item"][k] = int(rec["index_shift_by_item"][k]) + 3
    _save(p, rec)


def _mut_flip_verdict(rr, b):
    p = _first_cell(rr); rec = _load(p)
    flip = {"PASS": "FAIL", "WARN": "PASS", "FAIL": "PASS"}
    rec["verdict"] = flip.get(rec.get("verdict"), "PASS")
    _save(p, rec)


def _mut_hide_coverage(rr, b):
    p = _first_cell(rr); rec = _load(p)
    rec["coverage"]["clean"] = not rec["coverage"]["clean"]
    _save(p, rec)


def _mut_tamper_ceiling(rr, b):
    p = _first_cell(rr); rec = _load(p)
    key = next(iter(rec["ceiling_flags"]))
    rec["ceiling_flags"][key] = not rec["ceiling_flags"][key]
    _save(p, rec)


def _mut_wrong_family_ci(rr, b):
    agg = _load(rr / "aggregate.json"); agg["family"]["ci"] = [99.0, 100.0]; _save(rr / "aggregate.json", agg)


def _mut_wrong_family_verdict(rr, b):
    agg = _load(rr / "aggregate.json")
    agg["family"]["verdict"] = "FAIL" if agg["family"]["verdict"] != "FAIL" else "PASS"
    _save(rr / "aggregate.json", agg)


def _mut_wrong_release(rr, b):
    agg = _load(rr / "aggregate.json")
    agg["release_status"] = "INVALID" if agg["release_status"] == "VALID" else "VALID"
    _save(rr / "aggregate.json", agg)


def _mut_dual_backend(rr, b):
    shutil.copy(rr / "run_manifest.json", rr / "command_manifest.json")


def _mut_missing_backend_manifest(rr, b):
    (rr / "run_manifest.json").unlink()


def _mut_wrong_seed(rr, b):
    pm = _load(rr / "bootstrap_plan.json"); pm["identity"]["seed"] = 2
    pm["id"] = compute_id(pm["identity"]); _save(rr / "bootstrap_plan.json", pm)


def _mut_wrong_replicates(rr, b):
    pm = _load(rr / "bootstrap_plan.json"); pm["identity"]["replicate_count"] = 999
    pm["id"] = compute_id(pm["identity"]); _save(rr / "bootstrap_plan.json", pm)


def _mut_tamper_run_spec_id(rr, b):
    sm = _load(rr / "run_spec.json"); sm["id"] = _hex("0"); _save(rr / "run_spec.json", sm)


def _mut_tamper_plan_hash(rr, b):
    pm = _load(rr / "bootstrap_plan.json")
    pm["identity"]["resample_index_sha256"] = _hex("0")
    pm["id"] = compute_id(pm["identity"]); _save(rr / "bootstrap_plan.json", pm)


def _mut_fresh_with_resume(rr, b):
    (rr / "attempts.jsonl").write_text(
        json.dumps({"attempt": 1, "mode": "fresh", "command": ["dp_sweep", "--resume"]}) + "\n",
        encoding="utf-8")


def _mut_resume_without_bare(rr, b):
    with open(rr / "attempts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": 2, "mode": "resume", "command": ["dp_sweep"]}) + "\n")


def _mut_overwrite(rr, b):
    with open(rr / "attempts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": 2, "mode": "resume", "command": ["dp_sweep", "--resume", "--overwrite"]}) + "\n")


def _mut_unsafe_checksum(rr, b):
    with open(rr / "SHA256SUMS", "a", encoding="utf-8") as f:
        f.write(f"{_hex('a')}  ../evil.txt\n")


def _mut_duplicate_checksum(rr, b):
    lines = (rr / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    lines.append(lines[0])
    (rr / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_symlink(rr, b):
    target = rr / "aggregate.json"
    link = rr / "evil_link.json"
    os.symlink(target, link)
    with open(rr / "SHA256SUMS", "a", encoding="utf-8") as f:
        f.write(f"{sha256_file(target)}  evil_link.json\n")


def _mut_checksum_value(rr, b):
    lines = (rr / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    parts = lines[0].split()
    lines[0] = f"{_hex('0')}  {parts[1]}"
    (rr / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_invalid_png(rr, b):
    for png in (rr / "figures").glob("*.png"):
        png.write_bytes(b"not a png")
        break


def _mut_missing_external_artifacts(rr, b):
    (rr / "external_artifacts.json").unlink()


def _mut_unconverged_completed(rr, b):
    agg = _load(rr / "aggregate.json"); agg["fvi_selected"]["max_iterations"] = 1
    _save(rr / "aggregate.json", agg)


def _mut_fingerprint(rr, b):
    p = _first_cell(rr); rec = _load(p); rec["fingerprint_id"] = _hex("0"); _save(p, rec)


def _mut_adapter_row_hash(rr, b):
    # corrupt fit rows after manifest was written -> validate-adapter must reject
    with open(b / "fit_rows.jsonl.gz", "ab") as f:
        f.write(b"\x00corrupt")


_RUN_MUTATIONS: dict[str, Callable] = {
    "stale_cache": _mut_stale_cache,
    "cell_verdict_serialized_not_trusted": _mut_flip_verdict,
    "coverage_clean_serialized_not_trusted": _mut_hide_coverage,
    "ceiling_flags_tampered": _mut_tamper_ceiling,
    "wrong_family_maximum_statistic": _mut_wrong_family_ci,
    "family_verdict_hides_cell_warn": _mut_wrong_family_verdict,
    "wrong_release_status": _mut_wrong_release,
    "dual_backend_manifests": _mut_dual_backend,
    "missing_backend_manifest": _mut_missing_backend_manifest,
    "wrong_bootstrap_seed": _mut_wrong_seed,
    "wrong_bootstrap_count": _mut_wrong_replicates,
    "tampered_run_spec_id": _mut_tamper_run_spec_id,
    "wrong_bootstrap_plan_hash": _mut_tamper_plan_hash,
    "fresh_attempt_with_resume": _mut_fresh_with_resume,
    "resume_without_bare_resume": _mut_resume_without_bare,
    "overwrite_in_evidence_run": _mut_overwrite,
    "unsafe_checksum_traversal": _mut_unsafe_checksum,
    "duplicate_checksum_entry": _mut_duplicate_checksum,
    "symlink_in_package": _mut_symlink,
    "checksum_value_mismatch": _mut_checksum_value,
    "invalid_png": _mut_invalid_png,
    "missing_external_artifacts": _mut_missing_external_artifacts,
    "unconverged_fvi_marked_completed": _mut_unconverged_completed,
    "cell_fingerprint_tampered": _mut_fingerprint,
}


def run_self_test(base_dir: Path) -> tuple[bool, list[dict[str, Any]]]:
    base_dir = Path(base_dir)
    valid_dir = base_dir / "valid"
    built = build_valid_package(valid_dir)

    results: list[dict[str, Any]] = []
    baseline = checker.validate_run(
        built["run_root"], backend="modal", adapter_bundle=built["adapter_bundle"],
        require_final_profile=False, require_package=True,
    )
    results.append({"mutation": "<baseline valid>", "expected": "PASS",
                    "passed_check": baseline.passed, "ok": baseline.passed,
                    "errors": baseline.errors[:3]})

    all_ok = baseline.passed

    for name, fn in _RUN_MUTATIONS.items():
        mdir = base_dir / f"mut_{name}"
        if mdir.exists():
            shutil.rmtree(mdir)
        shutil.copytree(valid_dir, mdir, symlinks=True)
        rr, bundle = mdir / "run", mdir / "adapter_bundle"
        fn(rr, bundle)
        res = checker.validate_run(
            rr, backend="modal", adapter_bundle=bundle,
            require_final_profile=False, require_package=True,
        )
        rejected = not res.passed
        results.append({"mutation": name, "expected": "REJECT", "passed_check": res.passed,
                        "ok": rejected, "errors": res.errors[:2]})
        all_ok = all_ok and rejected

    # adapter-level mutation
    adir = base_dir / "mut_adapter_row_hash"
    if adir.exists():
        shutil.rmtree(adir)
    shutil.copytree(valid_dir, adir, symlinks=True)
    _mut_adapter_row_hash(adir / "run", adir / "adapter_bundle")
    ares = checker.validate_adapter(adir / "adapter_bundle")
    results.append({"mutation": "invalid_adapter_row_hash", "expected": "REJECT",
                    "passed_check": ares.passed, "ok": not ares.passed, "errors": ares.errors[:2]})
    all_ok = all_ok and not ares.passed

    return all_ok, results
