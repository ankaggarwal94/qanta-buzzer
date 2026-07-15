#!/usr/bin/env python3
"""Local (CPU) end-to-end reproduction of the StopDFF v5 bucketed-DP paired audit.

Runs the same pipeline as the Modal backend, in-process, without Modal:
  stage raw inputs -> source snapshot -> model snapshot -> adapter bundle ->
  FVI study (or fixed params) -> bootstrap plan -> sweep (backend=local) ->
  package -> standalone validation (backend=local).

Requires the nine raw inputs (see docs/stopdff_v5/REPRODUCTION.md). GPU is not required;
all-MiniLM-L6-v2 runs on CPU (slower). See --help for options.
"""
from __future__ import annotations

import argparse
import importlib.metadata as im
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.stopdff_v5 import producers, adapter_build, fvi_study, profile, sweep, writers, checker  # noqa: E402
from scripts.stopdff_v5.bootstrap import build_bootstrap_plan  # noqa: E402
from scripts.stopdff_v5.identity import build_manifest, compute_id, sha256_file  # noqa: E402
from scripts.stopdff_v5.manifests import (  # noqa: E402
    environment_contract_identity, fvi_study_identity, run_spec_identity,
)

_PKGS = ["numpy", "scipy", "scikit-learn", "pandas", "sentence-transformers",
         "transformers", "huggingface_hub"]


def _versions() -> dict[str, str]:
    out = {}
    for name in _PKGS:
        try:
            out[name] = im.version(name)
        except im.PackageNotFoundError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=_REPO / "data" / "processed")
    ap.add_argument("--paper-exports", type=Path, default=_REPO / "paper_exports")
    ap.add_argument("--repo-root", type=Path, default=_REPO)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=["smoke", "final"], default="smoke")
    ap.add_argument("--skip-fvi-study", action="store_true",
                    help="use --fvi-tolerance/--fvi-max-iterations directly (fast; skips the selector)")
    ap.add_argument("--fvi-tolerance", default="1e-6")
    ap.add_argument("--fvi-max-iterations", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_sha = subprocess.run(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()

    print("== source snapshot ==")
    src_man = producers.build_source_snapshot(args.repo_root, run_sha, out / "source_snapshot")
    source_id = src_man["id"]
    print("  source_manifest_id", source_id)

    print("== stage raw inputs ==")
    roles = {
        "mc_dataset.json": args.data_dir / "mc_dataset.json",
        "val_dataset.json": args.data_dir / "val_dataset.json",
        "test_dataset.json": args.data_dir / "test_dataset.json",
        "build_metadata.json": args.data_dir / "build_metadata.json",
        "split_metadata.json": args.data_dir / "split_metadata.json",
        "calibration.json": args.paper_exports / "calibration.json",
        "stopdff.json": args.paper_exports / "stopdff.json",
        "threshold_manifest.json": args.repo_root / "threshold_manifest.json",
        "threshold_manifest.json.sha256": args.repo_root / "threshold_manifest.json.sha256",
    }
    raw_man = producers.stage_raw_inputs(roles, out / "raw_inputs")
    raw_id = raw_man["id"]
    myopic_sha = next(f["sha256"] for f in raw_man["identity"]["files"] if f["role"] == "stopdff.json")
    raw_dir = out / "raw_inputs" / "raw"
    print("  raw_input_bundle_id", raw_id)

    print("== model snapshot (all-MiniLM-L6-v2, pinned revision) ==")
    model_man = adapter_build.freeze_model_snapshot(out / "model")
    model_id = model_man["id"]
    print("  model_snapshot_id", model_id, "rev", model_man["identity"]["model_revision"])

    print("== adapter bundle (CPU scoring) ==")
    adapter_dir = out / "adapter_bundle"
    adapter_man = adapter_build.build_adapter_bundle(
        mc_dataset_path=raw_dir / "mc_dataset.json", val_dataset_path=raw_dir / "val_dataset.json",
        test_dataset_path=raw_dir / "test_dataset.json", calibration_path=raw_dir / "calibration.json",
        model_snapshot_dir=out / "model" / "snapshot", out_dir=adapter_dir,
        source_manifest_id=source_id, raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        producer_hashes={"adapter_build.py": sha256_file(_REPO / "scripts/stopdff_v5/adapter_build.py")})
    adapter_id = adapter_man["id"]
    fit_sha = adapter_man["identity"]["fit_rows_sha256"]
    eval_sha = adapter_man["identity"]["eval_rows_sha256"]
    print("  adapter_bundle_id", adapter_id)

    rows = checker.load_adapter_rows(adapter_dir)
    calibration = json.loads((adapter_dir / "calibration.json").read_text())

    if args.skip_fvi_study:
        selected = {"tolerance": args.fvi_tolerance, "max_iterations": args.fvi_max_iterations}
        fvi_id = compute_id({"kind": "fvi_study_fixed", "adapter_bundle_id": adapter_id, "selected": selected})
        print("== FVI: fixed params (study skipped) ==", selected)
    else:
        print("== FVI candidate study + selector (slow on CPU) ==")
        study = fvi_study.run_fvi_study(rows=rows, calibration_json=calibration)
        selected = study["selected_parameters"]
        if selected is None:
            print("FVI selector found no eligible candidate", file=sys.stderr)
            return 1
        fvi_id = compute_id(fvi_study_identity(
            adapter_bundle_id=adapter_id, candidate_grid=study["candidate_grid"],
            representative_generator=study["representative_cell_generator"],
            candidate_results=study["candidate_convergence_results"],
            strict_reference_results=study["strict_reference"], selector_rule=study["selector_rule"],
            selected_parameters=selected, all96_validation=study["all96_fit_only_validation"],
            producer_hashes={}))
        print("  selected", selected, "fvi_study_id", fvi_id)

    env_id = compute_id(environment_contract_identity(
        python_version="%d.%d.%d" % sys.version_info[:3], package_versions=_versions()))

    replicates = 1000 if args.variant == "final" else 100
    mc = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "MC"}
    qa = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "QA"}
    plan = build_bootstrap_plan(sorted(mc & qa), replicates=replicates, seed=1)
    print("== bootstrap plan ==", "reps", replicates, "items", plan.n_items)

    run_spec = run_spec_identity(
        source_manifest_id=source_id, raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        adapter_bundle_id=adapter_id, fvi_study_id=fvi_id, bootstrap_plan_id=compute_id(_plan_ident(plan)),
        environment_contract_id=env_id, fvi_selected=selected, replicate_count=replicates,
        profile_variant=args.variant)
    run_spec_id = compute_id(run_spec)
    run_id = f"{args.variant}_local_{run_spec_id[:12]}"
    run_root = out / "runs" / run_id

    print("== sweep (backend=local) ==")
    ctx = sweep.SweepContext(
        rows=rows, calibration_json=calibration, run_spec=run_spec, run_spec_id=run_spec_id,
        bootstrap_plan=plan, output_dir=run_root, fvi_tolerance=str(selected["tolerance"]),
        fvi_max_iterations=int(selected["max_iterations"]), backend="local", profile_variant=args.variant,
        adapter_fit_rows_sha256=fit_sha, adapter_eval_rows_sha256=eval_sha, myopic_artifact_sha256=myopic_sha,
        producer_hashes={"sweep.py": sha256_file(_REPO / "scripts/stopdff_v5/sweep.py")},
        cells=(profile.full_grid() if args.variant == "final" else profile.smoke_cells()),
        environment=_versions(), resource_summary={"backend": "local"},
        attempt={"attempt": 1, "mode": "fresh", "command": ["run_stopdff_v5_local"], "run_spec_id": run_spec_id})
    agg = sweep.run_sweep(ctx)
    writers.package_run(run_root, agg, resource_summary={"backend": "local"}, external_artifacts=[])

    print("== validate (backend=local) ==")
    res = checker.validate_run(run_root, backend="local", adapter_bundle=adapter_dir,
                              require_final_profile=(args.variant == "final"), require_package=True)
    print(json.dumps({"release_status": agg["release_status"], "requested": agg["requested"],
                      "completed": agg["completed"], "failed": agg["failed"],
                      "family": agg.get("family"), "checker_passed": res.passed,
                      "checker_errors": res.errors[:10]}, indent=2))
    if not (res.passed and agg["release_status"] == "VALID"):
        return 1
    print(f"\nLOCAL REPRODUCTION OK -> {run_root}")
    return 0


def _plan_ident(plan):
    from scripts.stopdff_v5.bootstrap import plan_identity
    return plan_identity(plan)


if __name__ == "__main__":
    raise SystemExit(main())
