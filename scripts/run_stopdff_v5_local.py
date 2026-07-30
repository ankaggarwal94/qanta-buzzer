#!/usr/bin/env python3
"""Local (CPU) end-to-end reproduction of the StopDFF v5 bucketed-DP paired audit.

Runs the same pipeline as the Modal backend, in-process, without Modal:
  stage raw inputs -> source snapshot -> model snapshot -> adapter bundle ->
  FVI study (or fixed params) -> bootstrap plan -> sweep (backend=local) ->
  package -> standalone validation (backend=local).

Requires the ten raw inputs (see docs/stopdff_v5/REPRODUCTION.md). GPU is not required;
all-MiniLM-L6-v2 runs on CPU (slower). See --help for options.
"""
from __future__ import annotations

import argparse
import importlib.metadata as im
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.stopdff_v5 import (  # noqa: E402
    adapter_build,
    checker,
    fvi_study,
    producers,
    profile,
    selftest,
    sweep,
    writers,
)
from scripts.stopdff_v5.bootstrap import build_bootstrap_plan  # noqa: E402
from scripts.stopdff_v5.identity import (  # noqa: E402
    build_manifest,
    canonical_bytes,
    compute_id,
    sha256_bytes,
    sha256_file,
)
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


def _run_bound_sweep(
    *,
    adapter_dir: Path,
    run_spec: dict,
    plan,
    run_root: Path,
    myopic_sha256: str,
    producer_hashes: dict[str, str],
    environment: dict[str, str],
    cells: list[dict[str, str]],
    command: list[str],
) -> tuple[dict, "checker.CheckResult"]:
    """Resolve manifests before writing, run the sweep, and verify its output."""
    from scripts.stopdff_v5.bootstrap import plan_identity

    run_spec_manifest = {
        "id": compute_id(run_spec),
        "identity": run_spec,
    }
    bootstrap_identity = plan_identity(plan)
    bootstrap_manifest = {
        "id": compute_id(bootstrap_identity),
        "identity": bootstrap_identity,
        "item_ids": plan.item_ids,
    }
    binding = checker.resolve_run_binding(
        run_spec_manifest=run_spec_manifest,
        adapter_bundle=adapter_dir,
        bootstrap_plan_manifest=bootstrap_manifest,
    )
    ctx = sweep.SweepContext(
        rows=binding["rows"],
        calibration_json=binding["calibration"],
        run_spec=binding["run_spec_identity"],
        run_spec_id=binding["run_spec_id"],
        bootstrap_plan=binding["bootstrap_plan"],
        output_dir=run_root,
        fvi_tolerance=binding["fvi_tolerance"],
        fvi_max_iterations=binding["fvi_max_iterations"],
        backend="local",
        profile_variant=binding["variant"],
        adapter_bundle_id=binding["adapter_bundle_id"],
        adapter_fit_rows_sha256=binding["fit_rows_sha256"],
        adapter_eval_rows_sha256=binding["eval_rows_sha256"],
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=producer_hashes,
        cells=cells,
        environment=environment,
        resource_summary={"backend": "local"},
        attempt={
            "attempt": 1,
            "mode": "fresh",
            "command": command,
            "run_spec_id": binding["run_spec_id"],
            "adapter_id": binding["adapter_bundle_id"],
            "bootstrap_plan_id": binding["bootstrap_plan_id"],
        },
    )
    aggregate = sweep.run_sweep(ctx)
    result = checker.validate_run(
        run_root,
        backend="local",
        adapter_bundle=adapter_dir,
        require_final_profile=binding["variant"] == "final",
        require_package=False,
    )
    return aggregate, result


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

    if args.repo_root.resolve() != _REPO.resolve():
        raise ValueError(
            "--repo-root must be the repository whose code is executing"
        )
    if args.variant == "final" and args.skip_fvi_study:
        raise ValueError("final runs cannot skip the FVI selection study")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(args.repo_root),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("local reproduction requires a clean source worktree")

    run_sha = subprocess.run(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="stopdff_v5_selftest_") as work:
        mutation_ok, mutation_results = selftest.run_self_test(Path(work))
    if not mutation_ok:
        failed = [
            result["mutation"]
            for result in mutation_results
            if not result["ok"]
        ]
        raise ValueError(f"v5 mutation gate failed: {failed}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)

    print("== source snapshot ==")
    src_man = producers.build_source_snapshot(args.repo_root, run_sha, out / "source_snapshot")
    source_id = src_man["id"]
    print("  source_manifest_id", source_id)

    print("== stage raw inputs ==")
    roles = {
        "mc_dataset.json": args.data_dir / "mc_dataset.json",
        "train_dataset.json": args.data_dir / "train_dataset.json",
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
    adapter_result = checker.validate_adapter(adapter_dir)
    if (
        not adapter_result.passed
        or adapter_result.recomputed.get("adapter_bundle_id") != adapter_id
    ):
        raise ValueError(
            "new adapter failed validation: "
            + "; ".join(adapter_result.errors)
        )
    print("  adapter_bundle_id", adapter_id)

    rows = checker.load_adapter_rows(adapter_dir)
    calibration = json.loads((adapter_dir / "calibration.json").read_text())

    if args.skip_fvi_study:
        selected = {"tolerance": args.fvi_tolerance, "max_iterations": args.fvi_max_iterations}
        fvi_identity = {
            "kind": "fvi_study_fixed",
            "adapter_bundle_id": adapter_id,
            "selected": selected,
        }
        fvi_manifest = build_manifest(fvi_identity)
        fvi_id = fvi_manifest["id"]
        print("== FVI: fixed params (study skipped) ==", selected)
    else:
        print("== FVI candidate study + selector (slow on CPU) ==")
        study = fvi_study.run_fvi_study(rows=rows, calibration_json=calibration)
        selected = study["selected_parameters"]
        if selected is None:
            print("FVI selector found no eligible candidate", file=sys.stderr)
            return 1
        fvi_identity = fvi_study_identity(
            adapter_bundle_id=adapter_id, candidate_grid=study["candidate_grid"],
            representative_generator=study["representative_cell_generator"],
            candidate_results=study["candidate_convergence_results"],
            strict_reference_results=study["strict_reference"], selector_rule=study["selector_rule"],
            selected_parameters=selected, all96_validation=study["all96_fit_only_validation"],
            producer_hashes={})
        fvi_manifest = build_manifest(fvi_identity)
        fvi_id = fvi_manifest["id"]
        print("  selected", selected, "fvi_study_id", fvi_id)

    versions = _versions()
    environment_identity = environment_contract_identity(
        python_version="%d.%d.%d" % sys.version_info[:3],
        package_versions=versions,
    )
    environment_manifest = build_manifest(environment_identity)
    env_id = environment_manifest["id"]
    environment_record = {
        "python_version": "%d.%d.%d" % sys.version_info[:3],
        "package_versions": versions,
    }

    mc = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "MC"}
    qa = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "QA"}
    paired_items = sorted(mc & qa)
    producer_hashes = {
        "sweep.py": sha256_file(_REPO / "scripts/stopdff_v5/sweep.py"),
        "checker.py": sha256_file(_REPO / "scripts/stopdff_v5/checker.py"),
    }
    common_bindings = {
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
        "adapter_bundle_id": adapter_id,
        "fvi_study_id": fvi_id,
        "environment_contract_id": env_id,
    }
    receipt_ids: dict[str, str] = {}

    def persist_receipt(gate: str, receipt: dict) -> None:
        path = out / "receipts" / gate / f"{receipt['id']}.json"
        sweep._write_bound_json(path, receipt, resume=True)
        receipt_ids[gate] = receipt["id"]

    if args.variant == "final":
        print("== required deterministic two-build adapter gate ==")
        second_adapter_dir = out / "adapter_bundle_determinism"
        second_adapter = adapter_build.build_adapter_bundle(
            mc_dataset_path=raw_dir / "mc_dataset.json",
            val_dataset_path=raw_dir / "val_dataset.json",
            test_dataset_path=raw_dir / "test_dataset.json",
            calibration_path=raw_dir / "calibration.json",
            model_snapshot_dir=out / "model" / "snapshot",
            out_dir=second_adapter_dir,
            source_manifest_id=source_id,
            raw_input_bundle_id=raw_id,
            model_snapshot_id=model_id,
            producer_hashes={
                "adapter_build.py": sha256_file(
                    _REPO / "scripts/stopdff_v5/adapter_build.py"
                )
            },
        )
        compared = ("fit_rows.jsonl.gz", "eval_rows.jsonl.gz", "calibration.json")
        first_hashes = {
            name: sha256_file(adapter_dir / name)
            for name in compared
        }
        second_hashes = {
            name: sha256_file(second_adapter_dir / name)
            for name in compared
        }
        if second_adapter["id"] != adapter_id or second_hashes != first_hashes:
            raise ValueError("two-build adapter determinism gate failed")
        persist_receipt(
            "determinism",
            writers.build_prerequisite_receipt(
                gate="determinism",
                bindings={
                    key: common_bindings[key]
                    for key in (
                        "source_manifest_id",
                        "raw_input_bundle_id",
                        "model_snapshot_id",
                        "adapter_bundle_id",
                    )
                },
                evidence={
                    "bundle_files_sha256": sha256_bytes(
                        canonical_bytes(first_hashes)
                    )
                },
            ),
        )
        persist_receipt(
            "mutation",
            writers.build_prerequisite_receipt(
                gate="mutation",
                bindings=common_bindings,
                evidence={
                    "result_sha256": sha256_bytes(
                        canonical_bytes(mutation_results)
                    )
                },
            ),
        )
        print("== required bounded smoke before final sweep ==")
        smoke_plan = build_bootstrap_plan(
            paired_items,
            replicates=100,
            seed=1,
        )
        smoke_spec = run_spec_identity(
            source_manifest_id=source_id,
            raw_input_bundle_id=raw_id,
            model_snapshot_id=model_id,
            adapter_bundle_id=adapter_id,
            fvi_study_id=fvi_id,
            bootstrap_plan_id=compute_id(_plan_ident(smoke_plan)),
            environment_contract_id=env_id,
            fvi_selected=selected,
            replicate_count=100,
            profile_variant="smoke",
            myopic_artifact_sha256=myopic_sha,
            producer_hashes=producer_hashes,
            prerequisite_receipts={},
        )
        smoke_id = compute_id(smoke_spec)
        smoke_root = out / "runs" / f"smoke_local_{smoke_id[:12]}"
        smoke_aggregate, smoke_result = _run_bound_sweep(
            adapter_dir=adapter_dir,
            run_spec=smoke_spec,
            plan=smoke_plan,
            run_root=smoke_root,
            myopic_sha256=myopic_sha,
            producer_hashes=producer_hashes,
            environment=environment_record,
            cells=profile.smoke_cells(),
            command=["run_stopdff_v5_local", "--variant", "smoke"],
        )
        if (
            not smoke_result.passed
            or smoke_aggregate["release_status"] != "VALID"
        ):
            raise ValueError(
                "bounded smoke failed: "
                + "; ".join(smoke_result.errors[:10])
            )
        persist_receipt(
            "smoke",
            writers.build_prerequisite_receipt(
                gate="smoke",
                bindings=common_bindings,
                evidence={
                    "aggregate_sha256": sha256_file(
                        smoke_root / "aggregate.json"
                    )
                },
            ),
        )

    replicates = 1000 if args.variant == "final" else 100
    plan = build_bootstrap_plan(paired_items, replicates=replicates, seed=1)
    print("== bootstrap plan ==", "reps", replicates, "items", plan.n_items)

    run_spec = run_spec_identity(
        source_manifest_id=source_id, raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        adapter_bundle_id=adapter_id, fvi_study_id=fvi_id, bootstrap_plan_id=compute_id(_plan_ident(plan)),
        environment_contract_id=env_id, fvi_selected=selected, replicate_count=replicates,
        profile_variant=args.variant,
        myopic_artifact_sha256=myopic_sha,
        producer_hashes=producer_hashes,
        prerequisite_receipts=(
            {key: receipt_ids[key] for key in sorted(receipt_ids)}
            if args.variant == "final"
            else {}
        ),
    )
    run_spec_id = compute_id(run_spec)
    run_id = f"{args.variant}_local_{run_spec_id[:12]}"
    run_root = out / "runs" / run_id

    print("== sweep (backend=local) ==")
    agg, prepackage_result = _run_bound_sweep(
        adapter_dir=adapter_dir,
        run_spec=run_spec,
        plan=plan,
        run_root=run_root,
        myopic_sha256=myopic_sha,
        producer_hashes=producer_hashes,
        environment=environment_record,
        cells=(
            profile.full_grid()
            if args.variant == "final"
            else profile.smoke_cells()
        ),
        command=["run_stopdff_v5_local", "--variant", args.variant],
    )
    if not prepackage_result.passed:
        raise ValueError(
            "run failed validation before packaging: "
            + "; ".join(prepackage_result.errors[:10])
        )
    fvi_bytes = (
        json.dumps(fvi_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    environment_bytes = (
        json.dumps(environment_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    source_manifest_path = out / "source_snapshot" / "source_manifest.json"
    raw_manifest_path = out / "raw_inputs" / "raw_input_manifest.json"
    model_manifest_path = out / "model" / "model_snapshot_manifest.json"
    external_artifacts = [
        _ledger_entry(
            "source_manifest",
            source_id,
            source_manifest_path,
            "source_snapshot/source_manifest.json",
        ),
        _ledger_entry(
            "raw_input_manifest",
            raw_id,
            raw_manifest_path,
            "raw_inputs/raw_input_manifest.json",
        ),
        _ledger_entry(
            "model_snapshot_manifest",
            model_id,
            model_manifest_path,
            "model/model_snapshot_manifest.json",
        ),
        {
            "role": "fvi_study",
            "content_id": fvi_id,
            "sha256": sha256_bytes(fvi_bytes),
            "byte_size": len(fvi_bytes),
            "retrieval_path": "evidence/fvi_study.json",
        },
        {
            "role": "environment_contract",
            "content_id": env_id,
            "sha256": sha256_bytes(environment_bytes),
            "byte_size": len(environment_bytes),
            "retrieval_path": "evidence/environment_contract.json",
        },
    ]
    writers.package_run(
        run_root,
        agg,
        resource_summary={"backend": "local"},
        external_artifacts=external_artifacts,
        evidence_files={
            "evidence/fvi_study.json": fvi_bytes,
            "evidence/environment_contract.json": environment_bytes,
        },
    )

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


def _ledger_entry(
    role: str,
    content_id: str,
    path: Path,
    retrieval_path: str,
) -> dict:
    return {
        "role": role,
        "content_id": content_id,
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
        "retrieval_path": retrieval_path,
    }


if __name__ == "__main__":
    raise SystemExit(main())
