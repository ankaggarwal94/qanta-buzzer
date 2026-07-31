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
from pathlib import Path, PurePosixPath

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
    loads_no_duplicate_keys,
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
    resume: bool = False,
    attempt_number: int = 1,
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
            "attempt": attempt_number,
            "mode": "resume" if resume else "fresh",
            "command": command,
            "run_spec_id": binding["run_spec_id"],
            "adapter_id": binding["adapter_bundle_id"],
            "bootstrap_plan_id": binding["bootstrap_plan_id"],
        },
        resume=resume,
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


def main(argv: list[str] | None = None) -> int:
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
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume one compatible interrupted run from --out-dir",
    )
    ap.add_argument(
        "--allow-low-mc-retention",
        action="store_true",
        help="identity-bind and allow a below-threshold MC retention decision",
    )
    args = ap.parse_args(argv)

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
    out = Path(args.out_dir)
    if args.resume:
        return _resume_local_run(
            args=args,
            out=out,
            run_sha=run_sha,
        )

    with tempfile.TemporaryDirectory(prefix="stopdff_v5_selftest_") as work:
        mutation_ok, mutation_results = selftest.run_self_test(Path(work))
    if not mutation_ok:
        failed = [
            result["mutation"]
            for result in mutation_results
            if not result["ok"]
        ]
        raise ValueError(f"v5 mutation gate failed: {failed}")

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
        producer_hashes={"adapter_build.py": sha256_file(_REPO / "scripts/stopdff_v5/adapter_build.py")},
        allow_low_mc_retention=args.allow_low_mc_retention,
    )
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
    sweep._write_bound_json(
        out / "fvi_study.json",
        fvi_manifest,
        resume=False,
    )

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
            allow_low_mc_retention=args.allow_low_mc_retention,
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
            gate_overrides={
                "allow_low_mc_retention": args.allow_low_mc_retention,
            },
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
        gate_overrides={
            "allow_low_mc_retention": args.allow_low_mc_retention,
        },
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
    return _package_and_validate_local_run(
        out=out,
        run_root=run_root,
        adapter_dir=adapter_dir,
        aggregate=agg,
        fvi_manifest=fvi_manifest,
        environment_manifest=environment_manifest,
        source_id=source_id,
        raw_id=raw_id,
        model_id=model_id,
        require_final=args.variant == "final",
    )


def _load_bound_content_manifest(
    base: Path,
    *,
    manifest_name: str,
    expected_kind: str,
    file_key: str,
    name_key: str,
    content_subdir: str = "",
) -> dict:
    """Load one content manifest and rehash every declared regular file."""
    base = Path(base)
    path = base / manifest_name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"resume manifest is missing or noncanonical: {path}")
    manifest = loads_no_duplicate_keys(path.read_text(encoding="utf-8"))
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if (
        not isinstance(identity, dict)
        or compute_id(identity) != manifest.get("id")
        or identity.get("kind") != expected_kind
    ):
        raise ValueError(f"resume manifest identity is invalid: {path}")
    entries = identity.get(file_key)
    if not isinstance(entries, list):
        raise ValueError(f"resume manifest lacks {file_key}: {path}")
    seen: set[str] = set()
    content_root = base / content_subdir if content_subdir else base
    for entry in entries:
        name = entry.get(name_key) if isinstance(entry, dict) else None
        parsed = PurePosixPath(name) if isinstance(name, str) else None
        if (
            not isinstance(name, str)
            or not name
            or parsed is None
            or parsed.is_absolute()
            or ".." in parsed.parts
            or name in seen
        ):
            raise ValueError(f"resume manifest has an unsafe file entry: {path}")
        seen.add(name)
        declared_size = entry.get("size")
        target = content_root / name
        if (
            isinstance(declared_size, bool)
            or not isinstance(declared_size, int)
            or target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != declared_size
            or sha256_file(target) != entry.get("sha256")
        ):
            raise ValueError(f"resume manifest file mismatch: {target}")
    return manifest


def _next_resume_attempt(
    run_root: Path,
    *,
    run_spec_id: str,
    adapter_id: str,
    bootstrap_plan_id: str,
) -> int:
    """Validate the append-only started records and derive the next attempt."""
    attempts_path = Path(run_root) / "attempts.jsonl"
    if attempts_path.is_symlink() or not attempts_path.is_file():
        raise ValueError("resume requires a canonical attempts.jsonl")
    records = [
        loads_no_duplicate_keys(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("resume requires at least one prior attempt")
    for number, record in enumerate(records, start=1):
        command = record.get("command") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("attempt") != number
            or record.get("state") != "started"
            or record.get("run_spec_id") != run_spec_id
            or record.get("adapter_id") != adapter_id
            or record.get("bootstrap_plan_id") != bootstrap_plan_id
            or not isinstance(command, list)
            or not all(isinstance(part, str) for part in command)
            or "--overwrite" in command
        ):
            raise ValueError("resume attempt history invariant mismatch")
        if number == 1:
            if record.get("mode") != "fresh" or "--resume" in command:
                raise ValueError("resume initial attempt invariant mismatch")
        elif (
            record.get("mode") != "resume"
            or command.count("--resume") != 1
        ):
            raise ValueError("resume attempt mode invariant mismatch")
    return len(records) + 1


def _package_and_validate_local_run(
    *,
    out: Path,
    run_root: Path,
    adapter_dir: Path,
    aggregate: dict,
    fvi_manifest: dict,
    environment_manifest: dict,
    source_id: str,
    raw_id: str,
    model_id: str,
    require_final: bool,
) -> int:
    fvi_id = fvi_manifest["id"]
    env_id = environment_manifest["id"]
    fvi_bytes = (
        json.dumps(fvi_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    environment_bytes = (
        json.dumps(environment_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    external_artifacts = [
        _ledger_entry(
            "source_manifest",
            source_id,
            out / "source_snapshot" / "source_manifest.json",
            "source_snapshot/source_manifest.json",
        ),
        _ledger_entry(
            "raw_input_manifest",
            raw_id,
            out / "raw_inputs" / "raw_input_manifest.json",
            "raw_inputs/raw_input_manifest.json",
        ),
        _ledger_entry(
            "model_snapshot_manifest",
            model_id,
            out / "model" / "model_snapshot_manifest.json",
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
        aggregate,
        resource_summary={"backend": "local"},
        external_artifacts=external_artifacts,
        evidence_files={
            "evidence/fvi_study.json": fvi_bytes,
            "evidence/environment_contract.json": environment_bytes,
        },
    )

    print("== validate (backend=local) ==")
    result = checker.validate_run(
        run_root,
        backend="local",
        adapter_bundle=adapter_dir,
        require_final_profile=require_final,
        require_package=True,
    )
    print(json.dumps({
        "release_status": aggregate["release_status"],
        "requested": aggregate["requested"],
        "completed": aggregate["completed"],
        "failed": aggregate["failed"],
        "family": aggregate.get("family"),
        "checker_passed": result.passed,
        "checker_errors": result.errors[:10],
    }, indent=2))
    if not (result.passed and aggregate["release_status"] == "VALID"):
        return 1
    print(f"\nLOCAL REPRODUCTION OK -> {run_root}")
    return 0


def _resume_local_run(*, args, out: Path, run_sha: str) -> int:
    """Resume exactly one compatible local sweep; never recreate stage inputs."""
    if out.is_symlink() or not out.is_dir():
        raise ValueError("--resume requires an existing canonical --out-dir")
    source_manifest = _load_bound_content_manifest(
        out / "source_snapshot",
        manifest_name="source_manifest.json",
        expected_kind="source_snapshot",
        file_key="files",
        name_key="path",
        content_subdir="source",
    )
    raw_manifest = _load_bound_content_manifest(
        out / "raw_inputs",
        manifest_name="raw_input_manifest.json",
        expected_kind="raw_input_bundle",
        file_key="files",
        name_key="role",
        content_subdir="raw",
    )
    model_manifest = _load_bound_content_manifest(
        out / "model",
        manifest_name="model_snapshot_manifest.json",
        expected_kind="model_snapshot",
        file_key="files",
        name_key="path",
        content_subdir="snapshot",
    )
    if source_manifest["identity"].get("git_sha") != run_sha:
        raise ValueError("resume source snapshot does not match executing commit")
    semantic_checks = raw_manifest["identity"].get("semantic_checks")
    if (
        not isinstance(semantic_checks, dict)
        or semantic_checks.get("all_semantic_checks_pass") is not True
    ):
        raise ValueError("resume raw-input semantic checks are not passing")

    adapter_dir = out / "adapter_bundle"
    adapter_result = checker.validate_adapter(adapter_dir)
    adapter_manifest = checker.load_json(adapter_dir / "manifest.json")
    adapter_identity = (
        adapter_manifest.get("identity")
        if isinstance(adapter_manifest, dict)
        else None
    )
    expected_adapter_bindings = {
        "source_manifest_id": source_manifest["id"],
        "raw_input_bundle_id": raw_manifest["id"],
        "model_snapshot_id": model_manifest["id"],
    }
    if (
        not adapter_result.passed
        or not isinstance(adapter_identity, dict)
        or adapter_result.recomputed.get("adapter_bundle_id")
        != adapter_manifest.get("id")
        or any(
            adapter_identity.get(key) != value
            for key, value in expected_adapter_bindings.items()
        )
    ):
        raise ValueError(
            "resume adapter is invalid or bound to different inputs: "
            + "; ".join(adapter_result.errors[:10])
        )
    adapter_id = adapter_manifest["id"]

    runs_dir = out / "runs"
    prefix = f"{args.variant}_local_"
    candidates = (
        sorted(
            path
            for path in runs_dir.iterdir()
            if (
                path.name.startswith(prefix)
                and not path.is_symlink()
                and path.is_dir()
            )
        )
        if runs_dir.is_dir() and not runs_dir.is_symlink()
        else []
    )
    if len(candidates) != 1:
        raise ValueError(
            "--resume requires exactly one compatible variant run directory"
        )
    run_root = candidates[0]
    spec_path = run_root / "run_spec.json"
    spec_result = checker.validate_spec(
        spec_path,
        require_final_profile=args.variant == "final",
    )
    if not spec_result.passed:
        raise ValueError(
            "resume run spec is invalid: " + "; ".join(spec_result.errors[:10])
        )
    spec_manifest = checker.load_json(spec_path)
    run_spec = spec_manifest["identity"]
    run_spec_id = spec_manifest["id"]
    if (
        run_spec.get("gate", {}).get("allow_low_mc_retention")
        is not args.allow_low_mc_retention
    ):
        raise ValueError(
            "resume low-retention override does not match the run spec"
        )
    if run_root.name != f"{args.variant}_local_{run_spec_id[:12]}":
        raise ValueError("resume run directory is not bound to its run spec")
    spec_ids = run_spec["identity"]
    if (
        spec_ids.get("source_manifest_id") != source_manifest["id"]
        or spec_ids.get("raw_input_bundle_id") != raw_manifest["id"]
        or spec_ids.get("model_snapshot_id") != model_manifest["id"]
        or spec_ids.get("adapter_bundle_id") != adapter_id
    ):
        raise ValueError("resume run spec input bindings do not match staged evidence")

    fvi_path = out / "fvi_study.json"
    if fvi_path.is_symlink() or not fvi_path.is_file():
        raise ValueError("resume requires the durable pre-sweep FVI manifest")
    fvi_manifest = checker.load_json(fvi_path)
    fvi_identity = fvi_manifest.get("identity")
    selected = {
        "tolerance": run_spec["fvi_selected"]["tolerance"],
        "max_iterations": run_spec["fvi_selected"]["max_iterations"],
    }
    recorded_selected = (
        fvi_identity.get("selected_parameters")
        if isinstance(fvi_identity, dict)
        and fvi_identity.get("kind") == "fvi_study"
        else fvi_identity.get("selected")
        if isinstance(fvi_identity, dict)
        else None
    )
    if (
        not isinstance(fvi_identity, dict)
        or compute_id(fvi_identity) != fvi_manifest.get("id")
        or fvi_manifest.get("id") != spec_ids.get("fvi_study_id")
        or fvi_identity.get("adapter_bundle_id") != adapter_id
        or recorded_selected != selected
    ):
        raise ValueError("resume FVI manifest does not match the run spec")

    versions = _versions()
    environment_record = {
        "python_version": "%d.%d.%d" % sys.version_info[:3],
        "package_versions": versions,
    }
    environment_manifest = build_manifest(environment_contract_identity(
        python_version=environment_record["python_version"],
        package_versions=versions,
    ))
    if environment_manifest["id"] != spec_ids.get("environment_contract_id"):
        raise ValueError("resume environment does not match the run spec")

    raw_files = {
        entry["role"]: entry
        for entry in raw_manifest["identity"]["files"]
    }
    myopic = raw_files.get("stopdff.json")
    if (
        not isinstance(myopic, dict)
        or run_spec["evidence_roots"].get("myopic_artifact_sha256")
        != myopic.get("sha256")
    ):
        raise ValueError("resume myopic artifact does not match the run spec")
    producer_hashes = {
        "sweep.py": sha256_file(_REPO / "scripts/stopdff_v5/sweep.py"),
        "checker.py": sha256_file(_REPO / "scripts/stopdff_v5/checker.py"),
    }
    if run_spec["evidence_roots"].get("producer_hashes") != producer_hashes:
        raise ValueError("resume producer hashes do not match executing source")

    receipt_ids = run_spec["evidence_roots"]["prerequisite_receipts"]
    receipts: dict[str, dict] = {}
    for gate, receipt_id in receipt_ids.items():
        receipt_path = out / "receipts" / gate / f"{receipt_id}.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError(f"resume prerequisite receipt is missing: {gate}")
        receipt = loads_no_duplicate_keys(
            receipt_path.read_text(encoding="utf-8")
        )
        if not isinstance(receipt, dict):
            raise ValueError(f"resume prerequisite receipt is invalid: {gate}")
        receipts[gate] = receipt
    writers.validate_prerequisite_receipts(
        profile_variant=args.variant,
        identity_bindings={
            key: spec_ids[key]
            for key in (
                "source_manifest_id",
                "raw_input_bundle_id",
                "model_snapshot_id",
                "adapter_bundle_id",
                "fvi_study_id",
                "environment_contract_id",
            )
        },
        receipt_ids=receipt_ids,
        receipts=receipts,
    )

    rows = checker.load_adapter_rows(adapter_dir)
    mc = {
        row["item_id"]
        for row in rows
        if row["split"] == "test" and row["format"] == "MC"
    }
    qa = {
        row["item_id"]
        for row in rows
        if row["split"] == "test" and row["format"] == "QA"
    }
    replicate_count = run_spec["bootstrap"]["replicate_count"]
    plan = build_bootstrap_plan(
        sorted(mc & qa),
        replicates=replicate_count,
        seed=1,
    )
    bootstrap_plan_id = compute_id(_plan_ident(plan))
    if bootstrap_plan_id != spec_ids.get("bootstrap_plan_id"):
        raise ValueError("resume bootstrap plan does not match the run spec")

    aggregate_path = run_root / "aggregate.json"
    if (run_root / "SHA256SUMS").is_file() and aggregate_path.is_file():
        complete = checker.validate_run(
            run_root,
            backend="local",
            adapter_bundle=adapter_dir,
            require_final_profile=args.variant == "final",
            require_package=True,
        )
        aggregate = checker.load_json(aggregate_path)
        if complete.passed and aggregate.get("release_status") == "VALID":
            print(f"LOCAL REPRODUCTION ALREADY COMPLETE -> {run_root}")
            return 0

    attempt_number = _next_resume_attempt(
        run_root,
        run_spec_id=run_spec_id,
        adapter_id=adapter_id,
        bootstrap_plan_id=bootstrap_plan_id,
    )
    print(f"== resume sweep (backend=local, attempt={attempt_number}) ==")
    aggregate, prepackage = _run_bound_sweep(
        adapter_dir=adapter_dir,
        run_spec=run_spec,
        plan=plan,
        run_root=run_root,
        myopic_sha256=myopic["sha256"],
        producer_hashes=producer_hashes,
        environment=environment_record,
        cells=(
            profile.full_grid()
            if args.variant == "final"
            else profile.smoke_cells()
        ),
        command=[
            "run_stopdff_v5_local",
            "--variant",
            args.variant,
            "--resume",
        ],
        resume=True,
        attempt_number=attempt_number,
    )
    if not prepackage.passed:
        raise ValueError(
            "resumed run failed validation before packaging: "
            + "; ".join(prepackage.errors[:10])
        )
    return _package_and_validate_local_run(
        out=out,
        run_root=run_root,
        adapter_dir=adapter_dir,
        aggregate=aggregate,
        fvi_manifest=fvi_manifest,
        environment_manifest=environment_manifest,
        source_id=source_manifest["id"],
        raw_id=raw_manifest["id"],
        model_id=model_manifest["id"],
        require_final=args.variant == "final",
    )


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
