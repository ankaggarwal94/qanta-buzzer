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
import os
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from types import ModuleType

_REPO = Path(__file__).resolve().parents[1]
_REPO_IMPORT_ROOT = str(_REPO)
# Membership is not precedence: checkout B can otherwise remain before the
# reviewed checkout A on PYTHONPATH.  Make this entrypoint's checkout the
# authoritative import root before loading evidentiary producer code.
sys.path[:] = [entry for entry in sys.path if entry != _REPO_IMPORT_ROOT]
sys.path.insert(0, _REPO_IMPORT_ROOT)

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
from scripts.stopdff_v5.attempt_history import load_attempt_history  # noqa: E402
from scripts.stopdff_v5.content_manifest import git_mode_for_path  # noqa: E402
from scripts.stopdff_v5.identity import (  # noqa: E402
    build_manifest,
    compute_id,
    loads_no_duplicate_keys,
    sha256_bytes,
    sha256_file,
)
from scripts.stopdff_v5.manifests import (  # noqa: E402
    ENVIRONMENT_PACKAGES,
    FVI_PRODUCER_FILES,
    environment_contract_identity,
    fvi_study_identity,
    run_spec_identity,
)
from scripts.stopdff_v5.receipt_evidence import MUTATION_ROSTER  # noqa: E402


_IMPORTED_PRODUCER_MODULES: tuple[ModuleType, ...] = (
    adapter_build,
    checker,
    fvi_study,
    producers,
    profile,
    selftest,
    sweep,
    writers,
    sys.modules[build_bootstrap_plan.__module__],
    sys.modules[build_manifest.__module__],
    sys.modules[environment_contract_identity.__module__],
)


def _verify_imported_producer_origins() -> None:
    """Fail unless every imported producer came from this exact checkout."""
    repo = _REPO.resolve()
    modules = {module.__name__: module for module in _IMPORTED_PRODUCER_MODULES}
    modules.update(
        {
            name: module
            for name, module in sys.modules.items()
            if isinstance(module, ModuleType)
            and (
                name == "scripts.stopdff_v5"
                or name.startswith("scripts.stopdff_v5.")
                or name == "qb_data"
                or name.startswith("qb_data.")
            )
        }
    )
    for name, module in sorted(modules.items()):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            raise ValueError(
                f"imported producer {name} has no filesystem origin"
            )
        stem = repo.joinpath(*name.split("."))
        expected = (
            stem / "__init__.py"
            if (stem / "__init__.py").is_file()
            else stem.with_suffix(".py")
        )
        try:
            actual = Path(origin).resolve(strict=True)
            expected = expected.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ValueError(
                f"imported producer {name} has an invalid origin"
            ) from exc
        if actual != expected:
            raise ValueError(
                "imported producer does not originate from the executing "
                f"repository: {name} ({actual} != {expected})"
            )


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    missing: list[str] = []
    for name in ENVIRONMENT_PACKAGES:
        try:
            out[name] = im.version(name)
        except im.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise ValueError(
            "required environment distributions are missing: "
            + ", ".join(missing)
        )
    if set(out) != set(ENVIRONMENT_PACKAGES):
        raise ValueError("environment package set is not the declared closed set")
    return {name: out[name] for name in ENVIRONMENT_PACKAGES}


def _verified_local_source_execution(
    repo_root: Path,
    source_manifest: dict,
) -> dict[str, str]:
    """Rehash the executing clean checkout against its source snapshot."""
    _verify_imported_producer_origins()
    identity = source_manifest.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("kind") != "source_snapshot"
        or not isinstance(source_manifest.get("id"), str)
    ):
        raise ValueError("local source manifest is invalid")
    for entry in identity.get("files", []):
        runtime_path = repo_root / entry["path"]
        if (
            runtime_path.is_symlink()
            or not runtime_path.is_file()
            or sha256_file(runtime_path) != entry["sha256"]
            or git_mode_for_path(runtime_path) != entry["mode"]
        ):
            raise ValueError(
                f"executing source does not match source manifest: {entry['path']}"
            )
    source_id = source_manifest["id"]
    return {
        "environment": "local_clean_worktree",
        "executing_source_manifest_id": source_id,
        "runtime_source_manifest_id": source_id,
    }


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
        gate_overrides=binding["gate_overrides"],
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


_LOCAL_LIFECYCLE_FILE = "local_lifecycle.json"
_LOCAL_LIFECYCLE_SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, value: dict) -> None:
    """Publish a small lifecycle checkpoint with replace semantics."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _lifecycle_contract(*, args, run_sha: str) -> dict:
    return {
        "schema_version": _LOCAL_LIFECYCLE_SCHEMA_VERSION,
        "run_sha": run_sha,
        "variant": args.variant,
        "skip_fvi_study": args.skip_fvi_study,
        "fvi_tolerance": args.fvi_tolerance,
        "fvi_max_iterations": args.fvi_max_iterations,
        "allow_low_mc_retention": args.allow_low_mc_retention,
    }


def _load_or_create_lifecycle(
    *,
    out: Path,
    args,
    run_sha: str,
    resume: bool,
) -> dict:
    path = out / _LOCAL_LIFECYCLE_FILE
    expected = _lifecycle_contract(args=args, run_sha=run_sha)
    if resume:
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "--resume before sweep creation requires local_lifecycle.json"
            )
        state = loads_no_duplicate_keys(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("local lifecycle checkpoint is invalid")
        actual_contract = {
            key: state.get(key)
            for key in expected
        }
        if actual_contract != expected:
            raise ValueError("local lifecycle checkpoint does not match this command")
        executions = state.get("adapter_executions")
        if not isinstance(executions, dict):
            raise ValueError("local lifecycle adapter execution map is invalid")
        return state
    state = {**expected, "adapter_executions": {}}
    _atomic_write_json(path, state)
    return state


def _validate_checkpointed_mutation_results(results: object) -> list[dict]:
    """Return a complete successful mutation roster or fail closed."""
    if not isinstance(results, list) or tuple(
        result.get("mutation") if isinstance(result, dict) else None
        for result in results
    ) != MUTATION_ROSTER:
        raise ValueError("local lifecycle mutation roster is invalid")
    expected_fields = {"mutation", "expected", "passed_check", "ok", "errors"}
    for index, result in enumerate(results):
        expected = "PASS" if index == 0 else "REJECT"
        passed_check = index == 0
        if (
            not isinstance(result, dict)
            or set(result) != expected_fields
            or result.get("expected") != expected
            or result.get("passed_check") is not passed_check
            or result.get("ok") is not True
            or not isinstance(result.get("errors"), list)
            or any(
                not isinstance(error, str)
                for error in result.get("errors", [])
            )
        ):
            raise ValueError("local lifecycle mutation outcome is invalid")
    return results


def _checkpoint_adapter_execution(
    *,
    out: Path,
    state: dict,
    stage: str,
    execution_id: str,
    adapter_id: str,
) -> None:
    executions = state["adapter_executions"]
    existing = executions.get(stage)
    value = {"execution_id": execution_id, "adapter_id": adapter_id}
    if existing is not None and existing != value:
        raise ValueError(f"local lifecycle {stage} checkpoint mismatch")
    executions[stage] = value
    _atomic_write_json(out / _LOCAL_LIFECYCLE_FILE, state)


def _adapter_execution_id(
    *,
    state: dict,
    stage: str,
    adapter_id: str,
) -> str:
    record = state["adapter_executions"].get(stage)
    if (
        not isinstance(record, dict)
        or record.get("adapter_id") != adapter_id
        or not isinstance(record.get("execution_id"), str)
        or not record["execution_id"].startswith("local-")
    ):
        raise ValueError(f"local lifecycle {stage} execution is not checkpointed")
    return record["execution_id"]


def _variant_run_candidates(out: Path, variant: str) -> list[Path]:
    runs_dir = out / "runs"
    prefix = f"{variant}_local_"
    if not runs_dir.is_dir() or runs_dir.is_symlink():
        return []
    return sorted(
        path
        for path in runs_dir.iterdir()
        if (
            path.name.startswith(prefix)
            and not path.is_symlink()
            and path.is_dir()
        )
    )


def _fsync_staged_tree(root: Path) -> None:
    """Make a closed staged tree durable before publishing its directory entry."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("local stage builder did not produce a canonical directory")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    for directory_name, child_directories, filenames in os.walk(
        root,
        topdown=False,
        followlinks=False,
    ):
        directory = Path(directory_name)
        for name in child_directories:
            child = directory / name
            if child.is_symlink():
                raise ValueError(f"local stage contains a symlink: {child}")
        for name in filenames:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"local stage contains a non-file: {path}")
            descriptor = os.open(path, os.O_RDONLY | nofollow)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError(
                        f"local stage contains a non-regular file: {path}"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(
            directory,
            os.O_RDONLY | directory_flag | nofollow,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_stage_directory(
    *,
    out: Path,
    target_name: str,
    build,
):
    """Build away from the public path and atomically publish the directory."""
    target = out / target_name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"local stage already exists: {target_name}")
    with tempfile.TemporaryDirectory(prefix=f".{target_name}-", dir=out) as holder:
        staged = Path(holder) / "artifact"
        result = build(staged)
        _fsync_staged_tree(staged)
        staged.replace(target)
        directory = os.open(out, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return result


def _load_valid_adapter_stage(
    path: Path,
    *,
    source_id: str,
    raw_id: str,
    model_id: str,
) -> dict:
    result = checker.validate_adapter(path)
    manifest = checker.load_json(path / "manifest.json")
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    expected = {
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
    }
    if (
        not result.passed
        or not isinstance(identity, dict)
        or result.recomputed.get("adapter_bundle_id") != manifest.get("id")
        or any(identity.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(
            "local adapter stage is invalid or bound to different inputs: "
            + "; ".join(result.errors[:10])
        )
    return manifest


def _worktree_status_command(
    *,
    repo_root: Path,
    out: Path,
    resume: bool,
) -> list[str]:
    """Build the clean-tree query while excluding only resumed output bytes."""
    command = [
        "git",
        "-C",
        str(repo_root),
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".",
    ]
    if not resume:
        return command
    try:
        relative_out = out.relative_to(repo_root)
    except ValueError:
        return command
    if relative_out == Path("."):
        raise ValueError("--out-dir cannot be the repository root")
    command.append(f":(top,exclude){relative_out.as_posix()}")
    return command


def main(argv: list[str] | None = None) -> int:
    _verify_imported_producer_origins()
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
    repo_root = args.repo_root.resolve()
    out = Path(args.out_dir).absolute()
    if out.resolve(strict=False) != out:
        raise ValueError("--out-dir must not traverse symlinked path components")
    status_command = _worktree_status_command(
        repo_root=repo_root,
        out=out,
        resume=args.resume,
    )
    status = subprocess.run(
        status_command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("local reproduction requires a clean source worktree")

    run_sha = subprocess.run(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    runs_dir = out / "runs"
    if runs_dir.is_symlink():
        raise ValueError("local runs directory must not be a symlink")
    if args.resume and _variant_run_candidates(out, args.variant):
        _load_or_create_lifecycle(
            out=out,
            args=args,
            run_sha=run_sha,
            resume=True,
        )
        return _resume_local_run(
            args=args,
            out=out,
            run_sha=run_sha,
        )
    if args.resume:
        if out.is_symlink() or not out.is_dir():
            raise ValueError("--resume requires an existing canonical --out-dir")
    else:
        out.mkdir(parents=True, exist_ok=False)
    lifecycle = _load_or_create_lifecycle(
        out=out,
        args=args,
        run_sha=run_sha,
        resume=args.resume,
    )

    print("== source snapshot ==")
    source_stage = out / "source_snapshot"
    if source_stage.exists() or source_stage.is_symlink():
        src_man = _load_bound_content_manifest(
            source_stage,
            manifest_name="source_manifest.json",
            expected_kind="source_snapshot",
            file_key="files",
            name_key="path",
            content_subdir="source",
        )
        if src_man["identity"].get("git_sha") != run_sha:
            raise ValueError("local source snapshot does not match executing commit")
    else:
        src_man = _publish_stage_directory(
            out=out,
            target_name="source_snapshot",
            build=lambda staged: producers.build_source_snapshot(
                args.repo_root,
                run_sha,
                staged,
            ),
        )
    source_id = src_man["id"]
    source_execution = _verified_local_source_execution(args.repo_root, src_man)
    print("  source_manifest_id", source_id)

    recorded_mutations = lifecycle.get("mutation_results")
    if recorded_mutations is not None:
        mutation_results = _validate_checkpointed_mutation_results(
            recorded_mutations
        )
        mutation_ok = True
    else:
        with tempfile.TemporaryDirectory(prefix="stopdff_v5_selftest_") as work:
            mutation_ok, mutation_results = selftest.run_self_test(Path(work))
        if mutation_ok:
            lifecycle["mutation_results"] = mutation_results
            _atomic_write_json(out / _LOCAL_LIFECYCLE_FILE, lifecycle)
    if not mutation_ok:
        failed = [
            result["mutation"]
            for result in mutation_results
            if not result["ok"]
        ]
        raise ValueError(f"v5 mutation gate failed: {failed}")

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
    raw_stage = out / "raw_inputs"
    if raw_stage.exists() or raw_stage.is_symlink():
        raw_man = _load_bound_content_manifest(
            raw_stage,
            manifest_name="raw_input_manifest.json",
            expected_kind="raw_input_bundle",
            file_key="files",
            name_key="role",
            content_subdir="raw",
        )
    else:
        raw_man = _publish_stage_directory(
            out=out,
            target_name="raw_inputs",
            build=lambda staged: producers.stage_raw_inputs(roles, staged),
        )
    raw_id = raw_man["id"]
    myopic_sha = next(f["sha256"] for f in raw_man["identity"]["files"] if f["role"] == "stopdff.json")
    raw_dir = out / "raw_inputs" / "raw"
    print("  raw_input_bundle_id", raw_id)

    print("== model snapshot (all-MiniLM-L6-v2, pinned revision) ==")
    model_stage = out / "model"
    if model_stage.exists() or model_stage.is_symlink():
        model_man = _load_bound_content_manifest(
            model_stage,
            manifest_name="model_snapshot_manifest.json",
            expected_kind="model_snapshot",
            file_key="files",
            name_key="path",
            content_subdir="snapshot",
        )
    else:
        def build_model_snapshot(staged: Path) -> dict:
            manifest = adapter_build.freeze_model_snapshot(staged)
            manifest["snapshot_dir"] = str(model_stage / "snapshot")
            _atomic_write_json(
                staged / "model_snapshot_manifest.json",
                manifest,
            )
            return manifest

        model_man = _publish_stage_directory(
            out=out,
            target_name="model",
            build=build_model_snapshot,
        )
    model_id = model_man["id"]
    print("  model_snapshot_id", model_id, "rev", model_man["identity"]["model_revision"])

    print("== adapter bundle (CPU scoring) ==")
    adapter_dir = out / "adapter_bundle"
    if adapter_dir.exists() or adapter_dir.is_symlink():
        adapter_man = _load_valid_adapter_stage(
            adapter_dir,
            source_id=source_id,
            raw_id=raw_id,
            model_id=model_id,
        )
    else:
        def build_primary_adapter(staged: Path) -> dict:
            return adapter_build.build_adapter_bundle(
                mc_dataset_path=raw_dir / "mc_dataset.json",
                val_dataset_path=raw_dir / "val_dataset.json",
                test_dataset_path=raw_dir / "test_dataset.json",
                calibration_path=raw_dir / "calibration.json",
                model_snapshot_dir=out / "model" / "snapshot",
                out_dir=staged,
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

        adapter_man = _publish_stage_directory(
            out=out,
            target_name="adapter_bundle",
            build=build_primary_adapter,
        )
    adapter_id = adapter_man["id"]
    _load_valid_adapter_stage(
        adapter_dir,
        source_id=source_id,
        raw_id=raw_id,
        model_id=model_id,
    )
    if "adapter_bundle" not in lifecycle["adapter_executions"]:
        _checkpoint_adapter_execution(
            out=out,
            state=lifecycle,
            stage="adapter_bundle",
            execution_id=f"local-{uuid.uuid4().hex}",
            adapter_id=adapter_id,
        )
    first_build_execution_id = _adapter_execution_id(
        state=lifecycle,
        stage="adapter_bundle",
        adapter_id=adapter_id,
    )
    print("  adapter_bundle_id", adapter_id)

    rows = checker.load_adapter_rows(adapter_dir)
    calibration = json.loads((adapter_dir / "calibration.json").read_text())

    fvi_path = out / "fvi_study.json"
    if fvi_path.exists() or fvi_path.is_symlink():
        if fvi_path.is_symlink() or not fvi_path.is_file():
            raise ValueError("local FVI stage is not a canonical file")
        fvi_manifest = checker.load_json(fvi_path)
        fvi_identity = (
            fvi_manifest.get("identity")
            if isinstance(fvi_manifest, dict)
            else None
        )
        if (
            not isinstance(fvi_identity, dict)
            or compute_id(fvi_identity) != fvi_manifest.get("id")
            or fvi_identity.get("adapter_bundle_id") != adapter_id
        ):
            raise ValueError("local FVI stage is invalid or bound to another adapter")
        selected = (
            fvi_identity.get("selected_parameters")
            if fvi_identity.get("kind") == "fvi_study"
            else fvi_identity.get("selected")
        )
        expected_kind = "fvi_study_fixed" if args.skip_fvi_study else "fvi_study"
        if fvi_identity.get("kind") != expected_kind or not isinstance(selected, dict):
            raise ValueError("local FVI stage does not match this command")
        if args.skip_fvi_study and selected != {
            "tolerance": args.fvi_tolerance,
            "max_iterations": args.fvi_max_iterations,
        }:
            raise ValueError("local fixed FVI parameters do not match this command")
        fvi_id = fvi_manifest["id"]
        print("== FVI: reused verified stage ==", selected)
    else:
        if args.skip_fvi_study:
            selected = {
                "tolerance": args.fvi_tolerance,
                "max_iterations": args.fvi_max_iterations,
            }
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
            study = fvi_study.run_fvi_study(
                rows=rows,
                calibration_json=calibration,
            )
            selected = study["selected_parameters"]
            if selected is None:
                print("FVI selector found no eligible candidate", file=sys.stderr)
                return 1
            fvi_identity = fvi_study_identity(
                adapter_bundle_id=adapter_id,
                candidate_grid=study["candidate_grid"],
                representative_generator=study["representative_cell_generator"],
                candidate_results=study["candidate_convergence_results"],
                strict_reference_results=study["strict_reference"],
                selector_rule=study["selector_rule"],
                selected_parameters=selected,
                all96_validation=study["all96_fit_only_validation"],
                producer_hashes={
                    name: sha256_file(_REPO / "scripts" / "stopdff_v5" / name)
                    for name in FVI_PRODUCER_FILES
                },
            )
            fvi_manifest = build_manifest(fvi_identity)
            fvi_id = fvi_manifest["id"]
            print("  selected", selected, "fvi_study_id", fvi_id)
        sweep._write_bound_json(fvi_path, fvi_manifest, resume=False)

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

    def persist_receipt(
        gate: str,
        receipt: dict,
        evidence: dict,
    ) -> None:
        path = out / "receipts" / gate / f"{receipt['id']}.json"
        sweep._write_bound_json(
            path.with_suffix(".evidence.json"),
            evidence,
            resume=True,
        )
        sweep._write_bound_json(path, receipt, resume=True)
        receipt_ids[gate] = receipt["id"]

    if args.variant == "final":
        print("== required deterministic two-build adapter gate ==")
        second_adapter_dir = out / "adapter_bundle_determinism"
        if second_adapter_dir.exists() or second_adapter_dir.is_symlink():
            second_adapter = _load_valid_adapter_stage(
                second_adapter_dir,
                source_id=source_id,
                raw_id=raw_id,
                model_id=model_id,
            )
        else:
            def build_second_adapter(staged: Path) -> dict:
                return adapter_build.build_adapter_bundle(
                    mc_dataset_path=raw_dir / "mc_dataset.json",
                    val_dataset_path=raw_dir / "val_dataset.json",
                    test_dataset_path=raw_dir / "test_dataset.json",
                    calibration_path=raw_dir / "calibration.json",
                    model_snapshot_dir=out / "model" / "snapshot",
                    out_dir=staged,
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

            second_adapter = _publish_stage_directory(
                out=out,
                target_name="adapter_bundle_determinism",
                build=build_second_adapter,
            )
        if "adapter_bundle_determinism" not in lifecycle["adapter_executions"]:
            _checkpoint_adapter_execution(
                out=out,
                state=lifecycle,
                stage="adapter_bundle_determinism",
                execution_id=f"local-{uuid.uuid4().hex}",
                adapter_id=second_adapter["id"],
            )
        second_build_execution_id = _adapter_execution_id(
            state=lifecycle,
            stage="adapter_bundle_determinism",
            adapter_id=second_adapter["id"],
        )
        compared = (
            "fit_rows.jsonl.gz",
            "eval_rows.jsonl.gz",
            "calibration.json",
            "build_metadata.json",
        )
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
        determinism_bindings = {
            key: common_bindings[key]
            for key in (
                "source_manifest_id",
                "raw_input_bundle_id",
                "model_snapshot_id",
                "adapter_bundle_id",
            )
        }
        determinism_evidence = writers.build_prerequisite_evidence(
            gate="determinism",
            bindings=determinism_bindings,
            details={
                "source_execution": source_execution,
                "first_build_execution": {
                    "environment": "local_process",
                    "execution_id": first_build_execution_id,
                    "adapter_subdir": "adapter_bundle",
                    **determinism_bindings,
                    "cached": False,
                    "output_sha256": first_hashes,
                },
                "second_build_execution": {
                    "environment": "local_process",
                    "execution_id": second_build_execution_id,
                    "adapter_subdir": "adapter_bundle_determinism",
                    **determinism_bindings,
                    "cached": False,
                    "output_sha256": second_hashes,
                },
                "first_adapter_manifest": adapter_man,
                "second_adapter_manifest": second_adapter,
                "first_file_sha256": first_hashes,
                "second_file_sha256": second_hashes,
            },
        )
        persist_receipt(
            "determinism",
            writers.build_evidenced_prerequisite_receipt(
                gate="determinism",
                bindings=determinism_bindings,
                evidence=determinism_evidence,
            ),
            determinism_evidence,
        )
        mutation_evidence = writers.build_prerequisite_evidence(
            gate="mutation",
            bindings=common_bindings,
            details={
                "source_execution": source_execution,
                "results": mutation_results,
            },
        )
        persist_receipt(
            "mutation",
            writers.build_evidenced_prerequisite_receipt(
                gate="mutation",
                bindings=common_bindings,
                evidence=mutation_evidence,
            ),
            mutation_evidence,
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
            resource_summary_id=compute_id({"backend": "local"}),
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
        if smoke_root.resolve(strict=False) != smoke_root.absolute():
            raise ValueError("local smoke run path must not traverse symlinks")
        smoke_result = None
        smoke_aggregate = None
        if args.resume and smoke_root.is_dir() and not smoke_root.is_symlink():
            existing_smoke = checker.validate_run(
                smoke_root,
                backend="local",
                adapter_bundle=adapter_dir,
                require_final_profile=False,
                require_package=False,
            )
            aggregate_path = smoke_root / "aggregate.json"
            if existing_smoke.passed and aggregate_path.is_file():
                candidate = checker.load_json(aggregate_path)
                if candidate.get("release_status") == "VALID":
                    smoke_result = existing_smoke
                    smoke_aggregate = candidate
        if smoke_aggregate is None:
            smoke_resume = args.resume and smoke_root.is_dir()
            smoke_attempt = (
                _next_resume_attempt(
                    smoke_root,
                    run_spec_id=smoke_id,
                    adapter_id=adapter_id,
                    bootstrap_plan_id=compute_id(_plan_ident(smoke_plan)),
                )
                if smoke_resume
                else 1
            )
            smoke_aggregate, smoke_result = _run_bound_sweep(
                adapter_dir=adapter_dir,
                run_spec=smoke_spec,
                plan=smoke_plan,
                run_root=smoke_root,
                myopic_sha256=myopic_sha,
                producer_hashes=producer_hashes,
                environment=environment_record,
                cells=profile.smoke_cells(),
                command=[
                    "run_stopdff_v5_local",
                    "--variant",
                    "smoke",
                    *(["--resume"] if smoke_resume else []),
                ],
                resume=smoke_resume,
                attempt_number=smoke_attempt,
            )
        assert smoke_result is not None
        assert smoke_aggregate is not None
        if (
            not smoke_result.passed
            or smoke_aggregate["release_status"] != "VALID"
        ):
            raise ValueError(
                "bounded smoke failed: "
                + "; ".join(smoke_result.errors[:10])
            )
        smoke_evidence = writers.build_prerequisite_evidence(
            gate="smoke",
            bindings=common_bindings,
            details={
                "run_spec": {"id": smoke_id, "identity": smoke_spec},
                "aggregate": smoke_aggregate,
            },
        )
        persist_receipt(
            "smoke",
            writers.build_evidenced_prerequisite_receipt(
                gate="smoke",
                bindings=common_bindings,
                evidence=smoke_evidence,
            ),
            smoke_evidence,
        )

    replicates = 1000 if args.variant == "final" else 100
    plan = build_bootstrap_plan(paired_items, replicates=replicates, seed=1)
    print("== bootstrap plan ==", "reps", replicates, "items", plan.n_items)

    run_spec = run_spec_identity(
        source_manifest_id=source_id, raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        adapter_bundle_id=adapter_id, fvi_study_id=fvi_id, bootstrap_plan_id=compute_id(_plan_ident(plan)),
        environment_contract_id=env_id,
        resource_summary_id=compute_id({"backend": "local"}),
        fvi_selected=selected, replicate_count=replicates,
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
    """Load one canonical content manifest and rehash its exhaustive inventory."""
    from scripts.stopdff_v5.content_manifest import (
        validate_bound_content_manifest,
    )

    return validate_bound_content_manifest(
        base,
        manifest_name=manifest_name,
        expected_id=None,
        expected_kind=expected_kind,
        file_key=file_key,
        name_key=name_key,
        content_subdir=content_subdir,
        require_semantic_pass=expected_kind == "raw_input_bundle",
    )


def _next_resume_attempt(
    run_root: Path,
    *,
    run_spec_id: str,
    adapter_id: str,
    bootstrap_plan_id: str,
) -> int:
    """Validate the append-only started records and derive the next attempt."""
    attempts_path = Path(run_root) / "attempts.jsonl"
    try:
        _, records = load_attempt_history(attempts_path)
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("resume requires a canonical attempts.jsonl") from exc
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
    if source_manifest["identity"].get("git_sha") != run_sha:
        raise ValueError("resume source snapshot does not match executing commit")
    _verified_local_source_execution(args.repo_root, source_manifest)
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
    if args.variant == "final":
        for gate, receipt in receipts.items():
            evidence_path = (
                out
                / "receipts"
                / gate
                / f"{receipt['id']}.evidence.json"
            )
            if evidence_path.is_symlink() or not evidence_path.is_file():
                raise ValueError(
                    f"resume prerequisite evidence is missing: {gate}"
                )
            writers.verify_prerequisite_evidence_bytes(
                gate=gate,
                bindings=receipt["identity"]["bindings"],
                receipt_evidence=receipt["identity"]["evidence"],
                data=evidence_path.read_bytes(),
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
