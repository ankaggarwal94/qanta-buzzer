"""Integration test: full v5 scientific pipeline on synthetic adapter rows."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import (  # noqa: E402
    bootstrap,
    cellcompute,
    checker,
    identity,
    profile,
    selftest,
    sweep,
    writers,
)
from scripts.stopdff_v5.calibrators import CalibratorFitError  # noqa: E402
from scripts.stopdff_v5.fvi import FVIResult  # noqa: E402
from scripts.stopdff_v5.fvi_study import order_eligible  # noqa: E402
from scripts.stopdff_v5.manifests import (  # noqa: E402
    ENVIRONMENT_PACKAGES,
    environment_contract_identity,
    run_spec_identity,
)

CATEGORIES = ["history", "science", "arts"]
PREFIX_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def _synth_rows(n_items: int = 40, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_items):
        qid = f"q{i:03d}"
        split = "val" if i < n_items // 2 else "test"
        cat = CATEGORIES[i % len(CATEGORIES)]
        item_off = rng.uniform(-0.15, 0.15)
        for t, frac in enumerate(PREFIX_FRACS):
            mc_sim = float(np.clip(0.25 + 0.55 * frac + item_off + rng.uniform(-0.05, 0.05), 0.0, 1.0))
            qa_sim = float(np.clip(0.20 + 0.60 * frac + item_off + rng.uniform(-0.05, 0.05), 0.0, 1.0))
            mc_correct = int(mc_sim + rng.uniform(-0.15, 0.15) > 0.55)
            rows.append({
                "item_id": qid, "prefix_idx": t, "prefix_fraction": frac, "format": "MC",
                "split": split, "raw_similarity": mc_sim, "correct": mc_correct, "category": cat,
            })
            rows.append({
                "item_id": qid, "prefix_idx": t, "prefix_fraction": frac, "format": "QA",
                "split": split, "raw_similarity": qa_sim, "correct": 1, "category": cat,
            })
    return rows


def _calibration_json() -> dict:
    block = {"platt_coef": 5.0, "platt_intercept": -2.5}
    return {"per_bucket": {"early": dict(block), "mid": dict(block), "late": dict(block)}}


def _test_item_ids(rows: list[dict]) -> list[str]:
    mc = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "MC"}
    qa = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "QA"}
    return sorted(mc & qa)


def _make_ctx(tmp_path: Path, rows, cells, replicates=100) -> sweep.SweepContext:
    plan = bootstrap.build_bootstrap_plan(_test_item_ids(rows), replicates=replicates, seed=1)
    adapter_id = "e" * 64
    myopic_sha256 = "c" * 64
    producer_hashes = {
        "checker.py": "d" * 64,
        "sweep.py": "f" * 64,
    }
    environment = {
        "python_version": "3.11.0",
        "package_versions": {name: "test" for name in ENVIRONMENT_PACKAGES},
    }
    environment_id = identity.compute_id(
        environment_contract_identity(**environment)
    )
    run_spec = run_spec_identity(
        source_manifest_id="1" * 64,
        raw_input_bundle_id="2" * 64,
        model_snapshot_id="3" * 64,
        adapter_bundle_id=adapter_id,
        fvi_study_id="4" * 64,
        bootstrap_plan_id=identity.compute_id(bootstrap.plan_identity(plan)),
        environment_contract_id=environment_id,
        resource_summary_id=identity.compute_id({"backend": "modal"}),
        fvi_selected={"tolerance": "1e-8", "max_iterations": 100},
        replicate_count=replicates,
        profile_variant="smoke",
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=producer_hashes,
        prerequisite_receipts={},
    )
    return sweep.SweepContext(
        rows=rows, calibration_json=_calibration_json(),
        run_spec=run_spec, run_spec_id=identity.compute_id(run_spec),
        bootstrap_plan=plan, output_dir=tmp_path / "run",
        fvi_tolerance="1e-8", fvi_max_iterations=100, backend="modal",
        profile_variant="smoke", adapter_bundle_id=adapter_id,
        adapter_fit_rows_sha256="a" * 64,
        adapter_eval_rows_sha256="b" * 64,
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=producer_hashes,
        cells=cells,
        environment=environment,
        resource_summary={"backend": "modal"},
        attempt={"attempt": 1, "mode": "fresh", "command": ["dp_sweep"]},
    )


def _run_hard_exit_child(
    tmp_path: Path,
    *,
    exit_after_commit: int,
) -> subprocess.CompletedProcess[str]:
    script = """
import os
import sys
from pathlib import Path

from scripts.stopdff_v5 import profile, sweep
from tests.test_stopdff_v5_pipeline import _make_ctx, _synth_rows

base = Path(sys.argv[1])
exit_after_commit = int(sys.argv[2])
calls = 0

def commit():
    global calls
    calls += 1
    if calls == exit_after_commit:
        os._exit(91)

ctx = _make_ctx(base, _synth_rows(), profile.smoke_cells())
ctx.commit_fn = commit
sweep.run_sweep(ctx)
"""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(exit_after_commit),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )


def test_fresh_initialization_failure_never_exposes_a_canonical_run(
    tmp_path,
    monkeypatch,
):
    ctx = _make_ctx(tmp_path, _synth_rows(), [])
    original_write = sweep.atomic_write_bytes

    def fail_run_spec(path, data):
        if Path(path).name == "run_spec.json":
            raise RuntimeError("interrupted while staging run identity")
        original_write(path, data)

    monkeypatch.setattr(sweep, "atomic_write_bytes", fail_run_spec)
    with pytest.raises(RuntimeError, match="staging run identity"):
        sweep.run_sweep(ctx)
    assert not ctx.output_dir.exists()


def test_first_visible_run_contains_identity_and_attempt_and_is_resumable(
    tmp_path,
):
    rows = _synth_rows()
    ctx = _make_ctx(tmp_path, rows, [])
    publications = 0

    def interrupt_after_first_publication():
        nonlocal publications
        publications += 1
        if publications == 1:
            raise RuntimeError("interrupted after first publication")

    ctx.commit_fn = interrupt_after_first_publication
    with pytest.raises(RuntimeError, match="after first publication"):
        sweep.run_sweep(ctx)

    assert (ctx.output_dir / "run_spec.json").is_file()
    assert (ctx.output_dir / "bootstrap_plan.json").is_file()
    attempts_before = (ctx.output_dir / "attempts.jsonl").read_bytes()
    assert attempts_before.count(b"\n") == 1

    resumed = _make_ctx(tmp_path, rows, [])
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    aggregate = sweep.run_sweep(resumed)
    assert aggregate["requested"] == 0
    assert (ctx.output_dir / "attempts.jsonl").read_bytes().startswith(
        attempts_before
    )


def _resume_package_context(
    built: dict,
    *,
    attempt_number: int,
    commit_fn=None,
) -> sweep.SweepContext:
    run_root = built["run_root"]
    binding = checker.resolve_run_binding(
        run_spec_manifest=checker.load_json(run_root / "run_spec.json"),
        adapter_bundle=built["adapter_bundle"],
        bootstrap_plan_manifest=checker.load_json(
            run_root / "bootstrap_plan.json"
        ),
    )
    evidence_roots = binding["evidence_roots"]
    return sweep.SweepContext(
        rows=binding["rows"],
        calibration_json=binding["calibration"],
        run_spec=binding["run_spec_identity"],
        run_spec_id=binding["run_spec_id"],
        bootstrap_plan=binding["bootstrap_plan"],
        output_dir=run_root,
        fvi_tolerance=binding["fvi_tolerance"],
        fvi_max_iterations=binding["fvi_max_iterations"],
        backend="modal",
        profile_variant=binding["variant"],
        adapter_bundle_id=binding["adapter_bundle_id"],
        adapter_fit_rows_sha256=binding["fit_rows_sha256"],
        adapter_eval_rows_sha256=binding["eval_rows_sha256"],
        myopic_artifact_sha256=evidence_roots["myopic_artifact_sha256"],
        producer_hashes=evidence_roots["producer_hashes"],
        gate_overrides=binding["gate_overrides"],
        cells=profile.smoke_cells(),
        commit_fn=commit_fn,
        environment=checker.load_json(run_root / "environment.json"),
        resource_summary=checker.load_json(
            run_root / "resource_summary.json"
        ),
        attempt={
            "attempt": attempt_number,
            "mode": "resume",
            "command": ["dp_sweep", "--resume"],
        },
        resume=True,
    )


def _fresh_package_context(tmp_path: Path, built: dict) -> sweep.SweepContext:
    ctx = _resume_package_context(built, attempt_number=1)
    ctx.output_dir = tmp_path / "failed_run"
    ctx.resume = False
    ctx.attempt = {
        "attempt": 1,
        "mode": "fresh",
        "command": ["dp_sweep"],
    }
    return ctx


def _assert_failure_pipeline(
    *,
    ctx: sweep.SweepContext,
    adapter_bundle: Path,
    status: str,
    release_reason: str,
) -> None:
    aggregate = sweep.run_sweep(ctx)
    assert aggregate["completed"] == 0
    assert aggregate["failed"] == len(profile.smoke_cells())
    assert aggregate["family"] is None
    assert aggregate["release_status"] == "INVALID"
    assert release_reason in aggregate["release_reasons"]
    assert all(
        summary == {"status": status, "verdict": "INVALID"}
        for summary in aggregate["cells"].values()
    )

    for cell in profile.smoke_cells():
        path = ctx.output_dir / "cells" / f"{profile.cell_key_str(cell)}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["status"] == status
        assert record["reason"]
        if status == "calibrator_failed":
            assert "fvi" not in record
        else:
            assert record["fvi"] == {
                "status": "max_iterations_reached",
                "converged": False,
                "iterations": ctx.fvi_max_iterations,
                "final_delta": 1.0,
            }

    attempt_result = json.loads(
        (ctx.output_dir / "attempt_results" / "1.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt_result == {
        "attempt": 1,
        "state": "completed",
        "run_spec_id": ctx.run_spec_id,
        "completed": 0,
        "failed": len(profile.smoke_cells()),
    }

    checked = checker.validate_run(
        ctx.output_dir,
        backend="modal",
        adapter_bundle=adapter_bundle,
        require_final_profile=False,
        require_package=False,
    )
    assert not checked.passed
    assert checked.recomputed["release_status"] == "INVALID"
    assert "release invalid: not all requested cells completed" in checked.errors
    assert (
        f"release invalid: {len(profile.smoke_cells())} cell(s) failed"
        in checked.errors
    )
    assert f"release invalid: {release_reason}" in checked.errors


def test_calibrator_failure_propagates_through_sweep_and_checker(
    tmp_path,
    monkeypatch,
):
    built = selftest.build_valid_package(tmp_path / "fixture")
    ctx = _fresh_package_context(tmp_path, built)

    def fail_calibrator(*_args, **_kwargs):
        raise CalibratorFitError("forced calibrator failure")

    monkeypatch.setattr(cellcompute, "fit_calibrator", fail_calibrator)
    _assert_failure_pipeline(
        ctx=ctx,
        adapter_bundle=built["adapter_bundle"],
        status="calibrator_failed",
        release_reason="a calibrator failed to fit",
    )


def test_fvi_failure_propagates_through_sweep_and_checker(
    tmp_path,
    monkeypatch,
):
    built = selftest.build_valid_package(tmp_path / "fixture")
    ctx = _fresh_package_context(tmp_path, built)

    def fail_fvi(
        _estimator,
        _trajectories,
        _schedule,
        *,
        tolerance,
        max_iterations,
        tolerance_label="",
        **_kwargs,
    ):
        del tolerance
        return FVIResult(
            status="max_iterations_reached",
            converged=False,
            iterations=max_iterations,
            final_delta=1.0,
            tolerance=tolerance_label,
            max_iterations=max_iterations,
        )

    monkeypatch.setattr(cellcompute, "run_fvi", fail_fvi)
    _assert_failure_pipeline(
        ctx=ctx,
        adapter_bundle=built["adapter_bundle"],
        status="fvi_failed",
        release_reason="an FVI fit did not converge",
    )


def test_full_pipeline_smoke_cells(tmp_path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    ctx = _make_ctx(tmp_path, rows, cells)
    agg = sweep.run_sweep(ctx)

    assert agg["requested"] == 2
    assert agg["completed"] == 2
    assert agg["skipped"] == 0
    assert agg["failed"] == 0
    assert agg["release_status"] == "VALID"
    assert agg["family"] is not None
    assert agg["family"]["verdict"] in {"PASS", "WARN", "FAIL"}
    for key, summary in agg["cells"].items():
        assert summary["verdict"] in {"PASS", "WARN", "FAIL"}
    # backend manifest exclusivity
    assert (ctx.output_dir / "run_manifest.json").exists()
    assert not (ctx.output_dir / "command_manifest.json").exists()
    # per-cell json present and re-derivable
    for cell in cells:
        p = ctx.output_dir / "cells" / f"{profile.cell_key_str(cell)}.json"
        assert p.exists()
        rec = json.loads(p.read_text())
        assert rec["status"] == "completed"
        assert "index_shift_by_item" in rec


def test_pipeline_determinism(tmp_path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    a1 = sweep.run_sweep(_make_ctx(tmp_path / "a", rows, cells))
    a2 = sweep.run_sweep(_make_ctx(tmp_path / "b", rows, cells))
    # Compare scientific content (family + per-cell verdicts/points).
    assert a1["family"] == a2["family"]
    assert a1["cells"] == a2["cells"]


def test_resume_accepts_only_byte_identical_evidence_and_appends_attempt(tmp_path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    first = _make_ctx(tmp_path, rows, cells)
    sweep.run_sweep(first)

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    sweep.run_sweep(resumed)

    attempts = (
        resumed.output_dir / "attempts.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(attempts) == 2


def test_resume_rejects_changed_existing_cell_without_appending_attempt(tmp_path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    first = _make_ctx(tmp_path, rows, cells)
    sweep.run_sweep(first)
    cell_path = sorted((first.output_dir / "cells").glob("*.json"))[0]
    record = json.loads(cell_path.read_text(encoding="utf-8"))
    record["run_spec_id"] = "f" * 64
    cell_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    with pytest.raises(ValueError, match="resume evidence mismatch"):
        sweep.run_sweep(resumed)

    attempts = (
        resumed.output_dir / "attempts.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(attempts) == 1


def test_interrupted_first_attempt_is_durable_and_resumable(tmp_path, monkeypatch):
    rows = _synth_rows()
    ctx = _make_ctx(tmp_path, rows, profile.smoke_cells())

    original = sweep._cell_record
    calls = 0

    def interrupted(context, cell, *, prepared=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated interruption")
        return original(context, cell, prepared=prepared)

    monkeypatch.setattr(sweep, "_cell_record", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sweep.run_sweep(ctx)

    attempts = (ctx.output_dir / "attempts.jsonl").read_text().splitlines()
    assert len(attempts) == 1
    assert json.loads(attempts[0])["state"] == "started"
    failure = json.loads(
        (ctx.output_dir / "attempt_results" / "1.json").read_text()
    )
    assert failure["state"] == "failed"

    monkeypatch.setattr(sweep, "_cell_record", original)
    resumed = _make_ctx(tmp_path, rows, profile.smoke_cells())
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    sweep.run_sweep(resumed)
    completion = json.loads(
        (ctx.output_dir / "attempt_results" / "2.json").read_text()
    )
    assert completion["state"] == "completed"


def test_hard_exit_without_terminal_result_is_classified_create_once_on_resume(
    tmp_path,
):
    """Resume must terminalize only the latest unterminated durable attempt."""
    rows = _synth_rows()
    cells = profile.smoke_cells()
    first = _make_ctx(tmp_path, rows, cells)

    exited = _run_hard_exit_child(tmp_path, exit_after_commit=1)
    assert exited.returncode == 91, (exited.stdout, exited.stderr)

    def hard_exit_after_commit():
        raise SystemExit("simulated hard process exit")

    attempts_path = first.output_dir / "attempts.jsonl"
    original_history = attempts_path.read_bytes()
    assert len(original_history.splitlines()) == 1
    assert not (first.output_dir / "attempt_results" / "1.json").exists()

    # A second hard exit after recovery classification but before attempt 2 is
    # appended must leave the append-only history unchanged and the new
    # classification durable.
    interrupted_resume = _make_ctx(tmp_path, rows, cells)
    interrupted_resume.resume = True
    interrupted_resume.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    interrupted_resume.commit_fn = hard_exit_after_commit
    with pytest.raises(SystemExit, match="simulated hard process exit"):
        sweep.run_sweep(interrupted_resume)

    interrupted_path = first.output_dir / "attempt_results" / "1.json"
    interrupted_bytes = interrupted_path.read_bytes()
    assert attempts_path.read_bytes() == original_history
    assert json.loads(interrupted_bytes) == {
        "attempt": 1,
        "reason": "terminal_result_missing_at_resume",
        "run_spec_id": first.run_spec_id,
        "state": "interrupted",
    }
    intermediate_errors: list[str] = []
    assert not checker._check_attempts(
        first.output_dir,
        intermediate_errors,
        run_spec_id=first.run_spec_id,
        adapter_bundle_id=first.adapter_bundle_id,
        bootstrap_plan_id=identity.compute_id(
            bootstrap.plan_identity(first.bootstrap_plan)
        ),
        aggregate={},
    )
    assert "latest attempt did not complete" in intermediate_errors

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    aggregate = sweep.run_sweep(resumed)

    assert interrupted_path.read_bytes() == interrupted_bytes
    assert len(attempts_path.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(
        (first.output_dir / "attempt_results" / "2.json").read_text()
    )["state"] == "completed"

    errors: list[str] = []
    assert checker._check_attempts(
        first.output_dir,
        errors,
        run_spec_id=first.run_spec_id,
        adapter_bundle_id=first.adapter_bundle_id,
        bootstrap_plan_id=identity.compute_id(
            bootstrap.plan_identity(first.bootstrap_plan)
        ),
        aggregate=aggregate,
    )
    assert errors == []


def test_resume_preflight_corruption_after_hard_exit_writes_nothing(tmp_path):
    exited = _run_hard_exit_child(tmp_path, exit_after_commit=2)
    assert exited.returncode == 91, (exited.stdout, exited.stderr)

    rows = _synth_rows()
    cells = profile.smoke_cells()
    first = _make_ctx(tmp_path, rows, cells)
    cell_path = next((first.output_dir / "cells").glob("*.json"))
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["run_spec_id"] = "0" * 64
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempts_path = first.output_dir / "attempts.jsonl"
    history_before = attempts_path.read_bytes()

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    with pytest.raises(ValueError, match="resume evidence mismatch"):
        sweep.run_sweep(resumed)

    assert attempts_path.read_bytes() == history_before
    assert not (first.output_dir / "attempt_results" / "1.json").exists()
    assert not (first.output_dir / "attempt_results" / "2.json").exists()


def test_resume_rejects_tampered_interruption_without_overwriting(tmp_path):
    exited = _run_hard_exit_child(tmp_path, exit_after_commit=1)
    assert exited.returncode == 91, (exited.stdout, exited.stderr)

    rows = _synth_rows()
    cells = profile.smoke_cells()

    def stop_after_interruption_commit():
        raise SystemExit("stop after interruption classification")

    classifying = _make_ctx(tmp_path, rows, cells)
    classifying.resume = True
    classifying.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    classifying.commit_fn = stop_after_interruption_commit
    with pytest.raises(SystemExit, match="interruption classification"):
        sweep.run_sweep(classifying)

    result_path = classifying.output_dir / "attempt_results" / "1.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["reason"] = "operator_claimed_interruption"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_bytes = result_path.read_bytes()
    history_before = (
        classifying.output_dir / "attempts.jsonl"
    ).read_bytes()

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    with pytest.raises(ValueError, match="noncanonical reason"):
        sweep.run_sweep(resumed)

    assert result_path.read_bytes() == tampered_bytes
    assert (
        classifying.output_dir / "attempts.jsonl"
    ).read_bytes() == history_before
    assert not (classifying.output_dir / "attempt_results" / "2.json").exists()


def test_full_package_accepts_historical_interruption_and_latest_completion(
    tmp_path,
):
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]

    def hard_exit_after_started_commit():
        raise SystemExit("simulated attempt-2 process exit")

    second = _resume_package_context(
        built,
        attempt_number=2,
        commit_fn=hard_exit_after_started_commit,
    )
    with pytest.raises(SystemExit, match="attempt-2 process exit"):
        sweep.run_sweep(second)
    assert not (run_root / "attempt_results" / "2.json").exists()

    third = _resume_package_context(built, attempt_number=3)
    aggregate = sweep.run_sweep(third)
    assert checker.load_json(
        run_root / "attempt_results" / "2.json"
    )["state"] == "interrupted"
    assert checker.load_json(
        run_root / "attempt_results" / "3.json"
    )["state"] == "completed"

    external_artifacts = checker.load_json(
        run_root / "external_artifacts.json"
    )["artifacts"]
    evidence_files = {
        path.relative_to(run_root).as_posix(): path.read_bytes()
        for path in (run_root / "evidence").rglob("*")
        if path.is_file()
    }
    (run_root / "SHA256SUMS").unlink()
    writers.package_run(
        run_root,
        aggregate,
        resource_summary=checker.load_json(
            run_root / "resource_summary.json"
        ),
        external_artifacts=external_artifacts,
        evidence_files=evidence_files,
    )

    checked = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=True,
    )
    assert checked.passed, checked.errors


def test_resume_rejects_ambiguous_nonlatest_missing_terminal_result(tmp_path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    first = _make_ctx(tmp_path, rows, cells)
    sweep.run_sweep(first)

    second = _make_ctx(tmp_path, rows, cells)
    second.resume = True
    second.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    sweep.run_sweep(second)

    first_result = first.output_dir / "attempt_results" / "1.json"
    first_result.unlink()
    history_before = (first.output_dir / "attempts.jsonl").read_bytes()

    third = _make_ctx(tmp_path, rows, cells)
    third.resume = True
    third.attempt = {
        "attempt": 3,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    with pytest.raises(ValueError, match="unterminated attempt history"):
        sweep.run_sweep(third)

    assert (first.output_dir / "attempts.jsonl").read_bytes() == history_before
    assert not first_result.exists()
    assert not (first.output_dir / "attempt_results" / "3.json").exists()


def test_resume_rejects_unhashable_terminal_state_without_writing(tmp_path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    first = _make_ctx(tmp_path, rows, cells)
    sweep.run_sweep(first)

    result_path = first.output_dir / "attempt_results" / "1.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["state"] = []
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    history_before = (first.output_dir / "attempts.jsonl").read_bytes()

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    with pytest.raises(ValueError, match="attempt result mismatch"):
        sweep.run_sweep(resumed)

    assert (first.output_dir / "attempts.jsonl").read_bytes() == history_before
    assert not (first.output_dir / "attempt_results" / "2.json").exists()


def test_resume_rejects_unexpected_attempt_result_entry_without_writing(
    tmp_path,
):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    first = _make_ctx(tmp_path, rows, cells)
    sweep.run_sweep(first)
    unexpected = first.output_dir / "attempt_results" / "orphan.tmp"
    unexpected.write_bytes(b"partial terminal evidence")
    history_before = (first.output_dir / "attempts.jsonl").read_bytes()

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    with pytest.raises(ValueError, match="attempt result is invalid"):
        sweep.run_sweep(resumed)

    assert unexpected.read_bytes() == b"partial terminal evidence"
    assert (first.output_dir / "attempts.jsonl").read_bytes() == history_before
    assert not (first.output_dir / "attempt_results" / "2.json").exists()


def test_resume_preflights_every_existing_cell_before_recreating_any(tmp_path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    first = _make_ctx(tmp_path, rows, cells)
    sweep.run_sweep(first)
    paths = [
        first.output_dir / "cells" / f"{profile.cell_key_str(cell)}.json"
        for cell in cells
    ]
    paths[0].unlink()
    record = json.loads(paths[1].read_text(encoding="utf-8"))
    record["fingerprint_id"] = "f" * 64
    paths[1].write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    with pytest.raises(ValueError, match="resume evidence mismatch"):
        sweep.run_sweep(resumed)

    assert not paths[0].exists()
    assert len(
        (resumed.output_dir / "attempts.jsonl").read_text().splitlines()
    ) == 1


def test_successful_attempt_has_explicit_completion_state(tmp_path):
    rows = _synth_rows()
    ctx = _make_ctx(tmp_path, rows, profile.smoke_cells())
    aggregate = sweep.run_sweep(ctx)
    result = json.loads(
        (ctx.output_dir / "attempt_results" / "1.json").read_text()
    )
    assert result == {
        "attempt": 1,
        "failed": aggregate["failed"],
        "state": "completed",
        "completed": aggregate["completed"],
        "run_spec_id": ctx.run_spec_id,
    }


def test_package_is_create_once_or_byte_identical(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    aggregate = {
        "profile_variant": "smoke",
        "backend": "local",
        "cells": {},
        "family": {},
        "fvi_selected": {},
        "gate_overrides": {},
    }
    artifacts = [{
        "role": "source_manifest",
        "content_id": "1" * 64,
        "sha256": "2" * 64,
        "byte_size": 1,
        "retrieval_path": "source_manifest.json",
    }]
    writers.package_run(
        root,
        aggregate,
        resource_summary={"backend": "local"},
        external_artifacts=artifacts,
    )
    before = (root / "external_artifacts.json").read_bytes()
    writers.package_run(
        root,
        aggregate,
        resource_summary={"backend": "local"},
        external_artifacts=artifacts,
    )
    assert (root / "external_artifacts.json").read_bytes() == before

    report = root / "reports" / "report.md"
    report.write_text("corrupted", encoding="utf-8")
    checksum_before = (root / "SHA256SUMS").read_bytes()
    with pytest.raises(ValueError, match="package evidence mismatch"):
        writers.package_run(
            root,
            aggregate,
            resource_summary={"backend": "local"},
            external_artifacts=artifacts,
        )
    assert (root / "SHA256SUMS").read_bytes() == checksum_before


def test_bad_attempt_binding_fails_before_any_output(tmp_path):
    rows = _synth_rows()
    ctx = _make_ctx(tmp_path, rows, profile.smoke_cells())
    ctx.attempt = {
        "attempt": 1,
        "mode": "fresh",
        "command": ["dp_sweep"],
        "adapter_id": "f" * 64,
    }

    with pytest.raises(ValueError, match="attempt adapter_id"):
        sweep.run_sweep(ctx)
    assert not ctx.output_dir.exists()


def test_selector_ordering_pure():
    cands = [
        {"tolerance": "1e-10", "max_iterations": 200, "total_iterations": 50},
        {"tolerance": "1e-6", "max_iterations": 50, "total_iterations": 20},
        {"tolerance": "1e-8", "max_iterations": 100, "total_iterations": 20},
    ]
    ordered = order_eligible(cands)
    # smallest total_iterations first; tie -> larger tolerance (1e-6 > 1e-8)
    assert ordered[0]["tolerance"] == "1e-6"
    assert ordered[1]["tolerance"] == "1e-8"
    assert ordered[2]["total_iterations"] == 50
