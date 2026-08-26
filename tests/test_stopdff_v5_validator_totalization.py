"""Regression controls for the PR #30 successor integrity repair."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_stopdff_v5_local as local_runner
from scripts.stopdff_v5 import checker, locking, selftest, writers
from tests.harness_control_plane import (
    _fake_control_api,
    _load_modal_runner,
    _plan,
)


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "scripts" / "validate_stopdff_bucketed_sweep.py"


def _validate_package(built: dict) -> checker.CheckResult:
    return checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=True,
    )


@pytest.mark.parametrize(
    "mutation",
    ["markdown", "latex", "figure", "missing_figure", "extra_figure"],
)
def test_package_display_bytes_are_bound_to_validated_inputs(
    tmp_path: Path,
    mutation: str,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    baseline = _validate_package(built)
    assert baseline.passed, baseline.errors

    if mutation == "markdown":
        path = run_root / "reports" / "report.md"
        path.write_text(path.read_text(encoding="utf-8") + "false claim\n")
    elif mutation == "latex":
        path = run_root / "reports" / "report.tex"
        path.write_text(path.read_text(encoding="utf-8") + "% false claim\n")
    elif mutation == "figure":
        writers.write_min_png(
            run_root / "figures" / "cell_median_index_shift.png",
            width=7,
            height=5,
            rgb=(200, 10, 20),
        )
    elif mutation == "missing_figure":
        (run_root / "figures" / "cell_median_index_shift.png").unlink()
    else:
        writers.write_min_png(run_root / "figures" / "unrelated.png")
    writers.write_sha256sums(run_root)

    checked = _validate_package(built)
    assert not checked.passed
    assert any(
        "canonical rendered content" in error
        or "missing figures/" in error
        or "unexpected figures/" in error
        for error in checked.errors
    ), checked.errors


def test_reports_render_per_cell_verdicts_with_warn_reason_qualifiers() -> None:
    aggregate = {
        "profile_variant": "smoke",
        "backend": "modal",
        "requested": 4,
        "completed": 3,
        "skipped": 0,
        "failed": 1,
        "fvi_selected": {"tolerance": "1e-8", "max_iterations": 100},
        "cells": {
            "cell_ceiling": {
                "status": "completed",
                "verdict": "WARN",
                "abs_median_point": 0.0,
                "abs_median_ci": [0.0, 0.5],
                "coverage_clean": True,
                "ceiling_any": True,
            },
            "cell_coverage_ci": {
                "status": "completed",
                "verdict": "WARN",
                "abs_median_point": 0.6,
                "abs_median_ci": [0.2, 1.4],
                "coverage_clean": False,
                "ceiling_any": False,
            },
            "cell_clean": {
                "status": "completed",
                "verdict": "PASS",
                "abs_median_point": 0.0,
                "abs_median_ci": [0.0, 0.4],
                "coverage_clean": True,
                "ceiling_any": False,
            },
            "cell_invalid": {"status": "calibrator_failed", "verdict": "INVALID"},
        },
        "family": {"M": 0.6, "ci": [0.2, 1.4], "verdict": "WARN"},
        "gate_overrides": {
            "allow_low_mc_retention": False,
            "allow_incomplete_mc_coverage": False,
        },
        "release_status": "INVALID",
    }

    markdown = writers.render_markdown(
        aggregate, resource_summary={"backend": "modal"}
    )
    latex = writers.render_latex(aggregate)
    for rendered in (markdown, latex):
        assert "WARN (ceiling)" in rendered
        assert "WARN (coverage, ci_above_threshold)" in rendered
    # Qualified verdicts stay per-cell: clean and invalid cells render bare.
    assert "- cell_clean:" not in markdown
    assert "cell clean & PASS \\\\" in latex
    assert "cell invalid & INVALID \\\\" in latex

    # An active MC gate override is a run-level qualifier on completed cells.
    overridden = json.loads(json.dumps(aggregate))
    overridden["gate_overrides"]["allow_low_mc_retention"] = True
    overridden["cells"]["cell_clean"]["verdict"] = "WARN"
    assert "WARN (ceiling, override)" in writers.render_latex(overridden)
    assert "WARN (override)" in writers.render_markdown(
        overridden, resource_summary={"backend": "modal"}
    )


def test_validate_run_totalizes_over_nested_json(tmp_path: Path) -> None:
    built = selftest.build_valid_package(tmp_path)
    depth = 10_000
    (built["run_root"] / "aggregate.json").write_text(
        '{"nested":' * depth + "0" + "}" * depth,
        encoding="utf-8",
    )

    checked = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )
    assert not checked.passed
    assert any("RecursionError" in error for error in checked.errors)

    command = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "validate",
            str(built["run_root"]),
            "--backend",
            "modal",
            "--adapter-bundle",
            str(built["adapter_bundle"]),
            "--json",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert command.returncode == 1
    assert command.stderr == ""
    payload = json.loads(command.stdout)
    assert payload["schema_version"] == 1
    assert payload["passed"] is False
    assert any("RecursionError" in error for error in payload["errors"])


def test_validate_run_totalizes_numeric_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = selftest.build_valid_package(tmp_path)

    def overflow(*_args, **_kwargs):
        raise OverflowError("adversarial finite-range value")

    monkeypatch.setattr(checker, "compute_cell", overflow)
    checked = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )
    assert not checked.passed
    assert checked.errors == [
        "run evidence cannot be validated safely: "
        "OverflowError: adversarial finite-range value"
    ]


def test_package_rejects_rewritten_resource_claims(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    forged = {
        "backend": "forged",
        "claim": "live Modal execution certified",
        "gpu_hours": 999_999,
    }
    (run_root / "resource_summary.json").write_text(
        json.dumps(forged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = checker.load_json(run_root / "aggregate.json")
    (run_root / "reports" / "report.md").write_text(
        writers.render_markdown(aggregate, resource_summary=forged),
        encoding="utf-8",
    )
    writers.write_sha256sums(run_root)

    checked = _validate_package(built)
    assert not checked.passed
    assert any("resource_summary_id" in error for error in checked.errors)


def test_malformed_aggregate_cannot_escape_package_renderer(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    aggregate_path = built["run_root"] / "aggregate.json"
    aggregate = checker.load_json(aggregate_path)
    aggregate["cells"] = []
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checked = _validate_package(built)
    assert not checked.passed
    assert any(
        "canonical reports/figures cannot be regenerated" in error
        for error in checked.errors
    )


def test_unhashable_run_spec_variant_is_structured_failure(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    spec_path = built["run_root"] / "run_spec.json"
    spec = checker.load_json(spec_path)
    spec["identity"]["profile_variant"] = []
    spec_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checked = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )
    assert not checked.passed
    assert checked.errors


def test_validate_spec_totalizes_over_nested_json(tmp_path: Path) -> None:
    spec_path = tmp_path / "run_spec.json"
    depth = 10_000
    spec_path.write_text(
        '{"nested":' * depth + "0" + "}" * depth,
        encoding="utf-8",
    )

    checked = checker.validate_spec(
        spec_path,
        require_final_profile=False,
    )
    assert not checked.passed
    assert any("RecursionError" in error for error in checked.errors)

    command = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "validate-spec",
            str(spec_path),
            "--json",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert command.returncode == 1
    assert command.stderr == ""
    assert json.loads(command.stdout)["passed"] is False


def test_validate_adapter_totalizes_truncated_gzip(tmp_path: Path) -> None:
    built = selftest.build_valid_package(tmp_path)
    (built["adapter_bundle"] / "fit_rows.jsonl.gz").write_bytes(
        b"\x1f\x8b\x08\x00"
    )

    checked = checker.validate_adapter(built["adapter_bundle"])
    assert not checked.passed
    assert checked.errors

    command = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "validate-adapter",
            str(built["adapter_bundle"]),
            "--json",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert command.returncode == 1
    assert command.stderr == ""
    assert json.loads(command.stdout)["passed"] is False


def test_invalid_fvi_limit_never_reaches_cell_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    spec_path = built["run_root"] / "run_spec.json"
    spec = checker.load_json(spec_path)
    spec["identity"]["fvi_selected"]["max_iterations"] = 10**12
    spec_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def forbidden_compute(*_args, **_kwargs):
        raise AssertionError("invalid FVI limit reached compute_cell")

    monkeypatch.setattr(checker, "compute_cell", forbidden_compute)
    checked = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )
    assert not checked.passed
    assert any("FVI" in error for error in checked.errors)


def test_invalid_replicate_limit_never_builds_bootstrap_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    plan_path = built["run_root"] / "bootstrap_plan.json"
    plan = checker.load_json(plan_path)
    plan["identity"]["replicate_count"] = 10**12
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def forbidden_plan(*_args, **_kwargs):
        raise AssertionError("invalid replicate limit reached plan builder")

    monkeypatch.setattr(checker, "build_bootstrap_plan", forbidden_plan)
    checked = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )
    assert not checked.passed
    assert any("replicate_count" in error for error in checked.errors)


def test_unbound_item_universe_never_builds_bootstrap_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    plan_path = built["run_root"] / "bootstrap_plan.json"
    plan = checker.load_json(plan_path)
    plan["item_ids"] = sorted([*plan["item_ids"], "bogus-item-id"])
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def forbidden_plan(*_args, **_kwargs):
        raise AssertionError("unbound item universe reached plan builder")

    monkeypatch.setattr(checker, "build_bootstrap_plan", forbidden_plan)
    checked = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )
    assert not checked.passed
    assert any("paired adapter eval items" in error for error in checked.errors)


def test_stage_tree_is_fsynced_before_directory_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    out = tmp_path / "out"
    out.mkdir()
    events: list[tuple[str, str]] = []
    descriptors: dict[int, str] = {}
    real_open = local_runner.os.open
    real_close = local_runner.os.close
    real_fsync = local_runner.os.fsync
    real_rename = local_runner.os.rename

    def tracked_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        descriptors[descriptor] = str(path)
        return descriptor

    def tracked_fsync(descriptor):
        events.append(("fsync", descriptors.get(descriptor, "file-handle")))
        return real_fsync(descriptor)

    def tracked_close(descriptor):
        try:
            return real_close(descriptor)
        finally:
            descriptors.pop(descriptor, None)

    # _publish_stage_directory now publishes via fileio.publish_dir_create_once,
    # whose publication step is os.rename (create-once: os.mkdir claim + rename).
    # local_runner.os is the shared os module, so patching it here also captures
    # the primitive's rename/fsync in fileio.
    def tracked_rename(src, dst, *args, **kwargs):
        events.append(("rename", str(dst)))
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(local_runner.os, "open", tracked_open)
    monkeypatch.setattr(local_runner.os, "fsync", tracked_fsync)
    monkeypatch.setattr(local_runner.os, "close", tracked_close)
    monkeypatch.setattr(local_runner.os, "rename", tracked_rename)

    def build(staged: Path) -> str:
        nested = staged / "nested"
        nested.mkdir(parents=True)
        (nested / "payload.bin").write_bytes(b"durable payload")
        return "built"

    result = local_runner._publish_stage_directory(
        out=out,
        target_name="source",
        build=build,
    )
    assert result == "built"
    publish_index = next(
        index for index, event in enumerate(events) if event[0] == "rename"
    )
    before = events[:publish_index]
    after = events[publish_index + 1:]
    assert any("payload.bin" in path for kind, path in before if kind == "fsync")
    assert any("nested" in path for kind, path in before if kind == "fsync")
    # POSIX exposes the parent-directory descriptor and retains the historical
    # post-rename fsync.  Python on Windows rejects directory os.open; the
    # platform helper returns only for that unsupported operation.
    if os.name != "nt":
        assert any(kind == "fsync" for kind, _path in after)
    assert (
        out / "source" / "nested" / "payload.bin"
    ).read_bytes() == b"durable payload"


def test_publish_stage_directory_fails_closed_when_peer_wins_empty_slot(
    tmp_path: Path,
) -> None:
    """A peer or stale supervisor can create the same empty stage slot in the
    window between the entry precheck and publication. Publication must fail
    closed (create-once) rather than let os.replace silently overwrite the
    empty peer directory and claim a slot that was no longer absent."""
    out = tmp_path / "out"
    out.mkdir()

    def build(staged: Path) -> str:
        staged.mkdir()
        (staged / "manifest.json").write_text("{}", encoding="utf-8")
        # A peer wins the slot AFTER the entry precheck, inside the
        # build->publish window, so the precheck cannot mask the race.
        (out / "source").mkdir()
        return "built"

    with pytest.raises(FileExistsError):
        local_runner._publish_stage_directory(
            out=out,
            target_name="source",
            build=build,
        )
    # The peer's empty directory is left intact, never replaced by staged bytes.
    assert (out / "source").is_dir()
    assert list((out / "source").iterdir()) == []


def test_control_journal_rejects_ambiguous_historical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {"status": "initialized", "sequence": 0}
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )
    runner._record_control_event(
        state_path,
        state,
        event="stage_started",
        stage="verify_source",
        detail={"attempt": 1},
    )
    journal = state_path.with_name("control.json.jsonl")
    lines = journal.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace(
        '"event": "control_initialized"',
        '"event": "control_initialized", "event": "control_initialized"',
    )
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        runner._reconcile_control_journal(
            state_path,
            runner._load_control_json(state_path),
        )


def test_control_plane_uses_only_post_package_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api()
    state = runner.run_control_plane(
        _plan(ids),
        tmp_path / "control.json",
        resume=False,
        stage_api=api,
    )
    assert state["status"] == "completed"
    assert [name for name, _args in calls].count("validate") == 1
    assert "validate_unpacked" not in state["completed"]
    assert "validate_package" in state["completed"]


def test_schema_v3_control_state_fails_closed_before_remote_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api()
    state_path = tmp_path / "control.json"
    state = {
        "schema_version": 3,
        "plan": _plan(ids),
        "plan_digest": runner._control_plan_digest(_plan(ids)),
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    state_path.with_name("control.json.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported control-state schema"):
        runner.run_control_plane(
            _plan(ids),
            state_path,
            resume=True,
            stage_api=api,
        )
    assert calls == []


@pytest.mark.parametrize(
    "resource_summary",
    [
        {"backend": "modal", "usd": 0.25},
        {"backend": "modal", "usd": float("nan")},
    ],
)
def test_noncanonical_resource_summary_fails_before_remote_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_summary: dict,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api()
    plan = _plan(ids)
    plan["resource_summary"] = resource_summary

    with pytest.raises(ValueError, match="identity-safe"):
        runner.run_control_plane(
            plan,
            tmp_path / "control.json",
            resume=False,
            stage_api=api,
        )
    assert calls == []


@pytest.mark.parametrize("invalid_stage", ["invented_stage", []])
def test_schema_v3_journal_binds_terminal_status_and_known_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_stage: object,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    api, _calls, ids = _fake_control_api()
    state_path = tmp_path / "control.json"
    state = runner.run_control_plane(
        _plan(ids),
        state_path,
        resume=False,
        stage_api=api,
    )
    state["status"] = "running"
    runner._write_control_state(state_path, state)

    with pytest.raises(ValueError, match="status disagrees"):
        runner._reconcile_control_journal(
            state_path,
            runner._load_control_json(state_path),
        )

    invalid = {
        "sequence": 2,
        "event": "stage_started",
        "stage": invalid_stage,
        "utc_epoch_seconds": 1,
        "detail": {"attempt": 1},
        "previous_event_sha256": None,
    }
    with pytest.raises(ValueError, match="stage is invalid"):
        runner._validate_control_event_record(
            invalid,
            expected_sequence=2,
            previous_record=None,
        )


def test_lifecycle_lock_fails_fast_when_another_driver_holds_it(
    tmp_path, monkeypatch
):
    """The local driver refuses to race another process for the journal."""
    import os

    monkeypatch.setattr(local_runner, "_LIFECYCLE_LOCK_FDS", {})
    out = tmp_path / "workspace"
    out.mkdir()
    lock_path = out / "local_lifecycle.json.lock"
    # A separate owner map opens an independent descriptor, just as another
    # process would, through the host's canonical lock backend.
    foreign_locks: dict[str, int] = {}
    try:
        locking.acquire_process_lock(
            lock_path,
            foreign_locks,
            busy_label="foreign lifecycle owner",
        )
        with pytest.raises(
            RuntimeError, match="another local StopDFF driver holds"
        ):
            local_runner._acquire_lifecycle_lock(out)
        assert local_runner._LIFECYCLE_LOCK_FDS == {}
    finally:
        os.close(foreign_locks.pop(os.path.realpath(lock_path)))

    local_runner._acquire_lifecycle_lock(out)
    assert os.path.realpath(lock_path) in local_runner._LIFECYCLE_LOCK_FDS
    # Sequential re-entry by the same process reuses the held lock instead
    # of self-deadlocking (flock conflicts across open-file-descriptions).
    local_runner._acquire_lifecycle_lock(out)


def test_lifecycle_journal_creation_is_guarded_by_the_driver_lock(
    tmp_path, monkeypatch
):
    import os
    import types

    monkeypatch.setattr(local_runner, "_LIFECYCLE_LOCK_FDS", {})
    out = tmp_path / "workspace"
    out.mkdir()
    args = types.SimpleNamespace(
        variant="smoke",
        skip_fvi_study=False,
        fvi_tolerance="1e-6",
        fvi_max_iterations=100,
        allow_low_mc_retention=False,
    )
    lock_path = out / "local_lifecycle.json.lock"
    foreign_locks: dict[str, int] = {}
    try:
        locking.acquire_process_lock(
            lock_path,
            foreign_locks,
            busy_label="foreign lifecycle owner",
        )
        with pytest.raises(
            RuntimeError, match="another local StopDFF driver holds"
        ):
            local_runner._load_or_create_lifecycle(
                out=out, args=args, run_sha="a" * 40, resume=False
            )
        # Fail-fast happens before any journal byte exists.
        assert not (out / "local_lifecycle.json").exists()
    finally:
        os.close(foreign_locks.pop(os.path.realpath(lock_path)))

    state = local_runner._load_or_create_lifecycle(
        out=out, args=args, run_sha="a" * 40, resume=False
    )
    assert state["adapter_executions"] == {}
    assert (out / "local_lifecycle.json").exists()
    assert os.path.realpath(lock_path) in local_runner._LIFECYCLE_LOCK_FDS


def test_lifecycle_lock_reentrant_across_symlinked_spellings(tmp_path, monkeypatch):
    """Re-entrancy keys on file identity, not text: a second acquire reached
    through a symlinked spelling of the same workspace reuses the held fd
    instead of opening a new descriptor that self-conflicts on LOCK_EX and
    spuriously fails fast. Mirrors the control-plane twin
    (test_control_plane_lock_reentrant_across_symlinked_spellings); the local
    lifecycle lock lacked this coverage (L-V4-01, PR #30 round 4)."""
    import os

    monkeypatch.setattr(local_runner, "_LIFECYCLE_LOCK_FDS", {})
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    key = os.path.realpath(real_dir / "local_lifecycle.json.lock")
    try:
        local_runner._acquire_lifecycle_lock(real_dir)
        assert key in local_runner._LIFECYCLE_LOCK_FDS
        held_fd = local_runner._LIFECYCLE_LOCK_FDS[key]

        # Different text, same inode (link_dir -> real_dir): the second acquire
        # must be a no-op that reuses the held fd. Keyed on the raw textual path
        # this would open a second descriptor and fail fast on LOCK_EX.
        local_runner._acquire_lifecycle_lock(link_dir)
        assert local_runner._LIFECYCLE_LOCK_FDS[key] == held_fd
        assert list(local_runner._LIFECYCLE_LOCK_FDS) == [key]
    finally:
        fd = local_runner._LIFECYCLE_LOCK_FDS.pop(key, None)
        if fd is not None:
            os.close(fd)
