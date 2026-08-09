"""Regression controls for the PR #30 successor integrity repair."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_stopdff_v5_local as local_runner
from scripts.stopdff_v5 import checker, selftest, writers
from tests.test_pr30_control_repairs import _fake_control_api, _load_modal_runner
from tests.test_pr30_modal_recovery_v6 import _plan


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
    out = tmp_path / "out"
    out.mkdir()
    events: list[tuple[str, str]] = []
    descriptors: dict[int, str] = {}
    real_open = local_runner.os.open
    real_close = local_runner.os.close
    real_fsync = local_runner.os.fsync
    real_replace = Path.replace

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

    def tracked_replace(path, target):
        events.append(("replace", str(target)))
        return real_replace(path, target)

    monkeypatch.setattr(local_runner.os, "open", tracked_open)
    monkeypatch.setattr(local_runner.os, "fsync", tracked_fsync)
    monkeypatch.setattr(local_runner.os, "close", tracked_close)
    monkeypatch.setattr(Path, "replace", tracked_replace)

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
    replace_index = next(
        index for index, event in enumerate(events) if event[0] == "replace"
    )
    before = events[:replace_index]
    after = events[replace_index + 1:]
    assert any("payload.bin" in path for kind, path in before if kind == "fsync")
    assert any("nested" in path for kind, path in before if kind == "fsync")
    assert any(kind == "fsync" for kind, _path in after)
    assert (out / "source" / "nested" / "payload.bin").read_bytes() == b"durable payload"


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


def test_schema_v2_control_state_fails_closed_before_remote_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api()
    state_path = tmp_path / "control.json"
    state = {
        "schema_version": 2,
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
