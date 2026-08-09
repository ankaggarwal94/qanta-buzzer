"""Regression controls for the final PR #30 P0 repair transaction."""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.stopdff_v5 import checker, profile, selftest, sweep
from tests.test_pr30_control_repairs import _fake_control_api, _load_modal_runner
from tests.test_pr30_modal_recovery_v6 import _plan
from tests.test_stopdff_v5_pipeline import _make_ctx, _synth_rows


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "scripts" / "validate_stopdff_bucketed_sweep.py"


def _interrupt_after_initial_publication(tmp_path: Path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    ctx = _make_ctx(tmp_path, rows, cells)
    publications = 0

    def interrupt() -> None:
        nonlocal publications
        publications += 1
        if publications == 1:
            raise RuntimeError("interrupted after initial publication")

    ctx.commit_fn = interrupt
    with pytest.raises(RuntimeError, match="after initial publication"):
        sweep.run_sweep(ctx)
    return rows, cells, ctx.output_dir


def test_resume_rejects_symlinked_cells_parent_before_any_mutation(
    tmp_path: Path,
) -> None:
    rows, cells, run_root = _interrupt_after_initial_publication(tmp_path)
    attempts_path = run_root / "attempts.jsonl"
    attempts_before = attempts_path.read_bytes()
    external = tmp_path / "external-cells"
    external.mkdir()
    sentinel = external / "sentinel.bin"
    sentinel.write_bytes(b"outside-run\x00sentinel")
    (run_root / "cells").symlink_to(external, target_is_directory=True)

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    with pytest.raises(ValueError, match="cells directory.*symlink"):
        sweep.run_sweep(resumed)

    assert attempts_path.read_bytes() == attempts_before
    assert sentinel.read_bytes() == b"outside-run\x00sentinel"
    assert sorted(path.name for path in external.iterdir()) == ["sentinel.bin"]
    assert not (run_root / "attempt_results").exists()
    assert (run_root / "cells").is_symlink()


def test_resume_rejects_regular_file_cells_before_any_mutation(
    tmp_path: Path,
) -> None:
    rows, cells, run_root = _interrupt_after_initial_publication(tmp_path)
    attempts_path = run_root / "attempts.jsonl"
    attempts_before = attempts_path.read_bytes()
    (run_root / "cells").write_bytes(b"not a directory")

    resumed = _make_ctx(tmp_path, rows, cells)
    resumed.resume = True
    resumed.attempt = {
        "attempt": 2,
        "mode": "resume",
        "command": ["dp_sweep", "--resume"],
    }
    with pytest.raises(ValueError, match="cells path.*directory"):
        sweep.run_sweep(resumed)

    assert attempts_path.read_bytes() == attempts_before
    assert (run_root / "cells").read_bytes() == b"not a directory"
    assert not (run_root / "attempt_results").exists()


def test_control_plan_rejects_embedded_nul_before_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    api, _calls, ids = _fake_control_api()
    plan = _plan(ids)
    plan["adapter_subdirs"] = ["build\x00a", "build_b"]
    state_path = tmp_path / "control.json"

    with pytest.raises(ValueError, match="NUL"):
        runner.run_control_plane(
            plan,
            state_path,
            resume=False,
            stage_api=api,
        )

    assert not state_path.exists()
    assert not state_path.with_name(state_path.name + ".jsonl").exists()


@pytest.mark.parametrize("attempt", [1, 2])
def test_adapter_attempt_subdirs_reject_embedded_nul(
    monkeypatch: pytest.MonkeyPatch,
    attempt: int,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    with pytest.raises(ValueError, match="NUL"):
        runner._adapter_attempt_subdirs(["build\x00a", "build_b"], attempt)


def _first_cell(run_root: Path) -> Path:
    return sorted((run_root / "cells").glob("*.json"))[0]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_checksum(run_root: Path, changed_path: Path) -> None:
    relative = changed_path.relative_to(run_root).as_posix()
    digest = hashlib.sha256(changed_path.read_bytes()).hexdigest()
    checksum_path = run_root / "SHA256SUMS"
    lines = [
        line
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if not line.endswith(f"  {relative}")
    ]
    lines.append(f"{digest}  {relative}")
    checksum_path.write_text(
        "\n".join(sorted(lines)) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "world",
    [
        "duplicate_cell",
        "nonobject_cell",
        "nonobject_coverage",
        "nonobject_bootstrap",
        "nonobject_index_shifts",
        "noncanonical_fingerprint",
        "nonobject_aggregate_cells",
        "nonobject_family",
        "noncoercible_skipped",
        "nonobject_backend_manifest",
        "nonobject_backend_identity",
    ],
)
def test_validate_run_normalizes_malformed_evidence(
    tmp_path: Path,
    world: str,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    cell_path = _first_cell(run_root)

    if world == "duplicate_cell":
        cell_path.write_text('{"cell_key":"a","cell_key":"b"}\n', encoding="utf-8")
    elif world == "nonobject_cell":
        cell_path.write_text("[]\n", encoding="utf-8")
    elif world in {
        "nonobject_coverage",
        "nonobject_bootstrap",
        "nonobject_index_shifts",
        "noncanonical_fingerprint",
    }:
        cell = checker.load_json(cell_path)
        if world == "nonobject_coverage":
            cell["coverage"] = []
        elif world == "nonobject_bootstrap":
            cell["bootstrap"] = []
        elif world == "nonobject_index_shifts":
            cell["index_shift_by_item"] = []
        else:
            cell["fingerprint_identity"]["noncanonical_float"] = 0.5
        _write_json(cell_path, cell)
    elif world in {
        "nonobject_aggregate_cells",
        "nonobject_family",
        "noncoercible_skipped",
    }:
        aggregate_path = run_root / "aggregate.json"
        aggregate = checker.load_json(aggregate_path)
        if world == "nonobject_aggregate_cells":
            aggregate["cells"] = []
        elif world == "nonobject_family":
            aggregate["family"] = "not-an-object"
        else:
            aggregate["skipped"] = "not-an-integer"
        _write_json(aggregate_path, aggregate)
    else:
        manifest_path = run_root / "run_manifest.json"
        if world == "nonobject_backend_manifest":
            manifest_path.write_text("[]\n", encoding="utf-8")
        else:
            manifest = checker.load_json(manifest_path)
            manifest["identity"] = []
            _write_json(manifest_path, manifest)

    result = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_package=False,
    )

    assert not result.passed
    assert result.errors


def test_validate_run_normalizes_duplicate_key_adapter_row(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    row_path = built["adapter_bundle"] / "fit_rows.jsonl.gz"
    with gzip.open(row_path, "wt", encoding="utf-8") as handle:
        handle.write('{"item_id":"a","item_id":"b"}\n')

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_package=False,
    )

    assert not result.passed
    assert any("adapter rows cannot be decoded" in error for error in result.errors)


def test_validate_json_cli_normalizes_malformed_cell(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    _first_cell(built["run_root"]).write_text(
        '{"cell_key":"a","cell_key":"b"}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
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
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["command"] == "validate"
    assert payload["passed"] is False
    assert payload["errors"]


def _mutate_scientific_record(cell: dict, mutation: str) -> None:
    if mutation == "coverage_fraction":
        cell["coverage"]["primary_fraction"] += 0.125
    elif mutation == "coverage_count":
        cell["coverage"]["decision_points"] += 1
    elif mutation == "descriptive":
        cell["descriptive"]["never_buzz_mc"] += 1
    elif mutation == "fvi":
        cell["fvi"]["iterations"] += 1
    elif mutation == "status":
        cell["status"] = "calibrator_failed"
    elif mutation == "bootstrap_replicate":
        cell["bootstrap"]["abs_median_replicates"][0] += 1.0
    elif mutation == "other_ci":
        cell["bootstrap"]["ci"]["signed_index_mean"][0] += 0.25
    elif mutation == "string_index_shift":
        key = sorted(cell["index_shift_by_item"])[0]
        cell["index_shift_by_item"][key] = str(cell["index_shift_by_item"][key])
    elif mutation == "string_point":
        cell["bootstrap"]["point"]["signed_index_mean"] = str(
            cell["bootstrap"]["point"]["signed_index_mean"]
        )
    elif mutation == "integer_ceiling_flag":
        key = sorted(cell["ceiling_flags"])[0]
        cell["ceiling_flags"][key] = int(cell["ceiling_flags"][key])
    else:
        raise AssertionError(f"unknown mutation {mutation}")


@pytest.mark.parametrize(
    "mutation",
    [
        "coverage_fraction",
        "coverage_count",
        "descriptive",
        "fvi",
        "status",
        "bootstrap_replicate",
        "other_ci",
        "string_index_shift",
        "string_point",
        "integer_ceiling_flag",
    ],
)
def test_checksum_regenerated_scientific_mutation_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    cell_path = _first_cell(built["run_root"])
    cell = checker.load_json(cell_path)
    _mutate_scientific_record(cell, mutation)
    _write_json(cell_path, cell)
    _refresh_checksum(built["run_root"], cell_path)

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_package=True,
    )

    assert not result.passed, mutation
    assert any(cell_path.stem in error for error in result.errors)


@pytest.mark.parametrize(
    "mutation",
    [
        "profile_name",
        "release_reasons",
        "extra_cell_summary",
        "string_skipped",
        "integer_gate_override",
        "numeric_fvi_tolerance",
    ],
)
def test_checksum_regenerated_aggregate_contract_mutation_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    aggregate_path = built["run_root"] / "aggregate.json"
    aggregate = checker.load_json(aggregate_path)
    if mutation == "profile_name":
        aggregate["profile_name"] = "lookalike-profile"
    elif mutation == "release_reasons":
        aggregate["release_reasons"] = ["invented"]
    elif mutation == "extra_cell_summary":
        aggregate["cells"]["extra-cell"] = {"status": "completed"}
    elif mutation == "string_skipped":
        aggregate["skipped"] = "0"
    elif mutation == "integer_gate_override":
        aggregate["gate_overrides"] = {
            key: int(value)
            for key, value in aggregate["gate_overrides"].items()
        }
    else:
        aggregate["fvi_selected"]["tolerance"] = 1e-10
    _write_json(aggregate_path, aggregate)
    _refresh_checksum(built["run_root"], aggregate_path)

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_package=True,
    )

    assert not result.passed, mutation


def test_scientific_control_uses_independent_run_copy(tmp_path: Path) -> None:
    """Sanity-check the checksum helper without mutating the accepted fixture."""
    built = selftest.build_valid_package(tmp_path / "source")
    copied = tmp_path / "copied-run"
    shutil.copytree(built["run_root"], copied)
    assert checker.validate_run(
        copied,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_package=True,
    ).passed
