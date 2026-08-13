"""Focused regressions for Round 2 scientific acceptance invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.stopdff_v5 import (
    adapter_build,
    checker,
    checker_package,
    fvi_study,
    selftest,
)
from scripts.stopdff_v5.calibrators import CalibratorFitError
from scripts.stopdff_v5.checker_calibration import platt_phase_errors
from scripts.stopdff_v5.identity import compute_id, sha256_file
from scripts.stopdff_v5.rowio import write_jsonl_gz
from scripts.stopdff_v5.verdicts import ReleaseCheck


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _producer_phase(**updates: object) -> dict:
    block = {
        "ece": 0.025,
        "n_samples": 200,
        "platt_coef": 4.5,
        "platt_intercept": -2.0,
        "platt_model_type": "logistic",
        "platt_fallback_reason": None,
        "platt_constant_probability": None,
    }
    block.update(updates)
    return block


def test_platt_checker_accepts_logistic_and_rejects_constant_phases() -> None:
    assert platt_phase_errors(_producer_phase(), phase="early") == []
    assert "constant model is forbidden" in platt_phase_errors(
        _producer_phase(
            ece=0.0,
            n_samples=0,
            platt_coef=None,
            platt_intercept=None,
            platt_model_type="constant",
            platt_fallback_reason="empty_validation_bucket",
            platt_constant_probability=0.0,
        ),
        phase="mid",
    )[0]
    assert "constant model is forbidden" in platt_phase_errors(
        _producer_phase(
            n_samples=12,
            platt_coef=None,
            platt_intercept=None,
            platt_model_type="constant",
            platt_fallback_reason="single_class_validation_bucket",
            platt_constant_probability=1.0,
        ),
        phase="late",
    )[0]


@pytest.mark.parametrize(
    "updates",
    [
        {"ece": float("nan")},
        {"ece": 1.1},
        {"n_samples": True},
        {"n_samples": -1},
        {"platt_fallback_reason": "single_class_validation_bucket"},
        {
            "n_samples": 0,
            "platt_coef": None,
            "platt_intercept": None,
            "platt_model_type": "constant",
            "platt_fallback_reason": "empty_validation_bucket",
            "platt_constant_probability": 1.0,
        },
    ],
)
def test_platt_checker_rejects_invalid_producer_semantics(updates: dict) -> None:
    assert platt_phase_errors(_producer_phase(**updates), phase="early")


def test_adapter_checker_requires_terminal_prefix_fraction_one(tmp_path: Path) -> None:
    built = selftest.build_valid_package(tmp_path)
    bundle = built["adapter_bundle"]
    rows_path = bundle / "fit_rows.jsonl.gz"
    rows = checker.load_jsonl_gz(rows_path)
    item_id = rows[0]["item_id"]
    terminal_idx = max(
        row["prefix_idx"] for row in rows if row["item_id"] == item_id
    )
    for row in rows:
        if row["item_id"] == item_id and row["prefix_idx"] == terminal_idx:
            row["prefix_fraction"] = 0.9
    write_jsonl_gz(rows_path, rows)

    manifest_path = bundle / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["identity"]["fit_rows_sha256"] = sha256_file(rows_path)
    manifest["id"] = compute_id(manifest["identity"])
    _write_json(manifest_path, manifest)

    result = checker.validate_adapter(bundle)
    assert not result.passed
    assert any(
        "terminal prefix_fraction must be 1.0" in error
        for error in result.errors
    )


def test_fvi_study_stops_when_strict_reference_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def nonconverged_strict(**kwargs):
        calls.append(kwargs)
        return {
            "tolerance": kwargs["tolerance_label"],
            "max_iterations": kwargs["max_iterations"],
            "total_iterations": 1,
            "all_converged": False,
            "cells": {},
        }

    monkeypatch.setattr(
        fvi_study,
        "run_candidate_on_cells",
        nonconverged_strict,
    )
    with pytest.raises(ValueError, match="strict FVI reference"):
        fvi_study.run_fvi_study(rows=[], calibration_json=None)
    assert len(calls) == 1


def test_packaged_fvi_requires_converged_strict_reference(tmp_path: Path) -> None:
    built = selftest.build_valid_package(tmp_path)
    identity = _read_json(
        built["run_root"] / "evidence" / "fvi_study.json"
    )["identity"]
    identity["strict_reference_results"]["all_converged"] = False
    errors: list[str] = []
    checker_package._check_fvi_study_identity(identity, errors)
    assert "packaged FVI strict reference is noncanonical" in errors


def _phase_rows(*, n: int, one_class: bool) -> list[dict]:
    rows = []
    for phase_fraction in (0.1, 0.5, 0.9):
        for index in range(n):
            rows.append(
                {
                    "format": "MC",
                    "split": "val",
                    "prefix_fraction": phase_fraction,
                    "raw_similarity": index / max(1, n),
                    "correct": 1 if one_class else index % 2,
                }
            )
    return rows


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (_phase_rows(n=9, one_class=False), "has 9 rows < 10"),
        (_phase_rows(n=10, one_class=True), "lacks both correctness classes"),
    ],
)
def test_bound_platt_calibration_enforces_profile_phase_eligibility(
    rows,
    message,
) -> None:
    with pytest.raises(CalibratorFitError, match=message):
        adapter_build.derive_bound_calibration(
            fit_rows=rows,
            eval_rows=[],
            model_snapshot_id="a" * 64,
            fit_rows_sha256="b" * 64,
        )


def test_rehashed_terminal_prefix_must_equal_bound_full_question(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    bundle = built["adapter_bundle"]
    rows_path = bundle / "fit_rows.jsonl.gz"
    rows = checker.load_jsonl_gz(rows_path)
    item_id = rows[0]["item_id"]
    terminal_idx = max(
        row["prefix_idx"] for row in rows if row["item_id"] == item_id
    )
    for row in rows:
        if row["item_id"] == item_id and row["prefix_idx"] == terminal_idx:
            row["prefix_text_sha256"] = "f" * 64
    write_jsonl_gz(rows_path, rows)

    manifest_path = bundle / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["identity"]["fit_rows_sha256"] = sha256_file(rows_path)
    all_rows = rows + checker.load_jsonl_gz(bundle / "eval_rows.jsonl.gz")
    manifest["identity"]["question_trajectory_binding_id"] = (
        adapter_build.question_trajectory_binding_from_rows(all_rows)
    )
    calibration = adapter_build.derive_bound_calibration(
        fit_rows=rows,
        eval_rows=checker.load_jsonl_gz(bundle / "eval_rows.jsonl.gz"),
        model_snapshot_id=manifest["identity"]["model_snapshot_id"],
        fit_rows_sha256=manifest["identity"]["fit_rows_sha256"],
    )
    _write_json(bundle / "calibration.json", calibration)
    manifest["identity"]["calibration_sha256"] = sha256_file(
        bundle / "calibration.json"
    )
    manifest["id"] = compute_id(manifest["identity"])
    _write_json(manifest_path, manifest)

    result = checker.validate_adapter(bundle)
    assert not result.passed
    assert any(
        "terminal prefix is not bound to the full question" in error
        for error in result.errors
    )


def test_invalid_recomputed_release_fails_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    aggregate_path = built["run_root"] / "aggregate.json"
    aggregate = _read_json(aggregate_path)
    aggregate["release_status"] = "INVALID"
    _write_json(aggregate_path, aggregate)
    monkeypatch.setattr(
        checker,
        "release_validity",
        lambda **_kwargs: ReleaseCheck(
            valid=False,
            reasons=["forced scientific failure"],
        ),
    )

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )
    assert not result.passed
    assert "release invalid: forced scientific failure" in result.errors


def test_family_contract_requires_ci_and_exact_fields(tmp_path: Path) -> None:
    built = selftest.build_valid_package(tmp_path)
    aggregate_path = built["run_root"] / "aggregate.json"
    aggregate = _read_json(aggregate_path)
    del aggregate["family"]["ci"]
    _write_json(aggregate_path, aggregate)

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )
    assert not result.passed
    assert "family fields do not match the canonical contract" in result.errors
    assert "family CI mismatch" in result.errors


def test_adapter_retention_override_must_propagate_to_run_gate(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    bundle = built["adapter_bundle"]
    manifest_path = bundle / "manifest.json"
    manifest = _read_json(manifest_path)
    decision = manifest["identity"]["mc_retention_evidence"]["splits"]["fit"]
    retained = decision["retained_count"]
    decision.update(
        {
            "threshold": "1.0",
            "retention_rate": repr(retained / (retained + 1)),
            "raw_count": retained + 1,
            "dropped_count": 1,
            "passed": False,
            "overridden": True,
            "effective_pass": True,
        }
    )
    manifest["id"] = compute_id(manifest["identity"])
    _write_json(manifest_path, manifest)

    spec_path = built["run_root"] / "run_spec.json"
    spec = _read_json(spec_path)
    spec["identity"]["identity"]["adapter_bundle_id"] = manifest["id"]
    spec["identity"]["gate"]["allow_low_mc_retention"] = False
    spec["id"] = compute_id(spec["identity"])
    plan = _read_json(built["run_root"] / "bootstrap_plan.json")

    with pytest.raises(
        ValueError,
        match="adapter low-retention override is not enabled by the run gate",
    ):
        checker.resolve_run_binding(
            run_spec_manifest=spec,
            adapter_bundle=bundle,
            bootstrap_plan_manifest=plan,
        )
