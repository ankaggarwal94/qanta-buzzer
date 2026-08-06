"""Focused regressions for Round 2 scientific acceptance invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.stopdff_v5 import checker, selftest
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


def test_platt_checker_accepts_actual_seven_field_producer_shapes() -> None:
    assert platt_phase_errors(_producer_phase(), phase="early") == []
    assert platt_phase_errors(
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
    ) == []
    assert platt_phase_errors(
        _producer_phase(
            n_samples=12,
            platt_coef=None,
            platt_intercept=None,
            platt_model_type="constant",
            platt_fallback_reason="single_class_validation_bucket",
            platt_constant_probability=1.0,
        ),
        phase="late",
    ) == []


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
