"""Identity-graph regressions for the StopDFF v5 adapter and run package."""
from __future__ import annotations

import json
import gzip
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import checker, identity, selftest  # noqa: E402


def _rewrite_adapter_manifest(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _mutate_first_fit_row(bundle: Path, field: str, value) -> None:
    fit_path = bundle / "fit_rows.jsonl.gz"
    rows = [
        json.loads(line)
        for line in gzip.decompress(fit_path.read_bytes())
        .decode("utf-8")
        .splitlines()
        if line
    ]
    rows[0][field] = value
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    fit_path.write_bytes(gzip.compress(payload, mtime=0))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["fit_rows_sha256"] = identity.sha256_file(fit_path)
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("fit_row_count", 999_999),
        ("schema_columns", ["item_id"]),
        ("fit_split", "test"),
        ("eval_split", "val"),
    ],
)
def test_adapter_counts_schema_and_split_roles_derive_from_row_bytes(
    tmp_path,
    field,
    replacement,
):
    """Recomputing a manifest ID cannot bless false claims about its row bytes."""
    built = selftest.build_valid_package(tmp_path)
    manifest_path = built["adapter_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = replacement
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed, field


def test_adapter_manifest_binds_calibration_bytes(tmp_path):
    """Calibration bytes are a scientific adapter input and need a hash edge."""
    built = selftest.build_valid_package(tmp_path)
    calibration = built["adapter_bundle"] / "calibration.json"
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    payload["tampered"] = True
    calibration.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any("calibration sha mismatch" in error for error in result.errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("item_id", 17),
        ("prefix_idx", True),
        ("prefix_fraction", float("nan")),
        ("prefix_fraction", 1.01),
        ("prefix_fraction", 10**1000),
        ("correct", True),
        ("correct", 2),
    ],
)
def test_adapter_rejects_rehashed_invalid_row_domains(tmp_path, field, value):
    built = selftest.build_valid_package(tmp_path)
    _mutate_first_fit_row(built["adapter_bundle"], field, value)

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed, (field, value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scoring_spec", {"model_id": "forged"}),
        ("producer_hashes", {"adapter_build.py": "not-a-hash"}),
    ],
)
def test_adapter_rejects_rehashed_noncanonical_producer_contract(
    tmp_path,
    field,
    value,
):
    built = selftest.build_valid_package(tmp_path)
    manifest_path = built["adapter_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = value
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed, field


def test_adapter_invalid_count_is_rejected_without_checker_exception(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    manifest_path = built["adapter_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["fit_row_count"] = {"not": "an integer"}
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any("fit_row_count" in error for error in result.errors)


def test_adapter_calibration_semantics_are_canonical(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    calibration_path = built["adapter_bundle"] / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["per_bucket"]["early"]["platt_coef"] = float("nan")
    calibration_path.write_text(
        json.dumps(calibration, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path = built["adapter_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["calibration_sha256"] = identity.sha256_file(
        calibration_path
    )
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any("calibration" in error for error in result.errors)
