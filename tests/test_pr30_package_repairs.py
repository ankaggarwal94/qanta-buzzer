"""Regression coverage for PR #30 package-contract repairs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import checker, checker_package, selftest, writers  # noqa: E402
from scripts.stopdff_v5.checker_calibration import platt_phase_errors  # noqa: E402
from scripts.stopdff_v5.identity import build_manifest, compute_id  # noqa: E402
from scripts.stopdff_v5.manifests import environment_contract_identity  # noqa: E402


def _dump(path: Path, value: object) -> bytes:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _ledger(
    role: str,
    manifest: dict,
    path: Path,
    *,
    legacy_retrieval: bool = False,
) -> dict:
    entry = {
        "role": role,
        "content_id": manifest["id"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_size": path.stat().st_size,
    }
    entry["retrieval" if legacy_retrieval else "retrieval_path"] = str(path)
    return entry


def test_valid_smoke_package_binds_all_input_manifests(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    root = built["run_root"]
    artifacts = checker.load_json(root / "external_artifacts.json")["artifacts"]

    assert {entry["role"] for entry in artifacts} == {
        "source_manifest",
        "raw_input_manifest",
        "model_snapshot_manifest",
        "fvi_study",
        "environment_contract",
    }
    assert {
        entry["retrieval_path"]
        for entry in artifacts
        if entry["role"].endswith("manifest")
    } == {
        "evidence/source_manifest.json",
        "evidence/raw_input_manifest.json",
        "evidence/model_snapshot_manifest.json",
    }
    result = checker.validate_run(
        root,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_package=True,
    )
    assert result.passed, result.errors


def test_checksum_inventory_rejects_unlisted_files_and_symlinks(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "bound.txt").write_text("bound", encoding="utf-8")
    writers.write_sha256sums(root)
    errors: list[str] = []
    checker_package.check_complete_checksums(root, errors)
    assert errors == []

    (root / "unlisted.txt").write_text("unlisted", encoding="utf-8")
    errors = []
    checker_package.check_complete_checksums(root, errors)
    assert "unlisted package file: 'unlisted.txt'" in errors

    (root / "link").symlink_to(root / "bound.txt")
    errors = []
    checker_package.check_complete_checksums(root, errors)
    assert "symlink in package: 'link'" in errors


def test_rehashed_raw_manifest_still_requires_semantic_pass(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    root = built["run_root"]
    raw_path = root / "evidence" / "raw_input_manifest.json"
    raw_manifest = checker.load_json(raw_path)
    raw_manifest["identity"]["semantic_checks"]["all_semantic_checks_pass"] = False
    raw_manifest["id"] = compute_id(raw_manifest["identity"])
    raw_bytes = _dump(raw_path, raw_manifest)

    ledger_path = root / "external_artifacts.json"
    ledger = checker.load_json(ledger_path)
    for entry in ledger["artifacts"]:
        if entry["role"] == "raw_input_manifest":
            entry["content_id"] = raw_manifest["id"]
            entry["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
            entry["byte_size"] = len(raw_bytes)
    _dump(ledger_path, ledger)
    writers.write_sha256sums(root)

    spec = checker.load_json(root / "run_spec.json")["identity"]
    spec_ids = dict(spec["identity"])
    spec_ids["raw_input_bundle_id"] = raw_manifest["id"]
    errors: list[str] = []
    checker_package.check_external_artifacts(
        root,
        errors,
        spec_ids=spec_ids,
        evidence_roots=spec["evidence_roots"],
        profile_variant="smoke",
        fvi_selected=spec["fvi_selected"],
        environment_claims=checker.load_json(root / "environment.json"),
    )
    assert "packaged raw-input semantic checks did not pass" in errors


def test_final_package_carries_and_revalidates_receipts(tmp_path, monkeypatch):
    root = tmp_path / "runs" / "final"
    root.mkdir(parents=True)
    manifests = {
        "source_manifest": build_manifest({"kind": "source_snapshot"}),
        "raw_input_manifest": build_manifest(
            {
                "kind": "raw_input_bundle",
                "semantic_checks": {"all_semantic_checks_pass": True},
            }
        ),
        "model_snapshot_manifest": build_manifest({"kind": "model_snapshot"}),
    }
    manifest_paths = {
        role: tmp_path / "inputs" / f"{role}.json"
        for role in manifests
    }
    for role, manifest in manifests.items():
        _dump(manifest_paths[role], manifest)

    adapter_id = "a" * 64
    fvi = build_manifest(
        {
            "kind": "fvi_study",
            "adapter_bundle_id": adapter_id,
            "selected_parameters": {
                "tolerance": "1e-8",
                "max_iterations": 100,
            },
        }
    )
    environment_claims = {
        "python_version": "3.11.0",
        "package_versions": {"numpy": "2.0.0"},
    }
    environment = build_manifest(
        environment_contract_identity(**environment_claims)
    )
    fvi_bytes = _dump(tmp_path / "fvi.json", fvi)
    environment_bytes = _dump(tmp_path / "environment.json", environment)
    bindings = {
        "source_manifest_id": manifests["source_manifest"]["id"],
        "raw_input_bundle_id": manifests["raw_input_manifest"]["id"],
        "model_snapshot_id": manifests["model_snapshot_manifest"]["id"],
        "adapter_bundle_id": adapter_id,
        "fvi_study_id": fvi["id"],
        "environment_contract_id": environment["id"],
    }
    receipts = {}
    for gate in ("smoke", "mutation", "determinism"):
        receipt_bindings = (
            {
                key: bindings[key]
                for key in (
                    "source_manifest_id",
                    "raw_input_bundle_id",
                    "model_snapshot_id",
                    "adapter_bundle_id",
                )
            }
            if gate == "determinism"
            else bindings
        )
        receipt = writers.build_prerequisite_receipt(
            gate=gate,
            bindings=receipt_bindings,
            evidence={"fixture": gate},
        )
        receipts[gate] = receipt
        _dump(
            tmp_path / "receipts" / gate / f"{receipt['id']}.json",
            receipt,
        )
    receipt_ids = {gate: receipt["id"] for gate, receipt in receipts.items()}
    run_spec_identity = {
        "profile_variant": "final",
        "identity": bindings,
        "evidence_roots": {"prerequisite_receipts": receipt_ids},
    }
    _dump(
        root / "run_spec.json",
        {"id": compute_id(run_spec_identity), "identity": run_spec_identity},
    )
    external_artifacts = [
        _ledger(
            role,
            manifests[role],
            manifest_paths[role],
            legacy_retrieval=True,
        )
        for role in manifests
    ]
    external_artifacts.extend(
        [
            {
                "role": "fvi_study",
                "content_id": fvi["id"],
                "sha256": hashlib.sha256(fvi_bytes).hexdigest(),
                "byte_size": len(fvi_bytes),
                "retrieval_path": "evidence/fvi_study.json",
            },
            {
                "role": "environment_contract",
                "content_id": environment["id"],
                "sha256": hashlib.sha256(environment_bytes).hexdigest(),
                "byte_size": len(environment_bytes),
                "retrieval_path": "evidence/environment_contract.json",
            },
        ]
    )
    monkeypatch.setattr(writers, "write_figures", lambda *_args, **_kwargs: [])
    writers.package_run(
        root,
        {
            "profile_variant": "final",
            "cells": {},
            "family": {},
            "fvi_selected": {
                "tolerance": "1e-8",
                "max_iterations": 100,
            },
            "gate_overrides": {},
        },
        resource_summary={"backend": "local"},
        external_artifacts=external_artifacts,
        evidence_files={
            "evidence/fvi_study.json": fvi_bytes,
            "evidence/environment_contract.json": environment_bytes,
        },
    )

    ledger = checker.load_json(root / "external_artifacts.json")
    assert {
        entry["role"]
        for entry in ledger["artifacts"]
        if entry["role"].startswith("prerequisite_receipt_")
    } == {
        "prerequisite_receipt_smoke",
        "prerequisite_receipt_mutation",
        "prerequisite_receipt_determinism",
    }
    errors: list[str] = []
    checker_package.check_external_artifacts(
        root,
        errors,
        spec_ids=bindings,
        evidence_roots={"prerequisite_receipts": receipt_ids},
        profile_variant="final",
        fvi_selected={"tolerance": "1e-8", "max_iterations": 100},
        environment_claims=environment_claims,
    )
    assert errors == []


def test_constant_platt_phase_and_schema_envelope_are_supported():
    assert platt_phase_errors(
        {
            "platt_coef": None,
            "platt_intercept": None,
            "platt_model_type": "constant",
            "platt_constant_probability": 0.7,
        },
        phase="late",
    ) == []
    assert platt_phase_errors(
        {
            "platt_coef": None,
            "platt_intercept": None,
            "platt_model_type": "constant",
            "platt_constant_probability": 1.7,
        },
        phase="late",
    )

    schema = json.loads(
        (REPO / "schemas" / "stopdff_run_spec.schema.json").read_text()
    )
    assert schema["required"] == ["id", "identity"]
    assert schema["properties"]["identity"]["$ref"] == "#/$defs/identity"
    assert schema["$defs"]["identity"]["properties"]["kind"]["const"] == "run_spec"


def test_unbound_ledger_alias_is_normalized_and_zero_size_is_rejected(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "report"
    root.mkdir()
    artifact = {
        "role": "source_manifest",
        "content_id": "1" * 64,
        "sha256": "2" * 64,
        "byte_size": 1,
        "retrieval": "source.json",
    }
    monkeypatch.setattr(writers, "write_figures", lambda *_args, **_kwargs: [])
    writers.package_run(
        root,
        {"cells": {}, "family": {}, "fvi_selected": {}, "gate_overrides": {}},
        resource_summary={},
        external_artifacts=[artifact],
    )
    stored = checker.load_json(root / "external_artifacts.json")["artifacts"][0]
    assert "retrieval" not in stored
    assert stored["retrieval_path"] == "source.json"

    bad_root = tmp_path / "bad-report"
    bad_root.mkdir()
    with pytest.raises(ValueError, match="invalid external-artifact"):
        writers.package_run(
            bad_root,
            {"cells": {}, "family": {}, "fvi_selected": {}, "gate_overrides": {}},
            resource_summary={},
            external_artifacts=[{**artifact, "byte_size": 0}],
        )
