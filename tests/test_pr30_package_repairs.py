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
from scripts.stopdff_v5.manifests import (  # noqa: E402
    FVI_PRODUCER_FILES,
    environment_contract_identity,
    fvi_study_identity,
)
from scripts.stopdff_v5.receipt_evidence import (  # noqa: E402
    DETERMINISM_FILES,
    MUTATION_ROSTER,
)


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


def _canonical_fvi_identity(
    adapter_id: str,
    producer_hashes: dict[str, str],
) -> dict:
    selected = {"tolerance": "1e-8", "max_iterations": 100}
    candidates = []
    for tolerance in ("1e-6", "1e-8", "1e-10"):
        for max_iterations in (50, 100, 200):
            is_selected = (
                tolerance == selected["tolerance"]
                and max_iterations == selected["max_iterations"]
            )
            candidates.append(
                {
                    "tolerance": tolerance,
                    "max_iterations": max_iterations,
                    "total_iterations": max_iterations,
                    "all_converged": True,
                    "eligible": is_selected,
                    "ineligibility_reasons": (
                        [] if is_selected else ["synthetic_fixture"]
                    ),
                }
            )
    return fvi_study_identity(
        adapter_bundle_id=adapter_id,
        candidate_grid={
            "tolerance": ["1e-6", "1e-8", "1e-10"],
            "max_iterations": [50, 100, 200],
        },
        representative_generator="representative_24_parity",
        candidate_results=candidates,
        strict_reference_results={
            "tolerance": "1e-10",
            "max_iterations": 200,
            "total_iterations": 200,
            "all_converged": True,
        },
        selector_rule=(
            "min_total_iterations__then_larger_tolerance__then_smaller_max_iter"
        ),
        selected_parameters=selected,
        all96_validation={
            **selected,
            "all_converged": True,
            "total_iterations": 9600,
        },
        producer_hashes=producer_hashes,
    )


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
        adapter_identity=checker.load_json(
            built["adapter_bundle"] / "manifest.json"
        )["identity"],
    )
    assert "packaged raw-input semantic checks did not pass" in errors


def test_final_package_carries_and_revalidates_receipts(tmp_path, monkeypatch):
    root = tmp_path / "runs" / "final"
    root.mkdir(parents=True)
    producer_hashes = {
        name: hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in FVI_PRODUCER_FILES
    }
    adapter_producer_hashes = {"adapter_build.py": "d" * 64}
    run_producer_hashes = {"checker.py": "e" * 64, "sweep.py": "f" * 64}
    source_producer_hashes = {
        **producer_hashes,
        **adapter_producer_hashes,
        **run_producer_hashes,
    }
    build_metadata_sha256 = "b" * 64
    manifests = {
        "source_manifest": build_manifest(
            {
                "kind": "source_snapshot",
                "files": [
                    {
                        "path": f"scripts/stopdff_v5/{name}",
                        "sha256": digest,
                    }
                    for name, digest in source_producer_hashes.items()
                ],
            }
        ),
        "raw_input_manifest": build_manifest(
            {
                "kind": "raw_input_bundle",
                "files": [
                    {
                        "role": "build_metadata.json",
                        "size": 1,
                        "sha256": build_metadata_sha256,
                    }
                ],
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

    determinism_hashes = {
        name: hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in DETERMINISM_FILES
    }
    determinism_hashes["build_metadata.json"] = build_metadata_sha256
    adapter_identity = {
        "kind": "adapter_bundle",
        "source_manifest_id": manifests["source_manifest"]["id"],
        "raw_input_bundle_id": manifests["raw_input_manifest"]["id"],
        "model_snapshot_id": manifests["model_snapshot_manifest"]["id"],
        "fit_rows_sha256": determinism_hashes["fit_rows.jsonl.gz"],
        "eval_rows_sha256": determinism_hashes["eval_rows.jsonl.gz"],
        "calibration_sha256": determinism_hashes["calibration.json"],
        "producer_hashes": adapter_producer_hashes,
        "mc_retention_evidence": {
            "build_metadata_sha256": determinism_hashes[
                "build_metadata.json"
            ],
        },
    }
    adapter_manifest = build_manifest(adapter_identity)
    adapter_id = adapter_manifest["id"]
    fvi = build_manifest(_canonical_fvi_identity(adapter_id, producer_hashes))
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
    smoke_spec_identity = {
        "kind": "run_spec",
        "profile_variant": "smoke",
        "identity": bindings,
        "evidence_roots": {"prerequisite_receipts": {}},
    }
    smoke_spec = {
        "id": compute_id(smoke_spec_identity),
        "identity": smoke_spec_identity,
    }
    mutation_results = [
        {
            "mutation": name,
            "expected": "PASS" if index == 0 else "REJECT",
            "passed_check": index == 0,
            "ok": True,
            "errors": [],
        }
        for index, name in enumerate(MUTATION_ROSTER)
    ]
    gate_evidence = {
        "smoke": writers.build_prerequisite_evidence(
            gate="smoke",
            bindings=bindings,
            details={
                "run_spec": smoke_spec,
                "aggregate": {
                    "profile_variant": "smoke",
                    "run_spec_id": smoke_spec["id"],
                    "adapter_bundle_id": adapter_id,
                    "fvi_study_id": fvi["id"],
                    "requested": 1,
                    "completed": 1,
                    "failed": 0,
                    "skipped": 0,
                    "release_status": "VALID",
                    "release_reasons": [],
                },
            },
        ),
        "mutation": writers.build_prerequisite_evidence(
            gate="mutation",
            bindings=bindings,
            details={"results": mutation_results},
        ),
        "determinism": writers.build_prerequisite_evidence(
            gate="determinism",
            bindings={
                key: bindings[key]
                for key in (
                    "source_manifest_id",
                    "raw_input_bundle_id",
                    "model_snapshot_id",
                    "adapter_bundle_id",
                )
            },
            details={
                "first_adapter_manifest": adapter_manifest,
                "second_adapter_manifest": adapter_manifest,
                "first_file_sha256": determinism_hashes,
                "second_file_sha256": determinism_hashes,
            },
        ),
    }
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
        receipt = writers.build_evidenced_prerequisite_receipt(
            gate=gate,
            bindings=receipt_bindings,
            evidence=gate_evidence[gate],
        )
        receipts[gate] = receipt
        _dump(
            tmp_path / "receipts" / gate / f"{receipt['id']}.json",
            receipt,
        )
        _dump(
            tmp_path
            / "receipts"
            / gate
            / f"{receipt['id']}.evidence.json",
            gate_evidence[gate],
        )
    receipt_ids = {gate: receipt["id"] for gate, receipt in receipts.items()}
    run_spec_identity = {
        "profile_variant": "final",
        "identity": bindings,
        "evidence_roots": {
            "prerequisite_receipts": receipt_ids,
            "producer_hashes": run_producer_hashes,
        },
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
    assert {
        path.name
        for path in (root / "evidence" / "prerequisite_receipts").glob(
            "*.evidence.json"
        )
    } == {
        "smoke.evidence.json",
        "mutation.evidence.json",
        "determinism.evidence.json",
    }
    errors: list[str] = []
    checker_package.check_external_artifacts(
        root,
        errors,
        spec_ids=bindings,
        evidence_roots={
            "prerequisite_receipts": receipt_ids,
            "producer_hashes": run_producer_hashes,
        },
        profile_variant="final",
        fvi_selected={"tolerance": "1e-8", "max_iterations": 100},
        environment_claims=environment_claims,
        adapter_identity={
            **adapter_identity,
            "mc_retention_evidence": {
                "build_metadata_sha256": build_metadata_sha256,
            },
        },
    )
    assert errors == []

    (root / "evidence" / "prerequisite_receipts" / "mutation.evidence.json").unlink()
    errors = []
    checker_package.check_external_artifacts(
        root,
        errors,
        spec_ids=bindings,
        evidence_roots={
            "prerequisite_receipts": receipt_ids,
            "producer_hashes": run_producer_hashes,
        },
        profile_variant="final",
        fvi_selected={"tolerance": "1e-8", "max_iterations": 100},
        environment_claims=environment_claims,
        adapter_identity={
            **adapter_identity,
            "mc_retention_evidence": {
                "build_metadata_sha256": build_metadata_sha256,
            },
        },
    )
    assert any("missing packaged mutation prerequisite evidence" in error for error in errors)


@pytest.mark.parametrize("producer_hashes", [None, {}, {"fvi_study.py": "a" * 64}])
def test_fvi_study_requires_complete_producer_inventory(producer_hashes):
    complete = {
        name: hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in FVI_PRODUCER_FILES
    }
    identity = _canonical_fvi_identity("a" * 64, complete)
    if producer_hashes is None:
        del identity["producer_hashes"]
    else:
        identity["producer_hashes"] = producer_hashes
    errors: list[str] = []
    checker_package._check_fvi_study_identity(identity, errors)
    assert any("canonical producer set" in error for error in errors)


def test_final_fvi_rejects_label_only_study_identity():
    errors: list[str] = []
    checker_package._check_fvi_study_identity(
        {
            "kind": "fvi_study",
            "adapter_bundle_id": "a" * 64,
            "selected_parameters": {
                "tolerance": "1e-8",
                "max_iterations": 100,
            },
        },
        errors,
    )
    assert "packaged FVI study fields do not match the canonical contract" in errors


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
