"""Regression coverage for the round-6 public API contract repairs."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import validate_stopdff_bucketed_sweep as acceptance_cli
from scripts.stopdff_v5 import checker, profile, receipt_evidence, selftest, writers
from scripts.stopdff_v5.identity import compute_id
from scripts.stopdff_v5.manifests import ENVIRONMENT_PACKAGES, run_spec_identity


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "scripts" / "validate_stopdff_bucketed_sweep.py"


def _valid_smoke_spec() -> dict[str, object]:
    identity = run_spec_identity(
        source_manifest_id="1" * 64,
        raw_input_bundle_id="2" * 64,
        model_snapshot_id="3" * 64,
        adapter_bundle_id="4" * 64,
        fvi_study_id="5" * 64,
        bootstrap_plan_id="6" * 64,
        environment_contract_id="7" * 64,
        fvi_selected={"tolerance": "1e-8", "max_iterations": 100},
        replicate_count=100,
        profile_variant="smoke",
        myopic_artifact_sha256="8" * 64,
        producer_hashes={"checker.py": "9" * 64, "sweep.py": "a" * 64},
        prerequisite_receipts={},
    )
    return {"id": compute_id(identity), "identity": identity}


def _determinism_evidence() -> tuple[dict[str, str], dict[str, object]]:
    source_id = "1" * 64
    raw_id = "2" * 64
    model_id = "3" * 64
    file_hashes = {
        "build_metadata.json": "4" * 64,
        "calibration.json": "5" * 64,
        "eval_rows.jsonl.gz": "6" * 64,
        "fit_rows.jsonl.gz": "7" * 64,
    }
    adapter_identity = {
        "kind": "adapter_bundle",
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
        "fit_rows_sha256": file_hashes["fit_rows.jsonl.gz"],
        "eval_rows_sha256": file_hashes["eval_rows.jsonl.gz"],
        "calibration_sha256": file_hashes["calibration.json"],
        "mc_retention_evidence": {
            "build_metadata_sha256": file_hashes["build_metadata.json"],
        },
    }
    adapter_manifest = {
        "id": compute_id(adapter_identity),
        "identity": adapter_identity,
    }
    bindings = {
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
        "adapter_bundle_id": adapter_manifest["id"],
    }

    def execution(execution_id: str, subdir: str) -> dict[str, object]:
        return {
            "environment": "modal_function_call",
            "execution_id": execution_id,
            "adapter_subdir": subdir,
            **bindings,
            "cached": False,
            "output_sha256": file_hashes,
        }

    evidence = receipt_evidence.build_prerequisite_evidence(
        gate="determinism",
        bindings=bindings,
        details={
            "source_execution": {
                "environment": "modal_image",
                "executing_source_manifest_id": source_id,
                "runtime_source_manifest_id": source_id,
            },
            "first_build_execution": execution("fc-invented-a", "build_a"),
            "second_build_execution": execution("fc-invented-b", "build_b"),
            "first_adapter_manifest": adapter_manifest,
            "second_adapter_manifest": adapter_manifest,
            "first_file_sha256": file_hashes,
            "second_file_sha256": file_hashes,
        },
    )
    return bindings, evidence


def test_validate_adapter_normalizes_manifest_decoder_errors(tmp_path: Path) -> None:
    bundle = tmp_path / "adapter"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        '{"identity": {}, "identity": {}}\n',
        encoding="utf-8",
    )

    result = checker.validate_adapter(bundle)

    assert not result.passed
    assert result.recomputed == {}
    assert any("adapter manifest cannot be decoded" in error for error in result.errors)


def test_environment_requires_exact_declared_package_set(tmp_path: Path) -> None:
    built = selftest.build_valid_package(tmp_path)
    environment_path = built["run_root"] / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    assert set(environment["package_versions"]) == set(ENVIRONMENT_PACKAGES)
    environment["package_versions"].pop(ENVIRONMENT_PACKAGES[-1])
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )

    assert not result.passed
    assert any("exactly the declared evidence-affecting packages" in e for e in result.errors)


def test_static_and_run_bound_gate_schemas_are_distinct() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    gate_schema = json.loads(
        (REPO / "schemas" / "stopdff_gate_policy.schema.json").read_text(
            encoding="utf-8"
        )
    )
    profile_schema = json.loads(
        (REPO / "schemas" / "stopdff_scientific_profile.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(gate_schema)
    run_gate = {**profile.GATE, "allow_low_mc_retention": True}
    jsonschema.Draft202012Validator(gate_schema).validate(run_gate)
    static_validator = jsonschema.Draft202012Validator(
        gate_schema["$defs"]["static_gate"]
    )
    static_validator.validate(profile.GATE)
    with pytest.raises(jsonschema.ValidationError):
        static_validator.validate(run_gate)
    assert profile_schema["properties"]["gate"]["$ref"].endswith(
        "#/$defs/static_gate"
    )


def test_unknown_evidence_gate_has_value_error_contract() -> None:
    with pytest.raises(ValueError, match="unknown prerequisite gate"):
        receipt_evidence.build_prerequisite_evidence(
            gate="not-a-gate",
            bindings={},
            details={},
        )


def test_execution_assertions_are_explicitly_trusted_producer_only() -> None:
    bindings, evidence = _determinism_evidence()
    receipt_evidence.validate_prerequisite_evidence(
        gate="determinism",
        bindings=bindings,
        evidence=evidence,
    )
    reproduction = (
        REPO / "docs" / "stopdff_v5" / "REPRODUCTION.md"
    ).read_text(encoding="utf-8")
    contract = (REPO / "ACCEPTANCE_CONTRACT.md").read_text(encoding="utf-8")
    assert "trusted-producer boundary" in reproduction
    assert "unsigned assertions" in reproduction
    assert "does not independently authenticate" in reproduction
    assert "unsigned trusted-producer assertions" in contract


def test_receipt_envelope_validation_is_neutral_not_writer_owned() -> None:
    from scripts.stopdff_v5 import checker_package

    assert (
        writers.validate_prerequisite_receipts
        is receipt_evidence.validate_prerequisite_receipts
    )
    source = inspect.getsource(checker_package)
    assert "from .writers import validate_prerequisite_receipts" not in source
    receipt_evidence.validate_prerequisite_receipts(
        profile_variant="smoke",
        identity_bindings={},
        receipt_ids={},
        receipts={},
    )


def test_public_validators_use_numpy_style_contract_sections() -> None:
    for function in (
        checker.validate_spec,
        checker.validate_adapter,
        checker.resolve_run_binding,
        checker.validate_run,
        receipt_evidence.validate_prerequisite_bindings,
        receipt_evidence.validate_prerequisite_evidence,
        receipt_evidence.validate_prerequisite_receipts,
        receipt_evidence.verify_prerequisite_evidence_bytes,
    ):
        doc = inspect.getdoc(function) or ""
        assert "Parameters\n----------" in doc, function.__name__
        assert "Returns\n-------" in doc, function.__name__


def test_checker_has_no_hand_maintained_helper_digest_table() -> None:
    source = (REPO / "scripts" / "stopdff_v5" / "checker.py").read_text(
        encoding="utf-8"
    )
    assert "_FOCUSED_CHECKER_HASHES" not in source
    assert "prepared=prepared_cell_inputs" in source


def test_main_routes_all_public_commands_and_preserves_human_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    def validate_spec(path: Path, *, require_final_profile: bool):
        calls.append(("validate-spec", (path, require_final_profile)))
        return checker.CheckResult(True, recomputed={"run_spec_id": "1" * 64})

    def validate_adapter(path: Path):
        calls.append(("validate-adapter", path))
        return checker.CheckResult(False, errors=["bad adapter"])

    def validate_run(path: Path, **kwargs):
        calls.append(("validate", (path, kwargs)))
        return checker.CheckResult(True, recomputed={"release_status": "VALID"})

    monkeypatch.setattr(acceptance_cli.checker, "validate_spec", validate_spec)
    monkeypatch.setattr(acceptance_cli.checker, "validate_adapter", validate_adapter)
    monkeypatch.setattr(acceptance_cli.checker, "validate_run", validate_run)
    monkeypatch.setattr(acceptance_cli.selftest, "run_self_test", lambda _path: (True, []))

    assert acceptance_cli.main(
        ["validate-spec", "spec.json", "--require-final-profile"]
    ) == 0
    assert "validate-spec: PASS" in capsys.readouterr().out
    assert acceptance_cli.main(["validate-adapter", "adapter"]) == 1
    assert "validate-adapter: FAIL" in capsys.readouterr().err
    assert acceptance_cli.main(
        [
            "validate",
            "run",
            "--backend",
            "modal",
            "--adapter-bundle",
            "adapter",
            "--require-package",
        ]
    ) == 0
    assert "validate: PASS" in capsys.readouterr().out
    assert acceptance_cli.main(
        ["self-test", "--work-dir", str(tmp_path / "self-test")]
    ) == 0
    assert "SELF_TEST=PASS" in capsys.readouterr().out
    assert [name for name, _ in calls] == [
        "validate-spec",
        "validate-adapter",
        "validate",
    ]


def test_cli_json_and_subprocess_exit_contract(tmp_path: Path) -> None:
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(
        json.dumps(_valid_smoke_spec(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    success = subprocess.run(
        [sys.executable, str(CLI), "validate-spec", str(spec_path), "--json"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert success.returncode == 0, success.stderr
    payload = json.loads(success.stdout)
    assert payload == {
        "schema_version": 1,
        "command": "validate-spec",
        "passed": True,
        "errors": [],
        "recomputed": {},
    }

    failure = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "validate-adapter",
            str(tmp_path / "missing-adapter"),
            "--json",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failure.returncode == 1
    failure_payload = json.loads(failure.stdout)
    assert failure_payload["passed"] is False
    assert failure_payload["command"] == "validate-adapter"
    assert failure_payload["errors"] == ["adapter manifest.json missing"]
