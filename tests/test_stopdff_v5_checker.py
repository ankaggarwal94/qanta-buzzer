"""Checker + negative mutation suite (synthetic fixtures)."""
from __future__ import annotations

import copy
import gzip
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import checker, fvi_study, identity, selftest  # noqa: E402
from scripts.stopdff_v5.adapter_build import derive_bound_calibration  # noqa: E402
from scripts.stopdff_v5.checker_package import inspect_packaged_fvi_manifest_kind  # noqa: E402


_EXPECTED_RUN_MUTATIONS = frozenset({
    "stale_cache",
    "cell_verdict_serialized_not_trusted",
    "coverage_clean_serialized_not_trusted",
    "ceiling_flags_tampered",
    "wrong_family_maximum_statistic",
    "family_verdict_hides_cell_warn",
    "wrong_release_status",
    "dual_backend_manifests",
    "missing_backend_manifest",
    "wrong_bootstrap_seed",
    "wrong_bootstrap_count",
    "tampered_run_spec_id",
    "wrong_bootstrap_plan_hash",
    "fresh_attempt_with_resume",
    "resume_without_bare_resume",
    "overwrite_in_evidence_run",
    "unsafe_checksum_traversal",
    "duplicate_checksum_entry",
    "symlink_in_package",
    "checksum_value_mismatch",
    "invalid_png",
    "truncated_png_after_ihdr",
    "missing_external_artifacts",
    "unconverged_fvi_marked_completed",
    "cell_fingerprint_tampered",
    "adapter_calibration_bytes_tampered",
    "backend_adapter_binding",
    "attempt_adapter_binding",
    "cell_adapter_binding",
    "fingerprint_adapter_hash_binding",
    "aggregate_adapter_binding",
    "aggregate_fvi_binding",
    "unknown_attempt_mode",
    "fingerprint_kind",
    "fingerprint_producer_binding",
    "cell_gate_override",
    "backend_environment_binding",
    "missing_fvi_evidence",
    "missing_attempt_result",
    "attempt_result_counts",
})
_EXPECTED_ADAPTER_MUTATIONS = frozenset({"invalid_adapter_row_hash"})


def test_valid_package_passes(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    res = checker.validate_run(
        built["run_root"], backend="modal", adapter_bundle=built["adapter_bundle"],
        require_final_profile=False, require_package=True,
    )
    assert res.passed, res.errors


def test_fixed_fvi_smoke_package_skips_selector_recomputation(tmp_path, monkeypatch):
    built = selftest.build_valid_package(tmp_path, fixed_fvi=True)
    checker._FVI_STUDY_CACHE.clear()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_fvi_study should not run for fvi_study_fixed")

    monkeypatch.setattr(fvi_study, "run_fvi_study", fail_if_called)

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=True,
    )

    assert result.passed, result.errors


def test_genuine_fvi_package_still_recomputes_selector(tmp_path, monkeypatch):
    built = selftest.build_valid_package(tmp_path)
    checker._FVI_STUDY_CACHE.clear()
    expected = copy.deepcopy(selftest._SYNTH_FVI_STUDY)
    calls: list[tuple[list[dict], dict]] = []

    def record_and_return(rows, calibration_json):
        calls.append((rows, calibration_json))
        return copy.deepcopy(expected)

    monkeypatch.setattr(fvi_study, "run_fvi_study", record_and_return)

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=True,
    )

    assert result.passed, result.errors
    assert len(calls) == 1


def test_fixed_fvi_final_package_rejected(tmp_path):
    """A fixed-FVI manifest in a final-profile package must be rejected."""
    built = selftest.build_valid_package(tmp_path, fixed_fvi=True)
    run_spec = json.loads(
        (built["run_root"] / "run_spec.json").read_text(encoding="utf-8")
    )
    fvi_study_id = run_spec["identity"]["identity"]["fvi_study_id"]
    with pytest.raises(ValueError, match="kind mismatch"):
        inspect_packaged_fvi_manifest_kind(
            built["run_root"],
            expected_id=fvi_study_id,
            profile_variant="final",
        )


def test_checker_rejects_unhashable_attempt_result_state_without_exception(
    tmp_path,
):
    built = selftest.build_valid_package(tmp_path)
    result_path = built["run_root"] / "attempt_results" / "1.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["state"] = []
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checked = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )

    assert not checked.passed
    assert any("invalid state" in error for error in checked.errors)


def test_checker_rejects_unexpected_attempt_result_entry(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    unexpected = built["run_root"] / "attempt_results" / "orphan.tmp"
    unexpected.write_bytes(b"partial terminal evidence")

    checked = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )

    assert not checked.passed
    assert any(
        "invalid attempt result filename" in error
        for error in checked.errors
    )


def test_validate_adapter_ok(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    res = checker.validate_adapter(built["adapter_bundle"])
    assert res.passed, res.errors


def test_negative_mutation_suite(tmp_path):
    registered = list(selftest._RUN_MUTATIONS)
    assert len(registered) == len(set(registered)) == len(_EXPECTED_RUN_MUTATIONS)
    assert set(registered) == _EXPECTED_RUN_MUTATIONS

    ok, results = selftest.run_self_test(tmp_path)
    failures = [r for r in results if not r["ok"]]
    assert ok, f"mutations not rejected: {[(r['mutation'], r['errors']) for r in failures]}"

    names = [result["mutation"] for result in results]
    assert names[0] == "<baseline valid>"
    assert results[0]["expected"] == "PASS"
    assert len(names) == len(set(names))
    assert set(names[1:]) == (
        _EXPECTED_RUN_MUTATIONS | _EXPECTED_ADAPTER_MUTATIONS
    )
    assert len(results) == 1 + len(_EXPECTED_RUN_MUTATIONS) + len(
        _EXPECTED_ADAPTER_MUTATIONS
    )
    assert all(result["expected"] == "REJECT" for result in results[1:])


def test_validate_spec_placeholder_rejected(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text('{"id":"x","identity":{"kind":"run_spec","profile_name":"stopdff_bucketed_dp_paired_v2",'
                    '"identity":{"adapter_bundle_id":"<ADAPTER_ID>"}}}', encoding="utf-8")
    res = checker.validate_spec(spec, require_final_profile=False)
    assert not res.passed


@pytest.mark.parametrize(
    "mutation",
    [
        "bootstrap_bit_generator",
        "attempt_mode",
        "fingerprint_kind",
        "fingerprint_producers",
        "fingerprint_myopic",
        "cell_gate_override",
        "family_M",
        "aggregate_myopic",
        "backend_environment",
    ],
)
def test_checker_rejects_self_consistent_identity_and_summary_bypasses(
    tmp_path,
    mutation,
):
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]

    if mutation == "bootstrap_bit_generator":
        path = run_root / "bootstrap_plan.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["identity"]["bit_generator"] = "MT19937"
        record["id"] = identity.compute_id(record["identity"])
    elif mutation == "attempt_mode":
        path = run_root / "attempts.jsonl"
        record = json.loads(path.read_text(encoding="utf-8").strip())
        record["mode"] = "replay"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        record = None
    elif mutation in {
        "fingerprint_kind",
        "fingerprint_producers",
        "fingerprint_myopic",
    }:
        path = sorted((run_root / "cells").glob("*.json"))[0]
        record = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "fingerprint_kind":
            record["fingerprint_identity"]["kind"] = "unreviewed"
        elif mutation == "fingerprint_producers":
            record["fingerprint_identity"]["producer_hashes"] = {
                "sweep.py": "f" * 64
            }
        else:
            record["fingerprint_identity"]["myopic_artifact_sha256"] = "f" * 64
        record["fingerprint_id"] = identity.compute_id(
            record["fingerprint_identity"]
        )
    elif mutation == "cell_gate_override":
        path = sorted((run_root / "cells").glob("*.json"))[0]
        record = json.loads(path.read_text(encoding="utf-8"))
        record["mc_gate_overridden"] = not record["mc_gate_overridden"]
    elif mutation in {"family_M", "aggregate_myopic"}:
        path = run_root / "aggregate.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "family_M":
            record["family"]["M"] += 10
        else:
            record["myopic_artifact_sha256"] = "f" * 64
    else:
        path = run_root / "run_manifest.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["environment"] = {"python_version": "0.0.0"}

    if record is not None:
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    result = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )
    assert not result.passed, mutation


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bit_generator", "MT19937", "bit_generator"),
        ("resample_dtype", "uint64", "resample_dtype"),
        ("numpy_version_contract", "0.0.0", "NumPy version"),
    ],
)
def test_bootstrap_metadata_rejected_after_graph_is_self_consistently_rebound(
    tmp_path,
    field,
    value,
    message,
):
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    plan = json.loads((run_root / "bootstrap_plan.json").read_text())
    plan["identity"][field] = value
    plan["id"] = identity.compute_id(plan["identity"])

    spec = json.loads((run_root / "run_spec.json").read_text())
    spec["identity"]["identity"]["bootstrap_plan_id"] = plan["id"]
    spec["id"] = identity.compute_id(spec["identity"])

    with pytest.raises(ValueError, match=message):
        checker.resolve_run_binding(
            run_spec_manifest=spec,
            adapter_bundle=built["adapter_bundle"],
            bootstrap_plan_manifest=plan,
        )


def test_package_fvi_ledger_binding_rejected_with_valid_checksums(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    ledger = run_root / "external_artifacts.json"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    next(
        item for item in payload["artifacts"] if item["role"] == "fvi_study"
    )["content_id"] = "f" * 64
    ledger.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sums = run_root / "SHA256SUMS"
    lines = sums.read_text(encoding="utf-8").splitlines()
    lines = [
        (
            f"{identity.sha256_file(ledger)}  external_artifacts.json"
            if line.endswith("  external_artifacts.json")
            else line
        )
        for line in lines
    ]
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=True,
    )
    assert not result.passed
    assert any(
        "fvi_study does not match run spec" in error
        for error in result.errors
    )


def test_validate_run_rejects_self_valid_but_unbound_adapter(tmp_path):
    """A valid adapter B cannot be substituted for run spec A."""
    built = selftest.build_valid_package(tmp_path / "original")
    substituted = tmp_path / "substituted_adapter"
    shutil.copytree(built["adapter_bundle"], substituted)

    manifest_path = substituted / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["model_snapshot_id"] = "9" * 64
    calibration_path = substituted / "calibration.json"
    calibration = derive_bound_calibration(
        fit_rows=checker.load_jsonl_gz(substituted / "fit_rows.jsonl.gz"),
        eval_rows=checker.load_jsonl_gz(substituted / "eval_rows.jsonl.gz"),
        model_snapshot_id=manifest["identity"]["model_snapshot_id"],
        fit_rows_sha256=manifest["identity"]["fit_rows_sha256"],
    )
    calibration_path.write_text(
        json.dumps(calibration, sort_keys=True),
        encoding="utf-8",
    )
    manifest["identity"]["calibration_sha256"] = identity.sha256_file(
        calibration_path
    )
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    adapter_result = checker.validate_adapter(substituted)
    assert adapter_result.passed, adapter_result.errors

    run_result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=substituted,
        require_final_profile=False,
        require_package=False,
    )
    assert not run_result.passed
    assert any("adapter_bundle_id" in error for error in run_result.errors)


def test_validate_run_rejects_adapter_byte_hash_mismatch_with_equivalent_rows(tmp_path):
    """Parsed-equivalent gzip bytes still have to match the bound byte hash."""
    built = selftest.build_valid_package(tmp_path)
    fit_rows = built["adapter_bundle"] / "fit_rows.jsonl.gz"
    raw = bytearray(fit_rows.read_bytes())
    assert raw[:2] == b"\x1f\x8b"
    raw[4:8] = (1).to_bytes(4, "little")
    fit_rows.write_bytes(raw)

    run_result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )
    assert not run_result.passed
    assert any("fit_rows sha mismatch" in error for error in run_result.errors)


@pytest.mark.parametrize(
    "edge",
    [
        "aggregate_run_spec",
        "backend_run_spec",
        "attempt_run_spec",
        "cell_run_spec",
        "fingerprint_run_spec",
        "aggregate_bootstrap_plan",
        "aggregate_adapter",
        "backend_adapter",
        "attempt_adapter",
        "cell_adapter",
        "fingerprint_adapter",
        "fingerprint_fit_hash",
        "cell_bootstrap_plan",
        "aggregate_fvi_settings",
        "aggregate_gate_overrides",
    ],
)
def test_validate_run_binds_all_produced_records_to_run_spec(tmp_path, edge):
    """Locally self-consistent records must still match the run identity graph."""
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    wrong_id = "f" * 64

    if edge == "aggregate_run_spec":
        path = run_root / "aggregate.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["run_spec_id"] = wrong_id
    elif edge == "backend_run_spec":
        path = run_root / "run_manifest.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["identity"]["run_spec_id"] = wrong_id
        record["id"] = identity.compute_id(record["identity"])
    elif edge == "backend_adapter":
        path = run_root / "run_manifest.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["identity"]["adapter_bundle_id"] = wrong_id
        record["id"] = identity.compute_id(record["identity"])
    elif edge == "attempt_run_spec":
        path = run_root / "attempts.jsonl"
        record = json.loads(path.read_text(encoding="utf-8").strip())
        record["run_spec_id"] = wrong_id
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        record = None
    elif edge == "attempt_adapter":
        path = run_root / "attempts.jsonl"
        record = json.loads(path.read_text(encoding="utf-8").strip())
        record["adapter_id"] = wrong_id
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        record = None
    elif edge in {"cell_run_spec", "fingerprint_run_spec"}:
        path = sorted((run_root / "cells").glob("*.json"))[0]
        record = json.loads(path.read_text(encoding="utf-8"))
        if edge == "cell_run_spec":
            record["run_spec_id"] = wrong_id
        else:
            record["fingerprint_identity"]["run_spec_id"] = wrong_id
            record["fingerprint_id"] = identity.compute_id(
                record["fingerprint_identity"]
            )
    elif edge in {
        "cell_adapter",
        "fingerprint_adapter",
        "fingerprint_fit_hash",
        "cell_bootstrap_plan",
    }:
        path = sorted((run_root / "cells").glob("*.json"))[0]
        record = json.loads(path.read_text(encoding="utf-8"))
        if edge == "cell_adapter":
            record["adapter_bundle_id"] = wrong_id
        elif edge == "cell_bootstrap_plan":
            record["bootstrap_plan_id"] = wrong_id
        elif edge == "fingerprint_adapter":
            record["fingerprint_identity"]["adapter_bundle_id"] = wrong_id
            record["fingerprint_id"] = identity.compute_id(
                record["fingerprint_identity"]
            )
        else:
            record["fingerprint_identity"]["adapter_fit_rows_sha256"] = wrong_id
            record["fingerprint_id"] = identity.compute_id(
                record["fingerprint_identity"]
            )
    elif edge == "aggregate_adapter":
        path = run_root / "aggregate.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["adapter_bundle_id"] = wrong_id
    elif edge == "aggregate_fvi_settings":
        path = run_root / "aggregate.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        current = record["fvi_selected"]["max_iterations"]
        record["fvi_selected"]["max_iterations"] = (
            50 if current != 50 else 100
        )
    elif edge == "aggregate_gate_overrides":
        path = run_root / "aggregate.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["gate_overrides"]["allow_low_mc_retention"] = True
    else:
        path = run_root / "aggregate.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["bootstrap_plan_id"] = wrong_id

    if record is not None:
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    result = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )
    assert not result.passed, edge


def test_validate_adapter_rejects_duplicate_json_keys_even_with_rebound_hash(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    fit_path = built["adapter_bundle"] / "fit_rows.jsonl.gz"
    lines = gzip.decompress(fit_path.read_bytes()).decode("utf-8").splitlines()
    lines[0] = lines[0].replace(
        '"item_id":',
        '"item_id":"forged","item_id":',
        1,
    )
    fit_path.write_bytes(
        gzip.compress(("\n".join(lines) + "\n").encode("utf-8"), mtime=0)
    )

    manifest_path = built["adapter_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["fit_rows_sha256"] = identity.sha256_file(fit_path)
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any("duplicate JSON key" in error for error in result.errors)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_version",
        "profile_variant",
        "scientific_profile",
        "identity_id",
        "fvi_tolerance",
        "gate_field",
        "myopic_root",
        "producer_root",
        "smoke_receipts",
    ],
)
def test_validate_spec_rejects_self_hashed_noncanonical_contract(
    tmp_path,
    mutation,
):
    built = selftest.build_valid_package(tmp_path)
    source = json.loads(
        (built["run_root"] / "run_spec.json").read_text(encoding="utf-8")
    )
    candidate = copy.deepcopy(source)
    body = candidate["identity"]
    if mutation == "schema_version":
        body["schema_version"] = 999
    elif mutation == "profile_variant":
        body["profile_variant"] = "trial"
    elif mutation == "scientific_profile":
        body["scientific_profile"] = {}
    elif mutation == "identity_id":
        body["identity"]["adapter_bundle_id"] = "not-a-hash"
    elif mutation == "fvi_tolerance":
        body["fvi_selected"]["tolerance"] = "1e-3"
    elif mutation == "myopic_root":
        body["evidence_roots"]["myopic_artifact_sha256"] = "not-a-hash"
    elif mutation == "producer_root":
        body["evidence_roots"]["producer_hashes"] = {
            "checker.py": "f" * 64,
        }
    elif mutation == "smoke_receipts":
        body["evidence_roots"]["prerequisite_receipts"] = {
            "smoke": "f" * 64,
        }
    else:
        body["gate"]["unreviewed_override"] = True
    candidate["id"] = identity.compute_id(body)

    path = tmp_path / f"invalid-{mutation}.json"
    path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result = checker.validate_spec(path, require_final_profile=False)
    assert not result.passed, mutation
