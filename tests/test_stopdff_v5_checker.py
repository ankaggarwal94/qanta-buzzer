"""Checker + negative mutation suite (synthetic fixtures)."""
from __future__ import annotations

import copy
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import checker, fvi_study, identity, selftest, writers  # noqa: E402
from scripts.stopdff_v5.adapter_build import derive_bound_calibration  # noqa: E402
from scripts.stopdff_v5.checker_package import inspect_packaged_fvi_manifest_kind  # noqa: E402
from scripts.stopdff_v5.receipt_evidence import (  # noqa: E402
    build_prerequisite_evidence,
)


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
_EXPECTED_CHECKSUM_CONTENT_MUTATIONS = frozenset({
    "unsafe_checksum_traversal",
    "duplicate_checksum_entry",
    "checksum_value_mismatch",
})
_EXPECTED_FINAL_RECEIPT_MUTATIONS = frozenset({
    "final_receipt_evidence_bytes_tampered",
    "final_receipt_id_forged",
    "final_receipt_binding_mismatch",
    "final_missing_prerequisite_receipt_role",
    "final_spec_drops_prerequisite_receipts",
    "final_extra_evidence_file",
    "final_extra_evidence_dir",
})
# The final phase covers two classes: receipt/evidence-ledger forgeries
# (rejected by the receipt lane) and evidence-namespace tampers (rejected by
# the package path policy's evidence recursion).
_FINAL_MUTATION_ERROR_MARKERS = {
    "final_extra_evidence_file": ("unaudited package file",),
    "final_extra_evidence_dir": ("unaudited empty package directory",),
}
_DEFAULT_FINAL_MUTATION_MARKERS = ("receipt", "prerequisite")


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
    # run_spec.json is a manifest: {"id": ..., "identity": <run_spec_identity>}.
    # run_spec_identity itself contains a nested "identity" dict with the IDs.
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


def test_validate_adapter_rejects_symlinked_bundle_root(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    link = tmp_path / "symlinked_bundle"
    link.symlink_to(built["adapter_bundle"])
    res = checker.validate_adapter(link)
    assert not res.passed
    assert any("non-symlink directory" in e for e in res.errors)


@pytest.mark.parametrize(
    "payload",
    [
        "manifest.json",
        "fit_rows.jsonl.gz",
        "eval_rows.jsonl.gz",
        "calibration.json",
        "build_metadata.json",
    ],
)
def test_validate_adapter_rejects_symlinked_payload(tmp_path, payload):
    built = selftest.build_valid_package(tmp_path)
    bundle = built["adapter_bundle"]
    target = bundle / payload
    tmp_copy = tmp_path / f"real_{payload}"
    tmp_copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(tmp_copy)
    res = checker.validate_adapter(bundle)
    assert not res.passed
    assert any("non-symlink" in e for e in res.errors)


def test_validate_adapter_symlinked_payload_no_hash_or_decode(tmp_path, monkeypatch):
    """sha256_file and load_jsonl_gz must not be invoked for a symlinked payload."""
    built = selftest.build_valid_package(tmp_path)
    bundle = built["adapter_bundle"]
    target = bundle / "fit_rows.jsonl.gz"
    tmp_copy = tmp_path / "real_fit_rows.jsonl.gz"
    tmp_copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(tmp_copy)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("sha256_file must not follow symlinks")

    monkeypatch.setattr(checker, "sha256_file", fail_if_called)

    res = checker.validate_adapter(bundle)
    assert not res.passed
    assert any("non-symlink" in e for e in res.errors)


def test_validate_run_rejects_symlinked_run_root_before_decode(tmp_path, monkeypatch):
    built = selftest.build_valid_package(tmp_path)
    link = tmp_path / "symlinked_run"
    link.symlink_to(built["run_root"], target_is_directory=True)
    calls: list[Path] = []
    original = checker.load_json

    def record(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(checker, "load_json", record)
    result = checker.validate_run(
        link,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )

    assert not result.passed
    assert result.errors == ["run root must be a non-symlink directory"]
    assert calls == []


@pytest.mark.parametrize(
    "filename",
    ["aggregate.json", "run_spec.json", "bootstrap_plan.json"],
)
def test_validate_run_rejects_symlinked_required_json_before_decode(
    tmp_path,
    monkeypatch,
    filename,
):
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    target = run_root / filename
    tmp_copy = tmp_path / f"real_{filename}"
    tmp_copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(tmp_copy)
    calls: list[Path] = []
    original = checker.load_json

    def record(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(checker, "load_json", record)
    result = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )

    assert not result.passed
    assert f"{filename} must be a non-symlink regular file" in result.errors
    assert target not in calls


@pytest.mark.parametrize(
    "filename",
    ["aggregate.json", "run_spec.json", "bootstrap_plan.json"],
)
def test_validate_run_missing_required_json_retains_contract(tmp_path, filename):
    built = selftest.build_valid_package(tmp_path)
    target = built["run_root"] / filename
    target.unlink()

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )

    assert not result.passed
    assert f"missing {filename}" in result.errors


@pytest.mark.parametrize(
    ("backend", "filename"),
    [
        ("modal", "run_manifest.json"),
        ("local", "command_manifest.json"),
    ],
)
def test_validate_run_rejects_symlinked_backend_manifest_before_decode(
    tmp_path,
    monkeypatch,
    backend,
    filename,
):
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    if backend == "local":
        modal_path = run_root / "run_manifest.json"
        manifest = json.loads(modal_path.read_text(encoding="utf-8"))
        manifest["identity"]["kind"] = "command_manifest"
        manifest["identity"]["backend"] = "local"
        manifest["id"] = identity.compute_id(manifest["identity"])
        (run_root / filename).write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        modal_path.unlink()
        aggregate_path = run_root / "aggregate.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate["backend"] = "local"
        aggregate_path.write_text(
            json.dumps(aggregate, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    target = run_root / filename
    external = tmp_path / f"external_{backend}_{filename}"
    external.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(external)
    calls: list[Path] = []
    original = checker.load_json

    def record(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(checker, "load_json", record)
    result = checker.validate_run(
        run_root,
        backend=backend,
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )

    assert not result.passed
    assert f"{filename} must be a non-symlink regular file" in result.errors
    assert target not in calls


def test_validate_run_rejects_symlinked_environment_before_decode(
    tmp_path,
    monkeypatch,
):
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    target = run_root / "environment.json"
    external = tmp_path / "external_environment.json"
    external.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(external)
    calls: list[Path] = []
    original = checker.load_json

    def record(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(checker, "load_json", record)
    result = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )

    assert not result.passed
    assert (
        "environment.json must be a non-symlink regular file"
        in result.errors
    )
    assert target not in calls


@pytest.mark.parametrize("forbidden_kind", ["dangling_symlink", "fifo"])
def test_validate_run_rejects_noncanonical_forbidden_backend_manifest(
    tmp_path,
    forbidden_kind,
):
    built = selftest.build_valid_package(tmp_path)
    forbidden = built["run_root"] / "command_manifest.json"
    if forbidden_kind == "dangling_symlink":
        forbidden.symlink_to(tmp_path / "missing_command_manifest.json")
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO creation is unavailable on this platform")
        os.mkfifo(forbidden)

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )

    assert not result.passed
    assert "modal backend forbids command_manifest.json" in result.errors


def test_validate_run_missing_backend_and_environment_retains_diagnostics(
    tmp_path,
):
    built = selftest.build_valid_package(tmp_path)
    (built["run_root"] / "run_manifest.json").unlink()
    (built["run_root"] / "environment.json").unlink()

    result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )

    assert not result.passed
    assert "modal backend requires run_manifest.json" in result.errors
    assert "missing environment.json" in result.errors


def test_negative_mutation_suite(tmp_path, monkeypatch):
    registered = list(selftest._RUN_MUTATIONS)
    assert len(registered) == len(set(registered)) == len(_EXPECTED_RUN_MUTATIONS)
    assert set(registered) == _EXPECTED_RUN_MUTATIONS
    assert (
        selftest._CHECKSUM_CONTENT_MUTATIONS
        == _EXPECTED_CHECKSUM_CONTENT_MUTATIONS
    )
    assert (
        set(selftest._FINAL_RECEIPT_MUTATIONS)
        == _EXPECTED_FINAL_RECEIPT_MUTATIONS
    )

    # Spy on the gate's two-layer structure: every non-SHA256SUMS-content
    # mutation (and every final receipt mutation) must be re-validated with a
    # regenerated checksum inventory, so the semantic layer alone rejects it.
    regenerated: list[Path] = []
    real_write_sha256sums = selftest.write_sha256sums

    def counting_write_sha256sums(root):
        regenerated.append(Path(root))
        return real_write_sha256sums(root)

    monkeypatch.setattr(
        selftest, "write_sha256sums", counting_write_sha256sums
    )

    validate_calls = {"smoke": 0, "final": 0}
    real_validate_run = checker.validate_run

    def counting_validate_run(*args, **kwargs):
        key = "final" if kwargs.get("require_final_profile") else "smoke"
        validate_calls[key] += 1
        return real_validate_run(*args, **kwargs)

    monkeypatch.setattr(checker, "validate_run", counting_validate_run)

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
    assert all(result["passed_check"] is False for result in results[1:])

    # Two-layer accounting: 37 semantic re-validations in the run-mutation
    # loop plus one per final receipt mutation.
    run_semantic = len(selftest._RUN_MUTATIONS) - len(
        selftest._CHECKSUM_CONTENT_MUTATIONS
    )
    final_mutations = len(selftest._FINAL_RECEIPT_MUTATIONS)
    assert len(regenerated) == run_semantic + final_mutations
    assert validate_calls["smoke"] == (
        1 + len(selftest._RUN_MUTATIONS) + run_semantic
    )
    assert validate_calls["final"] == 1 + final_mutations

    # Lockstep with the mutation-gate receipt contract: the strengthened
    # gate's successful results must still mint mutation-gate evidence
    # (receipt_evidence pins the roster and per-entry fields).
    source_id = "1" * 64
    evidence = build_prerequisite_evidence(
        gate="mutation",
        bindings={
            "source_manifest_id": source_id,
            "raw_input_bundle_id": "2" * 64,
            "model_snapshot_id": "3" * 64,
            "adapter_bundle_id": "4" * 64,
            "fvi_study_id": "5" * 64,
            "environment_contract_id": "6" * 64,
        },
        details={
            "source_execution": {
                "environment": "modal_image",
                "executing_source_manifest_id": source_id,
                "runtime_source_manifest_id": source_id,
            },
            "results": results,
        },
    )
    assert evidence["kind"] == "mutation_gate_evidence"


def test_semantic_recompute_rejects_checksum_consistent_verdict_flip(tmp_path):
    """H-1 headline scenario: an adversary who regenerates SHA256SUMS after
    flipping a serialized cell verdict must still be rejected — by the
    semantic recompute layer, with no checksum error in sight."""
    built = selftest.build_valid_package(tmp_path)
    selftest._mut_flip_verdict(built["run_root"], built["adapter_bundle"])
    writers.write_sha256sums(built["run_root"])

    res = checker.validate_run(
        built["run_root"], backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False, require_package=True,
    )

    assert not res.passed
    assert not any("checksum" in error.lower() for error in res.errors)
    assert any("verdict" in error.lower() for error in res.errors)


@pytest.fixture(scope="module")
def final_built(tmp_path_factory):
    """One receipt-bearing final package, shared read-only across tests."""
    base = tmp_path_factory.mktemp("final_pkg")
    return base / "valid", selftest.build_valid_package(
        base / "valid", final_variant=True
    )


def test_final_profile_package_validates_end_to_end(final_built):
    _, built = final_built
    res = checker.validate_run(
        built["run_root"], backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=True, require_package=True,
    )
    assert res.passed, res.errors
    assert set(built["prerequisite_receipt_ids"]) == {
        "smoke", "mutation", "determinism",
    }


def test_smoke_package_rejected_under_require_final_profile(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    res = checker.validate_run(
        built["run_root"], backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=True, require_package=True,
    )
    assert not res.passed
    assert any("final validation requires" in error for error in res.errors)


@pytest.mark.parametrize("mutation", sorted(_EXPECTED_FINAL_RECEIPT_MUTATIONS))
def test_final_receipt_mutations_rejected_with_regenerated_checksums(
    final_built,
    tmp_path,
    mutation,
):
    """Receipt/evidence-ledger forgeries and evidence-namespace tampers must
    be rejected by their semantic lane itself: SHA256SUMS is regenerated, so
    no stale-checksum rejection can mask a regression in that lane."""
    valid_dir, _ = final_built
    mdir = tmp_path / f"mut_{mutation}"
    shutil.copytree(valid_dir, mdir, symlinks=True)
    rr, bundle = mdir / "runs" / "run", mdir / "adapter_bundle"

    selftest._FINAL_RECEIPT_MUTATIONS[mutation](rr, bundle)
    writers.write_sha256sums(rr)

    res = checker.validate_run(
        rr, backend="modal", adapter_bundle=bundle,
        require_final_profile=True, require_package=True,
    )
    assert not res.passed, mutation
    assert not any("checksum" in error.lower() for error in res.errors)
    markers = _FINAL_MUTATION_ERROR_MARKERS.get(
        mutation, _DEFAULT_FINAL_MUTATION_MARKERS
    )
    assert any(
        marker in error for error in res.errors for marker in markers
    ), res.errors


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


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        # int -> bool coercion: True == 1 under Python ``==``.
        ("bootstrap_seed_true", "run spec bootstrap contract mismatch"),
        # bool -> int coercion: 1 == True under Python ``==``.
        (
            "bootstrap_common_resamples_one",
            "run spec bootstrap contract mismatch",
        ),
        (
            "calibration_both_classes_one",
            "run spec calibration contract mismatch",
        ),
        (
            "profile_nested_bootstrap_seed_true",
            "run spec scientific_profile does not match the canonical profile",
        ),
    ],
)
def test_validate_spec_rejects_bool_int_coerced_constants(
    tmp_path,
    mutation,
    expected_error,
):
    """Bool/int-coerced constants are byte-distinct canonical identities and
    must fail validation even though Python ``==`` treats True == 1."""
    built = selftest.build_valid_package(tmp_path)
    source = json.loads(
        (built["run_root"] / "run_spec.json").read_text(encoding="utf-8")
    )
    candidate = copy.deepcopy(source)
    body = candidate["identity"]
    if mutation == "bootstrap_seed_true":
        assert body["bootstrap"]["seed"] == 1
        body["bootstrap"]["seed"] = True
    elif mutation == "bootstrap_common_resamples_one":
        assert body["bootstrap"]["common_resamples_across_cells"] is True
        body["bootstrap"]["common_resamples_across_cells"] = 1
    elif mutation == "calibration_both_classes_one":
        assert body["calibration"]["both_classes_required"] is True
        body["calibration"]["both_classes_required"] = 1
    else:
        assert body["scientific_profile"]["bootstrap"]["seed"] == 1
        body["scientific_profile"]["bootstrap"]["seed"] = True
    # Self-hash the mutated identity so only the strict constant-block
    # comparison can reject it, not the id check.
    candidate["id"] = identity.compute_id(body)

    path = tmp_path / f"coerced-{mutation}.json"
    path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result = checker.validate_spec(path, require_final_profile=False)
    assert not result.passed, mutation
    assert expected_error in result.errors


def test_validate_spec_rejects_symlink_before_decode(tmp_path, monkeypatch):
    built = selftest.build_valid_package(tmp_path)
    canonical = built["run_root"] / "run_spec.json"
    external = tmp_path / "external-run-spec.json"
    external.write_bytes(canonical.read_bytes())
    selected = tmp_path / "selected-run-spec.json"
    selected.symlink_to(external)

    monkeypatch.setattr(
        checker,
        "load_json",
        lambda _path: pytest.fail("symlinked run spec was decoded"),
    )
    result = checker.validate_spec(
        selected,
        require_final_profile=False,
    )
    assert not result.passed
    assert result.errors == [
        "run spec path must be a non-symlink regular file"
    ]


def test_validate_spec_preserves_missing_path_diagnostic(tmp_path):
    result = checker.validate_spec(
        tmp_path / "missing-run-spec.json",
        require_final_profile=False,
    )
    assert not result.passed
    assert result.errors == ["run spec is missing"]


def test_validate_run_rejects_orphaned_temp_file_in_cells(tmp_path):
    """A mkstemp orphan under cells/ must fail both validation modes."""
    built = selftest.build_valid_package(tmp_path)
    orphan = built["run_root"] / "cells" / "tmpa1b2c3d4"
    orphan.write_bytes(b"orphaned partial write")

    packaged = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=True,
    )
    assert not packaged.passed
    assert any(
        "unexpected non-cell entry in cells/" in error
        for error in packaged.errors
    ), packaged.errors
    assert any(
        "unaudited entry in cells/" in error for error in packaged.errors
    ), packaged.errors

    unpackaged = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=False,
        require_package=False,
    )
    assert not unpackaged.passed
    assert any(
        "unexpected non-cell entry in cells/" in error
        for error in unpackaged.errors
    ), unpackaged.errors


def test_validate_run_rejects_unaudited_top_level_package_entries(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]

    def _validate() -> checker.CheckResult:
        return checker.validate_run(
            run_root,
            backend="modal",
            adapter_bundle=built["adapter_bundle"],
            require_final_profile=False,
            require_package=True,
        )

    stray_file = run_root / "tmpz9y8x7w6"
    stray_file.write_bytes(b"orphaned partial write")
    result = _validate()
    assert not result.passed
    assert any(
        "unaudited package file: 'tmpz9y8x7w6'" in error
        for error in result.errors
    ), result.errors
    stray_file.unlink()

    stray_dir = run_root / "scratch"
    stray_dir.mkdir()
    result = _validate()
    assert not result.passed
    assert any(
        "unaudited package directory: 'scratch'" in error
        for error in result.errors
    ), result.errors
    stray_dir.rmdir()

    orphan_result = run_root / "attempt_results" / "tmpq5w6e7r8"
    orphan_result.write_bytes(b"orphaned partial write")
    result = _validate()
    assert not result.passed
    assert any(
        "unaudited entry in attempt_results/" in error
        for error in result.errors
    ), result.errors
    orphan_result.unlink()

    recovered = _validate()
    assert recovered.passed, recovered.errors


def test_validate_run_rejects_unaudited_evidence_namespace_entries(tmp_path):
    """Extras under evidence/ must fail even when SHA256SUMS is regenerated.

    The checksum bijection constrains hashes, never which paths may appear:
    a checksum-consistent adversary can attest arbitrary bytes under
    evidence/ unless the path policy recurses the namespace for exact
    membership (and rejects entry-free directories, which SHA256SUMS never
    sees at all).
    """
    built = selftest.build_valid_package(tmp_path)
    run_root = built["run_root"]
    evidence = run_root / "evidence"

    def _validate() -> checker.CheckResult:
        writers.write_sha256sums(run_root)
        return checker.validate_run(
            run_root,
            backend="modal",
            adapter_bundle=built["adapter_bundle"],
            require_final_profile=False,
            require_package=True,
        )

    def _assert_rejected(expected_error: str) -> None:
        result = _validate()
        assert not result.passed
        assert any(
            expected_error in error for error in result.errors
        ), (expected_error, result.errors)

    stray_file = evidence / "backdoor.bin"
    stray_file.write_bytes(b"attested but unaudited")
    _assert_rejected("unaudited package file: 'evidence/backdoor.bin'")
    stray_file.unlink()

    stray_tree = evidence / "rogue"
    stray_tree.mkdir()
    (stray_tree / "payload.bin").write_bytes(b"attested but unaudited")
    _assert_rejected("unaudited package directory: 'evidence/rogue'")
    shutil.rmtree(stray_tree)

    bound_sibling = evidence / "source_snapshot" / "extra.bin"
    bound_sibling.write_bytes(b"attested but unaudited")
    _assert_rejected(
        "unaudited package file: 'evidence/source_snapshot/extra.bin'"
    )
    bound_sibling.unlink()

    empty_dir = evidence / "source_snapshot" / "source" / "rogue"
    empty_dir.mkdir()
    _assert_rejected(
        "unaudited empty package directory: "
        "'evidence/source_snapshot/source/rogue'"
    )
    empty_dir.rmdir()

    # A receipts subtree in a smoke package is unaudited even when every
    # file inside uses a canonical receipt name: no smoke lane ever reads it.
    smoke_receipts = evidence / "prerequisite_receipts"
    smoke_receipts.mkdir()
    (smoke_receipts / "smoke.json").write_bytes(b"{}")
    _assert_rejected(
        "unaudited package directory: 'evidence/prerequisite_receipts'"
    )
    shutil.rmtree(smoke_receipts)

    recovered = _validate()
    assert recovered.passed, recovered.errors


def test_final_package_rejects_unaudited_prerequisite_receipt_entries(
    final_built,
    tmp_path,
):
    """Non-canonical names under prerequisite_receipts/ must be rejected:
    the receipt lane audits only the fixed per-gate paths, so extra entries
    would otherwise be attested without any lane reading them."""
    valid_dir, _ = final_built

    def _validate_copy(tamper) -> checker.CheckResult:
        mdir = tmp_path / f"case_{len(list(tmp_path.iterdir()))}"
        shutil.copytree(valid_dir, mdir, symlinks=True)
        rr = mdir / "runs" / "run"
        tamper(rr / "evidence" / "prerequisite_receipts")
        writers.write_sha256sums(rr)
        return checker.validate_run(
            rr,
            backend="modal",
            adapter_bundle=mdir / "adapter_bundle",
            require_final_profile=True,
            require_package=True,
        )

    result = _validate_copy(
        lambda receipts: (receipts / "rogue.json").write_bytes(b"{}")
    )
    assert not result.passed
    assert any(
        "unaudited package file: 'evidence/prerequisite_receipts/rogue.json'"
        in error
        for error in result.errors
    ), result.errors

    def _nested_dir(receipts):
        nested = receipts / "nested"
        nested.mkdir()
        (nested / "payload.bin").write_bytes(b"attested but unaudited")

    result = _validate_copy(_nested_dir)
    assert not result.passed
    assert any(
        "unaudited package directory: 'evidence/prerequisite_receipts/nested'"
        in error
        for error in result.errors
    ), result.errors

    result = _validate_copy(lambda receipts: (receipts / "empty").mkdir())
    assert not result.passed
    assert any(
        "unaudited empty package directory: "
        "'evidence/prerequisite_receipts/empty'" in error
        for error in result.errors
    ), result.errors
