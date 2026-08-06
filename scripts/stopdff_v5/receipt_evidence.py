"""Canonical, byte-bound evidence for prerequisite success receipts."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .identity import compute_id, loads_no_duplicate_keys

FULL_RECEIPT_BINDINGS = {
    "source_manifest_id",
    "raw_input_bundle_id",
    "model_snapshot_id",
    "adapter_bundle_id",
    "fvi_study_id",
    "environment_contract_id",
}
DETERMINISM_BINDINGS = {
    "source_manifest_id",
    "raw_input_bundle_id",
    "model_snapshot_id",
    "adapter_bundle_id",
}
DETERMINISM_FILES = (
    "build_metadata.json",
    "calibration.json",
    "eval_rows.jsonl.gz",
    "fit_rows.jsonl.gz",
)
MUTATION_ROSTER = (
    "<baseline valid>",
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
    "invalid_adapter_row_hash",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_FIELDS = {
    "smoke": {"kind", "schema_version", "bindings", "run_spec", "aggregate"},
    "mutation": {"kind", "schema_version", "bindings", "results"},
    "determinism": {
        "kind",
        "schema_version",
        "bindings",
        "first_adapter_manifest",
        "second_adapter_manifest",
        "first_file_sha256",
        "second_file_sha256",
    },
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def prerequisite_evidence_bytes(evidence: dict[str, Any]) -> bytes:
    """Return the single accepted byte encoding for packaged gate evidence."""
    if not isinstance(evidence, dict):
        raise ValueError("prerequisite evidence must be an object")
    try:
        encoded = json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"prerequisite evidence is not canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def prerequisite_evidence_sha256(evidence: dict[str, Any]) -> str:
    """Hash the exact canonical evidence bytes persisted beside a receipt."""
    return hashlib.sha256(prerequisite_evidence_bytes(evidence)).hexdigest()


def _validate_bindings(gate: str, bindings: Any) -> dict[str, str]:
    required = DETERMINISM_BINDINGS if gate == "determinism" else FULL_RECEIPT_BINDINGS
    if not isinstance(bindings, dict) or set(bindings) != required:
        raise ValueError(f"{gate} prerequisite evidence bindings mismatch")
    if any(not _is_sha256(value) for value in bindings.values()):
        raise ValueError(f"{gate} prerequisite evidence binding is not SHA-256")
    return {key: bindings[key] for key in sorted(bindings)}


def _validate_smoke(evidence: dict[str, Any], bindings: dict[str, str]) -> None:
    run_spec = evidence.get("run_spec")
    spec_identity = run_spec.get("identity") if isinstance(run_spec, dict) else None
    if (
        not isinstance(spec_identity, dict)
        or set(run_spec) != {"id", "identity"}
        or compute_id(spec_identity) != run_spec.get("id")
        or spec_identity.get("kind") != "run_spec"
        or spec_identity.get("profile_variant") != "smoke"
    ):
        raise ValueError("smoke evidence run spec is invalid")
    spec_bindings = spec_identity.get("identity")
    if not isinstance(spec_bindings, dict) or any(
        spec_bindings.get(key) != value for key, value in bindings.items()
    ):
        raise ValueError("smoke evidence run spec bindings mismatch")
    roots = spec_identity.get("evidence_roots")
    if (
        not isinstance(roots, dict)
        or roots.get("prerequisite_receipts") != {}
    ):
        raise ValueError("smoke evidence run spec claims prerequisite receipts")

    aggregate = evidence.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("smoke evidence aggregate must be an object")
    requested = aggregate.get("requested")
    completed = aggregate.get("completed")
    if (
        aggregate.get("profile_variant") != "smoke"
        or aggregate.get("run_spec_id") != run_spec["id"]
        or aggregate.get("adapter_bundle_id") != bindings["adapter_bundle_id"]
        or aggregate.get("fvi_study_id") != bindings["fvi_study_id"]
        or aggregate.get("release_status") != "VALID"
        or aggregate.get("release_reasons") != []
        or isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested <= 0
        or completed != requested
        or aggregate.get("failed") != 0
        or aggregate.get("skipped") != 0
    ):
        raise ValueError("smoke evidence does not prove a complete VALID run")


def _validate_mutation(evidence: dict[str, Any]) -> None:
    results = evidence.get("results")
    if not isinstance(results, list) or tuple(
        result.get("mutation") if isinstance(result, dict) else None
        for result in results
    ) != MUTATION_ROSTER:
        raise ValueError("mutation evidence roster mismatch")
    expected_fields = {"mutation", "expected", "passed_check", "ok", "errors"}
    for index, result in enumerate(results):
        if not isinstance(result, dict) or set(result) != expected_fields:
            raise ValueError("mutation evidence result fields mismatch")
        expected = "PASS" if index == 0 else "REJECT"
        passed = index == 0
        errors = result.get("errors")
        if (
            result.get("expected") != expected
            or result.get("passed_check") is not passed
            or result.get("ok") is not True
            or not isinstance(errors, list)
            or any(not isinstance(error, str) for error in errors)
        ):
            raise ValueError(f"mutation evidence outcome mismatch: {result.get('mutation')}")


def _validate_adapter_manifest(
    manifest: Any,
    *,
    bindings: dict[str, str],
) -> dict[str, Any]:
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if (
        not isinstance(identity, dict)
        or set(manifest) != {"id", "identity"}
        or compute_id(identity) != manifest.get("id")
        or manifest.get("id") != bindings["adapter_bundle_id"]
        or identity.get("kind") != "adapter_bundle"
    ):
        raise ValueError("determinism evidence adapter manifest is invalid")
    for key in ("source_manifest_id", "raw_input_bundle_id", "model_snapshot_id"):
        if identity.get(key) != bindings[key]:
            raise ValueError("determinism evidence adapter bindings mismatch")
    return identity


def _validate_determinism(
    evidence: dict[str, Any],
    bindings: dict[str, str],
) -> None:
    first_manifest = evidence.get("first_adapter_manifest")
    second_manifest = evidence.get("second_adapter_manifest")
    first_identity = _validate_adapter_manifest(first_manifest, bindings=bindings)
    second_identity = _validate_adapter_manifest(second_manifest, bindings=bindings)
    if first_identity != second_identity:
        raise ValueError("determinism evidence adapter identities differ")

    first_hashes = evidence.get("first_file_sha256")
    second_hashes = evidence.get("second_file_sha256")
    expected_files = set(DETERMINISM_FILES)
    if (
        not isinstance(first_hashes, dict)
        or not isinstance(second_hashes, dict)
        or set(first_hashes) != expected_files
        or set(second_hashes) != expected_files
        or any(not _is_sha256(value) for value in first_hashes.values())
        or any(not _is_sha256(value) for value in second_hashes.values())
        or first_hashes != second_hashes
    ):
        raise ValueError("determinism evidence file hashes differ or are incomplete")
    retention = first_identity.get("mc_retention_evidence")
    expected_hashes = {
        "fit_rows.jsonl.gz": first_identity.get("fit_rows_sha256"),
        "eval_rows.jsonl.gz": first_identity.get("eval_rows_sha256"),
        "calibration.json": first_identity.get("calibration_sha256"),
        "build_metadata.json": (
            retention.get("build_metadata_sha256")
            if isinstance(retention, dict)
            else None
        ),
    }
    if first_hashes != expected_hashes:
        raise ValueError("determinism evidence hashes do not match adapter identity")


def validate_prerequisite_evidence(
    *,
    gate: str,
    bindings: dict[str, str],
    evidence: dict[str, Any],
) -> None:
    """Validate one gate's exact self-contained evidence contract."""
    if gate not in _EVIDENCE_FIELDS:
        raise ValueError(f"unknown prerequisite gate {gate!r}")
    normalized_bindings = _validate_bindings(gate, bindings)
    if not isinstance(evidence, dict) or set(evidence) != _EVIDENCE_FIELDS[gate]:
        raise ValueError(f"{gate} prerequisite evidence fields mismatch")
    if evidence.get("kind") != f"{gate}_gate_evidence" or evidence.get("schema_version") != 1:
        raise ValueError(f"{gate} prerequisite evidence envelope mismatch")
    if evidence.get("bindings") != normalized_bindings:
        raise ValueError(f"{gate} prerequisite evidence bindings mismatch")
    if gate == "smoke":
        _validate_smoke(evidence, normalized_bindings)
    elif gate == "mutation":
        _validate_mutation(evidence)
    else:
        _validate_determinism(evidence, normalized_bindings)
    prerequisite_evidence_bytes(evidence)


def build_prerequisite_evidence(
    *,
    gate: str,
    bindings: dict[str, str],
    details: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate a gate evidence object before a receipt is issued."""
    normalized_bindings = _validate_bindings(gate, bindings)
    evidence = {
        "kind": f"{gate}_gate_evidence",
        "schema_version": 1,
        "bindings": normalized_bindings,
        **details,
    }
    validate_prerequisite_evidence(
        gate=gate,
        bindings=normalized_bindings,
        evidence=evidence,
    )
    return evidence


def verify_prerequisite_evidence_bytes(
    *,
    gate: str,
    bindings: dict[str, str],
    receipt_evidence: dict[str, Any],
    data: bytes,
) -> dict[str, Any]:
    """Verify canonical packaged bytes, receipt digest, and gate semantics."""
    if set(receipt_evidence) != {"evidence_sha256"} or not _is_sha256(
        receipt_evidence.get("evidence_sha256")
    ):
        raise ValueError(f"{gate} receipt evidence digest is invalid")
    if hashlib.sha256(data).hexdigest() != receipt_evidence["evidence_sha256"]:
        raise ValueError(f"{gate} prerequisite evidence digest mismatch")
    try:
        evidence = loads_no_duplicate_keys(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{gate} prerequisite evidence is invalid JSON: {exc}") from exc
    if not isinstance(evidence, dict) or prerequisite_evidence_bytes(evidence) != data:
        raise ValueError(f"{gate} prerequisite evidence bytes are noncanonical")
    validate_prerequisite_evidence(
        gate=gate,
        bindings=bindings,
        evidence=evidence,
    )
    return evidence
