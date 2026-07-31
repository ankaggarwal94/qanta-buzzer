"""Focused package-integrity checks for the StopDFF v5 checker."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .identity import compute_id, loads_no_duplicate_keys, sha256_file
from .manifests import environment_contract_identity
from .writers import validate_prerequisite_receipts

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RECEIPT_GATES = ("smoke", "mutation", "determinism")
_RECEIPT_BINDINGS = (
    "source_manifest_id",
    "raw_input_bundle_id",
    "model_snapshot_id",
    "adapter_bundle_id",
    "fvi_study_id",
    "environment_contract_id",
)


def _err(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _load_json(path: Path) -> Any:
    return loads_no_duplicate_keys(path.read_text(encoding="utf-8"))


def check_complete_checksums(run_root: Path, errors: list[str]) -> None:
    """Validate SHA256SUMS as the complete recursive regular-file inventory."""
    run_root = Path(run_root)
    sums_path = run_root / "SHA256SUMS"
    if sums_path.is_symlink() or not sums_path.is_file():
        errors.append("missing or non-regular SHA256SUMS")
        return

    listed: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        errors.append(f"SHA256SUMS cannot be decoded: {exc}")
        return
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            errors.append(f"malformed SHA256SUMS line: {line!r}")
            continue
        digest, name = parts
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or name == "SHA256SUMS"
        ):
            errors.append(f"unsafe checksum path: {name!r}")
            continue
        if not _is_sha256(digest):
            errors.append(f"invalid checksum digest for {name!r}")
            continue
        if name in listed:
            errors.append(f"duplicate checksum entry: {name!r}")
            continue
        listed[name] = digest

    actual: set[str] = set()
    for path in sorted(run_root.rglob("*")):
        relative = path.relative_to(run_root).as_posix()
        if path.is_symlink():
            errors.append(f"symlink in package: {relative!r}")
            continue
        if path.is_file() and relative != "SHA256SUMS":
            actual.add(relative)

    missing = sorted(actual - set(listed))
    unexpected = sorted(set(listed) - actual)
    for name in missing:
        errors.append(f"unlisted package file: {name!r}")
    for name in unexpected:
        errors.append(f"checksum target missing: {name!r}")
    for name in sorted(actual & set(listed)):
        if sha256_file(run_root / name) != listed[name]:
            errors.append(f"checksum mismatch: {name!r}")


def _packaged_manifest(
    *,
    run_root: Path,
    errors: list[str],
    by_role: dict[str, dict[str, Any]],
    role: str,
    expected_path: str,
    expected_id: Any,
    expected_kind: str | set[str],
) -> dict[str, Any]:
    artifact = by_role.get(role)
    if artifact is None:
        return {}
    if artifact.get("retrieval_path") != expected_path:
        errors.append(f"external artifact {role} must use {expected_path}")
        return {}
    evidence_path = run_root / expected_path
    if evidence_path.is_symlink() or not evidence_path.is_file():
        errors.append(f"packaged {role} evidence is missing")
        return {}
    _err(
        errors,
        evidence_path.stat().st_size == artifact.get("byte_size"),
        f"packaged {role} byte_size mismatch",
    )
    _err(
        errors,
        sha256_file(evidence_path) == artifact.get("sha256"),
        f"packaged {role} sha256 mismatch",
    )
    try:
        manifest = _load_json(evidence_path)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"packaged {role} cannot be decoded: {exc}")
        return {}
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if not isinstance(identity, dict):
        errors.append(f"packaged {role} must be a manifest object")
        return {}
    try:
        recomputed_id = compute_id(identity)
    except (TypeError, ValueError) as exc:
        errors.append(f"packaged {role} identity cannot be canonicalized: {exc}")
        return {}
    _err(
        errors,
        recomputed_id == manifest.get("id"),
        f"packaged {role} manifest id mismatch",
    )
    _err(
        errors,
        manifest.get("id") == artifact.get("content_id") == expected_id,
        f"packaged {role} content id mismatch",
    )
    expected_kinds = expected_kind if isinstance(expected_kind, set) else {expected_kind}
    _err(
        errors,
        identity.get("kind") in expected_kinds,
        f"packaged {role} kind mismatch",
    )
    return manifest


def check_external_artifacts(
    run_root: Path,
    errors: list[str],
    *,
    spec_ids: dict[str, Any],
    evidence_roots: dict[str, Any],
    profile_variant: Any,
    fvi_selected: dict[str, Any],
    environment_claims: dict[str, Any],
) -> None:
    """Recompute all packaged manifest and prerequisite-receipt bindings."""
    run_root = Path(run_root)
    path = run_root / "external_artifacts.json"
    if path.is_symlink() or not path.is_file():
        errors.append("missing external_artifacts.json")
        return
    try:
        payload = _load_json(path)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"external_artifacts.json cannot be decoded: {exc}")
        return
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(artifacts, list):
        errors.append("external_artifacts.json must contain an artifacts list")
        return

    exact_fields = {
        "role",
        "content_id",
        "sha256",
        "byte_size",
        "retrieval_path",
    }
    by_role: dict[str, dict[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or set(artifact) != exact_fields:
            errors.append(f"external artifact {index} fields are noncanonical")
            continue
        role = artifact.get("role")
        if not isinstance(role, str) or not role or role in by_role:
            errors.append(f"external artifact {index} has invalid/duplicate role")
            continue
        by_role[role] = artifact
        _err(
            errors,
            _is_sha256(artifact.get("content_id")),
            f"external artifact {role} content_id must be 64-hex",
        )
        _err(
            errors,
            _is_sha256(artifact.get("sha256")),
            f"external artifact {role} sha256 must be 64-hex",
        )
        _err(
            errors,
            _is_positive_int(artifact.get("byte_size")),
            f"external artifact {role} byte_size must be positive",
        )
        retrieval_path = artifact.get("retrieval_path")
        relative = Path(retrieval_path) if isinstance(retrieval_path, str) else None
        _err(
            errors,
            relative is not None
            and bool(retrieval_path)
            and not relative.is_absolute()
            and ".." not in relative.parts,
            f"external artifact {role} retrieval_path is unsafe",
        )

    receipt_ids = (
        evidence_roots.get("prerequisite_receipts")
        if isinstance(evidence_roots, dict)
        else None
    )
    if not isinstance(receipt_ids, dict):
        receipt_ids = {}
        errors.append("run spec prerequisite_receipts must be an object")
    required = {
        "source_manifest": spec_ids.get("source_manifest_id"),
        "raw_input_manifest": spec_ids.get("raw_input_bundle_id"),
        "model_snapshot_manifest": spec_ids.get("model_snapshot_id"),
        "fvi_study": spec_ids.get("fvi_study_id"),
        "environment_contract": spec_ids.get("environment_contract_id"),
    }
    if profile_variant == "final":
        required.update(
            {
                f"prerequisite_receipt_{gate}": receipt_ids.get(gate)
                for gate in _RECEIPT_GATES
            }
        )
    _err(
        errors,
        set(by_role) == set(required),
        "external artifact roles do not match the canonical package set",
    )
    for role, expected_id in required.items():
        if role not in by_role:
            errors.append(f"missing external artifact role {role}")
        elif by_role[role].get("content_id") != expected_id:
            errors.append(f"external artifact {role} does not match run spec")

    source_manifest = _packaged_manifest(
        run_root=run_root,
        errors=errors,
        by_role=by_role,
        role="source_manifest",
        expected_path="evidence/source_manifest.json",
        expected_id=spec_ids.get("source_manifest_id"),
        expected_kind="source_snapshot",
    )
    raw_manifest = _packaged_manifest(
        run_root=run_root,
        errors=errors,
        by_role=by_role,
        role="raw_input_manifest",
        expected_path="evidence/raw_input_manifest.json",
        expected_id=spec_ids.get("raw_input_bundle_id"),
        expected_kind="raw_input_bundle",
    )
    _packaged_manifest(
        run_root=run_root,
        errors=errors,
        by_role=by_role,
        role="model_snapshot_manifest",
        expected_path="evidence/model_snapshot_manifest.json",
        expected_id=spec_ids.get("model_snapshot_id"),
        expected_kind="model_snapshot",
    )
    raw_identity = raw_manifest.get("identity", {})
    semantic_checks = (
        raw_identity.get("semantic_checks")
        if isinstance(raw_identity, dict)
        else None
    )
    _err(
        errors,
        isinstance(semantic_checks, dict)
        and semantic_checks.get("all_semantic_checks_pass") is True,
        "packaged raw-input semantic checks did not pass",
    )

    fvi_manifest = _packaged_manifest(
        run_root=run_root,
        errors=errors,
        by_role=by_role,
        role="fvi_study",
        expected_path="evidence/fvi_study.json",
        expected_id=spec_ids.get("fvi_study_id"),
        expected_kind={"fvi_study", "fvi_study_fixed"},
    )
    fvi_identity = fvi_manifest.get("identity", {})
    if isinstance(fvi_identity, dict) and fvi_identity:
        _err(
            errors,
            fvi_identity.get("adapter_bundle_id")
            == spec_ids.get("adapter_bundle_id"),
            "packaged FVI evidence does not match the adapter",
        )
        selected = (
            fvi_identity.get("selected_parameters")
            if fvi_identity.get("kind") == "fvi_study"
            else fvi_identity.get("selected")
        )
        _err(
            errors,
            selected == fvi_selected,
            "packaged FVI selection does not match the run spec",
        )

    environment_manifest = _packaged_manifest(
        run_root=run_root,
        errors=errors,
        by_role=by_role,
        role="environment_contract",
        expected_path="evidence/environment_contract.json",
        expected_id=spec_ids.get("environment_contract_id"),
        expected_kind="environment_contract",
    )
    environment_identity = environment_manifest.get("identity", {})
    if isinstance(environment_identity, dict) and environment_identity:
        try:
            expected_environment = environment_contract_identity(
                python_version=environment_claims["python_version"],
                package_versions=environment_claims["package_versions"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"packaged environment cannot be compared: {exc}")
        else:
            _err(
                errors,
                environment_identity == expected_environment,
                "packaged environment evidence does not match environment.json",
            )

    receipts: dict[str, dict[str, Any]] = {}
    if profile_variant == "final":
        for gate in _RECEIPT_GATES:
            role = f"prerequisite_receipt_{gate}"
            receipts[gate] = _packaged_manifest(
                run_root=run_root,
                errors=errors,
                by_role=by_role,
                role=role,
                expected_path=f"evidence/prerequisite_receipts/{gate}.json",
                expected_id=receipt_ids.get(gate),
                expected_kind="prerequisite_receipt",
            )
    identity_bindings = {
        key: spec_ids.get(key)
        for key in _RECEIPT_BINDINGS
    }
    try:
        validate_prerequisite_receipts(
            profile_variant=str(profile_variant),
            identity_bindings=identity_bindings,
            receipt_ids=receipt_ids,
            receipts=receipts,
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"packaged prerequisite receipts are invalid: {exc}")
