"""Focused package-integrity checks for the StopDFF v5 checker."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION
from .identity import compute_id, loads_no_duplicate_keys, sha256_file
from .manifests import FVI_PRODUCER_FILES, environment_contract_identity
from .profile import FVI_MAX_ITERATIONS, FVI_STRICT_REFERENCE, FVI_TOLERANCES
from .receipt_evidence import (
    validate_prerequisite_receipts,
    verify_prerequisite_evidence_bytes,
)

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


def _check_fvi_study_identity(
    identity: dict[str, Any],
    errors: list[str],
) -> None:
    """Validate the complete canonical FVI-study envelope.

    This is deliberately structural: the checker cannot rerun the expensive
    study from a package alone, but it must not accept a label-only manifest as
    proof that the preregistered study occurred.
    """
    expected_fields = {
        "kind",
        "adapter_bundle_id",
        "scientific_protocol_version",
        "candidate_grid",
        "representative_cell_generator",
        "candidate_convergence_results",
        "strict_reference_results",
        "selector_rule",
        "selected_parameters",
        "all96_fit_only_validation",
        "producer_hashes",
    }
    _err(
        errors,
        set(identity) == expected_fields,
        "packaged FVI study fields do not match the canonical contract",
    )
    _err(
        errors,
        identity.get("scientific_protocol_version") == PROTOCOL_VERSION,
        "packaged FVI study protocol version mismatch",
    )
    expected_grid = {
        "tolerance": list(FVI_TOLERANCES),
        "max_iterations": list(FVI_MAX_ITERATIONS),
    }
    _err(
        errors,
        identity.get("candidate_grid") == expected_grid,
        "packaged FVI candidate grid mismatch",
    )
    _err(
        errors,
        identity.get("representative_cell_generator")
        == "representative_24_parity",
        "packaged FVI representative generator mismatch",
    )
    _err(
        errors,
        identity.get("selector_rule")
        == "min_total_iterations__then_larger_tolerance__then_smaller_max_iter",
        "packaged FVI selector rule mismatch",
    )

    expected_pairs = {
        (tolerance, max_iterations)
        for tolerance in FVI_TOLERANCES
        for max_iterations in FVI_MAX_ITERATIONS
    }
    candidate_pairs: set[tuple[str, int]] = set()
    eligible_pairs: set[tuple[str, int]] = set()
    candidates = identity.get("candidate_convergence_results")
    candidates_valid = isinstance(candidates, list) and len(candidates) == len(
        expected_pairs
    )
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {
                "tolerance",
                "max_iterations",
                "total_iterations",
                "all_converged",
                "eligible",
                "ineligibility_reasons",
            }:
                candidates_valid = False
                continue
            pair = (
                candidate.get("tolerance"),
                candidate.get("max_iterations"),
            )
            reasons = candidate.get("ineligibility_reasons")
            if (
                pair not in expected_pairs
                or pair in candidate_pairs
                or not _is_positive_int(candidate.get("total_iterations"))
                or not isinstance(candidate.get("all_converged"), bool)
                or not isinstance(candidate.get("eligible"), bool)
                or not isinstance(reasons, list)
                or not all(isinstance(reason, str) for reason in reasons)
                or (candidate.get("eligible") is True and reasons)
            ):
                candidates_valid = False
            else:
                candidate_pairs.add(pair)
                if candidate["eligible"]:
                    eligible_pairs.add(pair)
    _err(
        errors,
        candidates_valid and candidate_pairs == expected_pairs,
        "packaged FVI candidate results are noncanonical",
    )

    strict = identity.get("strict_reference_results")
    strict_valid = (
        isinstance(strict, dict)
        and set(strict)
        == {"tolerance", "max_iterations", "total_iterations", "all_converged"}
        and strict.get("tolerance") == FVI_STRICT_REFERENCE["tolerance"]
        and strict.get("max_iterations")
        == FVI_STRICT_REFERENCE["max_iterations"]
        and _is_positive_int(strict.get("total_iterations"))
        and strict.get("all_converged") is True
    )
    _err(errors, strict_valid, "packaged FVI strict reference is noncanonical")

    selected = identity.get("selected_parameters")
    selected_pair = (
        selected.get("tolerance"),
        selected.get("max_iterations"),
    ) if isinstance(selected, dict) else (None, None)
    _err(
        errors,
        isinstance(selected, dict)
        and set(selected) == {"tolerance", "max_iterations"}
        and selected_pair in eligible_pairs,
        "packaged FVI selected parameters are not an eligible candidate",
    )
    all96 = identity.get("all96_fit_only_validation")
    _err(
        errors,
        isinstance(all96, dict)
        and set(all96)
        == {"tolerance", "max_iterations", "all_converged", "total_iterations"}
        and (
            all96.get("tolerance"),
            all96.get("max_iterations"),
        )
        == selected_pair
        and all96.get("all_converged") is True
        and _is_positive_int(all96.get("total_iterations")),
        "packaged FVI all-96 validation is noncanonical",
    )

    producer_hashes = identity.get("producer_hashes")
    _err(
        errors,
        isinstance(producer_hashes, dict)
        and set(producer_hashes) == set(FVI_PRODUCER_FILES)
        and all(_is_sha256(value) for value in producer_hashes.values()),
        "packaged FVI producer_hashes do not match the canonical producer set",
    )


def _load_json(path: Path) -> Any:
    return loads_no_duplicate_keys(path.read_text(encoding="utf-8"))


def _check_source_producer_map(
    errors: list[str],
    *,
    source_hashes: dict[str, str],
    claimed: Any,
    expected_basenames: set[str],
    label: str,
) -> None:
    if not isinstance(claimed, dict) or set(claimed) != expected_basenames:
        errors.append(f"{label} does not match the canonical producer set")
        return
    for basename in sorted(expected_basenames):
        expected_path = f"scripts/stopdff_v5/{basename}"
        digest = claimed.get(basename)
        _err(
            errors,
            _is_sha256(digest) and source_hashes.get(expected_path) == digest,
            f"{label} {basename!r} does not match packaged source",
        )


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
    content_kinds = {"source_snapshot", "raw_input_bundle", "model_snapshot"}
    if len(expected_kinds) == 1 and next(iter(expected_kinds)) in content_kinds:
        from .content_manifest import (
            validate_bound_content_manifest,
            validate_content_manifest_document,
        )

        try:
            validate_content_manifest_document(
                manifest,
                manifest_name=f"packaged {role}",
                expected_id=expected_id if isinstance(expected_id, str) else None,
                expected_kind=next(iter(expected_kinds)),
                require_semantic_pass=role == "raw_input_manifest",
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"packaged {role} identity envelope is invalid: {exc}")
            return {}
        packaged_subdirs = {
            "source_manifest": "source_snapshot/source",
            "raw_input_manifest": "raw_inputs/raw",
            "model_snapshot_manifest": "model_snapshot/snapshot",
        }
        name_keys = {
            "source_manifest": "path",
            "raw_input_manifest": "role",
            "model_snapshot_manifest": "path",
        }
        try:
            validate_bound_content_manifest(
                evidence_path.parent,
                manifest_name=evidence_path.name,
                expected_id=expected_id if isinstance(expected_id, str) else None,
                expected_kind=next(iter(expected_kinds)),
                file_key="files",
                name_key=name_keys[role],
                content_subdir=packaged_subdirs[role],
                require_semantic_pass=role == "raw_input_manifest",
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"packaged {role} content inventory is invalid: {exc}")
            return {}
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
    adapter_identity: dict[str, Any],
    recomputed_fvi_study: dict[str, Any] | None = None,
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
    source_entries = source_manifest.get("identity", {}).get("files", [])
    source_hashes: dict[str, str] = {}
    source_entries_valid = isinstance(source_entries, list)
    if isinstance(source_entries, list):
        for entry in source_entries:
            path_value = entry.get("path") if isinstance(entry, dict) else None
            digest = entry.get("sha256") if isinstance(entry, dict) else None
            if (
                not isinstance(path_value, str)
                or not path_value
                or path_value in source_hashes
                or not _is_sha256(digest)
            ):
                source_entries_valid = False
                continue
            source_hashes[path_value] = digest
    _err(
        errors,
        source_entries_valid,
        "packaged source manifest file inventory is noncanonical",
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
    recomputed_trajectory_id = None
    try:
        from .producers import raw_question_trajectory_binding

        recomputed_trajectory_id = raw_question_trajectory_binding(
            run_root / "evidence" / "raw_inputs" / "raw"
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(
            "packaged raw-input question trajectory cannot be recomputed: "
            f"{exc}"
        )
    _err(
        errors,
        isinstance(semantic_checks, dict)
        and semantic_checks.get("all_semantic_checks_pass") is True,
        "packaged raw-input semantic checks did not pass",
    )
    _err(
        errors,
        isinstance(semantic_checks, dict)
        and semantic_checks.get("question_trajectory_binding_id")
        == recomputed_trajectory_id
        == adapter_identity.get("question_trajectory_binding_id"),
        "adapter question trajectory does not match packaged raw inputs",
    )
    raw_files = raw_identity.get("files") if isinstance(raw_identity, dict) else None
    build_metadata_entries = [
        entry
        for entry in raw_files
        if isinstance(entry, dict) and entry.get("role") == "build_metadata.json"
    ] if isinstance(raw_files, list) else []
    retention = (
        adapter_identity.get("mc_retention_evidence")
        if isinstance(adapter_identity, dict)
        else None
    )
    _err(
        errors,
        len(build_metadata_entries) == 1
        and isinstance(retention, dict)
        and build_metadata_entries[0].get("sha256")
        == retention.get("build_metadata_sha256"),
        "adapter retention evidence does not match packaged build metadata",
    )
    _check_source_producer_map(
        errors,
        source_hashes=source_hashes,
        claimed=(
            adapter_identity.get("producer_hashes")
            if isinstance(adapter_identity, dict)
            else None
        ),
        expected_basenames={"adapter_build.py"},
        label="adapter producer_hashes",
    )
    _check_source_producer_map(
        errors,
        source_hashes=source_hashes,
        claimed=(
            evidence_roots.get("producer_hashes")
            if isinstance(evidence_roots, dict)
            else None
        ),
        expected_basenames={"checker.py", "sweep.py"},
        label="run-spec producer_hashes",
    )

    fvi_manifest = _packaged_manifest(
        run_root=run_root,
        errors=errors,
        by_role=by_role,
        role="fvi_study",
        expected_path="evidence/fvi_study.json",
        expected_id=spec_ids.get("fvi_study_id"),
        expected_kind=(
            "fvi_study"
            if profile_variant == "final"
            else {"fvi_study", "fvi_study_fixed"}
        ),
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
        if fvi_identity.get("kind") == "fvi_study":
            _check_fvi_study_identity(fvi_identity, errors)
            if recomputed_fvi_study is not None:
                scientific_fields = {
                    "candidate_grid": "candidate_grid",
                    "representative_cell_generator": (
                        "representative_cell_generator"
                    ),
                    "candidate_convergence_results": (
                        "candidate_convergence_results"
                    ),
                    "strict_reference_results": "strict_reference",
                    "selector_rule": "selector_rule",
                    "selected_parameters": "selected_parameters",
                    "all96_fit_only_validation": (
                        "all96_fit_only_validation"
                    ),
                }
                for identity_field, recomputed_field in scientific_fields.items():
                    _err(
                        errors,
                        fvi_identity.get(identity_field)
                        == recomputed_fvi_study.get(recomputed_field),
                        "packaged FVI study does not match independent "
                        f"recomputation: {identity_field}",
                    )
            producer_hashes = fvi_identity.get("producer_hashes")
            _check_source_producer_map(
                errors,
                source_hashes=source_hashes,
                claimed=producer_hashes,
                expected_basenames=set(FVI_PRODUCER_FILES),
                label="packaged FVI producer_hashes",
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
        if profile_variant == "final":
            for gate in _RECEIPT_GATES:
                evidence_path = (
                    run_root
                    / "evidence"
                    / "prerequisite_receipts"
                    / f"{gate}.evidence.json"
                )
                if evidence_path.is_symlink() or not evidence_path.is_file():
                    raise ValueError(f"missing packaged {gate} prerequisite evidence")
                receipt_identity = receipts[gate]["identity"]
                verify_prerequisite_evidence_bytes(
                    gate=gate,
                    bindings=receipt_identity["bindings"],
                    receipt_evidence=receipt_identity["evidence"],
                    data=evidence_path.read_bytes(),
                )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"packaged prerequisite receipts are invalid: {exc}")
