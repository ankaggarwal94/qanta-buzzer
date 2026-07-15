"""Content-addressed identity builders (IDENTITY_AND_ARTIFACT_CONTRACT.md 2-10).

Each function returns an ``identity`` dict (scientific-decimal quantities as strings).
Use identity.build_manifest(identity, **volatile) to attach an ``id`` and volatile fields.
"""
from __future__ import annotations

from typing import Any

from . import PROFILE_NAME, PROTOCOL_VERSION, SCHEMA_VERSION
from .profile import (
    BOOTSTRAP,
    CALIBRATION,
    CONTINUATION,
    FORMAT_CONDITION,
    GATE,
    profile_static_identity,
)
from .rewards import REWARD_SCHEDULE_STRINGS


def source_manifest_identity(
    *,
    git_sha: str,
    files: list[dict[str, Any]],  # each: {path, mode, size(int), sha256}
    pyproject_sha256: str,
    uv_lock_sha256: str,
) -> dict[str, Any]:
    ordered = sorted(files, key=lambda f: f["path"])
    return {
        "kind": "source_snapshot",
        "protocol_version": PROTOCOL_VERSION,
        "git_sha": git_sha,
        "files": [
            {
                "path": f["path"],
                "mode": str(f["mode"]),
                "size": int(f["size"]),
                "sha256": f["sha256"],
            }
            for f in ordered
        ],
        "pyproject_toml_sha256": pyproject_sha256,
        "uv_lock_sha256": uv_lock_sha256,
    }


def raw_input_identity(
    *,
    files: list[dict[str, Any]],  # each: {role, size, sha256}
    semantic_checks: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(files, key=lambda f: f["role"])
    return {
        "kind": "raw_input_bundle",
        "files": [
            {"role": f["role"], "size": int(f["size"]), "sha256": f["sha256"]} for f in ordered
        ],
        "semantic_checks": semantic_checks,
    }


def model_snapshot_identity(
    *,
    model_id: str,
    revision: str,
    files: list[dict[str, Any]],  # each: {path, size, sha256}
    sentence_transformers_version: str,
    transformers_version: str,
) -> dict[str, Any]:
    ordered = sorted(files, key=lambda f: f["path"])
    return {
        "kind": "model_snapshot",
        "model_id": model_id,
        "model_revision": revision,
        "trust_remote_code": False,
        "files": [
            {"path": f["path"], "size": int(f["size"]), "sha256": f["sha256"]} for f in ordered
        ],
        "sentence_transformers_version": sentence_transformers_version,
        "transformers_version": transformers_version,
    }


def adapter_identity(
    *,
    source_manifest_id: str,
    raw_input_bundle_id: str,
    model_snapshot_id: str,
    scoring_spec: dict[str, Any],
    fit_split: str,
    eval_split: str,
    schema_columns: list[str],
    fit_row_count: int,
    eval_row_count: int,
    fit_rows_sha256: str,
    eval_rows_sha256: str,
    mc_coverage: dict[str, Any],
    mc_retention: dict[str, Any],
    producer_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "kind": "adapter_bundle",
        "source_manifest_id": source_manifest_id,
        "raw_input_bundle_id": raw_input_bundle_id,
        "model_snapshot_id": model_snapshot_id,
        "scoring_spec": scoring_spec,
        "fit_split": fit_split,
        "eval_split": eval_split,
        "schema_columns": list(schema_columns),
        "fit_row_count": int(fit_row_count),
        "eval_row_count": int(eval_row_count),
        "fit_rows_sha256": fit_rows_sha256,
        "eval_rows_sha256": eval_rows_sha256,
        "mc_coverage_evidence": mc_coverage,
        "mc_retention_evidence": mc_retention,
        "producer_hashes": producer_hashes,
    }


def fvi_study_identity(
    *,
    adapter_bundle_id: str,
    candidate_grid: dict[str, Any],
    representative_generator: str,
    candidate_results: list[dict[str, Any]],
    strict_reference_results: dict[str, Any],
    selector_rule: str,
    selected_parameters: dict[str, Any],
    all96_validation: dict[str, Any],
    producer_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "kind": "fvi_study",
        "adapter_bundle_id": adapter_bundle_id,
        "scientific_protocol_version": PROTOCOL_VERSION,
        "candidate_grid": candidate_grid,
        "representative_cell_generator": representative_generator,
        "candidate_convergence_results": candidate_results,
        "strict_reference_results": strict_reference_results,
        "selector_rule": selector_rule,
        "selected_parameters": selected_parameters,
        "all96_fit_only_validation": all96_validation,
        "producer_hashes": producer_hashes,
    }


def environment_contract_identity(
    *,
    python_version: str,
    package_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "kind": "environment_contract",
        "python_version": python_version,
        "package_versions": {k: package_versions[k] for k in sorted(package_versions)},
    }


def run_spec_identity(
    *,
    source_manifest_id: str,
    raw_input_bundle_id: str,
    model_snapshot_id: str,
    adapter_bundle_id: str,
    fvi_study_id: str,
    bootstrap_plan_id: str,
    environment_contract_id: str,
    fvi_selected: dict[str, Any],  # {tolerance, max_iterations} as strings/ints
    replicate_count: int,
    profile_variant: str,  # "final" | "smoke"
) -> dict[str, Any]:
    static = profile_static_identity()
    return {
        "kind": "run_spec",
        "profile_name": PROFILE_NAME,
        "schema_version": SCHEMA_VERSION,
        "profile_variant": profile_variant,
        "scientific_profile": static,
        "identity": {
            "source_manifest_id": source_manifest_id,
            "raw_input_bundle_id": raw_input_bundle_id,
            "model_snapshot_id": model_snapshot_id,
            "adapter_bundle_id": adapter_bundle_id,
            "fvi_study_id": fvi_study_id,
            "bootstrap_plan_id": bootstrap_plan_id,
            "environment_contract_id": environment_contract_id,
        },
        "reward_schedules": {n: dict(REWARD_SCHEDULE_STRINGS[n]) for n in REWARD_SCHEDULE_STRINGS},
        "calibration": CALIBRATION,
        "continuation": CONTINUATION,
        "gate": GATE,
        "format_condition": [FORMAT_CONDITION],
        "splits": {"fit": "val", "eval": "test"},
        "fvi_selected": {
            "damping": "0.5",
            "tolerance": str(fvi_selected["tolerance"]),
            "max_iterations": int(fvi_selected["max_iterations"]),
            "required_consecutive_converged_iterations": 2,
            "numeric_dtype": "float64",
        },
        "bootstrap": {**BOOTSTRAP, "replicate_count": int(replicate_count)},
    }


def cell_fingerprint_identity(
    *,
    run_spec_id: str,
    cell: dict[str, str],
    adapter_fit_rows_sha256: str,
    adapter_eval_rows_sha256: str,
    calibrator_parameters: dict[str, Any],
    fvi_settings: dict[str, Any],
    producer_hashes: dict[str, str],
    myopic_artifact_sha256: str,
) -> dict[str, Any]:
    from .profile import CELL_AXES

    normalized_cell = {axis: cell[axis] for axis in CELL_AXES}
    return {
        "kind": "cell_fingerprint",
        "run_spec_id": run_spec_id,
        "cell": normalized_cell,
        "adapter_fit_rows_sha256": adapter_fit_rows_sha256,
        "adapter_eval_rows_sha256": adapter_eval_rows_sha256,
        "calibrator_parameters": calibrator_parameters,
        "fvi_settings": fvi_settings,
        "producer_hashes": producer_hashes,
        "myopic_artifact_sha256": myopic_artifact_sha256,
    }
