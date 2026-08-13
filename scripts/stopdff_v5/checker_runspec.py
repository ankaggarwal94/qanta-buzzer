"""Run-spec contract validation (ACCEPTANCE_CONTRACT.md).

Validates an untrusted run-spec manifest against the canonical profile
contract: envelope shape, constant blocks, identity graph, evidence roots,
FVI selection, gate, and bootstrap contract. Extracted verbatim from
``checker``, which re-exports ``validate_spec`` at its historical path.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .checker_common import (
    CheckResult,
    _canonical_path_issue,
    _err,
    _producer_hash_errors,
    _scientific_equal,
    load_json,
)
from .identity import compute_id, is_sha256_hex
from .profile import (
    BOOTSTRAP,
    CALIBRATION,
    CONTINUATION,
    FINAL_REPLICATES,
    FORMAT_CONDITION,
    FVI_MAX_ITERATIONS,
    FVI_TOLERANCES,
    GATE,
    SMOKE_REPLICATES,
    profile_static_identity,
)
from .rewards import REWARD_SCHEDULE_STRINGS

def _run_spec_manifest_envelope_errors(manifest: Any) -> list[str]:
    if not isinstance(manifest, dict) or set(manifest) != {"id", "identity"}:
        return ["run spec manifest fields do not match the canonical envelope"]
    return []


def _run_spec_errors(
    spec_identity: Any,
    spec_id: Any,
    *,
    require_final_profile: bool,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec_identity, dict):
        return ["run spec identity must be an object"]
    try:
        _err(
            errors,
            compute_id(spec_identity) == spec_id,
            "run-spec id does not match its identity",
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"run spec identity cannot be canonicalized: {exc}")

    expected_keys = {
        "kind",
        "profile_name",
        "schema_version",
        "profile_variant",
        "scientific_profile",
        "identity",
        "evidence_roots",
        "reward_schedules",
        "calibration",
        "continuation",
        "gate",
        "format_condition",
        "splits",
        "fvi_selected",
        "bootstrap",
    }
    _err(
        errors,
        set(spec_identity) == expected_keys,
        "run spec fields do not match the canonical contract",
    )
    _err(errors, spec_identity.get("kind") == "run_spec", "unexpected run spec kind")
    _err(
        errors,
        spec_identity.get("profile_name")
        == "stopdff_bucketed_dp_paired_v2",
        f"unexpected profile_name {spec_identity.get('profile_name')!r}",
    )
    _err(
        errors,
        spec_identity.get("schema_version") == 2,
        "run spec schema_version must be 2",
    )
    variant = spec_identity.get("profile_variant")
    _err(
        errors,
        isinstance(variant, str) and variant in {"smoke", "final"},
        "invalid run spec profile_variant",
    )
    if require_final_profile:
        _err(errors, variant == "final", "final validation requires final run spec")

    # Constant blocks use _scientific_equal: Python's ``==`` coerces bool/int
    # (True == 1), so a spec with byte-distinct canonical identity (e.g.
    # ``"seed": true``) would otherwise still validate as canonical.
    _err(
        errors,
        _scientific_equal(
            spec_identity.get("scientific_profile"), profile_static_identity()
        ),
        "run spec scientific_profile does not match the canonical profile",
    )
    _err(
        errors,
        _scientific_equal(
            spec_identity.get("reward_schedules"),
            {
                name: dict(REWARD_SCHEDULE_STRINGS[name])
                for name in REWARD_SCHEDULE_STRINGS
            },
        ),
        "run spec reward_schedules mismatch",
    )
    _err(
        errors,
        _scientific_equal(spec_identity.get("calibration"), CALIBRATION),
        "run spec calibration contract mismatch",
    )
    _err(
        errors,
        _scientific_equal(spec_identity.get("continuation"), CONTINUATION),
        "run spec continuation contract mismatch",
    )
    _err(
        errors,
        spec_identity.get("format_condition") == [FORMAT_CONDITION],
        "run spec format_condition mismatch",
    )
    _err(
        errors,
        spec_identity.get("splits") == {"fit": "val", "eval": "test"},
        "run spec split roles mismatch",
    )

    ids = spec_identity.get("identity")
    required_ids = {
        "source_manifest_id",
        "raw_input_bundle_id",
        "model_snapshot_id",
        "adapter_bundle_id",
        "fvi_study_id",
        "bootstrap_plan_id",
        "environment_contract_id",
        "resource_summary_id",
    }
    _err(
        errors,
        isinstance(ids, dict) and set(ids) == required_ids,
        "run spec identity graph fields mismatch",
    )
    if isinstance(ids, dict):
        for key in sorted(required_ids):
            _err(
                errors,
                is_sha256_hex(ids.get(key)),
                f"run spec {key} must be a 64-hex ID",
            )

    evidence_roots = spec_identity.get("evidence_roots")
    expected_root_keys = {
        "myopic_artifact_sha256",
        "producer_hashes",
        "prerequisite_receipts",
    }
    _err(
        errors,
        isinstance(evidence_roots, dict)
        and set(evidence_roots) == expected_root_keys,
        "run spec evidence_roots fields mismatch",
    )
    if isinstance(evidence_roots, dict):
        _err(
            errors,
            is_sha256_hex(evidence_roots.get("myopic_artifact_sha256")),
            "run spec myopic_artifact_sha256 must be 64-hex",
        )
        errors.extend(
            _producer_hash_errors(
                evidence_roots.get("producer_hashes"),
                label="run spec producer_hashes",
                required_keys={"checker.py", "sweep.py"},
            )
        )
        receipts = evidence_roots.get("prerequisite_receipts")
        expected_receipt_keys = (
            {"smoke", "mutation", "determinism"}
            if variant == "final"
            else set()
        )
        _err(
            errors,
            isinstance(receipts, dict)
            and set(receipts) == expected_receipt_keys,
            "run spec prerequisite_receipts do not match profile variant",
        )
        if isinstance(receipts, dict):
            for key, receipt_id in receipts.items():
                _err(
                    errors,
                    is_sha256_hex(receipt_id),
                    f"run spec prerequisite receipt {key!r} must be 64-hex",
                )

    fvi_selected = spec_identity.get("fvi_selected")
    expected_fvi_keys = {
        "damping",
        "tolerance",
        "max_iterations",
        "required_consecutive_converged_iterations",
        "numeric_dtype",
    }
    _err(
        errors,
        isinstance(fvi_selected, dict)
        and set(fvi_selected) == expected_fvi_keys,
        "run spec FVI fields mismatch",
    )
    if isinstance(fvi_selected, dict):
        _err(errors, fvi_selected.get("damping") == "0.5", "FVI damping mismatch")
        _err(
            errors,
            fvi_selected.get("tolerance") in FVI_TOLERANCES,
            "FVI tolerance is not an allowed candidate",
        )
        _err(
            errors,
            fvi_selected.get("max_iterations") in FVI_MAX_ITERATIONS,
            "FVI max_iterations is not an allowed candidate",
        )
        _err(
            errors,
            fvi_selected.get("required_consecutive_converged_iterations") == 2,
            "FVI convergence-count contract mismatch",
        )
        _err(
            errors,
            fvi_selected.get("numeric_dtype") == "float64",
            "FVI numeric_dtype mismatch",
        )

    gate = spec_identity.get("gate")
    _err(errors, isinstance(gate, dict) and set(gate) == set(GATE), "gate fields mismatch")
    if isinstance(gate, dict):
        for key, expected in GATE.items():
            if key in (
                "allow_low_mc_retention",
                "allow_incomplete_mc_coverage",
            ):
                _err(errors, isinstance(gate.get(key), bool), f"gate {key} must be boolean")
            else:
                _err(errors, gate.get(key) == expected, f"gate {key} mismatch")

    bootstrap = spec_identity.get("bootstrap")
    expected_replicates = (
        FINAL_REPLICATES if variant == "final" else SMOKE_REPLICATES
    )
    expected_bootstrap = {
        **BOOTSTRAP,
        "replicate_count": expected_replicates,
    }
    _err(
        errors,
        _scientific_equal(bootstrap, expected_bootstrap),
        "run spec bootstrap contract mismatch",
    )

    def _contains_placeholder(value: Any) -> bool:
        if isinstance(value, str):
            return bool(re.search(r"<[^<>]+>", value))
        if isinstance(value, dict):
            return any(_contains_placeholder(item) for item in value.values())
        if isinstance(value, list):
            return any(_contains_placeholder(item) for item in value)
        return False

    _err(
        errors,
        not _contains_placeholder(spec_identity),
        "run spec contains an unresolved template placeholder",
    )
    return errors


def _validate_spec_impl(
    spec_path: Path,
    *,
    require_final_profile: bool,
) -> CheckResult:
    """Validate a run-spec manifest against the canonical profile contract.

    Parameters
    ----------
    spec_path
        Path to the JSON run-spec manifest.
    require_final_profile
        Whether the manifest must select the final rather than smoke profile.

    Returns
    -------
    CheckResult
        Structured validation status and any contract errors.
    """
    spec_path = Path(spec_path)
    path_issue = _canonical_path_issue(
        spec_path,
        expect_directory=False,
    )
    if path_issue == "missing":
        return CheckResult(passed=False, errors=["run spec is missing"])
    if path_issue is not None:
        return CheckResult(
            passed=False,
            errors=["run spec path must be a non-symlink regular file"],
        )
    try:
        spec = load_json(spec_path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return CheckResult(passed=False, errors=[f"run spec cannot be decoded: {exc}"])
    if not isinstance(spec, dict):
        return CheckResult(passed=False, errors=["run spec manifest must be an object"])
    errors = _run_spec_manifest_envelope_errors(spec)
    errors.extend(_run_spec_errors(
        spec.get("identity"),
        spec.get("id"),
        require_final_profile=require_final_profile,
    ))
    return CheckResult(passed=not errors, errors=errors)


def validate_spec(spec_path: Path, *, require_final_profile: bool) -> CheckResult:
    """Validate an untrusted run-spec manifest.

    Parameters
    ----------
    spec_path
        Path to the JSON run-spec manifest.
    require_final_profile
        Whether the manifest must select the final rather than smoke profile.

    Returns
    -------
    CheckResult
        Structured validation status; malformed data never escapes this boundary.
    """
    try:
        return _validate_spec_impl(
            spec_path,
            require_final_profile=require_final_profile,
        )
    except (
        AttributeError,
        EOFError,
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        return CheckResult(
            passed=False,
            errors=[
                "run spec cannot be validated safely: "
                f"{type(exc).__name__}: {exc}"
            ],
        )
