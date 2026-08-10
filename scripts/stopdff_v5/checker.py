"""Standalone checker (ACCEPTANCE_CONTRACT.md).

Independently recomputes cell/family/release statistics from the adapter rows and the
run package, never trusting serialized verdict fields. Also validates package structure:
backend-manifest exclusivity, attempt history, bootstrap plan, safe checksums,
external_artifacts, report semantics, and PNG validity.

No standalone validation requires another backend or comparison policy.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import struct
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .attempt_history import load_attempt_history
from .bootstrap import build_bootstrap_plan, cell_bootstrap_stats, family_statistic
from .cellcompute import compute_cell, prepare_cell_inputs
from .checker_calibration import platt_phase_errors
from .checker_package import (
    check_complete_checksums,
    check_external_artifacts,
    inspect_packaged_fvi_manifest_kind,
)
from .identity import compute_id, loads_no_duplicate_keys, sha256_file
from .manifests import (
    ADAPTER_SCORING_SPEC,
    ENVIRONMENT_PACKAGES,
    environment_contract_identity,
)
from .profile import (
    BOOTSTRAP,
    CALIBRATION,
    CONTINUATION,
    EXPECTED_CELLS,
    FINAL_REPLICATES,
    FORMAT_CONDITION,
    FVI_MAX_ITERATIONS,
    FVI_TOLERANCES,
    GATE,
    SMOKE_REPLICATES,
    cell_key_str,
    full_grid,
    profile_static_identity,
    smoke_cells,
)
from .rewards import REWARD_SCHEDULE_STRINGS
from .verdicts import (
    ceiling_flags,
    cell_verdict,
    coverage_clean,
    family_verdict,
    release_validity,
)

_FLOAT_TOL = 1e-9
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_INTERRUPTED_REASON = "terminal_result_missing_at_resume"
_FVI_STUDY_CACHE: dict[tuple[str, str, str, str], dict[str, Any]] = {}


@dataclass
class CheckResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    recomputed: dict[str, Any] = field(default_factory=dict)


def _err(errors: list[str], cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_strict_int(value: Any, *, minimum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _is_finite_number(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return (
        math.isfinite(number)
        and (minimum is None or number >= minimum)
        and (maximum is None or number <= maximum)
    )


def _scientific_equal(actual: Any, expected: Any) -> bool:
    """Compare recomputable JSON claims without Python's coercive equality."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return _is_strict_int(actual) and actual == expected
    if isinstance(expected, float):
        return _is_finite_number(actual) and math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        )
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _scientific_equal(actual[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _scientific_equal(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _is_quantized_number(value: Any, *, decimal_places: int) -> bool:
    """Return whether a finite number is unchanged by producer rounding."""
    if not _is_finite_number(value):
        return False
    number = float(value)
    return number == round(number, decimal_places)


def _mc_retention_errors(
    value: Any,
    *,
    bundle_dir: Path,
    fit_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    fit_items: set[str],
    eval_items: set[str],
) -> list[str]:
    """Independently validate the identity-bound MC retention decision."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["adapter MC retention evidence must be an object"]
    expected_fields = {
        "build_metadata_sha256",
        "threshold_profile",
        "splits",
        "fit_rows",
        "eval_rows",
    }
    _err(
        errors,
        set(value) == expected_fields,
        "adapter MC retention evidence fields mismatch",
    )
    _err(
        errors,
        _is_sha256(value.get("build_metadata_sha256")),
        "adapter MC retention build-metadata hash is invalid",
    )
    _err(
        errors,
        value.get("threshold_profile") == "full",
        "adapter MC retention threshold profile must be full",
    )
    _err(
        errors,
        _is_strict_int(value.get("fit_rows"), minimum=0)
        and value.get("fit_rows") == len(fit_rows),
        "adapter MC retention fit_rows does not match row bytes",
    )
    _err(
        errors,
        _is_strict_int(value.get("eval_rows"), minimum=0)
        and value.get("eval_rows") == len(eval_rows),
        "adapter MC retention eval_rows does not match row bytes",
    )

    splits = value.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"fit", "eval"}:
        errors.append("adapter MC retention splits must be exactly fit/eval")
        return errors

    try:
        from scripts._audit_gates import (
            build_retention_metadata,
            load_mc_build_metadata,
        )

        build_metadata = load_mc_build_metadata(bundle_dir)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"adapter build_metadata.json cannot be validated: {exc}")
        build_metadata = None
    if isinstance(build_metadata, dict):
        _err(
            errors,
            build_metadata.get("status") == "loaded"
            and build_metadata.get("source_sha256")
            == value.get("build_metadata_sha256"),
            "adapter MC retention build-metadata bytes do not match evidence",
        )
    expected_decision_fields = {
        "applies",
        "split",
        "threshold",
        "retention_rate",
        "raw_count",
        "retained_count",
        "dropped_count",
        "passed",
        "overridden",
        "override_flag",
        "effective_pass",
    }
    for role, split, retained_items in (
        ("fit", "val", len(fit_items)),
        ("eval", "test", len(eval_items)),
    ):
        decision = splits.get(role)
        if not isinstance(decision, dict):
            errors.append(f"adapter MC retention {role} decision must be an object")
            continue
        _err(
            errors,
            set(decision) == expected_decision_fields,
            f"adapter MC retention {role} decision fields mismatch",
        )
        _err(
            errors,
            decision.get("applies") is True and decision.get("split") == split,
            f"adapter MC retention {role} split binding mismatch",
        )
        raw = decision.get("raw_count")
        retained = decision.get("retained_count")
        dropped = decision.get("dropped_count")
        counts_valid = all(
            _is_strict_int(count, minimum=0)
            for count in (raw, retained, dropped)
        )
        _err(
            errors,
            counts_valid
            and retained == retained_items
            and retained + dropped == raw,
            f"adapter MC retention {role} counts are inconsistent",
        )

        threshold_raw = decision.get("threshold")
        rate_raw = decision.get("retention_rate")
        try:
            threshold = float(threshold_raw)
            rate = float(rate_raw)
        except (TypeError, ValueError, OverflowError):
            threshold = math.nan
            rate = math.nan
        numeric_valid = (
            isinstance(threshold_raw, str)
            and isinstance(rate_raw, str)
            and math.isfinite(threshold)
            and math.isfinite(rate)
            and 0.0 <= threshold <= 1.0
            and 0.0 <= rate <= 1.0
            and threshold_raw == repr(threshold)
            and rate_raw == repr(rate)
        )
        expected_rate = (
            retained / raw
            if counts_valid and raw
            else 0.0
        )
        _err(
            errors,
            numeric_valid
            and math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-12),
            f"adapter MC retention {role} rate is inconsistent",
        )
        passed = decision.get("passed")
        overridden = decision.get("overridden")
        effective = decision.get("effective_pass")
        bools_valid = all(
            isinstance(flag, bool)
            for flag in (passed, overridden, effective)
        )
        expected_passed = numeric_valid and rate >= threshold
        _err(
            errors,
            bools_valid
            and passed is expected_passed
            and overridden is (not expected_passed)
            and effective is (passed or overridden)
            and effective is True,
            f"adapter MC retention {role} gate decision is inconsistent",
        )
        _err(
            errors,
            decision.get("override_flag") == "--allow-low-mc-retention",
            f"adapter MC retention {role} override flag mismatch",
        )
        if isinstance(build_metadata, dict):
            try:
                derived = build_retention_metadata(
                    build_metadata,
                    split=split,
                    smoke=False,
                    explicit_threshold=None,
                    override=decision.get("overridden") is True,
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"adapter MC retention {role} cannot be derived: {exc}"
                )
            else:
                derived = dict(derived)
                derived["effective_pass"] = bool(
                    derived["passed"] or derived["overridden"]
                )
                derived["threshold"] = repr(float(derived["threshold"]))
                derived["retention_rate"] = repr(
                    float(derived["retention_rate"])
                )
                _err(
                    errors,
                    decision == derived,
                    f"adapter MC retention {role} decision does not match "
                    "build-metadata bytes",
                )
    return errors


def _producer_hash_errors(
    value: Any,
    *,
    label: str,
    required_keys: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict) or not value:
        return [f"{label} must be a nonempty object"]
    if required_keys is not None and set(value) != required_keys:
        errors.append(
            f"{label} keys do not match the canonical producer set"
        )
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not key
            or key.startswith("/")
            or ".." in Path(key).parts
        ):
            errors.append(f"{label} contains an invalid producer path")
        if not _is_sha256(digest):
            errors.append(f"{label} hash for {key!r} must be 64-hex")
    return errors


def load_json(path: Path) -> Any:
    return loads_no_duplicate_keys(Path(path).read_text(encoding="utf-8"))


def load_jsonl_gz(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = loads_no_duplicate_keys(line)
                if not isinstance(row, dict):
                    raise ValueError("adapter JSONL row must be an object")
                rows.append(row)
    return rows


def load_adapter_rows(bundle_dir: Path) -> list[dict]:
    rows = load_jsonl_gz(bundle_dir / "fit_rows.jsonl.gz")
    rows += load_jsonl_gz(bundle_dir / "eval_rows.jsonl.gz")
    return rows


# --- validate-spec ----------------------------------------------------------------


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

    _err(
        errors,
        spec_identity.get("scientific_profile") == profile_static_identity(),
        "run spec scientific_profile does not match the canonical profile",
    )
    _err(
        errors,
        spec_identity.get("reward_schedules")
        == {
            name: dict(REWARD_SCHEDULE_STRINGS[name])
            for name in REWARD_SCHEDULE_STRINGS
        },
        "run spec reward_schedules mismatch",
    )
    _err(
        errors,
        spec_identity.get("calibration") == CALIBRATION,
        "run spec calibration contract mismatch",
    )
    _err(
        errors,
        spec_identity.get("continuation") == CONTINUATION,
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
                isinstance(ids.get(key), str)
                and re.fullmatch(r"[0-9a-f]{64}", ids[key]) is not None,
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
            _is_sha256(evidence_roots.get("myopic_artifact_sha256")),
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
                    _is_sha256(receipt_id),
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
        bootstrap == expected_bootstrap,
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


# --- validate-adapter -------------------------------------------------------------


def _validate_adapter_impl(bundle_dir: Path) -> CheckResult:
    """Validate an adapter bundle without leaking decoder exceptions.

    Parameters
    ----------
    bundle_dir
        Directory containing the adapter manifest and bound payload files.

    Returns
    -------
    CheckResult
        Structured validation status, errors, and recomputed bundle metadata.
    """
    errors: list[str] = []
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    _err(errors, manifest_path.exists(), "adapter manifest.json missing")
    if not manifest_path.exists():
        return CheckResult(passed=False, errors=errors)
    try:
        manifest = load_json(manifest_path)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        return CheckResult(
            passed=False,
            errors=[f"adapter manifest cannot be decoded: {exc}"],
        )
    if not isinstance(manifest, dict):
        return CheckResult(
            passed=False,
            errors=["adapter manifest must be an object"],
        )
    _err(
        errors,
        set(manifest) == {"id", "identity"},
        "adapter manifest top-level fields do not match the canonical contract",
    )
    ident = manifest.get("identity", {})
    if not isinstance(ident, dict):
        return CheckResult(
            passed=False,
            errors=["adapter manifest identity must be an object"],
        )
    try:
        _err(errors, compute_id(ident) == manifest.get("id"), "adapter manifest id mismatch")
    except (TypeError, ValueError) as exc:
        errors.append(f"adapter identity cannot be canonicalized: {exc}")
    expected_identity_fields = {
        "kind",
        "source_manifest_id",
        "raw_input_bundle_id",
        "model_snapshot_id",
        "scoring_spec",
        "fit_split",
        "eval_split",
        "schema_columns",
        "fit_row_count",
        "eval_row_count",
        "fit_rows_sha256",
        "eval_rows_sha256",
        "calibration_sha256",
        "question_trajectory_binding_id",
        "mc_coverage_evidence",
        "mc_retention_evidence",
        "producer_hashes",
    }
    _err(
        errors,
        set(ident) == expected_identity_fields,
        "adapter identity fields do not match the canonical contract",
    )
    _err(errors, ident.get("kind") == "adapter_bundle", "unexpected adapter manifest kind")
    for identity_field in (
        "source_manifest_id",
        "raw_input_bundle_id",
        "model_snapshot_id",
    ):
        _err(
            errors,
            _is_sha256(ident.get(identity_field)),
            f"adapter {identity_field} must be canonical 64-hex",
        )
    _err(
        errors,
        ident.get("scoring_spec") == ADAPTER_SCORING_SPEC,
        "adapter scoring_spec does not match the canonical scoring contract",
    )
    errors.extend(
        _producer_hash_errors(
            ident.get("producer_hashes"),
            label="adapter producer_hashes",
            required_keys={"adapter_build.py"},
        )
    )

    required_files = (
        "fit_rows.jsonl.gz",
        "eval_rows.jsonl.gz",
        "calibration.json",
        "build_metadata.json",
    )
    for name in required_files:
        p = bundle_dir / name
        _err(errors, p.exists(), f"adapter bundle missing {name}")

    hash_bindings = (
        ("fit_rows.jsonl.gz", "fit_rows_sha256", "adapter fit_rows sha mismatch"),
        ("eval_rows.jsonl.gz", "eval_rows_sha256", "adapter eval_rows sha mismatch"),
        ("calibration.json", "calibration_sha256", "adapter calibration sha mismatch"),
    )
    for filename, identity_key, message in hash_bindings:
        path = bundle_dir / filename
        if path.exists():
            _err(errors, sha256_file(path) == ident.get(identity_key), message)

    fit_rows: list[dict] = []
    eval_rows: list[dict] = []
    try:
        if (bundle_dir / "fit_rows.jsonl.gz").exists():
            fit_rows = load_jsonl_gz(bundle_dir / "fit_rows.jsonl.gz")
    except (
        EOFError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        zlib.error,
    ) as exc:
        errors.append(f"adapter fit_rows cannot be decoded: {exc}")
    try:
        if (bundle_dir / "eval_rows.jsonl.gz").exists():
            eval_rows = load_jsonl_gz(bundle_dir / "eval_rows.jsonl.gz")
    except (
        EOFError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        zlib.error,
    ) as exc:
        errors.append(f"adapter eval_rows cannot be decoded: {exc}")

    from .adapter_build import ADAPTER_SCHEMA_COLUMNS

    schema_columns = ident.get("schema_columns")
    _err(
        errors,
        schema_columns == ADAPTER_SCHEMA_COLUMNS,
        "adapter schema_columns do not match the canonical adapter schema",
    )
    expected_columns = set(ADAPTER_SCHEMA_COLUMNS)

    def _validate_rows(
        rows: list[dict],
        *,
        label: str,
        expected_split: str,
    ) -> set[str]:
        item_ids: set[str] = set()
        row_keys: set[tuple[str, str, int]] = set()
        prefix_pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        canonical_order_keys: list[tuple[str, str, int]] = []
        round_decimals = int(ADAPTER_SCORING_SPEC["round_decimals"])
        rounding_quantum = 10.0 ** -round_decimals
        residual_tolerance = rounding_quantum + 1e-12
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"adapter {label} row {index} is not an object")
                continue
            _err(
                errors,
                set(row) == expected_columns,
                f"adapter {label} row {index} does not match schema_columns",
            )
            _err(
                errors,
                row.get("split") == expected_split,
                f"adapter {label} row {index} split does not match {expected_split!r}",
            )
            row_format = row.get("format")
            _err(
                errors,
                isinstance(row_format, str) and row_format in {"MC", "QA"},
                f"adapter {label} row {index} has invalid format",
            )
            raw_item_id = row.get("item_id")
            item_id = raw_item_id if isinstance(raw_item_id, str) else ""
            _err(
                errors,
                bool(item_id),
                f"adapter {label} row {index} item_id must be a nonempty string",
            )
            raw_prefix_idx = row.get("prefix_idx")
            if not _is_strict_int(raw_prefix_idx, minimum=0):
                errors.append(f"adapter {label} row {index} has invalid prefix_idx")
                continue
            prefix_idx = raw_prefix_idx
            _err(
                errors,
                _is_finite_number(
                    row.get("prefix_fraction"),
                    minimum=0.0,
                    maximum=1.0,
                ),
                f"adapter {label} row {index} has invalid prefix_fraction",
            )
            for digest_field in (
                "prefix_text_sha256",
                "full_question_sha256",
            ):
                _err(
                    errors,
                    _is_sha256(row.get(digest_field)),
                    f"adapter {label} row {index} has invalid {digest_field}",
                )
            prefix_count = row.get("prefix_char_count")
            full_count = row.get("full_question_char_count")
            counts_valid = (
                _is_strict_int(prefix_count, minimum=1)
                and _is_strict_int(full_count, minimum=1)
                and prefix_count <= full_count
            )
            _err(
                errors,
                counts_valid,
                f"adapter {label} row {index} has invalid question lengths",
            )
            if counts_valid and _is_finite_number(row.get("prefix_fraction")):
                expected_fraction = round(prefix_count / full_count, round_decimals)
                _err(
                    errors,
                    math.isclose(
                        float(row["prefix_fraction"]),
                        expected_fraction,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    f"adapter {label} row {index} prefix_fraction does not "
                    "match bound question lengths",
                )
            raw_similarity = row.get("raw_similarity")
            _err(
                errors,
                _is_finite_number(
                    raw_similarity,
                    minimum=-1.0,
                    maximum=1.0,
                ),
                f"adapter {label} row {index} raw_similarity outside cosine range",
            )
            _err(
                errors,
                _is_strict_int(row.get("correct"))
                and row.get("correct") in {0, 1},
                f"adapter {label} row {index} correct must be integer 0 or 1",
            )
            _err(
                errors,
                _is_strict_int(row.get("K"), minimum=2),
                f"adapter {label} row {index} has invalid K",
            )
            for numeric_field in ("p_second_best", "top2_margin"):
                _err(
                    errors,
                    _is_finite_number(row.get(numeric_field)),
                    f"adapter {label} row {index} has invalid {numeric_field}",
                )
            for numeric_field in (
                "prefix_fraction",
                "raw_similarity",
                "p_second_best",
                "top2_margin",
            ):
                _err(
                    errors,
                    _is_quantized_number(
                        row.get(numeric_field),
                        decimal_places=round_decimals,
                    ),
                    f"adapter {label} row {index} {numeric_field} is not "
                    f"quantized to {round_decimals} decimals",
                )
            for text_field in (
                "category",
                "option_set_id",
                "distractor_strategy",
            ):
                _err(
                    errors,
                    isinstance(row.get(text_field), str)
                    and bool(row.get(text_field).strip()),
                    f"adapter {label} row {index} has invalid {text_field}",
                )
            valid_k = _is_strict_int(row.get("K"), minimum=2)
            _err(
                errors,
                bool(item_id)
                and valid_k
                and row.get("option_set_id")
                == f"{item_id}:K{row.get('K')}",
                f"adapter {label} row {index} has invalid option-set identity",
            )

            if row_format == "MC":
                second_best = row.get("p_second_best")
                margin = row.get("top2_margin")
                mc_similarity_valid = (
                    _is_finite_number(
                        raw_similarity,
                        minimum=-1.0,
                        maximum=1.0,
                    )
                    and _is_finite_number(
                        second_best,
                        minimum=-1.0,
                        maximum=1.0,
                    )
                    and _is_finite_number(
                        margin,
                        minimum=0.0,
                        maximum=2.0,
                    )
                )
                if mc_similarity_valid:
                    mc_similarity_valid = (
                        float(second_best) <= float(raw_similarity)
                        and math.isclose(
                            float(margin),
                            float(raw_similarity) - float(second_best),
                            rel_tol=0.0,
                            abs_tol=residual_tolerance,
                        )
                    )
                _err(
                    errors,
                    mc_similarity_valid,
                    f"adapter {label} row {index} MC similarity fields "
                    "violate the top-two cosine contract",
                )
            elif row_format == "QA":
                _err(
                    errors,
                    row.get("correct") == 1
                    and row.get("p_second_best") == 0.0
                    and row.get("top2_margin") == 0.0,
                    f"adapter {label} row {index} QA sentinel fields "
                    "are noncanonical",
                )

            key = (item_id, str(row_format), prefix_idx)
            _err(
                errors,
                key not in row_keys,
                f"adapter {label} contains duplicate row key {key!r}",
            )
            row_keys.add(key)
            if (
                item_id
                and isinstance(row_format, str)
                and row_format in {"MC", "QA"}
            ):
                canonical_order_keys.append((item_id, row_format, prefix_idx))
                prefix_pair = prefix_pairs.setdefault(
                    (item_id, prefix_idx),
                    {},
                )
                prefix_pair.setdefault(row_format, row)
            item_ids.add(item_id)

        if len(canonical_order_keys) == len(rows):
            _err(
                errors,
                canonical_order_keys == sorted(canonical_order_keys),
                f"adapter {label} rows are not in canonical producer order",
            )

        shared_fields = (
            "prefix_fraction",
            "prefix_text_sha256",
            "prefix_char_count",
            "full_question_sha256",
            "full_question_char_count",
            "split",
            "category",
            "K",
            "option_set_id",
            "distractor_strategy",
        )
        for prefix_key, pair in sorted(prefix_pairs.items()):
            _err(
                errors,
                set(pair) == {"MC", "QA"},
                f"adapter {label} prefix {prefix_key!r} lacks paired MC/QA "
                "prefix rows",
            )
            if set(pair) == {"MC", "QA"}:
                _err(
                    errors,
                    all(
                        pair["MC"].get(field) == pair["QA"].get(field)
                        for field in shared_fields
                    ),
                    f"adapter {label} prefix {prefix_key!r} has mismatched "
                    "MC/QA shared fields",
                )

        prefixes_by_item: dict[
            str,
            list[tuple[int, dict[str, dict[str, Any]]]],
        ] = {}
        for (item_id, prefix_idx), pair in prefix_pairs.items():
            prefixes_by_item.setdefault(item_id, []).append((prefix_idx, pair))
        item_metadata_fields = (
            "category",
            "K",
            "option_set_id",
            "distractor_strategy",
        )
        for item_id, item_prefixes in sorted(prefixes_by_item.items()):
            item_prefixes.sort(key=lambda entry: entry[0])
            prefix_indices = [entry[0] for entry in item_prefixes]
            _err(
                errors,
                prefix_indices == list(range(len(prefix_indices))),
                f"adapter {label} item {item_id!r} lacks contiguous prefix indices",
            )

            representative_rows = [
                pair.get("MC") or pair.get("QA")
                for _, pair in item_prefixes
            ]
            fractions = [
                row.get("prefix_fraction")
                for row in representative_rows
                if row is not None
            ]
            if len(fractions) == len(representative_rows) and all(
                _is_finite_number(value, minimum=0.0, maximum=1.0)
                for value in fractions
            ):
                _err(
                    errors,
                    all(
                        float(current) <= float(following)
                        for current, following in zip(
                            fractions,
                            fractions[1:],
                        )
                    ),
                    f"adapter {label} item {item_id!r} has nonmonotonic "
                    "prefix fractions",
                )

            prefix_counts = [
                row.get("prefix_char_count")
                for row in representative_rows
                if row is not None
            ]
            if len(prefix_counts) == len(representative_rows) and all(
                _is_strict_int(value, minimum=1) for value in prefix_counts
            ):
                _err(
                    errors,
                    all(
                        current < following
                        for current, following in zip(
                            prefix_counts,
                            prefix_counts[1:],
                        )
                    ),
                    f"adapter {label} item {item_id!r} prefix lengths do not "
                    "strictly increase",
                )

            terminal_fraction = fractions[-1] if fractions else None
            _err(
                errors,
                _is_finite_number(
                    terminal_fraction,
                    minimum=1.0,
                    maximum=1.0,
                ),
                f"adapter {label} item {item_id!r} terminal "
                "prefix_fraction must be 1.0",
            )
            full_bindings = {
                (
                    row.get("full_question_sha256"),
                    row.get("full_question_char_count"),
                )
                for row in representative_rows
                if row is not None
            }
            _err(
                errors,
                len(full_bindings) == 1,
                f"adapter {label} item {item_id!r} full-question binding "
                "changes across prefixes",
            )
            terminal_row = representative_rows[-1] if representative_rows else None
            _err(
                errors,
                terminal_row is not None
                and terminal_row.get("prefix_text_sha256")
                == terminal_row.get("full_question_sha256")
                and terminal_row.get("prefix_char_count")
                == terminal_row.get("full_question_char_count"),
                f"adapter {label} item {item_id!r} terminal prefix is not "
                "bound to the full question",
            )

            if representative_rows and representative_rows[0] is not None:
                reference = representative_rows[0]
                _err(
                    errors,
                    all(
                        row is not None
                        and all(
                            row.get(field) == reference.get(field)
                            for field in item_metadata_fields
                        )
                        for row in representative_rows
                    ),
                    f"adapter {label} item {item_id!r} metadata changes "
                    "across prefixes",
                )
        return item_ids

    fit_split = ident.get("fit_split")
    eval_split = ident.get("eval_split")
    _err(errors, fit_split == "val", "adapter fit_split must be 'val'")
    _err(errors, eval_split == "test", "adapter eval_split must be 'test'")
    fit_items = _validate_rows(fit_rows, label="fit", expected_split=str(fit_split))
    eval_items = _validate_rows(eval_rows, label="eval", expected_split=str(eval_split))

    for count_key, rows_for_count in (
        ("fit_row_count", fit_rows),
        ("eval_row_count", eval_rows),
    ):
        count = ident.get(count_key)
        _err(
            errors,
            _is_strict_int(count, minimum=0)
            and count == len(rows_for_count),
            f"adapter {count_key} does not match row bytes",
        )
    _err(
        errors,
        not (fit_items & eval_items),
        "adapter fit/eval item IDs overlap",
    )
    try:
        from .adapter_build import question_trajectory_binding_from_rows

        trajectory_id = question_trajectory_binding_from_rows(
            fit_rows + eval_rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"adapter question trajectory cannot be derived: {exc}")
    else:
        _err(
            errors,
            trajectory_id == ident.get("question_trajectory_binding_id"),
            "adapter question trajectory binding does not match row bytes",
        )

    eval_mc_prefixes = {
        (row["item_id"], row["prefix_idx"])
        for row in eval_rows
        if row.get("format") == "MC"
        and isinstance(row.get("item_id"), str)
        and _is_strict_int(row.get("prefix_idx"), minimum=0)
    }
    eval_qa_prefixes = {
        (row["item_id"], row["prefix_idx"])
        for row in eval_rows
        if row.get("format") == "QA"
        and isinstance(row.get("item_id"), str)
        and _is_strict_int(row.get("prefix_idx"), minimum=0)
    }
    expected_coverage = {
        "eval_mc_items": len(
            {item_id for item_id, _ in eval_mc_prefixes}
        ),
        "eval_qa_items": len(
            {item_id for item_id, _ in eval_qa_prefixes}
        ),
        "paired": eval_mc_prefixes == eval_qa_prefixes,
    }
    _err(
        errors,
        ident.get("mc_coverage_evidence") == expected_coverage,
        "adapter MC coverage evidence does not match eval row bytes",
    )
    errors.extend(
        _mc_retention_errors(
            ident.get("mc_retention_evidence"),
            bundle_dir=bundle_dir,
            fit_rows=fit_rows,
            eval_rows=eval_rows,
            fit_items=fit_items,
            eval_items=eval_items,
        )
    )

    calibration: dict[str, Any] | None = None
    cal_path = bundle_dir / "calibration.json"
    if cal_path.exists():
        try:
            loaded = load_json(cal_path)
            if isinstance(loaded, dict):
                calibration = loaded
            else:
                errors.append("adapter calibration.json must contain an object")
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(f"adapter calibration.json cannot be decoded: {exc}")
    if calibration is not None:
        metadata = calibration.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            errors.append("adapter calibration metadata must be an object")
            metadata = {}
        top_fit_split = calibration.get("fit_split")
        metadata_fit_split = (metadata or {}).get("fit_split")
        if (
            top_fit_split is not None
            and metadata_fit_split is not None
            and top_fit_split != metadata_fit_split
        ):
            errors.append("adapter calibration fit_split claims conflict")
        calibration_fit_split = top_fit_split or metadata_fit_split
        _err(
            errors,
            calibration_fit_split == fit_split,
            "adapter calibration fit split does not match adapter fit_split",
        )
        per_bucket = calibration.get("per_bucket")
        _err(
            errors,
            isinstance(per_bucket, dict)
            and set(per_bucket) == {"early", "mid", "late"},
            "adapter calibration per_bucket phases are noncanonical",
        )
        if isinstance(per_bucket, dict):
            for phase in ("early", "mid", "late"):
                block = per_bucket.get(phase)
                errors.extend(platt_phase_errors(block, phase=phase))
        try:
            from .adapter_build import derive_bound_calibration

            expected_calibration = derive_bound_calibration(
                fit_rows=fit_rows,
                eval_rows=eval_rows,
                model_snapshot_id=str(ident.get("model_snapshot_id")),
                fit_rows_sha256=str(ident.get("fit_rows_sha256")),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"adapter calibration cannot be recomputed: {exc}")
        else:
            _err(
                errors,
                calibration == expected_calibration,
                "adapter calibration is not derived from bound fit-row/model bytes",
            )

    return CheckResult(
        passed=not errors,
        errors=errors,
        recomputed={
            "adapter_bundle_id": manifest.get("id"),
            "fit_row_count": len(fit_rows),
            "eval_row_count": len(eval_rows),
            "fit_items": len(fit_items),
            "eval_items": len(eval_items),
        },
    )


def validate_adapter(bundle_dir: Path) -> CheckResult:
    """Validate an untrusted adapter bundle.

    Parameters
    ----------
    bundle_dir
        Directory containing the adapter manifest and bound payload files.

    Returns
    -------
    CheckResult
        Structured validation status; malformed data never escapes this boundary.
    """
    try:
        return _validate_adapter_impl(bundle_dir)
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
        zlib.error,
    ) as exc:
        return CheckResult(
            passed=False,
            errors=[
                "adapter bundle cannot be validated safely: "
                f"{type(exc).__name__}: {exc}"
            ],
        )


# --- validate (run) ---------------------------------------------------------------


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_MAX_DECODED_BYTES = 256 * 1024 * 1024
_PNG_ALLOWED_DEPTHS = {
    0: {1, 2, 4, 8, 16},
    2: {8, 16},
    3: {1, 2, 4, 8},
    4: {8, 16},
    6: {8, 16},
}
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_PNG_ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)


class _PNGError(ValueError):
    """A complete PNG structural or image-stream check failed."""


def _png_pass_geometry(
    width: int,
    height: int,
    interlace: int,
) -> list[tuple[int, int]]:
    if interlace == 0:
        return [(width, height)]
    passes: list[tuple[int, int]] = []
    for x_start, y_start, x_step, y_step in _PNG_ADAM7_PASSES:
        pass_width = (
            0
            if width <= x_start
            else (width - x_start + x_step - 1) // x_step
        )
        pass_height = (
            0
            if height <= y_start
            else (height - y_start + y_step - 1) // y_step
        )
        if pass_width and pass_height:
            passes.append((pass_width, pass_height))
    return passes


def _png_scanline_layout(
    *,
    width: int,
    height: int,
    bit_depth: int,
    color_type: int,
    interlace: int,
) -> tuple[list[tuple[int, int]], int]:
    bits_per_pixel = bit_depth * _PNG_CHANNELS[color_type]
    layout: list[tuple[int, int]] = []
    expected_size = 0
    for pass_width, pass_height in _png_pass_geometry(
        width,
        height,
        interlace,
    ):
        row_bytes = (pass_width * bits_per_pixel + 7) // 8
        layout.append((pass_height, row_bytes))
        expected_size += pass_height * (1 + row_bytes)
    return layout, expected_size


def _validate_png_bytes(data: bytes) -> None:
    """Validate a complete PNG without relying on an optional image library."""
    if data[:8] != _PNG_SIGNATURE:
        raise _PNGError("invalid signature")

    offset = len(_PNG_SIGNATURE)
    chunk_index = 0
    ihdr: tuple[int, int, int, int, int] | None = None
    saw_palette = False
    saw_idat = False
    idat_closed = False
    saw_iend = False
    idat_parts: list[bytes] = []

    while offset < len(data):
        if saw_iend:
            raise _PNGError("trailing bytes after IEND")
        if len(data) - offset < 12:
            raise _PNGError("truncated chunk framing")

        chunk_length = struct.unpack(">I", data[offset:offset + 4])[0]
        if chunk_length > 0x7FFFFFFF:
            raise _PNGError("chunk length exceeds the PNG limit")
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(data):
            raise _PNGError("truncated chunk payload")

        chunk_type = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + chunk_length]
        stored_crc = struct.unpack(">I", data[chunk_end - 4:chunk_end])[0]
        if not all(
            ord("A") <= byte <= ord("Z")
            or ord("a") <= byte <= ord("z")
            for byte in chunk_type
        ):
            raise _PNGError("invalid chunk type")
        if chunk_type[2] & 0x20:
            raise _PNGError("invalid reserved bit in chunk type")
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise _PNGError(f"CRC mismatch in {chunk_type.decode('ascii')}")

        if chunk_index == 0 and chunk_type != b"IHDR":
            raise _PNGError("IHDR is not the first chunk")
        if chunk_type == b"IHDR":
            if chunk_index != 0 or ihdr is not None or chunk_length != 13:
                raise _PNGError("invalid IHDR placement or length")
            (
                width,
                height,
                bit_depth,
                color_type,
                compression,
                filter_method,
                interlace,
            ) = struct.unpack(">IIBBBBB", payload)
            if (
                width == 0
                or height == 0
                or width > 0x7FFFFFFF
                or height > 0x7FFFFFFF
            ):
                raise _PNGError("invalid image dimensions")
            if bit_depth not in _PNG_ALLOWED_DEPTHS.get(color_type, set()):
                raise _PNGError("illegal bit-depth/color-type combination")
            if compression != 0 or filter_method != 0 or interlace not in {0, 1}:
                raise _PNGError("unsupported IHDR method")
            ihdr = (width, height, bit_depth, color_type, interlace)
        elif ihdr is None:
            raise _PNGError("chunk appears before IHDR")
        elif chunk_type == b"PLTE":
            if saw_palette or saw_idat:
                raise _PNGError("invalid PLTE placement")
            color_type = ihdr[3]
            entries = chunk_length // 3
            if (
                color_type in {0, 4}
                or chunk_length == 0
                or chunk_length % 3
                or entries > 256
                or (color_type == 3 and entries > 2 ** ihdr[2])
            ):
                raise _PNGError("invalid PLTE payload")
            saw_palette = True
        elif chunk_type == b"IDAT":
            if idat_closed:
                raise _PNGError("nonconsecutive IDAT chunks")
            if ihdr[3] == 3 and not saw_palette:
                raise _PNGError("indexed PNG is missing PLTE before IDAT")
            saw_idat = True
            idat_parts.append(payload)
        elif chunk_type == b"IEND":
            if chunk_length != 0 or not saw_idat:
                raise _PNGError("invalid IEND or missing IDAT")
            saw_iend = True
            if chunk_end != len(data):
                raise _PNGError("trailing bytes after IEND")
        else:
            if not (chunk_type[0] & 0x20):
                raise _PNGError("unknown critical chunk")
            if saw_idat:
                idat_closed = True

        offset = chunk_end
        chunk_index += 1

    if ihdr is None or not saw_idat or not saw_iend:
        raise _PNGError("missing required PNG chunk")
    if ihdr[3] == 3 and not saw_palette:
        raise _PNGError("indexed PNG is missing PLTE")

    layout, expected_size = _png_scanline_layout(
        width=ihdr[0],
        height=ihdr[1],
        bit_depth=ihdr[2],
        color_type=ihdr[3],
        interlace=ihdr[4],
    )
    if expected_size > _PNG_MAX_DECODED_BYTES:
        raise _PNGError("decoded image exceeds the package limit")

    compressed = b"".join(idat_parts)
    if not compressed:
        raise _PNGError("empty IDAT stream")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_size + 1)
        if decompressor.unconsumed_tail or len(raw) > expected_size:
            raise _PNGError("inflated image exceeds its IHDR dimensions")
        if not decompressor.eof:
            raise _PNGError("truncated zlib image stream")
        if decompressor.unused_data:
            raise _PNGError("trailing data in zlib image stream")
        raw += decompressor.flush()
    except zlib.error as exc:
        raise _PNGError(f"invalid zlib image stream: {exc}") from exc
    if len(raw) != expected_size:
        raise _PNGError("inflated image size does not match IHDR")

    cursor = 0
    for row_count, row_bytes in layout:
        stride = 1 + row_bytes
        for _ in range(row_count):
            if raw[cursor] > 4:
                raise _PNGError("invalid scanline filter")
            cursor += stride
    if cursor != len(raw):
        raise _PNGError("scanline layout does not consume the image stream")


def _check_png(path: Path, errors: list[str]) -> None:
    try:
        _validate_png_bytes(path.read_bytes())
    except (OSError, _PNGError) as exc:
        errors.append(f"invalid PNG {path.name}: {exc}")


def _check_attempts(
    run_root: Path,
    errors: list[str],
    *,
    run_spec_id: str,
    adapter_bundle_id: str,
    bootstrap_plan_id: str,
    aggregate: dict[str, Any],
) -> bool:
    error_count = len(errors)
    path = run_root / "attempts.jsonl"
    try:
        _, attempts = load_attempt_history(path)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        errors.append(f"attempts.jsonl is noncanonical: {exc}")
        return False
    if not attempts:
        errors.append("attempts.jsonl contains no attempts")
        return False
    last_num = 0
    attempt_numbers: list[int] = []
    for index, a in enumerate(attempts):
        if not isinstance(a, dict):
            errors.append("attempt record is not an object")
            continue
        raw_num = a.get("attempt")
        if not _is_strict_int(raw_num, minimum=1):
            errors.append("attempt number must be a positive integer")
            continue
        num = raw_num
        attempt_numbers.append(num)
        mode = a.get("mode")
        cmd = a.get("command", [])
        if not isinstance(mode, str) or mode not in {"fresh", "resume"}:
            errors.append(f"unknown attempt mode {mode!r}")
        if a.get("state") != "started":
            errors.append("attempt record state must be 'started'")
        if (
            not isinstance(cmd, list)
            or not all(isinstance(part, str) for part in cmd)
        ):
            errors.append("attempt command must be a string list")
            cmd = []
        if num != index + 1:
            errors.append("attempt numbers must be consecutive starting at 1")
        elif num <= last_num:
            errors.append("attempt numbers not monotonic")
        last_num = num
        if "--overwrite" in cmd:
            errors.append("--overwrite present in an evidence attempt")
        resume_count = list(cmd).count("--resume")
        if index == 0:
            if num != 1 or mode != "fresh":
                errors.append("first attempt must be attempt 1 in fresh mode")
            if resume_count != 0:
                errors.append("fresh attempt must omit --resume")
        elif mode == "resume":
            if resume_count != 1:
                errors.append("resume attempt must contain exactly one bare --resume")
        elif mode == "fresh":
            errors.append("only the first attempt may use fresh mode")
        _err(
            errors,
            a.get("run_spec_id") == run_spec_id,
            "attempt run_spec_id does not match run spec",
        )
        _err(
            errors,
            a.get("adapter_id") == adapter_bundle_id,
            "attempt adapter_id does not match run spec",
        )
        _err(
            errors,
            a.get("bootstrap_plan_id") == bootstrap_plan_id,
            "attempt bootstrap_plan_id does not match run spec",
        )

    results_dir = run_root / "attempt_results"
    if results_dir.is_symlink() or not results_dir.is_dir():
        errors.append("missing attempt_results directory")
        return False
    result_paths = sorted(results_dir.iterdir())
    result_numbers: set[int] = set()
    results: dict[int, dict[str, Any]] = {}
    for result_path in result_paths:
        try:
            result_number = int(result_path.stem)
        except ValueError:
            errors.append(f"invalid attempt result filename {result_path.name!r}")
            continue
        if (
            result_number < 1
            or result_path.name != f"{result_number}.json"
            or result_number in result_numbers
            or result_path.is_symlink()
            or not result_path.is_file()
        ):
            errors.append(f"invalid attempt result path {result_path.name!r}")
            continue
        try:
            result = load_json(result_path)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(
                f"attempt result {result_path.name!r} cannot be decoded: {exc}"
            )
            continue
        if not isinstance(result, dict):
            errors.append(f"attempt result {result_path.name!r} is not an object")
            continue
        result_numbers.add(result_number)
        results[result_number] = result
        _err(
            errors,
            result.get("attempt") == result_number,
            f"attempt result {result_number} number mismatch",
        )
        _err(
            errors,
            result.get("run_spec_id") == run_spec_id,
            f"attempt result {result_number} run_spec_id mismatch",
        )
        state = result.get("state")
        state_fields = {
            "completed": {"completed", "failed"},
            "failed": {"error_type", "error_message"},
            "interrupted": {"reason"},
        }
        valid_state = isinstance(state, str) and state in state_fields
        _err(
            errors,
            valid_state,
            f"attempt result {result_number} has invalid state",
        )
        if valid_state:
            _err(
                errors,
                set(result)
                == {"attempt", "state", "run_spec_id"} | state_fields[state],
                f"attempt result {result_number} fields do not match its state",
            )
        if state == "completed":
            _err(
                errors,
                _is_strict_int(result.get("completed"), minimum=0)
                and _is_strict_int(result.get("failed"), minimum=0),
                f"attempt result {result_number} has invalid counts",
            )
        elif state == "failed":
            _err(
                errors,
                isinstance(result.get("error_type"), str)
                and bool(result.get("error_type"))
                and isinstance(result.get("error_message"), str),
                f"attempt result {result_number} has invalid failure evidence",
            )
        elif state == "interrupted":
            _err(
                errors,
                result.get("reason") == _INTERRUPTED_REASON,
                f"attempt result {result_number} has invalid interruption evidence",
            )
    _err(
        errors,
        result_numbers == set(attempt_numbers),
        "attempt results do not match attempt history",
    )
    if attempt_numbers and attempt_numbers[-1] in results:
        final_result = results[attempt_numbers[-1]]
        _err(
            errors,
            final_result.get("state") == "completed",
            "latest attempt did not complete",
        )
        _err(
            errors,
            final_result.get("completed") == aggregate.get("completed")
            and final_result.get("failed") == aggregate.get("failed"),
            "latest attempt counts do not match aggregate",
        )
    return len(errors) == error_count


def _check_reports(
    run_root: Path,
    aggregate: dict[str, Any],
    resource_summary: dict[str, Any],
    errors: list[str],
) -> None:
    """Bind every displayed package byte to the validated scientific inputs."""
    from . import writers

    try:
        expected: dict[str, bytes] = {
            "reports/report.md": writers.render_markdown(
                aggregate,
                resource_summary=resource_summary,
            ).encode("utf-8"),
            "reports/report.tex": writers.render_latex(aggregate).encode(
                "utf-8"
            ),
        }
        with tempfile.TemporaryDirectory(prefix="stopdff_v5_check_figures_") as td:
            figure_root = Path(td)
            figure_paths = writers.write_figures(
                figure_root,
                aggregate,
                profile_variant=aggregate.get("profile_variant"),
            )
            for relative in figure_paths:
                expected[relative] = (figure_root / relative).read_bytes()
    except (
        AttributeError,
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            "canonical reports/figures cannot be regenerated: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    for directory_name in ("reports", "figures"):
        directory = run_root / directory_name
        expected_names = {
            Path(relative).name
            for relative in expected
            if Path(relative).parts[0] == directory_name
        }
        if directory.is_symlink() or not directory.is_dir():
            errors.append(f"missing or noncanonical {directory_name} directory")
            continue
        actual_names: set[str] = set()
        try:
            entries = list(directory.iterdir())
        except OSError as exc:
            errors.append(f"{directory_name} directory cannot be read: {exc}")
            continue
        for path in entries:
            if path.is_symlink() or not path.is_file():
                errors.append(
                    f"unexpected non-file package evidence: "
                    f"{directory_name}/{path.name}"
                )
                continue
            actual_names.add(path.name)
        for name in sorted(expected_names - actual_names):
            errors.append(f"missing {directory_name}/{name}")
        for name in sorted(actual_names - expected_names):
            errors.append(f"unexpected {directory_name}/{name}")

    for relative, expected_bytes in sorted(expected.items()):
        path = run_root / relative
        if path.is_symlink() or not path.is_file():
            continue
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            errors.append(f"{relative} cannot be inspected: {exc}")
            continue
        if actual_size != len(expected_bytes):
            errors.append(f"{relative} does not match canonical rendered content")
            continue
        try:
            actual_bytes = path.read_bytes()
        except OSError as exc:
            errors.append(f"{relative} cannot be read: {exc}")
            continue
        if actual_bytes != expected_bytes:
            errors.append(f"{relative} does not match canonical rendered content")
        if path.suffix == ".png":
            _check_png(path, errors)


def _resolve_run_binding(
    *,
    run_spec_manifest: dict[str, Any],
    adapter_bundle: Path,
    bootstrap_plan_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve canonical run inputs from self-valid manifests and local bytes."""
    errors: list[str] = []
    adapter_bundle = Path(adapter_bundle)
    errors.extend(_run_spec_manifest_envelope_errors(run_spec_manifest))

    adapter_result = validate_adapter(adapter_bundle)
    errors.extend(adapter_result.errors)
    adapter_manifest_path = adapter_bundle / "manifest.json"
    adapter_manifest: dict[str, Any] = {}
    if adapter_manifest_path.exists():
        try:
            loaded = load_json(adapter_manifest_path)
            if isinstance(loaded, dict):
                adapter_manifest = loaded
            else:
                errors.append("adapter manifest must contain an object")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"adapter manifest cannot be decoded: {exc}")

    spec_identity = run_spec_manifest.get("identity", {})
    spec_id = run_spec_manifest.get("id")
    if not isinstance(spec_identity, dict):
        errors.append("run spec identity must be an object")
        spec_identity = {}
    errors.extend(
        _run_spec_errors(
            spec_identity,
            spec_id,
            require_final_profile=False,
        )
    )
    try:
        _err(
            errors,
            compute_id(spec_identity) == spec_id,
            "run_spec id mismatch",
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"run_spec identity cannot be canonicalized: {exc}")

    adapter_identity = adapter_manifest.get("identity", {})
    adapter_id = adapter_manifest.get("id")
    if not isinstance(adapter_identity, dict):
        errors.append("adapter identity must be an object")
        adapter_identity = {}

    spec_ids = spec_identity.get("identity", {})
    if not isinstance(spec_ids, dict):
        errors.append("run spec identity graph must be an object")
        spec_ids = {}
    _err(
        errors,
        spec_ids.get("adapter_bundle_id") == adapter_id,
        "run spec adapter_bundle_id does not match supplied adapter_bundle_id",
    )
    for field in (
        "source_manifest_id",
        "raw_input_bundle_id",
        "model_snapshot_id",
    ):
        _err(
            errors,
            spec_ids.get(field) == adapter_identity.get(field),
            f"run spec {field} does not match adapter manifest",
        )

    plan_identity = bootstrap_plan_manifest.get("identity", {})
    plan_id = bootstrap_plan_manifest.get("id")
    if not isinstance(plan_identity, dict):
        errors.append("bootstrap plan identity must be an object")
        plan_identity = {}
    try:
        _err(
            errors,
            compute_id(plan_identity) == plan_id,
            "bootstrap_plan id mismatch",
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"bootstrap plan identity cannot be canonicalized: {exc}")
    _err(
        errors,
        spec_ids.get("bootstrap_plan_id") == plan_id,
        "run spec bootstrap_plan_id does not match supplied bootstrap plan",
    )
    expected_plan_keys = {
        "evaluation_item_id_list_sha256",
        "bit_generator",
        "seed",
        "replicate_count",
        "resample_index_sha256",
        "numpy_version_contract",
        "resample_dtype",
    }
    _err(
        errors,
        set(plan_identity) == expected_plan_keys,
        "bootstrap plan identity fields are noncanonical",
    )
    _err(
        errors,
        plan_identity.get("bit_generator") == "PCG64",
        "bootstrap bit_generator must be PCG64",
    )
    _err(
        errors,
        plan_identity.get("resample_dtype") == "int64",
        "bootstrap resample_dtype must be int64",
    )

    item_ids = bootstrap_plan_manifest.get("item_ids", [])
    if (
        not isinstance(item_ids, list)
        or not all(isinstance(item_id, str) and item_id for item_id in item_ids)
    ):
        errors.append("bootstrap plan item_ids must be nonempty strings")
        item_ids = []
    _err(
        errors,
        item_ids == sorted(set(item_ids)),
        "bootstrap plan item_ids must be sorted and unique",
    )

    seed_value = plan_identity.get("seed")
    replicate_value = plan_identity.get("replicate_count")
    variant = spec_identity.get("profile_variant")
    valid_variant = isinstance(variant, str) and variant in {"smoke", "final"}
    expected_replicates = (
        FINAL_REPLICATES
        if variant == "final"
        else SMOKE_REPLICATES if variant == "smoke" else None
    )
    if (
        not _is_strict_int(seed_value, minimum=0)
        or not _is_strict_int(replicate_value, minimum=1)
        or replicate_value not in {SMOKE_REPLICATES, FINAL_REPLICATES}
    ):
        seed = -1
        replicates = -1
        errors.append("bootstrap plan seed/replicate_count is invalid")
    else:
        seed = seed_value
        replicates = replicate_value
    if valid_variant and replicates != expected_replicates:
        errors.append(
            f"{variant} profile must use {expected_replicates} replicates "
            f"(got {replicates})"
        )
        replicates = -1
    _err(errors, seed == 1, f"bootstrap seed must be 1 (got {seed})")

    rows: list[dict] = []
    try:
        if (
            (adapter_bundle / "fit_rows.jsonl.gz").is_file()
            and (adapter_bundle / "eval_rows.jsonl.gz").is_file()
        ):
            rows = load_adapter_rows(adapter_bundle)
    except (
        EOFError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        zlib.error,
    ) as exc:
        errors.append(f"adapter rows cannot be decoded: {exc}")
    eval_mc_items = {
        str(row.get("item_id"))
        for row in rows
        if row.get("split") == "test" and row.get("format") == "MC"
    }
    eval_qa_items = {
        str(row.get("item_id"))
        for row in rows
        if row.get("split") == "test" and row.get("format") == "QA"
    }
    expected_item_ids = sorted(eval_mc_items & eval_qa_items)
    _err(
        errors,
        item_ids == expected_item_ids,
        "bootstrap item_ids do not match paired adapter eval items",
    )

    # Never allocate a manifest-selected resample matrix until its item universe
    # is proven to be the exact, bounded universe derived from the adapter rows.
    plan = None
    if (
        item_ids
        and item_ids == expected_item_ids
        and seed >= 0
        and replicates > 0
    ):
        try:
            plan = build_bootstrap_plan(item_ids, replicates=replicates, seed=seed)
            _err(
                errors,
                plan.resample_index_sha256
                == plan_identity.get("resample_index_sha256"),
                "bootstrap resample-index hash mismatch",
            )
            _err(
                errors,
                plan.item_id_list_sha256
                == plan_identity.get("evaluation_item_id_list_sha256"),
                "bootstrap item-id-list hash mismatch",
            )
            _err(
                errors,
                plan.numpy_version
                == plan_identity.get("numpy_version_contract"),
                "bootstrap NumPy version contract mismatch",
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"bootstrap plan cannot be rebuilt: {exc}")

    _err(
        errors,
        valid_variant,
        f"unsupported run spec profile_variant {variant!r}",
    )
    spec_bootstrap = spec_identity.get("bootstrap", {})
    spec_seed = (
        spec_bootstrap.get("seed", -1)
        if isinstance(spec_bootstrap, dict)
        else -1
    )
    spec_replicates = (
        spec_bootstrap.get("replicate_count", -1)
        if isinstance(spec_bootstrap, dict)
        else -1
    )
    if not _is_strict_int(spec_seed) or not _is_strict_int(spec_replicates):
        spec_seed = -1
        spec_replicates = -1
    _err(
        errors,
        spec_seed == seed,
        "run spec bootstrap seed does not match bootstrap plan",
    )
    _err(
        errors,
        spec_replicates == replicates,
        "run spec bootstrap replicate_count does not match bootstrap plan",
    )

    fvi_selected = spec_identity.get("fvi_selected", {})
    if not isinstance(fvi_selected, dict):
        errors.append("run spec fvi_selected must be an object")
        fvi_selected = {}
    raw_tolerance = fvi_selected.get("tolerance")
    raw_max_iterations = fvi_selected.get("max_iterations")
    if (
        raw_tolerance not in FVI_TOLERANCES
        or raw_max_iterations not in FVI_MAX_ITERATIONS
    ):
        tolerance = ""
        max_iterations = -1
        errors.append("run spec FVI settings are invalid")
    else:
        tolerance = str(raw_tolerance)
        max_iterations = int(raw_max_iterations)

    gate = spec_identity.get("gate", {})
    if not isinstance(gate, dict):
        errors.append("run spec gate must be an object")
        gate = {}
    gate_overrides: dict[str, bool] = {}
    for key in (
        "allow_low_mc_retention",
        "allow_incomplete_mc_coverage",
    ):
        value = gate.get(key)
        if not isinstance(value, bool):
            errors.append(f"run spec gate {key} must be boolean")
        gate_overrides[key] = bool(value)

    retention = adapter_identity.get("mc_retention_evidence", {})
    retention_splits = (
        retention.get("splits", {})
        if isinstance(retention, dict)
        else {}
    )
    adapter_retention_overridden = (
        isinstance(retention_splits, dict)
        and any(
            isinstance(decision, dict)
            and decision.get("overridden") is True
            for decision in retention_splits.values()
        )
    )
    _err(
        errors,
        not adapter_retention_overridden
        or gate_overrides.get("allow_low_mc_retention") is True,
        "adapter low-retention override is not enabled by the run gate",
    )

    calibration = None
    calibration_path = adapter_bundle / "calibration.json"
    if calibration_path.exists():
        try:
            calibration = load_json(calibration_path)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"calibration.json cannot be decoded: {exc}")

    try:
        required_consecutive = int(
            fvi_selected.get("required_consecutive_converged_iterations", -1)
        )
    except (TypeError, ValueError):
        required_consecutive = -1

    return (
        {
            "run_spec_identity": spec_identity,
            "run_spec_id": spec_id,
            "spec_ids": spec_ids,
            "evidence_roots": spec_identity.get("evidence_roots", {}),
            "adapter_identity": adapter_identity,
            "adapter_bundle_id": adapter_id,
            "fit_rows_sha256": adapter_identity.get("fit_rows_sha256"),
            "eval_rows_sha256": adapter_identity.get("eval_rows_sha256"),
            "bootstrap_plan_identity": plan_identity,
            "bootstrap_plan_id": plan_id,
            "bootstrap_plan": plan,
            "rows": rows,
            "calibration": calibration,
            "variant": variant,
            "replicates": replicates,
            "fvi_tolerance": tolerance,
            "fvi_max_iterations": max_iterations,
            "fvi_settings": {
                "damping": str(fvi_selected.get("damping")),
                "tolerance": tolerance,
                "max_iterations": max_iterations,
                "required_consecutive_converged_iterations": required_consecutive,
                "numeric_dtype": str(fvi_selected.get("numeric_dtype")),
            },
            "gate_overrides": gate_overrides,
        },
        errors,
    )


def resolve_run_binding(
    *,
    run_spec_manifest: dict[str, Any],
    adapter_bundle: Path,
    bootstrap_plan_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return verified canonical run inputs or fail before computation.

    Parameters
    ----------
    run_spec_manifest
        Parsed run-spec manifest.
    adapter_bundle
        Directory containing the adapter bundle.
    bootstrap_plan_manifest
        Parsed bootstrap-plan manifest.

    Returns
    -------
    dict[str, Any]
        Canonical identities, inputs, rows, and gate settings for computation.

    Raises
    ------
    ValueError
        If any manifest, identity binding, or local adapter byte is invalid.
    """
    binding, errors = _resolve_run_binding(
        run_spec_manifest=run_spec_manifest,
        adapter_bundle=adapter_bundle,
        bootstrap_plan_manifest=bootstrap_plan_manifest,
    )
    if errors:
        raise ValueError("run binding validation failed: " + "; ".join(errors))
    return binding


def _validate_run_impl(
    run_root: Path,
    *,
    backend: str,
    adapter_bundle: Path,
    require_final_profile: bool = False,
    require_package: bool = False,
) -> CheckResult:
    """Validate and independently recompute one StopDFF run.

    Parameters
    ----------
    run_root
        Directory containing the run artifacts.
    backend
        Expected execution backend, ``"local"`` or ``"modal"``.
    adapter_bundle
        Directory containing the identity-bound adapter bundle.
    require_final_profile
        Whether to require the final 96-cell profile.
    require_package
        Whether to enforce complete packaged-evidence and report checks.

    Returns
    -------
    CheckResult
        Structured validation status, errors, and recomputed release metadata.
    """
    run_root = Path(run_root)
    adapter_bundle = Path(adapter_bundle)
    errors: list[str] = []

    def _required_json(filename: str) -> dict[str, Any]:
        path = run_root / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            return {}
        try:
            loaded = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"{filename} cannot be decoded: {exc}")
            return {}
        if not isinstance(loaded, dict):
            errors.append(f"{filename} must contain an object")
            return {}
        return loaded

    aggregate = _required_json("aggregate.json")
    spec_manifest = _required_json("run_spec.json")
    plan_manifest = _required_json("bootstrap_plan.json")
    expected_aggregate_fields = {
        "profile_name",
        "profile_variant",
        "backend",
        "run_spec_id",
        "adapter_bundle_id",
        "bootstrap_plan_id",
        "fvi_study_id",
        "adapter_fit_rows_sha256",
        "adapter_eval_rows_sha256",
        "myopic_artifact_sha256",
        "requested",
        "completed",
        "skipped",
        "failed",
        "expected_cell_keys",
        "fvi_selected",
        "cells",
        "family",
        "gate_overrides",
        "release_status",
        "release_reasons",
    }
    _err(
        errors,
        set(aggregate) == expected_aggregate_fields,
        "aggregate fields do not match the canonical contract",
    )
    binding, binding_errors = _resolve_run_binding(
        run_spec_manifest=spec_manifest,
        adapter_bundle=adapter_bundle,
        bootstrap_plan_manifest=plan_manifest,
    )
    errors.extend(binding_errors)
    manifest_graph_valid = not binding_errors
    if binding_errors:
        return CheckResult(passed=False, errors=errors)

    run_spec_id = binding.get("run_spec_id")
    adapter_bundle_id = binding.get("adapter_bundle_id")
    bootstrap_plan_id = binding.get("bootstrap_plan_id")
    adapter_identity = binding.get("adapter_identity", {})
    spec_ids = binding.get("spec_ids", {})
    evidence_roots = binding.get("evidence_roots", {})
    variant = binding.get("variant")
    plan = binding.get("bootstrap_plan")
    rows = binding.get("rows", [])
    calibration = binding.get("calibration")
    tol_label = binding.get("fvi_tolerance", "")
    max_iter = int(binding.get("fvi_max_iterations", -1))
    expected_fvi_settings = binding.get("fvi_settings", {})
    gate_overrides = binding.get("gate_overrides", {})
    mc_overridden = bool(
        gate_overrides.get("allow_low_mc_retention")
        or gate_overrides.get("allow_incomplete_mc_coverage")
    )

    _err(
        errors,
        aggregate.get("backend") == backend,
        f"aggregate backend {aggregate.get('backend')!r} != requested {backend!r}",
    )
    _err(
        errors,
        aggregate.get("profile_variant") == variant,
        "aggregate profile_variant does not match run spec",
    )
    aggregate_bindings = {
        "run_spec_id": run_spec_id,
        "adapter_bundle_id": adapter_bundle_id,
        "bootstrap_plan_id": bootstrap_plan_id,
        "fvi_study_id": spec_ids.get("fvi_study_id"),
        "adapter_fit_rows_sha256": adapter_identity.get("fit_rows_sha256"),
        "adapter_eval_rows_sha256": adapter_identity.get("eval_rows_sha256"),
        "myopic_artifact_sha256": (
            evidence_roots.get("myopic_artifact_sha256")
            if isinstance(evidence_roots, dict)
            else None
        ),
    }
    for field, expected in aggregate_bindings.items():
        _err(
            errors,
            aggregate.get(field) == expected,
            f"aggregate {field} does not match bound run inputs",
        )
    aggregate_fvi = aggregate.get("fvi_selected", {})
    _err(
        errors,
        isinstance(aggregate_fvi, dict)
        and set(aggregate_fvi) == {"tolerance", "max_iterations"},
        "aggregate FVI fields do not match the canonical contract",
    )
    _err(
        errors,
        isinstance(aggregate_fvi, dict)
        and _scientific_equal(
            aggregate_fvi.get("tolerance"),
            tol_label,
        ),
        "aggregate FVI tolerance does not match run spec",
    )
    aggregate_max_iterations = (
        aggregate_fvi.get("max_iterations")
        if isinstance(aggregate_fvi, dict)
        else None
    )
    _err(
        errors,
        _is_strict_int(aggregate_max_iterations)
        and aggregate_max_iterations == max_iter,
        "aggregate FVI max_iterations does not match run spec",
    )
    _err(
        errors,
        _scientific_equal(aggregate.get("gate_overrides"), gate_overrides),
        "aggregate gate_overrides do not match run spec",
    )

    resource_summary_claims = _required_json("resource_summary.json")
    try:
        _err(
            errors,
            compute_id(resource_summary_claims)
            == spec_ids.get("resource_summary_id"),
            "resource_summary.json does not match resource_summary_id",
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"resource summary cannot be canonicalized: {exc}")

    # Backend manifest exclusivity and identity binding.
    run_manifest_path = run_root / "run_manifest.json"
    command_manifest_path = run_root / "command_manifest.json"
    backend_error_count = len(errors)
    expected_manifest_path = (
        run_manifest_path if backend == "modal" else command_manifest_path
    )
    forbidden_manifest_path = (
        command_manifest_path if backend == "modal" else run_manifest_path
    )
    _err(
        errors,
        expected_manifest_path.exists(),
        f"{backend} backend requires {expected_manifest_path.name}",
    )
    _err(
        errors,
        not forbidden_manifest_path.exists(),
        f"{backend} backend forbids {forbidden_manifest_path.name}",
    )
    backend_manifest: dict[str, Any] = {}
    if expected_manifest_path.exists():
        try:
            loaded_backend_manifest = load_json(expected_manifest_path)
            if not isinstance(loaded_backend_manifest, dict):
                errors.append("backend manifest must contain an object")
                loaded_backend_manifest = {}
            backend_manifest = loaded_backend_manifest
            backend_identity = backend_manifest.get("identity", {})
            if not isinstance(backend_identity, dict):
                errors.append("backend manifest identity must contain an object")
                backend_identity = {}
            _err(
                errors,
                compute_id(backend_identity) == backend_manifest.get("id"),
                "backend manifest id mismatch",
            )
            expected_kind = (
                "run_manifest" if backend == "modal" else "command_manifest"
            )
            _err(
                errors,
                backend_identity.get("kind") == expected_kind,
                "backend manifest kind mismatch",
            )
            backend_bindings = {
                "backend": backend,
                "run_spec_id": run_spec_id,
                "adapter_bundle_id": adapter_bundle_id,
                "bootstrap_plan_id": bootstrap_plan_id,
                "adapter_fit_rows_sha256": adapter_identity.get(
                    "fit_rows_sha256"
                ),
                "adapter_eval_rows_sha256": adapter_identity.get(
                    "eval_rows_sha256"
                ),
            }
            for field, expected in backend_bindings.items():
                _err(
                    errors,
                    backend_identity.get(field) == expected,
                    f"backend manifest {field} does not match bound run inputs",
                )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(f"backend manifest cannot be validated: {exc}")

    environment_path = run_root / "environment.json"
    environment_claims: dict[str, Any] = {}
    if not environment_path.is_file():
        errors.append("missing environment.json")
    else:
        try:
            loaded_environment = load_json(environment_path)
            if isinstance(loaded_environment, dict):
                environment_claims = loaded_environment
            else:
                errors.append("environment.json must contain an object")
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(f"environment.json cannot be decoded: {exc}")
    expected_environment_keys = {"python_version", "package_versions"}
    _err(
        errors,
        set(environment_claims) == expected_environment_keys,
        "environment.json fields are noncanonical",
    )
    package_versions = environment_claims.get("package_versions")
    _err(
        errors,
        isinstance(environment_claims.get("python_version"), str)
        and bool(environment_claims.get("python_version")),
        "environment python_version must be a nonempty string",
    )
    package_versions_valid = (
        isinstance(package_versions, dict)
        and set(package_versions) == set(ENVIRONMENT_PACKAGES)
        and all(
            isinstance(version, str) and bool(version)
            for version in package_versions.values()
        )
    )
    _err(
        errors,
        package_versions_valid,
        "environment package_versions must contain exactly the declared "
        "evidence-affecting packages with nonempty string versions",
    )
    if (
        isinstance(environment_claims.get("python_version"), str)
        and package_versions_valid
    ):
        try:
            environment_identity = environment_contract_identity(
                python_version=environment_claims["python_version"],
                package_versions=package_versions,
            )
            _err(
                errors,
                compute_id(environment_identity)
                == spec_ids.get("environment_contract_id"),
                "environment.json does not match environment_contract_id",
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"environment contract cannot be canonicalized: {exc}")
    _err(
        errors,
        backend_manifest.get("environment") == environment_claims,
        "backend manifest environment does not match environment.json",
    )
    _err(
        errors,
        backend_manifest.get("resource_summary") == resource_summary_claims,
        "backend manifest resource summary does not match resource_summary.json",
    )
    backend_manifest_valid = len(errors) == backend_error_count

    attempt_history_valid = _check_attempts(
        run_root,
        errors,
        run_spec_id=str(run_spec_id),
        adapter_bundle_id=str(adapter_bundle_id),
        bootstrap_plan_id=str(bootstrap_plan_id),
        aggregate=aggregate,
    )

    if require_final_profile:
        _err(errors, variant == "final", "final validation requires final profile")
    cells = smoke_cells() if variant == "smoke" else full_grid()
    if require_final_profile:
        _err(errors, len(cells) == EXPECTED_CELLS, "final profile must have 96 cells")
    expected_keys = {cell_key_str(cell) for cell in cells}
    _err(
        errors,
        aggregate.get("expected_cell_keys") == sorted(expected_keys),
        "aggregate expected_cell_keys mismatch",
    )
    aggregate_cells = aggregate.get("cells")
    if not isinstance(aggregate_cells, dict):
        errors.append("aggregate cells must contain an object")
        aggregate_cells = {}
    _err(
        errors,
        set(aggregate_cells) == expected_keys,
        "aggregate cell keys do not match the profile",
    )

    cells_dir = run_root / "cells"
    if cells_dir.is_symlink():
        errors.append("run cells directory must not be a symlink")
        actual_cell_keys: set[str] = set()
    elif not cells_dir.is_dir():
        errors.append("run cells path must be a directory")
        actual_cell_keys = set()
    else:
        actual_cell_keys = {
            path.stem
            for path in cells_dir.glob("*.json")
        }
    _err(
        errors,
        actual_cell_keys == expected_keys,
        "run cell file set does not match the profile",
    )

    abs_median_reps: dict[str, Any] = {}
    abs_median_point: dict[str, float] = {}
    recomputed_verdicts: dict[str, str] = {}
    completed: set[str] = set()
    failed: set[str] = set()
    all_calibrators_fitted = True
    all_fvi_converged = True
    try:
        prepared_cell_inputs = prepare_cell_inputs(rows, calibration)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"cell inputs cannot be prepared: {exc}")
        prepared_cell_inputs = None

    common_cell_fields = {
        "cell",
        "cell_key",
        "fingerprint_id",
        "fingerprint_identity",
        "status",
        "run_spec_id",
        "adapter_bundle_id",
        "bootstrap_plan_id",
        "calibrator_parameters",
    }
    completed_cell_fields = common_cell_fields | {
        "fvi",
        "coverage",
        "ceiling_flags",
        "index_shift_by_item",
        "bootstrap",
        "descriptive",
        "mc_gate_overridden",
        "verdict",
    }

    for cell in cells:
        key = cell_key_str(cell)
        cell_path = cells_dir / f"{key}.json"
        if cell_path.is_symlink() or not cell_path.is_file():
            errors.append(f"{key}: cell evidence must be a regular non-symlink file")
            failed.add(key)
            continue
        try:
            stored = load_json(cell_path)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(f"{key}: cell evidence cannot be decoded: {exc}")
            failed.add(key)
            continue
        if not isinstance(stored, dict):
            errors.append(f"{key}: cell evidence must contain an object")
            failed.add(key)
            continue
        _err(errors, stored.get("cell_key") == key, f"{key}: cell_key mismatch")
        _err(errors, stored.get("cell") == cell, f"{key}: cell payload mismatch")
        _err(
            errors,
            stored.get("run_spec_id") == run_spec_id,
            f"{key}: run_spec_id does not match run spec",
        )
        _err(
            errors,
            stored.get("adapter_bundle_id") == adapter_bundle_id,
            f"{key}: adapter_bundle_id does not match run spec",
        )
        _err(
            errors,
            stored.get("bootstrap_plan_id") == bootstrap_plan_id,
            f"{key}: bootstrap_plan_id does not match run spec",
        )

        fingerprint = stored.get("fingerprint_identity", {})
        fingerprint_id_matches = False
        if isinstance(fingerprint, dict):
            try:
                fingerprint_id_matches = (
                    compute_id(fingerprint) == stored.get("fingerprint_id")
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"{key}: fingerprint cannot be canonicalized: {exc}")
        _err(
            errors,
            fingerprint_id_matches,
            f"{key}: fingerprint id mismatch",
        )
        _err(
            errors,
            isinstance(fingerprint, dict)
            and fingerprint.get("kind") == "cell_fingerprint",
            f"{key}: fingerprint kind mismatch",
        )
        _err(
            errors,
            isinstance(fingerprint, dict)
            and fingerprint.get("producer_hashes")
            == (
                evidence_roots.get("producer_hashes")
                if isinstance(evidence_roots, dict)
                else None
            ),
            f"{key}: fingerprint producer_hashes do not match run spec",
        )
        fingerprint_bindings = {
            "run_spec_id": run_spec_id,
            "adapter_bundle_id": adapter_bundle_id,
            "bootstrap_plan_id": bootstrap_plan_id,
            "cell": cell,
            "adapter_fit_rows_sha256": adapter_identity.get(
                "fit_rows_sha256"
            ),
            "adapter_eval_rows_sha256": adapter_identity.get(
                "eval_rows_sha256"
            ),
            "fvi_settings": expected_fvi_settings,
            "myopic_artifact_sha256": (
                evidence_roots.get("myopic_artifact_sha256")
                if isinstance(evidence_roots, dict)
                else None
            ),
        }
        for field, expected in fingerprint_bindings.items():
            _err(
                errors,
                isinstance(fingerprint, dict)
                and fingerprint.get(field) == expected,
                f"{key}: fingerprint {field} does not match bound run inputs",
            )

        try:
            res = compute_cell(
                rows=rows,
                cell=cell,
                calibration_json=calibration,
                tolerance=float(tol_label),
                max_iterations=max_iter,
                tolerance_label=tol_label,
                metric_split="test",
                prepared=prepared_cell_inputs,
            )
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"{key}: independent cell recomputation failed: {exc}")
            failed.add(key)
            continue
        _err(
            errors,
            isinstance(fingerprint, dict)
            and _scientific_equal(
                fingerprint.get("calibrator_parameters"),
                res.calibrator_parameters,
            ),
            f"{key}: fingerprint calibrator_parameters mismatch",
        )
        _err(
            errors,
            _scientific_equal(
                stored.get("calibrator_parameters"),
                res.calibrator_parameters,
            ),
            f"{key}: stored calibrator_parameters mismatch",
        )
        if res.status != "completed":
            failed.add(key)
            if res.status == "calibrator_failed":
                all_calibrators_fitted = False
            if res.status == "fvi_failed":
                all_fvi_converged = False
            expected_failed_fields = common_cell_fields | {"reason"}
            expected_failed_claims: dict[str, Any] = {
                "status": res.status,
                "reason": res.reason,
            }
            if res.fvi is not None:
                expected_failed_fields.add("fvi")
                expected_failed_claims["fvi"] = {
                    "status": res.fvi.status,
                    "converged": res.fvi.converged,
                    "iterations": res.fvi.iterations,
                    "final_delta": res.fvi.final_delta,
                }
            _err(
                errors,
                set(stored) == expected_failed_fields,
                f"{key}: failed cell fields do not match the canonical contract",
            )
            for field, expected in expected_failed_claims.items():
                _err(
                    errors,
                    _scientific_equal(stored.get(field), expected),
                    f"{key}: stored {field} does not match independent recomputation",
                )
            expected_summary = {"status": res.status, "verdict": "INVALID"}
            _err(
                errors,
                _scientific_equal(aggregate_cells.get(key), expected_summary),
                f"{key}: aggregate cell summary mismatch",
            )
            continue

        completed.add(key)
        _err(
            errors,
            set(stored) == completed_cell_fields,
            f"{key}: completed cell fields do not match the canonical contract",
        )
        if plan is None:
            errors.append(f"{key}: bootstrap plan unavailable for recomputation")
            continue
        try:
            stats = cell_bootstrap_stats(res.index_shift_by_item, plan)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{key}: bootstrap cannot be recomputed: {exc}")
            continue
        flags = ceiling_flags(res.mc_stops, res.qa_stops, list(res.index_shift_by_item.values()))
        ceiling_any = any(flags.values())
        cov_clean = coverage_clean(res.coverage["fallback_fraction"], res.coverage["missing_fraction"])
        verdict = cell_verdict(
            abs_median_ci=stats["abs_median_ci"], coverage_is_clean=cov_clean,
            ceiling_any=ceiling_any, mc_gate_overridden=mc_overridden,
        )
        recomputed_verdicts[key] = verdict
        if res.fvi is None:
            errors.append(f"{key}: completed recomputation has no FVI result")
            continue
        expected_scientific_claims = {
            "status": "completed",
            "fvi": {
                "status": res.fvi.status,
                "converged": res.fvi.converged,
                "iterations": res.fvi.iterations,
                "final_delta": res.fvi.final_delta,
                "tolerance": tol_label,
                "max_iterations": max_iter,
            },
            "coverage": {**res.coverage, "clean": cov_clean},
            "ceiling_flags": flags,
            "index_shift_by_item": res.index_shift_by_item,
            "bootstrap": {
                "point": stats["point"],
                "ci": stats["ci"],
                "abs_median_replicates": stats[
                    "abs_median_replicates"
                ].tolist(),
            },
            "descriptive": res.descriptive,
            "mc_gate_overridden": mc_overridden,
            "verdict": verdict,
        }
        for field, expected in expected_scientific_claims.items():
            _err(
                errors,
                _scientific_equal(stored.get(field), expected),
                f"{key}: stored {field} does not match independent recomputation",
            )

        expected_summary = {
            "status": "completed",
            "verdict": verdict,
            "abs_median_point": stats["abs_median_point"],
            "abs_median_ci": stats["abs_median_ci"],
            "coverage_clean": cov_clean,
            "ceiling_any": ceiling_any,
        }
        _err(
            errors,
            _scientific_equal(aggregate_cells.get(key), expected_summary),
            f"{key}: aggregate cell summary mismatch",
        )

        abs_median_reps[key] = stats["abs_median_replicates"]
        abs_median_point[key] = stats["abs_median_point"]

    # Family recompute.
    recomputed_family: dict[str, Any] | None = None
    family_valid = bool(abs_median_reps)
    if family_valid:
        fam = family_statistic(abs_median_reps, abs_median_point)
        all_cells_pass = (completed == expected_keys) and all(
            recomputed_verdicts.get(k) == "PASS" for k in expected_keys
        )
        fam_verdict = family_verdict(
            family_ci=fam["ci"], all_cells_pass=all_cells_pass, mc_override_active=mc_overridden
        )
        recomputed_family = {
            "M": fam["M"],
            "ci": fam["ci"],
            "verdict": fam_verdict,
        }
        stored_family_value = aggregate.get("family")
        stored_family = (
            stored_family_value
            if isinstance(stored_family_value, dict)
            else {}
        )
        _err(
            errors,
            isinstance(stored_family_value, dict)
            and set(stored_family) == {"M", "ci", "verdict"},
            "family fields do not match the canonical contract",
        )
        _err(
            errors,
            _scientific_equal(stored_family.get("M"), fam["M"]),
            "family M mismatch",
        )
        _err(errors, stored_family.get("verdict") == fam_verdict,
             f"family verdict mismatch: stored {stored_family.get('verdict')!r} != recomputed {fam_verdict!r}")
        stored_family_ci = stored_family.get("ci")
        _err(
            errors,
            _scientific_equal(stored_family_ci, fam["ci"]),
            "family CI mismatch",
        )
    else:
        _err(
            errors,
            aggregate.get("family") is None,
            "family must be null when no cells completed",
        )

    # Counts.
    _err(
        errors,
        aggregate.get("profile_name") == "stopdff_bucketed_dp_paired_v2",
        "aggregate profile_name mismatch",
    )
    for field, expected_count in (
        ("requested", len(expected_keys)),
        ("completed", len(completed)),
        ("failed", len(failed)),
    ):
        value = aggregate.get(field)
        _err(
            errors,
            _is_strict_int(value, minimum=0) and value == expected_count,
            f"{field} count mismatch",
        )
    skipped = aggregate.get("skipped")
    _err(
        errors,
        _is_strict_int(skipped, minimum=0) and skipped == 0,
        "skipped must be 0",
    )

    # Release validity recompute.
    pre_release_valid = not errors
    graph_and_records_valid = manifest_graph_valid and backend_manifest_valid
    release = release_validity(
        expected_cell_keys=expected_keys, present_cell_keys=sorted(completed | failed),
        completed_keys=completed, failed_keys=failed, skipped_keys=set(),
        all_calibrators_fitted=all_calibrators_fitted,
        all_fvi_converged=all_fvi_converged,
        manifests_valid=graph_and_records_valid,
        cache_matches_aggregate=(
            pre_release_valid and (completed | failed) == expected_keys
        ),
        bootstrap_valid=family_valid and plan is not None,
        family_valid=family_valid,
        backend_manifest_valid=backend_manifest_valid,
        attempt_history_valid=attempt_history_valid,
    )
    recomputed_status = "VALID" if release.valid else "INVALID"
    _err(errors, aggregate.get("release_status") == recomputed_status,
         f"release_status mismatch: stored {aggregate.get('release_status')!r} != recomputed {recomputed_status!r}")
    _err(
        errors,
        _scientific_equal(aggregate.get("release_reasons"), release.reasons),
        "release_reasons mismatch",
    )
    if not release.valid:
        errors.extend(f"release invalid: {reason}" for reason in release.reasons)

    if require_package:
        recomputed_fvi_study = None
        packaged_fvi_kind = None
        try:
            packaged_fvi_kind = inspect_packaged_fvi_manifest_kind(
                run_root,
                expected_id=spec_ids.get("fvi_study_id"),
                profile_variant=variant,
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"packaged FVI preflight is invalid: {exc}")
        if (
            packaged_fvi_kind == "fvi_study"
            and manifest_graph_valid
            and rows
            and isinstance(calibration, dict)
        ):
            cache_key = (
                str(adapter_bundle_id),
                str(adapter_identity.get("fit_rows_sha256")),
                str(adapter_identity.get("eval_rows_sha256")),
                str(adapter_identity.get("calibration_sha256")),
            )
            recomputed_fvi_study = _FVI_STUDY_CACHE.get(cache_key)
            if recomputed_fvi_study is None:
                try:
                    from .fvi_study import run_fvi_study

                    recomputed_fvi_study = run_fvi_study(
                        rows=rows,
                        calibration_json=calibration,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(
                        f"FVI study cannot be independently recomputed: {exc}"
                    )
                else:
                    _FVI_STUDY_CACHE[cache_key] = recomputed_fvi_study
        check_complete_checksums(run_root, errors)
        check_external_artifacts(
            run_root,
            errors,
            spec_ids=spec_ids,
            evidence_roots=evidence_roots,
            profile_variant=variant,
            fvi_selected={
                "tolerance": tol_label,
                "max_iterations": max_iter,
            },
            environment_claims=environment_claims,
            adapter_identity=adapter_identity,
            recomputed_fvi_study=recomputed_fvi_study,
        )
        _check_reports(run_root, aggregate, resource_summary_claims, errors)

    return CheckResult(
        passed=not errors, errors=errors,
        recomputed={"release_status": recomputed_status,
                    "family": recomputed_family,
                    "completed": len(completed), "failed": len(failed),
                    "run_spec_id": run_spec_id,
                    "adapter_bundle_id": adapter_bundle_id,
                    "bootstrap_plan_id": bootstrap_plan_id},
    )


def validate_run(
    run_root: Path,
    *,
    backend: str,
    adapter_bundle: Path,
    require_final_profile: bool = False,
    require_package: bool = False,
) -> CheckResult:
    """Validate and independently recompute untrusted run evidence.

    Parameters
    ----------
    run_root
        Directory containing the run artifacts.
    backend
        Expected execution backend, ``"local"`` or ``"modal"``.
    adapter_bundle
        Directory containing the identity-bound adapter bundle.
    require_final_profile
        Whether to require the final 96-cell profile.
    require_package
        Whether to enforce complete packaged-evidence and report checks.

    Returns
    -------
    CheckResult
        Structured validation status. Data-derived recursion and numeric
        overflow failures are normalized rather than escaping this boundary.
    """
    if not isinstance(backend, str) or backend not in {"local", "modal"}:
        return CheckResult(
            passed=False,
            errors=["backend must be exactly 'local' or 'modal'"],
        )
    try:
        return _validate_run_impl(
            run_root,
            backend=backend,
            adapter_bundle=adapter_bundle,
            require_final_profile=require_final_profile,
            require_package=require_package,
        )
    except (
        AttributeError,
        EOFError,
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
        zlib.error,
    ) as exc:
        return CheckResult(
            passed=False,
            errors=[
                "run evidence cannot be validated safely: "
                f"{type(exc).__name__}: {exc}"
            ],
        )
