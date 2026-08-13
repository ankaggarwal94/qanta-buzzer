"""Standalone checker (ACCEPTANCE_CONTRACT.md).

Independently recomputes cell/family/release statistics from the adapter rows and the
run package, never trusting serialized verdict fields. Also validates package structure:
backend-manifest exclusivity, attempt history, bootstrap plan, safe checksums,
external_artifacts, report semantics, and PNG validity.

No standalone validation requires another backend or comparison policy.

This module keeps the adapter and run validators and acts as the facade for
the checker module family: shared primitives live in ``checker_common``; the
run-spec, PNG, attempt/report, and package/calibration sections live in
``checker_runspec``, ``checker_png``, ``checker_attempts``,
``checker_package``, and ``checker_calibration``. Every historical public
name remains importable from this module.
"""
from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bootstrap import build_bootstrap_plan, cell_bootstrap_stats, family_statistic
from .cellcompute import compute_cell, prepare_cell_inputs
from .checker_attempts import (
    _check_attempts,
    _check_package_path_policy,
    _check_reports,
)
from .checker_calibration import platt_phase_errors
from .checker_common import (
    CheckResult,
    _canonical_path_issue,
    _is_finite_number,
    _is_quantized_number,
    _is_strict_int,
    _producer_hash_errors,
    _scientific_equal,
    load_adapter_rows,
    load_json,
    load_jsonl_gz,
)
from .checker_package import (
    _err,
    check_complete_checksums,
    check_external_artifacts,
    inspect_packaged_fvi_manifest_kind,
)
# Facade re-exports: `_check_png` and `validate_spec` are historical
# checker.py entry points that now live in sibling modules.
from .checker_png import _check_png  # noqa: F401
from .checker_runspec import (
    _run_spec_errors,
    _run_spec_manifest_envelope_errors,
    validate_spec,  # noqa: F401
)
from .identity import compute_id, is_sha256_hex, sha256_file
from .manifests import (
    ADAPTER_SCORING_SPEC,
    ENVIRONMENT_PACKAGES,
    environment_contract_identity,
)
from .profile import (
    EXPECTED_CELLS,
    FINAL_REPLICATES,
    FVI_MAX_ITERATIONS,
    FVI_TOLERANCES,
    SMOKE_REPLICATES,
    cell_key_str,
    full_grid,
    smoke_cells,
)
from .verdicts import (
    ceiling_flags,
    cell_verdict,
    coverage_clean,
    family_verdict,
    release_validity,
)

_FVI_STUDY_CACHE: dict[tuple[str, str, str, str], dict[str, Any]] = {}


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
        is_sha256_hex(value.get("build_metadata_sha256")),
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


# --- validate-adapter -------------------------------------------------------------


def _adapter_identity_errors(
    manifest: dict[str, Any],
    ident: dict[str, Any],
    errors: list[str],
) -> None:
    """Check the manifest id binding and the canonical identity contract."""
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
            is_sha256_hex(ident.get(identity_field)),
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


def _adapter_payload_binding_errors(
    bundle_dir: Path,
    ident: dict[str, Any],
    errors: list[str],
) -> bool:
    """Require the bound payload files and their identity hash bindings.

    Returns False when any payload file is unusable (fatal for the bundle).
    """
    required_files = (
        "fit_rows.jsonl.gz",
        "eval_rows.jsonl.gz",
        "calibration.json",
        "build_metadata.json",
    )
    payload_errors: list[str] = []
    for name in required_files:
        p = bundle_dir / name
        if p.is_symlink() or not p.is_file():
            payload_errors.append(
                f"adapter bundle {name} must be a non-symlink regular file"
            )
    if payload_errors:
        errors.extend(payload_errors)
        return False

    hash_bindings = (
        ("fit_rows.jsonl.gz", "fit_rows_sha256", "adapter fit_rows sha mismatch"),
        ("eval_rows.jsonl.gz", "eval_rows_sha256", "adapter eval_rows sha mismatch"),
        ("calibration.json", "calibration_sha256", "adapter calibration sha mismatch"),
    )
    for filename, identity_key, message in hash_bindings:
        path = bundle_dir / filename
        _err(errors, sha256_file(path) == ident.get(identity_key), message)
    return True


def _load_adapter_split_rows(
    bundle_dir: Path,
    errors: list[str],
) -> tuple[list[dict], list[dict]]:
    """Decode both row files, normalizing decoder failures into errors."""
    fit_rows: list[dict] = []
    eval_rows: list[dict] = []
    try:
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
    return fit_rows, eval_rows


def _adapter_row_errors(
    row: Any,
    index: int,
    *,
    label: str,
    expected_split: str,
    expected_columns: set[str],
    round_decimals: int,
    residual_tolerance: float,
    errors: list[str],
) -> tuple[str, Any, int] | None:
    """Validate one adapter row's fields against the scoring contract.

    Returns the row's ``(item_id, format, prefix_idx)`` key fields, or None
    when the row is too malformed to participate in cross-row checks.
    """
    if not isinstance(row, dict):
        errors.append(f"adapter {label} row {index} is not an object")
        return None
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
        return None
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
            is_sha256_hex(row.get(digest_field)),
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

    return item_id, row_format, prefix_idx


def _adapter_prefix_pair_errors(
    prefix_pairs: dict[tuple[str, int], dict[str, dict[str, Any]]],
    *,
    label: str,
    errors: list[str],
) -> None:
    """Require complete MC/QA pairs whose shared prefix fields agree."""
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


def _adapter_item_trajectory_errors(
    prefix_pairs: dict[tuple[str, int], dict[str, dict[str, Any]]],
    *,
    label: str,
    errors: list[str],
) -> None:
    """Check per-item prefix trajectories: contiguity, monotonicity, binding."""
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


def _adapter_rows_errors(
    rows: list[dict],
    *,
    label: str,
    expected_split: str,
    expected_columns: set[str],
    errors: list[str],
) -> set[str]:
    """Validate one split's rows and return the item ids they claim."""
    item_ids: set[str] = set()
    row_keys: set[tuple[str, str, int]] = set()
    prefix_pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    canonical_order_keys: list[tuple[str, str, int]] = []
    round_decimals = int(ADAPTER_SCORING_SPEC["round_decimals"])
    rounding_quantum = 10.0 ** -round_decimals
    residual_tolerance = rounding_quantum + 1e-12
    for index, row in enumerate(rows):
        row_key_fields = _adapter_row_errors(
            row,
            index,
            label=label,
            expected_split=expected_split,
            expected_columns=expected_columns,
            round_decimals=round_decimals,
            residual_tolerance=residual_tolerance,
            errors=errors,
        )
        if row_key_fields is None:
            continue
        item_id, row_format, prefix_idx = row_key_fields
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

    _adapter_prefix_pair_errors(prefix_pairs, label=label, errors=errors)
    _adapter_item_trajectory_errors(prefix_pairs, label=label, errors=errors)
    return item_ids


def _adapter_row_count_errors(
    ident: dict[str, Any],
    *,
    fit_rows: list[dict],
    eval_rows: list[dict],
    fit_items: set[str],
    eval_items: set[str],
    errors: list[str],
) -> None:
    """Bind manifest row counts to row bytes and forbid split overlap."""
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


def _adapter_trajectory_binding_errors(
    ident: dict[str, Any],
    *,
    fit_rows: list[dict],
    eval_rows: list[dict],
    errors: list[str],
) -> None:
    """Recompute the question-trajectory binding from row bytes."""
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


def _adapter_mc_coverage_errors(
    ident: dict[str, Any],
    *,
    eval_rows: list[dict],
    errors: list[str],
) -> None:
    """Recompute MC/QA eval coverage evidence from row bytes."""
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


def _adapter_calibration_errors(
    bundle_dir: Path,
    ident: dict[str, Any],
    *,
    fit_split: Any,
    fit_rows: list[dict],
    eval_rows: list[dict],
    errors: list[str],
) -> None:
    """Validate calibration.json claims and rederive it from bound bytes."""
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
    bundle_issue = _canonical_path_issue(bundle_dir, expect_directory=True)
    if bundle_issue == "symlink":
        return CheckResult(
            passed=False,
            errors=["adapter bundle root must be a non-symlink directory"],
        )
    if bundle_issue == "missing":
        return CheckResult(
            passed=False,
            errors=["adapter manifest.json missing"],
        )
    if bundle_issue is not None:
        return CheckResult(
            passed=False,
            errors=["adapter bundle root must be a non-symlink directory"],
        )
    manifest_path = bundle_dir / "manifest.json"
    if _canonical_path_issue(manifest_path, expect_directory=False) is not None:
        return CheckResult(
            passed=False,
            errors=["adapter manifest.json must be a non-symlink regular file"],
        )
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
    _adapter_identity_errors(manifest, ident, errors)

    if not _adapter_payload_binding_errors(bundle_dir, ident, errors):
        return CheckResult(passed=False, errors=errors)

    fit_rows, eval_rows = _load_adapter_split_rows(bundle_dir, errors)

    from .adapter_build import ADAPTER_SCHEMA_COLUMNS

    schema_columns = ident.get("schema_columns")
    _err(
        errors,
        schema_columns == ADAPTER_SCHEMA_COLUMNS,
        "adapter schema_columns do not match the canonical adapter schema",
    )
    expected_columns = set(ADAPTER_SCHEMA_COLUMNS)

    fit_split = ident.get("fit_split")
    eval_split = ident.get("eval_split")
    _err(errors, fit_split == "val", "adapter fit_split must be 'val'")
    _err(errors, eval_split == "test", "adapter eval_split must be 'test'")
    fit_items = _adapter_rows_errors(
        fit_rows,
        label="fit",
        expected_split=str(fit_split),
        expected_columns=expected_columns,
        errors=errors,
    )
    eval_items = _adapter_rows_errors(
        eval_rows,
        label="eval",
        expected_split=str(eval_split),
        expected_columns=expected_columns,
        errors=errors,
    )

    _adapter_row_count_errors(
        ident,
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        fit_items=fit_items,
        eval_items=eval_items,
        errors=errors,
    )
    _adapter_trajectory_binding_errors(
        ident,
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        errors=errors,
    )
    _adapter_mc_coverage_errors(ident, eval_rows=eval_rows, errors=errors)
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
    _adapter_calibration_errors(
        bundle_dir,
        ident,
        fit_split=fit_split,
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        errors=errors,
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


def _load_required_run_json(
    run_root: Path,
    filename: str,
    errors: list[str],
) -> dict[str, Any]:
    """Load one required run-root JSON object, recording why it is unusable."""
    path = run_root / filename
    path_issue = _canonical_path_issue(path, expect_directory=False)
    if path_issue == "missing":
        errors.append(f"missing {filename}")
        return {}
    if path_issue is not None:
        errors.append(f"{filename} must be a non-symlink regular file")
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


def _aggregate_fields_errors(
    aggregate: dict[str, Any],
    errors: list[str],
) -> None:
    """Check the aggregate's top-level field set against the contract."""
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


def _aggregate_binding_errors(
    aggregate: dict[str, Any],
    *,
    backend: str,
    variant: Any,
    run_spec_id: Any,
    adapter_bundle_id: Any,
    bootstrap_plan_id: Any,
    spec_ids: dict[str, Any],
    adapter_identity: dict[str, Any],
    evidence_roots: Any,
    tol_label: str,
    max_iter: int,
    gate_overrides: dict[str, bool],
    errors: list[str],
) -> None:
    """Bind every aggregate claim to the resolved run inputs."""
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


def _resource_summary_errors(
    run_root: Path,
    spec_ids: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """Load resource_summary.json and bind it to resource_summary_id."""
    resource_summary_claims = _load_required_run_json(
        run_root,
        "resource_summary.json",
        errors,
    )
    try:
        _err(
            errors,
            compute_id(resource_summary_claims)
            == spec_ids.get("resource_summary_id"),
            "resource_summary.json does not match resource_summary_id",
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"resource summary cannot be canonicalized: {exc}")
    return resource_summary_claims


def _backend_and_environment_errors(
    run_root: Path,
    *,
    backend: str,
    run_spec_id: Any,
    adapter_bundle_id: Any,
    bootstrap_plan_id: Any,
    adapter_identity: dict[str, Any],
    spec_ids: dict[str, Any],
    resource_summary_claims: dict[str, Any],
    errors: list[str],
) -> tuple[bool, dict[str, Any]]:
    """Check backend-manifest exclusivity, identity, and environment binding.

    Returns whether every backend/environment record was valid, plus the
    decoded environment claims for downstream package checks.
    """
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
    expected_manifest_issue = _canonical_path_issue(
        expected_manifest_path,
        expect_directory=False,
    )
    forbidden_manifest_issue = _canonical_path_issue(
        forbidden_manifest_path,
        expect_directory=False,
    )
    if expected_manifest_issue == "missing":
        errors.append(
            f"{backend} backend requires {expected_manifest_path.name}"
        )
    elif expected_manifest_issue is not None:
        errors.append(
            f"{expected_manifest_path.name} must be a non-symlink regular file"
        )
    _err(
        errors,
        forbidden_manifest_issue == "missing",
        f"{backend} backend forbids {forbidden_manifest_path.name}",
    )
    backend_manifest: dict[str, Any] = {}
    if expected_manifest_issue is None:
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

    environment_claims = _load_required_run_json(
        run_root,
        "environment.json",
        errors,
    )
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
    return len(errors) == backend_error_count, environment_claims


def _expected_cell_sets(
    aggregate: dict[str, Any],
    *,
    variant: Any,
    require_final_profile: bool,
    errors: list[str],
) -> tuple[list[dict], set[str], dict[str, Any]]:
    """Derive the profile cell grid and bind aggregate cell keys to it."""
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
    return cells, expected_keys, aggregate_cells


def _run_cells_dir_errors(
    cells_dir: Path,
    expected_keys: set[str],
    errors: list[str],
) -> None:
    """Require cells/ to hold exactly the profile's audited cell records."""
    if cells_dir.is_symlink():
        errors.append("run cells directory must not be a symlink")
        actual_cell_keys: set[str] = set()
    elif not cells_dir.is_dir():
        errors.append("run cells path must be a directory")
        actual_cell_keys = set()
    else:
        # Enumerate every entry, not just *.json: an orphaned atomic-write
        # temp file (hard kill between mkstemp and rename) must fail
        # validation instead of hiding beside the audited cell records.
        actual_cell_keys = set()
        for path in sorted(cells_dir.iterdir()):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix != ".json"
            ):
                errors.append(
                    f"unexpected non-cell entry in cells/: {path.name!r}"
                )
                continue
            actual_cell_keys.add(path.stem)
    _err(
        errors,
        actual_cell_keys == expected_keys,
        "run cell file set does not match the profile",
    )


# Canonical field sets for stored cell evidence (failed cells add "reason";
# completed cells add the recomputed scientific claims).
_COMMON_CELL_FIELDS = {
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
_COMPLETED_CELL_FIELDS = _COMMON_CELL_FIELDS | {
    "fvi",
    "coverage",
    "ceiling_flags",
    "index_shift_by_item",
    "bootstrap",
    "descriptive",
    "mc_gate_overridden",
    "verdict",
}


@dataclass
class _CellScan:
    """Mutable accumulator state for the per-cell recompute loop."""

    completed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    recomputed_verdicts: dict[str, str] = field(default_factory=dict)
    abs_median_reps: dict[str, Any] = field(default_factory=dict)
    abs_median_point: dict[str, float] = field(default_factory=dict)
    all_calibrators_fitted: bool = True
    all_fvi_converged: bool = True


def _load_cell_evidence(
    cell_path: Path,
    *,
    key: str,
    errors: list[str],
) -> dict[str, Any] | None:
    """Load one stored cell record, or None when it cannot participate."""
    if cell_path.is_symlink() or not cell_path.is_file():
        errors.append(f"{key}: cell evidence must be a regular non-symlink file")
        return None
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
        return None
    if not isinstance(stored, dict):
        errors.append(f"{key}: cell evidence must contain an object")
        return None
    return stored


def _cell_binding_errors(
    stored: dict[str, Any],
    *,
    key: str,
    cell: dict[str, str],
    run_spec_id: Any,
    adapter_bundle_id: Any,
    bootstrap_plan_id: Any,
    adapter_identity: dict[str, Any],
    evidence_roots: Any,
    expected_fvi_settings: dict[str, Any],
    errors: list[str],
) -> None:
    """Bind one stored cell record and its fingerprint to the run inputs."""
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


def _recompute_cell(
    *,
    cell: dict[str, str],
    key: str,
    rows: list[dict],
    calibration: Any,
    tol_label: str,
    max_iter: int,
    prepared_cell_inputs: Any,
    errors: list[str],
) -> Any | None:
    """Independently recompute one cell, or None when recomputation fails."""
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
        return None
    return res


def _cell_calibrator_errors(
    stored: dict[str, Any],
    res: Any,
    *,
    key: str,
    errors: list[str],
) -> None:
    """Compare stored and fingerprinted calibrator parameters to recompute."""
    fingerprint = stored.get("fingerprint_identity", {})
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


def _failed_cell_errors(
    stored: dict[str, Any],
    res: Any,
    *,
    key: str,
    aggregate_cells: dict[str, Any],
    scan: _CellScan,
    errors: list[str],
) -> None:
    """Record one failed cell and bind its stored claims to recomputation."""
    scan.failed.add(key)
    if res.status == "calibrator_failed":
        scan.all_calibrators_fitted = False
    if res.status == "fvi_failed":
        scan.all_fvi_converged = False
    expected_failed_fields = _COMMON_CELL_FIELDS | {"reason"}
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


def _completed_cell_errors(
    stored: dict[str, Any],
    res: Any,
    *,
    key: str,
    plan: Any,
    tol_label: str,
    max_iter: int,
    mc_overridden: bool,
    aggregate_cells: dict[str, Any],
    scan: _CellScan,
    errors: list[str],
) -> None:
    """Record one completed cell and bind every stored scientific claim."""
    scan.completed.add(key)
    _err(
        errors,
        set(stored) == _COMPLETED_CELL_FIELDS,
        f"{key}: completed cell fields do not match the canonical contract",
    )
    if plan is None:
        errors.append(f"{key}: bootstrap plan unavailable for recomputation")
        return
    try:
        stats = cell_bootstrap_stats(res.index_shift_by_item, plan)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{key}: bootstrap cannot be recomputed: {exc}")
        return
    flags = ceiling_flags(res.mc_stops, res.qa_stops, list(res.index_shift_by_item.values()))
    ceiling_any = any(flags.values())
    cov_clean = coverage_clean(res.coverage["fallback_fraction"], res.coverage["missing_fraction"])
    verdict = cell_verdict(
        abs_median_ci=stats["abs_median_ci"], coverage_is_clean=cov_clean,
        ceiling_any=ceiling_any, mc_gate_overridden=mc_overridden,
    )
    scan.recomputed_verdicts[key] = verdict
    if res.fvi is None:
        errors.append(f"{key}: completed recomputation has no FVI result")
        return
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

    scan.abs_median_reps[key] = stats["abs_median_replicates"]
    scan.abs_median_point[key] = stats["abs_median_point"]


def _family_errors(
    aggregate: dict[str, Any],
    *,
    scan: _CellScan,
    expected_keys: set[str],
    mc_overridden: bool,
    errors: list[str],
) -> tuple[dict[str, Any] | None, bool]:
    """Recompute the family statistic and bind the aggregate family claims."""
    # Family recompute.
    recomputed_family: dict[str, Any] | None = None
    family_valid = bool(scan.abs_median_reps)
    if family_valid:
        fam = family_statistic(scan.abs_median_reps, scan.abs_median_point)
        all_cells_pass = (scan.completed == expected_keys) and all(
            scan.recomputed_verdicts.get(k) == "PASS" for k in expected_keys
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
    return recomputed_family, family_valid


def _aggregate_count_errors(
    aggregate: dict[str, Any],
    *,
    expected_keys: set[str],
    scan: _CellScan,
    errors: list[str],
) -> None:
    """Bind aggregate profile name and counts to the recomputed cell scan."""
    # Counts.
    _err(
        errors,
        aggregate.get("profile_name") == "stopdff_bucketed_dp_paired_v2",
        "aggregate profile_name mismatch",
    )
    for field, expected_count in (
        ("requested", len(expected_keys)),
        ("completed", len(scan.completed)),
        ("failed", len(scan.failed)),
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


def _release_status_errors(
    aggregate: dict[str, Any],
    *,
    expected_keys: set[str],
    scan: _CellScan,
    family_valid: bool,
    plan: Any,
    manifest_graph_valid: bool,
    backend_manifest_valid: bool,
    attempt_history_valid: bool,
    errors: list[str],
) -> str:
    """Recompute release validity and bind the aggregate release claims."""
    # Release validity recompute.
    pre_release_valid = not errors
    graph_and_records_valid = manifest_graph_valid and backend_manifest_valid
    release = release_validity(
        expected_cell_keys=expected_keys,
        present_cell_keys=sorted(scan.completed | scan.failed),
        completed_keys=scan.completed, failed_keys=scan.failed, skipped_keys=set(),
        all_calibrators_fitted=scan.all_calibrators_fitted,
        all_fvi_converged=scan.all_fvi_converged,
        manifests_valid=graph_and_records_valid,
        cache_matches_aggregate=(
            pre_release_valid and (scan.completed | scan.failed) == expected_keys
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
    return recomputed_status


def _package_evidence_errors(
    run_root: Path,
    *,
    aggregate: dict[str, Any],
    spec_ids: dict[str, Any],
    evidence_roots: Any,
    variant: Any,
    adapter_bundle_id: Any,
    adapter_identity: dict[str, Any],
    rows: list[dict],
    calibration: Any,
    manifest_graph_valid: bool,
    tol_label: str,
    max_iter: int,
    environment_claims: dict[str, Any],
    resource_summary_claims: dict[str, Any],
    errors: list[str],
) -> None:
    """Run the packaged-evidence lanes: FVI study, checksums, reports."""
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
    _check_package_path_policy(run_root, errors)
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
    run_root_issue = _canonical_path_issue(run_root, expect_directory=True)
    if run_root_issue == "missing":
        return CheckResult(passed=False, errors=["run root does not exist"])
    if run_root_issue is not None:
        return CheckResult(
            passed=False,
            errors=["run root must be a non-symlink directory"],
        )

    aggregate = _load_required_run_json(run_root, "aggregate.json", errors)
    spec_manifest = _load_required_run_json(run_root, "run_spec.json", errors)
    plan_manifest = _load_required_run_json(run_root, "bootstrap_plan.json", errors)
    _aggregate_fields_errors(aggregate, errors)
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

    _aggregate_binding_errors(
        aggregate,
        backend=backend,
        variant=variant,
        run_spec_id=run_spec_id,
        adapter_bundle_id=adapter_bundle_id,
        bootstrap_plan_id=bootstrap_plan_id,
        spec_ids=spec_ids,
        adapter_identity=adapter_identity,
        evidence_roots=evidence_roots,
        tol_label=tol_label,
        max_iter=max_iter,
        gate_overrides=gate_overrides,
        errors=errors,
    )
    resource_summary_claims = _resource_summary_errors(run_root, spec_ids, errors)
    backend_manifest_valid, environment_claims = _backend_and_environment_errors(
        run_root,
        backend=backend,
        run_spec_id=run_spec_id,
        adapter_bundle_id=adapter_bundle_id,
        bootstrap_plan_id=bootstrap_plan_id,
        adapter_identity=adapter_identity,
        spec_ids=spec_ids,
        resource_summary_claims=resource_summary_claims,
        errors=errors,
    )

    attempt_history_valid = _check_attempts(
        run_root,
        errors,
        run_spec_id=str(run_spec_id),
        adapter_bundle_id=str(adapter_bundle_id),
        bootstrap_plan_id=str(bootstrap_plan_id),
        aggregate=aggregate,
    )

    cells, expected_keys, aggregate_cells = _expected_cell_sets(
        aggregate,
        variant=variant,
        require_final_profile=require_final_profile,
        errors=errors,
    )
    cells_dir = run_root / "cells"
    _run_cells_dir_errors(cells_dir, expected_keys, errors)

    scan = _CellScan()
    try:
        prepared_cell_inputs = prepare_cell_inputs(rows, calibration)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"cell inputs cannot be prepared: {exc}")
        prepared_cell_inputs = None

    for cell in cells:
        key = cell_key_str(cell)
        stored = _load_cell_evidence(
            cells_dir / f"{key}.json",
            key=key,
            errors=errors,
        )
        if stored is None:
            scan.failed.add(key)
            continue
        _cell_binding_errors(
            stored,
            key=key,
            cell=cell,
            run_spec_id=run_spec_id,
            adapter_bundle_id=adapter_bundle_id,
            bootstrap_plan_id=bootstrap_plan_id,
            adapter_identity=adapter_identity,
            evidence_roots=evidence_roots,
            expected_fvi_settings=expected_fvi_settings,
            errors=errors,
        )
        res = _recompute_cell(
            cell=cell,
            key=key,
            rows=rows,
            calibration=calibration,
            tol_label=tol_label,
            max_iter=max_iter,
            prepared_cell_inputs=prepared_cell_inputs,
            errors=errors,
        )
        if res is None:
            scan.failed.add(key)
            continue
        _cell_calibrator_errors(stored, res, key=key, errors=errors)
        if res.status != "completed":
            _failed_cell_errors(
                stored,
                res,
                key=key,
                aggregate_cells=aggregate_cells,
                scan=scan,
                errors=errors,
            )
            continue
        _completed_cell_errors(
            stored,
            res,
            key=key,
            plan=plan,
            tol_label=tol_label,
            max_iter=max_iter,
            mc_overridden=mc_overridden,
            aggregate_cells=aggregate_cells,
            scan=scan,
            errors=errors,
        )

    recomputed_family, family_valid = _family_errors(
        aggregate,
        scan=scan,
        expected_keys=expected_keys,
        mc_overridden=mc_overridden,
        errors=errors,
    )
    _aggregate_count_errors(
        aggregate,
        expected_keys=expected_keys,
        scan=scan,
        errors=errors,
    )
    recomputed_status = _release_status_errors(
        aggregate,
        expected_keys=expected_keys,
        scan=scan,
        family_valid=family_valid,
        plan=plan,
        manifest_graph_valid=manifest_graph_valid,
        backend_manifest_valid=backend_manifest_valid,
        attempt_history_valid=attempt_history_valid,
        errors=errors,
    )

    if require_package:
        _package_evidence_errors(
            run_root,
            aggregate=aggregate,
            spec_ids=spec_ids,
            evidence_roots=evidence_roots,
            variant=variant,
            adapter_bundle_id=adapter_bundle_id,
            adapter_identity=adapter_identity,
            rows=rows,
            calibration=calibration,
            manifest_graph_valid=manifest_graph_valid,
            tol_label=tol_label,
            max_iter=max_iter,
            environment_claims=environment_claims,
            resource_summary_claims=resource_summary_claims,
            errors=errors,
        )

    return CheckResult(
        passed=not errors, errors=errors,
        recomputed={"release_status": recomputed_status,
                    "family": recomputed_family,
                    "completed": len(scan.completed), "failed": len(scan.failed),
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
