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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bootstrap import build_bootstrap_plan, cell_bootstrap_stats, family_statistic
from .cellcompute import compute_cell
from .identity import compute_id, loads_no_duplicate_keys, sha256_file
from .manifests import (
    ADAPTER_SCORING_SPEC,
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


def _is_quantized_number(value: Any, *, decimal_places: int) -> bool:
    """Return whether a finite number is unchanged by producer rounding."""
    if not _is_finite_number(value):
        return False
    number = float(value)
    return number == round(number, decimal_places)


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
    _err(errors, variant in {"smoke", "final"}, "invalid run spec profile_variant")
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


def validate_spec(spec_path: Path, *, require_final_profile: bool) -> CheckResult:
    try:
        spec = load_json(spec_path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return CheckResult(passed=False, errors=[f"run spec cannot be decoded: {exc}"])
    if not isinstance(spec, dict):
        return CheckResult(passed=False, errors=["run spec manifest must be an object"])
    errors = _run_spec_errors(
        spec.get("identity"),
        spec.get("id"),
        require_final_profile=require_final_profile,
    )
    return CheckResult(passed=not errors, errors=errors)


# --- validate-adapter -------------------------------------------------------------


def validate_adapter(bundle_dir: Path) -> CheckResult:
    errors: list[str] = []
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    _err(errors, manifest_path.exists(), "adapter manifest.json missing")
    if not manifest_path.exists():
        return CheckResult(passed=False, errors=errors)
    manifest = load_json(manifest_path)
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
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(f"adapter fit_rows cannot be decoded: {exc}")
    try:
        if (bundle_dir / "eval_rows.jsonl.gz").exists():
            eval_rows = load_jsonl_gz(bundle_dir / "eval_rows.jsonl.gz")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
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
                    and (
                        text_field == "category"
                        or bool(row.get(text_field))
                    ),
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
    expected_retention = {
        "fit_rows": len(fit_rows),
        "eval_rows": len(eval_rows),
    }
    _err(
        errors,
        ident.get("mc_retention_evidence") == expected_retention,
        "adapter MC retention evidence does not match row bytes",
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
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
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
                _err(
                    errors,
                    isinstance(block, dict)
                    and set(block) == {"platt_coef", "platt_intercept"},
                    f"adapter calibration {phase} parameters are noncanonical",
                )
                if isinstance(block, dict):
                    for parameter in ("platt_coef", "platt_intercept"):
                        _err(
                            errors,
                            _is_finite_number(block.get(parameter)),
                            f"adapter calibration {phase} {parameter} is invalid",
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


# --- validate (run) ---------------------------------------------------------------


def _check_png(path: Path, errors: list[str]) -> None:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        errors.append(f"invalid PNG signature: {path.name}")
        return
    # IHDR chunk: length(4) 'IHDR'(4) width(4) height(4)
    if len(data) < 24 or data[12:16] != b"IHDR":
        errors.append(f"invalid PNG IHDR: {path.name}")
        return
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        errors.append(f"PNG has non-positive dimensions: {path.name}")


def _check_checksums(run_root: Path, errors: list[str]) -> None:
    sums_path = run_root / "SHA256SUMS"
    if not sums_path.exists():
        errors.append("missing SHA256SUMS")
        return
    listed: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            errors.append(f"malformed SHA256SUMS line: {line!r}")
            continue
        digest, name = parts[0], parts[1]
        if name.startswith("/") or ".." in Path(name).parts:
            errors.append(f"unsafe checksum path: {name!r}")
        if name in listed:
            errors.append(f"duplicate checksum entry: {name!r}")
        listed[name] = digest
    # Every listed file must exist, be a regular file (no symlink), and match.
    for name, digest in listed.items():
        p = run_root / name
        if p.is_symlink():
            errors.append(f"symlink in checksums: {name!r}")
            continue
        if not p.is_file():
            errors.append(f"checksum target missing: {name!r}")
            continue
        if sha256_file(p) != digest:
            errors.append(f"checksum mismatch: {name!r}")


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
    if not path.exists():
        errors.append("missing attempts.jsonl")
        return False
    try:
        attempts = [
            loads_no_duplicate_keys(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(f"attempts.jsonl cannot be decoded: {exc}")
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
        if mode not in {"fresh", "resume"}:
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


def _check_reports(run_root: Path, errors: list[str]) -> None:
    md = run_root / "reports" / "report.md"
    tex = run_root / "reports" / "report.tex"
    required_md = [
        "profile", "paired", "reward", "calibrat", "continuation", "fvi",
        "family", "verdict", "override", "resource",
    ]
    if not md.exists():
        errors.append("missing reports/report.md")
    else:
        text = md.read_text(encoding="utf-8").lower()
        for token in required_md:
            if token not in text:
                errors.append(f"report.md missing required content: {token!r}")
    if not tex.exists():
        errors.append("missing reports/report.tex")
    figs = run_root / "figures"
    if figs.exists():
        for png in sorted(figs.glob("*.png")):
            _check_png(png, errors)


def _check_external_artifacts(
    run_root: Path,
    errors: list[str],
    *,
    spec_ids: dict[str, Any],
    fvi_selected: dict[str, Any],
    environment_claims: dict[str, Any],
) -> None:
    path = run_root / "external_artifacts.json"
    if not path.is_file():
        errors.append("missing external_artifacts.json")
        return
    try:
        payload = load_json(path)
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
    by_role: dict[str, dict[str, Any]] = {}
    exact_fields = {
        "role",
        "content_id",
        "sha256",
        "byte_size",
        "retrieval_path",
    }
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
            _is_strict_int(artifact.get("byte_size"), minimum=1),
            f"external artifact {role} byte_size must be positive",
        )
        _err(
            errors,
            isinstance(artifact.get("retrieval_path"), str)
            and bool(artifact.get("retrieval_path")),
            f"external artifact {role} retrieval_path must be nonempty",
        )
    required = {
        "source_manifest": spec_ids.get("source_manifest_id"),
        "raw_input_manifest": spec_ids.get("raw_input_bundle_id"),
        "model_snapshot_manifest": spec_ids.get("model_snapshot_id"),
        "fvi_study": spec_ids.get("fvi_study_id"),
        "environment_contract": spec_ids.get("environment_contract_id"),
    }
    for role, expected_id in required.items():
        _err(errors, role in by_role, f"missing external artifact role {role}")
        if role in by_role:
            _err(
                errors,
                by_role[role].get("content_id") == expected_id,
                f"external artifact {role} does not match run spec",
            )

    def _packaged_manifest(
        role: str,
        expected_path: str,
    ) -> dict[str, Any]:
        artifact = by_role.get(role)
        if artifact is None:
            return {}
        retrieval_path = artifact.get("retrieval_path")
        if retrieval_path != expected_path:
            errors.append(
                f"external artifact {role} must use {expected_path}"
            )
            return {}
        relative = Path(retrieval_path)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(
                f"external artifact {role} has an unsafe packaged path"
            )
            return {}
        evidence_path = run_root / relative
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
            manifest = load_json(evidence_path)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(f"packaged {role} cannot be decoded: {exc}")
            return {}
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("identity"),
            dict,
        ):
            errors.append(f"packaged {role} must be a manifest object")
            return {}
        try:
            _err(
                errors,
                compute_id(manifest["identity"]) == manifest.get("id"),
                f"packaged {role} manifest id mismatch",
            )
        except (TypeError, ValueError) as exc:
            errors.append(
                f"packaged {role} identity cannot be canonicalized: {exc}"
            )
            return {}
        _err(
            errors,
            manifest.get("id") == artifact.get("content_id"),
            f"packaged {role} content id mismatch",
        )
        return manifest

    fvi_manifest = _packaged_manifest(
        "fvi_study",
        "evidence/fvi_study.json",
    )
    fvi_identity = fvi_manifest.get("identity", {})
    if isinstance(fvi_identity, dict) and fvi_identity:
        _err(
            errors,
            fvi_identity.get("kind") in {"fvi_study", "fvi_study_fixed"},
            "packaged FVI evidence has an invalid kind",
        )
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
        "environment_contract",
        "evidence/environment_contract.json",
    )
    environment_identity = environment_manifest.get("identity", {})
    if isinstance(environment_identity, dict) and environment_identity:
        try:
            expected_environment_identity = environment_contract_identity(
                python_version=environment_claims["python_version"],
                package_versions=environment_claims["package_versions"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                f"packaged environment cannot be compared: {exc}"
            )
        else:
            _err(
                errors,
                environment_identity == expected_environment_identity,
                "packaged environment evidence does not match environment.json",
            )


def _resolve_run_binding(
    *,
    run_spec_manifest: dict[str, Any],
    adapter_bundle: Path,
    bootstrap_plan_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve canonical run inputs from self-valid manifests and local bytes."""
    errors: list[str] = []
    adapter_bundle = Path(adapter_bundle)

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
    if (
        not _is_strict_int(seed_value, minimum=0)
        or not _is_strict_int(replicate_value, minimum=1)
    ):
        seed = -1
        replicates = -1
        errors.append("bootstrap plan seed/replicate_count is invalid")
    else:
        seed = seed_value
        replicates = replicate_value
    _err(errors, seed == 1, f"bootstrap seed must be 1 (got {seed})")

    plan = None
    if item_ids and seed >= 0 and replicates > 0:
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

    rows: list[dict] = []
    try:
        if (
            (adapter_bundle / "fit_rows.jsonl.gz").is_file()
            and (adapter_bundle / "eval_rows.jsonl.gz").is_file()
        ):
            rows = load_adapter_rows(adapter_bundle)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
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

    variant = spec_identity.get("profile_variant")
    _err(
        errors,
        variant in {"smoke", "final"},
        f"unsupported run spec profile_variant {variant!r}",
    )
    expected_replicates = (
        FINAL_REPLICATES if variant == "final" else SMOKE_REPLICATES
    )
    _err(
        errors,
        replicates == expected_replicates,
        f"{variant} profile must use {expected_replicates} replicates (got {replicates})",
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
    try:
        tolerance = str(fvi_selected["tolerance"])
        max_iterations = int(fvi_selected["max_iterations"])
    except (KeyError, TypeError, ValueError):
        tolerance = ""
        max_iterations = -1
        errors.append("run spec FVI settings are invalid")

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
    """Return verified canonical run inputs or fail before computation."""
    binding, errors = _resolve_run_binding(
        run_spec_manifest=run_spec_manifest,
        adapter_bundle=adapter_bundle,
        bootstrap_plan_manifest=bootstrap_plan_manifest,
    )
    if errors:
        raise ValueError("run binding validation failed: " + "; ".join(errors))
    return binding


def validate_run(
    run_root: Path,
    *,
    backend: str,
    adapter_bundle: Path,
    require_final_profile: bool = False,
    require_package: bool = False,
) -> CheckResult:
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
    binding, binding_errors = _resolve_run_binding(
        run_spec_manifest=spec_manifest,
        adapter_bundle=adapter_bundle,
        bootstrap_plan_manifest=plan_manifest,
    )
    errors.extend(binding_errors)
    manifest_graph_valid = not binding_errors

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
        and str(aggregate_fvi.get("tolerance")) == tol_label,
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
        aggregate.get("gate_overrides") == gate_overrides,
        "aggregate gate_overrides do not match run spec",
    )

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
            backend_manifest = load_json(expected_manifest_path)
            backend_identity = backend_manifest.get("identity", {})
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
    _err(
        errors,
        isinstance(package_versions, dict)
        and bool(package_versions)
        and all(
            isinstance(name, str)
            and bool(name)
            and isinstance(version, str)
            and bool(version)
            for name, version in package_versions.items()
        ),
        "environment package_versions must be a nonempty string map",
    )
    if (
        isinstance(environment_claims.get("python_version"), str)
        and isinstance(package_versions, dict)
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
    actual_cell_keys = {
        path.stem
        for path in (run_root / "cells").glob("*.json")
    } if (run_root / "cells").is_dir() else set()
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

    for cell in cells:
        key = cell_key_str(cell)
        cell_path = run_root / "cells" / f"{key}.json"
        if not cell_path.exists():
            errors.append(f"missing cell file: {key}")
            failed.add(key)
            continue
        stored = load_json(cell_path)
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
        _err(
            errors,
            isinstance(fingerprint, dict)
            and compute_id(fingerprint) == stored.get("fingerprint_id"),
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
            )
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"{key}: independent cell recomputation failed: {exc}")
            failed.add(key)
            continue
        _err(
            errors,
            isinstance(fingerprint, dict)
            and fingerprint.get("calibrator_parameters")
            == res.calibrator_parameters,
            f"{key}: fingerprint calibrator_parameters mismatch",
        )
        _err(
            errors,
            stored.get("calibrator_parameters") == res.calibrator_parameters,
            f"{key}: stored calibrator_parameters mismatch",
        )
        if res.status != "completed":
            failed.add(key)
            _err(errors, stored.get("status") == res.status,
                 f"{key}: stored status {stored.get('status')!r} != recomputed {res.status!r}")
            # A non-completed cell must not be serialized as completed with a verdict.
            _err(errors, stored.get("status") != "completed",
                 f"{key}: non-converged/failed cell serialized as completed")
            continue

        completed.add(key)
        # index shifts must match exactly (no trusting serialized).
        stored_shifts = {str(k): int(v) for k, v in stored.get("index_shift_by_item", {}).items()}
        _err(errors, stored_shifts == res.index_shift_by_item,
             f"{key}: index_shift_by_item mismatch (cache stale or tampered)")

        stats = cell_bootstrap_stats(res.index_shift_by_item, plan)
        flags = ceiling_flags(res.mc_stops, res.qa_stops, list(res.index_shift_by_item.values()))
        ceiling_any = any(flags.values())
        cov_clean = coverage_clean(res.coverage["fallback_fraction"], res.coverage["missing_fraction"])
        verdict = cell_verdict(
            abs_median_ci=stats["abs_median_ci"], coverage_is_clean=cov_clean,
            ceiling_any=ceiling_any, mc_gate_overridden=mc_overridden,
        )
        recomputed_verdicts[key] = verdict
        _err(errors, stored.get("verdict") == verdict,
             f"{key}: stored verdict {stored.get('verdict')!r} != recomputed {verdict!r}")
        # coverage/ceiling serialized must match recomputed (no hiding a WARN).
        stored_cov = stored.get("coverage", {})
        _err(errors, bool(stored_cov.get("clean")) == cov_clean,
             f"{key}: coverage clean flag mismatch")
        _err(errors, stored.get("ceiling_flags") == flags, f"{key}: ceiling flags mismatch")
        _err(
            errors,
            stored.get("mc_gate_overridden") is mc_overridden,
            f"{key}: mc_gate_overridden mismatch",
        )
        stored_fvi = stored.get("fvi", {})
        _err(
            errors,
            str(stored_fvi.get("tolerance")) == tol_label
            and int(stored_fvi.get("max_iterations", -1)) == max_iter,
            f"{key}: stored FVI settings do not match run spec",
        )

        # Bootstrap point estimates and CIs must match independent recomputation.
        stored_point = stored.get("bootstrap", {}).get("point", {})
        for metric, expected in stats["point"].items():
            try:
                matches = abs(float(stored_point.get(metric)) - expected) < _FLOAT_TOL
            except (TypeError, ValueError):
                matches = False
            _err(errors, matches, f"{key}: bootstrap point mismatch for {metric}")
        stored_ci = stored.get("bootstrap", {}).get("ci", {}).get("absolute_index_median", [None, None])
        try:
            ci_matches = (
                len(stored_ci) == 2
                and abs(float(stored_ci[0]) - stats["abs_median_ci"][0])
                < _FLOAT_TOL
                and abs(float(stored_ci[1]) - stats["abs_median_ci"][1])
                < _FLOAT_TOL
            )
        except (TypeError, ValueError):
            ci_matches = False
        _err(errors, ci_matches, f"{key}: abs-median CI mismatch")

        aggregate_summary = (aggregate.get("cells") or {}).get(key, {})
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
            aggregate_summary == expected_summary,
            f"{key}: aggregate cell summary mismatch",
        )

        abs_median_reps[key] = stats["abs_median_replicates"]
        abs_median_point[key] = stats["abs_median_point"]

    # Family recompute.
    family_valid = bool(abs_median_reps)
    if family_valid:
        fam = family_statistic(abs_median_reps, abs_median_point)
        all_cells_pass = (completed == expected_keys) and all(
            recomputed_verdicts.get(k) == "PASS" for k in expected_keys
        )
        fam_verdict = family_verdict(
            family_ci=fam["ci"], all_cells_pass=all_cells_pass, mc_override_active=mc_overridden
        )
        stored_family = aggregate.get("family") or {}
        try:
            family_m_matches = (
                abs(float(stored_family.get("M")) - fam["M"]) < _FLOAT_TOL
            )
        except (TypeError, ValueError):
            family_m_matches = False
        _err(errors, family_m_matches, "family M mismatch")
        _err(errors, stored_family.get("verdict") == fam_verdict,
             f"family verdict mismatch: stored {stored_family.get('verdict')!r} != recomputed {fam_verdict!r}")
        if "ci" in stored_family:
            _err(errors,
                 abs(float(stored_family["ci"][0]) - fam["ci"][0]) < _FLOAT_TOL
                 and abs(float(stored_family["ci"][1]) - fam["ci"][1]) < _FLOAT_TOL,
                 "family CI mismatch")

    # Counts.
    _err(errors, aggregate.get("requested") == len(expected_keys), "requested count mismatch")
    _err(errors, aggregate.get("completed") == len(completed), "completed count mismatch")
    _err(errors, aggregate.get("failed") == len(failed), "failed count mismatch")
    _err(errors, int(aggregate.get("skipped", 0)) == 0, "skipped must be 0")

    # Release validity recompute.
    pre_release_valid = not errors
    graph_and_records_valid = manifest_graph_valid and backend_manifest_valid
    release = release_validity(
        expected_cell_keys=expected_keys, present_cell_keys=sorted(completed | failed),
        completed_keys=completed, failed_keys=failed, skipped_keys=set(),
        all_calibrators_fitted=not any("calibrator_failed" == load_json(run_root / "cells" / f"{k}.json").get("status") for k in failed),
        all_fvi_converged=not any("fvi_failed" == load_json(run_root / "cells" / f"{k}.json").get("status") for k in failed),
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

    if require_package:
        _check_checksums(run_root, errors)
        _check_external_artifacts(
            run_root,
            errors,
            spec_ids=spec_ids,
            fvi_selected={
                "tolerance": tol_label,
                "max_iterations": max_iter,
            },
            environment_claims=environment_claims,
        )
        _check_reports(run_root, errors)

    return CheckResult(
        passed=not errors, errors=errors,
        recomputed={"release_status": recomputed_status,
                    "family": aggregate.get("family"),
                    "completed": len(completed), "failed": len(failed),
                    "run_spec_id": run_spec_id,
                    "adapter_bundle_id": adapter_bundle_id,
                    "bootstrap_plan_id": bootstrap_plan_id},
    )
