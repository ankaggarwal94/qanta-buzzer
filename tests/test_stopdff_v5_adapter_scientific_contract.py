"""Fail-closed scientific-contract tests for StopDFF v5 adapter rows."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Callable

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import adapter_build, checker, identity, selftest  # noqa: E402


def _read_rows(bundle: Path, role: str) -> list[dict]:
    return [
        json.loads(line)
        for line in gzip.decompress((bundle / f"{role}_rows.jsonl.gz").read_bytes())
        .decode("utf-8")
        .splitlines()
        if line
    ]


def _rewrite_rows(
    bundle: Path,
    role: str,
    transform: Callable[[list[dict]], None],
) -> None:
    """Rewrite adapter rows and all count/hash/retention manifest claims."""
    rows = _read_rows(bundle, role)
    transform(rows)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    rows_path = bundle / f"{role}_rows.jsonl.gz"
    rows_path.write_bytes(gzip.compress(payload, mtime=0))

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][f"{role}_rows_sha256"] = identity.sha256_file(
        rows_path
    )
    manifest["identity"][f"{role}_row_count"] = len(rows)
    manifest["identity"]["mc_retention_evidence"][f"{role}_rows"] = len(rows)
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _rewrite_fit_rows(
    bundle: Path,
    transform: Callable[[list[dict]], None],
) -> None:
    _rewrite_rows(bundle, "fit", transform)


def _rewrite_eval_rows(
    bundle: Path,
    transform: Callable[[list[dict]], None],
) -> None:
    _rewrite_rows(bundle, "eval", transform)


def _rebind_calibration(bundle: Path) -> None:
    """Rebuild calibration when a test intends row changes to remain valid."""
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = adapter_build.derive_bound_calibration(
        fit_rows=_read_rows(bundle, "fit"),
        eval_rows=_read_rows(bundle, "eval"),
        model_snapshot_id=manifest["identity"]["model_snapshot_id"],
        fit_rows_sha256=manifest["identity"]["fit_rows_sha256"],
    )
    calibration_path = bundle / "calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
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


def _update_first(
    rows: list[dict],
    *,
    fmt: str,
    **updates,
) -> None:
    row = next(row for row in rows if row["format"] == fmt)
    row.update(updates)


def _update_prefix_pair(
    rows: list[dict],
    *,
    item_id: str,
    at_prefix_idx: int,
    **updates,
) -> None:
    matched = [
        row
        for row in rows
        if row["item_id"] == item_id and row["prefix_idx"] == at_prefix_idx
    ]
    assert {row["format"] for row in matched} == {"MC", "QA"}
    for row in matched:
        row.update(updates)


def _update_row(
    rows: list[dict],
    *,
    item_id: str,
    prefix_idx: int,
    fmt: str,
    **updates,
) -> None:
    row = next(
        row
        for row in rows
        if row["item_id"] == item_id
        and row["prefix_idx"] == prefix_idx
        and row["format"] == fmt
    )
    row.update(updates)


def test_adapter_rejects_rehashed_missing_qa_prefix_partner(tmp_path):
    built = selftest.build_valid_package(tmp_path)

    def delete_one_qa_prefix(rows: list[dict]) -> None:
        index = next(
            index for index, row in enumerate(rows) if row["format"] == "QA"
        )
        del rows[index]

    _rewrite_fit_rows(built["adapter_bundle"], delete_one_qa_prefix)

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any("paired MC/QA prefix rows" in error for error in result.errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prefix_fraction", 0.99),
        ("category", "mismatched-category"),
        ("K", 5),
        ("option_set_id", "mismatched-option-set"),
        ("distractor_strategy", "mismatched-strategy"),
    ],
)
def test_adapter_rejects_rehashed_mc_qa_shared_field_mismatch(
    tmp_path,
    field,
    value,
):
    built = selftest.build_valid_package(tmp_path)
    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_first(rows, fmt="QA", **{field: value}),
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed, field
    assert any("MC/QA shared fields" in error for error in result.errors)


@pytest.mark.parametrize("fmt", ["MC", "QA"])
@pytest.mark.parametrize("value", [-1.01, 1.01])
def test_adapter_rejects_rehashed_out_of_range_cosine_similarity(
    tmp_path,
    fmt,
    value,
):
    built = selftest.build_valid_package(tmp_path)
    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_first(
            rows,
            fmt=fmt,
            raw_similarity=value,
        ),
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed, (fmt, value)
    assert any("raw_similarity outside cosine range" in error for error in result.errors)


@pytest.mark.parametrize(
    "updates",
    [
        {"top2_margin": -0.1},
        {"p_second_best": 1.01},
        {"p_second_best": -1.01},
        {"top2_margin": 0.25},
    ],
)
def test_adapter_rejects_rehashed_impossible_mc_similarity_fields(
    tmp_path,
    updates,
):
    built = selftest.build_valid_package(tmp_path)
    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_first(rows, fmt="MC", **updates),
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed, updates
    assert any("MC similarity fields" in error for error in result.errors)


def test_adapter_rejects_rehashed_second_best_above_raw_within_rounding_quantum(
    tmp_path,
):
    """The residual tolerance must not relax the exact top-score ordering."""
    built = selftest.build_valid_package(tmp_path)
    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_first(
            rows,
            fmt="MC",
            raw_similarity=0.2,
            p_second_best=0.200001,
            top2_margin=0.0,
        ),
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any("MC similarity fields" in error for error in result.errors)


@pytest.mark.parametrize(
    "updates",
    [
        {"correct": 0},
        {"p_second_best": 0.1},
        {"top2_margin": 0.1},
    ],
)
def test_adapter_rejects_rehashed_noncanonical_qa_sentinels(
    tmp_path,
    updates,
):
    built = selftest.build_valid_package(tmp_path)
    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_first(rows, fmt="QA", **updates),
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed, updates
    assert any("QA sentinel fields" in error for error in result.errors)


@pytest.mark.parametrize(
    ("fmt", "updates"),
    [
        ("MC", {"K": 1, "option_set_id": "q000:K1"}),
        ("MC", {"option_set_id": "forged"}),
        ("QA", {"option_set_id": "forged"}),
    ],
)
def test_adapter_rejects_rehashed_noncanonical_option_set_identity(
    tmp_path,
    fmt,
    updates,
):
    built = selftest.build_valid_package(tmp_path)
    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_first(rows, fmt=fmt, **updates),
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed, (fmt, updates)
    assert any("option-set identity" in error for error in result.errors)


@pytest.mark.parametrize(
    ("prefix_idx", "updates", "error_fragment"),
    [
        (0, {"prefix_idx": 99}, "contiguous prefix indices"),
        (1, {"prefix_fraction": 0.05}, "nonmonotonic prefix fractions"),
        (1, {"category": "drifted"}, "metadata changes across prefixes"),
    ],
)
def test_adapter_rejects_rehashed_noncanonical_prefix_trajectory(
    tmp_path,
    prefix_idx,
    updates,
    error_fragment,
):
    built = selftest.build_valid_package(tmp_path)
    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_prefix_pair(
            rows,
            item_id="q000",
            at_prefix_idx=prefix_idx,
            **updates,
        ),
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed, updates
    assert any(error_fragment in error for error in result.errors)


def test_adapter_rejects_rehashed_extra_manifest_identity_field(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    manifest_path = built["adapter_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["unbound_claim"] = "forged"
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any(
        "identity fields do not match" in error for error in result.errors
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_manifest_id", "a" * 63),
        ("raw_input_bundle_id", "A" * 64),
        ("model_snapshot_id", "g" * 64),
    ],
)
def test_adapter_rejects_rehashed_noncanonical_upstream_identity(
    tmp_path,
    field,
    value,
):
    built = selftest.build_valid_package(tmp_path)
    manifest_path = built["adapter_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = value
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any(
        f"{field} must be canonical 64-hex" in error
        for error in result.errors
    )


def test_adapter_rejects_extra_manifest_top_level_field(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    manifest_path = built["adapter_bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unbound_claim"] = "forged"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any(
        "top-level fields do not match" in error for error in result.errors
    )


@pytest.mark.parametrize(
    ("field", "fmt"),
    [
        ("prefix_fraction", None),
        ("raw_similarity", "MC"),
        ("p_second_best", "MC"),
        ("top2_margin", "MC"),
    ],
)
def test_adapter_rejects_rehashed_seventh_decimal(field, fmt, tmp_path):
    built = selftest.build_valid_package(tmp_path)

    def mutate(rows: list[dict]) -> None:
        if fmt is None:
            _update_prefix_pair(
                rows,
                item_id="q000",
                at_prefix_idx=0,
                **{field: 0.1234567},
            )
        else:
            _update_row(
                rows,
                item_id="q000",
                prefix_idx=0,
                fmt=fmt,
                **{field: 0.1234567},
            )

    _rewrite_fit_rows(built["adapter_bundle"], mutate)

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any("quantized to 6 decimals" in error for error in result.errors)


def test_adapter_rejects_rehashed_noncanonical_row_order(tmp_path):
    built = selftest.build_valid_package(tmp_path)

    def swap_rows(rows: list[dict]) -> None:
        rows[0], rows[1] = rows[1], rows[0]

    _rewrite_fit_rows(built["adapter_bundle"], swap_rows)

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any("canonical producer order" in error for error in result.errors)


def test_adapter_coverage_pairing_uses_prefix_keys(tmp_path):
    built = selftest.build_valid_package(tmp_path)

    def delete_one_qa_prefix(rows: list[dict]) -> None:
        index = next(
            index for index, row in enumerate(rows) if row["format"] == "QA"
        )
        del rows[index]

    _rewrite_eval_rows(built["adapter_bundle"], delete_one_qa_prefix)

    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any("MC coverage evidence" in error for error in result.errors)


def test_adapter_producer_coverage_pairing_uses_prefix_keys():
    evidence = adapter_build._mc_coverage_evidence(
        [
            {"item_id": "q", "prefix_idx": 0, "format": "MC"},
            {"item_id": "q", "prefix_idx": 1, "format": "QA"},
        ]
    )

    assert evidence == {
        "eval_mc_items": 1,
        "eval_qa_items": 1,
        "paired": False,
    }


def test_adapter_accepts_cosine_endpoints_and_negative_values(tmp_path):
    built = selftest.build_valid_package(tmp_path)

    def use_boundary_values(rows: list[dict]) -> None:
        _update_row(
            rows,
            item_id="q000",
            prefix_idx=0,
            fmt="MC",
            raw_similarity=-1.0,
            p_second_best=-1.0,
            top2_margin=0.0,
        )
        _update_row(
            rows,
            item_id="q000",
            prefix_idx=0,
            fmt="QA",
            raw_similarity=-1.0,
        )
        _update_row(
            rows,
            item_id="q000",
            prefix_idx=1,
            fmt="MC",
            raw_similarity=1.0,
            p_second_best=-1.0,
            top2_margin=2.0,
        )
        _update_row(
            rows,
            item_id="q000",
            prefix_idx=1,
            fmt="QA",
            raw_similarity=1.0,
        )

    _rewrite_fit_rows(built["adapter_bundle"], use_boundary_values)
    _rebind_calibration(built["adapter_bundle"])

    result = checker.validate_adapter(built["adapter_bundle"])
    assert result.passed, result.errors


def test_adapter_rejects_fraction_not_derived_from_bound_content(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_prefix_pair(
            rows,
            item_id="q000",
            at_prefix_idx=1,
            prefix_fraction=0.1,
        ),
    )
    result = checker.validate_adapter(built["adapter_bundle"])
    assert not result.passed
    assert any(
        "prefix_fraction does not match bound question lengths" in error
        for error in result.errors
    )


def test_adapter_top_two_rounding_residual_boundary(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_row(
            rows,
            item_id="q000",
            prefix_idx=0,
            fmt="MC",
            raw_similarity=0.300001,
            p_second_best=0.2,
            top2_margin=0.1,
        ),
    )
    _rebind_calibration(built["adapter_bundle"])
    accepted = checker.validate_adapter(built["adapter_bundle"])
    assert accepted.passed, accepted.errors

    _rewrite_fit_rows(
        built["adapter_bundle"],
        lambda rows: _update_row(
            rows,
            item_id="q000",
            prefix_idx=0,
            fmt="MC",
            raw_similarity=0.300002,
        ),
    )
    rejected = checker.validate_adapter(built["adapter_bundle"])
    assert not rejected.passed
    assert any("MC similarity fields" in error for error in rejected.errors)


def test_adapter_producer_rejects_decreasing_cumulative_prefix_lengths():
    question = {
        "qid": "q",
        "question": "a sufficiently long complete question",
        "answer_primary": "answer",
        "cumulative_prefixes": ["a sufficiently long prefix", "short"],
        "options": ["answer", "distractor"],
        "gold_index": 0,
    }

    with pytest.raises(ValueError, match="nondecreasing cumulative_prefixes"):
        adapter_build._validate_scoring_question(question)


@pytest.mark.parametrize(
    ("prefixes", "error_fragment"),
    [
        (
            ["alpha beta", "gamma delta"],
            "strictly extending cumulative_prefixes",
        ),
        (
            ["alpha beta", "alpha delta epsilon"],
            "canonical question-token prefix",
        ),
        (
            ["alpha beta", "alpha beta gamma"],
            "canonical question-token prefix",
        ),
    ],
)
def test_adapter_producer_rejects_nonprefix_cumulative_text(
    prefixes,
    error_fragment,
):
    question = {
        "qid": "q",
        "question": "alpha beta delta epsilon",
        "answer_primary": "answer",
        "cumulative_prefixes": prefixes,
        "options": ["answer", "distractor"],
        "gold_index": 0,
    }

    with pytest.raises(ValueError, match=error_fragment):
        adapter_build._validate_scoring_question(question)


def test_adapter_producer_rejects_equal_canonical_prefixes():
    question = {
        "qid": "q",
        "question": "alpha beta gamma",
        "answer_primary": "answer",
        "cumulative_prefixes": ["alpha beta", "ALPHA  BETA"],
        "options": ["answer", "distractor"],
        "gold_index": 0,
    }

    with pytest.raises(ValueError, match="strictly extending cumulative_prefixes"):
        adapter_build._validate_scoring_question(question)


def test_adapter_producer_accepts_canonical_text_variants():
    question = {
        "qid": "q",
        "question": "Ａlpha  BETA gamma",
        "answer_primary": "answer",
        "cumulative_prefixes": [
            "alpha",
            "ALPHA beta",
            "Ａlpha  beta gamma",
        ],
        "options": ["answer", "distractor"],
        "gold_index": 0,
        "category": "Test",
    }

    adapter_build._validate_scoring_question(question)
