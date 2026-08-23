"""Regression tests for the Phase-4 build-metadata ledger-burn gap.

The producer hashes ``build_metadata.json`` while writing provenance after
scoring.  Pinned mode therefore has to include it in the pre-model staged
coverage gate; otherwise a missing or changed file can consume the one-shot
exception and fail only after model work has completed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from reproducibility.colm_aims_2026 import phase4, schema
from scripts import stopdff_fair_qa_retest as producer


def _args(tmp_path):
    return SimpleNamespace(
        data_dir=tmp_path / "staged_data",
        calibration=tmp_path / "calibration_train.json",
        fit_split="val",
        eval_split="test",
    )


def _eligibility():
    return {"derived_from": {"test_dataset_sha256": "1" * 64}}


def test_consumed_input_inventory_includes_build_metadata(tmp_path):
    consumed = producer._enumerate_consumed_inputs(_args(tmp_path), _eligibility())

    assert [entry["label"] for entry in consumed] == [
        "calibration_train",
        "eval_split",
        "fit_split",
        "mc_dataset",
        "answer_profiles",
        "build_metadata",
    ]
    assert consumed[-1]["path"] == tmp_path / "staged_data" / "build_metadata.json"
    assert consumed[-1]["frozen_sha256"] is None


def test_build_metadata_requires_operator_digest_before_model_work(tmp_path):
    consumed = producer._enumerate_consumed_inputs(_args(tmp_path), _eligibility())
    staged = [
        {
            "label": entry["label"],
            "path": entry["path"],
            "expected_sha256": str(index + 2) * 64,
        }
        for index, entry in enumerate(consumed)
        if entry["frozen_sha256"] is None and entry["label"] != "build_metadata"
    ]

    with pytest.raises(schema.TypedIngressError) as excinfo:
        phase4.required_staged_coverage(consumed, staged)

    assert "build_metadata.json" in str(excinfo.value)
