"""Focused regressions for the PR #30 adapter repair cluster."""
from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import (  # noqa: E402
    adapter_build,
    checker,
    identity,
    rowio,
    selftest,
)


def _scoring_question() -> dict:
    return {
        "qid": "q",
        "question": "alpha beta gamma",
        "answer_primary": "answer",
        "cumulative_prefixes": ["alpha", "alpha beta", "alpha beta gamma"],
        "options": ["answer", "distractor"],
        "gold_index": 0,
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_build_metadata(
    data_dir: Path,
    *,
    val_counts: tuple[int, int, int] = (1, 1, 0),
    test_counts: tuple[int, int, int] = (1, 1, 0),
    thresholds: dict[str, float] | None = None,
) -> None:
    def block(counts: tuple[int, int, int]) -> dict:
        raw, retained, dropped = counts
        return {
            "raw_count": raw,
            "retained_count": retained,
            "dropped_count": dropped,
            "retention_rate": retained / raw if raw else 0.0,
        }

    payload = {
        "splits": {
            "train": block((1, 1, 0)),
            "val": block(val_counts),
            "test": block(test_counts),
        },
    }
    if thresholds is not None:
        payload["retention_thresholds"] = thresholds
    _write_json(data_dir / "build_metadata.json", payload)


def test_scoring_question_binds_gold_answer_and_unique_normalized_options():
    mismatched = _scoring_question()
    mismatched["options"] = ["wrong", "answer"]
    with pytest.raises(ValueError, match="gold_index does not identify"):
        adapter_build._validate_scoring_question(mismatched)

    duplicated = _scoring_question()
    duplicated["options"] = ["The Answer", "answer!"]
    with pytest.raises(ValueError, match="duplicate normalized options"):
        adapter_build._validate_scoring_question(duplicated)


def test_scoring_question_requires_terminal_full_canonical_prefix():
    question = _scoring_question()
    question["cumulative_prefixes"] = ["alpha", "alpha beta"]

    with pytest.raises(ValueError, match="final cumulative prefix"):
        adapter_build._validate_scoring_question(question)


def test_adapter_binds_retention_decisions_and_canonical_prefix_fractions(
    tmp_path,
    monkeypatch,
):
    class FakeSentenceTransformer:
        def __init__(self, model_dir, *, trust_remote_code):
            assert Path(model_dir).name == "model"
            assert trust_remote_code is False

        def encode(self, values, *, convert_to_numpy):
            assert convert_to_numpy is True
            return np.asarray(
                [
                    [1.0, float((sum(map(ord, value)) % 7) + 1)]
                    for value in values
                ],
                dtype=float,
            )

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val = {
        "qid": "val",
        "question": "Ａlpha  BETA gamma",
        "answer_primary": "Answer",
    }
    test = {
        "qid": "test",
        "question": "Ｄelta  EPSILON zeta",
        "answer_primary": "Answer",
    }
    _write_json(data_dir / "val_dataset.json", [val])
    _write_json(data_dir / "test_dataset.json", [test])
    _write_json(
        data_dir / "mc_dataset.json",
        [
            {
                **val,
                "cumulative_prefixes": [
                    "alpha",
                    "ALPHA beta",
                    "Ａlpha beta gamma",
                ],
                "options": ["answer", "distractor"],
                "gold_index": 0,
            },
            {
                **test,
                "cumulative_prefixes": [
                    "delta",
                    "DELTA epsilon",
                    "Ｄelta epsilon zeta",
                ],
                "options": ["answer", "distractor"],
                "gold_index": 0,
            },
        ],
    )
    _write_build_metadata(
        data_dir,
        thresholds={"smoke": 0.5, "full": 0.8},
    )
    _write_json(data_dir / "calibration.json", {"fit_split": "val"})
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    manifest = adapter_build.build_adapter_bundle(
        mc_dataset_path=data_dir / "mc_dataset.json",
        val_dataset_path=data_dir / "val_dataset.json",
        test_dataset_path=data_dir / "test_dataset.json",
        calibration_path=data_dir / "calibration.json",
        model_snapshot_dir=model_dir,
        out_dir=tmp_path / "bundle",
        source_manifest_id="1" * 64,
        raw_input_bundle_id="2" * 64,
        model_snapshot_id="3" * 64,
        producer_hashes={"adapter_build.py": "4" * 64},
    )

    retention = manifest["identity"]["mc_retention_evidence"]
    assert retention["threshold_profile"] == "full"
    assert retention["build_metadata_sha256"]
    assert retention["fit_rows"] == 6
    assert retention["eval_rows"] == 6
    for role, split in (("fit", "val"), ("eval", "test")):
        decision = retention["splits"][role]
        assert decision["split"] == split
        assert decision["raw_count"] == 1
        assert decision["retained_count"] == 1
        assert float(decision["threshold"]) == pytest.approx(0.8)
        assert decision["passed"] is True
        assert decision["overridden"] is False
        assert decision["effective_pass"] is True

    fit_rows = rowio.read_jsonl_gz(tmp_path / "bundle" / "fit_rows.jsonl.gz")
    mc_fractions = [
        row["prefix_fraction"]
        for row in fit_rows
        if row["format"] == "MC"
    ]
    assert mc_fractions == [0.3125, 0.625, 1.0]

    eval_rows = rowio.read_jsonl_gz(
        tmp_path / "bundle" / "eval_rows.jsonl.gz"
    )
    retention_errors = checker._mc_retention_errors(
        retention,
        bundle_dir=tmp_path / "bundle",
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        fit_items={row["item_id"] for row in fit_rows},
        eval_items={row["item_id"] for row in eval_rows},
    )
    assert retention_errors == []

    mutated = json.loads(json.dumps(retention))
    mutated["splits"]["fit"]["retention_rate"] = "0.5"
    rejected = checker._mc_retention_errors(
        mutated,
        bundle_dir=tmp_path / "bundle",
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        fit_items={row["item_id"] for row in fit_rows},
        eval_items={row["item_id"] for row in eval_rows},
    )
    assert any("retention fit rate" in error for error in rejected)


def test_adapter_rejects_low_retention_unless_explicitly_overridden(tmp_path):
    _write_build_metadata(
        tmp_path,
        val_counts=(10, 1, 9),
        test_counts=(10, 1, 9),
        thresholds={"smoke": 0.5, "full": 0.8},
    )

    with pytest.raises(ValueError, match="below the full-profile threshold"):
        adapter_build._mc_retention_evidence(
            tmp_path,
            fit_item_count=1,
            eval_item_count=1,
            allow_low_mc_retention=False,
        )

    evidence = adapter_build._mc_retention_evidence(
        tmp_path,
        fit_item_count=1,
        eval_item_count=1,
        allow_low_mc_retention=True,
    )
    for role in ("fit", "eval"):
        decision = evidence["splits"][role]
        assert float(decision["threshold"]) == pytest.approx(0.8)
        assert decision["passed"] is False
        assert decision["overridden"] is True
        assert decision["effective_pass"] is True


def test_rowio_is_byte_deterministic_for_key_order_and_unicode():
    rows_a = [
        {"z": "雪", "a": 1},
        {"word": "café", "nested": {"β": 2, "a": 1}},
    ]
    rows_b = [
        {"a": 1, "z": "雪"},
        {"nested": {"a": 1, "β": 2}, "word": "café"},
    ]

    first = rowio.dumps_rows(rows_a)
    assert first == rowio.dumps_rows(rows_a)
    assert first == rowio.dumps_rows(rows_b)
    decoded = gzip.decompress(first).decode("utf-8")
    assert "雪" in decoded
    assert "café" in decoded
    assert "\\u96ea" not in decoded


def test_rowio_atomic_replace_failure_preserves_target_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "rows.jsonl.gz"
    target.write_bytes(b"existing-bytes")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(rowio.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        rowio.write_jsonl_gz(target, [{"answer": "雪"}])

    assert target.read_bytes() == b"existing-bytes"
    assert list(tmp_path.iterdir()) == [target]


def test_adapter_rejects_stale_calibration_after_all_hashes_are_rebound(
    tmp_path,
):
    built = selftest.build_valid_package(tmp_path)
    bundle = built["adapter_bundle"]
    calibration_path = bundle / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["per_bucket"]["early"]["platt_intercept"] += 0.125
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["calibration_sha256"] = identity.sha256_file(
        calibration_path
    )
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(bundle)
    assert result.passed is False
    assert any("not derived from bound fit-row" in error for error in result.errors)


def test_adapter_rejects_rebound_fit_rows_with_unrelated_calibration(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    bundle = built["adapter_bundle"]
    fit_path = bundle / "fit_rows.jsonl.gz"
    fit_rows = rowio.read_jsonl_gz(fit_path)
    fit_rows[0]["raw_similarity"] = round(
        min(1.0, float(fit_rows[0]["raw_similarity"]) + 0.2),
        6,
    )
    rowio.write_jsonl_gz(fit_path, fit_rows)

    calibration_path = bundle / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    new_fit_hash = identity.sha256_file(fit_path)
    calibration["metadata"]["fit_rows_sha256"] = new_fit_hash
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["fit_rows_sha256"] = new_fit_hash
    manifest["identity"]["calibration_sha256"] = identity.sha256_file(
        calibration_path
    )
    manifest["id"] = identity.compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(bundle)
    assert result.passed is False
    assert any("not derived from bound fit-row" in error for error in result.errors)
