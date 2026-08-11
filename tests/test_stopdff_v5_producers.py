"""Raw-input split-integrity tests for the StopDFF v5 control plane."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import producers  # noqa: E402


def _record(
    qid: str,
    question: str,
    answer: str,
    category: str = "General",
) -> dict:
    return {
        "qid": qid,
        "question": question,
        "answer_primary": answer,
        "category": category,
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_paths(tmp_path: Path) -> dict[str, Path]:
    sources = tmp_path / "sources"
    sources.mkdir()
    datasets = {
        "train_dataset.json": [_record("tr", "Train only?", "Train")],
        "val_dataset.json": [_record("va", "Validation only?", "Validation")],
        "test_dataset.json": [_record("te", "Test only?", "Test")],
        "mc_dataset.json": [
            {
                **_record("va", "Validation only?", "Validation"),
                "cumulative_prefixes": ["Validation only?"],
            },
            {
                **_record("te", "Test only?", "Test"),
                "cumulative_prefixes": ["Test only?"],
            },
        ],
    }
    for name, payload in datasets.items():
        _write_json(sources / name, payload)
    _write_json(
        sources / "build_metadata.json",
        {
            "splits": {
                split: {
                    "raw_count": 1,
                    "retained_count": 1,
                    "dropped_count": 0,
                }
                for split in ("train", "val", "test")
            }
        },
    )
    _write_json(
        sources / "split_metadata.json",
        {
            "train": {"count": 1, "categories": {"General": 1}},
            "val": {"count": 1, "categories": {"General": 1}},
            "test": {"count": 1, "categories": {"General": 1}},
            "total_questions": 3,
            "split_ratios": [1 / 3, 1 / 3, 1 / 3],
        },
    )
    _write_json(
        sources / "calibration.json",
        {"metadata": {"fit_split": "val"}},
    )
    _write_json(
        sources / "stopdff.json",
        {
            "median_abs_prefix_shift": 0.125,
            "metadata": {
                "metric_type": "diagnostic_only",
                "stopping_policy": "myopic_threshold",
            },
        },
    )
    threshold = sources / "threshold_manifest.json"
    _write_json(threshold, {"threshold": 0.5})
    digest = hashlib.sha256(threshold.read_bytes()).hexdigest()
    (sources / "threshold_manifest.json.sha256").write_text(
        f"{digest}  threshold_manifest.json\n",
        encoding="utf-8",
    )
    return {path.name: path for path in sources.iterdir()}


@pytest.mark.parametrize("overlap", ["qid", "normalized_text", "conflicting_answer"])
def test_stage_raw_inputs_recomputes_three_way_split_integrity(tmp_path, overlap):
    """Staged bytes, rather than split metadata claims, are the integrity oracle."""
    source_paths = _source_paths(tmp_path)
    train_path = source_paths["train_dataset.json"]
    val_path = source_paths["val_dataset.json"]

    if overlap == "qid":
        _write_json(val_path, [_record("tr", "Validation only?", "Validation")])
    elif overlap == "normalized_text":
        _write_json(
            val_path,
            [_record("va", "  ＴＲＡＩＮ   only? ", "Train")],
        )
    else:
        _write_json(
            train_path,
            [
                _record("tr1", "Who wrote Hamlet?", "William Shakespeare"),
                _record("tr2", "ｗｈｏ wrote HAMLET?", "Christopher Marlowe"),
            ],
        )

    with pytest.raises(ValueError, match="raw-input semantic checks failed"):
        producers.stage_raw_inputs(source_paths, tmp_path / "staged")


def test_stage_raw_inputs_binds_train_split_bytes(tmp_path):
    """The raw-input identity must cover train, validation, and test artifacts."""
    manifest = producers.stage_raw_inputs(
        _source_paths(tmp_path),
        tmp_path / "staged",
    )
    roles = {entry["role"] for entry in manifest["identity"]["files"]}
    assert "train_dataset.json" in roles


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", 123),
        ("answer_primary", ["not", "a", "scalar"]),
        ("qid", {"not": "a scalar"}),
    ],
)
def test_stage_raw_inputs_rejects_non_scalar_question_identity_fields(
    tmp_path,
    field,
    value,
):
    source_paths = _source_paths(tmp_path)
    train_path = source_paths["train_dataset.json"]
    record = _record("tr", "Train only?", "Train")
    record[field] = value
    _write_json(train_path, [record])

    with pytest.raises(ValueError, match="raw-input semantic checks failed"):
        producers.stage_raw_inputs(source_paths, tmp_path / "staged")


@pytest.mark.parametrize(
    "role",
    [
        "mc_dataset.json",
        "train_dataset.json",
        "val_dataset.json",
        "test_dataset.json",
        "build_metadata.json",
        "split_metadata.json",
        "calibration.json",
        "stopdff.json",
        "threshold_manifest.json",
    ],
)
def test_stage_raw_inputs_rejects_duplicate_keys_in_every_json_role(
    tmp_path,
    role,
):
    source_paths = _source_paths(tmp_path)
    source_paths[role].write_text(
        '{"duplicate": 1, "duplicate": 2}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        producers.stage_raw_inputs(source_paths, tmp_path / "staged")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("splits"),
        lambda payload: payload["splits"].pop("train"),
        lambda payload: payload["splits"].pop("val"),
        lambda payload: payload["splits"].pop("test"),
        lambda payload: payload["splits"]["train"].pop("retained_count"),
        lambda payload: payload["splits"]["val"].update(retained_count=0),
        lambda payload: payload["splits"]["test"].update(
            raw_count=2,
            retained_count=1,
            dropped_count=0,
        ),
    ],
)
def test_stage_raw_inputs_requires_and_recomputes_all_retained_counts(
    tmp_path,
    mutation,
):
    source_paths = _source_paths(tmp_path)
    metadata_path = source_paths["build_metadata.json"]
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(metadata_path, payload)

    with pytest.raises(ValueError, match="raw-input semantic checks failed"):
        producers.stage_raw_inputs(source_paths, tmp_path / "staged")


@pytest.mark.parametrize(
    "payload",
    [
        {
            "metadata": {
                "metric_type": "diagnostic_only",
                "stopping_policy": "myopic_threshold",
            },
        },
        {
            "median_abs_prefix_shift": "0.125",
            "metadata": {
                "metric_type": "diagnostic_only",
                "stopping_policy": "myopic_threshold",
            },
        },
        {
            "median_abs_prefix_shift": -0.125,
            "metadata": {
                "metric_type": "diagnostic_only",
                "stopping_policy": "myopic_threshold",
            },
        },
        {
            "median_abs_prefix_shift": 0.125,
            "metadata": {
                "metric_type": "finite_horizon_dp",
                "stopping_policy": "myopic_threshold",
            },
        },
        {
            "median_abs_prefix_shift": 0.125,
            "metadata": {
                "metric_type": "diagnostic_only",
                "stopping_policy": "learned_value",
            },
        },
    ],
)
def test_stage_raw_inputs_gates_on_canonical_myopic_metric(tmp_path, payload):
    source_paths = _source_paths(tmp_path)
    _write_json(source_paths["stopdff.json"], payload)

    with pytest.raises(ValueError, match="raw-input semantic checks failed"):
        producers.stage_raw_inputs(source_paths, tmp_path / "staged")



@pytest.mark.parametrize(
    "mutation",
    ["count", "category", "total", "ratio", "shape"],
)
def test_stage_raw_inputs_rejects_stale_split_metadata(tmp_path, mutation):
    """The retained-split sidecar must agree with the staged dataset bytes."""
    source_paths = _source_paths(tmp_path)
    metadata_path = source_paths["split_metadata.json"]
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mutation == "count":
        payload["train"]["count"] = 2
    elif mutation == "category":
        payload["val"]["categories"] = {"Stale": 1}
    elif mutation == "total":
        payload["total_questions"] = 4
    elif mutation == "ratio":
        payload["split_ratios"] = [0.5, 0.25, 0.25]
    else:
        payload["unexpected"] = True
    _write_json(metadata_path, payload)

    with pytest.raises(ValueError, match="raw-input semantic checks failed"):
        producers.stage_raw_inputs(source_paths, tmp_path / "staged")
