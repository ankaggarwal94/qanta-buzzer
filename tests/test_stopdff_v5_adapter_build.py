"""Boundary tests for byte-derived StopDFF v5 adapter split bindings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import adapter_build  # noqa: E402


def _entry(qid: str, question: str, answer: str) -> dict:
    return {
        "qid": qid,
        "question": question,
        "answer_primary": answer,
    }


def test_adapter_split_binding_requires_every_retained_split_qid():
    val = {"v": {"text": "validation?", "answer": "validation"}}
    test = {"t": {"text": "test?", "answer": "test"}}

    with pytest.raises(ValueError, match="missing val split qids"):
        adapter_build._validate_split_bindings(
            val,
            test,
            [_entry("t", "Test?", "Test")],
        )


def test_adapter_split_binding_rejects_normalized_text_overlap():
    val = {"v": {"text": "shared question?", "answer": "same"}}
    test = {"t": {"text": "shared question?", "answer": "same"}}

    with pytest.raises(ValueError, match="normalized question-text overlap"):
        adapter_build._validate_split_bindings(
            val,
            test,
            [
                _entry("v", "Shared question?", "Same"),
                _entry("t", "ＳＨＡＲＥＤ QUESTION?", "Same"),
            ],
        )


def test_adapter_split_binding_rejects_mc_source_mismatch():
    val = {"v": {"text": "validation?", "answer": "validation"}}
    test = {"t": {"text": "test?", "answer": "test"}}

    with pytest.raises(ValueError, match="answer does not match"):
        adapter_build._validate_split_bindings(
            val,
            test,
            [
                _entry("v", "Validation?", "Wrong"),
                _entry("t", "Test?", "Test"),
            ],
        )


@pytest.mark.parametrize("field", ["question", "answer_primary"])
def test_dataset_index_rejects_non_string_text_and_answer(tmp_path, field):
    record = _entry("v", "Validation?", "Validation")
    record[field] = ["not", "a", "string"]
    path = tmp_path / "val.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(ValueError, match="non-string question text or answer"):
        adapter_build._dataset_index(path, split="val")


def test_adapter_similarity_fields_keep_six_decimal_identity_contract():
    class FakeModel:
        def encode(self, values, *, convert_to_numpy):
            assert convert_to_numpy is True
            vectors = {
                "option one": [1.0, 0.0],
                "option two": [1.0, 2.0],
                "prefix": [1.0, 1.0],
                "prefix extended": [1.0, 3.0],
            }
            return np.asarray([vectors[value] for value in values], dtype=float)

    rows = adapter_build._score_question_rows(
        {
            "qid": "q",
            "question": "prefix extended",
            "cumulative_prefixes": ["prefix", "prefix extended"],
            "options": ["option one", "option two"],
            "gold_index": 1,
            "answer_primary": "option two",
            "category": "test",
        },
        FakeModel(),
        "val",
    )

    for row in rows:
        for field in (
            "prefix_fraction",
            "raw_similarity",
            "p_second_best",
            "top2_margin",
        ):
            assert row[field] == round(row[field], 6)


@pytest.mark.parametrize("loader", ["mc", "dataset", "calibration"])
def test_adapter_inputs_reject_duplicate_json_keys(tmp_path, loader):
    path = tmp_path / f"{loader}.json"
    path.write_text('{"duplicate": 1, "duplicate": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        if loader == "mc":
            adapter_build._load_mc_questions(path)
        elif loader == "dataset":
            adapter_build._dataset_index(path, split="val")
        else:
            adapter_build._load_calibration(path)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"metadata": []},
        {"metadata": {}},
        {"metadata": {"fit_split": "test"}},
    ],
)
def test_adapter_calibration_requires_object_bound_to_val(tmp_path, payload):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration"):
        adapter_build._load_calibration(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("question", 7, "question"),
        ("answer_primary", None, "answer"),
        ("cumulative_prefixes", [], "cumulative_prefixes"),
        ("cumulative_prefixes", ["valid", 7], "cumulative_prefixes"),
        ("options", ["only one"], "options"),
        ("options", ["one", 2], "options"),
        ("gold_index", True, "gold_index"),
        ("gold_index", 2, "gold_index"),
    ],
)
def test_adapter_scoring_rows_fail_closed_before_model_use(field, value, match):
    question = {
        "qid": "v",
        "question": "Validation?",
        "answer_primary": "Validation",
        "cumulative_prefixes": ["Validation?"],
        "options": ["Validation", "Test"],
        "gold_index": 0,
    }
    question[field] = value

    with pytest.raises(ValueError, match=match):
        adapter_build._validate_scoring_question(question)
