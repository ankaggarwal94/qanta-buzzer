"""Boundary tests for byte-derived StopDFF v5 adapter split bindings."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import adapter_build  # noqa: E402


def _entry(
    qid: str,
    question: str,
    answer: str,
    category: str = "Test",
) -> dict:
    return {
        "qid": qid,
        "question": question,
        "answer_primary": answer,
        "category": category,
    }


def test_adapter_split_binding_requires_every_retained_split_qid():
    val = {"v": {"text": "validation?", "answer": "validation", "category": "Test"}}
    test = {"t": {"text": "test?", "answer": "test", "category": "Test"}}

    with pytest.raises(ValueError, match="missing val split qids"):
        adapter_build._validate_split_bindings(
            val,
            test,
            [_entry("t", "Test?", "Test")],
        )


def test_adapter_split_binding_rejects_normalized_text_overlap():
    val = {"v": {"text": "shared question?", "answer": "same", "category": "Test"}}
    test = {"t": {"text": "shared question?", "answer": "same", "category": "Test"}}

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
    val = {"v": {"text": "validation?", "answer": "validation", "category": "Test"}}
    test = {"t": {"text": "test?", "answer": "test", "category": "Test"}}

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
        calls = 0

        def encode(
            self,
            values,
            *,
            batch_size,
            convert_to_numpy,
            show_progress_bar,
        ):
            self.calls += 1
            assert batch_size == adapter_build._ENCODE_BATCH_SIZE
            assert convert_to_numpy is True
            assert show_progress_bar is False
            vectors = {
                "option one": [1.0, 0.0],
                "option two": [1.0, 2.0],
                "prefix": [1.0, 1.0],
                "prefix extended": [1.0, 3.0],
            }
            return np.asarray([vectors[value] for value in values], dtype=float)

    model = FakeModel()
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
        model,
        "val",
    )

    assert model.calls == 1
    for row in rows:
        for field in (
            "prefix_fraction",
            "raw_similarity",
            "p_second_best",
            "top2_margin",
        ):
            assert row[field] == round(row[field], 6)


class _AliasFakeModel:
    """Deterministic encoder mapping the fixture strings to fixed vectors."""

    def encode(
        self,
        values,
        *,
        batch_size,
        convert_to_numpy,
        show_progress_bar,
    ):
        vectors = {
            "option one": [1.0, 0.0],
            "option two": [1.0, 2.0],
            "prefix": [1.0, 1.0],
            "prefix extended": [1.0, 3.0],
        }
        return np.asarray([vectors[value] for value in values], dtype=float)


def _canonical_scoring_question() -> dict:
    return {
        "qid": "q",
        "question": "prefix extended",
        "cumulative_prefixes": ["prefix", "prefix extended"],
        "options": ["option one", "option two"],
        "gold_index": 1,
        "answer_primary": "option two",
        "category": "test",
    }


def _alias_scoring_question() -> dict:
    """Same row as the canonical fixture but via the text/answer aliases that
    producers._raw_question_trajectory_binding and
    adapter_build._validate_split_bindings already accept."""
    return {
        "qid": "q",
        "text": "prefix extended",
        "cumulative_prefixes": ["prefix", "prefix extended"],
        "options": ["option one", "option two"],
        "gold_index": 1,
        "answer": "option two",
        "category": "test",
    }


def test_scoring_validator_accepts_text_answer_aliases():
    """A bundle staged via the text/answer aliases (accepted by the raw and
    split gates) must not be rejected by the adapter-build scoring validator;
    otherwise it is stageable but impossible to build into an adapter."""
    adapter_build._validate_scoring_question(_alias_scoring_question())


def test_scoring_rows_resolve_aliases_and_match_canonical_bytes():
    """The downstream scoring lookup must resolve the same aliases, and a
    canonical bundle's scored rows must be byte-identical (hash-attested rows
    never drift because _record_value prefers the canonical key)."""
    canonical_rows = adapter_build._score_question_rows(
        _canonical_scoring_question(), _AliasFakeModel(), "val"
    )
    alias_rows = adapter_build._score_question_rows(
        _alias_scoring_question(), _AliasFakeModel(), "val"
    )
    assert alias_rows == canonical_rows


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
def test_adapter_retained_selection_accepts_qid_aliases():
    questions = [
        {"question_id": "val-b"},
        {"id": "test-a"},
        {"qid": "ignored"},
    ]
    retained = adapter_build._select_retained_questions(
        questions,
        {"val-b"},
        {"test-a"},
    )
    assert [
        (adapter_build._record_qid(question), split)
        for question, split in retained
    ] == [("test-a", "test"), ("val-b", "val")]


@pytest.mark.parametrize("qid_field", ["question_id", "id"])
def test_adapter_scoring_accepts_qid_aliases(qid_field):
    class FakeModel:
        def encode(
            self,
            values,
            *,
            batch_size,
            convert_to_numpy,
            show_progress_bar,
        ):
            assert batch_size == adapter_build._ENCODE_BATCH_SIZE
            assert convert_to_numpy is True
            assert show_progress_bar is False
            vectors = {
                "correct": [1.0, 0.0],
                "wrong": [0.0, 1.0],
                "prefix": [1.0, 0.0],
            }
            return np.asarray([vectors[value] for value in values], dtype=float)

    question = {
        qid_field: "alias-qid",
        "question": "prefix",
        "cumulative_prefixes": ["prefix"],
        "options": ["correct", "wrong"],
        "gold_index": 0,
        "answer_primary": "correct",
        "category": "Test",
    }
    rows = adapter_build._score_question_rows(
        question,
        FakeModel(),
        "val",
    )
    assert rows
    assert {row["item_id"] for row in rows} == {"alias-qid"}


def test_freeze_model_snapshot_prunes_volatile_hub_cache_from_identity(
    tmp_path, monkeypatch
):
    """huggingface_hub local_dir transport metadata (.cache/) is volatile
    (etag + wall-clock timestamps) and must not enter the content-addressed
    model-snapshot identity, while pinned dot-files stay inventoried."""
    from scripts.stopdff_v5.content_manifest import validate_bound_content_manifest

    pinned_content = {
        ".gitattributes": b"*.safetensors filter=lfs\n",
        "config.json": b'{"hidden_size": 384}\n',
        "model.safetensors": b"\x00fixed-model-bytes\xff\n",
    }

    class _UnusedApi:
        def __init__(self):
            raise AssertionError("HfApi must not be constructed for a pinned revision")

    def _freeze(out_dir: Path, volatile: bytes) -> dict:
        def snapshot_download(*, repo_id: str, revision: str, local_dir: str) -> None:
            root = Path(local_dir)
            for name, data in pinned_content.items():
                (root / name).write_bytes(data)
            meta_dir = root / ".cache" / "huggingface" / "download"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "model.safetensors.metadata").write_bytes(volatile)

        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = _UnusedApi
        hub.snapshot_download = snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        for name, version in (
            ("sentence_transformers", "0.0.st-test"),
            ("transformers", "0.0.tf-test"),
        ):
            module = types.ModuleType(name)
            module.__version__ = version
            monkeypatch.setitem(sys.modules, name, module)
        return adapter_build.freeze_model_snapshot(out_dir, revision="a" * 40)

    first = _freeze(tmp_path / "one", b"etag-1 timestamp=1111111111.111\n")
    second = _freeze(tmp_path / "two", b"etag-2 timestamp=2222222222.222\n")

    # Identical pinned revisions freeze to the identical identity even when
    # the hub's transport metadata differs between downloads.
    assert first["id"] == second["id"]
    inventory = [entry["path"] for entry in first["identity"]["files"]]
    assert inventory == sorted(pinned_content)
    assert not any(".cache" in Path(path).parts for path in inventory)
    assert not (tmp_path / "one" / "snapshot" / ".cache").exists()

    # The pruned tree still satisfies the exhaustive bound-content walk that
    # every downstream consumer reproduces.
    validate_bound_content_manifest(
        tmp_path / "one",
        manifest_name="model_snapshot_manifest.json",
        expected_id=first["id"],
        file_key="files",
        name_key="path",
        content_subdir="snapshot",
        expected_kind="model_snapshot",
    )


def test_write_jsonl_gz_fsyncs_file_and_directory_around_publish(
    tmp_path, monkeypatch
):
    """Adapter row publication is crash-durable, not just rename-atomic.

    The row writer routes through the canonical durable-publish primitive
    (``fileio.publish_bytes``): flush + fsync the temp file before the
    rename, then fsync the directory so the published name survives a crash.
    """
    import os as os_module

    from scripts.stopdff_v5 import fileio, rowio

    events: list[str] = []
    real_fsync = os_module.fsync
    real_replace = os_module.replace

    def recording_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def recording_replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "fsync", recording_fsync)
    monkeypatch.setattr(fileio.os, "replace", recording_replace)

    path = tmp_path / "rows.jsonl.gz"
    rowio.write_jsonl_gz(path, [{"b": 2, "a": 1}])

    assert events == ["fsync", "replace", "fsync"]
    assert rowio.read_jsonl_gz(path) == [{"a": 1, "b": 2}]
