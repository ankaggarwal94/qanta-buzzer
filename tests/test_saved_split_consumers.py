"""Regression coverage for saved split checks before scoring or capping."""

import importlib
import json
import sys

import pytest


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("consumer", [
    "compute_csli", "compute_stopdff", "compute_prefix_calibration",
    "compute_stopdff_dp", "sweep_stopdff_dp",
])
def test_legacy_compute_rejects_train_test_text_copy_before_smoke_scoring(
    tmp_path, monkeypatch, capsys, consumer, wrapped
):
    """Changing the ID and placing a copy beyond the smoke cap cannot hide it."""
    # Other test modules may import evaluation.controls during collection.
    # Preserve its identity while importing CSLI in the isolated context its
    # DATA-05 import guard requires (same convention as the CSLI guard tests).
    saved_controls = sys.modules.pop("evaluation.controls", None)
    try:
        module = importlib.import_module(f"scripts.{consumer}")
    finally:
        if saved_controls is not None:
            sys.modules["evaluation.controls"] = saved_controls

    def row(qid):
        return {"qid": qid, "question": f"Distinct clue for {qid}", "answer_primary": "Alpha"}

    train = [row("train-original")]
    val = [row("validation")]
    test = [row(f"test-{i:03d}") for i in range(50)]
    test[-1]["question"] = " ＤＩＳＴＩＮＣＴ  CLUE\nFOR TRAIN-ORIGINAL "
    for split, rows in {"train": train, "val": val, "test": test}.items():
        payload = {"questions": rows, "metadata": {"split": split}} if wrapped else rows
        (tmp_path / f"{split}_dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "mc_dataset.json").write_text(json.dumps(train + val + test), encoding="utf-8")

    scoring_calls = []

    def reject_scoring(*args, **kwargs):
        scoring_calls.append(True)
        raise AssertionError("scoring must not start on overlapping splits")

    if consumer == "compute_csli":
        monkeypatch.setattr(module, "run_choices_only_prompt", reject_scoring)
    elif consumer in {"compute_stopdff", "compute_prefix_calibration"}:
        monkeypatch.setattr(module, "_get_sbert_model", reject_scoring)
    else:
        from scripts.stopdff_dp import adapter
        monkeypatch.setattr(adapter, "build_dataframe", reject_scoring)

    output = tmp_path / "result.json"
    args = ["--data-dir", str(tmp_path), "--smoke"]
    if consumer == "compute_stopdff_dp":
        args += ["--identity-calibration", "--out", str(output),
                 "--out-md", str(tmp_path / "result.md"),
                 "--out-tex", str(tmp_path / "result.tex")]
    elif consumer == "sweep_stopdff_dp":
        args += ["--identity-calibration", "--out", str(output), "--max-cells", "1"]
    else:
        args += ["--output", str(output)]
        if consumer == "compute_csli":
            args += ["--models", "tfidf"]
        elif consumer == "compute_stopdff":
            args += ["--report-output", str(tmp_path / "report.json")]

    if consumer == "sweep_stopdff_dp":
        code = module.main(args)
    else:
        from scripts.run_audited_metric import main
        code = main([consumer, *args])
    assert code != 0
    assert not scoring_calls
    if consumer == "sweep_stopdff_dp":
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["metadata"]["status"] == "failed"
        assert "1 normalized question-text overlap groups" in payload["metadata"]["error"]
        assert payload["cells"] == []
    else:
        assert "1 normalized question-text overlap groups" in capsys.readouterr().err
        assert not output.exists()


@pytest.mark.parametrize("consumer", ["train_ppo", "run_baselines", "evaluate_all", "train_t5_policy"])
def test_training_entrypoints_check_siblings_before_reading_mc_fields(tmp_path, monkeypatch, consumer):
    from argparse import Namespace

    module = importlib.import_module(f"scripts.{consumer}")
    for split in ("train", "val", "test"):
        row = {"qid": split, "question": "validation" if split == "val" else "ＳＨＡＲＥＤ",
               "answer_primary": "Hydrogen"}
        (tmp_path / f"{split}_dataset.json").write_text(json.dumps([row]), encoding="utf-8")
    args = Namespace(mc_path=str(tmp_path / "mc_dataset.json"), smoke=False, config=None,
                     output_dir=str(tmp_path), overrides=[], seed=None)
    with pytest.raises(ValueError, match="1 normalized question-text overlap groups"):
        if consumer == "train_t5_policy":
            module.load_question_splits_with_metadata(args, {"data": {"max_questions": 1}})
        else:
            monkeypatch.setattr(module, "parse_args", lambda: args)
            module.main()


def test_combined_manifest_replays_recorded_qids_and_rejects_copied_text(tmp_path, sample_mc_question):
    from dataclasses import replace
    from scripts.compare_policies import resolve_t5_reference_questions, resolve_t5_test_questions

    pool = [replace(sample_mc_question, qid=qid, question=text) for qid, text in (
        ("t1", "test clue"), ("v1", "validation clue"), ("r1", "training clue"),
        ("t2", "another test clue"),
    )]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    path = tmp_path / "moved_mc.json"
    manifest = {"source": "random_split_fallback", "mc_path": "original_mc.json",
                "train_qids": ["r1"], "val_qids": ["v1"], "test_qids": ["t2", "t1"]}
    (checkpoint / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    test, name = resolve_t5_test_questions(checkpoint, pool, path)
    assert name == "split_manifest"
    assert [q.qid for q in test] == ["t2", "t1"]
    train, name = resolve_t5_reference_questions(checkpoint, pool, path, test)
    assert name == "split_manifest"
    assert [q.qid for q in train] == ["r1"]
    pool[1] = replace(pool[1], question="  ＴＲＡＩＮＩＮＧ  clue")
    with pytest.raises(ValueError, match="question-text overlap"):
        resolve_t5_reference_questions(checkpoint, pool, path, test)
    with pytest.raises(ValueError, match="missing"):
        resolve_t5_test_questions(checkpoint, pool[:-1], path)


def test_combined_reference_excludes_reidentified_test_copy(tmp_path, sample_mc_question):
    from dataclasses import replace
    from scripts.compare_policies import resolve_t5_reference_questions

    held_out = replace(sample_mc_question, qid="test", question="Held out clue")
    copy = replace(held_out, qid="another-id", question="HELD    OUT CLUE")
    other = replace(held_out, qid="train", question="Different clue")
    reference, _ = resolve_t5_reference_questions(
        tmp_path, [copy, other, held_out], tmp_path / "mc_dataset.json", [held_out],
    )
    assert reference == [other]


def test_t5_combined_training_keeps_equivalent_clues_in_one_split(sample_mc_question):
    from dataclasses import replace
    from scripts.train_t5_policy import split_questions

    pool = [replace(sample_mc_question, qid=str(i), question=f"Question {i // 2}")
            for i in range(24)]
    splits = split_questions(pool, {"data": {"seed": 1}})
    assert all(splits)
    assert sum(map(len, splits)) == len(pool)
    texts = [{q.question for q in part} for part in splits]
    assert texts[0].isdisjoint(texts[1] | texts[2]) and texts[1].isdisjoint(texts[2])

def test_manifest_caps_cannot_hide_a_copy_in_an_external_split_directory(tmp_path):
    from scripts.compare_policies import _check_saved_split_paths

    checkpoint = tmp_path / "run"
    checkpoint.mkdir()
    other = tmp_path / "external"
    other.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [{"qid": "selected", "question": "Train clue", "answer_primary": "Helium"},
            {"qid": "hidden", "question": "Duplicate", "answer_primary": "Helium"}]
    (other / "train_dataset.json").write_text(json.dumps(rows), encoding="utf-8")
    (artifacts / "test_dataset.json").write_text(
        json.dumps([{**rows[-1], "qid": "different"}]), encoding="utf-8",
    )
    manifest = {"train_path": "../external/train_dataset.json", "train_qids": ["selected"]}
    (checkpoint / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="question-text overlap"):
        _check_saved_split_paths(checkpoint, artifacts)
