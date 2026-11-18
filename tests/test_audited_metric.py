"""Frozen producers and active metrics must check their actual selected partitions."""

import json
from argparse import Namespace

import pytest


def _split(directory, name, text):
    path = directory / f"{name}_dataset.json"
    path.write_text(json.dumps([{"qid": name, "question": text, "answer_primary": "Hydrogen"}]),
                    encoding="utf-8")
    return path


@pytest.mark.parametrize("operation", ["dp", "sweep", "train", "learned"])
def test_custom_splits_checked_before_scoring(tmp_path, operation):
    _split(tmp_path, "fold_a", "case FOLD")
    _split(tmp_path, "fold_b", "CASE   fold")
    # This consumer takes its fitting reference from canonical training data by default.
    if operation == "learned":
        _split(tmp_path, "train", "case fold")
    if operation == "dp":
        from scripts.run_audited_metric import main
        assert main(["compute_stopdff_dp", "--data-dir", str(tmp_path), "--fit-split",
                     "fold_a", "--split", "fold_b", "--identity-calibration", "--smoke"]) == 1
        return
    with pytest.raises(ValueError, match="question-text overlap"):
        if operation == "sweep":
            from scripts.sweep_stopdff_dp import _load_dataframes
            _load_dataframes(Namespace(data_dir=str(tmp_path), fit_split="fold_a", eval_split="fold_b"))
        elif operation == "train":
            from scripts.train_stopdff_value_model import _load_questions_for_training
            _load_questions_for_training(data_dir=tmp_path, train_split="fold_a", val_split="fold_b",
                                        smoke=True, allow_incomplete_mc_coverage=False,
                                        allow_low_mc_retention=False)
        else:
            from scripts.compute_stopdff_learned_value import _load_eval_dataframe
            _load_eval_dataframe(Namespace(data_dir=str(tmp_path), eval_split="fold_b"), artifact_dir=tmp_path)


def test_preflight_uses_frozen_parser_defaults_and_forwards_arguments(tmp_path, monkeypatch):
    from scripts import compute_stopdff_dp, run_audited_metric
    _split(tmp_path, "val", "clean validation")
    _split(tmp_path, "test", "clean test")
    monkeypatch.setattr(compute_stopdff_dp, "DEFAULT_DATA_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(compute_stopdff_dp, "main", lambda argv: calls.append(argv) or 17)
    assert run_audited_metric.main(["compute_stopdff_dp", "--identity-calibration"]) == 17
    assert calls == [["--identity-calibration"]]
    _split(tmp_path, "test", "clean validation")
    assert run_audited_metric.main(["compute_stopdff_dp", "--identity-calibration"]) == 1
    assert calls == [["--identity-calibration"]]

def test_learned_value_compares_held_out_data_to_actual_checkpoint_training_directory(tmp_path):
    from scripts.compute_stopdff_learned_value import _load_eval_dataframe
    training = tmp_path / "fit"
    training.mkdir()
    _split(training, "fit_fold", "same clue")
    _split(training, "validation_fold", "independent clue")
    _split(tmp_path, "test_fold", "SAME  clue")
    with pytest.raises(ValueError, match="question-text overlap"):
        _load_eval_dataframe(
            Namespace(data_dir=str(tmp_path), eval_split="test_fold"), artifact_dir=tmp_path,
            training_partitions=[(str(training), "fit_fold", "validation_fold")],
        )

@pytest.mark.parametrize("content", [None, "[]"])
def test_preflight_requires_selected_files_even_when_canonical_files_are_clean(
    tmp_path, monkeypatch, content,
):
    from scripts import compute_stopdff_dp, run_audited_metric
    _split(tmp_path, "val", "clean validation")
    _split(tmp_path, "test", "clean test")
    if content is not None:
        (tmp_path / "fold_a_dataset.json").write_text(content, encoding="utf-8")
    monkeypatch.setattr(compute_stopdff_dp, "main", lambda argv: pytest.fail("unaudited input"))
    assert run_audited_metric.main([
        "compute_stopdff_dp", "--data-dir", str(tmp_path), "--fit-split", "fold_a",
    ]) == 1
