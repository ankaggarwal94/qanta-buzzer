"""Identity, label and row-count contracts for the saved split audit."""

import json

import pytest

from qb_data.data_audit import audit_split_files


def _write_splits(tmp_path, rows):
    paths = {}
    for name, questions in rows.items():
        paths[name] = tmp_path / f"{name}_dataset.json"
        paths[name].write_text(json.dumps(questions), encoding="utf-8")
    return paths


def _row(qid, text, answer="Hydrogen"):
    return {"qid": qid, "question": text, "answer_primary": answer}


def test_counts_leaked_test_rows_once_despite_multiple_train_copies(tmp_path):
    paths = _write_splits(tmp_path, {
        "train": [_row("a", "Shared clue"), _row("b", "SHARED  clue")],
        "val": [_row("c", "unrelated clue")],
        "test": [_row("d", "shared clue"), _row("e", "shared\tclue"), _row("f", "fresh clue")],
    })
    report = audit_split_files(paths)
    assert report["status"] == "FAIL"
    assert report["duplicate_ids"] == []
    assert len(report["cross_split_text_groups"]) == 1
    assert report["row_overlap"]["test"]["train"] == {"rows": 2, "fraction": 2 / 3}
    assert report["within_split_duplicate_text_groups"] == {"train": 1, "val": 0, "test": 1}
    assert report["prior_results"] == "recheck_results_using_these_splits"


def test_conflicts_and_dirty_labels_fail_even_within_one_split(tmp_path):
    paths = _write_splits(tmp_path, {"train": [
        _row("a", "duplicate", "A"), _row("b", "duplicate", "Helium"),
        _row("c", "same text", "Mercury"), _row("d", "same text", "Venus"),
    ]})
    report = audit_split_files(paths)
    assert report["status"] == "FAIL"
    assert report["issues"][0]["code"] == "empty_or_invalid_answer"
    assert len(report["conflicting_answer_groups"]) == 2


def test_valid_wrapped_splits_have_hashes_and_do_not_certify_old_results(tmp_path):
    paths = _write_splits(tmp_path, {
        "train": {"questions": [_row(0, "unique train")]},
        "val": [_row("v", "unique val")], "test": [_row("t", "unique test")],
    })
    report = audit_split_files(paths)
    assert report["status"] == "PASS"
    assert all(len(source["sha256"]) == 64 for source in report["sources"].values())
    assert report["prior_results"] == "not_certified_by_this_audit"


def test_empty_audit_scope_is_not_a_pass():
    with pytest.raises(ValueError, match="No split files"):
        audit_split_files({})


@pytest.mark.parametrize("document,reason", [
    ('[{"qid":"first","qid":"hidden","question":"unique","answer":"Hydrogen"}]', "Duplicate JSON key"),
    ('[{"qid":"q","question":"unique","answer":"Hydrogen","metadata":{"score":NaN}}]', "Non-finite JSON number"),
    ('[{"qid":"q","question":"unique","answer":"Hydrogen","metadata":{"score":Infinity}}]', "Non-finite JSON number"),
])
def test_split_audit_rejects_ambiguous_or_nonstandard_json(tmp_path, document, reason):
    source = tmp_path / "train_dataset.json"
    source.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        audit_split_files({"train": source})

def test_canonical_reference_in_other_directory_is_checked(tmp_path):
    from qb_data.data_audit import require_clean_split_directory

    _write_splits(tmp_path, {"train": [_row("local", "unique train")],
                             "test": [_row("test", "held out")]})
    other = tmp_path / "other"
    other.mkdir()
    reference = _write_splits(other, {"train": [_row("copy", "HELD   OUT")]})["train"]
    require_clean_split_directory(tmp_path)
    with pytest.raises(ValueError, match="1 normalized question-text overlap groups"):
        require_clean_split_directory(tmp_path, train=reference)
