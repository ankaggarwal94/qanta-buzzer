"""Exercise the source auditor without changing the training loader's intake policy."""

import json

import pytest

from scripts.audit_dataset import main


def run_audit(tmp_path, source, *options):
    report = tmp_path / "report.json"
    code = main(["--input", str(source), "--report", str(report), *options])
    return code, json.loads(report.read_bytes())["raw_data"]


def test_csv_conflicts_preserve_both_sources_and_report_original_records(tmp_path):
    source = tmp_path / "source.csv"
    data = b"Question ID,Text,Answer\n1,Two||| clues,Mercury\n2,TWO   clues,Mars\n3,Another clue,Venus\n"
    source.write_bytes(data)
    quarantine = tmp_path / "quarantine.jsonl"
    code, report = run_audit(tmp_path, source, "--quarantine", str(quarantine))
    assert code == 1
    assert (report["total_records"], report["rejected_records"]) == (3, 2)
    assert report["conflicting_answer_groups"][0]["qids"] == ["1", "2"]
    assert [
        json.loads(line)["record"]["Answer"]
        for line in quarantine.read_text().splitlines()
    ] == ["Mercury", "Mars"]
    assert source.read_bytes() == data


def test_valid_duplicate_clues_and_distinct_ids_can_stay_together(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "qid,question,answer\na,Clue,The Moon\nb,CLUE,moon\n", encoding="utf-8"
    )
    code, report = run_audit(tmp_path, source)
    assert code == 0
    assert report["accepted_records"] == 2
    assert len(report["duplicate_text_groups"]) == 1


def test_dirty_rows_and_repeated_ids_are_all_reported(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "qid,question,answer\nx,,Mercury\nx,Clue,Mars\ny,Other,The\n", encoding="utf-8"
    )
    code, report = run_audit(tmp_path, source)
    assert code == 1
    assert report["rejected_records"] == 3
    assert report["duplicate_qid_groups"] == [{"qid": "x", "record_indices": [1, 2]}]


def test_qanta_auto_uses_one_label_field_for_the_entire_export(tmp_path):
    source = tmp_path / "qanta.json"
    source.write_text(
        json.dumps(
            {
                "questions": [
                    {"qanta_id": 0, "text": "Clue", "answer": "Raw", "page": "Entity"},
                    {"qanta_id": 1, "text": ["Other", "clue"], "answer": "Another"},
                ]
            }
        )
    )
    code, report = run_audit(tmp_path, source, "--format", "qanta")
    assert code == 1
    assert report["answer_field"] == "page" and report["rejected_records"] == 1
    assert (
        run_audit(
            tmp_path, source, "--format", "qanta", "--qanta-answer-field", "answer"
        )[0]
        == 0
    )


def test_qbreader_query_uses_formatted_tossups_and_preserves_labels(tmp_path):
    source = tmp_path / "qbreader.json"
    rows = [
        {
            "_id": {"$oid": "0123456789abcdef01234567"},
            "question": "<b>Æon</b> clue",
            "answer": "<i>Mercury</i>",
        },
        {"_id": "other", "question": "ÆON clue", "answer": "Mars [accept Ares]"},
    ]
    source.write_text(json.dumps({"tossups": {"questionArray": rows}, "bonuses": []}))
    code, report = run_audit(tmp_path, source, "--format", "qbreader")
    assert code == 1
    assert report["conflicting_answer_groups"][0]["qids"] == [
        "0123456789abcdef01234567",
        "other",
    ]
    assert report["rejected_records"] == 2


@pytest.mark.parametrize(
    "data", ['[{"text":"a","text":"b","answer":"c"}]', "NaN", "[]"]
)
def test_unreadable_or_empty_export_cannot_pass(tmp_path, data):
    source = tmp_path / "qanta.json"
    source.write_text(data)
    dest = tmp_path / "report.json"
    assert (
        main(["--input", str(source), "--format", "qanta", "--report", str(dest)]) == 2
    )
    assert json.loads(dest.read_bytes())["status"] == "ERROR"


def test_outputs_cannot_replace_input(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("question,answer\na,b\n")
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--report", str(source)])
    assert source.read_text() == "question,answer\na,b\n"


@pytest.mark.parametrize(
    "raw, expected_status",
    [("question,answer\n,Venus\n", "FAIL"), ("wrong_header\nVenus\n", "ERROR")],
)
def test_combined_report_keeps_split_findings_when_raw_is_dirty(
    tmp_path, raw, expected_status
):
    source = tmp_path / "questions.csv"
    source.write_text(raw)
    for name in ("train", "val", "test"):
        rows = [
            {
                "qid": name,
                "question": "Overlap" if name != "val" else "Different",
                "answer": "Venus",
            }
        ]
        (tmp_path / f"{name}_dataset.json").write_text(json.dumps(rows))
    report = tmp_path / "audit.json"
    code = main(
        ["--input", str(source), "--splits-dir", str(tmp_path), "--report", str(report)]
    )
    result = json.loads(report.read_bytes())
    assert code == (2 if expected_status == "ERROR" else 1)
    assert result["status"] == expected_status
    assert result["split_integrity"]["row_overlap"]["test"]["train"]["rows"] == 1
