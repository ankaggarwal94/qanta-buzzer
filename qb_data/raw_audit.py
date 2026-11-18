"""Read-only checks for clue text, source identifiers and answer labels."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from qb_data.audit_json import loads
from qb_data.data_loader import _generate_qid
from qb_data.dataset_splits import normalize_question_text, normalize_split_answer


class _RenderedText(HTMLParser):
    """Keep clues and answer directives while removing QBReader display markup."""

    inline = frozenset(("b", "i", "u", "em", "strong", "sup", "sub", "span", "font"))

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "div"}:
            self.parts.append(" ")
        elif tag not in self.inline:
            self.parts.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag in {"p", "div"}:
            self.parts.append(" ")
        elif tag not in self.inline and tag != "br":
            self.parts.append(f"</{tag}>")


def _read_rows(data: bytes, path: Path, source_format: str) -> list[Any]:
    text = data.decode("utf-8-sig")
    if source_format == "csv":
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate CSV column names")
        if not (
            {"question", "Text"} & set(columns)
            and {"answer", "Answer", "answer_primary"} & set(columns)
        ):
            raise ValueError("Expected CSV columns for question and answer")
        return list(reader)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        # Split on line feeds, preserving Unicode separators within JSON strings.
        return [loads(line) for line in io.StringIO(text, newline=None)]

    document = loads(text)
    if isinstance(document, dict):
        if source_format == "qbreader" and "tossups" in document:
            document = document["tossups"]
            if isinstance(document, dict):
                document = document.get("questionArray")
        elif source_format == "qanta" and "questions" in document:
            document = document["questions"]
        elif any(
            key in document
            for key in ("question", "text", "question_text", "tossup_text")
        ):
            document = [document]
    if not isinstance(document, list):
        raise TypeError(
            "Expected a question array, a tossup, or a supported export envelope"
        )
    return document


def _first(row, keys):
    return next((row[k] for k in keys if k in row), None)


def _text(value, *, formatted: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    if formatted:
        parser = _RenderedText()
        parser.feed(value)
        parser.close()
        return "".join(parser.parts)
    return value


def audit_raw_file(
    path: Path, source_format: str, answer_field: str = "auto"
) -> tuple[dict, list[dict]]:
    """Audit source rows and return a summary plus the originals of rejected rows.

    Parameters
    ----------
    path
        CSV, JSON or JSONL input. The digest and rows use the same byte snapshot.
    source_format
        One of csv, qanta or qbreader.
    answer_field
        For QANTA, auto selects mapped page labels source-wide if present.
        Explicit page and answer select exactly that field.
    """
    if source_format not in {"csv", "qanta", "qbreader"}:
        raise ValueError(f"Unknown question format: {source_format}")
    if answer_field not in {"auto", "page", "answer"} or (
        answer_field != "auto" and source_format != "qanta"
    ):
        raise ValueError("Explicit answer selection requires QANTA page or answer")
    data = path.read_bytes()
    rows = _read_rows(data, path, source_format)
    if not rows:
        raise ValueError("Source contains no question records")
    if answer_field == "auto":
        has_pages = source_format == "qanta" and any(
            isinstance(r, dict) and "page" in r for r in rows
        )
        answer_field = "page" if has_pages else "generic"

    by_id: dict[str, list[int]] = defaultdict(list)
    by_text: dict[str, list[int]] = defaultdict(list)
    ids: dict[int, str] = {}
    answers: dict[int, str] = {}
    issues: dict[int, list[str]] = defaultdict(list)

    qbreader = source_format == "qbreader"
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            issues[index].append("invalid_record")
            continue
        if None in row:
            issues[index].append("csv_column_mismatch")
        if qbreader and any(field in row for field in ("leadin", "parts", "answers")):
            issues[index].append("not_a_tossup")
        raw_text = _first(
            row, ("question", "Text", "text", "question_text", "tossup_text")
        )
        if (
            source_format == "qanta"
            and isinstance(raw_text, list)
            and all(isinstance(c, str) for c in raw_text)
        ):
            raw_text = "|||".join(raw_text)
        question = _text(raw_text, formatted=qbreader)
        clues = question.split("|||") if not qbreader else [question]
        key = normalize_question_text(" ".join(clues))
        if key:
            by_text[key].append(index)
        if not key or any(not clue.strip() for clue in clues):
            issues[index].append("empty_or_invalid_clue")

        id_fields = ("Question ID", "qid", "question_id", "id", "qanta_id", "_id")
        source_id = _first(row, id_fields)
        if qbreader and isinstance(source_id, dict):
            oid = source_id.get("$oid")
            source_id = (
                oid
                if len(source_id) == 1
                and isinstance(oid, str)
                and re.fullmatch(r"[a-fA-F0-9]{24}", oid)
                else {}
            )
        if source_id is None or (source_format == "csv" and source_id == ""):
            if question:
                ids[index] = _generate_qid(question)
        elif type(source_id) in {int, str} and str(source_id).strip():
            ids[index] = str(source_id).strip()
        else:
            issues[index].append("invalid_id")
        if index in ids:
            by_id[ids[index]].append(index)

        if answer_field != "generic":
            value = row.get(answer_field)
        elif source_format == "csv":
            value = _first(row, ("answer", "Answer", "answer_primary"))
        else:
            label_keys = ("answer_primary", "answer", "clean_answer")
            value = _first(row, label_keys)
            if not any(key in row for key in label_keys):
                aliases = row.get("clean_answers")
                if isinstance(aliases, list) and aliases:
                    value = aliases[0]
        answers[index] = normalize_split_answer(_text(value, formatted=qbreader))
        if not answers[index]:
            issues[index].append("empty_or_invalid_answer")

    repeated_ids = []
    for qid, indices in by_id.items():
        if len(indices) > 1:
            repeated_ids.append({"qid": qid, "record_indices": indices})
            for index in indices:
                issues[index].append("duplicate_qid")

    duplicates, conflicts = [], []
    for key, indices in by_text.items():
        if len(indices) < 2:
            continue
        labels = sorted({answers[i] for i in indices})
        group = {
            "text_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "qids": [ids.get(i) for i in indices],
            "record_indices": indices,
            "normalized_answers": labels,
        }
        duplicates.append(group)
        if len(labels) > 1:
            conflicts.append(group)
            for index in indices:
                issues[index].append("conflicting_answers")

    rejected = [
        {"record_index": i, "qid": ids.get(i), "issues": issues[i]}
        for i in sorted(issues)
    ]
    report = {
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": source_format,
        "answer_field": answer_field,
        "total_records": len(rows),
        "accepted_records": len(rows) - len(rejected),
        "rejected_records": len(rejected),
        "issues": rejected,
        "issue_counts": dict(
            Counter(code for row in rejected for code in row["issues"])
        ),
        "duplicate_qid_groups": repeated_ids,
        "duplicate_text_groups": duplicates,
        "conflicting_answer_groups": conflicts,
    }
    return report, [
        dict(item, record=rows[item["record_index"] - 1]) for item in rejected
    ]
