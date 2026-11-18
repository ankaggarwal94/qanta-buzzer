"""Report duplicate identities, overlapping clues and conflicting split labels."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qb_data.audit_json import loads
from qb_data.dataset_splits import normalize_question_text, normalize_split_answer


def audit_split_files(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Audit the supplied split files, recording the SHA-256 of the bytes parsed.

    Both a plain array and an object containing a questions array are accepted.
    Locations refer to zero-based row indices in each named split.
    """
    if not paths:
        raise ValueError("No split files were supplied for auditing")

    def read_splits():
        for name, entry in sorted(paths.items()):
            path = Path(entry)
            raw = path.read_bytes()
            payload = loads(raw.decode("utf-8-sig"))
            rows = payload.get("questions") if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                raise ValueError(f"{name}: expected a JSON list or a questions list")
            yield name, {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}, rows

    return _audit_rows(read_splits())


def _audit_rows(splits: Iterable[tuple[str, dict, Sequence]]) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, info, rows in splits:
        sources[name] = {**info, "rows": len(rows)}
        if not rows:
            issues.append({"split": name, "code": "empty_split"})
        for index, row in enumerate(rows):
            location: dict[str, Any] = {"split": name, "row_index": index}
            if not isinstance(row, Mapping):
                issues.append({**location, "code": "invalid_record_type"})
                continue
            qid = row.get("qid", row.get("qanta_id", row.get("id")))
            valid_id = (
                isinstance(qid, (str, int)) and not isinstance(qid, bool)
                and bool(str(qid).strip())
            )
            if valid_id:
                location["qid"] = str(qid).strip()
                by_id[location["qid"]].append(location)
            else:
                issues.append({**location, "code": "invalid_qid"})
            text = row.get("question", row.get("text"))
            answer = row.get("answer_primary", row.get("answer", row.get("page")))
            text_key = normalize_question_text(text) if isinstance(text, str) else ""
            if not text_key:
                issues.append({**location, "code": "empty_or_invalid_question"})
                continue
            if not isinstance(answer, str) or not normalize_split_answer(answer):
                issues.append({**location, "code": "empty_or_invalid_answer"})
            location["answer"] = answer if isinstance(answer, str) else None
            by_text[text_key].append(location)

    duplicate_ids = [
        {"qid": qid, "locations": locations}
        for qid, locations in sorted(by_id.items()) if len(locations) > 1
    ]
    cross_text: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    within_counts = {name: 0 for name in sources}
    directional = {
        left: {right: set() for right in sources if right != left}
        for left in sources
    }
    for text, locations in sorted(by_text.items()):
        if len(locations) < 2:
            continue
        group = {
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "locations": locations,
        }
        names = {item["split"] for item in locations}
        answers = {
            normalize_split_answer(item["answer"])
            for item in locations if isinstance(item["answer"], str)
        }
        if len(answers) > 1:
            conflicts.append({**group, "normalized_answers": sorted(answers)})
        for name in names:
            if sum(item["split"] == name for item in locations) > 1:
                within_counts[name] += 1
        if len(names) > 1:
            cross_text.append(group)
            for item in locations:
                for other in names - {item["split"]}:
                    directional[item["split"]][other].add(item["row_index"])
    overlap = {
        left: {
            right: {
                "rows": len(indices),
                "fraction": len(indices) / sources[left]["rows"]
                if sources[left]["rows"] else 0.0,
            }
            for right, indices in comparisons.items()
        }
        for left, comparisons in directional.items()
    }
    failed = bool(issues or duplicate_ids or cross_text or conflicts)
    return {
        "status": "FAIL" if failed else "PASS",
        "scope": "question IDs, nonempty labels, and normalized exact question text",
        "normalization": "Unicode NFKC, casefold, whitespace collapse; punctuation retained",
        "sources": sources,
        "issues": issues,
        "duplicate_ids": duplicate_ids,
        "cross_split_text_groups": cross_text,
        "conflicting_answer_groups": conflicts,
        "within_split_duplicate_text_groups": within_counts,
        "row_overlap": overlap,
        "prior_results": "recheck_results_using_these_splits" if failed else "not_certified_by_this_audit",
    }


def _check_report(report: dict[str, Any]) -> None:
    # Stages with optional or empty cohorts implement their own size policy.
    invalid_rows = sum(issue["code"] != "empty_split" for issue in report["issues"])
    duplicate_ids = len(report["duplicate_ids"])
    shared_texts = len(report["cross_split_text_groups"])
    conflicts = len(report["conflicting_answer_groups"])
    if invalid_rows or duplicate_ids or shared_texts or conflicts:
        raise ValueError(
            f"Dataset split integrity failed: {duplicate_ids} qid overlap groups, "
            f"{shared_texts} normalized question-text overlap groups, "
            f"{conflicts} conflicting-answer groups, {invalid_rows} invalid rows. "
            "Run scripts/audit_dataset.py for row locations."
        )


def require_clean_split_directory(data_dir: str | Path, **selected: Path) -> None:
    """Check all available named splits, with explicit files for overridden roles.

    Some legacy analyses supply just validation and test. An absent role cannot
    establish a boundary. The combined artifact is never itself a held-out split.
    """
    directory = Path(data_dir)
    paths = {name: directory / f"{name}_dataset.json" for name in ("train", "val", "test")}
    roles_by_filename = {path.name: name for name, path in paths.items()}
    for role, path in selected.items():
        if path.name != "mc_dataset.json":
            paths[roles_by_filename.get(path.name, role)] = path
    present = {name: path for name, path in paths.items() if path.exists()}
    if present:
        _check_report(audit_split_files(present))


def require_metric_splits(data_dir: Path, **selected: Path) -> None:
    """Check the files a metric actually uses and any available canonical peers."""
    for path in selected.values():
        if path.name == "mc_dataset.json":
            raise ValueError("The combined MC dataset cannot be used as a metric split")
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset {path}")
    paths = dict(selected)
    active = {path.resolve() for path in selected.values()}
    for name in ("train", "val", "test"):
        peer = data_dir / f"{name}_dataset.json"
        if name not in paths and peer.is_file() and peer.resolve() not in active:
            paths[name] = peer
    report = audit_split_files(paths)
    _check_report(report)
    for name in selected:
        if report["sources"][name]["rows"] == 0:
            raise ValueError(f"Selected {name} dataset is empty: {paths[name]}")


def validate_question_splits(**splits: Sequence) -> None:
    """Check the effective partitions when their sources span different directories."""
    def records(questions):
        return [
            row if isinstance(row, Mapping) else {
                "qid": row.qid, "question": row.question, "answer_primary": row.answer_primary
            }
            for row in questions
        ]

    _check_report(_audit_rows((name, {}, records(rows)) for name, rows in splits.items()))
