"""Optional source- and menu-bound records of individual answerline judgments."""

from __future__ import annotations

from copy import deepcopy
from unicodedata import normalize

from qb_data.acl_menus import AnswerClasses
from qb_data.acl_protocol import Prepared, digest


def _answer_key(value: str) -> str:
    """Use the pinned Jane answer-rule comparison for the source records."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("answer rule must be a nonempty string")
    return " ".join(normalize("NFKC", value).casefold().split())


def review_template(prepared: Prepared, dataset: dict) -> dict:
    """Return private context for review with all judgments explicitly unresolved."""
    if digest(dataset) != prepared.identity["dataset_sha256"]:
        raise ValueError("review dataset differs from the prepared freeze")
    if digest(prepared.menus) != prepared.identity["menus_sha256"]:
        raise ValueError("prepared menu content differs from its recorded identity")
    source = {q["qid"]: q for q in dataset["questions"]}
    questions = {}
    for split, rows in prepared.questions.items():
        for q in rows:
            row = source[q.qid]
            if q.qid in questions or prepared.identity["split_roles"].get(row["split"]) != split:
                raise ValueError(f"{q.qid}: duplicate prepared question or frozen split mismatch")
            menu = prepared.menus[q.qid]
            if (list(menu["options"]) != q.options or menu["gold_index"] != q.gold_index
                    or row["question"] != q.question):
                raise ValueError(f"{q.qid}: prepared inputs differ from the frozen review context")
            questions[q.qid] = {
                "source_sha256": digest(row),
                "menu_sha256": menu["digest"],
                "split": split,
                "question": row["question"],
                "answer_rules": deepcopy(row["answer"]),
                "reviewer": "",
                "status": "unresolved",
                "options": [{"slot": i, "label": label, "judgment": "unresolved", "rationale": ""}
                            for i, label in enumerate(q.options, 1)],
            }
    if set(questions) != set(source):
        raise ValueError("review needs one prepared question for each source question")
    return {"schema_version": "acl-source-menu-review-v1", "dataset_sha256": digest(dataset),
            "menus_sha256": digest(prepared.menus), "questions": questions}


def verify_review(prepared: Prepared, dataset: dict, review: dict) -> dict:
    """Verify evidence coverage, identity and consistency and return a private receipt."""
    template = review_template(prepared, dataset)
    if (not isinstance(review, dict) or review.keys() != template.keys()
            or any(review[k] != template[k] for k in template if k != "questions")):
        raise ValueError("review schema, source digest or menu digest differs")
    rows = review["questions"]
    if not isinstance(rows, dict) or rows.keys() != template["questions"].keys():
        raise ValueError("review must cover every prepared qid exactly once")
    registry = prepared.registry
    if digest(registry) != prepared.identity["oracle_sha256"]:
        raise ValueError("source review equivalence registry digest mismatch")
    catalog = AnswerClasses(registry)
    local_terms = registry.get("source_review", {}).get("context_only_by_qid", {})
    for qid, expected in template["questions"].items():
        record = rows[qid]
        if (not isinstance(record, dict) or record.keys() != expected.keys()
                or any(record[k] != expected[k] for k in expected
                       if k not in {"reviewer", "status", "options"})):
            raise ValueError(f"{qid}: review context identity differs")
        if (record["status"] != "reviewed" or not isinstance(record["reviewer"], str)
                or not record["reviewer"].strip()):
            raise ValueError(f"{qid}: unresolved review or missing reviewer")
        options = record["options"]
        if not isinstance(options, list) or len(options) != len(expected["options"]):
            raise ValueError(f"{qid}: incomplete option judgments")
        rules = expected["answer_rules"]
        accepted, rejected, prompt = (
            {_answer_key(value) for value in rules[name]} for name in ("accepted","rejected","prompt")
        )
        if (accepted & rejected) or (accepted & prompt) or (rejected & prompt):
            raise ValueError(f"{qid}: conflicting source answer rules")
        ac, rc, pc = (catalog.rule_classes(rules[name], context_only=local_terms.get(qid, []))
                      for name in ("accepted", "rejected", "prompt"))
        gold = prepared.menus[qid]["gold_index"]
        for i, (item, target) in enumerate(zip(options, expected["options"])):
            if (not isinstance(item, dict) or item.keys() != target.keys()
                    or type(item["slot"]) is not int or item["slot"] != i+1
                    or item["label"] != target["label"]):
                raise ValueError(f"{qid}: judgment refers to a different slot or label")
            label = _answer_key(item["label"])
            key = catalog.class_id(item["label"])
            if (label in prompt or key in pc or (i != gold and (label in accepted or key in ac))
                    or (i == gold and (label in rejected or key in rc))):
                raise ValueError(f"{qid}: selected option conflicts with source-specific rule")
            if (item["judgment"] != ("accepted" if i == gold else "rejected")
                    or not isinstance(item["rationale"], str) or not item["rationale"].strip()):
                raise ValueError(f"{qid}: unresolved, contradictory or unexplained option judgment")
    return {"schema_version": review["schema_version"], "dataset_sha256": review["dataset_sha256"],
            "menus_sha256": review["menus_sha256"], "review_sha256": digest(review),
            "reviewed_questions": len(rows), "evidence_scope": dataset.get("evidence_scope", "unspecified")}
