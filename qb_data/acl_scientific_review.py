"""Project source-scoped reviewed class decisions to regenerated ordered menus."""
from __future__ import annotations

from copy import deepcopy

from qb_data.acl_menus import AnswerClasses
from qb_data.acl_oracle_review import review_template, verify_review
from qb_data.acl_protocol import Prepared, digest


SCHEMA = "acl-question-class-review-v1"
def _header(prepared, dataset, registry):
    if (digest(dataset) != prepared.identity["dataset_sha256"] or
            digest(registry) != prepared.identity["oracle_sha256"]):
        raise ValueError("source dataset or equivalence registry review digest mismatch")
    return {"schema": SCHEMA, "dataset_sha256": prepared.identity["dataset_sha256"],
            "oracle_sha256": prepared.identity["oracle_sha256"],
            "reservoir_sha256": prepared.identity["reservoir_sha256"],
            "split_roles": prepared.identity["split_roles"]}


def extend_template(prepared: Prepared, dataset: dict, registry: dict, *, existing=None):
    """Add newly displayed classes as unresolved without modifying existing judgments."""
    header = _header(prepared, dataset, registry)
    out = {**header, "questions": {}} if existing is None else deepcopy(existing)
    if {k: v for k, v in out.items() if k != "questions"} != header:
        raise ValueError("class review belongs to different source inputs or splits")
    catalog = AnswerClasses(registry)
    for source in dataset["questions"]:
        qid = source["qid"]
        row = out["questions"].setdefault(qid, {"source_sha256": digest(source), "classes": {}})
        if row["source_sha256"] != digest(source):
            raise ValueError("class review question identity mismatch")
        for label in prepared.menus[qid]["options"]:
            row["classes"].setdefault(catalog.class_id(label), {
                "judgment": "unresolved", "reviewer": "", "rationale": ""})
    return out


def compile_review(prepared: Prepared, dataset: dict, registry: dict, reviewed: dict):
    """Build a per-slot sidecar from prior explicit decisions and verify its gold agreement."""
    header = _header(prepared, dataset, registry)
    if (not isinstance(reviewed, dict) or
            {k: v for k, v in reviewed.items() if k != "questions"} != header):
        raise ValueError("class review belongs to different source inputs or splits")
    sidecar = review_template(prepared, dataset)
    records = reviewed.get("questions")
    if not isinstance(records, dict) or set(records) != set(sidecar["questions"]):
        raise ValueError("class review must cover exactly the source questions")
    catalog = AnswerClasses(registry)
    for qid, target in sidecar["questions"].items():
        row = records[qid]
        if (not isinstance(row, dict) or set(row) != {"source_sha256", "classes"} or
                row["source_sha256"] != target["source_sha256"] or not isinstance(row["classes"], dict)):
            raise ValueError(f"{qid}: source-specific review identity mismatch")
        reviewers = set()
        rules = target["answer_rules"]
        local = registry.get("source_review", {}).get("context_only_by_qid", {}).get(qid, [])
        accepted = catalog.rule_classes(rules["accepted"], context_only=local)
        prompted = catalog.rule_classes(rules["prompt"], context_only=local)
        rejected = catalog.rule_classes(rules["rejected"], context_only=local)
        for item in target["options"]:
            key = catalog.class_id(item["label"])
            decision = row["classes"].get(key)
            if not isinstance(decision, dict) or set(decision) != {"judgment", "reviewer", "rationale"}:
                raise ValueError(f"{qid}: no reviewed class decision for {key!r}")
            if decision["judgment"] not in ("accepted", "rejected"):
                raise ValueError(f"{qid}: conditional, prompt-only or unresolved class {key!r} was displayed")
            if not all(isinstance(decision[k], str) and decision[k].strip()
                       for k in ("reviewer", "rationale")):
                raise ValueError(f"{qid}: incomplete review attribution for {key!r}")
            if (key in prompted or (key in accepted and decision["judgment"] != "accepted")
                    or (key in rejected and decision["judgment"] != "rejected")):
                raise ValueError(f"{qid}: class decision conflicts with the source answer rules for {key!r}")
            reviewers.add(decision["reviewer"])
            item.update(judgment=decision["judgment"], rationale=decision["rationale"])
        target.update(status="reviewed", reviewer=", ".join(sorted(reviewers)))
    # This checks the source rules, menu order and exactly one acceptable option.
    receipt = verify_review(prepared, dataset, sidecar)
    return sidecar, {**receipt, "class_review_sha256": digest(reviewed)}
