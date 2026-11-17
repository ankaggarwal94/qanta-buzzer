"""Audit the hash-pinned historical ACL archive locally, publishing aggregate counts only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

from qb_data.acl_protocol import ArmSpec, digest, prepare, publish_json
from qb_data.acl_menus import _label_key

ARCHIVE_SHA256 = "cd2f72b9a1fd774ccd1da30aefca418b60c5c8c35ffaf88b625773dd13f82cc3"
MANIFEST_SHA256 = "aebe327f6e4b953cbe32c2a9640fa33f4bb381deca9226f5109f3f1a422c2d0a"
CROSSWALK = {"calibration":"train", "selection":"val", "test":"test"}


def _read(z, name):
    return json.loads(z.read(name))


def verify_manifest(z):
    outer_bytes = z.read("MANIFEST.json")
    if hashlib.sha256(outer_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("archive manifest digest mismatch")
    outer = json.loads(outer_bytes)
    names = z.namelist()
    if len(names) != len(set(names)) or set(outer) | {"MANIFEST.json"} != set(names):
        raise ValueError("duplicate or unlisted archive member")
    for name, expected in outer.items():
        with z.open(name) as fp:
            sha = hashlib.file_digest(fp,"sha256").hexdigest()
        if sha != expected["sha256"] or z.getinfo(name).file_size != expected["bytes"]:
            raise ValueError("archive member hash or byte count mismatch")
    for dirname, manifest in (("frozen","manifest.json"),("tokenizer","MANIFEST.json"),
                              ("code","SOURCE_MANIFEST.json")):
        for path, expected in _read(z,f"{dirname}/{manifest}")["files"].items():
            ref = outer[f"{dirname}/{path}"]
            if isinstance(expected,str):
                equal = ref["sha256"] == expected
            else:
                equal = (ref["sha256"] == expected["sha256"]
                         and ref["bytes"] == expected.get("bytes",expected.get("byte_count")))
            if not equal:
                raise ValueError(f"{dirname} manifest disagrees with pinned package manifest")
    return len(outer)


def audit(path):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("selected archive differs from the pinned historical bytes")
    with ZipFile(path) as z:
        files = verify_manifest(z)
        dataset = _read(z,"frozen/evaluator/main_dataset.json")
        reservoir = _read(z,"frozen/evaluator/reservoir.json")
        queue = _read(z,"frozen/evaluator/review_queue.json")["questions"]
        identity_groups = len(_read(z,"identity_overrides.json")["equivalences"])
        type_review = _read(z,"type_matching/validation_receipt.json")
        file_manifest = _read(z,"MANIFEST.json")
    rows = dataset["questions"]
    questions = {q["qid"]:q for q in rows}
    targets = {sid for q in rows for sid in q["source"]["component_source_ids"]}
    donor_ids = {r["source"]["_id"] for r in reservoir}
    donor_members = {sid for r in reservoir for sid in r["component_source_ids"]}
    if len(questions) != len(rows) or targets & donor_members:
        raise ValueError("duplicate qid or shared target/reservoir source member")
    by_qid = {entry["qid"]:entry for entry in queue}
    if len(by_qid) != len(queue) or by_qid.keys() != questions.keys():
        raise ValueError("historical review queue coverage differs from dataset")
    for q in rows:
        entry = by_qid[q["qid"]]
        if (entry["answerline"] != q["answer"]["raw"] or entry["split"] != q["split"]
                or entry["syntactically_accepted"] != q["answer"]["accepted"]):
            raise ValueError("source review queue refers to different question evidence")
    classes = defaultdict(set)
    for q in rows:
        classes[q["source"]["canonical_answer"]].update(q["answer"]["accepted"])
    for r in reservoir:
        classes[r["canonical_answer"]].update(r["aliases"])
    owners = defaultdict(set)
    for canonical, labels in classes.items():
        for label in {canonical, *labels}:
            owners[_label_key(label)].add(canonical)
    conflicts = sum(len(v)>1 for v in owners.values())
    if conflicts:
        raise ValueError(f"{conflicts} normalized aliases have multiple mechanical owners")
    # Existing schema requires a reviewer field. The sentinel explicitly states
    # no review, and the mapping exists only in memory for this structural test.
    alias_input = {"version":"UNREVIEWED-archive-structural-test-v1",
                   "reviewer":"NO REVIEWER: mechanically projected upstream labels",
                   "classes":{c:sorted(labels-{c}) for c,labels in classes.items()}}
    counts = {}
    for condition in ("independent_pool","same_category_pool"):
        spec = ArmSpec.from_dict({"version":1,"k":4,"seed":1,"menu":"baseline",
                                  "baseline_condition":condition,"reward":{"arm":"baseline"}})
        prepared = prepare(spec,dataset,alias_input,reservoir=reservoir,split_roles=CROSSWALK)
        slots, distractors, prefixes = Counter(), 0, 0
        visited = set()
        for role, pool in prepared.questions.items():
            for question in pool:
                if question.qid in visited:
                    raise ValueError("repeated question in structural comparison")
                visited.add(question.qid)
                source = questions[question.qid]
                old = next(m for m in source["menus"] if m["condition"] == condition)
                gold = question.gold_index
                ids = old["provenance"]["candidate_source_ids"]
                expected_ids = [[sid] for sid in ids]
                expected_ids.insert(gold, [])
                if (CROSSWALK[source["split"]] != role or source["group_id"] != prepared.groups[question.qid]
                        or question.options != [o["text"] for o in old["options"]]
                        or old["options"][gold]["id"] != old["gold_option_id"]
                        or expected_ids != list(map(list,prepared.menus[question.qid]["source_ids"]))
                        or len(ids) != 3 or not set(ids) <= donor_ids
                        or question.cumulative_prefixes != [p["text"] for p in source["prefixes"]]
                        or len(question.cumulative_prefixes) != 10):
                    raise ValueError("structural comparison against frozen baseline failed")
                slots[str(gold)] += 1
                distractors += len(ids)
                prefixes += len(question.cumulative_prefixes)
        if visited != questions.keys():
            raise ValueError("structural comparison is missing archive questions")
        counts[condition] = {"gold_slots_zero_based":dict(sorted(slots.items())),
                             "distractor_source_bindings":distractors,"prefixes_matched":prefixes}
    return {"classification":"private historical data; aggregate structural compatibility only",
            "archive_sha256":ARCHIVE_SHA256,"manifest_sha256":MANIFEST_SHA256,
            "outer_manifest_members_verified":files,
            "dataset_member_sha256":file_manifest["frozen/evaluator/main_dataset.json"]["sha256"],
            "reservoir_member_sha256":file_manifest["frozen/evaluator/reservoir.json"]["sha256"],
            "question_count":len(rows),"reserved_donors":len(reservoir),
            "target_component_source_ids":len(targets),"donor_component_source_ids":len(donor_members),
            "shared_component_source_ids":len(targets & donor_members),
            "source_splits":dict(Counter(q["split"] for q in rows)),
            "target_categories":dict(Counter(q["source"]["category"] for q in rows)),
            "donors_by_category":dict(Counter(r["source"]["category"] for r in reservoir)),
            "baseline_replay":counts,
            "mechanical_unique_classes":len(classes),"mechanical_normalized_labels":len(owners),
            "mechanical_normalized_label_conflicts":conflicts,
            "structured_rules_nonempty":{k:sum(bool(q["answer"][k]) for q in rows)
                                         for k in ("accepted","prompt","rejected")},
            "raw_answerlines_with_open_bracket":sum("[" in q["answer"]["raw"] for q in rows),
            "raw_answerlines_with_rule_words":{
                t:sum(bool(re.search(r"\b"+t+r"\b",q["answer"]["raw"],re.I)) for q in rows)
                for t in ("prompt","reject","before","after","until")},
            "source_review_status":dict(Counter(r["human_status"] for r in queue)),
            "source_answer_type_reviewed":sum(r["answer_type_reviewed"] is True for r in queue),
            "upstream_identity_override_groups":identity_groups,
            "upstream_type_candidates":type_review["counts"]["covered_menu_candidates"],
            "upstream_type_human_or_semantic_validation":type_review["human_or_semantic_validation"],
            "completed_reviewed_global_registry_present":False,
            "sidecar_semantic_review_completed":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.archive)
    publish_json(args.receipt,result)
    print(f"Structural comparison complete for {result['question_count']} questions. Receipt {digest(result)}")


if __name__ == "__main__":
    main()
