"""Source-bound oracle evidence must survive exact data/menu identity checks."""
from copy import deepcopy

import pytest

from qb_data.acl_oracle_review import review_template, verify_review
from qb_data.acl_protocol import ArmSpec, prepare
from scripts.acl_smoke_data import inputs


def reviewed_fixture():
    dataset = inputs()
    aliases = {"version":"test", "reviewer":"test", "classes":{
        q["source"]["canonical_answer"]:[] for q in dataset["questions"]}}
    spec = ArmSpec.from_dict({"version":1,"k":4,"seed":1,"menu":"baseline",
                              "reward":{"arm":"baseline"}})
    prepared = prepare(spec, dataset, aliases)
    review = review_template(prepared,dataset)
    assert all(all(d["judgment"] == "unresolved" for d in rec["options"])
               for rec in review["questions"].values())
    for q in dataset["questions"]:
        record = review["questions"][q["qid"]]
        record["reviewer"] = "synthetic test adjudicator"
        record["status"] = "reviewed"
        for item in record["options"]:
            item["judgment"] = ("accepted" if item["label"] == q["source"]["canonical_answer"]
                                else "rejected")
            item["rationale"] = "Synthetic fixture lookup for this unit test."
    return prepared,dataset,review


def test_complete_scoped_review_and_no_scientific_claim_for_fixture():
    prepared,dataset,review = reviewed_fixture()
    receipt = verify_review(prepared,dataset,review)
    assert receipt["reviewed_questions"] == len(dataset["questions"])
    assert receipt["evidence_scope"] == "synthetic_fixture"
    assert receipt["review_sha256"]


@pytest.mark.parametrize("mutation", ["missing","stale","unknown","incomplete","permuted",
                                      "ambiguous","silent-reviewer","unresolved","empty-reason"])
def test_scoped_review_fails_closed(mutation):
    prepared,dataset,review = reviewed_fixture()
    rec = next(iter(review["questions"].values()))
    if mutation == "missing":
        review["questions"].popitem()
    if mutation == "stale":
        dataset["questions"][0]["answer"]["raw"] = "newly disclosed conditional answer"
    if mutation == "unknown":
        review["questions"]["hidden"] = deepcopy(rec)
    if mutation == "incomplete":
        rec["options"].pop()
    if mutation == "permuted":
        rec["options"].reverse()
    if mutation == "ambiguous":
        rec["options"][1]["judgment"] = "accepted"
    if mutation == "silent-reviewer":
        rec["reviewer"] = ""
    if mutation == "unresolved":
        rec["status"] = "awaiting_adjudication"
    if mutation == "empty-reason":
        rec["options"][0]["rationale"] = " "
    with pytest.raises(ValueError):
        verify_review(prepared,dataset,review)


def test_contextual_alternate_or_prompt_alias_fails_before_play():
    prepared, dataset, _ = reviewed_fixture()
    registry = prepared.registry
    registry["classes"]["Venus"] = ["Evening Star"]
    for kind in ("accepted", "prompt"):
        modified = deepcopy(dataset)
        modified["questions"][0]["answer"][kind].append("Evening Star")
        with pytest.raises(ValueError, match="displayed option"):
            prepare(prepared.spec, modified, registry, contextual_review=True)


def test_review_checks_split_and_multiple_group_text():
    prepared, dataset, review = reviewed_fixture()
    prepared.questions["val"].append(prepared.questions["train"].pop())
    with pytest.raises(ValueError, match="split mismatch"):
        verify_review(prepared, dataset, review)
    a, b = dataset["questions"][:2]
    b["question"], b["prefixes"] = a["question"], a["prefixes"]
    with pytest.raises(ValueError, match="text overlap"):
        prepare(prepared.spec, dataset, prepared.registry)


def test_scientific_review_projects_complete_judgments_and_rejects_new_class():
    from dataclasses import replace
    from qb_data.acl_scientific_review import compile_review, extend_template
    prepared, dataset, _ = reviewed_fixture()
    reviewed = extend_template(prepared, dataset, prepared.registry)
    for q in dataset["questions"]:
        for key, decision in reviewed["questions"][q["qid"]]["classes"].items():
            decision.update(reviewer="test", rationale="fixture identity",
                            judgment="accepted" if key == q["source"]["canonical_answer"] else "rejected")
    sidecar, receipt = compile_review(prepared, dataset, prepared.registry, reviewed)
    assert receipt["reviewed_questions"] == len(dataset["questions"])
    assert len(sidecar["questions"]) == len(dataset["questions"])
    regenerated = prepare(replace(prepared.spec, menu="uniform_global"), dataset, prepared.registry)
    with pytest.raises(ValueError, match="no reviewed class decision"):
        compile_review(regenerated, dataset, prepared.registry, reviewed)

def test_direct_review_rejects_displayed_alias_on_legacy_prepared_menu():
    from qb_data.acl_protocol import digest
    prepared, dataset, review = reviewed_fixture()
    prepared.registry["classes"]["Venus"] = ["Evening Star"]
    prepared = prepare(prepared.spec, dataset, prepared.registry)
    source = dataset["questions"][0]
    source["answer"]["prompt"] = ["Evening Star"]
    # Recreate an older freeze whose menu preparation permitted this prompt class.
    prepared.identity.update(dataset_sha256=digest(dataset), oracle_sha256=digest(prepared.registry))
    updated = review_template(prepared, dataset)
    for qid, record in updated["questions"].items():
        record.update({field: review["questions"][qid][field] for field in ("status", "reviewer", "options")})
    with pytest.raises(ValueError, match="source-specific rule"):
        verify_review(prepared, dataset, updated)
