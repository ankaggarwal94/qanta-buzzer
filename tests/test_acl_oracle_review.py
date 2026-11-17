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
