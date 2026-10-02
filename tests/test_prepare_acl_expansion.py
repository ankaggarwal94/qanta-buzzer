"""Focused regressions for leakage boundaries and deterministic dataset freeze."""
from itertools import combinations
import json
from pathlib import Path
import random

import pytest

from scripts.prepare_acl_expansion import (
    Config, allocate, allocate_bounded, assert_public_jobs, entity_key, extract_answer,
    group_candidates, load_candidates, near_duplicate_edges, prepare,
    primary_required_form, primary_identity_forms, word_prefixes,
)


def synthetic_question(index: int, count: int = 61) -> str:
    return " ".join(f"q{index}word{j}" for j in range(count)) + "?"


def source(index: int, *, category="History", year=2023, answer=None) -> dict:
    return {"_id": str(index), "question_sanitized": synthetic_question(index),
            "answer": answer or f"<u>Entity{index}</u>", "category": category,
            "year": year, "difficulty": 3 + index % 2, "standard": True,
            "set": {"name": "Synthetic fixture set", "year": year}}


def test_prefixes_are_ten_exact_endpoints_preserving_whitespace():
    text = "  " + synthetic_question(1).replace(" ", "  ") + "  "
    prefixes = word_prefixes(text)
    assert len(prefixes) == 10
    assert [p["token_count"] for p in prefixes] == [61 * k // 10 for k in range(1, 11)]
    assert all(text.startswith(p["text"]) for p in prefixes)
    assert prefixes[-1]["text"] == text
    assert prefixes[1]["fraction"] <= .2


def test_alias_parser_is_conservative_and_preserves_required_name_distinction():
    raw = "Winston <u>Churchill</u> [accept <u>Winston Spencer Churchill</u>] [accept Churchill before speech] [prompt on Winston]"
    primary, aliases, _ = extract_answer(raw)
    assert primary == "Winston Churchill"
    assert aliases == ["Winston Churchill", "Winston Spencer Churchill"]
    assert primary_required_form(raw) == "Churchill"
    assert "Churchill" not in aliases
    assert extract_answer("<u>Paris</u> [accept either Lutetia] [accept Paris for first sentence]")[1] == ["Paris"]
    assert extract_answer("<u>Romulus</u> and <u>Remus</u>")[0] is None
    assert extract_answer("<u>Danger</u><script>payload</script>")[0] is None


def test_near_duplicate_index_matches_bruteforce_including_boundary():
    rng = random.Random(5)
    values = [frozenset(rng.sample(range(30), rng.randrange(5, 25))) for _ in range(100)]
    values += [frozenset(range(20)), frozenset(range(19)), frozenset(range(17))]
    for threshold in (.5, .8, .85, 1.0):
        expected = {frozenset((a,b)) for a,b in combinations(range(len(values)),2)
                    if len(values[a]&values[b])/len(values[a]|values[b]) >= threshold}
        actual = {frozenset((a,b)) for a,b,_ in near_duplicate_edges(values, threshold)}
        assert actual == expected


def test_allocation_is_exact_bounded_and_handles_empty_strata():
    assert allocate(0,{}) == {}
    assert allocate(5,{"a":2,"b":4,"c":7}) == {"a":1,"b":1,"c":3}
    with pytest.raises(ValueError):
        allocate(20,{"a":3})


def test_source_weight_allocation_caps_unavailable_strata_and_redistributes():
    weights={"a":900,"b":90,"c":10}
    capacities={"a":10,"b":100,"c":100}
    result=allocate_bounded(100,weights,capacities)
    assert result == {"a":10,"b":81,"c":9}
    assert allocate_bounded(0,weights,capacities) == {"a":0,"b":0,"c":0}
    with pytest.raises(ValueError):
        allocate_bounded(300,weights,capacities)


def test_alias_bridge_propagates_prior_exclusion_before_component_selection(tmp_path):
    rows = [source(1,answer="<u>Alpha</u> [accept <u>Beta</u>]"),
            source(2,answer="<u>Beta</u> [accept <u>Gamma</u>]"), source(3)]
    path=tmp_path/"source.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    exclusions={"source_ids":[],"aliases":["Alpha"],"questions":[]}
    candidates,_=load_candidates(path, exclusions, Config())
    grouped,audit=group_candidates(candidates, exclusions, Config())
    assert [r["source"]["_id"] for r in grouped] == ["3"]
    assert audit["prior_near_duplicate_excluded_records"] == 2


def test_primary_required_form_blocks_freshness_but_not_grading_alias(tmp_path):
    row=source(1,answer="Winston <u>Churchill</u>")
    path=tmp_path/"source.jsonl"; path.write_text(json.dumps(row)+"\n")
    exclusions={"source_ids":[],"aliases":["Churchill"],"questions":[]}
    candidates,_=load_candidates(path, exclusions, Config())
    assert candidates[0]["aliases"] == ["Winston Churchill"]
    grouped,_=group_candidates(candidates,exclusions,Config())
    assert grouped == []


def test_explicit_either_underlined_portion_blocks_macmillan_without_grading_alias(tmp_path):
    raw = ("Maurice Harold <b><u>Macmillan</u></b>, 1st <b><u>Earl of Stockton</u></b> "
           "[accept either underlined portion; prompt on <b><u>Supermac</u></b>]")
    assert primary_identity_forms(raw) == ["Macmillan Earl of Stockton", "Macmillan", "Earl of Stockton"]
    assert extract_answer(raw)[1] == ["Maurice Harold Macmillan, 1st Earl of Stockton"]
    assert primary_identity_forms("<u>Winston</u> <u>Churchill</u>") == ["Winston Churchill"]
    assert primary_identity_forms("<u>Winston</u> <u>Churchill</u> [accept either before third clue]") == ["Winston Churchill"]
    assert primary_identity_forms("<u>Name</u>, <u>Title</u> [accept either]") == ["Name Title", "Name", "Title"]
    row=source(1, answer=raw)
    path=tmp_path/"source.jsonl"; path.write_text(json.dumps(row)+"\n")
    exclusions={"source_ids":[],"aliases":["Macmillan"],"questions":[]}
    candidates,_=load_candidates(path, exclusions, Config())
    grouped,_=group_candidates(candidates,exclusions,Config())
    assert grouped == []


def test_curated_identity_override_requires_exact_source_evidence_and_propagates_prior(tmp_path):
    rows=[source(1,answer="<u>Alpha</u>"),source(2,answer="<u>Beta</u>"),source(3)]
    path=tmp_path/"source.jsonl"; path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    exclusions={"source_ids":[],"aliases":["Alpha"],"questions":[]}
    candidates,_=load_candidates(path,exclusions,Config())
    overrides={"schema_version":"acl-identity-overrides-v1", "reviewer":"AI-assisted source answerline review; not human",
               "equivalences":[{"source_ids":["1","2"],"rationale":"Synthetic fixture equivalence for propagation test",
                   "evidence":[{"source_id":str(i),"answerline":rows[i-1]["answer"],"span":rows[i-1]["answer"]} for i in (1,2)]}]}
    grouped,audit=group_candidates(candidates,exclusions,Config(),overrides)
    assert [r["source"]["_id"] for r in grouped] == ["3"]
    assert len(audit["curated_identity_override_groups"]) == 1
    overrides["equivalences"][0]["evidence"][0]["span"] = "not in source"
    with pytest.raises(ValueError,match="exact source answerline/span"):
        group_candidates(candidates,exclusions,Config(),overrides)


def test_end_to_end_identity_safe_exact_stratification_and_create_once(tmp_path):
    rows=[source(i,category="History" if i%2 else "Science") for i in range(60)]
    rows += [source(i,year=2026) for i in range(60,66)]
    source_path=tmp_path/"source.jsonl"
    source_path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    exclusions_path=tmp_path/"exclusions.json"
    exclusions_path.write_text(json.dumps({"source_ids":["0"],"aliases":["Entity1"],"questions":[synthetic_question(2)]}))
    config=Config(main_count=30,calibration_count=6,selection_count=6,reservoir_per_category=3,holdout_count=5)
    out=tmp_path/"frozen"
    summary=prepare(source_path,exclusions_path,out,config)
    dataset=json.loads((out/"evaluator/main_dataset.json").read_text())
    jobs=json.loads((out/"public/main_jobs.json").read_text())["jobs"]
    assert summary["main"]["question_count"] == 30
    assert {k:v["question_count"] for k,v in summary["splits"].items()} == {"calibration":6,"selection":6,"test":18}
    assert summary["holdout"]["question_count"] == 5
    assert len(jobs) == 900
    assert_public_jobs(jobs)
    assert not {q["source"]["id"] for q in dataset["questions"]} & {"0","1","2"}
    assert all(q["group_id"].startswith("component:") for q in dataset["questions"])
    assert all("answer" not in json.loads(j["prompt"].rsplit("\n\n",1)[1]) for j in jobs)
    before=(out/"manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(source_path,exclusions_path,out,config)
    assert before == (out/"manifest.json").read_bytes()
    out2=tmp_path/"frozen2"
    prepare(source_path,exclusions_path,out2,config)
    assert (out2/"manifest.json").read_bytes() == before
