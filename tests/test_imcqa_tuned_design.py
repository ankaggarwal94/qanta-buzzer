"""Fresh sampling and public-only inference boundary regressions."""
from copy import deepcopy
import json

import pytest

from qb_data.jane_paired import build_jobs, validate_dataset
from scripts import imcqa_tuned_design as design
from scripts import prepare_acl_expansion as original
from scripts import prepare_imcqa_tuned as prep


def record(index, category="History", *, year=2023, question=None):
    text = question or " ".join(f"q{index}word{j}" for j in range(61))
    source = {"_id": str(index), "question_sanitized": text, "answer": f"<u>Entity{index}</u>",
              "category": category, "year": year, "difficulty": 3, "standard": True,
              "set": {"name": "Synthetic fixture set", "year": year}}
    return {"source": source, "canonical_answer": f"Entity{index}", "aliases": [f"Entity{index}"],
            "extraction_reason": "synthetic fixture", "alias_keys": [f"entity{index}"],
            "component_alias_keys": [f"entity{index}"], "component_source_ids": [str(index)],
            "group_id": f"synthetic:{index}", "split": "test", "freshness_blocked": False, "source_line": index}


def build_fixture():
    """Return a valid four-question public package and separate gold dataset."""
    rows = [record(i) for i in range(4)]
    reservoir = [record(i) for i in range(100, 104)]
    dataset = original.build_dataset(rows, reservoir, original.Config(), {"synthetic_fixture": True}, required_splits={"test"})
    dataset["evidence_scope"] = "synthetic_fixture"
    selection = {"salt": design.SALT, "manifest_sha256": "a" * 64,
                 "selected_qids": [q["qid"] for q in dataset["questions"]],
                 "selected_group_ids": {q["qid"]: q["group_id"] for q in dataset["questions"]},
                 "selected_normalized_text_sha256": {q["qid"]: design.base.sha(design.base.canonical(design.normalized_tokens(q["question"]))) for q in dataset["questions"]},
                 "qid_category": {q["qid"]: "History" for q in dataset["questions"]}, "outcomes_used_for_selection": False}
    jobs = build_jobs(dataset, required_splits={"test"})
    public = design.build_public_package(jobs, selection, source_input_sha256="b" * 64,
             main_dataset_sha256=design.base.sha(design.base.canonical(dataset)),
             policy_lock_sha256="c" * 64, sample_size_planning_sha256="d" * 64)
    return public, dataset


def test_plain_grid_is_complete_and_gold_stays_separate():
    public, dataset = build_fixture()
    rows = design.validate_public_package(public)
    assert len(rows) == 160
    assert {r["arm"] for r in rows} == {"plain"}
    assert all(r["execution"] == "new" and r["split"] == "test" for r in rows)
    assert all(set(r) == design.PUBLIC_JOB_KEYS for r in rows)
    assert "gold_option_id" not in json.dumps(public)
    assert {r["rotation"] for r in rows} == set(range(4))
    assert {r["prefix_id"] for r in rows} == set(design.PREFIX_IDS)
    assert dataset["questions"][0]["menus"][0]["gold_option_id"] == "A"


def test_existing_default_still_requires_all_three_splits():
    _, dataset = build_fixture()
    with pytest.raises(ValueError, match="required splits"):
        validate_dataset(dataset)
    with pytest.raises(ValueError, match="required splits"):
        build_jobs(dataset)
    validate_dataset(dataset, required_splits={"test"})
    with pytest.raises(ValueError, match="nonempty subset"):
        validate_dataset(dataset, required_splits={"unknown"})


@pytest.mark.parametrize("mutation", ["gold_field", "drop", "order", "prompt", "selection_hash", "group", "reward", "split", "model"])
def test_public_rejects_leaks_mutations_and_missing_cells(mutation):
    public, _ = build_fixture()
    if mutation == "gold_field":
        public["jobs"][0]["gold_option_id"] = "A"
    elif mutation == "drop":
        public["jobs"].pop()
    elif mutation == "order":
        public["jobs"][0], public["jobs"][1] = public["jobs"][1], public["jobs"][0]
    elif mutation == "prompt":
        public["jobs"][0]["prompt"] += "tampered"
        public["jobs"][0]["prompt_sha256"] = design.base.sha(public["jobs"][0]["prompt"].encode())
    elif mutation == "selection_hash":
        public["selection"]["manifest_sha256"] = "e" * 64
    elif mutation == "group":
        public["jobs"][0]["group_id"] = "other"
    elif mutation == "reward":
        public["jobs"][0]["reward"] = 99
    elif mutation == "split":
        public["jobs"][0]["split"] = "selection"
    else:
        public["model"]["revision"] = "e" * 40
    with pytest.raises(ValueError):
        design.validate_public_package(public)


def test_jaccard_exact_inclusive_boundary_and_unicode_normalization():
    assert design.near_duplicate(frozenset(range(4)), frozenset(range(5)))
    assert not design.near_duplicate(frozenset(range(3)), frozenset(range(5)))
    assert design.shingles("ＦOO bar café!!!") == design.shingles("foo BAR café")


def test_freshness_uses_component_members_aliases_and_all_prior_texts():
    old = record(0)
    prior = original.build_dataset([old], [record(i) for i in range(100, 104)], original.Config(), {"synthetic_fixture": True}, required_splits={"test"})
    members, alias, near, excluded, newyear = [record(i) for i in range(1, 6)]
    members["component_source_ids"].append("0")
    alias["component_alias_keys"].append("entity0")
    near["source"]["question_sanitized"] = old["source"]["question_sanitized"] + " extra"
    excluded["source"]["question_sanitized"] = record(77)["source"]["question_sanitized"]
    newyear["source"]["year"] = 2026
    duplicates = [record(6), record(7)]
    duplicates[1]["source"]["question_sanitized"] = duplicates[0]["source"]["question_sanitized"]
    candidates, audit = prep.fresh_candidates([old, members, alias, near, excluded, newyear, *duplicates], prior, [],
                {"source_ids": [], "aliases": [], "questions": [record(77)["source"]["question_sanitized"]]})
    assert len(candidates) == 1
    assert candidates[0]["source"]["_id"] in {"6", "7"}
    assert audit["rejection_counts"] == {"prior_or_reservoir_group": 1, "prior_component_source_id": 1,
          "prior_component_alias": 1, "outside_historical_year_frame": 1, "prior_text_jaccard_ge_0.8": 2,
          "fresh_text_jaccard_ge_0.8": 1}
    again, _ = prep.fresh_candidates(list(reversed([old, members, alias, near, excluded, newyear, *duplicates])), prior, [],
                {"source_ids": [], "aliases": [], "questions": [record(77)["source"]["question_sanitized"]]})
    assert [r["source"]["_id"] for r in again] == [r["source"]["_id"] for r in candidates]


def test_stratification_is_capacity_bounded_and_deterministic():
    candidates = [record(i, "History" if i < 2 else "Science") for i in range(8)]
    weights = {("History", 3): 9, ("Science", 3): 1}
    chosen, audit = prep.select_fresh(candidates, 4, weights)
    assert [r["source"]["category"] for r in chosen].count("History") == 2
    assert [r["selected"] for r in audit] == [2, 2]
    assert prep.select_fresh(list(reversed(candidates)), 4, weights)[0] == chosen
    with pytest.raises(ValueError):
        prep.select_fresh(candidates, 9, weights)


def test_reconstructed_identity_mismatch_fails_closed():
    _, prior = build_fixture()
    records = [record(i) for i in range(4)]
    prep.verify_original_partition(records, [], prior, [])
    changed = deepcopy(records)
    changed[0]["component_alias_keys"].append("hidden_alias")
    with pytest.raises(ValueError, match="identities differ"):
        prep.verify_original_partition(changed, [], prior, [])


def test_missing_or_changed_source_cannot_create_cohort(tmp_path):
    path = tmp_path / "substitute.jsonl"
    path.write_text("{}\n")
    out = tmp_path / "fresh"
    with pytest.raises(ValueError, match="hash-pinned input"):
        prep.prepare({"source_jsonl": path}, out)
    assert not out.exists()


def test_complete_preparation_on_hash_bound_synthetic_source(tmp_path, monkeypatch):
    source_path, exclusions_path, overrides_path = [tmp_path / n for n in ("source.jsonl", "exclusions.json", "overrides.json")]
    source_path.write_text("".join(json.dumps(record(i)["source"]) + "\n" for i in range(30)))
    prep.write_json(exclusions_path, {"source_ids": [], "aliases": [], "questions": []})
    prep.write_json(overrides_path, {"schema_version": "acl-identity-overrides-v1", "reviewer": "synthetic fixture", "equivalences": []})
    config = original.Config(main_count=6, calibration_count=2, selection_count=2, reservoir_per_category=4, holdout_count=0)
    original.prepare(source_path, exclusions_path, tmp_path / "old", config, overrides_path)
    paths = {"source_jsonl": source_path, "exclusions": exclusions_path, "identity_overrides": overrides_path,
             "prior_dataset": tmp_path / "old/evaluator/main_dataset.json", "reservoir": tmp_path / "old/evaluator/reservoir.json"}
    monkeypatch.setattr(prep, "INPUT_PINS", {k: original.file_sha(p) for k, p in paths.items()})
    monkeypatch.setattr(original, "Config", lambda: config)
    paths.update({k: tmp_path / (k + ".json") for k in prep.LOCK_INPUTS})
    prep.write_json(paths["policy_lock"], {"schema_version": "imcqa-independently-tuned-policies-v1", "status": "frozen_for_fresh_evaluation", "new_calibration_fits": 0})
    prep.write_json(paths["sample_size_planning"], {"planned_questions": 4})
    prep.write_json(paths["selection_receipt"], {"status": "complete", "validation_passed": True, "outputs_sha256": {
        "frozen_policies.json": original.file_sha(paths["policy_lock"]), "sample_size_planning.json": original.file_sha(paths["sample_size_planning"])}})
    summary = prep.prepare(paths, tmp_path / "new")
    assert summary["status"] == "prepared_not_scored" and summary["n_jobs"] == 160
    public = prep.read_json(tmp_path / "new/public/pilot.json")
    assert public["main_dataset_sha256"] == original.file_sha(tmp_path / "new/evaluator/main_dataset.json")
    assert public["policy_lock_sha256"] == original.file_sha(paths["policy_lock"])
    assert len(design.validate_public_package(public)) == 160
    prior = prep.read_json(paths["prior_dataset"])
    fresh = prep.read_json(tmp_path / "new/evaluator/main_dataset.json")
    assert not {q["group_id"] for q in prior["questions"]} & {q["group_id"] for q in fresh["questions"]}
    with pytest.raises(ValueError, match="already exists"):
        prep.prepare(paths, tmp_path / "new")
