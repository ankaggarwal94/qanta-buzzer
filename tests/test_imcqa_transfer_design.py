"""Meaningful integrity and separation gates for frozen-policy transfer inputs."""
from copy import deepcopy

import pytest

from scripts import acl_option_scoring as base
from scripts import imcqa_transfer_design as design
from scripts import imcqa_protocol_design as old


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(design, "N_QUESTIONS", 3)
    questions, jobs = [], []
    groups = {"calibration": [f"cal-{i}" for i in range(100)],
              "selection": [f"old-{i}" for i in range(100)]}
    qspecs = [(q, s) for s, qs in groups.items() for q in qs]
    qspecs += [(f"new-{i}", "selection") for i in range(4)] + [("test-item", "test")]
    options = [{"id": k, "text": v} for k, v in zip("ABCD", ("One", "Two", "Three", "Four"))]
    for qid, split in qspecs:
        words = [f"token{qid.replace('-', '')}w{n}" for n in range(30)]
        text = " ".join(words)
        questions.append({"qid": qid, "group_id": f"group:{qid}", "split": split, "question": text,
                          "source": {"category": "History"}, "answer": "EVALUATOR_ONLY_CORRECT_ANSWER"})
        if qid.startswith("new-"):
            for condition in design.CONDITIONS:
                for rn, prefix in enumerate(design.PREFIX_IDS, 1):
                    payload = {"question_prefix": " ".join(words[:6*rn]), "options": options}
                    prompt = "Frozen original prompt.\n\n" + base.canonical(payload).decode().rstrip()
                    jobs.append({"job_id": f"{qid}:{condition}:{prefix}", "format": "mc", "qid": qid,
                                 "group_id": f"group:{qid}", "split": split, "condition": condition,
                                 "menu_id": f"menu:{qid}:{condition}", "prefix_id": prefix,
                                 "fraction": rn/5, "prompt": prompt, "prompt_sha256": base.sha(prompt.encode())})
    fitted = {"parameters": []}
    for condition in design.CONDITIONS:
        intercept, slope, threshold, fixed = design.POLICY_VALUES[condition]
        fitted["parameters"].append({"answer_source": "plain", "condition": condition,
            "intercept": intercept, "slope": slope, "feature_clip": [1e-6, .999999],
            "selected_threshold": threshold, "selected_fixed_policy": fixed,
            "fit_qids": groups["calibration"][:20], "n_fit_questions": 20,
            "fit_split": "calibration", "method": "monotone_regularized_logistic_correctness"})
    result = {"main_jobs": {"jobs": jobs}, "main_dataset": {"questions": questions},
              "prior_public": {"selection": {"selected_qids": groups}}, "fitted_parameters": fitted}
    monkeypatch.setattr(design, "CANONICAL_SOURCE_SHA256", {k: base.sha(base.canonical(v)) for k, v in result.items()})
    return result


@pytest.fixture
def package(inputs):
    return design.build_public_package(inputs["main_jobs"], inputs["main_dataset"],
                                       inputs["prior_public"], inputs["fitted_parameters"])


def test_selection_is_frozen_and_outcome_blind(inputs, package):
    chosen = package["selection"]["selected_qids"]["selection"]
    assert chosen == sorted((f"new-{i}" for i in range(4)), key=lambda q: (design.rank(q), q))[:3]
    assert not set(chosen) & set(package["selection"]["manifest"]["excluded_prior_qids"])
    changed = deepcopy(inputs["main_dataset"])
    for q in changed["questions"]:
        q["answer"] = "OTHER PRIVATE GOLD"
    assert design.select_questions(changed, inputs["prior_public"]) == package["selection"]["manifest"]


def test_public_has_exact_grid_and_no_gold(package):
    assert len(package["jobs"]) == 3 * 80
    assert b"EVALUATOR_ONLY_CORRECT_ANSWER" not in base.canonical(package)
    assert all(set(j) == old.PUBLIC_JOB_KEYS for j in package["jobs"])
    assert all(j["split"] == "selection" and j["execution"] == "new" and j["source_score_id"] is None for j in package["jobs"])
    assert len({j["score_id"] for j in package["jobs"]}) == 240
    for plain, wait in zip(package["jobs"][::2], package["jobs"][1::2]):
        assert old._payload(plain["prompt"]) == old._payload(wait["prompt"])
        assert (plain["arm"], wait["arm"]) == ("plain", "wait")
    assert design.validate_public_package(package) == package["jobs"]


def test_inclusive_jaccard_boundary_and_normalization():
    ten = frozenset((str(i),) for i in range(10))
    assert design.near_duplicate(ten, frozenset((str(i),) for i in range(8)))
    assert not design.near_duplicate(ten, frozenset((str(i),) for i in range(7)))
    assert design.shingles("Ａlpha, BETA_gamma delta epsilon.") == design.shingles("alpha beta gamma DELTA EPSILON")


def test_full_question_near_duplicate_and_group_are_excluded(inputs):
    dataset = deepcopy(inputs["main_dataset"])
    rows = {q["qid"]: q for q in dataset["questions"]}
    rows["new-0"]["question"] = rows["old-0"]["question"].upper()
    manifest = design.select_questions(dataset, inputs["prior_public"])
    assert "new-0" not in manifest["selected_qids"]
    if design.rank("new-0") < max(design.rank(q) for q in manifest["selected_qids"]):
        assert any(r["qid"] == "new-0" and r["reference_qid"] == "old-0" for r in manifest["skipped"])
    rows["new-0"]["group_id"] = rows["old-0"]["group_id"]
    grouped = design.select_questions(dataset, inputs["prior_public"])
    assert "new-0" not in grouped["eligible_ranked_qids"]


@pytest.mark.parametrize("mutation", ["gold", "missing", "duplicate", "split", "group", "mapping", "prompt", "round", "reuse", "threshold", "fit_qid", "manifest", "source_hash", "full_text_hash"])
def test_fail_closed_on_scientific_mutation(package, mutation):
    p = deepcopy(package); j = p["jobs"][0]
    if mutation == "gold": j["gold_option_id"] = "A"
    if mutation == "missing": p["jobs"].pop()
    if mutation == "duplicate": p["jobs"][1] = deepcopy(j)
    if mutation == "split": j["split"] = "test"
    if mutation == "group": j["group_id"] = "group:old-0"
    if mutation == "mapping": j["option_source_ids"] = old.mapping(1)
    if mutation == "prompt":
        j["prompt"] += " Change the instruction."; j["prompt_sha256"] = base.sha(j["prompt"].encode())
    if mutation == "round": j["round"] = 2
    if mutation == "reuse": j["execution"] = "reuse"
    if mutation == "threshold": p["frozen_policy"]["parameters"][0]["selected_threshold"] = .5
    if mutation == "fit_qid": p["frozen_policy"]["parameters"][0]["fit_qids"][0] = j["qid"]
    if mutation == "manifest": p["selection"]["manifest"]["selected_qids"].reverse()
    if mutation == "source_hash": p["source_input_sha256"] = "0" * 64
    if mutation == "full_text_hash":
        p["selection"]["manifest"]["selected_normalized_text_sha256"][j["qid"]] = "0" * 64
        p["selection"]["manifest_sha256"] = base.sha(base.canonical(p["selection"]["manifest"]))
    with pytest.raises(ValueError):
        design.validate_public_package(p)


def test_source_content_binding_rejects_even_unselected_mutation(inputs):
    dataset = deepcopy(inputs["main_dataset"])
    dataset["questions"][0]["question"] += " changed"
    with pytest.raises(ValueError, match="hash-pinned main_dataset"):
        design.build_public_package(inputs["main_jobs"], dataset, inputs["prior_public"], inputs["fitted_parameters"])


def test_frozen_policy_has_no_refitting(package):
    assert package["frozen_policy"]["refitting_permitted"] is False
    for f in package["frozen_policy"]["parameters"]:
        assert (f["intercept"], f["slope"], f["selected_threshold"], f["selected_fixed_policy"]) == design.POLICY_VALUES[f["condition"]]
