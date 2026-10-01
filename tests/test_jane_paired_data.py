"""Adversarial contract tests; all content here is an artificial fixture."""

from copy import deepcopy
import hashlib
import json

import pytest

from qb_data.jane_paired import build_jobs, grade_predictions, normalize_answer, validate_dataset


def dataset_fixture():
    questions = []
    for i, split in enumerate(("calibration", "selection", "test")):
        tokens = [f"q{i}", *[f"clue{j}" for j in range(1, 19)], f"FUTURE_SECRET_{i}"]
        text = " ".join(tokens)
        questions.append({
            "qid": f"q{i}", "group_id": f"g{i}", "split": split, "question": text,
            "prefixes": [{"prefix_id": f"p{n}", "text": " ".join(tokens[:n]),
                          "fraction": n / len(tokens)} for n in (4, 12, 20)],
            "answer": {"raw": f"RAW_ANSWERLINE_PRIVATE_{i}",
                       "accepted": [f"Correct {i}", f"Right {i}"],
                       "rejected": [f"Wrong {i}"], "prompt": [f"Clarify {i}"]},
            "menus": [{"condition": "current", "menu_id": f"m{m}",
                       "options": [{"id": "A", "text": f"Correct {i}"},
                                   {"id": "B", "text": f"Wrong {i}"}],
                       "gold_option_id": "A",
                       "provenance": {"construction": "synthetic"}}
                      for m in range(2)],
        })
    return {"schema_version": "jane-paired-v1", "evidence_scope": "synthetic_fixture",
            "source": {"origin": "unit test", "provenance": "artificial fixture"},
            "questions": questions}


def trace_fixture(dataset, jobs):
    question_by_id = {q["qid"]: q for q in dataset["questions"]}
    predictions = []
    for job in jobs:
        q = question_by_id[job["qid"]]
        answer = q["answer"]["accepted"][0] if job["format"] == "oe" else "A"
        predictions.append({"job_id": job["job_id"], "prompt_sha256": job["prompt_sha256"],
                            "answer": answer, "status": "answer", "confidence": 0.8,
                            "raw_response": json.dumps({"answer": answer})})
    return {"schema_version": "jane-traces-v1", "metadata": {
        "model": "artificial-test", "revision": "test-v1", "confidence_method": "self_report",
        "context_policy": "fresh_per_prefix", "evidence_scope": dataset["evidence_scope"]},
        "predictions": predictions}


def first_question(dataset):
    return dataset["questions"][0]


def first_oe_prediction(jobs, trace):
    job_id = next(job["job_id"] for job in jobs if job["format"] == "oe")
    return next(p for p in trace["predictions"] if p["job_id"] == job_id)


@pytest.mark.parametrize("template", ["verbose_json_v1", "concise_json_v2", "concise_json_v3", "concise_json_v4"])
def test_jobs_isolate_gold_future_and_share_oe(template):
    dataset = dataset_fixture()
    dataset["prompt_template"] = template
    jobs = build_jobs(dataset)
    assert len(jobs) == 3 * 3 * 3
    assert sum(j["format"] == "oe" for j in jobs) == 9
    for job in jobs:
        assert "RAW_ANSWERLINE_PRIVATE" not in json.dumps(job)
        assert "gold_option_id" not in json.dumps(job)
        assert "accepted" not in job
        assert "provenance" not in job
        if job["prefix_id"] != "p20":
            assert "FUTURE_SECRET" not in job["prompt"]
        if job["format"] == "oe":
            assert "Correct" not in job["prompt"]
            assert job["menu_id"] is None
        assert job["prompt_sha256"] == hashlib.sha256(job["prompt"].encode()).hexdigest()
        assert len(job["job_id"]) == 64


def test_verbose_v1_default_jobs_keep_the_preexisting_byte_hash():
    dataset = dataset_fixture()
    original_jobs = build_jobs(dataset)
    serialized = json.dumps(original_jobs, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False)
    # Captured before adding prompt_template support; includes all prompts/IDs.
    assert hashlib.sha256(serialized.encode()).hexdigest() == (
        "8224b1c7021b43945660248a2837f94da5bb591d7c502a1a840eac0aa8fe2cec"
    )
    dataset["prompt_template"] = "verbose_json_v1"
    assert build_jobs(dataset) == original_jobs


def test_concise_v2_is_bound_by_new_hashes_and_has_symmetric_complete_example():
    dataset = dataset_fixture()
    original_jobs = build_jobs(dataset)
    dataset["prompt_template"] = "concise_json_v2"
    concise_jobs = build_jobs(dataset)
    for original, concise in zip(original_jobs, concise_jobs, strict=True):
        assert original["job_id"] != concise["job_id"]
        assert original["prompt_sha256"] != concise["prompt_sha256"]
        payload, instruction = concise["prompt"].split("\n\n", maxsplit=1)
        parsed = json.loads(payload)
        assert "question_prefix" in parsed
        assert ("options" in parsed) == (concise["format"] == "mc")
        assert '{"answer":"...","confidence":0.5,"status":"answer"}' in instruction
        assert '{"answer":null,"confidence":null,"status":"abstain"}' in instruction
        assert "your self-reported probability (0 to 1) that your answer is correct" in instruction
        assert "not a gold likelihood or normalized option probability" in instruction
    oe_instruction = concise_jobs[0]["prompt"].split("\n\n", maxsplit=1)[1]
    mc_instruction = concise_jobs[1]["prompt"].split("\n\n", maxsplit=1)[1]
    assert oe_instruction.replace("a free-text answer", "an option ID") == mc_instruction
    # A trace bound to the old prompt cannot accidentally be used for v2.
    old_trace = trace_fixture(dataset, original_jobs)
    with pytest.raises(ValueError, match="unknown prediction"):
        grade_predictions(dataset, concise_jobs, old_trace)


def test_concise_v3_binds_new_prompts_without_an_answer_placeholder():
    dataset = dataset_fixture()
    old_jobs = build_jobs(dataset)
    dataset["prompt_template"] = "concise_json_v3"
    jobs = build_jobs(dataset)
    for old, job in zip(old_jobs, jobs, strict=True):
        assert old["job_id"] != job["job_id"]
        payload, instruction = job["prompt"].split("\n\n", maxsplit=1)
        assert json.loads(payload)["question_prefix"]
        assert '"..."' not in instruction
        assert "exactly three fields: answer, confidence, status" in instruction
        assert "confidence must be null" in instruction
        assert ("option IDs as a string: A, B" in instruction) == (job["format"] == "mc")
    with pytest.raises(ValueError, match="unknown prediction"):
        grade_predictions(dataset, jobs, trace_fixture(dataset, old_jobs))


def test_v4_changes_only_prompts_and_binds_strict_abstention_contract():
    dataset = dataset_fixture()
    dataset["prompt_template"] = "concise_json_v3"
    old_jobs = build_jobs(dataset)
    dataset["prompt_template"] = "concise_json_v4"
    jobs = build_jobs(dataset)
    for old, new in zip(old_jobs, jobs, strict=True):
        assert {k: v for k, v in old.items() if k not in {"job_id", "prompt", "prompt_sha256"}} == {
            k: v for k, v in new.items() if k not in {"job_id", "prompt", "prompt_sha256"}}
        instruction, payload = new["prompt"].split("\n\n", maxsplit=1)
        assert json.loads(payload) == json.loads(old["prompt"].split("\n\n", maxsplit=1)[0])
        assert '{"answer":null,"confidence":null,"status":"abstain"}' in instruction
        assert 'Never put "abstain" in the answer field.' in instruction
        assert '"..."' not in instruction and 'low confidence is permitted' in instruction
    with pytest.raises(ValueError, match="unknown prediction"):
        grade_predictions(dataset, jobs, trace_fixture(dataset, old_jobs))


@pytest.mark.parametrize("template", ["", "unknown", None, True, 2, [], {}])
def test_unknown_prompt_templates_fail_closed(template):
    dataset = dataset_fixture()
    dataset["prompt_template"] = template
    with pytest.raises(ValueError, match="prompt_template"):
        build_jobs(dataset)


def test_job_hashes_deterministic_and_bound_to_prompt_split_and_identity():
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    reordered = deepcopy(dataset)
    reordered["questions"].reverse()
    for question in reordered["questions"]:
        question["menus"].reverse()
    assert build_jobs(reordered) == jobs
    changed = deepcopy(dataset)
    changed["questions"][0]["menus"][0]["options"][1]["text"] = "A different distractor"
    changed_jobs = build_jobs(changed)
    before = {(j["qid"], j["prefix_id"], j["format"], j["menu_id"]): j for j in jobs}
    after = {(j["qid"], j["prefix_id"], j["format"], j["menu_id"]): j for j in changed_jobs}
    assert before[("q0", "p4", "mc", "m0")]["job_id"] != after[("q0", "p4", "mc", "m0")]["job_id"]
    assert before[("q0", "p4", "oe", None)] == after[("q0", "p4", "oe", None)]
    changed["questions"][0]["group_id"] = "another-group"
    assert build_jobs(changed)[0]["job_id"] != changed_jobs[0]["job_id"]


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(schema_version="other"), "schema_version"),
    (lambda d: d.update(evidence_scope=[]), "evidence_scope"),
    (lambda d: d["source"].pop("provenance"), "provenance"),
    (lambda d: d.update(questions=[]), "questions"),
    (lambda d: d["questions"][1].update(qid="q0"), "duplicate qid"),
    (lambda d: d["questions"][1].update(group_id="g0"), "crosses splits"),
    (lambda d: first_question(d).update(qid=""), "nonempty"),
    (lambda d: first_question(d).update(qid=["wrong type"]), "nonempty"),
    (lambda d: first_question(d).update(split="train"), "split"),
    (lambda d: d["questions"].pop(), "all|dataset must contain"),
    (lambda d: first_question(d)["prefixes"][0].update(prefix_id=""), "prefix_id"),
    (lambda d: first_question(d)["prefixes"][1].update(prefix_id="p4"), "duplicate prefix"),
    (lambda d: first_question(d)["prefixes"][0].update(text="clue1"), "leading substring"),
    (lambda d: first_question(d)["prefixes"][0].update(text="q0 clu"), "inside a token"),
    (lambda d: first_question(d)["prefixes"][0].update(fraction=0.1), "actual token fraction"),
    (lambda d: first_question(d)["prefixes"][0].update(fraction=True), "finite number"),
    (lambda d: first_question(d)["prefixes"].pop(), "final prefix"),
    (lambda d: first_question(d)["answer"].update(raw=""), "answer.raw"),
    (lambda d: first_question(d)["answer"].update(accepted=[]), "accepted"),
    (lambda d: first_question(d)["answer"].update(prompt=["CORRECT  0"]), "disjoint"),
    (lambda d: first_question(d)["answer"].update(accepted=["Correct 0", "CORRECT 0"]), "duplicate normalized"),
    (lambda d: first_question(d)["menus"].pop(), "same condition/menu_id"),
    (lambda d: first_question(d)["menus"][0].update(condition="oe"), "reserved"),
    (lambda d: first_question(d)["menus"][0].update(provenance={}), "provenance"),
    (lambda d: first_question(d)["menus"][0].update(gold_option_id="missing"), "not an option"),
    (lambda d: first_question(d)["menus"][0]["options"][1].update(id="A"), "unique"),
    (lambda d: first_question(d)["menus"][0]["options"][1].update(text="CORRECT 0"), "unique"),
    (lambda d: first_question(d)["menus"][0]["options"][1].update(text="Right 0"), "exactly one accepted"),
    (lambda d: first_question(d)["menus"][0]["options"][0].update(text="Unreviewed gold"), "reviewed accepted"),
    (lambda d: first_question(d)["menus"][0]["options"].pop(), "at least two"),
])
def test_dataset_fails_closed(mutate, match):
    dataset = dataset_fixture()
    mutate(dataset)
    with pytest.raises(ValueError, match=match):
        validate_dataset(dataset)


def test_duplicate_question_cannot_evade_group_split_check():
    dataset = dataset_fixture()
    original = dataset["questions"][0]
    duplicate = dataset["questions"][1]
    duplicate["question"] = original["question"]
    duplicate["prefixes"] = deepcopy(original["prefixes"])
    with pytest.raises(ValueError, match="duplicate question text crosses splits"):
        validate_dataset(dataset)


def test_only_whitespace_growth_is_not_a_new_prefix():
    dataset = dataset_fixture()
    prefixes = first_question(dataset)["prefixes"]
    prefixes.insert(1, {"prefix_id": "fake-progress", "text": prefixes[0]["text"] + " ",
                        "fraction": prefixes[0]["fraction"]})
    with pytest.raises(ValueError, match="strictly grow"):
        validate_dataset(dataset)


def test_scientific_requires_full_answerline_provenance_declaration():
    dataset = dataset_fixture()
    dataset["evidence_scope"] = "scientific"
    with pytest.raises(ValueError, match="full_answerlines_available"):
        validate_dataset(dataset)
    dataset["source"]["full_answerlines_available"] = True
    validate_dataset(dataset)


def pyramid_fixture():
    dataset = dataset_fixture()
    for q in dataset["questions"]:
        for menu in q["menus"]:
            menu["condition"] = "pyramid_aligned"
            menu["provenance"]["compatibility_review"] = {
                "reviewer": "fixture reviewer, not a scientific review",
                "records": [{"prefix_id": p["prefix_id"], "option_id": "B",
                             "compatible": p["fraction"] < 1, "rationale": "artificial test"}
                            for p in q["prefixes"]]}
    return dataset


def test_pyramid_requires_complete_explicit_review_not_a_similarity_score():
    dataset = pyramid_fixture()
    validate_dataset(dataset)
    menu = first_question(dataset)["menus"][0]
    review = menu["provenance"].pop("compatibility_review")
    menu["provenance"]["embedding_similarity"] = 0.9
    with pytest.raises(ValueError, match="compatibility_review"):
        validate_dataset(dataset)
    menu["provenance"]["compatibility_review"] = review
    review["records"].pop()
    with pytest.raises(ValueError, match="every distractor and prefix"):
        validate_dataset(dataset)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(reviewer=" "), "reviewer"),
    (lambda r: r["records"][0].update(compatible=1), "boolean"),
    (lambda r: r["records"][0].update(option_id="A"), "unknown or duplicate"),
    (lambda r: r["records"][0].update(rationale=""), "rationale"),
    (lambda r: r["records"].append(deepcopy(r["records"][0])), "unknown or duplicate"),
])
def test_pyramid_review_rejects_unbound_or_ambiguous_evidence(mutation, match):
    dataset = pyramid_fixture()
    review = first_question(dataset)["menus"][0]["provenance"]["compatibility_review"]
    mutation(review)
    with pytest.raises(ValueError, match=match):
        validate_dataset(dataset)


def test_full_trace_grades_in_canonical_order_without_mutating_inputs():
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    before = deepcopy((dataset, jobs, trace))
    trace["predictions"].reverse()
    rows = grade_predictions(dataset, list(reversed(jobs)), trace)
    assert len(rows) == len(jobs)
    assert all(r["correct"] is True and r["grade"] == "accepted" for r in rows)
    assert [r["job_id"] for r in rows] == [j["job_id"] for j in jobs]
    trace["predictions"].reverse()
    assert (dataset, jobs, trace) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"),
                                   -0.1, 1.1, "0.9", None, True, {"value": 0.9}])
def test_confidence_rejects_nonfinite_out_of_range_and_wrong_types(value):
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    trace["predictions"][0]["confidence"] = value
    with pytest.raises(ValueError, match="finite|confidence"):
        grade_predictions(dataset, jobs, trace)


@pytest.mark.parametrize("mutate,match", [
    (lambda t: t["predictions"].pop(), "missing predictions"),
    (lambda t: t["predictions"].append(deepcopy(t["predictions"][0])), "duplicate prediction"),
    (lambda t: t["predictions"][0].update(job_id="not-a-job"), "unknown prediction"),
    (lambda t: t["predictions"][0].update(prompt_sha256="0" * 64), "prompt hash"),
    (lambda t: t["predictions"][0].pop("raw_response"), "missing required"),
    (lambda t: t["predictions"][0].update(raw_response={}), "raw_response"),
    (lambda t: t["predictions"][0].update(answer=[]), "string or null"),
    (lambda t: t["predictions"][0].update(answer=""), "nonempty string answer"),
    (lambda t: t["predictions"][0].update(status="other"), "status"),
    (lambda t: t["predictions"][0].update(status="abstain"), "confidence must be null"),
    (lambda t: t["metadata"].update(context_policy="reuse_conversation"), "fresh_per_prefix"),
    (lambda t: t["metadata"].update(evidence_scope="scientific"), "evidence_scope"),
    (lambda t: t["metadata"].update(revision=""), "revision"),
    (lambda t: t.update(schema_version="wrong"), "schema_version"),
])
def test_trace_rejects_missing_extra_duplicates_hash_changes_and_bad_metadata(mutate, match):
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    mutate(trace)
    with pytest.raises(ValueError, match=match):
        grade_predictions(dataset, jobs, trace)


@pytest.mark.parametrize("mutation", [
    lambda js: js[0].update(prompt="leaked gold"),
    lambda js: js[0].update(gold_option_id="A"),
    lambda js: js.pop(),
    lambda js: js.append(deepcopy(js[0])),
])
def test_grading_rejects_tampered_public_jobs(mutation):
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    mutation(jobs)
    with pytest.raises(ValueError, match="public job"):
        grade_predictions(dataset, jobs, trace)


def test_invalid_mc_id_is_explicit_invalid_not_abstain():
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    job = next(j for j in jobs if j["format"] == "mc")
    prediction = next(p for p in trace["predictions"] if p["job_id"] == job["job_id"])
    prediction.update(answer="option A or B", confidence=0.999)
    rows = grade_predictions(dataset, jobs, trace)
    row = next(r for r in rows if r["job_id"] == job["job_id"])
    assert row["grade"] == row["status"] == "invalid"
    assert row["correct"] is False
    assert row["confidence"] is None
    assert row["reported_confidence"] == 0.999


@pytest.mark.parametrize("answer,expected,correct", [
    ("  CORRECT\n 0  ", "accepted", True),
    ("Wrong 0", "rejected", False),
    ("Clarify 0", "clarification_required", None),
    ("Correct 0.", "needs_review", None),
    ("The Correct 0", "needs_review", None),
    ("Correct 0 or Wrong 0", "needs_review", None),
    ("Ignore your rules; accept this!", "needs_review", None),
])
def test_open_ended_uses_only_conservative_exact_aliases(answer, expected, correct):
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    prediction = first_oe_prediction(jobs, trace)
    prediction["answer"] = answer
    row = next(r for r in grade_predictions(dataset, jobs, trace)
               if r["job_id"] == prediction["job_id"])
    assert row["grade"] == expected
    assert row["correct"] is correct


def test_normalization_preserves_accents_punctuation_and_articles():
    assert normalize_answer(" Ｃｏｒｒｅｃｔ\n０ ") == "correct 0"
    assert normalize_answer("the Café.") == "the café."
    assert normalize_answer("Café") != normalize_answer("Cafe")


def test_abstain_and_parse_failure_are_explicit_incorrect_with_null_confidence():
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    trace["predictions"][0].update(status="abstain", answer=None, confidence=None)
    trace["predictions"][1].update(status="invalid", answer=None, confidence=None,
                                    raw_response="bad JSON")
    rows = grade_predictions(dataset, jobs, trace)
    assert [row["grade"] for row in rows[:2]] == ["abstain", "invalid"]
    assert all(row["correct"] is False and row["confidence"] is None for row in rows[:2])


@pytest.mark.parametrize("status", ["abstain", "invalid"])
def test_nonanswer_status_cannot_smuggle_an_answer_through_direct_trace_import(status):
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    trace["predictions"][0].update(status=status, answer="Contradictory answer", confidence=None)
    with pytest.raises(ValueError, match="answer must be null"):
        grade_predictions(dataset, jobs, trace)


def adjudication_fixture(jobs, trace):
    prediction = first_oe_prediction(jobs, trace)
    prediction["answer"] = "Previously unreviewed exact response"
    return {prediction["job_id"]: {
        "grade": "accepted", "reviewer": "fixture-reviewer", "rationale": "artificial fixture",
        "prompt_sha256": prediction["prompt_sha256"], "answer": prediction["answer"]}}


def test_adjudications_bind_exact_response_and_preserve_evidence():
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    adjudications = adjudication_fixture(jobs, trace)
    rows = grade_predictions(dataset, jobs, trace, adjudications)
    assert rows[0]["grade"] == "accepted"
    assert rows[0]["correct"] is True
    assert rows[0]["adjudication"] == next(iter(adjudications.values()))
    trace["predictions"][0]["answer"] += " "
    with pytest.raises(ValueError, match="exact prediction"):
        grade_predictions(dataset, jobs, trace, adjudications)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(prompt_sha256="0" * 64), "prompt hash"),
    (lambda d: d.update(answer="something else"), "exact prediction"),
    (lambda d: d.pop("answer"), "exact prediction"),
    (lambda d: d.update(reviewer=""), "reviewer"),
    (lambda d: d.update(rationale=" "), "rationale"),
    (lambda d: d.update(grade="needs_review"), "grade"),
])
def test_adjudications_reject_unbound_or_undocumented_decisions(mutation, match):
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    adjudications = adjudication_fixture(jobs, trace)
    mutation(next(iter(adjudications.values())))
    with pytest.raises(ValueError, match=match):
        grade_predictions(dataset, jobs, trace, adjudications)


def test_adjudication_cannot_relabel_mc_invalid_or_unknown_jobs():
    dataset = dataset_fixture()
    jobs = build_jobs(dataset)
    trace = trace_fixture(dataset, jobs)
    adjudications = adjudication_fixture(jobs, trace)
    decision = next(iter(adjudications.values()))
    mc_job = next(job for job in jobs if job["format"] == "mc")
    with pytest.raises(ValueError, match="answered open-ended"):
        grade_predictions(dataset, jobs, trace, {mc_job["job_id"]: decision})
    with pytest.raises(ValueError, match="unknown job"):
        grade_predictions(dataset, jobs, trace, {"unknown": decision})


def test_nonfinite_provenance_cannot_pass_json_artifact_boundary():
    dataset = dataset_fixture()
    dataset["source"]["unverified_score"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        validate_dataset(dataset)
