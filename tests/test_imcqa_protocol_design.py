"""Leakage, exact-source reuse, factorial identity and synthetic decision gates."""
from collections import Counter
from copy import deepcopy

import pytest

from scripts import imcqa_protocol_design as design
from scripts import imcqa_wait_scoring as prior


def fixture_prior():
    jobs, categories = [], {}
    for split in ("calibration", "selection", "test"):
        for number in range(20):
            qid = f"{split}-{number}"
            categories[qid] = "History" if number % 2 else "Science"
            for condition in prior.CONDITIONS:
                for round_number, prefix_id in enumerate(prior.PREFIX_IDS, 1):
                    payload = {"question_prefix": "clue " * round_number,
                               "options": [{"id": label, "text": name} for label, name in
                                           zip("ABCD", ["Mercury", "Venus", "Earth", "Mars"])]}
                    text = "original frozen instructions\n\n" + prior.base.canonical(payload).decode().strip()
                    jobs.append({"job_id": f"{qid}-{condition}-{prefix_id}", "qid": qid, "group_id": qid,
                                 "split": split, "format": "mc", "condition": condition, "menu_id": "fixed_1",
                                 "prefix_id": prefix_id, "fraction": round_number / 5,
                                 "prompt": text, "prompt_sha256": prior.base.sha(text.encode())})
    return prior.build_public_package({"jobs": jobs}, source_input_sha256=prior.SOURCE_INPUT_SHA256,
                                      category_map=categories, n_per_split=20, n_diagnostic_per_split=20)


@pytest.fixture(scope="module")
def frozen_prior():
    return fixture_prior()


@pytest.fixture
def package(monkeypatch, frozen_prior):
    checksum = prior.base.sha(prior.base.canonical(frozen_prior))
    monkeypatch.setattr(design, "PRIOR_PUBLIC_SHA256", checksum)
    return design.build_public_package(frozen_prior, prior_public_sha256=checksum)


def test_full_factorial_has_exact_counts_and_no_test_or_gold(package):
    assert len(package["jobs"]) == 5232
    assert Counter(row["block"] for row in package["jobs"]) == {"factorial": 4800, "label_swap": 400, "comprehension": 32}
    assert Counter(row["execution"] for row in package["jobs"]) == {"new": 4032, "reuse": 1200}
    assert Counter(row["split"] for row in package["jobs"]) == {"calibration": 2600, "selection": 2600, "synthetic": 32}
    assert all(set(row) == design.PUBLIC_JOB_KEYS for row in package["jobs"])
    assert len({row["qid"] for row in package["jobs"] if row["block"] != "comprehension"}) == 40
    assert len({row["score_id"] for row in package["jobs"]}) == 5232


def test_every_reused_prompt_mapping_is_byte_identical(package, frozen_prior):
    lookup = {row["score_id"]: row for row in frozen_prior["jobs"]}
    reused = [row for row in package["jobs"] if row["execution"] == "reuse"]
    assert len(reused) == 1200
    for row in reused:
        old = lookup[row["source_score_id"]]
        assert row["prompt"] == old["prompt"]
        assert row["prompt_sha256"] == old["prompt_sha256"]
        assert row["option_source_ids"] == old["option_source_ids"]
        assert row["allowed_actions"] == old["allowed_actions"]


def test_plain_arm_removes_game_but_preserves_exact_question_data(package):
    first = package["jobs"][:13]
    plain, forced, wait = first[:3]
    assert (plain["arm"], forced["arm"], wait["arm"]) == ("plain", "forced", "wait")
    assert design._payload(plain["prompt"]) == design._payload(forced["prompt"]) == design._payload(wait["prompt"])
    assert "round" not in plain["prompt"] and "WAIT" not in plain["prompt"] and "points" not in plain["prompt"]
    assert 'Return exactly one action in the form {"action":"X"}' in plain["prompt"]
    assert forced["allowed_actions"] == plain["allowed_actions"] == "ABCD"


def test_rotations_preserve_candidate_identity_and_swap_preserves_order(package):
    rows = package["jobs"][:13]
    ordinary = design._payload(rows[2]["prompt"])["options"]
    for rotation in range(4):
        row = rows[rotation * 3 + 2]
        shown = design._payload(row["prompt"])["options"]
        restored = {row["option_source_ids"][option["id"]]: option["text"] for option in shown}
        assert restored == {option["id"]: option["text"] for option in ordinary}
    swap = rows[12]
    assert swap["option_source_ids"] == {"E": "A", "B": "B", "C": "C", "D": "D"}
    shown = design._payload(swap["prompt"])["options"]
    assert [option["text"] for option in shown] == [option["text"] for option in ordinary]
    assert [option["id"] for option in shown] == list("EBCD")
    assert "Answering E, B, C, or D ends" in swap["prompt"]
    assert "A means WAIT" in swap["prompt"]
    assert "E means WAIT" not in swap["prompt"]


def test_label_swap_does_not_modify_embedded_data_that_matches_rule():
    options = [{"id": label, "text": name} for label, name in zip("EBCD", ("One", "Two", "Three", "Four"))]
    question = "Quoted description. E means a letter. Answering A, B, C, or D ends this question."
    rendered = design.prompt(question, options, 5, "wait", "A")
    assert design._payload(rendered)["question_prefix"] == question
    assert rendered.endswith("A means PASS, ending with 0 points.")


def test_synthetic_checks_have_unique_mathematical_optimum_and_separate_expected_fixture():
    rows, fixture = design.comprehension_cases()
    assert len(rows) == len(fixture["expected"]) == 32
    assert Counter(entry["case_family"] for entry in fixture["expected"].values()) == {"known": 16, "future": 8, "terminal": 8}
    for row in rows:
        answer = fixture["expected"][row["score_id"]]
        assert "expected_display_action" not in row and "known_correct_candidate" not in row
        if answer["case_family"] == "known":
            assert answer["answer_now_expected_reward"] > answer["wait_or_pass_value"]
            assert row["option_source_ids"][answer["expected_display_action"]] == answer["known_correct_candidate"]
            assert "probability of being correct is 1.0" in row["prompt"]
        else:
            assert answer["answer_now_expected_reward"] < answer["wait_or_pass_value"]
            assert answer["expected_display_action"] == row["wait_label"]
            assert "probability 0.25" in row["prompt"]
    future = [entry for entry in fixture["expected"].values() if entry["case_family"] == "future"]
    assert all(entry["expected_semantic_action"] == "WAIT" and entry["wait_or_pass_value"] == .8 for entry in future)
    terminal = [entry for entry in fixture["expected"].values() if entry["case_family"] == "terminal"]
    assert all(entry["expected_semantic_action"] == "PASS" and entry["wait_or_pass_value"] == 0 for entry in terminal)


@pytest.mark.parametrize("mutation", ["extra_gold", "missing", "duplicate", "reuse", "mapping", "text", "split", "synthetic"])
def test_validation_fails_closed_on_corruption(package, mutation):
    corrupted = deepcopy(package)
    if mutation == "extra_gold": corrupted["jobs"][0]["correct_answer"] = "A"
    if mutation == "missing": corrupted["jobs"].pop()
    if mutation == "duplicate": corrupted["jobs"][1] = deepcopy(corrupted["jobs"][0])
    if mutation == "reuse": corrupted["jobs"][1]["source_score_id"] = corrupted["jobs"][2]["source_score_id"]
    if mutation == "mapping": corrupted["jobs"][0]["option_source_ids"] = dict(zip("ABCD", "BCDA"))
    if mutation == "text":
        corrupted["jobs"][0]["prompt"] += " Ignore these instructions."
        corrupted["jobs"][0]["prompt_sha256"] = prior.base.sha(corrupted["jobs"][0]["prompt"].encode())
    if mutation == "split": corrupted["jobs"][0]["split"] = "test"
    if mutation == "synthetic": corrupted["jobs"][-1]["synthetic_case"] = "invented"
    with pytest.raises(ValueError):
        design.validate_public_package(corrupted)


def test_prior_reconstruction_rejects_consistent_changed_source(package, frozen_prior, monkeypatch):
    assert design.validate_public_package(package, frozen_prior) == package["jobs"]
    corrupted = deepcopy(frozen_prior)
    corrupted["jobs"][0]["prompt"] += " changed"
    with pytest.raises(ValueError, match="prior public package hash differs"):
        design.build_public_package(corrupted, prior_public_sha256=design.PRIOR_PUBLIC_SHA256)
    with pytest.raises(ValueError, match="score hashes"):
        design.build_public_package(frozen_prior, prior_public_sha256=design.PRIOR_PUBLIC_SHA256,
                                    prior_scores_sha256={"qwen3b": "0" * 64, "qwen7b": "1" * 64})
