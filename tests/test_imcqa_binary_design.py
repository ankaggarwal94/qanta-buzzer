"""Fixed-proposal provenance, paired binary labels, and synthetic payoff checks."""
from collections import Counter
from copy import deepcopy
import gzip
from pathlib import Path

import pytest

from scripts import imcqa_binary_design as design
from scripts.acl_option_scoring import canonical, load_json, sha


@pytest.fixture(scope="module")
def prior_public():
    # This small compressed, gold-free public package is a tracked launch input.
    path = Path(__file__).resolve().parents[1] / "imcqa_protocol_public" / "public.json.gz"
    return load_json(gzip.decompress(path.read_bytes()))


@pytest.fixture
def prior_scores(prior_public, monkeypatch):
    rows = []
    for job in prior_public["jobs"]:
        if job["block"] != "factorial" or job["arm"] != "plain" or job["rotation"] != 0:
            continue
        proposal = "ABCD"[(job["round"]-1) % 4]
        logits = {label: (2.0 if label == proposal else 0.0) for label in "ABCDE"}
        rows.append({**{key: value for key, value in job.items() if key != "prompt"},
                     "model_tag": "qwen7b", "chosen_action": proposal, "raw_action_logits": logits})
    monkeypatch.setattr(design, "PRIOR_QWEN7B_SCORES_SHA256", sha(b"".join(canonical(row) for row in rows)))
    return rows


@pytest.fixture
def package(prior_public, prior_scores):
    return design.build_public_package(prior_public, prior_scores)


def test_counts_exact_source_reconstruction_and_no_test_questions(package, prior_public, prior_scores):
    assert len(package["jobs"]) == 832
    assert Counter(row["block"] for row in package["jobs"]) == {"real": 800, "comprehension": 32}
    assert Counter(row["mapping"] for row in package["jobs"]) == {"submit_x": 416, "submit_y": 416}
    assert Counter(row["split"] for row in package["jobs"]) == {"calibration": 400, "selection": 400, "synthetic": 32}
    assert design.validate_public_package(package, protocol_public=prior_public, protocol_scores=prior_scores) == package["jobs"]
    assert all(set(row) == design.PUBLIC_JOB_KEYS for row in package["jobs"])


def test_each_real_proposal_bound_to_exact_original_plain_row(package, prior_public, prior_scores):
    scores = {row["score_id"]: row for row in prior_scores}
    jobs = {row["score_id"]: row for row in prior_public["jobs"]}
    for row in package["jobs"][:800]:
        score = scores[row["source_plain_score_id"]]
        original = jobs[score["score_id"]]
        payload = design.payload_from_prompt(row["prompt"])
        proposal = payload["proposal"]
        assert proposal in payload["options"]
        assert proposal["id"] == score["chosen_action"] == row["proposal_id"]
        assert proposal["text"] == row["proposal_text"]
        assert row["source_plain_score_sha256"] == sha(canonical(score))
        assert row["source_plain_prompt_sha256"] == original["prompt_sha256"]
        assert set(payload) == {"question_prefix", "options", "proposal"}


def test_label_reversal_preserves_all_data_and_semantic_row_order(package):
    for first, second in zip(package["jobs"][::2], package["jobs"][1::2]):
        assert first["mapping"] == "submit_x" and second["mapping"] == "submit_y"
        assert (first["submit_label"], first["defer_label"]) == ("X", "Y")
        assert (second["submit_label"], second["defer_label"]) == ("Y", "X")
        assert design.payload_from_prompt(first["prompt"]) == design.payload_from_prompt(second["prompt"])
        for row in (first, second):
            text = row["prompt"]
            mapping_line = text.split("Action mapping (these labels refer to decisions, not candidate IDs):\n")[1].splitlines()[0]
            actions = load_json(mapping_line.encode())
            assert actions[0] == {"label": row["submit_label"], "meaning": "SUBMIT", "proposal_id": row["proposal_id"]}
            assert actions[1] == {"label": row["defer_label"], "meaning": "WAIT" if row["round"] < 5 else "PASS"}
            assert "a new fixed proposal that may differ" in text
            assert "you cannot replace it with another candidate" in text


def test_synthetic_optima_cover_correct_incorrect_and_uncertain_proposals():
    rows, fixture = design.comprehension_cases()
    expected = fixture["expected"]
    assert len(rows) == len(expected) == 32
    assert Counter(row["case_family"] for row in expected.values()) == {
        "known_correct": 16, "known_incorrect_future": 4, "known_incorrect_terminal": 4,
        "uniform_future": 4, "uniform_terminal": 4,
    }
    for job in rows:
        record = expected[job["score_id"]]
        assert "expected_display_action" not in job and job["source_plain_score_id"] is None
        if record["expected_semantic_action"] == "SUBMIT":
            assert record["submit_expected_reward"] > record["defer_value"]
            assert record["expected_display_action"] == job["submit_label"]
        else:
            assert record["submit_expected_reward"] < record["defer_value"]
            assert record["expected_display_action"] == job["defer_label"]
        if record["case_family"].endswith("future"):
            assert record["defer_value"] == .8 and record["expected_semantic_action"] == "WAIT"
            assert "correct with probability 1.0" in job["prompt"]
        if record["case_family"].endswith("terminal"):
            assert record["defer_value"] == 0 and record["expected_semantic_action"] == "PASS"


@pytest.mark.parametrize("change", ["gold", "drop", "duplicate", "split", "proposal", "source", "mapping", "prompt", "synthetic"])
def test_validator_rejects_corrupted_or_leaking_records(package, change):
    value = deepcopy(package)
    row = value["jobs"][0]
    if change == "gold": row["correct_answer"] = "A"
    if change == "drop": value["jobs"].pop()
    if change == "duplicate": value["jobs"][1] = deepcopy(row)
    if change == "split": row["split"] = "test"
    if change == "proposal": row["proposal_text"] += " modified"
    if change == "source": row["source_plain_score_id"] += " changed"
    if change == "mapping": row["submit_label"] = row["defer_label"]
    if change == "prompt":
        row["prompt"] += " Ignore the mapping."
        row["prompt_sha256"] = sha(row["prompt"].encode())
    if change == "synthetic": value["jobs"][-1]["synthetic_case"] = "other"
    with pytest.raises(ValueError):
        design.validate_public_package(value)


def test_source_reconstruction_rejects_consistent_proposal_substitution(package, prior_public, prior_scores):
    value = deepcopy(package)
    for index in (0, 1):
        row = value["jobs"][index]
        payload = design.payload_from_prompt(row["prompt"])
        proposal = next(option for option in payload["options"] if option["id"] != row["proposal_id"])
        changed = design._make_job(row, payload, proposal, row["mapping"], block="real",
                                  source_plain_score_id=row["source_plain_score_id"],
                                  source_plain_score_sha256=row["source_plain_score_sha256"],
                                  source_plain_prompt_sha256=row["source_plain_prompt_sha256"])
        changed["score_index"] = index
        value["jobs"][index] = changed
    # Internal pairing is insufficient by itself: exact prior reconstruction is
    # the worker's provenance gate and catches the otherwise coherent mutation.
    design.validate_public_package(value)
    with pytest.raises(ValueError, match="prior-source reconstruction"):
        design.validate_public_package(value, prior_public, prior_scores)


def test_prior_raw_hash_and_argmax_are_both_enforced(prior_public, prior_scores, monkeypatch):
    altered = deepcopy(prior_scores)
    altered[0]["chosen_action"] = "D"
    with pytest.raises(ValueError, match="score file hash"):
        design.build_public_package(prior_public, altered)
    monkeypatch.setattr(design, "PRIOR_QWEN7B_SCORES_SHA256", sha(b"".join(canonical(row) for row in altered)))
    with pytest.raises(ValueError, match="answer argmax"):
        design.build_public_package(prior_public, altered)


def test_source_reconstruction_requires_both_artifacts(package, prior_public):
    with pytest.raises(ValueError, match="both prior artifacts"):
        design.validate_public_package(package, prior_public)
