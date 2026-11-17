from hashlib import sha256
from types import SimpleNamespace

import pytest

from scripts.acl_paired_inference import paired_jobs, render_prompt, score_responses
from qb_data.acl_protocol import ArmSpec, digest


OPTIONS = ["Mars", "Earth", "Ceres", "Venus"]
PREFIX = "First clue mentions café."


@pytest.mark.parametrize("arm,round_number,expected", [
    ("oe", 1, "d4f8a4b56c8bb24b26c0857418b78eb2747828d3c070ad4c4883b19fd31bf369"),
    ("mc", 1, "c73edd9dc437d857377b914c9b7bafc4234162cbefbeb7d436c5f71cc2bc867b"),
    ("wait", 1, "8bbf5ba935cb69a1e667d7a554519dfe141daaeb05f168fbb94297d8bfb2954d"),
    ("wait", 5, "5f8eac1a5b7ad33efa35f9ee7cdfee88752b22219b66d4f9aa56a4655427af87"),
    ("forced", 1, "20929ccb19cdb3195db8a21e041002a345f3f878c421d289f10ea8165e8daff4"),
    ("forced", 5, "028341e5a3b563ed41258ee5739ce5327b11245f393d116e6f90899587a44830"),
])
def test_pinned_upstream_prompt_bytes(arm, round_number, expected):
    # SHA256 from executing each immutable upstream renderer on this input.
    prompt = render_prompt(PREFIX, OPTIONS, round_number, arm)
    assert sha256(prompt.encode()).hexdigest() == expected


def prepared(options=OPTIONS, prefixes=(PREFIX, "PRIVATE_FUTURE_SUFFIX")):
    q = SimpleNamespace(qid="q", cumulative_prefixes=prefixes, options=options)
    spec = ArmSpec.from_dict({"version": 1, "menu": "uniform_global", "k": 4, "seed": 1,
                              "reward": {"arm": "linear", "end": .2, "gamma": 1.},
                              "wrong_reward": -1., "wait_cost": 0.})
    return SimpleNamespace(questions={"test": [q]}, groups={"q": "group"}, spec=spec)


def test_oe_job_is_shared_across_menus_and_hides_future():
    first = paired_jobs(prepared(), "test")
    second = paired_jobs(prepared(OPTIONS[::-1]), "test")
    assert first[0] == second[0] and first[1]["job_id"] != second[1]["job_id"]
    assert "PRIVATE_FUTURE_SUFFIX" not in repr(first[:2])
    assert all(label not in first[0]["prompt"] for label in OPTIONS)
    with pytest.raises(ValueError, match="five"):
        paired_jobs(prepared(), "test", imcqa=True)


def test_imcqa_selects_the_original_five_even_deciles_and_terminal_pass():
    words = [f"word{x}" for x in range(40)]
    prefixes = [" ".join(words[:n]) for n in range(4, 41, 4)]
    instance = prepared(prefixes=prefixes)
    instance.questions["test"][0].gold_index = 1
    rules = {"accepted": ["Earth"], "prompt": ["planet"], "rejected": ["Venus"]}
    data = {"questions": [{"qid": "q", "answer": rules}]}
    instance.identity = {"dataset_sha256": digest(data)}
    jobs = paired_jobs(instance, "test", imcqa=True)
    assert sorted({j["prefix_index"] for j in jobs}) == [1, 3, 5, 7, 9]
    assert len(jobs) == 20
    responses = []
    for j in jobs:
        if j["arm"] in ("wait", "forced"):
            response = {"action": "E"}
        else:
            answer = "planet" if j["arm"] == "oe" else "B"
            response = {"answer": answer, "status": "answer", "confidence": .5}
        responses.append({**j, "response": response})
    scores = score_responses(instance, data, "test", responses, imcqa=True)
    assert [r["grade"] for r in scores if r["arm"] == "wait"] == ["wait"] * 4 + ["pass"]
    assert {r["grade"] for r in scores if r["arm"] == "forced"} == {"invalid"}
    assert {r["grade"] for r in scores if r["arm"] == "mc"} == {"accepted"}
    assert {r["correct"] for r in scores if r["arm"] == "oe"} == {None}
    assert {r["grade"] for r in scores if r["arm"] == "oe"} == {"clarification_required"}
    with pytest.raises(ValueError, match="exactly once"):
        score_responses(instance, data, "test", responses[:-1], imcqa=True)
    responses[0]["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        score_responses(instance, data, "test", responses, imcqa=True)


def test_noncanonical_five_prefixes_fail_imcqa_comparability():
    with pytest.raises(ValueError, match="word"):
        paired_jobs(prepared(prefixes=["a", "ab", "abc", "abcd", "abcde"]), "test", imcqa=True)


def test_frozen_prompt_cannot_be_used_for_an_unmatched_reward():
    from dataclasses import replace
    p = prepared(prefixes=["a", "a b", "a b c", "a b c d", "a b c d e"])
    p.spec = replace(p.spec, wrong_reward=-.5)
    with pytest.raises(ValueError, match="payoff"):
        paired_jobs(p, "test", imcqa=True)


def test_public_backend_rejects_augmented_private_fields_and_modified_prompts():
    from scripts.acl_paired_cpu_inference import validate_public
    jobs = paired_jobs(prepared(), "test")
    package = {"schema": "acl-paired-public-v1", "jobs": jobs}
    validate_public(package)
    jobs[0]["gold_index"] = 1
    with pytest.raises(ValueError, match="unexpected public"):
        validate_public(package)
    del jobs[0]["gold_index"]
    jobs[0]["prompt"] += " other"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_public(package)
