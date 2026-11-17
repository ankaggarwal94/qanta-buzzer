"""Exported jobs must describe the same horizon action as live observations."""
from dataclasses import replace

import numpy as np
import pytest

from qb_data.acl_protocol import Prepared, digest, policy_prompt
from qb_env.text_wrapper import TextObservationWrapper
from qb_env.tossup_env import TossupMCEnv
from tests.test_acl_prompt import arm


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("mode", ["no_buzz", "force_commit"])
def test_export_matches_live_observations(sample_mc_question, horizon, mode):
    question = replace(sample_mc_question,
                       cumulative_prefixes=sample_mc_question.cumulative_prefixes[:horizon],
                       run_indices=sample_mc_question.run_indices[:horizon])
    spec = arm("linear")
    prepared = Prepared(spec, {"test": [question]}, {}, {}, {"fixture": True})
    jobs = list(prepared.public_jobs("test", end_mode=mode))
    env = TossupMCEnv(
        [question], None, K=4, acl_reward=spec.reward, end_mode=mode,
        precomputed_beliefs={(0, i): np.full(4, .25) for i in range(horizon)},
        wait_penalty=-spec.wait_cost, buzz_correct=spec.correct_reward,
        buzz_incorrect=spec.wrong_reward, no_buzz_reward=spec.abstain_reward)
    wrapped = TextObservationWrapper(env, acl_spec=spec)
    prompt, _ = wrapped.reset()
    for job in jobs:
        assert job["prompt"] == prompt
        assert job["end_mode"] == mode
        prompt, reward, terminated, _, _ = wrapped.step(0)
    assert terminated
    outcome = spec.abstain_reward
    if mode == "force_commit":
        outcome = spec.reward.end if question.gold_index == 0 else spec.wrong_reward
    assert reward == pytest.approx(spec.wait_cost + outcome)


def test_export_mode_identity_and_compatible_default(sample_mc_question):
    prepared = Prepared(arm("constant"), {"test": [sample_mc_question]}, {}, {}, {})
    default = list(prepared.public_jobs("test"))
    assert default == list(prepared.public_jobs("test", end_mode="no_buzz"))
    forced = list(prepared.public_jobs("test", end_mode="force_commit"))
    assert {job["job_id"] for job in default}.isdisjoint(job["job_id"] for job in forced)
    for i, job in enumerate(default):
        old_prompt = policy_prompt(sample_mc_question.cumulative_prefixes[i],
                                   sample_mc_question.options, prepared.spec)
        assert job["job_id"] == digest([prepared.identity, "test", sample_mc_question.qid, i, old_prompt])
    with pytest.raises(ValueError, match="end mode"):
        list(prepared.public_jobs("test", end_mode="invalid"))
    prepared.questions["empty"] = []
    with pytest.raises(ValueError, match="end mode"):
        list(prepared.public_jobs("empty", end_mode="invalid"))
