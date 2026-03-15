"""Integration-style guards for the PR1 feature-port subset."""

from __future__ import annotations

import pytest
import torch

from models.t5_policy import T5PolicyModel
from models.likelihoods import TfIdfLikelihood
from qb_env import StopOnlyEnv, TossupMCEnv


@pytest.fixture(scope="module")
def t5_small_model():
    try:
        model = T5PolicyModel(
            {
                "model_name": "t5-small",
                "device": "cpu",
                "max_input_length": 128,
                "num_choices": 4,
            }
        )
    except OSError as exc:
        pytest.skip(f"t5-small unavailable in test environment: {exc}")
    model.eval()
    return model


def test_t5_wait_log_prob_does_not_depend_on_answer_logits(t5_small_model):
    """WAIT log-prob is independent of answer-head mass."""
    model = t5_small_model
    joint_log_prob = getattr(model, "_joint_action_log_prob")
    wait_logits = torch.tensor([[1.5, -0.5]], dtype=torch.float32, device=model.device)
    answer_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32, device=model.device)
    actions = torch.tensor([0], dtype=torch.long, device=model.device)

    lp1 = joint_log_prob(wait_logits, answer_logits, actions)
    lp2 = joint_log_prob(wait_logits, answer_logits.flip(dims=[-1]), actions)
    assert torch.allclose(lp1, lp2, atol=1e-6)


def test_t5_entropy_uses_chain_rule(t5_small_model):
    """Joint entropy follows H(wait) + p_buzz * H(answer)."""
    model = t5_small_model
    joint_entropy = getattr(model, "_joint_entropy")
    wait_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=model.device)
    answer_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], dtype=torch.float32, device=model.device)

    entropy = joint_entropy(wait_logits, answer_logits)
    wait_probs = torch.softmax(wait_logits, dim=-1)
    wait_log_probs = torch.log_softmax(wait_logits, dim=-1)
    answer_probs = torch.softmax(answer_logits, dim=-1)
    answer_log_probs = torch.log_softmax(answer_logits, dim=-1)
    expected = (
        -(wait_probs * wait_log_probs).sum(dim=-1)
        + wait_probs[:, 1] * (-(answer_probs * answer_log_probs).sum(dim=-1))
    )
    assert torch.allclose(entropy, expected, atol=1e-6)


def test_stop_only_env_has_discrete_2_action_space(sample_tfidf_env):
    env = StopOnlyEnv(sample_tfidf_env)
    assert env.action_space.n == 2


def test_flat_kplus1_mode_still_available(sample_tfidf_env):
    assert sample_tfidf_env.action_space.n == 5


def test_no_buzz_end_mode_does_not_force_choice(sample_mc_question):
    corpus = sample_mc_question.option_profiles[:]
    model = TfIdfLikelihood(corpus_texts=corpus)
    env = TossupMCEnv(
        questions=[sample_mc_question],
        likelihood_model=model,
        K=4,
        reward_mode="simple",
        end_mode="no_buzz",
        no_buzz_reward=0.0,
    )
    _obs, _info = env.reset(seed=0)
    while True:
        _obs, _reward, _term, truncated, info = env.step(0)
        if truncated:
            break
    assert info.get("no_buzz") is True
    assert info.get("forced_choice") == -1
    assert info.get("forced_correct") is False
