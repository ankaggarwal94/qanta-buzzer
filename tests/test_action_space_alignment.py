from __future__ import annotations

import numpy as np
import torch


import pytest

from models.t5_policy import T5PolicyModel


@pytest.fixture(scope="module")
def t5_small_model():
    try:
        model = T5PolicyModel({"model_name": "t5-small", "device": "cpu", "max_input_length": 128, "num_choices": 4})
    except OSError as exc:
        pytest.skip(f"t5-small unavailable in test environment: {exc}")
    model.eval()
    return model


@pytest.fixture
def sample_texts() -> list[str]:
    return ["CLUES: test clue | CHOICES: (1) A (2) B (3) C (4) D"]

from evaluation.metrics import calibration_at_buzz, system_score
from qb_env.stop_only_env import StopOnlyEnv
from scripts import train_ppo as train_ppo_script


def test_t5_wait_log_prob_does_not_depend_on_answer_mass(t5_small_model, sample_texts):
    model = t5_small_model
    enc = model.encode_input(sample_texts[:1])
    pooled = model.get_encoder_output(enc["input_ids"], enc["attention_mask"])
    wait_logits, answer_logits, _ = model.policy_head(pooled)
    wait_probs = torch.softmax(wait_logits, dim=-1)
    answer_probs = torch.softmax(answer_logits, dim=-1)
    actions = torch.tensor([0], dtype=torch.long, device=model.device)
    lp = model._joint_action_log_prob(wait_probs, answer_probs, actions)

    altered = answer_probs.flip(dims=[-1])
    lp2 = model._joint_action_log_prob(wait_probs, altered, actions)
    assert torch.allclose(lp, lp2, atol=1e-6)


def test_t5_entropy_uses_chain_rule(t5_small_model, sample_texts):
    model = t5_small_model
    enc = model.encode_input(sample_texts[:1])
    pooled = model.get_encoder_output(enc["input_ids"], enc["attention_mask"])
    wait_logits, answer_logits, _ = model.policy_head(pooled)
    wait_probs = torch.softmax(wait_logits, dim=-1)
    answer_probs = torch.softmax(answer_logits, dim=-1)
    ent = model._joint_entropy(wait_probs, answer_probs)
    wait_ent = -(wait_probs * torch.log(wait_probs.clamp_min(1e-12))).sum(dim=-1)
    ans_ent = -(answer_probs * torch.log(answer_probs.clamp_min(1e-12))).sum(dim=-1)
    assert torch.allclose(ent, wait_ent + wait_probs[:, 1] * ans_ent, atol=1e-6)


def test_stop_only_env_has_discrete_2_action_space(sample_tfidf_env):
    env = StopOnlyEnv(sample_tfidf_env)
    assert env.action_space.n == 2


def test_flat_kplus1_mode_still_available(sample_tfidf_env):
    assert sample_tfidf_env.action_space.n == 5


def test_metrics_use_p_correct_trace_not_binary_g_trace():
    score = system_score([0.5], [0.8])
    assert abs(score - 0.4) < 1e-6
    out = calibration_at_buzz([{"buzz_step": 0, "p_correct_trace": [0.7], "correct": True}])
    assert "ece" in out


def test_no_buzz_end_mode_does_not_force_choice(sample_tfidf_env):
    sample_tfidf_env.end_mode = "no_buzz"
    _obs, _ = sample_tfidf_env.reset(seed=0)
    while True:
        _obs, _r, _t, tr, info = sample_tfidf_env.step(0)
        if tr:
            break
    assert info.get("no_buzz") is True
    assert info.get("forced_choice") == -1


def test_train_ppo_defaults_to_stop_only(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_ppo.py"])
    args = train_ppo_script.parse_args()
    assert args.policy_mode == "stop_only"
