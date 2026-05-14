"""Focused tests for scripts/train_ppo.py runtime provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agents.ppo_buzzer import PPOEpisodeTrace
import scripts.train_ppo as train_ppo


def test_train_ppo_seed_override_updates_env_and_saved_config(
    tmp_path, monkeypatch, sample_mc_question
) -> None:
    """CLI --seed/--timesteps should flow into env construction and config_used."""
    captured_configs = []

    def fake_load_config(_config_path=None, smoke=False):
        return {
            "data": {"K": 4},
            "likelihood": {"model": "tfidf", "beta": 5.0},
            "environment": {
                "reward_mode": "time_penalty",
                "seed": 13,
                "wait_penalty": 0.05,
                "early_buzz_penalty": 0.2,
                "buzz_correct": 1.0,
                "buzz_incorrect": -0.5,
            },
            "bayesian": {"threshold_sweep": [0.5], "alpha": 10.0},
            "ppo": {
                "seed": 13,
                "total_timesteps": 999,
                "learning_rate": 3e-4,
                "n_steps": 8,
                "batch_size": 4,
                "n_epochs": 1,
                "gamma": 0.99,
                "policy_kwargs": {"net_arch": [8, 8]},
            },
        }

    def fake_make_env_from_config(mc_questions, likelihood_model, config, precomputed_beliefs=None):
        captured_configs.append(json.loads(json.dumps(config)))
        return object()

    class FakePPOBuzzer:
        def __init__(self, env, **kwargs):
            self.env = env

        def train(self, total_timesteps):
            self.total_timesteps = total_timesteps

        def save(self, path):
            (Path(str(path) + ".zip")).write_text("model", encoding="utf-8")

        @classmethod
        def load(cls, path, env, use_maskable_ppo=False):
            return cls(env)

        def run_episode(self, deterministic=False, question_idx=None):
            return PPOEpisodeTrace(
                qid=f"q{question_idx}",
                buzz_step=0,
                buzz_trace_idx=0,
                buzz_index=0,
                gold_index=0,
                correct=True,
                forced_commit=False,
                forced_step=-1,
                forced_choice=-1,
                forced_correct=False,
                episode_reward=1.0,
                c_trace=[1.0],
                g_trace=[1.0],
                top_p_trace=[0.9],
                entropy_trace=[0.0],
            )

    (tmp_path / "train_dataset.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(train_ppo, "load_config", fake_load_config)
    monkeypatch.setattr(train_ppo, "load_mc_questions", lambda _path: [sample_mc_question])
    monkeypatch.setattr(train_ppo, "build_likelihood_model", lambda config, qs: object())
    monkeypatch.setattr(train_ppo, "load_embedding_cache", lambda model, config: None)
    monkeypatch.setattr(train_ppo, "save_embedding_cache", lambda model, config: None)
    monkeypatch.setattr(train_ppo, "precompute_beliefs", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_ppo, "make_env_from_config", fake_make_env_from_config)
    monkeypatch.setattr(train_ppo, "PPOBuzzer", FakePPOBuzzer)
    monkeypatch.setattr(train_ppo, "summarize_buzz_metrics", lambda rows: {"buzz_accuracy": 1.0, "mean_sq": 1.0, "mean_buzz_step": 0.0, "mean_reward_like": 1.0})
    monkeypatch.setattr(train_ppo, "calibration_at_buzz", lambda rows: {"ece": 0.0, "brier": 0.0, "n_calibration": 1.0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--smoke",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "123",
            "--timesteps",
            "10",
        ],
    )

    train_ppo.main()

    config_used = json.loads((tmp_path / "config_used.json").read_text())
    assert config_used["ppo"]["seed"] == 123
    assert config_used["environment"]["seed"] == 123
    assert config_used["ppo"]["total_timesteps"] == 10
    assert captured_configs[0]["environment"]["seed"] == 123
    assert captured_configs[1]["environment"]["seed"] == 123
