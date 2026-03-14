"""Test suite for scripts/_common.py and agents/ppo_buzzer.py.

Covers:
- AGT-01: PPOBuzzer training, save, load, episode execution
- AGT-07: Shared utilities (config, JSON, MCQuestion serialization)
- S_q metric support: c_trace, g_trace, entropy_trace generation

Uses TF-IDF likelihood for fast test execution (< 10 seconds total).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from agents.ppo_buzzer import PPOBuzzer, PPOEpisodeTrace
from qb_data.mc_builder import MCQuestion
from qb_env.tossup_env import TossupMCEnv
from scripts._common import (
    ARTIFACT_DIR,
    PROJECT_ROOT,
    load_config,
    load_json,
    mc_question_from_dict,
    save_json,
    to_serializable,
)


# ------------------------------------------------------------------ #
# Tests: _common utilities (AGT-07)
# ------------------------------------------------------------------ #


class TestLoadConfig:
    """Tests for config loading utility."""

    def test_load_config_default(self) -> None:
        """load_config() without args loads default.yaml with expected keys."""
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "data" in cfg
        assert "ppo" in cfg
        assert "environment" in cfg
        assert "likelihood" in cfg

    def test_load_config_smoke(self) -> None:
        """load_config() can load smoke.yaml with reduced settings."""
        smoke_path = str(PROJECT_ROOT / "configs" / "smoke.yaml")
        cfg = load_config(smoke_path)
        assert cfg["data"]["max_questions"] == 50
        assert cfg["ppo"]["total_timesteps"] == 3000


class TestJsonUtilities:
    """Tests for JSON save/load round-trip."""

    def test_save_load_json_roundtrip(self, tmp_path: Path) -> None:
        """save_json/load_json round-trips nested dicts."""
        data = {"a": 1, "b": [2, 3], "c": {"d": "hello"}}
        path = tmp_path / "test.json"
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data

    def test_save_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_json creates missing parent directories."""
        path = tmp_path / "sub" / "dir" / "test.json"
        save_json(path, {"x": 1})
        assert path.exists()


class TestMCQuestionSerialization:
    """Tests for MCQuestion serialization and deserialization."""

    def test_to_serializable_on_mcquestion(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """to_serializable converts MCQuestion to a dict."""
        result = to_serializable(sample_mc_question)
        assert isinstance(result, dict)
        assert result["qid"] == "test_q1"
        assert result["gold_index"] == 0
        assert len(result["options"]) == 4

    def test_mc_question_roundtrip(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """MCQuestion survives serialization -> deserialization round-trip."""
        serialized = to_serializable(sample_mc_question)
        restored = mc_question_from_dict(serialized)
        assert restored.qid == sample_mc_question.qid
        assert restored.gold_index == sample_mc_question.gold_index
        assert restored.options == sample_mc_question.options
        assert restored.tokens == sample_mc_question.tokens

    def test_mc_question_json_roundtrip(
        self, sample_mc_question: MCQuestion, tmp_path: Path
    ) -> None:
        """MCQuestion survives save_json -> load_json -> mc_question_from_dict."""
        path = tmp_path / "mc.json"
        save_json(path, [sample_mc_question])
        raw = load_json(path)
        restored = mc_question_from_dict(raw[0])
        assert restored.qid == sample_mc_question.qid
        assert restored.answer_primary == sample_mc_question.answer_primary


class TestArtifactDir:
    """Tests for path constants."""

    def test_artifact_dir_constant(self) -> None:
        """ARTIFACT_DIR points to project/artifacts."""
        assert ARTIFACT_DIR.name == "artifacts"
        assert ARTIFACT_DIR.parent == PROJECT_ROOT


# ------------------------------------------------------------------ #
# Tests: PPOBuzzer initialization (AGT-01)
# ------------------------------------------------------------------ #


class TestPPOBuzzerInit:
    """Tests for PPOBuzzer construction."""

    def test_ppo_buzzer_init(self, sample_tfidf_env: TossupMCEnv) -> None:
        """PPOBuzzer instantiates with default hyperparameters."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        assert buzzer.model is not None
        assert buzzer.env is sample_tfidf_env

    def test_ppo_buzzer_custom_policy_kwargs(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """PPOBuzzer accepts custom policy_kwargs."""
        buzzer = PPOBuzzer(
            env=sample_tfidf_env,
            policy_kwargs={"net_arch": [128, 128, 64]},
        )
        assert buzzer.model is not None


# ------------------------------------------------------------------ #
# Tests: Episode trace generation
# ------------------------------------------------------------------ #


class TestActionProbabilities:
    """Tests for action probability extraction."""

    def test_action_probabilities_shape(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """action_probabilities returns K+1 probabilities that sum to 1."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        obs, _ = sample_tfidf_env.reset(seed=42)
        probs = buzzer.action_probabilities(obs)
        K = sample_tfidf_env.K
        assert probs.shape == (K + 1,), f"Expected ({K + 1},), got {probs.shape}"
        assert abs(probs.sum() - 1.0) < 1e-5, f"Probabilities sum to {probs.sum()}"
        assert (probs >= 0).all(), "All probabilities should be non-negative"

    def test_c_t_computation(self, sample_tfidf_env: TossupMCEnv) -> None:
        """c_t returns buzz probability in [0, 1]."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        obs, _ = sample_tfidf_env.reset(seed=42)
        c_val = buzzer.c_t(obs)
        assert 0.0 <= c_val <= 1.0, f"c_t={c_val} out of range"

    def test_g_t_computation(self, sample_tfidf_env: TossupMCEnv) -> None:
        """g_t returns correctness probability, handles near-zero c_t."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        obs, _ = sample_tfidf_env.reset(seed=42)
        gold_index = sample_tfidf_env.question.gold_index
        g_val = buzzer.g_t(obs, gold_index)
        assert g_val >= 0.0, f"g_t={g_val} should be non-negative"
        # g_t can be > 1.0 if P(gold) > P(buzz) in early steps, but
        # mathematically g_t = P(gold) / c_t <= 1.0 since P(gold) <= c_t
        # (gold action is one of the buzz actions)
        assert g_val <= 1.0 + 1e-5, f"g_t={g_val} should be <= 1.0"


class TestRunEpisode:
    """Tests for full episode execution with traces."""

    def test_run_episode_generates_traces(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """run_episode returns PPOEpisodeTrace with matching trace lengths."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)

        assert isinstance(trace, PPOEpisodeTrace)
        assert len(trace.c_trace) == len(trace.g_trace)
        assert len(trace.c_trace) == len(trace.top_p_trace)
        assert len(trace.c_trace) == len(trace.entropy_trace)
        assert len(trace.c_trace) > 0, "Episode should have at least one step"

    def test_run_episode_trace_values(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Trace values are in valid ranges."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)

        for c_val in trace.c_trace:
            assert 0.0 <= c_val <= 1.0, f"c_trace value {c_val} out of [0,1]"
        for g_val in trace.g_trace:
            assert g_val >= 0.0, f"g_trace value {g_val} should be non-negative"
        for top_p in trace.top_p_trace:
            assert 0.0 <= top_p <= 1.0, f"top_p_trace value {top_p} out of [0,1]"
        for ent in trace.entropy_trace:
            assert ent >= 0.0, f"entropy {ent} should be non-negative"

    def test_ppo_calibration_uses_top_p_trace(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """calibration_at_buzz on PPO traces uses top_p_trace, not c_trace."""
        from dataclasses import asdict
        from evaluation.metrics import calibration_at_buzz

        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)
        assert len(trace.top_p_trace) > 0, "top_p_trace must be populated"

        cal = calibration_at_buzz([asdict(trace)])
        assert cal["n_calibration"] == 1.0
        # Confidence should be top_p_trace[buzz_step], not c_trace[buzz_step]
        idx = min(max(0, trace.buzz_step), len(trace.top_p_trace) - 1)
        expected_conf = trace.top_p_trace[idx]
        expected_brier = (expected_conf - (1.0 if trace.correct else 0.0)) ** 2
        assert abs(cal["brier"] - expected_brier) < 1e-9

    def test_run_episode_deterministic(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Deterministic episodes with same seed produce same traces."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace1 = buzzer.run_episode(deterministic=True, seed=42)
        trace2 = buzzer.run_episode(deterministic=True, seed=42)

        assert trace1.buzz_step == trace2.buzz_step
        assert trace1.buzz_index == trace2.buzz_index
        np.testing.assert_allclose(trace1.c_trace, trace2.c_trace, atol=1e-6)

    def test_run_episode_has_qid(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Episode trace includes the question ID."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)
        assert trace.qid != "", "qid should not be empty"

    def test_run_episode_correct_field(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """correct field matches buzz_index vs gold_index."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)
        assert trace.correct == (trace.buzz_index == trace.gold_index)


# ------------------------------------------------------------------ #
# Tests: Checkpoint save/load
# ------------------------------------------------------------------ #


class TestCheckpointSaveLoad:
    """Tests for PPOBuzzer model persistence."""

    def test_ppo_checkpoint_save_load(
        self, sample_tfidf_env: TossupMCEnv, tmp_path: Path
    ) -> None:
        """PPOBuzzer saves and loads from checkpoint."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        save_path = tmp_path / "ppo_test"
        buzzer.save(save_path)

        # SB3 appends .zip
        assert (tmp_path / "ppo_test.zip").exists(), "Model file should exist"

        loaded = PPOBuzzer.load(save_path, env=sample_tfidf_env)
        assert loaded.model is not None

        # Verify loaded model produces valid probabilities
        obs, _ = sample_tfidf_env.reset(seed=42)
        probs = loaded.action_probabilities(obs)
        assert probs.shape == (sample_tfidf_env.K + 1,)
        assert abs(probs.sum() - 1.0) < 1e-5


class TestMaskablePPO:
    """Tests for optional MaskablePPO path."""

    def test_default_ppo_unchanged(self, sample_tfidf_env) -> None:
        buzzer = PPOBuzzer(env=sample_tfidf_env, use_maskable_ppo=False)
        assert not buzzer._use_maskable
        trace = buzzer.run_episode(seed=42)
        assert len(trace.c_trace) > 0

    def test_maskable_import_error(self, sample_tfidf_env) -> None:
        sb3_contrib = pytest.importorskip("sb3_contrib", reason="sb3-contrib not installed")
        buzzer = PPOBuzzer(env=sample_tfidf_env, use_maskable_ppo=True)
        assert buzzer._use_maskable
