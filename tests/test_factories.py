"""Test suite for factory functions — build_likelihood_from_config and make_env_from_config.

Covers:
- LIK-06: build_likelihood_from_config dispatches on config["likelihood"]["model"]
- CFG-02: make_env_from_config constructs TossupMCEnv from YAML config
"""

from __future__ import annotations

import numpy as np
import pytest

from models.likelihoods import (
    LikelihoodModel,
    SBERTLikelihood,
    TfIdfLikelihood,
    build_likelihood_from_config,
)
from qb_data.mc_builder import MCQuestion
from qb_env.tossup_env import TossupMCEnv, make_env_from_config


# ------------------------------------------------------------------ #
# Tests: build_likelihood_from_config (LIK-06)
# ------------------------------------------------------------------ #


class TestBuildLikelihoodFromConfig:
    """Tests for likelihood model factory function."""

    @pytest.fixture
    def stub_sbert_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub SBERT model loading so factory tests stay offline-safe."""

        def fake_init(self, model_name: str = "all-MiniLM-L6-v2") -> None:
            LikelihoodModel.__init__(self)
            self.model_name = model_name
            self.encoder = object()

        monkeypatch.setattr(SBERTLikelihood, "__init__", fake_init)

    def test_likelihood_factory_sbert(
        self, sample_config: dict, stub_sbert_init: None
    ) -> None:
        """Config with model='sbert' creates SBERTLikelihood."""
        sample_config["likelihood"]["model"] = "sbert"
        model = build_likelihood_from_config(sample_config)
        assert isinstance(model, SBERTLikelihood), (
            f"Expected SBERTLikelihood, got {type(model).__name__}"
        )

    def test_likelihood_factory_tfidf(
        self, sample_config: dict, sample_corpus: list[str]
    ) -> None:
        """Config with model='tfidf' creates TfIdfLikelihood (fitted)."""
        sample_config["likelihood"]["model"] = "tfidf"
        model = build_likelihood_from_config(sample_config, corpus_texts=sample_corpus)
        assert isinstance(model, TfIdfLikelihood), (
            f"Expected TfIdfLikelihood, got {type(model).__name__}"
        )
        assert model._is_fit is True, "TF-IDF model should be fitted after construction"

    def test_likelihood_factory_tfidf_missing_corpus(
        self, sample_config: dict
    ) -> None:
        """TF-IDF factory without corpus_texts raises ValueError."""
        sample_config["likelihood"]["model"] = "tfidf"
        with pytest.raises(ValueError, match="corpus_texts"):
            build_likelihood_from_config(sample_config)

    def test_likelihood_factory_unknown_model(self, sample_config: dict) -> None:
        """Unknown model name raises ValueError."""
        sample_config["likelihood"]["model"] = "unknown_model"
        with pytest.raises(ValueError, match="Unknown likelihood model"):
            build_likelihood_from_config(sample_config)

    def test_likelihood_factory_sbert_name_override(
        self, sample_config: dict, stub_sbert_init: None
    ) -> None:
        """sbert_name config key overrides default model name."""
        sample_config["likelihood"]["model"] = "sbert"
        sample_config["likelihood"]["sbert_name"] = "all-MiniLM-L6-v2"
        model = build_likelihood_from_config(sample_config)
        assert isinstance(model, SBERTLikelihood)
        assert model.model_name == "all-MiniLM-L6-v2", (
            f"Expected all-MiniLM-L6-v2, got {model.model_name}"
        )

    def test_likelihood_factory_embedding_model_key(
        self, sample_config: dict, stub_sbert_init: None
    ) -> None:
        """embedding_model config key works as fallback for sbert_name."""
        sample_config["likelihood"]["model"] = "sbert"
        sample_config["likelihood"]["embedding_model"] = "all-MiniLM-L6-v2"
        # Remove sbert_name if present to test fallback
        sample_config["likelihood"].pop("sbert_name", None)
        model = build_likelihood_from_config(sample_config)
        assert isinstance(model, SBERTLikelihood)
        assert model.model_name == "all-MiniLM-L6-v2"


# ------------------------------------------------------------------ #
# Tests: make_env_from_config (CFG-02)
# ------------------------------------------------------------------ #


class TestMakeEnvFromConfig:
    """Tests for environment factory function."""

    def _make_model_and_env(
        self, mc_question: MCQuestion, config: dict
    ) -> TossupMCEnv:
        """Helper to create a model and env from config."""
        corpus = mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        return make_env_from_config([mc_question], model, config)

    def test_env_factory_creates_tossup_env(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory creates a TossupMCEnv instance."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert isinstance(env, TossupMCEnv), (
            f"Expected TossupMCEnv, got {type(env).__name__}"
        )

    def test_env_factory_config_values(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory correctly extracts config values."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.K == 4, f"Expected K=4, got {env.K}"
        assert env.reward_mode == "simple", (
            f"Expected 'simple', got '{env.reward_mode}'"
        )
        assert env.belief_mode == "from_scratch", (
            f"Expected 'from_scratch', got '{env.belief_mode}'"
        )
        assert env.beta == 5.0, f"Expected beta=5.0, got {env.beta}"

    def test_env_factory_reward_mode_override(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Config overrides reward mode."""
        sample_config["environment"]["reward"] = "human_grounded"
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.reward_mode == "human_grounded", (
            f"Expected 'human_grounded', got '{env.reward_mode}'"
        )

    def test_env_factory_beta_override(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Config overrides beta value."""
        sample_config["likelihood"]["beta"] = 10.0
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.beta == 10.0, f"Expected beta=10.0, got {env.beta}"

    def test_env_factory_wait_penalty_override(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Config overrides wait_penalty value."""
        sample_config["environment"]["wait_penalty"] = 0.05
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.wait_penalty == 0.05, (
            f"Expected wait_penalty=0.05, got {env.wait_penalty}"
        )

    def test_env_factory_reset_works(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory-created env can reset and produce valid observation."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert "qid" in info, "Info should contain 'qid'"
        assert np.all(np.isfinite(obs)), "All observations should be finite"

    def test_env_factory_step_works(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory-created env can step and return valid results."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
        assert terminated is False, "WAIT should not terminate"

    def test_env_factory_reward_mode_key_fallback(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Factory supports 'reward_mode' key (default.yaml uses this)."""
        config = {
            "data": {"K": 4},
            "environment": {
                "reward_mode": "time_penalty",
                "wait_penalty": 0.1,
                "buzz_correct": 1.0,
                "buzz_incorrect": -0.5,
            },
            "likelihood": {"beta": 5.0},
        }
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = make_env_from_config([sample_mc_question], model, config)
        assert env.reward_mode == "time_penalty", (
            f"Expected 'time_penalty', got '{env.reward_mode}'"
        )


class TestDSPyFactoryIntegration:
    """Factory dispatches to DSPyLikelihood when configured."""

    def test_factory_returns_dspy_likelihood(self):
        from models.dspy_likelihood import DSPyLikelihood

        config = {
            "likelihood": {"model": "dspy"},
            "dspy": {"cache_dir": None, "program_fingerprint": "test"},
        }
        model = build_likelihood_from_config(config)
        assert isinstance(model, DSPyLikelihood)

    def test_default_paths_unchanged(self, sample_corpus):
        config = {"likelihood": {"model": "tfidf"}}
        model = build_likelihood_from_config(config, corpus_texts=sample_corpus)
        assert isinstance(model, TfIdfLikelihood)
