"""Compatibility bridge tests for qb-rl surfaces ported into qanta-buzzer."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import agents.bayesian_buzzer as bayesian_buzzer
import models.answer_profiles as compat_answer_profiles
import models.likelihoods as likelihoods
import qb_data.answer_profiles as qb_answer_profiles
import qb_data.data_loader as qb_data_loader
import qb_env.data_loader as compat_data_loader
import qb_env.mc_builder as compat_mc_builder
import qb_env.text_utils as compat_text_utils
from agents.softmax_profile_buzzer import (
    SequentialBayesBuzzer as CompatSequentialBayesBuzzer,
)
from agents.softmax_profile_buzzer import (
    SoftmaxEpisodeResult as CompatSoftmaxEpisodeResult,
)
from agents.softmax_profile_buzzer import (
    SoftmaxProfileBuzzer as CompatSoftmaxProfileBuzzer,
)
from models.likelihoods import OpenAILikelihood, build_likelihood_from_config
from qb_data.mc_builder import MCBuilder


def _install_fake_openai(monkeypatch, vectors: dict[str, list[float]], calls: list[tuple[str, tuple[str, ...]]]) -> None:
    """Install a fake ``openai`` module that serves deterministic embeddings."""

    class FakeEmbeddingsClient:
        def create(self, model: str, input: list[str]):
            calls.append((model, tuple(input)))
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=vectors[text])
                    for text in input
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.embeddings = FakeEmbeddingsClient()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))


class TestOpenAILikelihood:
    """Tests for optional OpenAI embedding support."""

    def test_openai_likelihood_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAILikelihood()

    def test_openai_likelihood_scores_and_reuses_cache(self, monkeypatch) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []
        vectors = {
            "first president": [2.0, 0.0],
            "george washington": [3.0, 0.0],
            "albert einstein": [0.0, 4.0],
        }
        _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        model = OpenAILikelihood(model="fake-embedding-model")

        embeddings = model._embed_batch(["first president", "george washington"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)
        calls_before_score = len(calls)

        scores_1 = model.score(
            "first president",
            ["george washington", "albert einstein"],
        )
        assert scores_1[0] > scores_1[1]
        assert len(calls) == calls_before_score + 2, (
            "first score should call the embeddings API twice"
        )

        scores_2 = model.score(
            "first president",
            ["george washington", "albert einstein"],
        )
        np.testing.assert_allclose(scores_1, scores_2, atol=1e-6)
        assert len(calls) == calls_before_score + 2, "second score should be served from cache"

    def test_likelihood_factory_openai(self, monkeypatch) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []
        vectors = {"a": [1.0, 0.0]}
        _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        config = {"likelihood": {"model": "openai", "openai_model": "fake-openai"}}
        model = build_likelihood_from_config(config)

        assert isinstance(model, OpenAILikelihood)
        assert model.model == "fake-openai"


class TestOpenAIProfileStrategy:
    """Tests for OpenAI-backed distractor ranking."""

    def test_openai_profile_uses_openai_embeddings(self, monkeypatch) -> None:
        calls: list[str] = []
        embeddings = {
            "gold profile": np.array([1.0, 0.0], dtype=np.float32),
            "near distractor": np.array([0.9, 0.1], dtype=np.float32),
            "far distractor": np.array([0.0, 1.0], dtype=np.float32),
        }

        class FakeOpenAILikelihood:
            def __init__(self, model: str = "unused") -> None:
                calls.append(model)

            def embed_and_cache(self, texts: list[str]) -> np.ndarray:
                return np.stack([embeddings[text] for text in texts]).astype(np.float32)

        monkeypatch.setattr(likelihoods, "OpenAILikelihood", FakeOpenAILikelihood)

        builder = MCBuilder(strategy="openai_profile", openai_model="fake-openai")
        rankings = builder._compute_rankings(
            answers=["gold", "near", "far"],
            answer_profiles={
                "gold": "gold profile",
                "near": "near distractor",
                "far": "far distractor",
            },
            answer_to_category={},
        )

        assert calls == ["fake-openai"]
        assert rankings["gold"][0] == "near"
        assert rankings["gold"][1] == "far"


class TestQBRLCompatibilityModules:
    """Tests for qb-rl import-path shims."""

    def test_module_aliases_resolve_expected_symbols(self) -> None:
        assert compat_answer_profiles.AnswerProfileBuilder is qb_answer_profiles.AnswerProfileBuilder
        assert compat_data_loader.parse_row is qb_data_loader.parse_row
        assert compat_mc_builder.MCBuilder.__name__ == "MCBuilder"
        assert compat_text_utils.normalize_answer("The Answer") == "answer"
        assert CompatSoftmaxProfileBuzzer is bayesian_buzzer.SoftmaxProfileBuzzer
        assert CompatSequentialBayesBuzzer is bayesian_buzzer.SequentialBayesBuzzer
        assert CompatSoftmaxEpisodeResult is bayesian_buzzer.SoftmaxEpisodeResult

    def test_parse_row_supports_qb_rl_metadata(self) -> None:
        question = compat_data_loader.parse_row(
            {
                "qid": "q-1",
                "question": "alpha beta gamma",
                "answer_primary": "George Washington",
                "clean_answers": ["George Washington", "Washington"],
                "run_indices": [1, 2],
                "metadata": {
                    "category": "History",
                    "human_buzz_positions": [{"position": 4, "count": 2}],
                },
            }
        )

        assert question.qid == "q-1"
        assert question.category == "History"
        assert question.human_buzz_positions == [(4, 2)]
        assert question.cumulative_prefixes == ["alpha beta", "alpha beta gamma"]

    def test_load_tossup_questions_from_config_prefers_dataset_smoke(
        self, monkeypatch
    ) -> None:
        captured: dict[str, object] = {}
        sample_question = compat_data_loader.TossupQuestion(
            qid="hf-1",
            question="alpha beta",
            tokens=["alpha", "beta"],
            answer_primary="Answer",
            clean_answers=["Answer"],
            run_indices=[1],
            human_buzz_positions=None,
            category="History",
            cumulative_prefixes=["alpha beta"],
        )

        def fake_load_tossup_questions(
            dataset: str,
            dataset_config: str | None = None,
            split: str = "eval",
            limit: int | None = None,
        ):
            captured["dataset"] = dataset
            captured["dataset_config"] = dataset_config
            captured["split"] = split
            captured["limit"] = limit
            return [sample_question]

        monkeypatch.setattr(qb_data_loader, "load_tossup_questions", fake_load_tossup_questions)

        config = {
            "data": {
                "dataset": "main-dataset",
                "dataset_config": "main-config",
                "dataset_smoke": "smoke-dataset",
                "dataset_smoke_config": "smoke-config",
                "split": "train",
            }
        }

        questions = compat_data_loader.load_tossup_questions_from_config(config, smoke=True)

        assert len(questions) == 1
        assert captured == {
            "dataset": "smoke-dataset",
            "dataset_config": "smoke-config",
            "split": "train",
            "limit": None,
        }
