"""Unit tests for split-aware likelihood fitting in pipeline scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import evaluation.metrics as metrics
import scripts.evaluate_all as evaluate_all
import scripts.run_baselines as run_baselines


@dataclass
class _FakeQuestion:
    """Minimal question object for split-selection tests."""

    qid: str
    question: str
    option_profiles: list[str]
    tokens: list[str] = field(default_factory=lambda: ["clue"])
    run_indices: list[int] = field(default_factory=lambda: [0])
    cumulative_prefixes: list[str] = field(default_factory=lambda: ["clue"])


@dataclass
class _FakeEpisode:
    """Minimal episode payload that survives asdict()."""

    qid: str = "q1"
    correct: bool = True
    buzz_step: int = 0
    reward_like: float = 1.0
    top_p_trace: list[float] = field(default_factory=lambda: [0.9])
    c_trace: list[float] = field(default_factory=lambda: [0.9])
    g_trace: list[float] = field(default_factory=lambda: [1.0])
    entropy_trace: list[float] = field(default_factory=lambda: [0.1])


class _FakeLikelihoodModel:
    """Likelihood stub with the methods pipeline scripts call."""

    def precompute_embeddings(
        self, texts: list[str], batch_size: int = 64
    ) -> None:
        del texts, batch_size

    def load_cache(self, path: Path) -> int:
        del path
        return 0

    def save_cache(self, path: Path) -> int:
        del path
        return 0


def _config() -> dict[str, object]:
    """Return a minimal config accepted by both pipeline scripts."""

    return {
        "likelihood": {"model": "tfidf", "beta": 5.0},
        "bayesian": {"alpha": 10.0, "threshold_sweep": [0.5]},
        "environment": {"reward_mode": "standard"},
        "ppo": {},
    }


def test_run_baselines_builds_tfidf_from_train_split_when_val_selected(
    tmp_path,
    monkeypatch,
) -> None:
    """Validation runs should still fit TF-IDF on the sibling train split."""

    (tmp_path / "train_dataset.json").write_text("[]", encoding="utf-8")
    (tmp_path / "val_dataset.json").write_text("[]", encoding="utf-8")

    train_questions = [_FakeQuestion("train-q", "train clue", ["train profile"])]
    val_questions = [_FakeQuestion("val-q", "val clue", ["val profile"])]
    calls: dict[str, object] = {}

    def fake_load_mc_questions(path: Path):
        name = Path(path).name
        if name == "train_dataset.json":
            return train_questions
        if name == "val_dataset.json":
            return val_questions
        raise AssertionError(f"Unexpected dataset path: {path}")

    def fake_builder(config, mc_questions):
        calls["builder_config"] = config
        calls["builder_questions"] = mc_questions
        return _FakeLikelihoodModel()

    def fake_precompute(questions, likelihood_model, beta):
        del likelihood_model, beta
        calls["precompute_questions"] = questions
        return ["precomputed"]

    monkeypatch.setattr(
        run_baselines,
        "parse_args",
        lambda: SimpleNamespace(
            config=None,
            smoke=False,
            mc_path=None,
            output_dir=str(tmp_path),
            overrides=[],
        ),
    )
    monkeypatch.setattr(run_baselines, "load_config", lambda *args, **kwargs: _config())
    monkeypatch.setattr(
        run_baselines,
        "resolve_default_dataset_path",
        lambda *args, **kwargs: (tmp_path / "val_dataset.json", "val", None),
    )
    monkeypatch.setattr(run_baselines, "load_mc_questions", fake_load_mc_questions)
    monkeypatch.setattr(run_baselines, "build_likelihood_model", fake_builder)
    monkeypatch.setattr(run_baselines, "load_embedding_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_baselines, "save_embedding_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_baselines, "precompute_beliefs", fake_precompute)
    monkeypatch.setattr(
        run_baselines,
        "sweep_thresholds",
        lambda **kwargs: {0.5: [_FakeEpisode()]},
    )
    monkeypatch.setattr(
        run_baselines,
        "_softmax_episode_from_precomputed",
        lambda *args, **kwargs: _FakeEpisode(),
    )
    monkeypatch.setattr(
        run_baselines,
        "precompute_sequential_beliefs",
        lambda *args, **kwargs: ["seq"],
    )
    monkeypatch.setattr(
        run_baselines,
        "sweep_sequential_thresholds",
        lambda **kwargs: {0.5: [_FakeEpisode()]},
    )
    monkeypatch.setattr(
        run_baselines,
        "_always_final_from_precomputed",
        lambda *args, **kwargs: _FakeEpisode(),
    )
    monkeypatch.setattr(
        run_baselines,
        "summarize_buzz_metrics",
        lambda results: {"buzz_accuracy": 1.0, "mean_sq": 1.0},
    )
    monkeypatch.setattr(
        run_baselines,
        "calibration_at_buzz",
        lambda results: {"ece": 0.0, "brier": 0.0},
    )
    monkeypatch.setattr(run_baselines, "save_json", lambda *args, **kwargs: None)

    run_baselines.main()

    assert calls["builder_questions"] == train_questions
    assert calls["precompute_questions"] == val_questions


def test_evaluate_all_builds_tfidf_from_train_split_when_test_selected(
    tmp_path,
    monkeypatch,
) -> None:
    """Test evaluation should fit TF-IDF on the sibling train split."""

    (tmp_path / "train_dataset.json").write_text("[]", encoding="utf-8")
    (tmp_path / "test_dataset.json").write_text("[]", encoding="utf-8")

    train_questions = [_FakeQuestion("train-q", "train clue", ["train profile"])]
    test_questions = [_FakeQuestion("test-q", "test clue", ["test profile"])]
    calls: dict[str, object] = {}

    def fake_load_mc_questions(path: Path):
        name = Path(path).name
        if name == "train_dataset.json":
            return train_questions
        if name == "test_dataset.json":
            return test_questions
        raise AssertionError(f"Unexpected dataset path: {path}")

    def fake_builder(config, mc_questions):
        calls["builder_config"] = config
        calls["builder_questions"] = mc_questions
        return _FakeLikelihoodModel()

    def fake_precompute(questions, likelihood_model, beta):
        del likelihood_model, beta
        calls["precompute_questions"] = questions
        return ["precomputed"]

    monkeypatch.setattr(
        evaluate_all,
        "parse_args",
        lambda: SimpleNamespace(
            config=None,
            smoke=False,
            mc_path=None,
            output_dir=str(tmp_path),
            overrides=[],
        ),
    )
    monkeypatch.setattr(evaluate_all, "load_config", lambda *args, **kwargs: _config())
    monkeypatch.setattr(
        evaluate_all,
        "resolve_default_dataset_path",
        lambda *args, **kwargs: (tmp_path / "test_dataset.json", "test", None),
    )
    monkeypatch.setattr(evaluate_all, "load_mc_questions", fake_load_mc_questions)
    monkeypatch.setattr(evaluate_all, "build_likelihood_model", fake_builder)
    monkeypatch.setattr(evaluate_all, "load_embedding_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate_all, "precompute_beliefs", fake_precompute)
    monkeypatch.setattr(
        evaluate_all,
        "_softmax_episode_from_precomputed",
        lambda *args, **kwargs: _FakeEpisode(),
    )
    monkeypatch.setattr(
        evaluate_all,
        "summarize_buzz_metrics",
        lambda results: {"buzz_accuracy": 1.0, "mean_sq": 1.0},
    )
    monkeypatch.setattr(
        evaluate_all,
        "calibration_at_buzz",
        lambda results: {"ece": 0.0, "brier": 0.0},
    )
    monkeypatch.setattr(evaluate_all, "per_category_accuracy", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        evaluate_all,
        "run_shuffle_control_precomputed",
        lambda *args, **kwargs: {"mean_sq": 1.0, "runs": []},
    )
    monkeypatch.setattr(
        evaluate_all,
        "run_choices_only_control",
        lambda *args, **kwargs: {"skipped": True},
    )
    monkeypatch.setattr(
        evaluate_all,
        "plot_entropy_vs_clue_index",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_all,
        "plot_calibration_curve",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_all,
        "save_comparison_table",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(evaluate_all, "save_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        metrics,
        "calibration_pairs_at_buzz",
        lambda runs: ([0.9], [1]),
    )

    evaluate_all.main()

    assert calls["builder_questions"] == train_questions
    assert calls["precompute_questions"] == test_questions
