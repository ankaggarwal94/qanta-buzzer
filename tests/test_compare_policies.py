"""Tests for compare_policies helper functions."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_policies import (
    resolve_mlp_eval_config,
    resolve_t5_reference_questions,
    resolve_t5_test_questions,
)


def test_resolve_mlp_eval_config_prefers_checkpoint_sidecar(tmp_path):

    sidecar_config = {"likelihood": {"model": "t5-base"}, "ppo": {"seed": 99}}
    sidecar_path = tmp_path / "config_used.json"
    sidecar_path.write_text(json.dumps(sidecar_config))

    fake_checkpoint = tmp_path / "ppo_model.zip"
    fake_checkpoint.touch()

    fallback = {"likelihood": {"model": "tfidf"}}
    resolved = resolve_mlp_eval_config(str(fake_checkpoint), fallback)
    assert resolved["likelihood"]["model"] == "t5-base"
    assert resolved["ppo"]["seed"] == 99


def test_resolve_mlp_eval_config_uses_fallback_when_no_sidecar(tmp_path):
    fake_checkpoint = tmp_path / "ppo_model.zip"
    fake_checkpoint.touch()

    fallback = {"likelihood": {"model": "tfidf"}}
    resolved = resolve_mlp_eval_config(str(fake_checkpoint), fallback)
    assert resolved is fallback


def test_resolve_mlp_eval_config_handles_directory_checkpoint(tmp_path):
    """When checkpoint_path is a directory, look for sidecar inside it."""
    ckpt_dir = tmp_path / "best_model"
    ckpt_dir.mkdir()
    sidecar_config = {"likelihood": {"model": "sbert"}, "ppo": {"seed": 7}}
    (ckpt_dir / "config_used.json").write_text(json.dumps(sidecar_config))

    fallback = {"likelihood": {"model": "tfidf"}}
    resolved = resolve_mlp_eval_config(str(ckpt_dir), fallback)
    assert resolved["likelihood"]["model"] == "sbert"


def test_resolve_mlp_eval_config_survives_corrupt_json(tmp_path):
    """Corrupt sidecar JSON should fall back gracefully, not crash."""
    (tmp_path / "config_used.json").write_text("{bad json")
    fake_checkpoint = tmp_path / "ppo_model.zip"
    fake_checkpoint.touch()

    fallback = {"likelihood": {"model": "tfidf"}}
    resolved = resolve_mlp_eval_config(str(fake_checkpoint), fallback)
    assert resolved is fallback


def test_evaluate_mlp_policy_uses_builder_and_question_idx(monkeypatch):
    import scripts.compare_policies as cp
    from agents.ppo_buzzer import PPOBuzzer
    import qb_env.tossup_env as te

    calls: dict[str, object] = {
        "builder_count": 0,
        "builder_args": None,
        "question_idx": [],
    }

    def fake_builder(config, test_questions):
        calls["builder_count"] += 1
        calls["builder_args"] = (config, test_questions)
        return object()

    def fake_make_env_from_config(mc_questions, likelihood_model, config):
        return object()

    class FakeAgent:
        def run_episode(self, deterministic=True, question_idx=None):
            calls["question_idx"].append(question_idx)
            return {"buzz_step": 0, "correct": True, "top_p_trace": [0.9]}

    monkeypatch.setattr(cp, "build_likelihood_model", fake_builder)
    monkeypatch.setattr(cp, "load_embedding_cache", lambda model, config: None)
    monkeypatch.setattr(te, "make_env_from_config", fake_make_env_from_config)
    monkeypatch.setattr(
        PPOBuzzer,
        "load",
        classmethod(lambda cls, checkpoint_path, env, use_maskable_ppo=False: FakeAgent()),
    )
    monkeypatch.setattr(cp, "summarize_buzz_metrics", lambda results: {"buzz_accuracy": 1.0, "mean_sq": 1.0, "mean_buzz_step": 0.0, "mean_reward_like": 0.0})
    monkeypatch.setattr(cp, "calibration_pairs_at_buzz", lambda results: ([0.9], [1]))
    monkeypatch.setattr(cp, "expected_calibration_error", lambda c, o: 0.0)
    monkeypatch.setattr(cp, "brier_score", lambda c, o: 0.0)

    out = cp.evaluate_mlp_policy(
        checkpoint_path="artifacts/main/ppo_model",
        test_questions=[object(), object(), object()],
        config={"likelihood": {"model": "tfidf"}, "ppo": {}},
    )

    assert calls["builder_count"] == 1
    assert calls["question_idx"] == [0, 1, 2]
    assert out["accuracy"] == 1.0


def test_resolve_t5_test_questions_prefers_split_manifest(tmp_path, monkeypatch):
    import scripts.compare_policies as cp

    train_path = tmp_path / "train_dataset.json"
    test_path = tmp_path / "test_dataset.json"
    train_path.write_text("[]")
    test_path.write_text("[]")
    manifest = {
        "source": "persisted_artifacts",
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_qids": ["train_qid"],
        "test_qids": ["test_qid_b", "test_qid_a"],
    }
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest))
    checkpoint_dir = tmp_path / "best_model"
    checkpoint_dir.mkdir()

    def fake_load_mc_questions(path: str | Path):
        name = Path(path).name
        if name == "test_dataset.json":
            return [
                type("Q", (), {"qid": "test_qid_a"})(),
                type("Q", (), {"qid": "test_qid_b"})(),
            ]
        return [type("Q", (), {"qid": "train_qid"})()]

    monkeypatch.setattr(cp, "load_mc_questions", fake_load_mc_questions)

    questions, source = resolve_t5_test_questions(
        checkpoint_dir,
        all_questions=[],
        mc_path=tmp_path / "mc_dataset.json",
    )
    assert source == "split_manifest"
    assert [q.qid for q in questions] == ["test_qid_b", "test_qid_a"]


def test_resolve_t5_reference_questions_prefers_split_manifest(tmp_path, monkeypatch):
    import scripts.compare_policies as cp

    train_path = tmp_path / "train_dataset.json"
    train_path.write_text("[]")
    manifest = {
        "source": "persisted_artifacts",
        "train_path": str(train_path),
        "train_qids": ["train_qid_b", "train_qid_a"],
    }
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest))
    checkpoint_dir = tmp_path / "best_model"
    checkpoint_dir.mkdir()

    monkeypatch.setattr(
        cp,
        "load_mc_questions",
        lambda _path: [
            type("Q", (), {"qid": "train_qid_a"})(),
            type("Q", (), {"qid": "train_qid_b"})(),
        ],
    )

    questions, source = resolve_t5_reference_questions(
        checkpoint_dir,
        all_questions=[],
        mc_path=tmp_path / "mc_dataset.json",
    )
    assert source == "split_manifest"
    assert [q.qid for q in questions] == ["train_qid_b", "train_qid_a"]


def _fake_q(qid: str):
    return type("Q", (), {"qid": qid})()


def test_resolve_t5_reference_questions_combined_fallback_subtracts_test_qids(tmp_path):
    """Combined-dataset fallback must exclude test qids to avoid leakage."""
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    all_questions = [_fake_q(f"q{i}") for i in range(10)]
    test_questions = [all_questions[1], all_questions[4], all_questions[7]]

    ref, source = resolve_t5_reference_questions(
        checkpoint_dir,
        all_questions=all_questions,
        mc_path=tmp_path / "mc_dataset.json",
        test_questions=test_questions,
    )
    assert source == "combined_dataset_minus_test_fallback"
    ref_qids = {q.qid for q in ref}
    assert ref_qids.isdisjoint({"q1", "q4", "q7"})
    assert len(ref) == 7


def test_resolve_t5_reference_questions_combined_fallback_raises_when_empty(tmp_path):
    """When test_questions == all_questions, the filter empties the reference; raise."""
    import pytest as _pytest
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    all_questions = [_fake_q("q1"), _fake_q("q2")]

    with _pytest.raises(ValueError, match="fully contained"):
        resolve_t5_reference_questions(
            checkpoint_dir,
            all_questions=all_questions,
            mc_path=tmp_path / "mc_dataset.json",
            test_questions=list(all_questions),
        )


def test_resolve_t5_reference_questions_legacy_no_test_questions_warns(tmp_path, capsys):
    """Without ``test_questions``, the legacy combined fallback path is preserved
    but emits a loud warning so silent contamination is visible."""
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    all_questions = [_fake_q("q1"), _fake_q("q2")]

    ref, source = resolve_t5_reference_questions(
        checkpoint_dir,
        all_questions=all_questions,
        mc_path=tmp_path / "mc_dataset.json",
    )
    assert source == "combined_dataset_fallback"
    assert ref is all_questions
    out = capsys.readouterr().out
    assert "falling back to combined" in out and "in-sample" in out.lower()


def test_resolve_manifest_questions_treats_empty_qids_as_unresolved(
    tmp_path, monkeypatch, capsys
):
    """Empty qids list must fall through to sibling/random resolution.

    Previously ``_resolve_manifest_questions`` returned ``[]`` for an
    empty ``train_qids``/``test_qids``, which downstream consumers
    treated as a valid split and reported zero-everything metrics.
    """
    import scripts.compare_policies as cp

    train_path = tmp_path / "train_dataset.json"
    train_path.write_text("[]")
    manifest = {
        "source": "persisted_artifacts",
        "train_path": str(train_path),
        "train_qids": [],
    }
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest))
    checkpoint_dir = tmp_path / "best_model"
    checkpoint_dir.mkdir()
    monkeypatch.setattr(cp, "load_mc_questions", lambda _path: [])

    # When the manifest is unresolved AND no sibling train_dataset.json
    # provides a non-zero corpus, the function falls through to the
    # combined-fallback path. We assert the source is NOT split_manifest.
    _ref, source = resolve_t5_reference_questions(
        checkpoint_dir,
        all_questions=[_fake_q("q1")],
        mc_path=tmp_path / "mc_dataset.json",
    )
    assert source != "split_manifest"
    # The warning trail must mention the empty qids list explicitly.
    out = capsys.readouterr().out
    assert "empty" in out.lower()


def test_evaluate_mlp_policy_warns_on_empty_reference_questions(monkeypatch, capsys):
    """An empty ``reference_questions`` list must trigger the leakage
    warning AND fall back to ``test_questions`` for the likelihood fit.

    Regression for the truthiness-vs-``is None`` bug where the warning
    only fired on ``None`` and an empty list silently used test text.
    """
    import scripts.compare_policies as cp
    from agents.ppo_buzzer import PPOBuzzer
    import qb_env.tossup_env as te

    captured = {"builder_corpus": None}

    def fake_builder(config, corpus_questions):
        captured["builder_corpus"] = corpus_questions
        return object()

    def fake_make_env_from_config(mc_questions, likelihood_model, config):
        return object()

    class FakeAgent:
        def run_episode(self, deterministic=True, question_idx=None):
            return {"buzz_step": 0, "correct": True, "top_p_trace": [0.9]}

    monkeypatch.setattr(cp, "build_likelihood_model", fake_builder)
    monkeypatch.setattr(cp, "load_embedding_cache", lambda model, config: None)
    monkeypatch.setattr(te, "make_env_from_config", fake_make_env_from_config)
    monkeypatch.setattr(
        PPOBuzzer,
        "load",
        classmethod(lambda cls, checkpoint_path, env, use_maskable_ppo=False: FakeAgent()),
    )
    monkeypatch.setattr(
        cp,
        "summarize_buzz_metrics",
        lambda results: {
            "buzz_accuracy": 0.5,
            "mean_sq": 0.1,
            "mean_buzz_step": 0.0,
            "mean_reward_like": 0.0,
        },
    )
    monkeypatch.setattr(cp, "calibration_pairs_at_buzz", lambda results: ([0.9], [1]))
    monkeypatch.setattr(cp, "expected_calibration_error", lambda c, o: 0.0)
    monkeypatch.setattr(cp, "brier_score", lambda c, o: 0.0)

    test_questions = [_fake_q("test_q0"), _fake_q("test_q1")]
    cp.evaluate_mlp_policy(
        checkpoint_path="artifacts/main/ppo_model",
        test_questions=test_questions,
        config={"likelihood": {"model": "tfidf"}, "ppo": {}},
        reference_questions=[],  # empty list — the bug scenario
    )

    # Builder must have been called with test_questions (the fallback)
    # AND the leakage warning must have fired.
    assert captured["builder_corpus"] is test_questions
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "empty reference_questions" in out


def test_evaluate_mlp_policy_passes_reference_questions_to_builder(monkeypatch):
    """evaluate_mlp_policy must fit the likelihood model on the train
    reference set, not the held-out test set."""
    import scripts.compare_policies as cp
    from agents.ppo_buzzer import PPOBuzzer
    import qb_env.tossup_env as te

    captured = {"builder_corpus": None}

    def fake_builder(config, corpus_questions):
        captured["builder_corpus"] = corpus_questions
        return object()

    def fake_make_env_from_config(mc_questions, likelihood_model, config):
        return object()

    class FakeAgent:
        def run_episode(self, deterministic=True, question_idx=None):
            return {"buzz_step": 0, "correct": True, "top_p_trace": [0.9]}

    monkeypatch.setattr(cp, "build_likelihood_model", fake_builder)
    monkeypatch.setattr(cp, "load_embedding_cache", lambda model, config: None)
    monkeypatch.setattr(te, "make_env_from_config", fake_make_env_from_config)
    monkeypatch.setattr(
        PPOBuzzer,
        "load",
        classmethod(lambda cls, checkpoint_path, env, use_maskable_ppo=False: FakeAgent()),
    )
    monkeypatch.setattr(
        cp,
        "summarize_buzz_metrics",
        lambda results: {
            "buzz_accuracy": 1.0,
            "mean_sq": 1.0,
            "mean_buzz_step": 0.0,
            "mean_reward_like": 0.0,
        },
    )
    monkeypatch.setattr(cp, "calibration_pairs_at_buzz", lambda results: ([0.9], [1]))
    monkeypatch.setattr(cp, "expected_calibration_error", lambda c, o: 0.0)
    monkeypatch.setattr(cp, "brier_score", lambda c, o: 0.0)

    test_questions = [_fake_q("test_q0"), _fake_q("test_q1")]
    reference_questions = [_fake_q("train_q0"), _fake_q("train_q1"), _fake_q("train_q2")]
    result = cp.evaluate_mlp_policy(
        checkpoint_path="artifacts/main/ppo_model",
        test_questions=test_questions,
        config={"likelihood": {"model": "tfidf"}, "ppo": {}},
        reference_questions=reference_questions,
        reference_source="sibling_train_dataset",
        test_set_source="sibling_test_dataset",
    )

    # The TF-IDF corpus must be the train/reference list, not the test list.
    assert captured["builder_corpus"] is reference_questions
    assert result["reference_source"] == "sibling_train_dataset"
    assert result["test_set_source"] == "sibling_test_dataset"
