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
