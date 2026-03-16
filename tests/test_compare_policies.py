"""Tests for compare_policies helper functions."""

import json
import pytest
from pathlib import Path


def test_resolve_mlp_eval_config_prefers_checkpoint_sidecar(tmp_path):
    from scripts.compare_policies import resolve_mlp_eval_config

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
    from scripts.compare_policies import resolve_mlp_eval_config

    fake_checkpoint = tmp_path / "ppo_model.zip"
    fake_checkpoint.touch()

    fallback = {"likelihood": {"model": "tfidf"}}
    resolved = resolve_mlp_eval_config(str(fake_checkpoint), fallback)
    assert resolved is fallback
