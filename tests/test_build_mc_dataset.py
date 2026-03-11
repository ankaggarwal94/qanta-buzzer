"""Regression tests for scripts/build_mc_dataset.py CLI defaults."""

from __future__ import annotations

from pathlib import Path

from qb_data.config import load_config as load_yaml_config
from scripts.build_mc_dataset import parse_args, resolve_output_dir


class TestBuildMcDatasetArgs:
    """Tests for smoke-aware argument resolution."""

    def test_parse_args_smoke_uses_dynamic_defaults(self) -> None:
        args = parse_args(["--smoke"])

        assert args.smoke is True
        assert args.config is None
        assert args.output_dir is None
        assert args.overrides == []

    def test_parse_args_explicit_overrides_win(self) -> None:
        args = parse_args(
            [
                "--smoke",
                "--config",
                "configs/custom.yaml",
                "--output-dir",
                "custom/output",
                "data.K=5",
            ]
        )

        assert args.smoke is True
        assert args.config == "configs/custom.yaml"
        assert args.output_dir == "custom/output"
        assert args.overrides == ["data.K=5"]

    def test_resolve_output_dir_defaults_to_smoke_artifacts(self) -> None:
        assert resolve_output_dir(None, smoke=True) == Path("artifacts/smoke")

    def test_resolve_output_dir_defaults_to_processed_data(self) -> None:
        assert resolve_output_dir(None, smoke=False) == Path("data/processed")

    def test_resolve_output_dir_preserves_explicit_override(self) -> None:
        assert resolve_output_dir("custom/output", smoke=True) == Path("custom/output")

    def test_load_config_smoke_without_explicit_path(self) -> None:
        cfg = load_yaml_config(None, smoke=True)

        assert cfg["data"]["max_questions"] == 50
        assert cfg["ppo"]["total_timesteps"] == 3000
