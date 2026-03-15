"""Regression tests for scripts/build_mc_dataset.py CLI defaults."""

from __future__ import annotations

from pathlib import Path

from qb_data.config import load_config as load_yaml_config, merge_overrides
from scripts.build_mc_dataset import parse_args, parse_overrides, resolve_output_dir


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


class TestParseOverrides:
    """Tests for the fixed flat-key override parsing."""

    def test_returns_dotted_keys(self) -> None:
        """parse_overrides must return flat dotted keys, not nested dicts."""
        args = parse_args(["data.K=5", "environment.reward_mode=simple"])
        overrides = parse_overrides(args)
        assert "data.K" in overrides
        assert overrides["data.K"] == 5
        assert "environment.reward_mode" in overrides
        assert overrides["environment.reward_mode"] == "simple"
        assert "data" not in overrides, "Must not nest into a 'data' sub-dict"

    def test_preserves_sibling_sections(self) -> None:
        """Overriding data.K must not clobber data.csv_path."""
        base = {
            "data": {"K": 4, "csv_path": "questions.csv", "distractor_strategy": "sbert_profile"},
            "environment": {"reward_mode": "time_penalty", "seed": 13},
        }
        args = parse_args(["data.K=5"])
        overrides = parse_overrides(args)
        merged = merge_overrides(dict(base), overrides)
        assert merged["data"]["K"] == 5
        assert merged["data"]["csv_path"] == "questions.csv"
        assert merged["data"]["distractor_strategy"] == "sbert_profile"
        assert merged["environment"]["reward_mode"] == "time_penalty"

    def test_value_types(self) -> None:
        """Values are parsed as int, float, bool, or string."""
        args = parse_args(["data.K=5", "likelihood.beta=3.5", "data.shuffle=true", "data.name=foo"])
        overrides = parse_overrides(args)
        assert overrides["data.K"] == 5
        assert isinstance(overrides["data.K"], int)
        assert overrides["likelihood.beta"] == 3.5
        assert isinstance(overrides["likelihood.beta"], float)
        assert overrides["data.shuffle"] is True
        assert overrides["data.name"] == "foo"

    def test_no_overrides_returns_empty(self) -> None:
        args = parse_args(["--smoke"])
        overrides = parse_overrides(args)
        assert overrides == {}

    def test_merge_overrides_leaf_only(self) -> None:
        """merge_overrides with dotted keys updates only targeted leaves."""
        config = {
            "data": {"K": 4, "csv_path": "q.csv"},
            "environment": {"reward_mode": "simple"},
        }
        result = merge_overrides(config, {"data.K": 6, "environment.reward_mode": "time_penalty"})
        assert result["data"]["K"] == 6
        assert result["data"]["csv_path"] == "q.csv"
        assert result["environment"]["reward_mode"] == "time_penalty"
