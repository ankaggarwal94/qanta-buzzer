"""Regression tests for scripts/build_mc_dataset.py CLI defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from qb_data.config import load_config as load_yaml_config, merge_overrides
from scripts.build_mc_dataset import (
    build_metadata_entry,
    make_mc_builder,
    parse_args,
    parse_overrides,
    resolve_output_dir,
)
from scripts.test_mc_builder import make_demo_mc_builder


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
        assert cfg["mc_guards"]["max_repair_attempts"] == 10_000

    def test_default_repair_budget_is_forwarded_to_mc_builder(self) -> None:
        cfg = load_yaml_config(None, smoke=False)

        builder = make_mc_builder(cfg)

        assert cfg["mc_guards"]["max_repair_attempts"] == 10_000
        assert builder.max_repair_attempts == 10_000

    def test_integer_variable_k_overrides_preserve_config_semantics(self) -> None:
        """Documented CLI integers remain valid without factory coercion."""
        cfg = load_yaml_config(None, smoke=False)
        args = parse_args(
            [
                "data.K=5",
                "data.variable_K=true",
                "data.min_K=1",
                "data.max_K=null",
            ]
        )
        cfg = merge_overrides(cfg, parse_overrides(args))

        builder = make_mc_builder(cfg)

        assert builder.K == 5
        assert builder.variable_K is True
        assert builder.min_K == 2
        assert builder.max_K == 5

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ("data.K=4.5", "K must be an integer"),
            ("data.min_K=2.5", "min_K must be an integer"),
            ("data.max_K=4.5", "max_K must be an integer"),
        ],
    )
    def test_non_integer_k_override_fails_in_builder_configuration(
        self,
        override: str,
        message: str,
    ) -> None:
        """The config factory must not truncate invalid K bounds."""
        cfg = load_yaml_config(None, smoke=False)
        args = parse_args(["data.variable_K=true", override])
        cfg = merge_overrides(cfg, parse_overrides(args))

        with pytest.raises(ValueError, match=message):
            make_mc_builder(cfg)

    def test_demo_builder_forwards_configured_repair_budget(self) -> None:
        cfg = load_yaml_config(None, smoke=False)
        cfg["mc_guards"]["max_repair_attempts"] = 7

        builder = make_demo_mc_builder(cfg)

        assert builder.max_repair_attempts == 7

    def test_demo_builder_defaults_repair_budget_for_legacy_config(self) -> None:
        cfg = load_yaml_config(None, smoke=False)
        cfg["mc_guards"].pop("max_repair_attempts")

        builder = make_demo_mc_builder(cfg)

        assert builder.max_repair_attempts == 10_000


def test_build_metadata_preserves_repair_telemetry() -> None:
    """Published split metadata must carry the stable repair schema."""
    repair = {
        "attempted_questions": 3,
        "succeeded_questions": 1,
        "ranked_successes": 1,
        "fallback_successes": 0,
        "budget_exhausted_questions": 1,
        "candidate_attempts": 37,
        "candidate_scans": 41,
        "length_ratio_triggers": 2,
        "question_overlap_triggers": 2,
        "simultaneous_guard_triggers": 1,
        "failed_questions": 2,
        "exhaustive_no_solution_questions": 1,
        "unrecoverable_gold_overlap_questions": 0,
    }
    stats = {
        "reference_answer_count": 11,
        "drop_reasons": {
            "repair_budget_exhausted": 1,
            "guard_repair_failed": 1,
        },
        "repair": repair,
    }

    metadata = build_metadata_entry([object(), object(), object()], [object()], stats)

    assert metadata["raw_count"] == 3
    assert metadata["retained_count"] == 1
    assert metadata["repair"] == repair
    assert metadata["repair"] is not repair


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


class TestPrintStatisticsPerSplitStats:
    """Regression for the stale-shared-MCBuilder reference in print_statistics."""

    def test_print_statistics_reports_per_split_drop_reasons(self, capsys) -> None:
        """When ``build_stats`` is provided, drop reasons must be reported
        for every split — not just whichever split happened to be built
        last by the shared MCBuilder."""
        from scripts.build_mc_dataset import print_statistics

        class _Q:
            def __init__(self, category: str = "X", question: str = "q", **_):
                self.category = category
                self.question = question
                self.answer_primary = "a"
                self.options = ["a", "b", "c", "d"]

        train = [_Q()]
        val = [_Q()]
        test = [_Q()]
        build_stats = {
            "train": {"drop_reasons": {"unseen_gold_answer": 5}},
            "val": {"drop_reasons": {"length_ratio_guard": 2}},
            "test": {"drop_reasons": {"question_overlap_guard": 1}},
        }
        print_statistics(
            train, val, test,
            profile_builder=None,
            build_stats=build_stats,
        )
        out = capsys.readouterr().out
        # All three splits' drop reasons must be visible, not just the
        # last builder's stats.
        assert "train:" in out
        assert "unseen_gold_answer: 5 rejections" in out
        assert "val:" in out
        assert "length_ratio_guard: 2 rejections" in out
        assert "test:" in out
        assert "question_overlap_guard: 1 rejections" in out

    def test_print_statistics_build_stats_overrides_stale_mc_builder(self, capsys) -> None:
        """If both ``mc_builder`` (stale shared reference) and
        ``build_stats`` (per-split snapshots) are passed, the snapshots
        win so the original bug is impossible to reproduce."""
        from scripts.build_mc_dataset import print_statistics

        class _Q:
            def __init__(self, category: str = "X", question: str = "q", **_):
                self.category = category
                self.question = question
                self.answer_primary = "a"
                self.options = ["a", "b", "c", "d"]

        class _SharedBuilder:
            # Mirrors how MCBuilder's last_build_stats mutates between
            # split builds — exposing only the LAST split's data.
            last_build_stats = {"drop_reasons": {"unseen_gold_answer": 999}}

        build_stats = {
            "train": {"drop_reasons": {"unseen_gold_answer": 5}},
            "val": {"drop_reasons": {}},
            "test": {"drop_reasons": {}},
        }
        print_statistics(
            [_Q()], [_Q()], [_Q()],
            profile_builder=None,
            mc_builder=_SharedBuilder(),
            build_stats=build_stats,
        )
        out = capsys.readouterr().out
        # Stale shared-builder value (999) must NOT appear; per-split
        # snapshot (5) must.
        assert "unseen_gold_answer: 999 rejections" not in out
        assert "unseen_gold_answer: 5 rejections" in out
