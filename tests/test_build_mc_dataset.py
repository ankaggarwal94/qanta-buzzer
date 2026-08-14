"""Regression tests for scripts/build_mc_dataset.py CLI defaults."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qb_data.config import load_config as load_yaml_config, merge_overrides
from scripts.build_mc_dataset import (
    build_metadata_entry,
    build_retained_split_metadata,
    make_mc_builder,
    parse_args,
    parse_overrides,
    resolve_output_dir,
)
from scripts.test_mc_builder import make_demo_mc_builder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_mc_dataset.py"
DEMO_SCRIPT = PROJECT_ROOT / "scripts" / "test_mc_builder.py"


def test_build_script_remains_executable() -> None:
    """The shebang entrypoint must retain its executable repository mode."""
    assert BUILD_SCRIPT.stat().st_mode & 0o111


def test_script_module_imports_do_not_mutate_sys_path() -> None:
    """Ordinary package imports must not change process-global precedence."""
    code = """
import sys
before = list(sys.path)
import scripts.build_mc_dataset
import scripts.test_mc_builder
assert sys.path == before, (before, sys.path)
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_direct_script_paths_resolve_repository_imports(tmp_path: Path) -> None:
    """Direct path execution must still bootstrap imports outside the repo."""
    help_result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr

    probe = (
        "import runpy, sys; "
        f"root = {str(PROJECT_ROOT)!r}; "
        "sys.path.append(root); "
        f"runpy.run_path({str(DEMO_SCRIPT)!r}, run_name='__probe__'); "
        "assert sys.path[0] == root, sys.path"
    )
    demo_result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert demo_result.returncode == 0, demo_result.stderr


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

    def test_displayed_override_command_matches_parser_contract(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The exact help example must parse and apply alongside known options."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0
        help_text = capsys.readouterr().out
        command = next(
            line.strip()
            for line in help_text.splitlines()
            if line.strip().startswith("python scripts/build_mc_dataset.py")
            and "data.K=5" in line
        )
        command_tokens = shlex.split(command)
        assert command_tokens[:2] == ["python", "scripts/build_mc_dataset.py"]
        override_tokens = command_tokens[2:]
        assert all(not token.startswith("--data.") for token in override_tokens)

        args = parse_args(
            [
                "--smoke",
                "--config",
                "configs/custom.yaml",
                "--output-dir",
                "custom/output",
                *override_tokens,
            ]
        )
        assert args.smoke is True
        assert args.config == "configs/custom.yaml"
        assert args.output_dir == "custom/output"
        merged = merge_overrides(
            load_yaml_config(None, smoke=True),
            parse_overrides(args),
        )
        assert merged["data"]["K"] == 5
        assert merged["data"]["distractor_strategy"] == "tfidf_profile"

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

    def test_boolean_false_variable_k_override_remains_disabled(self) -> None:
        """The documented unquoted CLI boolean must remain a real False."""
        cfg = load_yaml_config(None, smoke=False)
        args = parse_args(["data.variable_K=false"])
        cfg = merge_overrides(cfg, parse_overrides(args))

        builder = make_mc_builder(cfg)

        assert builder.variable_K is False

    def test_missing_variable_k_uses_disabled_default(self) -> None:
        """Legacy configs without variable_K keep the documented default."""
        cfg = load_yaml_config(None, smoke=False)
        cfg["data"].pop("variable_K")

        builder = make_mc_builder(cfg)

        assert builder.variable_K is False

    @pytest.mark.parametrize("value", ["false", "true", "yes", "0"])
    def test_string_variable_k_config_is_rejected(self, value: str) -> None:
        """Quoted YAML/CLI strings must not be coerced by Python truthiness."""
        cfg = load_yaml_config(None, smoke=False)
        cfg["data"]["variable_K"] = value

        with pytest.raises(ValueError, match="variable_K must be a boolean"):
            make_mc_builder(cfg)

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


def test_empty_target_stats_preserve_reference_counts_without_fitting() -> None:
    """An empty target split still reports its supplied reference corpus."""
    cfg = load_yaml_config(None, smoke=False)
    builder = make_mc_builder(cfg)
    references = [
        SimpleNamespace(answer_primary="Alpha"),
        SimpleNamespace(answer_primary="Alpha"),
        SimpleNamespace(answer_primary="Beta"),
    ]

    class ProfileBuilderThatMustNotFit:
        def fit(self, _questions) -> None:
            raise AssertionError("empty-target builds must not fit reference profiles")

    built = builder.build(
        [],
        ProfileBuilderThatMustNotFit(),
        reference_questions=references,
    )

    assert built == []
    assert builder.last_build_stats["target_questions"] == 0
    assert builder.last_build_stats["reference_questions"] == 3
    assert builder.last_build_stats["reference_answer_count"] == 2
    assert builder.last_build_stats["repair"] == {
        "attempted_questions": 0,
        "succeeded_questions": 0,
        "ranked_successes": 0,
        "fallback_successes": 0,
        "budget_exhausted_questions": 0,
        "candidate_attempts": 0,
        "candidate_scans": 0,
        "length_ratio_triggers": 0,
        "question_overlap_triggers": 0,
        "simultaneous_guard_triggers": 0,
        "failed_questions": 0,
        "exhaustive_no_solution_questions": 0,
        "unrecoverable_gold_overlap_questions": 0,
    }


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



def test_retained_split_metadata_uses_retained_mc_outputs() -> None:
    """The sidecar describes retained splits with deterministic categories."""
    train = [
        SimpleNamespace(category="Science"),
        SimpleNamespace(category="Arts"),
        SimpleNamespace(category="Science"),
    ]
    val = [SimpleNamespace(category="History")]
    test = [SimpleNamespace(category="Arts"), SimpleNamespace(category="History")]

    metadata = build_retained_split_metadata(train, val, test)

    assert metadata == {
        "train": {"count": 3, "categories": {"Arts": 1, "Science": 2}},
        "val": {"count": 1, "categories": {"History": 1}},
        "test": {"count": 2, "categories": {"Arts": 1, "History": 1}},
        "total_questions": 6,
        "split_ratios": [0.5, 1 / 6, 1 / 3],
    }
    assert list(metadata["train"]["categories"]) == ["Arts", "Science"]


def test_retained_split_metadata_handles_all_empty_splits() -> None:
    """An all-filtered build has a finite, deterministic zero-ratio sidecar."""
    assert build_retained_split_metadata([], [], []) == {
        "train": {"count": 0, "categories": {}},
        "val": {"count": 0, "categories": {}},
        "test": {"count": 0, "categories": {}},
        "total_questions": 0,
        "split_ratios": [0.0, 0.0, 0.0],
    }
