"""Hazard-efficacy harness: orchestration rules (R-003, R-005, R-008, R-011,
R-013, R-014).

Filesystem/subprocess-shaped RED tests for ``scripts/run_hazard_efficacy.py``.
All child processes are replaced by monkeypatching the single injectable
seam ``scripts.run_hazard_efficacy._run_child(argv, log_path)``; run dirs
are fabricated with sidecars that replicate the real producers
(``config_used.json`` / ``split_manifest.json`` per
``scripts/train_t5_policy.py``, AP-031 format pinning — see
``tests/_hazard_efficacy_fixtures.py``).

- R-003: cross-arm config/split sidecar assertions name the offending arm.
- R-005: one ``evaluate_t5_policy`` call per arm x seed with identical
  test split + kwargs (except checkpoint path), ``return_runs=True``.
- R-008: distinct run dirs per arm x seed; fail-loud provenance; the
  supervised split manifest is persisted and train/test disjointness holds.
- R-011: argv LISTS with smoke injecting ONLY ``ppo.eval_interval=1``;
  ``shell=True`` nowhere; missing checkpoints / nonzero exits fail loud
  naming the run and log.
- R-013: preflight before any child, ``--dry-run``, resume markers
  (incl. ``wall_clock_seconds``), partial-dir fail-loud, ``--force``,
  split-source assertion, git-sha drift.
- R-014: per-run ``eval_result.json`` persistence, ``--report-only``,
  ``--prune-checkpoints`` semantics.
- main() composition: one shared supervised child FIRST, arm children
  branch from its checkpoint, eval runs on the TEST artifact's questions
  (never train's), report + plot written at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_hazard_efficacy as harness
from tests._hazard_efficacy_fixtures import (
    DEFAULT_SPLIT_QIDS,
    current_git_sha,
    fabricating_runner,
    make_hazard_dynamics,
    make_plan_record,
    make_run_dir,
    make_split_manifest,
    write_json,
    write_split_artifacts,
)

CONFIG_PATH = str(harness.PROJECT_ROOT / "configs" / "t5_policy.yaml")


def _namespace(out_dir: Path, **overrides) -> argparse.Namespace:
    """A parsed-args namespace with the pinned dest names and defaults."""
    values = {
        "config": CONFIG_PATH,
        "smoke": True,
        "seeds": [1, 2, 3],
        "arms": ["A", "B", "C"],
        "beta_terminal": 1.0,
        "freeze_answer_head": False,
        "out_dir": str(out_dir),
        "force": False,
        "dry_run": False,
        "report_only": False,
        "prune_checkpoints": False,
        "variant": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _argv_kwargs(run_dir: Path, **overrides) -> dict:
    kwargs = {
        "arm": "B",
        "seed": 2,
        "run_dir": run_dir,
        "shared_supervised_path": run_dir.parent / "shared_supervised" / "best_model",
        "config_path": CONFIG_PATH,
        "smoke": False,
        "hazard": True,
        "beta_terminal": None,
        "freeze_answer_head": False,
        "ablation": None,
        "extra_flags": None,
    }
    kwargs.update(overrides)
    return kwargs


def _flag_value(argv: list[str], flag: str) -> str:
    """Return the token following ``flag`` in an argv list."""
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------------------
# CLI surface (R-011 / spec-mandated interface)
# ---------------------------------------------------------------------------


# Tests R-011 [unit]: pinned CLI defaults.
def test_r011_parse_args_defaults() -> None:
    args = harness.parse_args([])
    assert str(args.config).endswith("configs/t5_policy.yaml")
    assert args.smoke is False
    assert args.seeds == [1, 2, 3]
    assert args.arms == ["A", "B", "C"]
    assert args.beta_terminal == pytest.approx(1.0)
    assert args.freeze_answer_head is False
    assert str(args.out_dir) == "results/hazard_efficacy"
    assert args.force is False
    assert args.dry_run is False
    assert args.report_only is False
    assert args.prune_checkpoints is False
    assert args.variant == []


# Tests R-011 [unit]: nargs lists parse as typed lists; --variant repeats.
def test_r011_parse_args_lists_and_repeatable_variant() -> None:
    args = harness.parse_args(
        [
            "--seeds", "1", "2",
            "--arms", "A", "B",
            "--beta-terminal", "2.0",
            "--freeze-answer-head",
            "--smoke",
            "--variant", "Bfz:--freeze-answer-head",
            "--variant", "Bb2:--beta-terminal 2.0",
        ]
    )
    assert args.seeds == [1, 2]
    assert args.arms == ["A", "B"]
    assert args.beta_terminal == pytest.approx(2.0)
    assert args.freeze_answer_head is True
    assert args.smoke is True
    assert args.variant == ["Bfz:--freeze-answer-head", "Bb2:--beta-terminal 2.0"]


# ---------------------------------------------------------------------------
# R-011 argv construction
# ---------------------------------------------------------------------------


# Tests R-011 [integration]: smoke argv injects ONLY ppo.eval_interval=1 —
# never ppo.save_interval — and carries the pinned branch/seed flags.
def test_r011_smoke_argv_injects_eval_interval_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "B_seed2"
    argv = harness.build_child_argv(**_argv_kwargs(run_dir, smoke=True))

    assert isinstance(argv, list)
    assert all(isinstance(token, str) for token in argv)
    assert argv[0] == sys.executable
    assert "train_t5_policy.py" in argv[1]
    assert "--smoke" in argv
    assert "ppo.eval_interval=1" in argv
    assert not any("ppo.save_interval" in token for token in argv)
    assert "--skip-supervised" in argv
    assert _flag_value(argv, "--model-path") == str(
        tmp_path / "shared_supervised" / "best_model"
    )
    assert _flag_value(argv, "--seed") == "2"
    assert _flag_value(argv, "--config") == CONFIG_PATH
    assert f"supervised.checkpoint_dir={run_dir}" in argv


# Tests R-011 [integration]: non-smoke argv injects neither --smoke nor the
# eval-interval override.
def test_r011_non_smoke_argv_has_no_injected_overrides(tmp_path: Path) -> None:
    argv = harness.build_child_argv(
        **_argv_kwargs(tmp_path / "A_seed1", arm="A", seed=1, hazard=False,
                       smoke=False)
    )
    assert "--smoke" not in argv
    assert not any("ppo.eval_interval" in token for token in argv)
    assert not any("ppo.save_interval" in token for token in argv)


# Tests R-011/R-003 [integration]: hazard-arm flag mapping (B knobs, C
# ablation, A bare).
def test_r011_arm_flag_mapping(tmp_path: Path) -> None:
    arm_b = harness.build_child_argv(
        **_argv_kwargs(tmp_path / "B_seed1", arm="B", seed=1, hazard=True,
                       beta_terminal=2.0, freeze_answer_head=True)
    )
    assert "--hazard-pretrain" in arm_b
    assert _flag_value(arm_b, "--beta-terminal") == "2.0"
    assert "--freeze-answer-head" in arm_b
    assert "--hazard-ablation" not in arm_b

    arm_c = harness.build_child_argv(
        **_argv_kwargs(tmp_path / "C_seed1", arm="C", seed=1, hazard=True,
                       ablation="shuffled_nll")
    )
    assert "--hazard-pretrain" in arm_c
    assert _flag_value(arm_c, "--hazard-ablation") == "shuffled_nll"

    arm_a = harness.build_child_argv(
        **_argv_kwargs(tmp_path / "A_seed1", arm="A", seed=1, hazard=False)
    )
    assert "--hazard-pretrain" not in arm_a
    assert "--hazard-ablation" not in arm_a
    assert "--freeze-answer-head" not in arm_a


# Tests R-011 [unit]: source-level — the subprocess module is actually
# imported (code-level evidence, not docstring prose) and shell=True never
# appears anywhere in the harness module.
def test_r011_subprocess_used_and_no_shell_true() -> None:
    src = inspect.getsource(harness)
    assert "import subprocess" in src, "children must run via subprocess"
    assert "shell=True" not in src, "argv lists only — never shell strings"


# Tests R-011 [integration]: the REAL _run_child (no monkeypatch) runs the
# argv list via subprocess, tees the child's stdout AND stderr into the
# given log path (QA-008: the stderr half of the tee is asserted, not
# assumed), and returns the child's exit code verbatim.
def test_r011_real_run_child_tees_log_and_returns_exit_code(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "train.log"
    argv = [
        sys.executable,
        "-c",
        "import sys; print('xmarker'); "
        "sys.stderr.write('emarker\\n'); sys.exit(3)",
    ]

    returncode = harness._run_child(argv, log_path)

    assert returncode == 3
    assert log_path.exists(), "child output must be tee'd to the log path"
    log_text = log_path.read_text()
    assert "xmarker" in log_text
    assert "emarker" in log_text, (
        "the child's STDERR must land in the log too (a traceback-only "
        "failure would otherwise leave an empty log)"
    )


# Tests R-011 [integration]: a missing <run_dir>/ppo_t5/best_model after a
# child run raises naming the run and the log path.
def test_r011_missing_best_model_fails_naming_run(tmp_path: Path) -> None:
    record = make_plan_record(tmp_path, "B", 2)
    Path(record["run_dir"]).mkdir(parents=True)
    with pytest.raises(harness.ChildRunError) as excinfo:
        harness.check_child_outputs(record)
    message = str(excinfo.value)
    assert "B" in message and "2" in message
    assert str(record["log_path"]) in message

    # With the checkpoint present the check passes silently.
    ok = make_plan_record(tmp_path, "A", 1)
    make_run_dir(tmp_path, "A", 1, marker=False, write_eval_result=False)
    assert harness.check_child_outputs(ok) is None


# Tests R-011 [integration]: a nonzero child exit fails loud with the run
# name, exit code, log path, and log tail.
def test_r011_nonzero_child_exit_fails_with_log_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = make_plan_record(tmp_path, "B", 1)
    runner, calls = fabricating_runner(
        [record], exit_code=3, log_text="traceback: boom-tail-marker\n"
    )
    monkeypatch.setattr(harness, "_run_child", runner)

    with pytest.raises(harness.ChildRunError) as excinfo:
        harness.execute_plan([record])
    message = str(excinfo.value)
    assert "B" in message
    assert "3" in message
    assert str(record["log_path"]) in message
    assert "boom-tail-marker" in message
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# R-003 arm-control sidecar assertions
# ---------------------------------------------------------------------------


def _control_records(tmp_path: Path, **arm_kwargs) -> list[dict]:
    """A/B/C x seeds 1-2 fixture runs; per-arm make_run_dir overrides."""
    records = []
    for arm in ("A", "B", "C"):
        for seed in (1, 2):
            kwargs = dict(arm_kwargs.get(arm, {}))
            # Vary BOTH checkpoint_dir keys per run: they are exempt.
            kwargs.setdefault("ppo_checkpoint_dir", f"checkpoints/{arm}_{seed}")
            run_dir = make_run_dir(tmp_path, arm, seed, **kwargs)
            records.append({"arm": arm, "seed": seed, "run_dir": run_dir,
                            "hazard": arm != "A"})
    return records


# Tests R-003 [integration]: clean sidecars — equal except hazard block,
# seed, and checkpoint-dir keys — pass the assertion.
def test_r003_clean_sidecars_pass(tmp_path: Path) -> None:
    records = _control_records(tmp_path)
    assert harness.assert_arm_control(records) is None


# Tests R-003 [integration]: a doctored model_name raises naming the arm.
def test_r003_model_name_mismatch_names_offending_arm(tmp_path: Path) -> None:
    records = _control_records(
        tmp_path, B={"model_name": "t5-base"}
    )
    with pytest.raises(harness.ArmControlError) as excinfo:
        harness.assert_arm_control(records)
    assert "B" in str(excinfo.value)


# Tests R-003 [integration]: doctored split qids raise naming the arm.
def test_r003_split_qid_mismatch_names_offending_arm(tmp_path: Path) -> None:
    doctored = {
        "train": DEFAULT_SPLIT_QIDS["train"],
        "val": DEFAULT_SPLIT_QIDS["val"],
        "test": ["q1", "q2", "zz-doctored"],
    }
    records = _control_records(tmp_path, C={"split_qids": doctored})
    with pytest.raises(harness.ArmControlError) as excinfo:
        harness.assert_arm_control(records)
    assert "C" in str(excinfo.value)


# Tests R-003 [integration]: ANY unexpected differing key (e.g. ppo.lr)
# raises naming the arm and the key.
def test_r003_unexpected_key_difference_names_arm_and_key(tmp_path: Path) -> None:
    records = _control_records(
        tmp_path, B={"config_mutations": {"ppo.lr": 5e-4}}
    )
    with pytest.raises(harness.ArmControlError) as excinfo:
        harness.assert_arm_control(records)
    message = str(excinfo.value)
    assert "B" in message
    assert "lr" in message


# ---------------------------------------------------------------------------
# R-005 identical eval path
# ---------------------------------------------------------------------------


def _fake_eval_factory(calls: list):
    """Capture fake returning the REAL evaluate_t5_policy output key names
    (scripts/compare_policies.py) plus the return_runs records."""
    def fake_eval(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        qids = [q.qid for q in kwargs["test_questions"]]
        return {
            "accuracy": 0.5,
            "mean_sq": 0.1,
            "ece": 0.10,
            "brier": 0.20,
            "avg_buzz_pos": 2.0,
            "n_questions": len(qids),
            "test_set_source": "persisted_artifacts",
            "runs": [
                {"qid": qid, "sq": 0.1, "buzz_position": 2, "buzzed": True,
                 "correct": True, "forced_correct": False, "confidence": 0.9,
                 "episode_reward": 1.0, "n_steps": 2}
                for qid in qids
            ],
        }
    return fake_eval


# Tests R-005 [integration]: exactly one evaluate_t5_policy call per
# arm x seed, keyword-only, return_runs=True, identical test qids and
# kwargs across calls except the per-run checkpoint path.
def test_r005_one_call_per_run_identical_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = []
    for arm in ("A", "B"):
        for seed in (1, 2):
            run_dir = make_run_dir(tmp_path, arm, seed, write_eval_result=False)
            records.append({"arm": arm, "seed": seed, "run_dir": run_dir,
                            "hazard": arm != "A"})
    questions = [SimpleNamespace(qid=q) for q in ("q1", "q2", "q3")]

    calls: list = []
    monkeypatch.setattr(
        harness, "evaluate_t5_policy", _fake_eval_factory(calls), raising=False
    )

    harness.evaluate_all_runs(records, questions, {"smoke": True})

    assert len(calls) == 4
    assert all(args == () for args, _ in calls), "call with keywords only"
    assert all(kw["return_runs"] is True for _, kw in calls)

    qid_lists = [[q.qid for q in kw["test_questions"]] for _, kw in calls]
    assert all(qids == qid_lists[0] for qids in qid_lists)

    checkpoint_paths = [str(kw["checkpoint_path"]) for _, kw in calls]
    assert len(set(checkpoint_paths)) == 4, "distinct checkpoints per run"
    for record, path in zip(records, checkpoint_paths):
        assert str(record["run_dir"]) in path
        assert path.endswith("ppo_t5/best_model"), (
            "each run must be evaluated at its PPO best_model checkpoint "
            f"(<run_dir>/ppo_t5/best_model), got {path}"
        )

    stripped = []
    for _, kw in calls:
        rest = {k: v for k, v in kw.items()
                if k not in ("checkpoint_path", "test_questions")}
        stripped.append(rest)
    assert all(rest == stripped[0] for rest in stripped)


# Tests R-005/R-014 [integration]: evaluate_run passes the payload through,
# enriches it with the degeneracy diagnostics, and persists it.
def test_r005_eval_result_enrichment_and_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = make_run_dir(tmp_path, "A", 1, write_eval_result=False)
    record = {"arm": "A", "seed": 1, "run_dir": run_dir, "hazard": False,
              "log_path": run_dir / "train.log"}
    questions = [SimpleNamespace(qid=q) for q in ("q1", "q2", "q3")]

    def fake_eval(*args, **kwargs):
        return {
            "policy_accuracy": 0.5,
            "mean_sq": 0.125,
            "runs": [
                {"qid": "q1", "sq": 0.2, "buzz_position": 4, "buzzed": True,
                 "correct": True, "forced_correct": False, "confidence": 0.8,
                 "episode_reward": 1.0, "n_steps": 4},
                {"qid": "q2", "sq": 0.1, "buzz_position": 2, "buzzed": True,
                 "correct": False, "forced_correct": False, "confidence": 0.6,
                 "episode_reward": -1.0, "n_steps": 2},
                {"qid": "q3", "sq": 0.05, "buzz_position": None, "buzzed": False,
                 "correct": False, "forced_correct": True, "confidence": None,
                 "episode_reward": 0.5, "n_steps": 6},
            ],
        }

    monkeypatch.setattr(harness, "evaluate_t5_policy", fake_eval, raising=False)
    result = harness.evaluate_run(record, questions, {})

    eval_path = run_dir / "eval_result.json"
    assert eval_path.exists(), "eval_result.json must be written per run"
    on_disk = json.loads(eval_path.read_text())

    for payload in (result, on_disk):
        assert payload["policy_accuracy"] == pytest.approx(0.5)  # pass-through
        assert payload["mean_sq"] == pytest.approx(0.125)
        assert payload["arm"] == "A"
        assert payload["seed"] == 1
        assert payload["policy_buzz_rate"] == pytest.approx(2.0 / 3.0)
        assert payload["forced_commit_rate"] == pytest.approx(1.0 / 3.0)
        assert payload["n_correct_policy_buzzes"] == 1
        assert payload["mean_correct_buzz_position"] == pytest.approx(4.0)
        assert len(payload["runs"]) == 3


# ---------------------------------------------------------------------------
# R-008 dir planning + provenance
# ---------------------------------------------------------------------------


# Tests R-008 [integration]: every arm x seed gets a DISTINCT pinned run dir
# and a shared supervised checkpoint path.
def test_r008_distinct_run_dirs_per_arm_seed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    plan = harness.plan_runs(_namespace(out))
    assert len(plan) == 9

    run_dirs = [Path(rec["run_dir"]) for rec in plan]
    assert len(set(run_dirs)) == 9, "run dirs must be distinct per arm x seed"
    for rec in plan:
        expected = out / f"{rec['arm']}_seed{rec['seed']}"
        assert Path(rec["run_dir"]) == expected
        assert Path(rec["log_path"]) == expected / "train.log"
        assert isinstance(rec["argv"], list)

    model_paths = {_flag_value(rec["argv"], "--model-path") for rec in plan}
    assert len(model_paths) == 1, "all arms branch from ONE shared checkpoint"
    shared = model_paths.pop()
    assert str(out) in shared
    assert "supervised" in shared


# Tests R-008/R-011/R-003 [integration]: plan argv CONTENT for the core arms
# — the arm identity flags, the namespace knob values, the per-record seed,
# and the smoke-only injected overrides are all pinned per record.
def test_r008_plan_argv_flag_mapping(tmp_path: Path) -> None:
    out = tmp_path / "out"
    plan = harness.plan_runs(_namespace(out, seeds=[1, 2], beta_terminal=2.5))
    assert len(plan) == 6  # A, B, C x seeds 1, 2

    by_arm: dict[str, list[dict]] = {}
    for rec in plan:
        by_arm.setdefault(rec["arm"], []).append(rec)
    assert set(by_arm) == {"A", "B", "C"}

    for rec in plan:
        argv = rec["argv"]
        # Every record: --seed threads the RECORD's own seed value.
        assert _flag_value(argv, "--seed") == str(rec["seed"])
        # Smoke namespace: --smoke + ppo.eval_interval=1 injected in every
        # child argv; ppo.save_interval never appears (R-011).
        assert "--smoke" in argv
        assert "ppo.eval_interval=1" in argv
        assert not any("ppo.save_interval" in token for token in argv)

    for rec in by_arm["A"]:
        argv = rec["argv"]
        # Arm A is the bare control: NO hazard flag of any kind.
        assert "--hazard-pretrain" not in argv
        assert "--hazard-ablation" not in argv
        assert "--beta-terminal" not in argv
        assert "--freeze-answer-head" not in argv

    for rec in by_arm["B"]:
        argv = rec["argv"]
        assert "--hazard-pretrain" in argv
        assert _flag_value(argv, "--beta-terminal") == "2.5", (
            "arm B must thread the namespace --beta-terminal value"
        )
        assert "--hazard-ablation" not in argv
        # freeze_answer_head=False in the namespace => flag absent (passing
        # it would force freezing on in the child).
        assert "--freeze-answer-head" not in argv

    for rec in by_arm["C"]:
        argv = rec["argv"]
        assert "--hazard-pretrain" in argv
        assert _flag_value(argv, "--hazard-ablation") == "shuffled_nll"


# Tests R-008 [integration]: repeatable --variant NAME:FLAGS adds hazard
# variant runs with their own dirs and extra hazard flags.
# AMENDED in mini-audit fix round (MA-008): variants now live in a namespace
# DISTINCT from the role-bearing core arms — run dirs are
# variant_<NAME>_seed<k>, arm label variant:<NAME> — and they INHERIT the
# invocation's hazard knobs with FLAGS overriding (an overridden knob is not
# re-injected, so each knob flag appears exactly once).
def test_r008_variant_plan_adds_hazard_variant_runs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    args = _namespace(
        out, seeds=[1, 2],
        variant=["Bfz:--beta-terminal 2.0 --freeze-answer-head"],
    )
    plan = harness.plan_runs(args)
    assert len(plan) == 8  # (A, B, C, Bfz) x 2 seeds

    variants = [rec for rec in plan if rec["arm"] == "variant:Bfz"]
    assert len(variants) == 2
    for rec in variants:
        assert rec["variant"] == "Bfz"
        assert Path(rec["run_dir"]) == out / f"variant_Bfz_seed{rec['seed']}"
        argv = rec["argv"]
        assert "--hazard-pretrain" in argv
        # FLAGS override the inherited knobs: exactly one occurrence each.
        assert argv.count("--beta-terminal") == 1
        assert _flag_value(argv, "--beta-terminal") == "2.0"
        assert argv.count("--freeze-answer-head") == 1
        # MA-001: the planned hazard identity mirrors the parsed argv.
        assert rec["hazard_knobs"] == {
            "pretrain": True,
            "beta_terminal": 2.0,
            "freeze_answer_head": True,
            "ablation": None,
        }


# Tests R-008 [unit]: defensive — variant names that would escape the out
# dir are rejected at plan time.
@pytest.mark.parametrize("bad_name", ["b/z", ".."])
def test_r008_variant_name_path_safety(tmp_path: Path, bad_name: str) -> None:
    args = _namespace(tmp_path / "out", variant=[f"{bad_name}:--freeze-answer-head"])
    with pytest.raises(harness.PreflightError):
        harness.plan_runs(args)


# Tests R-008 [integration]: provenance is complete (real git/torch/platform)
# or fails loud with no report.
def test_r008_provenance_complete_or_fail_loud(tmp_path: Path) -> None:
    from tests._hazard_efficacy_fixtures import make_config_used

    config_used = make_config_used(tmp_path / "A_seed1", arm="A", seed=1)
    prov = harness.collect_provenance(config_used)
    assert prov["model_name"] == "t5-small"
    assert prov["seed"] == 1
    assert prov["device"] == "cpu"
    assert prov["git_sha"] == current_git_sha()
    assert isinstance(prov["git_dirty"], bool)
    assert prov["torch_version"]
    assert prov["platform"]

    broken = make_config_used(tmp_path / "A_seed1", arm="A", seed=1)
    del broken["model"]["model_name"]
    with pytest.raises(harness.ProvenanceError):
        harness.collect_provenance(broken)


# Tests R-008 [integration]: the supervised split manifest is persisted next
# to the shared checkpoint and train/test disjointness is asserted.
def test_r008_supervised_manifest_persisted_and_disjoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "shared_supervised" / "best_model"
    ckpt.mkdir(parents=True)

    manifest = make_split_manifest()
    written = harness.write_supervised_split_manifest(ckpt, manifest)
    assert Path(written).exists()
    assert Path(written).name == "split_manifest.json"
    assert Path(written).parent == ckpt
    on_disk = json.loads(Path(written).read_text())
    assert on_disk["train_qids"] == manifest["train_qids"]
    assert on_disk["test_qids"] == manifest["test_qids"]

    overlapping = make_split_manifest(
        split_qids={"train": ["t1", "q1"], "val": ["v1"], "test": ["q1", "q2"]}
    )
    with pytest.raises(harness.ProvenanceError):
        harness.write_supervised_split_manifest(ckpt, overlapping)


# ---------------------------------------------------------------------------
# R-013 preflight / dry-run / resume
# ---------------------------------------------------------------------------


# Tests R-013 [integration]: missing split artifacts produce an actionable
# error naming the missing location and build_mc_dataset.
def test_r013_missing_split_artifacts_actionable_error(tmp_path: Path) -> None:
    empty = tmp_path / "no_artifacts"
    empty.mkdir()
    with pytest.raises(harness.PreflightError) as excinfo:
        harness.resolve_split_artifacts(smoke=True, search_dirs=[empty])
    message = str(excinfo.value)
    assert "build_mc_dataset" in message
    assert str(empty) in message

    populated = tmp_path / "artifacts"
    populated.mkdir()
    for name in ("train_dataset.json", "val_dataset.json", "test_dataset.json"):
        (populated / name).write_text("[]")
    paths = harness.resolve_split_artifacts(smoke=True, search_dirs=[populated])
    assert set(paths) == {"train", "val", "test"}
    assert paths["train"] == populated / "train_dataset.json"


# Tests R-013 [integration]: the preflight runs BEFORE any child launch —
# zero subprocess invocations when it fails.
def test_r013_preflight_blocks_before_any_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launches: list = []
    monkeypatch.setattr(
        harness, "_run_child", lambda argv, log_path: launches.append(argv) or 0
    )

    def boom(**kwargs):
        raise harness.PreflightError(
            "split artifacts missing; run scripts/build_mc_dataset.py --smoke"
        )

    monkeypatch.setattr(harness, "resolve_split_artifacts", boom)
    with pytest.raises((harness.PreflightError, SystemExit)):
        harness.main(["--smoke", "--out-dir", str(tmp_path / "out")])
    assert launches == []


# Tests R-013 [integration]: --dry-run prints the full plan and launches
# zero children.
def test_r013_dry_run_prints_plan_zero_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    launches: list = []
    monkeypatch.setattr(
        harness, "_run_child", lambda argv, log_path: launches.append(argv) or 0
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    split_paths = {}
    for split in ("train", "val", "test"):
        p = artifacts / f"{split}_dataset.json"
        p.write_text("[]")
        split_paths[split] = p
    monkeypatch.setattr(
        harness, "resolve_split_artifacts", lambda **kwargs: split_paths
    )

    harness.main(["--smoke", "--dry-run", "--out-dir", str(tmp_path / "out")])

    assert launches == []
    output = capsys.readouterr().out
    assert "A_seed1" in output
    assert "C_seed3" in output


# Tests R-013 [integration]: run-dir classification — fresh / partial /
# complete (hazard arms also need hazard/best_model).
def test_r013_classify_run_dir_states(tmp_path: Path) -> None:
    assert harness.classify_run_dir(tmp_path / "missing", hazard=False) == "fresh"

    complete = make_run_dir(tmp_path, "A", 1)
    assert harness.classify_run_dir(complete, hazard=False) == "complete"

    partial = make_run_dir(tmp_path, "A", 2, marker=False)
    assert harness.classify_run_dir(partial, hazard=False) == "partial"

    # Hazard arm without hazard/best_model is NOT complete even with marker.
    hazardless = make_run_dir(tmp_path, "B", 1, hazard=False)
    assert harness.classify_run_dir(hazardless, hazard=True) != "complete"


# Tests R-013 [integration]: complete runs are skipped (zero child calls),
# marked resumed, and still pass the sidecar assertions.
def test_r013_complete_runs_resumed_without_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [make_plan_record(tmp_path, arm, 1) for arm in ("A", "B")]
    for rec in records:
        make_run_dir(tmp_path, rec["arm"], 1)

    runner, calls = fabricating_runner(records)
    monkeypatch.setattr(harness, "_run_child", runner)

    updated = harness.execute_plan(records)
    assert calls == []
    assert all(rec["resumed"] is True for rec in updated)


# Tests R-013 [integration]: a resumed dir is still subject to the full
# R-003 sidecar assertions — a stale/doctored dir fails loud.
def test_r013_resumed_doctored_dir_fails_arm_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [make_plan_record(tmp_path, arm, 1) for arm in ("A", "B")]
    make_run_dir(tmp_path, "A", 1)
    make_run_dir(tmp_path, "B", 1, model_name="t5-base")  # stale config

    runner, calls = fabricating_runner(records)
    monkeypatch.setattr(harness, "_run_child", runner)

    with pytest.raises(harness.ArmControlError) as excinfo:
        harness.execute_plan(records)
    assert "B" in str(excinfo.value)
    assert calls == []


# Tests R-013 [integration]: a partial dir (outputs, no marker) fails loud
# with delete-or---force instructions.
def test_r013_partial_dir_fails_loud_with_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = make_plan_record(tmp_path, "A", 1)
    make_run_dir(tmp_path, "A", 1, marker=False)

    runner, calls = fabricating_runner([record])
    monkeypatch.setattr(harness, "_run_child", runner)

    with pytest.raises(harness.PartialRunError) as excinfo:
        harness.execute_plan([record])
    message = str(excinfo.value)
    assert "--force" in message
    assert calls == []


# Tests R-013 [integration]: --force re-runs everything, complete dirs
# included.
def test_r013_force_reruns_complete_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [make_plan_record(tmp_path, arm, 1) for arm in ("A", "B")]
    for rec in records:
        make_run_dir(tmp_path, rec["arm"], 1)

    runner, calls = fabricating_runner(records)
    monkeypatch.setattr(harness, "_run_child", runner)

    updated = harness.execute_plan(records, force=True)
    assert len(calls) == 2
    assert all(rec["resumed"] is False for rec in updated)


# Tests R-013 [integration]: fresh dirs invoke one child each; the harness
# (not the child) writes RUN_COMPLETE.json after exit 0 + checkpoint check.
# EXTENDED per PR #41 review r3806602894: the marker additionally carries the
# TRAINING-time provenance snapshot (git_dirty/torch_version/platform/device).
def test_r013_fresh_runs_invoke_children_and_write_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import platform as platform_mod

    import torch

    records = [make_plan_record(tmp_path, arm, 1) for arm in ("A", "B")]
    runner, calls = fabricating_runner(records)
    monkeypatch.setattr(harness, "_run_child", runner)

    updated = harness.execute_plan(records)
    assert len(calls) == 2
    for rec in updated:
        assert rec["resumed"] is False
        marker_path = Path(rec["run_dir"]) / "RUN_COMPLETE.json"
        assert marker_path.exists(), "harness writes the completion marker"
        marker = json.loads(marker_path.read_text())
        assert {"git_sha", "arm", "seed", "wall_clock_seconds", "smoke",
                "git_dirty", "torch_version", "platform",
                "device"} <= set(marker)
        assert marker["arm"] == rec["arm"]
        # QA-R2-2: the marker persists the invocation's smoke flag (the
        # direct-caller default here is False).
        assert marker["smoke"] is False
        # The marker records the child's MEASURED elapsed seconds — a real
        # float >= 0 (the fabricating runner returns near-instantly).
        wall_clock = marker["wall_clock_seconds"]
        assert isinstance(wall_clock, float) and not isinstance(wall_clock, bool)
        assert wall_clock >= 0.0
        # PR #41 r3806602894: the TRAINING-time provenance snapshot is
        # captured at completion — real git dirty state, the training
        # interpreter's torch, this machine's platform, and the child's
        # RESOLVED device from its own config_used.json (fixture: "cpu").
        assert isinstance(marker["git_dirty"], bool)
        assert marker["torch_version"] == str(torch.__version__)
        assert marker["platform"] == platform_mod.platform()
        assert marker["device"] == "cpu"


# Tests R-013 [integration]: the silent random-split fallback invalidates
# provenance — split_manifest.source must be persisted_artifacts.
def test_r013_random_split_fallback_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = make_plan_record(tmp_path, "A", 1)
    runner, calls = fabricating_runner(
        [record], split_source="random_split_fallback"
    )
    monkeypatch.setattr(harness, "_run_child", runner)

    with pytest.raises(harness.HarnessError) as excinfo:
        harness.execute_plan([record])
    assert "persisted_artifacts" in str(excinfo.value)


# Tests R-013 [integration]: a git-SHA mismatch on a reused run is recorded
# and warned — never a silent mix, never fatal.
def test_r013_git_sha_mismatch_recorded_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [make_plan_record(tmp_path, "A", 1)]
    make_run_dir(tmp_path, "A", 1, marker_git_sha="0" * 40)

    runner, calls = fabricating_runner(records)
    monkeypatch.setattr(harness, "_run_child", runner)

    updated = harness.execute_plan(records)
    assert calls == []
    assert updated[0]["resumed"] is True
    assert updated[0]["git_sha_mismatch"] is True


# ---------------------------------------------------------------------------
# R-014 eval persistence / report-only / prune
# ---------------------------------------------------------------------------


# Tests R-014 [integration]: --report-only reassembles report + plot from
# existing run dirs with ZERO training or eval calls (and no split preflight).
def test_r014_report_only_zero_children_zero_evals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    for arm in ("A", "B"):
        make_run_dir(out, arm, 1, include_hazard_dynamics=(arm == "B"))

    launches: list = []
    evals: list = []
    monkeypatch.setattr(
        harness, "_run_child", lambda argv, log_path: launches.append(argv) or 0
    )
    monkeypatch.setattr(
        harness,
        "evaluate_t5_policy",
        lambda *a, **k: evals.append(1) or {},
        raising=False,
    )

    def no_preflight(**kwargs):
        raise harness.PreflightError("report-only must not need split artifacts")

    monkeypatch.setattr(harness, "resolve_split_artifacts", no_preflight)

    harness.main(
        ["--report-only", "--smoke", "--out-dir", str(out),
         "--arms", "A", "B", "--seeds", "1"]
    )

    assert launches == []
    assert evals == []
    assert (out / "hazard_efficacy_report.json").exists()
    assert (out / "hazard_efficacy_plot.png").exists()


# Tests R-014 [integration]: report assembly reads EXCLUSIVELY the per-run
# eval_result.json files — it works with no checkpoints on disk and never
# calls the eval entry point.
def test_r014_assembly_reads_only_per_run_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    records = []
    for arm in ("A", "B"):
        run_dir = make_run_dir(
            out, arm, 1,
            write_checkpoints=False,  # nothing but sidecars + eval files
            include_hazard_dynamics=(arm == "B"),
        )
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A", "resumed": True})

    def boom(*args, **kwargs):
        raise AssertionError("assembly must not evaluate")

    monkeypatch.setattr(harness, "evaluate_t5_policy", boom, raising=False)

    report = harness.assemble_report(out, records, smoke=True)
    assert report["schema_version"] == 1
    assert len(report["runs"]) == 2


# Tests R-014 [integration]: a failure later in the eval stage never deletes
# or skips already-written per-run eval artifacts.
def test_r014_eval_artifacts_survive_downstream_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = []
    for arm in ("A", "B"):
        run_dir = make_run_dir(tmp_path, arm, 1, write_eval_result=False)
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A"})
    questions = [SimpleNamespace(qid=q) for q in ("q1", "q2", "q3")]

    calls: list = []
    good = _fake_eval_factory(calls)

    def flaky(*args, **kwargs):
        if calls:
            raise RuntimeError("second eval exploded")
        return good(*args, **kwargs)

    monkeypatch.setattr(harness, "evaluate_t5_policy", flaky, raising=False)

    with pytest.raises(RuntimeError):
        harness.evaluate_all_runs(records, questions, {})

    first_eval = Path(records[0]["run_dir"]) / "eval_result.json"
    assert first_eval.exists(), "run 1's eval artifact must survive run 2's failure"


# Tests R-014 [integration]: prune deletes iter_N/epoch_N/training_state.pt
# but keeps best_model weights, sidecars, history, and eval_result.json.
def test_r014_prune_reclaims_only_regenerable_state(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, "B", 1, include_prunables=True)
    ppo = run_dir / "ppo_t5"

    harness.prune_run_checkpoints(run_dir)

    assert not (ppo / "iter_1").exists()
    assert not (ppo / "iter_2").exists()
    assert not (ppo / "epoch_1").exists()
    assert not list(run_dir.rglob("training_state.pt")), (
        "every optimizer-state file is reclaimed"
    )
    assert (ppo / "best_model" / "policy_head.pt").exists()
    assert (run_dir / "hazard" / "best_model" / "policy_head.pt").exists()
    assert (ppo / "config_used.json").exists()
    assert (ppo / "split_manifest.json").exists()
    assert (ppo / "history.json").exists()
    assert (run_dir / "eval_result.json").exists()


# Tests R-014 [integration]: prune refuses (deletes nothing) while
# eval_result.json is absent — the report must stay regenerable.
def test_r014_prune_refuses_without_eval_result(tmp_path: Path) -> None:
    run_dir = make_run_dir(
        tmp_path, "B", 1, include_prunables=True, write_eval_result=False
    )
    with contextlib.suppress(harness.HarnessError):
        harness.prune_run_checkpoints(run_dir)
    assert (run_dir / "ppo_t5" / "iter_1").exists(), (
        "nothing may be deleted before eval_result.json exists"
    )


# ---------------------------------------------------------------------------
# main() happy-path composition (R-005 / R-008 / R-011 / R-013 / R-014)
# ---------------------------------------------------------------------------


def _sup_ckpt_root(argv: list[str]) -> Path:
    """The ``supervised.checkpoint_dir=<root>`` override value of an argv."""
    overrides = [
        token for token in argv
        if token.startswith("supervised.checkpoint_dir=")
    ]
    assert overrides, (
        "every child argv must direct its outputs via a "
        f"supervised.checkpoint_dir=<root> override; argv={argv}"
    )
    return Path(overrides[-1].split("=", 1)[1])


def _composition_runner(calls: list):
    """Fabricating ``_run_child`` for full main() runs (no pre-known plan).

    Arm children (``--skip-supervised`` present) get real-producer-shaped
    run dirs via ``make_run_dir``; the ONE supervised child gets the real
    trainer's layout ``<root>/supervised/best_model`` (see
    ``training/train_supervised_t5.py``: best model saved under
    ``<checkpoint_dir>/supervised/best_model/``).
    """

    def runner(argv, log_path):
        argv = [str(token) for token in argv]
        log_path = Path(log_path)
        calls.append((argv, log_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("child ok\n")
        root = _sup_ckpt_root(argv)
        if "--skip-supervised" in argv:
            arm, _, seed = root.name.rpartition("_seed")
            make_run_dir(
                root.parent,
                arm,
                int(seed),
                hazard="--hazard-pretrain" in argv,
                marker=False,
                write_eval_result=False,
            )
        else:
            best = root / "supervised" / "best_model"
            best.mkdir(parents=True, exist_ok=True)
            (best / "policy_head.pt").write_bytes(b"stub-weights")
            write_json(root / "supervised" / "history.json", [])
        return 0

    return runner


# Tests R-005/R-008/R-011/R-013/R-014 [integration]: the full main() smoke
# happy path — real split artifacts on disk, real config file, children and
# eval faked at the pinned seams. Pins: (i) eval receives the TEST
# artifact's questions (NOT train's — R-005's Through: the test split is
# resolved from the shared persisted-artifact manifest); (ii) exactly ONE
# supervised child runs, strictly BEFORE every arm child, and every arm
# branches from ITS checkpoint; (iii) report + plot exist afterward.
def test_main_smoke_happy_path_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    split_paths = write_split_artifacts(tmp_path / "artifacts")
    monkeypatch.setattr(
        harness,
        "resolve_split_artifacts",
        lambda **kwargs: dict(split_paths),
    )

    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(harness, "_run_child", _composition_runner(calls))

    eval_calls: list = []
    monkeypatch.setattr(
        harness, "evaluate_t5_policy", _fake_eval_factory(eval_calls),
        raising=False,
    )

    # The R-010b probe stage must route through probe_and_write_hazard_
    # dynamics (fabricated checkpoints are not loadable model weights).
    # QA-008: the monkeypatched seam records its calls so the composition
    # can assert the probe stage actually ran (a silently skipped stage
    # would otherwise pass).
    probe_calls: list = []

    def fake_probe(*args, **kwargs):
        probe_calls.append((args, kwargs))
        out_path = Path(
            kwargs["out_path"] if "out_path" in kwargs else args[4]
        )
        block = make_hazard_dynamics()
        write_json(out_path, block)
        return block

    monkeypatch.setattr(harness, "probe_and_write_hazard_dynamics", fake_probe)

    harness.main(
        [
            "--smoke",
            "--out-dir", str(out),
            "--config", CONFIG_PATH,
            "--seeds", "1",
            "--arms", "A", "B", "C",
        ]
    )

    # (ii) exactly ONE supervised child, strictly before ALL arm children.
    supervised_calls = [
        (argv, log) for argv, log in calls if "--skip-supervised" not in argv
    ]
    arm_calls = [
        (argv, log) for argv, log in calls if "--skip-supervised" in argv
    ]
    assert len(supervised_calls) == 1, (
        "the shared supervised warm-start must run exactly once"
    )
    assert len(arm_calls) == 3  # A, B, C x seed 1
    assert "--skip-supervised" not in calls[0][0], (
        "the supervised child must run BEFORE any arm child"
    )
    assert all("--skip-supervised" in argv for argv, _ in calls[1:])

    # Every arm branches from the ONE supervised checkpoint the shared
    # child produced (<root>/supervised/best_model — the real trainer's
    # layout).
    shared_ckpt = str(
        _sup_ckpt_root(supervised_calls[0][0]) / "supervised" / "best_model"
    )
    for argv, log_path in arm_calls:
        assert _flag_value(argv, "--model-path") == shared_ckpt
        assert log_path == _sup_ckpt_root(argv) / "train.log"

    # (i) eval runs once per arm on the TEST artifact's questions.
    assert len(eval_calls) == 3
    test_qids = sorted(
        rec["qid"] for rec in json.loads(split_paths["test"].read_text())
    )
    train_qids = sorted(
        rec["qid"] for rec in json.loads(split_paths["train"].read_text())
    )
    assert test_qids and test_qids != train_qids  # fixture sanity
    for _, kwargs in eval_calls:
        got = sorted(q.qid for q in kwargs["test_questions"])
        assert got == test_qids, (
            "eval must receive the TEST artifact's questions resolved from "
            f"the shared manifest; got {got}"
        )
        assert got != train_qids, "evaluating on the TRAIN split is R-005-fatal"

    # (iii) the report and plot exist afterward.
    assert (out / "hazard_efficacy_report.json").exists()
    assert (out / "hazard_efficacy_plot.png").exists()
    report = json.loads((out / "hazard_efficacy_report.json").read_text())
    assert {rec["arm"] for rec in report["runs"]} == {"A", "B", "C"}

    # (iv) QA-008: the probe seam was invoked once per HAZARD run (arms B
    # and C x seed 1) and the report's hazard_dynamics block is non-null.
    assert len(probe_calls) == 2, (
        "probe_and_write_hazard_dynamics must run once per hazard run; "
        f"got {len(probe_calls)} call(s)"
    )
    assert isinstance(report["hazard_dynamics"], dict) and report[
        "hazard_dynamics"
    ], "the report must carry a non-null hazard_dynamics block"


# ---------------------------------------------------------------------------
# QA fix round 1 regressions (QA-001..QA-005, QA-007..QA-010, QA-012)
# ---------------------------------------------------------------------------


def _supervised_fabricating_runner(calls: list):
    """Fake ``_run_child`` for direct ``_run_shared_supervised`` tests."""

    def runner(argv, log_path):
        argv = [str(token) for token in argv]
        calls.append(argv)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("supervised ok\n")
        best = _sup_ckpt_root(argv) / "supervised" / "best_model"
        best.mkdir(parents=True, exist_ok=True)
        (best / "policy_head.pt").write_bytes(b"stub-weights")
        return 0

    return runner


# Tests QA-001 [integration]: invalidate-before-mutate — force -> crash ->
# resume must raise PartialRunError, never resume a half-trained dir under
# the OLD completion marker.
def test_qa001_force_crash_then_resume_raises_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = make_plan_record(tmp_path, "A", 1)
    make_run_dir(tmp_path, "A", 1)  # complete: marker + eval_result on disk
    run_dir = Path(record["run_dir"])
    assert (run_dir / "RUN_COMPLETE.json").exists()
    assert (run_dir / "eval_result.json").exists()

    crash_runner, crash_calls = fabricating_runner([record], exit_code=3)
    monkeypatch.setattr(harness, "_run_child", crash_runner)
    with pytest.raises(harness.ChildRunError):
        harness.execute_plan([record], force=True)
    assert len(crash_calls) == 1
    # The stale markers were unlinked BEFORE the child launched, so the
    # crashed dir is an honest partial — not a fake complete.
    assert not (run_dir / "RUN_COMPLETE.json").exists()
    assert not (run_dir / "eval_result.json").exists()

    ok_runner, ok_calls = fabricating_runner([record])
    monkeypatch.setattr(harness, "_run_child", ok_runner)
    with pytest.raises(harness.PartialRunError):
        harness.execute_plan([record])  # resume WITHOUT --force
    assert ok_calls == []


# Tests QA-002 [integration]: resume answers "does this dir match what THIS
# invocation would produce" — a stale model or drifted split qids fail loud;
# a matching dir resumes cleanly.
# AMENDED per PR #41 review r3806602891: the split check is ORDERED EQUALITY
# per split against the current invocation's own capped selection
# (expected_run_context["split_qids"]), no longer membership in the raw
# artifacts.
def test_qa002_stale_resumed_dir_vs_current_invocation_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "model_name": "t5-small",
        "split_qids": {
            split: list(qids) for split, qids in DEFAULT_SPLIT_QIDS.items()
        },
    }

    # (a) stale model identity: dir trained for t5-base, invocation t5-small.
    records = [make_plan_record(tmp_path / "a", "A", 1)]
    make_run_dir(tmp_path / "a", "A", 1, model_name="t5-base")
    runner, calls = fabricating_runner(records)
    monkeypatch.setattr(harness, "_run_child", runner)
    with pytest.raises(harness.ProvenanceError) as excinfo:
        harness.execute_plan(records, expected_run_context=expected)
    message = str(excinfo.value)
    assert "t5-base" in message and "t5-small" in message
    assert calls == []

    # (b) split drift: the resumed manifest names a qid outside the split
    # the CURRENT invocation would select (dir predates an artifact rebuild).
    records = [make_plan_record(tmp_path / "b", "A", 1)]
    make_run_dir(
        tmp_path / "b", "A", 1,
        split_qids={"train": ["OLD-qid"], "val": ["v1"], "test": ["q1"]},
    )
    runner, calls = fabricating_runner(records)
    monkeypatch.setattr(harness, "_run_child", runner)
    with pytest.raises(harness.ProvenanceError) as excinfo:
        harness.execute_plan(records, expected_run_context=expected)
    assert "OLD-qid" in str(excinfo.value)
    assert calls == []

    # (c) a dir matching the current invocation resumes with zero children.
    records = [make_plan_record(tmp_path / "c", "A", 1)]
    make_run_dir(tmp_path / "c", "A", 1)
    runner, calls = fabricating_runner(records)
    monkeypatch.setattr(harness, "_run_child", runner)
    updated = harness.execute_plan(records, expected_run_context=expected)
    assert calls == []
    assert updated[0]["resumed"] is True


# Tests PR #41 review r3806602891 [integration]: a resumed manifest with the
# SAME qid set as the current selection but a different order (rebuilt/
# reordered artifacts; data.max_questions would select a different prefix)
# is rejected with ProvenanceError — membership alone must never pass it.
def test_pr41_resumed_manifest_same_set_different_order_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "split_qids": {
            split: list(qids) for split, qids in DEFAULT_SPLIT_QIDS.items()
        },
    }
    reordered = {
        "train": list(reversed(DEFAULT_SPLIT_QIDS["train"])),  # same SET
        "val": list(DEFAULT_SPLIT_QIDS["val"]),
        "test": list(DEFAULT_SPLIT_QIDS["test"]),
    }
    assert set(reordered["train"]) == set(DEFAULT_SPLIT_QIDS["train"])
    assert reordered["train"] != DEFAULT_SPLIT_QIDS["train"]

    records = [make_plan_record(tmp_path, "A", 1)]
    make_run_dir(tmp_path, "A", 1, split_qids=reordered)
    runner, calls = fabricating_runner(records)
    monkeypatch.setattr(harness, "_run_child", runner)

    with pytest.raises(harness.ProvenanceError) as excinfo:
        harness.execute_plan(records, expected_run_context=expected)
    message = str(excinfo.value)
    assert "ORDER" in message, message
    assert "A_seed1" in message
    assert calls == [], "the stale dir must be rejected, never re-trained"


# Tests QA-002 [integration]: the shared supervised checkpoint carries an
# identity marker validated on every reuse — mismatch or absence fails loud.
def test_qa002_shared_supervised_marker_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(harness, "_run_child", _supervised_fabricating_runner(calls))
    out = tmp_path / "out"
    args = _namespace(out)

    # Fresh build writes the identity marker beside the checkpoint root.
    harness._run_shared_supervised(args, out)
    assert len(calls) == 1
    marker_path = (
        harness.shared_supervised_root(out) / harness.SHARED_SUPERVISED_MARKER
    )
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text())
    assert {"config_hash", "git_sha", "model_name",
            "supervised_seed"} <= set(marker)
    assert marker["model_name"] == "t5-small"  # smoke resolution of the YAML
    assert marker["git_sha"] == current_git_sha()
    # QA-R2-3: the build seed (seeds[0]) is recorded in the identity marker.
    assert marker["supervised_seed"] == 1

    # Matching marker => reuse, zero further children.
    harness._run_shared_supervised(args, out)
    assert len(calls) == 1

    # Doctored model_name => fail loud (a t5-base checkpoint must never
    # silently seed a t5-small comparison or vice versa).
    marker_path.write_text(json.dumps({**marker, "model_name": "t5-base"}))
    with pytest.raises(harness.ProvenanceError) as excinfo:
        harness._run_shared_supervised(args, out)
    assert "t5-base" in str(excinfo.value)

    # Doctored config hash => fail loud.
    marker_path.write_text(json.dumps({**marker, "config_hash": "0" * 64}))
    with pytest.raises(harness.ProvenanceError):
        harness._run_shared_supervised(args, out)

    # Missing marker on an existing checkpoint => unvalidatable => fail loud.
    marker_path.unlink()
    with pytest.raises(harness.ProvenanceError) as excinfo:
        harness._run_shared_supervised(args, out)
    assert "--force" in str(excinfo.value)
    assert len(calls) == 1, "no silent rebuild on validation failure"


# Tests QA-002 [integration]: --force rebuilds the shared supervised
# checkpoint (and refreshes its identity marker) instead of reusing it.
def test_qa002_force_rebuilds_shared_supervised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(harness, "_run_child", _supervised_fabricating_runner(calls))
    out = tmp_path / "out"

    harness._run_shared_supervised(_namespace(out), out)
    assert len(calls) == 1

    harness._run_shared_supervised(_namespace(out, force=True), out)
    assert len(calls) == 2, "--force must re-run the shared supervised child"
    marker_path = (
        harness.shared_supervised_root(out) / harness.SHARED_SUPERVISED_MARKER
    )
    assert marker_path.exists(), "the rebuilt checkpoint gets a fresh marker"


# Tests QA-003 [integration]: a config-override variant composes an argv the
# REAL child parser accepts, with every positional key=value override
# contiguous at the argv tail.
# AMENDED in mini-audit fix round (MA-008): variant run dirs/arm labels moved
# to the distinct variant_<NAME>_seed<k> / variant:<NAME> namespace.
def test_qa003_variant_positional_override_argv_parses_and_stays_tail(
    tmp_path: Path,
) -> None:
    import scripts.train_t5_policy as train_t5_policy

    out = tmp_path / "out"
    plan = harness.plan_runs(
        _namespace(out, seeds=[1], variant=["lr_sweep:ppo.lr=2e-5"])
    )
    variant = next(rec for rec in plan if rec["arm"] == "variant:lr_sweep")
    argv = variant["argv"]

    positional = [t for t in argv[2:] if "=" in t and not t.startswith("-")]
    assert positional == [
        "ppo.lr=2e-5",
        "ppo.eval_interval=1",
        f"supervised.checkpoint_dir={out / 'variant_lr_sweep_seed1'}",
    ]
    assert argv[-len(positional):] == positional, (
        "positional overrides must be CONTIGUOUS at the argv tail; "
        f"argv={argv}"
    )

    # The REAL parser accepts every planned argv and receives ALL the
    # positional overrides in its trailing ``overrides`` group.
    for rec in plan:
        parsed = train_t5_policy.parse_args(
            argv=[str(t) for t in rec["argv"][2:]]
        )
        assert f"supervised.checkpoint_dir={rec['run_dir']}" in parsed.overrides
    parsed = train_t5_policy.parse_args(argv=[str(t) for t in argv[2:]])
    assert "ppo.lr=2e-5" in parsed.overrides
    assert "ppo.eval_interval=1" in parsed.overrides
    assert parsed.smoke is True
    assert parsed.hazard_pretrain is True


# Tests QA-003 [integration]: plan_runs round-trips every argv through the
# real child parser at preflight — a bad variant flag dies at PLAN time
# (zero children), never after the shared supervised phase + nine runs.
def test_qa003_preflight_rejects_argv_the_real_parser_rejects(
    tmp_path: Path,
) -> None:
    args = _namespace(tmp_path / "out", variant=["bad:--no-such-flag"])
    with pytest.raises(harness.PreflightError) as excinfo:
        harness.plan_runs(args)
    message = str(excinfo.value)
    assert "bad_seed1" in message
    assert "no-such-flag" in message


# Tests QA-004 [integration]: with artifacts LARGER than the children's
# data.max_questions-capped split, the harness selects eval/probe/manifest
# questions BY the child split manifest (manifest order) — never the raw
# artifact lists.
# AMENDED per PR #41 review r3806602891: the harness now resolves the capped
# selection ITSELF through the child trainer's own loader and asserts the
# first run's manifest EQUALS it (ordered) before subsetting — so the test
# runs under a config whose cap really bites (max_questions=8 over 13
# artifact questions; the real global-scope allocation of 6/2/5 at cap 8 is
# exactly 4/1/3 = the fixture DEFAULT_SPLIT_QIDS) and asserts ordered
# equality end to end.
def test_qa004_eval_probe_and_manifest_follow_child_split_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    capped_qids = dict(DEFAULT_SPLIT_QIDS)  # what the children trained on
    big_qids = {
        "train": [f"t{i}" for i in range(1, 7)],  # t1..t6 (artifacts larger)
        "val": ["v1", "v2"],
        "test": [f"q{i}" for i in range(1, 6)],  # q1..q5
    }
    for split in ("train", "val", "test"):
        assert set(capped_qids[split]) < set(big_qids[split])  # strict subset

    # A real config whose smoke cap bites on the 13-question artifacts: the
    # trainer's global-scope allocation of (6, 2, 5) under max_questions=8
    # is (4, 1, 3) — the exact capped_qids prefixes above.
    config = yaml.safe_load(Path(CONFIG_PATH).read_text())
    config["smoke"]["data"]["max_questions"] = 8
    capped_config = tmp_path / "t5_policy_capped.yaml"
    capped_config.write_text(yaml.safe_dump(config))

    out = tmp_path / "out"
    split_paths = write_split_artifacts(tmp_path / "artifacts", split_qids=big_qids)
    monkeypatch.setattr(
        harness, "resolve_split_artifacts", lambda **kwargs: dict(split_paths)
    )

    def runner(argv, log_path):
        argv = [str(token) for token in argv]
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("child ok\n")
        root = _sup_ckpt_root(argv)
        if "--skip-supervised" in argv:
            arm, _, seed = root.name.rpartition("_seed")
            make_run_dir(
                root.parent, arm, int(seed),
                hazard="--hazard-pretrain" in argv,
                marker=False, write_eval_result=False,
                split_qids=capped_qids,  # children honored the cap
            )
        else:
            best = root / "supervised" / "best_model"
            best.mkdir(parents=True, exist_ok=True)
            (best / "policy_head.pt").write_bytes(b"stub-weights")
        return 0

    monkeypatch.setattr(harness, "_run_child", runner)

    eval_calls: list = []
    monkeypatch.setattr(
        harness, "evaluate_t5_policy", _fake_eval_factory(eval_calls),
        raising=False,
    )

    probe_calls: list = []

    def fake_probe(*args, **kwargs):
        probe_calls.append((args, kwargs))
        out_path = Path(kwargs["out_path"] if "out_path" in kwargs else args[4])
        block = make_hazard_dynamics()
        write_json(out_path, block)
        return block

    monkeypatch.setattr(harness, "probe_and_write_hazard_dynamics", fake_probe)

    harness.main(
        ["--smoke", "--out-dir", str(out), "--config", str(capped_config),
         "--seeds", "1", "--arms", "A", "B"]
    )

    # Eval receives the CHILD manifest's capped test/train splits, in
    # manifest order (ORDERED list equality — PR #41 r3806602891) — never
    # the larger raw artifact lists.
    assert len(eval_calls) == 2
    for _, kwargs in eval_calls:
        got_test = [q.qid for q in kwargs["test_questions"]]
        assert got_test == capped_qids["test"]
        assert got_test != big_qids["test"]
        got_ref = [q.qid for q in kwargs["reference_questions"]]
        assert got_ref == capped_qids["train"]

    # The probe sample is drawn from the capped train subset.
    probe_args, probe_kwargs = probe_calls[0]
    probe_questions = (
        probe_kwargs["questions"] if "questions" in probe_kwargs else probe_args[2]
    )
    assert [q.qid for q in probe_questions] == capped_qids["train"]

    # The supervised split manifest records the capped qids (ordered) and
    # counts.
    sup_manifest = json.loads(
        (harness.shared_supervised_checkpoint(out) / "split_manifest.json")
        .read_text()
    )
    assert sup_manifest["train_qids"] == capped_qids["train"]
    assert sup_manifest["test_qids"] == capped_qids["test"]
    assert sup_manifest["test_count"] == len(capped_qids["test"])


# Tests PR #41 review r3806602891 [integration]: the current invocation's
# own capped selection — resolved via the trainer's loader — is the
# authority the eval/probe subsetting checks against. Children whose
# manifests carry the SAME qid set in a DIFFERENT order (the rebuilt/
# reordered-artifact signature) fail loud before any eval, with no report.
def test_pr41_child_manifest_order_drift_fails_before_subsetting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    split_paths = write_split_artifacts(tmp_path / "artifacts")
    monkeypatch.setattr(
        harness, "resolve_split_artifacts", lambda **kwargs: dict(split_paths)
    )

    drifted_qids = {
        "train": list(reversed(DEFAULT_SPLIT_QIDS["train"])),  # same SET
        "val": list(DEFAULT_SPLIT_QIDS["val"]),
        "test": list(DEFAULT_SPLIT_QIDS["test"]),
    }

    def runner(argv, log_path):
        argv = [str(token) for token in argv]
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("child ok\n")
        root = _sup_ckpt_root(argv)
        if "--skip-supervised" in argv:
            arm, _, seed = root.name.rpartition("_seed")
            make_run_dir(
                root.parent, arm, int(seed),
                hazard="--hazard-pretrain" in argv,
                marker=False, write_eval_result=False,
                split_qids=drifted_qids,
            )
        else:
            best = root / "supervised" / "best_model"
            best.mkdir(parents=True, exist_ok=True)
            (best / "policy_head.pt").write_bytes(b"stub-weights")
        return 0

    monkeypatch.setattr(harness, "_run_child", runner)

    eval_calls: list = []
    monkeypatch.setattr(
        harness, "evaluate_t5_policy", _fake_eval_factory(eval_calls),
        raising=False,
    )

    with pytest.raises(harness.ProvenanceError) as excinfo:
        harness.main(
            ["--smoke", "--out-dir", str(out), "--config", CONFIG_PATH,
             "--seeds", "1", "--arms", "A"]
        )
    message = str(excinfo.value)
    assert "ORDER" in message, message
    assert "A_seed1" in message
    assert eval_calls == [], "no eval may run on an unverified subsetting"
    assert not (out / "hazard_efficacy_report.json").exists()


# Tests QA-005 [integration]: --report-only --dry-run --prune-checkpoints is
# a ZERO-action combination — nothing written, nothing deleted, no children,
# no evals.
def test_qa005_report_only_dry_run_is_zero_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    for arm in ("A", "B"):
        make_run_dir(
            out, arm, 1, include_prunables=True,
            include_hazard_dynamics=(arm == "B"),
        )

    launches: list = []
    evals: list = []
    monkeypatch.setattr(
        harness, "_run_child", lambda argv, log_path: launches.append(argv) or 0
    )
    monkeypatch.setattr(
        harness, "evaluate_t5_policy",
        lambda *a, **k: evals.append(1) or {}, raising=False,
    )

    snapshot_before = {str(p) for p in out.rglob("*")}
    harness.main(
        ["--report-only", "--dry-run", "--prune-checkpoints", "--smoke",
         "--out-dir", str(out), "--arms", "A", "B", "--seeds", "1"]
    )
    snapshot_after = {str(p) for p in out.rglob("*")}

    assert snapshot_after == snapshot_before, (
        "--report-only --dry-run must neither write nor delete anything"
    )
    assert launches == [] and evals == []
    assert not (out / "hazard_efficacy_report.json").exists()
    assert not (out / "hazard_efficacy_plot.png").exists()
    assert (out / "A_seed1" / "ppo_t5" / "iter_1").exists(), (
        "--prune-checkpoints must delete NOTHING under --dry-run"
    )
    assert (out / "A_seed1" / "eval_result.json").exists()


# Tests QA-005 [unit]: the central flag-compatibility matrix rejects the
# contradictory --report-only --force combination right after parse.
def test_qa005_report_only_force_rejected(tmp_path: Path) -> None:
    with pytest.raises(harness.PreflightError):
        harness.validate_flag_compatibility(
            _namespace(tmp_path / "out", report_only=True, force=True)
        )
    # Allowed combination: report-only + dry-run (zero-action semantics).
    harness.validate_flag_compatibility(
        _namespace(tmp_path / "out", report_only=True, dry_run=True)
    )
    # main() applies the matrix before doing anything else.
    with pytest.raises(harness.PreflightError):
        harness.main(
            ["--report-only", "--force", "--out-dir", str(tmp_path / "out")]
        )


# Tests QA-007 [integration]: the shared supervised child stops at the
# branch point — its (discarded) PPO phase is capped via --ppo-iterations 1.
def test_qa007_shared_supervised_child_stops_at_branch_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(harness, "_run_child", _supervised_fabricating_runner(calls))
    out = tmp_path / "out"

    shared = harness._run_shared_supervised(_namespace(out), out)

    assert shared == harness.shared_supervised_checkpoint(out)
    assert len(calls) == 1
    argv = calls[0]
    assert "--skip-supervised" not in argv
    assert _flag_value(argv, "--ppo-iterations") == "1", (
        "the shared child's PPO output is discarded (every arm re-runs PPO "
        "from the branch point); a full PPO budget here is pure waste"
    )


# Tests QA-008 [integration]: --report-only reads each run's marker for the
# REAL git-sha drift instead of recording False unconditionally.
def test_qa008_report_only_reads_markers_for_git_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    make_run_dir(out, "A", 1)  # marker at the CURRENT sha
    make_run_dir(out, "B", 1, marker_git_sha="0" * 40,
                 include_hazard_dynamics=True)  # stale marker

    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )

    harness.main(
        ["--report-only", "--smoke", "--out-dir", str(out),
         "--arms", "A", "B", "--seeds", "1"]
    )

    report = json.loads((out / "hazard_efficacy_report.json").read_text())
    flags = {rec["arm"]: rec["git_sha_mismatch"] for rec in report["runs"]}
    assert flags == {"A": False, "B": True}, (
        "git_sha_mismatch must reflect each run's RUN_COMPLETE.json marker"
    )


# Tests QA-009 [integration]: a tee-loop exception KILLS the child (Popen
# lifetime via context manager) instead of leaking it to keep writing into
# the run dir.
def test_qa009_run_child_kills_child_on_tee_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time as time_mod

    heartbeat = tmp_path / "heartbeat.txt"
    script = (
        "import pathlib, time\n"
        f"hb = pathlib.Path({str(heartbeat)!r})\n"
        "print('first-line', flush=True)\n"
        "for i in range(200):\n"
        "    hb.write_text(str(i))\n"
        "    time.sleep(0.05)\n"
    )
    argv = [sys.executable, "-u", "-c", script]

    def boom(_text):
        raise RuntimeError("tee sink failed")

    monkeypatch.setattr(sys.stdout, "write", boom)
    try:
        with pytest.raises(RuntimeError, match="tee sink failed"):
            harness._run_child(argv, tmp_path / "train.log")
    finally:
        monkeypatch.undo()

    # A killed child stops heartbeating; a leaked one keeps writing for
    # ~10s. Two spaced snapshots must be identical.
    time_mod.sleep(0.4)
    first = heartbeat.read_text() if heartbeat.exists() else "<never wrote>"
    time_mod.sleep(0.6)
    second = heartbeat.read_text() if heartbeat.exists() else "<never wrote>"
    assert first == second, (
        "the child must be KILLED when the tee loop raises — a leaked child "
        "keeps writing into the run dir"
    )


# Tests QA-010 [integration]: prune skips symlinks, never descends through
# symlinked dirs, resolve-and-contains every candidate under the run root,
# and terminates on symlink cycles.
def test_qa010_prune_never_deletes_through_symlinks(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, "A", 1, include_prunables=True)
    ppo = run_dir / "ppo_t5"

    outside = tmp_path / "outside"
    victim_dir = outside / "iter_99"
    victim_dir.mkdir(parents=True)
    (victim_dir / "weights.pt").write_bytes(b"outside-weights")
    victim_state = outside / "training_state.pt"
    victim_state.write_bytes(b"outside-state")

    # A dir symlink NAMED like a prunable, aliasing content outside run_dir.
    (ppo / "iter_5").symlink_to(victim_dir, target_is_directory=True)
    # A dir symlink into the outside tree (walk must not descend through it).
    (ppo / "link_out").symlink_to(outside, target_is_directory=True)
    # A file symlink named training_state.pt pointing outside.
    escape = ppo / "escape"
    escape.mkdir()
    (escape / "training_state.pt").symlink_to(victim_state)
    # A symlink CYCLE back to an ancestor (the walk must terminate).
    (ppo / "loop").symlink_to(run_dir, target_is_directory=True)

    harness.prune_run_checkpoints(run_dir)

    # Real prunables inside the run dir are reclaimed.
    assert not (ppo / "iter_1").exists()
    assert not (ppo / "epoch_1").exists()
    assert not (ppo / "best_model" / "training_state.pt").exists()
    # NOTHING outside the run dir was touched.
    assert victim_dir.exists()
    assert (victim_dir / "weights.pt").exists()
    assert victim_state.exists()
    assert victim_state.read_bytes() == b"outside-state"


# Tests QA-012 [integration]: relative --config/--out-dir are absolutized
# ONCE at the CLI boundary, so child argvs carry CWD-independent paths.
def test_qa012_relative_config_and_out_dir_absolutized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("cfg.yaml").write_text("model:\n  model_name: t5-small\n")
    artifacts = Path("arts")
    artifacts.mkdir()
    split_paths = {}
    for split in ("train", "val", "test"):
        p = artifacts / f"{split}_dataset.json"
        p.write_text("[]")
        split_paths[split] = p
    monkeypatch.setattr(
        harness, "resolve_split_artifacts", lambda **kwargs: dict(split_paths)
    )
    launches: list = []
    monkeypatch.setattr(
        harness, "_run_child", lambda argv, log_path: launches.append(argv) or 0
    )

    harness.main(
        ["--smoke", "--dry-run", "--config", "cfg.yaml",
         "--out-dir", "rel_out", "--seeds", "1", "--arms", "A"]
    )

    assert launches == []
    output = capsys.readouterr().out
    cfg_abs = str(Path("cfg.yaml").resolve())
    out_abs = Path("rel_out").resolve()
    assert cfg_abs in output, "--config must reach children as an ABSOLUTE path"
    assert f"supervised.checkpoint_dir={out_abs / 'A_seed1'}" in output
    assert "supervised.checkpoint_dir=rel_out" not in output


# ---------------------------------------------------------------------------
# QA fix round 2 (QA-R2-1 .. QA-R2-3)
# ---------------------------------------------------------------------------


# Tests QA-R2-1 [integration]: --report-only reconciles the surviving dirs
# against the FULL plan — a planned arm/seed whose run dir is wholly missing
# becomes a printed report warning (QA-011 wording family), never a silent
# drop; a fully-present plan keeps warnings == [].
def test_qa_r2_1_report_only_reconciles_wholly_missing_planned_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )

    # Plan: arms A, B x seeds 1, 3 — B_seed3's dir is wholly absent.
    out = tmp_path / "out"
    make_run_dir(out, "A", 1)
    make_run_dir(out, "A", 3)
    make_run_dir(out, "B", 1, include_hazard_dynamics=True)

    harness.main(
        ["--report-only", "--smoke", "--out-dir", str(out),
         "--arms", "A", "B", "--seeds", "1", "3"]
    )

    report = json.loads((out / "hazard_efficacy_report.json").read_text())
    warnings = report["warnings"]
    assert isinstance(warnings, list) and warnings
    assert any("arm B" in w and "[3]" in w for w in warnings), (
        f"the wholly-missing B_seed3 must be named in the warnings: {warnings}"
    )
    assert not any("arm A" in w for w in warnings), "arm A is fully present"
    # The warning is printed too, never report-only-silent.
    printed = capsys.readouterr().out
    assert any(
        "WARNING" in line and "arm B" in line and "[3]" in line
        for line in printed.splitlines()
    ), f"the plan-vs-disk warning must be printed: {printed!r}"
    # The report still assembles from the three surviving dirs.
    assert len(report["runs"]) == 3

    # Clean case: every planned dir exists -> warnings unchanged ([]).
    clean = tmp_path / "clean"
    make_run_dir(clean, "A", 1)
    make_run_dir(clean, "A", 3)
    make_run_dir(clean, "B", 1, include_hazard_dynamics=True)
    make_run_dir(clean, "B", 3)
    harness.main(
        ["--report-only", "--smoke", "--out-dir", str(clean),
         "--arms", "A", "B", "--seeds", "1", "3"]
    )
    clean_report = json.loads(
        (clean / "hazard_efficacy_report.json").read_text()
    )
    assert clean_report["warnings"] == []


# Tests QA-R2-2 [integration]: --report-only derives the report's smoke
# labeling (smoke caveat + verdict scale note) from the runs' persisted
# RUN_COMPLETE.json "smoke" markers — never from the CURRENT invocation's
# --smoke flag.
def test_qa_r2_2_report_only_derives_smoke_label_from_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )

    # Smoke-trained dirs, report-only WITHOUT --smoke: the report must still
    # carry the smoke caveat and the smoke-scoped verdict.
    out = tmp_path / "out"
    make_run_dir(out, "A", 1, marker_smoke=True)
    make_run_dir(out, "B", 1, marker_smoke=True, include_hazard_dynamics=True)
    harness.main(
        ["--report-only", "--out-dir", str(out),
         "--arms", "A", "B", "--seeds", "1"]
    )
    report = json.loads((out / "hazard_efficacy_report.json").read_text())
    assert any(
        "plumbing/training-dynamics" in caveat for caveat in report["caveats"]
    ), f"smoke-trained dirs must keep the smoke caveat: {report['caveats']}"
    scope = report["verdict"]["scope"]
    assert "smoke" in scope and "plumbing/training-dynamics" in scope, (
        f"the verdict must stay scoped to smoke evidence: {scope!r}"
    )

    # Control: non-smoke markers under the same flag-less invocation keep
    # the non-smoke labeling (derivation, not a constant).
    full = tmp_path / "full"
    make_run_dir(full, "A", 1, marker_smoke=False)
    make_run_dir(full, "B", 1, marker_smoke=False, include_hazard_dynamics=True)
    harness.main(
        ["--report-only", "--out-dir", str(full),
         "--arms", "A", "B", "--seeds", "1"]
    )
    full_report = json.loads((full / "hazard_efficacy_report.json").read_text())
    assert not any(
        "plumbing/training-dynamics" in caveat
        for caveat in full_report["caveats"]
    )
    assert "smoke" not in full_report["verdict"]["scope"]


# Tests QA-R2-2 [integration]: mixed smoke provenance across the existing
# markers fails loud — one report cannot honestly label both cohorts.
def test_qa_r2_2_mixed_smoke_markers_fail_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )
    out = tmp_path / "out"
    make_run_dir(out, "A", 1, marker_smoke=True)
    make_run_dir(out, "B", 1, marker_smoke=False, include_hazard_dynamics=True)

    with pytest.raises(harness.ProvenanceError) as excinfo:
        harness.main(
            ["--report-only", "--out-dir", str(out),
             "--arms", "A", "B", "--seeds", "1"]
        )
    message = str(excinfo.value)
    assert "smoke" in message
    assert "A_seed1" in message and "B_seed1" in message
    assert not (out / "hazard_efficacy_report.json").exists(), (
        "no report may be written under mixed smoke provenance"
    )


# Tests QA-R2-3 [integration]: the shared supervised identity marker records
# the build seed (seeds[0]); reuse under a different seeds[0] WARNS and
# records shared_supervised_seed_mismatch — it never raises and never
# rebuilds (any fixed shared prefix preserves the paired contrast).
def test_qa_r2_3_shared_supervised_seed_mismatch_warns_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(harness, "_run_child", _supervised_fabricating_runner(calls))
    out = tmp_path / "out"
    marker_path = (
        harness.shared_supervised_root(out) / harness.SHARED_SUPERVISED_MARKER
    )

    # Fresh build records the build seed; no mismatch flag yet.
    harness._run_shared_supervised(_namespace(out, seeds=[1, 2, 3]), out)
    assert len(calls) == 1
    marker = json.loads(marker_path.read_text())
    assert marker["supervised_seed"] == 1
    assert "shared_supervised_seed_mismatch" not in marker

    # Same-seed reuse: no warning, no flag.
    capsys.readouterr()
    harness._run_shared_supervised(_namespace(out, seeds=[1, 2, 3]), out)
    assert len(calls) == 1
    assert "WARNING" not in capsys.readouterr().out
    assert "shared_supervised_seed_mismatch" not in json.loads(
        marker_path.read_text()
    )

    # Different-seed reuse: WARN + record, never raise, never rebuild.
    harness._run_shared_supervised(_namespace(out, seeds=[7, 8]), out)
    assert len(calls) == 1, "seed drift must reuse, never rebuild or raise"
    warned = capsys.readouterr().out
    assert "WARNING" in warned and "seed" in warned
    marker = json.loads(marker_path.read_text())
    assert marker["shared_supervised_seed_mismatch"] is True
    assert marker["supervised_seed"] == 1, "the BUILD seed stays recorded"
    # Enforced identity fields survive the recorded-and-warn rewrite.
    assert marker["model_name"] == "t5-small"

    # The recorded mismatch is sticky provenance — a later matching-seed
    # reuse keeps it.
    harness._run_shared_supervised(_namespace(out, seeds=[1, 2, 3]), out)
    assert len(calls) == 1
    assert json.loads(marker_path.read_text())[
        "shared_supervised_seed_mismatch"
    ] is True


# ---------------------------------------------------------------------------
# PR #41 review r3806602894 — training-time provenance sourced from markers
# ---------------------------------------------------------------------------


# Tests PR #41 r3806602894 [integration]: --report-only sources each row's
# provenance (git_sha/git_dirty/torch_version/platform/device) from the
# run's OWN RUN_COMPLETE.json training-time snapshot — never from the
# report-generation process — and flags the source. git_sha_mismatch
# semantics are unchanged (recorded, never fatal).
def test_pr41_report_only_provenance_sourced_from_run_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )
    out = tmp_path / "out"
    stale_sha = "a" * 40  # training-time sha differs from the current HEAD
    sentinel = {
        "git_dirty": True,
        "torch_version": "9.9.9-training-sentinel",
        "platform": "TrainingOS-1.0-arm64",
        "device": "mps",
    }
    for arm in ("A", "B"):
        make_run_dir(
            out, arm, 1,
            marker_git_sha=stale_sha,
            marker_extra=dict(sentinel),
            include_hazard_dynamics=(arm == "B"),
        )

    harness.main(
        ["--report-only", "--smoke", "--out-dir", str(out),
         "--arms", "A", "B", "--seeds", "1"]
    )

    report = json.loads((out / "hazard_efficacy_report.json").read_text())
    for row in report["runs"]:
        prov = row["provenance"]
        assert prov["provenance_source"] == "run_marker"
        # Training-time values verbatim — NOT this process's torch/platform
        # or the current checkout.
        assert prov["torch_version"] == sentinel["torch_version"]
        assert prov["torch_version"] != str(torch.__version__)
        assert prov["platform"] == sentinel["platform"]
        assert prov["git_dirty"] is True
        assert prov["device"] == "mps"
        assert prov["git_sha"] == stale_sha
        assert prov["git_sha"] != current_git_sha()
        # Existing drift semantics unchanged: recorded and non-fatal.
        assert row["git_sha_mismatch"] is True
    # No legacy warning — every marker carried the snapshot.
    assert not any("report_time_legacy" in w for w in report["warnings"])


# Tests PR #41 r3806602894 [integration]: legacy markers WITHOUT the
# training-time snapshot fall back to report-time provenance, are flagged
# provenance_source=report_time_legacy per run, and a report warning lists
# them (printed too).
def test_pr41_report_only_legacy_marker_falls_back_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import torch

    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )
    out = tmp_path / "out"
    for arm in ("A", "B"):
        run_dir = make_run_dir(
            out, arm, 1, include_hazard_dynamics=(arm == "B")
        )
        # Rewrite the marker WITHOUT the snapshot fields (legacy marker
        # predating PR #41 r3806602894).
        marker_path = run_dir / "RUN_COMPLETE.json"
        marker = json.loads(marker_path.read_text())
        for field in ("git_dirty", "torch_version", "platform", "device"):
            marker.pop(field, None)
        marker_path.write_text(json.dumps(marker, indent=2))

    capsys.readouterr()
    harness.main(
        ["--report-only", "--smoke", "--out-dir", str(out),
         "--arms", "A", "B", "--seeds", "1"]
    )

    report = json.loads((out / "hazard_efficacy_report.json").read_text())
    for row in report["runs"]:
        prov = row["provenance"]
        assert prov["provenance_source"] == "report_time_legacy"
        # Report-time fallback values (the pre-existing behavior).
        assert prov["torch_version"] == str(torch.__version__)
        assert prov["git_sha"] == current_git_sha()
        assert prov["device"] == "cpu"  # config_used.json fallback source
    # The fallback is flagged in the report warnings AND printed.
    legacy_warnings = [
        w for w in report["warnings"] if "report_time_legacy" in w
    ]
    assert len(legacy_warnings) == 1
    assert "A_seed1" in legacy_warnings[0]
    assert "B_seed1" in legacy_warnings[0]
    printed = capsys.readouterr().out
    assert "report_time_legacy" in printed
