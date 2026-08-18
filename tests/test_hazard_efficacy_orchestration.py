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
# argv list via subprocess, tees the child's stdout into the given log path,
# and returns the child's exit code verbatim.
def test_r011_real_run_child_tees_log_and_returns_exit_code(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "train.log"
    argv = [sys.executable, "-c", "import sys; print('xmarker'); sys.exit(3)"]

    returncode = harness._run_child(argv, log_path)

    assert returncode == 3
    assert log_path.exists(), "child output must be tee'd to the log path"
    assert "xmarker" in log_path.read_text()


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


# Tests R-008 [integration]: repeatable --variant NAME:FLAGS adds B-variant
# runs with their own dirs and extra hazard flags.
def test_r008_variant_plan_adds_hazard_variant_runs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    args = _namespace(
        out, seeds=[1, 2],
        variant=["Bfz:--beta-terminal 2.0 --freeze-answer-head"],
    )
    plan = harness.plan_runs(args)
    assert len(plan) == 8  # (A, B, C, Bfz) x 2 seeds

    variants = [rec for rec in plan if rec["arm"] == "Bfz"]
    assert len(variants) == 2
    for rec in variants:
        assert Path(rec["run_dir"]) == out / f"Bfz_seed{rec['seed']}"
        argv = rec["argv"]
        assert "--hazard-pretrain" in argv
        assert _flag_value(argv, "--beta-terminal") == "2.0"
        assert "--freeze-answer-head" in argv


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
def test_r013_fresh_runs_invoke_children_and_write_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        assert {"git_sha", "arm", "seed", "wall_clock_seconds"} <= set(marker)
        assert marker["arm"] == rec["arm"]
        # The marker records the child's MEASURED elapsed seconds — a real
        # float >= 0 (the fabricating runner returns near-instantly).
        wall_clock = marker["wall_clock_seconds"]
        assert isinstance(wall_clock, float) and not isinstance(wall_clock, bool)
        assert wall_clock >= 0.0


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
    def fake_probe(*args, **kwargs):
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
