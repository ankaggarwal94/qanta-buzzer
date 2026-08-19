"""Mini-audit fix round regressions (MA-001..MA-018) for the hazard-efficacy
harness (``scripts/run_hazard_efficacy.py``) and its MA-017 companions.

One regression per CRITICAL/HIGH finding (MA-001..MA-008) plus smaller tests
for the MEDIUM/LOW batch. Fixture idioms follow
``tests/_hazard_efficacy_fixtures.py`` (AP-031 format pinning); children and
eval are faked at the pinned seams only.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_hazard_efficacy as harness
from tests._hazard_efficacy_fixtures import (
    fabricating_runner,
    make_hazard_dynamics,
    make_hazard_history,
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


def _fake_eval_factory(calls: list):
    """Capture fake mirroring the REAL evaluate_t5_policy output names."""

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


def _sup_runner(calls: list):
    """Fake ``_run_child`` fabricating the shared supervised child's layout."""

    def runner(argv, log_path):
        argv = [str(token) for token in argv]
        calls.append(argv)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("supervised ok\n")
        override = [
            t for t in argv if t.startswith("supervised.checkpoint_dir=")
        ][-1]
        root = Path(override.split("=", 1)[1])
        best = root / "supervised" / "best_model"
        best.mkdir(parents=True, exist_ok=True)
        (best / "policy_head.pt").write_bytes(b"weights-v1")
        return 0

    return runner


_QUESTIONS = [SimpleNamespace(qid=q) for q in ("q1", "q2", "q3")]


# ---------------------------------------------------------------------------
# MA-001 — positive arm/run identity
# ---------------------------------------------------------------------------


# Tests MA-001 [integration]: a dir whose config_used seed differs from the
# plan slot (a copied dir fabricating a replication) fails loud on resume.
def test_ma001_copied_dir_seed_mismatch_fails_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    plan = harness.plan_runs(_namespace(out, arms=["B"], seeds=[2]))
    # B_seed1's content copied into the B_seed2 slot: config says seed=1.
    make_run_dir(out, "B", 2, config_mutations={"seed": 1})
    runner, calls = fabricating_runner(plan)
    monkeypatch.setattr(harness, "_run_child", runner)

    with pytest.raises(harness.ProvenanceError, match="MA-001") as excinfo:
        harness.execute_plan(plan)
    message = str(excinfo.value)
    assert "B_seed2" in message and "seed" in message
    assert calls == []


# Tests MA-001 [integration]: a C-trained checkpoint masquerading as B (its
# config hazard block carries the ablation) fails the hazard-identity check.
def test_ma001_arm_role_swap_fails_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    plan = harness.plan_runs(_namespace(out, arms=["B"], seeds=[1]))
    make_run_dir(
        out, "B", 1, config_mutations={"hazard.ablation": "shuffled_nll"}
    )
    runner, calls = fabricating_runner(plan)
    monkeypatch.setattr(harness, "_run_child", runner)

    with pytest.raises(harness.ProvenanceError, match="hazard"):
        harness.execute_plan(plan)
    assert calls == []


# Tests MA-001 [integration]: the completion marker is self-identifying — a
# marker naming another arm/seed fails loud on resume.
def test_ma001_marker_arm_mismatch_fails_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    plan = harness.plan_runs(_namespace(out, arms=["B"], seeds=[1]))
    make_run_dir(out, "B", 1, marker_extra={"arm": "C"})
    runner, calls = fabricating_runner(plan)
    monkeypatch.setattr(harness, "_run_child", runner)

    with pytest.raises(harness.ProvenanceError, match="MA-001"):
        harness.execute_plan(plan)
    assert calls == []


# Tests MA-001 [unit]: report rows echo each run's hazard identity block.
def test_ma001_report_rows_carry_hazard_identity(tmp_path: Path) -> None:
    out = tmp_path / "out"
    records = []
    for arm in ("A", "B"):
        run_dir = make_run_dir(out, arm, 1, include_hazard_dynamics=(arm == "B"))
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A", "resumed": False})
    report = harness.assemble_report(out, records, smoke=True)
    by_arm = {rec["arm"]: rec for rec in report["runs"]}
    assert by_arm["B"]["hazard"] == {
        "pretrain": True,
        "beta_terminal": 1.0,
        "freeze_answer_head": False,
        "ablation": None,
    }
    assert by_arm["A"]["hazard"]["pretrain"] is False


# Tests MA-001 [unit]: the eval sidecar is self-identifying — a payload
# naming another seed fails assembly loud.
def test_ma001_eval_sidecar_self_identity_asserted(tmp_path: Path) -> None:
    out = tmp_path / "out"
    records = []
    for arm in ("A", "B"):
        overrides = {"seed": 99} if arm == "B" else {}
        run_dir = make_run_dir(
            out, arm, 1, eval_overrides=overrides,
            include_hazard_dynamics=(arm == "B"),
        )
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A", "resumed": False})
    with pytest.raises(harness.ProvenanceError, match="MA-001"):
        harness.assemble_report(out, records, smoke=True)


# ---------------------------------------------------------------------------
# MA-002 — shared verify_run_records() gate
# ---------------------------------------------------------------------------


# Tests MA-002 [integration]: --report-only routes through the gate — a
# heterogeneous tree (doctored model_name) can no longer assemble a report.
def test_ma002_report_only_heterogeneous_tree_fails_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )
    out = tmp_path / "out"
    make_run_dir(out, "A", 1)
    make_run_dir(out, "B", 1, model_name="t5-base", include_hazard_dynamics=True)

    with pytest.raises(harness.ArmControlError):
        harness.main(
            ["--report-only", "--smoke", "--out-dir", str(out),
             "--arms", "A", "B", "--seeds", "1"]
        )
    assert not (out / "hazard_efficacy_report.json").exists()


# Tests MA-002 [integration]: --report-only rejects a PARTIAL dir (marker
# but no checkpoints) instead of silently assembling around it.
def test_ma002_report_only_partial_dir_fails_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )
    out = tmp_path / "out"
    make_run_dir(out, "A", 1)
    make_run_dir(out, "B", 1, write_checkpoints=False,
                 include_hazard_dynamics=True)

    with pytest.raises(harness.PartialRunError, match="MA-002"):
        harness.main(
            ["--report-only", "--smoke", "--out-dir", str(out),
             "--arms", "A", "B", "--seeds", "1"]
        )
    assert not (out / "hazard_efficacy_report.json").exists()


# Tests MA-002 [unit]: BOTH report-producing entry paths route through the
# single gate (structural pin).
def test_ma002_both_entry_paths_route_through_verify_gate() -> None:
    assert "verify_run_records(records)" in inspect.getsource(
        harness._main_execute
    )
    assert "verify_run_records(records)" in inspect.getsource(
        harness._main_report_only
    )


# ---------------------------------------------------------------------------
# MA-003 — shared-checkpoint weight fingerprint
# ---------------------------------------------------------------------------


# Tests MA-003 [integration]: the shared identity marker records the weight
# fingerprint; a mutated/rebuilt checkpoint is detected on reuse.
def test_ma003_shared_marker_fingerprint_detects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list = []
    monkeypatch.setattr(harness, "_run_child", _sup_runner(calls))
    out = tmp_path / "out"

    shared = harness._run_shared_supervised(_namespace(out), out)
    marker_path = (
        harness.shared_supervised_root(out) / harness.SHARED_SUPERVISED_MARKER
    )
    marker = json.loads(marker_path.read_text())
    assert marker["weights_sha256"] == harness._weights_fingerprint(shared)
    # MA-015: the atomic write leaves no temp file behind.
    assert not list(marker_path.parent.glob("*.tmp"))

    # Unchanged content: reuse without a rebuild.
    harness._run_shared_supervised(_namespace(out), out)
    assert len(calls) == 1

    # Mutated weights: reuse fails loud instead of silently seeding arms.
    (Path(shared) / "policy_head.pt").write_bytes(b"weights-v2-mutated")
    with pytest.raises(harness.ProvenanceError, match="MA-003"):
        harness._run_shared_supervised(_namespace(out), out)
    assert len(calls) == 1, "no silent rebuild on fingerprint mismatch"


# Tests MA-003 [integration]: resumed arm runs assert their recorded branch
# fingerprint against the CURRENT shared checkpoint; fresh runs get stamped.
def test_ma003_resumed_run_fingerprint_asserted_and_fresh_stamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # (a) mismatch: the run branched from a rebuilt checkpoint.
    record = make_plan_record(tmp_path / "a", "A", 1)
    make_run_dir(
        tmp_path / "a", "A", 1,
        marker_extra={"shared_supervised_weights_sha256": "0" * 64},
    )
    runner, calls = fabricating_runner([record])
    monkeypatch.setattr(harness, "_run_child", runner)
    with pytest.raises(harness.ProvenanceError, match="MA-003"):
        harness.execute_plan(
            [record], expected_run_context={"shared_weights_sha256": "f" * 64}
        )
    assert calls == []

    # (b) match: resumes cleanly.
    record = make_plan_record(tmp_path / "b", "A", 1)
    make_run_dir(
        tmp_path / "b", "A", 1,
        marker_extra={"shared_supervised_weights_sha256": "f" * 64},
    )
    runner, calls = fabricating_runner([record])
    monkeypatch.setattr(harness, "_run_child", runner)
    updated = harness.execute_plan(
        [record], expected_run_context={"shared_weights_sha256": "f" * 64}
    )
    assert updated[0]["resumed"] is True and calls == []

    # (c) legacy marker without the field: warns, never raises.
    record = make_plan_record(tmp_path / "c", "A", 1)
    make_run_dir(tmp_path / "c", "A", 1)
    runner, calls = fabricating_runner([record])
    monkeypatch.setattr(harness, "_run_child", runner)
    capsys.readouterr()
    harness.execute_plan(
        [record], expected_run_context={"shared_weights_sha256": "f" * 64}
    )
    assert "legacy marker" in capsys.readouterr().out

    # (d) fresh runs get the fingerprint stamped into their marker.
    record = make_plan_record(tmp_path / "d", "A", 1)
    runner, calls = fabricating_runner([record])
    monkeypatch.setattr(harness, "_run_child", runner)
    harness.execute_plan(
        [record], expected_run_context={"shared_weights_sha256": "a" * 64}
    )
    marker = json.loads(
        (Path(record["run_dir"]) / "RUN_COMPLETE.json").read_text()
    )
    assert marker["shared_supervised_weights_sha256"] == "a" * 64
    assert not list(Path(record["run_dir"]).glob("*.tmp"))


# ---------------------------------------------------------------------------
# MA-004 — producer-manifest disjointness on reuse
# ---------------------------------------------------------------------------


# Tests MA-004 [unit]: on reuse the disjointness proof comes from the shared
# child's OWN recorded split — an overlap with the CURRENT test qids fails
# loud; a disjoint producer manifest is returned (and becomes the persisted
# manifest); a missing producer manifest falls back to None.
def test_ma004_reuse_disjointness_from_producer_manifest(tmp_path: Path) -> None:
    out = tmp_path / "out"
    producer_path = (
        harness.shared_supervised_root(out) / "ppo_t5" / "split_manifest.json"
    )
    overlapping = make_split_manifest(
        split_qids={"train": ["t1", "q1"], "val": ["v1"], "test": ["x1"]}
    )
    write_json(producer_path, overlapping)
    with pytest.raises(harness.ProvenanceError, match="MA-004"):
        harness.shared_manifest_for_reuse(out, ["q1", "q2"])

    clean = make_split_manifest()
    write_json(producer_path, clean)
    got = harness.shared_manifest_for_reuse(out, ["q1", "q2", "q3"])
    assert got is not None
    assert got["train_qids"] == clean["train_qids"]

    assert harness.shared_manifest_for_reuse(tmp_path / "other", ["q1"]) is None


# ---------------------------------------------------------------------------
# MA-005 — hazard_dynamics.json joins the invalidation set
# ---------------------------------------------------------------------------


# Tests MA-005 [integration]: a force re-run unlinks the stale
# hazard_dynamics.json alongside the marker and eval result BEFORE the child
# launches — a crash then leaves no stale dynamics for --report-only.
def test_ma005_force_rerun_unlinks_stale_hazard_dynamics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = make_plan_record(tmp_path, "B", 1)
    run_dir = make_run_dir(tmp_path, "B", 1, include_hazard_dynamics=True)
    assert (run_dir / "hazard_dynamics.json").exists()

    crash, calls = fabricating_runner([record], exit_code=3)
    monkeypatch.setattr(harness, "_run_child", crash)
    with pytest.raises(harness.ChildRunError):
        harness.execute_plan([record], force=True)
    assert len(calls) == 1
    assert not (run_dir / "hazard_dynamics.json").exists()
    assert not (run_dir / "RUN_COMPLETE.json").exists()
    assert not (run_dir / "eval_result.json").exists()


# ---------------------------------------------------------------------------
# MA-006 — child watchdog
# ---------------------------------------------------------------------------


# Tests MA-006 [integration]: a child that goes silent is killed promptly
# with a ChildRunError naming the last-output age and the log.
def test_ma006_stalled_child_killed_with_named_age(tmp_path: Path) -> None:
    script = "import time; print('one line', flush=True); time.sleep(60)"
    argv = [sys.executable, "-u", "-c", script]
    log_path = tmp_path / "train.log"

    start = time.monotonic()
    with pytest.raises(harness.ChildRunError) as excinfo:
        harness._run_child(argv, log_path, stall_timeout_seconds=0.4)
    elapsed = time.monotonic() - start
    assert elapsed < 20.0, "the stalled child must be killed promptly"

    message = str(excinfo.value)
    assert "stall" in message.lower()
    assert "no output" in message
    assert str(log_path) in message


# Tests MA-006 [integration]: a healthy child is unaffected by the watchdog;
# the optional total-runtime cap kills a chatty-but-endless child.
def test_ma006_healthy_child_ok_and_total_cap_enforced(tmp_path: Path) -> None:
    ok = [sys.executable, "-c", "print('fine')"]
    assert harness._run_child(ok, tmp_path / "ok.log",
                              stall_timeout_seconds=30.0) == 0
    assert "fine" in (tmp_path / "ok.log").read_text()

    chatty = [
        sys.executable, "-u", "-c",
        "import time\nfor i in range(600):\n    print(i, flush=True)\n"
        "    time.sleep(0.05)\n",
    ]
    start = time.monotonic()
    with pytest.raises(harness.ChildRunError, match="total runtime"):
        harness._run_child(
            chatty, tmp_path / "chat.log", child_timeout_seconds=0.5
        )
    assert time.monotonic() - start < 20.0


# ---------------------------------------------------------------------------
# MA-007 — disk budget / per-run + shared prune
# ---------------------------------------------------------------------------


# Tests MA-007 [integration]: with --prune-checkpoints semantics, each run is
# pruned right after ITS eval_result.json exists — run 1 is reclaimed even
# when run 2's eval later fails (peak disk never spans the whole pipeline).
def test_ma007_per_run_prune_interleaved_with_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = []
    for arm in ("A", "B"):
        run_dir = make_run_dir(
            tmp_path, arm, 1, include_prunables=True, write_eval_result=False
        )
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A"})

    calls: list = []
    good = _fake_eval_factory(calls)

    def flaky(*args, **kwargs):
        if calls:
            raise RuntimeError("second eval exploded")
        return good(*args, **kwargs)

    monkeypatch.setattr(harness, "evaluate_t5_policy", flaky, raising=False)

    with pytest.raises(RuntimeError):
        harness.evaluate_all_runs(records, _QUESTIONS, {}, prune=True)

    run1 = Path(records[0]["run_dir"])
    run2 = Path(records[1]["run_dir"])
    assert not (run1 / "ppo_t5" / "iter_1").exists(), (
        "run 1 must be pruned immediately after its own eval"
    )
    assert (run1 / "eval_result.json").exists()
    assert (run1 / "ppo_t5" / "best_model" / "policy_head.pt").exists()
    assert (run2 / "ppo_t5" / "iter_1").exists(), "run 2 was never evaluated"


# Tests MA-007 [unit]: the shared supervised tree's epoch_*/iter_*/
# training_state.pt are reclaimed while best_model dirs survive; a missing
# shared root is a silent no-op.
def test_ma007_shared_supervised_prune(tmp_path: Path) -> None:
    out = tmp_path / "out"
    root = harness.shared_supervised_root(out)
    best = root / "supervised" / "best_model"
    best.mkdir(parents=True)
    (best / "policy_head.pt").write_bytes(b"w")
    (best / "training_state.pt").write_bytes(b"s")
    epoch = root / "supervised" / "epoch_1"
    epoch.mkdir()
    (epoch / "policy_head.pt").write_bytes(b"w")
    ppo_iter = root / "ppo_t5" / "iter_1"
    ppo_iter.mkdir(parents=True)
    (ppo_iter / "policy_head.pt").write_bytes(b"w")

    harness.prune_shared_supervised_checkpoints(out)

    assert not epoch.exists()
    assert not ppo_iter.exists()
    assert (best / "policy_head.pt").exists()
    assert not (best / "training_state.pt").exists()

    harness.prune_shared_supervised_checkpoints(tmp_path / "absent")  # no-op


# Tests MA-007 [unit]: the preflight disk budget prints the estimate and
# warns when it exceeds 80% of free space — and stays quiet otherwise.
def test_ma007_disk_preflight_prints_estimate_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "out"
    args = _namespace(out, seeds=[1], arms=["A", "B"])
    plan = harness.plan_runs(args)

    monkeypatch.setattr(
        harness.shutil, "disk_usage",
        lambda p: SimpleNamespace(total=100, used=90, free=10),
    )
    harness._print_disk_preflight(args, plan, out)
    printed = capsys.readouterr().out
    assert "disk preflight" in printed
    assert "WARNING" in printed and "80%" in printed

    monkeypatch.setattr(
        harness.shutil, "disk_usage",
        lambda p: SimpleNamespace(total=10**15, used=0, free=10**15),
    )
    harness._print_disk_preflight(args, plan, out)
    printed = capsys.readouterr().out
    assert "disk preflight" in printed and "WARNING" not in printed


# ---------------------------------------------------------------------------
# MA-008 — variant hardening
# ---------------------------------------------------------------------------


# Tests MA-008 [unit]: reserved FLAGS tokens, reserved overrides, and bare
# no-op tokens are all rejected at plan time.
# EXTENDED in PR #41 round-2 resolve: negative-number-shaped tokens and
# unknown flag bases ([5] — they parsed as silent positionals), overrides of
# arm-control NON-exempt keys ([3] — guaranteed post-training
# ArmControlError), and silently-overwritten seed=/hazard.* overrides
# (P3 [14]) are all rejected at plan time too.
# EXTENDED in PR #41 round-3 resolve: known FLAGS whose config effect lands
# on a non-exempt key (R2-3 / codex r3809591787 — --ppo-iterations writes
# ppo.iterations) and ANY *.checkpoint_dir= override (R2-4 — exempt but
# INERT in the child: a silent duplicate-of-B variant) are rejected too.
@pytest.mark.parametrize(
    ("flags", "needle"),
    [
        ("--seed 9", "reserved"),
        ("--model-path /tmp/x", "reserved"),
        ("--skip-supervised", "reserved"),
        ("--smoke", "reserved"),
        ("supervised.checkpoint_dir=/tmp/x", "reserved"),
        ("ppo.eval_interval=5", "reserved"),
        ("lr2e-5", "bare token"),
        # PR #41 round-2 [5]: negative-number-shaped / unknown flag bases.
        ("-3", "unrecognized"),
        ("-0.5", "unrecognized"),
        ("--no-such-flag", "unrecognized"),
        # PR #41 round-2 [3]: arm-control non-exempt config overrides.
        ("ppo.lr=2e-5", "NOT exempt"),
        ("data.max_questions=5", "NOT exempt"),
        ("model.model_name=t5-large", "NOT exempt"),
        # PR #41 round-2 P3 [14]: silently-overwritten overrides.
        ("seed=5", "OVERWRITTEN"),
        ("hazard.beta_terminal=9.0", "OVERWRITTEN"),
        # PR #41 round-3 R2-3 (codex r3809591787): flag-form config writes
        # onto non-exempt keys (two-token and =-joined spellings).
        ("--ppo-iterations 3", "NOT exempt"),
        ("--ppo-iterations=3", "NOT exempt"),
        # PR #41 round-3 R2-4: *.checkpoint_dir= overrides are inert in the
        # child (flatten_config reads supervised.checkpoint_dir only).
        ("ppo.checkpoint_dir=elsewhere", "DUPLICATE"),
        ("data.checkpoint_dir=elsewhere", "DUPLICATE"),
        ("checkpoint_dir=elsewhere", "DUPLICATE"),
    ],
)
def test_ma008_variant_reserved_and_bare_tokens_rejected(
    tmp_path: Path, flags: str, needle: str
) -> None:
    args = _namespace(tmp_path / "out", variant=[f"V:{flags}"])
    with pytest.raises(harness.PreflightError, match=needle):
        harness.plan_runs(args)


# Tests MA-008 [unit]: variant names shadowing core arms A/B/C are rejected
# (they could otherwise capture control/treatment roles).
def test_ma008_variant_shadowing_core_arm_rejected(tmp_path: Path) -> None:
    for name in ("A", "B", "C"):
        with pytest.raises(harness.PreflightError, match="shadows"):
            harness.plan_runs(
                _namespace(tmp_path / "out", variant=[f"{name}:ppo.lr=2e-5"])
            )


# Tests MA-008 [unit]: variants inherit the invocation's hazard knobs; FLAGS
# override with exactly one occurrence of each knob flag; the recorded
# hazard identity reflects the parsed result.
def test_ma008_variant_inherits_hazard_knobs_flags_override(
    tmp_path: Path,
) -> None:
    # AMENDED in PR #41 round-2 resolve [3]: the carrier variant is now
    # FLAGS-free (pure knob inheritance) — its old ppo.lr= override targets
    # an arm-control non-exempt key and is rejected at plan time.
    out = tmp_path / "out"
    plan = harness.plan_runs(
        _namespace(
            out, seeds=[1], beta_terminal=2.5, freeze_answer_head=True,
            variant=["Bv:"],
        )
    )
    variant = next(rec for rec in plan if rec["arm"] == "variant:Bv")
    argv = variant["argv"]
    assert argv[argv.index("--beta-terminal") + 1] == "2.5"
    assert "--freeze-answer-head" in argv
    assert variant["hazard_knobs"] == {
        "pretrain": True,
        "beta_terminal": 2.5,
        "freeze_answer_head": True,
        "ablation": None,
    }

    plan2 = harness.plan_runs(
        _namespace(
            out, seeds=[1], beta_terminal=2.5,
            variant=["Bo:--beta-terminal 9.0"],
        )
    )
    override = next(rec for rec in plan2 if rec["arm"] == "variant:Bo")
    assert override["argv"].count("--beta-terminal") == 1
    assert override["hazard_knobs"]["beta_terminal"] == pytest.approx(9.0)


# Tests PR #41 round-3 R2-3 (codex r3809591787) [unit]: a variant FLAGS
# token that is a KNOWN child flag but whose config effect lands on an
# arm-control non-exempt key (--ppo-iterations -> ppo.iterations) dies at
# PLAN time with an error naming the flag, the effect key, and the variant
# — zero children. It used to pass the known-flag gate AND the parser
# roundtrip, then abort hours later at the post-children arm-control diff
# (the child writes ppo.iterations into config_used.json).
def test_pr41_r3_variant_flag_config_effect_rejected_at_plan_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail(
            "the rejection must cost ZERO children"
        ),
    )
    with pytest.raises(harness.PreflightError) as excinfo:
        harness.plan_runs(
            _namespace(tmp_path / "out", variant=["it3:--ppo-iterations 3"])
        )
    message = str(excinfo.value)
    assert "--ppo-iterations" in message, message
    assert "'it3'" in message, "the error must name the variant"
    assert "ppo.iterations" in message, "the error must name the effect key"
    assert "NOT exempt" in message, message


# Tests PR #41 round-3 R2-5 [unit]: the _CHILD_KNOWN_FLAGS /
# _CHILD_VALUE_FLAGS mirrors stay in lockstep with the REAL child parser
# (MA-018 tuple-parity precedent: mirror sets are asserted against their
# source of truth, here scripts.train_t5_policy.parse_args's parser
# introspected via parser._actions) — a new child flag that is not
# mirrored would otherwise be silently unplannable (known-flags) or
# mis-classified as a bare positional (value-flags).
def test_pr41_r3_child_flag_mirrors_parity_with_real_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.train_t5_policy as train_t5_policy

    captured: dict = {}
    real_parse_args = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        captured.setdefault("parser", self)
        return real_parse_args(self, args, namespace)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    train_t5_policy.parse_args(argv=["--skip-supervised"])
    parser = captured["parser"]

    known: set[str] = set()
    value_taking: set[str] = set()
    for action in parser._actions:
        if not action.option_strings:
            continue  # the trailing positional ``overrides`` group
        if "--help" in action.option_strings:
            continue
        known.update(action.option_strings)
        if action.nargs != 0:  # store_true actions have nargs == 0
            value_taking.update(action.option_strings)

    assert known == set(harness._CHILD_KNOWN_FLAGS), (
        "_CHILD_KNOWN_FLAGS must equal the real child parser's full option "
        f"surface; parser={sorted(known)} "
        f"mirror={sorted(harness._CHILD_KNOWN_FLAGS)}"
    )
    assert value_taking == set(harness._CHILD_VALUE_FLAGS), (
        "_CHILD_VALUE_FLAGS must equal the real child parser's value-taking "
        f"options; parser={sorted(value_taking)} "
        f"mirror={sorted(harness._CHILD_VALUE_FLAGS)}"
    )
    # Structural sanity: every flag with a config effect is a known flag.
    assert set(harness._CHILD_FLAG_CONFIG_EFFECTS) <= known


# ---------------------------------------------------------------------------
# MA-009 — plot renders missing as missing
# ---------------------------------------------------------------------------


# Tests MA-009 [unit]: a no-data arm yields None (never 0.0 = optimal on the
# lower-is-earlier axis) and the plot still renders (NaN gap + annotation).
def test_ma009_plot_missing_renders_as_missing(tmp_path: Path) -> None:
    runs = [
        {"arm": "A", "seed": 1, "mean_correct_buzz_position": None,
         "accuracy": 0.5},
        {"arm": "B", "seed": 1, "mean_correct_buzz_position": 3.0,
         "accuracy": 0.6},
    ]
    assert harness._plot_arm_mean(runs, "A", "mean_correct_buzz_position") is None
    assert harness._plot_arm_mean(
        runs, "B", "mean_correct_buzz_position"
    ) == pytest.approx(3.0)

    report = {"runs": runs, "scale": {"model_name": "t5-small", "n_test": 3}}
    plot_path = harness.write_plot(report, tmp_path)
    assert plot_path.exists()


# ---------------------------------------------------------------------------
# MA-010 — C step parity + source labels
# ---------------------------------------------------------------------------


# Tests MA-010 [integration]: a C run whose hazard step count differs from
# B's at the same seed voids the compute control and fails report assembly.
def test_ma010_step_parity_mismatch_fails_report(tmp_path: Path) -> None:
    out = tmp_path / "out"
    records = []
    specs = {
        "A": None,
        "B": [1.0, 1.0, 1.0, 1.0],
        "C": [1.0] * 6,  # NOT step-matched
    }
    for arm, losses in specs.items():
        run_dir = make_run_dir(
            out, arm, 1,
            hazard_history_losses=losses,
            include_hazard_dynamics=(arm == "B"),
        )
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A", "resumed": False})
    with pytest.raises(harness.ProvenanceError, match="MA-010") as excinfo:
        harness.assemble_report(out, records, smoke=True)
    message = str(excinfo.value)
    assert "6" in message and "4" in message and "seed 1" in message


# Tests MA-010 [unit]: hazard_compute and hazard_dynamics carry explicit
# source_arm/source_seed labels for their single-replicate values.
def test_ma010_hazard_blocks_carry_source_labels(tmp_path: Path) -> None:
    out = tmp_path / "out"
    records = []
    for arm in ("A", "B"):
        run_dir = make_run_dir(out, arm, 1, include_hazard_dynamics=(arm == "B"))
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A", "resumed": False})
    report = harness.assemble_report(out, records, smoke=True)
    assert report["hazard_compute"]["source_arm"] == "B"
    assert report["hazard_compute"]["source_seed"] == 1
    assert report["hazard_dynamics"]["source_arm"] == "B"
    assert report["hazard_dynamics"]["source_seed"] == 1


# ---------------------------------------------------------------------------
# MA-011 — endpoint / dynamics conventions named
# ---------------------------------------------------------------------------


# Tests MA-011 [unit]: the endpoint definition names the exact payload keys
# and index base; hazard_dynamics names its own (different) index base.
def test_ma011_endpoint_and_dynamics_name_keys_and_index_conventions() -> None:
    for token in (
        "mean_correct_buzz_position",
        "accuracy",
        "n_correct_policy_buzzes",
        "0-indexed",
    ):
        assert token in harness.ENDPOINT_DEFINITION, token

    block = harness.build_hazard_dynamics(
        [[0.1, 0.2]], [[0.2, 0.3]], make_hazard_history([1.0, 2.0])
    )
    assert "1-indexed" in block["index_convention"]
    assert "0-indexed" in block["index_convention"]


# ---------------------------------------------------------------------------
# MA-012 — read-before-recompute for eval + probe
# ---------------------------------------------------------------------------


# Tests MA-012 [integration]: resumed runs with existing eval artifacts are
# read, not recomputed; --re-eval forces the recompute.
def test_ma012_resumed_eval_skipped_and_re_eval_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "out"
    records = []
    for arm in ("A", "B"):
        run_dir = make_run_dir(out, arm, 1, include_hazard_dynamics=(arm == "B"))
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A", "resumed": True,
                        "log_path": run_dir / "train.log"})
    calls: list = []
    monkeypatch.setattr(
        harness, "evaluate_t5_policy", _fake_eval_factory(calls), raising=False
    )

    results = harness.evaluate_all_runs(records, _QUESTIONS, {})
    assert calls == [], "resumed runs with eval artifacts must not recompute"
    assert len(results) == 2
    assert "eval resumed" in capsys.readouterr().out

    harness.evaluate_all_runs(records, _QUESTIONS, {}, re_eval=True)
    assert len(calls) == 2, "--re-eval must force the recompute"


# Tests MA-012/MA-018 [integration]: the probe stage prints [j/M] banners,
# skips resumed runs with existing dynamics, and --re-eval overrides.
def test_ma012_ma018_probe_skip_and_banners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "out"
    records = []
    for arm in ("B", "C"):
        run_dir = make_run_dir(out, arm, 1, include_hazard_dynamics=True)
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": True, "resumed": True})

    probe_calls: list = []

    def fake_probe(*args, **kwargs):
        probe_calls.append(1)
        out_path = Path(kwargs["out_path"] if "out_path" in kwargs else args[4])
        write_json(out_path, make_hazard_dynamics())
        return {}

    monkeypatch.setattr(harness, "probe_and_write_hazard_dynamics", fake_probe)

    harness._probe_all_hazard_runs(records, out / "shared", [], re_eval=False)
    printed = capsys.readouterr().out
    assert probe_calls == []
    assert "[probe 1/2]" in printed and "probe resumed" in printed

    harness._probe_all_hazard_runs(records, out / "shared", [], re_eval=True)
    printed = capsys.readouterr().out
    assert len(probe_calls) == 2
    assert "[probe 2/2]" in printed


# ---------------------------------------------------------------------------
# MA-013 — everything decidable in preflight is decided there
# ---------------------------------------------------------------------------


# Tests MA-013 [integration]: a partial arm dir aborts BEFORE the shared
# supervised child (zero children), not fail-late mid-execution.
def test_ma013_partial_dir_fails_before_any_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    split_paths = write_split_artifacts(tmp_path / "artifacts")
    monkeypatch.setattr(
        harness, "resolve_split_artifacts", lambda **kwargs: dict(split_paths)
    )
    launches: list = []
    monkeypatch.setattr(
        harness, "_run_child", lambda argv, log_path: launches.append(argv) or 0
    )
    make_run_dir(out, "A", 1, marker=False)  # partial

    with pytest.raises(harness.PartialRunError, match="preflight"):
        harness.main(
            ["--smoke", "--out-dir", str(out), "--config", CONFIG_PATH,
             "--seeds", "1", "--arms", "A"]
        )
    assert launches == [], "the failure must precede EVERY child launch"


# Tests MA-013 [integration]: the dry-run plan includes the shared supervised
# argv (round-tripped, --skip-test-eval included) and the disk preflight.
def test_ma013_dry_run_prints_supervised_argv_and_disk_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    artifacts = tmp_path / "artifacts"
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
        ["--smoke", "--dry-run", "--out-dir", str(tmp_path / "out"),
         "--seeds", "1", "--arms", "A"]
    )
    assert launches == []
    printed = capsys.readouterr().out
    assert "[shared supervised]" in printed
    assert "--ppo-iterations 1" in printed
    assert "--skip-test-eval" in printed  # MA-017 composed into the argv
    assert "disk preflight" in printed  # MA-007


# Tests MA-013 [integration]: a typo'd --config dies at preflight (even under
# --dry-run), never after the supervised phase.
def test_ma013_config_typo_dies_in_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    split_paths = {}
    for split in ("train", "val", "test"):
        p = artifacts / f"{split}_dataset.json"
        p.write_text("[]")
        split_paths[split] = p
    monkeypatch.setattr(
        harness, "resolve_split_artifacts", lambda **kwargs: dict(split_paths)
    )
    with pytest.raises(harness.PreflightError, match="Config file"):
        harness.main(
            ["--smoke", "--dry-run", "--config", str(tmp_path / "nope.yaml"),
             "--out-dir", str(tmp_path / "out"), "--seeds", "1", "--arms", "A"]
        )


# ---------------------------------------------------------------------------
# MA-014 — closing summary / resume affordance / prune narration
# ---------------------------------------------------------------------------


# Tests MA-014 [integration]: the closing summary carries verdict + disk +
# cleanup hint + BOTH artifact paths (report-only included).
def test_ma014_closing_summary_on_report_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "out"
    for arm in ("A", "B"):
        make_run_dir(out, arm, 1, include_hazard_dynamics=(arm == "B"))
    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )

    harness.main(
        ["--report-only", "--smoke", "--out-dir", str(out),
         "--arms", "A", "B", "--seeds", "1"]
    )
    printed = capsys.readouterr().out
    assert "VERDICT:" in printed
    assert "disk usage" in printed
    assert "Report written to" in printed
    assert "Plot written to" in printed, "report-only must print the plot line"
    assert "--prune-checkpoints" in printed  # cleanup hint when not pruning


# Tests MA-014 [integration]: child-failure errors state the resume
# affordance; prune narrates what it reclaimed and kept.
def test_ma014_resume_guidance_and_prune_narration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    record = make_plan_record(tmp_path / "runs", "A", 1)
    crash, _calls = fabricating_runner([record], exit_code=2)
    monkeypatch.setattr(harness, "_run_child", crash)
    with pytest.raises(harness.ChildRunError, match="resumes"):
        harness.execute_plan([record])

    run_dir = make_run_dir(tmp_path, "B", 1, include_prunables=True)
    capsys.readouterr()
    harness.prune_run_checkpoints(run_dir)
    printed = capsys.readouterr().out
    assert "[prune]" in printed
    assert "reclaimed" in printed and "kept" in printed


# ---------------------------------------------------------------------------
# MA-015 — atomic/typed markers + disk-minus-plan reconciliation
# ---------------------------------------------------------------------------


# Tests MA-015 [integration]: a corrupt (truncated) marker raises
# PartialRunError WITH remediation instead of an unexplained decode error.
def test_ma015_corrupt_marker_raises_with_remediation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = make_plan_record(tmp_path, "A", 1)
    run_dir = make_run_dir(tmp_path, "A", 1)
    (run_dir / "RUN_COMPLETE.json").write_text("{truncated")
    runner, calls = fabricating_runner([record])
    monkeypatch.setattr(harness, "_run_child", runner)

    with pytest.raises(harness.PartialRunError, match="--force"):
        harness.execute_plan([record])
    assert calls == []


# Tests MA-015 [integration]: marker fields are TYPE-validated — a string
# "smoke" or boolean wall-clock fails loud instead of skewing consumers.
def test_ma015_typed_marker_fields_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = make_plan_record(tmp_path, "A", 1)
    make_run_dir(tmp_path, "A", 1, marker_extra={"smoke": "yes"})
    runner, calls = fabricating_runner([record])
    monkeypatch.setattr(harness, "_run_child", runner)
    with pytest.raises(harness.PartialRunError, match="smoke"):
        harness.execute_plan([record])
    assert calls == []

    record = make_plan_record(tmp_path / "b", "A", 1)
    make_run_dir(tmp_path / "b", "A", 1, marker_extra={"wall_clock_seconds": True})
    runner, calls = fabricating_runner([record])
    monkeypatch.setattr(harness, "_run_child", runner)
    with pytest.raises(harness.PartialRunError, match="wall_clock_seconds"):
        harness.execute_plan([record])
    assert calls == []


# Tests MA-015 [integration]: --report-only warns about run dirs on disk
# that the plan does NOT name (disk-minus-plan reconciliation).
def test_ma015_report_only_warns_on_unplanned_run_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness, "_run_child",
        lambda argv, log_path: pytest.fail("report-only must not train"),
    )
    out = tmp_path / "out"
    make_run_dir(out, "A", 1)
    make_run_dir(out, "B", 1, include_hazard_dynamics=True)
    make_run_dir(out, "A", 7)  # NOT in the plan below

    harness.main(
        ["--report-only", "--smoke", "--out-dir", str(out),
         "--arms", "A", "B", "--seeds", "1"]
    )
    report = json.loads((out / "hazard_efficacy_report.json").read_text())
    assert any(
        "A_seed7" in w and "NOT in the current plan" in w
        for w in report["warnings"]
    ), report["warnings"]


# ---------------------------------------------------------------------------
# MA-016 — qid uniqueness at every ingestion boundary
# ---------------------------------------------------------------------------


# Tests MA-016 [unit+integration]: duplicate qids fail loud at the eval
# boundary, the manifest-read boundary, and via the shared helper.
def test_ma016_duplicate_qids_fail_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = make_run_dir(tmp_path, "A", 1, write_eval_result=False)
    record = {"arm": "A", "seed": 1, "run_dir": run_dir, "hazard": False,
              "log_path": run_dir / "train.log"}

    def dup_eval(*args, **kwargs):
        row = {"qid": "q1", "sq": 0.1, "buzz_position": 2, "buzzed": True,
               "correct": True, "forced_correct": False, "confidence": 0.9,
               "episode_reward": 1.0, "n_steps": 2}
        return {"accuracy": 0.5, "runs": [dict(row), dict(row)]}

    monkeypatch.setattr(harness, "evaluate_t5_policy", dup_eval, raising=False)
    with pytest.raises(harness.HarnessError, match="MA-016"):
        harness.evaluate_run(record, _QUESTIONS, {})

    with pytest.raises(harness.ProvenanceError, match="MA-016"):
        harness.subset_questions_by_manifest(
            list(_QUESTIONS), ["q1", "q1"], split="test",
            manifest_path=tmp_path / "m.json",
        )

    with pytest.raises(harness.PreflightError, match="MA-016"):
        harness.assert_unique_qids(
            ["a", "a"], where="artifact", error_cls=harness.PreflightError
        )
    harness.assert_unique_qids(["a", "b"], where="artifact")  # clean passes


# ---------------------------------------------------------------------------
# MA-017 — shared child skips test eval; strict-JSON writers
# ---------------------------------------------------------------------------


# Tests MA-017 [unit]: the shared supervised argv composes --skip-test-eval
# and the REAL child parser accepts the full argv.
def test_ma017_shared_supervised_argv_skips_test_eval(tmp_path: Path) -> None:
    import scripts.train_t5_policy as train_t5_policy

    argv = harness.build_shared_supervised_argv(
        _namespace(tmp_path / "out"), tmp_path / "out"
    )
    assert "--skip-test-eval" in argv
    assert "--ppo-iterations" in argv
    parsed = train_t5_policy.parse_args(argv=[str(t) for t in argv[2:]])
    assert parsed.skip_test_eval is True
    assert parsed.ppo_iterations == 1


# Tests MA-017 [unit]: save_json(allow_nan=False) rejects non-finite floats;
# the default keeps every existing caller's behavior.
def test_ma017_save_json_allow_nan_false_rejects_non_finite(
    tmp_path: Path,
) -> None:
    from scripts._common import save_json

    with pytest.raises(ValueError):
        save_json(tmp_path / "x.json", {"v": float("-inf")}, allow_nan=False)
    # PR #41 round-2 resolve (P3 [23]): the old `... or True` here asserted
    # nothing. The real invariant: whatever the aborted strict write left
    # behind (nothing, or a truncated prefix) must NOT be a loadable JSON
    # document — no reader can mistake it for a complete artifact.
    strict_path = tmp_path / "x.json"
    if strict_path.exists():
        with pytest.raises(json.JSONDecodeError):
            json.loads(strict_path.read_text())

    legacy = save_json(tmp_path / "y.json", {"v": float("-inf")})
    assert "-Infinity" in legacy.read_text()


# ---------------------------------------------------------------------------
# MA-018 — LOW batch
# ---------------------------------------------------------------------------


# Tests MA-018 [unit]: git_dirty state pair (clean -> False, dirty -> True).
def test_ma018_git_dirty_state_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "_git_output", lambda *args: "")
    assert harness._git_dirty() is False
    monkeypatch.setattr(harness, "_git_output", lambda *args: " M scripts/x.py")
    assert harness._git_dirty() is True


# Tests MA-018 [unit]: seed domain (0 <= seed < 2**32) and non-finite
# --beta-terminal rejected at preflight.
def test_ma018_seed_domain_and_beta_finite_preflight(tmp_path: Path) -> None:
    with pytest.raises(harness.PreflightError, match="seed domain"):
        harness.plan_runs(_namespace(tmp_path / "o", seeds=[-1]))
    with pytest.raises(harness.PreflightError, match="seed domain"):
        harness.plan_runs(_namespace(tmp_path / "o", seeds=[2**32]))
    with pytest.raises(harness.PreflightError, match="beta-terminal"):
        harness.plan_runs(
            _namespace(tmp_path / "o", beta_terminal=float("inf"))
        )


# Tests MA-018 [unit]: the disk walk never follows symlinks (cycles, escapes)
# and counts only real files inside the tree.
def test_ma018_measure_disk_usage_symlink_hardened(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"12345")
    (root / "sub" / "b.bin").write_bytes(b"123")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"x" * 1000)
    (root / "loop").symlink_to(root, target_is_directory=True)
    (root / "esc").symlink_to(outside, target_is_directory=True)
    (root / "link.bin").symlink_to(outside / "big.bin")

    assert harness._measure_disk_usage(root) == 8


# Tests MA-018 [unit]: the metric-float validator for eval payload consumers.
def test_ma018_validated_metric() -> None:
    assert harness._validated_metric(None, key="x", where="w") is None
    assert harness._validated_metric(1.5, key="x", where="w") == pytest.approx(1.5)
    with pytest.raises(harness.ProvenanceError, match="non-finite"):
        harness._validated_metric(float("nan"), key="x", where="w")
    with pytest.raises(harness.ProvenanceError):
        harness._validated_metric(True, key="x", where="w")
    with pytest.raises(harness.ProvenanceError):
        harness._validated_metric("0.5", key="x", where="w")


# Tests MA-018 [integration]: the shared git-drift warning no longer claims
# the drift is "recorded" (nothing persists it for the shared checkpoint).
def test_ma018_shared_git_drift_wording_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    calls: list = []
    monkeypatch.setattr(harness, "_run_child", _sup_runner(calls))
    out = tmp_path / "out"
    harness._run_shared_supervised(_namespace(out), out)
    marker_path = (
        harness.shared_supervised_root(out) / harness.SHARED_SUPERVISED_MARKER
    )
    marker = json.loads(marker_path.read_text())
    marker["git_sha"] = "0" * 40
    marker_path.write_text(json.dumps(marker))

    capsys.readouterr()
    harness._run_shared_supervised(_namespace(out), out)
    warned = capsys.readouterr().out
    assert "WARNING" in warned
    assert "NOT persisted" in warned
    assert "drift recorded" not in warned


# ---------------------------------------------------------------------------
# Mini-audit-verify round (F1..F6)
# ---------------------------------------------------------------------------


# Tests F1 [unit+integration]: the weights fingerprint ignores
# training_state.pt (optimizer state, not weight content), so a
# --prune-checkpoints pass over the shared tree no longer breaks
# shared-checkpoint reuse.
def test_f1_fingerprint_ignores_training_state_and_reuse_survives_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unit: fingerprint invariant to optimizer-state presence.
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "policy_head.pt").write_bytes(b"weights-v1")
    (ckpt / "training_state.pt").write_bytes(b"optimizer-state")
    with_state = harness._weights_fingerprint(ckpt)
    (ckpt / "training_state.pt").unlink()
    assert harness._weights_fingerprint(ckpt) == with_state

    # Integration: marker built over a best_model CONTAINING
    # training_state.pt; pruning deletes the state file; reuse succeeds
    # (no MA-003 raise, no silent rebuild).
    calls: list = []

    def runner(argv, log_path):
        argv = [str(token) for token in argv]
        calls.append(argv)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("supervised ok\n")
        override = [
            t for t in argv if t.startswith("supervised.checkpoint_dir=")
        ][-1]
        best = Path(override.split("=", 1)[1]) / "supervised" / "best_model"
        best.mkdir(parents=True, exist_ok=True)
        (best / "policy_head.pt").write_bytes(b"weights-v1")
        (best / "training_state.pt").write_bytes(b"optimizer-state")
        return 0

    monkeypatch.setattr(harness, "_run_child", runner)
    out = tmp_path / "out"
    shared = harness._run_shared_supervised(_namespace(out), out)
    assert len(calls) == 1
    assert (Path(shared) / "training_state.pt").exists()

    harness.prune_shared_supervised_checkpoints(out)
    assert not (Path(shared) / "training_state.pt").exists()

    harness._run_shared_supervised(_namespace(out), out)
    assert len(calls) == 1, (
        "pruned optimizer state must not read as a checkpoint mutation "
        "(F1: reuse succeeds without a rebuild)"
    )


# Tests F2 [integration]: children launch UNBUFFERED (PYTHONUNBUFFERED=1 in
# the Popen env) while the parent environment is still inherited.
def test_f2_run_child_env_unbuffered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("F2_CANARY", "present")
    script = (
        "import os, sys\n"
        "ok = (os.environ.get('PYTHONUNBUFFERED') == '1'\n"
        "      and os.environ.get('F2_CANARY') == 'present')\n"
        "print('env-checked', flush=True)\n"
        "sys.exit(0 if ok else 7)\n"
    )
    log_path = tmp_path / "train.log"
    assert harness._run_child([sys.executable, "-c", script], log_path) == 0
    assert "env-checked" in log_path.read_text()


# Tests F2 [unit]: the default stall budget is 120 minutes (raised from 60).
def test_f2_stall_timeout_default_raised_to_120() -> None:
    assert harness.DEFAULT_STALL_TIMEOUT_MINUTES == 120.0
    assert harness.parse_args([]).stall_timeout_minutes == 120.0


# Tests F3 [integration]: with allow_abbrev=False on the child parser, an
# abbreviated identity flag smuggled through variant FLAGS dies at
# preflight as unrecognized instead of silently rebinding --model-path.
def test_f3_abbreviated_variant_flag_dies_at_preflight(tmp_path: Path) -> None:
    args = _namespace(tmp_path / "out", variant=["X:--model-pat=/other"])
    with pytest.raises(harness.PreflightError, match="unrecognized"):
        harness.plan_runs(args)


# Tests F3 [unit]: a doctored argv whose --model-path differs from the
# planned shared checkpoint dies at the roundtrip identity check.
def test_f3_doctored_model_path_dies_at_roundtrip(tmp_path: Path) -> None:
    plan = harness.plan_runs(_namespace(tmp_path / "out", arms=["A"], seeds=[1]))
    record = dict(plan[0])
    argv = [str(token) for token in record["argv"]]
    argv[argv.index("--model-path") + 1] = str(tmp_path / "evil" / "best_model")
    record["argv"] = argv
    with pytest.raises(harness.PreflightError, match="model-path"):
        harness._roundtrip_child_argv(record)


# Tests F4 [integration]: markers disagreeing on the MA-003 branch-point
# fingerprint fail verify_run_records loud; agreeing markers pass with a
# warning for legacy field-less ones.
def test_f4_mismatched_shared_fingerprints_fail_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "out"
    records = []
    for arm, fingerprint in (("A", "a" * 64), ("B", "b" * 64)):
        run_dir = make_run_dir(
            out, arm, 1,
            marker_extra={"shared_supervised_weights_sha256": fingerprint},
            include_hazard_dynamics=(arm == "B"),
        )
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A"})
    with pytest.raises(
        harness.ProvenanceError, match="mini-audit-verify F4"
    ) as excinfo:
        harness.verify_run_records(records)
    message = str(excinfo.value)
    assert "A_seed1" in message and "B_seed1" in message

    # One stamped + one legacy marker: passes with a legacy warning.
    out2 = tmp_path / "out2"
    records2 = []
    for arm, extra in (
        ("A", {"shared_supervised_weights_sha256": "c" * 64}),
        ("B", {}),
    ):
        run_dir = make_run_dir(out2, arm, 1, marker_extra=extra,
                               include_hazard_dynamics=(arm == "B"))
        records2.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                         "hazard": arm != "A"})
    capsys.readouterr()
    harness.verify_run_records(records2)
    printed = capsys.readouterr().out
    assert "legacy markers" in printed and "B_seed1" in printed


# Tests F5 [unit]: the =-joined knob form in variant FLAGS suppresses the
# harness's knob injection — exactly one --beta-terminal occurrence, and
# the recorded hazard identity reflects the variant's value.
def test_f5_eq_form_knob_flag_suppresses_injection(tmp_path: Path) -> None:
    plan = harness.plan_runs(
        _namespace(
            tmp_path / "out", seeds=[1], beta_terminal=2.5,
            variant=["Bq:--beta-terminal=0.5"],
        )
    )
    variant = next(rec for rec in plan if rec["arm"] == "variant:Bq")
    knob_tokens = [
        t for t in variant["argv"]
        if str(t).split("=", 1)[0] == "--beta-terminal"
    ]
    assert knob_tokens == ["--beta-terminal=0.5"], variant["argv"]
    assert variant["hazard_knobs"]["beta_terminal"] == pytest.approx(0.5)


# Tests F6 [integration]: a child that closes its output stream (EOF on the
# pipe) but never exits is killed after the stall budget instead of hanging
# the harness in an unbounded wait; the tee'd output survives in the log.
# EXTENDED per PR #41 review r3806602901: the post-EOF wait is bounded by
# the SMALLER of the stall budget and the REMAINING total budget — with the
# stall watchdog DISABLED, --child-timeout-minutes alone still kills the
# wedged child, and the error names whichever limit expired.
def test_f6_child_eof_without_exit_is_killed(tmp_path: Path) -> None:
    script = (
        "import os, sys, time\n"
        "print('bye', flush=True)\n"
        "os.close(1)\n"
        "os.close(2)\n"
        "time.sleep(60)\n"
    )
    log_path = tmp_path / "train.log"
    start = time.monotonic()
    with pytest.raises(harness.ChildRunError, match="did not exit") as excinfo:
        harness._run_child(
            [sys.executable, "-c", script], log_path,
            stall_timeout_seconds=0.5,
        )
    assert time.monotonic() - start < 20.0
    assert "stall-timeout-minutes" in str(excinfo.value)
    assert "bye" in log_path.read_text()

    # PR #41 r3806602901: total-cap-only configuration (stall watchdog
    # disabled) — proc.wait(timeout=None) used to block forever here
    # despite the configured hard runtime cap. The wait must be bounded by
    # the remaining total budget and the error must name the total cap.
    total_log = tmp_path / "train_total_cap.log"
    start = time.monotonic()
    with pytest.raises(harness.ChildRunError, match="did not exit") as excinfo:
        harness._run_child(
            [sys.executable, "-c", script], total_log,
            stall_timeout_seconds=0,  # <= 0 disables the stall watchdog
            child_timeout_seconds=1.0,
        )
    assert time.monotonic() - start < 20.0
    message = str(excinfo.value)
    assert "child-timeout-minutes" in message, message
    assert "stall-timeout-minutes" not in message, message
    assert "bye" in total_log.read_text()


# Tests PR #41 round-2 P3 [24] [integration]: the post-EOF wait with BOTH
# limits configured is bounded by the smaller of the two REMAINING budgets,
# and the error names whichever limit expired (the stall-only and
# total-only legs live in test_f6_child_eof_without_exit_is_killed).
# REWRITTEN in PR #41 round-3 resolve (R2-6 / codex r3809591780) to pin the
# documented CONTRACT, not the implementation: the stall limit is ONE
# CONTINUOUS no-output window — a child that goes silent and THEN EOFs must
# die when the continuous-silence window (measured from the last output
# line) crosses stall_limit, NOT stall_limit after EOF. The old
# implementation granted a FRESH full stall budget at EOF, stretching the
# silence window to silence-before-EOF + stall_limit.
def test_pr41_post_eof_wait_bounded_by_smaller_of_both_limits(
    tmp_path: Path,
) -> None:
    # (a) CONTRACT leg: 'bye', then ~2.0s of silence, then EOF, then a wedge
    # (never exits). stall=3.0s => ~1.0s of the silence window remains at
    # EOF, so the kill lands ~3s after the last output line. A fresh
    # post-EOF budget (the R2-6 bug) would let it live ~2.0 + 3.0 = ~5s.
    script = (
        "import os, sys, time\n"
        "print('bye', flush=True)\n"
        "time.sleep(2.0)\n"
        "os.close(1)\n"
        "os.close(2)\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    with pytest.raises(harness.ChildRunError, match="did not exit") as excinfo:
        harness._run_child(
            [sys.executable, "-c", script], tmp_path / "stall_smaller.log",
            stall_timeout_seconds=3.0,
            child_timeout_seconds=60.0,
        )
    elapsed = time.monotonic() - start
    assert elapsed >= 2.0, "the wait must reach the child's late EOF first"
    assert elapsed < 4.5, (
        "a continuous-silence window of 3.0s must kill ~3s after the last "
        "output line — a fresh post-EOF stall budget (R2-6) would stretch "
        f"it to ~5s; elapsed={elapsed:.2f}s"
    )
    message = str(excinfo.value)
    assert "stall-timeout-minutes" in message, message
    assert "child-timeout-minutes" not in message, message

    # (b) remaining total budget < remaining stall budget => the total cap
    # expires and is the one named (immediate-EOF child, so the stall
    # window is still nearly whole).
    eof_now_script = (
        "import os, sys, time\n"
        "print('bye', flush=True)\n"
        "os.close(1)\n"
        "os.close(2)\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    with pytest.raises(harness.ChildRunError, match="did not exit") as excinfo:
        harness._run_child(
            [sys.executable, "-c", eof_now_script],
            tmp_path / "total_smaller.log",
            stall_timeout_seconds=60.0,
            child_timeout_seconds=1.0,
        )
    assert time.monotonic() - start < 20.0
    message = str(excinfo.value)
    assert "child-timeout-minutes" in message, message
    assert "stall-timeout-minutes" not in message, message
