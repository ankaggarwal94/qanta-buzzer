"""Hazard-efficacy harness: pure-logic rules (R-006, R-007, R-009, R-012).

RED-phase tests pinning the interface of ``scripts/run_hazard_efficacy.py``:

- R-006: pre-committed primary endpoint (position gain >= 1.0 prefix AND
  accuracy tolerance 0.01, replicated in >= 2 of 3 seeds; zero-correct-buzz
  seeds are non-successes with ``undefined_position``).
- R-007: paired-by-qid bootstrap significance, seed-averaged pooling,
  gated at n >= 50 questions, deterministic, fail-loud on unpaired qids;
  ``bootstrap_ci`` identity-pinned to ``evaluation.controls``.
- R-009: report assembly schema (schema_version, endpoint_definition,
  scale block incl. disk usage, Device-2 caveat verbatim, structured
  verdict, plot path) for smoke and non-smoke inputs; per-run ece/brier
  surfaced under the REAL eval names; hazard_compute.wall_clock_seconds
  sourced from arm B's hazard_history.json (the hazard-phase wall clock —
  QA-006), with the PPO-dominated child total carried separately as
  child_total_wall_clock_seconds from arm B's RUN_COMPLETE.json marker.
- R-012: ``evaluate_t5_policy`` is the sole eval entry point; the harness
  module never touches ``TossupMCEnv``/``TextObservationWrapper``.

All fixtures are fabricated; no model or subprocess is touched.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import scripts.run_hazard_efficacy as harness
from tests._hazard_efficacy_fixtures import (
    current_git_sha,
    deep_key_scan,
    make_run_dir,
)

# Pinned verbatim in the test (not via the module constant) so the pin
# lives on the test side (R-009).
DEVICE2_SENTENCE = "Full-scale t5-large efficacy remains a Device-2 (RTX 5090) run."


# ---------------------------------------------------------------------------
# R-006 helpers
# ---------------------------------------------------------------------------


def _seed_record(
    seed: int,
    *,
    control_pos: float | None = 5.0,
    treatment_pos: float | None = 4.0,
    control_acc: float = 0.60,
    treatment_acc: float = 0.60,
    control_buzzes: int = 10,
    treatment_buzzes: int = 10,
) -> dict:
    return {
        "seed": seed,
        "control": {
            "mean_correct_buzz_position": control_pos,
            "accuracy": control_acc,
            "n_correct_policy_buzzes": control_buzzes,
        },
        "treatment": {
            "mean_correct_buzz_position": treatment_pos,
            "accuracy": treatment_acc,
            "n_correct_policy_buzzes": treatment_buzzes,
        },
    }


# Tests R-006 [unit]: inclusive thresholds — exactly 1.0 earlier and exactly
# -0.01 accuracy is a success.
def test_r006_exact_threshold_boundaries_are_successes() -> None:
    per_seed = [
        _seed_record(s, control_pos=5.0, treatment_pos=4.0,
                     control_acc=0.60, treatment_acc=0.59)
        for s in (1, 2, 3)
    ]
    result = harness.compute_primary_endpoint(per_seed)
    assert result["success"] is True
    assert result["n_seeds"] == 3
    assert result["n_seeds_replicated"] == 3
    assert all(rec["seed_success"] is True for rec in result["per_seed"])
    assert all(rec["undefined_position"] is False for rec in result["per_seed"])


# Tests R-006 [unit]: 0.99 prefixes earlier misses the 1.0 threshold.
def test_r006_position_gain_below_threshold_is_not_success() -> None:
    per_seed = [
        _seed_record(s, control_pos=5.0, treatment_pos=4.01)
        for s in (1, 2, 3)
    ]
    result = harness.compute_primary_endpoint(per_seed)
    assert result["success"] is False
    assert result["n_seeds_replicated"] == 0


# Tests R-006 [unit]: replication in exactly 2 of 3 seeds is a success.
def test_r006_two_of_three_replication_succeeds() -> None:
    per_seed = [
        _seed_record(1, control_pos=5.0, treatment_pos=3.5),
        _seed_record(2, control_pos=5.0, treatment_pos=3.9),
        _seed_record(3, control_pos=5.0, treatment_pos=5.0),  # no gain
    ]
    result = harness.compute_primary_endpoint(per_seed)
    assert result["success"] is True
    assert result["n_seeds_replicated"] == 2


# Tests R-006 [unit]: 1 of 3 seeds is not replicated.
def test_r006_one_of_three_is_not_replicated() -> None:
    per_seed = [
        _seed_record(1, control_pos=5.0, treatment_pos=3.0),
        _seed_record(2, control_pos=5.0, treatment_pos=5.0),
        _seed_record(3, control_pos=5.0, treatment_pos=4.9),
    ]
    result = harness.compute_primary_endpoint(per_seed)
    assert result["success"] is False
    assert result["n_seeds_replicated"] == 1


# Tests R-006 [unit]: an accuracy drop beyond 0.01 fails the seed even with
# a large position gain.
def test_r006_accuracy_drop_beyond_tolerance_fails_seed() -> None:
    per_seed = [
        _seed_record(s, control_pos=6.0, treatment_pos=3.0,
                     control_acc=0.60, treatment_acc=0.58)
        for s in (1, 2, 3)
    ]
    result = harness.compute_primary_endpoint(per_seed)
    assert result["success"] is False
    assert result["n_seeds_replicated"] == 0


# Tests R-006 [unit]: a zero-correct-buzz arm makes the seed a non-success
# carrying undefined_position; the other seeds can still replicate.
@pytest.mark.parametrize("zero_side", ["treatment", "control"])
def test_r006_zero_correct_buzz_seed_is_undefined_nonsuccess(zero_side: str) -> None:
    degenerate = _seed_record(3)
    degenerate[zero_side]["n_correct_policy_buzzes"] = 0
    degenerate[zero_side]["mean_correct_buzz_position"] = None
    per_seed = [
        _seed_record(1, control_pos=5.0, treatment_pos=3.5),
        _seed_record(2, control_pos=5.0, treatment_pos=3.5),
        degenerate,
    ]
    result = harness.compute_primary_endpoint(per_seed)
    assert result["success"] is True  # 2 of 3 still replicate
    assert result["n_seeds_replicated"] == 2
    rec3 = next(r for r in result["per_seed"] if r["seed"] == 3)
    assert rec3["seed_success"] is False
    assert rec3["undefined_position"] is True


# Tests R-006 [unit]: per-seed position x accuracy pairs are stored so the
# frontier is auditable.
def test_r006_per_seed_pairs_are_auditable() -> None:
    per_seed = [
        _seed_record(1, control_pos=5.5, treatment_pos=4.25,
                     control_acc=0.61, treatment_acc=0.62),
    ]
    result = harness.compute_primary_endpoint(per_seed)
    rec = result["per_seed"][0]
    assert rec["seed"] == 1
    assert rec["control_position"] == pytest.approx(5.5)
    assert rec["treatment_position"] == pytest.approx(4.25)
    assert rec["control_accuracy"] == pytest.approx(0.61)
    assert rec["treatment_accuracy"] == pytest.approx(0.62)


# Tests R-006 [unit]: defensive — empty input fails loud.
def test_r006_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        harness.compute_primary_endpoint([])


# Tests R-006 [unit] (PR #41 round-4, r3809814764): the endpoint's "2 of 3"
# has a DENOMINATOR — with n_seeds_planned=3, a 2-of-2 paired input (e.g.
# --seeds 1 2, or a planned run dir wholly missing under --report-only,
# intersected away by _endpoint_pairs_from_evals) is success=False with the
# additive coverage fields, even though BOTH surviving pairs replicate.
def test_r006_incomplete_seed_coverage_never_succeeds() -> None:
    per_seed = [
        _seed_record(1, control_pos=5.0, treatment_pos=3.5),
        _seed_record(2, control_pos=5.0, treatment_pos=3.5),
    ]
    result = harness.compute_primary_endpoint(per_seed, n_seeds_planned=3)
    assert result["success"] is False
    assert result["incomplete_seed_coverage"] is True
    assert result["n_seeds_planned"] == 3
    assert result["endpoint_definition_denominator"] == 3
    # The replication threshold itself is unchanged (>= 2): both pairs DID
    # replicate — coverage, not replication, is what failed.
    assert result["n_seeds"] == 2
    assert result["n_seeds_replicated"] == 2


# Tests R-006 [unit] (PR #41 round-4): full planned coverage keeps the
# pre-existing 2-of-3 success semantics and records the denominator.
def test_r006_full_planned_coverage_keeps_two_of_three_semantics() -> None:
    per_seed = [
        _seed_record(1, control_pos=5.0, treatment_pos=3.5),
        _seed_record(2, control_pos=5.0, treatment_pos=3.9),
        _seed_record(3, control_pos=5.0, treatment_pos=5.0),  # no gain
    ]
    result = harness.compute_primary_endpoint(per_seed, n_seeds_planned=3)
    assert result["success"] is True
    assert result["incomplete_seed_coverage"] is False
    assert result["n_seeds_planned"] == 3
    assert result["endpoint_definition_denominator"] == 3
    assert result["n_seeds"] == 3
    assert result["n_seeds_replicated"] == 2


# Tests R-006 [unit] (PR #41 round-4): omitting n_seeds_planned (legacy
# callers) preserves the PRIOR payload exactly — same success semantics,
# no coverage keys (additive-only signature change).
def test_r006_legacy_no_planned_denominator_keeps_prior_payload() -> None:
    per_seed = [
        _seed_record(1, control_pos=5.0, treatment_pos=3.5),
        _seed_record(2, control_pos=5.0, treatment_pos=3.5),
    ]
    result = harness.compute_primary_endpoint(per_seed)
    assert result["success"] is True  # prior behavior: >= 2 replicate
    assert set(result) == {
        "success", "n_seeds", "n_seeds_replicated", "per_seed"
    }


# Tests R-006 [unit] (PR #41 round-4): defensive — a non-positive, bool, or
# non-int planned count fails loud (a vacuous denominator would defeat the
# coverage requirement).
@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_r006_invalid_n_seeds_planned_raises(bad) -> None:
    with pytest.raises(ValueError, match="n_seeds_planned"):
        harness.compute_primary_endpoint(
            [_seed_record(1)], n_seeds_planned=bad
        )


# ---------------------------------------------------------------------------
# R-007 significance
# ---------------------------------------------------------------------------


def _sq_maps(n: int, *, delta: float, seeds=(1,)) -> tuple[dict, dict]:
    """Control ~0 with tiny index jitter; treatment = control + delta."""
    control = {
        s: {f"q{i}": 0.01 * (i % 3) for i in range(n)} for s in seeds
    }
    treatment = {
        s: {f"q{i}": 0.01 * (i % 3) + delta for i in range(n)} for s in seeds
    }
    return control, treatment


# Tests R-007 [unit]: n=100 with a large clean effect — CI excludes 0 and
# brackets the mean delta; labeled evaluable.
def test_r007_large_clean_effect_ci_excludes_zero() -> None:
    control, treatment = _sq_maps(100, delta=0.5)
    result = harness.compute_significance(control, treatment)
    assert result["n_questions"] == 100
    assert result["significance_evaluable"] is True
    assert result["significance"] == "paired_bootstrap_ci"
    assert result["mean_delta"] == pytest.approx(0.5)
    assert result["ci_low"] <= result["mean_delta"] <= result["ci_high"]
    assert result["ci_low"] > 0.0  # excludes zero for the obvious effect


# Tests R-007 [unit]: the >= 50 questions gate — 50 evaluable, 49 not.
def test_r007_scale_gate_boundary() -> None:
    control, treatment = _sq_maps(50, delta=0.5)
    at_50 = harness.compute_significance(control, treatment)
    assert at_50["significance_evaluable"] is True

    control, treatment = _sq_maps(49, delta=0.5)
    at_49 = harness.compute_significance(control, treatment)
    assert at_49["significance_evaluable"] is False
    assert at_49["significance"] == "not_evaluable_at_this_scale"
    # The numbers are still carried (only the significance LABEL is gated).
    assert at_49["n_questions"] == 49
    assert at_49["mean_delta"] == pytest.approx(0.5)


# Tests R-007 [unit]: pooling is seed-averaged S_q per qid per arm — never
# 3xN independent pairs.
def test_r007_seed_averaged_pooling() -> None:
    n = 60
    control = {
        1: {f"q{i}": 0.0 for i in range(n)},
        2: {f"q{i}": 0.2 for i in range(n)},
    }
    treatment = {
        1: {f"q{i}": 0.4 for i in range(n)},
        2: {f"q{i}": 0.6 for i in range(n)},
    }
    # per-qid arm means: control 0.1, treatment 0.5 -> delta exactly 0.4
    result = harness.compute_significance(control, treatment)
    assert result["mean_delta"] == pytest.approx(0.4)
    assert result["n_questions"] == n


# Tests R-007 [unit]: mismatched qid sets between arms fail loud.
def test_r007_mismatched_qids_across_arms_raise() -> None:
    control = {1: {"q1": 0.1, "q2": 0.2}}
    treatment = {1: {"q1": 0.3, "qX": 0.4}}
    with pytest.raises(ValueError):
        harness.compute_significance(control, treatment)


# Tests R-007 [unit]: a seed missing a qid within one arm fails loud.
def test_r007_missing_qid_within_arm_raises() -> None:
    control = {1: {"q1": 0.1, "q2": 0.2}, 2: {"q1": 0.1}}  # seed 2 lost q2
    treatment = {1: {"q1": 0.3, "q2": 0.4}, 2: {"q1": 0.3, "q2": 0.4}}
    with pytest.raises(ValueError):
        harness.compute_significance(control, treatment)


# Tests R-007 [unit]: the bootstrap is deterministic under the fixed seed.
def test_r007_bootstrap_deterministic() -> None:
    control, treatment = _sq_maps(64, delta=0.25)
    first = harness.compute_significance(control, treatment)
    second = harness.compute_significance(control, treatment)
    assert (first["ci_low"], first["ci_high"]) == (
        second["ci_low"], second["ci_high"]
    )


# Tests R-007 [unit]: defensive — empty inputs fail loud.
def test_r007_empty_inputs_raise() -> None:
    with pytest.raises(ValueError):
        harness.compute_significance({}, {})


# Tests R-007 [unit]: bootstrap reuse is MECHANICAL — the harness module
# binds evaluation/controls.py::bootstrap_ci as a module attribute (same
# identity-pin idiom as the R-012 evaluate_t5_policy assert), so the
# resampler cannot be silently reimplemented. Strict object identity is
# safe to assert again: the collection-time sys.modules eviction of
# evaluation.controls that used to false-fail this pin under full-suite
# runs is fixed (the CSLI test files now save/evict/restore the module
# object around their guarded scripts.compute_csli imports).
def test_r007_bootstrap_ci_imported_from_evaluation_controls() -> None:
    from evaluation import controls

    assert getattr(harness, "bootstrap_ci") is controls.bootstrap_ci, (
        "harness must import bootstrap_ci from evaluation.controls"
    )


# ---------------------------------------------------------------------------
# R-009 report assembly (fixture run dirs; no models, no subprocesses)
# ---------------------------------------------------------------------------


# Per-arm marker wall clocks: DISTINCT values so the report's provenance
# for hazard_compute.wall_clock_seconds (arm B's marker) is discriminable
# from picking A's or C's — or summing.
_ASSEMBLY_WALL_CLOCKS = {"A": 3.0, "B": 42.25, "C": 7.5}


def _assembly_fixture(tmp_path: Path) -> tuple[Path, list[dict]]:
    """Out dir with A/B/C seed-1 runs; B carries hazard dynamics."""
    out = tmp_path / "efficacy_out"
    records = []
    specs = {
        "A": {"mean_sq": 0.10, "accuracy": 0.50},
        "B": {"mean_sq": 0.30, "accuracy": 0.60},
        "C": {"mean_sq": 0.05, "accuracy": 0.40},
    }
    for arm, overrides in specs.items():
        run_dir = make_run_dir(
            out, arm, 1,
            eval_overrides=overrides,
            include_hazard_dynamics=(arm == "B"),
            marker_wall_clock_seconds=_ASSEMBLY_WALL_CLOCKS[arm],
        )
        records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                        "hazard": arm != "A", "resumed": False})
    return out, records


# Tests R-009 [unit]: full report schema — schema_version, git sha,
# endpoint_definition, scale block incl. disk usage, the Device-2 caveat
# VERBATIM, structured verdict, plot path — for smoke and non-smoke inputs.
@pytest.mark.parametrize("smoke", [True, False])
def test_r009_report_schema_caveat_and_plot(tmp_path: Path, smoke: bool) -> None:
    out, records = _assembly_fixture(tmp_path)
    report = harness.assemble_report(out, records, smoke=smoke)

    assert report["schema_version"] == 1
    assert report["git_sha"] == current_git_sha()

    endpoint_def = report["endpoint_definition"]
    assert isinstance(endpoint_def, str) and endpoint_def
    assert "1.0" in endpoint_def and "0.01" in endpoint_def

    scale = report["scale"]
    assert scale["model_name"] == "t5-small"
    assert scale["n_train"] == 4
    assert scale["n_val"] == 1
    assert scale["n_test"] == 3
    assert scale["ppo_iterations"] == 5
    assert scale["device"] == "cpu"
    assert isinstance(scale["disk_usage_bytes"], int)
    assert scale["disk_usage_bytes"] > 0

    assert isinstance(report["caveats"], list)
    assert DEVICE2_SENTENCE in report["caveats"]

    verdict = report["verdict"]
    assert isinstance(verdict, dict)
    assert set(verdict) >= {"verdict", "scope", "evidence"}
    assert isinstance(verdict["verdict"], str)

    assert report["plot_path"] == "hazard_efficacy_plot.png"
    assert (out / "hazard_efficacy_plot.png").exists()

    # The report itself is persisted and re-readable.
    on_disk = json.loads((out / "hazard_efficacy_report.json").read_text())
    assert on_disk["schema_version"] == 1

    # Endpoint + significance blocks are embedded; 3 test qids gate the
    # significance label (R-007 wiring).
    assert isinstance(report["endpoint"], dict)
    assert "success" in report["endpoint"]
    assert report["significance"]["significance"] == "not_evaluable_at_this_scale"

    # Per-run records carry resume + degeneracy diagnostics + provenance,
    # and surface the per-run calibration metrics under the REAL
    # evaluate_t5_policy names (ece / brier — from eval_result.json).
    assert len(report["runs"]) == len(records)
    for rec in report["runs"]:
        assert {"arm", "seed", "resumed", "policy_buzz_rate",
                "forced_commit_rate", "provenance"} <= set(rec)
        assert rec["ece"] == pytest.approx(0.10)
        assert rec["brier"] == pytest.approx(0.20)
        prov = rec["provenance"]
        assert {"model_name", "seed", "device", "git_sha", "git_dirty",
                "torch_version", "platform"} <= set(prov)


# Tests R-009/R-004-report [unit]: B-vs-A and C-vs-A deltas side by side,
# plus B's hazard optimizer-step count and wall-clock field.
def test_r009_arm_deltas_and_hazard_compute(tmp_path: Path) -> None:
    out, records = _assembly_fixture(tmp_path)
    report = harness.assemble_report(out, records, smoke=True)

    deltas = report["arm_deltas"]
    assert set(deltas) >= {"B_vs_A", "C_vs_A"}
    assert deltas["B_vs_A"]["mean_sq_delta"] == pytest.approx(0.20)
    assert deltas["B_vs_A"]["accuracy_delta"] == pytest.approx(0.10)
    assert deltas["C_vs_A"]["mean_sq_delta"] == pytest.approx(-0.05)
    assert deltas["C_vs_A"]["accuracy_delta"] == pytest.approx(-0.10)

    hazard_compute = report["hazard_compute"]
    # B's fixture hazard_history.json has exactly 4 optimizer steps.
    assert hazard_compute["optimizer_steps"] == 4
    # QA-006: wall_clock_seconds has a pinned HAZARD-PHASE data source —
    # arm B's hazard_history.json (fixture 3.75). It is NEVER the child-
    # total marker elapsed (B: 42.25), nor A's (3.0) or C's (7.5) markers,
    # nor a sum or mean across arms.
    wall_clock = hazard_compute["wall_clock_seconds"]
    assert isinstance(wall_clock, float) and not isinstance(wall_clock, bool)
    assert wall_clock == pytest.approx(3.75)
    assert wall_clock != pytest.approx(_ASSEMBLY_WALL_CLOCKS["B"])
    # The PPO-dominated child total keeps its own renamed field, still
    # sourced from arm B's RUN_COMPLETE.json marker (fixture 42.25).
    child_total = hazard_compute["child_total_wall_clock_seconds"]
    assert isinstance(child_total, float) and not isinstance(child_total, bool)
    assert child_total == pytest.approx(_ASSEMBLY_WALL_CLOCKS["B"])


# Tests QA-011 [unit]: aggregations reconcile against the PLAN — an arm
# contributing fewer seeds than planned to the per-qid S_q pool (missing or
# empty per-question runs records) surfaces as a report warning, never a
# silently smaller CI pool. A clean fixture yields zero warnings.
def test_qa011_report_warns_when_arm_contributes_fewer_seeds(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    records = []
    for arm in ("A", "B"):
        for seed in (1, 2):
            # B seed 2's eval payload lost its per-question runs records.
            eval_overrides = {"runs": []} if (arm, seed) == ("B", 2) else {}
            run_dir = make_run_dir(
                out, arm, seed,
                eval_overrides=eval_overrides,
                include_hazard_dynamics=(arm, seed) == ("B", 1),
            )
            records.append({"arm": arm, "seed": seed, "run_dir": run_dir,
                            "hazard": arm != "A", "resumed": False})

    report = harness.assemble_report(out, records, smoke=True)

    warnings = report["warnings"]
    assert isinstance(warnings, list) and warnings
    assert any("B" in w and "[2]" in w for w in warnings), (
        f"the dropped arm-B seed 2 must be named in the warnings: {warnings}"
    )
    assert not any("arm A" in w for w in warnings), "arm A is fully covered"

    # A fully-covered plan carries an EMPTY warnings list.
    clean_out = tmp_path / "clean"
    clean_records = []
    for arm in ("A", "B"):
        run_dir = make_run_dir(
            clean_out, arm, 1, include_hazard_dynamics=(arm == "B")
        )
        clean_records.append({"arm": arm, "seed": 1, "run_dir": run_dir,
                              "hazard": arm != "A", "resumed": False})
    clean_report = harness.assemble_report(clean_out, clean_records, smoke=True)
    assert clean_report["warnings"] == []


# Tests R-005/R-009 [unit]: Expected Wins is excluded from the default report.
def test_r009_no_expected_wins_key_anywhere(tmp_path: Path) -> None:
    out, records = _assembly_fixture(tmp_path)
    report = harness.assemble_report(out, records, smoke=True)
    assert deep_key_scan(report, "expected_wins") == []
    assert deep_key_scan(report, "expectedwins") == []


# Tests R-010b/R-009 [unit]: the report hazard_dynamics block carries the
# per-position means before/after, the expected-buzz-time delta, and the
# first/second-half hazard-loss means from the per-run fixture.
def test_r009_hazard_dynamics_block_from_fixture(tmp_path: Path) -> None:
    out, records = _assembly_fixture(tmp_path)
    report = harness.assemble_report(out, records, smoke=True)

    dyn = report["hazard_dynamics"]
    assert dyn["per_position_mean_before"] == pytest.approx([0.1, 0.1, 0.1, 0.1])
    assert dyn["per_position_mean_after"] == pytest.approx([0.3, 0.4, 0.5, 0.6])
    assert dyn["expected_buzz_time_delta"] == pytest.approx(-1.4)
    assert dyn["first_half_mean_loss"] == pytest.approx(4.0)
    assert dyn["second_half_mean_loss"] == pytest.approx(2.0)


# Tests R-009 [unit]: the plot is produced headlessly via savefig, never
# plt.show(). Code-level evidence is required (docstrings don't count).
def test_r009_plot_uses_agg_savefig_never_show() -> None:
    src = inspect.getsource(harness)
    assert (
        'matplotlib.use("Agg")' in src or "matplotlib.use('Agg')" in src
    ), "plot code must select the headless Agg backend before pyplot"
    assert "savefig(" in src, "plot must be written via savefig(...)"
    assert "plt.show" not in src and ".show()" not in src


# ---------------------------------------------------------------------------
# R-012 reuse enforcement
# ---------------------------------------------------------------------------


# Tests R-012 [unit]: evaluate_t5_policy is imported from
# scripts.compare_policies as the sole eval entry point, and the harness
# module never constructs environments/rollouts itself.
def test_r012_shared_eval_entrypoint_and_no_env_reimplementation() -> None:
    import scripts.compare_policies as compare_policies

    assert (
        getattr(harness, "evaluate_t5_policy")
        is compare_policies.evaluate_t5_policy
    ), "harness must import evaluate_t5_policy from scripts.compare_policies"

    src = inspect.getsource(harness)
    assert "TossupMCEnv" not in src, "harness must not build environments"
    assert "TextObservationWrapper" not in src, "harness must not run rollouts"


# Tests R-012 [unit]: the eval stage is a thin wrapper — the monkeypatched
# entry point is called exactly once per run and nothing else evaluates.
def test_r012_eval_stage_calls_entrypoint_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    run_dir = make_run_dir(tmp_path, "A", 1, write_eval_result=False)
    record = {"arm": "A", "seed": 1, "run_dir": run_dir, "hazard": False,
              "log_path": run_dir / "train.log"}
    questions = [SimpleNamespace(qid=q) for q in ("q1", "q2", "q3")]

    calls: list = []

    def fake_eval(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "policy_accuracy": 0.5,
            "mean_sq": 0.1,
            "runs": [
                {"qid": q.qid, "sq": 0.1, "buzz_position": 2, "buzzed": True,
                 "correct": True, "forced_correct": False, "confidence": 0.9,
                 "episode_reward": 1.0, "n_steps": 2}
                for q in questions
            ],
        }

    monkeypatch.setattr(harness, "evaluate_t5_policy", fake_eval, raising=False)
    harness.evaluate_run(record, questions, {})
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# PR #41 round-2 resolve — real-producer scale mirror [8], R-005
# buzz-position histogram (P3 [27]), _arm_metric_mean guard (P3 [6])
# ---------------------------------------------------------------------------


# Tests PR #41 round-2 [8] [unit]: the report scale block's manifest count
# fields are pinned against the REAL producer — _build_scale consumes a
# manifest built by scripts/train_t5_policy.py::_build_split_manifest itself
# (not a fixture replica), so a producer field rename can never ship behind
# fixture-only green.
def test_pr41_scale_block_consumes_real_producer_manifest_counts(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from scripts.train_t5_policy import _build_split_manifest

    train = [SimpleNamespace(qid=f"t{i}") for i in range(4)]
    val = [SimpleNamespace(qid="v0")]
    test = [SimpleNamespace(qid=f"q{i}") for i in range(3)]
    manifest = _build_split_manifest(
        source="persisted_artifacts",
        mc_path=None,
        train_questions=train,
        val_questions=val,
        test_questions=test,
    )
    # The REAL producer emits the exact count fields the scale block reads.
    assert manifest["train_count"] == 4
    assert manifest["val_count"] == 1
    assert manifest["test_count"] == 3

    config_used = {
        "model": {"model_name": "t5-small", "device": "cpu"},
        "ppo": {"iterations": 5},
    }
    scale = harness._build_scale(config_used, manifest, tmp_path)
    assert scale["n_train"] == 4
    assert scale["n_val"] == 1
    assert scale["n_test"] == 3
    assert scale["ppo_iterations"] == 5
    assert scale["device"] == "cpu"


# Tests PR #41 round-2 P3 [27] / R-005 [unit]: the buzz-position histogram
# helper counts ONLY real policy buzzes (0-indexed positions as string keys)
# and ignores forced commits, null/bool/non-finite positions.
def test_pr41_r005_buzz_position_histogram_helper() -> None:
    runs = [
        {"qid": "q1", "buzzed": True, "buzz_position": 2},
        {"qid": "q2", "buzzed": True, "buzz_position": 2},
        {"qid": "q3", "buzzed": True, "buzz_position": 4},
        {"qid": "q4", "buzzed": False, "buzz_position": None},  # forced
        {"qid": "q5", "buzzed": True, "buzz_position": None},   # degenerate
        {"qid": "q6", "buzzed": True, "buzz_position": True},   # bool guard
        {"qid": "q7", "buzzed": True, "buzz_position": float("nan")},
    ]
    assert harness._buzz_position_histogram(runs) == {"2": 2, "4": 1}
    assert harness._buzz_position_histogram([]) == {}


# Tests PR #41 round-2 P3 [27] / R-005 [unit]: the report carries the
# per-ARM buzz-position histograms (aggregated across seeds from the runs
# records) and each per-run row carries its own histogram — both additive.
def test_pr41_r005_report_carries_buzz_position_histograms(
    tmp_path: Path,
) -> None:
    out, records = _assembly_fixture(tmp_path)
    report = harness.assemble_report(out, records, smoke=True)

    # Fixture eval runs per arm: one buzz at position 4 (correct), one at
    # position 2 (incorrect), one forced commit.
    assert set(report["buzz_position_histograms"]) == {"A", "B", "C"}
    for arm in ("A", "B", "C"):
        assert report["buzz_position_histograms"][arm] == {"2": 1, "4": 1}
    for row in report["runs"]:
        assert row["buzz_position_histogram"] == {"2": 1, "4": 1}

    # The persisted report carries them too.
    on_disk = json.loads((out / "hazard_efficacy_report.json").read_text())
    assert on_disk["buzz_position_histograms"]["A"] == {"2": 1, "4": 1}


# Tests PR #41 round-2 P3 [27] / R-005 [unit]: evaluate_run enriches (and
# persists) the per-run buzz_position_histogram alongside the correct-buzz
# mean.
def test_pr41_r005_evaluate_run_persists_histogram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    run_dir = make_run_dir(tmp_path, "A", 1, write_eval_result=False)
    record = {"arm": "A", "seed": 1, "run_dir": run_dir, "hazard": False,
              "log_path": run_dir / "train.log"}
    questions = [SimpleNamespace(qid=q) for q in ("q1", "q2", "q3")]

    def fake_eval(*args, **kwargs):
        return {
            "accuracy": 0.5,
            "mean_sq": 0.1,
            "runs": [
                {"qid": "q1", "sq": 0.1, "buzz_position": 2, "buzzed": True,
                 "correct": True, "forced_correct": False, "confidence": 0.9,
                 "episode_reward": 1.0, "n_steps": 2},
                {"qid": "q2", "sq": 0.1, "buzz_position": 2, "buzzed": True,
                 "correct": False, "forced_correct": False, "confidence": 0.5,
                 "episode_reward": -1.0, "n_steps": 2},
                {"qid": "q3", "sq": 0.1, "buzz_position": None,
                 "buzzed": False, "correct": False, "forced_correct": True,
                 "confidence": None, "episode_reward": 0.5, "n_steps": 6},
            ],
        }

    monkeypatch.setattr(harness, "evaluate_t5_policy", fake_eval, raising=False)
    enriched = harness.evaluate_run(record, questions, {})
    assert enriched["buzz_position_histogram"] == {"2": 2}
    on_disk = json.loads((run_dir / "eval_result.json").read_text())
    assert on_disk["buzz_position_histogram"] == {"2": 2}


# Tests PR #41 round-2 P3 [6] [unit]: _arm_metric_mean mirrors the plot
# helper's guard — bools never average as 0/1 and non-finite values never
# poison an arm delta.
def test_pr41_arm_metric_mean_excludes_bools_and_non_finite() -> None:
    eval_by_run = {
        ("A", 1): {"mean_sq": 0.2},
        ("A", 2): {"mean_sq": True},          # bool: int subclass, excluded
        ("A", 3): {"mean_sq": float("nan")},  # non-finite: excluded
        ("B", 1): {"mean_sq": 0.4},
    }
    assert harness._arm_metric_mean(eval_by_run, "A", "mean_sq") == pytest.approx(0.2)
    assert harness._arm_metric_mean(eval_by_run, "B", "mean_sq") == pytest.approx(0.4)
    only_junk = {("A", 1): {"mean_sq": True}, ("A", 2): {"mean_sq": float("inf")}}
    assert harness._arm_metric_mean(only_junk, "A", "mean_sq") is None
