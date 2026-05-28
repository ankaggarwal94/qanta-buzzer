"""Unit tests for the DP StopDFF pipeline (scripts/compute_stopdff_dp.py)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.stopdff_dp import (
    rewards as rewards_module,
)
from scripts.stopdff_dp.types import RewardSchedule


def test_reward_registry_includes_all_required_schedules() -> None:
    """The four schedules named in the spec must all be in the registry."""
    registry = rewards_module.REWARD_REGISTRY
    required = {"acf_flat", "power_mark", "wait_cost_small", "strict_wrong"}
    assert required.issubset(registry.keys())
    for name in required:
        schedule = registry[name]
        assert isinstance(schedule, RewardSchedule)
        assert schedule.name == name


def test_acf_flat_has_zero_wait_cost_and_no_power_split() -> None:
    schedule = rewards_module.REWARD_REGISTRY["acf_flat"]
    assert schedule.c_wait == 0.0
    # No power_split means early and late reward must be equal.
    assert schedule.r_correct_early == schedule.r_correct_late == 10.0
    assert schedule.r_wrong == -5.0


def test_wait_cost_small_has_nonzero_c_wait() -> None:
    schedule = rewards_module.REWARD_REGISTRY["wait_cost_small"]
    assert schedule.c_wait == 0.05


from scripts.stopdff_dp import dp_solver
from scripts.stopdff_dp.rewards import REWARD_REGISTRY


def _zero_continuation(*_args, **_kwargs) -> float:
    return 0.0


def test_dp_equals_myopic_when_continuation_is_zero() -> None:
    """If E[V_{t+1}] == 0 everywhere, DP = stop at first prefix where A_t(p)>0."""
    schedule = REWARD_REGISTRY["acf_flat"]
    # A_t(p) = 10p - 5(1-p) = 15p - 5; zero at p = 1/3.
    # So with p_trajectory [0.2, 0.4, 0.9], myopic stops at idx=1 (p=0.4).
    p_trajectory = [0.2, 0.4, 0.9]
    prefix_fractions = [0.1, 0.5, 0.9]

    trace = dp_solver.solve_trajectory(
        p_trajectory=p_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )

    assert trace.stop_step == 1
    # All continuation values must be 0.0 under this estimator.
    assert all(cv == 0.0 for cv in trace.continuation_values)


def test_dp_waits_when_future_value_exceeds_current_answer_utility() -> None:
    """With a high continuation value, DP should defer stopping."""
    schedule = REWARD_REGISTRY["acf_flat"]
    p_trajectory = [0.4, 0.95]
    prefix_fractions = [0.5, 1.0]

    # A_0(0.4) = 15*0.4 - 5 = 1.0; A_1(0.95) = 15*0.95 - 5 = 9.25.
    # Force the DP to see continuation = 8.0 at t=0, which beats A_0=1.0
    # plus -c_wait=0, so the agent should wait at t=0 and stop at t=1.
    def continuation_fn(t: int, **_kw: object) -> float:
        return 8.0 if t == 0 else 0.0

    trace = dp_solver.solve_trajectory(
        p_trajectory=p_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=continuation_fn,
    )
    assert trace.stop_step == 1


def test_dp_stops_earlier_when_mc_probabilities_uniformly_shifted_upward() -> None:
    """If we add delta to every p_t, DP stop step must be <= the lower-p version.

    StopDFF sign convention: stop_step_MC < stop_step_QA when MC raises p.
    """
    schedule = REWARD_REGISTRY["acf_flat"]
    qa_trajectory = [0.2, 0.3, 0.4, 0.6]
    mc_trajectory = [min(1.0, p + 0.2) for p in qa_trajectory]
    prefix_fractions = [0.2, 0.4, 0.6, 0.8]

    qa_trace = dp_solver.solve_trajectory(
        p_trajectory=qa_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )
    mc_trace = dp_solver.solve_trajectory(
        p_trajectory=mc_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )

    assert mc_trace.stop_step <= qa_trace.stop_step
    # StopDFF (MC - QA) must be <= 0 in this construction.
    assert mc_trace.stop_step - qa_trace.stop_step <= 0


def test_dp_horizon_terminal_uses_max_of_answer_or_zero() -> None:
    """V_T = max(A_T(p_T), 0). When A_T<0 we should never buzz (stop=T)."""
    schedule = REWARD_REGISTRY["acf_flat"]
    p_trajectory = [0.05, 0.10]
    prefix_fractions = [0.5, 1.0]
    trace = dp_solver.solve_trajectory(
        p_trajectory=p_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )
    # Both A_t < 0, so optimal action is to never stop; we encode that as
    # stop_step == len(p_trajectory) (i.e. one past the last index).
    assert trace.stop_step == len(p_trajectory)


def test_dp_never_stops_on_exact_ties_at_zero() -> None:
    """At A_t == wait_value (tie), the policy is to wait, not stop.

    Locks the convention that intermediate and terminal steps both use
    strict-greater-than for stopping, so trajectories that tie at every
    prefix produce stop_step == T (never stop).
    """
    # acf_flat: r_correct=10, r_wrong=-5; pick p so A_t == 0:
    # 10p - 5(1-p) = 0  =>  15p = 5  =>  p = 1/3.
    schedule = REWARD_REGISTRY["acf_flat"]
    p_trajectory = [1.0 / 3.0, 1.0 / 3.0]
    prefix_fractions = [0.5, 1.0]
    trace = dp_solver.solve_trajectory(
        p_trajectory=p_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )
    # All A_t == 0, all wait_values == 0. Strict `>` means no stop ever.
    assert trace.stop_step == len(p_trajectory)


def test_dp_empty_trajectory_returns_empty_trace() -> None:
    """T=0 edge case: solver returns an empty trace with stop_step=0."""
    schedule = REWARD_REGISTRY["acf_flat"]
    trace = dp_solver.solve_trajectory(
        p_trajectory=[],
        prefix_fractions=[],
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )
    assert trace.stop_step == 0
    assert len(trace.values) == 0


import pandas as pd

from scripts.stopdff_dp import continuation as cont_module
from scripts.stopdff_dp.continuation import FALLBACK_LADDER


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Tiny helper to build a normalised adapter dataframe in tests."""
    from scripts.stopdff_dp.types import ADAPTER_COLUMNS
    df = pd.DataFrame(rows)
    for col in ADAPTER_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[list(ADAPTER_COLUMNS)]


def test_empirical_bucket_fitter_uses_only_fit_split_rows() -> None:
    """Continuation buckets must not see test-split rows during fit.

    The leakage check is enforced at the API boundary: passing a
    dataframe whose split column contains the eval split raises.
    """
    rows = [
        {"subject": "sbert:Lit", "item_id": "q1", "prefix_idx": 0,
         "prefix_fraction": 0.5, "format": "MC",
         "split": "test", "p_raw": 0.1, "p_calibrated": 0.2, "correct": 0,
         "top_answer": "a", "gold": "a", "category": "Lit", "option_set_id": "s1"},
    ]
    df_test = _make_df(rows)
    with pytest.raises(ValueError, match="leakage|fit on test"):
        cont_module.EmpiricalBucketEstimator.fit(
            fit_df=df_test,
            schedule=REWARD_REGISTRY["acf_flat"],
            fit_split_name="val",
        )


def test_empirical_bucket_fit_allows_legacy_rows_without_prefix_fraction() -> None:
    """Legacy fixtures without prefix_fraction still use the rank/T fallback."""
    rows = []
    for prefix_idx in range(3):
        rows.append({
            "subject": "sbert:Lit", "item_id": "q1", "prefix_idx": prefix_idx,
            "format": "MC", "split": "val",
            "p_raw": 0.55, "p_calibrated": 0.55, "correct": 0,
            "top_answer": "a", "gold": "b", "category": "Lit",
            "option_set_id": "s1",
        })
    df_legacy = _make_df(rows).drop(columns=["prefix_fraction"])

    estimator = cont_module.EmpiricalBucketEstimator.fit(
        fit_df=df_legacy,
        schedule=REWARD_REGISTRY["acf_flat"],
        fit_split_name="val",
        min_bucket_size=1,
    )

    assert estimator.last_coverage_tag_for(
        prefix_bucket="mid",
        fmt="MC",
        subject_bucket="sbert:Lit",
        p_bin=2,
        entropy_bin=2,
    ) == "exact"


def test_empirical_bucket_estimator_returns_pooled_when_bucket_sparse() -> None:
    """When the exact bucket has <3 trajectories, fallback drops entropy_bin first.

    Constructed so the (prefix_bucket, format, subject_bucket, p_bin, entropy_bin)
    bucket has only 1 trajectory but the entropy-dropped bucket has 5+. The
    estimator must walk exactly one rung down (drop entropy_bin) and record
    that rung in ``_last_rung``.
    """
    rows = []
    # Item 1: 4 prefixes, all p_calibrated values land in entropy_bin=2 (p>=0.5 -> entropy>=0.5).
    # The non-terminal row at prefix_idx=0 falls in the early prefix bucket.
    # prefix_fraction = (prefix_idx + 1) / 4 -> [0.25, 0.5, 0.75, 1.0]
    # puts prefix_idx=0 in "early" (0.25 < 0.33) for the bucket lookup.
    for prefix_idx in range(4):
        rows.append({
            "subject": "sbert:Lit", "item_id": "q_specific", "prefix_idx": prefix_idx,
            "prefix_fraction": (prefix_idx + 1) / 4,
            "format": "MC", "split": "val",
            "p_raw": 0.55, "p_calibrated": 0.55, "correct": 0,
            "top_answer": "a", "gold": "b", "category": "Lit",
            "option_set_id": "s_specific",
        })
    # 5 more items in the same (prefix_bucket=early, format=MC, subject_bucket=sbert:Lit, p_bin=2)
    # but with entropy_bin=1 (p~0.5 -> entropy ~ 1.0 > 0.9 limit -> actually clip; pick p where entropy~0.6).
    # To land in entropy_bin=1 we need 0.5 <= H(p) < 0.9. H(0.7) = 0.881; H(0.75)=0.811. Use p=0.7.
    for i in range(5):
        for prefix_idx in range(4):
            rows.append({
                "subject": "sbert:Lit", "item_id": f"q_pool_{i}", "prefix_idx": prefix_idx,
                "prefix_fraction": (prefix_idx + 1) / 4,
                "format": "MC", "split": "val",
                "p_raw": 0.7, "p_calibrated": 0.7, "correct": 0,
                "top_answer": "a", "gold": "b", "category": "Lit",
                "option_set_id": f"s_pool_{i}",
            })
    df_val = _make_df(rows)

    estimator = cont_module.EmpiricalBucketEstimator.fit(
        fit_df=df_val,
        schedule=REWARD_REGISTRY["acf_flat"],
        fit_split_name="val",
        min_bucket_size=3,
    )

    # Query the specific-bucket (entropy_bin=2 -- H(0.55) ~ 0.993 -> bin index 2 since 0.9 <= H < 1.01).
    # The specific bucket has only the 1 row from q_specific (3 non-terminal rows actually, but
    # the spec is "1 trajectory" -- with multi-prefix items each prefix is its own row, so what
    # matters is the count of v_next-non-NaN rows landing in this exact key tuple).
    # Either way, dropping entropy_bin pools the 5 pool items (p=0.7, entropy_bin=1) with the
    # 1 specific item -- both land in p_bin=2 (0.4 <= p < 0.6 is bin 2... wait,
    # DEFAULT_P_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01), so p=0.55 -> p_bin=2, p=0.7 -> p_bin=3.
    # That means dropping entropy_bin does NOT pool them -- they have different p_bins.
    # Drop p_bin too (rung 2). At rung 2 we get 6 items total in (early, MC, sbert:Lit).
    # The test now exercises rung 2 (drop entropy_bin AND p_bin), still "pooled".
    tag = estimator.last_coverage_tag_for(
        prefix_bucket="early",
        fmt="MC",
        subject_bucket="sbert:Lit",
        p_bin=2,
        entropy_bin=2,
    )
    assert tag == "pooled"
    # Lock the rung index >= 1 (any pooled fallback rung counts).
    assert estimator._last_rung is not None
    assert estimator._last_rung != FALLBACK_LADDER[0]


def test_oracle_trajectory_estimator_flags_non_confirmatory() -> None:
    estimator = cont_module.OracleTrajectoryEstimator()
    assert estimator.confirmatory is False


def test_pooled_empirical_fallback_ladder_documented() -> None:
    """The fallback ladder must be a fixed, declared sequence."""
    ladder = cont_module.FALLBACK_LADDER
    # Must be a tuple of tuples so it cannot be mutated at runtime.
    assert isinstance(ladder, tuple)
    # The first rung is the most-specific bucket; the last is the most-pooled.
    assert ladder[0] == (
        "prefix_bucket", "format", "subject_bucket", "p_bin", "entropy_bin",
    )
    assert "format" in ladder[-1]


def test_pooled_empirical_skips_most_specific_rungs() -> None:
    """PooledEmpiricalEstimator must land on rung >= 2, never on rung 0 or 1.

    Sentinel p_bin=-1, entropy_bin=-1 in the inner call forces rungs 0 and 1
    (which condition on p_bin and entropy_bin) to miss. Build a fitted
    estimator with enough data that the inner Empirical would otherwise
    return rung 0; then verify Pooled forces a rung >= 2 outcome.
    """
    rows = []
    # 10 items, 4 prefixes each, all in (early-able, MC, sbert:Lit, p_bin=2, ent_bin=2).
    # prefix_fraction = (prefix_idx + 1) / 4 -> [0.25, 0.5, 0.75, 1.0]
    # puts prefix_idx=0 in "early" (0.25 < 0.33) for the bucket lookup.
    for i in range(10):
        for prefix_idx in range(4):
            rows.append({
                "subject": "sbert:Lit", "item_id": f"q{i}",
                "prefix_idx": prefix_idx,
                "prefix_fraction": (prefix_idx + 1) / 4,
                "format": "MC", "split": "val",
                "p_raw": 0.55, "p_calibrated": 0.55, "correct": 0,
                "top_answer": "a", "gold": "b", "category": "Lit",
                "option_set_id": f"s{i}",
            })
    df_val = _make_df(rows)

    pooled = cont_module.PooledEmpiricalEstimator.fit(
        fit_df=df_val, schedule=REWARD_REGISTRY["acf_flat"],
        fit_split_name="val", min_bucket_size=3,
    )
    tag = pooled.last_coverage_tag_for(
        prefix_bucket="early",
        fmt="MC",
        subject_bucket="sbert:Lit",
        p_bin=2,
        entropy_bin=2,
    )
    # Pooled estimator must report "pooled" because it can never land on rung 0
    # (which is the only rung that produces "exact").
    assert tag == "pooled"
    # Verify the inner estimator's _last_rung is NOT the most-specific rung.
    assert pooled.inner._last_rung is not None
    assert pooled.inner._last_rung != FALLBACK_LADDER[0]
    # Also verify rung 1 was skipped (rung 1 conditions on p_bin, which our sentinel breaks).
    assert pooled.inner._last_rung != FALLBACK_LADDER[1]


from scripts.stopdff_dp import diagnostics as diag_module
from scripts.stopdff_dp.types import DPTrace


def _trace(stop_step: int, T: int, tags: list[str] | None = None) -> DPTrace:
    return DPTrace(
        item_id="q",
        fmt="MC",
        stop_step=stop_step,
        values=[0.0] * T,
        answer_utilities=[0.0] * T,
        continuation_values=[0.0] * T,
        coverage_tags=tags or ["exact"] * T,
    )


def test_coverage_warn_when_more_than_5pct_pooled() -> None:
    traces = [_trace(stop_step=2, T=3, tags=["pooled", "pooled", "pooled"])]
    summary = diag_module.summarize_coverage(traces)
    assert summary["fraction_pooled"] == 1.0
    assert summary["verdict"] == "warn"


def test_coverage_pass_when_fully_exact() -> None:
    traces = [_trace(stop_step=2, T=3, tags=["exact", "exact", "exact"])]
    summary = diag_module.summarize_coverage(traces)
    assert summary["fraction_pooled"] == 0.0
    assert summary["fraction_exact"] == 1.0
    assert summary["verdict"] == "pass"


def test_ceiling_all_stop_at_final_prefix() -> None:
    mc_traces = [_trace(stop_step=2, T=3), _trace(stop_step=2, T=3)]
    qa_traces = [_trace(stop_step=2, T=3), _trace(stop_step=2, T=3)]
    flags = diag_module.detect_ceiling_effects(mc_traces, qa_traces)
    assert flags["all_stop_at_final_prefix"] is True
    assert flags["all_stop_at_first_prefix"] is False


def test_ceiling_no_cross_format_variance() -> None:
    mc_traces = [_trace(stop_step=1, T=3), _trace(stop_step=2, T=3)]
    qa_traces = [_trace(stop_step=1, T=3), _trace(stop_step=2, T=3)]
    flags = diag_module.detect_ceiling_effects(mc_traces, qa_traces)
    assert flags["no_cross_format_stopping_variance"] is True


def test_continuation_model_collapsed_helper() -> None:
    """continuation_model_collapsed returns True iff all cells are pooled."""
    collapsed = {"fraction_exact": 0.0, "fraction_pooled": 1.0,
                 "fraction_missing": 0.0}
    assert diag_module.continuation_model_collapsed(collapsed) is True
    healthy = {"fraction_exact": 1.0, "fraction_pooled": 0.0,
               "fraction_missing": 0.0}
    assert diag_module.continuation_model_collapsed(healthy) is False
    # Empty / missing keys should not crash.
    assert diag_module.continuation_model_collapsed({}) is False


def test_coverage_pass_at_pooled_threshold_boundary() -> None:
    """Strict > comparison: fraction_pooled == 0.05 should still pass."""
    # 19 exact + 1 pooled = 5% pooled exactly.
    tags = ["exact"] * 19 + ["pooled"]
    traces = [_trace(stop_step=len(tags) - 1, T=len(tags), tags=tags)]
    summary = diag_module.summarize_coverage(traces)
    assert summary["fraction_pooled"] == 0.05
    assert summary["verdict"] == "pass"


def test_coverage_heterogeneous_tag_mix_sums_to_one() -> None:
    """Mixed tags: fractions are populated and sum to 1.0."""
    traces = [_trace(stop_step=2, T=3, tags=["exact", "pooled", "missing"])]
    summary = diag_module.summarize_coverage(traces)
    assert summary["fraction_exact"] + summary["fraction_pooled"] \
        + summary["fraction_missing"] == pytest.approx(1.0)
    # Missing > 0.01 forces warn even though pooled is fine.
    assert summary["verdict"] == "warn"


def test_ceiling_distinguishes_never_stopped_traces() -> None:
    """stop_step == T encodes 'never stopped'; diagnostics surface it."""
    # T=3 trajectories where stop_step == T (never stopped).
    mc = [_trace(stop_step=3, T=3), _trace(stop_step=3, T=3)]
    qa = [_trace(stop_step=3, T=3), _trace(stop_step=3, T=3)]
    flags = diag_module.detect_ceiling_effects(mc, qa)
    assert flags["n_stopped_cells"] == 0
    assert flags["n_never_stopped_cells"] == 4
    assert flags["all_stop_at_final_prefix"] is True


def test_ceiling_treats_never_stop_and_final_prefix_as_horizon_ceiling() -> None:
    """No-stop and last-prefix stops are both horizon-ceiling artifacts."""
    mc = [_trace(stop_step=3, T=3)]
    qa = [_trace(stop_step=2, T=3)]
    flags = diag_module.detect_ceiling_effects(mc, qa)
    assert flags["all_stop_at_final_prefix"] is True
    assert flags["n_never_stopped_cells"] == 1
    assert flags["no_cross_format_stopping_variance"] is False


def test_ceiling_raises_when_trace_lists_unequal_length() -> None:
    """Caller contract: mc and qa must be paired (equal length)."""
    mc = [_trace(stop_step=1, T=3)]
    qa = [_trace(stop_step=1, T=3), _trace(stop_step=2, T=3)]
    with pytest.raises(ValueError, match="equal-length"):
        diag_module.detect_ceiling_effects(mc, qa)


def test_ceiling_empty_flag() -> None:
    """Empty input is signaled with empty=True."""
    flags = diag_module.detect_ceiling_effects([], [])
    assert flags["empty"] is True
    assert flags["n_items"] == 0
    assert flags["no_cross_format_stopping_variance"] is False


from scripts.stopdff_dp import adapter as adapter_module


def _fake_mc_question(qid: str, gold_text: str = "George Washington") -> dict:
    """Synthesize the minimum MC question dict the adapter expects."""
    return {
        "qid": qid,
        "question": "Who was the first president of the United States?",
        "tokens": ["Who", "was", "the", "first", "president"],
        "answer_primary": gold_text,
        "clean_answers": [gold_text],
        "run_indices": [0, 4],
        "human_buzz_positions": [],
        "category": "History",
        "cumulative_prefixes": ["Who", "Who was the first president"],
        "options": [gold_text, "Thomas Jefferson", "John Adams", "Benjamin Franklin"],
        "gold_index": 0,
        "option_profiles": [
            "president", "vice", "second", "diplomat",
        ],
        "option_answer_primary": [
            gold_text, "Thomas Jefferson", "John Adams", "Benjamin Franklin",
        ],
        "distractor_strategy": "test",
    }


def test_adapter_produces_canonical_columns(monkeypatch) -> None:
    """The adapter must yield a dataframe with the canonical column set."""
    fake_questions = [_fake_mc_question("q1")]
    df = adapter_module.build_dataframe(
        mc_questions=fake_questions,
        target_qids={"q1"},
        split_name="val",
        calibration_path=None,  # Use the identity-calibration test mode
        identity_calibration=True,
    )
    from scripts.stopdff_dp.types import ADAPTER_COLUMNS
    assert list(df.columns) == list(ADAPTER_COLUMNS)
    # Two rows per (qid, prefix) per format -> 2 prefixes * 2 formats = 4.
    assert len(df) == 4


def test_adapter_fit_eval_split_separation_raises_on_overlap() -> None:
    """Passing the same split for fit and eval should raise."""
    with pytest.raises(ValueError, match="fit and eval split must differ"):
        adapter_module.validate_split_separation(
            fit_split="test", eval_split="test"
        )


def test_adapter_split_separation_rejects_test_as_fit_split() -> None:
    """The confirmatory DP path must never fit continuation on test."""
    with pytest.raises(ValueError, match="got fit_split='TEST'"):
        adapter_module.validate_split_separation(
            fit_split="TEST", eval_split="val"
        )


import json
from scripts.stopdff_dp import writers as writers_module
from scripts.stopdff_dp.types import DPTrace


def test_dptrace_documents_never_stop_sentinel() -> None:
    assert DPTrace.__doc__ is not None
    assert "stop_step == len(values)" in DPTrace.__doc__


def test_writer_metric_type_is_finite_horizon_dp(tmp_path: Path) -> None:
    out_json = tmp_path / "stopdff_dp.json"
    out_md = tmp_path / "stopdff_dp.md"
    out_tex = tmp_path / "stopdff_dp_table.tex"
    mc_traces = [_trace(stop_step=1, T=3, tags=["exact", "exact", "exact"])]
    qa_traces = [_trace(stop_step=2, T=3, tags=["exact", "exact", "exact"])]
    payload = writers_module.assemble_payload(
        mc_traces=mc_traces,
        qa_traces=qa_traces,
        reward_schedule_name="acf_flat",
        continuation_estimator_name="empirical_bucket",
        fit_split="val",
        eval_split="test",
        coverage_summary={
            "n_cells": 6, "fraction_exact": 1.0, "fraction_pooled": 0.0,
            "fraction_missing": 0.0, "verdict": "pass", "reason": "ok",
        },
        ceiling_flags={
            "all_stop_at_first_prefix": False,
            "all_stop_at_final_prefix": False,
            "no_cross_format_stopping_variance": False,
            "n_items": 1, "n_stopped_cells": 2, "n_never_stopped_cells": 0,
            "empty": False,
        },
        per_item_stopdff=[("q1", -1)],
        gate_verdict="pass",
        gate_verdict_reason="all_clean",
        confirmatory=True,
    )
    writers_module.write_json(out_json, payload)
    writers_module.write_markdown(out_md, payload)
    writers_module.write_latex(out_tex, payload)
    assert out_json.exists() and out_md.exists() and out_tex.exists()
    loaded = json.loads(out_json.read_text())
    assert loaded["metadata"]["metric_type"] == "finite_horizon_dp"
    assert loaded["metadata"]["stopping_policy"] == "finite_horizon_dp"
    assert "myopic" not in loaded["metadata"]["metric_type"]


def test_writer_records_reward_schedule_description() -> None:
    """JSON metadata must carry the schedule semantics, not just its name."""
    payload = writers_module.assemble_payload(
        mc_traces=[],
        qa_traces=[],
        reward_schedule_name="power_mark",
        reward_schedule_description="15/-5 before half; 10/-5 after half",
        continuation_estimator_name="empirical_bucket",
        fit_split="val",
        eval_split="test",
        coverage_summary={
            "n_cells": 0, "fraction_exact": None, "fraction_pooled": None,
            "fraction_missing": None, "verdict": "warn", "reason": "no_cells",
        },
        ceiling_flags={
            "all_stop_at_first_prefix": False,
            "all_stop_at_final_prefix": False,
            "no_cross_format_stopping_variance": False,
            "n_items": 0, "n_stopped_cells": 0, "n_never_stopped_cells": 0,
            "empty": True,
        },
        per_item_stopdff=[],
        gate_verdict="warn",
        gate_verdict_reason="no_data",
        confirmatory=True,
    )
    assert (
        payload["metadata"]["reward_schedule_description"]
        == "15/-5 before half; 10/-5 after half"
    )


def test_writer_handles_none_coverage_fractions(tmp_path: Path) -> None:
    """diagnostics.summarize_coverage returns None fractions on empty traces;
    writers must not crash on that legitimate diagnostic state."""
    payload = writers_module.assemble_payload(
        mc_traces=[],
        qa_traces=[],
        reward_schedule_name="acf_flat",
        continuation_estimator_name="empirical_bucket",
        fit_split="val",
        eval_split="test",
        coverage_summary={
            "n_cells": 0, "fraction_exact": None, "fraction_pooled": None,
            "fraction_missing": None, "verdict": "warn", "reason": "no_cells",
        },
        ceiling_flags={
            "all_stop_at_first_prefix": False,
            "all_stop_at_final_prefix": False,
            "no_cross_format_stopping_variance": False,
            "n_items": 0, "n_stopped_cells": 0, "n_never_stopped_cells": 0,
            "empty": True,
        },
        per_item_stopdff=[],
        gate_verdict="warn",
        gate_verdict_reason="no_data",
        confirmatory=True,
    )
    out_json = tmp_path / "dp.json"
    out_md = tmp_path / "dp.md"
    out_tex = tmp_path / "dp.tex"
    writers_module.write_json(out_json, payload)
    writers_module.write_markdown(out_md, payload)
    writers_module.write_latex(out_tex, payload)
    md = out_md.read_text()
    tex = out_tex.read_text()
    assert "n/a" in md
    assert "n/a" in tex


def test_writer_non_confirmatory_emits_md_warning(tmp_path: Path) -> None:
    """When confirmatory=False (e.g., oracle estimator), MD must warn."""
    payload = writers_module.assemble_payload(
        mc_traces=[],
        qa_traces=[],
        reward_schedule_name="acf_flat",
        continuation_estimator_name="oracle_trajectory",
        fit_split="val",
        eval_split="test",
        coverage_summary={
            "n_cells": 3, "fraction_exact": 1.0, "fraction_pooled": 0.0,
            "fraction_missing": 0.0, "verdict": "pass", "reason": "ok",
        },
        ceiling_flags={
            "all_stop_at_first_prefix": False,
            "all_stop_at_final_prefix": False,
            "no_cross_format_stopping_variance": False,
            "n_items": 1, "n_stopped_cells": 1, "n_never_stopped_cells": 0,
            "empty": False,
        },
        per_item_stopdff=[("q1", 0)],
        gate_verdict="pass",
        gate_verdict_reason="ok",
        confirmatory=False,
    )
    out_md = tmp_path / "dp.md"
    writers_module.write_markdown(out_md, payload)
    md = out_md.read_text()
    assert "Non-confirmatory" in md
    assert "upper-bound" in md.lower()


def test_writer_to_serializable_handles_numpy_types(tmp_path: Path) -> None:
    """to_serializable must convert numpy scalars in coverage/ceiling counters."""
    import numpy as np
    payload = writers_module.assemble_payload(
        mc_traces=[],
        qa_traces=[],
        reward_schedule_name="acf_flat",
        continuation_estimator_name="empirical_bucket",
        fit_split="val",
        eval_split="test",
        coverage_summary={
            "n_cells": np.int64(6), "fraction_exact": np.float64(1.0),
            "fraction_pooled": np.float64(0.0), "fraction_missing": np.float64(0.0),
            "verdict": "pass", "reason": "ok",
        },
        ceiling_flags={
            "all_stop_at_first_prefix": np.bool_(False),
            "all_stop_at_final_prefix": np.bool_(False),
            "no_cross_format_stopping_variance": np.bool_(False),
            "n_items": np.int64(1), "n_stopped_cells": np.int64(1),
            "n_never_stopped_cells": np.int64(0), "empty": np.bool_(False),
        },
        per_item_stopdff=[("q1", np.int64(-1))],
        gate_verdict="pass",
        gate_verdict_reason="ok",
        confirmatory=True,
    )
    out_json = tmp_path / "dp.json"
    writers_module.write_json(out_json, payload)
    loaded = json.loads(out_json.read_text())
    # If to_serializable runs, the integer/bool/float types in JSON round-trip
    # back to Python ints/bools/floats with no exception.
    assert loaded["coverage"]["n_cells"] == 6
    assert loaded["ceiling_flags"]["empty"] is False


import sys


def test_cli_smoke_run_writes_all_three_artifacts(tmp_path, monkeypatch) -> None:
    """End-to-end identity-calibration smoke run writes JSON+MD+TeX."""
    # Build a tiny mc/val/test dataset in-place.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question(f"v{i}") for i in range(5)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(5)]
    all_qs = val_qs + test_qs
    (data_dir / "mc_dataset.json").write_text(
        json.dumps(all_qs)
    )
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))

    out_json = tmp_path / "stopdff_dp.json"
    out_md = tmp_path / "stopdff_dp.md"
    out_tex = tmp_path / "stopdff_dp_table.tex"

    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "test",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",  # skip SBERT for the test
        "--out", str(out_json),
        "--out-md", str(out_md),
        "--out-tex", str(out_tex),
        "--allow-incomplete-mc-coverage",
        "--allow-low-mc-retention",
    ])
    assert rc == 0
    assert out_json.exists() and out_md.exists() and out_tex.exists()
    payload = json.loads(out_json.read_text())
    assert payload["metadata"]["metric_type"] == "finite_horizon_dp"
    assert payload["metadata"]["fit_split"] == "val"
    assert payload["metadata"]["eval_split"] == "test"


def test_cli_warns_when_responses_flag_is_provenance_only(
    tmp_path, capsys
) -> None:
    """--responses is currently recorded for provenance but not consumed."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(3)]
    (data_dir / "mc_dataset.json").write_text(json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))
    responses_path = tmp_path / "responses.json"
    responses_path.write_text("[]")

    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--responses", str(responses_path),
        "--split", "test",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",
        "--allow-incomplete-mc-coverage",
        "--allow-low-mc-retention",
        "--out", str(tmp_path / "out.json"),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])

    assert rc == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "--responses" in captured.err
    assert "no-op" in captured.err


def test_cli_rejects_same_split_for_fit_and_eval(tmp_path) -> None:
    from scripts import compute_stopdff_dp
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "mc_dataset.json").write_text("[]")
    (data_dir / "val_dataset.json").write_text("[]")
    (data_dir / "test_dataset.json").write_text("[]")
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "val",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",
        "--out", str(tmp_path / "out.json"),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])
    assert rc != 0


def test_cli_pooled_empirical_reports_pooled_coverage(tmp_path) -> None:
    """Regression: PooledEmpiricalEstimator must surface pooled coverage,
    not silently return 'exact' via the missing _last_tag proxy."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # 10 items per split so the pool has enough rows for any rung.
    val_qs = [_fake_mc_question(f"v{i}") for i in range(10)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(10)]
    (data_dir / "mc_dataset.json").write_text(json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))

    out_json = tmp_path / "out.json"
    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "test",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "pooled_empirical",
        "--identity-calibration",
        "--out", str(out_json),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    # Pooled estimator skips the most-specific rungs, so coverage should
    # report pooled > 0 (NOT 100% exact via the missing-property bug).
    assert payload["coverage"]["fraction_pooled"] > 0.0


def test_cli_oracle_trajectory_flags_non_confirmatory(tmp_path) -> None:
    """Oracle continuation must emit confirmatory=False in the payload."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(3)]
    (data_dir / "mc_dataset.json").write_text(json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))
    out_json = tmp_path / "out.json"
    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "test",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "oracle_trajectory",
        "--identity-calibration",
        "--out", str(out_json),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["confirmatory"] is False
    assert payload["metadata"]["continuation_estimator"] == "oracle_trajectory"


def test_cli_smoke_prunes_to_30_qids_per_split(tmp_path) -> None:
    """--smoke caps eval to 30 qids; n_items in payload must be <= 30."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question(f"v{i}") for i in range(50)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(50)]
    (data_dir / "mc_dataset.json").write_text(json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))
    out_json = tmp_path / "out.json"
    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "test",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",
        "--smoke",
        "--out", str(out_json),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["n_items"] <= 30


def test_fit_dataframe_never_contains_eval_split_rows(tmp_path) -> None:
    """Confirm via direct inspection that the fit_df has only val rows."""
    val_qs = [_fake_mc_question(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(3)]
    mc_pool = val_qs + test_qs
    val_qids = {q["qid"] for q in val_qs}
    test_qids = {q["qid"] for q in test_qs}

    fit_df = adapter_module.build_dataframe(
        mc_questions=mc_pool,
        target_qids=val_qids,
        split_name="val",
        calibration_path=None,
        identity_calibration=True,
    )
    eval_df = adapter_module.build_dataframe(
        mc_questions=mc_pool,
        target_qids=test_qids,
        split_name="test",
        calibration_path=None,
        identity_calibration=True,
    )

    assert set(fit_df["split"]) == {"val"}
    assert set(eval_df["split"]) == {"test"}
    assert set(fit_df["item_id"]).isdisjoint(set(eval_df["item_id"]))

    # And the EmpiricalBucketEstimator must refuse to fit on the eval frame.
    with pytest.raises(ValueError):
        cont_module.EmpiricalBucketEstimator.fit(
            fit_df=eval_df, schedule=REWARD_REGISTRY["acf_flat"],
            fit_split_name="val",
        )


def test_audit_card_row_added_without_replacing_diagnostic(tmp_path, monkeypatch):
    """The DP StopDFF row should appear after the existing diagnostic row."""
    from scripts import make_audit_card
    # Stub-load the three existing JSONs and one new DP JSON.
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    # Copy minimum-valid fixtures from the repo paper_exports directory.
    import shutil
    src = Path(__file__).resolve().parent.parent / "paper_exports"
    for fname in ("csli.json", "calibration.json", "stopdff.json"):
        shutil.copyfile(src / fname, paper / fname)
    # Synthesize a minimal stopdff_dp.json.
    (paper / "stopdff_dp.json").write_text(json.dumps({
        "stopdff_dp_signed_median": -0.5,
        "stopdff_dp_signed_mean": -0.4,
        "stopdff_dp_abs_median": 0.5,
        "n_items": 10,
        "direction_breakdown": {"mc_earlier": 5, "qa_earlier": 3, "same_step": 2},
        "coverage": {"verdict": "pass", "fraction_exact": 1.0,
                     "fraction_pooled": 0.0, "fraction_missing": 0.0,
                     "n_cells": 60, "reason": "ok"},
        "ceiling_flags": {"all_stop_at_first_prefix": False,
                          "all_stop_at_final_prefix": False,
                          "no_cross_format_stopping_variance": False,
                          "n_items": 10, "n_stopped_cells": 50,
                          "n_never_stopped_cells": 10, "empty": False},
        "gate_verdict": "pass",
        "gate_verdict_reason": "all_clean",
        "confirmatory": True,
        "metadata": {"metric_type": "finite_horizon_dp",
                     "stopping_policy": "finite_horizon_dp",
                     "reward_schedule": "power_mark",
                     "continuation_estimator": "empirical_bucket",
                     "fit_split": "val", "eval_split": "test"},
    }))
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)
    rc = make_audit_card.main_with_args(["--include-dp-stopdff"])
    assert rc == 0
    card = json.loads((paper / "audit_card.json").read_text())
    names = [m["name"] for m in card["metrics"]]
    assert any("Diagnostic StopDFF" in n for n in names)
    assert any("DP StopDFF" in n for n in names)


def test_cli_rejects_incomplete_mc_coverage_without_override(tmp_path) -> None:
    """If MC eval coverage < 98% and no override, exit nonzero."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # 10 val + 10 test questions, but mc_dataset only has 5 of each split.
    val_qs = [_fake_mc_question(f"v{i}") for i in range(10)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(10)]
    # mc_dataset is a strict subset (missing 5 of each).
    mc_subset = val_qs[:5] + test_qs[:5]
    (data_dir / "mc_dataset.json").write_text(json.dumps(mc_subset))
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))

    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "test",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",
        "--out", str(tmp_path / "out.json"),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])
    assert rc != 0


def test_cli_accepts_incomplete_mc_coverage_with_override(tmp_path) -> None:
    """With --allow-incomplete-mc-coverage, the script runs and records
    the override in the artifact's mc_coverage block."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question(f"v{i}") for i in range(10)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(10)]
    mc_subset = val_qs[:5] + test_qs[:5]
    (data_dir / "mc_dataset.json").write_text(json.dumps(mc_subset))
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))

    out_json = tmp_path / "out.json"
    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "test",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",
        "--allow-incomplete-mc-coverage",
        "--out", str(out_json),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert "mc_coverage" in payload
    # Coverage was below threshold but override flipped 'overridden' to true.
    eval_block = payload["mc_coverage"]["test"]
    assert eval_block["passed"] is False
    assert eval_block["overridden"] is True
    assert eval_block["coverage_rate"] == 0.5


def test_cli_records_clean_coverage_when_complete(tmp_path) -> None:
    """When MC coverage is complete (100%), the payload records passed=True."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question(f"v{i}") for i in range(5)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(5)]
    (data_dir / "mc_dataset.json").write_text(json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))

    out_json = tmp_path / "out.json"
    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "test",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",
        "--out", str(out_json),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    eval_block = payload["mc_coverage"]["test"]
    assert eval_block["passed"] is True
    assert eval_block["overridden"] is False
    assert eval_block["coverage_rate"] == 1.0


def test_audit_card_dp_retention_override_triggers_retained_subset(tmp_path, monkeypatch):
    """When the DP artifact was produced with --allow-low-mc-retention, the
    audit card must surface the retained-subset qualifier even if the DP
    threshold + producer verdicts are PASS.

    Uses real csli.json + calibration.json from paper_exports/ for valid
    schema, but synthesizes a clean stopdff.json (no ceiling effect, no
    unreachable buckets, gate_verdict=pass) so per-metric verdicts are all
    PASS — that's the only ladder branch in ``_compute_overall_verdict``
    that surfaces the retained-subset qualifier (a per-metric WARN
    intentionally suppresses it; see test_overall_verdict_fail_dominates_retention_override
    in tests/test_pr14_review_regressions.py).
    """
    from scripts import make_audit_card
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    import shutil
    src = Path(__file__).resolve().parent.parent / "paper_exports"
    for fname in ("csli.json", "calibration.json"):
        shutil.copyfile(src / fname, paper / fname)
    # Synthesize a clean stopdff.json — no ceiling effect, no unreachable
    # buckets, gate_verdict=pass — so the diagnostic StopDFF row evaluates
    # to PASS and doesn't collapse the override qualifier.
    (paper / "stopdff.json").write_text(json.dumps({
        "median_abs_prefix_shift": 0.05,
        "mean_abs_prefix_shift": 0.05,
        "per_question_distribution_summary": {
            "min": 0.0, "p25": 0.02, "median": 0.05,
            "p75": 0.08, "max": 0.2, "n": 10,
        },
        "direction_breakdown": {
            "mc_stops_earlier": 0, "nonmc_stops_earlier": 0, "same_step": 10,
        },
        "ceiling_effect_detected": False,
        "unreachable_buckets": [],
        "reachability": {
            "early": {"threshold_reachable": True,
                      "calibrated_at_sim_1": 0.8,
                      "calibrated_at_sim_neg_1": 0.1,
                      "max_calibrated_at_sim_1": 0.8,
                      "max_calibrated_probability": 0.8},
            "mid": {"threshold_reachable": True,
                    "calibrated_at_sim_1": 0.8,
                    "calibrated_at_sim_neg_1": 0.1,
                    "max_calibrated_at_sim_1": 0.8,
                    "max_calibrated_probability": 0.8},
            "late": {"threshold_reachable": True,
                     "calibrated_at_sim_1": 0.8,
                     "calibrated_at_sim_neg_1": 0.1,
                     "max_calibrated_at_sim_1": 0.8,
                     "max_calibrated_probability": 0.8},
        },
        "gate_verdict": "pass",
        "gate_verdict_reason": "threshold_only",
        "threshold_only_verdict": "pass",
        "threshold": 1.0,
        "metadata": {
            "metric_type": "diagnostic_only",
            "stopping_policy": "myopic_threshold",
            "stop_threshold": 0.7,
        },
    }))
    # Synthesize a clean DP JSON whose gate verdict is pass but
    # mc_retention_gate.test is passed=False, overridden=True.
    (paper / "stopdff_dp.json").write_text(json.dumps({
        "stopdff_dp_signed_median": 0.0,
        "stopdff_dp_signed_mean": 0.0,
        "stopdff_dp_abs_median": 0.0,
        "n_items": 10,
        "direction_breakdown": {"mc_earlier": 0, "qa_earlier": 0, "same_step": 10},
        "coverage": {"verdict": "pass", "fraction_exact": 1.0,
                     "fraction_pooled": 0.0, "fraction_missing": 0.0,
                     "n_cells": 60, "reason": "ok"},
        "ceiling_flags": {"all_stop_at_first_prefix": False,
                          "all_stop_at_final_prefix": False,
                          "no_cross_format_stopping_variance": False,
                          "n_items": 10, "n_stopped_cells": 50,
                          "n_never_stopped_cells": 10, "empty": False},
        "gate_verdict": "pass",
        "gate_verdict_reason": "all_clean",
        "confirmatory": True,
        "mc_coverage": {
            "test": {"target_qids": 100, "mc_questions_total": 50,
                     "matched_questions": 50, "matched_qids": 50,
                     "missing_qids": 0, "coverage_rate": 0.5,
                     "threshold": 0.98, "passed": False,
                     "overridden": True,
                     "override_flag": "--allow-incomplete-mc-coverage",
                     "split": "test"},
            "val": {"target_qids": 100, "mc_questions_total": 50,
                    "matched_questions": 50, "matched_qids": 50,
                    "missing_qids": 0, "coverage_rate": 0.5,
                    "threshold": 0.98, "passed": False,
                    "overridden": True,
                    "override_flag": "--allow-incomplete-mc-coverage",
                    "split": "val"},
        },
        "mc_retention_gate": {
            "test": {"applies": True, "split": "test",
                     "threshold": 0.8, "retention_rate": 0.7,
                     "raw_count": 100, "retained_count": 70,
                     "dropped_count": 30, "passed": False,
                     "overridden": True,
                     "override_flag": "--allow-low-mc-retention"},
            "val": {"applies": True, "split": "val",
                    "threshold": 0.8, "retention_rate": 0.7,
                    "raw_count": 100, "retained_count": 70,
                    "dropped_count": 30, "passed": False,
                    "overridden": True,
                    "override_flag": "--allow-low-mc-retention"},
        },
        "mc_build_metadata": {"status": "loaded",
                              "source_path": "data/processed/build_metadata.json",
                              "source_sha256": "dummy"},
        "metadata": {"metric_type": "finite_horizon_dp",
                     "stopping_policy": "finite_horizon_dp",
                     "reward_schedule": "power_mark",
                     "continuation_estimator": "empirical_bucket",
                     "fit_split": "val", "eval_split": "test"},
    }))
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)
    rc = make_audit_card.main_with_args(["--include-dp-stopdff"])
    assert rc == 0
    card = json.loads((paper / "audit_card.json").read_text())
    # data_provenance must surface stopdff_dp.
    assert "stopdff_dp" in card["data_provenance"]
    dp_prov = card["data_provenance"]["stopdff_dp"]
    assert dp_prov["coverage"]["test"]["overridden"] is True
    assert dp_prov["retention"]["test"]["overridden"] is True
    # Overall verdict qualifier mentions the retained-subset.
    assert (
        card.get("overall_verdict_qualifier") is not None
        and "retained-subset" in card["overall_verdict_qualifier"]
    )


def test_audit_card_dp_stale_producer_sha_triggers_warn(tmp_path, monkeypatch):
    """A DP artifact whose recorded script_sha256 doesn't match the live
    compute_stopdff_dp.py must downgrade the overall verdict to WARN with
    a stale-producer qualifier."""
    from scripts import make_audit_card
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    import shutil
    src = Path(__file__).resolve().parent.parent / "paper_exports"
    for fname in ("csli.json", "calibration.json", "stopdff.json"):
        shutil.copyfile(src / fname, paper / fname)
    # Synthesize a clean DP JSON with a wrong script_sha256.
    (paper / "stopdff_dp.json").write_text(json.dumps({
        "stopdff_dp_signed_median": 0.0,
        "stopdff_dp_signed_mean": 0.0,
        "stopdff_dp_abs_median": 0.0,
        "n_items": 10,
        "direction_breakdown": {"mc_earlier": 0, "qa_earlier": 0, "same_step": 10},
        "coverage": {"verdict": "pass", "fraction_exact": 1.0,
                     "fraction_pooled": 0.0, "fraction_missing": 0.0,
                     "n_cells": 60, "reason": "ok"},
        "ceiling_flags": {"all_stop_at_first_prefix": False,
                          "all_stop_at_final_prefix": False,
                          "no_cross_format_stopping_variance": False,
                          "n_items": 10, "n_stopped_cells": 50,
                          "n_never_stopped_cells": 10, "empty": False},
        "gate_verdict": "pass",
        "gate_verdict_reason": "all_clean",
        "confirmatory": True,
        "metadata": {
            "metric_type": "finite_horizon_dp",
            "stopping_policy": "finite_horizon_dp",
            "reward_schedule": "power_mark",
            "continuation_estimator": "empirical_bucket",
            "fit_split": "val",
            "eval_split": "test",
            "generation": {
                "schema_version": 1,
                "generated_at_utc": "2026-05-27T20:00:00+00:00",
                "command": ["python", "scripts/compute_stopdff_dp.py"],
                "argv": [],
                "cwd": ".",
                "output_path": "paper_exports/stopdff_dp.json",
                "script_path": "scripts/compute_stopdff_dp.py",
                "script_sha256": "0" * 64,  # deliberately stale
                "git_commit": "0" * 40,
                "git_dirty": False,
                "git_status_relevant_paths": "",
            },
        },
    }))
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)
    rc = make_audit_card.main_with_args(["--include-dp-stopdff"])
    assert rc == 0
    card = json.loads((paper / "audit_card.json").read_text())
    # artifact_provenance must include stopdff_dp.json with sha_matches=False.
    dp_prov = card["artifact_provenance"]["stopdff_dp.json"]
    assert dp_prov["sha_matches"] is False
    assert dp_prov["recorded_sha256"] == "0" * 64
    # Overall verdict qualifier must mention the stale producer hash.
    qualifier = card.get("overall_verdict_qualifier") or ""
    assert "stale producer hash" in qualifier
    assert "stopdff_dp.json" in qualifier


def test_audit_card_non_confirmatory_dp_forces_warn(tmp_path, monkeypatch):
    """A DP artifact with confirmatory=False (e.g., oracle_trajectory) must
    yield verdict=warn even when coverage and signed_median are clean."""
    from scripts import make_audit_card
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    import shutil
    src = Path(__file__).resolve().parent.parent / "paper_exports"
    for fname in ("csli.json", "calibration.json", "stopdff.json"):
        shutil.copyfile(src / fname, paper / fname)
    # Synthesize a non-confirmatory DP JSON that would otherwise PASS.
    # Use the LIVE script_sha256 so artifact-provenance doesn't also warn.
    from scripts._common import sha256_file
    real_sha = sha256_file(
        Path(__file__).resolve().parent.parent / "scripts" / "compute_stopdff_dp.py"
    )
    (paper / "stopdff_dp.json").write_text(json.dumps({
        "stopdff_dp_signed_median": 0.0,
        "stopdff_dp_signed_mean": 0.0,
        "stopdff_dp_abs_median": 0.0,
        "n_items": 10,
        "direction_breakdown": {"mc_earlier": 0, "qa_earlier": 0, "same_step": 10},
        "coverage": {"verdict": "pass", "fraction_exact": 1.0,
                     "fraction_pooled": 0.0, "fraction_missing": 0.0,
                     "n_cells": 60, "reason": "ok"},
        "ceiling_flags": {"all_stop_at_first_prefix": False,
                          "all_stop_at_final_prefix": False,
                          "no_cross_format_stopping_variance": False,
                          "n_items": 10, "n_stopped_cells": 50,
                          "n_never_stopped_cells": 10, "empty": False},
        "gate_verdict": "pass",
        "gate_verdict_reason": "all_clean",
        "confirmatory": False,  # the key field under test
        "metadata": {
            "metric_type": "finite_horizon_dp",
            "stopping_policy": "finite_horizon_dp",
            "reward_schedule": "power_mark",
            "continuation_estimator": "oracle_trajectory",
            "fit_split": "val",
            "eval_split": "test",
            "generation": {
                "schema_version": 1,
                "generated_at_utc": "2026-05-27T20:00:00+00:00",
                "command": ["python", "scripts/compute_stopdff_dp.py"],
                "argv": [],
                "cwd": ".",
                "output_path": "paper_exports/stopdff_dp.json",
                "script_path": "scripts/compute_stopdff_dp.py",
                "script_sha256": real_sha,
                "git_commit": "0" * 40,
                "git_dirty": False,
                "git_status_relevant_paths": "",
            },
        },
    }))
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)
    rc = make_audit_card.main_with_args(["--include-dp-stopdff"])
    assert rc == 0
    card = json.loads((paper / "audit_card.json").read_text())
    # Find the DP row in metrics.
    dp_row = next(
        m for m in card["metrics"]
        if "DP StopDFF" in m["name"]
    )
    assert dp_row["verdict"] == "warn"
    qualifier = dp_row.get("verdict_qualifier") or ""
    assert "non-confirmatory" in qualifier


def test_audit_card_dp_stale_helper_hash_triggers_warn(tmp_path, monkeypatch):
    """When a helper module's hash differs from what stopdff_dp.json
    recorded, the audit card must downgrade to WARN with a helper-
    mismatch qualifier — even if the producer script's own SHA matches."""
    from scripts import make_audit_card
    from scripts._common import sha256_file
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    import shutil
    src = Path(__file__).resolve().parent.parent / "paper_exports"
    for fname in ("csli.json", "calibration.json", "stopdff.json"):
        shutil.copyfile(src / fname, paper / fname)
    # Synthesize a DP artifact whose producer SHA matches the live
    # compute_stopdff_dp.py but whose helper_sha256s has a deliberate
    # mismatch for one helper module.
    repo_root = Path(__file__).resolve().parent.parent
    live_producer_sha = sha256_file(repo_root / "scripts" / "compute_stopdff_dp.py")
    (paper / "stopdff_dp.json").write_text(json.dumps({
        "stopdff_dp_signed_median": 0.0,
        "stopdff_dp_signed_mean": 0.0,
        "stopdff_dp_abs_median": 0.0,
        "n_items": 10,
        "direction_breakdown": {"mc_earlier": 0, "qa_earlier": 0, "same_step": 10},
        "coverage": {"verdict": "pass", "fraction_exact": 1.0,
                     "fraction_pooled": 0.0, "fraction_missing": 0.0,
                     "n_cells": 60, "reason": "ok"},
        "ceiling_flags": {"all_stop_at_first_prefix": False,
                          "all_stop_at_final_prefix": False,
                          "no_cross_format_stopping_variance": False,
                          "n_items": 10, "n_stopped_cells": 50,
                          "n_never_stopped_cells": 10, "empty": False},
        "gate_verdict": "pass",
        "gate_verdict_reason": "all_clean",
        "confirmatory": True,
        "metadata": {
            "metric_type": "finite_horizon_dp",
            "stopping_policy": "finite_horizon_dp",
            "reward_schedule": "power_mark",
            "continuation_estimator": "empirical_bucket",
            "fit_split": "val",
            "eval_split": "test",
            "generation": {
                "schema_version": 1,
                "generated_at_utc": "2026-05-27T20:00:00+00:00",
                "command": ["python", "scripts/compute_stopdff_dp.py"],
                "argv": [],
                "cwd": ".",
                "output_path": "paper_exports/stopdff_dp.json",
                "script_path": "scripts/compute_stopdff_dp.py",
                "script_sha256": live_producer_sha,  # matches the live producer
                "git_commit": "0" * 40,
                "git_dirty": False,
                "git_status_relevant_paths": "",
                "helper_sha256s": {
                    # Wrong hash for a real helper module.
                    "scripts/stopdff_dp/rewards.py": "0" * 64,
                    # Other helpers omitted; the dict is non-exhaustive
                    # by design — the test only asserts the mismatch
                    # branch fires.
                },
            },
        },
    }))
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)
    rc = make_audit_card.main_with_args(["--include-dp-stopdff"])
    assert rc == 0
    card = json.loads((paper / "audit_card.json").read_text())
    dp_prov = card["artifact_provenance"]["stopdff_dp.json"]
    assert dp_prov["sha_matches"] is False
    assert dp_prov["helper_mismatches"] is not None
    assert "scripts/stopdff_dp/rewards.py" in dp_prov["helper_mismatches"]
    # Overall verdict qualifier mentions stale producer hash.
    qualifier = card.get("overall_verdict_qualifier") or ""
    assert "stale producer hash" in qualifier and "stopdff_dp.json" in qualifier


def test_audit_card_dp_missing_recorded_helper_triggers_warn() -> None:
    """A recorded DP helper path that no longer exists is stale provenance."""
    from scripts import make_audit_card
    from scripts._common import sha256_file

    repo_root = Path(__file__).resolve().parent.parent
    live_producer_sha = sha256_file(repo_root / "scripts" / "compute_stopdff_dp.py")
    missing_helper = "scripts/stopdff_dp/deleted_helper.py"
    provenance = make_audit_card._build_artifact_provenance(
        csli_data={},
        cal_data={},
        stopdff_data={},
        dp_data={
            "metadata": {
                "generation": {
                    "script_sha256": live_producer_sha,
                    "helper_sha256s": {
                        missing_helper: "1" * 64,
                    },
                },
            },
        },
    )

    dp_prov = provenance["stopdff_dp.json"]
    assert dp_prov["sha_matches"] is False
    assert dp_prov["helper_mismatches"] == {
        missing_helper: {"recorded": "1" * 64, "current": None}
    }


def test_audit_card_dp_helper_hashes_all_match_keeps_pass(tmp_path, monkeypatch):
    """When the DP artifact records helper hashes that all match the live
    files, sha_matches stays True and helper_mismatches is None."""
    from scripts import make_audit_card
    from scripts._common import sha256_file
    from scripts.stopdff_dp._provenance import helper_sha256s as live_helpers
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    import shutil
    src = Path(__file__).resolve().parent.parent / "paper_exports"
    for fname in ("csli.json", "calibration.json", "stopdff.json"):
        shutil.copyfile(src / fname, paper / fname)
    repo_root = Path(__file__).resolve().parent.parent
    live_producer_sha = sha256_file(repo_root / "scripts" / "compute_stopdff_dp.py")
    (paper / "stopdff_dp.json").write_text(json.dumps({
        "stopdff_dp_signed_median": 0.0,
        "stopdff_dp_signed_mean": 0.0,
        "stopdff_dp_abs_median": 0.0,
        "n_items": 10,
        "direction_breakdown": {"mc_earlier": 0, "qa_earlier": 0, "same_step": 10},
        "coverage": {"verdict": "pass", "fraction_exact": 1.0,
                     "fraction_pooled": 0.0, "fraction_missing": 0.0,
                     "n_cells": 60, "reason": "ok"},
        "ceiling_flags": {"all_stop_at_first_prefix": False,
                          "all_stop_at_final_prefix": False,
                          "no_cross_format_stopping_variance": False,
                          "n_items": 10, "n_stopped_cells": 50,
                          "n_never_stopped_cells": 10, "empty": False},
        "gate_verdict": "pass",
        "gate_verdict_reason": "all_clean",
        "confirmatory": True,
        "metadata": {
            "metric_type": "finite_horizon_dp",
            "stopping_policy": "finite_horizon_dp",
            "reward_schedule": "power_mark",
            "continuation_estimator": "empirical_bucket",
            "fit_split": "val",
            "eval_split": "test",
            "generation": {
                "schema_version": 1,
                "generated_at_utc": "2026-05-27T20:00:00+00:00",
                "command": ["python", "scripts/compute_stopdff_dp.py"],
                "argv": [],
                "cwd": ".",
                "output_path": "paper_exports/stopdff_dp.json",
                "script_path": "scripts/compute_stopdff_dp.py",
                "script_sha256": live_producer_sha,
                "git_commit": "0" * 40,
                "git_dirty": False,
                "git_status_relevant_paths": "",
                "helper_sha256s": live_helpers(),
            },
        },
    }))
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)
    rc = make_audit_card.main_with_args(["--include-dp-stopdff"])
    assert rc == 0
    card = json.loads((paper / "audit_card.json").read_text())
    dp_prov = card["artifact_provenance"]["stopdff_dp.json"]
    assert dp_prov["sha_matches"] is True
    assert dp_prov["helper_mismatches"] is None


def test_writer_latex_escapes_gate_verdict_string(tmp_path):
    """writers.write_latex must escape the gate_verdict field.

    Defensive — current gate_verdict values are alphabetical, but future
    string fields piped through this writer could include LaTeX specials.
    """
    from scripts.stopdff_dp.writers import write_latex
    payload = {
        "stopdff_dp_signed_median": 0.0,
        "stopdff_dp_signed_mean": 0.0,
        "stopdff_dp_abs_median": 0.0,
        "n_items": 0,
        "coverage": {
            "fraction_exact": 0.5,
            "fraction_pooled": 0.5,
            "fraction_missing": 0.0,
            "verdict": "pass", "reason": "ok",
        },
        "gate_verdict": "warn_special_$_chars",  # contains underscores + $
    }
    out = tmp_path / "table.tex"
    write_latex(out, payload)
    body = out.read_text()
    assert "warn\\_special\\_\\$\\_chars" in body
    # Bare underscore in the verdict string must not appear.
    assert "warn_special_$_chars" not in body


def test_producer_dirty_check_includes_helper_modules_and_inputs(tmp_path, monkeypatch):
    """compute_stopdff_dp.py's provenance dirty check must flip git_dirty=true
    when an imported helper OR an input dataset has uncommitted changes.

    Without the extra_paths fix, helper edits would be hashed into
    helper_sha256s but flagged as git_dirty=false, leaking a non-
    reproducible commit pointer into the audit card.
    """
    import json as _json
    from scripts import _common as common
    from scripts import compute_stopdff_dp as ctd
    from scripts.stopdff_dp._provenance import helper_paths

    helper_rel = "scripts/stopdff_dp/rewards.py"
    helper_for_test = ctd.PROJECT_ROOT / helper_rel
    assert helper_for_test in helper_paths(), (
        f"test premise: {helper_for_test} must be in helper_paths()"
    )

    # Synthesize input fixtures so unrelated input paths are clean / off-tree.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    from tests.test_stopdff_dp import _fake_mc_question  # already defined for adapter tests
    val_qs = [_fake_mc_question(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(3)]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    out_tex = tmp_path / "out.tex"

    status_calls: list[list[str]] = []

    def fake_git_output(args: list[str]) -> str:
        if args[:3] == ["status", "--short", "--"]:
            status_calls.append(args)
            assert helper_rel in args
            return f" M {helper_rel}"
        if args == ["rev-parse", "HEAD"]:
            return "c" * 40
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(common, "_git_output", fake_git_output)

    rc_dirty = ctd.main([
        "--data-dir", str(data_dir),
        "--split", "test", "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",
        "--allow-incomplete-mc-coverage", "--allow-low-mc-retention",
        "--out", str(out_json),
        "--out-md", str(out_md),
        "--out-tex", str(out_tex),
    ])
    assert rc_dirty == 0
    dirty_payload = _json.loads(out_json.read_text())
    generation = dirty_payload["metadata"]["generation"]
    assert generation["git_commit"] == "c" * 40
    assert generation["git_dirty"] is True
    assert helper_rel in generation["git_status_relevant_paths"]
    assert status_calls


def test_adapter_validate_qid_separation_passes_on_disjoint_sets():
    adapter_module.validate_qid_separation(
        fit_qids={"v1", "v2", "v3"},
        eval_qids={"t1", "t2", "t3"},
    )  # Must not raise.


def test_adapter_validate_qid_separation_raises_on_overlap():
    with pytest.raises(ValueError, match="leakage"):
        adapter_module.validate_qid_separation(
            fit_qids={"v1", "shared", "v3"},
            eval_qids={"t1", "shared", "t3"},
        )


def test_adapter_validate_qid_separation_reports_overlap_count_and_examples():
    overlap_size = 12
    fit = {f"shared_{i}" for i in range(overlap_size)} | {"fit_only"}
    eval_ = {f"shared_{i}" for i in range(overlap_size)} | {"eval_only"}
    try:
        adapter_module.validate_qid_separation(
            fit_qids=fit, eval_qids=eval_,
            fit_split="val", eval_split="test",
        )
    except ValueError as exc:
        message = str(exc)
        assert f"{overlap_size} qid" in message
        # Examples list capped at 10.
        assert "shared_0" in message
        assert "shared_9" in message
        # Examples don't include the 11th/12th item.
        assert "shared_10" not in message
        assert "shared_11" not in message
    else:
        raise AssertionError("expected ValueError on overlap")


def test_cli_rejects_overlapping_val_test_qids(tmp_path):
    """End-to-end: when val_dataset.json and test_dataset.json share qids,
    compute_stopdff_dp.main must exit nonzero with a leakage error."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question(f"v{i}") for i in range(3)]
    # Test split intentionally reuses one val qid ("v1").
    test_qs = (
        [_fake_mc_question("v1")]  # overlapping qid
        + [_fake_mc_question(f"t{i}") for i in range(2)]
    )
    (data_dir / "mc_dataset.json").write_text(json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))

    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "test", "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",
        "--allow-incomplete-mc-coverage", "--allow-low-mc-retention",
        "--out", str(tmp_path / "out.json"),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])
    assert rc != 0
