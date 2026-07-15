"""Unit tests for the stopdff_v5 data-independent core (fixtures/synthetic only)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import (  # noqa: E402
    bootstrap,
    calibrators,
    continuation,
    fvi,
    identity,
    policy,
    profile,
    verdicts,
)
from scripts.stopdff_v5.rewards import answer_utility, get_schedule  # noqa: E402


# --- identity ---------------------------------------------------------------------


def test_canonical_bytes_sorts_keys_and_is_stable():
    a = {"b": 1, "a": "x"}
    b = {"a": "x", "b": 1}
    assert identity.canonical_bytes(a) == identity.canonical_bytes(b)
    assert identity.canonical_bytes(a) == b'{"a":"x","b":1}'


def test_identity_rejects_float():
    with pytest.raises(identity.IdentityError):
        identity.canonical_bytes({"x": 0.5})


def test_identity_rejects_duplicate_keys_on_load():
    with pytest.raises(identity.IdentityError):
        identity.loads_no_duplicate_keys('{"a":1,"a":2}')


def test_manifest_roundtrip_and_verify():
    ident = {"kind": "demo", "value": "0.05"}
    man = identity.build_manifest(ident, path="/tmp/x", ts="now")
    assert man["id"] == identity.compute_id(ident)
    assert identity.verify_manifest_id(man) == man["id"]
    man["id"] = "deadbeef"
    with pytest.raises(identity.IdentityError):
        identity.verify_manifest_id(man)


# --- rewards ----------------------------------------------------------------------


def test_reward_schedules_values():
    acf = get_schedule("acf_flat")
    assert (acf.correct_early, acf.correct_late, acf.wrong, acf.split, acf.wait_cost) == (
        10.0, 10.0, -5.0, 1.0, 0.0,
    )
    pm = get_schedule("power_mark")
    assert pm.r_correct(0.4) == 15.0 and pm.r_correct(0.6) == 10.0
    wcs = get_schedule("wait_cost_small")
    assert wcs.wait_cost == 0.05
    sw = get_schedule("strict_wrong")
    assert sw.wrong == -10.0


def test_answer_utility_formula():
    s = get_schedule("power_mark")
    assert answer_utility(1.0, 0.1, s) == 15.0
    assert answer_utility(0.0, 0.1, s) == -5.0


# --- policy (3-action) ------------------------------------------------------------


def _const_cont(value):
    return lambda t, p, prefix_fraction: value


def test_terminal_abstain_when_answer_nonpositive():
    s = get_schedule("acf_flat")
    tr = policy.solve_trajectory(
        p_trajectory=[1.0 / 3.0], prefix_fractions=[1.0], schedule=s, continuation_fn=_const_cont(0.0)
    )
    # A_0 = 10*(1/3) - 5*(2/3) = 0.0 -> not strictly > abstain -> abstain
    assert tr.never_buzz is True and tr.stop_index == 1


def test_terminal_answer_when_positive():
    s = get_schedule("acf_flat")
    tr = policy.solve_trajectory(
        p_trajectory=[0.9], prefix_fractions=[1.0], schedule=s, continuation_fn=_const_cont(0.0)
    )
    assert tr.stop_index == 0 and tr.never_buzz is False


def test_nonterminal_wait_beats_answer_on_high_continuation():
    s = get_schedule("acf_flat")
    tr = policy.solve_trajectory(
        p_trajectory=[0.9, 0.9], prefix_fractions=[0.5, 1.0], schedule=s,
        continuation_fn=_const_cont(100.0),
    )
    assert tr.stop_index == 1  # waits at t=0, answers at terminal


def test_nonterminal_tie_prefers_wait():
    s = get_schedule("acf_flat")
    # A_0 = 8.5; set continuation so wait_value == A_0 exactly -> tie -> wait.
    tr = policy.solve_trajectory(
        p_trajectory=[0.9, 0.9], prefix_fractions=[0.5, 1.0], schedule=s,
        continuation_fn=_const_cont(8.5),
    )
    assert tr.stop_index == 1


def test_nonterminal_abstain_beats_answer_when_answer_nonpositive():
    s = get_schedule("wait_cost_small")  # c_wait = 0.05
    # p chosen so A_0 in (-0.05, 0): 20p - 5 = -0.01 -> p = 0.2495
    tr = policy.solve_trajectory(
        p_trajectory=[0.2495, 0.9], prefix_fractions=[0.1, 1.0], schedule=s,
        continuation_fn=_const_cont(0.0),
    )
    # wait_value at t0 = -0.05; A_0 ~ -0.01 > wait but <= abstain(0) -> not answer
    assert tr.stop_index == 1


def test_signed_index_shift():
    s = get_schedule("acf_flat")
    mc = policy.solve_trajectory(p_trajectory=[0.9], prefix_fractions=[1.0], schedule=s, continuation_fn=_const_cont(0.0))
    qa = policy.solve_trajectory(p_trajectory=[1.0 / 3.0], prefix_fractions=[1.0], schedule=s, continuation_fn=_const_cont(0.0))
    assert policy.signed_index_shift(mc, qa) == 0 - 1


# --- profile ----------------------------------------------------------------------


def test_full_grid_has_96_unique_cells():
    grid = profile.full_grid()
    assert len(grid) == 96
    keys = {profile.cell_key_str(c) for c in grid}
    assert len(keys) == 96


def test_smoke_cells_two_explicit():
    cells = profile.smoke_cells()
    assert len(cells) == 2
    assert profile.cell_key_str(cells[0]) != profile.cell_key_str(cells[1])


def test_representative_24():
    assert len(profile.representative_24()) == 24


def test_legacy_alias_normalization_no_duplicate():
    c = {
        "reward_schedule": "acf_flat",
        "continuation": "empirical_bucket",
        "calibrator": "isotonic",
        "prefix_bucketing": "exact_prefix",
        "category_pooling": "per_subject",
    }
    norm = profile.normalize_cell(c)
    assert norm["category_pooling"] == "per_category"


def test_unknown_axis_value_rejected():
    with pytest.raises(ValueError):
        profile.normalize_cell({
            "reward_schedule": "nope", "continuation": "empirical_bucket",
            "calibrator": "isotonic", "prefix_bucketing": "exact_prefix",
            "category_pooling": "per_category",
        })


# --- calibrators ------------------------------------------------------------------


def _mc_rows(phase_frac, n, split_correct):
    rows = []
    for i in range(n):
        rows.append({
            "raw_similarity": 0.1 + 0.8 * (i / max(1, n - 1)),
            "correct": 1 if i >= split_correct else 0,
            "prefix_fraction": phase_frac,
        })
    return rows


def test_platt_logistic_and_constant():
    cal = {
        "per_bucket": {
            "early": {"platt_coef": 4.0, "platt_intercept": -2.0},
            "mid": {"platt_coef": 4.0, "platt_intercept": -2.0},
            "late": {"platt_coef": None, "platt_intercept": None,
                     "platt_model_type": "constant", "platt_constant_probability": 0.7},
        }
    }
    c = calibrators.fit_platt(cal)
    assert 0.0 <= c.apply(0.5, 0.1) <= 1.0
    assert abs(c.apply(0.5, 0.9) - 0.7) < 1e-9  # late constant


def test_similarity_temperature_prereq_error():
    with pytest.raises(calibrators.CalibratorFitError):
        calibrators.fit_similarity_temperature(_mc_rows(0.1, 5, 2))  # <10 rows


def test_similarity_temperature_fits_all_phases():
    rows = _mc_rows(0.1, 12, 6) + _mc_rows(0.5, 12, 6) + _mc_rows(0.9, 12, 6)
    c = calibrators.fit_similarity_temperature(rows)
    assert set(c.phase_params) == {"early", "mid", "late"}
    assert 0.0 <= c.apply(0.5, 0.5) <= 1.0


def test_isotonic_monotone():
    rows = _mc_rows(0.1, 20, 10) + _mc_rows(0.5, 20, 10) + _mc_rows(0.9, 20, 10)
    c = calibrators.fit_isotonic(rows)
    lo = c.apply(0.15, 0.1)
    hi = c.apply(0.85, 0.1)
    assert hi >= lo - 1e-9


# --- continuation -----------------------------------------------------------------


def test_bins():
    assert continuation.p_bin_of(0.0) == 0
    assert continuation.p_bin_of(0.19) == 0
    assert continuation.p_bin_of(0.2) == 1
    assert continuation.p_bin_of(1.0) == 4
    # binary entropy at p=0.5 is 1.0 -> falls in the last bin [0.9, 1.000000000001)
    assert continuation.entropy_bin_of(0.5) == 2
    assert continuation.entropy_bin_of(0.99) == 0  # near-zero entropy
    assert continuation.entropy_bin_of(0.8) == 1  # H(0.8) ~ 0.72 -> [0.5, 0.9)


def test_prefix_key_modes():
    assert continuation.prefix_key("exact_prefix", prefix_idx=3, prefix_fraction=0.9) == 3
    assert continuation.prefix_key("early_mid_late", prefix_idx=3, prefix_fraction=0.1) == "early"
    assert continuation.prefix_key("early_mid_late", prefix_idx=3, prefix_fraction=0.5) == "mid"
    assert continuation.prefix_key("early_mid_late", prefix_idx=3, prefix_fraction=0.9) == "late"


def test_continuation_coverage_tags():
    est = continuation.ContinuationEstimator("empirical_bucket", "pooled_category")
    obs = continuation.make_observation(
        prefix_bucketing="early_mid_late", prefix_idx=0, prefix_fraction=0.1,
        fmt="MC", category="hist", p_calibrated=0.5,
    )
    # No counts -> missing
    assert est.coverage_tag(obs) == "missing"
    # Give the top rung enough count -> primary
    for key in est.rung_keys(obs):
        est.bucket_counts[key] = 3
        est.bucket_means[key] = 1.0
    assert est.coverage_tag(obs) == "primary"
    assert est.estimate(obs) == 1.0


# --- fvi --------------------------------------------------------------------------


def _fit_trajectory(item_id, fmt, p_traj, fracs, prefix_bucketing="early_mid_late", category="c"):
    obs = [
        continuation.make_observation(
            prefix_bucketing=prefix_bucketing, prefix_idx=t, prefix_fraction=fracs[t],
            fmt=fmt, category=category, p_calibrated=p_traj[t],
        )
        for t in range(len(p_traj) - 1)  # nonterminal observations
    ]
    return fvi.FitTrajectory(
        item_id=item_id, fmt=fmt, category=category,
        p_trajectory=p_traj, prefix_fractions=fracs, obs_at_t=obs,
    )


def test_fvi_converges_and_bounds():
    s = get_schedule("acf_flat")
    est = continuation.ContinuationEstimator("empirical_bucket", "pooled_category")
    trajs = [
        _fit_trajectory("i1", "MC", [0.6, 0.7, 0.9], [0.2, 0.5, 1.0]),
        _fit_trajectory("i2", "MC", [0.5, 0.8, 0.95], [0.2, 0.5, 1.0]),
        _fit_trajectory("i3", "MC", [0.55, 0.75, 0.9], [0.2, 0.5, 1.0]),
        _fit_trajectory("i4", "MC", [0.4, 0.6, 0.85], [0.2, 0.5, 1.0]),
    ]
    continuation.build_counts(est, [t.obs_at_t for t in trajs])
    res = fvi.run_fvi(est, trajs, s, tolerance=1e-10, max_iterations=200, tolerance_label="1e-10")
    assert res.status == "converged"
    for v in est.bucket_means.values():
        assert -1e-12 <= v <= s.max_correct_reward + 1e-12


# --- bootstrap --------------------------------------------------------------------


def test_bootstrap_plan_deterministic():
    ids = [f"q{i}" for i in range(20)]
    p1 = bootstrap.build_bootstrap_plan(ids, replicates=100, seed=1)
    p2 = bootstrap.build_bootstrap_plan(ids, replicates=100, seed=1)
    assert np.array_equal(p1.resample_indices, p2.resample_indices)
    assert p1.resample_index_sha256 == p2.resample_index_sha256
    assert p1.item_id_list_sha256 == p2.item_id_list_sha256
    assert p1.resample_indices.shape == (100, 20)


def test_cell_stats_and_family():
    ids = [f"q{i}" for i in range(30)]
    plan = bootstrap.build_bootstrap_plan(ids, replicates=200, seed=1)
    shifts = {q: (i % 3 - 1) for i, q in enumerate(ids)}  # -1,0,1 pattern
    stats = bootstrap.cell_bootstrap_stats(shifts, plan)
    lo, hi = stats["abs_median_ci"]
    assert lo <= hi and math.isfinite(lo) and math.isfinite(hi)
    fam = bootstrap.family_statistic(
        {"cellA": stats["abs_median_replicates"]}, {"cellA": stats["abs_median_point"]}
    )
    assert fam["ci"][0] <= fam["ci"][1]


# --- verdicts ---------------------------------------------------------------------


def test_cell_verdict_logic():
    assert verdicts.cell_verdict(abs_median_ci=[1.2, 1.5], coverage_is_clean=True, ceiling_any=False, mc_gate_overridden=False) == "FAIL"
    assert verdicts.cell_verdict(abs_median_ci=[0.0, 1.0], coverage_is_clean=True, ceiling_any=False, mc_gate_overridden=False) == "PASS"
    assert verdicts.cell_verdict(abs_median_ci=[0.0, 1.0], coverage_is_clean=False, ceiling_any=False, mc_gate_overridden=False) == "WARN"
    assert verdicts.cell_verdict(abs_median_ci=[0.5, 2.0], coverage_is_clean=True, ceiling_any=False, mc_gate_overridden=False) == "WARN"


def test_ceiling_flags():
    T = verdicts.TraceStop
    flags = verdicts.ceiling_flags(
        [T(0, False, 3), T(0, False, 3)], [T(0, False, 3)], [0, 0, 0]
    )
    assert flags["all_answer_first_prefix"] is True
    assert flags["all_paired_index_shifts_zero"] is True


def test_family_verdict_and_release():
    assert verdicts.family_verdict(family_ci=[1.1, 1.4], all_cells_pass=True, mc_override_active=False) == "FAIL"
    assert verdicts.family_verdict(family_ci=[0.0, 1.0], all_cells_pass=True, mc_override_active=False) == "PASS"
    assert verdicts.family_verdict(family_ci=[0.0, 1.0], all_cells_pass=False, mc_override_active=False) == "WARN"
    keys = {"a", "b"}
    rc = verdicts.release_validity(
        expected_cell_keys=keys, present_cell_keys=["a", "b"], completed_keys=keys,
        failed_keys=set(), skipped_keys=set(), all_calibrators_fitted=True, all_fvi_converged=True,
        manifests_valid=True, cache_matches_aggregate=True, bootstrap_valid=True, family_valid=True,
        backend_manifest_valid=True, attempt_history_valid=True,
    )
    assert rc.valid is True
    rc2 = verdicts.release_validity(
        expected_cell_keys=keys, present_cell_keys=["a"], completed_keys={"a"},
        failed_keys=set(), skipped_keys=set(), all_calibrators_fitted=True, all_fvi_converged=True,
        manifests_valid=True, cache_matches_aggregate=True, bootstrap_valid=True, family_valid=True,
        backend_manifest_valid=True, attempt_history_valid=True,
    )
    assert rc2.valid is False
