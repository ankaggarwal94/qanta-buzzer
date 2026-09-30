"""Tests for the TAE camera-ready diagnostics artifact builder.

Unit tests cover the pure functions with synthetic fixtures (phase bucketing
including the exact 0.33/0.66 edge semantics, the run's 10-bin ECE including
last-bin right-inclusivity, calibrator application, never-buzz bounds, the
myopic p* formula, and per-item cross-cell dispersion).  The integration test
runs the mandatory regression gate against the real frozen export and SKIPS
cleanly when the export is absent (CI has no export).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import tae_camera_ready_diagnostics as tcd  # noqa: E402

EXPORT_ROOT = tcd.DEFAULT_EXPORT_ROOT
HAS_EXPORT = EXPORT_ROOT.is_dir()


# ---------------------------------------------------------------------------
# Phase bucketing
# ---------------------------------------------------------------------------

def test_phase_of_edges():
    assert tcd.phase_of(0.0) == "early"
    assert tcd.phase_of(0.3299999) == "early"
    assert tcd.phase_of(0.33) == "mid"  # exact boundary goes UP
    assert tcd.phase_of(0.5) == "mid"
    assert tcd.phase_of(0.6599999) == "mid"
    assert tcd.phase_of(0.66) == "late"  # exact boundary goes UP
    assert tcd.phase_of(1.0) == "late"


def test_phase_of_matches_run_convention():
    """Contract test against the run's own phase_of (calibrators.py)."""
    from scripts.stopdff_v5 import calibrators

    grid = [0.0, 0.1, 0.329999, 0.33, 0.330001, 0.5, 0.659999, 0.66, 0.660001, 0.99, 1.0]
    for value in grid:
        assert tcd.phase_of(value) == calibrators.phase_of(value), value


def test_phase_mask_matches_phase_of():
    fractions = np.array([0.0, 0.32, 0.33, 0.34, 0.65, 0.66, 0.67, 1.0])
    for phase in tcd.PHASES:
        mask = tcd.phase_mask(fractions, phase)
        expected = np.array([tcd.phase_of(v) == phase for v in fractions])
        assert np.array_equal(mask, expected), phase
    with pytest.raises(ValueError):
        tcd.phase_mask(fractions, "bogus")


# ---------------------------------------------------------------------------
# ECE (the run's 10-bin definition)
# ---------------------------------------------------------------------------

def test_ece_empty_returns_zero():
    assert tcd.ece_10bin(np.array([]), np.array([])) == 0.0


def test_ece_hand_computed_two_bins():
    p = np.array([0.05, 0.95])
    y = np.array([0, 1])
    # bins 0 and 9, each weight 0.5: 0.5*|0-0.05| + 0.5*|1-0.95| = 0.05
    assert tcd.ece_10bin(p, y) == pytest.approx(0.05)


def test_ece_bin_edge_goes_to_upper_bin():
    """p == 0.1 belongs to bin [0.1, 0.2), not bin [0.0, 0.1)."""
    p = np.array([0.05, 0.1])
    y = np.array([1, 0])
    # split bins: 0.5*|1-0.05| + 0.5*|0-0.1| = 0.525 (same-bin would give 0.425)
    assert tcd.ece_10bin(p, y) == pytest.approx(0.525)


def test_ece_last_bin_right_inclusive():
    """p == 1.0 must be counted (last bin is [0.9, 1.0])."""
    assert tcd.ece_10bin(np.array([1.0]), np.array([0])) == pytest.approx(1.0)
    assert tcd.ece_10bin(np.array([1.0]), np.array([1])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Calibrator application
# ---------------------------------------------------------------------------

def test_apply_platt_scalar_and_vector_agree():
    a, b = 2.0, -1.0
    assert tcd.apply_platt_scalar(0.5, a, b) == pytest.approx(0.5)
    s = np.array([0.0, 0.25, 0.5, 0.685533])
    vec = tcd.apply_calibrator("platt-logistic", {"a": "2.0", "b": "-1.0"}, s)
    scalar = np.array([tcd.apply_platt_scalar(v, a, b) for v in s])
    assert np.allclose(vec, scalar, atol=1e-12)


def test_apply_similarity_temperature():
    s = np.array([1.0])
    p = tcd.apply_calibrator("similarity-temperature", {"T": "2.0"}, s)
    assert p[0] == pytest.approx(1.0 / (1.0 + math.exp(-0.5)))


def test_apply_isotonic_interp_and_clip():
    params = {"x_thresholds": ["0.0", "1.0"], "y_thresholds": ["0.2", "0.8"]}
    s = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    p = tcd.apply_calibrator("isotonic", params, s)
    assert p[0] == pytest.approx(0.2)  # left edge clip
    assert p[1] == pytest.approx(0.2)
    assert p[2] == pytest.approx(0.5)  # linear interpolation
    assert p[3] == pytest.approx(0.8)
    assert p[4] == pytest.approx(0.8)  # right edge clip


def test_apply_isotonic_flat_block():
    """A flat block (equal y at both knots) yields the block value inside it."""
    params = {
        "x_thresholds": ["0.0", "0.3", "0.73"],
        "y_thresholds": ["0.1", "0.4", "0.4"],
    }
    p = tcd.apply_calibrator("isotonic", params, np.array([0.5, 0.685533]))
    assert np.allclose(p, 0.4)


def test_apply_isotonic_rejects_nonincreasing_knots():
    params = {"x_thresholds": ["0.0", "0.0"], "y_thresholds": ["0.1", "0.2"]}
    with pytest.raises(tcd.AssumptionError):
        tcd.apply_calibrator("isotonic", params, np.array([0.5]))


def test_apply_calibrator_unknown_name():
    with pytest.raises(ValueError):
        tcd.apply_calibrator("bogus", {}, np.array([0.5]))


# ---------------------------------------------------------------------------
# Never-buzz inclusion-exclusion bounds
# ---------------------------------------------------------------------------

def test_mutual_never_buzz_bounds_binding():
    b = tcd.mutual_never_buzz_bounds(2000, 1500, 2200, 3037)
    assert b["mutual_lb"] == 463  # 2000 + 1500 - 3037
    assert b["mutual_ub"] == 1500
    assert b["same_finite_stop_bounds"] == [700, 1737]


def test_mutual_never_buzz_bounds_nonbinding():
    b = tcd.mutual_never_buzz_bounds(100, 200, 150, 3037)
    assert b["mutual_lb"] == 0
    assert b["mutual_ub"] == 100  # min(nb_mc, nb_qa, zero_count)
    assert b["same_finite_stop_bounds"] == [50, 150]


def test_mutual_never_buzz_bounds_ub_capped_by_zero_mass():
    """The upper bound is capped by the D=0 mass (mutual NB implies D=0)."""
    b = tcd.mutual_never_buzz_bounds(2118, 2405, 1581, 3037)
    assert b["mutual_lb"] == 1486  # 2118 + 2405 - 3037
    assert b["mutual_ub"] == 1581  # min(2118, 2405, 1581): capped by zero_count
    assert b["same_finite_stop_bounds"] == [0, 95]


# ---------------------------------------------------------------------------
# Myopic p* and applicability
# ---------------------------------------------------------------------------

def test_p_star_values():
    assert tcd.p_star(5.0, 10.0) == pytest.approx(1.0 / 3.0)
    assert tcd.p_star(10.0, 10.0) == pytest.approx(0.5)
    assert tcd.p_star(5.0, 15.0) == pytest.approx(0.25)
    assert tcd.p_star(10.0, 15.0) == pytest.approx(0.4)


def test_applicable_pstars_split_half():
    sched = {"correct_early": "15", "correct_late": "10", "wrong": "-5", "split": "0.5"}
    assert tcd.applicable_pstars("early", sched) == [pytest.approx(0.25)]
    mid = tcd.applicable_pstars("mid", sched)
    assert mid == [pytest.approx(0.25), pytest.approx(1.0 / 3.0)]  # straddles split
    assert tcd.applicable_pstars("late", sched) == [pytest.approx(1.0 / 3.0)]


def test_applicable_pstars_split_one():
    # acf_flat: split=1.0 and equal rewards -> single p* everywhere
    sched = {"correct_early": "10", "correct_late": "10", "wrong": "-5", "split": "1.0"}
    for phase in tcd.PHASES:
        assert tcd.applicable_pstars(phase, sched) == [pytest.approx(1.0 / 3.0)]
    # split=1.0 with distinct rewards: late reward applies only at fraction 1.0
    sched2 = {"correct_early": "15", "correct_late": "10", "wrong": "-5", "split": "1.0"}
    assert tcd.applicable_pstars("early", sched2) == [pytest.approx(0.25)]
    assert tcd.applicable_pstars("mid", sched2) == [pytest.approx(0.25)]
    assert tcd.applicable_pstars("late", sched2) == [
        pytest.approx(0.25),
        pytest.approx(1.0 / 3.0),
    ]


# ---------------------------------------------------------------------------
# Dispersion and accuracy helpers
# ---------------------------------------------------------------------------

def test_per_item_dispersion():
    matrix = np.array([[0, 0, 0], [1, -1, 0]])
    d = tcd.per_item_dispersion(matrix)
    assert d["sd"][0] == pytest.approx(0.0)
    assert d["sd"][1] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert d["rng"].tolist() == [0.0, 2.0]
    assert d["iqr"][0] == pytest.approx(0.0)
    assert d["iqr"][1] == pytest.approx(1.0)  # p75(-1,0,1)=0.5, p25=-0.5


def test_accuracy_by_phase_boundary_rows():
    fractions = np.array([0.0, 0.33, 0.5, 0.66, 1.0])
    correct = np.array([1, 0, 1, 0, 1])
    out = tcd.accuracy_by_phase(fractions, correct)
    assert out["n"] == 5 and out["n_correct"] == 3
    assert out["accuracy"] == pytest.approx(0.6)
    assert out["by_phase"]["early"] == {"n": 1, "n_correct": 1, "accuracy": 1.0}
    assert out["by_phase"]["mid"]["n"] == 2  # 0.33 and 0.5
    assert out["by_phase"]["mid"]["accuracy"] == pytest.approx(0.5)
    assert out["by_phase"]["late"]["n"] == 2  # 0.66 and 1.0
    assert out["by_phase"]["late"]["accuracy"] == pytest.approx(0.5)


def test_accuracy_by_phase_empty_phase():
    out = tcd.accuracy_by_phase(np.array([0.9]), np.array([1]))
    assert out["by_phase"]["early"]["accuracy"] is None
    assert out["by_phase"]["late"]["accuracy"] == 1.0


def test_dist_summary():
    s = tcd.dist_summary(np.array([0.0, 1.0, 2.0, 3.0]))
    assert s["mean"] == pytest.approx(1.5)
    assert s["median"] == pytest.approx(1.5)
    assert s["min"] == 0.0 and s["max"] == 3.0


# ---------------------------------------------------------------------------
# Integration: mandatory gate against the real frozen export
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_EXPORT, reason="frozen evidence export not present (CI)")
def test_integration_full_pipeline_and_gate(tmp_path):
    out_dir = tmp_path / "diag"
    rc = tcd.main(["--out", str(out_dir)])
    assert rc == 0

    with open(out_dir / "provenance.json", encoding="utf-8") as handle:
        prov = json.load(handle)
    assert prov["regression_gate"]["status"] == "PASS"
    gate = prov["regression_gate"]["phases"]
    assert gate["early"]["recomputed_ece"] == 0.013747
    assert gate["mid"]["recomputed_ece"] == 0.013958
    assert gate["late"]["recomputed_ece"] == 0.031919
    assert all(c["status"] == "PASS" for c in prov["cross_checks"])
    assert prov["cells"]["count"] == 96

    with open(out_dir / "mc_accuracy.json", encoding="utf-8") as handle:
        acc = json.load(handle)
    assert acc["test"]["accuracy"] == 0.397118
    assert acc["test"]["n_correct"] == 5925
    assert acc["val"]["accuracy"] == 0.403402

    with open(out_dir / "calibration_diagnostics.json", encoding="utf-8") as handle:
        calib = json.load(handle)
    platt = calib["per_calibrator"]["platt-logistic"]["phases"]
    assert platt["early"]["mc_ece"] == 0.013747
    gap_block = platt["late"]["qa_gold_idealized_gap"]
    assert "warning" in gap_block and "NOT a calibration ECE" in gap_block["warning"]
    simtemp = calib["per_calibrator"]["similarity-temperature"]
    assert simtemp["temperature_note"]["saturated_at_grid_max"] is True
    assert calib["per_calibrator"]["isotonic"]["knot_counts"] == {
        "early": 18, "mid": 28, "late": 40,
    }

    with open(out_dir / "shift_specification_variability.json", encoding="utf-8") as handle:
        shift = json.load(handle)
    assert shift["n_cells"] == 96 and shift["n_items"] == 3037
    assert len(shift["per_cell"]) == 96
    assert shift["signed_mean_summary"]["min"] == -0.676655
    assert shift["signed_mean_summary"]["max"] == 0.109648
    assert "myopic" in shift["reachability_bound"]["framing"].lower()
    assert "not a recomputation" in shift["reachability_bound"]["framing"].lower()

    tex = (out_dir / "tables.tex").read_text(encoding="utf-8")
    assert "\\usepackage" not in tex
    assert tex.count("\\begin{table}") == 3
    assert "\\toprule" in tex and "\\bottomrule" in tex
