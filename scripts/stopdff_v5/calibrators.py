"""Calibrator registry (indexed by SCIENTIFIC_CONTRACT.md).

Every calibrator is fitted on VALIDATION MC ROWS ONLY, per phase (early/mid/late), and
the resulting phase-specific map is applied without refitting to both MC and QA rows.

    early: prefix_fraction < 0.33
    mid:   0.33 <= prefix_fraction < 0.66
    late:  prefix_fraction >= 0.66

Calibrators:
  platt-logistic         : uses staged calibration.json parameters; MUST NOT refit.
  similarity-temperature : p = sigma(s / T_phase), T fit over a fixed grid (min log loss).
  isotonic               : one isotonic model per phase (increasing, clip, y in [0,1]).

A phase fit requires >= 10 MC rows and both correctness classes, except a platt phase that
explicitly records a constant model. Failure invalidates the cell (no fallback).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .profile import CALIBRATION

PHASE_EARLY_MAX = 0.33
PHASE_MID_MAX = 0.66
MIN_PHASE_ROWS = int(CALIBRATION["minimum_phase_rows"])
SIMTEMP_GRID = tuple(float(x) for x in CALIBRATION["similarity_temperature_grid"])
PHASES = ("early", "mid", "late")
_EPS = 1e-12


class CalibratorFitError(ValueError):
    """A fit prerequisite failed (too few rows / one class / missing params). Cell invalid."""


def phase_of(prefix_fraction: float) -> str:
    if prefix_fraction < PHASE_EARLY_MAX:
        return "early"
    if prefix_fraction < PHASE_MID_MAX:
        return "mid"
    return "late"


def _sigmoid(z: float) -> float:
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _binary_log_loss(ys: np.ndarray, ps: np.ndarray) -> float:
    ps = np.clip(ps, _EPS, 1.0 - _EPS)
    return float(np.mean(-(ys * np.log(ps) + (1.0 - ys) * np.log(1.0 - ps))))


@dataclass
class Calibrator:
    """Fitted per-phase calibrator with a deterministic apply() and identity params."""

    name: str
    phase_params: dict[str, dict[str, Any]]  # phase -> serializable params (fingerprint)
    _apply_phase: Any = None  # dict[str, callable]

    def apply(self, raw_similarity: float, prefix_fraction: float) -> float:
        phase = phase_of(prefix_fraction)
        fn = self._apply_phase[phase]
        return float(min(1.0, max(0.0, fn(float(raw_similarity)))))

    def parameters(self) -> dict[str, Any]:
        """Deterministic, canonical (string-decimal) parameter block for the fingerprint."""
        return {"calibrator": self.name, "phases": self.phase_params}


def _rows_by_phase(mc_val_rows: Sequence[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {p: [] for p in PHASES}
    for row in mc_val_rows:
        out[phase_of(float(row["prefix_fraction"]))].append(row)
    return out


def _require_fit_prereq(rows: list[dict], phase: str, calibrator: str) -> None:
    if len(rows) < MIN_PHASE_ROWS:
        raise CalibratorFitError(
            f"{calibrator}: phase {phase!r} has {len(rows)} rows < {MIN_PHASE_ROWS}"
        )
    classes = {int(r["correct"]) for r in rows}
    if classes != {0, 1}:
        raise CalibratorFitError(
            f"{calibrator}: phase {phase!r} lacks both correctness classes (saw {sorted(classes)})"
        )


def _fmt_num(x: float) -> str:
    return repr(float(x))


# --- platt-logistic ---------------------------------------------------------------


def fit_platt(calibration_json: dict) -> Calibrator:
    """Load staged Platt parameters from calibration.json (per phase). No refit."""
    per_bucket = calibration_json.get("per_bucket")
    if not isinstance(per_bucket, dict):
        raise CalibratorFitError("calibration.json missing 'per_bucket'")
    apply_fns: dict[str, Any] = {}
    params: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        block = per_bucket.get(phase)
        if not isinstance(block, dict):
            raise CalibratorFitError(f"platt-logistic: calibration.json missing phase {phase!r}")
        coef = block.get("platt_coef")
        intercept = block.get("platt_intercept")
        model_type = block.get("platt_model_type")
        if coef is None or intercept is None:
            if model_type != "constant":
                raise CalibratorFitError(
                    f"platt-logistic: phase {phase!r} has null params without model_type=constant"
                )
            prob = float(block.get("platt_constant_probability", 0.0))
            prob = min(1.0, max(0.0, prob))
            apply_fns[phase] = (lambda s, _p=prob: _p)
            params[phase] = {"model": "constant", "probability": _fmt_num(prob)}
        else:
            a = float(coef)
            b = float(intercept)
            apply_fns[phase] = (lambda s, _a=a, _b=b: _sigmoid(_a * s + _b))
            params[phase] = {"model": "logistic", "a": _fmt_num(a), "b": _fmt_num(b)}
    return Calibrator(name="platt-logistic", phase_params=params, _apply_phase=apply_fns)


# --- similarity-temperature -------------------------------------------------------


def fit_similarity_temperature(mc_val_rows: Sequence[dict]) -> Calibrator:
    by_phase = _rows_by_phase(mc_val_rows)
    apply_fns: dict[str, Any] = {}
    params: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        rows = by_phase[phase]
        _require_fit_prereq(rows, phase, "similarity-temperature")
        s = np.array([float(r["raw_similarity"]) for r in rows], dtype=np.float64)
        y = np.array([float(int(r["correct"])) for r in rows], dtype=np.float64)
        best_T = None
        best_loss = math.inf
        for T in SIMTEMP_GRID:  # ascending; strict-min keeps smallest T on tie
            ps = 1.0 / (1.0 + np.exp(-np.clip(s / T, -500.0, 500.0)))
            loss = _binary_log_loss(y, ps)
            if loss < best_loss - 1e-15:
                best_loss = loss
                best_T = T
        assert best_T is not None
        apply_fns[phase] = (lambda sim, _T=best_T: _sigmoid(sim / _T))
        params[phase] = {"model": "similarity_temperature", "T": _fmt_num(best_T)}
    return Calibrator(name="similarity-temperature", phase_params=params, _apply_phase=apply_fns)


# --- isotonic ---------------------------------------------------------------------


def fit_isotonic(mc_val_rows: Sequence[dict]) -> Calibrator:
    from sklearn.isotonic import IsotonicRegression

    by_phase = _rows_by_phase(mc_val_rows)
    apply_fns: dict[str, Any] = {}
    params: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        rows = by_phase[phase]
        _require_fit_prereq(rows, phase, "isotonic")
        s = np.array([float(r["raw_similarity"]) for r in rows], dtype=np.float64)
        y = np.array([float(int(r["correct"])) for r in rows], dtype=np.float64)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(s, y)
        apply_fns[phase] = (lambda sim, _m=iso: float(_m.predict([sim])[0]))
        params[phase] = {
            "model": "isotonic",
            "x_thresholds": [_fmt_num(x) for x in np.asarray(iso.X_thresholds_).tolist()],
            "y_thresholds": [_fmt_num(v) for v in np.asarray(iso.y_thresholds_).tolist()],
        }
    return Calibrator(name="isotonic", phase_params=params, _apply_phase=apply_fns)


def fit_calibrator(
    name: str,
    *,
    mc_val_rows: Sequence[dict],
    calibration_json: dict | None,
) -> Calibrator:
    if name == "platt-logistic":
        if calibration_json is None:
            raise CalibratorFitError("platt-logistic requires calibration.json")
        return fit_platt(calibration_json)
    if name == "similarity-temperature":
        return fit_similarity_temperature(mc_val_rows)
    if name == "isotonic":
        return fit_isotonic(mc_val_rows)
    raise ValueError(f"unknown calibrator {name!r}")
