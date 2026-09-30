#!/usr/bin/env python
"""TAE camera-ready post-hoc diagnostics (artifact ``tae_camera_ready_diag_v1``).

Computes three additive analyses for the NeurIPS 2026 TAE workshop camera-ready
from the frozen, checksummed evidence export of run ``final_modal_5d5328102912``
(read-only; nothing under ``artifacts/`` is modified):

1. ``mc_accuracy.json`` — MC-arm accuracy (test + val): aggregate, per phase
   (character-fraction convention, boundaries 0.33/0.66), and an index-based
   per-``prefix_idx`` curve.  QA-arm accuracy is NOT reportable (QA rows carry
   ``correct == 1`` by construction: gold-idealized reference arm).
2. ``calibration_diagnostics.json`` — per calibrator x phase: MC ECE (the
   run's own 10-bin definition), attained calibrated-probability ceilings and
   level statistics per format (the level-gap evidence), the sim-temp
   grid-saturation note, isotonic knot counts, and the QA gold-idealization
   gap (explicitly NOT an ECE).
3. ``shift_specification_variability.json`` — the RESCOPED item 3:
   cross-specification variability of the paired shift D = tau_MC - tau_QA
   across the 96 registered cells (per-item SD/range/IQR; per-axis attribution
   of cell signed means), never-buzz inclusion-exclusion bounds, and the
   myopic reachability bound p* = c_wrong / (R_t + c_wrong) per reward
   schedule versus attained calibrated-probability ceilings.

Plus ``provenance.json`` (sha256 + byte size of every input read, run id, git
rev, versions, gate result, method notes) and ``tables.tex`` (booktabs LaTeX,
no preamble lines).

Method notes
------------
- Isotonic calibrators are reconstructed from the stored per-phase knots
  (``x_thresholds``/``y_thresholds``) via linear interpolation with edge
  clipping (``np.interp`` + clip to [0, 1]) — numerically equivalent to
  ``sklearn.isotonic.IsotonicRegression(out_of_bounds="clip").predict`` on the
  stored thresholds; sklearn itself is never imported.
- The MANDATORY regression gate reproduces the stored Platt ECE triple
  0.013747 / 0.013958 / 0.031919 (early/mid/late) from the eval rows + stored
  parameters using the run's own ECE definition before any new number is
  emitted; the script aborts nonzero on any mismatch.
- Deterministic: no sampling, no RNG.  Only numpy is required beyond the
  standard library.

Usage
-----
``.venv/bin/python scripts/tae_camera_ready_diagnostics.py``
(optionally ``--export-root PATH --out DIR``).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "final_modal_5d5328102912"
DEFAULT_EXPORT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "stopdff_modal_controller_0017b89d_20260827"
    / f"modal_export_{RUN_ID}"
)
DEFAULT_OUT = REPO_ROOT / "results" / "tae_camera_ready_diag_v1"

PHASES = ("early", "mid", "late")
PHASE_EARLY_MAX = 0.33
PHASE_MID_MAX = 0.66
CALIBRATORS = ("platt-logistic", "similarity-temperature", "isotonic")
AXES = (
    "reward_schedule",
    "continuation",
    "calibrator",
    "prefix_bucketing",
    "category_pooling",
)

# ---------------------------------------------------------------------------
# Pre-validated expectations (Lane I verdict, briefing 2026-09-30).  Any
# mismatch at runtime means schema/content drift: the script stops nonzero.
# ---------------------------------------------------------------------------
EXPECTED_PLATT_ECE = {"early": 0.013747, "mid": 0.013958, "late": 0.031919}
EXPECTED_EVAL = {"rows": 29840, "items": 3037, "MC": 14920, "QA": 14920, "split": "test"}
EXPECTED_FIT = {"rows": 30104, "items": 3050, "MC": 15052, "QA": 15052, "split": "val"}
EXPECTED_TEST_MC_PHASE_N = {"early": 3386, "mid": 4479, "late": 7055}
EXPECTED_TEST_MC_CORRECT = 5925
EXPECTED_TEST_MC_ACC = 0.397118
EXPECTED_VAL_MC_ACC = 0.403402
EXPECTED_TEST_PHASE_ACC_4DP = {"early": 0.3334, "mid": 0.3717, "late": 0.4438}
EXPECTED_VAL_PHASE_ACC_4DP = {"early": 0.3443, "mid": 0.3761, "late": 0.4489}
EXPECTED_MAX_RAW_SIM = 0.685533
EXPECTED_N_CELLS = 96
EXPECTED_N_ITEMS = 3037
EXPECTED_ISOTONIC_KNOTS = {"early": 18, "mid": 28, "late": 40}
EXPECTED_SIMTEMP_T = 5.0
EXPECTED_CEILINGS_4DP = {
    "platt-logistic": {"early": 0.4752, "mid": 0.4730, "late": 0.7179},
    "similarity-temperature": {"early": 0.5342, "mid": 0.5341, "late": 0.5342},
    "isotonic": {"early": 0.3996, "mid": 0.4169, "late": 0.6545},
}
EXPECTED_SIGNED_MEAN_MIN = -0.676655
EXPECTED_SIGNED_MEAN_MIN_CELL = (
    "reward_schedule=power_mark__continuation=empirical_bucket__calibrator=isotonic"
    "__prefix_bucketing=early_mid_late__category_pooling=pooled_category"
)
EXPECTED_SIGNED_MEAN_MAX = 0.109648
EXPECTED_ZERO_FRACTION_MIN_PCT = 52.058
EXPECTED_ZERO_FRACTION_MAX_PCT = 99.967

QA_GAP_WARNING = (
    "NOT a calibration ECE. QA-arm labels are identically 1 by construction "
    "(gold-idealized reference arm: cosine similarity to the constructed gold "
    "answer, correctness hard-coded to 1), so any 'QA ECE' degenerates to "
    "1 - mean(calibrated p). This value therefore measures the gap between "
    "the calibrated confidence level and the idealized always-correct label, "
    "not per-format calibration quality."
)

MYOPIC_FRAMING = (
    "Myopic reachability bound, NOT a recomputation of the Bellman policy: "
    "p* = c_wrong / (R_t + c_wrong) is the probability at which a single "
    "immediate ANSWER action becomes utility-positive against ABSTAIN "
    "(value 0) under reward R_t for a correct answer and -c_wrong for a "
    "wrong one. WAIT values, continuation estimates, and the DP recursion "
    "are deliberately not recomputed; wait_cost does not enter this bound. "
    "A calibrator x phase whose attained calibrated-probability ceiling "
    "lies below every applicable p* can never make answering myopically "
    "utility-positive in that phase for that reward schedule."
)


class AssumptionError(RuntimeError):
    """A pre-validated briefing assumption failed at runtime (schema drift)."""


class GateError(RuntimeError):
    """The mandatory Platt-ECE regression gate failed to reproduce."""


def _require(condition: bool, message: str) -> None:
    """Raise :class:`AssumptionError` with ``message`` unless ``condition``."""
    if not condition:
        raise AssumptionError(message)


# ---------------------------------------------------------------------------
# Pure functions (unit-tested)
# ---------------------------------------------------------------------------

def phase_of(prefix_fraction: float) -> str:
    """Map a character-fraction to the run's phase convention.

    Parameters
    ----------
    prefix_fraction : float
        Prefix character count divided by full-question character count.

    Returns
    -------
    str
        ``"early"`` if ``prefix_fraction < 0.33``, ``"mid"`` if
        ``prefix_fraction < 0.66``, else ``"late"`` — identical edge
        semantics to ``scripts/stopdff_v5/calibrators.phase_of``
        (0.33 -> mid, 0.66 -> late).
    """
    if prefix_fraction < PHASE_EARLY_MAX:
        return "early"
    if prefix_fraction < PHASE_MID_MAX:
        return "mid"
    return "late"


def phase_mask(fractions: np.ndarray, phase: str) -> np.ndarray:
    """Vectorized boolean mask matching :func:`phase_of` exactly.

    Parameters
    ----------
    fractions : numpy.ndarray
        Array of prefix fractions.
    phase : str
        One of ``"early"``, ``"mid"``, ``"late"``.
    """
    if phase == "early":
        return fractions < PHASE_EARLY_MAX
    if phase == "mid":
        return (fractions >= PHASE_EARLY_MAX) & (fractions < PHASE_MID_MAX)
    if phase == "late":
        return fractions >= PHASE_MID_MAX
    raise ValueError(f"unknown phase {phase!r}")


def ece_10bin(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """The run's own 10-bin expected calibration error.

    Exact port of ``scripts/stopdff_v5/adapter_build.py`` (lines 183-204):
    bins are ``np.linspace(0, 1, 11)``; each bin is ``[lower, upper)`` except
    the last, which is ``[0.9, 1.0]`` (right-inclusive); the contribution is
    count-weighted ``|mean(label) - mean(p)|``; an empty input returns 0.0.

    Parameters
    ----------
    probabilities : numpy.ndarray
        Calibrated probabilities in [0, 1].
    labels : numpy.ndarray
        Binary correctness labels (0/1).

    Returns
    -------
    float
        The expected calibration error (unrounded).
    """
    if len(probabilities) == 0:
        return 0.0
    total = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        count = int(mask.sum())
        if count:
            total += (count / len(probabilities)) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return total


def sigmoid_scalar(z: float) -> float:
    """Scalar sigmoid with the run's +/-500 clip (mirrors ``calibrators._sigmoid``)."""
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def apply_platt_scalar(raw_similarity: float, a: float, b: float) -> float:
    """Scalar Platt application ``sigmoid(a * s + b)`` (gate code path).

    Mirrors ``scripts/stopdff_v5/calibrators.apply_platt_logistic`` exactly so
    the regression gate reproduces the stored artifact byte-for-byte at the
    stored 6-decimal rounding.
    """
    return sigmoid_scalar(float(a) * float(raw_similarity) + float(b))


def sigmoid_vec(z: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid with the same +/-500 clip as :func:`sigmoid_scalar`."""
    z = np.clip(np.asarray(z, dtype=np.float64), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def apply_calibrator(
    calibrator: str, params: dict, similarities: np.ndarray
) -> np.ndarray:
    """Apply one stored per-phase calibrator map to raw similarities.

    Parameters
    ----------
    calibrator : str
        ``"platt-logistic"``, ``"similarity-temperature"``, or ``"isotonic"``.
    params : dict
        The stored single-phase parameter block from
        ``calibrator_parameters.phases[<phase>]`` of a cell file (values are
        decimal strings, as serialized by the run).
    similarities : numpy.ndarray
        Raw similarity scores.

    Returns
    -------
    numpy.ndarray
        Calibrated probabilities, clipped to [0, 1] (matching
        ``calibrators.Calibrator.apply``).

    Notes
    -----
    Isotonic is reconstructed via linear interpolation on the stored knots
    with edge clipping (``np.interp``), the documented sklearn-equivalent
    reconstruction; the x knots are required to be strictly increasing.
    """
    s = np.asarray(similarities, dtype=np.float64)
    if calibrator == "platt-logistic":
        p = sigmoid_vec(float(params["a"]) * s + float(params["b"]))
    elif calibrator == "similarity-temperature":
        p = sigmoid_vec(s / float(params["T"]))
    elif calibrator == "isotonic":
        x = np.asarray([float(v) for v in params["x_thresholds"]], dtype=np.float64)
        y = np.asarray([float(v) for v in params["y_thresholds"]], dtype=np.float64)
        if not np.all(np.diff(x) > 0):
            raise AssumptionError(
                "isotonic x_thresholds are not strictly increasing; "
                "np.interp reconstruction is not well-defined"
            )
        p = np.interp(s, x, y)
    else:
        raise ValueError(f"unknown calibrator {calibrator!r}")
    return np.clip(p, 0.0, 1.0)


def mutual_never_buzz_bounds(
    nb_mc: int, nb_qa: int, zero_count: int, n_items: int
) -> dict:
    """Inclusion-exclusion bounds on mutual never-buzz and finite agreement.

    Never-buzz is coded ``stop_index == item horizon`` in both arms
    (``scripts/stopdff_v5/policy.py``), and the paired arms share each item's
    prefix set, so mutual never-buzz implies D = 0 and the D = 0 mass
    decomposes exactly into {same finite stop} plus {mutual never-buzz}.

    Parameters
    ----------
    nb_mc, nb_qa : int
        Per-cell never-buzz counts for the MC and QA arms.
    zero_count : int
        Number of items with D = 0 in the cell.
    n_items : int
        Paired item count (3037).

    Returns
    -------
    dict
        ``mutual_lb`` = max(0, nb_mc + nb_qa - n_items);
        ``mutual_ub`` = min(nb_mc, nb_qa, zero_count);
        ``same_finite_stop_bounds`` = [zero_count - mutual_ub,
        zero_count - mutual_lb] (counts).
    """
    mutual_lb = max(0, nb_mc + nb_qa - n_items)
    mutual_ub = min(nb_mc, nb_qa, zero_count)
    return {
        "mutual_lb": mutual_lb,
        "mutual_ub": mutual_ub,
        "same_finite_stop_bounds": [zero_count - mutual_ub, zero_count - mutual_lb],
    }


def p_star(c_wrong: float, reward_correct: float) -> float:
    """Myopic answer-positivity threshold ``c_wrong / (R + c_wrong)``.

    The probability above which one immediate ANSWER action has positive
    expected value against ABSTAIN (value 0):
    ``p * R - (1 - p) * c_wrong > 0``.
    """
    return c_wrong / (reward_correct + c_wrong)


def applicable_pstars(phase: str, schedule: dict) -> list:
    """Which p* values can apply inside a calibration phase.

    The reward switch is ``R = correct_early if prefix_fraction < split else
    correct_late`` (``scripts/stopdff_v5/rewards.py``) on the same character
    fraction as the calibration phases (early [0, 0.33), mid [0.33, 0.66),
    late [0.66, 1.0]).  A phase interval contributes the early-reward p* when
    its lower edge lies below ``split`` and the late-reward p* when its upper
    edge (1.0 is attainable at the final prefix) reaches ``split``.

    Parameters
    ----------
    phase : str
        Calibration phase name.
    schedule : dict
        Parsed reward schedule with float fields ``correct_early``,
        ``correct_late``, ``wrong`` (negative), ``split``.

    Returns
    -------
    list of float
        Sorted unique applicable p* values for the phase.
    """
    intervals = {"early": (0.0, PHASE_EARLY_MAX), "mid": (PHASE_EARLY_MAX, PHASE_MID_MAX), "late": (PHASE_MID_MAX, 1.0)}
    lo, hi = intervals[phase]
    split = float(schedule["split"])
    c_wrong = -float(schedule["wrong"])
    out = set()
    if lo < split:
        out.add(p_star(c_wrong, float(schedule["correct_early"])))
    if hi >= split:
        out.add(p_star(c_wrong, float(schedule["correct_late"])))
    return sorted(out)


def per_item_dispersion(shift_matrix: np.ndarray) -> dict:
    """Per-item dispersion of D across cells.

    Parameters
    ----------
    shift_matrix : numpy.ndarray
        Integer array of shape (n_items, n_cells).

    Returns
    -------
    dict of numpy.ndarray
        ``sd`` (population SD, ddof=0), ``rng`` (max - min), ``iqr``
        (p75 - p25, numpy linear interpolation), each of length n_items.
    """
    m = np.asarray(shift_matrix, dtype=np.float64)
    sd = m.std(axis=1, ddof=0)
    rng = m.max(axis=1) - m.min(axis=1)
    q75, q25 = np.percentile(m, [75, 25], axis=1)
    return {"sd": sd, "rng": rng, "iqr": q75 - q25}


def dist_summary(values: np.ndarray) -> dict:
    """Mean/median/p90/min/max summary of a 1-D array, rounded to 6 dp."""
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(v.mean()), 6),
        "median": round(float(np.median(v)), 6),
        "p90": round(float(np.percentile(v, 90)), 6),
        "min": round(float(v.min()), 6),
        "max": round(float(v.max()), 6),
    }


def accuracy_by_phase(fractions: np.ndarray, correct: np.ndarray) -> dict:
    """Aggregate and per-phase accuracy under the run's phase convention.

    Parameters
    ----------
    fractions : numpy.ndarray
        Prefix character fractions.
    correct : numpy.ndarray
        Binary correctness labels.

    Returns
    -------
    dict
        ``{"n", "n_correct", "accuracy", "by_phase": {phase: {...}}}``.
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.int64)
    out = {
        "n": int(len(correct)),
        "n_correct": int(correct.sum()),
        "accuracy": round(float(correct.mean()), 6) if len(correct) else None,
        "by_phase": {},
    }
    for phase in PHASES:
        mask = phase_mask(fractions, phase)
        n = int(mask.sum())
        out["by_phase"][phase] = {
            "n": n,
            "n_correct": int(correct[mask].sum()),
            "accuracy": round(float(correct[mask].mean()), 6) if n else None,
        }
    return out


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Streamed sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_gz(path: Path) -> list:
    """Load a gzipped JSONL file into a list of dicts."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def write_json(path: Path, payload: dict) -> None:
    """Write deterministic JSON (sorted keys, indent 2, trailing newline)."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def tex_escape(text: str) -> str:
    """Escape the LaTeX-special characters that appear in cell/axis names."""
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def rows_to_arrays(rows: list) -> dict:
    """Columnar views of the adapter rows needed by the diagnostics."""
    return {
        "format": np.asarray([r["format"] for r in rows]),
        "correct": np.asarray([int(r["correct"]) for r in rows], dtype=np.int64),
        "raw_similarity": np.asarray(
            [float(r["raw_similarity"]) for r in rows], dtype=np.float64
        ),
        "prefix_fraction": np.asarray(
            [float(r["prefix_fraction"]) for r in rows], dtype=np.float64
        ),
        "prefix_idx": np.asarray([int(r["prefix_idx"]) for r in rows], dtype=np.int64),
        "item_id": np.asarray([str(r["item_id"]) for r in rows]),
    }


def assert_row_file(rows: list, expected: dict, name: str) -> None:
    """Assert the briefing's row/item counts, split purity, and QA label law."""
    _require(len(rows) == expected["rows"], f"{name}: row count {len(rows)} != {expected['rows']}")
    fmt_counts = defaultdict(int)
    items = set()
    for r in rows:
        fmt_counts[r["format"]] += 1
        items.add(str(r["item_id"]))
        _require(
            r["split"] == expected["split"],
            f"{name}: row split {r['split']!r} != {expected['split']!r}",
        )
    _require(fmt_counts["MC"] == expected["MC"], f"{name}: MC count {fmt_counts['MC']} != {expected['MC']}")
    _require(fmt_counts["QA"] == expected["QA"], f"{name}: QA count {fmt_counts['QA']} != {expected['QA']}")
    _require(len(items) == expected["items"], f"{name}: item count {len(items)} != {expected['items']}")
    qa_bad = sum(1 for r in rows if r["format"] == "QA" and int(r["correct"]) != 1)
    _require(qa_bad == 0, f"{name}: {qa_bad} QA rows with correct != 1 (gold-idealized law violated)")
    per_item = defaultdict(lambda: [0, 0])
    for r in rows:
        per_item[str(r["item_id"])][0 if r["format"] == "MC" else 1] += 1
    unbalanced = [i for i, (mc, qa) in per_item.items() if mc != qa or mc == 0]
    _require(
        not unbalanced,
        f"{name}: {len(unbalanced)} items without equal MC/QA prefix rows "
        f"(first: {unbalanced[:3]}); paired-horizon assumption violated",
    )


def run_regression_gate(eval_rows: list, calibration: dict) -> dict:
    """MANDATORY gate: reproduce the stored Platt ECE triple exactly.

    Mirrors the adapter's own computation (scalar sigmoid on the stored
    6-dp-rounded coefficients over test MC rows, the run's 10-bin ECE,
    rounded to 6 dp).  Raises :class:`GateError` on any mismatch.
    """
    result = {}
    mc_test = [r for r in eval_rows if r["format"] == "MC" and r["split"] == "test"]
    for phase in PHASES:
        block = calibration["per_bucket"][phase]
        a = float(block["platt_coef"])
        b = float(block["platt_intercept"])
        sub = [r for r in mc_test if phase_of(float(r["prefix_fraction"])) == phase]
        probs = np.asarray(
            [apply_platt_scalar(float(r["raw_similarity"]), a, b) for r in sub],
            dtype=np.float64,
        )
        labels = np.asarray([int(r["correct"]) for r in sub], dtype=np.int64)
        recomputed = round(ece_10bin(probs, labels), 6)
        stored = float(block["ece"])
        expected = EXPECTED_PLATT_ECE[phase]
        if not (recomputed == stored == expected and len(sub) == int(block["n_samples"])):
            raise GateError(
                f"regression gate FAILED at phase {phase!r}: recomputed={recomputed} "
                f"stored={stored} briefing={expected} n={len(sub)} stored_n={block['n_samples']}"
            )
        result[phase] = {
            "stored_ece": stored,
            "recomputed_ece": recomputed,
            "n_samples": len(sub),
        }
    return result


def load_cells(cells_dir: Path, item_ids: list) -> list:
    """Load all 96 cell files, asserting the briefing's per-cell invariants."""
    files = sorted(cells_dir.glob("*.json"))
    _require(len(files) == EXPECTED_N_CELLS, f"cells: found {len(files)} files != {EXPECTED_N_CELLS}")
    cells = []
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            cell = json.load(handle)
        key = cell["cell_key"]
        _require(path.stem == key, f"cell file name {path.name} != cell_key {key}")
        _require(cell["status"] == "completed", f"{key}: status {cell['status']!r}")
        _require(cell["verdict"] == "PASS", f"{key}: verdict {cell['verdict']!r}")
        desc = cell["descriptive"]
        _require(desc["metric_split"] == "test", f"{key}: metric_split {desc['metric_split']!r}")
        _require(
            desc["n_paired_items"] == EXPECTED_N_ITEMS,
            f"{key}: n_paired_items {desc['n_paired_items']} != {EXPECTED_N_ITEMS}",
        )
        shift_keys = list(cell["index_shift_by_item"].keys())
        _require(
            shift_keys == item_ids,
            f"{key}: index_shift_by_item key order differs from bootstrap_plan.item_ids",
        )
        cells.append(cell)
    return cells


def assert_calibrator_identity(cells: list, calibration: dict) -> dict:
    """Assert per-calibrator parameter identity across its 32 cells.

    Returns the canonical ``{calibrator: phases}`` parameter map.
    """
    canonical = {}
    counts = defaultdict(int)
    for cell in cells:
        name = cell["cell"]["calibrator"]
        counts[name] += 1
        params = cell["calibrator_parameters"]
        _require(
            params["calibrator"] == name,
            f"{cell['cell_key']}: calibrator_parameters.calibrator mismatch",
        )
        blob = json.dumps(params["phases"], sort_keys=True)
        if name not in canonical:
            canonical[name] = (blob, params["phases"])
        else:
            _require(
                canonical[name][0] == blob,
                f"{cell['cell_key']}: calibrator parameters differ across "
                f"{name} cells (identity assumption violated)",
            )
    _require(
        set(counts) == set(CALIBRATORS) and all(c == 32 for c in counts.values()),
        f"calibrator cell counts {dict(counts)} != 32 each of {CALIBRATORS}",
    )
    phases_map = {name: canonical[name][1] for name in canonical}
    # Cross-check the Platt cells against the staged calibration.json.
    for phase in PHASES:
        cell_p = phases_map["platt-logistic"][phase]
        blk = calibration["per_bucket"][phase]
        _require(
            float(cell_p["a"]) == float(blk["platt_coef"])
            and float(cell_p["b"]) == float(blk["platt_intercept"]),
            f"platt {phase}: cell params ({cell_p['a']}, {cell_p['b']}) != "
            f"calibration.json ({blk['platt_coef']}, {blk['platt_intercept']})",
        )
    # Isotonic knot counts and sim-temp grid saturation.
    for phase in PHASES:
        n_knots = len(phases_map["isotonic"][phase]["x_thresholds"])
        _require(
            n_knots == EXPECTED_ISOTONIC_KNOTS[phase],
            f"isotonic {phase}: {n_knots} knots != {EXPECTED_ISOTONIC_KNOTS[phase]}",
        )
        t_val = float(phases_map["similarity-temperature"][phase]["T"])
        _require(
            t_val == EXPECTED_SIMTEMP_T,
            f"similarity-temperature {phase}: T={t_val} != {EXPECTED_SIMTEMP_T}",
        )
    return phases_map


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="frozen evidence export root (read-only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output directory for the versioned diagnostics artifact",
    )
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_root: Path = args.export_root
    out_dir: Path = args.out
    if not export_root.is_dir():
        print(f"ERROR: export root not found: {export_root}", file=sys.stderr)
        return 2

    adapter = export_root / "canonical_adapter"
    verified = export_root / "verified_export"
    paths = {
        "eval_rows": adapter / "eval_rows.jsonl.gz",
        "fit_rows": adapter / "fit_rows.jsonl.gz",
        "calibration": adapter / "calibration.json",
        "run_spec": verified / "run_spec.json",
        "bootstrap_plan": verified / "bootstrap_plan.json",
        "aggregate": verified / "aggregate.json",
        "sha256sums": verified / "SHA256SUMS",
        "receipt": export_root / "local_export_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            print(f"ERROR: missing input {name}: {path}", file=sys.stderr)
            return 2

    # --- provenance: hash every input file we read -------------------------
    input_records = []
    input_sha = {}
    for name, path in sorted(paths.items()):
        digest = sha256_file(path)
        input_sha[name] = digest
        input_records.append(
            {
                "name": name,
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "verified_against": [],
            }
        )

    with open(paths["receipt"], "r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    _require(receipt["run_id"] == RUN_ID, f"receipt run_id {receipt['run_id']!r} != {RUN_ID!r}")
    receipt_map = {kf["path"]: kf["sha256"] for kf in receipt.get("key_files", [])}
    sums_text = paths["sha256sums"].read_text(encoding="utf-8")
    sums_map = {}
    for line in sums_text.splitlines():
        line = line.strip()
        if line:
            digest, rel = line.split(None, 1)
            sums_map[rel.strip()] = digest
    for record in input_records:
        export_rel = str((REPO_ROOT / record["path"]).relative_to(export_root))
        if export_rel in receipt_map:
            _require(
                receipt_map[export_rel] == record["sha256"],
                f"{record['name']}: sha256 mismatch vs local_export_receipt.json",
            )
            record["verified_against"].append("local_export_receipt.json")
        if export_rel.startswith("verified_export/"):
            inner = export_rel[len("verified_export/"):]
            if inner in sums_map:
                _require(
                    sums_map[inner] == record["sha256"],
                    f"{record['name']}: sha256 mismatch vs SHA256SUMS",
                )
                record["verified_against"].append("SHA256SUMS")

    # Verify the 96 cell files against SHA256SUMS (binds them transitively).
    cell_entries = {k: v for k, v in sums_map.items() if k.startswith("cells/")}
    _require(
        len(cell_entries) == EXPECTED_N_CELLS,
        f"SHA256SUMS lists {len(cell_entries)} cells/ entries != {EXPECTED_N_CELLS}",
    )
    for rel, expected_digest in sorted(cell_entries.items()):
        actual = sha256_file(verified / rel)
        _require(actual == expected_digest, f"cell {rel}: sha256 mismatch vs SHA256SUMS")

    # --- load inputs -------------------------------------------------------
    eval_rows = load_jsonl_gz(paths["eval_rows"])
    fit_rows = load_jsonl_gz(paths["fit_rows"])
    with open(paths["calibration"], "r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    with open(paths["run_spec"], "r", encoding="utf-8") as handle:
        run_spec = json.load(handle)
    with open(paths["bootstrap_plan"], "r", encoding="utf-8") as handle:
        bootstrap_plan = json.load(handle)
    with open(paths["aggregate"], "r", encoding="utf-8") as handle:
        aggregate = json.load(handle)

    assert_row_file(eval_rows, EXPECTED_EVAL, "eval_rows")
    assert_row_file(fit_rows, EXPECTED_FIT, "fit_rows")
    _require(aggregate["completed"] == EXPECTED_N_CELLS, "aggregate.completed != 96")
    _require(aggregate["failed"] == 0, "aggregate.failed != 0")
    item_ids = [str(i) for i in bootstrap_plan["item_ids"]]
    _require(len(item_ids) == EXPECTED_N_ITEMS, "bootstrap_plan item_ids count != 3037")
    eval_item_set = {str(r["item_id"]) for r in eval_rows}
    _require(set(item_ids) == eval_item_set, "bootstrap_plan item_ids != eval row item set")

    # --- MANDATORY regression gate (before any new number is emitted) ------
    gate = run_regression_gate(eval_rows, calibration)

    # --- cells + calibrator identity ---------------------------------------
    cells = load_cells(verified / "cells", item_ids)
    calibrator_phases = assert_calibrator_identity(cells, calibration)

    cross_checks = []

    def check(name: str, computed, expected, hard: bool = True) -> None:
        status = "PASS" if computed == expected else "FAIL"
        cross_checks.append(
            {"name": name, "computed": computed, "expected": expected, "status": status}
        )
        if hard:
            _require(status == "PASS", f"cross-check {name}: computed={computed} expected={expected}")

    # --- (2) MC accuracy ---------------------------------------------------
    ev = rows_to_arrays(eval_rows)
    ft = rows_to_arrays(fit_rows)
    mc_ev = ev["format"] == "MC"
    mc_ft = ft["format"] == "MC"
    test_acc = accuracy_by_phase(ev["prefix_fraction"][mc_ev], ev["correct"][mc_ev])
    val_acc = accuracy_by_phase(ft["prefix_fraction"][mc_ft], ft["correct"][mc_ft])
    test_acc["n_items"] = len({i for i in ev["item_id"][mc_ev]})
    val_acc["n_items"] = len({i for i in ft["item_id"][mc_ft]})

    check("test_mc_n_correct", test_acc["n_correct"], EXPECTED_TEST_MC_CORRECT)
    check("test_mc_accuracy", test_acc["accuracy"], EXPECTED_TEST_MC_ACC)
    check("val_mc_accuracy", val_acc["accuracy"], EXPECTED_VAL_MC_ACC)
    for phase in PHASES:
        check(
            f"test_mc_phase_n_{phase}", test_acc["by_phase"][phase]["n"], EXPECTED_TEST_MC_PHASE_N[phase]
        )
        check(
            f"test_mc_phase_acc_{phase}",
            round(test_acc["by_phase"][phase]["accuracy"], 4),
            EXPECTED_TEST_PHASE_ACC_4DP[phase],
        )
        check(
            f"val_mc_phase_acc_{phase}",
            round(val_acc["by_phase"][phase]["accuracy"], 4),
            EXPECTED_VAL_PHASE_ACC_4DP[phase],
        )

    def per_prefix_idx(arrays: dict, mask: np.ndarray) -> list:
        out = []
        for idx in range(int(arrays["prefix_idx"].max()) + 1):
            sel = mask & (arrays["prefix_idx"] == idx)
            n = int(sel.sum())
            out.append(
                {
                    "prefix_idx": idx,
                    "n": n,
                    "accuracy": round(float(arrays["correct"][sel].mean()), 6) if n else None,
                }
            )
        return out

    mc_accuracy = {
        "artifact": "tae_camera_ready_diag_v1",
        "run_id": RUN_ID,
        "definitions": {
            "mc_accuracy": (
                "mean of `correct` over MC adapter rows; per adapter_build, "
                "MC correct = int(predicted_idx == gold_index)"
            ),
            "qa_note": (
                "QA-arm accuracy is NOT reportable: QA rows carry correct == 1 "
                "by construction (gold-idealized reference arm)"
            ),
            "phase_convention": (
                "early/mid/late by CHARACTER fraction of the question "
                "(prefix_char_count / full_question_char_count); "
                "early < 0.33 <= mid < 0.66 <= late"
            ),
        },
        "test": test_acc,
        "val": val_acc,
        "per_prefix_idx": {
            "note": (
                "INDEX-BASED curve over prefix_idx (0-9, variable horizons per "
                "item); this indexing does NOT match the run's "
                "character-fraction phase convention above"
            ),
            "test_mc": per_prefix_idx(ev, mc_ev),
            "val_mc": per_prefix_idx(ft, mc_ft),
        },
    }

    # --- (3) calibration diagnostics ----------------------------------------
    max_raw_sim = round(float(ev["raw_similarity"].max()), 6)
    check("max_raw_similarity_eval", max_raw_sim, EXPECTED_MAX_RAW_SIM)

    per_calibrator = {}
    attained_max = {}
    ceiling_checks_ok = True
    for name in CALIBRATORS:
        phases_out = {}
        attained_max[name] = {}
        for phase in PHASES:
            pmask = phase_mask(ev["prefix_fraction"], phase)
            params = calibrator_phases[name][phase]
            stats = {}
            for fmt in ("MC", "QA"):
                sel = pmask & (ev["format"] == fmt)
                probs = apply_calibrator(name, params, ev["raw_similarity"][sel])
                stats[fmt] = {
                    "n": int(sel.sum()),
                    "max_p": round(float(probs.max()), 6),
                    "mean_p": round(float(probs.mean()), 6),
                    "median_p": round(float(np.median(probs)), 6),
                    "p90_p": round(float(np.percentile(probs, 90)), 6),
                }
            sel_mc = pmask & mc_ev
            probs_mc = apply_calibrator(name, params, ev["raw_similarity"][sel_mc])
            mc_ece = round(ece_10bin(probs_mc, ev["correct"][sel_mc]), 6)
            if name == "platt-logistic":
                _require(
                    mc_ece == EXPECTED_PLATT_ECE[phase],
                    f"vectorized platt ECE {phase}={mc_ece} != gate value "
                    f"{EXPECTED_PLATT_ECE[phase]} (vectorized/scalar divergence)",
                )
            max_all = max(stats["MC"]["max_p"], stats["QA"]["max_p"])
            attained_max[name][phase] = {
                "MC": stats["MC"]["max_p"],
                "QA": stats["QA"]["max_p"],
                "all_formats": max_all,
            }
            if round(max_all, 4) != EXPECTED_CEILINGS_4DP[name][phase]:
                ceiling_checks_ok = False
            qa_gap = round(1.0 - stats["QA"]["mean_p"], 6)
            phases_out[phase] = {
                "mc_ece": mc_ece,
                "n_mc": int(sel_mc.sum()),
                "by_format": stats,
                "attained_max_p_all_formats": max_all,
                "level_gap_mean_p_mc_minus_qa": round(
                    stats["MC"]["mean_p"] - stats["QA"]["mean_p"], 6
                ),
                "qa_gold_idealized_gap": {
                    "value": qa_gap,
                    "definition": "1 - mean(calibrated p) over QA rows in the phase",
                    "warning": QA_GAP_WARNING,
                },
            }
        entry = {"phases": phases_out, "parameters_stored_per_cell": True}
        if name == "similarity-temperature":
            grid = [float(x) for x in run_spec["identity"]["calibration"]["similarity_temperature_grid"]]
            entry["temperature_note"] = {
                "T_by_phase": {ph: float(calibrator_phases[name][ph]["T"]) for ph in PHASES},
                "fit_grid": grid,
                "saturated_at_grid_max": all(
                    float(calibrator_phases[name][ph]["T"]) == max(grid) for ph in PHASES
                ),
                "note": (
                    "T equals the fit grid's MAXIMUM (5.0) in every phase: the "
                    "grid-constrained log-loss optimum is saturated/truncated at "
                    "the boundary, compressing calibrated probabilities toward 0.5"
                ),
            }
        if name == "isotonic":
            entry["knot_counts"] = {
                ph: len(calibrator_phases[name][ph]["x_thresholds"]) for ph in PHASES
            }
            entry["reconstruction_note"] = (
                "isotonic applied via linear interpolation on the stored "
                "(x_thresholds, y_thresholds) knots with edge clipping "
                "(np.interp + clip to [0,1]); sklearn-equivalent, sklearn not imported"
            )
        per_calibrator[name] = entry
    check("attained_ceilings_4dp_all_formats", ceiling_checks_ok, True)

    calibration_diagnostics = {
        "artifact": "tae_camera_ready_diag_v1",
        "run_id": RUN_ID,
        "method": {
            "ece_definition": (
                "the run's own 10-bin ECE (scripts/stopdff_v5/adapter_build.py): "
                "np.linspace(0,1,11) edges, last bin right-inclusive, "
                "count-weighted |mean(label) - mean(p)|"
            ),
            "phase_convention": mc_accuracy["definitions"]["phase_convention"],
            "fit_application": (
                "one shared per-phase map fit on validation MC rows, applied "
                "unchanged to both MC and QA test rows (run_spec: "
                "fit_rows=validation_mc_only, "
                "application=shared_map_applied_to_mc_and_qa)"
            ),
            "calibrator_application": {
                "platt-logistic": "p = sigmoid(a*s + b), z clipped to [-500, 500]",
                "similarity-temperature": "p = sigmoid(s / T), same clip",
                "isotonic": per_calibrator["isotonic"]["reconstruction_note"],
            },
            "split": "all statistics on TEST eval rows",
        },
        "regression_gate": {"name": "platt_ece_triple", "status": "PASS", "phases": gate},
        "max_raw_similarity_test": max_raw_sim,
        "per_calibrator": per_calibrator,
    }

    # --- (4) shift specification variability -------------------------------
    cells_sorted = sorted(cells, key=lambda c: c["cell_key"])
    shift_matrix = np.asarray(
        [[int(c["index_shift_by_item"][i]) for i in item_ids] for c in cells_sorted],
        dtype=np.int64,
    ).T  # (n_items, n_cells)
    _require(shift_matrix.shape == (EXPECTED_N_ITEMS, EXPECTED_N_CELLS), "shift matrix shape")
    obs_min, obs_max = int(shift_matrix.min()), int(shift_matrix.max())
    _require(-9 <= obs_min and obs_max <= 8, f"shift range [{obs_min},{obs_max}] outside [-9,8]")

    disp = per_item_dispersion(shift_matrix)
    dispersion_out = {
        "definitions": {
            "sd": "population SD (ddof=0) of D across the 96 cells, per item",
            "range": "max - min of D across the 96 cells, per item",
            "iqr": "p75 - p25 (numpy linear interpolation) of D across the 96 cells, per item",
        },
        "sd": dist_summary(disp["sd"]),
        "range": dist_summary(disp["rng"]),
        "iqr": dist_summary(disp["iqr"]),
        "fraction_items_D_constant_across_cells": round(
            float((disp["rng"] == 0).mean()), 6
        ),
        "fraction_items_D_zero_in_all_cells": round(
            float((np.abs(shift_matrix).max(axis=1) == 0).mean()), 6
        ),
        "fraction_items_D_nonzero_in_at_least_one_cell": round(
            float((np.abs(shift_matrix).max(axis=1) > 0).mean()), 6
        ),
    }

    per_cell = []
    signed_means = {}
    for col, cell in enumerate(cells_sorted):
        vals = shift_matrix[:, col]
        signed_mean = float(vals.mean())
        zero_count = int((vals == 0).sum())
        nb_mc = int(cell["descriptive"]["never_buzz_mc"])
        nb_qa = int(cell["descriptive"]["never_buzz_qa"])
        point = cell.get("bootstrap", {}).get("point", {})
        if "signed_index_mean" in point:
            _require(
                abs(float(point["signed_index_mean"]) - signed_mean) < 1e-6,
                f"{cell['cell_key']}: recomputed signed mean {signed_mean} != "
                f"stored bootstrap point {point['signed_index_mean']}",
            )
        bounds = mutual_never_buzz_bounds(nb_mc, nb_qa, zero_count, EXPECTED_N_ITEMS)
        signed_means[cell["cell_key"]] = signed_mean
        per_cell.append(
            {
                "cell_key": cell["cell_key"],
                "axes": cell["cell"],
                "signed_mean_shift": round(signed_mean, 6),
                "zero_fraction": round(zero_count / EXPECTED_N_ITEMS, 6),
                "zero_count": zero_count,
                "never_buzz_mc": nb_mc,
                "never_buzz_qa": nb_qa,
                "mutual_never_buzz_lower_bound": bounds["mutual_lb"],
                "mutual_never_buzz_upper_bound": bounds["mutual_ub"],
                "mutual_nb_lb_fraction_of_items": round(
                    bounds["mutual_lb"] / EXPECTED_N_ITEMS, 6
                ),
                "mutual_nb_lb_share_of_zero_mass": round(
                    bounds["mutual_lb"] / zero_count, 6
                )
                if zero_count
                else None,
                "same_finite_stop_count_bounds": bounds["same_finite_stop_bounds"],
                "same_finite_stop_fraction_bounds": [
                    round(b / EXPECTED_N_ITEMS, 6) for b in bounds["same_finite_stop_bounds"]
                ],
            }
        )

    sm_values = np.asarray([signed_means[c["cell_key"]] for c in cells_sorted])
    zf_values = np.asarray([pc["zero_fraction"] for pc in per_cell])
    min_idx, max_idx = int(sm_values.argmin()), int(sm_values.argmax())
    check("signed_mean_min", round(float(sm_values.min()), 6), EXPECTED_SIGNED_MEAN_MIN)
    check("signed_mean_min_cell", cells_sorted[min_idx]["cell_key"], EXPECTED_SIGNED_MEAN_MIN_CELL)
    check("signed_mean_max", round(float(sm_values.max()), 6), EXPECTED_SIGNED_MEAN_MAX)
    check(
        "zero_fraction_min_pct",
        round(float(zf_values.min()) * 100, 3),
        EXPECTED_ZERO_FRACTION_MIN_PCT,
    )
    check(
        "zero_fraction_max_pct",
        round(float(zf_values.max()) * 100, 3),
        EXPECTED_ZERO_FRACTION_MAX_PCT,
    )

    axis_attribution = {}
    for axis in AXES:
        levels = {}
        for cell in cells_sorted:
            levels.setdefault(cell["cell"][axis], []).append(signed_means[cell["cell_key"]])
        level_stats = {}
        for level, vals_list in sorted(levels.items()):
            arr = np.asarray(vals_list)
            level_stats[level] = {
                "n_cells": int(arr.size),
                "mean_of_cell_signed_means": round(float(arr.mean()), 6),
                "min": round(float(arr.min()), 6),
                "max": round(float(arr.max()), 6),
            }
        means = [v["mean_of_cell_signed_means"] for v in level_stats.values()]
        axis_attribution[axis] = {
            "levels": level_stats,
            "level_mean_spread": round(max(means) - min(means), 6),
        }

    # Never-buzz bound highlights.
    positive_lb = [pc for pc in per_cell if pc["mutual_never_buzz_lower_bound"] > 0]
    top_by_share = sorted(
        (pc for pc in positive_lb if pc["mutual_nb_lb_share_of_zero_mass"] is not None),
        key=lambda pc: pc["mutual_nb_lb_share_of_zero_mass"],
        reverse=True,
    )[:5]
    never_buzz_bounds = {
        "method": (
            "mutual never-buzz >= max(0, nb_mc + nb_qa - 3037) per cell "
            "(inclusion-exclusion). Never-buzz is coded stop_index == item "
            "horizon in BOTH arms over the same prefix set "
            "(scripts/stopdff_v5/policy.py), so mutual never-buzz implies "
            "D = 0 and the D = 0 mass decomposes exactly into {same finite "
            "stop} + {mutual never-buzz}; the same-finite-stop (finite "
            "agreement) fraction is therefore bounded between "
            "(zero - min(nb_mc, nb_qa, zero)) / n and (zero - LB) / n"
        ),
        "n_cells_with_positive_mutual_lb": len(positive_lb),
        "max_never_buzz_mc": int(max(pc["never_buzz_mc"] for pc in per_cell)),
        "max_never_buzz_qa": int(max(pc["never_buzz_qa"] for pc in per_cell)),
        "top_cells_by_mutual_nb_lb_share_of_zero_mass": [
            {
                "cell_key": pc["cell_key"],
                "mutual_never_buzz_lower_bound": pc["mutual_never_buzz_lower_bound"],
                "mutual_nb_lb_share_of_zero_mass": pc["mutual_nb_lb_share_of_zero_mass"],
                "zero_fraction": pc["zero_fraction"],
                "never_buzz_mc": pc["never_buzz_mc"],
                "never_buzz_qa": pc["never_buzz_qa"],
                "same_finite_stop_fraction_bounds": pc["same_finite_stop_fraction_bounds"],
            }
            for pc in top_by_share
        ],
    }

    # Reachability bound.
    schedules = run_spec["identity"]["reward_schedules"]
    p_star_by_schedule = {}
    for sched_name, sched in sorted(schedules.items()):
        c_wrong = -float(sched["wrong"])
        p_star_by_schedule[sched_name] = {
            "c_wrong": c_wrong,
            "R_early": float(sched["correct_early"]),
            "R_late": float(sched["correct_late"]),
            "split": float(sched["split"]),
            "wait_cost": float(sched["wait_cost"]),
            "p_star_early_reward": round(p_star(c_wrong, float(sched["correct_early"])), 6),
            "p_star_late_reward": round(p_star(c_wrong, float(sched["correct_late"])), 6),
            "applicable_p_star_by_phase": {
                ph: [round(v, 6) for v in applicable_pstars(ph, sched)] for ph in PHASES
            },
        }
    flags = []
    for sched_name in sorted(schedules):
        sched = schedules[sched_name]
        for name in CALIBRATORS:
            for ph in PHASES:
                stars = applicable_pstars(ph, sched)
                for fmt in ("MC", "QA"):
                    ceiling = attained_max[name][ph][fmt]
                    never_positive = ceiling < min(stars)
                    below_some = ceiling < max(stars)
                    if below_some:
                        flags.append(
                            {
                                "reward_schedule": sched_name,
                                "calibrator": name,
                                "phase": ph,
                                "format": fmt,
                                "max_attained_p": ceiling,
                                "applicable_p_star": [round(v, 6) for v in stars],
                                "never_myopically_positive": bool(never_positive),
                                "below_max_applicable_p_star_only": bool(
                                    below_some and not never_positive
                                ),
                            }
                        )
    n_never = sum(1 for f in flags if f["never_myopically_positive"])

    shift_spec_variability = {
        "artifact": "tae_camera_ready_diag_v1",
        "run_id": RUN_ID,
        "rescope_note": (
            "Per-format stopping indices are not recoverable from the frozen "
            "artifacts (cells store only the paired shift D = tau_MC - tau_QA "
            "per item); this file therefore reports cross-SPECIFICATION "
            "variability of D across the 96 registered cells, never-buzz "
            "inclusion-exclusion bounds, and a myopic reachability bound"
        ),
        "n_cells": EXPECTED_N_CELLS,
        "n_items": EXPECTED_N_ITEMS,
        "shift_value_range_observed": [obs_min, obs_max],
        "per_item_cross_cell_dispersion": dispersion_out,
        "per_axis_attribution": {
            "note": (
                "cell signed means (mean of D per cell) grouped by each "
                "specification axis level; level_mean_spread = max - min of "
                "the level means for that axis"
            ),
            "axes": axis_attribution,
        },
        "never_buzz_bounds": never_buzz_bounds,
        "zero_fraction_summary": dist_summary(zf_values),
        "signed_mean_summary": {
            **dist_summary(sm_values),
            "argmin_cell": cells_sorted[min_idx]["cell_key"],
            "argmax_cell": cells_sorted[max_idx]["cell_key"],
        },
        "reachability_bound": {
            "framing": MYOPIC_FRAMING,
            "p_star_by_schedule": p_star_by_schedule,
            "attained_max_calibrated_p": attained_max,
            "flags": flags,
            "n_flags_never_myopically_positive": n_never,
            "n_flags_total": len(flags),
        },
        "per_cell": per_cell,
    }

    # --- outputs ------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "mc_accuracy.json", mc_accuracy)
    write_json(out_dir / "calibration_diagnostics.json", calibration_diagnostics)
    write_json(out_dir / "shift_specification_variability.json", shift_spec_variability)
    tables_tex = render_tables(
        mc_accuracy, calibration_diagnostics, shift_spec_variability
    )
    (out_dir / "tables.tex").write_text(tables_tex, encoding="utf-8")

    try:
        git_rev = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "branch", "--show-current"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - git absent
        git_rev, git_branch = None, None

    output_records = []
    for fname in (
        "mc_accuracy.json",
        "calibration_diagnostics.json",
        "shift_specification_variability.json",
        "tables.tex",
    ):
        fpath = out_dir / fname
        output_records.append(
            {"path": fname, "bytes": fpath.stat().st_size, "sha256": sha256_file(fpath)}
        )

    provenance = {
        "artifact": "tae_camera_ready_diag_v1",
        "run_id": RUN_ID,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {
            "script": "scripts/tae_camera_ready_diagnostics.py",
            "git_rev": git_rev,
            "git_branch": git_branch,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "export_root": str(export_root.relative_to(REPO_ROOT))
        if export_root.is_relative_to(REPO_ROOT)
        else str(export_root),
        "inputs": input_records,
        "cells": {
            "count": EXPECTED_N_CELLS,
            "verified_against_sha256sums": True,
            "sha256sums_sha256": input_sha["sha256sums"],
        },
        "regression_gate": {
            "name": "platt_ece_triple",
            "status": "PASS",
            "definition": (
                "recompute the stored Platt ECE (early/mid/late) from test MC "
                "eval rows + stored 6dp platt coefficients using the run's "
                "10-bin ECE, rounded to 6dp; abort on any mismatch"
            ),
            "phases": gate,
        },
        "cross_checks": cross_checks,
        "method_notes": [
            "deterministic; no sampling, no RNG; numpy-only (no sklearn)",
            calibration_diagnostics["method"]["calibrator_application"]["isotonic"],
            calibration_diagnostics["method"]["ece_definition"],
            mc_accuracy["definitions"]["phase_convention"],
            never_buzz_bounds["method"],
            MYOPIC_FRAMING,
            (
                "scientific invariants untouched: all values here are post-hoc "
                "summaries of the frozen export; no registered result changes"
            ),
        ],
        "outputs": output_records,
    }
    write_json(out_dir / "provenance.json", provenance)

    gate_str = ", ".join(f"{p}={gate[p]['recomputed_ece']}" for p in PHASES)
    n_pass = sum(1 for c in cross_checks if c["status"] == "PASS")
    print(f"tae_camera_ready_diag_v1 written to {out_dir}")
    print(f"regression gate: PASS ({gate_str})")
    print(f"cross-checks: {n_pass}/{len(cross_checks)} PASS")
    return 0


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def render_tables(mc_acc: dict, calib: dict, shift: dict) -> str:
    """Render the three camera-ready booktabs tables (no preamble lines)."""
    lines = []
    lines.append("% Auto-generated by scripts/tae_camera_ready_diagnostics.py")
    lines.append("% Artifact: tae_camera_ready_diag_v1 (run final_modal_5d5328102912)")
    lines.append("% Requires the booktabs package; no preamble lines are emitted here.")
    lines.append("")

    # T1: MC accuracy.
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{MC-arm accuracy of the frozen scorer (aggregate and by "
        "question phase; phases are character-fraction terciles with "
        "boundaries 0.33/0.66). QA-arm accuracy is not reportable: QA rows "
        "are a gold-idealized reference arm with correctness fixed to 1. "
        "Artifact \\texttt{tae\\_camera\\_ready\\_diag\\_v1}.}")
    lines.append("\\label{tab:tae-mc-accuracy}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Split & Aggregate & Early & Mid & Late \\\\")
    lines.append("\\midrule")
    for split_name, block in (("Test", mc_acc["test"]), ("Val", mc_acc["val"])):
        cells = [f"{block['accuracy']:.3f} ({block['n']:,})"]
        for ph in PHASES:
            b = block["by_phase"][ph]
            cells.append(f"{b['accuracy']:.3f} ({b['n']:,})")
        lines.append(f"{split_name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # T2: calibration diagnostics.
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Calibration diagnostics per calibrator and phase (test "
        "rows): MC ECE (the run's 10-bin definition), attained calibrated-"
        "probability ceilings per format, and mean calibrated probability "
        "per format with the MC$-$QA level gap. The similarity-temperature "
        "fit saturates at the grid maximum ($T{=}5.0$ in every phase); "
        "isotonic maps are reconstructed from the stored knots (18/28/40 "
        "knots for early/mid/late) by linear interpolation with edge "
        "clipping. The QA column is a gold-idealization level, not a "
        "calibration quality measure.}")
    lines.append("\\label{tab:tae-calibration-diagnostics}")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Calibrator & Phase & ECE (MC) & $\\max \\hat p$ MC & $\\max \\hat p$ QA "
        "& $\\bar{\\hat p}$ MC & $\\bar{\\hat p}$ QA & Gap \\\\")
    lines.append("\\midrule")
    display = {
        "platt-logistic": "Platt",
        "similarity-temperature": "Sim-temp",
        "isotonic": "Isotonic",
    }
    for name in CALIBRATORS:
        block = calib["per_calibrator"][name]["phases"]
        for i, ph in enumerate(PHASES):
            b = block[ph]
            label = display[name] if i == 0 else ""
            lines.append(
                f"{label} & {ph} & {b['mc_ece']:.4f} & "
                f"{b['by_format']['MC']['max_p']:.3f} & {b['by_format']['QA']['max_p']:.3f} & "
                f"{b['by_format']['MC']['mean_p']:.3f} & {b['by_format']['QA']['mean_p']:.3f} & "
                f"{b['level_gap_mean_p_mc_minus_qa']:.3f} \\\\")
        if name != CALIBRATORS[-1]:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # T3: specification variability + never-buzz bounds + reachability.
    disp = shift["per_item_cross_cell_dispersion"]
    nb = shift["never_buzz_bounds"]
    reach = shift["reachability_bound"]
    zf = shift["zero_fraction_summary"]
    sm = shift["signed_mean_summary"]
    top = nb["top_cells_by_mutual_nb_lb_share_of_zero_mass"]
    n_never = reach["n_flags_never_myopically_positive"]
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Cross-specification variability of the paired stop-index "
        "shift $D=\\tau_{\\mathrm{MC}}-\\tau_{\\mathrm{QA}}$ across the 96 "
        "registered cells, never-buzz inclusion--exclusion bounds, and the "
        "myopic reachability bound (a diagnostic bound, not a policy "
        "recomputation). Mutual never-buzz per cell is bounded below by "
        "$\\max(0,\\,nb_{\\mathrm{MC}}+nb_{\\mathrm{QA}}-3037)$; because "
        "never-buzz is coded as stopping at the item horizon in both arms, "
        "$D{=}0$ decomposes exactly into same-finite-stop and mutual "
        "never-buzz.}")
    lines.append("\\label{tab:tae-spec-variability}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("\\multicolumn{2}{l}{\\emph{Per-item dispersion of $D$ across 96 cells} (3{,}037 items)} \\\\")
    lines.append("\\midrule")
    lines.append(
        f"SD: mean / median / p90 / max & "
        f"{disp['sd']['mean']:.3f} / {disp['sd']['median']:.3f} / "
        f"{disp['sd']['p90']:.3f} / {disp['sd']['max']:.3f} \\\\")
    lines.append(
        f"Range: mean / median / p90 / max & "
        f"{disp['range']['mean']:.3f} / {disp['range']['median']:.3f} / "
        f"{disp['range']['p90']:.3f} / {disp['range']['max']:.0f} \\\\")
    lines.append(
        f"IQR: mean / median / p90 / max & "
        f"{disp['iqr']['mean']:.3f} / {disp['iqr']['median']:.3f} / "
        f"{disp['iqr']['p90']:.3f} / {disp['iqr']['max']:.3f} \\\\")
    lines.append(
        f"Items with $D$ constant across all cells & "
        f"{100 * disp['fraction_items_D_constant_across_cells']:.1f}\\% \\\\")
    lines.append(
        f"Items with $D=0$ in every cell & "
        f"{100 * disp['fraction_items_D_zero_in_all_cells']:.1f}\\% \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{2}{l}{\\emph{Cell-level summaries} (96 cells)} \\\\")
    lines.append("\\midrule")
    lines.append(
        f"Cell signed mean of $D$: min / max & "
        f"${sm['min']:.3f}$ / ${sm['max']:.3f}$ \\\\")
    lines.append(
        f"Cell zero-shift fraction: min / max & "
        f"{100 * zf['min']:.2f}\\% / {100 * zf['max']:.2f}\\% \\\\")
    lines.append(
        f"Cells with positive mutual never-buzz bound & "
        f"{nb['n_cells_with_positive_mutual_lb']} / 96 \\\\")
    if top:
        t0 = top[0]
        lines.append(
            f"Strongest bound: mutual NB $\\geq$ {t0['mutual_never_buzz_lower_bound']:,} items & "
            f"{100 * t0['mutual_nb_lb_share_of_zero_mass']:.1f}\\% of that cell's $D{{=}}0$ mass \\\\")
        lo, hi = t0["same_finite_stop_fraction_bounds"]
        lines.append(
            f"\\quad implied same-finite-stop fraction there & "
            f"[{100 * lo:.1f}\\%, {100 * hi:.1f}\\%] \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{2}{l}{\\emph{Myopic reachability bound} ($p^* = c_\\mathrm{wrong}/(R_t + c_\\mathrm{wrong})$)} \\\\")
    lines.append("\\midrule")
    for sched_name, sp in sorted(reach["p_star_by_schedule"].items()):
        lines.append(
            f"{tex_escape(sched_name)}: $p^*$ (early / late reward) & "
            f"{sp['p_star_early_reward']:.3f} / {sp['p_star_late_reward']:.3f} \\\\")
    lines.append(
        f"(schedule, calibrator, phase, format) combos never myopically positive & "
        f"{n_never} / {4 * 3 * 3 * 2} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"REGRESSION GATE FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except AssumptionError as exc:
        print(f"BRIEFING ASSUMPTION FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
