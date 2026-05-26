#!/usr/bin/env python3
"""
Compute Stopping-Decision Fairness (StopDFF) diagnostic metric.

DIAGNOSTIC ONLY: This metric is NOT an ex-ante deployable decision policy.
It uses a myopic-threshold stopping rule (stop when calibrated confidence > 0.7)
to measure whether MC reformulation shifts the stop-step compared to the non-MC
open-ended condition. A trained continuation model (e.g., RL-based optimal
stopping policy) is deferred to future work. The myopic-threshold is diagnostic
only and does not represent how an actual quiz-bowl buzzer system would operate.

Purpose:
    Third and final scientific metric for the pilot benchmark translation audit.
    Determines if providing explicit answer choices causes a model to commit earlier
    or later than it would without choices visible.

Algorithm:
    For each test-split MC question, at each cumulative prefix step:
      MC condition: SBERT cosine similarity between prefix embedding and each of
        the 4 options; max similarity = raw confidence; apply Platt scaling;
        stop_step = first prefix where calibrated confidence > 0.7
      non-MC condition: SBERT cosine similarity between prefix embedding and the
        correct answer text (answer_primary); that single similarity = raw confidence;
        apply same Platt scaling; stop_step = first prefix where calibrated
        confidence > 0.7

    StopDFF per question = |stop_step_MC - stop_step_nonMC| (absolute prefix-index shift)
    Panel-level = median of all absolute shifts across test-split questions

Usage:
    python scripts/compute_stopdff.py --help
    python scripts/compute_stopdff.py --smoke      # 20 test questions
    python scripts/compute_stopdff.py --dry-run    # Parse args, no compute
    python scripts/compute_stopdff.py              # Full run

Inputs:
    data/processed/mc_dataset.json     (MC questions with options, gold_index, cumulative_prefixes)
    data/processed/test_dataset.json   (test split for qid filtering)
    paper_exports/calibration.json     (per-bucket Platt coef/intercept from Phase 5)
    threshold_manifest.json            (frozen stopdff_median_abs_prefix threshold = 1)

Outputs:
    paper_exports/stopdff.json         (metric results for Phase 7 audit card)
    stopdff_report.json                (full report with v10 section 13 attestation)

Exit codes:
    0 = success
    1 = runtime error
    2 = argument error
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_exports" / "stopdff.json"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "stopdff_report.json"
THRESHOLD_MANIFEST = PROJECT_ROOT / "threshold_manifest.json"
CALIBRATION_JSON = PROJECT_ROOT / "paper_exports" / "calibration.json"

# Lazy-loaded model cache
_SBERT_MODEL = None

# Reproducibility seed
SEED = 789685

# Myopic stop threshold (calibrated probability)
STOP_THRESHOLD = 0.7

# Provenance constants from SPLIT_PROVENANCE.md (hardcoded per T-06-03 mitigation)
FRESH_SPLIT_SEED = 789685
FRESH_SPLIT_COMMIT_SHA = "a43be19733509c52e0820c44e322044b4304dc18"
THRESHOLD_FREEZE_TIMESTAMP = "2026-05-26T07:53:23Z"


def _get_sbert_model():
    """Lazy-load SBERT model (all-MiniLM-L6-v2).

    Returns
    -------
    SentenceTransformer
        Loaded sentence transformer model.
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print("[STOP] Loading SBERT model (all-MiniLM-L6-v2)...", flush=True)
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def load_platt_coefficients(calibration_path: Path) -> dict[str, tuple[float, float]]:
    """Load per-bucket Platt scaling coefficients from calibration.json.

    Parameters
    ----------
    calibration_path : Path
        Path to paper_exports/calibration.json (produced by Phase 5).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping of bucket_name -> (coef, intercept) tuples.
    """
    with open(calibration_path, "r") as f:
        data = json.load(f)

    platt_params = {}
    for bucket_name, bucket_data in data["per_bucket"].items():
        coef = bucket_data["platt_coef"]
        intercept = bucket_data["platt_intercept"]
        platt_params[bucket_name] = (coef, intercept)

    return platt_params


def calibrate_score(raw_score: float, coef: float, intercept: float) -> float:
    """Apply Platt scaling logistic transform to a raw score.

    Replicates sklearn LogisticRegression.predict_proba without needing
    sklearn at inference time:  P(correct) = 1 / (1 + exp(-(coef * x + intercept)))

    Parameters
    ----------
    raw_score : float
        Raw cosine similarity score.
    coef : float
        Platt scaling coefficient (logistic regression weight).
    intercept : float
        Platt scaling intercept (logistic regression bias).

    Returns
    -------
    float
        Calibrated probability in [0, 1].
    """
    z = coef * raw_score + intercept
    # Clamp to avoid overflow
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def assign_bucket(frac: float) -> str:
    """Assign a prefix fraction to early/mid/late bucket.

    Parameters
    ----------
    frac : float
        Prefix fraction (prefix_char_length / full_question_char_length).

    Returns
    -------
    str
        Bucket name: "early" if frac < 0.33, "mid" if frac < 0.66, "late" otherwise.
    """
    if frac < 0.33:
        return "early"
    elif frac < 0.66:
        return "mid"
    else:
        return "late"


def compute_stop_step_mc(
    question: dict,
    model,
    platt_params: dict[str, tuple[float, float]],
) -> int:
    """Compute myopic-threshold stop-step for MC condition.

    At each cumulative prefix, compute SBERT cosine similarity between prefix
    embedding and each of the 4 option embeddings; take max similarity as raw
    confidence; apply Platt calibration; stop at first prefix where calibrated
    confidence exceeds STOP_THRESHOLD (0.7).

    Parameters
    ----------
    question : dict
        MC question with 'cumulative_prefixes', 'options', 'question' fields.
    model : SentenceTransformer
        Loaded SBERT model.
    platt_params : dict[str, tuple[float, float]]
        Per-bucket (coef, intercept) from calibration.json.

    Returns
    -------
    int
        0-based prefix index where threshold first exceeded, or len(prefixes)-1
        if never exceeded.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    prefixes = question["cumulative_prefixes"]
    options = question["options"]
    full_len = len(question["question"])

    if full_len == 0 or len(prefixes) == 0:
        return 0

    # Encode all options once (shared across prefixes)
    option_embs = model.encode(options, convert_to_numpy=True)

    for idx, prefix in enumerate(prefixes):
        prefix_fraction = len(prefix) / full_len
        bucket = assign_bucket(prefix_fraction)
        coef, intercept = platt_params[bucket]

        # Encode prefix
        prefix_emb = model.encode([prefix], convert_to_numpy=True)

        # Cosine similarity between prefix and each option
        sims = cosine_similarity(prefix_emb, option_embs)[0]
        max_sim = float(np.max(sims))

        # Apply Platt calibration
        calibrated_conf = calibrate_score(max_sim, coef, intercept)

        if calibrated_conf > STOP_THRESHOLD:
            return idx

    # Never exceeded threshold -- return last prefix index
    return len(prefixes) - 1


def compute_stop_step_nonmc(
    question: dict,
    model,
    platt_params: dict[str, tuple[float, float]],
) -> int:
    """Compute myopic-threshold stop-step for non-MC condition.

    At each cumulative prefix, compute SBERT cosine similarity between prefix
    embedding and the correct answer text (answer_primary); that single similarity
    is raw confidence; apply Platt calibration; stop at first prefix where
    calibrated confidence exceeds STOP_THRESHOLD (0.7).

    The non-MC condition means the model still processes prefixes but WITHOUT
    seeing explicit choices. The confidence signal comes from embedding similarity
    to the known correct answer text (simulating "the model knows what it's looking
    for but doesn't have a menu to pick from").

    Parameters
    ----------
    question : dict
        MC question with 'cumulative_prefixes', 'answer_primary', 'question' fields.
    model : SentenceTransformer
        Loaded SBERT model.
    platt_params : dict[str, tuple[float, float]]
        Per-bucket (coef, intercept) from calibration.json.

    Returns
    -------
    int
        0-based prefix index where threshold first exceeded, or len(prefixes)-1
        if never exceeded.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    prefixes = question["cumulative_prefixes"]
    answer_text = question["answer_primary"]
    full_len = len(question["question"])

    if full_len == 0 or len(prefixes) == 0:
        return 0

    # Encode answer_primary ONCE (shared across prefixes)
    answer_emb = model.encode([answer_text], convert_to_numpy=True)

    for idx, prefix in enumerate(prefixes):
        prefix_fraction = len(prefix) / full_len
        bucket = assign_bucket(prefix_fraction)
        coef, intercept = platt_params[bucket]

        # Encode prefix
        prefix_emb = model.encode([prefix], convert_to_numpy=True)

        # Cosine similarity between prefix and answer text
        sim = cosine_similarity(prefix_emb, answer_emb)[0][0]
        raw_conf = float(sim)

        # Apply Platt calibration
        calibrated_conf = calibrate_score(raw_conf, coef, intercept)

        if calibrated_conf > STOP_THRESHOLD:
            return idx

    # Never exceeded threshold -- return last prefix index
    return len(prefixes) - 1


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Optional[list[str]]
        Argument list (defaults to sys.argv[1:]).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute StopDFF diagnostic metric (myopic-threshold stopping fairness)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing processed datasets (default: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output path for StopDFF results JSON (default: paper_exports/stopdff.json)",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default=str(DEFAULT_REPORT_OUTPUT),
        help="Output path for full report with attestation (default: stopdff_report.json)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick mode: 20 test questions (for pipeline testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and validate paths, but do not run compute",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for StopDFF diagnostic computation.

    Parameters
    ----------
    argv : Optional[list[str]]
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code (0 = success, 1 = error, 2 = argument error).
    """
    args = _parse_args(argv)

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    report_output_path = Path(args.report_output)

    # Validate data directory exists
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[STOP] Dry run -- data_dir={data_dir}")
        print(f"[STOP] Output would be written to: {output_path}")
        print(f"[STOP] Report would be written to: {report_output_path}")
        # Validate required data files exist
        required = ["mc_dataset.json", "test_dataset.json"]
        for fname in required:
            fpath = data_dir / fname
            exists = fpath.exists()
            print(f"[STOP]   {fname}: {'FOUND' if exists else 'MISSING'}")
            if not exists:
                return 1
        print(f"[STOP] Calibration JSON: {'FOUND' if CALIBRATION_JSON.exists() else 'MISSING'}")
        if not CALIBRATION_JSON.exists():
            return 1
        print(f"[STOP] Threshold manifest: {'FOUND' if THRESHOLD_MANIFEST.exists() else 'MISSING'}")
        print(f"[STOP] STOP_THRESHOLD: {STOP_THRESHOLD}")
        print(f"[STOP] SEED: {SEED}")
        return 0

    # ========================================================================
    # Load data
    # ========================================================================
    print("[STOP] Loading datasets...", flush=True)

    mc_path = data_dir / "mc_dataset.json"
    if not mc_path.exists():
        print(f"ERROR: MC dataset not found: {mc_path}", file=sys.stderr)
        return 1

    with open(mc_path, "r") as f:
        mc_questions = json.load(f)

    test_path = data_dir / "test_dataset.json"
    if not test_path.exists():
        print(f"ERROR: Test dataset not found: {test_path}", file=sys.stderr)
        return 1

    with open(test_path, "r") as f:
        test_data = json.load(f)

    # Extract test qid set
    test_qids = set(str(q["qid"]) for q in test_data["questions"])

    # Filter MC questions to test split
    mc_test = [q for q in mc_questions if str(q["qid"]) in test_qids]

    print(f"[STOP] MC total: {len(mc_questions)}, MC test: {len(mc_test)}", flush=True)

    if args.smoke:
        mc_test = mc_test[:20]
        print(f"[STOP] Smoke mode: trimmed to {len(mc_test)} test questions", flush=True)

    # ========================================================================
    # Load Platt coefficients from Phase 5 calibration
    # ========================================================================
    print("[STOP] Loading Platt coefficients from calibration.json...", flush=True)

    if not CALIBRATION_JSON.exists():
        print(f"ERROR: Calibration JSON not found: {CALIBRATION_JSON}", file=sys.stderr)
        return 1

    platt_params = load_platt_coefficients(CALIBRATION_JSON)
    for bucket_name, (coef, intercept) in platt_params.items():
        print(f"[STOP]   {bucket_name}: coef={coef:.6f}, intercept={intercept:.6f}", flush=True)

    # ========================================================================
    # Compute stop steps for MC and non-MC conditions
    # ========================================================================
    model = _get_sbert_model()

    print(f"[STOP] Computing stop steps for {len(mc_test)} test questions...", flush=True)
    print(f"[STOP] Stop threshold: {STOP_THRESHOLD}", flush=True)

    abs_shifts = []
    mc_stop_steps = []
    nonmc_stop_steps = []

    for i, q in enumerate(mc_test):
        # MC condition: max similarity to options
        mc_step = compute_stop_step_mc(q, model, platt_params)
        # non-MC condition: similarity to answer_primary
        nonmc_step = compute_stop_step_nonmc(q, model, platt_params)

        abs_shift = abs(mc_step - nonmc_step)
        abs_shifts.append(abs_shift)
        mc_stop_steps.append(mc_step)
        nonmc_stop_steps.append(nonmc_step)

        if (i + 1) % 200 == 0:
            print(f"[STOP]   Processed {i+1}/{len(mc_test)} questions "
                  f"(running median shift: {float(np.median(abs_shifts)):.2f})", flush=True)

    # ========================================================================
    # Compute panel-level statistics
    # ========================================================================
    abs_shifts_arr = np.array(abs_shifts, dtype=np.float64)

    median_abs_prefix_shift = float(np.median(abs_shifts_arr))
    mean_abs_prefix_shift = float(np.mean(abs_shifts_arr))
    p25 = float(np.percentile(abs_shifts_arr, 25))
    p50 = float(np.percentile(abs_shifts_arr, 50))
    p75 = float(np.percentile(abs_shifts_arr, 75))
    max_shift = float(np.max(abs_shifts_arr))

    # Also compute direction breakdown
    mc_earlier = sum(1 for ms, ns in zip(mc_stop_steps, nonmc_stop_steps) if ms < ns)
    nonmc_earlier = sum(1 for ms, ns in zip(mc_stop_steps, nonmc_stop_steps) if ns < ms)
    same_step = sum(1 for ms, ns in zip(mc_stop_steps, nonmc_stop_steps) if ms == ns)

    print(f"\n[STOP] Panel statistics:", flush=True)
    print(f"[STOP]   Median |shift|: {median_abs_prefix_shift:.4f}", flush=True)
    print(f"[STOP]   Mean |shift|: {mean_abs_prefix_shift:.4f}", flush=True)
    print(f"[STOP]   P25/P50/P75/Max: {p25:.1f}/{p50:.1f}/{p75:.1f}/{max_shift:.1f}", flush=True)
    print(f"[STOP]   MC stops earlier: {mc_earlier} ({100*mc_earlier/len(mc_test):.1f}%)", flush=True)
    print(f"[STOP]   non-MC stops earlier: {nonmc_earlier} ({100*nonmc_earlier/len(mc_test):.1f}%)", flush=True)
    print(f"[STOP]   Same stop step: {same_step} ({100*same_step/len(mc_test):.1f}%)", flush=True)

    # ========================================================================
    # Gate verdict
    # ========================================================================
    threshold = 1  # default
    if THRESHOLD_MANIFEST.exists():
        with open(THRESHOLD_MANIFEST, "r") as f:
            manifest = json.load(f)
        for t in manifest.get("thresholds", []):
            if t["metric"] == "stopdff_median_abs_prefix":
                threshold = int(t["threshold"])
                break

    gate_verdict = "pass" if median_abs_prefix_shift <= threshold else "warn"
    print(f"\n[STOP] Gate verdict: {gate_verdict} "
          f"(median={median_abs_prefix_shift:.4f} vs threshold={threshold})", flush=True)

    # ========================================================================
    # Write paper_exports/stopdff.json
    # ========================================================================
    output_data = {
        "median_abs_prefix_shift": round(median_abs_prefix_shift, 6),
        "mean_abs_prefix_shift": round(mean_abs_prefix_shift, 6),
        "per_question_distribution_summary": {
            "p25": round(p25, 4),
            "p50": round(p50, 4),
            "p75": round(p75, 4),
            "max": round(max_shift, 4),
        },
        "direction_breakdown": {
            "mc_stops_earlier": mc_earlier,
            "nonmc_stops_earlier": nonmc_earlier,
            "same_step": same_step,
        },
        "gate_verdict": gate_verdict,
        "threshold": threshold,
        "metadata": {
            "seed": SEED,
            "n_test": len(mc_test),
            "model": "all-MiniLM-L6-v2",
            "stop_threshold": STOP_THRESHOLD,
            "metric_type": "diagnostic_only",
            "stopping_policy": "myopic_threshold",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[STOP] Results written to: {output_path}", flush=True)

    # ========================================================================
    # Write stopdff_report.json with attestation header
    # ========================================================================
    owner_attestation = {
        "attested_by": "Ankit Aggarwal",
        "attested_at": THRESHOLD_FREEZE_TIMESTAMP,
        "fresh_split_seed": FRESH_SPLIT_SEED,
        "fresh_split_commit_sha": FRESH_SPLIT_COMMIT_SHA,
        "test_split_inspected_for_tuning": False,
        "thresholds_frozen_before_any_test_inspection": True,
        "calibration_fit_split": "validation",
        "platt_scaling_fit_on_val_applied_to_test": True,
        "stopdff_information_set": "myopic_threshold",
        "stopdff_continuation_source": "myopic_threshold",
        "stopdff_severity_in_paper": "diagnostic",
    }

    report_data = {
        "owner_attestation": owner_attestation,
        **output_data,
    }

    with open(report_output_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"[STOP] Report written to: {report_output_path}", flush=True)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("[STOP] STOPPING-DECISION FAIRNESS (StopDFF) RESULTS")
    print("[STOP] ** DIAGNOSTIC ONLY -- myopic threshold, not ex-ante DP **")
    print("=" * 60)
    print(f"  Median |prefix shift|: {median_abs_prefix_shift:.4f}")
    print(f"  Mean |prefix shift|: {mean_abs_prefix_shift:.4f}")
    print(f"  Distribution: P25={p25:.1f}, P50={p50:.1f}, P75={p75:.1f}, Max={max_shift:.1f}")
    print(f"  Direction: MC earlier {mc_earlier}, non-MC earlier {nonmc_earlier}, same {same_step}")
    print(f"  Threshold: {threshold}, Verdict: {gate_verdict}")
    print(f"  n_test: {len(mc_test)}, model: all-MiniLM-L6-v2")
    print(f"  Stop threshold: {STOP_THRESHOLD}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
