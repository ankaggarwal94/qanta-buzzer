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

# Myopic stop threshold (calibrated probability).
# NOTE (CR-01 limitation): The 0.7 threshold is pre-registered and frozen. Given the
# fitted Platt coefficients from calibration.json, this threshold is mathematically
# unreachable for "early" and "mid" buckets because cosine similarity is bounded in
# [-1, 1] and the calibrated probability cannot exceed ~0.5 for those buckets with
# typical raw scores. This means the StopDFF metric is degenerate (ceiling effect):
# both MC and non-MC conditions always time out to the last prefix step, yielding
# zero shift for all questions. The metric is reported as "diagnostic_only" in the
# manuscript. A reachability check is computed at runtime and recorded in the output.
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
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    platt_params = {}
    for bucket_name, bucket_data in data["per_bucket"].items():
        coef = bucket_data["platt_coef"]
        intercept = bucket_data["platt_intercept"]
        if coef is None or intercept is None:
            if bucket_data.get("platt_model_type") != "constant":
                raise ValueError(
                    f"Bucket '{bucket_name}' has null Platt parameters "
                    "without platt_model_type='constant'."
                )
            probability = float(bucket_data.get("platt_constant_probability", 0.0))
            coef = 0.0
            if probability <= 0.0:
                intercept = -500.0
            elif probability >= 1.0:
                intercept = 500.0
            else:
                intercept = math.log(probability / (1.0 - probability))
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


def check_threshold_reachability(
    platt_params: dict[str, tuple[float, float]],
    threshold: float = STOP_THRESHOLD,
) -> dict[str, dict]:
    """Check whether the stop threshold is reachable for each bucket.

    Given that cosine similarity is bounded in [-1, 1], compute the maximum
    calibrated probability achievable for each bucket and determine whether
    the stop threshold can ever be exceeded.

    Parameters
    ----------
    platt_params : dict[str, tuple[float, float]]
        Per-bucket (coef, intercept) tuples from calibration.json.
    threshold : float
        The stop threshold to check reachability against.

    Returns
    -------
    dict[str, dict]
        Mapping of bucket_name -> reachability diagnostics over cosine [-1, 1].
    """
    reachability = {}
    for bucket_name, (coef, intercept) in platt_params.items():
        # The logistic mapping is monotone in coef * x + intercept. For positive
        # coefficients the max over cosine [-1, 1] is at x=1; for negative
        # coefficients it is at x=-1.
        max_raw_score = 1.0 if coef >= 0 else -1.0
        max_cal = calibrate_score(max_raw_score, coef, intercept)
        cal_at_sim_1 = calibrate_score(1.0, coef, intercept)
        cal_at_sim_neg_1 = calibrate_score(-1.0, coef, intercept)

        # Compute required raw score to reach threshold:
        # threshold = 1 / (1 + exp(-(coef * x + intercept)))
        # => coef * x + intercept = -log(1/threshold - 1)
        # => x = (-log(1/threshold - 1) - intercept) / coef
        if threshold >= 1.0 or threshold <= 0.0:
            required_raw = None
        elif coef == 0.0:
            required_raw = None
        else:
            logit = -math.log(1.0 / threshold - 1.0)
            required_raw = (logit - intercept) / coef

        reachability[bucket_name] = {
            "max_calibrated_probability": round(max_cal, 6),
            "max_calibrated_raw_score": max_raw_score,
            "calibrated_at_sim_1": round(cal_at_sim_1, 6),
            "calibrated_at_sim_neg_1": round(cal_at_sim_neg_1, 6),
            # Backward-compatible alias retained for older readers; no longer
            # used as the max when coef < 0.
            "max_calibrated_at_sim_1": round(cal_at_sim_1, 6),
            "threshold_reachable": max_cal > threshold,
            "required_raw_score": round(required_raw, 6) if required_raw is not None else None,
        }

    return reachability


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
        "--calibration",
        type=str,
        default=str(CALIBRATION_JSON),
        help="Path to calibration JSON from compute_prefix_calibration.py (default: paper_exports/calibration.json)",
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
    parser.add_argument(
        "--min-mc-coverage",
        type=float,
        default=0.98,
        help=(
            "Minimum fraction of test-split qids that must have a matching "
            "MCQuestion in mc_dataset.json (default: 0.98). When coverage is "
            "below this fraction the script refuses to run, because StopDFF "
            "would be evaluated on a non-random subset (PR #14 Blocker 3). "
            "Pass --allow-incomplete-mc-coverage to override."
        ),
    )
    parser.add_argument(
        "--min-mc-retention",
        type=float,
        default=None,
        help=(
            "Minimum raw-test retention rate required by build_metadata.json. "
            "Defaults to build_metadata.retention_thresholds.full (or .smoke "
            "in --smoke mode) when present, otherwise 0.98."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-mc-coverage",
        action="store_true",
        help=(
            "Override the --min-mc-coverage gate. Use only when downstream "
            "interpretation accounts for the non-random subset."
        ),
    )
    parser.add_argument(
        "--allow-low-mc-retention",
        action="store_true",
        help=(
            "Override the build_metadata.json raw-test retention gate. Use "
            "only when reporting StopDFF explicitly as a retained-MC-test-"
            "subset metric."
        ),
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
    # Capture argv BEFORE _parse_args mutates it so generation provenance
    # records the exact effective invocation.
    effective_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = _parse_args(argv)

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    report_output_path = Path(args.report_output)
    calibration_path = Path(args.calibration)

    # Validate data directory exists
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[STOP] Dry run -- data_dir={data_dir}")
        print(f"[STOP] Output would be written to: {output_path}")
        print(f"[STOP] Report would be written to: {report_output_path}")
        print(f"[STOP] Calibration JSON: {calibration_path}")
        # Validate required data files exist
        required = ["mc_dataset.json", "test_dataset.json"]
        for fname in required:
            fpath = data_dir / fname
            exists = fpath.exists()
            print(f"[STOP]   {fname}: {'FOUND' if exists else 'MISSING'}")
            if not exists:
                return 1
        print(f"[STOP] Calibration JSON: {'FOUND' if calibration_path.exists() else 'MISSING'}")
        if not calibration_path.exists():
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

    with open(mc_path, "r", encoding="utf-8") as f:
        mc_questions = json.load(f)

    test_path = data_dir / "test_dataset.json"
    if not test_path.exists():
        print(f"ERROR: Test dataset not found: {test_path}", file=sys.stderr)
        return 1

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Iter2 IN-01: accept both on-disk shapes for test_dataset.json.
    # See scripts/_common.iter_split_questions for the rationale; the
    # producer last-write determines whether the file is wrapped or
    # a plain list and this consumer must handle both, matching the
    # WR-05 fix already applied in compute_csli.py.
    from scripts._common import iter_split_questions

    try:
        test_questions_iter = iter_split_questions(
            test_data, source_path=test_path
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Extract test qid set
    test_qids = set(str(q["qid"]) for q in test_questions_iter)

    # PR #14 Blocker 3: shared coverage + retention gates, mirroring
    # the CSLI script's defaults. StopDFF is evaluated only on the
    # test split, so only the test qid set is gated.
    from scripts._audit_gates import (
        build_coverage_metadata,
        build_retention_metadata,
        filter_mc_questions_to_split,
        load_mc_build_metadata,
    )

    mc_test, test_coverage = filter_mc_questions_to_split(mc_questions, test_qids)

    print(
        f"[STOP] MC total: {len(mc_questions)}, MC test: {len(mc_test)} "
        f"({test_coverage['coverage_rate']:.1%} of test qids)",
        flush=True,
    )

    if len(mc_test) == 0:
        print("ERROR: No test-split MC questions found after filtering. "
              "Check that mc_dataset.json and test_dataset.json have overlapping qids.",
              file=sys.stderr)
        return 1

    if (
        test_coverage["coverage_rate"] < args.min_mc_coverage
        and not args.allow_incomplete_mc_coverage
    ):
        print(
            f"ERROR: PR-14-B3 violation: MC test coverage is "
            f"{test_coverage['coverage_rate']:.1%} "
            f"(threshold: {args.min_mc_coverage:.1%}). StopDFF would be "
            f"evaluated on a non-random subset selected against 'hard to "
            f"find distractors for'. Pass --allow-incomplete-mc-coverage "
            f"to override.",
            file=sys.stderr,
        )
        return 1

    try:
        build_metadata = load_mc_build_metadata(data_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    test_retention_meta = build_retention_metadata(
        build_metadata,
        split="test",
        smoke=args.smoke,
        explicit_threshold=args.min_mc_retention,
        override=args.allow_low_mc_retention,
    )
    if (
        test_retention_meta["applies"]
        and test_retention_meta["passed"] is False
        and not args.allow_low_mc_retention
    ):
        print(
            f"ERROR: PR-14-B3 violation: raw-test MC retention is "
            f"{test_retention_meta['retention_rate']:.1%} (threshold: "
            f"{test_retention_meta['threshold']:.1%}). StopDFF would be "
            f"evaluated on the retained MC subset. Pass "
            f"--allow-low-mc-retention only if the artifact/report "
            f"explicitly qualifies the result as retained-subset StopDFF.",
            file=sys.stderr,
        )
        return 1

    if args.smoke:
        mc_test = mc_test[:20]
        print(f"[STOP] Smoke mode: trimmed to {len(mc_test)} test questions", flush=True)

    # PR #14 follow-up review (Issue C): enforce K=4 at runtime in stopdff.
    # The MC condition (compute_stop_step_mc) takes max cosine similarity
    # over K=4 option embeddings, and Platt calibration loaded below is fit
    # on the K=4 raw-score distribution. If any question has a different K,
    # the stop-step decisions are derived from a misaligned calibrator and
    # the resulting median absolute prefix shift is scientifically invalid.
    # The non-MC condition (compute_stop_step_nonmc) is K-independent, so the
    # guard applies only to the MC iteration.
    K = 4
    bad_k = [
        (q.get("qid"), len(q.get("options") or []))
        for q in mc_test
        if len(q.get("options") or []) != K
    ]
    if bad_k:
        first_qid, first_count = bad_k[0]
        print(
            f"ERROR: StopDFF assumes K={K} options per MC question, but "
            f"{len(bad_k)} test-split questions have a different K "
            f"(first: qid={first_qid}, K={first_count}). The MC condition "
            f"takes max similarity over {first_count} options against Platt "
            f"coefficients fit on K={K}; the resulting stop steps would be "
            f"misaligned. Rebuild the MC dataset so every retained question "
            f"has exactly K options.",
            file=sys.stderr,
        )
        return 1

    # ========================================================================
    # Load Platt coefficients from Phase 5 calibration
    # ========================================================================
    print(f"[STOP] Loading Platt coefficients from {calibration_path}...", flush=True)

    if not calibration_path.exists():
        print(f"ERROR: Calibration JSON not found: {calibration_path}", file=sys.stderr)
        return 1

    platt_params = load_platt_coefficients(calibration_path)
    for bucket_name, (coef, intercept) in platt_params.items():
        print(f"[STOP]   {bucket_name}: coef={coef:.6f}, intercept={intercept:.6f}", flush=True)

    # ========================================================================
    # Reachability check (CR-01): report which buckets can reach the threshold
    # ========================================================================
    reachability = check_threshold_reachability(platt_params, STOP_THRESHOLD)
    unreachable_buckets = [b for b, r in reachability.items() if not r["threshold_reachable"]]

    if unreachable_buckets:
        print(f"[STOP] WARNING: Stop threshold {STOP_THRESHOLD} is UNREACHABLE for buckets: "
              f"{unreachable_buckets}", flush=True)
        for b in unreachable_buckets:
            r = reachability[b]
            print(
                f"[STOP]   {b}: max calibrated prob over cosine [-1, 1] "
                f"is {r['max_calibrated_probability']:.4f} at "
                f"raw_score={r['max_calibrated_raw_score']} (< {STOP_THRESHOLD}), "
                f"requires raw_score={r['required_raw_score']}",
                flush=True,
            )
        print("[STOP]   This is a known limitation (pre-registered threshold). "
              "The metric may show ceiling effects.", flush=True)
    else:
        print(f"[STOP] Threshold {STOP_THRESHOLD} is reachable for all buckets.", flush=True)

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

    print("\n[STOP] Panel statistics:", flush=True)
    print(f"[STOP]   Median |shift|: {median_abs_prefix_shift:.4f}", flush=True)
    print(f"[STOP]   Mean |shift|: {mean_abs_prefix_shift:.4f}", flush=True)
    print(f"[STOP]   P25/P50/P75/Max: {p25:.1f}/{p50:.1f}/{p75:.1f}/{max_shift:.1f}", flush=True)
    print(f"[STOP]   MC stops earlier: {mc_earlier} ({100*mc_earlier/len(mc_test):.1f}%)", flush=True)
    print(f"[STOP]   non-MC stops earlier: {nonmc_earlier} ({100*nonmc_earlier/len(mc_test):.1f}%)", flush=True)
    print(f"[STOP]   Same stop step: {same_step} ({100*same_step/len(mc_test):.1f}%)", flush=True)

    # ========================================================================
    # Detect ceiling effect (CR-01): all questions timed out to last prefix
    # ========================================================================
    # Ceiling effect = no question in either condition stopped before the last step
    all_mc_maxed = all(
        mc_step == len(q["cumulative_prefixes"]) - 1
        for mc_step, q in zip(mc_stop_steps, mc_test)
    )
    all_nonmc_maxed = all(
        ns == len(q["cumulative_prefixes"]) - 1
        for ns, q in zip(nonmc_stop_steps, mc_test)
    )
    ceiling_effect_detected = all_mc_maxed and all_nonmc_maxed

    if ceiling_effect_detected:
        print("[STOP] CEILING EFFECT DETECTED: No question in either condition stopped "
              "before the final prefix step. The metric is degenerate at this threshold.",
              flush=True)

    # Identify any bucket whose calibrated stop threshold is unreachable
    # (Platt-mapped probability over [-1, 1] cosine cannot cross STOP_THRESHOLD).
    unreachable_buckets = sorted(
        bucket
        for bucket, info in reachability.items()
        if isinstance(info, dict) and info.get("threshold_reachable") is False
    )
    if unreachable_buckets:
        print(
            f"[STOP] UNREACHABLE BUCKETS: {unreachable_buckets} -- calibrated "
            "stop threshold cannot be crossed by any cosine in [-1, 1].",
            flush=True,
        )

    # ========================================================================
    # Gate verdict
    # ========================================================================
    from scripts.threshold_manifest import (
        load_frozen_threshold_manifest,
        threshold_value,
    )

    manifest = load_frozen_threshold_manifest(THRESHOLD_MANIFEST, strict=True)
    # WR-02: fail closed (no silent fallback to hardcoded 1).
    threshold = int(threshold_value(manifest, "stopdff_median_abs_prefix"))

    threshold_only_verdict = (
        "pass" if median_abs_prefix_shift <= threshold else "warn"
    )

    # PR #14 follow-up review (Blocker 1): the producer now emits the
    # final scientific verdict, not just the threshold check. When the
    # ceiling effect fires or any bucket's calibrated threshold is
    # unreachable, the metric has no power to detect prefix shifts even
    # in principle, so a threshold-only PASS is scientifically
    # misleading. Downgrade to ``"warn"`` and record the reason; the
    # downstream audit card surfaces the qualifier in the verdict cell
    # and consequently the overall verdict.
    if ceiling_effect_detected or unreachable_buckets:
        gate_verdict = "warn"
        gate_verdict_reason = "diagnostic_null: " + ", ".join(
            filter(
                None,
                [
                    "ceiling_effect" if ceiling_effect_detected else "",
                    (
                        f"unreachable_buckets={unreachable_buckets}"
                        if unreachable_buckets
                        else ""
                    ),
                ],
            )
        )
    else:
        gate_verdict = threshold_only_verdict
        gate_verdict_reason = "threshold_only"

    print(
        f"\n[STOP] Gate verdict: {gate_verdict} "
        f"(median={median_abs_prefix_shift:.4f} vs threshold={threshold}; "
        f"reason={gate_verdict_reason})",
        flush=True,
    )

    # ========================================================================
    # Write paper_exports/stopdff.json
    # ========================================================================
    test_coverage_metadata = build_coverage_metadata(
        test_coverage,
        threshold=args.min_mc_coverage,
        override=args.allow_incomplete_mc_coverage,
    )
    test_coverage_metadata["split"] = "test"

    from scripts._common import build_generation_provenance

    generation = build_generation_provenance(
        __file__,
        effective_argv,
        output_path=output_path,
        extra_paths=[
            THRESHOLD_MANIFEST,
            data_dir / "build_metadata.json",
            calibration_path,
        ],
    )

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
        "ceiling_effect_detected": ceiling_effect_detected,
        "unreachable_buckets": unreachable_buckets,
        "reachability": reachability,
        "gate_verdict": gate_verdict,
        "gate_verdict_reason": gate_verdict_reason,
        "threshold_only_verdict": threshold_only_verdict,
        "threshold": threshold,
        "mc_coverage": {"test": test_coverage_metadata},
        "mc_retention_gate": {"test": test_retention_meta},
        "mc_build_metadata": {
            "status": build_metadata["status"],
            "source_path": build_metadata["source_path"],
            "source_sha256": build_metadata["source_sha256"],
        },
        "metadata": {
            "seed": SEED,
            "n_test": len(mc_test),
            "model": "all-MiniLM-L6-v2",
            "stop_threshold": STOP_THRESHOLD,
            "metric_type": "diagnostic_only",
            "stopping_policy": "myopic_threshold",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
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

    with open(report_output_path, "w", encoding="utf-8") as f:
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
