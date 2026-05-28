#!/usr/bin/env python3
"""
Compute prefix-wise calibration via Platt scaling and per-bucket ECE.

This metric assesses whether calibrated confidence scores from SBERT similarity
are reliable at different information-reveal stages (early/mid/late prefix) of
quizbowl tossups reformulated as MC questions. High ECE indicates the MC format
distorts confidence calibration at that prefix stage.

Pipeline:
  1. For each MC question, compute cosine similarity between SBERT embeddings
     of each cumulative_prefix and each option; max similarity = raw confidence.
  2. Assign each prefix to a bucket (early/mid/late) based on fractional position.
  3. Fit Platt scaling (LogisticRegression C=1.0) per bucket on VAL split.
  4. Apply Platt-calibrated probabilities to TEST split per bucket.
  5. Compute ECE per bucket (10 equal-width bins).
  6. Generate reliability diagrams per bucket.
  7. Compare max(bucket ECEs) against frozen threshold 0.10.

Usage:
    python scripts/compute_prefix_calibration.py --help
    python scripts/compute_prefix_calibration.py --smoke      # 20 val + 20 test questions
    python scripts/compute_prefix_calibration.py --dry-run    # Parse args, no compute
    python scripts/compute_prefix_calibration.py              # Full run

Inputs:
    data/processed/mc_dataset.json   (MC questions with options, gold_index, cumulative_prefixes)
    data/processed/val_dataset.json  (val split for qid filtering)
    data/processed/test_dataset.json (test split for qid filtering)
    threshold_manifest.json          (frozen prefix_ece threshold = 0.10)

Outputs:
    paper_exports/calibration.json              (per-bucket ECE + gate verdict)
    paper_exports/reliability_early.png         (reliability diagram for early bucket)
    paper_exports/reliability_mid.png           (reliability diagram for mid bucket)
    paper_exports/reliability_late.png          (reliability diagram for late bucket)

Exit codes:
    0 = success
    1 = runtime error
    2 = argument error
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_exports" / "calibration.json"
THRESHOLD_MANIFEST = PROJECT_ROOT / "threshold_manifest.json"

# Lazy-loaded model cache
_SBERT_MODEL = None

# Reproducibility seed
SEED = 789685


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
        print("[CALI] Loading SBERT model (all-MiniLM-L6-v2)...", flush=True)
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def compute_prefix_confidence(
    question: dict,
    model,
) -> list[tuple[float, float, int]]:
    """Compute confidence scores at each prefix of an MC question.

    For each cumulative prefix, encodes the prefix and all options via SBERT,
    computes cosine similarity between prefix embedding and each option embedding,
    and returns the max similarity as raw confidence along with correctness.

    Parameters
    ----------
    question : dict
        MC question dict with 'cumulative_prefixes', 'options', 'gold_index',
        'question' fields.
    model : SentenceTransformer
        Loaded SBERT model for encoding.

    Returns
    -------
    list[tuple[float, float, int]]
        List of (prefix_fraction, max_similarity_score, correct) tuples where:
        - prefix_fraction = len(prefix) / len(full_question)
        - max_similarity_score = max cosine sim across options
        - correct = 1 if argmax similarity matches gold_index, else 0
    """
    from sklearn.metrics.pairwise import cosine_similarity

    full_question = question["question"]
    prefixes = question["cumulative_prefixes"]
    options = question["options"]
    gold_index = question["gold_index"]
    full_len = len(full_question)

    if full_len == 0:
        return []

    # Encode all options once (shared across prefixes)
    option_embs = model.encode(options, convert_to_numpy=True)

    results = []
    for prefix in prefixes:
        prefix_fraction = len(prefix) / full_len

        # Encode prefix
        prefix_emb = model.encode([prefix], convert_to_numpy=True)

        # Cosine similarity between prefix and each option
        sims = cosine_similarity(prefix_emb, option_embs)[0]

        max_sim = float(np.max(sims))
        predicted_idx = int(np.argmax(sims))
        correct = 1 if predicted_idx == gold_index else 0

        results.append((prefix_fraction, max_sim, correct))

    return results


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


def fit_platt(raw_scores: np.ndarray, labels: np.ndarray):
    """Fit Platt scaling via logistic regression.

    Parameters
    ----------
    raw_scores : np.ndarray
        1-D array of raw confidence scores (cosine similarities).
    labels : np.ndarray
        1-D binary array (1 = correct prediction, 0 = incorrect).

    Returns
    -------
    LogisticRegression
        Fitted logistic regression model mapping raw scores to calibrated
        probabilities.
    """
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
    lr.fit(raw_scores.reshape(-1, 1), labels)
    return lr


class ConstantCalibrationModel:
    """Fallback calibrator for empty or single-class validation buckets."""

    def __init__(self, probability: float, reason: str) -> None:
        self.probability = float(probability)
        self.reason = reason

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        n = len(raw_scores)
        positive = np.full(n, self.probability, dtype=float)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])


def _fit_bucket_calibrator(
    bucket_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[object, dict[str, object]]:
    """Fit Platt scaling or a safe constant fallback for one bucket."""
    if len(labels) == 0:
        print(
            f"[CALI] WARNING: Bucket '{bucket_name}' is empty; using constant "
            "0.0 calibration fallback",
            flush=True,
        )
        return ConstantCalibrationModel(0.0, "empty_validation_bucket"), {
            "platt_model_type": "constant",
            "platt_fallback_reason": "empty_validation_bucket",
        }

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        probability = float(labels[0])
        label_name = "correct" if probability == 1.0 else "incorrect"
        print(
            f"[CALI] WARNING: Bucket '{bucket_name}' has only one class "
            f"(all {label_name}); using constant {probability:.1f} "
            "calibration fallback",
            flush=True,
        )
        return ConstantCalibrationModel(
            probability,
            "single_class_validation_bucket",
        ), {
            "platt_model_type": "constant",
            "platt_fallback_reason": "single_class_validation_bucket",
        }

    return fit_platt(scores, labels), {
        "platt_model_type": "logistic",
        "platt_fallback_reason": None,
    }


def _calibrate_bucket_scores(model: object, raw_scores: np.ndarray) -> np.ndarray:
    """Apply bucket calibrator, returning an empty vector for empty buckets."""
    if len(raw_scores) == 0:
        return np.array([], dtype=float)
    return model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


def compute_ece(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Uses 10 equal-width bins in [0, 1]. ECE is the weighted mean of
    |accuracy_bin - confidence_bin| weighted by bin count / total.

    Parameters
    ----------
    calibrated_probs : np.ndarray
        1-D array of calibrated probability values in [0, 1].
    labels : np.ndarray
        1-D binary array (1 = correct, 0 = incorrect).
    n_bins : int
        Number of equal-width bins (default 10).

    Returns
    -------
    float
        ECE value in [0, 1].
    """
    n = len(calibrated_probs)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (calibrated_probs >= lo) & (calibrated_probs < hi)
        else:
            # Last bin includes the right boundary
            mask = (calibrated_probs >= lo) & (calibrated_probs <= hi)

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        bin_accuracy = labels[mask].mean()
        bin_confidence = calibrated_probs[mask].mean()
        ece += (bin_count / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def plot_reliability_diagram(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    bucket_name: str,
    n_bins: int = 10,
) -> None:
    """Generate and save a reliability diagram.

    Plots (mean_predicted_confidence, mean_accuracy) per bin with a diagonal
    reference line representing perfect calibration.

    Parameters
    ----------
    calibrated_probs : np.ndarray
        1-D array of calibrated probability values in [0, 1].
    labels : np.ndarray
        1-D binary array (1 = correct, 0 = incorrect).
    output_path : Path
        File path for the saved PNG.
    bucket_name : str
        Bucket name for the plot title (e.g., "early", "mid", "late").
    n_bins : int
        Number of equal-width bins (default 10).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs = []
    ys = []
    sizes = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (calibrated_probs >= lo) & (calibrated_probs < hi)
        else:
            mask = (calibrated_probs >= lo) & (calibrated_probs <= hi)

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        mean_conf = calibrated_probs[mask].mean()
        mean_acc = labels[mask].mean()
        xs.append(mean_conf)
        ys.append(mean_acc)
        sizes.append(bin_count)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.scatter(xs, ys, color="tab:blue", s=80, zorder=5, label="Bin mean")

    # Annotate bin counts
    for x, y, s in zip(xs, ys, sizes):
        ax.annotate(str(s), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="tab:gray")

    ax.set_title(f"Reliability Diagram: {bucket_name} prefix")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Mean empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
        description="Compute prefix-wise calibration via Platt scaling and per-bucket ECE",
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
        help="Output path for calibration results JSON (default: paper_exports/calibration.json)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick mode: 20 val + 20 test questions (for pipeline testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and validate data path, but do not run compute",
    )
    parser.add_argument(
        "--min-mc-coverage",
        type=float,
        default=0.98,
        help=(
            "Minimum fraction of test-split qids that must have a matching "
            "MCQuestion in mc_dataset.json (default: 0.98). When coverage is "
            "below this fraction the script refuses to run, because "
            "calibration would be fit/evaluated on a non-random subset "
            "(PR #14 Blocker 3). Pass --allow-incomplete-mc-coverage to "
            "override."
        ),
    )
    parser.add_argument(
        "--min-mc-retention",
        type=float,
        default=None,
        help=(
            "Minimum raw-test retention rate required by build_metadata.json. "
            "Defaults to build_metadata.retention_thresholds.full (or .smoke "
            "in --smoke mode) when present, otherwise 0.98. This gate is "
            "separate from --min-mc-coverage."
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
            "only when reporting calibration explicitly as a retained-MC-"
            "test-subset metric."
        ),
    )
    parser.add_argument(
        "--fit-split",
        choices=["val", "train"],
        default="val",
        help=(
            "Split to fit Platt coefficients on. Default 'val' preserves "
            "backward-compatible calibration.json provenance; 'train' "
            "produces the train-fit calibration artifact a learned-value "
            "StopDFF trainer (Prompt 5 prerequisite) requires. When "
            "--fit-split=train, pass --output paper_exports/"
            "calibration_train.json to avoid overwriting the val-fit "
            "artifact backing the audit card."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for prefix-wise calibration computation.

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

    # When the operator opts into a train-fit calibration AND leaves
    # --output at the default, warn (do NOT auto-redirect): writing the
    # train-fit artifact over paper_exports/calibration.json would
    # invalidate the val-fit artifact that backs the audit card. The
    # operator owns the path decision; we only surface the risk.
    if args.fit_split == "train" and output_path.resolve() == DEFAULT_OUTPUT.resolve():
        print(
            "WARNING: --fit-split=train with default --output will overwrite "
            "paper_exports/calibration.json (the val-fit artifact backing "
            "the audit card). Consider --output "
            "paper_exports/calibration_train.json.",
            file=sys.stderr,
            flush=True,
        )

    # Validate data directory exists
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    fit_split_filename = f"{args.fit_split}_dataset.json"

    if args.dry_run:
        print(f"[CALI] Dry run -- data_dir={data_dir}")
        print(f"[CALI] Output would be written to: {output_path}")
        print(f"[CALI] Fit split: {args.fit_split} (file: {fit_split_filename})")
        # Validate required data files exist (only the selected fit split).
        required = ["mc_dataset.json", fit_split_filename, "test_dataset.json"]
        for fname in required:
            fpath = data_dir / fname
            exists = fpath.exists()
            print(f"[CALI]   {fname}: {'FOUND' if exists else 'MISSING'}")
            if not exists:
                return 1
        print(f"[CALI] Threshold manifest: {'FOUND' if THRESHOLD_MANIFEST.exists() else 'MISSING'}")
        return 0

    # ========================================================================
    # Load data
    # ========================================================================
    print("[CALI] Loading datasets...", flush=True)

    mc_path = data_dir / "mc_dataset.json"
    if not mc_path.exists():
        print(f"ERROR: MC dataset not found: {mc_path}", file=sys.stderr)
        return 1

    with open(mc_path, "r", encoding="utf-8") as f:
        mc_questions = json.load(f)

    # Load the configured fit split. Default --fit-split=val preserves
    # backward-compatible behavior; --fit-split=train loads the train
    # dataset so a learned-value StopDFF trainer can consume a
    # leakage-free train-fit calibration artifact (Prompt 5).
    val_path = data_dir / fit_split_filename
    if not val_path.exists():
        print(
            f"ERROR: Fit-split dataset not found: {val_path} "
            f"(--fit-split={args.fit_split})",
            file=sys.stderr,
        )
        return 1

    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    test_path = data_dir / "test_dataset.json"
    if not test_path.exists():
        print(f"ERROR: Test dataset not found: {test_path}", file=sys.stderr)
        return 1

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Iter2 IN-01: accept both on-disk shapes for val/test_dataset.json.
    # See scripts/_common.iter_split_questions for the rationale; this
    # closes the cross-consumer gap with compute_csli.py / compute_stopdff.py.
    from scripts._common import iter_split_questions

    try:
        val_questions_iter = iter_split_questions(
            val_data, source_path=val_path
        )
        test_questions_iter = iter_split_questions(
            test_data, source_path=test_path
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Extract qid sets
    val_qids = set(str(q["qid"]) for q in val_questions_iter)
    test_qids = set(str(q["qid"]) for q in test_questions_iter)

    # PR #14 Blocker 3: shared coverage + retention gates, mirroring
    # the CSLI script's defaults. Calibration is dual-evaluated
    # (Platt fit on val, ECE evaluated on test) so both qid sets are
    # gated separately. The retention gate reads the same
    # data/processed/build_metadata.json the CSLI script uses.
    from scripts._audit_gates import (
        build_coverage_metadata,
        build_retention_metadata,
        filter_mc_questions_to_split,
        load_mc_build_metadata,
    )

    mc_val, val_coverage = filter_mc_questions_to_split(mc_questions, val_qids)
    mc_test, test_coverage = filter_mc_questions_to_split(mc_questions, test_qids)

    # PR #14 follow-up review (Issue C): enforce K=4 at runtime in calibration.
    # Platt coefficients are fit on the per-bucket distribution of max cosine
    # similarity across exactly K=4 option embeddings. If the underlying MC
    # data carries a different K for any question, the calibrators are
    # silently misaligned with the raw-score distribution they are applied to.
    # Fail closed so a variable-K artifact cannot reach the audit card.
    K = 4
    for split_name, mc_split in (("val", mc_val), ("test", mc_test)):
        bad_k = [
            (q.get("qid"), len(q.get("options") or []))
            for q in mc_split
            if len(q.get("options") or []) != K
        ]
        if bad_k:
            first_qid, first_count = bad_k[0]
            print(
                f"ERROR: Calibration assumes K={K} options per MC question, but "
                f"{len(bad_k)} {split_name}-split questions have a different K "
                f"(first: qid={first_qid}, K={first_count}). Platt scaling is fit "
                f"on the K={K} raw-confidence distribution and would be misaligned. "
                f"Rebuild the MC dataset so every retained question has exactly K "
                f"options.",
                file=sys.stderr,
            )
            return 1

    print(
        f"[CALI] MC total: {len(mc_questions)}, MC val: {len(mc_val)} "
        f"({val_coverage['coverage_rate']:.1%} of val qids), "
        f"MC test: {len(mc_test)} "
        f"({test_coverage['coverage_rate']:.1%} of test qids)",
        flush=True,
    )

    val_coverage_passed = (
        val_coverage["coverage_rate"] >= args.min_mc_coverage
    )
    test_coverage_passed = (
        test_coverage["coverage_rate"] >= args.min_mc_coverage
    )
    if (
        not (val_coverage_passed and test_coverage_passed)
        and not args.allow_incomplete_mc_coverage
    ):
        print(
            f"ERROR: PR-14-B3 violation: MC coverage below "
            f"{args.min_mc_coverage:.1%} for at least one split "
            f"(val={val_coverage['coverage_rate']:.1%}, "
            f"test={test_coverage['coverage_rate']:.1%}). Calibration "
            f"would be fit/evaluated on a non-random subset. Re-run "
            f"scripts/build_mc_dataset.py against the new split, or pass "
            f"--allow-incomplete-mc-coverage to override.",
            file=sys.stderr,
        )
        return 1

    # PR #14 follow-up review (Codex #3308770805): even with
    # ``--allow-incomplete-mc-coverage``, fail closed when zero MC questions
    # remain in the TEST split after filtering. The override is intended for
    # low-but-nonzero coverage; an empty test split would produce
    # ``max_ece=0`` / ``n_test=0`` reliability diagrams against zero held-out
    # labels, which is not a usable calibration evaluation. The VAL split
    # being empty is intentionally still allowed -- the existing
    # ConstantCalibrationModel fallback handles val=0 per
    # tests/test_pr14_review_regressions.py::test_prefix_calibration_uses_constant_model_for_empty_val_bucket
    # and the smoke fixture (val=0/3 retention) explicitly exercises this.
    # Mirrors the ``compute_csli.py:1682-1692`` and StopDFF guards.
    if len(mc_test) == 0:
        print(
            "ERROR: After filtering test split, zero MC questions remain. "
            "The --allow-incomplete-mc-coverage override is for "
            "low-but-nonzero coverage; an empty test split would produce "
            "an empty calibration evaluation (max_ece=0, n_test=0, empty "
            "reliability diagrams). Rebuild the MC dataset via "
            "scripts/build_mc_dataset.py against the current split.",
            file=sys.stderr,
        )
        return 1

    try:
        build_metadata = load_mc_build_metadata(data_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    val_retention_meta = build_retention_metadata(
        build_metadata,
        split=args.fit_split,
        smoke=args.smoke,
        explicit_threshold=args.min_mc_retention,
        override=args.allow_low_mc_retention,
    )
    test_retention_meta = build_retention_metadata(
        build_metadata,
        split="test",
        smoke=args.smoke,
        explicit_threshold=args.min_mc_retention,
        override=args.allow_low_mc_retention,
    )
    for split_meta in (val_retention_meta, test_retention_meta):
        if (
            split_meta["applies"]
            and split_meta["passed"] is False
            and not args.allow_low_mc_retention
        ):
            print(
                f"ERROR: PR-14-B3 violation: raw-{split_meta['split']} MC "
                f"retention is {split_meta['retention_rate']:.1%} "
                f"(threshold: {split_meta['threshold']:.1%}). Calibration "
                f"would be fit/evaluated on the retained MC subset, not "
                f"the full raw fresh split. Pass --allow-low-mc-retention "
                f"only if the artifact/report explicitly qualifies the "
                f"result as a retained-subset metric.",
                file=sys.stderr,
            )
            return 1

    if args.smoke:
        mc_val = mc_val[:20]
        mc_test = mc_test[:20]
        print(f"[CALI] Smoke mode: trimmed to {len(mc_val)} val + {len(mc_test)} test", flush=True)

    # ========================================================================
    # Compute prefix-wise confidence scores
    # ========================================================================
    model = _get_sbert_model()

    # Process validation set
    print("[CALI] Computing prefix-wise confidence on VAL set...", flush=True)
    val_buckets: dict[str, tuple[list[float], list[int]]] = {
        "early": ([], []),
        "mid": ([], []),
        "late": ([], []),
    }

    for i, q in enumerate(mc_val):
        prefix_results = compute_prefix_confidence(q, model)
        for frac, score, correct in prefix_results:
            bucket = assign_bucket(frac)
            val_buckets[bucket][0].append(score)
            val_buckets[bucket][1].append(correct)
        if (i + 1) % 200 == 0:
            print(f"[CALI]   Val: {i+1}/{len(mc_val)}", flush=True)

    print("[CALI] Val bucket sizes: " + ", ".join(
        f"{k}={len(v[0])}" for k, v in val_buckets.items()
    ), flush=True)

    # ========================================================================
    # Fit Platt scaling per bucket on VAL
    # ========================================================================
    print("[CALI] Fitting Platt scaling per bucket on VAL...", flush=True)
    platt_models: dict[str, object] = {}

    for bucket_name in ["early", "mid", "late"]:
        scores = np.array(val_buckets[bucket_name][0])
        labels = np.array(val_buckets[bucket_name][1])

        if len(scores) < 10:
            print(f"[CALI] WARNING: Bucket '{bucket_name}' has only {len(scores)} val samples, "
                  "Platt fit may be unreliable", flush=True)

        platt_models[bucket_name], model_metadata = _fit_bucket_calibrator(
            bucket_name,
            scores,
            labels,
        )

        # WR-01: Check class balance before fitting Platt scaling
        if model_metadata["platt_model_type"] == "logistic":
            n_pos = int(labels.sum())
            n_total = len(labels)
            class_ratio = n_pos / n_total
            if class_ratio < 0.1 or class_ratio > 0.9:
                print(f"[CALI] WARNING: Bucket '{bucket_name}' has imbalanced classes "
                      f"(positive rate={class_ratio:.3f}, {n_pos}/{n_total}), "
                      "Platt scaling may produce extreme coefficients", flush=True)

        calibrator = platt_models[bucket_name]
        if model_metadata["platt_model_type"] == "logistic":
            coef = float(calibrator.coef_[0][0])
            intercept = float(calibrator.intercept_[0])
            print(f"[CALI]   {bucket_name}: coef={coef:.4f}, intercept={intercept:.4f} "
                  f"(n={len(scores)})", flush=True)
        else:
            print(
                f"[CALI]   {bucket_name}: constant={calibrator.probability:.1f} "
                f"({calibrator.reason}, n={len(scores)})",
                flush=True,
            )

    # ========================================================================
    # Apply Platt to TEST set and compute ECE
    # ========================================================================
    print("[CALI] Computing prefix-wise confidence on TEST set...", flush=True)
    test_buckets: dict[str, tuple[list[float], list[int]]] = {
        "early": ([], []),
        "mid": ([], []),
        "late": ([], []),
    }

    for i, q in enumerate(mc_test):
        prefix_results = compute_prefix_confidence(q, model)
        for frac, score, correct in prefix_results:
            bucket = assign_bucket(frac)
            test_buckets[bucket][0].append(score)
            test_buckets[bucket][1].append(correct)
        if (i + 1) % 200 == 0:
            print(f"[CALI]   Test: {i+1}/{len(mc_test)}", flush=True)

    print("[CALI] Test bucket sizes: " + ", ".join(
        f"{k}={len(v[0])}" for k, v in test_buckets.items()
    ), flush=True)

    # ========================================================================
    # Compute ECE and generate reliability diagrams per bucket
    # ========================================================================
    print("[CALI] Computing ECE and generating reliability diagrams...", flush=True)
    per_bucket_results: dict[str, dict] = {}
    output_dir = output_path.parent

    for bucket_name in ["early", "mid", "late"]:
        raw_scores = np.array(test_buckets[bucket_name][0])
        labels = np.array(test_buckets[bucket_name][1])
        lr = platt_models[bucket_name]

        # Apply Platt calibration; empty buckets produce empty calibrated arrays.
        calibrated = _calibrate_bucket_scores(lr, raw_scores)

        # Compute ECE
        ece = compute_ece(calibrated, labels, n_bins=10)

        # Generate reliability diagram
        png_path = output_dir / f"reliability_{bucket_name}.png"
        plot_reliability_diagram(calibrated, labels, png_path, bucket_name, n_bins=10)

        # Store results
        if hasattr(lr, "coef_"):
            platt_coef = round(float(lr.coef_[0][0]), 6)
            platt_intercept = round(float(lr.intercept_[0]), 6)
            model_type = "logistic"
            fallback_reason = None
            constant_probability = None
        else:
            platt_coef = None
            platt_intercept = None
            model_type = "constant"
            fallback_reason = lr.reason
            constant_probability = lr.probability

        per_bucket_results[bucket_name] = {
            "ece": round(ece, 6),
            "n_samples": int(len(raw_scores)),
            "platt_coef": platt_coef,
            "platt_intercept": platt_intercept,
            "platt_model_type": model_type,
            "platt_fallback_reason": fallback_reason,
            "platt_constant_probability": constant_probability,
        }

        print(f"[CALI]   {bucket_name}: ECE={ece:.4f}, n={len(raw_scores)}, "
              f"diagram -> {png_path.name}", flush=True)

    # ========================================================================
    # Gate verdict
    # ========================================================================
    max_ece = max(v["ece"] for v in per_bucket_results.values())

    from scripts.threshold_manifest import (
        load_frozen_threshold_manifest,
        threshold_value,
    )

    manifest = load_frozen_threshold_manifest(THRESHOLD_MANIFEST, strict=True)
    # WR-02: fail closed (no silent fallback to hardcoded 0.10).
    threshold = float(threshold_value(manifest, "prefix_ece"))

    threshold_only_verdict = "pass" if max_ece <= threshold else "warn"

    # PR #14 follow-up review (Issue C): the producer now emits the
    # final scientific verdict, not just the threshold check. If any
    # per-bucket calibrator fell back to ``ConstantCalibrationModel``
    # (empty val bucket, single-class val bucket) the ECE for that
    # bucket is 0.0 by construction and the threshold gate would
    # falsely PASS. Downgrade to ``"warn"`` so any consumer that just
    # reads ``gate_verdict`` sees the limitation without having to
    # re-walk per-bucket fallback metadata.
    fallback_buckets = [
        name
        for name, bucket in per_bucket_results.items()
        if bucket.get("platt_model_type") == "constant"
    ]
    empty_buckets = [
        name
        for name, bucket in per_bucket_results.items()
        if bucket.get("n_samples") == 0
    ]
    if fallback_buckets or empty_buckets:
        gate_verdict = "warn"
        gate_verdict_reason = (
            "degenerate_calibrator_or_empty_bucket: "
            f"fallback={fallback_buckets}, empty={empty_buckets}"
        )
    else:
        gate_verdict = threshold_only_verdict
        gate_verdict_reason = "threshold_only"

    print(
        f"\n[CALI] Gate verdict: {gate_verdict} "
        f"(max_ece={max_ece:.4f} vs threshold={threshold}; "
        f"reason={gate_verdict_reason})",
        flush=True,
    )

    # ========================================================================
    # Write output JSON
    # ========================================================================
    val_coverage_metadata = build_coverage_metadata(
        val_coverage,
        threshold=args.min_mc_coverage,
        override=args.allow_incomplete_mc_coverage,
    )
    val_coverage_metadata["split"] = args.fit_split
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
        extra_paths=[THRESHOLD_MANIFEST, data_dir / "build_metadata.json"],
    )

    output_data = {
        "per_bucket": per_bucket_results,
        "max_ece": round(max_ece, 6),
        "threshold": threshold,
        "gate_verdict": gate_verdict,
        "gate_verdict_reason": gate_verdict_reason,
        "threshold_only_verdict": threshold_only_verdict,
        "degenerate_buckets": {
            "fallback": fallback_buckets,
            "empty": empty_buckets,
        },
        "mc_coverage": {
            args.fit_split: val_coverage_metadata,
            "test": test_coverage_metadata,
        },
        "mc_retention_gate": {
            args.fit_split: val_retention_meta,
            "test": test_retention_meta,
        },
        "mc_build_metadata": {
            "status": build_metadata["status"],
            "source_path": build_metadata["source_path"],
            "source_sha256": build_metadata["source_sha256"],
        },
        "metadata": {
            "seed": SEED,
            "n_val": len(mc_val),
            "n_test": len(mc_test),
            "n_fit": len(mc_val),
            "fit_split": args.fit_split,
            "model": "all-MiniLM-L6-v2",
            "n_bins": 10,
            "platt_C": 1.0,
            "bucket_boundaries": {"early": "[0.0, 0.33)", "mid": "[0.33, 0.66)", "late": "[0.66, 1.0]"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"[CALI] Results written to: {output_path}", flush=True)

    # Print summary
    print("\n" + "=" * 60)
    print("[CALI] PREFIX-WISE CALIBRATION RESULTS")
    print("=" * 60)
    for bucket_name in ["early", "mid", "late"]:
        r = per_bucket_results[bucket_name]
        print(f"  {bucket_name}: ECE={r['ece']:.4f} (n={r['n_samples']})")
    print(f"  max_ece={max_ece:.4f}, threshold={threshold}, verdict={gate_verdict}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
