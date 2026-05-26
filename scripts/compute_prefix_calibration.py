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
    args = _parse_args(argv)

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    # Validate data directory exists
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[CALI] Dry run -- data_dir={data_dir}")
        print(f"[CALI] Output would be written to: {output_path}")
        # Validate required data files exist
        required = ["mc_dataset.json", "val_dataset.json", "test_dataset.json"]
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

    with open(mc_path, "r") as f:
        mc_questions = json.load(f)

    val_path = data_dir / "val_dataset.json"
    if not val_path.exists():
        print(f"ERROR: Val dataset not found: {val_path}", file=sys.stderr)
        return 1

    with open(val_path, "r") as f:
        val_data = json.load(f)

    test_path = data_dir / "test_dataset.json"
    if not test_path.exists():
        print(f"ERROR: Test dataset not found: {test_path}", file=sys.stderr)
        return 1

    with open(test_path, "r") as f:
        test_data = json.load(f)

    # Extract qid sets
    val_qids = set(str(q["qid"]) for q in val_data["questions"])
    test_qids = set(str(q["qid"]) for q in test_data["questions"])

    # Filter MC questions by split
    mc_val = [q for q in mc_questions if str(q["qid"]) in val_qids]
    mc_test = [q for q in mc_questions if str(q["qid"]) in test_qids]

    print(f"[CALI] MC total: {len(mc_questions)}, MC val: {len(mc_val)}, MC test: {len(mc_test)}", flush=True)

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

        platt_models[bucket_name] = fit_platt(scores, labels)
        coef = float(platt_models[bucket_name].coef_[0][0])
        intercept = float(platt_models[bucket_name].intercept_[0])
        print(f"[CALI]   {bucket_name}: coef={coef:.4f}, intercept={intercept:.4f} "
              f"(n={len(scores)})", flush=True)

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

        # Apply Platt calibration
        calibrated = lr.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

        # Compute ECE
        ece = compute_ece(calibrated, labels, n_bins=10)

        # Generate reliability diagram
        png_path = output_dir / f"reliability_{bucket_name}.png"
        plot_reliability_diagram(calibrated, labels, png_path, bucket_name, n_bins=10)

        # Store results
        per_bucket_results[bucket_name] = {
            "ece": round(ece, 6),
            "n_samples": int(len(raw_scores)),
            "platt_coef": round(float(lr.coef_[0][0]), 6),
            "platt_intercept": round(float(lr.intercept_[0]), 6),
        }

        print(f"[CALI]   {bucket_name}: ECE={ece:.4f}, n={len(raw_scores)}, "
              f"diagram -> {png_path.name}", flush=True)

    # ========================================================================
    # Gate verdict
    # ========================================================================
    max_ece = max(v["ece"] for v in per_bucket_results.values())

    # Load frozen threshold
    threshold = 0.10  # default
    if THRESHOLD_MANIFEST.exists():
        with open(THRESHOLD_MANIFEST, "r") as f:
            manifest = json.load(f)
        for t in manifest.get("thresholds", []):
            if t["metric"] == "prefix_ece":
                threshold = float(t["threshold"])
                break

    gate_verdict = "pass" if max_ece <= threshold else "warn"
    print(f"\n[CALI] Gate verdict: {gate_verdict} (max_ece={max_ece:.4f} vs threshold={threshold})",
          flush=True)

    # ========================================================================
    # Write output JSON
    # ========================================================================
    output_data = {
        "per_bucket": per_bucket_results,
        "max_ece": round(max_ece, 6),
        "threshold": threshold,
        "gate_verdict": gate_verdict,
        "metadata": {
            "seed": SEED,
            "n_val": len(mc_val),
            "n_test": len(mc_test),
            "model": "all-MiniLM-L6-v2",
            "n_bins": 10,
            "platt_C": 1.0,
            "bucket_boundaries": {"early": "[0.0, 0.33)", "mid": "[0.33, 0.66)", "late": "[0.66, 1.0]"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
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
