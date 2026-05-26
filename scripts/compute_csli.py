#!/usr/bin/env python3
"""
Compute CSLI (Choice-Set Leakage Index) via local model panel.

CSLI = acc_full - acc_choices_only

This metric measures whether a model can identify the correct answer from
the multiple-choice options alone (without seeing the question text).
High choices-only accuracy (above 1/K + 0.05 = 0.30 for K=4) indicates
the answer choices leak information about the correct answer through
construction artifacts (e.g., length, specificity, plausibility patterns).

IMPORTANT -- DATA-05 SYMBOL COLLISION GUARD:
    DO NOT import evaluation.controls.run_choices_only_control -- that function
    uses surface-feature logistic regression (char n-gram TF-IDF sklearn), which
    is a DIFFERENT experiment measuring surface-feature artifacts. This script
    implements a local-model panel approach: TF-IDF similarity, SBERT embeddings,
    and T5-small log-likelihood scoring. The two approaches measure fundamentally
    different constructs.

Usage:
    python scripts/compute_csli.py --help
    python scripts/compute_csli.py --smoke           # 1 model, 10 questions
    python scripts/compute_csli.py --dry-run         # Parse args, no compute
    python scripts/compute_csli.py --models tfidf,sbert,t5-small

Inputs:
    data/processed/mc_dataset.json   (MC questions with options, gold_index)
    data/processed/test_dataset.json (test split for qid filtering)

Outputs:
    paper_exports/csli.json  (panel CSLI results with bootstrap CIs)

Exit codes:
    0 = success
    1 = runtime error
    2 = argument error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ============================================================================
# SYMBOL_COLLISION_GUARD
# ============================================================================
# DO NOT import evaluation.controls.run_choices_only_control
# DO NOT import from evaluation.controls or from evaluation import controls
#
# That function uses surface-feature logistic regression (char-trigram TF-IDF),
# which is a different experiment from the local-model panel approach here.
# See DATA-05 in MASTER_PLAN_v10 for the symbol collision prohibition.
# ============================================================================

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_exports" / "csli.json"
DEFAULT_MODELS = "tfidf,sbert,t5-small"
THRESHOLD_MANIFEST = PROJECT_ROOT / "threshold_manifest.json"

# Lazy-loaded model caches
_SBERT_MODEL = None
_T5_MODEL = None
_T5_TOKENIZER = None


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Uses the **percentile method** (not BCa or studentized bootstrap).
    The percentile method is simpler but may undercover for skewed
    distributions. For the CSLI per-question indicators (discrete values
    in {-1, -2/3, ..., 2/3, 1}), this provides a CI that captures
    sampling variability (which questions are in the test set) but does
    NOT account for model selection uncertainty or between-model variance.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed values (e.g., per-item correctness indicators).
    n_resamples : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (default 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        (mean, ci_lower, ci_upper) where ci_lower and ci_upper are the
        percentile-based confidence interval bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)

    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(means, 100 * alpha / 2))
    ci_upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    mean = float(np.mean(values))

    return (mean, ci_lower, ci_upper)


# ============================================================================
# MODEL IMPLEMENTATIONS
# ============================================================================


def _get_sbert_model():
    """Lazy-load SBERT model (all-MiniLM-L6-v2)."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print("[CSLI] Loading SBERT model (all-MiniLM-L6-v2)...", flush=True)
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def _get_t5_model():
    """Lazy-load T5-small model and tokenizer."""
    global _T5_MODEL, _T5_TOKENIZER
    if _T5_MODEL is None:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        print("[CSLI] Loading T5-small model...", flush=True)
        _T5_TOKENIZER = T5Tokenizer.from_pretrained("t5-small")
        _T5_MODEL = T5ForConditionalGeneration.from_pretrained("t5-small")
        _T5_MODEL.eval()
    return _T5_MODEL, _T5_TOKENIZER


def _score_tfidf_choices_only(options: list[str]) -> int:
    """TF-IDF choices-only: select option most dissimilar from others.

    Parameters
    ----------
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(options)
    except ValueError:
        # All options are empty or identical -- fall back to 0
        return 0

    sim_matrix = cosine_similarity(tfidf_matrix)
    # For each option, compute mean similarity to OTHER options
    n = len(options)
    mean_sims = np.zeros(n)
    for i in range(n):
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        mean_sims[i] = np.mean(others) if others else 0.0

    # Select option with LOWEST mean similarity (most dissimilar = most specific)
    return int(np.argmin(mean_sims))


def _score_tfidf_full(question: str, options: list[str]) -> int:
    """TF-IDF full: select option most similar to question text.

    Parameters
    ----------
    question : str
        The full question text.
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    documents = [question] + options
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return 0

    # Similarity between question (index 0) and each option (indices 1..K)
    question_vec = tfidf_matrix[0:1]
    option_vecs = tfidf_matrix[1:]
    sims = cosine_similarity(question_vec, option_vecs)[0]

    return int(np.argmax(sims))


def _score_sbert_choices_only(options: list[str]) -> int:
    """SBERT choices-only: select option most dissimilar from others.

    Parameters
    ----------
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    model = _get_sbert_model()
    embeddings = model.encode(options, convert_to_numpy=True)
    sim_matrix = cosine_similarity(embeddings)

    n = len(options)
    mean_sims = np.zeros(n)
    for i in range(n):
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        mean_sims[i] = np.mean(others) if others else 0.0

    return int(np.argmin(mean_sims))


def _score_sbert_full(question: str, options: list[str]) -> int:
    """SBERT full: select option most similar to question embedding.

    Parameters
    ----------
    question : str
        The full question text.
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    model = _get_sbert_model()
    all_texts = [question] + options
    embeddings = model.encode(all_texts, convert_to_numpy=True)

    question_emb = embeddings[0:1]
    option_embs = embeddings[1:]
    sims = cosine_similarity(question_emb, option_embs)[0]

    return int(np.argmax(sims))


def _score_t5_choices_only(options: list[str]) -> int:
    """T5-small choices-only: select option with highest generation likelihood.

    Constructs input without question text and scores each option by
    cross-entropy loss (lower loss = higher likelihood = predicted answer).

    Parameters
    ----------
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    import torch

    model, tokenizer = _get_t5_model()
    labels = ["A", "B", "C", "D"]
    input_text = "question: Which is most likely correct? " + " ".join(
        f"{labels[i]}: {opt}" for i, opt in enumerate(options)
    )

    input_ids = tokenizer(
        input_text, return_tensors="pt", max_length=512, truncation=True
    ).input_ids

    losses = []
    with torch.no_grad():
        for i, opt in enumerate(options):
            target_text = f"{labels[i]}: {opt}"
            target_ids = tokenizer(
                target_text, return_tensors="pt", max_length=128, truncation=True
            ).input_ids
            outputs = model(input_ids=input_ids, labels=target_ids)
            losses.append(outputs.loss.item())

    # Lowest loss = highest likelihood
    return int(np.argmin(losses))


def _score_t5_full(question: str, options: list[str]) -> int:
    """T5-small full: select option with highest generation likelihood given question.

    Parameters
    ----------
    question : str
        The full question text.
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    import torch

    model, tokenizer = _get_t5_model()
    labels = ["A", "B", "C", "D"]
    input_text = "question: " + question + " " + " ".join(
        f"{labels[i]}: {opt}" for i, opt in enumerate(options)
    )

    input_ids = tokenizer(
        input_text, return_tensors="pt", max_length=512, truncation=True
    ).input_ids

    losses = []
    with torch.no_grad():
        for i, opt in enumerate(options):
            target_text = f"{labels[i]}: {opt}"
            target_ids = tokenizer(
                target_text, return_tensors="pt", max_length=128, truncation=True
            ).input_ids
            outputs = model(input_ids=input_ids, labels=target_ids)
            losses.append(outputs.loss.item())

    return int(np.argmin(losses))


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================


def run_choices_only_prompt(
    questions: list[dict],
    model_name: str,
) -> tuple[float, np.ndarray]:
    """Run choices-only experiment for a given model.

    Presents ONLY the answer choices (no question text) to the model and
    determines which option it selects. Returns accuracy and per-question
    correctness array.

    Parameters
    ----------
    questions : list[dict]
        List of MC question dicts with 'options' and 'gold_index' fields.
    model_name : str
        Model identifier: 'tfidf', 'sbert', or 't5-small'.

    Returns
    -------
    tuple[float, np.ndarray]
        (accuracy, per_question_correct) where per_question_correct is a
        binary array of shape (n_questions,).
    """
    n = len(questions)
    correct = np.zeros(n, dtype=np.int32)

    for i, q in enumerate(questions):
        options = q["options"]
        gold_idx = q["gold_index"]

        if model_name == "tfidf":
            pred = _score_tfidf_choices_only(options)
        elif model_name == "sbert":
            pred = _score_sbert_choices_only(options)
        elif model_name == "t5-small":
            pred = _score_t5_choices_only(options)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        correct[i] = int(pred == gold_idx)

        if (i + 1) % 500 == 0:
            print(f"[CSLI] {model_name} choices-only: {i+1}/{n}", flush=True)

    accuracy = float(np.mean(correct))
    return (accuracy, correct)


def run_full_prompt(
    questions: list[dict],
    model_name: str,
) -> tuple[float, np.ndarray]:
    """Run full-question experiment (question text + choices) for a given model.

    Presents the full question text along with answer choices. Returns accuracy
    and per-question correctness array.

    Parameters
    ----------
    questions : list[dict]
        List of MC question dicts with 'question', 'options', 'gold_index'.
    model_name : str
        Model identifier: 'tfidf', 'sbert', or 't5-small'.

    Returns
    -------
    tuple[float, np.ndarray]
        (accuracy, per_question_correct) where per_question_correct is a
        binary array of shape (n_questions,).
    """
    n = len(questions)
    correct = np.zeros(n, dtype=np.int32)

    for i, q in enumerate(questions):
        question_text = q["question"]
        options = q["options"]
        gold_idx = q["gold_index"]

        if model_name == "tfidf":
            pred = _score_tfidf_full(question_text, options)
        elif model_name == "sbert":
            pred = _score_sbert_full(question_text, options)
        elif model_name == "t5-small":
            pred = _score_t5_full(question_text, options)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        correct[i] = int(pred == gold_idx)

        if (i + 1) % 500 == 0:
            print(f"[CSLI] {model_name} full: {i+1}/{n}", flush=True)

    accuracy = float(np.mean(correct))
    return (accuracy, correct)


def compute_panel_csli(
    results: dict[str, dict],
    per_question_csli: np.ndarray,
    n_questions: int,
    model_list: list[str],
) -> dict:
    """Compute panel-level CSLI from per-model results with bootstrap CI.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping from model_name -> {'acc_full', 'acc_choices_only', 'csli', 'leakage_flag'}.
    per_question_csli : np.ndarray
        Per-question CSLI values averaged across models (for bootstrap CI).
    n_questions : int
        Number of questions evaluated.
    model_list : list[str]
        Ordered list of model names.

    Returns
    -------
    dict
        Panel CSLI summary with per-model and aggregate statistics.
    """
    # Compute panel mean CSLI
    csli_values = [results[m]["csli"] for m in model_list]
    panel_mean = float(np.mean(csli_values))

    # Bootstrap CI on per-question CSLI array
    mean_ci, ci_lower, ci_upper = bootstrap_ci(
        per_question_csli, n_resamples=1000, seed=789685
    )

    from scripts.threshold_manifest import load_frozen_threshold_manifest

    manifest = load_frozen_threshold_manifest(THRESHOLD_MANIFEST, strict=True)
    threshold = 0.30  # default: 1/K + 0.05, K=4
    for t in manifest.get("thresholds", []):
        if t["metric"] == "choices_only_accuracy":
            threshold = float(t.get("numeric_value_K4", 0.30))
            break
    for m in model_list:
        results[m]["leakage_flag"] = results[m]["acc_choices_only"] > threshold

    return {
        "panel_csli": {
            "mean": panel_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        "per_model": results,
        "metadata": {
            "n_questions": n_questions,
            "n_models": len(model_list),
            "K": 4,
            "threshold": threshold,
            "bootstrap_resamples": 1000,
            "bootstrap_method": "percentile",
            "bootstrap_note": (
                "Percentile-method bootstrap (not BCa). CI captures "
                "sampling variability over questions but not model "
                "selection uncertainty."
            ),
            "seed": 789685,
            "test_split_seed": 789685,
            "models": model_list,
        },
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for CSLI computation.

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
        description="Compute CSLI via local model panel (TF-IDF, SBERT, T5-small)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing processed MC datasets (default: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output path for CSLI results JSON (default: paper_exports/csli.json)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated list of model names for the panel. "
            "Available: tfidf, sbert, t5-small (default: tfidf,sbert,t5-small)"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick mode: 1 model, 10 questions (for pipeline testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and validate data path, but do not run compute",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for CSLI computation.

    Parameters
    ----------
    argv : Optional[list[str]]
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    args = _parse_args(argv)

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    model_list = [m.strip() for m in args.models.split(",")]

    if args.smoke:
        model_list = model_list[:1]
        print(f"[CSLI] Smoke mode: 1 model ({model_list[0]}), 10 questions")

    # Validate data directory exists
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[CSLI] Dry run -- data_dir={data_dir}, models={model_list}")
        print(f"[CSLI] Output would be written to: {output_path}")
        return 0

    # Load MC dataset (has options and gold_index)
    mc_path = data_dir / "mc_dataset.json"
    if not mc_path.exists():
        print(f"ERROR: MC dataset not found: {mc_path}", file=sys.stderr)
        return 1

    with open(mc_path, "r") as f:
        mc_questions = json.load(f)

    # Load test split (for qid filtering)
    test_path = data_dir / "test_dataset.json"
    if not test_path.exists():
        print(f"ERROR: Test dataset not found: {test_path}", file=sys.stderr)
        return 1

    with open(test_path, "r") as f:
        test_data = json.load(f)

    # Extract test split qids
    test_qids = set(str(q["qid"]) for q in test_data["questions"])

    # Filter MC questions to only those in test split
    questions = [q for q in mc_questions if str(q["qid"]) in test_qids]
    print(f"[CSLI] Loaded {len(questions)} test-split MC questions "
          f"(from {len(mc_questions)} MC total, {len(test_qids)} test qids)")

    if args.smoke:
        questions = questions[:10]
        print(f"[CSLI] Smoke mode: trimmed to {len(questions)} questions")

    print(f"[CSLI] Panel models: {model_list}")

    # Run experiments per model
    results: dict[str, dict] = {}
    per_model_choices_correct: dict[str, np.ndarray] = {}
    per_model_full_correct: dict[str, np.ndarray] = {}

    for model_name in model_list:
        print(f"\n[CSLI] === Running model: {model_name} ===", flush=True)

        print(f"[CSLI] {model_name}: choices-only condition...", flush=True)
        acc_choices, choices_correct = run_choices_only_prompt(questions, model_name)

        print(f"[CSLI] {model_name}: full condition...", flush=True)
        acc_full, full_correct = run_full_prompt(questions, model_name)

        csli = acc_full - acc_choices
        results[model_name] = {
            "acc_full": round(acc_full, 6),
            "acc_choices_only": round(acc_choices, 6),
            "csli": round(csli, 6),
        }
        per_model_choices_correct[model_name] = choices_correct
        per_model_full_correct[model_name] = full_correct

        print(f"[CSLI] {model_name}: full={acc_full:.4f}, "
              f"choices_only={acc_choices:.4f}, CSLI={csli:.4f}", flush=True)

    # Compute per-question CSLI averaged across models for bootstrap
    n_questions = len(questions)
    per_question_csli = np.zeros(n_questions)
    for m in model_list:
        # per-question CSLI = correct_full - correct_choices_only
        per_question_csli += (
            per_model_full_correct[m].astype(float)
            - per_model_choices_correct[m].astype(float)
        )
    per_question_csli /= len(model_list)

    # Compute panel-level CSLI with bootstrap CI
    panel = compute_panel_csli(results, per_question_csli, n_questions, model_list)

    # Print summary
    print("\n" + "=" * 60)
    print("[CSLI] RESULTS SUMMARY")
    print("=" * 60)
    p = panel["panel_csli"]
    print(f"Panel CSLI: {p['mean']:.4f} [{p['ci_lower']:.4f}, {p['ci_upper']:.4f}]")
    for m, v in panel["per_model"].items():
        flag = " ** LEAKAGE WARNING **" if v["leakage_flag"] else ""
        print(f"  {m}: full={v['acc_full']:.4f}, choices_only={v['acc_choices_only']:.4f}, "
              f"csli={v['csli']:.4f}{flag}")
    print(f"Threshold: acc_choices_only > {panel['metadata']['threshold']:.2f} triggers flag")
    print("=" * 60)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(panel, f, indent=2)

    print(f"\n[CSLI] Results written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
