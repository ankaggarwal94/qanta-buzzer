#!/usr/bin/env python3
"""
Compute CSLI (Choice-Set Leakage Index) via LLM-prompt panel.

CSLI = acc_full - acc_choices_only

This metric measures whether an LLM can identify the correct answer from
the multiple-choice options alone (without seeing the question text).
High CSLI indicates the answer choices leak information about the correct
answer through construction artifacts (e.g., length, specificity, plausibility
patterns).

IMPORTANT — DATA-05 SYMBOL COLLISION GUARD:
    DO NOT import evaluation.controls.run_choices_only_control -- that function
    uses surface-feature logistic regression (char n-gram TF-IDF sklearn), which
    is a DIFFERENT experiment measuring surface-feature artifacts. This script
    implements the LLM-prompt approach: present ONLY the K answer choices to an
    LLM and ask it to select the most likely correct answer. The two approaches
    measure fundamentally different constructs.

Usage:
    python scripts/compute_csli.py --help
    python scripts/compute_csli.py --smoke           # 1 model, 10 questions
    python scripts/compute_csli.py --dry-run         # Parse args, no compute
    python scripts/compute_csli.py --models gemini-1.5-flash,gpt-4o-mini

Inputs:
    data/processed/test_dataset.json  (MC questions with options)

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
# which is a different experiment from the LLM-prompt approach implemented here.
# See DATA-05 in MASTER_PLAN_v10 for the symbol collision prohibition.
# ============================================================================

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_exports" / "csli.json"
DEFAULT_MODELS = "gemini-1.5-flash,gpt-4o-mini,claude-3-haiku-20240307"


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

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


def run_choices_only_prompt(
    questions: list[dict],
    model_name: str,
) -> float:
    """Run choices-only LLM prompt experiment.

    Presents ONLY the answer choices (no question text) to an LLM and asks
    it to select the most likely correct answer. Returns accuracy (fraction
    of questions where the LLM selects the correct option).

    Parameters
    ----------
    questions : list[dict]
        List of MC question dicts with 'options' and 'correct_idx' fields.
    model_name : str
        LLM model identifier (e.g., 'gemini-1.5-flash').

    Returns
    -------
    float
        Accuracy of the LLM on choices-only prompts (0.0 to 1.0).

    Raises
    ------
    NotImplementedError
        Phase 4 implementation pending -- LLM API calls not yet wired.
    """
    raise NotImplementedError(
        "Phase 4 implementation pending: LLM choices-only prompt experiment. "
        "This stub establishes the correct architecture (LLM prompt, NOT "
        "surface-feature logistic regression)."
    )


def run_full_prompt(
    questions: list[dict],
    model_name: str,
) -> float:
    """Run full-question LLM prompt experiment (question text + choices).

    Presents the full question text along with answer choices to an LLM.
    Returns accuracy (fraction of questions answered correctly).

    Parameters
    ----------
    questions : list[dict]
        List of MC question dicts with 'question', 'options', 'correct_idx'.
    model_name : str
        LLM model identifier (e.g., 'gemini-1.5-flash').

    Returns
    -------
    float
        Accuracy of the LLM on full prompts (0.0 to 1.0).

    Raises
    ------
    NotImplementedError
        Phase 4 implementation pending -- LLM API calls not yet wired.
    """
    raise NotImplementedError(
        "Phase 4 implementation pending: LLM full-question prompt experiment. "
        "This stub establishes the correct architecture."
    )


def compute_panel_csli(results: dict[str, dict[str, float]]) -> dict:
    """Compute panel-level CSLI from per-model full and choices-only accuracies.

    CSLI_model = acc_full(model) - acc_choices_only(model)
    Panel CSLI = median(CSLI across models)

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Mapping from model_name -> {'acc_full': float, 'acc_choices_only': float}.

    Returns
    -------
    dict
        Panel CSLI summary with per-model and aggregate statistics.

    Raises
    ------
    NotImplementedError
        Phase 4 implementation pending -- requires run_full_prompt and
        run_choices_only_prompt results.
    """
    raise NotImplementedError(
        "Phase 4 implementation pending: requires per-model accuracy results "
        "from run_choices_only_prompt and run_full_prompt."
    )


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
        description="Compute CSLI via LLM-prompt panel (NOT surface-feature logistic regression)",
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
        help="Comma-separated list of LLM model names for the panel",
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

    # Load test dataset
    test_path = data_dir / "test_dataset.json"
    if not test_path.exists():
        print(f"ERROR: Test dataset not found: {test_path}", file=sys.stderr)
        return 1

    with open(test_path, "r") as f:
        questions = json.load(f)

    if args.smoke:
        questions = questions[:10]

    print(f"[CSLI] Loaded {len(questions)} questions from {test_path}")
    print(f"[CSLI] Panel models: {model_list}")

    # Phase 4: Run LLM experiments per model
    # Each model gets choices-only and full prompts
    results: dict[str, dict[str, float]] = {}
    for model_name in model_list:
        print(f"[CSLI] Running model: {model_name}")
        acc_choices = run_choices_only_prompt(questions, model_name)
        acc_full = run_full_prompt(questions, model_name)
        results[model_name] = {
            "acc_full": acc_full,
            "acc_choices_only": acc_choices,
            "csli": acc_full - acc_choices,
        }

    # Compute panel-level CSLI
    panel = compute_panel_csli(results)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(panel, f, indent=2)

    print(f"[CSLI] Results written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
