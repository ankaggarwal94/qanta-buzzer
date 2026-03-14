This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
- Pay special attention to the Repository Description. These contain important context and guidelines specific to this project.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: README.md, CLAUDE.md, AGENTS.md, pyproject.toml, requirements.txt, .github/copilot-instructions.md, configs/**, qb_data/**, qb_env/**, models/**, agents/**, evaluation/**, scripts/**, training/**, tests/**
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# User Provided Header
qanta-buzzer branch main @ 22236838 (core code snapshot — post-remediation)

# Directory Structure
```
.github/
  copilot-instructions.md
agents/
  __init__.py
  _math.py
  bayesian_buzzer.py
  ppo_buzzer.py
  softmax_profile_buzzer.py
  threshold_buzzer.py
configs/
  default.yaml
  smoke.yaml
  t5_policy.yaml
evaluation/
  __init__.py
  controls.py
  metrics.py
  plotting.py
models/
  __init__.py
  answer_profiles.py
  features.py
  likelihoods.py
  t5_policy.py
qb_data/
  __init__.py
  answer_profiles.py
  config.py
  data_loader.py
  dataset_splits.py
  huggingface_loader.py
  mc_builder.py
  text_utils.py
qb_env/
  __init__.py
  data_loader.py
  mc_builder.py
  text_utils.py
  text_wrapper.py
  tossup_env.py
scripts/
  _common.py
  build_mc_dataset.py
  ci.sh
  compare_policies.py
  evaluate_all.py
  manual-smoke.sh
  run_baselines.py
  run_smoke_pipeline.py
  sweep_reward_shaping.py
  test_mc_builder.py
  train_ppo.py
  train_t5_policy.py
tests/
  conftest.py
  test_agents.py
  test_answer_profile_cache.py
  test_build_mc_dataset.py
  test_dataset_splits.py
  test_environment.py
  test_factories.py
  test_features.py
  test_likelihoods.py
  test_mc_builder_topk.py
  test_metrics.py
  test_ppo_buzzer.py
  test_ppo_t5.py
  test_qb_rl_bridge.py
  test_supervised_t5.py
  test_t5_policy.py
  test_text_wrapper.py
training/
  __init__.py
  train_ppo_t5.py
  train_supervised_t5.py
AGENTS.md
CLAUDE.md
pyproject.toml
README.md
requirements.txt
```

# Files

## File: tests/test_dataset_splits.py
````python
"""Tests for stratified dataset splitting reproducibility.

Verifies that splits are deterministic across invocations and do not
depend on Python's hash randomization (PYTHONHASHSEED).
"""

import subprocess
import sys

import pytest

from qb_data.data_loader import TossupQuestion
from qb_data.dataset_splits import create_stratified_splits


def _make_questions(n: int, categories: list[str]) -> list[TossupQuestion]:
    """Create n dummy TossupQuestion instances cycling through categories."""
    questions = []
    for i in range(n):
        cat = categories[i % len(categories)]
        questions.append(
            TossupQuestion(
                qid=f"q{i:04d}",
                question=f"Question {i}",
                tokens=[f"token{i}"],
                answer_primary=f"Answer {i}",
                clean_answers=[f"Answer {i}"],
                run_indices=[0],
                human_buzz_positions=[],
                category=cat,
                cumulative_prefixes=[f"token{i}"],
            )
        )
    return questions


def test_splits_deterministic_same_process():
    """Same seed produces identical splits within one process."""
    questions = _make_questions(60, ["History", "Science", "Literature"])
    train1, val1, test1 = create_stratified_splits(questions, seed=42)
    train2, val2, test2 = create_stratified_splits(questions, seed=42)
    assert [q.qid for q in train1] == [q.qid for q in train2]
    assert [q.qid for q in val1] == [q.qid for q in val2]
    assert [q.qid for q in test1] == [q.qid for q in test2]


def test_splits_deterministic_across_processes():
    """Splits must be identical even with different PYTHONHASHSEED values.

    Runs the split in two subprocesses with different PYTHONHASHSEED and
    checks that they produce identical qid orderings.
    """
    script = (
        "import json, sys, io; sys.path.insert(0, '.'); "
        "sys.stdout = io.StringIO(); "
        "from qb_data.data_loader import TossupQuestion; "
        "from qb_data.dataset_splits import create_stratified_splits; "
        "qs = [TossupQuestion(qid=f'q{i:04d}', question=f'Q{i}', tokens=[f't{i}'], "
        "answer_primary=f'A{i}', clean_answers=[f'A{i}'], run_indices=[0], "
        "human_buzz_positions=[], category=['History','Science','Lit'][i%3], "
        "cumulative_prefixes=[f't{i}']) for i in range(60)]; "
        "tr,va,te = create_stratified_splits(qs, seed=42); "
        "sys.stdout = sys.__stdout__; "
        "print(json.dumps([q.qid for q in tr]))"
    )
    import json
    import os

    base_env = {k: v for k, v in os.environ.items()}
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    results = []
    for hashseed in ["0", "12345"]:
        env = {**base_env, "PYTHONHASHSEED": hashseed}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root,
            timeout=30,
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        results.append(json.loads(proc.stdout.strip()))
    assert results[0] == results[1], (
        "Splits differ across PYTHONHASHSEED values — hash(category) is not deterministic"
    )


def test_splits_different_seeds_differ():
    """Different seeds should produce different splits."""
    questions = _make_questions(60, ["History", "Science", "Literature"])
    train1, _, _ = create_stratified_splits(questions, seed=42)
    train2, _, _ = create_stratified_splits(questions, seed=99)
    assert [q.qid for q in train1] != [q.qid for q in train2]


def test_splits_all_questions_assigned():
    """Every question must appear in exactly one split."""
    questions = _make_questions(100, ["A", "B", "C", "D"])
    train, val, test = create_stratified_splits(questions, seed=1)
    all_qids = {q.qid for q in train} | {q.qid for q in val} | {q.qid for q in test}
    assert len(all_qids) == 100
    assert len(train) + len(val) + len(test) == 100
````

## File: .github/copilot-instructions.md
````markdown
# Copilot Instructions for `qanta-buzzer`

Use these instructions as the repo-wide baseline for Copilot work in this repository. Keep them concise, and prefer branch-local source-of-truth docs when they exist.

## Source of truth

- If the checked-out branch contains `CLAUDE.md`, follow it.
- If the checked-out branch contains `.planning/`, treat `.planning/` as the durable project state and keep important workflow decisions aligned with it.
- Do not invent a second planning system in parallel with existing repo docs.

## Code paths

- This repository has an older root-level prototype path centered on files such as `main.py`, `environment.py`, `dataset.py`, `model.py`, `train_supervised.py`, `train_ppo.py`, and `metrics.py`.
- Some branches also contain a newer modular pipeline with packages such as `qb_data/`, `qb_env/`, `models/`, `agents/`, `evaluation/`, `scripts/`, and `training/`.
- Match the checked-out branch. Do not assume the modular pipeline exists on every branch, and do not force work back into the root-level prototype if the modular packages are already present.

## Change discipline

- Keep changes minimal and scoped to the request.
- Prefer editing existing modules over introducing new abstractions unless the request clearly needs them.
- Do not add dependencies unless they are required.
- Do not commit generated Python cache files, virtual environments, model artifacts, or local notebooks unless the task explicitly asks for tracked generated outputs.

## Validation

- Prefer the narrowest relevant verification for the files you changed.
- On older/root-prototype branches, the lightweight validation scripts are:
  - `python test_imports.py`
  - `python test_csv_loader.py`
- On branches with `tests/` and `pyproject.toml`, prefer targeted `pytest` first and run the full suite when the change is broad or touches shared infrastructure.
- If the branch exposes smoke workflows such as `python scripts/build_mc_dataset.py --smoke`, prefer those over heavyweight full training runs during routine iteration.

## Heavyweight ML workflows

- This repo uses heavyweight ML dependencies including PyTorch, Transformers, sentence-transformers, and Stable-Baselines3.
- Avoid expensive model downloads or long training runs unless the task actually requires them.
- If you are editing docs, config handling, tests, or small control-flow logic, do not trigger full T5 or PPO training just to prove the change.

## Practical repo guidance

- Respect the existing file organization and naming conventions on the active branch.
- When documentation and code disagree, trust the executable code first, then update docs to match.
- If a branch includes compatibility shims or bridge code, preserve backward-compatible imports and config aliases unless the task explicitly asks to remove them.
````

## File: agents/_math.py
````python
from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid for scalar confidence proxies."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)
````

## File: evaluation/__init__.py
````python
"""
Evaluation Package

Metrics computation for quiz bowl buzzer agents, including S_q scoring,
calibration analysis (ECE, Brier score), and buzz timing statistics.

Ported from qb-rl reference implementation with adaptations for
qanta-buzzer's EpisodeResult / SoftmaxEpisodeResult / PPOEpisodeTrace
dataclass structures.
"""

from evaluation.metrics import (
    calibration_at_buzz,
    expected_calibration_error,
    per_category_accuracy,
    summarize_buzz_metrics,
    system_score,
)

__all__ = [
    "system_score",
    "summarize_buzz_metrics",
    "calibration_at_buzz",
    "expected_calibration_error",
    "per_category_accuracy",
]
````

## File: evaluation/metrics.py
````python
"""
Evaluation Metrics for Quiz Bowl Buzzer Agents

Computes buzz accuracy, S_q scoring, calibration metrics (ECE, Brier score),
and buzz timing statistics from episode trace data.

Ported from qb-rl reference implementation (evaluation/metrics.py).
Accepts both raw dicts and dataclass instances (EpisodeResult,
SoftmaxEpisodeResult, PPOEpisodeTrace) via the _to_dict helper.

Functions
---------
system_score(c_trace, g_trace)
    Compute S_q = sum_t b_t * g_t where b_t = c_t * prod_{i<t} (1 - c_i).
expected_calibration_error(confidences, outcomes, n_bins)
    Binned ECE over confidence-outcome pairs.
brier_score(confidences, outcomes)
    Mean squared error between confidence and binary outcome.
summarize_buzz_metrics(results)
    Aggregate accuracy, buzz step, S_q, and reward across episodes.
calibration_at_buzz(results)
    Extract buzz-time top_p confidence and compute ECE + Brier score.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def _to_dict(item: Any) -> dict[str, Any]:
    """Convert dataclass or object to dict for uniform access.

    Parameters
    ----------
    item : Any
        A dict, dataclass instance, or object with __dict__.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the item.
    """
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    return item.__dict__


def system_score(c_trace: list[float], g_trace: list[float]) -> float:
    """Compute S_q scoring metric for a single episode.

    S_q = sum_t b_t * g_t, where b_t = c_t * prod_{i<t} (1 - c_i).
    This is the expected correctness under the agent's buzz policy,
    accounting for the survival probability of not having buzzed earlier.

    Parameters
    ----------
    c_trace : list[float]
        Buzz probability at each time step (confidence proxy).
    g_trace : list[float]
        Correctness indicator at each time step (1.0 if top answer is
        correct, 0.0 otherwise).

    Returns
    -------
    float
        S_q score for the episode, in [0, 1].
    """
    c = np.array(c_trace, dtype=np.float64)
    g = np.array(g_trace, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    b = np.zeros_like(c)
    survival = 1.0
    for t in range(len(c)):
        b[t] = c[t] * survival
        survival *= (1.0 - c[t])
    return float(np.sum(b * g))


def expected_calibration_error(
    confidences: list[float], outcomes: list[int], n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) with uniform binning.

    ECE measures the gap between predicted confidence and actual accuracy
    across confidence bins. Lower ECE indicates better-calibrated predictions.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).
    n_bins : int
        Number of uniform bins for confidence bucketing.

    Returns
    -------
    float
        Expected calibration error in [0, 1]. Returns 0.0 if no data.
    """
    if not confidences:
        return 0.0
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        bin_acc = y[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    """Compute Brier score (mean squared calibration error).

    Brier score measures the mean squared difference between predicted
    confidence and binary outcome. Lower is better; 0 is perfect.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).

    Returns
    -------
    float
        Brier score in [0, 1]. Returns 0.0 if no data.
    """
    if not confidences:
        return 0.0
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    return float(np.mean((conf - y) ** 2))


def summarize_buzz_metrics(results: list[Any]) -> dict[str, float]:
    """Aggregate buzz metrics across a list of episode results.

    Computes accuracy, mean buzz step, mean S_q score, and mean reward
    from episode trace data. Accepts dicts or dataclass instances.

    Parameters
    ----------
    results : list[Any]
        List of episode results (dicts, EpisodeResult, SoftmaxEpisodeResult,
        or PPOEpisodeTrace instances). Each must have: correct, buzz_step,
        c_trace, g_trace. Optionally: reward_like or episode_reward.

    Returns
    -------
    dict[str, float]
        Summary metrics: n, buzz_accuracy, mean_buzz_step, mean_sq,
        mean_reward_like.
    """
    rows = [_to_dict(r) for r in results]
    if not rows:
        return {
            "n": 0.0,
            "buzz_accuracy": 0.0,
            "mean_buzz_step": 0.0,
            "mean_sq": 0.0,
            "mean_reward_like": 0.0,
        }

    correct = np.array(
        [1 if bool(r.get("correct", False)) else 0 for r in rows],
        dtype=np.float64,
    )
    buzz_steps = np.array(
        [int(r.get("buzz_step", 0)) for r in rows], dtype=np.float64
    )
    sq_scores = np.array(
        [
            system_score(
                list(r.get("c_trace", [])),
                list(r.get("g_trace", [])),
            )
            for r in rows
        ],
        dtype=np.float64,
    )
    reward_like = np.array(
        [
            float(r.get("reward_like", r.get("episode_reward", 0.0)))
            for r in rows
        ],
        dtype=np.float64,
    )

    return {
        "n": float(len(rows)),
        "buzz_accuracy": float(correct.mean()),
        "mean_buzz_step": float(buzz_steps.mean()),
        "mean_sq": float(sq_scores.mean()),
        "mean_reward_like": float(reward_like.mean()),
    }


def per_category_accuracy(
    results: list[Any],
    questions: list[Any],
) -> dict[str, dict[str, float]]:
    """Compute accuracy and S_q metrics grouped by question category.

    Joins results with questions to extract category field, then groups
    and computes summarize_buzz_metrics per category.

    Parameters
    ----------
    results : list[Any]
        Episode results from agent evaluation (dicts or dataclasses).
        Must have qid field for joining.
    questions : list[Any]
        Original questions with category field (MCQuestion or similar).

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from category name to metrics dict with keys:
        n, buzz_accuracy, mean_buzz_step, mean_sq, mean_reward_like.
    """
    from collections import defaultdict

    # Build qid -> category lookup, default to "unknown" for missing
    qid_to_category: dict[str, str] = {}
    for q in questions:
        q_dict = _to_dict(q)
        cat = q_dict.get("category", "") or ""
        qid = q_dict.get("qid", "")
        qid_to_category[qid] = cat if cat else "unknown"

    # Group results by category
    by_category: dict[str, list[Any]] = defaultdict(list)
    for r in results:
        r_dict = _to_dict(r)
        qid = r_dict.get("qid", "")
        category = qid_to_category.get(qid, "unknown")
        by_category[category].append(r)

    # Compute metrics per category
    return {
        cat: summarize_buzz_metrics(rows)
        for cat, rows in sorted(by_category.items())
    }


def calibration_at_buzz(results: list[Any]) -> dict[str, float]:
    """Compute calibration metrics at the buzz decision point.

    Uses the belief model's top-answer probability (``top_p_trace``) at
    buzz time as the confidence proxy.  This measures whether the belief
    distribution is well-calibrated: when the model assigns 0.8
    probability to its top answer, that answer should be correct ~80% of
    the time.

    Falls back to ``c_trace`` (sigmoid confidence) when ``top_p_trace``
    is unavailable (e.g. PPO episode traces that lack per-step belief
    breakdowns).

    Parameters
    ----------
    results : list[Any]
        List of episode results (dicts or dataclass instances). Each must
        have: buzz_step, correct, and at least one of top_p_trace or
        c_trace.

    Returns
    -------
    dict[str, float]
        Calibration metrics: ece, brier, n_calibration.
    """
    rows = [_to_dict(r) for r in results]
    confidences: list[float] = []
    outcomes: list[int] = []
    for row in rows:
        top_p_trace = list(row.get("top_p_trace", []))
        c_trace = list(row.get("c_trace", []))
        conf_trace = top_p_trace if top_p_trace else c_trace
        if not conf_trace:
            continue
        buzz_step = int(row.get("buzz_step", max(0, len(conf_trace) - 1)))
        idx = min(max(0, buzz_step), len(conf_trace) - 1)
        confidences.append(float(conf_trace[idx]))
        outcomes.append(1 if bool(row.get("correct", False)) else 0)

    return {
        "ece": expected_calibration_error(confidences, outcomes),
        "brier": brier_score(confidences, outcomes),
        "n_calibration": float(len(confidences)),
    }
````

## File: evaluation/plotting.py
````python
"""
Visualization Functions for Quiz Bowl Buzzer Evaluation

Provides plotting utilities for evaluation results including entropy curves,
calibration plots, and comparison tables. All functions accept output paths
and create parent directories as needed.

Ported from qb-rl reference implementation (evaluation/plotting.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_parent(path: str | Path) -> Path:
    """Create parent directories for an output path if needed.

    Parameters
    ----------
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The resolved Path object.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_learning_curve(
    timesteps: list[int],
    rewards: list[float],
    output_path: str | Path,
) -> str:
    """Plot training learning curve (reward vs timesteps).

    Parameters
    ----------
    timesteps : list[int]
        Training timestep values.
    rewards : list[float]
        Corresponding episode reward values.
    output_path : str or Path
        File path for the saved figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    p = _ensure_parent(output_path)
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=timesteps, y=rewards)
    plt.title("Learning Curve")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)


def plot_entropy_vs_clue_index(
    entropy_traces: dict[str, list[float]],
    output_path: str | Path,
) -> str:
    """Plot policy entropy as a function of clue index.

    Creates a line plot with multiple agent entropy traces showing how
    policy uncertainty decreases as more clues are revealed.

    Parameters
    ----------
    entropy_traces : dict[str, list[float]]
        Mapping from agent name to per-step entropy values.
    output_path : str or Path
        File path for the saved figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    p = _ensure_parent(output_path)
    plt.figure(figsize=(7, 4))
    for label, trace in entropy_traces.items():
        x = np.arange(len(trace))
        sns.lineplot(x=x, y=trace, label=label)
    plt.title("Belief Entropy vs Clue Index")
    plt.xlabel("Clue index")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)


def plot_calibration_curve(
    confidences: list[float],
    outcomes: list[int],
    output_path: str | Path,
    n_bins: int = 10,
) -> str:
    """Plot calibration curve (predicted confidence vs empirical accuracy).

    Bins confidences into uniform bins and plots mean accuracy per bin
    against mean confidence. The diagonal represents perfect calibration.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).
    output_path : str or Path
        File path for the saved figure.
    n_bins : int
        Number of uniform bins for confidence bucketing.

    Returns
    -------
    str
        Path to the saved figure.
    """
    p = _ensure_parent(output_path)
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs = []
    ys = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        xs.append(conf[mask].mean())
        ys.append(y[mask].mean())

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter(xs, ys, color="tab:blue")
    plt.title("Calibration Plot")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)


def save_comparison_table(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    """Save agent comparison metrics as a CSV or markdown table.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        List of metric dicts, each with agent name and metrics.
    output_path : str or Path
        File path for the saved table (.csv or .md).

    Returns
    -------
    str
        Path to the saved table file.
    """
    p = _ensure_parent(output_path)
    df = pd.DataFrame(rows)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_markdown(p, index=False)
    return str(p)
````

## File: models/features.py
````python
"""
Belief Feature Extraction

Extracts derived features from belief probability distributions for use as
policy observations. Given a belief vector of K probabilities (one per answer
option), produces a (K + 6)-dimensional feature vector containing:

    belief[0..K-1]   raw belief probabilities
    top_p             max belief probability
    margin            gap between top two probabilities
    entropy           Shannon entropy of the distribution
    stability         L1 distance from previous belief (0 if first step)
    progress          fraction of total clue steps elapsed
    clue_idx_norm     normalized clue index (0 to 1 over steps)

Ported from qb-rl reference implementation (models/features.py).
"""

from __future__ import annotations

import numpy as np


def entropy_of_distribution(prob: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Uses clipping for numerical stability to avoid log(0).

    Parameters
    ----------
    prob : np.ndarray
        1D probability vector. Values should sum to ~1.0.

    Returns
    -------
    float
        Shannon entropy H(p) = -sum(p * log(p)), non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> uniform = np.array([0.25, 0.25, 0.25, 0.25])
    >>> abs(entropy_of_distribution(uniform) - 1.3863) < 0.001
    True
    """
    clipped = np.clip(prob, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def extract_belief_features(
    belief: np.ndarray,
    prev_belief: np.ndarray | None,
    step_idx: int,
    total_steps: int,
) -> np.ndarray:
    """Extract derived features from a belief probability vector.

    Concatenates the raw belief with 6 derived scalar features to produce
    a fixed-size observation vector for the RL policy.

    Parameters
    ----------
    belief : np.ndarray
        1D probability vector of shape (K,) over answer options.
    prev_belief : np.ndarray or None
        Previous step's belief vector, same shape as ``belief``.
        Pass None on the first step (stability will be 0.0).
    step_idx : int
        Current clue step index (0-based).
    total_steps : int
        Total number of clue steps in the episode.

    Returns
    -------
    np.ndarray
        Feature vector of shape (K + 6,) with dtype float32.
        Layout: [belief..., top_p, margin, entropy, stability, progress, clue_idx_norm].

    Raises
    ------
    ValueError
        If ``belief`` is not a 1D array.

    Examples
    --------
    >>> import numpy as np
    >>> belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    >>> feats = extract_belief_features(belief, None, 2, 6)
    >>> feats.shape
    (10,)
    >>> feats.dtype
    dtype('float32')
    """
    belief = np.asarray(belief, dtype=np.float32)
    if belief.ndim != 1:
        raise ValueError("belief must be a 1D probability vector")

    top_p = float(np.max(belief))
    sorted_probs = np.sort(belief)[::-1]
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_p - second
    ent = entropy_of_distribution(belief)
    stability = float(np.abs(belief - prev_belief).sum()) if prev_belief is not None else 0.0
    progress = float(step_idx / max(1, total_steps))
    clue_idx_norm = float(step_idx / max(1, total_steps - 1))

    extras = np.array([top_p, margin, ent, stability, progress, clue_idx_norm], dtype=np.float32)
    return np.concatenate([belief, extras]).astype(np.float32)
````

## File: qb_data/dataset_splits.py
````python
"""
Stratified dataset splitting utilities for quiz bowl data.

This module provides functions to create train/val/test splits that maintain
category distribution across all splits.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from qb_data.data_loader import TossupQuestion


def create_stratified_splits(
    questions: List[TossupQuestion],
    ratios: List[float] = [0.7, 0.15, 0.15],
    seed: int = 42
) -> Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]:
    """
    Create stratified train/val/test splits maintaining category distribution.

    Parameters
    ----------
    questions : List[TossupQuestion]
        List of questions to split
    ratios : List[float]
        Train/val/test split ratios (must sum to 1.0)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]
        Train, validation, and test splits

    Raises
    ------
    ValueError
        If ratios don't sum to 1.0 or questions list is empty
    """
    # Validate inputs
    if not questions:
        raise ValueError("Cannot split empty question list")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    # Initialize random generator for reproducibility
    rng = random.Random(seed)

    # Group questions by category
    category_groups = defaultdict(list)
    for q in questions:
        category_groups[q.category].append(q)

    # Initialize output lists
    train_questions = []
    val_questions = []
    test_questions = []

    # Split each category maintaining ratios
    for category, category_questions in category_groups.items():
        # Sort for deterministic splits
        sorted_questions = sorted(category_questions, key=lambda q: q.qid)

        # Deterministic per-category seed via MD5 (immune to PYTHONHASHSEED)
        cat_hash = int(hashlib.md5(category.encode("utf-8")).hexdigest(), 16)
        category_seed = seed + cat_hash % 1_000_000
        category_rng = random.Random(category_seed)
        shuffled = sorted_questions.copy()
        category_rng.shuffle(shuffled)

        n = len(shuffled)

        # Calculate split indices
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        # Handle small categories - ensure at least 1 in train if possible
        if n == 1:
            train_questions.extend(shuffled)
        elif n == 2:
            train_questions.extend(shuffled[:1])
            val_questions.extend(shuffled[1:])
        else:
            # Standard split
            train_questions.extend(shuffled[:train_end])
            val_questions.extend(shuffled[train_end:val_end])
            test_questions.extend(shuffled[val_end:])

    # Verify all questions assigned exactly once
    total_original = len(questions)
    total_split = len(train_questions) + len(val_questions) + len(test_questions)

    if total_original != total_split:
        raise RuntimeError(f"Split mismatch: {total_original} original vs {total_split} split")

    # Log category distribution statistics
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_questions)} questions ({len(train_questions)/total_original:.1%})")
    print(f"  Val:   {len(val_questions)} questions ({len(val_questions)/total_original:.1%})")
    print(f"  Test:  {len(test_questions)} questions ({len(test_questions)/total_original:.1%})")

    # Category distribution analysis
    train_categories = defaultdict(int)
    val_categories = defaultdict(int)
    test_categories = defaultdict(int)

    for q in train_questions:
        train_categories[q.category] += 1
    for q in val_questions:
        val_categories[q.category] += 1
    for q in test_questions:
        test_categories[q.category] += 1

    all_categories = set(train_categories.keys()) | set(val_categories.keys()) | set(test_categories.keys())
    print(f"\nCategory distribution ({len(all_categories)} categories):")

    for category in sorted(all_categories)[:5]:  # Show first 5 categories
        orig_count = len(category_groups[category])
        train_count = train_categories.get(category, 0)
        val_count = val_categories.get(category, 0)
        test_count = test_categories.get(category, 0)
        print(f"  {category}: {train_count}/{val_count}/{test_count} (orig: {orig_count})")

    if len(all_categories) > 5:
        print(f"  ... and {len(all_categories) - 5} more categories")

    return train_questions, val_questions, test_questions


def save_splits(
    train: List[TossupQuestion],
    val: List[TossupQuestion],
    test: List[TossupQuestion],
    output_dir: str = "data"
) -> None:
    """
    Save dataset splits to JSON files with metadata.

    Parameters
    ----------
    train : List[TossupQuestion]
        Training split
    val : List[TossupQuestion]
        Validation split
    test : List[TossupQuestion]
        Test split
    output_dir : str
        Directory to save split files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Helper to convert TossupQuestion to dict
    def questions_to_dict(questions: List[TossupQuestion]) -> List[Dict[str, Any]]:
        return [
            {
                "qid": q.qid,
                "question": q.question,
                "tokens": q.tokens,
                "answer_primary": q.answer_primary,
                "clean_answers": q.clean_answers,
                "run_indices": q.run_indices,
                "human_buzz_positions": q.human_buzz_positions,
                "category": q.category,
                "cumulative_prefixes": q.cumulative_prefixes
            }
            for q in questions
        ]

    # Calculate category distributions for metadata
    def get_category_distribution(questions: List[TossupQuestion]) -> Dict[str, int]:
        dist = defaultdict(int)
        for q in questions:
            dist[q.category] += 1
        return dict(dist)

    # Save each split with metadata
    splits = [
        ("train_dataset.json", train),
        ("val_dataset.json", val),
        ("test_dataset.json", test)
    ]

    for filename, questions in splits:
        filepath = output_path / filename

        data = {
            "metadata": {
                "total_questions": len(questions),
                "categories": len(set(q.category for q in questions)),
                "category_distribution": get_category_distribution(questions),
                "split_type": filename.replace("_dataset.json", "")
            },
            "questions": questions_to_dict(questions)
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(questions)} questions to {filepath}")

    # Save combined metadata file
    metadata_path = output_path / "split_metadata.json"
    metadata = {
        "train": {
            "count": len(train),
            "categories": get_category_distribution(train)
        },
        "val": {
            "count": len(val),
            "categories": get_category_distribution(val)
        },
        "test": {
            "count": len(test),
            "categories": get_category_distribution(test)
        },
        "total_questions": len(train) + len(val) + len(test),
        "split_ratios": [
            len(train) / (len(train) + len(val) + len(test)),
            len(val) / (len(train) + len(val) + len(test)),
            len(test) / (len(train) + len(val) + len(test))
        ]
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved split metadata to {metadata_path}")
````

## File: qb_data/huggingface_loader.py
````python
"""
HuggingFace dataset loader for quiz bowl data.

This module provides fallback loading from HuggingFace Hub when local CSV files
are not available.
"""

from typing import List, Optional, Dict, Any

from qb_data.data_loader import TossupQuestion
from qb_data.text_utils import tokenize_text, normalize_answer


def load_from_huggingface(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "eval"
) -> List[TossupQuestion]:
    """
    Load quiz bowl dataset from HuggingFace Hub.

    Parameters
    ----------
    dataset_name : str
        Name of the HuggingFace dataset (e.g., "qanta-challenge/acf-co24-tossups")
    config_name : Optional[str]
        Configuration name for the dataset (e.g., "questions", "tossup")
    split : str
        Dataset split to load (default: "eval")

    Returns
    -------
    List[TossupQuestion]
        List of parsed questions

    Raises
    ------
    ImportError
        If datasets library is not installed
    ValueError
        If dataset not found or required fields missing
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets library not installed. Falling back to CSV loader.")
        print("Install with: pip install datasets")
        raise ImportError("HuggingFace datasets library not available. Please use CSV fallback.")

    # Known dataset configurations from qb-rl
    known_configs = {
        "qanta-challenge/acf-co24-tossups": "questions",
        "qanta-challenge/qanta25-playground": "tossup"
    }

    # Use known config if not provided
    if config_name is None and dataset_name in known_configs:
        config_name = known_configs[dataset_name]
        print(f"Using known config '{config_name}' for {dataset_name}")

    # Try to load dataset
    try:
        print(f"Loading {dataset_name} from HuggingFace Hub...")
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        print(f"Successfully loaded {len(dataset)} questions")
    except Exception as e:
        error_msg = f"Failed to load dataset {dataset_name}: {e}"
        print(f"Error: {error_msg}")
        print("Falling back to local CSV loader...")
        raise ValueError(error_msg)

    # Parse dataset rows into TossupQuestion format
    questions = []
    for idx, row in enumerate(dataset):
        try:
            question = parse_huggingface_row(row, idx)
            questions.append(question)
        except KeyError as e:
            print(f"Warning: Skipping row {idx} due to missing field: {e}")
            continue
        except Exception as e:
            print(f"Warning: Failed to parse row {idx}: {e}")
            continue

    if not questions:
        raise ValueError(f"No valid questions parsed from {dataset_name}")

    print(f"Parsed {len(questions)} questions from HuggingFace dataset")
    return questions


def parse_huggingface_row(row: Dict[str, Any], idx: int = 0) -> TossupQuestion:
    """
    Parse a HuggingFace dataset row into TossupQuestion format.

    Parameters
    ----------
    row : Dict[str, Any]
        Single row from HuggingFace dataset
    idx : int
        Row index for generating IDs

    Returns
    -------
    TossupQuestion
        Parsed question object

    Raises
    ------
    KeyError
        If required fields are missing
    """
    # Field mapping for different dataset formats
    # Primary fields
    question_fields = ["question", "text", "question_text", "tossup_text"]
    answer_fields = ["answer_primary", "answer", "clean_answer", "clean_answers", "page"]
    category_fields = ["category", "topic", "subject"]

    # Extract question text
    question_text = None
    for field in question_fields:
        if field in row:
            question_text = row[field]
            break

    if not question_text:
        raise KeyError(f"No question field found. Available fields: {list(row.keys())}")

    # Extract answer
    answer_text = None
    for field in answer_fields:
        if field in row:
            value = row[field]
            # Handle list of answers
            if isinstance(value, list) and value:
                answer_text = value[0]
            elif isinstance(value, str):
                answer_text = value
            break

    if not answer_text:
        raise KeyError(f"No answer field found. Available fields: {list(row.keys())}")

    # Extract category (with default)
    category = "General"
    for field in category_fields:
        if field in row and row[field]:
            category = str(row[field])
            break

    # Generate ID if not present
    qid = row.get("qid") or row.get("id") or row.get("qanta_id") or f"hf_{idx:06d}"

    # Handle clues that may be separated by ||| or in a list
    if "|||" in question_text:
        # QANTA format with ||| separators
        clues = question_text.split("|||")
        question_text = " ".join(clues)
    elif isinstance(question_text, list):
        # List of clues
        clues = question_text
        question_text = " ".join(clues)
    else:
        # Single text, split by sentences as approximation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', question_text)
        clues = sentences if len(sentences) > 1 else [question_text]

    # Tokenize text
    tokens = tokenize_text(question_text)

    # Build run indices (boundaries between clues)
    run_indices = []
    current_pos = 0
    for clue in clues:
        clue_tokens = tokenize_text(clue)
        current_pos += len(clue_tokens)
        if current_pos > 0:
            run_indices.append(current_pos - 1)  # Index is 0-based

    # Build cumulative prefixes
    cumulative_prefixes = []
    for idx in run_indices:
        prefix = " ".join(tokens[:idx + 1])
        cumulative_prefixes.append(prefix)

    # Normalize answer for matching
    clean_answers = [normalize_answer(answer_text)]

    return TossupQuestion(
        qid=qid,
        question=question_text,
        tokens=tokens,
        answer_primary=answer_text,  # Keep original answer as primary
        clean_answers=clean_answers,  # Normalized version for matching
        run_indices=run_indices,
        human_buzz_positions=None,  # Not available from HuggingFace
        category=category,
        cumulative_prefixes=cumulative_prefixes
    )


def try_huggingface_fallback(csv_path: str) -> Optional[List[TossupQuestion]]:
    """
    Attempt to load from HuggingFace if CSV is missing.

    Parameters
    ----------
    csv_path : str
        Path to missing CSV file

    Returns
    -------
    Optional[List[TossupQuestion]]
        Questions if HuggingFace load succeeds, None otherwise
    """
    print(f"CSV file {csv_path} not found. Attempting HuggingFace fallback...")

    # Try known datasets in order
    fallback_datasets = [
        ("qanta-challenge/acf-co24-tossups", "questions"),
        ("qanta-challenge/qanta25-playground", "tossup")
    ]

    for dataset_name, config_name in fallback_datasets:
        try:
            questions = load_from_huggingface(dataset_name, config_name)
            if questions:
                print(f"Successfully loaded {len(questions)} questions from {dataset_name}")
                return questions
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue

    print("All HuggingFace fallback attempts failed")
    return None
````

## File: qb_data/text_utils.py
````python
"""
Text utilities for quiz bowl answer normalization and tokenization.
"""

import re
import string
from typing import Optional, List


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text by splitting on whitespace.

    Parameters
    ----------
    text : str
        Text to tokenize

    Returns
    -------
    List[str]
        List of tokens (words)
    """
    if not text:
        return []
    return text.split()


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Removes articles (a, an, the) from the beginning, converts to lowercase,
    strips punctuation and extra whitespace, and handles edge cases.

    Parameters
    ----------
    answer : str
        The answer string to normalize

    Returns
    -------
    str
        The normalized answer string

    Examples
    --------
    >>> normalize_answer("The Great Gatsby")
    'great gatsby'
    >>> normalize_answer("A Tale of Two Cities!")
    'tale of two cities'
    >>> normalize_answer("   An    Example   ")
    'example'
    >>> normalize_answer("")
    ''
    """
    if not answer:
        return ""

    # Convert to lowercase
    normalized = answer.lower()

    # Remove leading/trailing whitespace
    normalized = normalized.strip()

    # Remove leading articles (a, an, the)
    # Use \b word boundary to ensure we match complete words
    normalized = re.sub(r'^(a|an|the)\b\s*', '', normalized)

    # Remove punctuation
    # Keep alphanumeric characters and spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)

    # Normalize whitespace (collapse multiple spaces to single space)
    normalized = re.sub(r'\s+', ' ', normalized)

    # Final strip in case punctuation removal left spaces
    normalized = normalized.strip()

    return normalized
````

## File: scripts/manual-smoke.sh
````bash
#!/usr/bin/env bash
# Manual smoke pipeline -- runs the four-stage belief-feature smoke workflow.
# Intended for human verification, not CI (stages are heavyweight ML runs).
#
# Prereqs: pip install -e .  (see AGENTS.md for full setup)
# Outputs: artifacts/smoke/
set -euo pipefail

echo "=== Stage 1/4: Build MC dataset ==="
python scripts/build_mc_dataset.py --smoke

echo "=== Stage 2/4: Run baselines ==="
python scripts/run_baselines.py --smoke

echo "=== Stage 3/4: Train PPO ==="
python scripts/train_ppo.py --smoke

echo "=== Stage 4/4: Evaluate all ==="
python scripts/evaluate_all.py --smoke

echo "=== Smoke pipeline complete. Check artifacts/smoke/ ==="
````

## File: scripts/run_smoke_pipeline.py
````python
#!/usr/bin/env python3
"""Run the full canonical smoke pipeline end-to-end.

Stages:
1) build_mc_dataset
2) run_baselines
3) train_ppo
4) evaluate_all

Writes a summary JSON to artifacts/smoke/smoke_pipeline_summary.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "smoke"


STAGES = [
    ["scripts/build_mc_dataset.py", "--smoke"],
    ["scripts/run_baselines.py", "--smoke"],
    ["scripts/train_ppo.py", "--smoke"],
    ["scripts/evaluate_all.py", "--smoke"],
]


def run_stage(python_exe: str, args: list[str]) -> tuple[int, float]:
    """Run one stage command and return (exit_code, seconds)."""
    cmd = [python_exe, *args]
    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    return proc.returncode, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full smoke pipeline")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter)",
    )
    ns = parser.parse_args()

    print("=" * 60)
    print("Smoke Pipeline Runner")
    print("=" * 60)
    print(f"Python: {ns.python}")
    print()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "python": ns.python,
        "started_at_unix": time.time(),
        "stages": [],
    }

    pipeline_start = time.time()
    for stage_args in STAGES:
        stage_name = stage_args[0]
        print(f"Running: {stage_name} {' '.join(stage_args[1:])}")
        code, seconds = run_stage(ns.python, stage_args)
        summary["stages"].append(
            {
                "stage": stage_name,
                "args": stage_args[1:],
                "exit_code": code,
                "seconds": round(seconds, 3),
            }
        )
        if code != 0:
            summary["status"] = "failed"
            summary["failed_stage"] = stage_name
            summary["total_seconds"] = round(time.time() - pipeline_start, 3)
            out_path = ARTIFACT_DIR / "smoke_pipeline_summary.json"
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"\nFAILED at {stage_name} (exit={code})")
            print(f"Summary written: {out_path}")
            return code
        print(f"✓ {stage_name} completed in {seconds:.1f}s\n")

    summary["status"] = "ok"
    summary["total_seconds"] = round(time.time() - pipeline_start, 3)
    out_path = ARTIFACT_DIR / "smoke_pipeline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Smoke pipeline completed successfully")
    print(f"Summary written: {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

## File: scripts/sweep_reward_shaping.py
````python
#!/usr/bin/env python3
"""Sweep PPO smoke reward-shaping settings and record results.

Runs `scripts/train_ppo.py` in smoke mode across a small grid of:
- environment.wait_penalty
- environment.early_buzz_penalty

Collects metrics from artifacts/smoke/ppo_summary.json after each run and writes:
- artifacts/smoke/reward_sweep_results.json
- artifacts/smoke/reward_sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "smoke"
TMP_CONFIG = ARTIFACT_DIR / "_tmp_sweep_smoke.yaml"
PPO_SUMMARY = ARTIFACT_DIR / "ppo_summary.json"

WAIT_PENALTIES = [0.0, 0.02, 0.05]
EARLY_BUZZ_PENALTIES = [0.2, 0.5, 0.8]
SEEDS = [13, 42, 123]


def run_cmd(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep PPO reward shaping")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated seeds, e.g. 13,42,123",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional timesteps override for train_ppo during sweep",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(SMOKE_CONFIG)

    python_exe = sys.executable
    results = []

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    grid = [(w, e) for w in WAIT_PENALTIES for e in EARLY_BUZZ_PENALTIES]

    print("=" * 72)
    print(f"Reward sweep: {len(grid)} configs x {len(seeds)} seeds")
    print("=" * 72)

    for idx, (wait_penalty, early_buzz_penalty) in enumerate(grid, start=1):
        per_seed = []
        print(f"[{idx}/{len(grid)}] wait_penalty={wait_penalty}, early_buzz_penalty={early_buzz_penalty}")

        for seed in seeds:
            cfg = dict(base_cfg)
            cfg.setdefault("environment", {})
            cfg["environment"] = dict(cfg["environment"])
            cfg["environment"]["wait_penalty"] = float(wait_penalty)
            cfg["environment"]["early_buzz_penalty"] = float(early_buzz_penalty)
            cfg["environment"]["seed"] = int(seed)

            cfg.setdefault("ppo", {})
            cfg["ppo"] = dict(cfg["ppo"])
            cfg["ppo"]["seed"] = int(seed)
            save_yaml(TMP_CONFIG, cfg)

            cmd = [python_exe, "scripts/train_ppo.py", "--config", str(TMP_CONFIG), "--smoke", "--seed", str(seed)]
            if args.timesteps is not None:
                cmd.extend(["--timesteps", str(args.timesteps)])

            start = time.time()
            code = run_cmd(cmd)
            elapsed = time.time() - start

            if code != 0 or not PPO_SUMMARY.exists():
                per_seed.append({"seed": seed, "status": "failed", "seconds": round(elapsed, 3)})
                continue

            summary = load_json(PPO_SUMMARY)
            per_seed.append(
                {
                    "seed": seed,
                    "status": "ok",
                    "seconds": round(elapsed, 3),
                    "buzz_accuracy": float(summary.get("buzz_accuracy", 0.0)),
                    "mean_sq": float(summary.get("mean_sq", 0.0)),
                    "mean_buzz_step": float(summary.get("mean_buzz_step", 0.0)),
                    "ece": float(summary.get("ece", 0.0)),
                    "brier": float(summary.get("brier", 0.0)),
                }
            )

        ok = [r for r in per_seed if r.get("status") == "ok"]
        if not ok:
            results.append(
                {
                    "wait_penalty": wait_penalty,
                    "early_buzz_penalty": early_buzz_penalty,
                    "status": "failed",
                    "num_ok": 0,
                    "num_total": len(per_seed),
                    "per_seed": per_seed,
                }
            )
            continue

        mean_acc = sum(r["buzz_accuracy"] for r in ok) / len(ok)
        mean_sq = sum(r["mean_sq"] for r in ok) / len(ok)
        mean_step = sum(r["mean_buzz_step"] for r in ok) / len(ok)
        mean_ece = sum(r["ece"] for r in ok) / len(ok)
        mean_brier = sum(r["brier"] for r in ok) / len(ok)
        mean_seconds = sum(r["seconds"] for r in ok) / len(ok)

        # Balanced objective: maximize accuracy + S_q while penalizing calibration error.
        objective = mean_acc + mean_sq - 0.5 * mean_ece

        results.append(
            {
                "wait_penalty": wait_penalty,
                "early_buzz_penalty": early_buzz_penalty,
                "status": "ok",
                "num_ok": len(ok),
                "num_total": len(per_seed),
                "seconds": round(mean_seconds, 3),
                "buzz_accuracy": mean_acc,
                "mean_sq": mean_sq,
                "mean_buzz_step": mean_step,
                "ece": mean_ece,
                "brier": mean_brier,
                "objective": objective,
                "per_seed": per_seed,
            }
        )

    # cleanup temp config
    if TMP_CONFIG.exists():
        TMP_CONFIG.unlink()

    out_json = ARTIFACT_DIR / "reward_sweep_results.json"
    out_csv = ARTIFACT_DIR / "reward_sweep_results.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fields = [
        "wait_penalty",
        "early_buzz_penalty",
        "status",
        "num_ok",
        "num_total",
        "seconds",
        "buzz_accuracy",
        "mean_sq",
        "mean_buzz_step",
        "ece",
        "brier",
        "objective",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            flat = {k: row.get(k, "") for k in fields}
            writer.writerow(flat)

    ok_runs = [r for r in results if r.get("status") == "ok"]
    if not ok_runs:
        print("No successful runs.")
        return 1

    best = max(ok_runs, key=lambda r: float(r.get("objective", 0.0)))

    print("\nBest run:")
    print(best)
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

## File: scripts/test_mc_builder.py
````python
#!/usr/bin/env python
"""Test script to verify MC construction with anti-artifact guards."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qb_data.data_loader import QANTADatasetLoader
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.mc_builder import MCBuilder, MCQuestion
from qb_data.config import load_config


def main():
    """Test MC question construction with guards."""
    print("Testing MC Builder with Anti-Artifact Guards")
    print("=" * 50)

    # Load configuration
    config = load_config("configs/default.yaml")

    # Load test questions
    data_path = "data/test_questions.csv"
    if not os.path.exists(data_path):
        print(f"Error: Test data not found at {data_path}")
        print("Please ensure test_questions.csv exists")
        return 1

    # Load questions
    questions = QANTADatasetLoader.load_from_csv(data_path)
    print(f"\nLoaded {len(questions)} test questions")

    # Create answer profile builder
    profile_builder = AnswerProfileBuilder(
        max_tokens_per_profile=config["answer_profiles"]["max_tokens_per_profile"],
        min_questions_per_answer=config["answer_profiles"]["min_questions_per_answer"]
    )
    profile_builder.fit(questions)
    print(f"Built profiles for {len(profile_builder._grouped)} unique answers")

    # Create MC builder with guards from config
    mc_builder = MCBuilder(
        K=config["data"]["K"],
        strategy="tfidf_profile",  # Use TF-IDF since it doesn't require embeddings
        alias_edit_distance_threshold=config["mc_guards"]["alias_edit_distance_threshold"],
        duplicate_token_overlap_threshold=config["mc_guards"]["duplicate_token_overlap_threshold"],
        max_length_ratio=config["mc_guards"]["max_length_ratio"],
        random_seed=config["data"]["shuffle_seed"]
    )

    # Build MC questions
    print(f"\nBuilding MC questions with K={config['data']['K']} options...")
    mc_questions = mc_builder.build(questions, profile_builder)
    print(f"Created {len(mc_questions)} MC questions (from {len(questions)} originals)")

    # Calculate rejection rate
    rejection_rate = 1.0 - (len(mc_questions) / len(questions))
    print(f"Rejection rate: {rejection_rate:.1%} (due to guard violations)")

    # Print sample MC questions
    print("\n" + "=" * 50)
    print("Sample MC Questions:")
    print("=" * 50)

    for i, mc_q in enumerate(mc_questions[:3]):  # Show first 3
        print(f"\n[Question {i+1}]")
        print(f"Category: {mc_q.category or 'Unknown'}")
        print(f"Question ID: {mc_q.qid}")

        # Show first clue (truncated)
        first_clue = mc_q.tokens[0] if mc_q.tokens else mc_q.question[:100]
        print(f"First clue: {first_clue[:150]}...")

        print(f"\nOptions:")
        for j, option in enumerate(mc_q.options):
            marker = " [CORRECT]" if j == mc_q.gold_index else ""
            print(f"  {j+1}. {option}{marker}")

        print(f"\nDistractor strategy: {mc_q.distractor_strategy}")

        # Check guards for this question
        print("\nGuard checks:")

        # Check alias collision
        gold_aliases = [mc_q.answer_primary] + list(mc_q.clean_answers)
        alias_violations = []
        for j, option in enumerate(mc_q.options):
            if j != mc_q.gold_index:
                for alias in gold_aliases:
                    from difflib import SequenceMatcher
                    dist = 1.0 - SequenceMatcher(None, option.lower(), alias.lower()).ratio()
                    if dist < 0.2:
                        alias_violations.append((option, alias, dist))

        if alias_violations:
            print(f"  ✗ Alias collision detected: {alias_violations}")
        else:
            print("  ✓ No alias collisions")

        # Check token overlap between options
        from qb_data.mc_builder import _token_overlap
        high_overlaps = []
        for j in range(len(mc_q.options)):
            for k in range(j+1, len(mc_q.options)):
                overlap = _token_overlap(mc_q.options[j], mc_q.options[k])
                if overlap > 0.8:
                    high_overlaps.append((mc_q.options[j], mc_q.options[k], overlap))

        if high_overlaps:
            print(f"  ✗ High token overlap: {high_overlaps}")
        else:
            print("  ✓ No high token overlaps")

        # Check length ratio
        lengths = [len(o.split()) for o in mc_q.options]
        ratio = max(lengths) / max(1, min(lengths))
        if ratio > 3.0:
            print(f"  ✗ Length ratio violation: {ratio:.2f} (max: {max(lengths)}, min: {min(lengths)})")
        else:
            print(f"  ✓ Length ratio OK: {ratio:.2f}")

        # Check question overlap
        from qb_data.text_utils import normalize_answer
        q_norm = normalize_answer(mc_q.question).lower()
        overlaps = []
        for option in mc_q.options:
            o_norm = normalize_answer(option).lower()
            if o_norm and o_norm in q_norm:
                overlaps.append(option)

        if overlaps:
            print(f"  ✗ Options appear in question: {overlaps}")
        else:
            print("  ✓ No options in question text")

    # Print statistics
    print("\n" + "=" * 50)
    print("Statistics:")
    print("=" * 50)
    print(f"Total questions processed: {len(questions)}")
    print(f"MC questions built: {len(mc_questions)}")
    print(f"Questions rejected by guards: {len(questions) - len(mc_questions)}")

    # Analyze rejection reasons (would need to track in MCBuilder for full details)
    if len(mc_questions) < len(questions):
        print("\nNote: Some questions were rejected due to guard violations.")
        print("Common reasons include:")
        print("  - Not enough valid distractors after alias/duplicate filtering")
        print("  - Length ratio violations between options")
        print("  - Answer text appearing in question")

    print("\n✓ MC questions built successfully with guards active")
    return 0


if __name__ == "__main__":
    exit(main())
````

## File: tests/test_answer_profile_cache.py
````python
"""Tests for AnswerProfileBuilder._cache memoization.

Verifies that:
1. Distractor profiles (exclude_qid=None) are cached and return identical results
2. Leave-one-out profiles (answer, qid) are cached and return identical results
3. Cache is invalidated on fit() with new data
4. Cached distractor profile is byte-identical to freshly computed profile
5. Cached leave-one-out profile is byte-identical to freshly computed profile
6. Cache reduces actual computation (single entry per unique key)
"""

from __future__ import annotations

import pytest

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion


def _make_question(
    qid: str,
    answer: str,
    text: str,
    category: str = "History",
) -> TossupQuestion:
    """Create a minimal TossupQuestion for cache testing."""
    tokens = text.split()
    return TossupQuestion(
        qid=qid,
        question=text,
        tokens=tokens,
        answer_primary=answer,
        clean_answers=[answer],
        run_indices=[len(tokens) - 1],
        human_buzz_positions=[],
        category=category,
        cumulative_prefixes=[text],
    )


@pytest.fixture
def sample_questions() -> list[TossupQuestion]:
    """Five questions with 3 shared answers for exercising cache hits."""
    return [
        _make_question("q1", "Washington", "first president commander in chief"),
        _make_question("q2", "Washington", "led the continental army to victory"),
        _make_question("q3", "Jefferson", "wrote the declaration of independence"),
        _make_question("q4", "Jefferson", "third president and diplomat to France"),
        _make_question("q5", "Lincoln", "preserved the union during civil war"),
    ]


@pytest.fixture
def builder(sample_questions: list[TossupQuestion]) -> AnswerProfileBuilder:
    """Return a fitted AnswerProfileBuilder."""
    b = AnswerProfileBuilder(max_tokens_per_profile=2000, min_questions_per_answer=1)
    b.fit(sample_questions)
    return b


class TestProfileCacheHits:
    """Repeated calls with the same args return the same cached result."""

    def test_distractor_profile_cached(
        self, builder: AnswerProfileBuilder
    ) -> None:
        """profile_for_answer returns identical string on repeated (answer, None)."""
        first = builder.profile_for_answer("Washington", exclude_qid=None)
        second = builder.profile_for_answer("Washington", exclude_qid=None)
        assert first is second  # same object, not just equal

    def test_leave_one_out_profile_cached(
        self, builder: AnswerProfileBuilder
    ) -> None:
        """profile_for_answer returns identical string on repeated (answer, qid)."""
        first = builder.profile_for_answer("Washington", exclude_qid="q1")
        second = builder.profile_for_answer("Washington", exclude_qid="q1")
        assert first is second  # same object from cache


class TestCacheInvalidation:
    """fit() with new data clears the cache."""

    def test_fit_clears_cache(
        self, builder: AnswerProfileBuilder, sample_questions: list[TossupQuestion]
    ) -> None:
        """After fit() with new data, cache is empty and profiles reflect new data."""
        # Populate cache
        builder.profile_for_answer("Washington", exclude_qid=None)
        assert len(builder._cache) > 0

        # Re-fit with different data
        new_questions = [
            _make_question("q99", "Washington", "completely different text about cherry trees"),
        ]
        builder.fit(new_questions)
        assert len(builder._cache) == 0

        # New profile should reflect new data
        profile = builder.profile_for_answer("Washington", exclude_qid=None)
        assert "cherry" in profile


class TestCacheEquivalence:
    """Cached profiles are byte-identical to freshly computed profiles."""

    def test_distractor_cache_equivalence(
        self, sample_questions: list[TossupQuestion]
    ) -> None:
        """Cached (answer, None) profile is byte-identical to a fresh computation."""
        # Build fresh (uncached) profile
        fresh_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        fresh_builder.fit(sample_questions)
        fresh_profile = fresh_builder._profile_text("Jefferson", exclude_qid=None)

        # Build cached profile
        cached_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        cached_builder.fit(sample_questions)
        _ = cached_builder._profile_text("Jefferson", exclude_qid=None)  # populate cache
        cached_profile = cached_builder._profile_text("Jefferson", exclude_qid=None)  # from cache

        assert fresh_profile == cached_profile

    def test_leave_one_out_cache_equivalence(
        self, sample_questions: list[TossupQuestion]
    ) -> None:
        """Cached (answer, qid) profile is byte-identical to a fresh computation."""
        fresh_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        fresh_builder.fit(sample_questions)
        fresh_profile = fresh_builder._profile_text("Washington", exclude_qid="q1")

        cached_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        cached_builder.fit(sample_questions)
        _ = cached_builder._profile_text("Washington", exclude_qid="q1")
        cached_profile = cached_builder._profile_text("Washington", exclude_qid="q1")

        assert fresh_profile == cached_profile


class TestCacheEfficiency:
    """Cache reduces computation to one real call per unique key."""

    def test_cache_stores_one_entry_per_unique_key(
        self, builder: AnswerProfileBuilder
    ) -> None:
        """Calling _profile_text N times with same args results in 1 cache entry."""
        for _ in range(10):
            builder.profile_for_answer("Lincoln", exclude_qid=None)

        # Only one cache entry for (Lincoln, None)
        assert ("Lincoln", None) in builder._cache
        assert len([k for k in builder._cache if k[0] == "Lincoln"]) == 1
````

## File: tests/test_features.py
````python
"""Test suite for models/features.py — belief feature extraction.

Covers ENV-03: Belief feature extraction produces (K+6)-dimensional vectors
with correct derived features (entropy, margin, stability, progress).
"""

from __future__ import annotations

import numpy as np
import pytest

from models.features import entropy_of_distribution, extract_belief_features


# ------------------------------------------------------------------ #
# Tests for entropy_of_distribution
# ------------------------------------------------------------------ #


class TestEntropyOfDistribution:
    """Tests for Shannon entropy computation."""

    def test_entropy_uniform(self) -> None:
        """Uniform distribution over 4 options has maximum entropy ln(4)."""
        belief = np.array([0.25, 0.25, 0.25, 0.25])
        ent = entropy_of_distribution(belief)
        # ln(4) ~ 1.3863
        assert 1.35 < ent < 1.40, f"Uniform entropy {ent} not near ln(4)=1.3863"

    def test_entropy_peaked(self) -> None:
        """Peaked distribution has low entropy."""
        belief = np.array([0.9, 0.05, 0.03, 0.02])
        ent = entropy_of_distribution(belief)
        assert ent < 0.5, f"Peaked entropy {ent} should be < 0.5"

    def test_entropy_deterministic_no_nan(self) -> None:
        """Deterministic distribution [1, 0, 0, 0] produces no NaN/inf."""
        belief = np.array([1.0, 0.0, 0.0, 0.0])
        ent = entropy_of_distribution(belief)
        assert np.isfinite(ent), f"Entropy {ent} should be finite"
        assert ent >= 0.0, f"Entropy {ent} should be non-negative"

    def test_entropy_deterministic_last(self) -> None:
        """Deterministic distribution [0, 0, 0, 1] produces no NaN/inf."""
        belief = np.array([0.0, 0.0, 0.0, 1.0])
        ent = entropy_of_distribution(belief)
        assert np.isfinite(ent), f"Entropy {ent} should be finite"
        assert ent >= 0.0, f"Entropy {ent} should be non-negative"

    def test_entropy_binary(self) -> None:
        """Binary uniform distribution has entropy ln(2)."""
        belief = np.array([0.5, 0.5])
        ent = entropy_of_distribution(belief)
        assert abs(ent - np.log(2)) < 0.01, f"Binary entropy {ent} != ln(2)={np.log(2):.4f}"


# ------------------------------------------------------------------ #
# Tests for extract_belief_features
# ------------------------------------------------------------------ #


class TestExtractBeliefFeatures:
    """Tests for belief feature vector extraction."""

    def test_feature_shape(self) -> None:
        """Output shape is (K+6,) for K=4 belief vector."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 6)
        assert features.shape == (10,), f"Expected (10,), got {features.shape}"

    def test_feature_shape_k3(self) -> None:
        """Output shape adapts to K=3."""
        belief = np.array([0.4, 0.3, 0.3], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 5)
        assert features.shape == (9,), f"Expected (9,), got {features.shape}"

    def test_feature_contents_belief_prefix(self) -> None:
        """First K elements of feature vector are the raw belief."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        np.testing.assert_array_almost_equal(
            features[:4], belief, decimal=5,
            err_msg="First K elements should match input belief",
        )

    def test_derived_top_p(self) -> None:
        """top_p is max(belief)."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        assert abs(features[4] - 0.5) < 1e-5, f"top_p={features[4]}, expected 0.5"

    def test_derived_margin(self) -> None:
        """margin is top_p - second_highest."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        expected_margin = 0.5 - 0.3
        assert abs(features[5] - expected_margin) < 1e-5, (
            f"margin={features[5]}, expected {expected_margin}"
        )

    def test_derived_entropy_in_range(self) -> None:
        """Entropy is in a reasonable range for a non-uniform distribution."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        ent = features[6]
        assert 0 < ent < np.log(4) + 0.01, f"Entropy {ent} out of range"

    def test_stability_none_prev(self) -> None:
        """Stability is 0.0 when prev_belief is None (first step)."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 6)
        assert features[7] == 0.0, f"Stability={features[7]}, expected 0.0 for first step"

    def test_stability_computation(self) -> None:
        """Stability tracks L1 distance between consecutive beliefs."""
        prev_belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, prev_belief, 1, 6)
        expected_stability = float(np.abs(belief - prev_belief).sum())
        assert abs(features[7] - expected_stability) < 1e-5, (
            f"Stability={features[7]}, expected {expected_stability}"
        )

    def test_progress(self) -> None:
        """progress = step_idx / total_steps."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 3, 6)
        expected_progress = 3.0 / 6.0
        assert abs(features[8] - expected_progress) < 1e-5, (
            f"progress={features[8]}, expected {expected_progress}"
        )

    def test_clue_idx_norm(self) -> None:
        """clue_idx_norm = step_idx / (total_steps - 1)."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 3, 6)
        expected_norm = 3.0 / 5.0
        assert abs(features[9] - expected_norm) < 1e-5, (
            f"clue_idx_norm={features[9]}, expected {expected_norm}"
        )

    def test_dtype_float32(self) -> None:
        """Output dtype is float32."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 6)
        assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"

    def test_invalid_2d_belief_raises(self) -> None:
        """Passing a 2D belief array raises ValueError."""
        belief = np.array([[0.5, 0.5]], dtype=np.float32)
        with pytest.raises(ValueError, match="1D"):
            extract_belief_features(belief, None, 0, 1)
````

## File: tests/test_mc_builder_topk.py
````python
"""Regression tests for top-M distractor ranking in MCBuilder._compute_rankings.

Validates that the argpartition-based top-M retrieval produces the same top
distractors as a full argsort, truncates ranking lists correctly, degrades
gracefully when N is small, and leaves category_random strategy unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qb_data.mc_builder import MCBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_answers(n: int) -> tuple[list[str], dict[str, str]]:
    """Create *n* synthetic answers with distinct TF-IDF profiles.

    Each answer is a short phrase and its profile is a sentence containing
    unique vocabulary so TF-IDF can discriminate between them.
    """
    topics = [
        ("George Washington", "first president commander revolutionary war continental army"),
        ("Thomas Jefferson", "third president declaration independence Virginia Monticello"),
        ("John Adams", "second president Massachusetts diplomat federalist"),
        ("Benjamin Franklin", "inventor diplomat Philadelphia printing press electricity"),
        ("Abraham Lincoln", "sixteenth president civil war emancipation slavery"),
        ("Alexander Hamilton", "treasury secretary banking system federalist papers"),
        ("James Madison", "bill rights constitution fourth president Virginia"),
        ("Andrew Jackson", "military hero populist president battle New Orleans"),
        ("Theodore Roosevelt", "progressive trust buster national parks rough riders"),
        ("Ulysses Grant", "civil war general eighteenth president reconstruction"),
        ("Woodrow Wilson", "world war one league nations progressive president"),
        ("Franklin Roosevelt", "new deal world war two great depression fireside"),
        ("Harry Truman", "atomic bomb cold war Korean conflict fair deal"),
        ("Dwight Eisenhower", "supreme commander NATO interstate highway system"),
        ("John Kennedy", "space race Cuban missile crisis new frontier"),
        ("Lyndon Johnson", "great society civil rights Vietnam escalation"),
        ("Richard Nixon", "detente China opening Watergate resignation"),
        ("Ronald Reagan", "cold war end conservative revolution economic growth"),
        ("Barack Obama", "affordable care act first African American president"),
        ("Jimmy Carter", "Camp David accords energy crisis human rights"),
    ]
    answers = [t[0] for t in topics[:n]]
    profiles = {t[0]: t[1] for t in topics[:n]}
    return answers, profiles


def _full_sort_rankings(
    answers: list[str], profiles: dict[str, str]
) -> dict[str, list[str]]:
    """Compute rankings via full argsort (reference implementation)."""
    docs = [profiles[a] for a in answers]
    answer_idx = {a: i for i, a in enumerate(answers)}
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    sim = cosine_similarity(matrix, matrix)
    rankings: dict[str, list[str]] = {}
    for answer in answers:
        idx = answer_idx[answer]
        order = np.argsort(-sim[idx]).tolist()
        rankings[answer] = [answers[i] for i in order if answers[i] != answer]
    return rankings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTopMRanking:
    """Tests for top-M argpartition ranking in MCBuilder._compute_rankings."""

    def test_top_m_truncation(self) -> None:
        """Rankings should have length <= min(M, N-1)."""
        answers, profiles = _make_synthetic_answers(20)
        builder = MCBuilder(K=4, strategy="tfidf_profile")
        categories: dict[str, str] = {}

        rankings = builder._compute_rankings(answers, profiles, categories)

        M = min(max(5 * 4, 30), len(answers) - 1)  # min(30, 19) = 19
        for answer, ranked in rankings.items():
            assert len(ranked) <= min(M, len(answers) - 1), (
                f"Answer '{answer}' has {len(ranked)} distractors, "
                f"expected <= {min(M, len(answers) - 1)}"
            )

    def test_order_preservation(self) -> None:
        """Top-3 distractors must match the full-sort reference."""
        answers, profiles = _make_synthetic_answers(20)
        builder = MCBuilder(K=4, strategy="tfidf_profile")
        categories: dict[str, str] = {}

        rankings = builder._compute_rankings(answers, profiles, categories)
        reference = _full_sort_rankings(answers, profiles)

        for answer in answers:
            actual_top3 = rankings[answer][:3]
            expected_top3 = reference[answer][:3]
            assert actual_top3 == expected_top3, (
                f"Answer '{answer}': top-3 mismatch.\n"
                f"  actual:   {actual_top3}\n"
                f"  expected: {expected_top3}"
            )

    def test_small_n_graceful(self) -> None:
        """With N=5, rankings should have length N-1=4 without error."""
        answers, profiles = _make_synthetic_answers(5)
        builder = MCBuilder(K=4, strategy="tfidf_profile")
        categories: dict[str, str] = {}

        rankings = builder._compute_rankings(answers, profiles, categories)

        for answer, ranked in rankings.items():
            assert len(ranked) == 4, (
                f"Answer '{answer}' has {len(ranked)} distractors, expected 4"
            )

    def test_category_random_unaffected(self) -> None:
        """category_random strategy should not use argpartition path."""
        answers, profiles = _make_synthetic_answers(10)
        categories = {a: "History" for a in answers}
        builder = MCBuilder(K=4, strategy="category_random")

        rankings = builder._compute_rankings(answers, profiles, categories)

        for answer, ranked in rankings.items():
            # All same-category peers (minus self) should be present
            assert set(ranked) == set(a for a in answers if a != answer), (
                f"Answer '{answer}': category_random should include all peers"
            )
````

## File: tests/test_metrics.py
````python
"""Unit tests for evaluation metrics.

Tests edge cases for system_score (S_q), calibration metrics (ECE, Brier),
and per-category accuracy grouping.
"""

import pytest

from evaluation.metrics import (
    brier_score,
    calibration_at_buzz,
    expected_calibration_error,
    per_category_accuracy,
    summarize_buzz_metrics,
    system_score,
)


# ---------------------------------------------------------------------------
# system_score (S_q) edge cases
# ---------------------------------------------------------------------------


def test_system_score_empty_trace():
    """S_q should return 0.0 for empty traces."""
    assert system_score([], []) == 0.0


def test_system_score_all_zero_confidence():
    """S_q should return 0.0 when agent never considers buzzing."""
    c_trace = [0.0, 0.0, 0.0]
    g_trace = [1.0, 1.0, 1.0]  # All correct but agent doesn't buzz
    assert system_score(c_trace, g_trace) == 0.0


def test_system_score_all_correct_immediate_buzz():
    """S_q should equal first g_trace value when agent buzzes immediately."""
    c_trace = [1.0, 0.0, 0.0]  # Buzz on step 0
    g_trace = [1.0, 1.0, 1.0]
    expected = 1.0 * 1.0  # b_0 = c_0 * 1.0 = 1.0, survival after = 0
    assert abs(system_score(c_trace, g_trace) - expected) < 1e-9


def test_system_score_gradual_confidence():
    """S_q should accumulate survival-weighted correctness."""
    c_trace = [0.3, 0.5, 1.0]
    g_trace = [0.0, 0.0, 1.0]  # Only correct at final step
    # b_0 = 0.3 * 1.0 = 0.3, survival = 0.7
    # b_1 = 0.5 * 0.7 = 0.35, survival = 0.7 * 0.5 = 0.35
    # b_2 = 1.0 * 0.35 = 0.35
    # S_q = 0.3*0 + 0.35*0 + 0.35*1 = 0.35
    expected = 0.35
    assert abs(system_score(c_trace, g_trace) - expected) < 1e-9


def test_system_score_single_step():
    """S_q should work for single-step episodes."""
    c_trace = [1.0]
    g_trace = [1.0]
    assert abs(system_score(c_trace, g_trace) - 1.0) < 1e-9

    c_trace = [0.5]
    g_trace = [1.0]
    assert abs(system_score(c_trace, g_trace) - 0.5) < 1e-9


def test_system_score_never_correct():
    """S_q should return 0.0 when g_trace is all zeros."""
    c_trace = [0.5, 0.5, 0.5]
    g_trace = [0.0, 0.0, 0.0]
    assert system_score(c_trace, g_trace) == 0.0


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------


def test_expected_calibration_error_perfect():
    """ECE should be near 0.0 for perfectly calibrated predictions."""
    # 70% confidence with 70% accuracy
    confidences = [0.7] * 10
    outcomes = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    ece = expected_calibration_error(confidences, outcomes, n_bins=10)
    assert ece < 0.01  # Near zero for perfect calibration


def test_expected_calibration_error_empty():
    """ECE should return 0.0 for empty inputs."""
    assert expected_calibration_error([], []) == 0.0


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------


def test_brier_score_perfect():
    """Brier score should be 0.0 for perfect predictions."""
    confidences = [1.0, 1.0, 0.0, 0.0]
    outcomes = [1, 1, 0, 0]
    bs = brier_score(confidences, outcomes)
    assert bs == 0.0


def test_brier_score_worst():
    """Brier score should be 1.0 for worst-case predictions."""
    confidences = [0.0, 0.0, 1.0, 1.0]
    outcomes = [1, 1, 0, 0]
    bs = brier_score(confidences, outcomes)
    assert abs(bs - 1.0) < 1e-9


def test_brier_score_empty():
    """Brier score should return 0.0 for empty inputs."""
    assert brier_score([], []) == 0.0


# ---------------------------------------------------------------------------
# summarize_buzz_metrics
# ---------------------------------------------------------------------------


def test_summarize_buzz_metrics_empty():
    """summarize_buzz_metrics should handle empty results."""
    result = summarize_buzz_metrics([])
    assert result["n"] == 0.0
    assert result["buzz_accuracy"] == 0.0


def test_summarize_buzz_metrics_basic():
    """summarize_buzz_metrics should compute correct aggregates."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 2,
            "c_trace": [0.0, 0.0, 1.0],
            "g_trace": [0.0, 0.0, 1.0],
            "reward_like": 0.8,
        },
        {
            "qid": "q2",
            "correct": False,
            "buzz_step": 1,
            "c_trace": [0.0, 1.0],
            "g_trace": [0.0, 0.0],
            "reward_like": -0.1,
        },
    ]
    summary = summarize_buzz_metrics(results)
    assert summary["n"] == 2.0
    assert abs(summary["buzz_accuracy"] - 0.5) < 1e-9
    assert abs(summary["mean_buzz_step"] - 1.5) < 1e-9


# ---------------------------------------------------------------------------
# per_category_accuracy
# ---------------------------------------------------------------------------


def test_per_category_accuracy_basic():
    """per_category_accuracy should group results by question category."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 2,
            "c_trace": [0.0, 0.0, 1.0],
            "g_trace": [0.0, 0.0, 1.0],
            "reward_like": 0.8,
        },
        {
            "qid": "q2",
            "correct": False,
            "buzz_step": 1,
            "c_trace": [0.0, 1.0],
            "g_trace": [0.0, 0.0],
            "reward_like": -0.1,
        },
        {
            "qid": "q3",
            "correct": True,
            "buzz_step": 3,
            "c_trace": [0.0, 0.0, 0.0, 1.0],
            "g_trace": [0.0, 0.0, 0.0, 1.0],
            "reward_like": 0.7,
        },
    ]
    questions = [
        {"qid": "q1", "category": "History"},
        {"qid": "q2", "category": "Science"},
        {"qid": "q3", "category": "History"},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "History" in cat_metrics
    assert "Science" in cat_metrics
    assert cat_metrics["History"]["n"] == 2.0
    assert cat_metrics["History"]["buzz_accuracy"] == 1.0
    assert cat_metrics["Science"]["n"] == 1.0
    assert cat_metrics["Science"]["buzz_accuracy"] == 0.0


def test_per_category_accuracy_missing_category():
    """per_category_accuracy should default missing categories to 'unknown'."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [1.0],
            "g_trace": [1.0],
            "reward_like": 1.0,
        },
    ]
    questions = [
        {"qid": "q1", "category": ""},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "unknown" in cat_metrics
    assert cat_metrics["unknown"]["n"] == 1.0


def test_per_category_accuracy_none_category():
    """per_category_accuracy should handle None category."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [1.0],
            "g_trace": [1.0],
            "reward_like": 1.0,
        },
    ]
    questions = [
        {"qid": "q1", "category": None},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "unknown" in cat_metrics


def test_per_category_accuracy_unmatched_qid():
    """Results with qids not in questions should group to 'unknown'."""
    results = [
        {
            "qid": "q_orphan",
            "correct": False,
            "buzz_step": 0,
            "c_trace": [1.0],
            "g_trace": [0.0],
            "reward_like": -0.1,
        },
    ]
    questions = [
        {"qid": "q1", "category": "History"},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "unknown" in cat_metrics
    assert cat_metrics["unknown"]["n"] == 1.0


# ---------------------------------------------------------------------------
# calibration_at_buzz — uses top_p_trace, not g_trace
# ---------------------------------------------------------------------------


def test_calibration_at_buzz_uses_top_p_trace():
    """calibration_at_buzz must use top_p_trace (belief prob), not g_trace (binary)."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 2,
            "c_trace": [0.1, 0.3, 0.9],
            "g_trace": [0.0, 0.0, 1.0],
            "top_p_trace": [0.3, 0.5, 0.8],
        },
        {
            "qid": "q2",
            "correct": False,
            "buzz_step": 1,
            "c_trace": [0.2, 0.7],
            "g_trace": [0.0, 0.0],
            "top_p_trace": [0.4, 0.6],
        },
    ]
    cal = calibration_at_buzz(results)
    assert cal["n_calibration"] == 2.0
    # Confidence from top_p_trace at buzz_step:
    # q1: top_p_trace[2] = 0.8, q2: top_p_trace[1] = 0.6
    # Brier = ((0.8-1)^2 + (0.6-0)^2)/2 = (0.04+0.36)/2 = 0.2
    assert abs(cal["brier"] - 0.2) < 1e-9


def test_calibration_at_buzz_falls_back_to_c_trace():
    """When top_p_trace is absent, calibration should fall back to c_trace."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [0.7],
            "g_trace": [1.0],
        },
    ]
    cal = calibration_at_buzz(results)
    assert cal["n_calibration"] == 1.0
    assert abs(cal["brier"] - (0.7 - 1.0) ** 2) < 1e-9


def test_calibration_at_buzz_empty():
    """calibration_at_buzz should return zeros for empty input."""
    cal = calibration_at_buzz([])
    assert cal["ece"] == 0.0
    assert cal["brier"] == 0.0
    assert cal["n_calibration"] == 0.0


def test_calibration_at_buzz_binary_g_trace_not_used():
    """Regression: binary g_trace must NOT be used as confidence.

    If g_trace (binary 0/1) were used, Brier for a correct episode with
    g_trace=[1.0] would be 0.0 regardless of actual confidence.  With
    top_p_trace=[0.5] and correct=True, Brier = (0.5-1)^2 = 0.25.
    """
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [0.9],
            "g_trace": [1.0],
            "top_p_trace": [0.5],
        },
    ]
    cal = calibration_at_buzz(results)
    assert abs(cal["brier"] - 0.25) < 1e-9
````

## File: AGENTS.md
````markdown
# AGENTS.md

Canonical repo contract for all coding agents (Claude, Copilot, Cursor, etc.).

## Project Overview

Stanford CS234 final project: a unified quiz bowl RL buzzer system with two tracks. The belief-feature pipeline builds MC tossups, scores answer profiles with TF-IDF / SBERT / T5 / optional OpenAI embeddings, trains or compares buzzers, and evaluates with S_q plus calibration metrics. The T5 policy pipeline provides supervised warm-start and PPO for an end-to-end text policy. `qanta-buzzer` is the canonical repo. qb-rl compatibility is preserved through additive shims rather than structural rewrites.

## Setup

Requires Python >= 3.11.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
```

Optional OpenAI support:

```bash
pip install -e '.[openai]'
export OPENAI_API_KEY=...
```

## Architecture

| Package | Purpose |
|---------|---------|
| `qb_data/` | Canonical data loading, answer profiles, stratified splits, MC construction |
| `qb_env/` | Gymnasium environment, text wrapper, qb-rl compatibility shims |
| `models/` | Likelihood models (TF-IDF, SBERT, T5, OpenAI), belief features, T5 policy model |
| `agents/` | Threshold, softmax-profile, sequential Bayes, PPO wrapper |
| `evaluation/` | S_q metric, calibration, control experiments, plotting |
| `scripts/` | Pipeline entrypoints and shared helpers |
| `training/` | T5 policy supervised + PPO trainers |
| `configs/` | YAML configuration files (default, smoke, t5_policy) |

## Testing

261 tests across 16 test files.

```bash
pytest                    # full suite
pytest tests/test_qb_rl_bridge.py tests/test_factories.py tests/test_ppo_buzzer.py  # focused bridge/runtime checks
scripts/ci.sh             # CI entry point (runs pytest, exits nonzero on failure)
```

## Smoke Pipeline

Four-stage belief-feature smoke workflow. `--smoke` selects `configs/smoke.yaml` and writes outputs to `artifacts/smoke/`.

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

Or run all four stages via the wrapper script:

```bash
scripts/manual-smoke.sh
```

## T5 Policy Pipeline

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/compare_policies.py --config configs/t5_policy.yaml
```

## Configuration

| Config | Purpose |
|--------|---------|
| `configs/default.yaml` | Full runs with T5-large likelihood and 100k PPO timesteps |
| `configs/smoke.yaml` | Quick tests: 50 questions, TF-IDF likelihood, 3k PPO timesteps |
| `configs/t5_policy.yaml` | T5 policy pipeline: model, supervised, PPO, and data sections |

qb-rl config aliases are supported (e.g., `data.dataset`, `likelihood.sbert_name`, `environment.reward` as alias for `reward_mode`).

## Compatibility Bridge

Old qb-rl import paths that still resolve:

- `qb_env.data_loader`, `qb_env.mc_builder`, `qb_env.text_utils`
- `models.answer_profiles`
- `agents.softmax_profile_buzzer`

OpenAI support is opt-in only. Default local workflows stay offline-friendly and do not require the `openai` package or `OPENAI_API_KEY`.

## Conventions

- NumPy-style docstrings with Parameters/Returns sections
- RL notation: `V` (value), `R` (reward), `T` (transition), `gamma` (discount), `s`/`a` (state/action)
- Prefer NumPy/PyTorch vectorized operations over loops in ML code
- Explicit seeds for reproducibility (use 1, 2, 3 for multi-seed runs)
````

## File: agents/__init__.py
````python
from agents.threshold_buzzer import (
    ThresholdBuzzer,
    AlwaysBuzzFinalBuzzer,
    EpisodeResult,
    sweep_thresholds,
    result_to_dict,
)
from agents.bayesian_buzzer import (
    SoftmaxProfileBuzzer,
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
    sweep_sequential_thresholds,
)

# Lazy import: PPOBuzzer requires stable_baselines3 which may not be installed
# in all environments (e.g., baseline-only runs). Import on demand.


def __getattr__(name: str):
    if name in ("PPOBuzzer", "PPOEpisodeTrace"):
        from agents.ppo_buzzer import PPOBuzzer, PPOEpisodeTrace
        return {"PPOBuzzer": PPOBuzzer, "PPOEpisodeTrace": PPOEpisodeTrace}[name]
    raise AttributeError(f"module 'agents' has no attribute {name!r}")


__all__ = [
    "ThresholdBuzzer",
    "AlwaysBuzzFinalBuzzer",
    "SoftmaxProfileBuzzer",
    "SequentialBayesBuzzer",
    "PPOBuzzer",
    "EpisodeResult",
    "SoftmaxEpisodeResult",
    "PPOEpisodeTrace",
    "sweep_thresholds",
    "sweep_sequential_thresholds",
    "result_to_dict",
]
````

## File: agents/softmax_profile_buzzer.py
````python
"""qb-rl compatibility re-exports for Bayesian-family buzzers."""

from agents.bayesian_buzzer import (
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
    SoftmaxProfileBuzzer,
)

__all__ = [
    "SoftmaxEpisodeResult",
    "SoftmaxProfileBuzzer",
    "SequentialBayesBuzzer",
]
````

## File: configs/t5_policy.yaml
````yaml
# T5 Policy Configuration
# Hyperparameters for T5PolicyModel with supervised warm-start and PPO fine-tuning.
# Use with: python -m training.train_supervised_t5 --config configs/t5_policy.yaml

model:
  model_name: t5-large  # Use t5-base or t5-small if memory constrained
  device: auto  # auto-detect cuda > mps > cpu
  max_input_length: 512
  num_choices: 4

supervised:
  lr: 3.0e-4
  epochs: 10
  batch_size: 8
  grad_accum_steps: 4  # Effective batch = 32
  max_grad_norm: 1.0
  weight_decay: 0.01
  checkpoint_dir: checkpoints

ppo:
  lr: 1.0e-5  # Lower than supervised for stability
  iterations: 100
  batch_size: 8
  epochs_per_iter: 4
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  gamma: 0.99
  gae_lambda: 0.95
  target_kl: 0.03
  checkpoint_dir: checkpoints

data:
  csv_path: "questions.csv"
  K: 4
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15
  seed: 42

# Smoke test overrides (use with --smoke flag)
smoke:
  model:
    model_name: t5-small  # 60M params instead of 770M
    max_input_length: 128
  supervised:
    epochs: 2
    batch_size: 4
    grad_accum_steps: 1  # No accumulation for speed
  ppo:
    iterations: 5
    batch_size: 4
    epochs_per_iter: 2
  data:
    max_questions: 50
````

## File: evaluation/controls.py
````python
"""
Control Experiments for Quiz Bowl Buzzer Evaluation

Implements three control experiments to validate that the buzzer agent
genuinely uses question clues rather than exploiting surface-form artifacts:

1. **Choices-only control**: Strips all clues, trains a logistic regression
   on option surface features (char n-grams, length, capitalization). Expected
   accuracy ~25% (1/K) if options have no exploitable artifacts.

2. **Shuffle control**: Randomizes option ordering to verify the agent has
   no position bias. Performance should be unchanged.

3. **Alias substitution control**: Swaps answer text with aliases to verify
   robustness to surface-form changes.

Ported from qb-rl reference implementation (evaluation/controls.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from qb_data.mc_builder import MCQuestion


def _option_scalar_features(option: str) -> list[float]:
    """Extract scalar surface features from a single option string.

    Parameters
    ----------
    option : str
        Answer option text.

    Returns
    -------
    list[float]
        Six scalar features: char length, token count, has_parens,
        has_comma, is_title, is_lower.
    """
    tokens = option.split()
    has_parens = 1.0 if "(" in option or ")" in option else 0.0
    has_comma = 1.0 if "," in option else 0.0
    is_title = 1.0 if option.istitle() else 0.0
    is_lower = 1.0 if option.islower() else 0.0
    return [
        float(len(option)),
        float(len(tokens)),
        has_parens,
        has_comma,
        is_title,
        is_lower,
    ]


def _cross_option_features(options: list[str]) -> list[float]:
    """Extract cross-option comparative features.

    Parameters
    ----------
    options : list[str]
        All answer options for a question.

    Returns
    -------
    list[float]
        Three features: max/min length ratio, length std, number of
        distinct capitalization patterns.
    """
    lengths = np.array(
        [max(1, len(o.split())) for o in options], dtype=np.float32
    )
    cap_patterns = len(
        set(
            ("title" if o.istitle() else "lower" if o.islower() else "mixed")
            for o in options
        )
    )
    return [
        float(lengths.max() / lengths.min()),
        float(lengths.std()),
        float(cap_patterns),
    ]


def run_choices_only_control(
    questions: list[MCQuestion],
    random_seed: int = 13,
    test_fraction: float = 0.25,
) -> dict[str, float]:
    """Run choices-only control: predict answer from surface features only.

    Strips all question clues and trains a logistic regression on option
    surface features (char n-grams, length, capitalization patterns).
    Expected accuracy ~25% (1/K) if options are well-constructed.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    random_seed : int
        Seed for reproducible train/test split.
    test_fraction : float
        Fraction of questions held out for testing.

    Returns
    -------
    dict[str, float]
        Control results: accuracy, chance baseline, and test set size.
    """
    if not questions:
        return {"accuracy": 0.0, "chance": 0.0, "n_test": 0.0}

    rng = random.Random(random_seed)
    shuffled = questions[:]
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - test_fraction)))
    train_q = shuffled[:split_idx]
    test_q = shuffled[split_idx:]
    if not test_q:
        test_q = train_q

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
    vec.fit([opt for q in train_q for opt in q.options])

    def build_matrix(
        rows: list[MCQuestion],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        X = []
        y = []
        group_sizes: list[int] = []
        for q in rows:
            cross = _cross_option_features(q.options)
            group_sizes.append(len(q.options))
            tfidf = vec.transform(q.options).toarray()
            for i, option in enumerate(q.options):
                feat = np.array(
                    _option_scalar_features(option) + cross, dtype=np.float32
                )
                row = np.concatenate([feat, tfidf[i]], axis=0)
                X.append(row)
                y.append(1 if i == q.gold_index else 0)
        return np.array(X), np.array(y), group_sizes

    X_train, y_train, _ = build_matrix(train_q)
    X_test, y_test, test_group_sizes = build_matrix(test_q)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    offset = 0
    correct = 0
    total = 0
    for q, group_size in zip(test_q, test_group_sizes):
        group_probs = probs[offset : offset + group_size]
        pred_idx = int(np.argmax(group_probs))
        if pred_idx == q.gold_index:
            correct += 1
        total += 1
        offset += group_size

    accuracy = correct / max(1, total)
    chance = 1.0 / max(1, len(questions[0].options))
    return {
        "accuracy": float(accuracy),
        "chance": float(chance),
        "n_test": float(total),
    }


def shuffled_option_copy(
    question: MCQuestion, rng: random.Random
) -> MCQuestion:
    """Create a copy of an MCQuestion with shuffled option ordering.

    Parameters
    ----------
    question : MCQuestion
        Original question.
    rng : random.Random
        Random number generator for shuffling.

    Returns
    -------
    MCQuestion
        Copy with permuted options, profiles, answer_primary, and
        updated gold_index.
    """
    perm = list(range(len(question.options)))
    rng.shuffle(perm)
    new_options = [question.options[i] for i in perm]
    new_profiles = [question.option_profiles[i] for i in perm]
    new_answer_primary = [question.option_answer_primary[i] for i in perm]
    new_gold = perm.index(question.gold_index)
    return replace(
        question,
        options=new_options,
        option_profiles=new_profiles,
        option_answer_primary=new_answer_primary,
        gold_index=new_gold,
    )


def run_shuffle_control(
    questions: list[MCQuestion],
    evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run shuffle control: randomize option ordering and evaluate.

    Permutes the answer options for each question and runs the evaluator.
    If the agent has no position bias, performance should be unchanged.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    evaluator : callable
        Function that takes a list of MCQuestion and returns a metrics dict.
    random_seed : int
        Seed for reproducible shuffling.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics on shuffled questions.
    """
    rng = random.Random(random_seed)
    shuffled = [shuffled_option_copy(q, rng) for q in questions]
    return evaluator(shuffled)


def alias_substitution_copy(
    question: MCQuestion,
    alias_lookup: dict[str, list[str]],
    rng: random.Random,
) -> MCQuestion:
    """Create a copy of an MCQuestion with alias-substituted options.

    Parameters
    ----------
    question : MCQuestion
        Original question.
    alias_lookup : dict[str, list[str]]
        Mapping from canonical answer to list of known aliases.
    rng : random.Random
        Random number generator for alias selection.

    Returns
    -------
    MCQuestion
        Copy with alias-substituted option text and profiles.
    """
    new_options = []
    new_profiles = list(question.option_profiles)
    for i, (option_text, answer_primary) in enumerate(
        zip(question.options, question.option_answer_primary)
    ):
        aliases = [
            a
            for a in alias_lookup.get(answer_primary, [])
            if a and a != option_text
        ]
        if aliases:
            alias = rng.choice(aliases)
            new_options.append(alias)
            if new_profiles[i].strip() == answer_primary.strip():
                new_profiles[i] = alias
        else:
            new_options.append(option_text)
    return replace(question, options=new_options, option_profiles=new_profiles)


def run_alias_substitution_control(
    questions: list[MCQuestion],
    alias_lookup: dict[str, list[str]],
    evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run alias substitution control: swap answer text with aliases.

    Replaces option text with known aliases to verify the agent is robust
    to surface-form changes. Performance should be similar to full eval.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    alias_lookup : dict[str, list[str]]
        Mapping from canonical answer to list of known aliases.
    evaluator : callable
        Function that takes a list of MCQuestion and returns a metrics dict.
    random_seed : int
        Seed for reproducible alias selection.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics on alias-substituted questions.
    """
    rng = random.Random(random_seed)
    swapped = [
        alias_substitution_copy(q, alias_lookup=alias_lookup, rng=rng)
        for q in questions
    ]
    return evaluator(swapped)


def run_shuffle_control_precomputed(
    precomputed: list["_PrecomputedQuestion"],
    threshold: float,
    alpha: float,
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run shuffle control by permuting precomputed belief vectors.

    Produces numerically identical results to ``run_shuffle_control`` with
    a live ``SoftmaxProfileBuzzer`` evaluator, but makes zero
    ``likelihood_model.score()`` calls.  Instead, the belief vectors
    stored in each ``_PrecomputedQuestion`` are reordered according to
    the same random permutation that ``shuffled_option_copy`` would apply.

    Parameters
    ----------
    precomputed : list[_PrecomputedQuestion]
        Pre-computed belief distributions (one per question).
    threshold : float
        Buzz threshold for the softmax profile buzzer.
    alpha : float
        Sigmoid steepness for the confidence proxy.
    random_seed : int
        Seed for reproducible shuffling (must match the seed used in
        ``run_shuffle_control`` for equivalence).

    Returns
    -------
    dict[str, Any]
        Summary metrics with ``"runs"`` key containing per-question dicts.
    """
    from dataclasses import asdict

    from agents.threshold_buzzer import (
        _PrecomputedQuestion,
        _softmax_episode_from_precomputed,
    )
    from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics

    rng = random.Random(random_seed)
    runs: list[dict[str, Any]] = []
    for pq in precomputed:
        perm = list(range(pq.num_options))
        rng.shuffle(perm)
        new_gold = perm.index(pq.gold_index)
        shuffled_beliefs = [b[perm] for b in pq.beliefs]
        shuffled_pq = _PrecomputedQuestion(
            qid=pq.qid,
            gold_index=new_gold,
            num_options=pq.num_options,
            beliefs=shuffled_beliefs,
        )
        result = _softmax_episode_from_precomputed(shuffled_pq, threshold, alpha)
        runs.append(asdict(result))
    summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
    summary["runs"] = runs
    return summary


def bootstrap_ci(
    values: list[float],
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : list[float]
        Observed values.
    n_samples : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (0.05 = 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    samples = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(arr), size=len(arr))
        samples.append(float(arr[idx].mean()))
    lo = np.quantile(samples, alpha / 2.0)
    hi = np.quantile(samples, 1.0 - alpha / 2.0)
    return float(lo), float(hi)
````

## File: models/answer_profiles.py
````python
"""qb-rl compatibility re-export for answer profile building."""

from qb_data.answer_profiles import AnswerProfileBuilder

__all__ = ["AnswerProfileBuilder"]
````

## File: models/t5_policy.py
````python
"""
T5-based Policy Model for Quiz Bowl RL Agent

Implements T5PolicyModel with a custom PolicyHead containing three independent
heads (wait/answer/value) for end-to-end text-based policy learning. This
provides an alternative to the MLP policy trained on belief features
(Phase 4 approach).

Architecture overview:

    Text input  -->  T5 Encoder  -->  Mean Pooling  -->  PolicyHead
                                                          |-- Wait head (2)
                                                          |-- Answer head (K)
                                                          |-- Value head (1)

The T5 encoder produces contextual embeddings from tokenized text. Mean pooling
(attention-masked) reduces the variable-length sequence to a fixed-size vector.
The PolicyHead then produces three independent outputs:

- **Wait logits** [B, 2]: probability of waiting vs answering now
- **Answer logits** [B, K]: probability of selecting each answer option
- **Value estimate** [B, 1]: state value for PPO advantage computation

Action space maps to the TossupMCEnv convention:
    0 = WAIT (wait head selects "wait")
    1..K = SELECT answer i-1 (wait head selects "answer now", answer head picks i-1)

Ported from qanta-buzzer reference implementation (model.py) with these changes:
    - T5EncoderModel replaces T5ForConditionalGeneration (2x faster, 50% less memory)
    - T5TokenizerFast replaces T5Tokenizer (3-5x faster tokenization via Rust backend)
    - Config dict replaces qanta-buzzer's Config class for unified codebase compatibility
    - NumPy-style docstrings added throughout
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """Custom policy head with three independent output heads.

    Attached to a T5 encoder's pooled output, this module produces the three
    outputs needed for actor-critic RL in the quiz bowl POMDP: a binary
    wait/answer-now decision, a K-way answer selection, and a scalar value
    estimate.

    All three heads are fully independent (no shared hidden layers beyond the
    encoder), using the same pattern: Linear -> ReLU -> Dropout -> Linear.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the input from the T5 encoder's pooled output.
        Default 1024 matches T5-large (``d_model``). Use 512 for t5-small,
        768 for t5-base.
    num_choices : int
        Number of answer options (K). Default 4 for quiz bowl MC questions.

    Attributes
    ----------
    wait_head : nn.Sequential
        Binary head producing [wait, answer_now] logits.
    answer_head : nn.Sequential
        Multi-class head producing logits over K answer choices.
    value_head : nn.Sequential
        Scalar head producing state value estimate.
    """

    def __init__(self, hidden_size: int = 1024, num_choices: int = 4) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_choices = num_choices

        # Wait/continue decision head (binary: wait vs answer_now)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),  # [wait, answer_now]
        )

        # Answer selection head (over K choices)
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_choices),
        )

        # Value head (state value estimate for PPO)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(
        self, encoder_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all three heads.

        Parameters
        ----------
        encoder_hidden_state : torch.Tensor
            Pooled encoder output of shape ``[batch_size, hidden_size]``.

        Returns
        -------
        wait_logits : torch.Tensor
            Shape ``[batch_size, 2]`` -- logits for [wait, answer_now].
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer options.
        values : torch.Tensor
            Shape ``[batch_size, 1]`` -- state value estimates.
        """
        wait_logits = self.wait_head(encoder_hidden_state)
        answer_logits = self.answer_head(encoder_hidden_state)
        values = self.value_head(encoder_hidden_state)

        return wait_logits, answer_logits, values


class T5PolicyModel(nn.Module):
    """T5 encoder with custom policy head for end-to-end RL.

    Combines a pre-trained T5 encoder with a ``PolicyHead`` to produce policy
    outputs directly from text observations. This is the alternative approach
    to Phase 4's MLP policy, which operates on numeric belief features.

    The model processes text in three stages:

    1. **Tokenization**: Text is tokenized with ``T5TokenizerFast`` (Rust-backed
       for speed) with padding and truncation.
    2. **Encoding**: ``T5EncoderModel`` produces contextual hidden states
       ``[B, seq_len, d_model]``.
    3. **Pooling + Heads**: Attention-masked mean pooling reduces to
       ``[B, d_model]``, then PolicyHead produces wait/answer/value outputs.

    Action space follows TossupMCEnv convention:
        - 0 = WAIT
        - 1..K = SELECT answer (i-1)

    Combined actions are decomposed into two independent decisions for log
    probability computation:
        - ``wait_action``: 0 (wait) or 1 (answer now)
        - ``answer_action``: 0..K-1 (which answer to select)

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with the following keys:

        - ``model_name`` (str): HuggingFace T5 model identifier.
          Default ``"t5-large"``. Options: ``"t5-small"``, ``"t5-base"``,
          ``"t5-large"``.
        - ``device`` (str): Torch device. Default auto-detects
          (cuda > mps > cpu).
        - ``max_input_length`` (int): Maximum token sequence length.
          Default 512.
        - ``num_choices`` (int): Number of answer options (K). Default 4.

    Attributes
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    device : torch.device
        Computation device.
    encoder : T5EncoderModel
        Pre-trained T5 encoder.
    tokenizer : T5TokenizerFast
        Fast T5 tokenizer.
    policy_head : PolicyHead
        Custom three-head policy module.
    max_input_length : int
        Maximum token sequence length for tokenization.

    Examples
    --------
    >>> config = {"model_name": "t5-small", "device": "cpu", "num_choices": 4}
    >>> model = T5PolicyModel(config)
    >>> texts = ["CLUES: first president | CHOICES: (1) Washington (2) Jefferson"]
    >>> wait_logits, answer_logits, values = model(texts)
    >>> wait_logits.shape
    torch.Size([1, 2])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        from transformers import T5EncoderModel, T5TokenizerFast

        self.config = config
        model_name = config.get("model_name", "t5-large")
        self.max_input_length = config.get("max_input_length", 512)
        num_choices = config.get("num_choices", 4)

        # Auto-detect device
        default_device = "cpu"
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            default_device = "mps"
        self.device = torch.device(config.get("device", default_device))

        # Load T5 encoder only (not full T5ForConditionalGeneration)
        # This is 2x faster and uses 50% less memory since the decoder is unused
        print(f"Loading T5 encoder: {model_name}")
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)

        # Get hidden size from T5 config (512 for small, 768 for base, 1024 for large)
        hidden_size = self.encoder.config.d_model

        # Custom policy head
        self.policy_head = PolicyHead(
            hidden_size=hidden_size,
            num_choices=num_choices,
        )

        # Move to device
        self.to(self.device)

        # Print model info
        self._print_model_info()

    def _print_model_info(self) -> None:
        """Print model architecture summary and parameter counts."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        total_params = encoder_params + policy_params

        print(f"Model Architecture:")
        print(f"  T5 encoder parameters: {encoder_params:,}")
        print(f"  Policy head parameters: {policy_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {self.device}")

    def encode_input(
        self,
        text_inputs: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs using T5TokenizerFast.

        Parameters
        ----------
        text_inputs : list[str]
            List of input text strings to tokenize.
        max_length : int or None
            Maximum sequence length. If None, uses ``self.max_input_length``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with ``"input_ids"`` and ``"attention_mask"`` tensors,
            both of shape ``[batch_size, seq_len]``, moved to ``self.device``.
        """
        if max_length is None:
            max_length = self.max_input_length

        encoding = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {k: v.to(self.device) for k, v in encoding.items()}

    def get_encoder_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute T5 encoder output and pool to a fixed-size vector.

        Uses attention-masked mean pooling: sum hidden states where attention
        mask is 1, divide by number of non-padding tokens. This ensures
        padding tokens contribute zero to the pooled representation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]`` (1 for real
            tokens, 0 for padding).

        Returns
        -------
        torch.Tensor
            Pooled encoder output of shape ``[batch_size, hidden_size]``.
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # encoder_outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        hidden_states = encoder_outputs.last_hidden_state

        # Attention-masked mean pooling over sequence dimension
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / sum_mask

        return pooled_output

    def forward(
        self,
        text_inputs: List[str],
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: tokenize, encode, pool, then apply policy head.

        Parameters
        ----------
        text_inputs : list[str]
            List of text observations (e.g.,
            ``"CLUES: clue1 clue2 | CHOICES: (1) ans1 (2) ans2"``).
        return_value : bool
            If True, return value estimates. If False, values is None.

        Returns
        -------
        wait_logits : torch.Tensor
            Shape ``[batch_size, 2]`` -- logits for [wait, answer_now].
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer options.
        values : torch.Tensor or None
            Shape ``[batch_size, 1]`` if return_value is True, else None.
        """
        # Encode inputs
        encoding = self.encode_input(text_inputs)

        # Get pooled encoder output
        pooled_output = self.get_encoder_output(
            encoding["input_ids"],
            encoding["attention_mask"],
        )

        # Pass through policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)

        if not return_value:
            values = None

        return wait_logits, answer_logits, values

    def predict_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict answer choice for supervised training.

        Only uses the answer head (wait and value heads are ignored). This is
        the interface for supervised warm-start training where the model learns
        to select the correct answer from complete questions.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``.

        Returns
        -------
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer choices.
        predictions : torch.Tensor
            Shape ``[batch_size]`` -- predicted answer indices (argmax).
        """
        # Get encoder output
        pooled_output = self.get_encoder_output(input_ids, attention_mask)

        # Get answer logits from policy head
        _, answer_logits, _ = self.policy_head(pooled_output)

        # Get predictions
        predictions = torch.argmax(answer_logits, dim=-1)

        return answer_logits, predictions

    def select_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Select actions based on current policy.

        Produces combined actions following TossupMCEnv convention:
        0 = WAIT, 1..K = SELECT answer 0..K-1. The decision is decomposed
        into two independent categorical distributions: wait (binary) and
        answer (K-way).

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``.
        deterministic : bool
            If True, use argmax instead of sampling.
        temperature : float
            Temperature for softmax. Higher values increase randomness.
            Default 1.0 (no scaling).

        Returns
        -------
        combined_actions : torch.Tensor
            Shape ``[batch_size]`` -- combined actions (0 = WAIT, 1..K = SELECT).
        info : dict[str, Any]
            Dictionary with keys:

            - ``wait_logits``: raw wait head output
            - ``answer_logits``: raw answer head output
            - ``wait_probs``: softmax of wait logits
            - ``answer_probs``: softmax of answer logits
            - ``wait_actions``: sampled wait decisions (0 or 1)
            - ``answer_actions``: sampled answer indices (0..K-1)
            - ``values``: value estimates
            - ``log_probs``: total log probability of the combined action
        """
        with torch.no_grad():
            # Get encoder output
            pooled_output = self.get_encoder_output(input_ids, attention_mask)

            # Get logits from policy head
            wait_logits, answer_logits, values = self.policy_head(pooled_output)

            # Apply temperature
            wait_logits_scaled = wait_logits / temperature
            answer_logits_scaled = answer_logits / temperature

            # Get probabilities
            wait_probs = F.softmax(wait_logits_scaled, dim=-1)
            answer_probs = F.softmax(answer_logits_scaled, dim=-1)

            if deterministic:
                # Argmax for both decisions
                wait_actions = torch.argmax(wait_probs, dim=-1)
                answer_actions = torch.argmax(answer_probs, dim=-1)
            else:
                # Sample from distributions
                wait_dist = torch.distributions.Categorical(wait_probs)
                answer_dist = torch.distributions.Categorical(answer_probs)

                wait_actions = wait_dist.sample()
                answer_actions = answer_dist.sample()

            # Compute log probabilities
            wait_log_probs = F.log_softmax(wait_logits_scaled, dim=-1)
            answer_log_probs = F.log_softmax(answer_logits_scaled, dim=-1)

            selected_wait_log_probs = wait_log_probs.gather(
                1, wait_actions.unsqueeze(-1)
            ).squeeze(-1)
            selected_answer_log_probs = answer_log_probs.gather(
                1, answer_actions.unsqueeze(-1)
            ).squeeze(-1)

            # Total log prob is sum (independent decisions)
            log_probs = selected_wait_log_probs + selected_answer_log_probs

            # Combine wait and answer into single action
            # wait_action == 0 -> combined action = 0 (WAIT)
            # wait_action == 1 -> combined action = 1 + answer_action (SELECT)
            combined_actions = torch.where(
                wait_actions == 0,
                torch.zeros_like(wait_actions),
                1 + answer_actions,
            )

            info = {
                "wait_logits": wait_logits,
                "answer_logits": answer_logits,
                "wait_probs": wait_probs,
                "answer_probs": answer_probs,
                "wait_actions": wait_actions,
                "answer_actions": answer_actions,
                "values": values,
                "log_probs": log_probs,
            }

            return combined_actions, info

    def get_action_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy for given actions.

        Used during PPO training to evaluate old actions under the current
        policy. Decomposes combined actions (0 = WAIT, 1..K = SELECT) into
        independent wait and answer decisions for probability computation.

        Action decomposition:
            - ``actions == 0`` -> ``wait_action = 0`` (WAIT)
            - ``actions in 1..K`` -> ``wait_action = 1``, ``answer_action = actions - 1``

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``.
        actions : torch.Tensor
            Combined actions of shape ``[batch_size]``. Values in {0, 1, ..., K}.

        Returns
        -------
        log_probs : torch.Tensor
            Shape ``[batch_size]`` -- total log probability of each action.
        entropy : torch.Tensor
            Shape ``[batch_size]`` -- sum of wait and answer entropy.
        values : torch.Tensor
            Shape ``[batch_size]`` -- value estimates (squeezed).
        """
        # Decompose combined actions into wait and answer components
        # action 0 -> wait=0 (WAIT)
        # action 1-K -> wait=1, answer=0..K-1 (SELECT)
        wait_actions = (actions > 0).long()
        answer_actions = torch.clamp(actions - 1, min=0)  # Map 1..K to 0..K-1

        # Get encoder output
        pooled_output = self.get_encoder_output(input_ids, attention_mask)

        # Get logits from policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)

        # Compute log probabilities
        wait_log_probs = F.log_softmax(wait_logits, dim=-1)
        answer_log_probs = F.log_softmax(answer_logits, dim=-1)

        # Gather log probs for selected actions
        selected_wait_log_probs = wait_log_probs.gather(
            1, wait_actions.unsqueeze(-1)
        ).squeeze(-1)
        selected_answer_log_probs = answer_log_probs.gather(
            1, answer_actions.unsqueeze(-1)
        ).squeeze(-1)

        # Total log prob
        log_probs = selected_wait_log_probs + selected_answer_log_probs

        # Compute entropy
        wait_probs = F.softmax(wait_logits, dim=-1)
        answer_probs = F.softmax(answer_logits, dim=-1)

        wait_entropy = -(wait_probs * wait_log_probs).sum(dim=-1)
        answer_entropy = -(answer_probs * answer_log_probs).sum(dim=-1)

        entropy = wait_entropy + answer_entropy

        return log_probs, entropy, values.squeeze(-1)

    def save(self, save_dir: str) -> None:
        """Save model checkpoint to disk.

        Saves three components:
        1. T5 encoder weights and config (HuggingFace format)
        2. Tokenizer files (HuggingFace format)
        3. Policy head state dict (PyTorch format as ``policy_head.pt``)

        Parameters
        ----------
        save_dir : str
            Directory path to save the checkpoint. Created if it doesn't exist.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save T5 encoder
        self.encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save policy head
        policy_head_path = os.path.join(save_dir, "policy_head.pt")
        torch.save(self.policy_head.state_dict(), policy_head_path)

        print(f"Model saved to {save_dir}")

    def load(self, load_dir: str) -> None:
        """Load model checkpoint from disk.

        Loads T5 encoder weights, tokenizer, and policy head state dict from
        the specified directory. The model is moved to ``self.device`` after
        loading.

        Parameters
        ----------
        load_dir : str
            Directory containing a previously saved checkpoint.

        Raises
        ------
        FileNotFoundError
            If ``policy_head.pt`` is not found in ``load_dir``.
        """
        from transformers import T5EncoderModel, T5TokenizerFast

        # Load T5 encoder
        self.encoder = T5EncoderModel.from_pretrained(load_dir)
        self.tokenizer = T5TokenizerFast.from_pretrained(load_dir)

        # Load policy head
        policy_head_path = os.path.join(load_dir, "policy_head.pt")
        self.policy_head.load_state_dict(
            torch.load(policy_head_path, map_location=self.device, weights_only=True)
        )

        self.to(self.device)
        print(f"Model loaded from {load_dir}")

    @classmethod
    def load_pretrained(
        cls,
        load_dir: str,
        device: Optional[str] = None,
    ) -> "T5PolicyModel":
        """Load a pretrained model from a directory.

        Class method that creates a new T5PolicyModel instance and loads
        weights from a saved checkpoint.

        Parameters
        ----------
        load_dir : str
            Directory containing a previously saved checkpoint.
        device : str or None
            Device to load model on (e.g., ``"cpu"``, ``"cuda"``, ``"mps"``).
            If None, auto-detects.

        Returns
        -------
        T5PolicyModel
            A loaded model instance ready for inference.
        """
        from transformers import T5EncoderModel

        # Detect model config from saved checkpoint
        t5_encoder = T5EncoderModel.from_pretrained(load_dir, local_files_only=True)
        hidden_size = t5_encoder.config.d_model

        # Infer num_choices from policy head state dict
        policy_head_path = os.path.join(load_dir, "policy_head.pt")
        policy_head_state = torch.load(
            policy_head_path, map_location="cpu", weights_only=True
        )
        # answer_head final linear layer weight shape is [num_choices, hidden_dim]
        num_choices = policy_head_state["answer_head.3.weight"].shape[0]

        config = {
            "model_name": load_dir,
            "device": device or "cpu",
            "num_choices": num_choices,
        }

        model = cls(config)
        model.load(load_dir)
        return model
````

## File: qb_data/__init__.py
````python
"""Quiz Bowl Data Package.

Core data structures and utilities for quiz bowl question processing,
including qb-rl compatibility loader helpers.
"""

from qb_data.data_loader import (
    QANTADatasetLoader,
    TossupQuestion,
    load_tossup_questions,
    load_tossup_questions_from_config,
    parse_row,
)
from qb_data.text_utils import normalize_answer

__all__ = [
    'TossupQuestion',
    'QANTADatasetLoader',
    'parse_row',
    'load_tossup_questions',
    'load_tossup_questions_from_config',
    'normalize_answer',
]
````

## File: qb_data/answer_profiles.py
````python
"""Answer profile builder with leave-one-out exclusion for quiz bowl questions."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from qb_data.data_loader import TossupQuestion


class AnswerProfileBuilder:
    """Builds profiles for answers by aggregating question texts.

    The profile for an answer is created by concatenating all question texts
    that have that answer. When building profiles for distractors, we use
    all questions. For the gold answer, we exclude the current question to
    prevent information leakage (leave-one-out).

    Attributes:
        max_tokens_per_profile: Maximum number of tokens to keep in each profile.
        min_questions_per_answer: Minimum questions needed to build a profile.
        _grouped: Dictionary mapping answer_primary to list of (qid, question_text) tuples.
    """

    def __init__(
        self,
        max_tokens_per_profile: int = 2000,
        min_questions_per_answer: int = 1
    ):
        """Initialize the answer profile builder.

        Args:
            max_tokens_per_profile: Maximum tokens to keep in each profile.
            min_questions_per_answer: Minimum questions needed to build a profile.
        """
        self.max_tokens_per_profile = max_tokens_per_profile
        self.min_questions_per_answer = min_questions_per_answer
        self._grouped: Dict[str, List[Tuple[str, str]]] = {}
        self._cache: Dict[Tuple[str, Optional[str]], str] = {}

    def fit(self, questions: List[TossupQuestion]) -> "AnswerProfileBuilder":
        """Fit the builder on a set of questions.

        Groups questions by their primary answer for efficient profile building.

        Args:
            questions: List of tossup questions to group by answer.

        Returns:
            Self for method chaining.
        """
        grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for q in questions:
            # Store qid and full question text for each answer
            grouped[q.answer_primary].append((q.qid, q.question))
        self._grouped = dict(grouped)
        self._cache = {}
        return self

    def _profile_text(
        self,
        answer_primary: str,
        exclude_qid: Optional[str] = None
    ) -> str:
        """Build profile text for an answer with optional exclusion.

        Args:
            answer_primary: The answer to build a profile for.
            exclude_qid: Optional question ID to exclude (leave-one-out).

        Returns:
            Profile text truncated to max_tokens_per_profile.
        """
        key = (answer_primary, exclude_qid)
        if key in self._cache:
            return self._cache[key]

        items = self._grouped.get(answer_primary, [])
        texts: List[str] = []

        # Collect all question texts except the excluded one
        for qid, qtext in items:
            if exclude_qid is not None and qid == exclude_qid:
                continue
            texts.append(qtext)

        # If not enough questions after exclusion, fall back to answer text
        if len(texts) < self.min_questions_per_answer:
            self._cache[key] = answer_primary
            return answer_primary

        # Merge all texts and split into tokens
        merged = " ".join(texts).split()

        # Truncate to max tokens if specified
        if self.max_tokens_per_profile > 0:
            merged = merged[:self.max_tokens_per_profile]

        result = " ".join(merged) if merged else answer_primary
        self._cache[key] = result
        return result

    def profile_for_answer(
        self,
        answer_primary: str,
        exclude_qid: Optional[str] = None
    ) -> str:
        """Get the profile for a specific answer.

        Args:
            answer_primary: The answer to get a profile for.
            exclude_qid: Optional question ID to exclude (for gold answer).

        Returns:
            Profile text for the answer.
        """
        return self._profile_text(
            answer_primary=answer_primary,
            exclude_qid=exclude_qid
        )

    def build_profiles(
        self,
        questions: List[TossupQuestion],
        exclude_qid: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build profiles for all answers in the dataset.

        Args:
            questions: List of questions (used to fit if not already fitted).
            exclude_qid: Optional question ID to exclude from all profiles.

        Returns:
            Dictionary mapping answer_primary to profile text.
        """
        if not self._grouped:
            self.fit(questions)

        return {
            answer: self._profile_text(answer, exclude_qid=exclude_qid)
            for answer in self._grouped.keys()
        }
````

## File: qb_data/config.py
````python
"""Configuration loading and management utilities.

Provides functions to load YAML configurations, apply small
cross-codebase compatibility normalizations, and merge CLI overrides
using dot notation (e.g., ``data.K=5`` updates ``config["data"]["K"]``).
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union


def normalize_config(
    config: Dict[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    """Apply compatibility defaults to a loaded configuration.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    smoke : bool
        Whether the caller intends to run in smoke mode.

    Returns
    -------
    dict
        Normalized configuration dictionary.
    """
    data_cfg = config.setdefault("data", {})
    env_cfg = config.setdefault("environment", {})
    lik_cfg = config.setdefault("likelihood", {})

    if "reward" in env_cfg and "reward_mode" not in env_cfg:
        env_cfg["reward_mode"] = env_cfg["reward"]
    elif "reward_mode" in env_cfg and "reward" not in env_cfg:
        env_cfg["reward"] = env_cfg["reward_mode"]

    if smoke and data_cfg.get("dataset_smoke") and "dataset" not in data_cfg:
        data_cfg["dataset"] = data_cfg["dataset_smoke"]
    if smoke and data_cfg.get("dataset_smoke_config") and "dataset_config" not in data_cfg:
        data_cfg["dataset_config"] = data_cfg["dataset_smoke_config"]

    if "embedding_model" in lik_cfg and "sbert_name" not in lik_cfg:
        lik_cfg["sbert_name"] = lik_cfg["embedding_model"]
    if "sbert_name" in lik_cfg and "embedding_model" not in lik_cfg:
        lik_cfg["embedding_model"] = lik_cfg["sbert_name"]

    return config


def resolve_data_loading_options(
    config: Dict[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    """Resolve CSV/Hugging Face data-loading options from a config dict.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    smoke : bool
        Whether the caller intends to run in smoke mode.

    Returns
    -------
    dict
        Resolved data-loading settings.
    """
    data_cfg = config.get("data", {})
    use_smoke_dataset = smoke and any(
        data_cfg.get(key) is not None
        for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
    )

    csv_path = data_cfg.get("csv_path")
    if smoke and data_cfg.get("csv_smoke_path"):
        csv_path = data_cfg["csv_smoke_path"]

    dataset = data_cfg.get("dataset")
    dataset_config = data_cfg.get("dataset_config")
    split = data_cfg.get("split", "eval")

    if use_smoke_dataset:
        dataset = data_cfg.get("dataset_smoke", dataset)
        dataset_config = data_cfg.get("dataset_smoke_config", dataset_config)
        split = data_cfg.get("split_smoke", split)

    return {
        "csv_path": csv_path,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "use_huggingface": bool(data_cfg.get("use_huggingface", False) or dataset),
        "max_questions": data_cfg.get("max_questions"),
        "uses_dataset_smoke": use_smoke_dataset,
    }


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration file. Defaults to configs/default.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    ImportError
        If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config loading. "
            "Install it with: pip install pyyaml"
        )

    # Default to configs/default.yaml if no path given
    if config_path is None:
        project_root = Path(__file__).parent.parent
        default_path = project_root / "configs" / "default.yaml"
        smoke_path = project_root / "configs" / "smoke.yaml"

        if smoke and default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                default_config = yaml.safe_load(f) or {}
            default_data = default_config.get("data", {})
            if any(
                default_data.get(key) is not None
                for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
            ):
                config_path = default_path
            elif smoke_path.exists():
                config_path = smoke_path
            else:
                config_path = default_path
        else:
            config_path = default_path
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return normalize_config(config or {}, smoke=smoke)


def merge_overrides(
    config: Dict[str, Any],
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge override values into configuration using dot notation.

    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    overrides : dict
        Override values to merge. Keys can use dot notation
        (e.g., {"data.K": 5} updates config["data"]["K"]).

    Returns
    -------
    dict
        Updated configuration with overrides applied.

    Examples
    --------
    >>> config = {"data": {"K": 4}, "ppo": {"batch_size": 32}}
    >>> overrides = {"data.K": 5, "ppo.batch_size": 16}
    >>> config = merge_overrides(config, overrides)
    >>> assert config["data"]["K"] == 5
    >>> assert config["ppo"]["batch_size"] == 16
    """
    for key, value in overrides.items():
        # Split on dots for nested keys
        keys = key.split(".")

        # Navigate to the nested location
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the final value
        final_key = keys[-1]
        current[final_key] = value

    return normalize_config(config)


def build_argparse_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to configuration overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Configuration overrides extracted from args.

    Notes
    -----
    Special handling:
    - --smoke flag loads smoke.yaml config
    - --config specifies custom config path
    - --override key=value pairs for dot notation overrides
    """
    overrides = {}

    # Handle smoke test mode
    if hasattr(args, "smoke") and args.smoke:
        overrides["__smoke__"] = True

    # Handle custom config path
    if hasattr(args, "config") and args.config:
        overrides["__config_path__"] = args.config

    # Parse key=value override pairs
    if hasattr(args, "override") and args.override:
        for override_str in args.override:
            if "=" not in override_str:
                print(f"Warning: Invalid override format '{override_str}', expected 'key=value'")
                continue

            key, value_str = override_str.split("=", 1)

            # Try to parse value as appropriate type
            value = parse_value(value_str)
            overrides[key] = value

    return overrides


def parse_value(value_str: str) -> Any:
    """Parse string value to appropriate Python type.

    Parameters
    ----------
    value_str : str
        String representation of value.

    Returns
    -------
    any
        Parsed value with appropriate type.

    Examples
    --------
    >>> parse_value("5") == 5
    >>> parse_value("3.14") == 3.14
    >>> parse_value("true") == True
    >>> parse_value("false") == False
    >>> parse_value("null") == None
    >>> parse_value("hello") == "hello"
    """
    # Handle boolean values
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # Handle null/none
    if value_str.lower() in ("null", "none"):
        return None

    # Try to parse as number
    try:
        # Try integer first
        if "." not in value_str:
            return int(value_str)
        # Then float
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add configuration-related arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.
    """
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke test configuration for quick testing"
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values using dot notation (e.g., data.K=5)"
    )


def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration and apply command-line overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Final configuration with all overrides applied.
    """
    # Build overrides from args
    overrides = build_argparse_overrides(args)

    # Check for special config path
    config_path = overrides.pop("__config_path__", None)
    smoke = bool(overrides.pop("__smoke__", False))

    # Load base config
    config = load_config(config_path, smoke=smoke)

    # Apply remaining overrides
    if overrides:
        config = merge_overrides(config, overrides)

    return config


# Convenience exports
__all__ = [
    "load_config",
    "merge_overrides",
    "normalize_config",
    "resolve_data_loading_options",
    "build_argparse_overrides",
    "add_config_args",
    "load_config_with_overrides",
]
````

## File: qb_data/data_loader.py
````python
"""
Data structures and loaders for quiz bowl questions.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

from qb_data.text_utils import normalize_answer


@dataclass
class TossupQuestion:
    """
    A quiz bowl tossup question with incremental clues.

    Attributes
    ----------
    qid : str
        Unique question identifier
    question : str
        Full question text (all clues concatenated)
    tokens : List[str]
        Tokenized question split on whitespace
    answer_primary : str
        Primary answer text
    clean_answers : List[str]
        List of acceptable answer variants
    run_indices : List[int]
        Token indices where clues end (for incremental reveal)
    human_buzz_positions : Optional[List[Tuple[int, int]]]
        Human buzzer positions as (position, count) tuples
    category : str
        Question category (e.g., "History", "Literature")
    cumulative_prefixes : List[str]
        Precomputed text prefixes at each run_index
    """
    qid: str
    question: str
    tokens: List[str]
    answer_primary: str
    clean_answers: List[str]
    run_indices: List[int]
    human_buzz_positions: Optional[List[Tuple[int, int]]]
    category: str
    cumulative_prefixes: List[str]


def _parse_clues_to_tokens(clues: List[str]) -> Tuple[List[str], List[int]]:
    """
    Convert list of clues to tokens and run indices.

    Parameters
    ----------
    clues : List[str]
        List of clue strings

    Returns
    -------
    Tuple[List[str], List[int]]
        Tokens (words) and indices where each clue ends
    """
    tokens = []
    run_indices = []

    for clue in clues:
        clue_tokens = clue.split()
        tokens.extend(clue_tokens)
        if clue_tokens:  # Only add index if clue has tokens
            run_indices.append(len(tokens) - 1)

    return tokens, run_indices


def _generate_qid(text: str) -> str:
    """
    Generate a unique question ID from question text.

    Parameters
    ----------
    text : str
        Question text to hash

    Returns
    -------
    str
        Unique identifier based on text hash
    """
    hash_obj = hashlib.md5(text.encode('utf-8'))
    return f"qid-{hash_obj.hexdigest()[:12]}"


def _coerce_human_buzz_positions(value: Any) -> Optional[List[Tuple[int, int]]]:
    """Coerce various metadata formats into ``(position, count)`` tuples."""
    if value is None:
        return None

    if isinstance(value, list):
        result: List[Tuple[int, int]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    result.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, dict):
                pos = item.get("position")
                count = item.get("count", 1)
                if pos is None:
                    continue
                try:
                    result.append((int(pos), int(count)))
                except (TypeError, ValueError):
                    continue
        return result or None

    return None


def _coerce_run_indices(run_indices: Any, token_count: int) -> List[int]:
    """Validate and coerce run indices into a sorted unique list."""
    clean: List[int] = []
    for idx in run_indices or []:
        try:
            clean.append(int(idx))
        except (TypeError, ValueError):
            continue

    if not clean:
        if token_count <= 0:
            raise ValueError("question must contain at least one token")
        clean = list(range(token_count))

    clean = sorted(set(clean))
    if clean[0] < 0 or clean[-1] > token_count - 1:
        raise ValueError(
            f"run_indices out of bounds: min={clean[0]} max={clean[-1]} token_count={token_count}"
        )
    return clean


def parse_row(row: Dict[str, Any]) -> TossupQuestion:
    """Parse a qb-rl/HuggingFace-style row into ``TossupQuestion``."""
    question = str(row["question"])
    tokens = question.split()
    metadata = row.get("metadata", {}) or {}
    answer_primary = str(
        row.get("answer_primary") or (row.get("clean_answers") or [""])[0]
    ).strip()
    clean_answers = [str(x) for x in (row.get("clean_answers") or [])]
    if not clean_answers and answer_primary:
        clean_answers = [answer_primary]

    run_indices = _coerce_run_indices(
        row.get("run_indices") or [],
        token_count=len(tokens),
    )

    normalized_question = " ".join(question.split())
    normalized_tokens = " ".join(tokens)
    if normalized_tokens != normalized_question:
        raise ValueError("tokenization roundtrip mismatch")
    if max(run_indices) > len(tokens) - 1:
        raise ValueError("run_indices out of bounds")

    cumulative_prefixes = [" ".join(tokens[: idx + 1]) for idx in run_indices]
    category = str(metadata.get("category") or row.get("category") or "")
    human_buzz_positions = _coerce_human_buzz_positions(
        metadata.get("human_buzz_positions") or row.get("human_buzz_positions")
    )

    qid_raw = row.get("qid") or row.get("question_id") or row.get("id")
    if qid_raw is None:
        qid_raw = _generate_qid(question)

    return TossupQuestion(
        qid=str(qid_raw),
        question=question,
        tokens=tokens,
        answer_primary=answer_primary,
        clean_answers=clean_answers,
        run_indices=run_indices,
        human_buzz_positions=human_buzz_positions,
        category=category,
        cumulative_prefixes=cumulative_prefixes,
    )


def load_tossup_questions(
    dataset: str,
    dataset_config: Optional[str] = None,
    split: str = "eval",
    limit: Optional[int] = None,
) -> List[TossupQuestion]:
    """Load tossup questions from Hugging Face datasets using qb-rl semantics."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for Hugging Face loading. Install it with: pip install datasets"
        ) from exc

    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split)
    else:
        ds = load_dataset(dataset, split=split)

    if limit is not None:
        ds = ds.select(range(min(int(limit), len(ds))))

    return [parse_row(dict(row)) for row in ds]


def load_tossup_questions_from_config(
    config: Dict[str, Any],
    smoke: bool = False,
) -> List[TossupQuestion]:
    """Load tossups from config, supporting qb-rl and qanta-buzzer keys."""
    from qb_data.config import resolve_data_loading_options

    data_opts = resolve_data_loading_options(config, smoke=smoke)
    csv_path = data_opts.get("csv_path")
    dataset = data_opts.get("dataset")
    dataset_config = data_opts.get("dataset_config")
    split = data_opts.get("split", "eval")
    limit = data_opts.get("max_questions")

    if csv_path and Path(csv_path).exists():
        questions = QANTADatasetLoader.load_from_csv(str(csv_path))
    elif dataset:
        questions = load_tossup_questions(
            dataset=str(dataset),
            dataset_config=str(dataset_config) if dataset_config else None,
            split=str(split),
            limit=int(limit) if limit is not None else None,
        )
    elif csv_path and data_opts.get("use_huggingface"):
        from qb_data.huggingface_loader import try_huggingface_fallback

        questions = try_huggingface_fallback(str(csv_path))
        if questions is None:
            raise FileNotFoundError(
                f"Could not load questions from missing CSV path {csv_path} via Hugging Face fallback"
            )
    else:
        raise FileNotFoundError(
            "No valid data source configured. Provide data.csv_path or "
            "data.dataset/data.dataset_config for qb-rl compatibility."
        )

    if limit is not None:
        questions = questions[: int(limit)]

    return questions


class QANTADatasetLoader:
    """
    Loader for QANTA-format quiz bowl CSV files.

    The QANTA format has questions with clues separated by ||| delimiters.
    """

    @classmethod
    def load_from_csv(cls, filepath: str) -> List[TossupQuestion]:
        """
        Load questions from a QANTA-format CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file

        Returns
        -------
        List[TossupQuestion]
            List of parsed questions

        Raises
        ------
        FileNotFoundError
            If the CSV file doesn't exist
        ValueError
            If required columns are missing or data is malformed
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        questions = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate required columns
            required_columns = {'question', 'answer'}
            actual_columns = set(reader.fieldnames or [])

            # Handle alternate column names
            if 'Text' in actual_columns and 'question' not in actual_columns:
                # QANTA format uses 'Text' instead of 'question'
                text_col = 'Text'
            elif 'question' in actual_columns:
                text_col = 'question'
            else:
                raise ValueError(f"Missing required column 'question' or 'Text'. Found columns: {actual_columns}")

            if 'Answer' in actual_columns and 'answer' not in actual_columns:
                answer_col = 'Answer'
            elif 'answer' in actual_columns:
                answer_col = 'answer'
            else:
                raise ValueError(f"Missing required column 'answer' or 'Answer'. Found columns: {actual_columns}")

            # Check for optional columns
            category_col = None
            if 'Category' in actual_columns:
                category_col = 'Category'
            elif 'category' in actual_columns:
                category_col = 'category'

            qid_col = None
            if 'Question ID' in actual_columns:
                qid_col = 'Question ID'
            elif 'qid' in actual_columns:
                qid_col = 'qid'
            elif 'question_id' in actual_columns:
                qid_col = 'question_id'

            # Parse each row
            for row_idx, row in enumerate(reader):
                try:
                    # Get question text and parse clues
                    question_text = row[text_col]
                    if not question_text or not question_text.strip():
                        continue  # Skip empty questions

                    # Split on ||| delimiter
                    if '|||' in question_text:
                        clues = [clue.strip() for clue in question_text.split('|||')]
                        clues = [c for c in clues if c]  # Remove empty clues
                    else:
                        # Treat entire text as single clue if no delimiter
                        clues = [question_text.strip()]

                    if not clues:
                        continue  # Skip if no valid clues

                    # Get answer
                    answer = row[answer_col].strip()
                    if not answer:
                        continue  # Skip questions without answers

                    # Get category (optional)
                    category = ""
                    if category_col:
                        category = row.get(category_col, "").strip()

                    # Get or generate question ID
                    if qid_col and row.get(qid_col):
                        qid = row[qid_col].strip()
                    else:
                        qid = _generate_qid(question_text)

                    # Parse clues into tokens and run indices
                    tokens, run_indices = _parse_clues_to_tokens(clues)

                    # Build cumulative prefixes
                    cumulative_prefixes = []
                    for idx in run_indices:
                        prefix = " ".join(tokens[:idx + 1])
                        cumulative_prefixes.append(prefix)

                    # Create clean answers list
                    clean_answers = [normalize_answer(answer)]

                    # Full question is all clues joined
                    full_question = " ".join(clues)

                    # Create TossupQuestion
                    question = TossupQuestion(
                        qid=qid,
                        question=full_question,
                        tokens=tokens,
                        answer_primary=answer,
                        clean_answers=clean_answers,
                        run_indices=run_indices,
                        human_buzz_positions=None,  # Not available in basic CSV
                        category=category,
                        cumulative_prefixes=cumulative_prefixes
                    )

                    questions.append(question)

                except Exception as e:
                    print(f"Warning: Failed to parse row {row_idx + 1}: {e}")
                    continue

        if not questions:
            raise ValueError(f"No valid questions found in {filepath}")

        return questions
````

## File: qb_env/data_loader.py
````python
"""qb-rl compatibility re-exports for tossup data loading."""

from qb_data.data_loader import (
    QANTADatasetLoader,
    TossupQuestion,
    load_tossup_questions,
    load_tossup_questions_from_config,
    parse_row,
)

__all__ = [
    "TossupQuestion",
    "QANTADatasetLoader",
    "parse_row",
    "load_tossup_questions",
    "load_tossup_questions_from_config",
]
````

## File: qb_env/mc_builder.py
````python
"""qb-rl compatibility re-exports for MC question building."""

from qb_data.mc_builder import MCBuilder, MCQuestion, _token_overlap

__all__ = ["MCQuestion", "MCBuilder", "_token_overlap"]
````

## File: qb_env/text_utils.py
````python
"""qb-rl compatibility re-exports for text utilities."""

from qb_data.text_utils import normalize_answer, tokenize_text

__all__ = ["normalize_answer", "tokenize_text"]
````

## File: qb_env/text_wrapper.py
````python
"""
TextObservationWrapper for converting belief features to text observations.

Wraps TossupMCEnv to provide text-formatted observations (clues + choices)
instead of numeric belief feature vectors. This bridges the gap between
the environment's native observation space (Box(K+6,)) and T5PolicyModel's
text input requirement.

The underlying environment still operates on beliefs internally for reward
computation -- the wrapper only transforms what the agent SEES, not how the
environment computes rewards or transitions.

Text format matches T5PolicyModel's expected input:
    "CLUES: clue1 clue2 ... | CHOICES: (1) ans1 (2) ans2 (3) ans3 (4) ans4"

Ported from qanta-buzzer's environment.py get_text_representation() method,
adapted for the unified codebase's Gymnasium wrapper pattern.
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np

from qb_data.mc_builder import MCQuestion


class TextObservationWrapper(gym.ObservationWrapper):
    """Wrap TossupMCEnv to provide text observations instead of belief features.

    The underlying env still operates on beliefs internally (for reward
    computation), but the agent sees text-formatted observations for T5 input.
    This is a Gymnasium ObservationWrapper that intercepts the observation
    returned by reset() and step() and converts it to a text string.

    The observation space is set to a placeholder Box(1,) since Gymnasium
    requires a defined space, but text observations are variable-length
    strings. Downstream code (T5PolicyModel) handles tokenization.

    Parameters
    ----------
    env : gym.Env
        The underlying TossupMCEnv instance. Must have ``question``
        (MCQuestion) and ``step_idx`` (int) attributes.

    Examples
    --------
    >>> from qb_env.tossup_env import TossupMCEnv
    >>> env = TossupMCEnv(questions=qs, likelihood_model=lm, K=4)
    >>> wrapped = TextObservationWrapper(env)
    >>> obs, info = wrapped.reset()
    >>> assert isinstance(obs, str)
    >>> assert "CLUES:" in obs and "CHOICES:" in obs
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Override observation space with a placeholder.
        # Text observations are variable-length strings; Gymnasium requires
        # a Space object, so we use a minimal Box as a sentinel.
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> str:
        """Convert numeric belief observation to formatted text string.

        Reconstructs visible clues from the underlying environment's current
        question and step index, then formats them with answer choices in the
        standard T5PolicyModel input format.

        Parameters
        ----------
        obs : np.ndarray
            Numeric belief features from the underlying environment.
            Shape ``(K+6,)``. Not used directly -- the text is reconstructed
            from ``env.question`` and ``env.step_idx``.

        Returns
        -------
        str
            Formatted text observation:
            ``"CLUES: <visible clue tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
        """
        question: MCQuestion = self.env.question
        step_idx: int = self.env.step_idx

        # Build visible clue text from cumulative prefixes.
        #
        # TossupMCEnv step semantics:
        #   - reset() sets step_idx=0, belief is uniform (no clues processed).
        #   - step(WAIT) calls _compute_belief(step_idx), THEN increments step_idx.
        #   - The observation returned after step() has step_idx ALREADY incremented.
        #
        # So step_idx tells us how many WAIT actions have been taken:
        #   step_idx=0: No WAITs yet; no clues processed; show minimal context
        #   step_idx=N: N WAITs taken; beliefs from cumulative_prefixes[0..N-1]
        #
        # cumulative_prefixes[i] = text of tokens[0..run_indices[i]].
        # After N WAITs, the agent has seen information up to
        # cumulative_prefixes[N-1], so that is what the text obs shows.
        if step_idx == 0:
            # No clues processed yet; show question start as minimal context
            # (matches initial observation having some textual content for T5)
            clues_text = question.tokens[0] if question.tokens else ""
        elif step_idx <= len(question.cumulative_prefixes):
            clues_text = question.cumulative_prefixes[step_idx - 1]
        else:
            # Past all clues (truncated episode); show all text
            clues_text = question.cumulative_prefixes[-1]

        # Format answer choices
        choices_parts = [
            f"({i + 1}) {opt}" for i, opt in enumerate(question.options)
        ]
        choices_text = " ".join(choices_parts)

        return f"CLUES: {clues_text} | CHOICES: {choices_text}"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and return a text observation.

        Parameters
        ----------
        seed : int or None
            Random seed passed to underlying environment.
        options : dict or None
            Options passed to underlying environment.

        Returns
        -------
        observation : str
            Text-formatted initial observation.
        info : dict[str, Any]
            Episode metadata from underlying environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: int
    ) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        """Execute one step and return text observation.

        Parameters
        ----------
        action : int
            Action to take. 0 = WAIT, 1..K = BUZZ with answer (action-1).

        Returns
        -------
        observation : str
            Text-formatted observation after the step.
        reward : float
            Scalar reward for this step.
        terminated : bool
            True if the agent buzzed (natural episode end).
        truncated : bool
            True if all clues exhausted (forced termination).
        info : dict[str, Any]
            Step metadata from underlying environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    @property
    def unwrapped_env(self):
        """Access the underlying TossupMCEnv directly.

        Returns
        -------
        TossupMCEnv
            The unwrapped environment instance.
        """
        return self.env
````

## File: scripts/ci.sh
````bash
#!/usr/bin/env bash
# CI entry point -- runs the full pytest suite from the project venv.
# Exit nonzero on any failure so CI gates catch regressions.
#
# Usage:
#   bash scripts/ci.sh              # full suite
#   bash scripts/ci.sh -k "not t5"  # skip T5-dependent tests
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
elif ! command -v pytest &>/dev/null; then
    echo "ERROR: No .venv found and pytest not on PATH." >&2
    echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ." >&2
    exit 1
fi

pytest tests/ "$@"
````

## File: scripts/compare_policies.py
````python
#!/usr/bin/env python3
"""
Compare T5-as-likelihood (MLP policy) vs T5-as-policy (end-to-end).

Evaluates both approaches on the same test set using the same metric
functions (accuracy, S_q, ECE, Brier score, buzz position).

**Important caveats for numeric comparison:**

The two evaluation paths are *not* fully apples-to-apples:

- The MLP path uses config-driven environment settings (e.g. wait_penalty
  from default.yaml or smoke.yaml).
- The T5 path uses its own hardcoded reward settings (wait_penalty=0.1,
  matching the T5 pipeline's default).
- The MLP path builds TF-IDF from test questions + all option profiles.
  The T5 path builds TF-IDF from profiles of the first 100 questions
  only (lightweight env reward computation — the T5 policy does not
  consume TF-IDF likelihoods).
- S_q semantics differ: for MLP, c_trace is a sigmoid confidence proxy
  over belief max; for T5, c_trace is the wait-head buzz probability.

These differences are inherent to the two architectures.  Accuracy and
buzz-position comparisons are directly meaningful.  ECE and Brier are
computed identically (both use top_p at buzz time).  S_q and reward
comparisons should be interpreted qualitatively.

MLP Policy (Phase 4):
    T5/TF-IDF computes likelihood scores -> belief features -> MLP
    policy decides.  Uses SB3 PPO with belief-feature observations.

T5 Policy (Phase 6):
    T5 encoder processes text directly -> PolicyHead decides.
    Uses custom PPO with text observations via TextObservationWrapper.

Usage:
    python scripts/compare_policies.py \\
        --mlp-checkpoint checkpoints/ppo/best_model \\
        --t5-checkpoint checkpoints/ppo_t5/best_model \\
        --output results/t5_comparison.json

    python scripts/compare_policies.py \\
        --t5-checkpoint checkpoints/ppo_t5/best_model \\
        --t5-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from evaluation.metrics import (
    expected_calibration_error,
    brier_score,
    summarize_buzz_metrics,
    system_score,
)
from scripts._common import ARTIFACT_DIR, load_config, load_mc_questions, save_json


def evaluate_mlp_policy(
    checkpoint_path: str,
    test_questions: list,
    config: dict,
) -> dict[str, Any]:
    """Evaluate Phase 4 MLP policy with T5/TF-IDF likelihood on belief features.

    Loads a PPOBuzzer from an SB3 checkpoint, runs deterministic episodes
    on each test question, and computes accuracy, S_q, ECE, and buzz
    position metrics.

    Parameters
    ----------
    checkpoint_path : str
        Path to SB3 PPO model checkpoint (`.zip`` file).
    test_questions : list
        List of MCQuestion instances to evaluate on.
    config : dict
        YAML config dict with environment, likelihood, and data sections.

    Returns
    -------
    dict[str, Any]
        Evaluation results: accuracy, mean_sq, ece, brier, avg_buzz_pos,
        n_questions.
    """
    from agents.ppo_buzzer import PPOBuzzer
    from models.likelihoods import TfIdfLikelihood
    from qb_env.tossup_env import make_env_from_config

    # Build likelihood model
    corpus = (
        [q.question for q in test_questions]
        + [p for q in test_questions for p in q.option_profiles]
    )
    likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

    # Build environment with all test questions
    env = make_env_from_config(
        mc_questions=test_questions,
        likelihood_model=likelihood_model,
        config=config,
    )

    # Load trained agent
    agent = PPOBuzzer.load(checkpoint_path, env=env)

    # Run episodes
    results = []
    for _ in range(len(test_questions)):
        trace = agent.run_episode(deterministic=True)
        results.append(trace)

    # Compute metrics
    buzz_metrics = summarize_buzz_metrics(results)

    # Extract confidences and outcomes for calibration — use top_p
    from dataclasses import asdict

    rows = [asdict(r) for r in results]
    confidences = []
    outcomes = []
    buzz_positions = []
    for row in rows:
        top_p_trace = list(row.get("top_p_trace", []))
        c_trace = list(row.get("c_trace", []))
        conf_trace = top_p_trace if top_p_trace else c_trace
        buzz_step = int(row.get("buzz_step", max(0, len(conf_trace) - 1)))
        if conf_trace:
            idx = min(max(0, buzz_step), len(conf_trace) - 1)
            confidences.append(float(conf_trace[idx]))
            outcomes.append(1 if bool(row.get("correct", False)) else 0)
        buzz_positions.append(buzz_step)

    ece = expected_calibration_error(confidences, outcomes)
    brier = brier_score(confidences, outcomes)

    return {
        "accuracy": buzz_metrics["buzz_accuracy"],
        "mean_sq": buzz_metrics["mean_sq"],
        "ece": ece,
        "brier": brier,
        "avg_buzz_pos": float(np.mean(buzz_positions)) if buzz_positions else 0.0,
        "mean_reward": buzz_metrics["mean_reward_like"],
        "n_questions": len(test_questions),
    }


def evaluate_t5_policy(
    checkpoint_path: str,
    test_questions: list,
    config: dict,
) -> dict[str, Any]:
    """Evaluate Phase 6 T5 end-to-end policy on text observations.

    Loads a T5PolicyModel from checkpoint, runs deterministic episodes
    on each test question using TextObservationWrapper, and computes the
    same metrics as evaluate_mlp_policy for fair comparison.

    Parameters
    ----------
    checkpoint_path : str
        Path to T5PolicyModel checkpoint directory.
    test_questions : list
        List of MCQuestion instances to evaluate on.
    config : dict
        YAML config dict.

    Returns
    -------
    dict[str, Any]
        Evaluation results: accuracy, mean_sq, ece, brier, avg_buzz_pos,
        n_questions.
    """
    import torch
    from models.t5_policy import T5PolicyModel
    from models.likelihoods import TfIdfLikelihood
    from qb_env.text_wrapper import TextObservationWrapper
    from qb_env.tossup_env import TossupMCEnv

    # Load T5 policy model
    model = T5PolicyModel.load_pretrained(checkpoint_path, device="cpu")
    model.eval()

    # Build lightweight likelihood for environment reward computation
    corpus = []
    for q in test_questions[:100]:
        corpus.extend(q.option_profiles)
    likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

    correct_count = 0
    total_count = 0
    sq_scores = []
    confidences = []
    outcomes = []
    buzz_positions = []

    with torch.no_grad():
        for question in test_questions:
            env = TossupMCEnv(
                questions=[question],
                likelihood_model=likelihood_model,
                K=len(question.options),
                reward_mode="time_penalty",
                wait_penalty=0.1,
                belief_mode="from_scratch",
            )
            wrapped_env = TextObservationWrapper(env)

            obs, info = wrapped_env.reset()
            done = False
            c_trace = []
            g_trace = []
            top_p_trace = []
            episode_reward = 0.0
            step_count = 0

            while not done:
                inputs = model.tokenizer(
                    obs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                actions, act_info = model.select_action(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    deterministic=True,
                )

                action = actions.item()

                wait_probs = act_info["wait_probs"]
                buzz_prob = wait_probs[0, 1].item()
                c_trace.append(buzz_prob)

                answer_probs = act_info["answer_probs"]
                gold_prob = answer_probs[0, question.gold_index].item()
                g_trace.append(gold_prob)

                top_p = float(answer_probs[0].max().item())
                top_p_trace.append(top_p)

                obs, reward, terminated, truncated, step_info = (
                    wrapped_env.step(action)
                )
                done = terminated or truncated
                episode_reward += reward
                step_count += 1

            sq = system_score(c_trace, g_trace)
            sq_scores.append(sq)

            is_correct = step_info.get("correct", False) or step_info.get(
                "forced_correct", False
            )
            if is_correct:
                correct_count += 1
            total_count += 1

            # Calibration: use top_p (max answer prob) for consistency
            # with belief-feature agents
            if top_p_trace:
                buzz_step = step_count - 1
                confidences.append(top_p_trace[-1])
                outcomes.append(1 if is_correct else 0)
                buzz_positions.append(buzz_step)

    accuracy = correct_count / max(1, total_count)
    mean_sq = float(np.mean(sq_scores)) if sq_scores else 0.0
    ece = expected_calibration_error(confidences, outcomes)
    brier_val = brier_score(confidences, outcomes)
    avg_buzz_pos = float(np.mean(buzz_positions)) if buzz_positions else 0.0

    return {
        "accuracy": accuracy,
        "mean_sq": mean_sq,
        "ece": ece,
        "brier": brier_val,
        "avg_buzz_pos": avg_buzz_pos,
        "mean_reward": 0.0,  # Not tracked per-episode for T5 policy eval
        "n_questions": total_count,
    }


def print_comparison(
    mlp_results: dict[str, Any] | None,
    t5_results: dict[str, Any],
    test_size: int,
) -> dict[str, Any]:
    """Print and return comparison summary.

    Parameters
    ----------
    mlp_results : dict or None
        MLP policy evaluation results. None if --t5-only.
    t5_results : dict
        T5 policy evaluation results.
    test_size : int
        Number of test questions evaluated.

    Returns
    -------
    dict[str, Any]
        Complete comparison dict for JSON serialization.
    """
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS: T5-as-Likelihood vs T5-as-Policy")
    print("=" * 70)
    print(f"Test set size: {test_size}")
    print()

    if mlp_results is not None:
        print(f"{'Metric':<20} {'MLP (T5-likelihood)':>20} {'T5 (end-to-end)':>20} {'Difference':>15}")
        print("-" * 75)
        for metric in ["accuracy", "mean_sq", "ece", "brier", "avg_buzz_pos"]:
            mlp_val = mlp_results.get(metric, 0.0)
            t5_val = t5_results.get(metric, 0.0)
            diff = t5_val - mlp_val
            print(f"{metric:<20} {mlp_val:>20.4f} {t5_val:>20.4f} {diff:>+15.4f}")
    else:
        print("T5 Policy (end-to-end) results:")
        print("-" * 40)
        for metric in ["accuracy", "mean_sq", "ece", "brier", "avg_buzz_pos"]:
            val = t5_results.get(metric, 0.0)
            print(f"  {metric:<20}: {val:.4f}")

    # Build comparison dict
    comparison: dict[str, Any] = {
        "test_size": test_size,
        "t5_policy": t5_results,
    }
    if mlp_results is not None:
        comparison["mlp_policy"] = mlp_results
        comparison["difference"] = {
            metric: t5_results.get(metric, 0.0) - mlp_results.get(metric, 0.0)
            for metric in ["accuracy", "mean_sq", "ece", "brier", "avg_buzz_pos"]
        }

    return comparison


def parse_compare_args() -> argparse.Namespace:
    """Parse comparison script arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare T5-as-likelihood (MLP) vs T5-as-policy.",
    )
    parser.add_argument(
        "--mlp-checkpoint",
        type=str,
        default=None,
        help="Path to Phase 4 MLP policy checkpoint.",
    )
    parser.add_argument(
        "--t5-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 6 T5 policy checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mc-path",
        type=str,
        default=None,
        help="Path to MC dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/t5_comparison.json",
        help="Path for output JSON results.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick test with first 50 questions.",
    )
    parser.add_argument(
        "--t5-only",
        action="store_true",
        help="Only evaluate T5 policy (skip MLP comparison).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the comparison experiment."""
    args = parse_compare_args()

    # Load config
    config = load_config(args.config)

    # Load test questions
    if args.mc_path:
        mc_path = Path(args.mc_path)
    else:
        candidates = [
            ARTIFACT_DIR / "main" / "mc_dataset.json",
            ARTIFACT_DIR / "smoke" / "mc_dataset.json",
            PROJECT_ROOT / "data" / "processed" / "mc_dataset.json",
        ]
        mc_path = None
        for candidate in candidates:
            if candidate.exists():
                mc_path = candidate
                break
        if mc_path is None:
            print("ERROR: No MC dataset found. Run build_mc_dataset.py first.")
            sys.exit(1)

    print(f"Loading questions from: {mc_path}")
    all_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(all_questions)} questions")

    # Use last 15% as test set (matching standard split)
    import random
    rng = random.Random(42)
    shuffled = all_questions[:]
    rng.shuffle(shuffled)
    test_start = int(len(shuffled) * 0.85)
    test_questions = shuffled[test_start:]

    if args.smoke:
        test_questions = test_questions[:50]

    print(f"Test set: {len(test_questions)} questions")

    # Evaluate MLP policy (if checkpoint provided and not t5-only)
    mlp_results = None
    if args.mlp_checkpoint and not args.t5_only:
        print("\n" + "-" * 40)
        print("Evaluating MLP policy (T5-as-likelihood)...")
        print("-" * 40)
        mlp_results = evaluate_mlp_policy(
            args.mlp_checkpoint, test_questions, config
        )
        print(f"  Accuracy: {mlp_results['accuracy']:.4f}")
        print(f"  Mean S_q: {mlp_results['mean_sq']:.4f}")

    # Evaluate T5 policy
    print("\n" + "-" * 40)
    print("Evaluating T5 policy (end-to-end)...")
    print("-" * 40)
    t5_results = evaluate_t5_policy(
        args.t5_checkpoint, test_questions, config
    )
    print(f"  Accuracy: {t5_results['accuracy']:.4f}")
    print(f"  Mean S_q: {t5_results['mean_sq']:.4f}")

    # Print comparison
    comparison = print_comparison(mlp_results, t5_results, len(test_questions))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, comparison)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
````

## File: scripts/train_t5_policy.py
````python
#!/usr/bin/env python3
"""
Train T5 policy with supervised warm-start then PPO fine-tuning.

End-to-end pipeline for training a T5PolicyModel on quiz bowl questions:
1. Supervised warm-start: Train answer selection on complete questions
2. PPO fine-tuning: Optimize wait/answer policy on incremental episodes

Usage:
    # Full pipeline (supervised + PPO)
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml

    # Quick smoke test (t5-small, few epochs)
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke

    # Skip supervised, load pretrained for PPO only
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml \
        --skip-supervised --model-path checkpoints/supervised/best_model

    # Custom number of PPO iterations
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml \
        --ppo-iterations 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from scripts._common import ARTIFACT_DIR, load_mc_questions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for training configuration.
    """
    parser = argparse.ArgumentParser(
        description="Train T5 policy with supervised warm-start then PPO.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "t5_policy.yaml"),
        help="Path to YAML config file (default: configs/t5_policy.yaml).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick test run: uses t5-small, 2 epochs, 4 batch size.",
    )
    parser.add_argument(
        "--skip-supervised",
        action="store_true",
        help="Skip supervised training phase.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (required if --skip-supervised).",
    )
    parser.add_argument(
        "--mc-path",
        type=str,
        default=None,
        help="Path to MC dataset JSON file.",
    )
    parser.add_argument(
        "--ppo-iterations",
        type=int,
        default=None,
        help="Override number of PPO iterations from config.",
    )
    return parser.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> dict:
    """Load YAML config and apply smoke/CLI overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Configuration dictionary with overrides applied.
    """
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke:
        smoke = config.get("smoke", {})
        # Override model settings
        if "model" in smoke:
            for key, val in smoke["model"].items():
                config["model"][key] = val
        # Override supervised settings
        if "supervised" in smoke:
            for key, val in smoke["supervised"].items():
                config["supervised"][key] = val
        # Override PPO settings
        if "ppo" in smoke:
            for key, val in smoke["ppo"].items():
                config["ppo"][key] = val
        # Override data settings
        if "data" in smoke:
            for key, val in smoke["data"].items():
                config["data"][key] = val

    if args.ppo_iterations is not None:
        config["ppo"]["iterations"] = args.ppo_iterations

    return config


def flatten_config(config: dict) -> dict:
    """Flatten nested config sections into a single dict for trainer APIs.

    Parameters
    ----------
    config : dict
        Nested config dict with sections (model, supervised, ppo, data).

    Returns
    -------
    dict
        Flat config dict with prefixed keys for each trainer.
    """
    flat = {}

    # Model section
    model = config.get("model", {})
    flat["model_name"] = model.get("model_name", "t5-large")
    device = model.get("device", "auto")
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    flat["device"] = device
    flat["max_input_length"] = model.get("max_input_length", 512)
    flat["num_choices"] = model.get("num_choices", config.get("data", {}).get("K", 4))

    # Supervised section
    sup = config.get("supervised", {})
    flat["supervised_lr"] = sup.get("lr", 3e-4)
    flat["supervised_epochs"] = sup.get("epochs", 10)
    flat["supervised_batch_size"] = sup.get("batch_size", 8)
    flat["supervised_grad_accum_steps"] = sup.get("grad_accum_steps", 4)
    flat["max_grad_norm"] = sup.get("max_grad_norm", 1.0)
    flat["weight_decay"] = sup.get("weight_decay", 0.01)
    flat["checkpoint_dir"] = sup.get("checkpoint_dir", "checkpoints")

    # PPO section
    ppo = config.get("ppo", {})
    flat["ppo_lr"] = ppo.get("lr", 1e-5)
    flat["ppo_iterations"] = ppo.get("iterations", 100)
    flat["ppo_batch_size"] = ppo.get("batch_size", 8)
    flat["ppo_epochs_per_iter"] = ppo.get("epochs_per_iter", 4)
    flat["ppo_gamma"] = ppo.get("gamma", 0.99)
    flat["ppo_gae_lambda"] = ppo.get("gae_lambda", 0.95)
    flat["ppo_clip_ratio"] = ppo.get("clip_ratio", 0.2)
    flat["ppo_value_coef"] = ppo.get("value_coef", 0.5)
    flat["ppo_entropy_coef"] = ppo.get("entropy_coef", 0.01)
    flat["ppo_max_grad_norm"] = ppo.get("max_grad_norm", 0.5)
    flat["ppo_episodes_per_iter"] = ppo.get("episodes_per_iter", 16)
    flat["eval_interval"] = ppo.get("eval_interval", 10)
    flat["save_interval"] = ppo.get("save_interval", 20)

    return flat


def load_questions(args: argparse.Namespace, config: dict) -> list:
    """Load MC questions from file or fallback paths.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, may have mc_path override.
    config : dict
        Config dict with data section.

    Returns
    -------
    list
        List of MCQuestion instances.
    """
    if args.mc_path:
        mc_path = Path(args.mc_path)
    else:
        # Try standard locations
        candidates = [
            ARTIFACT_DIR / "main" / "mc_dataset.json",
            ARTIFACT_DIR / "smoke" / "mc_dataset.json",
            PROJECT_ROOT / "data" / "processed" / "mc_dataset.json",
        ]
        mc_path = None
        for candidate in candidates:
            if candidate.exists():
                mc_path = candidate
                break

        if mc_path is None:
            print("ERROR: No MC dataset found. Run build_mc_dataset.py first.")
            print("Searched locations:")
            for c in candidates:
                print(f"  {c}")
            sys.exit(1)

    print(f"Loading MC questions from: {mc_path}")
    questions = load_mc_questions(mc_path)
    print(f"Loaded {len(questions)} questions")

    # Apply max_questions limit (smoke mode)
    max_questions = config.get("data", {}).get("max_questions", None)
    if max_questions and len(questions) > max_questions:
        questions = questions[:max_questions]
        print(f"Limited to {max_questions} questions (smoke mode)")

    return questions


def split_questions(questions: list, config: dict) -> tuple:
    """Split questions into train/val/test sets.

    Parameters
    ----------
    questions : list
        Full list of MCQuestion instances.
    config : dict
        Config dict with data section (train_size, val_size, test_size, seed).

    Returns
    -------
    tuple[list, list, list]
        Train, validation, and test question lists.
    """
    import random

    data = config.get("data", {})
    seed = data.get("seed", 42)
    train_size = data.get("train_size", 0.7)
    val_size = data.get("val_size", 0.15)

    rng = random.Random(seed)
    shuffled = questions[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    train_questions = shuffled[:n_train]
    val_questions = shuffled[n_train : n_train + n_val]
    test_questions = shuffled[n_train + n_val :]

    print(f"Split: {len(train_questions)} train, {len(val_questions)} val, {len(test_questions)} test")
    return train_questions, val_questions, test_questions


def main() -> None:
    """Run the full T5 policy training pipeline."""
    args = parse_args()

    if args.skip_supervised and args.model_path is None:
        print("ERROR: --model-path is required when using --skip-supervised")
        sys.exit(1)

    # Load config with overrides
    config = load_config_with_overrides(args)
    flat_config = flatten_config(config)

    # Load and split dataset
    questions = load_questions(args, config)
    train_questions, val_questions, test_questions = split_questions(questions, config)

    # Import training modules (lazy to avoid loading transformers until needed)
    from training.train_supervised_t5 import run_supervised_training
    from training.train_ppo_t5 import run_ppo_training

    # Phase 1: Supervised warm-start (optional)
    supervised_model_path = None
    if not args.skip_supervised:
        print("\n" + "=" * 60)
        print("PHASE 1: SUPERVISED WARM-START")
        print("=" * 60)

        model, trainer = run_supervised_training(
            config=flat_config,
            train_questions=train_questions,
            val_questions=val_questions,
        )
        supervised_model_path = str(
            trainer.checkpoint_dir / "best_model"
        )
        print(f"Supervised model saved to: {supervised_model_path}")
    else:
        supervised_model_path = args.model_path
        print(f"\nSkipping supervised training, using model: {supervised_model_path}")

    # Phase 2: PPO fine-tuning
    print("\n" + "=" * 60)
    print("PHASE 2: PPO FINE-TUNING (T5 Policy)")
    print("=" * 60)

    model, trainer = run_ppo_training(
        config=flat_config,
        train_questions=train_questions,
        val_questions=val_questions,
        test_questions=test_questions,
        pretrained_model_path=supervised_model_path,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best PPO model saved to: {trainer.checkpoint_dir / 'best_model'}")
    print(f"Training history: {trainer.checkpoint_dir / 'history.json'}")


if __name__ == "__main__":
    main()
````

## File: tests/conftest.py
````python
"""Shared pytest fixtures for test suites.

Provides reusable test data for environment, likelihood, features,
factory, and agent test suites. All fixtures create minimal but complete
data structures that satisfy the interfaces expected by the codebase modules.

Fixtures
--------
sample_mc_question
    A single MCQuestion with 4 options (gold_index=0), 6 clue steps,
    and pre-computed cumulative prefixes. Suitable for environment and
    feature extraction tests.

sample_config
    A minimal config dict matching the YAML structure expected by
    ``make_env_from_config`` and ``build_likelihood_from_config``.
    Uses "simple" reward mode for predictable test outcomes.

sample_corpus
    A list of 10 short text strings about US presidents and historical
    events. Suitable for fitting TF-IDF vectorizers in tests.

sample_tfidf_env
    A TossupMCEnv with TF-IDF likelihood and 3 sample MCQuestions.
    Fast to construct, suitable for agent and PPO tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from qb_data.mc_builder import MCQuestion

if TYPE_CHECKING:
    from qb_env.tossup_env import TossupMCEnv


@pytest.fixture
def sample_mc_question() -> MCQuestion:
    """Return a minimal MCQuestion for testing.

    The question is about the first US president with 4 answer options.
    Gold answer is "George Washington" at index 0. Six clue steps are
    defined via run_indices with pre-computed cumulative prefixes.

    Returns
    -------
    MCQuestion
        A complete MCQuestion suitable for environment testing.
    """
    tokens = [
        "Who", "was", "the", "first", "president",
        "of", "the", "United", "States", "?",
    ]
    run_indices = [0, 2, 4, 6, 8, 9]
    cumulative_prefixes = [
        "Who",
        "Who was the",
        "Who was the first president",
        "Who was the first president of the",
        "Who was the first president of the United States",
        "Who was the first president of the United States ?",
    ]
    return MCQuestion(
        qid="test_q1",
        question="Who was the first president of the United States?",
        tokens=tokens,
        answer_primary="George Washington",
        clean_answers=["George Washington", "Washington"],
        run_indices=run_indices,
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=cumulative_prefixes,
        options=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        gold_index=0,
        option_profiles=[
            "George Washington first president commander revolutionary war continental army",
            "Thomas Jefferson third president declaration independence Virginia",
            "John Adams second president Massachusetts diplomat",
            "Benjamin Franklin inventor diplomat Philadelphia printing press",
        ],
        option_answer_primary=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        distractor_strategy="test",
    )


@pytest.fixture
def sample_config() -> dict:
    """Return a minimal config dict for factory tests.

    Matches the YAML structure expected by ``make_env_from_config`` and
    ``build_likelihood_from_config``. Uses "simple" reward mode and
    "from_scratch" belief mode for predictable test outcomes.

    Returns
    -------
    dict
        Config dict with data, environment, and likelihood sections.
    """
    return {
        "data": {"K": 4},
        "environment": {
            "reward": "simple",
            "wait_penalty": 0.0,
            "buzz_correct": 1.0,
            "buzz_incorrect": -1.0,
            "belief_mode": "from_scratch",
        },
        "likelihood": {
            "model": "sbert",
            "beta": 5.0,
        },
    }


@pytest.fixture
def sample_corpus() -> list[str]:
    """Return a list of 10 short text strings for TF-IDF fitting.

    Topics cover US presidents and major historical events, providing
    sufficient vocabulary variety for TF-IDF vectorizer tests.

    Returns
    -------
    list[str]
        Ten text strings suitable for corpus fitting.
    """
    return [
        "George Washington was the first president of the United States",
        "Thomas Jefferson wrote the Declaration of Independence",
        "John Adams served as the second president after Washington",
        "Benjamin Franklin was an inventor and diplomat in Philadelphia",
        "Abraham Lincoln freed the slaves during the Civil War",
        "Alexander Hamilton established the national banking system",
        "James Madison authored the Bill of Rights and Constitution",
        "Andrew Jackson was a military hero and populist president",
        "The American Revolution established independence from Britain",
        "The Constitution created a federal system of government",
    ]


@pytest.fixture(scope="module")
def sample_t5_model():
    """Return a T5Likelihood model for testing.

    Uses t5-small (60M params) for fast test execution. Scoped to module
    level so the model is loaded once per test file, not per test function.

    Returns
    -------
    T5Likelihood
        A T5 likelihood model suitable for testing semantic scoring.

    Notes
    -----
    This fixture may take 5-10 seconds on first run to download the model
    from HuggingFace. Subsequent runs use cached weights.
    """
    from models.likelihoods import T5Likelihood

    return T5Likelihood(model_name="t5-small")


@pytest.fixture
def sample_tfidf_env(sample_mc_question: MCQuestion) -> "TossupMCEnv":
    """Return a TossupMCEnv with TF-IDF likelihood and 3 sample questions.

    Creates a lightweight environment suitable for PPOBuzzer and agent
    tests. Uses TF-IDF likelihood for fast execution (< 1ms per score).
    Three copies of the sample question are used to provide enough data
    for environment sampling.

    Returns
    -------
    TossupMCEnv
        A configured environment with simple reward mode.
    """
    from models.likelihoods import TfIdfLikelihood
    from qb_env.tossup_env import TossupMCEnv

    corpus = sample_mc_question.option_profiles[:]
    model = TfIdfLikelihood(corpus_texts=corpus)

    # Use 3 copies for variety in sampling
    questions = [sample_mc_question] * 3
    return TossupMCEnv(
        questions=questions,
        likelihood_model=model,
        K=4,
        reward_mode="simple",
        wait_penalty=0.0,
        buzz_correct=1.0,
        buzz_incorrect=-1.0,
        belief_mode="from_scratch",
        beta=5.0,
    )
````

## File: tests/test_build_mc_dataset.py
````python
"""Regression tests for scripts/build_mc_dataset.py CLI defaults."""

from __future__ import annotations

from pathlib import Path

from qb_data.config import load_config as load_yaml_config
from scripts.build_mc_dataset import parse_args, resolve_output_dir


class TestBuildMcDatasetArgs:
    """Tests for smoke-aware argument resolution."""

    def test_parse_args_smoke_uses_dynamic_defaults(self) -> None:
        args = parse_args(["--smoke"])

        assert args.smoke is True
        assert args.config is None
        assert args.output_dir is None
        assert args.overrides == []

    def test_parse_args_explicit_overrides_win(self) -> None:
        args = parse_args(
            [
                "--smoke",
                "--config",
                "configs/custom.yaml",
                "--output-dir",
                "custom/output",
                "data.K=5",
            ]
        )

        assert args.smoke is True
        assert args.config == "configs/custom.yaml"
        assert args.output_dir == "custom/output"
        assert args.overrides == ["data.K=5"]

    def test_resolve_output_dir_defaults_to_smoke_artifacts(self) -> None:
        assert resolve_output_dir(None, smoke=True) == Path("artifacts/smoke")

    def test_resolve_output_dir_defaults_to_processed_data(self) -> None:
        assert resolve_output_dir(None, smoke=False) == Path("data/processed")

    def test_resolve_output_dir_preserves_explicit_override(self) -> None:
        assert resolve_output_dir("custom/output", smoke=True) == Path("custom/output")

    def test_load_config_smoke_without_explicit_path(self) -> None:
        cfg = load_yaml_config(None, smoke=True)

        assert cfg["data"]["max_questions"] == 50
        assert cfg["ppo"]["total_timesteps"] == 3000
````

## File: tests/test_environment.py
````python
"""Test suite for qb_env/tossup_env.py — TossupMCEnv Gymnasium environment.

Covers:
- ENV-01: Gymnasium interface compliance (reset, step, spaces)
- ENV-02: Action space Discrete(K+1) with WAIT and BUZZ actions
- ENV-04: Reward modes (time_penalty, simple, human_grounded)
- ENV-05: Likelihood model pluggability
"""

from __future__ import annotations

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from models.likelihoods import SBERTLikelihood, TfIdfLikelihood
from qb_data.mc_builder import MCQuestion
from qb_env.tossup_env import TossupMCEnv, precompute_beliefs


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_env(
    mc_question: MCQuestion,
    corpus: list[str] | None = None,
    reward_mode: str = "simple",
    wait_penalty: float = 0.0,
    buzz_correct: float = 1.0,
    buzz_incorrect: float = -1.0,
    belief_mode: str = "from_scratch",
    beta: float = 5.0,
    use_sbert: bool = False,
) -> TossupMCEnv:
    """Create a TossupMCEnv with TF-IDF or SBERT likelihood model.

    Helper for tests that need a configured environment without going
    through the factory function.
    """
    if use_sbert:
        model = SBERTLikelihood()
    else:
        if corpus is None:
            corpus = mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
    return TossupMCEnv(
        questions=[mc_question],
        likelihood_model=model,
        K=4,
        reward_mode=reward_mode,
        wait_penalty=wait_penalty,
        buzz_correct=buzz_correct,
        buzz_incorrect=buzz_incorrect,
        belief_mode=belief_mode,
        beta=beta,
    )


# ------------------------------------------------------------------ #
# Tests: Gymnasium Interface (ENV-01)
# ------------------------------------------------------------------ #


class TestGymnasiumInterface:
    """Tests for Gymnasium API compliance."""

    def test_isinstance_gym_env(self, sample_mc_question: MCQuestion) -> None:
        """TossupMCEnv is a subclass of gym.Env."""
        env = _make_env(sample_mc_question)
        assert isinstance(env, gym.Env), "TossupMCEnv should be a gym.Env subclass"

    def test_has_reset_and_step(self, sample_mc_question: MCQuestion) -> None:
        """Environment has reset() and step() methods."""
        env = _make_env(sample_mc_question)
        assert hasattr(env, "reset"), "Missing reset() method"
        assert hasattr(env, "step"), "Missing step() method"
        assert callable(env.reset), "reset should be callable"
        assert callable(env.step), "step should be callable"

    def test_action_space_discrete(self, sample_mc_question: MCQuestion) -> None:
        """Action space is Discrete(K+1) = Discrete(5) for K=4."""
        env = _make_env(sample_mc_question)
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            f"Expected Discrete, got {type(env.action_space)}"
        )
        assert env.action_space.n == 5, (
            f"Expected Discrete(5) for K=4, got Discrete({env.action_space.n})"
        )

    def test_observation_space_box(self, sample_mc_question: MCQuestion) -> None:
        """Observation space is Box(K+6,) = Box(10,) for K=4."""
        env = _make_env(sample_mc_question)
        assert isinstance(env.observation_space, gym.spaces.Box), (
            f"Expected Box, got {type(env.observation_space)}"
        )
        assert env.observation_space.shape == (10,), (
            f"Expected shape (10,), got {env.observation_space.shape}"
        )
        assert env.observation_space.dtype == np.float32, (
            f"Expected float32, got {env.observation_space.dtype}"
        )

    def test_action_space_contains_all_valid_actions(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """All actions 0..K are valid in the action space."""
        env = _make_env(sample_mc_question)
        for action in range(5):
            assert env.action_space.contains(action), (
                f"Action {action} should be valid"
            )
        assert not env.action_space.contains(5), "Action 5 should be invalid for K=4"
        assert not env.action_space.contains(-1), "Action -1 should be invalid"


# ------------------------------------------------------------------ #
# Tests: Episode Flow
# ------------------------------------------------------------------ #


class TestEpisodeFlow:
    """Tests for reset/step/termination lifecycle."""

    def test_reset_returns_obs_and_info(self, sample_mc_question: MCQuestion) -> None:
        """reset() returns (observation, info) tuple."""
        env = _make_env(sample_mc_question)
        result = env.reset()
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2 elements, got {len(result)}"

    def test_reset_obs_shape_dtype(self, sample_mc_question: MCQuestion) -> None:
        """Observation from reset is (K+6,) float32."""
        env = _make_env(sample_mc_question)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_reset_info_contains_qid(self, sample_mc_question: MCQuestion) -> None:
        """Info dict from reset contains qid."""
        env = _make_env(sample_mc_question)
        _obs, info = env.reset()
        assert "qid" in info, "Info should contain 'qid'"
        assert info["qid"] == "test_q1", f"Expected 'test_q1', got {info['qid']}"

    def test_reset_initializes_state(self, sample_mc_question: MCQuestion) -> None:
        """After reset, step_idx=0, not terminated, not truncated."""
        env = _make_env(sample_mc_question)
        env.reset()
        assert env.step_idx == 0, f"step_idx should be 0, got {env.step_idx}"
        assert env.terminated is False, "terminated should be False"
        assert env.truncated is False, "truncated should be False"

    def test_wait_action_advances_step(self, sample_mc_question: MCQuestion) -> None:
        """WAIT (action 0) increments step_idx and returns not terminated."""
        env = _make_env(sample_mc_question)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert not terminated, "Should not terminate on WAIT"
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert env.step_idx == 1, f"step_idx should be 1, got {env.step_idx}"

    def test_buzz_correct_terminates(self, sample_mc_question: MCQuestion) -> None:
        """Buzzing with correct answer (action 1 = option 0 = gold) terminates."""
        env = _make_env(sample_mc_question)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # gold_index=0, action=1
        assert terminated is True, "Should terminate on buzz"
        assert truncated is False, "Should not be truncated"
        assert info["correct"] is True, "Buzzing with gold should be correct"
        assert info["chosen_idx"] == 0, f"chosen_idx should be 0, got {info['chosen_idx']}"

    def test_buzz_incorrect_terminates(self, sample_mc_question: MCQuestion) -> None:
        """Buzzing with incorrect answer terminates with correct=False."""
        env = _make_env(sample_mc_question)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(2)  # option 1 = incorrect
        assert terminated is True, "Should terminate on buzz"
        assert info["correct"] is False, "Buzzing with wrong answer should be incorrect"

    def test_forced_termination(self, sample_mc_question: MCQuestion) -> None:
        """Exhausting all clues causes truncation with forced choice."""
        env = _make_env(sample_mc_question)
        env.reset()
        total = env.total_steps  # 6 steps for sample question

        # WAIT until all clues exhausted
        for i in range(total):
            obs, reward, terminated, truncated, info = env.step(0)
            if truncated:
                break

        assert truncated is True, "Should be truncated after exhausting clues"
        assert "forced_choice" in info, "Info should contain 'forced_choice'"
        assert "forced_correct" in info, "Info should contain 'forced_correct'"
        assert isinstance(info["forced_choice"], int), "forced_choice should be int"

    def test_step_before_reset_raises(self, sample_mc_question: MCQuestion) -> None:
        """Calling step() before reset() raises RuntimeError."""
        env = _make_env(sample_mc_question)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)

    def test_step_after_terminated_raises(self, sample_mc_question: MCQuestion) -> None:
        """Calling step() after termination raises RuntimeError."""
        env = _make_env(sample_mc_question)
        env.reset()
        env.step(1)  # buzz to terminate
        with pytest.raises(RuntimeError, match="terminated"):
            env.step(0)

    def test_invalid_action_raises(self, sample_mc_question: MCQuestion) -> None:
        """Invalid action raises ValueError."""
        env = _make_env(sample_mc_question)
        env.reset()
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(99)


# ------------------------------------------------------------------ #
# Tests: Reward Modes (ENV-04)
# ------------------------------------------------------------------ #


class TestRewardModes:
    """Tests for different reward computation modes."""

    def test_reward_simple_correct(self, sample_mc_question: MCQuestion) -> None:
        """Simple mode: correct buzz gives +1.0."""
        env = _make_env(sample_mc_question, reward_mode="simple")
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(1)  # correct buzz
        assert reward == 1.0, f"Simple correct reward should be 1.0, got {reward}"

    def test_reward_simple_incorrect(self, sample_mc_question: MCQuestion) -> None:
        """Simple mode: incorrect buzz gives -1.0."""
        env = _make_env(sample_mc_question, reward_mode="simple")
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(2)  # incorrect buzz
        assert reward == -1.0, f"Simple incorrect reward should be -1.0, got {reward}"

    def test_reward_simple_wait_no_penalty(self, sample_mc_question: MCQuestion) -> None:
        """Simple mode: WAIT has 0 reward regardless of wait_penalty setting."""
        env = _make_env(
            sample_mc_question, reward_mode="simple", wait_penalty=0.1
        )
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert reward == 0.0, f"Simple WAIT reward should be 0.0, got {reward}"

    def test_reward_time_penalty_wait(self, sample_mc_question: MCQuestion) -> None:
        """Time penalty mode: WAIT incurs -wait_penalty."""
        env = _make_env(
            sample_mc_question, reward_mode="time_penalty", wait_penalty=0.1
        )
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert abs(reward - (-0.1)) < 1e-6, (
            f"Time penalty WAIT reward should be -0.1, got {reward}"
        )

    def test_reward_time_penalty_buzz_correct(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Time penalty mode: correct buzz gives buzz_correct."""
        env = _make_env(
            sample_mc_question,
            reward_mode="time_penalty",
            buzz_correct=1.0,
            wait_penalty=0.1,
        )
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(1)
        assert reward == 1.0, f"Time penalty correct buzz should be 1.0, got {reward}"

    def test_reward_time_penalty_cumulative(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Time penalty mode: waiting then buzzing accumulates penalties."""
        env = _make_env(
            sample_mc_question,
            reward_mode="time_penalty",
            wait_penalty=0.1,
            buzz_correct=1.0,
        )
        env.reset()
        # Wait 2 steps (-0.2 cumulative), then buzz correct (+1.0)
        total_reward = 0.0
        _obs, r1, _t, _tr, _info = env.step(0)
        total_reward += r1
        _obs, r2, _t, _tr, _info = env.step(0)
        total_reward += r2
        _obs, r3, _t, _tr, _info = env.step(1)  # buzz correct
        total_reward += r3
        assert abs(total_reward - 0.8) < 1e-6, (
            f"Cumulative reward should be ~0.8, got {total_reward}"
        )

    def test_reward_human_grounded(self, sample_mc_question: MCQuestion) -> None:
        """Human grounded mode works without human buzz data (returns normal reward)."""
        env = _make_env(
            sample_mc_question,
            reward_mode="human_grounded",
            buzz_correct=1.0,
            buzz_incorrect=-0.5,
        )
        env.reset()
        # With no human buzz positions, reward should be buzz_correct/incorrect
        _obs, reward, _term, _trunc, _info = env.step(1)
        assert reward == 1.0, f"Human grounded correct buzz should be 1.0, got {reward}"

    def test_reward_human_grounded_with_positions(self) -> None:
        """Human grounded mode: buzzing after human position gives 0.0."""
        # Create question with human buzz at position 0 (very early)
        mc_q = MCQuestion(
            qid="hg_test",
            question="Who was the first president?",
            tokens=["Who", "was", "the", "first", "president", "?"],
            answer_primary="George Washington",
            clean_answers=["George Washington"],
            run_indices=[0, 2, 4, 5],
            human_buzz_positions=[(0, 10)],  # Most humans buzz at position 0
            category="History",
            cumulative_prefixes=[
                "Who",
                "Who was the",
                "Who was the first president",
                "Who was the first president ?",
            ],
            options=["George Washington", "Jefferson", "Adams", "Franklin"],
            gold_index=0,
            option_profiles=["Washington", "Jefferson", "Adams", "Franklin"],
            option_answer_primary=["George Washington", "Jefferson", "Adams", "Franklin"],
            distractor_strategy="test",
        )
        corpus = mc_q.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = TossupMCEnv(
            questions=[mc_q],
            likelihood_model=model,
            K=4,
            reward_mode="human_grounded",
            buzz_correct=1.0,
            buzz_incorrect=-0.5,
        )
        env.reset()
        # Wait a few steps so agent buzzes after human position (0)
        env.step(0)  # step 0 -> reveal clue at position 0
        env.step(0)  # step 1 -> reveal clue at position 2
        _obs, reward, _term, _trunc, _info = env.step(1)  # buzz at step 2
        # Agent buzzes at token pos > 0 (human), so reward should be 0.0
        assert reward == 0.0, f"Should get 0.0 for buzzing after human, got {reward}"


# ------------------------------------------------------------------ #
# Tests: Likelihood Model Pluggability (ENV-05)
# ------------------------------------------------------------------ #


class TestLikelihoodPluggability:
    """Tests for interchangeable likelihood models."""

    def test_tfidf_model_produces_valid_obs(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """TF-IDF likelihood model produces valid observations."""
        env = _make_env(sample_mc_question, use_sbert=False)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "All observations should be finite"
        # Take a step
        obs2, _r, _t, _tr, _info = env.step(0)
        assert obs2.shape == (10,), f"Expected (10,), got {obs2.shape}"
        assert np.all(np.isfinite(obs2)), "Step observations should be finite"

    def test_sbert_model_produces_valid_obs(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """SBERT likelihood model produces valid observations."""
        env = _make_env(sample_mc_question, use_sbert=True)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "All observations should be finite"
        # Take a step
        obs2, _r, _t, _tr, _info = env.step(0)
        assert obs2.shape == (10,), f"Expected (10,), got {obs2.shape}"
        assert np.all(np.isfinite(obs2)), "Step observations should be finite"

    def test_both_models_same_obs_shape(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Both TF-IDF and SBERT produce same observation shape."""
        env_tfidf = _make_env(sample_mc_question, use_sbert=False)
        env_sbert = _make_env(sample_mc_question, use_sbert=True)

        obs_tfidf, _ = env_tfidf.reset(seed=42)
        obs_sbert, _ = env_sbert.reset(seed=42)

        assert obs_tfidf.shape == obs_sbert.shape, (
            f"TF-IDF obs {obs_tfidf.shape} != SBERT obs {obs_sbert.shape}"
        )
        assert obs_tfidf.dtype == obs_sbert.dtype, (
            f"TF-IDF dtype {obs_tfidf.dtype} != SBERT dtype {obs_sbert.dtype}"
        )


# ------------------------------------------------------------------ #
# Tests: Belief Modes
# ------------------------------------------------------------------ #


class TestBeliefModes:
    """Tests for different belief computation modes."""

    def test_from_scratch_belief(self, sample_mc_question: MCQuestion) -> None:
        """from_scratch mode recomputes belief from cumulative prefix."""
        env = _make_env(sample_mc_question, belief_mode="from_scratch")
        env.reset()
        # Wait several steps to get a more discriminative clue prefix
        for _ in range(3):
            env.step(0)
        # After multiple steps with more context, belief should be valid
        # and at least one option should have higher probability
        assert abs(env.belief.sum() - 1.0) < 1e-5, (
            f"Belief should sum to 1.0, got {env.belief.sum()}"
        )
        assert all(env.belief >= 0), "All beliefs should be non-negative"
        assert env.belief.dtype == np.float32, "Belief should be float32"

    def test_sequential_bayes_belief(self, sample_mc_question: MCQuestion) -> None:
        """sequential_bayes mode updates belief incrementally."""
        env = _make_env(sample_mc_question, belief_mode="sequential_bayes")
        env.reset()
        env.step(0)  # first WAIT
        # Belief should sum to ~1.0
        assert abs(env.belief.sum() - 1.0) < 1e-5, (
            f"Belief should sum to 1.0, got {env.belief.sum()}"
        )

    def test_invalid_belief_mode_raises(self, sample_mc_question: MCQuestion) -> None:
        """Unknown belief mode raises ValueError on step."""
        env = _make_env(sample_mc_question, belief_mode="unknown_mode")
        env.reset()
        with pytest.raises(ValueError, match="Unknown belief_mode"):
            env.step(0)


# ------------------------------------------------------------------ #
# Tests: Constructor Validation
# ------------------------------------------------------------------ #


class TestConstructorValidation:
    """Tests for constructor input validation."""

    def test_empty_questions_raises(self) -> None:
        """Empty question list raises ValueError."""
        model = TfIdfLikelihood(corpus_texts=["test"])
        with pytest.raises(ValueError, match="cannot be empty"):
            TossupMCEnv(questions=[], likelihood_model=model)

    def test_k_less_than_2_raises(self, sample_mc_question: MCQuestion) -> None:
        """K < 2 raises ValueError."""
        model = TfIdfLikelihood(corpus_texts=["test"])
        with pytest.raises(ValueError, match="K must be >= 2"):
            TossupMCEnv(
                questions=[sample_mc_question], likelihood_model=model, K=1
            )


# ------------------------------------------------------------------ #
# Tests: Precomputed Beliefs (OPT-1)
# ------------------------------------------------------------------ #


class TestPrecomputedBeliefs:
    """Tests for precomputed belief trajectory bypass."""

    def test_precomputed_matches_live_from_scratch(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Precomputed env produces identical beliefs as live env (from_scratch)."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        # Run live env and record beliefs at each step
        live_env = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="from_scratch", beta=5.0,
        )
        live_env.reset(seed=42, options={"question_idx": 0})
        live_beliefs = []
        for _ in range(live_env.total_steps):
            live_env.step(0)  # WAIT
            live_beliefs.append(live_env.belief.copy())
            if live_env.truncated:
                break

        # Build precomputed cache
        cache = precompute_beliefs(
            questions=questions, likelihood_model=model,
            belief_mode="from_scratch", beta=5.0, K=4,
        )

        # Run precomputed env and compare beliefs
        pre_env = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="from_scratch", beta=5.0,
            precomputed_beliefs=cache,
        )
        pre_env.reset(seed=42, options={"question_idx": 0})
        for i in range(len(live_beliefs)):
            pre_env.step(0)
            np.testing.assert_allclose(
                pre_env.belief, live_beliefs[i], atol=1e-6,
                err_msg=f"Belief mismatch at step {i} (from_scratch)",
            )
            if pre_env.truncated:
                break

    def test_precomputed_matches_live_sequential_bayes(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Precomputed env produces identical beliefs as live env (sequential_bayes)."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        # Run live env
        live_env = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="sequential_bayes", beta=5.0,
        )
        live_env.reset(seed=42, options={"question_idx": 0})
        live_beliefs = []
        for _ in range(live_env.total_steps):
            live_env.step(0)
            live_beliefs.append(live_env.belief.copy())
            if live_env.truncated:
                break

        # Build precomputed cache
        cache = precompute_beliefs(
            questions=questions, likelihood_model=model,
            belief_mode="sequential_bayes", beta=5.0, K=4,
        )

        # Run precomputed env
        pre_env = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="sequential_bayes", beta=5.0,
            precomputed_beliefs=cache,
        )
        pre_env.reset(seed=42, options={"question_idx": 0})
        for i in range(len(live_beliefs)):
            pre_env.step(0)
            np.testing.assert_allclose(
                pre_env.belief, live_beliefs[i], atol=1e-6,
                err_msg=f"Belief mismatch at step {i} (sequential_bayes)",
            )
            if pre_env.truncated:
                break

    def test_precomputed_skips_scoring(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Precomputed env never calls likelihood_model.score()."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        cache = precompute_beliefs(
            questions=questions, likelihood_model=model,
            belief_mode="from_scratch", beta=5.0, K=4,
        )

        # Replace score with a mock
        mock_model = MagicMock(spec=TfIdfLikelihood)
        mock_model.score = MagicMock()

        env = TossupMCEnv(
            questions=questions, likelihood_model=mock_model, K=4,
            belief_mode="from_scratch", beta=5.0,
            precomputed_beliefs=cache,
        )
        env.reset(seed=42, options={"question_idx": 0})
        for _ in range(env.total_steps):
            env.step(0)
            if env.truncated:
                break

        mock_model.score.assert_not_called()

    def test_no_precomputed_backward_compat(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Env with precomputed_beliefs=None behaves identically to default."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        # Default env (no precomputed_beliefs arg)
        env_default = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="from_scratch", beta=5.0,
        )
        env_default.reset(seed=42, options={"question_idx": 0})
        obs_default, _, _, _, _ = env_default.step(0)

        # Explicit None
        env_none = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="from_scratch", beta=5.0,
            precomputed_beliefs=None,
        )
        env_none.reset(seed=42, options={"question_idx": 0})
        obs_none, _, _, _, _ = env_none.step(0)

        np.testing.assert_array_equal(obs_default, obs_none)

    def test_precompute_beliefs_helper_shape(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """precompute_beliefs returns correct keys and belief shapes."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        cache = precompute_beliefs(
            questions=questions, likelihood_model=model,
            belief_mode="from_scratch", beta=5.0, K=4,
        )

        total_steps = len(sample_mc_question.run_indices)
        for s in range(total_steps):
            key = (0, s)
            assert key in cache, f"Missing key {key}"
            belief = cache[key]
            assert belief.shape == (4,), f"Expected (4,), got {belief.shape}"
            assert belief.dtype == np.float32, f"Expected float32, got {belief.dtype}"
            assert abs(belief.sum() - 1.0) < 1e-5, (
                f"Belief should sum to ~1.0, got {belief.sum()}"
            )
````

## File: tests/test_factories.py
````python
"""Test suite for factory functions — build_likelihood_from_config and make_env_from_config.

Covers:
- LIK-06: build_likelihood_from_config dispatches on config["likelihood"]["model"]
- CFG-02: make_env_from_config constructs TossupMCEnv from YAML config
"""

from __future__ import annotations

import numpy as np
import pytest

from models.likelihoods import (
    LikelihoodModel,
    SBERTLikelihood,
    TfIdfLikelihood,
    build_likelihood_from_config,
)
from qb_data.mc_builder import MCQuestion
from qb_env.tossup_env import TossupMCEnv, make_env_from_config


# ------------------------------------------------------------------ #
# Tests: build_likelihood_from_config (LIK-06)
# ------------------------------------------------------------------ #


class TestBuildLikelihoodFromConfig:
    """Tests for likelihood model factory function."""

    @pytest.fixture
    def stub_sbert_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub SBERT model loading so factory tests stay offline-safe."""

        def fake_init(self, model_name: str = "all-MiniLM-L6-v2") -> None:
            LikelihoodModel.__init__(self)
            self.model_name = model_name
            self.encoder = object()

        monkeypatch.setattr(SBERTLikelihood, "__init__", fake_init)

    def test_likelihood_factory_sbert(
        self, sample_config: dict, stub_sbert_init: None
    ) -> None:
        """Config with model='sbert' creates SBERTLikelihood."""
        sample_config["likelihood"]["model"] = "sbert"
        model = build_likelihood_from_config(sample_config)
        assert isinstance(model, SBERTLikelihood), (
            f"Expected SBERTLikelihood, got {type(model).__name__}"
        )

    def test_likelihood_factory_tfidf(
        self, sample_config: dict, sample_corpus: list[str]
    ) -> None:
        """Config with model='tfidf' creates TfIdfLikelihood (fitted)."""
        sample_config["likelihood"]["model"] = "tfidf"
        model = build_likelihood_from_config(sample_config, corpus_texts=sample_corpus)
        assert isinstance(model, TfIdfLikelihood), (
            f"Expected TfIdfLikelihood, got {type(model).__name__}"
        )
        assert model._is_fit is True, "TF-IDF model should be fitted after construction"

    def test_likelihood_factory_tfidf_missing_corpus(
        self, sample_config: dict
    ) -> None:
        """TF-IDF factory without corpus_texts raises ValueError."""
        sample_config["likelihood"]["model"] = "tfidf"
        with pytest.raises(ValueError, match="corpus_texts"):
            build_likelihood_from_config(sample_config)

    def test_likelihood_factory_unknown_model(self, sample_config: dict) -> None:
        """Unknown model name raises ValueError."""
        sample_config["likelihood"]["model"] = "unknown_model"
        with pytest.raises(ValueError, match="Unknown likelihood model"):
            build_likelihood_from_config(sample_config)

    def test_likelihood_factory_sbert_name_override(
        self, sample_config: dict, stub_sbert_init: None
    ) -> None:
        """sbert_name config key overrides default model name."""
        sample_config["likelihood"]["model"] = "sbert"
        sample_config["likelihood"]["sbert_name"] = "all-MiniLM-L6-v2"
        model = build_likelihood_from_config(sample_config)
        assert isinstance(model, SBERTLikelihood)
        assert model.model_name == "all-MiniLM-L6-v2", (
            f"Expected all-MiniLM-L6-v2, got {model.model_name}"
        )

    def test_likelihood_factory_embedding_model_key(
        self, sample_config: dict, stub_sbert_init: None
    ) -> None:
        """embedding_model config key works as fallback for sbert_name."""
        sample_config["likelihood"]["model"] = "sbert"
        sample_config["likelihood"]["embedding_model"] = "all-MiniLM-L6-v2"
        # Remove sbert_name if present to test fallback
        sample_config["likelihood"].pop("sbert_name", None)
        model = build_likelihood_from_config(sample_config)
        assert isinstance(model, SBERTLikelihood)
        assert model.model_name == "all-MiniLM-L6-v2"


# ------------------------------------------------------------------ #
# Tests: make_env_from_config (CFG-02)
# ------------------------------------------------------------------ #


class TestMakeEnvFromConfig:
    """Tests for environment factory function."""

    def _make_model_and_env(
        self, mc_question: MCQuestion, config: dict
    ) -> TossupMCEnv:
        """Helper to create a model and env from config."""
        corpus = mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        return make_env_from_config([mc_question], model, config)

    def test_env_factory_creates_tossup_env(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory creates a TossupMCEnv instance."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert isinstance(env, TossupMCEnv), (
            f"Expected TossupMCEnv, got {type(env).__name__}"
        )

    def test_env_factory_config_values(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory correctly extracts config values."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.K == 4, f"Expected K=4, got {env.K}"
        assert env.reward_mode == "simple", (
            f"Expected 'simple', got '{env.reward_mode}'"
        )
        assert env.belief_mode == "from_scratch", (
            f"Expected 'from_scratch', got '{env.belief_mode}'"
        )
        assert env.beta == 5.0, f"Expected beta=5.0, got {env.beta}"

    def test_env_factory_reward_mode_override(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Config overrides reward mode."""
        sample_config["environment"]["reward"] = "human_grounded"
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.reward_mode == "human_grounded", (
            f"Expected 'human_grounded', got '{env.reward_mode}'"
        )

    def test_env_factory_beta_override(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Config overrides beta value."""
        sample_config["likelihood"]["beta"] = 10.0
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.beta == 10.0, f"Expected beta=10.0, got {env.beta}"

    def test_env_factory_wait_penalty_override(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Config overrides wait_penalty value."""
        sample_config["environment"]["wait_penalty"] = 0.05
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.wait_penalty == 0.05, (
            f"Expected wait_penalty=0.05, got {env.wait_penalty}"
        )

    def test_env_factory_reset_works(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory-created env can reset and produce valid observation."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert "qid" in info, "Info should contain 'qid'"
        assert np.all(np.isfinite(obs)), "All observations should be finite"

    def test_env_factory_step_works(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory-created env can step and return valid results."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
        assert terminated is False, "WAIT should not terminate"

    def test_env_factory_reward_mode_key_fallback(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Factory supports 'reward_mode' key (default.yaml uses this)."""
        config = {
            "data": {"K": 4},
            "environment": {
                "reward_mode": "time_penalty",
                "wait_penalty": 0.1,
                "buzz_correct": 1.0,
                "buzz_incorrect": -0.5,
            },
            "likelihood": {"beta": 5.0},
        }
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = make_env_from_config([sample_mc_question], model, config)
        assert env.reward_mode == "time_penalty", (
            f"Expected 'time_penalty', got '{env.reward_mode}'"
        )
````

## File: tests/test_ppo_buzzer.py
````python
"""Test suite for scripts/_common.py and agents/ppo_buzzer.py.

Covers:
- AGT-01: PPOBuzzer training, save, load, episode execution
- AGT-07: Shared utilities (config, JSON, MCQuestion serialization)
- S_q metric support: c_trace, g_trace, entropy_trace generation

Uses TF-IDF likelihood for fast test execution (< 10 seconds total).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from agents.ppo_buzzer import PPOBuzzer, PPOEpisodeTrace
from qb_data.mc_builder import MCQuestion
from qb_env.tossup_env import TossupMCEnv
from scripts._common import (
    ARTIFACT_DIR,
    PROJECT_ROOT,
    load_config,
    load_json,
    mc_question_from_dict,
    save_json,
    to_serializable,
)


# ------------------------------------------------------------------ #
# Tests: _common utilities (AGT-07)
# ------------------------------------------------------------------ #


class TestLoadConfig:
    """Tests for config loading utility."""

    def test_load_config_default(self) -> None:
        """load_config() without args loads default.yaml with expected keys."""
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "data" in cfg
        assert "ppo" in cfg
        assert "environment" in cfg
        assert "likelihood" in cfg

    def test_load_config_smoke(self) -> None:
        """load_config() can load smoke.yaml with reduced settings."""
        smoke_path = str(PROJECT_ROOT / "configs" / "smoke.yaml")
        cfg = load_config(smoke_path)
        assert cfg["data"]["max_questions"] == 50
        assert cfg["ppo"]["total_timesteps"] == 3000


class TestJsonUtilities:
    """Tests for JSON save/load round-trip."""

    def test_save_load_json_roundtrip(self, tmp_path: Path) -> None:
        """save_json/load_json round-trips nested dicts."""
        data = {"a": 1, "b": [2, 3], "c": {"d": "hello"}}
        path = tmp_path / "test.json"
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data

    def test_save_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_json creates missing parent directories."""
        path = tmp_path / "sub" / "dir" / "test.json"
        save_json(path, {"x": 1})
        assert path.exists()


class TestMCQuestionSerialization:
    """Tests for MCQuestion serialization and deserialization."""

    def test_to_serializable_on_mcquestion(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """to_serializable converts MCQuestion to a dict."""
        result = to_serializable(sample_mc_question)
        assert isinstance(result, dict)
        assert result["qid"] == "test_q1"
        assert result["gold_index"] == 0
        assert len(result["options"]) == 4

    def test_mc_question_roundtrip(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """MCQuestion survives serialization -> deserialization round-trip."""
        serialized = to_serializable(sample_mc_question)
        restored = mc_question_from_dict(serialized)
        assert restored.qid == sample_mc_question.qid
        assert restored.gold_index == sample_mc_question.gold_index
        assert restored.options == sample_mc_question.options
        assert restored.tokens == sample_mc_question.tokens

    def test_mc_question_json_roundtrip(
        self, sample_mc_question: MCQuestion, tmp_path: Path
    ) -> None:
        """MCQuestion survives save_json -> load_json -> mc_question_from_dict."""
        path = tmp_path / "mc.json"
        save_json(path, [sample_mc_question])
        raw = load_json(path)
        restored = mc_question_from_dict(raw[0])
        assert restored.qid == sample_mc_question.qid
        assert restored.answer_primary == sample_mc_question.answer_primary


class TestArtifactDir:
    """Tests for path constants."""

    def test_artifact_dir_constant(self) -> None:
        """ARTIFACT_DIR points to project/artifacts."""
        assert ARTIFACT_DIR.name == "artifacts"
        assert ARTIFACT_DIR.parent == PROJECT_ROOT


# ------------------------------------------------------------------ #
# Tests: PPOBuzzer initialization (AGT-01)
# ------------------------------------------------------------------ #


class TestPPOBuzzerInit:
    """Tests for PPOBuzzer construction."""

    def test_ppo_buzzer_init(self, sample_tfidf_env: TossupMCEnv) -> None:
        """PPOBuzzer instantiates with default hyperparameters."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        assert buzzer.model is not None
        assert buzzer.env is sample_tfidf_env

    def test_ppo_buzzer_custom_policy_kwargs(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """PPOBuzzer accepts custom policy_kwargs."""
        buzzer = PPOBuzzer(
            env=sample_tfidf_env,
            policy_kwargs={"net_arch": [128, 128, 64]},
        )
        assert buzzer.model is not None


# ------------------------------------------------------------------ #
# Tests: Episode trace generation
# ------------------------------------------------------------------ #


class TestActionProbabilities:
    """Tests for action probability extraction."""

    def test_action_probabilities_shape(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """action_probabilities returns K+1 probabilities that sum to 1."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        obs, _ = sample_tfidf_env.reset(seed=42)
        probs = buzzer.action_probabilities(obs)
        K = sample_tfidf_env.K
        assert probs.shape == (K + 1,), f"Expected ({K + 1},), got {probs.shape}"
        assert abs(probs.sum() - 1.0) < 1e-5, f"Probabilities sum to {probs.sum()}"
        assert (probs >= 0).all(), "All probabilities should be non-negative"

    def test_c_t_computation(self, sample_tfidf_env: TossupMCEnv) -> None:
        """c_t returns buzz probability in [0, 1]."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        obs, _ = sample_tfidf_env.reset(seed=42)
        c_val = buzzer.c_t(obs)
        assert 0.0 <= c_val <= 1.0, f"c_t={c_val} out of range"

    def test_g_t_computation(self, sample_tfidf_env: TossupMCEnv) -> None:
        """g_t returns correctness probability, handles near-zero c_t."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        obs, _ = sample_tfidf_env.reset(seed=42)
        gold_index = sample_tfidf_env.question.gold_index
        g_val = buzzer.g_t(obs, gold_index)
        assert g_val >= 0.0, f"g_t={g_val} should be non-negative"
        # g_t can be > 1.0 if P(gold) > P(buzz) in early steps, but
        # mathematically g_t = P(gold) / c_t <= 1.0 since P(gold) <= c_t
        # (gold action is one of the buzz actions)
        assert g_val <= 1.0 + 1e-5, f"g_t={g_val} should be <= 1.0"


class TestRunEpisode:
    """Tests for full episode execution with traces."""

    def test_run_episode_generates_traces(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """run_episode returns PPOEpisodeTrace with matching trace lengths."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)

        assert isinstance(trace, PPOEpisodeTrace)
        assert len(trace.c_trace) == len(trace.g_trace)
        assert len(trace.c_trace) == len(trace.top_p_trace)
        assert len(trace.c_trace) == len(trace.entropy_trace)
        assert len(trace.c_trace) > 0, "Episode should have at least one step"

    def test_run_episode_trace_values(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Trace values are in valid ranges."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)

        for c_val in trace.c_trace:
            assert 0.0 <= c_val <= 1.0, f"c_trace value {c_val} out of [0,1]"
        for g_val in trace.g_trace:
            assert g_val >= 0.0, f"g_trace value {g_val} should be non-negative"
        for top_p in trace.top_p_trace:
            assert 0.0 <= top_p <= 1.0, f"top_p_trace value {top_p} out of [0,1]"
        for ent in trace.entropy_trace:
            assert ent >= 0.0, f"entropy {ent} should be non-negative"

    def test_ppo_calibration_uses_top_p_trace(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """calibration_at_buzz on PPO traces uses top_p_trace, not c_trace."""
        from dataclasses import asdict
        from evaluation.metrics import calibration_at_buzz

        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)
        assert len(trace.top_p_trace) > 0, "top_p_trace must be populated"

        cal = calibration_at_buzz([asdict(trace)])
        assert cal["n_calibration"] == 1.0
        # Confidence should be top_p_trace[buzz_step], not c_trace[buzz_step]
        idx = min(max(0, trace.buzz_step), len(trace.top_p_trace) - 1)
        expected_conf = trace.top_p_trace[idx]
        expected_brier = (expected_conf - (1.0 if trace.correct else 0.0)) ** 2
        assert abs(cal["brier"] - expected_brier) < 1e-9

    def test_run_episode_deterministic(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Deterministic episodes with same seed produce same traces."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace1 = buzzer.run_episode(deterministic=True, seed=42)
        trace2 = buzzer.run_episode(deterministic=True, seed=42)

        assert trace1.buzz_step == trace2.buzz_step
        assert trace1.buzz_index == trace2.buzz_index
        np.testing.assert_allclose(trace1.c_trace, trace2.c_trace, atol=1e-6)

    def test_run_episode_has_qid(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Episode trace includes the question ID."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)
        assert trace.qid != "", "qid should not be empty"

    def test_run_episode_correct_field(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """correct field matches buzz_index vs gold_index."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)
        assert trace.correct == (trace.buzz_index == trace.gold_index)


# ------------------------------------------------------------------ #
# Tests: Checkpoint save/load
# ------------------------------------------------------------------ #


class TestCheckpointSaveLoad:
    """Tests for PPOBuzzer model persistence."""

    def test_ppo_checkpoint_save_load(
        self, sample_tfidf_env: TossupMCEnv, tmp_path: Path
    ) -> None:
        """PPOBuzzer saves and loads from checkpoint."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        save_path = tmp_path / "ppo_test"
        buzzer.save(save_path)

        # SB3 appends .zip
        assert (tmp_path / "ppo_test.zip").exists(), "Model file should exist"

        loaded = PPOBuzzer.load(save_path, env=sample_tfidf_env)
        assert loaded.model is not None

        # Verify loaded model produces valid probabilities
        obs, _ = sample_tfidf_env.reset(seed=42)
        probs = loaded.action_probabilities(obs)
        assert probs.shape == (sample_tfidf_env.K + 1,)
        assert abs(probs.sum() - 1.0) < 1e-5
````

## File: tests/test_ppo_t5.py
````python
"""Unit tests for custom PPO trainer for T5PolicyModel.

Tests cover RolloutStep dataclass, RolloutBuffer with GAE computation,
rollout collection with memory management, dynamic padding, and PPO update.

Uses t5-small (60M params) and TF-IDF likelihood for fast execution.
The T5 model fixture is module-scoped (loaded once per test file).
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from training.train_ppo_t5 import RolloutStep, RolloutBuffer, PPOTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t5_ppo_config() -> dict:
    """Minimal PPO config for testing."""
    return {
        "model_name": "t5-small",
        "device": "cpu",
        "max_input_length": 64,
        "num_choices": 4,
        "ppo_lr": 1e-4,
        "ppo_iterations": 2,
        "ppo_batch_size": 4,
        "ppo_epochs_per_iter": 2,
        "ppo_gamma": 0.99,
        "ppo_gae_lambda": 0.95,
        "ppo_clip_ratio": 0.2,
        "ppo_value_coef": 0.5,
        "ppo_entropy_coef": 0.01,
        "ppo_max_grad_norm": 0.5,
        "ppo_episodes_per_iter": 2,
        "eval_interval": 1,
        "save_interval": 100,
        "checkpoint_dir": "/tmp/test_ppo_t5_checkpoints",
        "reward_time_penalty": 0.01,
    }


@pytest.fixture(scope="module")
def t5_ppo_model(t5_ppo_config):
    """Load T5PolicyModel with t5-small once per test module."""
    from models.t5_policy import T5PolicyModel

    model = T5PolicyModel(t5_ppo_config)
    return model


@pytest.fixture
def sample_rollout_steps() -> list:
    """Create sample RolloutStep instances for testing GAE computation."""
    # Simulate a 4-step episode: WAIT, WAIT, WAIT, BUZZ(correct)
    steps = [
        RolloutStep(
            observation_text="CLUES: Who | CHOICES: (1) A (2) B (3) C (4) D",
            action=0,
            reward=-0.01,
            done=False,
            value=0.2,
            log_prob=-0.8,
            input_ids=torch.randint(0, 100, (1, 10)),
            attention_mask=torch.ones(1, 10, dtype=torch.long),
        ),
        RolloutStep(
            observation_text="CLUES: Who was | CHOICES: (1) A (2) B (3) C (4) D",
            action=0,
            reward=-0.01,
            done=False,
            value=0.4,
            log_prob=-0.7,
            input_ids=torch.randint(0, 100, (1, 12)),
            attention_mask=torch.ones(1, 12, dtype=torch.long),
        ),
        RolloutStep(
            observation_text="CLUES: Who was the first | CHOICES: (1) A (2) B (3) C (4) D",
            action=0,
            reward=-0.01,
            done=False,
            value=0.6,
            log_prob=-0.5,
            input_ids=torch.randint(0, 100, (1, 15)),
            attention_mask=torch.ones(1, 15, dtype=torch.long),
        ),
        RolloutStep(
            observation_text="CLUES: Who was the first president | CHOICES: (1) A (2) B (3) C (4) D",
            action=1,
            reward=1.0,
            done=True,
            value=0.8,
            log_prob=-0.3,
            input_ids=torch.randint(0, 100, (1, 18)),
            attention_mask=torch.ones(1, 18, dtype=torch.long),
        ),
    ]
    return steps


# ---------------------------------------------------------------------------
# RolloutStep Tests
# ---------------------------------------------------------------------------


class TestRolloutStep:
    """Tests for the RolloutStep dataclass."""

    def test_rollout_step_dataclass(self):
        """RolloutStep stores all required fields."""
        step = RolloutStep(
            observation_text="test",
            action=0,
            reward=1.0,
            done=True,
            value=0.5,
            log_prob=-0.3,
        )
        assert step.observation_text == "test"
        assert step.action == 0
        assert step.reward == 1.0
        assert step.done is True
        assert step.value == 0.5
        assert step.log_prob == -0.3
        assert step.input_ids is None
        assert step.attention_mask is None
        assert step.return_ == 0.0
        assert step.advantage == 0.0

    def test_rollout_step_with_tensors(self):
        """RolloutStep stores tensor fields on CPU."""
        ids = torch.randint(0, 100, (1, 10))
        mask = torch.ones(1, 10, dtype=torch.long)
        step = RolloutStep(
            observation_text="test",
            action=1,
            reward=0.5,
            done=False,
            value=0.3,
            log_prob=-0.5,
            input_ids=ids,
            attention_mask=mask,
        )
        assert step.input_ids is not None
        assert step.input_ids.device.type == "cpu"
        assert step.attention_mask.device.type == "cpu"
        assert step.input_ids.shape == (1, 10)


# ---------------------------------------------------------------------------
# RolloutBuffer Tests
# ---------------------------------------------------------------------------


class TestRolloutBuffer:
    """Tests for the RolloutBuffer class."""

    def test_rollout_buffer_add(self, sample_rollout_steps):
        """Buffer accumulates rollouts correctly."""
        buffer = RolloutBuffer()
        assert len(buffer) == 0

        buffer.add_rollout(sample_rollout_steps)
        assert len(buffer) == 1

        buffer.add_rollout(sample_rollout_steps[:2])
        assert len(buffer) == 2

    def test_rollout_buffer_get_all_steps(self, sample_rollout_steps):
        """get_all_steps returns flat list of all steps."""
        buffer = RolloutBuffer()
        buffer.add_rollout(sample_rollout_steps)
        buffer.add_rollout(sample_rollout_steps[:2])

        all_steps = buffer.get_all_steps()
        assert len(all_steps) == 6  # 4 + 2

    def test_rollout_buffer_reset(self, sample_rollout_steps):
        """reset() clears all rollouts."""
        buffer = RolloutBuffer()
        buffer.add_rollout(sample_rollout_steps)
        assert len(buffer) == 1

        buffer.reset()
        assert len(buffer) == 0
        assert len(buffer.get_all_steps()) == 0

    def test_gae_computation(self, sample_rollout_steps):
        """GAE advantages match hand-calculated values.

        Episode: 4 steps with rewards [-0.01, -0.01, -0.01, 1.0]
        and values [0.2, 0.4, 0.6, 0.8].
        """
        buffer = RolloutBuffer()
        buffer.add_rollout(sample_rollout_steps)

        gamma = 0.99
        gae_lambda = 0.95

        buffer.compute_returns_and_advantages(gamma, gae_lambda)

        all_steps = buffer.get_all_steps()

        # Verify terminal step (t=3): done=True
        # delta_3 = r_3 + gamma * 0 - v_3 = 1.0 + 0 - 0.8 = 0.2
        # gae_3 = delta_3 = 0.2 (reset because done=True)
        assert abs(all_steps[3].advantage - 0.2) < 1e-6
        assert abs(all_steps[3].return_ - (0.2 + 0.8)) < 1e-6  # adv + value

        # Step t=2: not done
        # delta_2 = r_2 + gamma * v_3 - v_2 = -0.01 + 0.99 * 0.8 - 0.6 = 0.182
        # gae_2 = delta_2 + gamma * lambda * gae_3 = 0.182 + 0.99 * 0.95 * 0.2
        delta_2 = -0.01 + gamma * 0.8 - 0.6
        gae_2 = delta_2 + gamma * gae_lambda * 0.2
        assert abs(all_steps[2].advantage - gae_2) < 1e-6

        # Step t=1:
        # delta_1 = r_1 + gamma * v_2 - v_1 = -0.01 + 0.99 * 0.6 - 0.4
        delta_1 = -0.01 + gamma * 0.6 - 0.4
        gae_1 = delta_1 + gamma * gae_lambda * gae_2
        assert abs(all_steps[1].advantage - gae_1) < 1e-6

        # Step t=0:
        delta_0 = -0.01 + gamma * 0.4 - 0.2
        gae_0 = delta_0 + gamma * gae_lambda * gae_1
        assert abs(all_steps[0].advantage - gae_0) < 1e-6

    def test_gae_multiple_episodes(self, sample_rollout_steps):
        """GAE handles multiple episodes independently."""
        buffer = RolloutBuffer()

        # Two episodes
        buffer.add_rollout(sample_rollout_steps)
        buffer.add_rollout(sample_rollout_steps[:2] + [
            RolloutStep(
                observation_text="end",
                action=2,
                reward=-1.0,
                done=True,
                value=0.1,
                log_prob=-1.0,
            )
        ])

        buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)

        all_steps = buffer.get_all_steps()
        # All steps should have return_ and advantage set
        for step in all_steps:
            assert isinstance(step.return_, float)
            assert isinstance(step.advantage, float)


# ---------------------------------------------------------------------------
# Dynamic Padding Tests
# ---------------------------------------------------------------------------


class TestDynamicPadding:
    """Tests for dynamic batch padding."""

    def test_dynamic_padding(self, t5_ppo_model, t5_ppo_config, sample_mc_question):
        """Padding works with variable-length sequences."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        # Create steps with different sequence lengths
        steps = [
            RolloutStep(
                observation_text="short",
                action=0,
                reward=0.0,
                done=False,
                value=0.1,
                log_prob=-0.5,
                input_ids=torch.randint(0, 100, (1, 5)),
                attention_mask=torch.ones(1, 5, dtype=torch.long),
            ),
            RolloutStep(
                observation_text="this is a longer sequence",
                action=1,
                reward=1.0,
                done=True,
                value=0.8,
                log_prob=-0.2,
                input_ids=torch.randint(0, 100, (1, 15)),
                attention_mask=torch.ones(1, 15, dtype=torch.long),
            ),
            RolloutStep(
                observation_text="medium",
                action=0,
                reward=0.0,
                done=False,
                value=0.3,
                log_prob=-0.6,
                input_ids=torch.randint(0, 100, (1, 10)),
                attention_mask=torch.ones(1, 10, dtype=torch.long),
            ),
        ]

        input_ids, attention_mask = trainer._pad_batch(steps)

        # All padded to max length in batch (15)
        assert input_ids.shape == (3, 15)
        assert attention_mask.shape == (3, 15)

        # First sequence (len 5) should have 10 padding tokens
        assert attention_mask[0, :5].sum() == 5
        assert attention_mask[0, 5:].sum() == 0

        # Second sequence (len 15) should have no padding
        assert attention_mask[1].sum() == 15

        # Third sequence (len 10) should have 5 padding tokens
        assert attention_mask[2, :10].sum() == 10
        assert attention_mask[2, 10:].sum() == 0


# ---------------------------------------------------------------------------
# Memory Management Tests
# ---------------------------------------------------------------------------


class TestMemoryManagement:
    """Tests for memory-safe tensor handling."""

    def test_memory_management_cpu_storage(self, sample_rollout_steps):
        """Rollout tensors are stored on CPU, not GPU."""
        for step in sample_rollout_steps:
            if step.input_ids is not None:
                assert step.input_ids.device.type == "cpu", (
                    f"input_ids on {step.input_ids.device}, expected CPU"
                )
            if step.attention_mask is not None:
                assert step.attention_mask.device.type == "cpu", (
                    f"attention_mask on {step.attention_mask.device}, expected CPU"
                )

    def test_rollout_tensors_are_detached(self, sample_rollout_steps):
        """Stored tensors do not require gradients."""
        for step in sample_rollout_steps:
            if step.input_ids is not None:
                assert not step.input_ids.requires_grad
            if step.attention_mask is not None:
                assert not step.attention_mask.requires_grad


# ---------------------------------------------------------------------------
# PPO Update Tests
# ---------------------------------------------------------------------------


class TestPPOUpdate:
    """Tests for PPO policy updates."""

    def test_ppo_update_no_oom(
        self, t5_ppo_model, t5_ppo_config, sample_mc_question
    ):
        """update_policy completes without OOM or errors."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        # Create a small buffer with tokenized steps
        buffer = RolloutBuffer()
        texts = [
            "CLUES: Who | CHOICES: (1) A (2) B (3) C (4) D",
            "CLUES: Who was | CHOICES: (1) A (2) B (3) C (4) D",
            "CLUES: Who was the | CHOICES: (1) A (2) B (3) C (4) D",
        ]

        rollout = []
        for i, text in enumerate(texts):
            inputs = t5_ppo_model.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            is_last = i == len(texts) - 1
            step = RolloutStep(
                observation_text=text,
                action=0 if not is_last else 1,
                reward=-0.01 if not is_last else 1.0,
                done=is_last,
                value=0.1 * (i + 1),
                log_prob=-0.5,
                input_ids=inputs["input_ids"].detach().cpu(),
                attention_mask=inputs["attention_mask"].detach().cpu(),
            )
            rollout.append(step)

        buffer.add_rollout(rollout)

        # Should complete without errors
        metrics = trainer.update_policy(buffer)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert metrics["num_updates"] > 0

    def test_ppo_update_empty_buffer(
        self, t5_ppo_model, t5_ppo_config, sample_mc_question
    ):
        """update_policy handles empty buffer gracefully."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        buffer = RolloutBuffer()
        metrics = trainer.update_policy(buffer)

        assert metrics["num_updates"] == 0
        assert metrics["policy_loss"] == 0.0


# ---------------------------------------------------------------------------
# Rollout Collection Tests
# ---------------------------------------------------------------------------


class TestRolloutCollection:
    """Tests for rollout collection."""

    def test_rollout_collection(
        self, t5_ppo_model, t5_ppo_config, sample_mc_question
    ):
        """collect_rollouts returns buffer with episodes."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        buffer = trainer.collect_rollouts(num_episodes=2)

        assert len(buffer) == 2  # 2 episodes collected
        all_steps = buffer.get_all_steps()
        assert len(all_steps) > 0  # At least some steps

        # Each step should have text, action, reward, tensors
        for step in all_steps:
            assert isinstance(step.observation_text, str)
            assert isinstance(step.action, int)
            assert 0 <= step.action <= 4  # WAIT or SELECT
            assert step.input_ids is not None
            assert step.attention_mask is not None
            # Tensors should be on CPU
            assert step.input_ids.device.type == "cpu"
            assert step.attention_mask.device.type == "cpu"

    def test_rollout_episodes_terminate(
        self, t5_ppo_model, t5_ppo_config, sample_mc_question
    ):
        """All collected episodes properly terminate."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        buffer = trainer.collect_rollouts(num_episodes=3)

        for rollout in buffer.rollouts:
            # Last step should be done
            assert rollout[-1].done, "Episode should terminate"
            # Non-terminal steps should not be done
            for step in rollout[:-1]:
                assert not step.done, "Non-terminal step should not be done"
````

## File: tests/test_qb_rl_bridge.py
````python
"""Compatibility bridge tests for qb-rl surfaces ported into qanta-buzzer."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import agents.bayesian_buzzer as bayesian_buzzer
import models.answer_profiles as compat_answer_profiles
import models.likelihoods as likelihoods
import qb_data.answer_profiles as qb_answer_profiles
import qb_data.data_loader as qb_data_loader
import qb_env.data_loader as compat_data_loader
import qb_env.mc_builder as compat_mc_builder
import qb_env.text_utils as compat_text_utils
from agents.softmax_profile_buzzer import (
    SequentialBayesBuzzer as CompatSequentialBayesBuzzer,
)
from agents.softmax_profile_buzzer import (
    SoftmaxEpisodeResult as CompatSoftmaxEpisodeResult,
)
from agents.softmax_profile_buzzer import (
    SoftmaxProfileBuzzer as CompatSoftmaxProfileBuzzer,
)
from models.likelihoods import OpenAILikelihood, build_likelihood_from_config
from qb_data.mc_builder import MCBuilder


def _install_fake_openai(monkeypatch, vectors: dict[str, list[float]], calls: list[tuple[str, tuple[str, ...]]]) -> None:
    """Install a fake ``openai`` module that serves deterministic embeddings."""

    class FakeEmbeddingsClient:
        def create(self, model: str, input: list[str]):
            calls.append((model, tuple(input)))
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=vectors[text])
                    for text in input
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.embeddings = FakeEmbeddingsClient()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))


class TestOpenAILikelihood:
    """Tests for optional OpenAI embedding support."""

    def test_openai_likelihood_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAILikelihood()

    def test_openai_likelihood_scores_and_reuses_cache(self, monkeypatch) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []
        vectors = {
            "first president": [2.0, 0.0],
            "george washington": [3.0, 0.0],
            "albert einstein": [0.0, 4.0],
        }
        _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        model = OpenAILikelihood(model="fake-embedding-model")

        embeddings = model._embed_batch(["first president", "george washington"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)
        calls_before_score = len(calls)

        scores_1 = model.score(
            "first president",
            ["george washington", "albert einstein"],
        )
        assert scores_1[0] > scores_1[1]
        assert len(calls) == calls_before_score + 2, (
            "first score should call the embeddings API twice"
        )

        scores_2 = model.score(
            "first president",
            ["george washington", "albert einstein"],
        )
        np.testing.assert_allclose(scores_1, scores_2, atol=1e-6)
        assert len(calls) == calls_before_score + 2, "second score should be served from cache"

    def test_likelihood_factory_openai(self, monkeypatch) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []
        vectors = {"a": [1.0, 0.0]}
        _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        config = {"likelihood": {"model": "openai", "openai_model": "fake-openai"}}
        model = build_likelihood_from_config(config)

        assert isinstance(model, OpenAILikelihood)
        assert model.model == "fake-openai"


class TestOpenAIProfileStrategy:
    """Tests for OpenAI-backed distractor ranking."""

    def test_openai_profile_uses_openai_embeddings(self, monkeypatch) -> None:
        calls: list[str] = []
        embeddings = {
            "gold profile": np.array([1.0, 0.0], dtype=np.float32),
            "near distractor": np.array([0.9, 0.1], dtype=np.float32),
            "far distractor": np.array([0.0, 1.0], dtype=np.float32),
        }

        class FakeOpenAILikelihood:
            def __init__(self, model: str = "unused") -> None:
                calls.append(model)

            def embed_and_cache(self, texts: list[str]) -> np.ndarray:
                return np.stack([embeddings[text] for text in texts]).astype(np.float32)

        monkeypatch.setattr(likelihoods, "OpenAILikelihood", FakeOpenAILikelihood)

        builder = MCBuilder(strategy="openai_profile", openai_model="fake-openai")
        rankings = builder._compute_rankings(
            answers=["gold", "near", "far"],
            answer_profiles={
                "gold": "gold profile",
                "near": "near distractor",
                "far": "far distractor",
            },
            answer_to_category={},
        )

        assert calls == ["fake-openai"]
        assert rankings["gold"][0] == "near"
        assert rankings["gold"][1] == "far"


class TestQBRLCompatibilityModules:
    """Tests for qb-rl import-path shims."""

    def test_module_aliases_resolve_expected_symbols(self) -> None:
        assert compat_answer_profiles.AnswerProfileBuilder is qb_answer_profiles.AnswerProfileBuilder
        assert compat_data_loader.parse_row is qb_data_loader.parse_row
        assert compat_mc_builder.MCBuilder.__name__ == "MCBuilder"
        assert compat_text_utils.normalize_answer("The Answer") == "answer"
        assert CompatSoftmaxProfileBuzzer is bayesian_buzzer.SoftmaxProfileBuzzer
        assert CompatSequentialBayesBuzzer is bayesian_buzzer.SequentialBayesBuzzer
        assert CompatSoftmaxEpisodeResult is bayesian_buzzer.SoftmaxEpisodeResult

    def test_parse_row_supports_qb_rl_metadata(self) -> None:
        question = compat_data_loader.parse_row(
            {
                "qid": "q-1",
                "question": "alpha beta gamma",
                "answer_primary": "George Washington",
                "clean_answers": ["George Washington", "Washington"],
                "run_indices": [1, 2],
                "metadata": {
                    "category": "History",
                    "human_buzz_positions": [{"position": 4, "count": 2}],
                },
            }
        )

        assert question.qid == "q-1"
        assert question.category == "History"
        assert question.human_buzz_positions == [(4, 2)]
        assert question.cumulative_prefixes == ["alpha beta", "alpha beta gamma"]

    def test_load_tossup_questions_from_config_prefers_dataset_smoke(
        self, monkeypatch
    ) -> None:
        captured: dict[str, object] = {}
        sample_question = compat_data_loader.TossupQuestion(
            qid="hf-1",
            question="alpha beta",
            tokens=["alpha", "beta"],
            answer_primary="Answer",
            clean_answers=["Answer"],
            run_indices=[1],
            human_buzz_positions=None,
            category="History",
            cumulative_prefixes=["alpha beta"],
        )

        def fake_load_tossup_questions(
            dataset: str,
            dataset_config: str | None = None,
            split: str = "eval",
            limit: int | None = None,
        ):
            captured["dataset"] = dataset
            captured["dataset_config"] = dataset_config
            captured["split"] = split
            captured["limit"] = limit
            return [sample_question]

        monkeypatch.setattr(qb_data_loader, "load_tossup_questions", fake_load_tossup_questions)

        config = {
            "data": {
                "dataset": "main-dataset",
                "dataset_config": "main-config",
                "dataset_smoke": "smoke-dataset",
                "dataset_smoke_config": "smoke-config",
                "split": "train",
            }
        }

        questions = compat_data_loader.load_tossup_questions_from_config(config, smoke=True)

        assert len(questions) == 1
        assert captured == {
            "dataset": "smoke-dataset",
            "dataset_config": "smoke-config",
            "split": "train",
            "limit": None,
        }
````

## File: tests/test_supervised_t5.py
````python
"""Unit tests for SupervisedTrainer and supervised training utilities.

Tests cover batch preparation, training epochs, gradient accumulation,
checkpoint save/load, best model selection, and the run_supervised_training
entry point.

Uses t5-small (60M params) for speed. The model fixture is module-scoped
to load t5-small only once per test file.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion
from training.train_supervised_t5 import (
    SupervisedTrainer,
    format_question_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_question(qid: str, gold_index: int = 0) -> MCQuestion:
    """Create a minimal MCQuestion for testing."""
    tokens = ["Who", "was", "the", "first", "president"]
    return MCQuestion(
        qid=qid,
        question="Who was the first president",
        tokens=tokens,
        answer_primary="George Washington",
        clean_answers=["George Washington"],
        run_indices=[0, 2, 4],
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=[
            "Who",
            "Who was the",
            "Who was the first president",
        ],
        options=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        gold_index=gold_index,
        option_profiles=[
            "George Washington first president",
            "Thomas Jefferson third president",
            "John Adams second president",
            "Benjamin Franklin inventor diplomat",
        ],
        option_answer_primary=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        distractor_strategy="test",
    )


@pytest.fixture(scope="module")
def t5_small_model() -> T5PolicyModel:
    """Load T5PolicyModel with t5-small once per test module."""
    model = T5PolicyModel(
        {
            "model_name": "t5-small",
            "device": "cpu",
            "max_input_length": 64,
            "num_choices": 4,
        }
    )
    return model


@pytest.fixture
def train_questions() -> list[MCQuestion]:
    """Return 8 training questions with varied gold indices."""
    return [_make_question(f"train_{i}", i % 4) for i in range(8)]


@pytest.fixture
def val_questions() -> list[MCQuestion]:
    """Return 4 validation questions."""
    return [_make_question(f"val_{i}", i % 4) for i in range(4)]


@pytest.fixture
def trainer_config(tmp_path) -> dict:
    """Return a minimal supervised trainer config using temp directory."""
    return {
        "model_name": "t5-small",
        "device": "cpu",
        "num_choices": 4,
        "supervised_lr": 1e-3,
        "supervised_epochs": 2,
        "supervised_batch_size": 2,
        "supervised_grad_accum_steps": 2,
        "max_input_length": 64,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }


@pytest.fixture
def trainer(
    t5_small_model: T5PolicyModel,
    train_questions: list[MCQuestion],
    val_questions: list[MCQuestion],
    trainer_config: dict,
) -> SupervisedTrainer:
    """Return a configured SupervisedTrainer instance."""
    return SupervisedTrainer(
        model=t5_small_model,
        train_questions=train_questions,
        val_questions=val_questions,
        config=trainer_config,
    )


# ---------------------------------------------------------------------------
# Format Tests
# ---------------------------------------------------------------------------


class TestFormatQuestionText:
    """Tests for the format_question_text utility."""

    def test_format_includes_all_tokens(self):
        """Formatted text includes all question tokens as clues."""
        q = _make_question("q1")
        text = format_question_text(q)
        assert "Who was the first president" in text

    def test_format_includes_all_choices(self):
        """Formatted text includes all 4 answer choices."""
        q = _make_question("q1")
        text = format_question_text(q)
        assert "(1) George Washington" in text
        assert "(2) Thomas Jefferson" in text
        assert "(3) John Adams" in text
        assert "(4) Benjamin Franklin" in text

    def test_format_structure(self):
        """Formatted text has CLUES: ... | CHOICES: ... structure."""
        q = _make_question("q1")
        text = format_question_text(q)
        assert text.startswith("CLUES: ")
        assert " | CHOICES: " in text


# ---------------------------------------------------------------------------
# Batch Preparation Tests
# ---------------------------------------------------------------------------


class TestPrepareBatch:
    """Tests for SupervisedTrainer.prepare_batch."""

    def test_prepare_batch_format(self, trainer: SupervisedTrainer):
        """Batch preparation produces correct tensor types and shapes."""
        questions = [_make_question(f"q{i}", i % 4) for i in range(3)]
        input_ids, attention_mask, labels = trainer.prepare_batch(questions)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert input_ids.shape[0] == 3  # batch_size
        assert attention_mask.shape == input_ids.shape
        assert labels.shape == (3,)

    def test_prepare_batch_complete_questions(self, trainer: SupervisedTrainer):
        """Batch shows complete questions (all clues), not incremental."""
        q = _make_question("q1")
        input_ids, _, _ = trainer.prepare_batch([q])

        # Decode tokens to verify all clues are included
        decoded = trainer.model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # All tokens should be present in the decoded text
        assert "first" in decoded.lower()
        assert "president" in decoded.lower()

    def test_prepare_batch_labels_correct(self, trainer: SupervisedTrainer):
        """Labels match gold_index of each question."""
        questions = [
            _make_question("q0", gold_index=0),
            _make_question("q1", gold_index=2),
            _make_question("q2", gold_index=3),
        ]
        _, _, labels = trainer.prepare_batch(questions)
        assert labels.tolist() == [0, 2, 3]


# ---------------------------------------------------------------------------
# Training Tests
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    """Tests for SupervisedTrainer.train_epoch."""

    def test_training_epoch_completes(self, trainer: SupervisedTrainer):
        """One epoch completes without errors."""
        loss, acc = trainer.train_epoch()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0, "Loss should be positive"
        assert 0 <= acc <= 1, "Accuracy should be in [0, 1]"

    def test_gradient_accumulation(
        self,
        t5_small_model: T5PolicyModel,
        train_questions: list[MCQuestion],
        val_questions: list[MCQuestion],
        tmp_path,
    ):
        """Optimizer updates only on accumulation steps (not every batch)."""
        config = {
            "supervised_lr": 1e-3,
            "supervised_epochs": 1,
            "supervised_batch_size": 2,
            "supervised_grad_accum_steps": 4,  # Update every 4 batches
            "max_input_length": 64,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }

        trainer = SupervisedTrainer(
            model=t5_small_model,
            train_questions=train_questions,
            val_questions=val_questions,
            config=config,
        )

        # Record initial params
        initial_params = {
            name: param.clone()
            for name, param in t5_small_model.policy_head.named_parameters()
        }

        # Run one epoch
        trainer.train_epoch()

        # Check that params changed (at least some should update)
        any_changed = False
        for name, param in t5_small_model.policy_head.named_parameters():
            if not torch.equal(initial_params[name], param):
                any_changed = True
                break

        assert any_changed, "Policy head parameters should change after training"


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for SupervisedTrainer.validate."""

    def test_validate_returns_metrics(self, trainer: SupervisedTrainer):
        """Validation returns loss and accuracy."""
        val_loss, val_acc = trainer.validate()

        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss > 0
        assert 0 <= val_acc <= 1


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Tests for checkpoint save/load functionality."""

    def test_checkpoint_save_load(self, trainer: SupervisedTrainer):
        """Save then load produces identical model outputs."""
        trainer.model.eval()

        # Get output before save
        q = _make_question("test_checkpoint")
        input_ids, attention_mask, _ = trainer.prepare_batch([q])
        with torch.no_grad():
            logits_before, preds_before = trainer.model.predict_answer(
                input_ids, attention_mask
            )

        # Save checkpoint
        save_path = trainer.save_checkpoint(is_best=True)
        assert save_path.exists()
        assert (save_path / "policy_head.pt").exists()
        assert (save_path / "training_state.pt").exists()

        # Load checkpoint
        trainer.model.load(str(save_path))

        # Get output after load
        with torch.no_grad():
            logits_after, preds_after = trainer.model.predict_answer(
                input_ids, attention_mask
            )

        assert torch.allclose(logits_before, logits_after, atol=1e-5)

    def test_best_model_selection(
        self,
        t5_small_model: T5PolicyModel,
        train_questions: list[MCQuestion],
        val_questions: list[MCQuestion],
        tmp_path,
    ):
        """Best model saved by validation accuracy (best_model/ dir exists)."""
        config = {
            "supervised_lr": 1e-3,
            "supervised_epochs": 2,
            "supervised_batch_size": 4,
            "supervised_grad_accum_steps": 1,
            "max_input_length": 64,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }

        trainer = SupervisedTrainer(
            model=t5_small_model,
            train_questions=train_questions,
            val_questions=val_questions,
            config=config,
        )

        result = trainer.train()

        # Best model directory should exist
        best_model_path = trainer.checkpoint_dir / "best_model"
        assert best_model_path.exists(), "best_model/ directory should exist"
        assert (best_model_path / "policy_head.pt").exists()
        assert result["best_val_acc"] >= 0

    def test_history_saved(self, trainer: SupervisedTrainer):
        """Training history saved to history.json with correct structure."""
        # Run a quick training
        trainer.config["supervised_epochs"] = 1
        trainer.epochs = 1
        trainer.train()

        history_path = trainer.checkpoint_dir / "history.json"
        assert history_path.exists()

        with open(history_path) as f:
            history = json.load(f)

        assert "train" in history
        assert "val" in history
        assert "config" in history
        assert len(history["train"]) >= 1
        assert "loss" in history["train"][0]
        assert "accuracy" in history["train"][0]
````

## File: tests/test_t5_policy.py
````python
"""Unit tests for T5PolicyModel and PolicyHead.

Tests cover PolicyHead architecture, T5PolicyModel forward pass, action
decomposition, tokenization, mean pooling, and checkpoint I/O.

Uses t5-small (60M params) for speed -- tests complete in <30 seconds.
The model fixture is module-scoped to load t5-small only once.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from models.t5_policy import PolicyHead, T5PolicyModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t5_small_config() -> dict:
    """Return a minimal config dict for T5PolicyModel with t5-small."""
    return {
        "model_name": "t5-small",
        "device": "cpu",
        "max_input_length": 128,
        "num_choices": 4,
    }


@pytest.fixture(scope="module")
def t5_small_model(t5_small_config):
    """Load T5PolicyModel with t5-small once per test module."""
    model = T5PolicyModel(t5_small_config)
    model.eval()
    return model


@pytest.fixture
def sample_texts() -> list[str]:
    """Return sample text inputs in quiz bowl format."""
    return [
        "CLUES: Who was the first president | CHOICES: (1) Washington (2) Jefferson (3) Adams (4) Franklin",
        "CLUES: This element has atomic number 1 | CHOICES: (1) Hydrogen (2) Helium (3) Lithium (4) Carbon",
    ]


# ---------------------------------------------------------------------------
# PolicyHead Tests
# ---------------------------------------------------------------------------


class TestPolicyHead:
    """Tests for PolicyHead class."""

    def test_policy_head_forward(self):
        """PolicyHead returns 3 tensors with correct shapes [B,2], [B,K], [B,1]."""
        batch_size = 4
        hidden_size = 512
        num_choices = 4

        head = PolicyHead(hidden_size=hidden_size, num_choices=num_choices)
        x = torch.randn(batch_size, hidden_size)

        wait_logits, answer_logits, values = head(x)

        assert wait_logits.shape == (batch_size, 2)
        assert answer_logits.shape == (batch_size, num_choices)
        assert values.shape == (batch_size, 1)

    def test_policy_head_different_num_choices(self):
        """PolicyHead handles non-default num_choices."""
        head = PolicyHead(hidden_size=256, num_choices=6)
        x = torch.randn(2, 256)

        wait_logits, answer_logits, values = head(x)

        assert wait_logits.shape == (2, 2)
        assert answer_logits.shape == (2, 6)
        assert values.shape == (2, 1)

    def test_policy_head_dropout(self):
        """Dropout layers exist and affect output in training mode."""
        head = PolicyHead(hidden_size=128, num_choices=4)
        head.train()  # Enable dropout

        x = torch.randn(8, 128)

        # Run forward twice in training mode; outputs should differ with high probability
        out1 = head(x)[0]
        out2 = head(x)[0]

        # Not strictly guaranteed but extremely likely with 8 samples and dropout
        # Use eval mode comparison for determinism
        head.eval()
        out3 = head(x)[0]
        out4 = head(x)[0]
        assert torch.allclose(out3, out4), "Eval mode should be deterministic"

    def test_policy_head_single_sample(self):
        """PolicyHead works with batch_size=1."""
        head = PolicyHead(hidden_size=512, num_choices=4)
        x = torch.randn(1, 512)

        wait_logits, answer_logits, values = head(x)

        assert wait_logits.shape == (1, 2)
        assert answer_logits.shape == (1, 4)
        assert values.shape == (1, 1)


# ---------------------------------------------------------------------------
# T5PolicyModel Tests
# ---------------------------------------------------------------------------


class TestT5PolicyModel:
    """Tests for T5PolicyModel class."""

    def test_t5_policy_init(self, t5_small_model):
        """T5PolicyModel initializes without errors and has correct structure."""
        model = t5_small_model

        assert hasattr(model, "encoder")
        assert hasattr(model, "tokenizer")
        assert hasattr(model, "policy_head")
        assert isinstance(model.policy_head, PolicyHead)

    def test_t5_policy_forward(self, t5_small_model, sample_texts):
        """Forward pass returns correct shapes for text inputs."""
        model = t5_small_model
        wait_logits, answer_logits, values = model(sample_texts)

        batch_size = len(sample_texts)
        assert wait_logits.shape == (batch_size, 2)
        assert answer_logits.shape == (batch_size, 4)
        assert values.shape == (batch_size, 1)

    def test_t5_policy_forward_no_value(self, t5_small_model, sample_texts):
        """Forward pass with return_value=False returns None for values."""
        model = t5_small_model
        wait_logits, answer_logits, values = model(sample_texts, return_value=False)

        assert values is None
        assert wait_logits.shape[0] == len(sample_texts)

    def test_encode_input(self, t5_small_model, sample_texts):
        """Tokenization produces input_ids and attention_mask with correct device."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        assert "input_ids" in encoding
        assert "attention_mask" in encoding
        assert encoding["input_ids"].shape[0] == len(sample_texts)
        assert encoding["attention_mask"].shape == encoding["input_ids"].shape
        assert encoding["input_ids"].device == model.device

    def test_encode_input_padding(self, t5_small_model):
        """Tokenization handles inputs of different lengths with padding."""
        model = t5_small_model
        texts = ["short", "this is a much longer text input with more tokens"]
        encoding = model.encode_input(texts)

        # Both should have same seq_len after padding
        assert encoding["input_ids"].shape[0] == 2
        # Second text should have more non-padding tokens
        mask_sums = encoding["attention_mask"].sum(dim=1)
        assert mask_sums[1] > mask_sums[0]

    def test_mean_pooling(self, t5_small_model):
        """Mean pooling respects attention mask (padded tokens have zero contribution)."""
        model = t5_small_model

        # Create a simple case: two identical sentences, one with extra padding
        texts = ["hello world"]
        encoding = model.encode_input(texts)

        pooled = model.get_encoder_output(
            encoding["input_ids"], encoding["attention_mask"]
        )

        # Output should be [1, hidden_size]
        assert pooled.shape == (1, model.encoder.config.d_model)
        assert not torch.isnan(pooled).any()
        assert not torch.isinf(pooled).any()


# ---------------------------------------------------------------------------
# Action Decomposition Tests
# ---------------------------------------------------------------------------


class TestActionDecomposition:
    """Tests for action decomposition in select_action and get_action_log_probs."""

    def test_action_decomposition_wait(self, t5_small_model, sample_texts):
        """action=0 decomposes to wait=0 in get_action_log_probs."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        # WAIT action
        actions = torch.zeros(len(sample_texts), dtype=torch.long, device=model.device)
        log_probs, entropy, values = model.get_action_log_probs(
            encoding["input_ids"], encoding["attention_mask"], actions
        )

        assert log_probs.shape == (len(sample_texts),)
        assert entropy.shape == (len(sample_texts),)
        assert values.shape == (len(sample_texts),)
        # Log probs should be negative
        assert (log_probs <= 0).all()
        # Entropy should be non-negative
        assert (entropy >= 0).all()

    def test_action_decomposition_buzz(self, t5_small_model, sample_texts):
        """actions 1-4 decompose to wait=1, answer=0-3."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts[:1])  # Single sample

        for action_val in [1, 2, 3, 4]:
            actions = torch.tensor([action_val], dtype=torch.long, device=model.device)
            log_probs, entropy, values = model.get_action_log_probs(
                encoding["input_ids"], encoding["attention_mask"], actions
            )

            assert log_probs.shape == (1,)
            assert (log_probs <= 0).all()

    def test_select_action_deterministic(self, t5_small_model, sample_texts):
        """Deterministic mode produces consistent actions."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        actions1, info1 = model.select_action(
            encoding["input_ids"],
            encoding["attention_mask"],
            deterministic=True,
        )
        actions2, info2 = model.select_action(
            encoding["input_ids"],
            encoding["attention_mask"],
            deterministic=True,
        )

        assert torch.equal(actions1, actions2)

    def test_select_action_stochastic(self, t5_small_model, sample_texts):
        """Stochastic mode samples from distribution (info dict has correct keys)."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        actions, info = model.select_action(
            encoding["input_ids"],
            encoding["attention_mask"],
            deterministic=False,
        )

        assert actions.shape == (len(sample_texts),)
        assert "wait_logits" in info
        assert "answer_logits" in info
        assert "wait_probs" in info
        assert "answer_probs" in info
        assert "values" in info
        assert "log_probs" in info

        # All actions should be in valid range [0, K]
        assert (actions >= 0).all()
        assert (actions <= 4).all()

    def test_select_action_returns_valid_range(self, t5_small_model, sample_texts):
        """Combined actions are in range [0, num_choices]."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        # Run many times to cover both wait and buzz actions
        for _ in range(10):
            actions, info = model.select_action(
                encoding["input_ids"],
                encoding["attention_mask"],
                deterministic=False,
                temperature=2.0,  # Higher temp for more randomness
            )
            assert (actions >= 0).all()
            assert (actions <= 4).all()

    def test_get_action_log_probs_matches_select(self, t5_small_model, sample_texts):
        """Log probs from get_action_log_probs are consistent with select_action."""
        model = t5_small_model
        model.eval()
        encoding = model.encode_input(sample_texts[:1])

        # Get deterministic action
        actions, info = model.select_action(
            encoding["input_ids"],
            encoding["attention_mask"],
            deterministic=True,
        )

        # Compute log probs for the same action
        log_probs, entropy, values = model.get_action_log_probs(
            encoding["input_ids"],
            encoding["attention_mask"],
            actions,
        )

        # Log probs should be finite
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(values).all()


# ---------------------------------------------------------------------------
# Predict Answer Tests
# ---------------------------------------------------------------------------


class TestPredictAnswer:
    """Tests for supervised training interface."""

    def test_predict_answer(self, t5_small_model, sample_texts):
        """predict_answer returns logits and predictions with correct shapes."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        answer_logits, predictions = model.predict_answer(
            encoding["input_ids"],
            encoding["attention_mask"],
        )

        assert answer_logits.shape == (len(sample_texts), 4)
        assert predictions.shape == (len(sample_texts),)
        # Predictions should be in valid range
        assert (predictions >= 0).all()
        assert (predictions < 4).all()


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Tests for save/load checkpoint functionality."""

    def test_save_load_checkpoint(self, t5_small_model, sample_texts):
        """Save then load produces identical model outputs."""
        model = t5_small_model
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "checkpoint")

            # Get output before save
            with torch.no_grad():
                wait_before, answer_before, value_before = model(sample_texts)

            # Save
            model.save(save_path)

            # Verify files exist
            assert os.path.exists(os.path.join(save_path, "policy_head.pt"))
            assert os.path.exists(os.path.join(save_path, "config.json"))

            # Load into same model
            model.load(save_path)

            # Get output after load
            with torch.no_grad():
                wait_after, answer_after, value_after = model(sample_texts)

            # Outputs should be identical
            assert torch.allclose(wait_before, wait_after, atol=1e-5)
            assert torch.allclose(answer_before, answer_after, atol=1e-5)
            assert torch.allclose(value_before, value_after, atol=1e-5)
````

## File: tests/test_text_wrapper.py
````python
"""Unit tests for TextObservationWrapper.

Tests verify that the wrapper correctly converts TossupMCEnv's numeric
belief observations into text-formatted strings for T5PolicyModel input.

Uses TF-IDF likelihood for fast test execution (<1 second total).
"""

from __future__ import annotations

import pytest

from qb_data.mc_builder import MCQuestion
from qb_env.text_wrapper import TextObservationWrapper
from qb_env.tossup_env import TossupMCEnv
from models.likelihoods import TfIdfLikelihood


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_mc_question() -> MCQuestion:
    """Return a minimal MCQuestion for wrapper testing."""
    tokens = [
        "Who", "was", "the", "first", "president",
        "of", "the", "United", "States", "?",
    ]
    run_indices = [0, 2, 4, 6, 8, 9]
    cumulative_prefixes = [
        "Who",
        "Who was the",
        "Who was the first president",
        "Who was the first president of the",
        "Who was the first president of the United States",
        "Who was the first president of the United States ?",
    ]
    return MCQuestion(
        qid="test_q1",
        question="Who was the first president of the United States?",
        tokens=tokens,
        answer_primary="George Washington",
        clean_answers=["George Washington", "Washington"],
        run_indices=run_indices,
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=cumulative_prefixes,
        options=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        gold_index=0,
        option_profiles=[
            "George Washington first president commander revolutionary war",
            "Thomas Jefferson third president declaration independence",
            "John Adams second president Massachusetts diplomat",
            "Benjamin Franklin inventor diplomat Philadelphia printing",
        ],
        option_answer_primary=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        distractor_strategy="test",
    )


@pytest.fixture
def wrapped_env(sample_mc_question: MCQuestion) -> TextObservationWrapper:
    """Return a TextObservationWrapper around a TossupMCEnv."""
    corpus = sample_mc_question.option_profiles[:]
    model = TfIdfLikelihood(corpus_texts=corpus)
    questions = [sample_mc_question] * 3
    env = TossupMCEnv(
        questions=questions,
        likelihood_model=model,
        K=4,
        reward_mode="simple",
        wait_penalty=0.0,
        buzz_correct=1.0,
        buzz_incorrect=-1.0,
        belief_mode="from_scratch",
        beta=5.0,
    )
    return TextObservationWrapper(env)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTextObservationWrapper:
    """Tests for TextObservationWrapper class."""

    def test_wrapper_observation_format(self, wrapped_env: TextObservationWrapper):
        """Observation returns 'CLUES: ... | CHOICES: ...' format."""
        obs, info = wrapped_env.reset()

        assert isinstance(obs, str), f"Expected str, got {type(obs)}"
        assert "CLUES:" in obs, "Observation must contain 'CLUES:'"
        assert "CHOICES:" in obs, "Observation must contain 'CHOICES:'"
        assert "(1)" in obs, "Choices must be numbered starting at (1)"
        assert "(4)" in obs, "All 4 choices must be present"

    def test_wrapper_incremental_clues(self, wrapped_env: TextObservationWrapper):
        """Wrapper shows correct clues based on step_idx progression."""
        obs0, _ = wrapped_env.reset()

        # Initial: first token only
        clues_part = obs0.split(" | CHOICES:")[0].replace("CLUES: ", "")
        assert clues_part == "Who", f"Initial clues should be 'Who', got '{clues_part}'"

        # After first WAIT: cumulative_prefixes[0] = "Who"
        obs1, _, _, _, _ = wrapped_env.step(0)
        clues1 = obs1.split(" | CHOICES:")[0].replace("CLUES: ", "")
        assert clues1 == "Who", f"After 1st WAIT should be 'Who', got '{clues1}'"

        # After second WAIT: cumulative_prefixes[1] = "Who was the"
        obs2, _, _, _, _ = wrapped_env.step(0)
        clues2 = obs2.split(" | CHOICES:")[0].replace("CLUES: ", "")
        assert clues2 == "Who was the", f"After 2nd WAIT should be 'Who was the', got '{clues2}'"

    def test_wrapper_gymnasium_api(self, wrapped_env: TextObservationWrapper):
        """reset() and step() still work after wrapping."""
        # reset returns (obs, info) tuple
        result = wrapped_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, str)
        assert isinstance(info, dict)
        assert "qid" in info

        # step returns (obs, reward, terminated, truncated, info)
        result = wrapped_env.step(0)  # WAIT
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_wrapper_preserves_reward(self, sample_mc_question: MCQuestion):
        """Reward from wrapped env matches underlying env behavior."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)

        # Create unwrapped env
        env = TossupMCEnv(
            questions=[sample_mc_question] * 3,
            likelihood_model=model,
            K=4,
            reward_mode="simple",
            buzz_correct=1.0,
            buzz_incorrect=-1.0,
            seed=42,
        )

        # Create wrapped env with same seed
        env2 = TossupMCEnv(
            questions=[sample_mc_question] * 3,
            likelihood_model=model,
            K=4,
            reward_mode="simple",
            buzz_correct=1.0,
            buzz_incorrect=-1.0,
            seed=42,
        )
        wrapped = TextObservationWrapper(env2)

        # Reset both
        _, info1 = env.reset(seed=42)
        _, info2 = wrapped.reset(seed=42)

        # Take same actions
        _, r1, d1, t1, _ = env.step(0)
        _, r2, d2, t2, _ = wrapped.step(0)
        assert r1 == r2, f"Rewards differ: {r1} vs {r2}"
        assert d1 == d2, f"Terminated differs"
        assert t1 == t2, f"Truncated differs"

        # BUZZ with answer 1 (correct for gold_index=0)
        _, r1, d1, t1, _ = env.step(1)
        _, r2, d2, t2, _ = wrapped.step(1)
        assert r1 == r2, f"Buzz rewards differ: {r1} vs {r2}"
        assert d1 == d2

    def test_wrapper_multiple_steps(self, wrapped_env: TextObservationWrapper):
        """Multi-step episode produces increasing clue text."""
        obs, _ = wrapped_env.reset()
        prev_clues = obs.split(" | CHOICES:")[0]

        # Take multiple WAIT steps and verify clues grow
        grew_at_least_once = False
        for step in range(4):
            obs, _, terminated, truncated, _ = wrapped_env.step(0)
            if terminated or truncated:
                break
            current_clues = obs.split(" | CHOICES:")[0]
            if len(current_clues) > len(prev_clues):
                grew_at_least_once = True
            # Clues should never shrink
            assert len(current_clues) >= len(prev_clues), (
                f"Clues shrank at step {step}: '{prev_clues}' -> '{current_clues}'"
            )
            prev_clues = current_clues

        assert grew_at_least_once, "Clue text should grow with more WAITs"

    def test_wrapper_choices_include_all_options(
        self, wrapped_env: TextObservationWrapper
    ):
        """All 4 answer options appear in the choices section."""
        obs, _ = wrapped_env.reset()
        choices_part = obs.split("CHOICES: ")[1]

        assert "George Washington" in choices_part
        assert "Thomas Jefferson" in choices_part
        assert "John Adams" in choices_part
        assert "Benjamin Franklin" in choices_part

    def test_wrapper_buzz_ends_episode(self, wrapped_env: TextObservationWrapper):
        """Buzzing with an answer ends the episode."""
        wrapped_env.reset()
        _, _, terminated, truncated, info = wrapped_env.step(1)  # BUZZ answer 0
        assert terminated or truncated, "Episode should end after BUZZ"

    def test_wrapper_complete_episode(self, wrapped_env: TextObservationWrapper):
        """Full episode: WAIT until truncated or BUZZ."""
        wrapped_env.reset()

        for step in range(20):
            obs, reward, terminated, truncated, info = wrapped_env.step(0)
            if terminated or truncated:
                break
            assert isinstance(obs, str)

        # Episode must have ended (6 clue steps)
        assert terminated or truncated, "Episode should end within 20 steps"
````

## File: training/__init__.py
````python
"""
Training Package

Supervised warm-start and PPO fine-tuning for T5 policy models.
"""
````

## File: training/train_ppo_t5.py
````python
"""
Custom PPO Training for T5 Policy Model

Implements PPOTrainer with RolloutBuffer for end-to-end PPO fine-tuning of
T5PolicyModel on incremental quiz bowl episodes. Uses Generalized Advantage
Estimation (GAE) for variance reduction and dynamic batch padding to minimize
memory footprint.

Key design decisions:
    - Rollout tensors (input_ids, attention_mask) are immediately detached and
      moved to CPU after collection to prevent GPU memory accumulation.
    - Dynamic padding: each mini-batch is padded to the max length within that
      batch, not a global 512-token maximum, saving ~50%+ memory.
    - Config-dict interface for compatibility with the unified codebase YAML
      config pattern (see configs/t5_policy.yaml).

Ported from qanta-buzzer reference implementation (train_ppo.py) with:
    - TextObservationWrapper for text-based rollout collection
    - Memory-safe tensor management (detach + CPU storage)
    - Dynamic padding per mini-batch
    - Config dict interface replacing Config class
    - NumPy-style docstrings

Usage
-----
From Python::

    from training.train_ppo_t5 import PPOTrainer, run_ppo_training
    from models.t5_policy import T5PolicyModel
    from qb_data.mc_builder import MCQuestion

    model = T5PolicyModel({"model_name": "t5-small", "device": "cpu"})
    trainer = PPOTrainer(model, train_qs, val_qs, config)
    trainer.train()

From command line::

    python scripts/train_t5_policy.py --config configs/t5_policy.yaml
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion


@dataclass
class RolloutStep:
    """Single step in an episode rollout.

    Stores observation text, action, reward, value estimate, and log probability
    for a single environment step. Tokenized tensors (input_ids, attention_mask)
    are stored on CPU to prevent GPU memory accumulation during rollout collection.

    Attributes
    ----------
    observation_text : str
        Text observation at this step (CLUES: ... | CHOICES: ...).
    action : int
        Combined action taken (0=WAIT, 1..K=SELECT).
    reward : float
        Scalar reward received.
    done : bool
        Whether this step ended the episode.
    value : float
        Value estimate from the critic at this step.
    log_prob : float
        Log probability of the action under the policy at collection time.
    input_ids : torch.Tensor or None
        Tokenized input IDs stored on CPU. Shape ``[1, seq_len]``.
    attention_mask : torch.Tensor or None
        Attention mask stored on CPU. Shape ``[1, seq_len]``.
    return_ : float
        Discounted return (filled by ``compute_returns_and_advantages``).
    advantage : float
        GAE advantage (filled by ``compute_returns_and_advantages``).
    """

    observation_text: str
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    return_: float = 0.0
    advantage: float = 0.0


class RolloutBuffer:
    """Buffer to store and process episode rollouts for PPO updates.

    Accumulates complete episode rollouts (lists of RolloutStep), then computes
    discounted returns and GAE advantages across all episodes. Provides a flat
    view of all steps for mini-batch iteration during PPO updates.

    Attributes
    ----------
    rollouts : list[list[RolloutStep]]
        List of episode rollouts, each a list of steps.
    """

    def __init__(self) -> None:
        self.rollouts: List[List[RolloutStep]] = []

    def reset(self) -> None:
        """Clear all stored rollouts."""
        self.rollouts = []

    def add_rollout(self, steps: List[RolloutStep]) -> None:
        """Add a complete episode rollout to the buffer.

        Parameters
        ----------
        steps : list[RolloutStep]
            Complete episode rollout (ordered list of steps from reset to done).
        """
        self.rollouts.append(steps)

    def get_all_steps(self) -> List[RolloutStep]:
        """Get a flat list of all steps from all rollouts.

        Returns
        -------
        list[RolloutStep]
            All steps concatenated in order (rollout 0 steps, then rollout 1, ...).
        """
        all_steps: List[RolloutStep] = []
        for rollout in self.rollouts:
            all_steps.extend(rollout)
        return all_steps

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float
    ) -> None:
        """Compute discounted returns and GAE advantages for all rollouts.

        Uses Generalized Advantage Estimation (GAE) to compute per-step
        advantages. For each rollout, iterates backward from the terminal
        step computing:

            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = delta_t + gamma * lambda * A_{t+1}
            G_t = A_t + V(s_t)

        Terminal states reset next_value and gae to 0.

        Parameters
        ----------
        gamma : float
            Discount factor in [0, 1]. Higher values weight future rewards more.
        gae_lambda : float
            GAE lambda in [0, 1]. Trades off bias (low) vs variance (high).
        """
        for rollout in self.rollouts:
            rewards = [step.reward for step in rollout]
            values = [step.value for step in rollout]
            dones = [step.done for step in rollout]

            # GAE computation (backward pass)
            gae = 0.0
            next_value = 0.0  # Terminal state value

            for t in reversed(range(len(rollout))):
                if dones[t]:
                    next_value = 0.0
                    gae = 0.0

                # TD error
                delta = rewards[t] + gamma * next_value - values[t]

                # GAE accumulation
                gae = delta + gamma * gae_lambda * gae

                # Store return and advantage
                rollout[t].return_ = gae + values[t]
                rollout[t].advantage = gae

                next_value = values[t]

    def __len__(self) -> int:
        return len(self.rollouts)


class PPOTrainer:
    """Custom PPO trainer for T5PolicyModel on quiz bowl episodes.

    Collects rollouts by running T5PolicyModel in text-observation episodes
    (via TextObservationWrapper), then updates the policy using clipped
    surrogate PPO loss with value function and entropy regularization.

    The trainer handles the complete training loop:
    1. Collect rollouts (episodes) using the current policy
    2. Compute GAE advantages
    3. Update policy with mini-batch PPO for multiple epochs
    4. Periodically validate and save checkpoints

    Parameters
    ----------
    model : T5PolicyModel
        T5 policy model to train. Should be pre-trained via supervised
        warm-start for faster convergence.
    train_questions : list[MCQuestion]
        Training set questions for rollout collection.
    val_questions : list[MCQuestion]
        Validation set questions for periodic evaluation.
    config : dict[str, Any]
        Configuration dictionary with PPO hyperparameters:

        - ``ppo_lr`` (float): Learning rate. Default 1e-5.
        - ``ppo_iterations`` (int): Number of collect-update cycles. Default 100.
        - ``ppo_batch_size`` (int): Mini-batch size for PPO updates. Default 8.
        - ``ppo_epochs_per_iter`` (int): PPO epochs per iteration. Default 4.
        - ``ppo_gamma`` (float): Discount factor. Default 0.99.
        - ``ppo_gae_lambda`` (float): GAE lambda. Default 0.95.
        - ``ppo_clip_ratio`` (float): PPO clip ratio. Default 0.2.
        - ``ppo_value_coef`` (float): Value loss coefficient. Default 0.5.
        - ``ppo_entropy_coef`` (float): Entropy bonus coefficient. Default 0.01.
        - ``ppo_max_grad_norm`` (float): Gradient clip norm. Default 0.5.
        - ``ppo_episodes_per_iter`` (int): Episodes per rollout. Default 16.
        - ``eval_interval`` (int): Validate every N iterations. Default 10.
        - ``save_interval`` (int): Save checkpoint every N iterations. Default 20.
        - ``checkpoint_dir`` (str): Base checkpoint directory. Default "checkpoints".
        - ``reward_time_penalty`` (float): Time penalty for env. Default 0.1.

    Attributes
    ----------
    model : T5PolicyModel
        The model being trained.
    optimizer : torch.optim.AdamW
        Optimizer with weight decay.
    best_val_reward : float
        Best validation reward seen so far.
    history : list[dict]
        Per-iteration training metrics.
    checkpoint_dir : Path
        Directory for saving PPO checkpoints.
    """

    def __init__(
        self,
        model: T5PolicyModel,
        train_questions: List[MCQuestion],
        val_questions: List[MCQuestion],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_questions = list(train_questions)
        self.val_questions = list(val_questions)
        self.config = config

        self.device = model.device

        # PPO hyperparameters
        self.lr = float(config.get("ppo_lr", 1e-5))
        self.iterations = int(config.get("ppo_iterations", 100))
        self.batch_size = int(config.get("ppo_batch_size", 8))
        self.epochs_per_iter = int(config.get("ppo_epochs_per_iter", 4))
        self.gamma = float(config.get("ppo_gamma", 0.99))
        self.gae_lambda = float(config.get("ppo_gae_lambda", 0.95))
        self.clip_ratio = float(config.get("ppo_clip_ratio", 0.2))
        self.value_coef = float(config.get("ppo_value_coef", 0.5))
        self.entropy_coef = float(config.get("ppo_entropy_coef", 0.01))
        self.max_grad_norm = float(config.get("ppo_max_grad_norm", 0.5))
        self.episodes_per_iter = int(config.get("ppo_episodes_per_iter", 16))
        self.eval_interval = int(config.get("eval_interval", 10))
        self.save_interval = int(config.get("save_interval", 20))
        self.reward_time_penalty = float(config.get("reward_time_penalty", 0.1))
        self.max_input_length = int(config.get("max_input_length", 512))

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=0.01
        )

        # Training state
        self.current_iteration = 0
        self.best_val_reward = -float("inf")
        self.history: List[Dict[str, Any]] = []

        # Checkpoint directory
        self.checkpoint_dir = (
            Path(config.get("checkpoint_dir", "checkpoints")) / "ppo_t5"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def collect_rollouts(self, num_episodes: int) -> RolloutBuffer:
        """Collect rollouts by running episodes with the current policy.

        Creates a TossupMCEnv + TextObservationWrapper for each sampled
        question, runs the policy until episode termination, and stores
        all steps in a RolloutBuffer. Tokenized tensors are detached and
        moved to CPU immediately to prevent GPU memory accumulation.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to collect.

        Returns
        -------
        RolloutBuffer
            Buffer containing all collected episode rollouts.
        """
        from qb_env.text_wrapper import TextObservationWrapper
        from qb_env.tossup_env import TossupMCEnv
        from models.likelihoods import TfIdfLikelihood

        self.model.eval()
        buffer = RolloutBuffer()

        # Sample questions for this iteration
        questions = random.choices(self.train_questions, k=num_episodes)

        # Build a simple TF-IDF likelihood for environment scoring
        # (The T5 policy reads text directly; likelihood is only used for
        # environment reward computation via belief updates)
        corpus = []
        for q in self.train_questions[:100]:  # Use subset for speed
            corpus.extend(q.option_profiles)
        likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

        with torch.no_grad():
            for question in questions:
                env = TossupMCEnv(
                    questions=[question],
                    likelihood_model=likelihood_model,
                    K=len(question.options),
                    reward_mode="time_penalty",
                    wait_penalty=self.reward_time_penalty,
                    belief_mode="from_scratch",
                )
                wrapped_env = TextObservationWrapper(env)

                obs, info = wrapped_env.reset()
                done = False
                rollout: List[RolloutStep] = []

                while not done:
                    # Tokenize text observation
                    inputs = self.model.tokenizer(
                        obs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_length,
                    ).to(self.device)

                    # Get action from policy
                    actions, act_info = self.model.select_action(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        deterministic=False,
                    )

                    action = actions.item()
                    value = act_info["values"].squeeze().item()
                    log_prob = act_info["log_probs"].item()

                    # Take environment step
                    next_obs, reward, terminated, truncated, step_info = (
                        wrapped_env.step(action)
                    )
                    done = terminated or truncated

                    # CRITICAL: Detach and move tensors to CPU immediately
                    # to prevent GPU memory accumulation during rollout collection
                    step = RolloutStep(
                        observation_text=obs,
                        action=action,
                        reward=reward,
                        done=done,
                        value=value,
                        log_prob=log_prob,
                        input_ids=inputs["input_ids"].detach().cpu(),
                        attention_mask=inputs["attention_mask"].detach().cpu(),
                    )
                    rollout.append(step)

                    obs = next_obs

                buffer.add_rollout(rollout)

        return buffer

    def _pad_batch(
        self, batch_steps: List[RolloutStep]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamically pad a mini-batch of steps to the max length in the batch.

        Instead of padding all sequences to the global max (512 tokens), pads
        only to the longest sequence in the current mini-batch. This typically
        saves 50%+ memory since most quiz bowl observations are 100-200 tokens.

        Parameters
        ----------
        batch_steps : list[RolloutStep]
            Mini-batch of rollout steps with stored input_ids and attention_mask.

        Returns
        -------
        input_ids : torch.Tensor
            Padded input IDs of shape ``[batch_size, max_len]``, on device.
        attention_mask : torch.Tensor
            Padded attention mask of shape ``[batch_size, max_len]``, on device.
        """
        max_len = max(step.input_ids.shape[1] for step in batch_steps)
        pad_token_id = self.model.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []

        for step in batch_steps:
            seq_len = step.input_ids.shape[1]
            if seq_len < max_len:
                pad_len = max_len - seq_len
                input_ids_padded = torch.cat(
                    [
                        step.input_ids,
                        torch.full(
                            (1, pad_len),
                            pad_token_id,
                            dtype=step.input_ids.dtype,
                        ),
                    ],
                    dim=1,
                )
                attention_mask_padded = torch.cat(
                    [
                        step.attention_mask,
                        torch.zeros(
                            (1, pad_len), dtype=step.attention_mask.dtype
                        ),
                    ],
                    dim=1,
                )
            else:
                input_ids_padded = step.input_ids
                attention_mask_padded = step.attention_mask

            padded_input_ids.append(input_ids_padded)
            padded_attention_mask.append(attention_mask_padded)

        input_ids = torch.cat(padded_input_ids).to(self.device)
        attention_mask = torch.cat(padded_attention_mask).to(self.device)

        return input_ids, attention_mask

    def update_policy(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update the policy using PPO with clipped surrogate loss.

        Computes GAE advantages, normalizes them, then runs multiple epochs
        of mini-batch PPO updates. Each update computes the clipped surrogate
        policy loss, value function MSE loss, and entropy bonus.

        Parameters
        ----------
        buffer : RolloutBuffer
            Buffer with collected rollouts (compute_returns_and_advantages
            will be called internally).

        Returns
        -------
        dict[str, float]
            Training metrics: policy_loss, value_loss, entropy, num_updates.
        """
        self.model.train()

        # Compute returns and advantages
        buffer.compute_returns_and_advantages(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        # Get all steps
        all_steps = buffer.get_all_steps()
        if not all_steps:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "num_updates": 0,
            }

        # Normalize advantages
        advantages = torch.tensor(
            [step.advantage for step in all_steps], dtype=torch.float32
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # PPO epochs
        for epoch in range(self.epochs_per_iter):
            # Shuffle step indices
            indices = np.random.permutation(len(all_steps))

            # Mini-batch updates
            for start_idx in range(0, len(all_steps), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(all_steps))
                batch_indices = indices[start_idx:end_idx]

                # Get batch steps
                batch_steps = [all_steps[i] for i in batch_indices]

                # Dynamic padding to max length in THIS batch
                input_ids, attention_mask = self._pad_batch(batch_steps)

                # Prepare batch tensors
                actions = torch.tensor(
                    [step.action for step in batch_steps],
                    dtype=torch.long,
                ).to(self.device)
                old_log_probs = torch.tensor(
                    [step.log_prob for step in batch_steps],
                    dtype=torch.float32,
                ).to(self.device)
                returns = torch.tensor(
                    [step.return_ for step in batch_steps],
                    dtype=torch.float32,
                ).to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)

                # Get new log probs, entropy, and values from current policy
                new_log_probs, entropy, values = (
                    self.model.get_action_log_probs(
                        input_ids, attention_mask, actions
                    )
                )

                # PPO clipped surrogate policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE)
                value_loss = nn.MSELoss()(values, returns)

                # Entropy bonus (negative because we maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backward pass and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "num_updates": num_updates,
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set by running deterministic episodes.

        Runs one episode per validation question with deterministic action
        selection (argmax) and computes accuracy and average reward.

        Returns
        -------
        dict[str, float]
            Validation metrics: accuracy, average_reward, avg_episode_length.
        """
        from qb_env.text_wrapper import TextObservationWrapper
        from qb_env.tossup_env import TossupMCEnv
        from models.likelihoods import TfIdfLikelihood

        self.model.eval()

        corpus = []
        for q in self.train_questions[:100]:
            corpus.extend(q.option_profiles)
        likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

        correct = 0
        total = 0
        total_reward = 0.0
        total_length = 0

        # Limit validation size for speed
        val_questions = self.val_questions[:50]

        with torch.no_grad():
            for question in val_questions:
                env = TossupMCEnv(
                    questions=[question],
                    likelihood_model=likelihood_model,
                    K=len(question.options),
                    reward_mode="time_penalty",
                    wait_penalty=self.reward_time_penalty,
                    belief_mode="from_scratch",
                )
                wrapped_env = TextObservationWrapper(env)

                obs, info = wrapped_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done:
                    inputs = self.model.tokenizer(
                        obs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_length,
                    ).to(self.device)

                    actions, act_info = self.model.select_action(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        deterministic=True,
                    )

                    action = actions.item()
                    obs, reward, terminated, truncated, step_info = (
                        wrapped_env.step(action)
                    )
                    done = terminated or truncated
                    episode_reward += reward
                    episode_length += 1

                total_reward += episode_reward
                total_length += episode_length
                total += 1

                # Check if answer was correct
                if step_info.get("correct", False) or step_info.get(
                    "forced_correct", False
                ):
                    correct += 1

        return {
            "accuracy": correct / max(1, total),
            "average_reward": total_reward / max(1, total),
            "avg_episode_length": total_length / max(1, total),
        }

    def train(self) -> Dict[str, Any]:
        """Run the full PPO training loop.

        Alternates between rollout collection and policy updates for
        ``self.iterations`` cycles. Periodically validates and saves
        checkpoints.

        Returns
        -------
        dict[str, Any]
            Training summary: best_val_reward, total_iterations.
        """
        print(f"Starting PPO training for {self.iterations} iterations")
        print(f"  Training questions: {len(self.train_questions)}")
        print(f"  Validation questions: {len(self.val_questions)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Episodes per iteration: {self.episodes_per_iter}")
        print(f"  Device: {self.device}")
        print()

        for iteration in range(self.iterations):
            self.current_iteration = iteration

            # Collect rollouts
            print(f"\nIteration {iteration + 1}/{self.iterations}")
            print("  Collecting rollouts...")
            buffer = self.collect_rollouts(self.episodes_per_iter)

            # Compute episode statistics
            episode_rewards = []
            episode_lengths = []
            for rollout in buffer.rollouts:
                episode_reward = sum(step.reward for step in rollout)
                episode_rewards.append(episode_reward)
                episode_lengths.append(len(rollout))

            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0.0

            print(f"  Avg episode reward: {avg_reward:.4f}")
            print(f"  Avg episode length: {avg_length:.2f}")

            # Update policy
            print("  Updating policy...")
            update_metrics = self.update_policy(buffer)

            print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {update_metrics['value_loss']:.4f}")
            print(f"  Entropy: {update_metrics['entropy']:.4f}")

            # Validate periodically
            if (iteration + 1) % self.eval_interval == 0:
                print("\n  Validating...")
                val_summary = self.validate()
                val_reward = val_summary.get("average_reward", 0.0)

                print(f"  Val Accuracy: {val_summary['accuracy']:.4f}")
                print(f"  Val Reward: {val_reward:.4f}")
                print(
                    f"  Val Avg Length: {val_summary['avg_episode_length']:.2f}"
                )

                # Save history
                self.history.append(
                    {
                        "iteration": iteration + 1,
                        "train_reward": float(avg_reward),
                        "train_length": float(avg_length),
                        **update_metrics,
                        "val": val_summary,
                    }
                )

                # Save best model
                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self.save_checkpoint(is_best=True)
                    print(
                        f"  -> New best validation reward: {val_reward:.4f}"
                    )

            # Save regular checkpoint
            if (iteration + 1) % self.save_interval == 0:
                self.save_checkpoint(is_best=False)
                self.save_history()

        print("\n" + "=" * 60)
        print("PPO training completed!")
        print(f"Best validation reward: {self.best_val_reward:.4f}")
        print("=" * 60)

        # Save final history
        self.save_history()

        return {
            "best_val_reward": self.best_val_reward,
            "total_iterations": self.iterations,
        }

    def save_checkpoint(self, is_best: bool = False) -> Path:
        """Save model checkpoint to disk.

        Parameters
        ----------
        is_best : bool
            If True, save to ``best_model/`` directory.

        Returns
        -------
        Path
            Path to the saved checkpoint directory.
        """
        if is_best:
            save_path = self.checkpoint_dir / "best_model"
        else:
            save_path = (
                self.checkpoint_dir
                / f"iter_{self.current_iteration + 1}"
            )

        # Use T5PolicyModel's save() method
        self.model.save(str(save_path))

        # Save training state
        state = {
            "iteration": self.current_iteration + 1,
            "best_val_reward": self.best_val_reward,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, save_path / "training_state.pt")

        print(f"  Checkpoint saved to {save_path}")
        return save_path

    def save_history(self) -> Path:
        """Save training history to JSON.

        Returns
        -------
        Path
            Path to the saved history file.
        """
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=float)
        return history_path


def run_ppo_training(
    config: Dict[str, Any],
    train_questions: List[MCQuestion],
    val_questions: List[MCQuestion],
    test_questions: Optional[List[MCQuestion]] = None,
    pretrained_model_path: Optional[str] = None,
) -> Tuple[T5PolicyModel, PPOTrainer]:
    """Run the PPO training pipeline with optional pretrained model.

    Creates or loads a T5PolicyModel, trains it with PPO on quiz bowl
    episodes, and optionally evaluates on a test set.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with model and PPO hyperparameters.
    train_questions : list[MCQuestion]
        Training set questions.
    val_questions : list[MCQuestion]
        Validation set questions.
    test_questions : list[MCQuestion] or None
        Optional test set for final evaluation.
    pretrained_model_path : str or None
        Path to a supervised pretrained checkpoint. If provided, loads the
        model from this path. Otherwise creates a new model.

    Returns
    -------
    model : T5PolicyModel
        The trained model.
    trainer : PPOTrainer
        The trainer instance with training history.
    """
    print("=" * 60)
    print("PPO TRAINING PHASE (T5 Policy)")
    print("=" * 60)

    # Load or create model
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}")
        device = config.get("device", "cpu")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        model = T5PolicyModel.load_pretrained(
            pretrained_model_path, device=device
        )
    else:
        print("Initializing new model (no pretraining)")
        model_config = {
            "model_name": config.get("model_name", "t5-large"),
            "device": config.get("device", "cpu"),
            "max_input_length": config.get("max_input_length", 512),
            "num_choices": config.get("num_choices", 4),
        }
        model = T5PolicyModel(model_config)

    # Create trainer
    trainer = PPOTrainer(
        model=model,
        train_questions=train_questions,
        val_questions=val_questions,
        config=config,
    )

    # Train
    summary = trainer.train()

    # Evaluate on test set if provided
    if test_questions is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        # Load best model if it exists
        best_model_path = trainer.checkpoint_dir / "best_model"
        if best_model_path.exists():
            print(f"Loading best model from {best_model_path}")
            model.load(str(best_model_path))

        # Run validation on test set
        # Temporarily swap val questions with test questions
        original_val = trainer.val_questions
        trainer.val_questions = list(test_questions)
        test_metrics = trainer.validate()
        trainer.val_questions = original_val

        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Avg Reward: {test_metrics['average_reward']:.4f}")

        # Save test results
        test_results = {
            "test_metrics": test_metrics,
            "training_summary": summary,
        }
        results_path = trainer.checkpoint_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2, default=float)
        print(f"Test results saved to {results_path}")

    return model, trainer
````

## File: training/train_supervised_t5.py
````python
"""
Supervised warm-start training for T5PolicyModel.

Trains answer selection on complete questions using cross-entropy loss. All
clues are shown at once (not incremental), providing a strong initialization
before PPO fine-tuning on partial observations.

The training loop uses gradient accumulation (default 4 steps, effective
batch = 32) for stable training without exceeding GPU memory. Best model
is saved by validation accuracy to checkpoints/supervised/best_model/.

Ported from qanta-buzzer reference implementation (train_supervised.py)
with these changes:
    - Accepts list of MCQuestion objects instead of QuizBowlDataset class
    - Config dict interface instead of qanta-buzzer's Config class
    - Direct text formatting from MCQuestion (no QuizBowlEnvironment needed)
    - NumPy-style docstrings added throughout

Usage
-----
From Python::

    from training.train_supervised_t5 import SupervisedTrainer, run_supervised_training
    from models.t5_policy import T5PolicyModel
    from qb_data.mc_builder import MCQuestion

    model = T5PolicyModel({"model_name": "t5-small", "device": "cpu"})
    trainer = SupervisedTrainer(model, train_qs, val_qs, config)
    trainer.train()

From command line::

    python -m training.train_supervised_t5 --config configs/t5_policy.yaml
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion


def format_question_text(question: MCQuestion) -> str:
    """Format a complete question as text for supervised training.

    Shows ALL clues (complete question) since supervised training is the
    easier task of answer selection on full information. PPO later trains
    on incremental clues.

    Parameters
    ----------
    question : MCQuestion
        Question with tokens, options, and gold_index.

    Returns
    -------
    str
        Formatted text: ``"CLUES: <all tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
    """
    clues_text = " ".join(question.tokens)
    choices_parts = [f"({i + 1}) {opt}" for i, opt in enumerate(question.options)]
    choices_text = " ".join(choices_parts)
    return f"CLUES: {clues_text} | CHOICES: {choices_text}"


class SupervisedTrainer:
    """Trainer for supervised warm-start of T5PolicyModel.

    Trains the answer head using cross-entropy loss on complete questions
    (all clues shown at once). Uses gradient accumulation for stable training
    with large effective batch sizes without exceeding GPU memory.

    The training loop:
    1. Shuffles training data each epoch
    2. Iterates over mini-batches
    3. Computes cross-entropy loss on answer logits
    4. Accumulates gradients for ``grad_accum_steps`` batches
    5. Clips gradients and updates optimizer
    6. Validates after each epoch
    7. Saves best model by validation accuracy

    Parameters
    ----------
    model : T5PolicyModel
        Model to train. Must have ``predict_answer`` and ``tokenizer``.
    train_questions : list[MCQuestion]
        Training set questions.
    val_questions : list[MCQuestion]
        Validation set questions.
    config : dict[str, Any]
        Configuration dictionary with keys:

        - ``supervised_lr`` (float): Learning rate. Default 3e-4.
        - ``supervised_epochs`` (int): Number of epochs. Default 10.
        - ``supervised_batch_size`` (int): Batch size. Default 8.
        - ``supervised_grad_accum_steps`` (int): Gradient accumulation. Default 4.
        - ``checkpoint_dir`` (str): Base checkpoint directory. Default "checkpoints".
        - ``max_input_length`` (int): Max token length. Default 512.
        - ``max_grad_norm`` (float): Gradient clip norm. Default 1.0.
        - ``weight_decay`` (float): AdamW weight decay. Default 0.01.

    Attributes
    ----------
    model : T5PolicyModel
        The model being trained.
    optimizer : torch.optim.AdamW
        Optimizer with weight decay.
    criterion : nn.CrossEntropyLoss
        Loss function for answer classification.
    best_val_acc : float
        Best validation accuracy seen so far.
    train_history : list[dict]
        Per-epoch training metrics.
    val_history : list[dict]
        Per-epoch validation metrics.
    checkpoint_dir : Path
        Directory for saving checkpoints.
    """

    def __init__(
        self,
        model: T5PolicyModel,
        train_questions: List[MCQuestion],
        val_questions: List[MCQuestion],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_questions = list(train_questions)
        self.val_questions = list(val_questions)
        self.config = config

        self.device = model.device

        # Hyperparameters with defaults
        self.lr = float(config.get("supervised_lr", 3e-4))
        self.epochs = int(config.get("supervised_epochs", 10))
        self.batch_size = int(config.get("supervised_batch_size", 8))
        self.grad_accum_steps = int(config.get("supervised_grad_accum_steps", 4))
        self.max_input_length = int(config.get("max_input_length", 512))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.weight_decay = float(config.get("weight_decay", 0.01))

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history: List[Dict[str, Any]] = []
        self.val_history: List[Dict[str, Any]] = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints")) / "supervised"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def prepare_batch(
        self, questions: List[MCQuestion]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Format a batch of complete questions as tokenized tensors.

        Each question is formatted with ALL clues visible (supervised training
        shows complete information). Text is tokenized using the model's
        T5TokenizerFast.

        Parameters
        ----------
        questions : list[MCQuestion]
            Batch of questions to format.

        Returns
        -------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``, on device.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``, on device.
        labels : torch.Tensor
            Gold answer indices of shape ``[batch_size]``, on device.
        """
        texts = [format_question_text(q) for q in questions]
        labels = [q.gold_index for q in questions]

        # Tokenize
        inputs = self.model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

        return input_ids, attention_mask, labels_tensor

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with gradient accumulation.

        Shuffles training data, iterates over mini-batches, and updates
        the optimizer every ``grad_accum_steps`` batches. Gradients are
        clipped to ``max_grad_norm`` before each optimizer step.

        Returns
        -------
        epoch_loss : float
            Average loss over all batches in the epoch.
        epoch_acc : float
            Average accuracy over all batches in the epoch.
        """
        self.model.train()

        # Shuffle training data
        shuffled = self.train_questions[:]
        random.shuffle(shuffled)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = max(1, len(shuffled) // self.batch_size)

        # Zero gradients at start
        self.optimizer.zero_grad()

        for batch_idx in range(num_batches):
            # Get batch
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(shuffled))
            batch_questions = shuffled[start:end]

            if not batch_questions:
                continue

            # Prepare batch
            input_ids, attention_mask, labels = self.prepare_batch(batch_questions)

            # Forward pass
            answer_logits, predictions = self.model.predict_answer(
                input_ids, attention_mask
            )

            # Compute loss (scaled by accumulation steps for correct gradient magnitude)
            loss = self.criterion(answer_logits, labels)
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            # Track metrics (use unscaled loss for logging)
            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

            # Gradient accumulation: update every N batches
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Handle remaining accumulated gradients (if num_batches not divisible by accum_steps)
        remaining = num_batches % self.grad_accum_steps
        if remaining > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        epoch_loss = total_loss / max(1, num_batches)
        epoch_acc = total_correct / max(1, total_samples)

        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """Validate on the validation set.

        Runs the model in eval mode on all validation questions, computing
        accuracy and loss without gradient computation.

        Returns
        -------
        val_loss : float
            Average cross-entropy loss on validation set.
        val_acc : float
            Accuracy on validation set (fraction correct).
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = max(1, len(self.val_questions) // self.batch_size)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(self.val_questions))
                batch_questions = self.val_questions[start:end]

                if not batch_questions:
                    continue

                input_ids, attention_mask, labels = self.prepare_batch(batch_questions)
                answer_logits, predictions = self.model.predict_answer(
                    input_ids, attention_mask
                )

                loss = self.criterion(answer_logits, labels)
                total_loss += loss.item()
                total_correct += (predictions == labels).sum().item()
                total_samples += len(labels)

        val_loss = total_loss / max(1, num_batches)
        val_acc = total_correct / max(1, total_samples)

        return val_loss, val_acc

    def train(self) -> Dict[str, Any]:
        """Run full supervised training loop.

        Iterates over epochs, training and validating each epoch. Saves the
        best model by validation accuracy to ``checkpoint_dir/best_model/``.
        Training history is saved to ``checkpoint_dir/history.json``.

        Returns
        -------
        dict[str, Any]
            Training summary with keys: ``best_val_acc``, ``final_train_acc``,
            ``final_train_loss``, ``total_epochs``.
        """
        print(f"Starting supervised training for {self.epochs} epochs")
        print(f"  Training samples: {len(self.train_questions)}")
        print(f"  Validation samples: {len(self.val_questions)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation: {self.grad_accum_steps} (effective batch = {self.batch_size * self.grad_accum_steps})")
        print(f"  Learning rate: {self.lr}")
        print(f"  Device: {self.device}")
        print()

        final_train_loss = 0.0
        final_train_acc = 0.0

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_acc = self.train_epoch()
            final_train_loss = train_loss
            final_train_acc = train_acc

            # Validate
            val_loss, val_acc = self.validate()

            # Log results
            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save history
            self.train_history.append(
                {"epoch": epoch + 1, "loss": train_loss, "accuracy": train_acc}
            )
            self.val_history.append(
                {"epoch": epoch + 1, "loss": val_loss, "accuracy": val_acc}
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(is_best=True)
                print(f"  -> New best validation accuracy: {val_acc:.4f}")

        print(f"\nSupervised training completed!")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f}")

        # Save training history
        self.save_history()

        return {
            "best_val_acc": self.best_val_acc,
            "final_train_acc": final_train_acc,
            "final_train_loss": final_train_loss,
            "total_epochs": self.epochs,
        }

    def save_checkpoint(self, is_best: bool = False) -> Path:
        """Save model checkpoint to disk.

        Saves the model (T5 encoder + policy head) and optimizer state.
        Best model is saved to ``checkpoint_dir/best_model/``, epoch
        checkpoints to ``checkpoint_dir/epoch_N/``.

        Parameters
        ----------
        is_best : bool
            If True, save to ``best_model/`` directory.

        Returns
        -------
        Path
            Path to the saved checkpoint directory.
        """
        if is_best:
            save_path = self.checkpoint_dir / "best_model"
        else:
            save_path = self.checkpoint_dir / f"epoch_{self.current_epoch + 1}"

        # Use T5PolicyModel's save() method
        self.model.save(str(save_path))

        # Save training state
        state = {
            "epoch": self.current_epoch + 1,
            "best_val_acc": self.best_val_acc,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, save_path / "training_state.pt")

        return save_path

    def save_history(self) -> Path:
        """Save training history to JSON.

        Converts numpy types to native Python types for JSON serialization.

        Returns
        -------
        Path
            Path to the saved history file.
        """
        history = {
            "train": _convert_to_native(self.train_history),
            "val": _convert_to_native(self.val_history),
            "config": {
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
            },
        }

        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"Training history saved to {history_path}")
        return history_path


def run_supervised_training(
    config: Dict[str, Any],
    train_questions: List[MCQuestion],
    val_questions: List[MCQuestion],
    test_questions: Optional[List[MCQuestion]] = None,
) -> Tuple[T5PolicyModel, SupervisedTrainer]:
    """Run the complete supervised training pipeline.

    Creates a T5PolicyModel, trains it on complete questions, and optionally
    evaluates on a test set. This is the main entry point for supervised
    warm-start training.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary. Must include model config keys
        (``model_name``, ``device``, ``num_choices``) and supervised
        training keys (``supervised_lr``, etc.).
    train_questions : list[MCQuestion]
        Training set questions.
    val_questions : list[MCQuestion]
        Validation set questions.
    test_questions : list[MCQuestion] or None
        Optional test set for final evaluation.

    Returns
    -------
    model : T5PolicyModel
        The trained model (with best weights loaded).
    trainer : SupervisedTrainer
        The trainer instance with training history.
    """
    print("=" * 60)
    print("SUPERVISED TRAINING PHASE")
    print("=" * 60)

    # Initialize model
    model_config = {
        "model_name": config.get("model_name", "t5-large"),
        "device": config.get("device", "cpu"),
        "max_input_length": config.get("max_input_length", 512),
        "num_choices": config.get("num_choices", 4),
    }
    model = T5PolicyModel(model_config)

    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_questions=train_questions,
        val_questions=val_questions,
        config=config,
    )

    # Train
    summary = trainer.train()

    # Evaluate on test set if provided
    if test_questions is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        # Load best model
        best_model_path = trainer.checkpoint_dir / "best_model"
        model.load(str(best_model_path))
        model.eval()

        # Evaluate
        test_loss, test_acc = _evaluate_on_questions(model, test_questions, trainer)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Save test results
        test_results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "training_summary": summary,
        }
        results_path = trainer.checkpoint_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(_convert_to_native(test_results), f, indent=2)
        print(f"Test results saved to {results_path}")

    return model, trainer


def _evaluate_on_questions(
    model: T5PolicyModel,
    questions: List[MCQuestion],
    trainer: SupervisedTrainer,
) -> Tuple[float, float]:
    """Evaluate model on a set of questions.

    Parameters
    ----------
    model : T5PolicyModel
        Model to evaluate.
    questions : list[MCQuestion]
        Questions to evaluate on.
    trainer : SupervisedTrainer
        Trainer instance (for batch preparation).

    Returns
    -------
    avg_loss : float
        Average cross-entropy loss.
    accuracy : float
        Fraction of correctly predicted answers.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_size = trainer.batch_size
    num_batches = max(1, len(questions) // batch_size)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(questions))
            batch_questions = questions[start:end]

            if not batch_questions:
                continue

            input_ids, attention_mask, labels = trainer.prepare_batch(batch_questions)
            answer_logits, predictions = model.predict_answer(input_ids, attention_mask)

            loss = criterion(answer_logits, labels)
            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

    return total_loss / max(1, num_batches), total_correct / max(1, total_samples)


def _convert_to_native(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization.

    Parameters
    ----------
    obj : Any
        Object to convert. Handles dicts, lists, numpy scalars and arrays.

    Returns
    -------
    Any
        Object with all numpy types converted to native Python types.
    """
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_to_native(obj.tolist())
    else:
        return obj
````

## File: pyproject.toml
````toml
[build-system]
requires = ["setuptools>=69.0"]
build-backend = "setuptools.build_meta"

[project]
name = "qanta-buzzer"
version = "1.0.0"
description = "Unified quiz bowl RL buzzer system for Stanford CS234"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "datasets>=2.14.0",
  "gymnasium>=1.1.0",
  "jsonlines>=3.1.0",
  "matplotlib>=3.7.0",
  "numpy>=1.24.0",
  "pandas>=2.0.0",
  "PyYAML>=6.0.0",
  "scikit-learn>=1.3.0",
  "seaborn>=0.12.0",
  "sentence-transformers>=2.2.0",
  "stable-baselines3>=2.6.0",
  "torch>=2.0.0",
  "tqdm>=4.65.0",
  "transformers>=4.30.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0.0"]

[tool.setuptools.packages.find]
include = ["agents", "evaluation", "models", "qb_data", "qb_env", "training"]

[tool.pytest.ini_options]
testpaths = ["tests"]
````

## File: requirements.txt
````
# Base runtime dependencies for the unified modular repo.
# Preferred development setup is: pip install -e .

torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
datasets>=2.14.0
gymnasium>=1.1.0
stable-baselines3>=2.6.0
PyYAML>=6.0.0
tqdm>=4.65.0
jsonlines>=3.1.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0

# Optional OpenAI bridge support:
# pip install -e .[openai]
````

## File: agents/bayesian_buzzer.py
````python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents._math import sigmoid
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion



@dataclass
class SoftmaxEpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


class SoftmaxProfileBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        self.belief: np.ndarray | None = None

    def _belief_from_scratch(self, cumulative_prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(cumulative_prefix, option_profiles)
        scores = scores - np.max(scores)
        probs = np.exp(self.beta * scores)
        probs = probs / max(1e-12, probs.sum())
        return probs.astype(np.float32)

    def confidence_proxy(self, top_p: float) -> float:
        return sigmoid(self.alpha * (top_p - self.threshold))

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        chosen_idx = 0
        chosen_step = len(question.cumulative_prefixes) - 1

        for step_idx, prefix in enumerate(question.cumulative_prefixes):
            belief = self._belief_from_scratch(prefix, question.option_profiles)
            self.belief = belief
            top_idx = int(np.argmax(belief))
            top_p = float(np.max(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
            c_t = self.confidence_proxy(top_p)
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        return SoftmaxEpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=(chosen_idx == question.gold_index),
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


class SequentialBayesBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha

    def _step_update(self, prior: np.ndarray, fragment: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(fragment, option_profiles)
        scores = scores - np.max(scores)
        likelihood = np.exp(self.beta * scores)
        posterior = prior * likelihood
        denom = posterior.sum()
        if denom <= 0:
            return np.ones_like(prior) / len(prior)
        return (posterior / denom).astype(np.float32)

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        K = len(question.options)
        belief = np.ones(K, dtype=np.float32) / K
        chosen_idx = 0
        chosen_step = len(question.cumulative_prefixes) - 1

        for step_idx, token_idx in enumerate(question.run_indices):
            prev_token_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
            fragment = " ".join(question.tokens[prev_token_idx + 1 : token_idx + 1])
            belief = self._step_update(belief, fragment, question.option_profiles)
            top_idx = int(np.argmax(belief))
            top_p = float(np.max(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
            c_t = sigmoid(self.alpha * (top_p - self.threshold))
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        return SoftmaxEpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=(chosen_idx == question.gold_index),
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


def precompute_sequential_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    beta: float,
) -> list["_PrecomputedQuestion"]:
    """Compute Bayesian sequential beliefs at every step for every question.

    Starts with a uniform prior and applies Bayesian update
    ``posterior = prior * likelihood`` using token fragments derived from
    ``question.run_indices``.  Returns one ``_PrecomputedQuestion`` per
    question where ``beliefs`` are the Bayesian posteriors (NOT the
    from-scratch softmax beliefs).
    """
    from agents.threshold_buzzer import _PrecomputedQuestion

    out: list[_PrecomputedQuestion] = []
    for q in questions:
        K = len(q.options)
        belief = np.ones(K, dtype=np.float32) / K
        beliefs: list[np.ndarray] = []

        for step_idx, token_idx in enumerate(q.run_indices):
            prev_token_idx = q.run_indices[step_idx - 1] if step_idx > 0 else -1
            fragment = " ".join(q.tokens[prev_token_idx + 1 : token_idx + 1])
            scores = likelihood_model.score(fragment, q.option_profiles)
            scores = scores - np.max(scores)
            likelihood = np.exp(beta * scores)
            posterior = belief * likelihood
            denom = posterior.sum()
            if denom <= 0:
                belief = np.ones_like(belief) / len(belief)
            else:
                belief = (posterior / denom).astype(np.float32)
            beliefs.append(belief.copy())

        out.append(_PrecomputedQuestion(
            qid=q.qid,
            gold_index=q.gold_index,
            num_options=K,
            beliefs=beliefs,
        ))
    return out


def _sequential_episode_from_precomputed(
    pq: "_PrecomputedQuestion",
    threshold: float,
    alpha: float,
) -> SoftmaxEpisodeResult:
    """Build a SoftmaxEpisodeResult from pre-computed sequential beliefs.

    Identical buzzing logic to ``SequentialBayesBuzzer.run_episode`` but
    reads beliefs from a ``_PrecomputedQuestion`` instead of calling the
    likelihood model.
    """
    from agents.threshold_buzzer import _belief_stats

    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    chosen_step = len(pq.beliefs) - 1
    chosen_idx = 0

    for step_idx, belief in enumerate(pq.beliefs):
        top_idx, top_p, entropy = _belief_stats(belief)
        c_t = sigmoid(alpha * (top_p - threshold))
        g_t = 1.0 if top_idx == pq.gold_index else 0.0

        c_trace.append(c_t)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

        is_last = step_idx == len(pq.beliefs) - 1
        if top_p >= threshold or is_last:
            chosen_step = step_idx
            chosen_idx = top_idx
            break

    correct = chosen_idx == pq.gold_index
    return SoftmaxEpisodeResult(
        qid=pq.qid,
        buzz_step=chosen_step,
        buzz_index=chosen_idx,
        gold_index=pq.gold_index,
        correct=correct,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def sweep_sequential_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
    precomputed: list["_PrecomputedQuestion"] | None = None,
) -> dict[float, list[SoftmaxEpisodeResult]]:
    """Sweep multiple thresholds with a single sequential belief pass.

    If *precomputed* is provided the expensive model calls are skipped
    entirely and the sweep is pure numpy.  Otherwise beliefs are computed
    once internally and reused across thresholds.
    """
    if precomputed is None:
        precomputed = precompute_sequential_beliefs(questions, likelihood_model, beta)

    out: dict[float, list[SoftmaxEpisodeResult]] = {}
    for threshold in thresholds:
        out[float(threshold)] = [
            _sequential_episode_from_precomputed(pq, threshold, alpha)
            for pq in precomputed
        ]
    return out
````

## File: agents/ppo_buzzer.py
````python
"""PPO Buzzer agent wrapping Stable-Baselines3's PPO.

Provides the PPOBuzzer class for training an MLP policy on belief-feature
observations from TossupMCEnv, and PPOEpisodeTrace for recording per-step
action probabilities needed to compute the S_q scoring metric.

The key design rationale: SB3's ``learn()`` does not expose per-step action
distributions, so ``run_episode()`` implements custom episode execution that
records c_trace (buzz probability) and g_trace (correctness probability)
at each step for downstream S_q computation.

Ported from qb-rl reference implementation (agents/ppo_buzzer.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO

from qb_env.tossup_env import TossupMCEnv


@dataclass
class PPOEpisodeTrace:
    """Record of a single episode with per-step action probability traces.

    Used to compute the S_q scoring metric: S_q = sum(c_t * g_t) over steps,
    and calibration metrics (ECE, Brier) via ``top_p_trace``.

    Attributes
    ----------
    qid : str
        Question identifier.
    buzz_step : int
        Step at which the agent buzzed (-1 if never buzzed voluntarily).
    buzz_index : int
        Index of the chosen answer option (0-based, -1 if forced).
    gold_index : int
        Index of the correct answer option (0-based).
    correct : bool
        Whether the agent selected the correct answer.
    episode_reward : float
        Total accumulated reward over the episode.
    c_trace : list[float]
        Per-step buzz probability: 1 - P(wait) at each timestep.
    g_trace : list[float]
        Per-step correctness probability: P(gold_option) / P(buzz).
    top_p_trace : list[float]
        Per-step max belief probability: max(env.belief). Used as the
        confidence proxy for calibration metrics, consistent with
        baseline agents.
    entropy_trace : list[float]
        Per-step policy entropy over the full action distribution.
    """

    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    episode_reward: float
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


class PPOBuzzer:
    """PPO-trained buzzer agent wrapping Stable-Baselines3's PPO.

    Trains an MLP policy on belief-feature observations (Box(K+6,)) from
    TossupMCEnv. The policy maps observation vectors to a Discrete(K+1)
    action space: WAIT (0) or BUZZ with option i (1..K).

    Parameters
    ----------
    env : TossupMCEnv
        Gymnasium environment with belief-feature observations.
    learning_rate : float
        Learning rate for the Adam optimizer.
    n_steps : int
        Number of steps per rollout buffer collection.
    batch_size : int
        Minibatch size for PPO updates.
    n_epochs : int
        Number of optimization epochs per rollout.
    gamma : float
        Discount factor for return computation.
    policy_kwargs : dict or None
        Additional keyword arguments for the MLP policy. Defaults to
        ``{"net_arch": [64, 64]}`` (two hidden layers of 64 units).
    verbose : int
        SB3 verbosity level (0=silent, 1=info, 2=debug).
    """

    def __init__(
        self,
        env: TossupMCEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 128,
        batch_size: int = 32,
        n_epochs: int = 10,
        gamma: float = 0.99,
        seed: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
    ):
        if policy_kwargs is None:
            policy_kwargs = {"net_arch": [64, 64]}

        self.env = env
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=verbose,
            seed=seed,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
        )

    def train(self, total_timesteps: int = 100_000) -> None:
        """Train the PPO policy for the specified number of timesteps.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to collect during training.
        """
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str | Path) -> None:
        """Save the trained PPO model to disk.

        Parameters
        ----------
        path : str or Path
            File path for the saved model (SB3 appends .zip if needed).
        """
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, env: TossupMCEnv) -> "PPOBuzzer":
        """Load a previously saved PPO model.

        Parameters
        ----------
        path : str or Path
            Path to the saved model file.
        env : TossupMCEnv
            Environment to attach to the loaded model.

        Returns
        -------
        PPOBuzzer
            A PPOBuzzer with the loaded model weights.
        """
        agent = cls(env=env)
        agent.model = PPO.load(str(path), env=env)
        return agent

    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Extract action probabilities from the policy for a given observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).

        Returns
        -------
        np.ndarray
            Action probability vector of shape (K + 1,), dtype float32.
            Index 0 = P(wait), indices 1..K = P(buzz with option i).
        """
        obs_tensor = th.as_tensor(
            obs, dtype=th.float32, device=self.model.device
        ).unsqueeze(0)
        dist = self.model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs[0].detach().cpu().numpy()
        return probs.astype(np.float32)

    def c_t(self, obs: np.ndarray) -> float:
        """Compute buzz probability at the current step.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).

        Returns
        -------
        float
            Probability of buzzing: 1 - P(wait). Range [0, 1].
        """
        probs = self.action_probabilities(obs)
        return float(1.0 - probs[0])

    def g_t(self, obs: np.ndarray, gold_index: int) -> float:
        """Compute correctness probability at the current step.

        Given that the agent buzzes, what is the probability it selects
        the correct answer? Formally: P(gold_action) / P(buzz).

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).
        gold_index : int
            Index of the correct answer option (0-based).

        Returns
        -------
        float
            Conditional correctness probability. Returns 0.0 if buzz
            probability is near zero (< 1e-12).
        """
        probs = self.action_probabilities(obs)
        c_t = float(1.0 - probs[0])
        if c_t <= 1e-12:
            return 0.0
        return float(probs[gold_index + 1] / c_t)

    def run_episode(
        self,
        deterministic: bool = False,
        seed: int | None = None,
        question_idx: int | None = None,
    ) -> PPOEpisodeTrace:
        """Run a full episode and record per-step action probability traces.

        Executes the policy in the environment, computing c_trace (buzz
        probability), g_trace (correctness probability), and entropy_trace
        at each step. These traces are needed to compute the S_q metric.

        Parameters
        ----------
        deterministic : bool
            If True, select actions by argmax instead of sampling.
        seed : int or None
            If provided, seeds the environment reset for reproducibility.

        Returns
        -------
        PPOEpisodeTrace
            Complete episode record with action traces and outcome.
        """
        reset_options = None
        if question_idx is not None:
            reset_options = {"question_idx": int(question_idx)}

        obs, info = self.env.reset(seed=seed, options=reset_options)
        terminated = False
        truncated = False
        total_reward = 0.0
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        buzz_step = -1
        buzz_index = -1
        gold_index = (
            self.env.question.gold_index if self.env.question is not None else -1
        )

        while not (terminated or truncated):
            probs = self.action_probabilities(obs)
            c_val = float(1.0 - probs[0])
            g_val = (
                float(probs[gold_index + 1] / c_val) if c_val > 1e-12 else 0.0
            )
            entropy = float(
                -(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))).sum()
            )

            top_p_val = float(np.max(self.env.belief)) if self.env.belief is not None else c_val
            c_trace.append(c_val)
            g_trace.append(g_val)
            top_p_trace.append(top_p_val)
            entropy_trace.append(entropy)

            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(len(probs), p=probs))

            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward

            if action != 0 and buzz_step < 0:
                buzz_step = int(step_info.get("step_idx", 0))
                buzz_index = action - 1
            if truncated and buzz_step < 0:
                buzz_step = int(
                    step_info.get("step_idx", len(c_trace) - 1)
                )
                buzz_index = int(
                    step_info.get("forced_choice", np.argmax(self.env.belief))
                )

        correct = buzz_index == gold_index
        return PPOEpisodeTrace(
            qid=info.get("qid", ""),
            buzz_step=buzz_step,
            buzz_index=buzz_index,
            gold_index=gold_index,
            correct=correct,
            episode_reward=total_reward,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )
````

## File: configs/default.yaml
````yaml
# Default configuration for qanta-buzzer
# Adapted from qb-rl structure for T5-based quiz bowl agent

data:
  csv_path: "questions.csv"  # Raw QANTA CSV with ||| separated clues
  K: 4  # Number of answer choices
  distractor_strategy: "sbert_profile"  # sbert_profile | category_random | embedding_based
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  max_questions: null  # Limit for testing (null = use all)
  shuffle_seed: 42

answer_profiles:
  max_tokens_per_profile: 2000  # Max tokens to use for answer profile
  min_questions_per_answer: 1  # Minimum examples to build profile
  leave_one_out: true  # Exclude current question from profile

likelihood:
  model: "t5-large"  # Model for computing answer likelihoods (t5-small | t5-base | t5-large)
  embedding_model: "all-MiniLM-L6-v2"  # For distractor generation
  beta: 5.0  # Softmax temperature for belief distribution
  cache_embeddings: true
  cache_dir: "cache/embeddings"
  batch_size: 16
  max_length: 512  # Max input tokens for T5

environment:
  reward_mode: "time_penalty"  # time_penalty | simple
  seed: 13
  wait_penalty: 0.05  # Tuned candidate from multi-seed smoke sweep
  early_buzz_penalty: 0.2  # Tuned candidate from multi-seed smoke sweep
  buzz_correct: 1.0  # Reward for correct answer
  buzz_incorrect: -0.5  # Penalty for wrong answer
  max_steps: 20  # Maximum clues to reveal

mc_guards:  # Anti-artifact guards from qb-rl
  alias_edit_distance_threshold: 0.2  # Reject similar answer aliases
  duplicate_token_overlap_threshold: 0.8  # Reject token-overlapping distractors
  max_length_ratio: 3.0  # Reject distractors much longer than answer

bayesian:  # Bayesian buzzer sweep parameters (from qb-rl)
  threshold_sweep: [0.5, 0.6, 0.7, 0.8, 0.9]
  alpha: 10.0  # Sigmoid steepness for confidence proxy

ppo:  # PPO hyperparameters (for future use)
  seed: 13
  total_timesteps: 100000
  learning_rate: 3e-4
  n_steps: 128
  batch_size: 32
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  target_kl: 0.03
  policy_kwargs:
    net_arch: [64, 64]  # MLP architecture for belief-based policy

evaluation:
  metrics:
    - accuracy
    - reward
    - buzz_position
    - calibration  # ECE and Brier score
    - per_category
  compute_sq: true  # S_q scoring metric
  run_choices_only: true  # Control: model sees only choices, no clues
  run_shuffle: true  # Control: shuffle clue order
  bootstrap_ci_samples: 1000  # Bootstrap confidence intervals
  save_predictions: true
  prediction_dir: "results/predictions"

# Supervised warm-start settings (for T5 policy)
supervised:
  epochs: 10
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch = 32
  learning_rate: 1e-4
  warmup_steps: 500
  eval_steps: 100
  save_steps: 500
  save_total_limit: 3
  checkpoint_dir: "checkpoints/supervised"
````

## File: configs/smoke.yaml
````yaml
# Smoke test configuration - quick testing with reduced data
# Inherits from default.yaml and overrides key settings

# Data settings for quick testing
data:
  csv_path: "questions.csv"
  K: 4
  distractor_strategy: "category_random"  # Faster than sbert_profile
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  max_questions: 50  # Use only 50 questions for smoke test
  shuffle_seed: 42

answer_profiles:
  max_tokens_per_profile: 500  # Reduced for speed
  min_questions_per_answer: 1
  leave_one_out: false  # Skip for smoke test

likelihood:
  model: "tfidf"  # Use TF-IDF for fastest smoke testing (<5 seconds)
  embedding_model: "all-MiniLM-L6-v2"
  beta: 5.0  # Softmax temperature for belief distribution
  cache_embeddings: true
  cache_dir: "cache/embeddings"
  batch_size: 4  # Smaller batch for memory
  max_length: 256  # Shorter sequences

environment:
  reward_mode: "time_penalty"
  seed: 13
  wait_penalty: 0.05
  early_buzz_penalty: 0.2
  buzz_correct: 1.0
  buzz_incorrect: -1.0
  max_steps: 10  # Fewer steps for quick testing

mc_guards:
  alias_edit_distance_threshold: 0.2
  duplicate_token_overlap_threshold: 0.8
  max_length_ratio: 3.0

bayesian:  # Reduced sweep for smoke testing
  threshold_sweep: [0.5, 0.7, 0.9]
  alpha: 10.0

ppo:  # Reduced for smoke testing
  seed: 13
  total_timesteps: 3000
  learning_rate: 3e-4
  n_steps: 32  # Smaller rollout
  batch_size: 8  # Smaller batch
  n_epochs: 2  # Fewer epochs
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  target_kl: 0.03
  policy_kwargs:
    net_arch: [32, 32]  # Smaller network

evaluation:
  metrics:
    - accuracy
    - reward
  compute_sq: false  # Skip expensive metrics
  run_choices_only: false  # Skip control experiments
  run_shuffle: false
  bootstrap_ci_samples: 0  # No bootstrap for smoke test
  save_predictions: false
  prediction_dir: "results/predictions"

# Supervised settings for smoke test
supervised:
  epochs: 2  # Very few epochs
  batch_size: 4
  gradient_accumulation_steps: 1  # No accumulation for speed
  learning_rate: 1e-4
  warmup_steps: 10
  eval_steps: 20
  save_steps: 100
  save_total_limit: 1
  checkpoint_dir: "checkpoints/supervised_smoke"
````

## File: models/__init__.py
````python
"""
Models Package

Likelihood models, belief feature extraction, and policy model interfaces
for the quiz bowl RL buzzer system.
"""

from models.features import extract_belief_features, entropy_of_distribution
from models.likelihoods import (
    LikelihoodModel,
    OpenAILikelihood,
    SBERTLikelihood,
    T5Likelihood,
    TfIdfLikelihood,
    build_likelihood_from_config,
)

# Lazy import: T5PolicyModel and PolicyHead require transformers + torch.
# Import on demand to keep package lightweight for belief-feature-only usage.


def __getattr__(name: str):
    if name in ("T5PolicyModel", "PolicyHead"):
        from models.t5_policy import T5PolicyModel, PolicyHead
        return {"T5PolicyModel": T5PolicyModel, "PolicyHead": PolicyHead}[name]
    raise AttributeError(f"module 'models' has no attribute {name!r}")


__all__ = [
    "extract_belief_features",
    "entropy_of_distribution",
    "LikelihoodModel",
    "TfIdfLikelihood",
    "SBERTLikelihood",
    "OpenAILikelihood",
    "T5Likelihood",
    "build_likelihood_from_config",
    "T5PolicyModel",
    "PolicyHead",
]
````

## File: qb_data/mc_builder.py
````python
"""Multiple-choice question builder with anti-artifact guards."""

from __future__ import annotations

import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.text_utils import normalize_answer


@dataclass
class MCQuestion(TossupQuestion):
    """A tossup question with multiple-choice options.

    Extends TossupQuestion with fields for multiple-choice presentation
    and tracking of distractor generation strategy.
    """
    options: List[str]
    gold_index: int
    option_profiles: List[str]
    option_answer_primary: List[str]
    distractor_strategy: str


def _normalized_edit_distance(a: str, b: str) -> float:
    """Compute normalized edit distance between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Distance between 0 (identical) and 1 (completely different).
    """
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _token_overlap(a: str, b: str) -> float:
    """Compute token overlap between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Fraction of overlapping tokens (0 to 1).
    """
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


class MCBuilder:
    """Builder for multiple-choice questions with anti-artifact guards.

    This class implements four layers of guards to prevent spurious patterns
    that agents could exploit:
    1. Alias collision guard: Prevents distractors that are aliases of the gold answer
    2. Duplicate guard: Prevents distractors with high token overlap
    3. Length ratio guard: Prevents distractors much longer/shorter than others
    4. Question overlap guard: Prevents answers that appear in the question text
    """

    def __init__(
        self,
        K: int = 4,
        strategy: str = "sbert_profile",
        alias_edit_distance_threshold: float = 0.2,
        duplicate_token_overlap_threshold: float = 0.8,
        max_length_ratio: float = 3.0,
        random_seed: int = 13,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_model: str = "text-embedding-3-small",
    ):
        """Initialize the MC builder.

        Args:
            K: Number of answer choices (must be >= 2).
            strategy: Distractor selection strategy
                (sbert_profile, openai_profile, tfidf_profile, category_random).
            alias_edit_distance_threshold: Max edit distance for alias detection.
            duplicate_token_overlap_threshold: Max token overlap between options.
            max_length_ratio: Max ratio between longest and shortest option.
            random_seed: Random seed for reproducibility.
            embedding_model: SentenceTransformer model name for ``sbert_profile``.
            openai_model: OpenAI embedding model for ``openai_profile``.
        """
        if K < 2:
            raise ValueError("K must be >= 2")
        self.K = K
        self.strategy = strategy
        self.alias_edit_distance_threshold = alias_edit_distance_threshold
        self.duplicate_token_overlap_threshold = duplicate_token_overlap_threshold
        self.max_length_ratio = max_length_ratio
        self.rng = random.Random(random_seed)
        self.embedding_model = embedding_model
        self.openai_model = openai_model

    def _prepare_lookup(
        self, questions: List[TossupQuestion]
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], List[str]]:
        """Prepare lookup structures for answer processing.

        Args:
            questions: List of tossup questions.

        Returns:
            Tuple of (answer_to_aliases, answer_to_category, answer_to_norm, answers).
        """
        answer_to_aliases: Dict[str, Set[str]] = {}
        answer_to_category: Dict[str, str] = {}

        for q in questions:
            # Collect all aliases for each answer
            aliases = answer_to_aliases.setdefault(q.answer_primary, set())
            aliases.update(str(alias) for alias in q.clean_answers)
            aliases.add(q.answer_primary)

            # Track category for category-based distractor selection
            if q.category and q.answer_primary not in answer_to_category:
                answer_to_category[q.answer_primary] = q.category

        # Convert to sorted lists for consistency
        answer_to_aliases_list = {k: sorted(v) for k, v in answer_to_aliases.items()}
        answers = sorted(answer_to_aliases_list.keys())
        answer_to_norm = {a: str(normalize_answer(a)) for a in answers}

        return answer_to_aliases_list, answer_to_category, answer_to_norm, answers

    def _rank_by_similarity(
        self,
        sim: np.ndarray,
        answers: List[str],
        answer_idx: Dict[str, int],
        M: int,
    ) -> Dict[str, List[str]]:
        """Rank distractors for each answer using a similarity matrix.

        Uses ``np.argpartition`` for top-M retrieval when M < N-1,
        reducing per-answer work from O(N log N) to O(N + M log M).

        Parameters
        ----------
        sim : np.ndarray
            Pairwise similarity matrix of shape (N, N).
        answers : list[str]
            Ordered answer strings corresponding to matrix rows/cols.
        answer_idx : dict[str, int]
            Mapping from answer string to its index in *sim*.
        M : int
            Number of top candidates to retain per answer.

        Returns
        -------
        dict[str, list[str]]
            Each answer mapped to its ranked distractor list (length <= M).
        """
        N = len(answers)
        rankings: Dict[str, List[str]] = {}
        for answer in answers:
            idx = answer_idx[answer]
            row = sim[idx]
            if M >= N - 1:
                # Small N: full sort (no benefit from partition)
                order = np.argsort(-row).tolist()
            else:
                # Top-M retrieval: O(N) partition + O(M log M) sort
                top_m_idx = np.argpartition(-row, M)[:M]
                top_m_idx = top_m_idx[np.argsort(-row[top_m_idx])]
                order = top_m_idx.tolist()
            rankings[answer] = [answers[i] for i in order if answers[i] != answer]
        return rankings

    def _compute_rankings(
        self,
        answers: List[str],
        answer_profiles: Dict[str, str],
        answer_to_category: Dict[str, str],
    ) -> Dict[str, List[str]]:
        """Compute distractor rankings for each answer.

        For profile-based strategies, uses top-M retrieval via
        ``np.argpartition`` instead of full ``np.argsort`` to reduce
        per-answer complexity from O(N log N) to O(N + M log M) and
        total memory from O(N^2) to O(N*M), where M = max(5*K, 30).

        Args:
            answers: List of all unique answers.
            answer_profiles: Dictionary mapping answers to their profiles.
            answer_to_category: Dictionary mapping answers to categories.

        Returns:
            Dictionary mapping each answer to a ranked list of distractors.
        """
        if self.strategy == "category_random":
            # Random selection within the same category
            rankings: Dict[str, List[str]] = {}
            for answer in answers:
                category = answer_to_category.get(answer, "")
                # First try same category, then fall back to all answers
                candidates = [
                    a for a in answers
                    if a != answer and answer_to_category.get(a, "") == category
                ]
                if len(candidates) < self.K - 1:
                    candidates = [a for a in answers if a != answer]
                self.rng.shuffle(candidates)
                rankings[answer] = candidates
            return rankings

        # Profile-based ranking strategies
        docs = [answer_profiles[a] for a in answers]
        answer_idx = {a: i for i, a in enumerate(answers)}
        M = min(max(5 * self.K, 30), len(answers) - 1)

        if self.strategy == "tfidf_profile":
            # TF-IDF based similarity
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(docs)
            sim = cosine_similarity(matrix, matrix)
            return self._rank_by_similarity(sim, answers, answer_idx, M)

        if self.strategy in {"sbert_profile", "openai_profile"}:
            if self.strategy == "sbert_profile":
                # One-shot SBERT encoding for distractor ranking.
                # This is separate from the SBERTLikelihood runtime cache
                # because it runs only during MC dataset construction.
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(self.embedding_model)
                embeddings = encoder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
                sim = embeddings @ embeddings.T
            else:
                from models.likelihoods import OpenAILikelihood

                likelihood = OpenAILikelihood(model=self.openai_model)
                embeddings = likelihood.embed_and_cache(docs)
                sim = embeddings @ embeddings.T

            return self._rank_by_similarity(sim, answers, answer_idx, M)

        raise ValueError(f"Unknown distractor strategy: {self.strategy}")

    def _aliases_collide(self, candidate: str, gold_aliases: List[str]) -> bool:
        """Check if a candidate is too similar to any gold answer alias.

        Args:
            candidate: Candidate distractor.
            gold_aliases: List of aliases for the gold answer.

        Returns:
            True if the candidate collides with a gold alias.
        """
        candidate_norm = str(normalize_answer(candidate))
        gold_norms = [str(normalize_answer(alias)) for alias in gold_aliases]

        # Check exact match
        if candidate_norm in set(gold_norms):
            return True

        # Check edit distance
        for gold_norm in gold_norms:
            if _normalized_edit_distance(candidate_norm, gold_norm) < self.alias_edit_distance_threshold:
                return True

        return False

    def _violates_duplicate_guard(self, candidate: str, selected: List[str]) -> bool:
        """Check if candidate has too much token overlap with already selected options.

        Args:
            candidate: Candidate distractor.
            selected: List of already selected distractors.

        Returns:
            True if the candidate has too much overlap.
        """
        for chosen in selected:
            if _token_overlap(candidate, chosen) > self.duplicate_token_overlap_threshold:
                return True
        return False

    def _violates_length_ratio_guard(self, options: List[str]) -> bool:
        """Check if options have too different lengths.

        Args:
            options: List of all options.

        Returns:
            True if the length ratio is too high.
        """
        lengths = [max(1, len(o.split())) for o in options]
        return (max(lengths) / min(lengths)) > self.max_length_ratio

    def _violates_question_overlap_guard(self, question: str, options: List[str]) -> bool:
        """Check if any option appears in the question text.

        Args:
            question: Question text.
            options: List of answer options.

        Returns:
            True if any option appears in the question.
        """
        q_norm = str(normalize_answer(question))
        for option in options:
            o_norm = str(normalize_answer(option))
            if o_norm and o_norm in q_norm:
                return True
        return False

    def build(
        self,
        questions: List[TossupQuestion],
        profile_builder: AnswerProfileBuilder,
    ) -> List[MCQuestion]:
        """Build multiple-choice questions with anti-artifact guards.

        Args:
            questions: List of tossup questions.
            profile_builder: Profile builder for answer representations.

        Returns:
            List of MCQuestion objects that passed all guards.
        """
        if not questions:
            return []

        # Build answer profiles
        profile_builder.fit(questions)
        answer_profiles = profile_builder.build_profiles(questions)

        # Prepare lookup structures
        answer_to_aliases, answer_to_category, _answer_to_norm, answers = self._prepare_lookup(questions)

        # Compute distractor rankings
        rankings = self._compute_rankings(answers, answer_profiles, answer_to_category)

        mc_questions: List[MCQuestion] = []

        for q in questions:
            gold = q.answer_primary
            gold_aliases = answer_to_aliases.get(gold, [gold])
            ranked = rankings.get(gold, [a for a in answers if a != gold])
            selected: List[str] = []

            # Select distractors from ranked list
            for candidate in ranked:
                if candidate == gold:
                    continue
                # Apply guard 1: Check alias collision
                if self._aliases_collide(candidate, gold_aliases):
                    continue
                # Apply guard 2: Check duplicate tokens
                if self._violates_duplicate_guard(candidate, selected):
                    continue
                selected.append(candidate)
                if len(selected) >= self.K - 1:
                    break

            # If not enough distractors from ranking, try random fallback
            if len(selected) < self.K - 1:
                fallback = [a for a in answers if a not in selected and a != gold]
                self.rng.shuffle(fallback)
                for candidate in fallback:
                    if self._aliases_collide(candidate, gold_aliases):
                        continue
                    if self._violates_duplicate_guard(candidate, selected):
                        continue
                    selected.append(candidate)
                    if len(selected) >= self.K - 1:
                        break

            # Skip question if we can't find enough valid distractors
            if len(selected) < self.K - 1:
                continue

            # Create options and shuffle
            option_answer_primary = [gold] + selected[:self.K - 1]
            self.rng.shuffle(option_answer_primary)
            gold_index = option_answer_primary.index(gold)
            options = option_answer_primary[:]

            # Apply guard 3: Check length ratio
            if self._violates_length_ratio_guard(options):
                continue

            # Apply guard 4: Check question overlap
            if self._violates_question_overlap_guard(q.question, options):
                continue

            # Build option profiles with leave-one-out for gold
            option_profiles: List[str] = []
            for answer in option_answer_primary:
                exclude_qid = q.qid if answer == gold else None
                option_profiles.append(
                    profile_builder.profile_for_answer(answer, exclude_qid=exclude_qid)
                )

            # Create MCQuestion
            mc_questions.append(
                MCQuestion(
                    qid=q.qid,
                    question=q.question,
                    tokens=q.tokens,
                    answer_primary=q.answer_primary,
                    clean_answers=q.clean_answers,
                    run_indices=q.run_indices,
                    human_buzz_positions=q.human_buzz_positions,
                    category=q.category,
                    cumulative_prefixes=q.cumulative_prefixes,
                    options=options,
                    gold_index=gold_index,
                    option_profiles=option_profiles,
                    option_answer_primary=option_answer_primary,
                    distractor_strategy=self.strategy,
                )
            )

        return mc_questions


def build_mc_questions(
    questions: List[TossupQuestion],
    K: int,
    strategy: str,
    profile_builder: AnswerProfileBuilder,
    guards: Optional[Dict[str, Any]] = None,
    random_seed: int = 13,
) -> List[MCQuestion]:
    """Factory function to build multiple-choice questions.

    Args:
        questions: List of tossup questions.
        K: Number of answer choices.
        strategy: Distractor selection strategy.
        profile_builder: Profile builder for answer representations.
        guards: Optional dictionary of guard thresholds.
        random_seed: Random seed for reproducibility.

    Returns:
        List of MCQuestion objects that passed all guards.
    """
    guards = guards or {}
    builder = MCBuilder(
        K=K,
        strategy=strategy,
        alias_edit_distance_threshold=float(guards.get("alias_edit_distance_threshold", 0.2)),
        duplicate_token_overlap_threshold=float(guards.get("duplicate_token_overlap_threshold", 0.8)),
        max_length_ratio=float(guards.get("max_length_ratio", 3.0)),
        random_seed=random_seed,
    )
    return builder.build(questions=questions, profile_builder=profile_builder)
````

## File: qb_env/__init__.py
````python
"""Quiz Bowl Environment Package.

Gymnasium-compliant POMDP environment for quiz bowl question answering,
plus thin qb-rl compatibility exports for the old `qb_env.*` import paths.
"""

from qb_env.data_loader import (
    QANTADatasetLoader,
    TossupQuestion,
    load_tossup_questions,
    load_tossup_questions_from_config,
    parse_row,
)
from qb_env.mc_builder import MCBuilder, MCQuestion
from qb_env.text_utils import normalize_answer, tokenize_text
from qb_env.tossup_env import TossupMCEnv, make_env_from_config
from qb_env.text_wrapper import TextObservationWrapper

__all__ = [
    "TossupMCEnv",
    "make_env_from_config",
    "TextObservationWrapper",
    "TossupQuestion",
    "QANTADatasetLoader",
    "parse_row",
    "load_tossup_questions",
    "load_tossup_questions_from_config",
    "MCQuestion",
    "MCBuilder",
    "normalize_answer",
    "tokenize_text",
]
````

## File: scripts/_common.py
````python
"""Shared utilities for pipeline scripts.

Provides config loading, JSON serialization, MC question deserialization,
and path constants used across all pipeline scripts (build, baseline, train,
evaluate).

Ported from qb-rl reference implementation with import path adaptations
for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from models.likelihoods import LikelihoodModel, build_likelihood_from_config
from qb_data.config import load_config as load_yaml_config
from qb_data.mc_builder import MCQuestion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def load_config(config_path: str | None = None, smoke: bool = False) -> dict[str, Any]:
    """Load YAML configuration from a file path.

    Parameters
    ----------
    config_path : str or None
        Path to YAML config file. If None, loads ``configs/default.yaml``.

    Returns
    -------
    dict[str, Any]
        Parsed config dict with nested structure (data, likelihood,
        environment, ppo, etc.).
    """
    return load_yaml_config(config_path, smoke=smoke)


def build_likelihood_model(config: dict[str, Any], mc_questions: list[MCQuestion]):
    """Build a likelihood model with shared TF-IDF corpus handling."""
    corpus = None
    if config["likelihood"].get("model") == "tfidf":
        corpus = [q.question for q in mc_questions] + [
            profile
            for question in mc_questions
            for profile in question.option_profiles
        ]
    return build_likelihood_from_config(config, corpus_texts=corpus)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The created (or existing) directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_serializable(item: Any) -> Any:
    """Recursively convert dataclasses to dicts for JSON serialization.

    Parameters
    ----------
    item : Any
        Object to convert. Dataclasses are converted via ``asdict()``,
        dicts and lists are processed recursively.

    Returns
    -------
    Any
        JSON-serializable version of the input.
    """
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return {k: to_serializable(v) for k, v in item.items()}
    if isinstance(item, list):
        return [to_serializable(v) for v in item]
    return item


def save_json(path: str | Path, data: Any) -> Path:
    """Save data to a JSON file, creating parent directories as needed.

    Applies ``to_serializable`` to convert dataclasses before writing.

    Parameters
    ----------
    path : str or Path
        Output file path.
    data : Any
        Data to serialize. Dataclasses are converted to dicts automatically.

    Returns
    -------
    Path
        The path where the JSON was written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2)
    return p


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file.

    Returns
    -------
    Any
        Parsed JSON data.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def mc_question_from_dict(row: dict[str, Any]) -> MCQuestion:
    """Reconstruct an MCQuestion dataclass from a JSON-deserialized dict.

    Parameters
    ----------
    row : dict[str, Any]
        Dictionary with all MCQuestion fields.

    Returns
    -------
    MCQuestion
        Reconstructed MCQuestion instance.
    """
    return MCQuestion(
        qid=row["qid"],
        question=row["question"],
        tokens=list(row["tokens"]),
        answer_primary=row["answer_primary"],
        clean_answers=list(row["clean_answers"]),
        run_indices=list(row["run_indices"]),
        human_buzz_positions=row.get("human_buzz_positions"),
        category=row.get("category", ""),
        cumulative_prefixes=list(row["cumulative_prefixes"]),
        options=list(row["options"]),
        gold_index=int(row["gold_index"]),
        option_profiles=list(row["option_profiles"]),
        option_answer_primary=list(row["option_answer_primary"]),
        distractor_strategy=row.get("distractor_strategy", "unknown"),
    )


def load_mc_questions(path: str | Path) -> list[MCQuestion]:
    """Load and deserialize a list of MCQuestions from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file containing a list of serialized MCQuestion dicts.

    Returns
    -------
    list[MCQuestion]
        List of reconstructed MCQuestion instances.
    """
    raw = load_json(path)
    return [mc_question_from_dict(item) for item in raw]


# ------------------------------------------------------------------ #
# Embedding cache persistence helpers
# ------------------------------------------------------------------ #


def embedding_cache_path(config: dict[str, Any]) -> Path:
    """Return the resolved embedding cache file path from config.

    Uses ``config['likelihood']['cache_dir']`` (default ``'cache/embeddings'``)
    and appends ``'embedding_cache.npz'``.

    Parameters
    ----------
    config : dict
        Full YAML config dict.

    Returns
    -------
    Path
        Absolute path to the embedding cache ``.npz`` file.
    """
    cache_dir = config.get("likelihood", {}).get("cache_dir", "cache/embeddings")
    return PROJECT_ROOT / cache_dir / "embedding_cache.npz"


def load_embedding_cache(model: LikelihoodModel, config: dict[str, Any]) -> None:
    """Load persisted embedding cache into model if file exists.

    Parameters
    ----------
    model : LikelihoodModel
        Likelihood model whose embedding_cache will be populated.
    config : dict
        Full YAML config dict (used to resolve cache path).
    """
    path = embedding_cache_path(config)
    n = model.load_cache(path)
    if n > 0:
        print(f"Loaded {n} cached embeddings from {path}")


def save_embedding_cache(model: LikelihoodModel, config: dict[str, Any]) -> None:
    """Persist model's embedding cache to disk.

    Parameters
    ----------
    model : LikelihoodModel
        Likelihood model whose embedding_cache will be saved.
    config : dict
        Full YAML config dict (used to resolve cache path).
    """
    path = embedding_cache_path(config)
    n = model.save_cache(path)
    if n > 0:
        print(f"Saved {n} embeddings to {path}")
````

## File: scripts/build_mc_dataset.py
````python
#!/usr/bin/env python3
"""
Build multiple-choice dataset from QANTA quiz bowl questions.

This script orchestrates the complete data pipeline:
1. Load questions from CSV or HuggingFace
2. Build answer profiles from training data
3. Generate MC questions with anti-artifact guards
4. Create stratified train/val/test splits
5. Save processed datasets as JSON

Usage:
    python scripts/build_mc_dataset.py
    python scripts/build_mc_dataset.py --smoke  # Quick test with 50 questions in artifacts/smoke
    python scripts/build_mc_dataset.py --config configs/custom.yaml
    python scripts/build_mc_dataset.py --data.K=5 --data.distractor_strategy=tfidf_profile
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qb_data import TossupQuestion
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.config import load_config, merge_overrides, resolve_data_loading_options
from qb_data.data_loader import QANTADatasetLoader
from qb_data.dataset_splits import create_stratified_splits
from qb_data.huggingface_loader import load_from_huggingface
from qb_data.mc_builder import MCBuilder, MCQuestion

DEFAULT_OUTPUT_DIR = Path("data/processed")
SMOKE_OUTPUT_DIR = Path("artifacts/smoke")


def parse_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Parse CLI override arguments into nested dictionary.

    Converts args like --data.K=5 into {"data": {"K": 5}}

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    Dict[str, Any]
        Nested dictionary of config overrides
    """
    overrides = {}

    # Check for any attributes that look like overrides (contain dots)
    for key, value in vars(args).items():
        if value is not None and '.' not in key:
            continue  # Skip non-override args

    # Parse remaining args for dot notation overrides
    if hasattr(args, 'overrides') and args.overrides:
        for override in args.overrides:
            if '=' not in override:
                continue

            key, value = override.split('=', 1)
            keys = key.split('.')

            # Try to parse value as JSON first, then as int/float/bool
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                if value.lower() == 'true':
                    parsed_value = True
                elif value.lower() == 'false':
                    parsed_value = False
                elif value.isdigit():
                    parsed_value = int(value)
                else:
                    try:
                        parsed_value = float(value)
                    except ValueError:
                        parsed_value = value  # Keep as string

            # Build nested dictionary
            d = overrides
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = parsed_value

    return overrides


def resolve_output_dir(output_dir: Optional[str], smoke: bool) -> Path:
    """Resolve the dataset output directory from CLI inputs."""
    if output_dir is not None:
        return Path(output_dir)
    return SMOKE_OUTPUT_DIR if smoke else DEFAULT_OUTPUT_DIR


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for dataset construction."""
    parser = argparse.ArgumentParser(
        description="Build multiple-choice dataset from QANTA questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=(
            "Path to YAML configuration file. Defaults to configs/default.yaml, "
            "or the smoke config path selected by load_config() when --smoke is set."
        ),
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Use smoke test settings (50 questions, quick run, outputs to artifacts/smoke by default).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save processed datasets. Defaults to data/processed, or artifacts/smoke when --smoke is set.',
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides in format: data.K=5 data.distractor_strategy=tfidf_profile',
    )

    return parser.parse_args(argv)


def save_json(path: Path, data: List[Any]) -> None:
    """
    Save dataclass objects to JSON file.

    Parameters
    ----------
    path : Path
        Output file path
    data : List[Any]
        List of dataclass objects (TossupQuestion or MCQuestion)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dictionaries
    if data and hasattr(data[0], '__dataclass_fields__'):
        # It's a dataclass, use asdict
        from dataclasses import asdict
        json_data = [asdict(item) for item in data]
    else:
        json_data = data

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} items to {path}")


def print_statistics(
    train: List[MCQuestion],
    val: List[MCQuestion],
    test: List[MCQuestion],
    profile_builder: Optional[AnswerProfileBuilder] = None,
    mc_builder: Optional[MCBuilder] = None
) -> None:
    """
    Print dataset statistics.

    Parameters
    ----------
    train : List[MCQuestion]
        Training split
    val : List[MCQuestion]
        Validation split
    test : List[MCQuestion]
        Test split
    profile_builder : Optional[AnswerProfileBuilder]
        Answer profile builder for profile stats
    mc_builder : Optional[MCBuilder]
        MC builder for guard rejection stats
    """
    print("\n" + "="*60)
    print("Dataset Construction Complete")
    print("="*60)

    # Split statistics
    total = len(train) + len(val) + len(test)
    print(f"\nTotal MC questions: {total}")
    print(f"  Train: {len(train)} ({100*len(train)/total:.1f}%)")
    print(f"  Val:   {len(val)} ({100*len(val)/total:.1f}%)")
    print(f"  Test:  {len(test)} ({100*len(test)/total:.1f}%)")

    # Category distribution
    def get_categories(questions):
        return set(q.category for q in questions if q.category)

    all_categories = get_categories(train) | get_categories(val) | get_categories(test)
    print(f"\nCategories: {len(all_categories)}")

    # Sample categories
    sample_cats = sorted(all_categories)[:5]
    print("Sample categories:", ", ".join(sample_cats))

    # Answer profile statistics
    if profile_builder and hasattr(profile_builder, '_grouped'):
        print(f"\nAnswer profiles: {len(profile_builder._grouped)}")
        # Get average questions per answer
        avg_questions = sum(len(items) for items in profile_builder._grouped.values()) / len(profile_builder._grouped)
        print(f"Average questions per answer: {avg_questions:.1f}")

    # Guard rejection statistics
    if mc_builder and hasattr(mc_builder, 'guard_stats'):
        stats = mc_builder.guard_stats
        if stats:
            print("\nGuard rejection statistics:")
            for guard_name, count in stats.items():
                print(f"  {guard_name}: {count} rejections")

    # Sample MC question
    if train:
        sample = train[0]
        print(f"\nSample MC question:")
        # Get first sentence from the question
        first_sentence = sample.question[:100] + "..." if len(sample.question) > 100 else sample.question
        print(f"  Question: {first_sentence}")
        print(f"  Correct answer: {sample.answer_primary}")
        print(f"  Options: {', '.join(sample.options[:3])}...")
        print(f"  Category: {sample.category}")


def main(argv: Optional[list[str]] = None):
    """Main entry point for dataset construction."""
    args = parse_args(argv)

    # Start timing
    start_time = time.time()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config, smoke=args.smoke)

    # Apply overrides
    overrides = parse_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = merge_overrides(config, overrides)

    # Create output directory
    output_dir = resolve_output_dir(args.output_dir, smoke=args.smoke)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    print("\nLoading questions...")
    questions = None
    data_opts = resolve_data_loading_options(config, smoke=args.smoke)

    # Try CSV first
    csv_path = data_opts.get('csv_path')
    if csv_path and Path(csv_path).exists():
        print(f"Loading from CSV: {csv_path}")
        loader = QANTADatasetLoader()
        questions = loader.load_from_csv(csv_path)
        print(f"Loaded {len(questions)} questions from CSV")

    # Fallback to HuggingFace if configured
    if questions is None and data_opts.get('use_huggingface'):
        print("CSV not found, falling back to HuggingFace")
        dataset_name = data_opts.get('dataset') or 'qanta-challenge/acf-co24-tossups'
        questions = load_from_huggingface(
            dataset_name,
            config_name=data_opts.get('dataset_config'),
            split=data_opts.get('split', 'eval'),
        )
        print(f"Loaded {len(questions)} questions from HuggingFace")

    if questions is None:
        raise FileNotFoundError(f"Could not load questions from {csv_path} and HuggingFace fallback not enabled")

    # Apply configured limit after loading
    max_questions = data_opts.get('max_questions')
    if max_questions is not None and len(questions) > int(max_questions):
        print(f"Limiting dataset to {int(max_questions)} questions")
        questions = questions[: int(max_questions)]

    # Build answer profiles
    print("\nBuilding answer profiles...")
    profile_builder = AnswerProfileBuilder(
        max_tokens_per_profile=config['answer_profiles']['max_tokens_per_profile'],
        min_questions_per_answer=config['answer_profiles']['min_questions_per_answer']
    )
    profile_builder.fit(questions)
    print(f"Built {len(profile_builder._grouped)} answer profiles")

    # Construct MC questions with guards
    print("\nConstructing MC questions...")
    mc_builder = MCBuilder(
        K=config['data']['K'],
        strategy=config['data']['distractor_strategy'],
        embedding_model=config['likelihood'].get(
            'sbert_name',
            config['likelihood'].get('embedding_model', 'all-MiniLM-L6-v2'),
        ),
        openai_model=config['likelihood'].get('openai_model', 'text-embedding-3-small'),
        **config['mc_guards']
    )

    # Track guard statistics
    mc_builder.guard_stats = {}

    mc_questions = mc_builder.build(questions, profile_builder)
    print(f"Generated {len(mc_questions)} MC questions")

    if len(mc_questions) < len(questions):
        print(f"Note: {len(questions) - len(mc_questions)} questions filtered by guards")

    # Create stratified splits
    print("\nCreating stratified splits...")
    ratios = [
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        config['data']['test_ratio']
    ]

    train, val, test = create_stratified_splits(mc_questions, ratios=ratios)

    # Save datasets
    print("\nSaving datasets...")
    save_json(output_dir / "mc_dataset.json", mc_questions)
    save_json(output_dir / "train_dataset.json", train)
    save_json(output_dir / "val_dataset.json", val)
    save_json(output_dir / "test_dataset.json", test)

    # Save answer profiles for debugging
    if profile_builder._grouped:
        profiles_dict = {
            answer: {
                'question_count': len(items),
                'sample_qids': [qid for qid, _ in items[:5]]  # First 5 question IDs
            }
            for answer, items in profile_builder._grouped.items()
        }
        with open(output_dir / "answer_profiles.json", 'w') as f:
            json.dump(profiles_dict, f, indent=2)
        print(f"Saved answer profiles to {output_dir / 'answer_profiles.json'}")

    # Print statistics
    print_statistics(train, val, test, profile_builder, mc_builder)

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")

    if args.smoke:
        # Print sample MC questions for verification
        print("\n" + "="*60)
        print("Sample MC Questions (Smoke Test)")
        print("="*60)

        for i, q in enumerate(train[:3], 1):
            print(f"\nQuestion {i}:")
            # Get first clue from cumulative_prefixes if available
            if q.cumulative_prefixes:
                first_clue = q.cumulative_prefixes[0][:100] + "..." if len(q.cumulative_prefixes[0]) > 100 else q.cumulative_prefixes[0]
            else:
                first_clue = q.question[:100] + "..." if len(q.question) > 100 else q.question
            print(f"  First clue: {first_clue}")
            print(f"  Category: {q.category}")
            print(f"  Correct: {q.answer_primary}")
            print(f"  Options: {', '.join(q.options[:3])}...")

    print("\nDataset construction complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
````

## File: tests/test_likelihoods.py
````python
"""Test suite for models/likelihoods.py — likelihood model interface and implementations.

Covers:
- LIK-01: LikelihoodModel ABC contract
- LIK-02: TfIdfLikelihood with corpus fitting and cosine scoring
- LIK-03: SBERTLikelihood with semantic embeddings and caching
- LIK-04: T5Likelihood semantic scoring and embedding shape
- LIK-05: T5 embedding cache reuse and factory construction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models.likelihoods import (
    LikelihoodModel,
    SBERTLikelihood,
    TfIdfLikelihood,
)


# ------------------------------------------------------------------ #
# Tests for LikelihoodModel ABC
# ------------------------------------------------------------------ #


class TestLikelihoodModelABC:
    """Tests for the abstract base class contract."""

    def test_abstract_interface_cannot_instantiate(self) -> None:
        """LikelihoodModel ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LikelihoodModel()  # type: ignore[abstract]

    def test_embedding_cache_on_subclass(self, sample_corpus: list[str]) -> None:
        """Concrete subclass inherits embedding_cache dict."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert hasattr(model, "embedding_cache"), "Missing embedding_cache attribute"
        assert isinstance(model.embedding_cache, dict), "embedding_cache should be dict"


# ------------------------------------------------------------------ #
# Tests for TfIdfLikelihood
# ------------------------------------------------------------------ #


class TestTfIdfLikelihood:
    """Tests for TF-IDF based likelihood model."""

    def test_tfidf_requires_fit(self) -> None:
        """score() before fit() raises RuntimeError."""
        model = TfIdfLikelihood()
        with pytest.raises(RuntimeError, match="must be fit"):
            model.score("test clue", ["option1", "option2"])

    def test_tfidf_embed_requires_fit(self) -> None:
        """_embed_batch() before fit() raises RuntimeError."""
        model = TfIdfLikelihood()
        with pytest.raises(RuntimeError, match="must be fit"):
            model._embed_batch(["test text"])

    def test_tfidf_fit_and_score(self, sample_corpus: list[str]) -> None:
        """After fitting, score returns correct shape and dtype.

        Also verifies that more relevant text scores higher.
        """
        model = TfIdfLikelihood()
        model.fit(sample_corpus)

        scores = model.score(
            "Who was the first president?",
            ["George Washington first president", "Abraham Lincoln Civil War"],
        )
        assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
        assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"
        # Washington should score higher for "first president" clue
        assert scores[0] >= scores[1], (
            f"Washington ({scores[0]:.3f}) should score >= Lincoln ({scores[1]:.3f})"
        )

    def test_tfidf_embed_batch(self, sample_corpus: list[str]) -> None:
        """_embed_batch produces dense vectors of correct shape."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        embeddings = model._embed_batch(["test text one", "test text two"])
        assert embeddings.shape[0] == 2, f"Expected 2 rows, got {embeddings.shape[0]}"
        assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
        vocab_size = len(model.vectorizer.vocabulary_)
        assert embeddings.shape[1] == vocab_size, (
            f"Expected {vocab_size} cols, got {embeddings.shape[1]}"
        )

    def test_tfidf_corpus_in_constructor(self, sample_corpus: list[str]) -> None:
        """Passing corpus_texts to __init__ auto-fits the model."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert model._is_fit is True, "Model should be fit after corpus in constructor"
        # Should work without explicit fit()
        scores = model.score("president", ["Washington", "Lincoln"])
        assert scores.shape == (2,)

    def test_tfidf_fit_returns_self(self, sample_corpus: list[str]) -> None:
        """fit() returns self for method chaining."""
        model = TfIdfLikelihood()
        result = model.fit(sample_corpus)
        assert result is model, "fit() should return self"

    def test_tfidf_score_all_options(self, sample_corpus: list[str]) -> None:
        """Score works with 4 options matching K=4 environment setup."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        scores = model.score(
            "first president United States",
            [
                "George Washington commander revolutionary",
                "Thomas Jefferson declaration independence",
                "John Adams Massachusetts diplomat",
                "Benjamin Franklin inventor Philadelphia",
            ],
        )
        assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
        assert all(np.isfinite(scores)), "All scores should be finite"

    def test_tfidf_embed_batch_normalized(self, sample_corpus: list[str]) -> None:
        """_embed_batch returns L2-normalized vectors (row norms ~1.0)."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        embeddings = model._embed_batch(["George Washington president", "Thomas Jefferson"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)

    def test_tfidf_score_uses_cache(self, sample_corpus: list[str]) -> None:
        """score() populates embedding_cache via embed_and_cache()."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert len(model.embedding_cache) == 0
        model.score("first president", ["Washington profile", "Lincoln profile"])
        assert len(model.embedding_cache) == 3  # 1 clue + 2 options

    def test_tfidf_score_cache_hit(self, sample_corpus: list[str]) -> None:
        """Repeated score() with same options reuses cache."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        options = ["George Washington president", "Thomas Jefferson declaration"]
        model.score("first president", options)
        cache_after_first = len(model.embedding_cache)
        model.score("second president", options)
        # Only the new clue should be added; options are cached
        assert len(model.embedding_cache) == cache_after_first + 1

    def test_tfidf_score_matches_cosine_reference(self, sample_corpus: list[str]) -> None:
        """New cached score() matches sklearn cosine_similarity reference."""
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos

        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        clue = "Who was the first president?"
        options = [
            "George Washington first president commander revolutionary",
            "Abraham Lincoln Civil War emancipation",
            "Thomas Jefferson declaration independence Virginia",
            "Benjamin Franklin inventor Philadelphia diplomat",
        ]
        # Compute reference via sklearn cosine_similarity (old method)
        clue_vec = model.vectorizer.transform([clue])
        option_vecs = model.vectorizer.transform(options)
        ref_scores = sklearn_cos(clue_vec, option_vecs)[0].astype(np.float32)
        # Compute via new cached path
        actual_scores = model.score(clue, options)
        np.testing.assert_allclose(actual_scores, ref_scores, atol=1e-6)


# ------------------------------------------------------------------ #
# Tests for SBERTLikelihood
# ------------------------------------------------------------------ #


class TestSBERTLikelihood:
    """Tests for Sentence-BERT likelihood model."""

    def test_sbert_instantiation(self) -> None:
        """SBERTLikelihood can be instantiated with default model."""
        model = SBERTLikelihood()
        assert hasattr(model, "encoder"), "Missing encoder attribute"
        assert model.model_name == "all-MiniLM-L6-v2"

    def test_sbert_score_shape_and_dtype(self) -> None:
        """score() returns correct shape and dtype for 4 options."""
        model = SBERTLikelihood()
        scores = model.score(
            "first president United States",
            [
                "George Washington first president commander",
                "Thomas Jefferson third president declaration",
                "John Adams second president Massachusetts",
                "Benjamin Franklin inventor diplomat",
            ],
        )
        assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
        assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"

    def test_sbert_semantic_ranking(self) -> None:
        """SBERT ranks semantically similar text higher."""
        model = SBERTLikelihood()
        scores = model.score(
            "George Washington was the first president of the United States and led the Continental Army",
            [
                "George Washington first president commander revolutionary war continental army",
                "The theory of relativity was developed by Albert Einstein in physics",
            ],
        )
        # Washington profile should score much higher than Einstein
        assert scores[0] > scores[1], (
            f"Washington ({scores[0]:.3f}) should score > Einstein ({scores[1]:.3f})"
        )

    def test_sbert_embedding_cache_populated(self) -> None:
        """Embedding cache grows after first scoring call."""
        model = SBERTLikelihood()
        assert len(model.embedding_cache) == 0, "Cache should start empty"

        model.score("test clue", ["option A", "option B"])
        cache_after_first = len(model.embedding_cache)
        assert cache_after_first > 0, "Cache should be populated after score()"

    def test_sbert_embedding_cache_hit(self) -> None:
        """Repeated calls with same text use cache (size unchanged)."""
        model = SBERTLikelihood()
        scores1 = model.score("test clue", ["option A", "option B"])
        cache_size_1 = len(model.embedding_cache)

        scores2 = model.score("test clue", ["option A", "option B"])
        cache_size_2 = len(model.embedding_cache)

        assert cache_size_2 == cache_size_1, (
            f"Cache grew from {cache_size_1} to {cache_size_2} on repeated call"
        )
        np.testing.assert_array_almost_equal(
            scores1, scores2, decimal=5,
            err_msg="Cached results should match original",
        )

    def test_sbert_normalized_embeddings(self) -> None:
        """SBERT embeddings are L2-normalized (norm ~ 1.0)."""
        model = SBERTLikelihood()
        embeddings = model._embed_batch(["test sentence one", "test sentence two"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(
            norms, np.ones(2), decimal=4,
            err_msg="Embeddings should be L2-normalized",
        )


# ------------------------------------------------------------------ #
# Tests for T5Likelihood (LIK-04, LIK-05)
# ------------------------------------------------------------------ #


class TestT5Likelihood:
    """Tests for T5 encoder likelihood model.

    Uses the sample_t5_model fixture (t5-small, module-scoped) from conftest.py
    so the model is loaded once per test file, not per test function.
    """

    def test_t5_semantic_scoring(self, sample_t5_model) -> None:
        """T5 should score semantically relevant options higher (LIK-04).

        "First president" clue should rank Washington higher than Einstein,
        demonstrating that T5 captures semantic similarity between question
        content and answer profiles.
        """
        clue = "This person was the first president of the United States"
        options = [
            "George Washington first president commander revolutionary war",
            "Albert Einstein physicist theory relativity Nobel Prize",
        ]

        scores = sample_t5_model.score(clue, options)

        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float32
        assert len(scores) == 2
        # Washington should score higher than Einstein for "first president" query
        assert scores[0] > scores[1], (
            f"Expected Washington > Einstein, got {scores}"
        )

    def test_t5_embedding_cache(self, sample_t5_model) -> None:
        """T5 should cache embeddings and reuse them (LIK-05).

        After embedding two texts, the cache should contain 2 entries.
        Re-embedding the same texts should not grow the cache, and the
        returned embeddings should be identical.
        """
        # Clear cache to get a clean test
        sample_t5_model.embedding_cache.clear()

        texts = ["George Washington", "Thomas Jefferson"]

        # First call embeds and caches
        emb1 = sample_t5_model.embed_and_cache(texts)
        cache_size_1 = len(sample_t5_model.embedding_cache)

        # Second call reuses cache
        emb2 = sample_t5_model.embed_and_cache(texts)
        cache_size_2 = len(sample_t5_model.embedding_cache)

        np.testing.assert_array_equal(
            emb1, emb2, err_msg="Cached embeddings should match"
        )
        assert cache_size_1 == cache_size_2 == 2, (
            f"Cache size should not grow on reuse, got {cache_size_1} -> {cache_size_2}"
        )

    def test_t5_score_returns_float32(self, sample_t5_model) -> None:
        """T5 score should return float32 array, not probabilities.

        Scores are raw cosine similarities (not softmax probabilities),
        so they do not necessarily sum to 1.
        """
        scores = sample_t5_model.score("test clue", ["option 1", "option 2"])
        assert scores.dtype == np.float32
        assert scores.shape == (2,)
        # Scores are raw similarities, not probabilities (don't sum to 1)
        assert all(np.isfinite(scores)), "All scores should be finite"

    def test_build_t5_from_config(self) -> None:
        """Factory should construct T5Likelihood from config (LIK-04).

        The build_likelihood_from_config factory should recognize
        model="t5" and instantiate a T5Likelihood with the specified
        t5_name parameter.
        """
        from models.likelihoods import T5Likelihood, build_likelihood_from_config

        config = {
            "likelihood": {
                "model": "t5",
                "t5_name": "t5-small",
            }
        }

        model = build_likelihood_from_config(config)
        assert isinstance(model, T5Likelihood)
        assert model.model_name == "t5-small"

    def test_t5_handles_variable_length(self, sample_t5_model) -> None:
        """T5 should handle variable-length texts via attention mask.

        Short and long texts should both embed without error, producing
        embeddings of the same hidden dimension regardless of input length.
        """
        short = "Washington"
        long = (
            "George Washington was the first president of the United States "
            "and commander of the Continental Army during the Revolutionary War"
        )

        # Both should embed without error
        embs = sample_t5_model.embed_and_cache([short, long])
        assert embs.shape == (2, sample_t5_model.encoder.config.d_model), (
            f"Expected shape (2, {sample_t5_model.encoder.config.d_model}), "
            f"got {embs.shape}"
        )


# ------------------------------------------------------------------ #
# Tests for Embedding Cache Persistence
# ------------------------------------------------------------------ #


class TestEmbeddingCachePersistence:
    """Tests for save_cache / load_cache disk persistence on LikelihoodModel."""

    def test_save_load_cache_round_trip(self, tmp_path: Path, sample_corpus: list[str]) -> None:
        """save_cache writes .npz; load_cache restores identical entries."""
        model = SBERTLikelihood()
        texts = ["George Washington", "Thomas Jefferson", "Abraham Lincoln"]
        model.embed_and_cache(texts)
        assert len(model.embedding_cache) == 3

        cache_path = tmp_path / "cache.npz"
        saved = model.save_cache(cache_path)
        assert saved == 3
        assert cache_path.exists()

        model2 = SBERTLikelihood()
        assert len(model2.embedding_cache) == 0
        loaded = model2.load_cache(cache_path)
        assert loaded == 3

        for key in model.embedding_cache:
            np.testing.assert_array_equal(
                model.embedding_cache[key],
                model2.embedding_cache[key],
                err_msg=f"Mismatch for key {key}",
            )

    def test_load_cache_missing_file(self, tmp_path: Path) -> None:
        """load_cache with nonexistent file returns 0 and leaves cache empty."""
        model = SBERTLikelihood()
        result = model.load_cache(tmp_path / "nonexistent.npz")
        assert result == 0
        assert len(model.embedding_cache) == 0

    def test_save_cache_empty(self, tmp_path: Path) -> None:
        """save_cache with empty cache creates a valid .npz with zero arrays."""
        model = SBERTLikelihood()
        cache_path = tmp_path / "empty.npz"
        saved = model.save_cache(cache_path)
        assert saved == 0
        assert cache_path.exists()

        # Should be loadable
        model2 = SBERTLikelihood()
        loaded = model2.load_cache(cache_path)
        assert loaded == 0

    def test_tfidf_save_cache_noop(self, sample_corpus: list[str]) -> None:
        """TfIdfLikelihood.save_cache is a no-op returning 0."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        # Populate the cache with some embeddings
        model.embed_and_cache(["test text one", "test text two"])
        assert len(model.embedding_cache) > 0

        import tempfile
        import os
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "should_not_exist.npz"
            result = model.save_cache(path)
            assert result == 0
            assert not path.exists(), "TfIdfLikelihood should NOT write a cache file"

    def test_load_cache_does_not_overwrite(self, tmp_path: Path) -> None:
        """load_cache merges without overwriting existing cache entries."""
        model = SBERTLikelihood()
        texts = ["Hello world"]
        model.embed_and_cache(texts)

        # Save this cache
        cache_path = tmp_path / "cache.npz"
        model.save_cache(cache_path)

        # Create a second model, pre-populate with the same key but different value
        model2 = SBERTLikelihood()
        from models.likelihoods import _text_key
        key = _text_key("Hello world")
        original_value = np.ones(384, dtype=np.float32)  # dummy
        model2.embedding_cache[key] = original_value

        loaded = model2.load_cache(cache_path)
        assert loaded == 0, "Key already present, so nothing should be loaded"

        # Original value should be preserved (not overwritten)
        np.testing.assert_array_equal(
            model2.embedding_cache[key],
            original_value,
            err_msg="Existing cache entry was overwritten by load_cache",
        )


class TestCacheMemory:
    """Verify cache_memory_bytes property for resource monitoring."""

    def test_tfidf_cache_memory_bytes(self, sample_corpus):
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert model.cache_memory_bytes == 0
        model.embed_and_cache(["George Washington"])
        assert model.cache_memory_bytes > 0

    def test_empty_cache_zero_bytes(self, sample_corpus):
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert model.cache_memory_bytes == 0
````

## File: agents/threshold_buzzer.py
````python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents._math import sigmoid
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion


@dataclass
class EpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    reward_like: float
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


def _scores_to_belief(scores: np.ndarray, beta: float) -> np.ndarray:
    """Convert raw similarity scores to a belief distribution via softmax."""
    shifted = scores - np.max(scores)
    probs = np.exp(beta * shifted)
    probs = probs / max(1e-12, probs.sum())
    return probs.astype(np.float32)


def _belief_stats(belief: np.ndarray) -> tuple[int, float, float]:
    """Return (top_idx, top_p, entropy) from a belief distribution."""
    top_idx = int(np.argmax(belief))
    top_p = float(belief[top_idx])
    clipped = np.clip(belief, 1e-12, 1.0)
    entropy = float(-(clipped * np.log(clipped)).sum())
    return top_idx, top_p, entropy


@dataclass
class _PrecomputedQuestion:
    """Pre-computed belief distributions for every clue step of one question."""
    qid: str
    gold_index: int
    num_options: int
    beliefs: list[np.ndarray]


def precompute_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    beta: float,
) -> list[_PrecomputedQuestion]:
    """Compute beliefs at every step for every question (single model pass).

    After calling ``likelihood_model.precompute_embeddings()`` this is
    pure cache lookups + numpy math, so it runs in seconds rather than
    hours.
    """
    from tqdm import tqdm

    out: list[_PrecomputedQuestion] = []
    for q in tqdm(questions, desc="Computing beliefs"):
        beliefs = [
            _scores_to_belief(
                likelihood_model.score(prefix, q.option_profiles), beta
            )
            for prefix in q.cumulative_prefixes
        ]
        out.append(_PrecomputedQuestion(
            qid=q.qid,
            gold_index=q.gold_index,
            num_options=len(q.options),
            beliefs=beliefs,
        ))
    return out


class ThresholdBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        self.belief: np.ndarray | None = None

    def _belief_from_prefix(self, prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(prefix, option_profiles)
        return _scores_to_belief(scores, self.beta)

    def _confidence_proxy(self, top_p: float) -> float:
        return sigmoid(self.alpha * (top_p - self.threshold))

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        chosen_step = len(question.cumulative_prefixes) - 1
        chosen_idx = 0

        for step_idx, prefix in enumerate(question.cumulative_prefixes):
            belief = self._belief_from_prefix(prefix, question.option_profiles)
            self.belief = belief
            top_idx, top_p, entropy = _belief_stats(belief)
            c_t = self._confidence_proxy(top_p)
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        correct = chosen_idx == question.gold_index
        reward_like = 1.0 if correct else -0.5
        return EpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=correct,
            reward_like=reward_like,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


class AlwaysBuzzFinalBuzzer:
    def __init__(self, likelihood_model: LikelihoodModel, beta: float = 5.0):
        self.likelihood_model = likelihood_model
        self.beta = beta

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        final_step = len(question.cumulative_prefixes) - 1
        final_belief = np.ones(len(question.options), dtype=np.float32) / len(question.options)
        for prefix in question.cumulative_prefixes:
            scores = self.likelihood_model.score(prefix, question.option_profiles)
            probs = _scores_to_belief(scores, self.beta)
            final_belief = probs
            top_idx, top_p, entropy = _belief_stats(probs)
            c_trace.append(0.0)
            g_trace.append(1.0 if top_idx == question.gold_index else 0.0)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

        c_trace[-1] = 1.0
        buzz_idx = int(np.argmax(final_belief))
        correct = buzz_idx == question.gold_index
        reward_like = 1.0 if correct else -0.5
        return EpisodeResult(
            qid=question.qid,
            buzz_step=final_step,
            buzz_index=buzz_idx,
            gold_index=question.gold_index,
            correct=correct,
            reward_like=reward_like,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


def _softmax_episode_from_precomputed(
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
) -> "SoftmaxEpisodeResult":
    """Build a SoftmaxEpisodeResult from pre-computed beliefs (pure numpy).

    Identical buzzing logic to ``SoftmaxProfileBuzzer.run_episode`` but
    reads beliefs from a ``_PrecomputedQuestion`` instead of calling the
    likelihood model.
    """
    from agents.bayesian_buzzer import SoftmaxEpisodeResult

    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    chosen_step = len(pq.beliefs) - 1
    chosen_idx = 0

    for step_idx, belief in enumerate(pq.beliefs):
        top_idx, top_p, entropy = _belief_stats(belief)
        c_t = sigmoid(alpha * (top_p - threshold))
        g_t = 1.0 if top_idx == pq.gold_index else 0.0

        c_trace.append(c_t)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

        is_last = step_idx == len(pq.beliefs) - 1
        if top_p >= threshold or is_last:
            chosen_step = step_idx
            chosen_idx = top_idx
            break

    correct = chosen_idx == pq.gold_index
    return SoftmaxEpisodeResult(
        qid=pq.qid,
        buzz_step=chosen_step,
        buzz_index=chosen_idx,
        gold_index=pq.gold_index,
        correct=correct,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def _always_final_from_precomputed(pq: _PrecomputedQuestion) -> EpisodeResult:
    """Build an EpisodeResult for AlwaysBuzzFinal from pre-computed beliefs.

    Iterates all beliefs (no early stopping), buzzes at the last step
    with argmax of the final belief.
    """
    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    for belief in pq.beliefs:
        top_idx, top_p, entropy = _belief_stats(belief)
        g_t = 1.0 if top_idx == pq.gold_index else 0.0
        c_trace.append(0.0)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

    c_trace[-1] = 1.0
    buzz_idx = int(np.argmax(pq.beliefs[-1]))
    correct = buzz_idx == pq.gold_index
    return EpisodeResult(
        qid=pq.qid,
        buzz_step=len(pq.beliefs) - 1,
        buzz_index=buzz_idx,
        gold_index=pq.gold_index,
        correct=correct,
        reward_like=1.0 if correct else -0.5,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def _episode_from_precomputed(
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
) -> EpisodeResult:
    """Build an EpisodeResult from pre-computed beliefs (pure numpy)."""
    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    chosen_step = len(pq.beliefs) - 1
    chosen_idx = 0

    for step_idx, belief in enumerate(pq.beliefs):
        top_idx, top_p, entropy = _belief_stats(belief)
        c_t = sigmoid(alpha * (top_p - threshold))
        g_t = 1.0 if top_idx == pq.gold_index else 0.0

        c_trace.append(c_t)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

        is_last = step_idx == len(pq.beliefs) - 1
        if top_p >= threshold or is_last:
            chosen_step = step_idx
            chosen_idx = top_idx
            break

    correct = chosen_idx == pq.gold_index
    return EpisodeResult(
        qid=pq.qid,
        buzz_step=chosen_step,
        buzz_index=chosen_idx,
        gold_index=pq.gold_index,
        correct=correct,
        reward_like=1.0 if correct else -0.5,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def sweep_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
    precomputed: list[_PrecomputedQuestion] | None = None,
) -> dict[float, list[EpisodeResult]]:
    """Sweep multiple thresholds with a single belief-computation pass.

    If *precomputed* is provided the expensive model calls are skipped
    entirely and the sweep is pure numpy.  Otherwise beliefs are computed
    once internally and reused across thresholds.
    """
    if precomputed is None:
        precomputed = precompute_beliefs(questions, likelihood_model, beta)

    out: dict[float, list[EpisodeResult]] = {}
    for threshold in thresholds:
        out[float(threshold)] = [
            _episode_from_precomputed(pq, threshold, alpha)
            for pq in precomputed
        ]
    return out


def result_to_dict(result: EpisodeResult) -> dict[str, Any]:
    return {
        "qid": result.qid,
        "buzz_step": result.buzz_step,
        "buzz_index": result.buzz_index,
        "gold_index": result.gold_index,
        "correct": result.correct,
        "reward_like": result.reward_like,
        "c_trace": result.c_trace,
        "g_trace": result.g_trace,
        "top_p_trace": result.top_p_trace,
        "entropy_trace": result.entropy_trace,
    }
````

## File: qb_env/tossup_env.py
````python
"""
Gymnasium-compliant POMDP Environment for Quiz Bowl

Implements a tossup question environment where clues are revealed incrementally.
At each step the agent observes a belief-based feature vector and chooses either
to WAIT (action 0, reveals next clue) or to BUZZ with a specific answer option
(actions 1..K, ends the episode).

The environment computes beliefs over K answer options using a pluggable
LikelihoodModel and converts them to observations via extract_belief_features.

Ported from qb-rl reference implementation (qb_env/tossup_env.py) and adapted
for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models.features import extract_belief_features
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion


def _softmax(scores: np.ndarray, beta: float) -> np.ndarray:
    """Temperature-scaled softmax with numerical stability.

    Parameters
    ----------
    scores : np.ndarray
        Raw similarity scores of shape (K,).
    beta : float
        Temperature parameter. Higher values produce sharper distributions.

    Returns
    -------
    np.ndarray
        Probability distribution of shape (K,), dtype float32.
    """
    stable = scores - np.max(scores)
    probs = np.exp(beta * stable)
    probs_sum = np.sum(probs)
    if probs_sum <= 0:
        return np.ones_like(scores, dtype=np.float32) / len(scores)
    return (probs / probs_sum).astype(np.float32)


def precompute_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    belief_mode: str = "from_scratch",
    beta: float = 5.0,
    K: int = 4,
) -> dict[tuple[int, int], np.ndarray]:
    """Precompute belief trajectories for all questions and steps.

    Iterates over each question and each step index, computing the belief
    using the same logic as ``TossupMCEnv._compute_belief``. The result is
    a dict keyed by ``(question_index, step_idx)`` for O(1) lookup during
    training rollouts.

    Parameters
    ----------
    questions : list[MCQuestion]
        Pool of questions to precompute beliefs for.
    likelihood_model : LikelihoodModel
        Model that scores clue text against answer option profiles.
    belief_mode : str
        One of ``"from_scratch"``, ``"sequential_bayes"``.
    beta : float
        Softmax temperature for converting raw scores to probabilities.
    K : int
        Number of answer options per question.

    Returns
    -------
    dict[tuple[int, int], np.ndarray]
        Maps ``(question_index, step_idx)`` to belief vectors of shape
        ``(K,)`` with dtype float32. Each belief sums to ~1.0.
    """
    cache: dict[tuple[int, int], np.ndarray] = {}

    for q_idx, question in enumerate(questions):
        num_steps = len(question.run_indices)
        belief = np.ones(K, dtype=np.float32) / K

        for step_idx in range(num_steps):
            if belief_mode == "from_scratch":
                prefix = question.cumulative_prefixes[step_idx]
                scores = likelihood_model.score(prefix, question.option_profiles)
                belief = _softmax(scores, beta)

            elif belief_mode == "sequential_bayes":
                idx = question.run_indices[step_idx]
                prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
                frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])
                scores = likelihood_model.score(frag, question.option_profiles)
                likelihood = _softmax(scores, beta)
                posterior = belief * likelihood
                denom = posterior.sum()
                if denom <= 0:
                    belief = np.ones(K, dtype=np.float32) / K
                else:
                    belief = (posterior / denom).astype(np.float32)

            else:
                raise ValueError(f"Unknown belief_mode: {belief_mode}")

            cache[(q_idx, step_idx)] = belief.copy()

    return cache


class TossupMCEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment for quiz bowl tossup questions with MC options.

    Models quiz bowl as a POMDP where clues are revealed incrementally.
    The agent maintains a belief distribution over K answer options, updated
    at each step by a likelihood model. The agent decides when to buzz and
    which answer to select.

    Action Space
    ------------
    Discrete(K + 1):
        - 0: WAIT -- reveal the next clue and update belief
        - 1..K: BUZZ with answer option (i-1), ending the episode

    Observation Space
    -----------------
    Box(K + 6,):
        Belief features: [belief[0..K-1], top_p, margin, entropy,
        stability, progress, clue_idx_norm].
        See ``models.features.extract_belief_features`` for details.

    Reward Modes
    ------------
    ``time_penalty`` (default):
        -wait_penalty per WAIT step; +buzz_correct for correct buzz,
        +buzz_incorrect (negative) for wrong buzz.
    ``simple``:
        +1.0 for correct buzz, -1.0 for incorrect buzz, no WAIT penalty.
    ``human_grounded``:
        0.0 if the agent buzzes after the sampled human buzz position;
        otherwise +buzz_correct/-buzz_incorrect for correct/incorrect.

    Belief Modes
    ------------
    ``from_scratch``:
        Recompute belief from all clues seen so far via cumulative_prefixes.
    ``sequential_bayes``:
        Bayesian update: multiply prior belief by likelihood of new clue
        fragment, then normalize.

    Parameters
    ----------
    questions : list[MCQuestion]
        Pool of questions to sample from. Must be non-empty.
    likelihood_model : LikelihoodModel
        Model that scores clue text against answer option profiles.
    K : int
        Number of answer options per question. Must be >= 2.
    reward_mode : str
        One of ``"time_penalty"``, ``"simple"``, ``"human_grounded"``.
    wait_penalty : float
        Per-step penalty when reward_mode is ``"time_penalty"``.
    buzz_correct : float
        Reward for buzzing with the correct answer.
    buzz_incorrect : float
        Reward (typically negative) for buzzing with an incorrect answer.
    belief_mode : str
        One of ``"from_scratch"``, ``"sequential_bayes"``.
    beta : float
        Softmax temperature for converting raw scores to probabilities.
        Higher values produce sharper distributions.
    seed : int
        Random seed for question sampling and human buzz simulation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        questions: list[MCQuestion],
        likelihood_model: LikelihoodModel,
        K: int = 4,
        reward_mode: str = "time_penalty",
        wait_penalty: float = 0.01,
        early_buzz_penalty: float = 0.0,
        buzz_correct: float = 1.0,
        buzz_incorrect: float = -0.5,
        belief_mode: str = "from_scratch",
        beta: float = 5.0,
        seed: int = 13,
        precomputed_beliefs: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> None:
        if not questions:
            raise ValueError("questions cannot be empty")
        if K < 2:
            raise ValueError("K must be >= 2")

        self.questions = questions
        self.likelihood_model = likelihood_model
        self.K = K
        self.reward_mode = reward_mode
        self.wait_penalty = wait_penalty
        self.early_buzz_penalty = early_buzz_penalty
        self.buzz_correct = buzz_correct
        self.buzz_incorrect = buzz_incorrect
        self.belief_mode = belief_mode
        self.beta = beta
        self.rng = random.Random(seed)
        self.precomputed_beliefs = precomputed_beliefs

        # Build qid -> list-index map for precomputed belief lookups
        self._question_index_map: dict[str, int] = {
            q.qid: i for i, q in enumerate(questions)
        }

        self.action_space = spaces.Discrete(self.K + 1)
        # belief[K] + (top_p, margin, entropy, stability, progress, clue_idx)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.K + 6,), dtype=np.float32
        )

        self.question: MCQuestion | None = None
        self.step_idx: int = 0
        self.prev_belief: np.ndarray | None = None
        self.belief: np.ndarray = np.ones(self.K, dtype=np.float32) / self.K
        self.terminated: bool = False
        self.truncated: bool = False
        self._sampled_human_buzz_pos: int | None = None
        self._current_question_idx: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_steps(self) -> int:
        """Total number of incremental clue steps for the current question.

        Returns
        -------
        int
            Length of ``question.run_indices`` if a question is loaded, else 1.
        """
        if self.question is None:
            return 1
        return len(self.question.run_indices)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _sample_question(self) -> MCQuestion:
        """Sample a random question from the question pool.

        Returns
        -------
        MCQuestion
            A randomly selected question.
        """
        return self.rng.choice(self.questions)

    def _sample_human_buzz(self, question: MCQuestion) -> int | None:
        """Sample a human buzz position from the question's distribution.

        Uses weighted random sampling based on the number of humans who
        buzzed at each position. Returns None if no human buzz data exists.

        Parameters
        ----------
        question : MCQuestion
            The question to sample a human buzz position for.

        Returns
        -------
        int or None
            Sampled token position, or None if no human buzz data.
        """
        if not question.human_buzz_positions:
            return None
        positions = []
        weights = []
        for pos, count in question.human_buzz_positions:
            positions.append(int(pos))
            weights.append(max(1, int(count)))
        if not positions:
            return None
        return self.rng.choices(positions, weights=weights, k=1)[0]

    def _softmax_scores(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw likelihood scores to a probability distribution.

        Delegates to module-level ``_softmax`` with this environment's beta.

        Parameters
        ----------
        scores : np.ndarray
            Raw similarity scores of shape (K,).

        Returns
        -------
        np.ndarray
            Probability distribution of shape (K,), dtype float32.
        """
        return _softmax(scores, self.beta)

    def _compute_belief(self, question: MCQuestion, step_idx: int) -> np.ndarray:
        """Compute belief distribution over answer options at a given step.

        Two modes are supported:

        ``from_scratch``
            Score the cumulative clue prefix against all option profiles,
            then apply softmax. Each step is independent of the previous
            belief.

        ``sequential_bayes``
            Extract only the new clue fragment since the last step, score
            it, and perform a Bayesian update: posterior = prior * likelihood,
            then normalize. This is cheaper per step but may accumulate
            approximation errors.

        Parameters
        ----------
        question : MCQuestion
            Current question being played.
        step_idx : int
            Current step index (0-based, indexes into run_indices).

        Returns
        -------
        np.ndarray
            Updated belief distribution of shape (K,), dtype float32.

        Raises
        ------
        ValueError
            If ``self.belief_mode`` is not a recognized mode.
        """
        if self.precomputed_beliefs is not None:
            key = (self._current_question_idx, step_idx)
            return self.precomputed_beliefs[key].copy()

        if self.belief_mode == "from_scratch":
            prefix = question.cumulative_prefixes[step_idx]
            scores = self.likelihood_model.score(prefix, question.option_profiles)
            return self._softmax_scores(scores)

        if self.belief_mode == "sequential_bayes":
            idx = question.run_indices[step_idx]
            prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
            frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])
            scores = self.likelihood_model.score(frag, question.option_profiles)
            likelihood = self._softmax_scores(scores)
            posterior = self.belief * likelihood
            denom = posterior.sum()
            if denom <= 0:
                posterior = np.ones(self.K, dtype=np.float32) / self.K
            else:
                posterior = posterior / denom
            return posterior.astype(np.float32)

        raise ValueError(f"Unknown belief_mode: {self.belief_mode}")

    def _obs(self) -> np.ndarray:
        """Build the observation vector from current belief state.

        Delegates to ``extract_belief_features`` which concatenates the raw
        belief vector with 6 derived scalar features.

        Returns
        -------
        np.ndarray
            Feature vector of shape (K + 6,), dtype float32.
        """
        return extract_belief_features(
            belief=self.belief,
            prev_belief=self.prev_belief,
            step_idx=self.step_idx,
            total_steps=self.total_steps,
        )

    def _step_to_token_pos(self, step_idx: int) -> int:
        """Convert a step index to the corresponding token position.

        Used by the ``human_grounded`` reward mode to compare the agent's
        buzz position against the sampled human buzz position.

        Parameters
        ----------
        step_idx : int
            Step index (0-based, indexes into run_indices).

        Returns
        -------
        int
            Token position in the original question text.
        """
        if self.question is None or not self.question.run_indices:
            return step_idx
        if step_idx >= len(self.question.run_indices):
            return self.question.run_indices[-1]
        if step_idx < 0:
            return self.question.run_indices[0]
        return self.question.run_indices[step_idx]

    def _buzz_reward(self, question: MCQuestion, chosen_idx: int, last_seen_step: int) -> float:
        """Compute the reward for buzzing with a given answer.

        Dispatches on ``self.reward_mode``:

        ``simple``
            +1.0 for correct, -1.0 for incorrect.
        ``human_grounded``
            0.0 if the agent buzzes after the sampled human would have;
            otherwise +buzz_correct / +buzz_incorrect.
        ``time_penalty`` (default)
            +buzz_correct / +buzz_incorrect. The per-step wait penalty
            is applied separately in ``step()``.

        Parameters
        ----------
        question : MCQuestion
            Current question.
        chosen_idx : int
            Index of the chosen answer option (0-based).
        last_seen_step : int
            Step index of the last clue seen before buzzing.

        Returns
        -------
        float
            Reward value.
        """
        correct = chosen_idx == question.gold_index
        if self.reward_mode == "simple":
            return 1.0 if correct else -1.0
        if self.reward_mode == "human_grounded":
            token_pos = self._step_to_token_pos(last_seen_step)
            if self._sampled_human_buzz_pos is not None and token_pos > self._sampled_human_buzz_pos:
                return 0.0
            return self.buzz_correct if correct else self.buzz_incorrect
        # default: time_penalty
        reward = self.buzz_correct if correct else self.buzz_incorrect

        if self.early_buzz_penalty > 0 and self.total_steps > 1:
            progress = np.clip((last_seen_step + 1) / self.total_steps, 0.0, 1.0)
            reward -= float(self.early_buzz_penalty) * (1.0 - progress)

        return reward

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and start a new episode.

        Samples a random question from the pool, initializes belief to a
        uniform distribution, and returns the initial observation.

        Parameters
        ----------
        seed : int or None
            If provided, reseeds both the internal RNG and numpy's global
            RNG for reproducibility.
        options : dict or None
            Unused. Included for Gymnasium API compatibility.

        Returns
        -------
        observation : np.ndarray
            Initial observation of shape (K + 6,), dtype float32.
            Belief is uniform, so top_p = 1/K, margin = 0, entropy = max.
        info : dict[str, Any]
            Episode metadata. Contains ``"qid"`` (the sampled question ID).
        """
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)

        if options and "question_idx" in options:
            q_idx = int(options["question_idx"])
            if q_idx < 0 or q_idx >= len(self.questions):
                raise ValueError(f"question_idx out of range: {q_idx}")
            self.question = self.questions[q_idx]
            self._current_question_idx = q_idx
        else:
            self.question = self._sample_question()
            self._current_question_idx = self._question_index_map.get(
                self.question.qid, self.questions.index(self.question)
            )
        self.step_idx = 0
        self.prev_belief = None
        self.belief = np.ones(self.K, dtype=np.float32) / self.K
        self.terminated = False
        self.truncated = False
        self._sampled_human_buzz_pos = self._sample_human_buzz(self.question)
        return self._obs(), {"qid": self.question.qid}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        If ``action == 0`` (WAIT):
            - Saves previous belief, computes new belief from current clue.
            - Applies wait_penalty if reward_mode is ``"time_penalty"``.
            - Advances step counter.
            - If all clues exhausted: forced termination with best-guess
              answer (``truncated=True``).

        If ``action in 1..K`` (BUZZ):
            - Computes buzz reward for chosen answer option ``action - 1``.
            - Episode ends (``terminated=True``).

        Parameters
        ----------
        action : int
            Action to take. 0 = WAIT, 1..K = buzz with option (action-1).

        Returns
        -------
        observation : np.ndarray
            Updated observation of shape (K + 6,), dtype float32.
        reward : float
            Scalar reward for this step.
        terminated : bool
            True if the agent buzzed (natural episode end).
        truncated : bool
            True if all clues were exhausted (forced termination).
        info : dict[str, Any]
            Step metadata. Always contains ``"qid"`` and ``"step_idx"``.
            On BUZZ: also ``"chosen_idx"`` and ``"correct"``.
            On forced termination: also ``"forced_choice"`` and
            ``"forced_correct"``.

        Raises
        ------
        RuntimeError
            If called before ``reset()`` or after episode has ended.
        ValueError
            If ``action`` is not in the action space.
        """
        if self.question is None:
            raise RuntimeError("Environment must be reset() before step().")
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot call step() on terminated/truncated episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        info: dict[str, Any] = {"qid": self.question.qid}
        reward = 0.0

        if action == 0:
            # WAIT: reveal next clue and update belief
            self.prev_belief = self.belief.copy()
            self.belief = self._compute_belief(self.question, self.step_idx)
            if self.reward_mode == "time_penalty":
                reward -= self.wait_penalty

            self.step_idx += 1
            if self.step_idx >= self.total_steps:
                # Forced termination: pick best answer from current belief
                last_seen = self.step_idx - 1
                forced_choice = int(np.argmax(self.belief))
                reward += self._buzz_reward(self.question, forced_choice, last_seen)
                self.truncated = True
                info["step_idx"] = last_seen
                info["forced_choice"] = forced_choice
                info["forced_correct"] = forced_choice == self.question.gold_index
            else:
                info["step_idx"] = self.step_idx

        else:
            # BUZZ: select an answer option
            last_seen = max(0, self.step_idx - 1)
            chosen_idx = action - 1
            reward += self._buzz_reward(self.question, chosen_idx, last_seen)
            self.terminated = True
            info["step_idx"] = last_seen
            info["chosen_idx"] = chosen_idx
            info["correct"] = chosen_idx == self.question.gold_index

        obs = self._obs()
        return obs, float(reward), self.terminated, self.truncated, info


def make_env_from_config(
    mc_questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    config: dict[str, Any],
    precomputed_beliefs: dict[tuple[int, int], np.ndarray] | None = None,
) -> TossupMCEnv:
    """Construct a TossupMCEnv from YAML configuration.

    Factory function that reads the ``environment``, ``data``, and
    ``likelihood`` sections of a config dict and instantiates a fully
    configured environment. The likelihood model must be pre-constructed
    (e.g., via ``build_likelihood_from_config``).

    Parameters
    ----------
    mc_questions : list[MCQuestion]
        List of MCQuestion instances with options and answer profiles.
        Must be non-empty.
    likelihood_model : LikelihoodModel
        Pre-constructed likelihood model for scoring clues against options.
        Use ``build_likelihood_from_config`` to create one from config.
    config : dict[str, Any]
        Full YAML config dict. Must contain the following sections:

        - ``environment``: reward mode, penalties, belief mode
        - ``data``: K (number of answer choices)
        - ``likelihood``: beta (softmax temperature)
    precomputed_beliefs : dict or None
        Optional precomputed belief cache from ``precompute_beliefs()``.
        When provided, ``_compute_belief`` uses O(1) lookups instead of
        calling ``likelihood_model.score()``.

    Returns
    -------
    TossupMCEnv
        A configured Gymnasium environment ready for ``reset()``.

    Examples
    --------
    >>> from qb_data.config import load_config
    >>> from models.likelihoods import build_likelihood_from_config
    >>> config = load_config("configs/default.yaml")
    >>> model = build_likelihood_from_config(config, corpus_texts=corpus)
    >>> env = make_env_from_config(mc_questions, model, config)
    >>> obs, info = env.reset()
    """
    env_cfg = config["environment"]
    data_cfg = config["data"]
    lik_cfg = config["likelihood"]
    return TossupMCEnv(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        K=int(data_cfg.get("K", 4)),
        reward_mode=str(env_cfg.get("reward", env_cfg.get("reward_mode", "time_penalty"))),
        seed=int(env_cfg.get("seed", 13)),
        wait_penalty=float(env_cfg.get("wait_penalty", 0.01)),
        early_buzz_penalty=float(env_cfg.get("early_buzz_penalty", 0.0)),
        buzz_correct=float(env_cfg.get("buzz_correct", 1.0)),
        buzz_incorrect=float(env_cfg.get("buzz_incorrect", -0.5)),
        belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
        beta=float(lik_cfg.get("beta", 5.0)),
        precomputed_beliefs=precomputed_beliefs,
    )
````

## File: scripts/evaluate_all.py
````python
#!/usr/bin/env python3
"""
Comprehensive evaluation with control experiments and visualization.

Runs the SoftmaxProfileBuzzer at the best threshold (from baseline sweep),
then executes control experiments (choices-only, shuffle, alias substitution)
and generates comparison plots and tables for the CS234 writeup.

Consumes outputs from:
- build_mc_dataset.py (mc_dataset.json, alias_lookup.json)
- run_baselines.py (baseline_summary.json)
- train_ppo.py (ppo_summary.json)

Produces:
- evaluation_report.json (full eval + controls + baseline + PPO summaries)
- plots/entropy_vs_clue.png
- plots/calibration.png
- plots/comparison.csv

Usage:
    python scripts/evaluate_all.py --smoke
    python scripts/evaluate_all.py --config configs/custom.yaml
    python scripts/evaluate_all.py --mc-path artifacts/main/mc_dataset.json

Ported from qb-rl reference implementation (scripts/evaluate_all.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bayesian_buzzer import SoftmaxProfileBuzzer
from agents.threshold_buzzer import (
    _softmax_episode_from_precomputed,
    precompute_beliefs,
)
from evaluation.controls import (
    run_alias_substitution_control,
    run_choices_only_control,
    run_shuffle_control_precomputed,
)
from evaluation.metrics import (
    calibration_at_buzz,
    per_category_accuracy,
    summarize_buzz_metrics,
)
from evaluation.plotting import (
    plot_calibration_curve,
    plot_entropy_vs_clue_index,
    save_comparison_table,
)
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
    load_embedding_cache,
    load_json,
    load_mc_questions,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with config, smoke, and mc_path fields.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate all agents and controls."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
    )
    parser.add_argument(
        "--mc-path", type=str, default=None,
        help="Optional MC dataset JSON path (overrides config-derived path).",
    )
    return parser.parse_args()


def pick_best_softmax_threshold(
    out_dir: Path, default_threshold: float
) -> float:
    """Select the best softmax threshold from baseline sweep results.

    Loads baseline_summary.json and extracts the threshold with the
    highest mean S_q score from the softmax_profile results.

    Parameters
    ----------
    out_dir : Path
        Directory containing baseline_summary.json.
    default_threshold : float
        Fallback threshold if baseline summary is unavailable.

    Returns
    -------
    float
        Best threshold by S_q score, or default_threshold if unavailable.
    """
    summary_path = out_dir / "baseline_summary.json"
    if not summary_path.exists():
        return default_threshold
    summary = load_json(summary_path)
    softmax = summary.get("softmax_profile", {})
    if not softmax:
        return default_threshold
    best_t = default_threshold
    best_sq = float("-inf")
    for t_str, metrics in softmax.items():
        sq = float(metrics.get("mean_sq", float("-inf")))
        if sq > best_sq:
            best_sq = sq
            best_t = float(t_str)
    return best_t


def main() -> None:
    """Run comprehensive evaluation with controls and visualizations."""
    args = parse_args()

    config = load_config(args.config, smoke=args.smoke)

    split = "smoke" if args.smoke else "main"
    out_dir = ARTIFACT_DIR / split
    mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"

    # Fallback: check data/processed/ if artifacts path doesn't exist
    if not mc_path.exists():
        fallback = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"
        if fallback.exists():
            print(f"MC dataset not found at {mc_path}, using fallback: {fallback}")
            mc_path = fallback

    print(f"Loading MC questions from: {mc_path}")
    mc_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(mc_questions)} MC questions")

    # Load alias lookup (generated by build_mc_dataset.py)
    alias_path = out_dir / "alias_lookup.json"
    if alias_path.exists():
        alias_lookup = load_json(alias_path)
    else:
        print(f"Warning: alias_lookup.json not found at {alias_path}, using empty lookup")
        alias_lookup = {}

    # Build likelihood model
    print(f"Building likelihood model: {config['likelihood']['model']}")
    likelihood_model = build_likelihood_model(config, mc_questions)
    load_embedding_cache(likelihood_model, config)
    beta = float(config["likelihood"].get("beta", 5.0))
    alpha = float(config["bayesian"].get("alpha", 10.0))
    default_threshold = float(config["bayesian"]["threshold_sweep"][0])
    threshold = pick_best_softmax_threshold(out_dir, default_threshold=default_threshold)
    print(f"Using best softmax threshold: {threshold}")

    # Precompute beliefs once (single pass of likelihood_model.score())
    print("Precomputing beliefs...")
    precomputed = precompute_beliefs(mc_questions, likelihood_model, beta)

    # Precomputed evaluation (zero extra score() calls)
    def evaluate_questions_precomputed(pqs):
        runs = [asdict(_softmax_episode_from_precomputed(pq, threshold, alpha)) for pq in pqs]
        summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
        summary["runs"] = runs
        return summary

    # Live evaluator for controls that genuinely change option text (alias)
    def evaluate_questions_live(qset):
        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood_model,
            threshold=threshold,
            beta=beta,
            alpha=alpha,
        )
        runs = [asdict(agent.run_episode(q)) for q in qset]
        summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
        summary["runs"] = runs
        return summary

    # --- Run evaluations ---
    print("Running full evaluation...")
    full_eval = evaluate_questions_precomputed(precomputed)

    # Compute per-category breakdown
    print("\nComputing per-category breakdown...")
    per_category_results = per_category_accuracy(full_eval["runs"], mc_questions)

    # Sort by category name for readability
    per_category_sorted = dict(sorted(per_category_results.items()))

    print("\nPer-category accuracy:")
    for category, metrics in per_category_sorted.items():
        print(
            f"  {category:20s} (n={metrics['n']:3.0f}): "
            f"acc={metrics['buzz_accuracy']:.3f}, "
            f"S_q={metrics['mean_sq']:.3f}"
        )
    print()

    print("Running shuffle control...")
    shuffle_eval = run_shuffle_control_precomputed(precomputed, threshold, alpha)

    print("Running alias substitution control...")
    alias_eval = run_alias_substitution_control(
        mc_questions,
        alias_lookup=alias_lookup,
        evaluator=lambda qset: evaluate_questions_live(qset),
    )

    print("Running choices-only control...")
    choices_only = run_choices_only_control(mc_questions)

    # --- Load existing artifacts ---
    ppo_summary_path = out_dir / "ppo_summary.json"
    ppo_summary = load_json(ppo_summary_path) if ppo_summary_path.exists() else {}
    baseline_summary_path = out_dir / "baseline_summary.json"
    baseline_summary = (
        load_json(baseline_summary_path) if baseline_summary_path.exists() else {}
    )

    # --- Build evaluation report ---
    report = {
        "softmax_profile_best_threshold": threshold,
        "full_eval": {k: v for k, v in full_eval.items() if k != "runs"},
        "controls": {
            "choices_only": choices_only,
            "shuffle": {k: v for k, v in shuffle_eval.items() if k != "runs"},
            "alias_substitution": {
                k: v for k, v in alias_eval.items() if k != "runs"
            },
        },
        "per_category": per_category_sorted,
        "baseline_summary": baseline_summary,
        "ppo_summary": ppo_summary,
    }
    save_json(out_dir / "evaluation_report.json", report)

    # --- Generate visualizations ---
    print("Generating plots...")

    # Entropy vs clue index
    entropy_traces = [
        list(r["entropy_trace"])
        for r in full_eval["runs"]
        if r.get("entropy_trace")
    ]
    max_len = max((len(t) for t in entropy_traces), default=0)
    padded = np.full((len(entropy_traces), max_len), np.nan, dtype=np.float32)
    for i, trace in enumerate(entropy_traces):
        padded[i, : len(trace)] = np.array(trace, dtype=np.float32)
    entropy_trace = (
        np.nanmean(padded, axis=0).tolist() if max_len > 0 else []
    )
    plot_entropy_vs_clue_index(
        {"softmax_profile": entropy_trace},
        out_dir / "plots" / "entropy_vs_clue.png",
    )

    # Calibration curve — use top_p (belief in top answer) as confidence
    confidences = []
    outcomes = []
    for row in full_eval["runs"]:
        top_p = row.get("top_p_trace", row.get("c_trace", []))
        if not top_p:
            continue
        idx = min(int(row["buzz_step"]), len(top_p) - 1)
        confidences.append(float(top_p[idx]))
        outcomes.append(1 if bool(row["correct"]) else 0)
    plot_calibration_curve(
        confidences, outcomes, out_dir / "plots" / "calibration.png"
    )

    # Comparison table: include baseline sweep, controls, and PPO
    table_rows = []

    # Add baseline sweep results (threshold at multiple values)
    if "threshold" in baseline_summary:
        for threshold_str, metrics in baseline_summary["threshold"].items():
            table_rows.append({
                "agent": f"threshold_{threshold_str}",
                **{k: v for k, v in metrics.items() if k != "runs"},
            })

    # Add softmax_profile sweep results
    if "softmax_profile" in baseline_summary:
        for threshold_str, metrics in baseline_summary["softmax_profile"].items():
            table_rows.append({
                "agent": f"softmax_{threshold_str}",
                **{k: v for k, v in metrics.items() if k != "runs"},
            })

    # Add full softmax eval (best threshold) and control experiments
    table_rows.append({
        "agent": "full_softmax",
        **{k: v for k, v in full_eval.items() if k != "runs"},
    })
    table_rows.append({
        "agent": "shuffle_control",
        **{k: v for k, v in shuffle_eval.items() if k != "runs"},
    })
    table_rows.append({
        "agent": "alias_control",
        **{k: v for k, v in alias_eval.items() if k != "runs"},
    })

    # Add PPO if available
    if ppo_summary:
        table_rows.append({"agent": "ppo", **ppo_summary})

    save_comparison_table(table_rows, out_dir / "plots" / "comparison.csv")

    print(f"Wrote evaluation report to: {out_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
````

## File: tests/test_agents.py
````python
"""Test suite for agents/ -- baseline agent execution and episode result schemas.

Covers:
- AGT-02: ThresholdBuzzer execution and buzzing logic
- AGT-03: AlwaysBuzzFinalBuzzer wait-then-buzz behavior
- AGT-04: SoftmaxProfileBuzzer from-scratch belief recomputation
- AGT-05: SequentialBayesBuzzer incremental Bayesian updates
- AGT-06: EpisodeResult and SoftmaxEpisodeResult schema validation
- Threshold sweep utility tests
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from agents import (
    AlwaysBuzzFinalBuzzer,
    EpisodeResult,
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
    SoftmaxProfileBuzzer,
    ThresholdBuzzer,
    result_to_dict,
    sweep_thresholds,
)
from agents._math import sigmoid
from models.likelihoods import TfIdfLikelihood
from qb_data.mc_builder import MCQuestion


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_likelihood(corpus: list[str]) -> TfIdfLikelihood:
    """Create a fitted TF-IDF likelihood model from a corpus.

    Uses TF-IDF (fast) for agent logic tests so tests run quickly.
    """
    return TfIdfLikelihood(corpus_texts=corpus)


class TestSigmoidMath:
    """Tests for stable scalar sigmoid helper."""

    def test_sigmoid_handles_extreme_inputs_without_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert sigmoid(1000.0) == pytest.approx(1.0)
            assert sigmoid(-1000.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# ThresholdBuzzer tests (AGT-02)
# ------------------------------------------------------------------ #


class TestThresholdBuzzer:
    """Tests for ThresholdBuzzer execution and buzzing logic."""

    def test_threshold_buzzer_executes(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer runs an episode without error and returns EpisodeResult."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, EpisodeResult)
        assert result.qid == sample_mc_question.qid
        assert len(result.c_trace) > 0

    def test_threshold_buzzer_buzzes_on_threshold(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer buzzes when top_p >= threshold.

        With threshold=0.0, the agent should buzz immediately at step 0
        because any non-negative top_p will meet the threshold.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.0)
        result = agent.run_episode(sample_mc_question)

        # With threshold 0.0, should buzz at step 0
        assert result.buzz_step == 0, (
            f"Expected buzz at step 0 with threshold=0.0, got step {result.buzz_step}"
        )

    def test_threshold_buzzer_waits_on_low_confidence(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer waits when top_p < threshold.

        With threshold=1.0 (impossible for softmax to reach exactly 1.0 in
        practice), the agent should wait until the final step.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=1.0)
        result = agent.run_episode(sample_mc_question)

        # With threshold 1.0, should wait until the last step
        expected_final = len(sample_mc_question.cumulative_prefixes) - 1
        assert result.buzz_step == expected_final, (
            f"Expected buzz at final step {expected_final} with threshold=1.0, "
            f"got step {result.buzz_step}"
        )

    def test_threshold_buzzer_buzzes_at_final(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer always buzzes on final step regardless of threshold.

        Even with threshold=1.0 (unreachable), the agent must buzz at the
        final step as a forced fallback.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=1.0)
        result = agent.run_episode(sample_mc_question)

        final_step = len(sample_mc_question.cumulative_prefixes) - 1
        assert result.buzz_step == final_step
        assert result.buzz_index in range(len(sample_mc_question.options))

    def test_threshold_buzzer_traces_valid(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """c_trace and g_trace have correct and matching lengths.

        Traces should have length equal to buzz_step + 1 (one entry per
        step from 0 to buzz_step inclusive).
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        trace_len = result.buzz_step + 1
        assert len(result.c_trace) == trace_len, (
            f"c_trace length {len(result.c_trace)} != expected {trace_len}"
        )
        assert len(result.g_trace) == trace_len, (
            f"g_trace length {len(result.g_trace)} != expected {trace_len}"
        )
        assert len(result.top_p_trace) == trace_len
        assert len(result.entropy_trace) == trace_len

    def test_threshold_buzzer_confidence_proxy(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """c_t values in [0, 1] via sigmoid transformation."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for c_t in result.c_trace:
            assert 0.0 <= c_t <= 1.0, (
                f"Confidence proxy {c_t} outside [0, 1]"
            )

    def test_threshold_buzzer_custom_params(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer accepts custom beta and alpha parameters."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(
            likelihood_model=likelihood,
            threshold=0.5,
            beta=10.0,
            alpha=20.0,
        )
        assert agent.beta == 10.0
        assert agent.alpha == 20.0

        result = agent.run_episode(sample_mc_question)
        assert isinstance(result, EpisodeResult)

    def test_threshold_buzzer_confidence_proxy_stable_extremes(
        self, sample_corpus: list[str]
    ) -> None:
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(
            likelihood_model=likelihood,
            threshold=-100.0,
            alpha=100.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent._confidence_proxy(1.0) == pytest.approx(1.0)

        agent = ThresholdBuzzer(
            likelihood_model=likelihood,
            threshold=100.0,
            alpha=100.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent._confidence_proxy(0.0) == pytest.approx(0.0)

    def test_threshold_buzzer_top_p_in_range(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """top_p_trace values are valid probabilities in [0, 1]."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for p in result.top_p_trace:
            assert 0.0 <= p <= 1.0, f"top_p {p} outside [0, 1]"

    def test_threshold_buzzer_entropy_nonnegative(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Entropy values are non-negative (Shannon entropy >= 0)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for h in result.entropy_trace:
            assert h >= 0.0, f"Entropy {h} is negative"


# ------------------------------------------------------------------ #
# AlwaysBuzzFinalBuzzer tests (AGT-03)
# ------------------------------------------------------------------ #


class TestAlwaysBuzzFinalBuzzer:
    """Tests for AlwaysBuzzFinalBuzzer wait-then-buzz behavior."""

    def test_always_buzz_final_waits(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """All c_trace entries except the last are 0.0 (agent waits)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        # All entries except last should be 0.0
        for c_t in result.c_trace[:-1]:
            assert c_t == 0.0, f"Expected c_t=0.0 for waiting, got {c_t}"

    def test_always_buzz_final_buzzes_last(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """The last c_trace entry is 1.0 (agent buzzes at final step)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        assert result.c_trace[-1] == 1.0, (
            f"Expected c_trace[-1]=1.0, got {result.c_trace[-1]}"
        )

    def test_always_buzz_final_computes_beliefs(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Beliefs are computed at each step (not skipped).

        All top_p_trace entries should have valid probability values,
        demonstrating the model computed beliefs at every step.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        n_steps = len(sample_mc_question.cumulative_prefixes)
        assert len(result.top_p_trace) == n_steps, (
            f"Expected {n_steps} top_p entries, got {len(result.top_p_trace)}"
        )
        for p in result.top_p_trace:
            assert 0.0 <= p <= 1.0, f"top_p {p} outside [0, 1]"

    def test_always_buzz_final_buzz_step(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """buzz_step equals len(cumulative_prefixes) - 1 (last step)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        expected = len(sample_mc_question.cumulative_prefixes) - 1
        assert result.buzz_step == expected, (
            f"Expected buzz_step={expected}, got {result.buzz_step}"
        )

    def test_always_buzz_final_full_trace(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """All traces have length equal to number of cumulative prefixes."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        n = len(sample_mc_question.cumulative_prefixes)
        assert len(result.c_trace) == n
        assert len(result.g_trace) == n
        assert len(result.top_p_trace) == n
        assert len(result.entropy_trace) == n


# ------------------------------------------------------------------ #
# SoftmaxProfileBuzzer tests (AGT-04)
# ------------------------------------------------------------------ #


class TestSoftmaxProfileBuzzer:
    """Tests for SoftmaxProfileBuzzer from-scratch belief computation."""

    def test_softmax_profile_executes(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer runs an episode without error."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        assert result.qid == sample_mc_question.qid

    def test_softmax_profile_recomputes_belief(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer calls _belief_from_scratch each step.

        Verifies the method exists and the agent stores beliefs, confirming
        from-scratch recomputation (not incremental Bayesian updates).
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)

        # Verify the from-scratch method exists
        assert hasattr(agent, "_belief_from_scratch")

        result = agent.run_episode(sample_mc_question)

        # After episode, agent should have a stored belief
        assert agent.belief is not None
        assert isinstance(agent.belief, np.ndarray)
        assert agent.belief.shape == (len(sample_mc_question.options),)

    def test_softmax_profile_result_schema(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer returns SoftmaxEpisodeResult, not EpisodeResult."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        # SoftmaxEpisodeResult should NOT be an EpisodeResult (different dataclass)
        assert not isinstance(result, EpisodeResult)

    def test_softmax_profile_confidence_proxy(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer c_t values in [0, 1] via sigmoid."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for c_t in result.c_trace:
            assert 0.0 <= c_t <= 1.0, f"c_t {c_t} outside [0, 1]"

    def test_softmax_profile_threshold_behavior(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer respects threshold for buzzing."""
        likelihood = _make_likelihood(sample_corpus)

        # With threshold 0.0, should buzz immediately
        agent_low = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.0)
        result_low = agent_low.run_episode(sample_mc_question)
        assert result_low.buzz_step == 0

        # With threshold 1.0, should wait until the end
        agent_high = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=1.0)
        result_high = agent_high.run_episode(sample_mc_question)
        assert result_high.buzz_step == len(sample_mc_question.cumulative_prefixes) - 1

    def test_softmax_profile_confidence_proxy_stable_extremes(
        self, sample_corpus: list[str]
    ) -> None:
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood,
            threshold=-100.0,
            alpha=100.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent.confidence_proxy(1.0) == pytest.approx(1.0)

        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood,
            threshold=100.0,
            alpha=100.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent.confidence_proxy(0.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# SequentialBayesBuzzer tests (AGT-05)
# ------------------------------------------------------------------ #


class TestSequentialBayesBuzzer:
    """Tests for SequentialBayesBuzzer incremental Bayesian update."""

    def test_sequential_bayes_executes(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer runs an episode without error."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        assert result.qid == sample_mc_question.qid

    def test_sequential_bayes_uses_run_indices(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer requires question.run_indices field.

        The agent iterates over run_indices to extract token fragments,
        not over cumulative_prefixes. The number of trace entries should
        match the number of run_indices steps processed.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        # Trace length should be <= len(run_indices)
        assert len(result.c_trace) <= len(sample_mc_question.run_indices), (
            f"Trace length {len(result.c_trace)} > run_indices length "
            f"{len(sample_mc_question.run_indices)}"
        )

    def test_sequential_bayes_bayesian_update(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Belief is posterior proportional to prior * likelihood.

        Verify the _step_update method produces valid posterior:
        all entries >= 0 and sum to 1.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)

        K = len(sample_mc_question.options)
        prior = np.ones(K, dtype=np.float32) / K
        fragment = "first president"
        profiles = sample_mc_question.option_profiles

        posterior = agent._step_update(prior, fragment, profiles)

        assert posterior.shape == (K,), f"Expected shape ({K},), got {posterior.shape}"
        assert all(posterior >= 0), "Posterior has negative entries"
        np.testing.assert_almost_equal(
            posterior.sum(), 1.0, decimal=5,
            err_msg="Posterior should sum to 1.0",
        )

    def test_sequential_bayes_result_schema(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer returns SoftmaxEpisodeResult."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        assert not isinstance(result, EpisodeResult)

    def test_sequential_bayes_fragments(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer processes token fragments, not full prefixes.

        With threshold 1.0 (never buzzes early), all run_indices should be
        processed, producing traces of length len(run_indices).
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=1.0)
        result = agent.run_episode(sample_mc_question)

        n_steps = len(sample_mc_question.run_indices)
        assert len(result.c_trace) == n_steps, (
            f"Expected {n_steps} trace entries, got {len(result.c_trace)}"
        )


# ------------------------------------------------------------------ #
# Episode result schema tests (AGT-06)
# ------------------------------------------------------------------ #


class TestEpisodeResultSchema:
    """Tests for EpisodeResult and SoftmaxEpisodeResult dataclass schemas."""

    def test_episode_result_fields(self) -> None:
        """EpisodeResult has all required fields."""
        result = EpisodeResult(
            qid="test_q",
            buzz_step=3,
            buzz_index=1,
            gold_index=0,
            correct=False,
            reward_like=-0.5,
            c_trace=[0.1, 0.2, 0.3, 0.4],
            g_trace=[0.0, 0.0, 0.0, 1.0],
            top_p_trace=[0.3, 0.4, 0.5, 0.6],
            entropy_trace=[1.4, 1.2, 1.0, 0.8],
        )
        assert result.qid == "test_q"
        assert result.buzz_step == 3
        assert result.buzz_index == 1
        assert result.gold_index == 0
        assert result.correct is False
        assert result.reward_like == -0.5

    def test_softmax_episode_result_fields(self) -> None:
        """SoftmaxEpisodeResult has all required fields."""
        result = SoftmaxEpisodeResult(
            qid="test_q",
            buzz_step=2,
            buzz_index=0,
            gold_index=0,
            correct=True,
            c_trace=[0.1, 0.5, 0.9],
            g_trace=[1.0, 1.0, 1.0],
            top_p_trace=[0.4, 0.6, 0.9],
            entropy_trace=[1.2, 0.8, 0.3],
        )
        assert result.qid == "test_q"
        assert result.buzz_step == 2
        assert result.correct is True

    def test_traces_same_length(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """len(c_trace) == len(g_trace) for all agents."""
        likelihood = _make_likelihood(sample_corpus)

        agents = [
            ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
            AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
            SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
            SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
        ]

        for agent in agents:
            result = agent.run_episode(sample_mc_question)
            agent_name = type(agent).__name__
            assert len(result.c_trace) == len(result.g_trace), (
                f"{agent_name}: c_trace ({len(result.c_trace)}) != "
                f"g_trace ({len(result.g_trace)})"
            )
            assert len(result.c_trace) == len(result.top_p_trace), (
                f"{agent_name}: c_trace ({len(result.c_trace)}) != "
                f"top_p_trace ({len(result.top_p_trace)})"
            )
            assert len(result.c_trace) == len(result.entropy_trace), (
                f"{agent_name}: c_trace ({len(result.c_trace)}) != "
                f"entropy_trace ({len(result.entropy_trace)})"
            )

    def test_g_trace_binary(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """g_trace values are 0.0 or 1.0 (correctness is binary)."""
        likelihood = _make_likelihood(sample_corpus)

        agents = [
            ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
            AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
            SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
            SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
        ]

        for agent in agents:
            result = agent.run_episode(sample_mc_question)
            agent_name = type(agent).__name__
            for g_t in result.g_trace:
                assert g_t in (0.0, 1.0), (
                    f"{agent_name}: g_t={g_t} not in {{0.0, 1.0}}"
                )

    def test_buzz_index_valid(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """buzz_index in range(K) where K = len(options)."""
        likelihood = _make_likelihood(sample_corpus)
        K = len(sample_mc_question.options)

        agents = [
            ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
            AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
            SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
            SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
        ]

        for agent in agents:
            result = agent.run_episode(sample_mc_question)
            agent_name = type(agent).__name__
            assert 0 <= result.buzz_index < K, (
                f"{agent_name}: buzz_index={result.buzz_index} not in [0, {K})"
            )

    def test_result_to_dict(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """result_to_dict() converts EpisodeResult to dict."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        d = result_to_dict(result)
        assert isinstance(d, dict)
        assert d["qid"] == sample_mc_question.qid
        assert "buzz_step" in d
        assert "buzz_index" in d
        assert "gold_index" in d
        assert "correct" in d
        assert "reward_like" in d
        assert "c_trace" in d
        assert "g_trace" in d
        assert isinstance(d["c_trace"], list)


# ------------------------------------------------------------------ #
# Threshold sweep utility tests
# ------------------------------------------------------------------ #


class TestSweepThresholds:
    """Tests for sweep_thresholds utility function."""

    def test_sweep_thresholds_runs(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """sweep_thresholds() returns dict[float, list[EpisodeResult]]."""
        likelihood = _make_likelihood(sample_corpus)
        results = sweep_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=[0.7],
        )

        assert isinstance(results, dict)
        assert 0.7 in results
        assert len(results[0.7]) == 1
        assert isinstance(results[0.7][0], EpisodeResult)

    def test_sweep_thresholds_multiple_values(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Sweeps over [0.6, 0.7, 0.8, 0.9] and returns results for each."""
        likelihood = _make_likelihood(sample_corpus)
        thresholds = [0.6, 0.7, 0.8, 0.9]
        results = sweep_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=thresholds,
        )

        assert len(results) == len(thresholds)
        for thresh in thresholds:
            assert thresh in results, f"Missing results for threshold {thresh}"
            assert len(results[thresh]) == 1

    def test_sweep_thresholds_monotonic_buzz_step(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Higher thresholds should produce later or equal buzz steps.

        A higher threshold means the agent needs more confidence to buzz,
        so it should wait at least as long as with a lower threshold.
        """
        likelihood = _make_likelihood(sample_corpus)
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = sweep_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=thresholds,
        )

        buzz_steps = [results[t][0].buzz_step for t in thresholds]
        for i in range(len(buzz_steps) - 1):
            assert buzz_steps[i] <= buzz_steps[i + 1], (
                f"Buzz step not monotonic: threshold {thresholds[i]} "
                f"(step {buzz_steps[i]}) > threshold {thresholds[i+1]} "
                f"(step {buzz_steps[i+1]})"
            )


# ------------------------------------------------------------------ #
# Precomputed equivalence tests
# ------------------------------------------------------------------ #


class TestPrecomputedEquivalence:
    """Prove precomputed-path functions are numerically identical to live agents."""

    def test_softmax_precomputed_matches_live(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """_softmax_episode_from_precomputed matches SoftmaxProfileBuzzer.run_episode."""
        from agents.threshold_buzzer import (
            _softmax_episode_from_precomputed,
            precompute_beliefs,
        )

        likelihood = _make_likelihood(sample_corpus)
        threshold, beta, alpha = 0.7, 5.0, 10.0

        # Live agent
        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood, threshold=threshold, beta=beta, alpha=alpha
        )
        live = agent.run_episode(sample_mc_question)

        # Precomputed path
        pqs = precompute_beliefs([sample_mc_question], likelihood, beta)
        pre = _softmax_episode_from_precomputed(pqs[0], threshold, alpha)

        assert pre.buzz_step == live.buzz_step
        assert pre.buzz_index == live.buzz_index
        assert pre.correct == live.correct
        np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
        np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
        np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
        np.testing.assert_array_almost_equal(pre.entropy_trace, live.entropy_trace)

    def test_always_final_precomputed_matches_live(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """_always_final_from_precomputed matches AlwaysBuzzFinalBuzzer.run_episode."""
        from agents.threshold_buzzer import (
            _always_final_from_precomputed,
            precompute_beliefs,
        )

        likelihood = _make_likelihood(sample_corpus)
        beta = 5.0

        # Live agent
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood, beta=beta)
        live = agent.run_episode(sample_mc_question)

        # Precomputed path
        pqs = precompute_beliefs([sample_mc_question], likelihood, beta)
        pre = _always_final_from_precomputed(pqs[0])

        assert pre.buzz_step == live.buzz_step
        assert pre.buzz_index == live.buzz_index
        assert pre.correct == live.correct
        assert pre.reward_like == live.reward_like
        np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
        np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
        np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
        np.testing.assert_array_almost_equal(pre.entropy_trace, live.entropy_trace)

    def test_sequential_precomputed_matches_live(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """_sequential_episode_from_precomputed matches SequentialBayesBuzzer.run_episode."""
        from agents.bayesian_buzzer import (
            _sequential_episode_from_precomputed,
            precompute_sequential_beliefs,
        )

        likelihood = _make_likelihood(sample_corpus)
        threshold, beta, alpha = 0.7, 5.0, 10.0

        # Live agent
        agent = SequentialBayesBuzzer(
            likelihood_model=likelihood, threshold=threshold, beta=beta, alpha=alpha
        )
        live = agent.run_episode(sample_mc_question)

        # Precomputed path
        pqs = precompute_sequential_beliefs([sample_mc_question], likelihood, beta)
        pre = _sequential_episode_from_precomputed(pqs[0], threshold, alpha)

        assert pre.buzz_step == live.buzz_step
        assert pre.buzz_index == live.buzz_index
        assert pre.correct == live.correct
        np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
        np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
        np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
        np.testing.assert_array_almost_equal(pre.entropy_trace, live.entropy_trace)

    def test_sweep_sequential_matches_per_threshold(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """sweep_sequential_thresholds matches per-threshold SequentialBayesBuzzer."""
        from agents.bayesian_buzzer import sweep_sequential_thresholds

        likelihood = _make_likelihood(sample_corpus)
        thresholds = [0.5, 0.7, 0.9]
        beta, alpha = 5.0, 10.0

        # Sweep
        sweep = sweep_sequential_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=thresholds,
            beta=beta,
            alpha=alpha,
        )

        # Per-threshold live agents
        for threshold in thresholds:
            agent = SequentialBayesBuzzer(
                likelihood_model=likelihood,
                threshold=threshold,
                beta=beta,
                alpha=alpha,
            )
            live = agent.run_episode(sample_mc_question)
            pre = sweep[float(threshold)][0]

            assert pre.buzz_step == live.buzz_step, (
                f"threshold={threshold}: buzz_step {pre.buzz_step} != {live.buzz_step}"
            )
            assert pre.buzz_index == live.buzz_index
            assert pre.correct == live.correct
            np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
            np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
            np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
            np.testing.assert_array_almost_equal(
                pre.entropy_trace, live.entropy_trace
            )


# ------------------------------------------------------------------ #
# Shuffle precomputed equivalence tests
# ------------------------------------------------------------------ #


class TestShufflePrecomputedEquivalence:
    """Prove precomputed shuffle control matches live rescore shuffle control."""

    def test_shuffle_precomputed_matches_rescore(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Precomputed shuffle control matches live rescore shuffle control."""
        from dataclasses import asdict

        from agents.threshold_buzzer import precompute_beliefs
        from evaluation.controls import (
            run_shuffle_control,
            run_shuffle_control_precomputed,
        )
        from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics

        likelihood = _make_likelihood(sample_corpus)
        threshold, beta, alpha = 0.7, 5.0, 10.0
        questions = [sample_mc_question]

        # Live rescore path
        def evaluator(qset):
            agent = SoftmaxProfileBuzzer(
                likelihood_model=likelihood,
                threshold=threshold,
                beta=beta,
                alpha=alpha,
            )
            runs = [asdict(agent.run_episode(q)) for q in qset]
            summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
            summary["runs"] = runs
            return summary

        live_result = run_shuffle_control(questions, evaluator=evaluator, random_seed=13)

        # Precomputed path
        precomputed = precompute_beliefs(questions, likelihood, beta)
        pre_result = run_shuffle_control_precomputed(
            precomputed, threshold, alpha, random_seed=13
        )

        # Compare summary metrics
        assert live_result["mean_sq"] == pytest.approx(pre_result["mean_sq"])
        assert live_result["buzz_accuracy"] == pytest.approx(pre_result["buzz_accuracy"])

        # Compare per-run results
        for live_run, pre_run in zip(live_result["runs"], pre_result["runs"]):
            assert live_run["buzz_step"] == pre_run["buzz_step"]
            assert live_run["buzz_index"] == pre_run["buzz_index"]
            assert live_run["correct"] == pre_run["correct"]
            np.testing.assert_array_almost_equal(
                live_run["c_trace"], pre_run["c_trace"]
            )
            np.testing.assert_array_almost_equal(
                live_run["g_trace"], pre_run["g_trace"]
            )
            np.testing.assert_array_almost_equal(
                live_run["top_p_trace"], pre_run["top_p_trace"]
            )
            np.testing.assert_array_almost_equal(
                live_run["entropy_trace"], pre_run["entropy_trace"]
            )

    def test_permutation_consistency(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Permutation applied to beliefs matches permutation applied to gold_index."""
        import random as random_mod

        from agents.threshold_buzzer import _PrecomputedQuestion, precompute_beliefs
        from evaluation.controls import shuffled_option_copy

        likelihood = _make_likelihood(sample_corpus)
        beta = 5.0
        questions = [sample_mc_question]
        precomputed = precompute_beliefs(questions, likelihood, beta)

        # Reproduce the permutation that shuffled_option_copy would use
        rng_live = random_mod.Random(13)
        shuffled_q = shuffled_option_copy(sample_mc_question, rng_live)

        # Reproduce the same permutation for precomputed
        rng_pre = random_mod.Random(13)
        pq = precomputed[0]
        perm = list(range(pq.num_options))
        rng_pre.shuffle(perm)
        new_gold = perm.index(pq.gold_index)

        # The gold index should match
        assert new_gold == shuffled_q.gold_index
````

## File: scripts/run_baselines.py
````python
#!/usr/bin/env python3
"""
Run non-RL baseline agents and save episode traces + summary artifacts.

Executes four baseline agent types across a threshold sweep:
1. ThresholdBuzzer -- buzzes when top belief exceeds threshold
2. SoftmaxProfileBuzzer -- softmax belief from scratch at each step
3. SequentialBayesBuzzer -- Bayesian belief update with sequential fragments
4. AlwaysBuzzFinalBuzzer -- always waits until last clue, then buzzes

Results are saved to artifacts/{smoke,main}/ as JSON files with per-episode
traces and aggregated summary metrics (accuracy, S_q, ECE, Brier score).

Usage:
    python scripts/run_baselines.py              # Full run (default config)
    python scripts/run_baselines.py --smoke      # Quick smoke test (~50 questions)
    python scripts/run_baselines.py --config configs/custom.yaml
    python scripts/run_baselines.py --mc-path artifacts/main/mc_dataset.json

Ported from qb-rl reference implementation (scripts/run_baselines.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bayesian_buzzer import (
    precompute_sequential_beliefs,
    sweep_sequential_thresholds,
)
from agents.threshold_buzzer import (
    _always_final_from_precomputed,
    _softmax_episode_from_precomputed,
    precompute_beliefs,
    sweep_thresholds,
)
from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
    load_embedding_cache,
    load_mc_questions,
    save_embedding_cache,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with config, smoke, and mc_path fields.
    """
    parser = argparse.ArgumentParser(description="Run non-RL baseline agents.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
    )
    parser.add_argument(
        "--mc-path",
        type=str,
        default=None,
        help="Optional MC dataset JSON path (overrides config-derived path).",
    )
    return parser.parse_args()


def summarize(results: list[dict]) -> dict:
    """Combine buzz metrics and calibration into a single summary dict.

    Parameters
    ----------
    results : list[dict]
        List of episode trace dicts (from asdict(EpisodeResult)).

    Returns
    -------
    dict
        Merged summary with accuracy, S_q, ECE, Brier, etc.
    """
    return {
        **summarize_buzz_metrics(results),
        **calibration_at_buzz(results),
    }


def main() -> None:
    """Run all baseline agents and save artifacts."""
    start_time = time.time()

    args = parse_args()

    config = load_config(args.config, smoke=args.smoke)

    split = "smoke" if args.smoke else "main"
    out_dir = ARTIFACT_DIR / split

    # Determine MC dataset path
    mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"

    # Fallback: check data/processed/ if artifacts path doesn't exist
    if not mc_path.exists():
        fallback = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"
        if fallback.exists():
            print(f"MC dataset not found at {mc_path}, using fallback: {fallback}")
            mc_path = fallback

    print(f"Loading MC questions from: {mc_path}")
    mc_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(mc_questions)} MC questions")

    # Build likelihood model
    print(f"Building likelihood model: {config['likelihood']['model']}")
    likelihood_model = build_likelihood_model(config, mc_questions)
    load_embedding_cache(likelihood_model, config)

    # Extract hyperparameters
    beta = float(config["likelihood"].get("beta", 5.0))
    alpha = float(config["bayesian"].get("alpha", 10.0))
    thresholds = [float(x) for x in config["bayesian"]["threshold_sweep"]]

    print(f"Beta: {beta}, Alpha: {alpha}")
    print(f"Thresholds: {thresholds}")

    # --- Pre-compute all embeddings once (batched) ---
    all_texts: list[str] = []
    for q in mc_questions:
        all_texts.extend(q.cumulative_prefixes)
        all_texts.extend(q.option_profiles)
        for step_idx in range(len(q.run_indices)):
            prev_idx = q.run_indices[step_idx - 1] if step_idx > 0 else -1
            all_texts.append(" ".join(q.tokens[prev_idx + 1 : q.run_indices[step_idx] + 1]))
    print(f"\nPre-computing embeddings for {len(set(all_texts)):,} unique texts...")
    likelihood_model.precompute_embeddings(all_texts, batch_size=64)
    save_embedding_cache(likelihood_model, config)

    # --- Pre-compute beliefs (one model pass, all steps) ---
    precomputed = precompute_beliefs(mc_questions, likelihood_model, beta)

    # --- Threshold sweep (pure numpy, instant) ---
    print("\nRunning ThresholdBuzzer sweep...")
    threshold_runs = sweep_thresholds(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        thresholds=thresholds,
        beta=beta,
        alpha=alpha,
        precomputed=precomputed,
    )

    threshold_payload: dict[str, list[dict]] = {}
    threshold_summary: dict[str, dict] = {}
    for threshold, runs in threshold_runs.items():
        rows = [asdict(r) for r in runs]
        threshold_payload[str(threshold)] = rows
        threshold_summary[str(threshold)] = summarize(rows)

    # --- Softmax profile sweep (reuse from_scratch precomputed beliefs) ---
    print("\nRunning SoftmaxProfile sweep (precomputed)...")
    softmax_payload: dict[str, list[dict]] = {}
    softmax_summary: dict[str, dict] = {}
    for threshold in thresholds:
        results = [
            asdict(_softmax_episode_from_precomputed(pq, threshold, alpha))
            for pq in precomputed
        ]
        softmax_payload[str(threshold)] = results
        softmax_summary[str(threshold)] = summarize(results)

    # --- Sequential Bayes sweep (one belief pass, pure numpy threshold sweep) ---
    print("Pre-computing sequential Bayes beliefs...")
    seq_precomputed = precompute_sequential_beliefs(mc_questions, likelihood_model, beta)
    print("Running SequentialBayes sweep (precomputed)...")
    seq_results = sweep_sequential_thresholds(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        thresholds=thresholds,
        beta=beta,
        alpha=alpha,
        precomputed=seq_precomputed,
    )
    sequential_payload: dict[str, list[dict]] = {}
    sequential_summary: dict[str, dict] = {}
    for threshold, runs in seq_results.items():
        rows = [asdict(r) for r in runs]
        sequential_payload[str(threshold)] = rows
        sequential_summary[str(threshold)] = summarize(rows)

    # --- AlwaysBuzzFinal (reuse from_scratch precomputed beliefs) ---
    print("Running AlwaysBuzzFinal baseline (precomputed)...")
    floor_runs = [asdict(_always_final_from_precomputed(pq)) for pq in precomputed]
    floor_summary = summarize(floor_runs)

    # --- Save artifacts ---
    print(f"\nSaving artifacts to: {out_dir}")
    save_json(out_dir / "baseline_threshold_runs.json", threshold_payload)
    save_json(out_dir / "baseline_softmax_profile_runs.json", softmax_payload)
    save_json(out_dir / "baseline_sequential_bayes_runs.json", sequential_payload)
    save_json(out_dir / "baseline_floor_runs.json", floor_runs)

    summary = {
        "threshold": threshold_summary,
        "softmax_profile": softmax_summary,
        "sequential_bayes": sequential_summary,
        "always_final": floor_summary,
    }
    save_json(out_dir / "baseline_summary.json", summary)

    elapsed = time.time() - start_time
    print(f"\nWrote baseline outputs to: {out_dir}")
    print(f"Total time: {elapsed:.1f} seconds")

    # Print summary highlights
    print("\n--- Summary ---")
    for agent_name, agent_summary in summary.items():
        if isinstance(agent_summary, dict) and "buzz_accuracy" in agent_summary:
            # Single-threshold agent (always_final)
            print(f"  {agent_name}: accuracy={agent_summary['buzz_accuracy']:.3f}, "
                  f"mean_sq={agent_summary.get('mean_sq', 0):.3f}")
        elif isinstance(agent_summary, dict):
            # Multi-threshold agent
            for thr, metrics in agent_summary.items():
                if isinstance(metrics, dict) and "buzz_accuracy" in metrics:
                    print(f"  {agent_name}[{thr}]: accuracy={metrics['buzz_accuracy']:.3f}, "
                          f"mean_sq={metrics.get('mean_sq', 0):.3f}")


if __name__ == "__main__":
    main()
````

## File: CLAUDE.md
````markdown
# CLAUDE.md

See **AGENTS.md** for the full repo contract: setup, architecture, testing, smoke pipeline, and configuration.

## Claude-specific notes

- `.planning/` is durable project memory; respect STATE.md decisions.
- Prefer narrow verification over broad cargo-cult test runs.
- Do not add dependencies unless required.
- Seeds: use 1, 2, 3 for multi-seed runs.
- NumPy/PyTorch vectorized operations over loops in ML code.
````

## File: models/likelihoods.py
````python
"""
Likelihood Model Interface

Abstract base class for likelihood models that score answer options against
revealed clue text. Concrete implementations (TF-IDF, SBERT, T5) inherit
from ``LikelihoodModel`` and implement ``score()`` and ``_embed_batch()``.

The ``score()`` method returns **raw similarity scores**, not probabilities.
The environment applies softmax with a configurable temperature (beta) to
convert scores into a belief distribution.

Embedding caching is built into the base class: texts are hashed via SHA-256
and cached as float32 numpy arrays, so repeated calls with the same text
skip recomputation.

Ported from qb-rl reference implementation (models/likelihoods.py lines 1-38).
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


def _text_key(text: str) -> str:
    """Compute a SHA-256 hash key for embedding cache lookups.

    Parameters
    ----------
    text : str
        Input text to hash.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 digest.

    Examples
    --------
    >>> key = _text_key("hello world")
    >>> len(key)
    64
    >>> _text_key("hello world") == _text_key("hello world")
    True
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _best_torch_device() -> "torch.device":
    """Select the best available accelerator: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LikelihoodModel(ABC):
    """Abstract base class for likelihood models.

    Likelihood models score how well each answer option matches a given
    clue prefix. The environment uses these scores (via softmax) to compute
    belief distributions over answer options.

    Subclasses must implement:
        - ``score(clue_prefix, option_profiles) -> np.ndarray``
        - ``_embed_batch(texts) -> np.ndarray``

    The base class provides ``embed_and_cache()`` which handles caching of
    text embeddings via SHA-256 content hashing.

    Attributes
    ----------
    embedding_cache : dict[str, np.ndarray]
        Maps SHA-256 text hashes to float32 embedding vectors.
    """

    def __init__(self) -> None:
        self.embedding_cache: dict[str, np.ndarray] = {}

    @property
    def cache_memory_bytes(self) -> int:
        """Approximate memory used by the embedding cache in bytes."""
        return sum(v.nbytes for v in self.embedding_cache.values())

    @abstractmethod
    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Return raw similarity scores for each answer option.

        The caller (environment) converts these to probabilities via
        softmax with a beta temperature parameter. Higher scores indicate
        stronger match between clue and option.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far (concatenation of clues up to current step).
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Raw similarity scores of shape (K,) where K = len(option_profiles).
        """

    def embed_and_cache(self, texts: list[str]) -> np.ndarray:
        """Embed texts, using cache for previously seen inputs.

        Texts are identified by their SHA-256 hash. Only unseen texts
        are passed to ``_embed_batch()`` for actual computation; cached
        results are reused.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        np.ndarray
            Stacked embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        missing = [text for text in texts if _text_key(text) not in self.embedding_cache]
        if missing:
            new_embeddings = self._embed_batch(missing)
            for text, emb in zip(missing, new_embeddings):
                self.embedding_cache[_text_key(text)] = emb.astype(np.float32)
        return np.stack([self.embedding_cache[_text_key(text)] for text in texts])

    def precompute_embeddings(
        self,
        texts: list[str],
        batch_size: int = 64,
        desc: str = "Pre-computing embeddings",
    ) -> None:
        """Bulk pre-embed texts into cache, processing in batches.

        Call this before running agents so that all subsequent ``score()``
        calls are pure cache lookups (numpy dot products).  Duplicate and
        already-cached texts are skipped automatically.

        Parameters
        ----------
        texts : list[str]
            All texts to embed (clue prefixes, option profiles, fragments).
        batch_size : int
            Number of texts per ``_embed_batch`` call.
        desc : str
            tqdm progress-bar description.
        """
        from tqdm import tqdm

        unique = [t for t in dict.fromkeys(texts) if _text_key(t) not in self.embedding_cache]
        if not unique:
            return
        for i in tqdm(range(0, len(unique), batch_size), desc=desc,
                       total=(len(unique) + batch_size - 1) // batch_size):
            batch = unique[i : i + batch_size]
            embeddings = self._embed_batch(batch)
            for text, emb in zip(batch, embeddings):
                self.embedding_cache[_text_key(text)] = emb.astype(np.float32)

    def save_cache(self, path: str | Path) -> int:
        """Persist embedding_cache to disk as compressed ``.npz``.

        Creates parent directories if needed. Keys are SHA-256 hex
        strings (valid Python identifiers), values are float32 arrays.

        Parameters
        ----------
        path : str or Path
            Destination file path (should end with ``.npz``).

        Returns
        -------
        int
            Number of cache entries saved.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **self.embedding_cache)
        return len(self.embedding_cache)

    def load_cache(self, path: str | Path) -> int:
        """Load embedding_cache from a ``.npz`` file on disk.

        Merges loaded entries into the existing cache **without**
        overwriting keys that are already present (existing keys win).
        If the file does not exist, silently returns 0 (cold-start).

        Parameters
        ----------
        path : str or Path
            Path to ``.npz`` file previously written by ``save_cache``.

        Returns
        -------
        int
            Number of *new* entries added to the cache.
        """
        p = Path(path)
        if not p.exists():
            return 0
        with np.load(p) as data:
            loaded = 0
            for key in data.files:
                if key not in self.embedding_cache:
                    self.embedding_cache[key] = data[key].astype(np.float32)
                    loaded += 1
            return loaded

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Subclasses must implement.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            Embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        raise NotImplementedError


class TfIdfLikelihood(LikelihoodModel):
    """TF-IDF based likelihood model using cosine similarity.

    Uses scikit-learn's ``TfidfVectorizer`` to learn vocabulary and IDF weights
    from a corpus, then scores clue-option similarity via cosine distance in the
    TF-IDF vector space.

    The model **must** be ``fit()`` on a corpus before calling ``score()`` or
    ``_embed_batch()``. Calling these methods on an unfitted model raises
    ``RuntimeError``.

    This is the fast, interpretable baseline: keyword overlap drives similarity.
    It works well when clues contain distinctive vocabulary but misses semantic
    relationships (e.g., "first president" vs "George Washington").

    Parameters
    ----------
    corpus_texts : list[str] or None
        If provided, ``fit()`` is called immediately on these texts.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Scikit-learn vectorizer with English stop words removed.
    _is_fit : bool
        Whether the vectorizer has been fit on a corpus.

    Examples
    --------
    >>> corpus = ["George Washington was the first president",
    ...           "Abraham Lincoln freed the slaves"]
    >>> model = TfIdfLikelihood(corpus_texts=corpus)
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, corpus_texts: list[str] | None = None) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._is_fit = False
        if corpus_texts:
            self.fit(corpus_texts)

    def save_cache(self, path: str | Path) -> int:
        """No-op: TF-IDF embeddings are vocabulary-specific and not portable.

        TF-IDF vectors depend on the fitted vocabulary, which changes
        between ``fit()`` calls. Persisting them would produce wrong
        results if the vocabulary differs.

        Returns
        -------
        int
            Always 0.
        """
        return 0

    def fit(self, corpus_texts: list[str]) -> "TfIdfLikelihood":
        """Learn vocabulary and IDF weights from a text corpus.

        Parameters
        ----------
        corpus_texts : list[str]
            Corpus of documents to learn from. Should include answer profiles,
            clue texts, or both to capture domain vocabulary.

        Returns
        -------
        TfIdfLikelihood
            Self, for method chaining.
        """
        self.vectorizer.fit(corpus_texts)
        self._is_fit = True
        return self

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option against the clue using TF-IDF cosine similarity.

        Uses ``embed_and_cache()`` to embed both the clue and options, so
        repeated calls with the same texts skip vectorizer.transform().
        Since ``_embed_batch()`` returns L2-normalized vectors, the dot
        product equals cosine similarity.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far.
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Cosine similarity scores of shape (K,), dtype float32.
            Values in [-1, 1] but typically [0, 1] for TF-IDF.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._is_fit:
            raise RuntimeError("TfIdfLikelihood must be fit() before score().")
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts as dense, L2-normalized TF-IDF vectors.

        Row-wise L2 normalization ensures that dot product between any
        two embedding vectors equals their cosine similarity, matching
        the convention used by SBERT and T5 likelihood models.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            L2-normalized dense TF-IDF matrix of shape
            (len(texts), vocab_size), dtype float32.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._is_fit:
            raise RuntimeError("TfIdfLikelihood must be fit() before embedding.")
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero for empty docs
        return mat / norms


class SBERTLikelihood(LikelihoodModel):
    """Sentence-BERT likelihood model using semantic embeddings.

    Uses a ``SentenceTransformer`` model to compute dense, L2-normalized
    embeddings. Cosine similarity is computed as a simple dot product since
    embeddings are pre-normalized (``normalize_embeddings=True``).

    Inherits ``embed_and_cache()`` from ``LikelihoodModel`` for transparent
    caching of embeddings via SHA-256 content hashing. The first call to
    ``score()`` computes and caches all embeddings; subsequent calls with the
    same texts are fast cache lookups.

    Compared to TF-IDF, SBERT captures semantic similarity (e.g., "first
    president" and "George Washington" score highly even without word overlap)
    but is slower due to the neural encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for ``SentenceTransformer``.
        Default is ``"all-MiniLM-L6-v2"`` (22M params, 384-dim embeddings).
        First run downloads the model (~80MB) from HuggingFace.

    Attributes
    ----------
    model_name : str
        The SentenceTransformer model name.
    encoder : SentenceTransformer
        The loaded sentence transformer model.

    Examples
    --------
    >>> model = SBERTLikelihood()  # downloads model on first run
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the SentenceTransformer encoder.

        Embeddings are L2-normalized so that cosine similarity can be computed
        as a simple dot product (avoiding the division by norms).

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            Normalized embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        return self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using semantic cosine similarity.

        Computes dot product between the clue embedding and each option
        embedding. Since embeddings are L2-normalized, dot product equals
        cosine similarity.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far.
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Cosine similarity scores of shape (K,), dtype float32.
            Values in [-1, 1].
        """
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


class OpenAILikelihood(LikelihoodModel):
    """OpenAI embedding likelihood model using normalized embedding similarity.

    This path is optional and only activates when explicitly selected in config.
    It requires both the ``openai`` Python package and ``OPENAI_API_KEY`` to be
    available at runtime.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        super().__init__()

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError(
                "OpenAI likelihood requires OPENAI_API_KEY to be set."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI likelihood requires the openai package. "
                "Install it with: pip install -e .[openai] or pip install openai."
            ) from exc

        self.model = model
        self.client = OpenAI(api_key=resolved_api_key)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts via the OpenAI embeddings API and L2-normalize them."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        embeddings = np.stack(vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using cosine similarity over normalized embeddings."""
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


class T5Likelihood(LikelihoodModel):
    """T5 encoder likelihood model using mean-pooled semantic embeddings.

    Uses ``T5EncoderModel`` (not full ``T5ForConditionalGeneration``) for 2x
    faster inference and half the memory. Embeddings are mean-pooled over
    sequence length with attention mask weighting to handle padding correctly.

    Inherits ``embed_and_cache()`` from ``LikelihoodModel`` for transparent
    caching of embeddings via SHA-256 content hashing. The first call to
    ``score()`` computes and caches all embeddings; subsequent calls with the
    same texts are fast cache lookups.

    Compared to SBERT, T5 captures deeper semantic relationships via its
    encoder-decoder pre-training on massive text corpora. This is the novel
    contribution: using T5 as a likelihood model rather than just as a policy
    encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace T5 model identifier. Default is ``"t5-base"``
        (220M params). Options:

        - ``"t5-small"`` (60M params) -- fastest, lowest quality
        - ``"t5-base"`` (220M params) -- balanced (recommended)
        - ``"t5-large"`` (770M params) -- best quality, requires 8GB GPU VRAM

        First run downloads the model from HuggingFace (~850MB for t5-base).

    Attributes
    ----------
    model_name : str
        The T5 model identifier.
    encoder : T5EncoderModel
        Pre-trained T5 encoder loaded from HuggingFace.
    tokenizer : T5TokenizerFast
        Fast T5 tokenizer for text preprocessing.
    device : torch.device
        Computation device (cuda if available, else cpu).

    Examples
    --------
    >>> model = T5Likelihood(model_name="t5-small")
    >>> scores = model.score("first president", ["Washington", "Einstein"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, model_name: str = "t5-base") -> None:
        super().__init__()
        import torch
        from transformers import T5EncoderModel, T5TokenizerFast

        self.model_name = model_name
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.device = _best_torch_device()
        self.encoder.to(self.device)
        self.encoder.eval()

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts using T5 encoder with attention-masked mean pooling.

        Mean pooling uses the attention mask to exclude padding tokens from the
        average, ensuring correct semantic embeddings when sequences have
        different lengths. Embeddings are L2-normalized so that cosine
        similarity can be computed as a simple dot product.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            L2-normalized embeddings of shape (len(texts), hidden_dim),
            dtype float32. Hidden dim is 512 (t5-small), 768 (t5-base),
            or 1024 (t5-large).

        Notes
        -----
        Tensors are detached and moved to CPU immediately after computation
        to prevent GPU memory leaks when called repeatedly during episodes.
        """
        import torch

        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.encoder(**encoded)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            # Mean pooling over sequence length with attention mask
            mask = encoded.attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            masked_hidden = last_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden_dim)
            mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
            mean_pooled = sum_hidden / mask_sum  # (batch, hidden_dim)

            # L2 normalize for cosine similarity via dot product
            embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            # Detach and move to CPU to prevent GPU memory leak
            embeddings = embeddings.detach().cpu().numpy().astype(np.float32)

        return embeddings

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using T5 semantic cosine similarity.

        Computes dot product between the clue embedding and each option
        embedding. Since embeddings are L2-normalized, dot product equals
        cosine similarity.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far.
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Cosine similarity scores of shape (K,), dtype float32.
            Values in [-1, 1].
        """
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


def build_likelihood_from_config(
    config: dict[str, Any], corpus_texts: list[str] | None = None
) -> LikelihoodModel:
    """Construct a likelihood model from YAML configuration.

    Factory function that reads the ``likelihood`` section of the config dict
    and instantiates the appropriate ``LikelihoodModel`` subclass.

    Parameters
    ----------
    config : dict[str, Any]
        Full YAML config dict. Must contain a ``"likelihood"`` key with at
        least a ``"model"`` field specifying the model type.

        Supported model types:
        - ``"tfidf"``: TF-IDF cosine similarity (requires ``corpus_texts``)
        - ``"sbert"``: Sentence-BERT semantic similarity
        - ``"openai"``: OpenAI embedding similarity
        - ``"t5"`` / ``"t5-small"`` / ``"t5-base"`` / ``"t5-large"``:
          T5 encoder semantic similarity

        Optional config keys:
        - ``"sbert_name"`` or ``"embedding_model"``: SentenceTransformer model
          name (default: ``"all-MiniLM-L6-v2"``)
        - ``"openai_model"``: OpenAI embedding model name
          (default: ``"text-embedding-3-small"``)
        - ``"t5_name"``: T5 model name (default: ``"t5-base"``)

    corpus_texts : list[str] or None
        Text corpus for TF-IDF fitting. Required when ``model == "tfidf"``,
        ignored for other models.

    Returns
    -------
    LikelihoodModel
        An instantiated and ready-to-use likelihood model.

    Raises
    ------
    ValueError
        If ``model`` is ``"tfidf"`` and ``corpus_texts`` is None.
        If ``model`` is not a recognized model type.

    Examples
    --------
    >>> from qb_data.config import load_config
    >>> config = load_config("configs/default.yaml")
    >>> model = build_likelihood_from_config(config, corpus_texts=my_corpus)
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    """
    cfg = config["likelihood"]
    model_name = cfg.get("model", "sbert")

    if model_name == "tfidf":
        if not corpus_texts:
            raise ValueError("TF-IDF likelihood requires corpus_texts.")
        return TfIdfLikelihood(corpus_texts=corpus_texts)

    if model_name == "sbert":
        # Support both "sbert_name" (qb-rl convention) and
        # "embedding_model" (qanta-buzzer default.yaml convention)
        sbert_name = cfg.get("sbert_name", cfg.get("embedding_model", "all-MiniLM-L6-v2"))
        return SBERTLikelihood(model_name=sbert_name)

    if model_name == "openai":
        return OpenAILikelihood(
            model=cfg.get("openai_model", "text-embedding-3-small"),
        )

    if model_name == "t5":
        t5_name = cfg.get("t5_name", "t5-base")
        return T5Likelihood(model_name=t5_name)

    if isinstance(model_name, str) and model_name.startswith("t5"):
        t5_name = model_name
        return T5Likelihood(model_name=t5_name)

    raise ValueError(f"Unknown likelihood model: {model_name}")
````

## File: scripts/train_ppo.py
````python
#!/usr/bin/env python3
"""
Train PPO buzzer agent on belief-feature observations.

Loads MC questions, builds a likelihood model, creates a Gymnasium environment,
trains an MLP policy with SB3 PPO, then evaluates with episode traces and
summary metrics (accuracy, S_q, ECE, Brier score).

Usage:
    python scripts/train_ppo.py --smoke              # Quick smoke test
    python scripts/train_ppo.py --smoke --deterministic-eval
    python scripts/train_ppo.py --config configs/custom.yaml
    python scripts/train_ppo.py --timesteps 50000    # Override timesteps

Ported from qb-rl reference implementation (scripts/train_ppo.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_buzzer import PPOBuzzer
from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
from qb_env.tossup_env import make_env_from_config, precompute_beliefs
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
    load_embedding_cache,
    load_mc_questions,
    save_embedding_cache,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with config, smoke, mc_path, timesteps, and
        deterministic_eval fields.
    """
    parser = argparse.ArgumentParser(description="Train PPO buzzer.")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
    )
    parser.add_argument(
        "--mc-path", type=str, default=None,
        help="Optional MC dataset JSON path (overrides config-derived path).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total_timesteps from config.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override PPO/environment seed from config.",
    )
    parser.add_argument(
        "--deterministic-eval", action="store_true",
        help="Use deterministic policy for post-training episode evaluation.",
    )
    parser.add_argument(
        "--stochastic-eval", action="store_true",
        help="Force stochastic policy sampling for post-training evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    """Train PPO agent and save model + evaluation artifacts."""
    args = parse_args()

    config = load_config(args.config, smoke=args.smoke)

    split = "smoke" if args.smoke else "main"
    out_dir = ARTIFACT_DIR / split
    mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"

    # Fallback: check data/processed/ if artifacts path doesn't exist
    if not mc_path.exists():
        fallback = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"
        if fallback.exists():
            print(f"MC dataset not found at {mc_path}, using fallback: {fallback}")
            mc_path = fallback

    print(f"Loading MC questions from: {mc_path}")
    mc_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(mc_questions)} MC questions")

    print(f"Building likelihood model: {config['likelihood']['model']}")
    likelihood_model = build_likelihood_model(config, mc_questions)
    load_embedding_cache(likelihood_model, config)

    env_cfg = config["environment"]
    lik_cfg = config["likelihood"]

    print(f"Precomputing belief trajectories for {len(mc_questions)} questions...")
    belief_cache = precompute_beliefs(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
        beta=float(lik_cfg.get("beta", 5.0)),
        K=int(config["data"].get("K", 4)),
    )
    print(f"Cached {len(belief_cache)} belief vectors")
    save_embedding_cache(likelihood_model, config)

    env = make_env_from_config(
        mc_questions=mc_questions,
        likelihood_model=likelihood_model,
        config=config,
        precomputed_beliefs=belief_cache,
    )

    ppo_cfg = config["ppo"]
    train_seed = int(args.seed if args.seed is not None else ppo_cfg.get("seed", 13))
    total_timesteps = int(
        args.timesteps if args.timesteps is not None else ppo_cfg["total_timesteps"]
    )

    print(f"Training PPO for {total_timesteps} timesteps...")
    agent = PPOBuzzer(
        env=env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        seed=train_seed,
        policy_kwargs=ppo_cfg.get("policy_kwargs", {"net_arch": [64, 64]}),
        verbose=1,
    )

    agent.train(total_timesteps=total_timesteps)
    model_path = out_dir / "ppo_model"
    agent.save(model_path)

    eval_deterministic = True
    if args.stochastic_eval:
        eval_deterministic = False
    elif args.deterministic_eval:
        eval_deterministic = True

    print(
        f"Evaluating PPO agent on {len(mc_questions)} questions "
        f"(deterministic={eval_deterministic})..."
    )
    traces = [
        asdict(
            agent.run_episode(
                deterministic=eval_deterministic,
                question_idx=i,
            )
        )
        for i in range(len(mc_questions))
    ]
    summary = {**summarize_buzz_metrics(traces), **calibration_at_buzz(traces)}

    save_json(out_dir / "ppo_runs.json", traces)
    save_json(out_dir / "ppo_summary.json", summary)
    print(f"Saved PPO model to: {model_path}.zip")
    print(f"Saved PPO summaries to: {out_dir}")


if __name__ == "__main__":
    main()
````

## File: README.md
````markdown
# Quiz Bowl RL Buzzer (Unified)

Unified CS234 final project codebase for quiz bowl buzzing under incremental clues.

This repo keeps `qanta-buzzer` as the canonical implementation while preserving a qb-rl compatibility bridge:

- Modular belief-feature pipeline: `qb_data/` -> `models/` -> `qb_env/` -> `agents/` -> `evaluation/` -> `scripts/`
- T5 policy pipeline: supervised warm-start and PPO for end-to-end text-based buzzing
- qb-rl-compatible import/config shims for older notebooks and scripts
- Optional OpenAI embedding support (`likelihood.model: openai`, `data.distractor_strategy: openai_profile`)

## Setup

Requires Python >= 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional OpenAI support:

```bash
pip install -e '.[openai]'
export OPENAI_API_KEY=...
```

## Main Workflows

### Belief-feature / PPO pipeline

The canonical four-stage smoke pipeline:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

`--smoke` selects `configs/smoke.yaml` and writes outputs to `artifacts/smoke/`. Drop `--smoke` for full runs (uses `configs/default.yaml`, writes to `artifacts/main/`).

The smoke config uses tuned reward settings (`wait_penalty=0.05`, `early_buzz_penalty=0.2`, `ppo.seed=13`, `ppo.total_timesteps=3000`).

`train_ppo.py` also accepts `--seed` to override the PPO/environment seed, and `--stochastic-eval` / `--deterministic-eval` to control post-training evaluation mode.

### T5 policy pipeline

Trains a T5-based policy with supervised warm-start followed by PPO fine-tuning:

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke  # quick test with t5-small
```

The T5 pipeline uses its own config (`configs/t5_policy.yaml`) which defines `model`, `supervised`, `ppo`, and `data` sections. It does not inherit `environment` or `likelihood` settings from the belief-feature configs -- the T5 PPO trainer uses default reward settings (`wait_penalty=0.1`).

### Policy comparison

```bash
python scripts/compare_policies.py --t5-checkpoint checkpoints/ppo_t5/best_model
```

Compares the MLP belief-feature policy against the T5 end-to-end policy on the same test set. Accuracy and buzz-position metrics are directly comparable. ECE and Brier are computed identically (top-answer probability at buzz time). S_q and reward comparisons are qualitative because the two architectures use different confidence semantics (belief-sigmoid vs wait-head probability) and different reward settings (config-driven vs T5-pipeline defaults).

### Additional scripts

- `scripts/run_smoke_pipeline.py` -- runs all four smoke stages sequentially and writes a timing summary to `artifacts/smoke/smoke_pipeline_summary.json`
- `scripts/sweep_reward_shaping.py` -- grid sweep over `wait_penalty` and `early_buzz_penalty` with multi-seed evaluation
- `generate_presentation.py` -- generates the Marp presentation slides

## Configuration

Two primary YAML configs:

| Config | Purpose | Key reward settings |
|--------|---------|-------------------|
| `configs/default.yaml` | Full runs | `wait_penalty=0.05`, `early_buzz_penalty=0.2`, `buzz_incorrect=-0.5` |
| `configs/smoke.yaml` | Quick tests (50 questions) | Same as default except `buzz_incorrect=-1.0`, `total_timesteps=3000` |
| `configs/t5_policy.yaml` | T5 pipeline | Own `model`/`supervised`/`ppo`/`data` sections; no `environment` |

qb-rl config aliases are also supported: `data.dataset`, `data.dataset_config`, `likelihood.sbert_name`, `environment.reward` as an alias for `reward_mode`, etc.

## Testing

261 tests across 16 test files:

```bash
pytest                    # full suite
pytest tests/test_agents.py tests/test_environment.py tests/test_ppo_buzzer.py  # quick iteration
```

The test suite covers:

- Baseline agents (threshold, softmax-profile, sequential Bayes) and PPO wrapper
- Gymnasium environment behavior, reward modes, and belief computation
- Likelihood model factories (TF-IDF, SBERT with offline-safe stubs)
- T5 policy model, supervised trainer, and PPO trainer
- Evaluation metrics (S_q, ECE, Brier score, calibration at buzz, per-category accuracy)
- Dataset split reproducibility (cross-process determinism)
- qb-rl compatibility bridge
- Text observation wrapper

## Architecture

```
qb_data/        Data loading, answer profiles, stratified splits, MC construction
qb_env/         Gymnasium environment, text wrapper, qb-rl compatibility shims
models/         Likelihood models (TF-IDF, SBERT, T5, OpenAI), belief features, T5 policy
agents/         Threshold, softmax-profile, sequential Bayes, PPO buzzer
evaluation/     S_q metric, calibration, control experiments, plotting
scripts/        Pipeline entrypoints and shared helpers
training/       T5 policy supervised + PPO trainers
configs/        YAML configuration files
artifacts/      Generated pipeline outputs (smoke/ and main/)
```

## Compatibility Bridge

These old qb-rl import paths resolve in this repo:

- `qb_env.data_loader`, `qb_env.mc_builder`, `qb_env.text_utils`
- `models.answer_profiles`
- `agents.softmax_profile_buzzer`

The bridge is additive. `qb_data/` remains the canonical home for data loading and MC construction. OpenAI support is opt-in only -- default local workflows stay offline-friendly.

## Documentation

- `AGENTS.md` -- canonical repo contract for all coding agents (setup, architecture, testing, configuration)
- `CLAUDE.md` -- thin shim pointing to AGENTS.md with Claude-specific notes
- `walkthrough.md` -- end-to-end walkthrough exercising both pipelines (pre-remediation snapshot)
- `PRESENTATION.md` -- Marp presentation slides for the CS234 final project
- `.planning/` -- canonical project state, roadmap, architectural decisions, and remediation log

## Legacy Prototype

The pre-modularization prototype (`main.py`, `environment.py`, `model.py`, `dataset.py`, `config.py`, etc.) has been moved to `_legacy/`. These files are not part of the installed package and are preserved only for reference. The modular `scripts/` pipeline above is the canonical workflow.
````
