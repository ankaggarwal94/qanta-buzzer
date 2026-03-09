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
