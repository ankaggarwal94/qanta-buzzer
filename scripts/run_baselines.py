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

from agents.bayesian_buzzer import SequentialBayesBuzzer, SoftmaxProfileBuzzer
from agents.threshold_buzzer import (
    AlwaysBuzzFinalBuzzer,
    precompute_beliefs,
    sweep_thresholds,
)
from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
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

    # --- Softmax profile and Sequential Bayes sweeps ---
    softmax_payload: dict[str, list[dict]] = {}
    softmax_summary: dict[str, dict] = {}
    sequential_payload: dict[str, list[dict]] = {}
    sequential_summary: dict[str, dict] = {}

    for threshold in thresholds:
        print(f"Running SoftmaxProfile and SequentialBayes at threshold={threshold}...")

        softmax_agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood_model,
            threshold=threshold,
            beta=beta,
            alpha=alpha,
        )
        softmax_runs = [asdict(softmax_agent.run_episode(q)) for q in mc_questions]
        softmax_payload[str(threshold)] = softmax_runs
        softmax_summary[str(threshold)] = summarize(softmax_runs)

        seq_agent = SequentialBayesBuzzer(
            likelihood_model=likelihood_model,
            threshold=threshold,
            beta=beta,
            alpha=alpha,
        )
        seq_runs = [asdict(seq_agent.run_episode(q)) for q in mc_questions]
        sequential_payload[str(threshold)] = seq_runs
        sequential_summary[str(threshold)] = summarize(seq_runs)

    # --- AlwaysBuzzFinal (floor baseline) ---
    print("Running AlwaysBuzzFinal baseline...")
    floor_agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood_model, beta=beta)
    floor_runs = [asdict(floor_agent.run_episode(q)) for q in mc_questions]
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
