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
from qb_data.config import merge_overrides
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    dataset_path_for_split,
    load_config,
    load_embedding_cache,
    load_mc_questions,
    parse_overrides,
    redirect_combined_to_split,
    resolve_default_dataset_path,
    save_embedding_cache,
    save_json,
    split_name_from_path,
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
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: artifacts/<split>).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides: key=value (e.g. likelihood.model=tfidf)",
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
    overrides = parse_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = merge_overrides(config, overrides)

    split = "smoke" if args.smoke else "main"
    out_dir = Path(args.output_dir) if args.output_dir else ARTIFACT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine MC dataset path
    if args.mc_path:
        mc_path, dataset_split, mc_warning = redirect_combined_to_split(
            Path(args.mc_path), preferred_split="val",
        )
        if mc_warning:
            print(mc_warning)
    else:
        mc_path, dataset_split, warning = resolve_default_dataset_path(
            out_dir,
            preferred_split="val",
        )
        if warning:
            print(warning)

    print(f"Loading MC questions from: {mc_path}")
    mc_questions = load_mc_questions(mc_path)
    if not mc_questions and not args.mc_path:
        fallback = dataset_path_for_split(mc_path.parent, "combined")
        if fallback.exists() and fallback != mc_path:
            print(
                f"Warning: {mc_path} contained 0 questions; "
                f"falling back to {fallback}"
            )
            mc_path = fallback
            dataset_split = "combined"
            mc_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(mc_questions)} MC questions")

    # Build TF-IDF from train split even when selecting thresholds on val/test.
    train_path = mc_path.parent / "train_dataset.json"
    if train_path.exists():
        likelihood_questions = load_mc_questions(train_path)
    if not train_path.exists() or not likelihood_questions:
        likelihood_questions = mc_questions
        print(
            "Building likelihood model: "
            f"{config['likelihood']['model']} "
            "(train split missing or empty; using selected eval split)"
        )
    else:
        print(
            "Building likelihood model: "
            f"{config['likelihood']['model']} "
            f"(fit on train split with {len(likelihood_questions)} questions)"
        )
    likelihood_model = build_likelihood_model(config, likelihood_questions)
    load_embedding_cache(likelihood_model, config)

    # Extract hyperparameters
    beta = float(config["likelihood"].get("beta", 5.0))
    alpha = float(config["bayesian"].get("alpha", 10.0))
    thresholds = [float(x) for x in config["bayesian"]["threshold_sweep"]]
    env_cfg = config.get("environment", {})
    reward_mode = str(env_cfg.get("reward_mode", "time_penalty"))
    wait_penalty = float(env_cfg.get("wait_penalty", 0.0))
    buzz_correct = float(env_cfg.get("buzz_correct", 1.0))
    buzz_incorrect = float(env_cfg.get("buzz_incorrect", -0.5))
    early_buzz_penalty = float(env_cfg.get("early_buzz_penalty", 0.0))

    if reward_mode not in {"time_penalty", "simple"}:
        print(
            "Warning: exact baseline reward parity is only supported for "
            "time_penalty and simple reward modes. Reported mean_reward_like "
            "will not be config-comparable in this run."
        )

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
        reward_mode=reward_mode,
        wait_penalty=wait_penalty,
        buzz_correct=buzz_correct,
        buzz_incorrect=buzz_incorrect,
        early_buzz_penalty=early_buzz_penalty,
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
            asdict(
                _softmax_episode_from_precomputed(
                    pq,
                    threshold,
                    alpha,
                    reward_mode=reward_mode,
                    wait_penalty=wait_penalty,
                    buzz_correct=buzz_correct,
                    buzz_incorrect=buzz_incorrect,
                    early_buzz_penalty=early_buzz_penalty,
                )
            )
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
        reward_mode=reward_mode,
        wait_penalty=wait_penalty,
        buzz_correct=buzz_correct,
        buzz_incorrect=buzz_incorrect,
        early_buzz_penalty=early_buzz_penalty,
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
    floor_runs = [
        asdict(
            _always_final_from_precomputed(
                pq,
                reward_mode=reward_mode,
                wait_penalty=wait_penalty,
                buzz_correct=buzz_correct,
                buzz_incorrect=buzz_incorrect,
                early_buzz_penalty=early_buzz_penalty,
            )
        )
        for pq in precomputed
    ]
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
        "dataset_split": dataset_split,
        "selection_metric": "mean_sq",
        "reward_supported": reward_mode in {"time_penalty", "simple"},
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
