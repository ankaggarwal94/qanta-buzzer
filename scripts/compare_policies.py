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
