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
  The T5 path uses a lightweight TF-IDF helper built from the recorded
  training reference split when available.
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
    calibration_pairs_at_buzz,
    expected_calibration_error,
    brier_score,
    summarize_buzz_metrics,
    system_score,
)
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
    load_checkpoint_sidecar,
    load_embedding_cache,
    load_mc_questions,
    save_json,
)


def resolve_mlp_eval_config(
    checkpoint_path: str | Path,
    fallback_config: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the config that was used to train an MLP checkpoint.

    If a ``config_used.json`` sidecar exists next to the checkpoint,
    load and return it. Otherwise return ``fallback_config`` unchanged.
    """
    cp = Path(checkpoint_path).resolve()
    candidates = [cp / "config_used.json"] if cp.is_dir() else []
    candidates.append(cp.parent / "config_used.json")

    for sidecar in candidates:
        if sidecar.exists():
            try:
                with open(sidecar, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return fallback_config


def _resolve_manifest_questions(
    checkpoint_path: str | Path,
    split_name: str,
) -> list | None:
    """Load manifest-backed split questions in the recorded qid order.

    Returns None (with a printed warning) for any error — missing file,
    corrupt JSON, incomplete fields, or missing split data on disk.
    Callers should fall back to sibling dataset files or random splitting.
    """
    manifest, _, manifest_error = load_checkpoint_sidecar(
        checkpoint_path, "split_manifest.json"
    )
    if manifest_error:
        print(
            f"Warning: split_manifest.json next to {checkpoint_path} could not "
            f"be read ({manifest_error}); falling back to other resolution."
        )
        return None
    if not isinstance(manifest, dict):
        return None

    split_path = manifest.get(f"{split_name}_path")
    qids = manifest.get(f"{split_name}_qids")
    if not split_path or not isinstance(qids, list):
        print(
            f"Warning: split_manifest.json is missing {split_name}_path or "
            f"{split_name}_qids; falling back to other resolution."
        )
        return None

    path = Path(split_path)
    if not path.exists():
        print(
            f"Warning: manifest-backed {split_name} split does not exist at "
            f"{path}; falling back to other resolution."
        )
        return None

    questions = load_mc_questions(path)
    qid_to_question = {q.qid: q for q in questions}
    missing_qids = [qid for qid in qids if qid not in qid_to_question]
    if missing_qids:
        print(
            f"Warning: manifest-backed {split_name} split at {path} is "
            f"missing {len(missing_qids)} recorded qids; falling back."
        )
        return None
    return [qid_to_question[qid] for qid in qids]


def resolve_t5_test_questions(
    checkpoint_path: str | Path,
    all_questions: list,
    mc_path: Path,
) -> tuple[list, str]:
    """Resolve the T5-held-out test set, preferring checkpoint provenance."""
    manifest_questions = _resolve_manifest_questions(checkpoint_path, "test")
    if manifest_questions is not None:
        return manifest_questions, "split_manifest"

    test_split_path = mc_path.parent / "test_dataset.json"
    if test_split_path.exists():
        return load_mc_questions(test_split_path), "sibling_test_dataset"

    import random

    rng = random.Random(42)
    shuffled = all_questions[:]
    rng.shuffle(shuffled)
    test_start = int(len(shuffled) * 0.85)
    return shuffled[test_start:], "random_split_fallback"


def resolve_t5_reference_questions(
    checkpoint_path: str | Path,
    all_questions: list,
    mc_path: Path,
) -> tuple[list, str]:
    """Resolve the train-side reference set for T5 env reward helpers."""
    manifest_questions = _resolve_manifest_questions(checkpoint_path, "train")
    if manifest_questions is not None:
        return manifest_questions, "split_manifest"

    train_split_path = mc_path.parent / "train_dataset.json"
    if train_split_path.exists():
        return load_mc_questions(train_split_path), "sibling_train_dataset"

    return all_questions, "combined_dataset_fallback"


def evaluate_mlp_policy(
    checkpoint_path: str,
    test_questions: list,
    config: dict,
) -> dict[str, Any]:
    """Evaluate Phase 4 MLP policy on belief features.

    Uses the likelihood model specified by the checkpoint's sidecar
    config (``config_used.json``) when available, otherwise falls back
    to the provided config. If the resolved config selects TF-IDF, the
    corpus is fit on the evaluation set's question/option text.

    Parameters
    ----------
    checkpoint_path : str
        Path to SB3 PPO model checkpoint (``.zip`` file).
    test_questions : list
        List of MCQuestion instances to evaluate on.
    config : dict
        YAML config dict (fallback if no checkpoint sidecar exists).

    Returns
    -------
    dict[str, Any]
        Evaluation results: accuracy, mean_sq, ece, brier, avg_buzz_pos,
        n_questions.
    """
    from agents.ppo_buzzer import PPOBuzzer
    from qb_env.tossup_env import make_env_from_config

    resolved_config = resolve_mlp_eval_config(checkpoint_path, config)
    likelihood_model = build_likelihood_model(resolved_config, test_questions)
    load_embedding_cache(likelihood_model, resolved_config)

    env = make_env_from_config(
        mc_questions=test_questions,
        likelihood_model=likelihood_model,
        config=resolved_config,
    )

    use_maskable = bool(resolved_config.get("ppo", {}).get("use_maskable_ppo", False))
    agent = PPOBuzzer.load(checkpoint_path, env=env, use_maskable_ppo=use_maskable)

    # Run episodes — one per test question, deterministic order
    results = [
        agent.run_episode(deterministic=True, question_idx=i)
        for i in range(len(test_questions))
    ]

    # Compute metrics
    buzz_metrics = summarize_buzz_metrics(results)
    confidences, outcomes = calibration_pairs_at_buzz(results)
    ece = expected_calibration_error(confidences, outcomes)
    brier = brier_score(confidences, outcomes)

    return {
        "accuracy": buzz_metrics["buzz_accuracy"],
        "mean_sq": buzz_metrics["mean_sq"],
        "ece": ece,
        "brier": brier,
        "avg_buzz_pos": buzz_metrics.get("mean_buzz_step", 0.0),
        "mean_reward": buzz_metrics["mean_reward_like"],
        "forced_correct_rate": buzz_metrics.get("forced_correct_rate", 0.0),
        "overall_outcome_accuracy": buzz_metrics.get(
            "overall_outcome_accuracy", buzz_metrics["buzz_accuracy"]
        ),
        "n_questions": len(test_questions),
    }


def evaluate_t5_policy(
    checkpoint_path: str,
    test_questions: list,
    reference_questions: list,
    test_set_source: str,
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
        Evaluation results including policy-only accuracy, forced outcome
        diagnostics, and test-set provenance.
    """
    import torch
    from models.t5_policy import T5PolicyModel
    from models.likelihoods import TfIdfLikelihood
    from qb_env.text_wrapper import TextObservationWrapper
    from qb_env.tossup_env import TossupMCEnv

    # Load T5 policy model
    model = T5PolicyModel.load_pretrained(checkpoint_path)
    model.eval()

    # Build lightweight likelihood for environment reward computation
    corpus = []
    for q in reference_questions:
        corpus.extend(q.option_profiles)
    likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

    correct_count = 0
    forced_correct_count = 0
    total_count = 0
    sq_scores = []
    confidences = []
    outcomes = []
    buzz_positions = []
    total_reward = 0.0

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
                inputs = model.encode_input([obs], max_length=512)
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
            total_reward += episode_reward

            policy_correct = bool(step_info.get("correct", False))
            forced_correct = bool(step_info.get("forced_correct", False))
            if policy_correct:
                correct_count += 1
            if forced_correct:
                forced_correct_count += 1
            total_count += 1

            # Calibration: use top_p (max answer prob) for consistency
            # with belief-feature agents
            if terminated and top_p_trace:
                buzz_step = step_count - 1
                confidences.append(top_p_trace[-1])
                outcomes.append(1 if policy_correct else 0)
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
        "mean_reward": total_reward / max(1, total_count),
        "forced_correct_rate": forced_correct_count / max(1, total_count),
        "overall_outcome_accuracy": (correct_count + forced_correct_count) / max(1, total_count),
        "test_set_source": test_set_source,
        "n_questions_evaluated": total_count,
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

    test_questions, test_set_source = resolve_t5_test_questions(
        args.t5_checkpoint,
        all_questions,
        mc_path,
    )
    reference_questions, reference_source = resolve_t5_reference_questions(
        args.t5_checkpoint,
        all_questions,
        mc_path,
    )
    print(
        f"Using T5 test set source {test_set_source}: {len(test_questions)} questions"
    )
    print(
        f"Using T5 reward reference source {reference_source}: "
        f"{len(reference_questions)} questions"
    )

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
        args.t5_checkpoint,
        test_questions,
        reference_questions,
        test_set_source,
        config,
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
