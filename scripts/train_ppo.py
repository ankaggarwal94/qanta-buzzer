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
    load_mc_questions,
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
