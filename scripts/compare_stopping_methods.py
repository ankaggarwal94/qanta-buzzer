"""Compare stopping policies on one frozen answerer and named question splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_buzzer import PPOBuzzer
from models.likelihoods import TfIdfLikelihood
from qb_data.config import merge_overrides
from qb_data.data_audit import validate_question_splits
from qb_data.mc_builder import validate_gold_option, validate_reveal_grid
from qb_env.tossup_env import make_env_from_config
from scripts._common import (
    build_likelihood_model, collect_env_texts, load_config, mc_questions_from_payload,
    parse_overrides, save_json,
)
from training.stopping_data import ReplayEnv, collect_episodes, evaluate_actions

METHODS = ("threshold", "final_buzz", "wait_to_horizon", "fitted_value",
           "stop_imitation", "double_dqn", "ppo", "dpo", "grpo")


def summarize(rows):
    """Separate policy decisions from forced outcomes; condition timing on correct buzzes."""
    voluntary = [row for row in rows if row["outcome"] == "voluntary"]
    correct = [row for row in voluntary if row["correct"]]
    return {
        "questions": len(rows), "mean_return": float(np.mean([row["return"] for row in rows])),
        "voluntary_correct": len(correct), "voluntary_wrong": len(voluntary) - len(correct),
        "voluntary_correct_per_100": 100 * len(correct) / len(rows),
        "forced": sum(row["outcome"] == "force_commit" for row in rows),
        "forced_correct": sum(row["outcome"] == "force_commit" and row["correct"] for row in rows),
        "no_buzz": sum(row["outcome"] == "no_buzz" for row in rows),
        "mean_visible_prefixes_given_correct_voluntary": (
            float(np.mean([row["visible_prefixes"] for row in correct])) if correct else None
        ),
    }


def threshold_actions(observations, threshold):
    # The final six belief features always begin with top probability, including variable K.
    return observations[:, -6] >= threshold


def fit_method(name, episodes, validation, args, config, output):
    """Fit on train, select the threshold on validation, and reload trainable policies."""
    if name == "threshold":
        grid = config.get("bayesian", {}).get("threshold_sweep", [0.5, 0.7, 0.9])
        if not grid or not all(np.isfinite(t) and 0 <= t <= 1 for t in grid):
            raise ValueError("threshold_sweep requires values in [0, 1]")
        def val_reward(t):
            return sum(evaluate_actions(e, threshold_actions(e.observations, t))["return"]
                       for e in validation)
        selected = max(sorted(grid), key=val_reward)
        save_json(output / "threshold.json", {"threshold": selected, "grid": grid})
        return lambda x: threshold_actions(x, selected)
    if name == "wait_to_horizon":
        return lambda x: np.zeros(len(x), dtype=bool)
    if name == "final_buzz":
        return lambda x: np.arange(len(x)) == len(x) - 1
    if name == "ppo":
        ppo = config["ppo"]
        env = ReplayEnv(episodes)
        args_ppo = {key: ppo[key] for key in ("n_steps", "batch_size", "n_epochs", "policy_kwargs")
                    if key in ppo}
        args_ppo["learning_rate"] = float(ppo.get("learning_rate", 3e-4))
        agent = PPOBuzzer(env, gamma=1.0, seed=args.seed, device="cpu", **args_ppo)
        agent.train(args.ppo_steps)
        model = agent.model
        save_json(output / "ppo_training.json", {
            "requested_steps": args.ppo_steps, "actual_steps": model.num_timesteps,
            "learning_rate": args_ppo["learning_rate"], "n_steps": model.n_steps,
            "batch_size": model.batch_size, "n_epochs": model.n_epochs,
            "gamma": model.gamma, "gae_lambda": model.gae_lambda,
            "clip_range": float(model.clip_range(1.0)), "ent_coef": model.ent_coef,
            "vf_coef": model.vf_coef, "target_kl": model.target_kl,
        })
        path = output / "ppo.zip"
        agent.save(path)
        loaded = PPOBuzzer.load(path, env, device="cpu")
        return lambda x: loaded.model.predict(x, deterministic=True)[0]

    if name in ("dpo", "grpo"):
        from training.stopping_preferences import build_preference_policy, fit_preferences
        fit, build = fit_preferences, build_preference_policy
    else:
        from training.stopping_value import fit_value, make_value_model
        fit, build = fit_value, make_value_model
    trained = fit(episodes, name, seed=args.seed, epochs=args.epochs)
    if hasattr(trained, "training_diagnostics"):
        save_json(output / f"{name}_training.json", trained.training_diagnostics)
    n_features = episodes[0].observations.shape[-1]
    path = output / f"{name}.pt"
    torch.save({"method": name, "n_features": n_features, "state_dict": trained.state_dict()}, path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    policy = build(saved["n_features"])
    policy.load_state_dict(saved["state_dict"])
    policy.eval()

    @torch.inference_mode()
    def actions(observations):
        scores = policy(torch.as_tensor(observations, dtype=torch.float32))
        if scores.shape != (len(observations), 2) or not torch.isfinite(scores).all():
            raise ValueError(f"Invalid action scores from {name}")
        return scores.argmax(dim=-1).numpy()

    return actions


def run(args, output):
    config = merge_overrides(load_config(args.config), parse_overrides(args))
    paths = {s: args.data_dir / f"{s}_dataset.json" for s in ("train", "val", "test")}
    questions, sources = {}, {}
    for split, path in paths.items():
        raw = path.read_bytes()
        rows = mc_questions_from_payload(json.loads(raw), path)
        if not rows:
            raise ValueError(f"Empty {split} split")
        questions[split] = rows
        sources[split] = {"rows": len(rows), "sha256": hashlib.sha256(raw).hexdigest()}
    validate_question_splits(**questions)
    likelihood = build_likelihood_model(config, questions["train"])
    env = make_env_from_config(questions["train"], likelihood, config)
    if env.reward_mode not in {"simple", "time_penalty", "expected_wins", "human_grounded"}:
        raise ValueError(f"Unsupported belief-environment reward mode: {env.reward_mode!r}")
    # Keep the same train-fitted opponent, scorer and observation width across splits.
    episodes = {}
    for split, rows in questions.items():
        for question in rows:
            validate_gold_option(question)
            validate_reveal_grid(question)
            k = len(question.options)
            width = env.observation_space.shape[0] - 6
            if not (2 <= k <= width and (env.variable_K or k == width) and len(question.option_profiles) == k):
                raise ValueError(f"Invalid menu for {question.qid} in {split}")
        episodes[split] = []
        # Dense TF-IDF vectors have vocabulary width. Discard each chunk after
        # extracting its small belief observations, keeping the fitted IDF fixed.
        chunk_size = 16 if isinstance(likelihood, TfIdfLikelihood) else 128
        for start in range(0, len(rows), chunk_size):
            env.questions = rows[start:start + chunk_size]
            likelihood.precompute_embeddings(collect_env_texts(env.questions),
                                             batch_size=config["likelihood"].get("batch_size", 64))
            episodes[split].extend(collect_episodes(env, seed=args.seed + start))
            if isinstance(likelihood, TfIdfLikelihood):
                likelihood.embedding_cache.clear()
    methods = args.methods.split(",")
    results = {}
    all_rows = {}
    for name in methods:
        act = fit_method(name, episodes["train"], episodes["val"], args, config, output)
        all_rows[name] = {}
        results[name] = {}
        for split in ("val", "test"):
            rows = [evaluate_actions(e, act(e.observations)) for e in episodes[split]]
            all_rows[name][split] = rows
            results[name][split] = summarize(rows)
    # Each paired difference uses the same questions and exact terminal convention.
    if "threshold" in methods:
        for name in methods:
            for split in ("val", "test"):
                current, baseline = all_rows[name][split], all_rows["threshold"][split]
                diffs = [a["return"] - b["return"] for a, b in zip(current, baseline, strict=True)]
                results[name][split]["mean_return_minus_threshold"] = float(np.mean(diffs))
    save_json(output / "decisions.json", all_rows)
    save_json(output / "run.json", {
        "methods": methods, "seed": args.seed, "epochs": args.epochs, "ppo_steps": args.ppo_steps,
        "discount": 1.0, "config": config, "results": results,
        "data": sources,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with train, val and test datasets")
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma-separated subset of " + ", ".join(METHODS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ppo-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("overrides", nargs="*", help="Dotted configuration overrides (key=value)")
    args = parser.parse_args()
    methods = args.methods.split(",")
    if set(methods) - set(METHODS) or len(methods) != len(set(methods)):
        parser.error("--methods must contain unique supported method names")
    if args.epochs < 1 or args.ppo_steps < 1 or any("=" not in s for s in args.overrides):
        parser.error("Training budgets must be positive and overrides must be key=value")
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as stage:
        root = Path(stage) / "run"
        root.mkdir()
        run(args, root)
        root.rename(args.output)
    print(f"Saved held-out comparisons and checkpoints to {args.output}")


if __name__ == "__main__":
    main()
