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
import copy
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_buzzer import PPOBuzzer
from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
from qb_env.stop_only_env import StopOnlyEnv
from qb_env.tossup_env import make_env_from_config, precompute_beliefs
from qb_data.config import merge_overrides
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    collect_env_texts,
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
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=["flat_kplus1", "stop_only"],
        default="flat_kplus1",
        help="Policy action space: flat K+1 actions (default) or binary stop_only.",
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


def main() -> None:
    """Train PPO agent and save model + evaluation artifacts."""
    args = parse_args()

    config = load_config(args.config, smoke=args.smoke)
    overrides = parse_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = merge_overrides(config, overrides)

    split = "smoke" if args.smoke else "main"
    out_dir = Path(args.output_dir) if args.output_dir else ARTIFACT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mc_path:
        train_path, train_split, train_warning = redirect_combined_to_split(
            Path(args.mc_path), preferred_split="train",
        )
    else:
        train_path, train_split, train_warning = resolve_default_dataset_path(
            out_dir,
            preferred_split="train",
        )
    if train_warning:
        print(train_warning)

    print(f"Loading training MC questions from: {train_path}")
    train_questions = load_mc_questions(train_path)
    if not train_questions:
        fallback_path = dataset_path_for_split(train_path.parent, "combined")
        if train_path != fallback_path and fallback_path.exists():
            print(
                f"Warning: {train_path} contained 0 questions; "
                f"falling back to {fallback_path}"
            )
            train_path = fallback_path
            train_split = "combined"
            train_questions = load_mc_questions(train_path)
    print(f"Loaded {len(train_questions)} training questions")

    if args.mc_path:
        explicit_split = split_name_from_path(train_path)
        eval_candidate = dataset_path_for_split(train_path.parent, "val")
        if explicit_split == "val":
            eval_path = train_path
            eval_split = "val"
            eval_warning = None
        elif train_split == "combined":
            # ``--mc-path`` pointed to a combined artifact and no sibling
            # train_dataset.json was found, so training is on the combined
            # corpus (which contains val). Refuse to evaluate on the val
            # sibling: that would let the model "validate" on data it just
            # trained on.
            eval_path = train_path
            eval_split = "combined"
            eval_warning = (
                "Warning: --mc-path resolved to the combined dataset (no "
                "sibling train_dataset.json found); reusing it for "
                "evaluation to avoid leaking val/test into a 'validation' "
                "summary. Build train_dataset.json next to mc_dataset.json "
                "for split-safe validation."
            )
        elif eval_candidate.exists() and eval_candidate != train_path:
            eval_path = eval_candidate
            eval_split = "val"
            eval_warning = None
        else:
            eval_path = train_path
            eval_split = explicit_split if explicit_split != "explicit" else train_split
            eval_warning = (
                "Warning: val_dataset.json not found alongside explicit --mc-path; "
                "post-training evaluation will reuse the training dataset."
            )
    else:
        eval_candidate = dataset_path_for_split(train_path.parent, "val")
        if train_split == "train" and eval_candidate.exists():
            eval_path = eval_candidate
            eval_split = "val"
            eval_warning = None
        else:
            eval_path = train_path
            eval_split = train_split
            eval_warning = (
                "Warning: validation split not found; post-training evaluation "
                "will reuse the training dataset."
            )
    if eval_warning:
        print(eval_warning)

    print(f"Loading evaluation MC questions from: {eval_path}")
    eval_questions = load_mc_questions(eval_path)
    if not eval_questions:
        print(
            f"Warning: {eval_path} contained 0 questions; "
            "falling back to the training dataset for evaluation."
        )
        eval_path = train_path
        eval_split = train_split
        eval_questions = train_questions
    print(f"Loaded {len(eval_questions)} evaluation questions")

    ppo_cfg = config["ppo"]
    train_seed = int(args.seed if args.seed is not None else ppo_cfg.get("seed", 13))
    total_timesteps = int(
        args.timesteps if args.timesteps is not None else ppo_cfg["total_timesteps"]
    )

    # Persist the resolved runtime config, including CLI overrides that bypass
    # merge_overrides(), before it is used to construct the environment.
    config = copy.deepcopy(config)
    config.setdefault("ppo", {})
    config.setdefault("environment", {})
    config["ppo"]["seed"] = train_seed
    config["environment"]["seed"] = train_seed
    config["ppo"]["total_timesteps"] = total_timesteps
    ppo_cfg = config["ppo"]

    print(f"Building likelihood model: {config['likelihood']['model']}")
    likelihood_model = build_likelihood_model(config, train_questions)
    load_embedding_cache(likelihood_model, config)

    env_cfg = config["environment"]
    lik_cfg = config["likelihood"]

    # Batch-encode every text the env will ever score before computing
    # beliefs. For SBERT / T5 likelihoods this turns ~N×~10 single-shot
    # encoder forward passes into ceil(unique_texts / 64) batches.
    # TF-IDF batches into a single sparse-to-dense transform; cost is
    # O(unique_texts × vocab_size) but matches the cache state that
    # ``score()`` would produce anyway.
    train_eval_texts = collect_env_texts(train_questions) + collect_env_texts(
        eval_questions
    )
    eval_label = "val" if eval_split == "val" else eval_split
    print(
        f"Pre-computing embeddings for {len(set(train_eval_texts)):,} unique texts "
        f"(train + {eval_label})..."
    )
    likelihood_model.precompute_embeddings(train_eval_texts, batch_size=64)

    print(f"Precomputing train belief trajectories for {len(train_questions)} questions...")
    belief_cache = precompute_beliefs(
        questions=train_questions,
        likelihood_model=likelihood_model,
        belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
        beta=float(lik_cfg.get("beta", 5.0)),
        K=int(config["data"].get("K", 4)),
    )
    print(f"Cached {len(belief_cache)} belief vectors")
    save_embedding_cache(likelihood_model, config)

    env = make_env_from_config(
        mc_questions=train_questions,
        likelihood_model=likelihood_model,
        config=config,
        precomputed_beliefs=belief_cache,
    )
    if args.policy_mode == "stop_only":
        print("Wrapping environment with StopOnlyEnv (WAIT/BUZZ only)...")
        env = StopOnlyEnv(env)

    use_maskable = bool(ppo_cfg.get("use_maskable_ppo", False))
    if use_maskable:
        print("Using MaskablePPO for variable-K action masking")
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
        use_maskable_ppo=use_maskable,
    )

    agent.train(total_timesteps=total_timesteps)
    model_path = out_dir / "ppo_model"
    # Write metadata BEFORE the model so a failed metadata write does
    # not leave an orphaned checkpoint that downstream evaluators
    # silently mis-resolve. ``config_used.json`` is also written before
    # the model for the same reason.
    #
    # ``training_completed=False`` is written first so the symmetric
    # failure mode -- agent.save() raising AFTER the metadata writes
    # have overwritten a prior run's files -- is detectable by
    # evaluate_all.py: the stale ``ppo_model.zip`` from the prior run
    # is now paired with run_metadata.json that reads False, and
    # evaluate_all.py refuses to replay that combination instead of
    # silently evaluating the stale model under fresh metadata.
    import uuid

    run_id = uuid.uuid4().hex
    save_json(out_dir / "config_used.json", config)
    run_metadata = {
        "schema_version": 2,
        "run_id": run_id,
        "training_completed": False,
        "policy_mode": args.policy_mode,
        "evaluation_mode": (
            "stochastic" if args.stochastic_eval else "deterministic"
        ),
        "smoke": bool(args.smoke),
        "likelihood_reference_split": train_split,
        "likelihood_reference_path": str(train_path.resolve()),
        "likelihood_reference_n_questions": len(train_questions),
    }
    save_json(out_dir / "run_metadata.json", run_metadata)
    agent.save(model_path)
    # Re-stamp run_metadata so successful runs flip the flag. Anything
    # downstream that sees training_completed=False knows the on-disk
    # ppo_model.zip is either stale or absent.
    run_metadata["training_completed"] = True
    save_json(out_dir / "run_metadata.json", run_metadata)

    eval_deterministic = True
    if args.stochastic_eval:
        eval_deterministic = False
    elif args.deterministic_eval:
        eval_deterministic = True

    print(
        f"Evaluating PPO agent on {len(eval_questions)} questions "
        f"(deterministic={eval_deterministic})..."
    )
    eval_belief_cache = precompute_beliefs(
        questions=eval_questions,
        likelihood_model=likelihood_model,
        belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
        beta=float(lik_cfg.get("beta", 5.0)),
        K=int(config["data"].get("K", 4)),
    )
    eval_env = make_env_from_config(
        mc_questions=eval_questions,
        likelihood_model=likelihood_model,
        config=config,
        precomputed_beliefs=eval_belief_cache,
    )
    if args.policy_mode == "stop_only":
        eval_env = StopOnlyEnv(eval_env)
    eval_agent = PPOBuzzer.load(
        model_path,
        env=eval_env,
        use_maskable_ppo=use_maskable,
    )
    traces = [
        asdict(
            eval_agent.run_episode(
                deterministic=eval_deterministic,
                question_idx=i,
            )
        )
        for i in range(len(eval_questions))
    ]
    # ``schema_version`` discriminates pre/post forced-commit semantics for
    # downstream consumers. ``buzz_accuracy`` and ``mean_buzz_step`` count
    # policy-buzz outcomes only on v2; forced-commit credit is reported via
    # ``overall_outcome_accuracy`` / ``forced_correct_rate``. See
    # docs/full-pipeline-runbook.md for the migration guide.
    summary = {
        "schema_version": 2,
        **summarize_buzz_metrics(traces),
        **calibration_at_buzz(traces),
        "train_split": train_split,
        "eval_split": eval_split,
    }

    save_json(out_dir / "ppo_runs.json", traces)
    save_json(out_dir / "ppo_summary.json", summary)
    print(f"Saved PPO model to: {model_path}.zip")
    print(f"Saved PPO summaries to: {out_dir}")


if __name__ == "__main__":
    main()
