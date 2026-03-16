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

from qb_data.config import merge_overrides
from scripts._common import ARTIFACT_DIR, load_mc_questions, parse_overrides


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
    parser.add_argument(
        "--hazard-pretrain",
        action="store_true",
        help="Enable the experimental hazard pretraining bridge before PPO.",
    )
    parser.add_argument(
        "--beta-terminal",
        type=float,
        default=1.0,
        help="Terminal survival penalty used by the hazard bridge.",
    )
    parser.add_argument(
        "--freeze-answer-head",
        action="store_true",
        help="Freeze the answer head during the hazard bridge phase.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides: key=value (e.g. model.model_name=t5-base)",
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


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments and reject unsupported bridge paths."""
    if args.skip_supervised and args.model_path is None:
        print("ERROR: --model-path is required when using --skip-supervised")
        sys.exit(1)
    if args.hazard_pretrain:
        raise NotImplementedError(
            "Hazard pretraining loop not yet implemented. "
            "The math utilities are available in training/hazard_pretrain.py, "
            "but the end-to-end bridge has not been wired into train_t5_policy.py yet."
        )


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
    validate_args(args)

    # Load config with overrides
    config = load_config_with_overrides(args)
    overrides = parse_overrides(args)
    if overrides:
        config = merge_overrides(config, overrides)
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
