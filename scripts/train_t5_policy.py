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
from scripts._common import (
    ARTIFACT_DIR,
    PROCESSED_DIR,
    dataset_path_for_split,
    load_mc_questions,
    parse_overrides,
    resolve_persisted_split_paths,
    save_json,
)


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


def load_questions(
    args: argparse.Namespace,
    config: dict,
    *,
    return_path: bool = False,
) -> list | tuple[list, Path]:
    """Load a combined MC dataset when persisted splits are unavailable.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, may have mc_path override.
    config : dict
        Config dict with data section.

    Returns
    -------
    list or tuple[list, Path]
        Loaded MCQuestion instances, and optionally the resolved path.
    """
    if args.mc_path:
        mc_path = Path(args.mc_path)
    else:
        candidates = (
            [ARTIFACT_DIR / "smoke" / "mc_dataset.json", ARTIFACT_DIR / "main" / "mc_dataset.json"]
            if args.smoke
            else [ARTIFACT_DIR / "main" / "mc_dataset.json", ARTIFACT_DIR / "smoke" / "mc_dataset.json"]
        ) + [PROCESSED_DIR / "mc_dataset.json"]
        mc_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if mc_path is None:
            print("ERROR: No MC dataset found. Run build_mc_dataset.py first.")
            print("Searched locations:")
            for c in candidates:
                print(f"  {c}")
            sys.exit(1)

    print(f"Loading MC questions from: {mc_path}")
    questions = load_mc_questions(mc_path)
    print(f"Loaded {len(questions)} questions")

    # Apply max_questions limit when falling back to a combined dataset.
    max_questions = config.get("data", {}).get("max_questions", None)
    if max_questions and len(questions) > max_questions:
        questions = questions[:max_questions]
        print(f"Limited to {max_questions} questions (smoke mode)")

    if return_path:
        return questions, Path(mc_path)
    return questions


def _build_split_manifest(
    *,
    source: str,
    mc_path: str | None,
    train_questions: list,
    val_questions: list,
    test_questions: list,
    train_path: str | None = None,
    val_path: str | None = None,
    test_path: str | None = None,
    config: dict | None = None,
) -> dict:
    """Build a split-manifest payload for persisted provenance."""
    qid = lambda q: getattr(q, "qid", str(q))
    manifest = {
        "source": source,
        "mc_path": mc_path,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "train_qids": [qid(q) for q in train_questions],
        "val_qids": [qid(q) for q in val_questions],
        "test_qids": [qid(q) for q in test_questions],
    }
    total = len(train_questions) + len(val_questions) + len(test_questions)
    manifest.update(
        {
            "train_count": len(train_questions),
            "val_count": len(val_questions),
            "test_count": len(test_questions),
            "effective_train_ratio": len(train_questions) / max(1, total),
            "effective_val_ratio": len(val_questions) / max(1, total),
            "effective_test_ratio": len(test_questions) / max(1, total),
        }
    )
    if source == "random_split_fallback":
        data = (config or {}).get("data", {})
        manifest["split_seed"] = int(data.get("seed", data.get("shuffle_seed", 42)))
    return manifest


def load_question_splits_with_metadata(
    args: argparse.Namespace, config: dict
) -> tuple[list, list, list, dict]:
    """Load persisted train/val/test artifacts when available.

    Resolution order:
    1. If ``--mc-path`` is given, inspect its parent directory for sibling
       ``train_dataset.json``, ``val_dataset.json``, and ``test_dataset.json``.
    2. Otherwise search the standard artifact directories, preferring the
       smoke directory in smoke mode and the main directory otherwise.
    3. Fall back to loading a combined ``mc_dataset.json`` and performing the
       legacy in-memory random split.
    """
    if args.mc_path:
        candidate_dirs = [Path(args.mc_path).parent]
    elif args.smoke:
        candidate_dirs = [
            ARTIFACT_DIR / "smoke",
            ARTIFACT_DIR / "main",
            PROCESSED_DIR,
        ]
    else:
        candidate_dirs = [
            ARTIFACT_DIR / "main",
            ARTIFACT_DIR / "smoke",
            PROCESSED_DIR,
        ]

    for base_dir in candidate_dirs:
        split_paths = resolve_persisted_split_paths(base_dir)
        if split_paths is None:
            continue
        train_questions = load_mc_questions(split_paths["train"])
        val_questions = load_mc_questions(split_paths["val"])
        test_questions = load_mc_questions(split_paths["test"])
        print(
            "Using persisted dataset splits from "
            f"{base_dir}: {len(train_questions)} train, "
            f"{len(val_questions)} val, {len(test_questions)} test"
        )
        combined_path = dataset_path_for_split(base_dir, "combined")
        manifest = _build_split_manifest(
            source="persisted_artifacts",
            mc_path=(
                str(combined_path)
                if combined_path.exists()
                else (str(args.mc_path) if args.mc_path else None)
            ),
            train_questions=train_questions,
            val_questions=val_questions,
            test_questions=test_questions,
            train_path=str(split_paths["train"]),
            val_path=str(split_paths["val"]),
            test_path=str(split_paths["test"]),
        )
        return train_questions, val_questions, test_questions, manifest

    if args.mc_path:
        print(
            "Warning: persisted train/val/test artifacts were not found "
            f"alongside {args.mc_path}; falling back to an internal random split."
        )
    else:
        print(
            "Warning: persisted train/val/test artifacts were not found in "
            "standard locations; falling back to an internal random split."
        )

    questions, combined_path = load_questions(args, config, return_path=True)
    train_questions, val_questions, test_questions = split_questions(questions, config)
    manifest = _build_split_manifest(
        source="random_split_fallback",
        mc_path=str(combined_path),
        train_questions=train_questions,
        val_questions=val_questions,
        test_questions=test_questions,
        config=config,
    )
    return train_questions, val_questions, test_questions, manifest


def load_question_splits(args: argparse.Namespace, config: dict) -> tuple[list, list, list]:
    """Backward-compatible wrapper returning only the split question lists."""
    train_questions, val_questions, test_questions, _manifest = (
        load_question_splits_with_metadata(args, config)
    )
    return train_questions, val_questions, test_questions


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
    seed = data.get("seed", data.get("shuffle_seed", 42))
    train_size = data.get("train_size", data.get("train_ratio", 0.7))
    val_size = data.get("val_size", data.get("val_ratio", 0.15))

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

    # Load canonical split artifacts when they exist, otherwise fall back to
    # the legacy combined-dataset random split.
    train_questions, val_questions, test_questions, split_manifest = (
        load_question_splits_with_metadata(
        args, config
        )
    )

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

    resolved_config = yaml.safe_load(yaml.safe_dump(config))
    resolved_config.setdefault("model", {})
    resolved_config["model"]["device"] = flat_config["device"]
    resolved_config["model"]["num_choices"] = flat_config["num_choices"]
    save_json(trainer.checkpoint_dir / "config_used.json", resolved_config)
    save_json(trainer.checkpoint_dir / "split_manifest.json", split_manifest)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best PPO model saved to: {trainer.checkpoint_dir / 'best_model'}")
    print(f"Training history: {trainer.checkpoint_dir / 'history.json'}")


if __name__ == "__main__":
    main()
