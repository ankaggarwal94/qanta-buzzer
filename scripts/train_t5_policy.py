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
import copy
import sys
from pathlib import Path
from typing import Any

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

# Known hazard-phase ablations (R-004). Mirrors
# ``training.hazard_pretrain._VALID_ABLATIONS`` — kept as a local literal so
# argparse never triggers the heavy lazy import of the training module.
KNOWN_HAZARD_ABLATIONS = ("shuffled_nll",)


def _seed_all_rngs(seed: int) -> None:
    """Seed the Python, NumPy, and torch global RNGs (R-001).

    Called immediately before each training phase (supervised / hazard / PPO)
    when ``--seed`` is set, so every phase starts from the same reproducible
    RNG state regardless of how much randomness earlier phases consumed.
    Never called when ``--seed`` is absent (the global RNGs stay untouched).

    Parameters
    ----------
    seed : int
        Seed applied to ``random.seed``, ``numpy.random.seed``, and
        ``torch.manual_seed`` (the latter also seeds all CUDA devices).
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] or None, optional
        Argument tokens to parse instead of ``sys.argv[1:]`` (QA-003:
        lets orchestrators round-trip a composed child argv through THIS
        parser at plan time). Default ``None`` keeps the CLI behavior.

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
        "--hazard-ablation",
        type=str,
        default=None,
        choices=list(KNOWN_HAZARD_ABLATIONS),
        help=(
            "Step-matched null-signal hazard ablation (requires "
            "--hazard-pretrain). 'shuffled_nll' permutes each question's "
            "per-prefix NLL vector before the loss."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Training seed. When set, seeds Python random, NumPy, and torch "
            "immediately before each training phase (supervised / hazard / "
            "PPO). Default None keeps the unseeded behavior. Separate from "
            "data.seed, which drives only the split."
        ),
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help=(
            "Skip the final test-set evaluation after PPO (MA-017). Used by "
            "orchestrators whose PPO phase is discarded (e.g. the hazard-"
            "efficacy shared supervised child) so the unconditional test-"
            "eval tail is not paid for a throwaway checkpoint."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides: key=value (e.g. model.model_name=t5-base)",
    )
    return parser.parse_args(argv)


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
        print(f"Limited to {max_questions} questions (data.max_questions)")

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

    def qid(q: Any) -> str:
        return getattr(q, "qid", str(q))

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


def _apply_max_questions(
    train: list, val: list, test: list,
    max_q: int, scope: str,
) -> tuple[list, list, list]:
    """Apply max_questions cap to split question lists.

    Parameters
    ----------
    scope : str
        ``"global"`` distributes the cap proportionally across splits,
        guaranteeing at least one item per non-empty split.
        ``"per_split"`` truncates each split independently (legacy).
    """
    if scope == "per_split":
        return train[:max_q], val[:max_q], test[:max_q]
    total = len(train) + len(val) + len(test)
    if total <= max_q:
        return train, val, test
    splits = [("train", train), ("val", val), ("test", test)]
    non_empty = [(name, s) for name, s in splits if s]
    if max_q < len(non_empty):
        # Refuse to silently empty val/test in global mode. The legacy
        # behaviour assigned ``[]`` to one or more splits, producing a
        # split-aware-looking run with zero held-out coverage; downstream
        # evaluation then reports zero-everything metrics indistinguishable
        # from a legitimate 0% test score (or, in smoke mode with
        # ``iterations < eval_interval``, no PPO checkpoint at all).
        non_empty_names = [name for name, _ in non_empty]
        raise ValueError(
            f"max_questions={max_q} with scope='global' is smaller than the "
            f"number of non-empty splits ({len(non_empty)}: "
            f"{non_empty_names}); this would silently empty held-out "
            "splits. Either raise data.max_questions to at least "
            f"{len(non_empty)} or set data.max_questions_scope='per_split'."
        )
    budget = max_q
    allocated = {name: 1 for name, _ in non_empty}
    budget -= len(non_empty)
    remainder_total = sum(len(s) - 1 for _, s in non_empty)
    if remainder_total > 0:
        fracs = {name: (len(s) - 1) / remainder_total for name, s in non_empty}
        raw = {name: fracs[name] * budget for name in fracs}
        floors = {name: int(raw[name]) for name in raw}
        remainders = sorted(raw.keys(), key=lambda n: raw[n] - floors[n], reverse=True)
        used = sum(floors.values())
        for name in remainders:
            if used >= budget:
                break
            floors[name] += 1
            used += 1
        for name in floors:
            allocated[name] += floors[name]
    result_map = {name: s[:allocated.get(name, 0)] for name, s in splits}
    print(
        f"  max_questions={max_q} (scope={scope}): "
        f"{len(result_map['train'])} train, "
        f"{len(result_map['val'])} val, "
        f"{len(result_map['test'])} test"
    )
    return result_map["train"], result_map["val"], result_map["test"]


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

    data_cfg = (config or {}).get("data", {})
    max_questions = data_cfg.get("max_questions", None)
    max_questions_scope = str(data_cfg.get("max_questions_scope", "global"))
    if max_questions_scope not in {"global", "per_split"}:
        # Reject typos like "globall" or "Global" before they silently
        # bypass the post-cap empty-split guard below (strict equality
        # check on "global" used to fall through for any other string).
        raise ValueError(
            f"data.max_questions_scope={max_questions_scope!r} is invalid; "
            "must be 'global' (proportional) or 'per_split' (truncate each)."
        )

    for base_dir in candidate_dirs:
        split_paths = resolve_persisted_split_paths(base_dir)
        if split_paths is None:
            continue
        train_questions = load_mc_questions(split_paths["train"])
        if not train_questions:
            print(
                f"Warning: persisted train split at {split_paths['train']} "
                "is empty; skipping this candidate and trying next."
            )
            continue
        val_questions = load_mc_questions(split_paths["val"])
        test_questions = load_mc_questions(split_paths["test"])
        cap_was_applied = bool(max_questions)
        if cap_was_applied:
            train_questions, val_questions, test_questions = _apply_max_questions(
                train_questions, val_questions, test_questions,
                max_questions, max_questions_scope,
            )
        if max_questions_scope == "global" and (not val_questions or not test_questions):
            cause = (
                f"data.max_questions={max_questions} cap left empty val/test "
                "splits"
                if cap_was_applied
                else "persisted val/test splits are already empty upstream"
            )
            raise ValueError(
                f"Held-out splits are empty under data.max_questions_scope="
                f"'global' ({cause}); refusing to proceed. Either rebuild the "
                "splits with non-empty val/test, raise data.max_questions, "
                "or set data.max_questions_scope='per_split' to bypass this "
                "guard."
            )
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
            train_path=str(split_paths["train"].resolve()),
            val_path=str(split_paths["val"].resolve()),
            test_path=str(split_paths["test"].resolve()),
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
    # ``getattr`` keeps hand-built namespaces from older callers valid; the
    # real parse_args always sets the attribute (default None).
    hazard_ablation = getattr(args, "hazard_ablation", None)
    if hazard_ablation is not None:
        if not getattr(args, "hazard_pretrain", False):
            print("ERROR: --hazard-ablation requires --hazard-pretrain")
            sys.exit(1)
        if hazard_ablation not in KNOWN_HAZARD_ABLATIONS:
            # Defense-in-depth behind argparse ``choices`` for callers that
            # build namespaces directly (R-004: unknown values fail loud).
            print(
                f"ERROR: unknown --hazard-ablation value {hazard_ablation!r}; "
                f"expected one of {list(KNOWN_HAZARD_ABLATIONS)}"
            )
            sys.exit(1)


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
        load_question_splits_with_metadata(args, config)
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

        # R-001: re-seed immediately before EACH phase (not once at the top of
        # main) so every phase starts from a reproducible RNG state no matter
        # how much randomness earlier phases consumed. Never touch the global
        # RNGs when --seed is unset.
        if args.seed is not None:
            _seed_all_rngs(args.seed)
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

    # Phase 1.5: Hazard-pretrain warm-start bridge (optional).
    # Slots between the supervised warm-start and PPO: teaches the buzz/stop
    # head *when to buzz* before PPO, then hands its checkpoint to PPO.
    if args.hazard_pretrain:
        print("\n" + "=" * 60)
        print("PHASE 1.5: HAZARD-PRETRAIN WARM-START BRIDGE")
        print("=" * 60)
        from training.hazard_pretrain import run_hazard_pretrain

        # R-001: re-seed immediately before the hazard phase.
        if args.seed is not None:
            _seed_all_rngs(args.seed)
        # DECISION: thread ``ablation`` only when set. The parameter is
        # additive (keyword-only, default None), and the pre-feature call
        # shape — no ablation kwarg at all — is pinned by
        # tests/test_hazard_pretrain.py::test_main_wires_hazard_between_supervised_and_ppo,
        # so omitting it when None keeps both contracts satisfied.
        hazard_kwargs: dict[str, Any] = {}
        if args.hazard_ablation is not None:
            hazard_kwargs["ablation"] = args.hazard_ablation
        supervised_model_path = run_hazard_pretrain(
            config=flat_config,
            train_questions=train_questions,
            pretrained_model_path=supervised_model_path,
            beta_terminal=args.beta_terminal,
            freeze_answer_head=args.freeze_answer_head,
            **hazard_kwargs,
        )
        print(f"Hazard-bridge model saved to: {supervised_model_path}")

    # Phase 2: PPO fine-tuning
    print("\n" + "=" * 60)
    print("PHASE 2: PPO FINE-TUNING (T5 Policy)")
    print("=" * 60)

    # R-001: re-seed immediately before the PPO phase.
    if args.seed is not None:
        _seed_all_rngs(args.seed)
    # MA-017: --skip-test-eval threads test_questions=None so run_ppo_training
    # skips its final test-eval tail (getattr keeps hand-built namespaces
    # from older callers valid; the real parse_args always sets the flag).
    if getattr(args, "skip_test_eval", False):
        print("Skipping final test-set evaluation (--skip-test-eval).")
        ppo_test_questions = None
    else:
        ppo_test_questions = test_questions
    model, trainer = run_ppo_training(
        config=flat_config,
        train_questions=train_questions,
        val_questions=val_questions,
        test_questions=ppo_test_questions,
        pretrained_model_path=supervised_model_path,
    )

    # ``copy.deepcopy`` preserves tuples and other types that
    # ``yaml.safe_load(yaml.safe_dump(...))`` would coerce to lists; the
    # snapshot is solely to freeze a config object before mutating it.
    resolved_config = copy.deepcopy(config)
    resolved_config.setdefault("model", {})
    resolved_config["model"]["device"] = flat_config["device"]
    resolved_config["model"]["num_choices"] = flat_config["num_choices"]
    # R-001 + R-003 producer contract: every run dir is self-describing. The
    # top-level seed (null when unset) and the four-key hazard block are
    # written on EVERY run — control arms included — so the efficacy harness
    # can diff arms on a stable key set.
    resolved_config["seed"] = args.seed
    resolved_config["hazard"] = {
        "pretrain": bool(args.hazard_pretrain),
        "beta_terminal": float(args.beta_terminal),
        "freeze_answer_head": bool(args.freeze_answer_head),
        "ablation": args.hazard_ablation,
    }
    save_json(trainer.checkpoint_dir / "config_used.json", resolved_config)
    save_json(trainer.checkpoint_dir / "split_manifest.json", split_manifest)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best PPO model saved to: {trainer.checkpoint_dir / 'best_model'}")
    print(f"Training history: {trainer.checkpoint_dir / 'history.json'}")


if __name__ == "__main__":
    main()
