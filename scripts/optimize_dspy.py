#!/usr/bin/env python3
"""Offline DSPy compile/optimize workflow.

Compiles a DSPy scorer program against quiz bowl training data.
Does NOT integrate with PPO rollouts — this is pure offline tooling.

Usage:
    python scripts/optimize_dspy.py --config configs/default.yaml
    python scripts/optimize_dspy.py --config configs/default.yaml --optimizer MIPROv2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_dspy_trainset(
    mc_questions: list,
    max_examples: int = 50,
) -> list[dict[str, Any]]:
    """Build training examples for DSPy optimization.

    Each example contains a clue prefix, option profiles, and the gold
    answer index — suitable for ``dspy.Example``.

    Parameters
    ----------
    mc_questions : list
        MC question objects with cumulative_prefixes, option_profiles,
        and gold_index.
    max_examples : int
        Cap on the number of examples.

    Returns
    -------
    list[dict]
        Training examples.
    """
    examples = []
    for q in mc_questions[:max_examples]:
        mid = len(q.cumulative_prefixes) // 2
        prefix = q.cumulative_prefixes[mid] if q.cumulative_prefixes else q.question
        examples.append({
            "clue_prefix": prefix,
            "option_profiles": q.option_profiles,
            "gold_index": q.gold_index,
        })
    return examples


def _score_metric(example, prediction, _trace=None):
    """Compare predicted scores against gold target via argmax match.

    Used as the optimization metric for DSPy ``BootstrapFewShot`` and
    ``MIPROv2``.  Returns 1.0 when the argmax of the predicted scores
    matches the argmax of the target scores, 0.0 otherwise.
    """
    try:
        pred_scores = json.loads(prediction.scores)
        target_scores = json.loads(example.scores)
    except (json.JSONDecodeError, AttributeError):
        return 0.0
    if not pred_scores or not target_scores:
        return 0.0
    return 1.0 if (
        max(range(len(pred_scores)), key=lambda i: pred_scores[i])
        == max(range(len(target_scores)), key=lambda i: target_scores[i])
    ) else 0.0


def compile_dspy_scorer(
    trainset: list[dict[str, Any]],
    dspy_config: dict[str, Any],
) -> dict[str, Any]:
    """Compile a DSPy scorer program.

    Requires the ``dspy`` package to be installed.

    Parameters
    ----------
    trainset : list[dict]
        Training examples from ``build_dspy_trainset()``.
    dspy_config : dict
        DSPy configuration section from YAML.

    Returns
    -------
    dict
        Compilation result with ``program_fingerprint`` and metadata.
    """
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "DSPy optimization requires the dspy package. "
            "Install with: pip install -e '.[dspy]'"
        ) from exc

    lm_name = dspy_config.get("model", "openai/gpt-4o-mini")
    optimizer_name = dspy_config.get("optimizer", "BootstrapFewShot")

    lm = dspy.LM(lm_name)
    dspy.configure(lm=lm)

    class MCScoreSignature(dspy.Signature):
        """Score how well each answer option matches the quiz clue."""
        clue_prefix: str = dspy.InputField(desc="partial quiz question clue text")
        options: str = dspy.InputField(desc="JSON list of answer option profile texts")
        scores: str = dspy.OutputField(desc="JSON list of float scores, one per option")

    scorer = dspy.Predict(MCScoreSignature)

    examples = []
    for ex in trainset:
        gold = ex["gold_index"]
        target_scores = [0.0] * len(ex["option_profiles"])
        target_scores[gold] = 1.0
        examples.append(dspy.Example(
            clue_prefix=ex["clue_prefix"],
            options=json.dumps(ex["option_profiles"]),
            scores=json.dumps(target_scores),
        ).with_inputs("clue_prefix", "options"))

    if optimizer_name == "MIPROv2":
        optimizer = dspy.MIPROv2(metric=_score_metric)
    else:
        optimizer = dspy.BootstrapFewShot(metric=_score_metric)

    compiled = optimizer.compile(scorer, trainset=examples)

    fingerprint = hashlib.md5(
        json.dumps(dspy_config, sort_keys=True).encode()
    ).hexdigest()[:12]

    return {
        "program_fingerprint": fingerprint,
        "optimizer": optimizer_name,
        "n_examples": len(examples),
        "compiled_program": compiled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DSPy optimization")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    from scripts._common import load_config, load_mc_questions, ARTIFACT_DIR

    config = load_config(args.config)
    dspy_cfg = config.get("dspy", {})
    if args.optimizer:
        dspy_cfg["optimizer"] = args.optimizer
    max_ex = args.max_examples or int(dspy_cfg.get("max_examples", 50))

    # Use the train split to avoid leaking val/test data into DSPy compilation
    train_path = ARTIFACT_DIR / "smoke" / "train_dataset.json"
    if not train_path.exists():
        train_path = ARTIFACT_DIR / "main" / "train_dataset.json"
    if not train_path.exists():
        # Fallback to combined dataset with warning
        train_path = ARTIFACT_DIR / "smoke" / "mc_dataset.json"
        if not train_path.exists():
            train_path = ARTIFACT_DIR / "main" / "mc_dataset.json"
        print(f"Warning: train split not found, using combined dataset: {train_path}")
    questions = load_mc_questions(train_path)
    trainset = build_dspy_trainset(questions, max_examples=max_ex)

    print(f"Built {len(trainset)} training examples")
    print(f"Compiling with {dspy_cfg.get('optimizer', 'BootstrapFewShot')}...")
    result = compile_dspy_scorer(trainset, dspy_cfg)
    print(f"Compiled. Fingerprint: {result['program_fingerprint']}")


if __name__ == "__main__":
    main()
