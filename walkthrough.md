# Quiz Bowl RL Buzzer - Code Walkthrough

*2026-03-06T21:02:09Z by Showboat 0.6.1*
<!-- showboat-id: 6c5491ff-bc49-40c4-90db-b786c15beead -->

## 1. Repo orientation

This walkthrough reads the repo in the same order a contributor would use it: first the canonical modular pipeline, then the deeper subsystems, then the older prototype files that still live at the repo root.

The current source of truth is the modular stack under `qb_data/`, `models/`, `qb_env/`, `agents/`, `evaluation/`, `scripts/`, and `training/`. The root-level prototype files still matter because they preserve the older qanta-buzzer implementation and explain where the T5 policy track came from.

The next commands show the top-level layout and the repo's own guidance so the rest of the walkthrough has a clear map.

```bash
rg --files . -g '!venv/**' -g '!**/__pycache__/**' -g '!artifacts/smoke/**' | sort | sed -n '1,160p'

```

```output
./CLAUDE.md
./IMPLEMENTATION_README.md
./PROJECT_OVERVIEW.md
./README.md
./agents/__init__.py
./agents/bayesian_buzzer.py
./agents/ppo_buzzer.py
./agents/softmax_profile_buzzer.py
./agents/threshold_buzzer.py
./config.py
./configs/default.yaml
./configs/smoke.yaml
./configs/t5_policy.yaml
./data/processed/.gitkeep
./data/processed_dataset.json
./data/test
./data/test_dataset.json
./data/test_questions.csv
./data/train_dataset.json
./data/val_dataset.json
./dataset.py
./demo.py
./environment.py
./evaluation/__init__.py
./evaluation/controls.py
./evaluation/metrics.py
./evaluation/plotting.py
./main.py
./metrics.py
./model.py
./models/__init__.py
./models/answer_profiles.py
./models/features.py
./models/likelihoods.py
./models/t5_policy.py
./pyproject.toml
./qb_data/__init__.py
./qb_data/answer_profiles.py
./qb_data/config.py
./qb_data/data_loader.py
./qb_data/dataset_splits.py
./qb_data/huggingface_loader.py
./qb_data/mc_builder.py
./qb_data/text_utils.py
./qb_env/__init__.py
./qb_env/data_loader.py
./qb_env/mc_builder.py
./qb_env/text_utils.py
./qb_env/text_wrapper.py
./qb_env/tossup_env.py
./questions.csv
./requirements.txt
./run.sh
./scripts/__init__.py
./scripts/_common.py
./scripts/build_mc_dataset.py
./scripts/compare_policies.py
./scripts/evaluate_all.py
./scripts/run_baselines.py
./scripts/test_mc_builder.py
./scripts/train_ppo.py
./scripts/train_t5_policy.py
./test_csv_loader.py
./test_imports.py
./tests/__init__.py
./tests/conftest.py
./tests/test_agents.py
./tests/test_environment.py
./tests/test_factories.py
./tests/test_features.py
./tests/test_likelihoods.py
./tests/test_metrics.py
./tests/test_ppo_buzzer.py
./tests/test_ppo_t5.py
./tests/test_qb_rl_bridge.py
./tests/test_supervised_t5.py
./tests/test_t5_policy.py
./tests/test_text_wrapper.py
./train_ppo.py
./train_supervised.py
./training/__init__.py
./training/train_ppo_t5.py
./training/train_supervised_t5.py
./verify_data_loader.py
./visualize.py
./walkthrough.md
```

```bash
sed -n '1,200p' README.md

```

````output
# Quiz Bowl RL Buzzer (Unified)

Unified CS234 final project codebase for quiz bowl buzzing under incremental clues.

This repo keeps `qanta-buzzer` as the canonical implementation while preserving a qb-rl compatibility bridge:

- Modular belief-feature pipeline with `qb_data/`, `qb_env/`, `models/`, `agents/`, `evaluation/`, and `scripts/`
- T5 likelihood and T5 policy training tracks from the original qanta-buzzer work
- qb-rl-compatible import/config shims for older notebooks and scripts
- Optional OpenAI embedding support for `likelihood.model: openai` and `data.distractor_strategy: openai_profile`

## Setup

Preferred development install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional OpenAI support:

```bash
pip install -e .[openai]
export OPENAI_API_KEY=...
```

Legacy install remains available:

```bash
pip install -r requirements.txt
```

## Main Workflows

Belief-feature / PPO pipeline:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

T5 policy track:

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/compare_policies.py --config configs/t5_policy.yaml
```

Drop `--smoke` for full runs.

## Config Notes

- Current canonical config style is CSV-first with optional Hugging Face fallback.
- qb-rl aliases are also supported:
  - `data.dataset`, `data.dataset_config`, `data.dataset_smoke`, `data.dataset_smoke_config`, `data.split`
  - `likelihood.sbert_name`, `likelihood.openai_model`
  - `environment.reward` as an alias for `reward_mode`
- `--smoke` keeps the existing `configs/smoke.yaml` behavior, but qb-rl-style `dataset_smoke*` keys are honored when present in a loaded config.

## Compatibility Bridge

These old qb-rl import paths now resolve in this repo:

- `qb_env.data_loader`
- `qb_env.mc_builder`
- `qb_env.text_utils`
- `models.answer_profiles`
- `agents.softmax_profile_buzzer`

The bridge is additive. `qb_data/` remains the canonical home for data loading and MC construction.

## Testing

The repo has a real pytest suite under `tests/`, including:

- likelihood models and factories
- environment behavior
- baseline agents and PPO wrapper
- T5 policy components
- qb-rl compatibility bridge and mocked OpenAI coverage

Run a narrow test slice while iterating:

```bash
pytest tests/test_qb_rl_bridge.py tests/test_factories.py tests/test_ppo_buzzer.py
```

Run the full suite when needed:

```bash
pytest
```
````

```bash
sed -n '1,160p' CLAUDE.md

```

````output
# CLAUDE.md

This file provides repo-local guidance for Claude Code and other coding agents.

## Project Overview

Stanford CS234 final project: a unified quiz bowl RL buzzer system with two tracks:

1. Belief-feature pipeline: build MC tossups, score answer profiles with TF-IDF / SBERT / T5 / optional OpenAI embeddings, train or compare buzzers, and evaluate with S_q plus calibration metrics.
2. T5 policy pipeline: supervised warm-start and PPO for an end-to-end text policy.

`qanta-buzzer` is the canonical repo. qb-rl compatibility is preserved through additive shims rather than structural rewrites.

## Setup

Preferred development install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional OpenAI support:

```bash
pip install -e .[openai]
export OPENAI_API_KEY=...
```

Legacy setup still works:

```bash
pip install -r requirements.txt
```

## High-Value Commands

Belief-feature smoke workflow:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

T5 policy workflow:

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/compare_policies.py --config configs/t5_policy.yaml
```

## Testing

There is a formal pytest suite in `tests/`.

Focused bridge/runtime checks:

```bash
pytest tests/test_qb_rl_bridge.py tests/test_factories.py tests/test_ppo_buzzer.py
```

Full suite:

```bash
pytest
```

## Architecture

- `qb_data/`: canonical data loading, answer profiles, stratified splits, MC construction
- `qb_env/`: Gymnasium environment plus text wrapper and qb-rl compatibility shims
- `models/`: likelihood models, belief features, T5 policy model, compatibility exports
- `agents/`: threshold, softmax-profile, sequential Bayes, PPO wrapper
- `evaluation/`: S_q, calibration, controls, plotting
- `scripts/`: pipeline entrypoints and shared helpers
- `training/`: T5 policy supervised + PPO trainers

## Compatibility Notes

- qb-rl config aliases are supported in addition to the canonical qanta-buzzer YAML shape.
- Old qb-rl imports like `qb_env.data_loader` and `models.answer_profiles` are thin re-exports over the canonical modules.
- OpenAI support is opt-in only. Default local workflows stay offline-friendly and do not require the `openai` package or `OPENAI_API_KEY`.
````

## 2. Canonical end-to-end flow

The canonical workflow is a four-stage pipeline.

1. `scripts/build_mc_dataset.py` loads raw tossups, builds answer profiles, constructs multiple-choice variants, and writes processed datasets.
2. `scripts/run_baselines.py` runs the non-learning agents to establish comparison floors.
3. `scripts/train_ppo.py` trains the belief-feature PPO policy.
4. `scripts/evaluate_all.py` computes metrics, runs controls, and emits plots and summary artifacts.

This section stays at orchestration level. Later sections drill into the implementation behind each stage.

```bash
sed -n '1,60p' scripts/build_mc_dataset.py

```

```output
#!/usr/bin/env python3
"""
Build multiple-choice dataset from QANTA quiz bowl questions.

This script orchestrates the complete data pipeline:
1. Load questions from CSV or HuggingFace
2. Build answer profiles from training data
3. Generate MC questions with anti-artifact guards
4. Create stratified train/val/test splits
5. Save processed datasets as JSON

Usage:
    python scripts/build_mc_dataset.py
    python scripts/build_mc_dataset.py --smoke  # Quick test with 50 questions
    python scripts/build_mc_dataset.py --config configs/custom.yaml
    python scripts/build_mc_dataset.py --data.K=5 --data.distractor_strategy=tfidf_profile
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qb_data import TossupQuestion
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.config import load_config, merge_overrides, resolve_data_loading_options
from qb_data.data_loader import QANTADatasetLoader
from qb_data.dataset_splits import create_stratified_splits
from qb_data.huggingface_loader import load_from_huggingface
from qb_data.mc_builder import MCBuilder, MCQuestion


def parse_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Parse CLI override arguments into nested dictionary.

    Converts args like --data.K=5 into {"data": {"K": 5}}

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    Dict[str, Any]
        Nested dictionary of config overrides
    """
    overrides = {}

    # Check for any attributes that look like overrides (contain dots)
    for key, value in vars(args).items():
        if value is not None and '.' not in key:
            continue  # Skip non-override args

```

```bash
rg -n "resolve_data_loading_options|AnswerProfileBuilder|MCBuilder|create_stratified_splits|save_json\(|print_statistics\(" scripts/build_mc_dataset.py

```

```output
30:from qb_data.answer_profiles import AnswerProfileBuilder
31:from qb_data.config import load_config, merge_overrides, resolve_data_loading_options
33:from qb_data.dataset_splits import create_stratified_splits
35:from qb_data.mc_builder import MCBuilder, MCQuestion
97:def save_json(path: Path, data: List[Any]) -> None:
124:def print_statistics(
128:    profile_builder: Optional[AnswerProfileBuilder] = None,
129:    mc_builder: Optional[MCBuilder] = None
142:    profile_builder : Optional[AnswerProfileBuilder]
144:    mc_builder : Optional[MCBuilder]
249:    data_opts = resolve_data_loading_options(config, smoke=args.smoke)
281:    profile_builder = AnswerProfileBuilder(
290:    mc_builder = MCBuilder(
318:    train, val, test = create_stratified_splits(mc_questions, ratios=ratios)
322:    save_json(output_dir / "mc_dataset.json", mc_questions)
323:    save_json(output_dir / "train_dataset.json", train)
324:    save_json(output_dir / "val_dataset.json", val)
325:    save_json(output_dir / "test_dataset.json", test)
341:    print_statistics(train, val, test, profile_builder, mc_builder)
```

```bash
sed -n '1,70p' scripts/run_baselines.py

```

```output
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

from agents.bayesian_buzzer import SequentialBayesBuzzer, SoftmaxProfileBuzzer
from agents.threshold_buzzer import AlwaysBuzzFinalBuzzer, sweep_thresholds
from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
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
```

```bash
rg -n "SoftmaxProfileBuzzer|SequentialBayesBuzzer|AlwaysBuzzFinalBuzzer|sweep_thresholds|save_json\(|baseline_summary" scripts/run_baselines.py

```

```output
7:2. SoftmaxProfileBuzzer -- softmax belief from scratch at each step
8:3. SequentialBayesBuzzer -- Bayesian belief update with sequential fragments
9:4. AlwaysBuzzFinalBuzzer -- always waits until last clue, then buzzes
35:from agents.bayesian_buzzer import SequentialBayesBuzzer, SoftmaxProfileBuzzer
36:from agents.threshold_buzzer import AlwaysBuzzFinalBuzzer, sweep_thresholds
134:    threshold_runs = sweep_thresholds(
158:        softmax_agent = SoftmaxProfileBuzzer(
168:        seq_agent = SequentialBayesBuzzer(
180:    floor_agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood_model, beta=beta)
186:    save_json(out_dir / "baseline_threshold_runs.json", threshold_payload)
187:    save_json(out_dir / "baseline_softmax_profile_runs.json", softmax_payload)
188:    save_json(out_dir / "baseline_sequential_bayes_runs.json", sequential_payload)
189:    save_json(out_dir / "baseline_floor_runs.json", floor_runs)
197:    save_json(out_dir / "baseline_summary.json", summary)
```

```bash
sed -n '1,70p' scripts/train_ppo.py

```

```output
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
from qb_env.tossup_env import make_env_from_config
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
        "--deterministic-eval", action="store_true",
        help="Use deterministic policy for post-training episode evaluation.",
```

```bash
rg -n "PPOBuzzer|make_env_from_config|build_likelihood_model|save_json\(|ppo_summary" scripts/train_ppo.py

```

```output
30:from agents.ppo_buzzer import PPOBuzzer
32:from qb_env.tossup_env import make_env_from_config
35:    build_likelihood_model,
97:    likelihood_model = build_likelihood_model(config, mc_questions)
98:    env = make_env_from_config(
110:    agent = PPOBuzzer(
132:    save_json(out_dir / "ppo_runs.json", traces)
133:    save_json(out_dir / "ppo_summary.json", summary)
```

```bash
sed -n '1,80p' scripts/evaluate_all.py

```

```output
#!/usr/bin/env python3
"""
Comprehensive evaluation with control experiments and visualization.

Runs the SoftmaxProfileBuzzer at the best threshold (from baseline sweep),
then executes control experiments (choices-only, shuffle, alias substitution)
and generates comparison plots and tables for the CS234 writeup.

Consumes outputs from:
- build_mc_dataset.py (mc_dataset.json, alias_lookup.json)
- run_baselines.py (baseline_summary.json)
- train_ppo.py (ppo_summary.json)

Produces:
- evaluation_report.json (full eval + controls + baseline + PPO summaries)
- plots/entropy_vs_clue.png
- plots/calibration.png
- plots/comparison.csv

Usage:
    python scripts/evaluate_all.py --smoke
    python scripts/evaluate_all.py --config configs/custom.yaml
    python scripts/evaluate_all.py --mc-path artifacts/main/mc_dataset.json

Ported from qb-rl reference implementation (scripts/evaluate_all.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bayesian_buzzer import SoftmaxProfileBuzzer
from evaluation.controls import (
    run_alias_substitution_control,
    run_choices_only_control,
    run_shuffle_control,
)
from evaluation.metrics import (
    calibration_at_buzz,
    per_category_accuracy,
    summarize_buzz_metrics,
)
from evaluation.plotting import (
    plot_calibration_curve,
    plot_entropy_vs_clue_index,
    save_comparison_table,
)
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
    load_json,
    load_mc_questions,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with config, smoke, and mc_path fields.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate all agents and controls."
    )
    parser.add_argument(
        "--config", type=str, default=None,
```

```bash
rg -n "run_choices_only_control|run_shuffle_control|run_alias_substitution_control|plot_calibration_curve|plot_entropy_vs_clue_index|evaluation_report" scripts/evaluate_all.py

```

```output
15:- evaluation_report.json (full eval + controls + baseline + PPO summaries)
44:    run_alias_substitution_control,
45:    run_choices_only_control,
46:    run_shuffle_control,
54:    plot_calibration_curve,
55:    plot_entropy_vs_clue_index,
203:    shuffle_eval = run_shuffle_control(
208:    alias_eval = run_alias_substitution_control(
215:    choices_only = run_choices_only_control(mc_questions)
240:    save_json(out_dir / "evaluation_report.json", report)
258:    plot_entropy_vs_clue_index(
270:    plot_calibration_curve(
313:    print(f"Wrote evaluation report to: {out_dir / 'evaluation_report.json'}")
```

```bash
find artifacts/smoke -maxdepth 2 -type f | sort

```

```output
artifacts/smoke/answer_profiles.json
artifacts/smoke/baseline_floor_runs.json
artifacts/smoke/baseline_sequential_bayes_runs.json
artifacts/smoke/baseline_softmax_profile_runs.json
artifacts/smoke/baseline_summary.json
artifacts/smoke/baseline_threshold_runs.json
artifacts/smoke/evaluation_report.json
artifacts/smoke/mc_dataset.json
artifacts/smoke/plots/calibration.png
artifacts/smoke/plots/comparison.csv
artifacts/smoke/plots/entropy_vs_clue.png
artifacts/smoke/ppo_model.zip
artifacts/smoke/ppo_runs.json
artifacts/smoke/ppo_summary.json
artifacts/smoke/test_dataset.json
artifacts/smoke/train_dataset.json
artifacts/smoke/val_dataset.json
```

## 3. Data and config layer

Everything below the scripts starts with configuration and data normalization. The data layer turns raw question rows into typed objects, then into answer profiles, then into multiple-choice tossups, then into stratified train/val/test splits.

The important design choice is that the repo is CSV-first with optional Hugging Face fallback, and all multiple-choice construction is guarded against easy answer-space artifacts.

```bash
rg -n "def normalize_config|def resolve_data_loading_options|def load_config|environment.reward|dataset_smoke|openai_model|sbert_name" qb_data/config.py

```

```output
13:def normalize_config(
40:    if smoke and data_cfg.get("dataset_smoke") and "dataset" not in data_cfg:
41:        data_cfg["dataset"] = data_cfg["dataset_smoke"]
42:    if smoke and data_cfg.get("dataset_smoke_config") and "dataset_config" not in data_cfg:
43:        data_cfg["dataset_config"] = data_cfg["dataset_smoke_config"]
45:    if "embedding_model" in lik_cfg and "sbert_name" not in lik_cfg:
46:        lik_cfg["sbert_name"] = lik_cfg["embedding_model"]
47:    if "sbert_name" in lik_cfg and "embedding_model" not in lik_cfg:
48:        lik_cfg["embedding_model"] = lik_cfg["sbert_name"]
53:def resolve_data_loading_options(
74:        for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
86:        dataset = data_cfg.get("dataset_smoke", dataset)
87:        dataset_config = data_cfg.get("dataset_smoke_config", dataset_config)
97:        "uses_dataset_smoke": use_smoke_dataset,
101:def load_config(
144:                for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
327:def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
```

```bash
sed -n '1,220p' qb_data/config.py

```

```output
"""Configuration loading and management utilities.

Provides functions to load YAML configurations, apply small
cross-codebase compatibility normalizations, and merge CLI overrides
using dot notation (e.g., ``data.K=5`` updates ``config["data"]["K"]``).
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union


def normalize_config(
    config: Dict[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    """Apply compatibility defaults to a loaded configuration.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    smoke : bool
        Whether the caller intends to run in smoke mode.

    Returns
    -------
    dict
        Normalized configuration dictionary.
    """
    data_cfg = config.setdefault("data", {})
    env_cfg = config.setdefault("environment", {})
    lik_cfg = config.setdefault("likelihood", {})

    if "reward" in env_cfg and "reward_mode" not in env_cfg:
        env_cfg["reward_mode"] = env_cfg["reward"]
    elif "reward_mode" in env_cfg and "reward" not in env_cfg:
        env_cfg["reward"] = env_cfg["reward_mode"]

    if smoke and data_cfg.get("dataset_smoke") and "dataset" not in data_cfg:
        data_cfg["dataset"] = data_cfg["dataset_smoke"]
    if smoke and data_cfg.get("dataset_smoke_config") and "dataset_config" not in data_cfg:
        data_cfg["dataset_config"] = data_cfg["dataset_smoke_config"]

    if "embedding_model" in lik_cfg and "sbert_name" not in lik_cfg:
        lik_cfg["sbert_name"] = lik_cfg["embedding_model"]
    if "sbert_name" in lik_cfg and "embedding_model" not in lik_cfg:
        lik_cfg["embedding_model"] = lik_cfg["sbert_name"]

    return config


def resolve_data_loading_options(
    config: Dict[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    """Resolve CSV/Hugging Face data-loading options from a config dict.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    smoke : bool
        Whether the caller intends to run in smoke mode.

    Returns
    -------
    dict
        Resolved data-loading settings.
    """
    data_cfg = config.get("data", {})
    use_smoke_dataset = smoke and any(
        data_cfg.get(key) is not None
        for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
    )

    csv_path = data_cfg.get("csv_path")
    if smoke and data_cfg.get("csv_smoke_path"):
        csv_path = data_cfg["csv_smoke_path"]

    dataset = data_cfg.get("dataset")
    dataset_config = data_cfg.get("dataset_config")
    split = data_cfg.get("split", "eval")

    if use_smoke_dataset:
        dataset = data_cfg.get("dataset_smoke", dataset)
        dataset_config = data_cfg.get("dataset_smoke_config", dataset_config)
        split = data_cfg.get("split_smoke", split)

    return {
        "csv_path": csv_path,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "use_huggingface": bool(data_cfg.get("use_huggingface", False) or dataset),
        "max_questions": data_cfg.get("max_questions"),
        "uses_dataset_smoke": use_smoke_dataset,
    }


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration file. Defaults to configs/default.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    ImportError
        If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config loading. "
            "Install it with: pip install pyyaml"
        )

    # Default to configs/default.yaml if no path given
    if config_path is None:
        project_root = Path(__file__).parent.parent
        default_path = project_root / "configs" / "default.yaml"
        smoke_path = project_root / "configs" / "smoke.yaml"

        if smoke and default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                default_config = yaml.safe_load(f) or {}
            default_data = default_config.get("data", {})
            if any(
                default_data.get(key) is not None
                for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
            ):
                config_path = default_path
            elif smoke_path.exists():
                config_path = smoke_path
            else:
                config_path = default_path
        else:
            config_path = default_path
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return normalize_config(config or {}, smoke=smoke)


def merge_overrides(
    config: Dict[str, Any],
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge override values into configuration using dot notation.

    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    overrides : dict
        Override values to merge. Keys can use dot notation
        (e.g., {"data.K": 5} updates config["data"]["K"]).

    Returns
    -------
    dict
        Updated configuration with overrides applied.

    Examples
    --------
    >>> config = {"data": {"K": 4}, "ppo": {"batch_size": 32}}
    >>> overrides = {"data.K": 5, "ppo.batch_size": 16}
    >>> config = merge_overrides(config, overrides)
    >>> assert config["data"]["K"] == 5
    >>> assert config["ppo"]["batch_size"] == 16
    """
    for key, value in overrides.items():
        # Split on dots for nested keys
        keys = key.split(".")

        # Navigate to the nested location
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the final value
        final_key = keys[-1]
        current[final_key] = value

    return normalize_config(config)


def build_argparse_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to configuration overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
```

```bash
sed -n '1,220p' qb_data/data_loader.py

```

```output
"""
Data structures and loaders for quiz bowl questions.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

from qb_data.text_utils import normalize_answer


@dataclass
class TossupQuestion:
    """
    A quiz bowl tossup question with incremental clues.

    Attributes
    ----------
    qid : str
        Unique question identifier
    question : str
        Full question text (all clues concatenated)
    tokens : List[str]
        Tokenized question split on whitespace
    answer_primary : str
        Primary answer text
    clean_answers : List[str]
        List of acceptable answer variants
    run_indices : List[int]
        Token indices where clues end (for incremental reveal)
    human_buzz_positions : Optional[List[Tuple[int, int]]]
        Human buzzer positions as (position, count) tuples
    category : str
        Question category (e.g., "History", "Literature")
    cumulative_prefixes : List[str]
        Precomputed text prefixes at each run_index
    """
    qid: str
    question: str
    tokens: List[str]
    answer_primary: str
    clean_answers: List[str]
    run_indices: List[int]
    human_buzz_positions: Optional[List[Tuple[int, int]]]
    category: str
    cumulative_prefixes: List[str]


def _parse_clues_to_tokens(clues: List[str]) -> Tuple[List[str], List[int]]:
    """
    Convert list of clues to tokens and run indices.

    Parameters
    ----------
    clues : List[str]
        List of clue strings

    Returns
    -------
    Tuple[List[str], List[int]]
        Tokens (words) and indices where each clue ends
    """
    tokens = []
    run_indices = []

    for clue in clues:
        clue_tokens = clue.split()
        tokens.extend(clue_tokens)
        if clue_tokens:  # Only add index if clue has tokens
            run_indices.append(len(tokens) - 1)

    return tokens, run_indices


def _generate_qid(text: str) -> str:
    """
    Generate a unique question ID from question text.

    Parameters
    ----------
    text : str
        Question text to hash

    Returns
    -------
    str
        Unique identifier based on text hash
    """
    hash_obj = hashlib.md5(text.encode('utf-8'))
    return f"qid-{hash_obj.hexdigest()[:12]}"


def _coerce_human_buzz_positions(value: Any) -> Optional[List[Tuple[int, int]]]:
    """Coerce various metadata formats into ``(position, count)`` tuples."""
    if value is None:
        return None

    if isinstance(value, list):
        result: List[Tuple[int, int]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    result.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, dict):
                pos = item.get("position")
                count = item.get("count", 1)
                if pos is None:
                    continue
                try:
                    result.append((int(pos), int(count)))
                except (TypeError, ValueError):
                    continue
        return result or None

    return None


def _coerce_run_indices(run_indices: Any, token_count: int) -> List[int]:
    """Validate and coerce run indices into a sorted unique list."""
    clean: List[int] = []
    for idx in run_indices or []:
        try:
            clean.append(int(idx))
        except (TypeError, ValueError):
            continue

    if not clean:
        if token_count <= 0:
            raise ValueError("question must contain at least one token")
        clean = list(range(token_count))

    clean = sorted(set(clean))
    if clean[0] < 0 or clean[-1] > token_count - 1:
        raise ValueError(
            f"run_indices out of bounds: min={clean[0]} max={clean[-1]} token_count={token_count}"
        )
    return clean


def parse_row(row: Dict[str, Any]) -> TossupQuestion:
    """Parse a qb-rl/HuggingFace-style row into ``TossupQuestion``."""
    question = str(row["question"])
    tokens = question.split()
    metadata = row.get("metadata", {}) or {}
    answer_primary = str(
        row.get("answer_primary") or (row.get("clean_answers") or [""])[0]
    ).strip()
    clean_answers = [str(x) for x in (row.get("clean_answers") or [])]
    if not clean_answers and answer_primary:
        clean_answers = [answer_primary]

    run_indices = _coerce_run_indices(
        row.get("run_indices") or [],
        token_count=len(tokens),
    )

    normalized_question = " ".join(question.split())
    normalized_tokens = " ".join(tokens)
    if normalized_tokens != normalized_question:
        raise ValueError("tokenization roundtrip mismatch")
    if max(run_indices) > len(tokens) - 1:
        raise ValueError("run_indices out of bounds")

    cumulative_prefixes = [" ".join(tokens[: idx + 1]) for idx in run_indices]
    category = str(metadata.get("category") or row.get("category") or "")
    human_buzz_positions = _coerce_human_buzz_positions(
        metadata.get("human_buzz_positions") or row.get("human_buzz_positions")
    )

    qid_raw = row.get("qid") or row.get("question_id") or row.get("id")
    if qid_raw is None:
        qid_raw = _generate_qid(question)

    return TossupQuestion(
        qid=str(qid_raw),
        question=question,
        tokens=tokens,
        answer_primary=answer_primary,
        clean_answers=clean_answers,
        run_indices=run_indices,
        human_buzz_positions=human_buzz_positions,
        category=category,
        cumulative_prefixes=cumulative_prefixes,
    )


def load_tossup_questions(
    dataset: str,
    dataset_config: Optional[str] = None,
    split: str = "eval",
    limit: Optional[int] = None,
) -> List[TossupQuestion]:
    """Load tossup questions from Hugging Face datasets using qb-rl semantics."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for Hugging Face loading. Install it with: pip install datasets"
        ) from exc

    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split)
    else:
        ds = load_dataset(dataset, split=split)

    if limit is not None:
        ds = ds.select(range(min(int(limit), len(ds))))

    return [parse_row(dict(row)) for row in ds]


def load_tossup_questions_from_config(
    config: Dict[str, Any],
    smoke: bool = False,
) -> List[TossupQuestion]:
    """Load tossups from config, supporting qb-rl and qanta-buzzer keys."""
```

```bash
venv/bin/python - <<'PY'
from qb_data.data_loader import QANTADatasetLoader
qs = QANTADatasetLoader().load_from_csv('data/test_questions.csv')
q = qs[0]
print('loaded_questions =', len(qs))
print('qid =', q.qid)
print('answer_primary =', q.answer_primary)
print('category =', q.category)
print('run_indices =', q.run_indices)
print('num_prefixes =', len(q.cumulative_prefixes))
print('first_prefix =', q.cumulative_prefixes[0][:120])
PY

```

```output
loaded_questions = 10
qid = hist_001
answer_primary = Napoleon Bonaparte
category = History
run_indices = [15, 26, 39, 50]
num_prefixes = 4
first_prefix = This military leader rose to power during the French Revolution and became First Consul in 1799.
```

```bash
sed -n '1,220p' qb_data/huggingface_loader.py

```

```output
"""
HuggingFace dataset loader for quiz bowl data.

This module provides fallback loading from HuggingFace Hub when local CSV files
are not available.
"""

from typing import List, Optional, Dict, Any

from qb_data.data_loader import TossupQuestion
from qb_data.text_utils import tokenize_text, normalize_answer


def load_from_huggingface(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "eval"
) -> List[TossupQuestion]:
    """
    Load quiz bowl dataset from HuggingFace Hub.

    Parameters
    ----------
    dataset_name : str
        Name of the HuggingFace dataset (e.g., "qanta-challenge/acf-co24-tossups")
    config_name : Optional[str]
        Configuration name for the dataset (e.g., "questions", "tossup")
    split : str
        Dataset split to load (default: "eval")

    Returns
    -------
    List[TossupQuestion]
        List of parsed questions

    Raises
    ------
    ImportError
        If datasets library is not installed
    ValueError
        If dataset not found or required fields missing
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets library not installed. Falling back to CSV loader.")
        print("Install with: pip install datasets")
        raise ImportError("HuggingFace datasets library not available. Please use CSV fallback.")

    # Known dataset configurations from qb-rl
    known_configs = {
        "qanta-challenge/acf-co24-tossups": "questions",
        "qanta-challenge/qanta25-playground": "tossup"
    }

    # Use known config if not provided
    if config_name is None and dataset_name in known_configs:
        config_name = known_configs[dataset_name]
        print(f"Using known config '{config_name}' for {dataset_name}")

    # Try to load dataset
    try:
        print(f"Loading {dataset_name} from HuggingFace Hub...")
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        print(f"Successfully loaded {len(dataset)} questions")
    except Exception as e:
        error_msg = f"Failed to load dataset {dataset_name}: {e}"
        print(f"Error: {error_msg}")
        print("Falling back to local CSV loader...")
        raise ValueError(error_msg)

    # Parse dataset rows into TossupQuestion format
    questions = []
    for idx, row in enumerate(dataset):
        try:
            question = parse_huggingface_row(row, idx)
            questions.append(question)
        except KeyError as e:
            print(f"Warning: Skipping row {idx} due to missing field: {e}")
            continue
        except Exception as e:
            print(f"Warning: Failed to parse row {idx}: {e}")
            continue

    if not questions:
        raise ValueError(f"No valid questions parsed from {dataset_name}")

    print(f"Parsed {len(questions)} questions from HuggingFace dataset")
    return questions


def parse_huggingface_row(row: Dict[str, Any], idx: int = 0) -> TossupQuestion:
    """
    Parse a HuggingFace dataset row into TossupQuestion format.

    Parameters
    ----------
    row : Dict[str, Any]
        Single row from HuggingFace dataset
    idx : int
        Row index for generating IDs

    Returns
    -------
    TossupQuestion
        Parsed question object

    Raises
    ------
    KeyError
        If required fields are missing
    """
    # Field mapping for different dataset formats
    # Primary fields
    question_fields = ["question", "text", "question_text", "tossup_text"]
    answer_fields = ["answer_primary", "answer", "clean_answer", "clean_answers", "page"]
    category_fields = ["category", "topic", "subject"]

    # Extract question text
    question_text = None
    for field in question_fields:
        if field in row:
            question_text = row[field]
            break

    if not question_text:
        raise KeyError(f"No question field found. Available fields: {list(row.keys())}")

    # Extract answer
    answer_text = None
    for field in answer_fields:
        if field in row:
            value = row[field]
            # Handle list of answers
            if isinstance(value, list) and value:
                answer_text = value[0]
            elif isinstance(value, str):
                answer_text = value
            break

    if not answer_text:
        raise KeyError(f"No answer field found. Available fields: {list(row.keys())}")

    # Extract category (with default)
    category = "General"
    for field in category_fields:
        if field in row and row[field]:
            category = str(row[field])
            break

    # Generate ID if not present
    qid = row.get("qid") or row.get("id") or row.get("qanta_id") or f"hf_{idx:06d}"

    # Handle clues that may be separated by ||| or in a list
    if "|||" in question_text:
        # QANTA format with ||| separators
        clues = question_text.split("|||")
        question_text = " ".join(clues)
    elif isinstance(question_text, list):
        # List of clues
        clues = question_text
        question_text = " ".join(clues)
    else:
        # Single text, split by sentences as approximation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', question_text)
        clues = sentences if len(sentences) > 1 else [question_text]

    # Tokenize text
    tokens = tokenize_text(question_text)

    # Build run indices (boundaries between clues)
    run_indices = []
    current_pos = 0
    for clue in clues:
        clue_tokens = tokenize_text(clue)
        current_pos += len(clue_tokens)
        if current_pos > 0:
            run_indices.append(current_pos - 1)  # Index is 0-based

    # Build cumulative prefixes
    cumulative_prefixes = []
    for idx in run_indices:
        prefix = " ".join(tokens[:idx + 1])
        cumulative_prefixes.append(prefix)

    # Normalize answer for matching
    clean_answers = [normalize_answer(answer_text)]

    return TossupQuestion(
        qid=qid,
        question=question_text,
        tokens=tokens,
        answer_primary=answer_text,  # Keep original answer as primary
        clean_answers=clean_answers,  # Normalized version for matching
        run_indices=run_indices,
        human_buzz_positions=None,  # Not available from HuggingFace
        category=category,
        cumulative_prefixes=cumulative_prefixes
    )


def try_huggingface_fallback(csv_path: str) -> Optional[List[TossupQuestion]]:
    """
    Attempt to load from HuggingFace if CSV is missing.

    Parameters
    ----------
    csv_path : str
        Path to missing CSV file

    Returns
    -------
    Optional[List[TossupQuestion]]
        Questions if HuggingFace load succeeds, None otherwise
    """
    print(f"CSV file {csv_path} not found. Attempting HuggingFace fallback...")
```

```bash
sed -n '1,180p' qb_data/answer_profiles.py; printf '\n'

```

```output
"""Answer profile builder with leave-one-out exclusion for quiz bowl questions."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from qb_data.data_loader import TossupQuestion


class AnswerProfileBuilder:
    """Builds profiles for answers by aggregating question texts.

    The profile for an answer is created by concatenating all question texts
    that have that answer. When building profiles for distractors, we use
    all questions. For the gold answer, we exclude the current question to
    prevent information leakage (leave-one-out).

    Attributes:
        max_tokens_per_profile: Maximum number of tokens to keep in each profile.
        min_questions_per_answer: Minimum questions needed to build a profile.
        _grouped: Dictionary mapping answer_primary to list of (qid, question_text) tuples.
    """

    def __init__(
        self,
        max_tokens_per_profile: int = 2000,
        min_questions_per_answer: int = 1
    ):
        """Initialize the answer profile builder.

        Args:
            max_tokens_per_profile: Maximum tokens to keep in each profile.
            min_questions_per_answer: Minimum questions needed to build a profile.
        """
        self.max_tokens_per_profile = max_tokens_per_profile
        self.min_questions_per_answer = min_questions_per_answer
        self._grouped: Dict[str, List[Tuple[str, str]]] = {}

    def fit(self, questions: List[TossupQuestion]) -> "AnswerProfileBuilder":
        """Fit the builder on a set of questions.

        Groups questions by their primary answer for efficient profile building.

        Args:
            questions: List of tossup questions to group by answer.

        Returns:
            Self for method chaining.
        """
        grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for q in questions:
            # Store qid and full question text for each answer
            grouped[q.answer_primary].append((q.qid, q.question))
        self._grouped = dict(grouped)
        return self

    def _profile_text(
        self,
        answer_primary: str,
        exclude_qid: Optional[str] = None
    ) -> str:
        """Build profile text for an answer with optional exclusion.

        Args:
            answer_primary: The answer to build a profile for.
            exclude_qid: Optional question ID to exclude (leave-one-out).

        Returns:
            Profile text truncated to max_tokens_per_profile.
        """
        items = self._grouped.get(answer_primary, [])
        texts: List[str] = []

        # Collect all question texts except the excluded one
        for qid, qtext in items:
            if exclude_qid is not None and qid == exclude_qid:
                continue
            texts.append(qtext)

        # If not enough questions after exclusion, fall back to answer text
        if len(texts) < self.min_questions_per_answer:
            return answer_primary

        # Merge all texts and split into tokens
        merged = " ".join(texts).split()

        # Truncate to max tokens if specified
        if self.max_tokens_per_profile > 0:
            merged = merged[:self.max_tokens_per_profile]

        return " ".join(merged) if merged else answer_primary

    def profile_for_answer(
        self,
        answer_primary: str,
        exclude_qid: Optional[str] = None
    ) -> str:
        """Get the profile for a specific answer.

        Args:
            answer_primary: The answer to get a profile for.
            exclude_qid: Optional question ID to exclude (for gold answer).

        Returns:
            Profile text for the answer.
        """
        return self._profile_text(
            answer_primary=answer_primary,
            exclude_qid=exclude_qid
        )

    def build_profiles(
        self,
        questions: List[TossupQuestion],
        exclude_qid: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build profiles for all answers in the dataset.

        Args:
            questions: List of questions (used to fit if not already fitted).
            exclude_qid: Optional question ID to exclude from all profiles.

        Returns:
            Dictionary mapping answer_primary to profile text.
        """
        if not self._grouped:
            self.fit(questions)

        return {
            answer: self._profile_text(answer, exclude_qid=exclude_qid)
            for answer in self._grouped.keys()
        }
```

```bash
sed -n '1,240p' qb_data/mc_builder.py

```

```output
"""Multiple-choice question builder with anti-artifact guards."""

from __future__ import annotations

import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.text_utils import normalize_answer


@dataclass
class MCQuestion(TossupQuestion):
    """A tossup question with multiple-choice options.

    Extends TossupQuestion with fields for multiple-choice presentation
    and tracking of distractor generation strategy.
    """
    options: List[str]
    gold_index: int
    option_profiles: List[str]
    option_answer_primary: List[str]
    distractor_strategy: str


def _normalized_edit_distance(a: str, b: str) -> float:
    """Compute normalized edit distance between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Distance between 0 (identical) and 1 (completely different).
    """
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _token_overlap(a: str, b: str) -> float:
    """Compute token overlap between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Fraction of overlapping tokens (0 to 1).
    """
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


class MCBuilder:
    """Builder for multiple-choice questions with anti-artifact guards.

    This class implements four layers of guards to prevent spurious patterns
    that agents could exploit:
    1. Alias collision guard: Prevents distractors that are aliases of the gold answer
    2. Duplicate guard: Prevents distractors with high token overlap
    3. Length ratio guard: Prevents distractors much longer/shorter than others
    4. Question overlap guard: Prevents answers that appear in the question text
    """

    def __init__(
        self,
        K: int = 4,
        strategy: str = "sbert_profile",
        alias_edit_distance_threshold: float = 0.2,
        duplicate_token_overlap_threshold: float = 0.8,
        max_length_ratio: float = 3.0,
        random_seed: int = 13,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_model: str = "text-embedding-3-small",
    ):
        """Initialize the MC builder.

        Args:
            K: Number of answer choices (must be >= 2).
            strategy: Distractor selection strategy
                (sbert_profile, openai_profile, tfidf_profile, category_random).
            alias_edit_distance_threshold: Max edit distance for alias detection.
            duplicate_token_overlap_threshold: Max token overlap between options.
            max_length_ratio: Max ratio between longest and shortest option.
            random_seed: Random seed for reproducibility.
            embedding_model: SentenceTransformer model name for ``sbert_profile``.
            openai_model: OpenAI embedding model for ``openai_profile``.
        """
        if K < 2:
            raise ValueError("K must be >= 2")
        self.K = K
        self.strategy = strategy
        self.alias_edit_distance_threshold = alias_edit_distance_threshold
        self.duplicate_token_overlap_threshold = duplicate_token_overlap_threshold
        self.max_length_ratio = max_length_ratio
        self.rng = random.Random(random_seed)
        self.embedding_model = embedding_model
        self.openai_model = openai_model

    def _prepare_lookup(
        self, questions: List[TossupQuestion]
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], List[str]]:
        """Prepare lookup structures for answer processing.

        Args:
            questions: List of tossup questions.

        Returns:
            Tuple of (answer_to_aliases, answer_to_category, answer_to_norm, answers).
        """
        answer_to_aliases: Dict[str, Set[str]] = {}
        answer_to_category: Dict[str, str] = {}

        for q in questions:
            # Collect all aliases for each answer
            aliases = answer_to_aliases.setdefault(q.answer_primary, set())
            aliases.update(str(alias) for alias in q.clean_answers)
            aliases.add(q.answer_primary)

            # Track category for category-based distractor selection
            if q.category and q.answer_primary not in answer_to_category:
                answer_to_category[q.answer_primary] = q.category

        # Convert to sorted lists for consistency
        answer_to_aliases_list = {k: sorted(v) for k, v in answer_to_aliases.items()}
        answers = sorted(answer_to_aliases_list.keys())
        answer_to_norm = {a: str(normalize_answer(a)) for a in answers}

        return answer_to_aliases_list, answer_to_category, answer_to_norm, answers

    def _compute_rankings(
        self,
        answers: List[str],
        answer_profiles: Dict[str, str],
        answer_to_category: Dict[str, str],
    ) -> Dict[str, List[str]]:
        """Compute distractor rankings for each answer.

        Args:
            answers: List of all unique answers.
            answer_profiles: Dictionary mapping answers to their profiles.
            answer_to_category: Dictionary mapping answers to categories.

        Returns:
            Dictionary mapping each answer to a ranked list of distractors.
        """
        if self.strategy == "category_random":
            # Random selection within the same category
            rankings: Dict[str, List[str]] = {}
            for answer in answers:
                category = answer_to_category.get(answer, "")
                # First try same category, then fall back to all answers
                candidates = [
                    a for a in answers
                    if a != answer and answer_to_category.get(a, "") == category
                ]
                if len(candidates) < self.K - 1:
                    candidates = [a for a in answers if a != answer]
                self.rng.shuffle(candidates)
                rankings[answer] = candidates
            return rankings

        # Profile-based ranking strategies
        docs = [answer_profiles[a] for a in answers]
        answer_idx = {a: i for i, a in enumerate(answers)}
        rankings = {}

        if self.strategy == "tfidf_profile":
            # TF-IDF based similarity
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(docs)
            sim = cosine_similarity(matrix, matrix)
            for answer in answers:
                idx = answer_idx[answer]
                order = np.argsort(-sim[idx]).tolist()
                rankings[answer] = [answers[i] for i in order if answers[i] != answer]
            return rankings

        if self.strategy in {"sbert_profile", "openai_profile"}:
            if self.strategy == "sbert_profile":
                # Sentence-BERT embeddings
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(self.embedding_model)
                embeddings = encoder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
                sim = embeddings @ embeddings.T
            else:
                from models.likelihoods import OpenAILikelihood

                likelihood = OpenAILikelihood(model=self.openai_model)
                embeddings = likelihood.embed_and_cache(docs)
                sim = embeddings @ embeddings.T

            for answer in answers:
                idx = answer_idx[answer]
                order = np.argsort(-sim[idx]).tolist()
                rankings[answer] = [answers[i] for i in order if answers[i] != answer]
            return rankings

        raise ValueError(f"Unknown distractor strategy: {self.strategy}")

    def _aliases_collide(self, candidate: str, gold_aliases: List[str]) -> bool:
        """Check if a candidate is too similar to any gold answer alias.

        Args:
            candidate: Candidate distractor.
            gold_aliases: List of aliases for the gold answer.

        Returns:
            True if the candidate collides with a gold alias.
        """
        candidate_norm = str(normalize_answer(candidate))
        gold_norms = [str(normalize_answer(alias)) for alias in gold_aliases]

        # Check exact match
        if candidate_norm in set(gold_norms):
            return True

        # Check edit distance
        for gold_norm in gold_norms:
            if _normalized_edit_distance(candidate_norm, gold_norm) < self.alias_edit_distance_threshold:
                return True

        return False

    def _violates_duplicate_guard(self, candidate: str, selected: List[str]) -> bool:
        """Check if candidate has too much token overlap with already selected options.

        Args:
            candidate: Candidate distractor.
            selected: List of already selected distractors.

```

```bash
sed -n '278,386p' qb_data/mc_builder.py

```

```output
    def build(
        self,
        questions: List[TossupQuestion],
        profile_builder: AnswerProfileBuilder,
    ) -> List[MCQuestion]:
        """Build multiple-choice questions with anti-artifact guards.

        Args:
            questions: List of tossup questions.
            profile_builder: Profile builder for answer representations.

        Returns:
            List of MCQuestion objects that passed all guards.
        """
        if not questions:
            return []

        # Build answer profiles
        profile_builder.fit(questions)
        answer_profiles = profile_builder.build_profiles(questions)

        # Prepare lookup structures
        answer_to_aliases, answer_to_category, _answer_to_norm, answers = self._prepare_lookup(questions)

        # Compute distractor rankings
        rankings = self._compute_rankings(answers, answer_profiles, answer_to_category)

        mc_questions: List[MCQuestion] = []

        for q in questions:
            gold = q.answer_primary
            gold_aliases = answer_to_aliases.get(gold, [gold])
            ranked = rankings.get(gold, [a for a in answers if a != gold])
            selected: List[str] = []

            # Select distractors from ranked list
            for candidate in ranked:
                if candidate == gold:
                    continue
                # Apply guard 1: Check alias collision
                if self._aliases_collide(candidate, gold_aliases):
                    continue
                # Apply guard 2: Check duplicate tokens
                if self._violates_duplicate_guard(candidate, selected):
                    continue
                selected.append(candidate)
                if len(selected) >= self.K - 1:
                    break

            # If not enough distractors from ranking, try random fallback
            if len(selected) < self.K - 1:
                fallback = [a for a in answers if a not in selected and a != gold]
                self.rng.shuffle(fallback)
                for candidate in fallback:
                    if self._aliases_collide(candidate, gold_aliases):
                        continue
                    if self._violates_duplicate_guard(candidate, selected):
                        continue
                    selected.append(candidate)
                    if len(selected) >= self.K - 1:
                        break

            # Skip question if we can't find enough valid distractors
            if len(selected) < self.K - 1:
                continue

            # Create options and shuffle
            option_answer_primary = [gold] + selected[:self.K - 1]
            self.rng.shuffle(option_answer_primary)
            gold_index = option_answer_primary.index(gold)
            options = option_answer_primary[:]

            # Apply guard 3: Check length ratio
            if self._violates_length_ratio_guard(options):
                continue

            # Apply guard 4: Check question overlap
            if self._violates_question_overlap_guard(q.question, options):
                continue

            # Build option profiles with leave-one-out for gold
            option_profiles: List[str] = []
            for answer in option_answer_primary:
                exclude_qid = q.qid if answer == gold else None
                option_profiles.append(
                    profile_builder.profile_for_answer(answer, exclude_qid=exclude_qid)
                )

            # Create MCQuestion
            mc_questions.append(
                MCQuestion(
                    qid=q.qid,
                    question=q.question,
                    tokens=q.tokens,
                    answer_primary=q.answer_primary,
                    clean_answers=q.clean_answers,
                    run_indices=q.run_indices,
                    human_buzz_positions=q.human_buzz_positions,
                    category=q.category,
                    cumulative_prefixes=q.cumulative_prefixes,
                    options=options,
                    gold_index=gold_index,
                    option_profiles=option_profiles,
                    option_answer_primary=option_answer_primary,
                    distractor_strategy=self.strategy,
                )
            )

        return mc_questions
```

```bash
venv/bin/python - <<'PY'
from qb_data.data_loader import QANTADatasetLoader
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.mc_builder import MCBuilder
qs = QANTADatasetLoader().load_from_csv('data/test_questions.csv')
pb = AnswerProfileBuilder().fit(qs)
mc = MCBuilder(K=4, strategy='category_random').build(qs, pb)
q = mc[0]
print('mc_questions =', len(mc))
print('qid =', q.qid)
print('gold_answer =', q.answer_primary)
print('gold_index =', q.gold_index)
print('options =', q.options)
print('strategy =', q.distractor_strategy)
PY

```

```output
mc_questions = 10
qid = hist_001
gold_answer = Napoleon Bonaparte
gold_index = 2
options = ['Hamlet', 'Impressionism', 'Napoleon Bonaparte', 'Ludwig van Beethoven']
strategy = category_random
```

```bash
sed -n '1,180p' qb_data/dataset_splits.py

```

```output
"""
Stratified dataset splitting utilities for quiz bowl data.

This module provides functions to create train/val/test splits that maintain
category distribution across all splits.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from qb_data.data_loader import TossupQuestion


def create_stratified_splits(
    questions: List[TossupQuestion],
    ratios: List[float] = [0.7, 0.15, 0.15],
    seed: int = 42
) -> Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]:
    """
    Create stratified train/val/test splits maintaining category distribution.

    Parameters
    ----------
    questions : List[TossupQuestion]
        List of questions to split
    ratios : List[float]
        Train/val/test split ratios (must sum to 1.0)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]
        Train, validation, and test splits

    Raises
    ------
    ValueError
        If ratios don't sum to 1.0 or questions list is empty
    """
    # Validate inputs
    if not questions:
        raise ValueError("Cannot split empty question list")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    # Initialize random generator for reproducibility
    rng = random.Random(seed)

    # Group questions by category
    category_groups = defaultdict(list)
    for q in questions:
        category_groups[q.category].append(q)

    # Initialize output lists
    train_questions = []
    val_questions = []
    test_questions = []

    # Split each category maintaining ratios
    for category, category_questions in category_groups.items():
        # Sort for deterministic splits
        sorted_questions = sorted(category_questions, key=lambda q: q.qid)

        # Shuffle with fixed seed for this category
        category_seed = seed + hash(category) % 1000000
        category_rng = random.Random(category_seed)
        shuffled = sorted_questions.copy()
        category_rng.shuffle(shuffled)

        n = len(shuffled)

        # Calculate split indices
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        # Handle small categories - ensure at least 1 in train if possible
        if n == 1:
            train_questions.extend(shuffled)
        elif n == 2:
            train_questions.extend(shuffled[:1])
            val_questions.extend(shuffled[1:])
        else:
            # Standard split
            train_questions.extend(shuffled[:train_end])
            val_questions.extend(shuffled[train_end:val_end])
            test_questions.extend(shuffled[val_end:])

    # Verify all questions assigned exactly once
    total_original = len(questions)
    total_split = len(train_questions) + len(val_questions) + len(test_questions)

    if total_original != total_split:
        raise RuntimeError(f"Split mismatch: {total_original} original vs {total_split} split")

    # Log category distribution statistics
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_questions)} questions ({len(train_questions)/total_original:.1%})")
    print(f"  Val:   {len(val_questions)} questions ({len(val_questions)/total_original:.1%})")
    print(f"  Test:  {len(test_questions)} questions ({len(test_questions)/total_original:.1%})")

    # Category distribution analysis
    train_categories = defaultdict(int)
    val_categories = defaultdict(int)
    test_categories = defaultdict(int)

    for q in train_questions:
        train_categories[q.category] += 1
    for q in val_questions:
        val_categories[q.category] += 1
    for q in test_questions:
        test_categories[q.category] += 1

    all_categories = set(train_categories.keys()) | set(val_categories.keys()) | set(test_categories.keys())
    print(f"\nCategory distribution ({len(all_categories)} categories):")

    for category in sorted(all_categories)[:5]:  # Show first 5 categories
        orig_count = len(category_groups[category])
        train_count = train_categories.get(category, 0)
        val_count = val_categories.get(category, 0)
        test_count = test_categories.get(category, 0)
        print(f"  {category}: {train_count}/{val_count}/{test_count} (orig: {orig_count})")

    if len(all_categories) > 5:
        print(f"  ... and {len(all_categories) - 5} more categories")

    return train_questions, val_questions, test_questions


def save_splits(
    train: List[TossupQuestion],
    val: List[TossupQuestion],
    test: List[TossupQuestion],
    output_dir: str = "data"
) -> None:
    """
    Save dataset splits to JSON files with metadata.

    Parameters
    ----------
    train : List[TossupQuestion]
        Training split
    val : List[TossupQuestion]
        Validation split
    test : List[TossupQuestion]
        Test split
    output_dir : str
        Directory to save split files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Helper to convert TossupQuestion to dict
    def questions_to_dict(questions: List[TossupQuestion]) -> List[Dict[str, Any]]:
        return [
            {
                "qid": q.qid,
                "question": q.question,
                "tokens": q.tokens,
                "answer_primary": q.answer_primary,
                "clean_answers": q.clean_answers,
                "run_indices": q.run_indices,
                "human_buzz_positions": q.human_buzz_positions,
                "category": q.category,
                "cumulative_prefixes": q.cumulative_prefixes
            }
            for q in questions
        ]

    # Calculate category distributions for metadata
    def get_category_distribution(questions: List[TossupQuestion]) -> Dict[str, int]:
        dist = defaultdict(int)
        for q in questions:
            dist[q.category] += 1
        return dict(dist)

```

## 4. Model layer

The model layer splits into two responsibilities.

- Likelihood models score how well the revealed clues match each answer option profile.
- Policy models decide when to buzz and which answer to choose.

That separation is what makes the canonical pipeline easy to swap between TF-IDF, SBERT, T5, and optional OpenAI embeddings while keeping the environment and agents stable.

```bash
rg -n "^class LikelihoodModel|^class TfIdfLikelihood|^class SBERTLikelihood|^class OpenAILikelihood|^class T5Likelihood|^def build_likelihood_from_config" models/likelihoods.py

```

```output
53:class LikelihoodModel(ABC):
138:class TfIdfLikelihood(LikelihoodModel):
259:class SBERTLikelihood(LikelihoodModel):
350:class OpenAILikelihood(LikelihoodModel):
399:class T5Likelihood(LikelihoodModel):
539:def build_likelihood_from_config(
```

```bash
sed -n '1,180p' models/likelihoods.py

```

```output
"""
Likelihood Model Interface

Abstract base class for likelihood models that score answer options against
revealed clue text. Concrete implementations (TF-IDF, SBERT, T5) inherit
from ``LikelihoodModel`` and implement ``score()`` and ``_embed_batch()``.

The ``score()`` method returns **raw similarity scores**, not probabilities.
The environment applies softmax with a configurable temperature (beta) to
convert scores into a belief distribution.

Embedding caching is built into the base class: texts are hashed via SHA-256
and cached as float32 numpy arrays, so repeated calls with the same text
skip recomputation.

Ported from qb-rl reference implementation (models/likelihoods.py lines 1-38).
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def _text_key(text: str) -> str:
    """Compute a SHA-256 hash key for embedding cache lookups.

    Parameters
    ----------
    text : str
        Input text to hash.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 digest.

    Examples
    --------
    >>> key = _text_key("hello world")
    >>> len(key)
    64
    >>> _text_key("hello world") == _text_key("hello world")
    True
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LikelihoodModel(ABC):
    """Abstract base class for likelihood models.

    Likelihood models score how well each answer option matches a given
    clue prefix. The environment uses these scores (via softmax) to compute
    belief distributions over answer options.

    Subclasses must implement:
        - ``score(clue_prefix, option_profiles) -> np.ndarray``
        - ``_embed_batch(texts) -> np.ndarray``

    The base class provides ``embed_and_cache()`` which handles caching of
    text embeddings via SHA-256 content hashing.

    Attributes
    ----------
    embedding_cache : dict[str, np.ndarray]
        Maps SHA-256 text hashes to float32 embedding vectors.
    """

    def __init__(self) -> None:
        self.embedding_cache: dict[str, np.ndarray] = {}

    @abstractmethod
    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Return raw similarity scores for each answer option.

        The caller (environment) converts these to probabilities via
        softmax with a beta temperature parameter. Higher scores indicate
        stronger match between clue and option.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far (concatenation of clues up to current step).
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Raw similarity scores of shape (K,) where K = len(option_profiles).
        """

    def embed_and_cache(self, texts: list[str]) -> np.ndarray:
        """Embed texts, using cache for previously seen inputs.

        Texts are identified by their SHA-256 hash. Only unseen texts
        are passed to ``_embed_batch()`` for actual computation; cached
        results are reused.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        np.ndarray
            Stacked embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        missing = [text for text in texts if _text_key(text) not in self.embedding_cache]
        if missing:
            new_embeddings = self._embed_batch(missing)
            for text, emb in zip(missing, new_embeddings):
                self.embedding_cache[_text_key(text)] = emb.astype(np.float32)
        return np.stack([self.embedding_cache[_text_key(text)] for text in texts])

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Subclasses must implement.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            Embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        raise NotImplementedError


class TfIdfLikelihood(LikelihoodModel):
    """TF-IDF based likelihood model using cosine similarity.

    Uses scikit-learn's ``TfidfVectorizer`` to learn vocabulary and IDF weights
    from a corpus, then scores clue-option similarity via cosine distance in the
    TF-IDF vector space.

    The model **must** be ``fit()`` on a corpus before calling ``score()`` or
    ``_embed_batch()``. Calling these methods on an unfitted model raises
    ``RuntimeError``.

    This is the fast, interpretable baseline: keyword overlap drives similarity.
    It works well when clues contain distinctive vocabulary but misses semantic
    relationships (e.g., "first president" vs "George Washington").

    Parameters
    ----------
    corpus_texts : list[str] or None
        If provided, ``fit()`` is called immediately on these texts.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Scikit-learn vectorizer with English stop words removed.
    _is_fit : bool
        Whether the vectorizer has been fit on a corpus.

    Examples
    --------
    >>> corpus = ["George Washington was the first president",
    ...           "Abraham Lincoln freed the slaves"]
    >>> model = TfIdfLikelihood(corpus_texts=corpus)
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, corpus_texts: list[str] | None = None) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._is_fit = False
```

```bash
sed -n '330,620p' models/likelihoods.py

```

```output

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far.
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Cosine similarity scores of shape (K,), dtype float32.
            Values in [-1, 1].
        """
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


class OpenAILikelihood(LikelihoodModel):
    """OpenAI embedding likelihood model using normalized embedding similarity.

    This path is optional and only activates when explicitly selected in config.
    It requires both the ``openai`` Python package and ``OPENAI_API_KEY`` to be
    available at runtime.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        super().__init__()

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError(
                "OpenAI likelihood requires OPENAI_API_KEY to be set."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI likelihood requires the openai package. "
                "Install it with: pip install -e .[openai] or pip install openai."
            ) from exc

        self.model = model
        self.client = OpenAI(api_key=resolved_api_key)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts via the OpenAI embeddings API and L2-normalize them."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        embeddings = np.stack(vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using cosine similarity over normalized embeddings."""
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


class T5Likelihood(LikelihoodModel):
    """T5 encoder likelihood model using mean-pooled semantic embeddings.

    Uses ``T5EncoderModel`` (not full ``T5ForConditionalGeneration``) for 2x
    faster inference and half the memory. Embeddings are mean-pooled over
    sequence length with attention mask weighting to handle padding correctly.

    Inherits ``embed_and_cache()`` from ``LikelihoodModel`` for transparent
    caching of embeddings via SHA-256 content hashing. The first call to
    ``score()`` computes and caches all embeddings; subsequent calls with the
    same texts are fast cache lookups.

    Compared to SBERT, T5 captures deeper semantic relationships via its
    encoder-decoder pre-training on massive text corpora. This is the novel
    contribution: using T5 as a likelihood model rather than just as a policy
    encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace T5 model identifier. Default is ``"t5-base"``
        (220M params). Options:

        - ``"t5-small"`` (60M params) -- fastest, lowest quality
        - ``"t5-base"`` (220M params) -- balanced (recommended)
        - ``"t5-large"`` (770M params) -- best quality, requires 8GB GPU VRAM

        First run downloads the model from HuggingFace (~850MB for t5-base).

    Attributes
    ----------
    model_name : str
        The T5 model identifier.
    encoder : T5EncoderModel
        Pre-trained T5 encoder loaded from HuggingFace.
    tokenizer : T5TokenizerFast
        Fast T5 tokenizer for text preprocessing.
    device : torch.device
        Computation device (cuda if available, else cpu).

    Examples
    --------
    >>> model = T5Likelihood(model_name="t5-small")
    >>> scores = model.score("first president", ["Washington", "Einstein"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, model_name: str = "t5-base") -> None:
        super().__init__()
        import torch
        from transformers import T5EncoderModel, T5TokenizerFast

        self.model_name = model_name
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts using T5 encoder with attention-masked mean pooling.

        Mean pooling uses the attention mask to exclude padding tokens from the
        average, ensuring correct semantic embeddings when sequences have
        different lengths. Embeddings are L2-normalized so that cosine
        similarity can be computed as a simple dot product.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            L2-normalized embeddings of shape (len(texts), hidden_dim),
            dtype float32. Hidden dim is 512 (t5-small), 768 (t5-base),
            or 1024 (t5-large).

        Notes
        -----
        Tensors are detached and moved to CPU immediately after computation
        to prevent GPU memory leaks when called repeatedly during episodes.
        """
        import torch

        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.encoder(**encoded)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            # Mean pooling over sequence length with attention mask
            mask = encoded.attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            masked_hidden = last_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden_dim)
            mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
            mean_pooled = sum_hidden / mask_sum  # (batch, hidden_dim)

            # L2 normalize for cosine similarity via dot product
            embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            # Detach and move to CPU to prevent GPU memory leak
            embeddings = embeddings.detach().cpu().numpy().astype(np.float32)

        return embeddings

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using T5 semantic cosine similarity.

        Computes dot product between the clue embedding and each option
        embedding. Since embeddings are L2-normalized, dot product equals
        cosine similarity.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far.
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Cosine similarity scores of shape (K,), dtype float32.
            Values in [-1, 1].
        """
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


def build_likelihood_from_config(
    config: dict[str, Any], corpus_texts: list[str] | None = None
) -> LikelihoodModel:
    """Construct a likelihood model from YAML configuration.

    Factory function that reads the ``likelihood`` section of the config dict
    and instantiates the appropriate ``LikelihoodModel`` subclass.

    Parameters
    ----------
    config : dict[str, Any]
        Full YAML config dict. Must contain a ``"likelihood"`` key with at
        least a ``"model"`` field specifying the model type.

        Supported model types:
        - ``"tfidf"``: TF-IDF cosine similarity (requires ``corpus_texts``)
        - ``"sbert"``: Sentence-BERT semantic similarity
        - ``"openai"``: OpenAI embedding similarity
        - ``"t5"`` / ``"t5-small"`` / ``"t5-base"`` / ``"t5-large"``:
          T5 encoder semantic similarity

        Optional config keys:
        - ``"sbert_name"`` or ``"embedding_model"``: SentenceTransformer model
          name (default: ``"all-MiniLM-L6-v2"``)
        - ``"openai_model"``: OpenAI embedding model name
          (default: ``"text-embedding-3-small"``)
        - ``"t5_name"``: T5 model name (default: ``"t5-base"``)

    corpus_texts : list[str] or None
        Text corpus for TF-IDF fitting. Required when ``model == "tfidf"``,
        ignored for other models.

    Returns
    -------
    LikelihoodModel
        An instantiated and ready-to-use likelihood model.

    Raises
    ------
    ValueError
        If ``model`` is ``"tfidf"`` and ``corpus_texts`` is None.
        If ``model`` is not a recognized model type.

    Examples
    --------
    >>> from qb_data.config import load_config
    >>> config = load_config("configs/default.yaml")
    >>> model = build_likelihood_from_config(config, corpus_texts=my_corpus)
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    """
    cfg = config["likelihood"]
    model_name = cfg.get("model", "sbert")

    if model_name == "tfidf":
        if not corpus_texts:
            raise ValueError("TF-IDF likelihood requires corpus_texts.")
        return TfIdfLikelihood(corpus_texts=corpus_texts)

    if model_name == "sbert":
        # Support both "sbert_name" (qb-rl convention) and
        # "embedding_model" (qanta-buzzer default.yaml convention)
        sbert_name = cfg.get("sbert_name", cfg.get("embedding_model", "all-MiniLM-L6-v2"))
        return SBERTLikelihood(model_name=sbert_name)

    if model_name == "openai":
        return OpenAILikelihood(
            model=cfg.get("openai_model", "text-embedding-3-small"),
        )

    if model_name == "t5":
        t5_name = cfg.get("t5_name", "t5-base")
        return T5Likelihood(model_name=t5_name)

    if isinstance(model_name, str) and model_name.startswith("t5"):
        t5_name = model_name
        return T5Likelihood(model_name=t5_name)

    raise ValueError(f"Unknown likelihood model: {model_name}")
```

```bash
sed -n '1,180p' models/features.py

```

```output
"""
Belief Feature Extraction

Extracts derived features from belief probability distributions for use as
policy observations. Given a belief vector of K probabilities (one per answer
option), produces a (K + 6)-dimensional feature vector containing:

    belief[0..K-1]   raw belief probabilities
    top_p             max belief probability
    margin            gap between top two probabilities
    entropy           Shannon entropy of the distribution
    stability         L1 distance from previous belief (0 if first step)
    progress          fraction of total clue steps elapsed
    clue_idx_norm     normalized clue index (0 to 1 over steps)

Ported from qb-rl reference implementation (models/features.py).
"""

from __future__ import annotations

import numpy as np


def entropy_of_distribution(prob: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Uses clipping for numerical stability to avoid log(0).

    Parameters
    ----------
    prob : np.ndarray
        1D probability vector. Values should sum to ~1.0.

    Returns
    -------
    float
        Shannon entropy H(p) = -sum(p * log(p)), non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> uniform = np.array([0.25, 0.25, 0.25, 0.25])
    >>> abs(entropy_of_distribution(uniform) - 1.3863) < 0.001
    True
    """
    clipped = np.clip(prob, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def extract_belief_features(
    belief: np.ndarray,
    prev_belief: np.ndarray | None,
    step_idx: int,
    total_steps: int,
) -> np.ndarray:
    """Extract derived features from a belief probability vector.

    Concatenates the raw belief with 6 derived scalar features to produce
    a fixed-size observation vector for the RL policy.

    Parameters
    ----------
    belief : np.ndarray
        1D probability vector of shape (K,) over answer options.
    prev_belief : np.ndarray or None
        Previous step's belief vector, same shape as ``belief``.
        Pass None on the first step (stability will be 0.0).
    step_idx : int
        Current clue step index (0-based).
    total_steps : int
        Total number of clue steps in the episode.

    Returns
    -------
    np.ndarray
        Feature vector of shape (K + 6,) with dtype float32.
        Layout: [belief..., top_p, margin, entropy, stability, progress, clue_idx_norm].

    Raises
    ------
    ValueError
        If ``belief`` is not a 1D array.

    Examples
    --------
    >>> import numpy as np
    >>> belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    >>> feats = extract_belief_features(belief, None, 2, 6)
    >>> feats.shape
    (10,)
    >>> feats.dtype
    dtype('float32')
    """
    belief = np.asarray(belief, dtype=np.float32)
    if belief.ndim != 1:
        raise ValueError("belief must be a 1D probability vector")

    top_p = float(np.max(belief))
    sorted_probs = np.sort(belief)[::-1]
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_p - second
    ent = entropy_of_distribution(belief)
    stability = float(np.abs(belief - prev_belief).sum()) if prev_belief is not None else 0.0
    progress = float(step_idx / max(1, total_steps))
    clue_idx_norm = float(step_idx / max(1, total_steps - 1))

    extras = np.array([top_p, margin, ent, stability, progress, clue_idx_norm], dtype=np.float32)
    return np.concatenate([belief, extras]).astype(np.float32)
```

```bash
venv/bin/python - <<'PY'
from pathlib import Path
import re
text = Path('models/likelihoods.py').read_text()
direct = re.findall(r'if model_name == "([^"]+)"', text)
print('direct_factory_keys =', direct)
print('literal_t5_names =', ['t5-small', 't5-base', 't5-large'])
PY

```

```output
direct_factory_keys = ['tfidf', 'sbert', 'openai', 't5']
literal_t5_names = ['t5-small', 't5-base', 't5-large']
```

```bash
sed -n '1,260p' models/t5_policy.py

```

```output
"""
T5-based Policy Model for Quiz Bowl RL Agent

Implements T5PolicyModel with a custom PolicyHead containing three independent
heads (wait/answer/value) for end-to-end text-based policy learning. This
provides an alternative to the MLP policy trained on belief features
(Phase 4 approach).

Architecture overview:

    Text input  -->  T5 Encoder  -->  Mean Pooling  -->  PolicyHead
                                                          |-- Wait head (2)
                                                          |-- Answer head (K)
                                                          |-- Value head (1)

The T5 encoder produces contextual embeddings from tokenized text. Mean pooling
(attention-masked) reduces the variable-length sequence to a fixed-size vector.
The PolicyHead then produces three independent outputs:

- **Wait logits** [B, 2]: probability of waiting vs answering now
- **Answer logits** [B, K]: probability of selecting each answer option
- **Value estimate** [B, 1]: state value for PPO advantage computation

Action space maps to the TossupMCEnv convention:
    0 = WAIT (wait head selects "wait")
    1..K = SELECT answer i-1 (wait head selects "answer now", answer head picks i-1)

Ported from qanta-buzzer reference implementation (model.py) with these changes:
    - T5EncoderModel replaces T5ForConditionalGeneration (2x faster, 50% less memory)
    - T5TokenizerFast replaces T5Tokenizer (3-5x faster tokenization via Rust backend)
    - Config dict replaces qanta-buzzer's Config class for unified codebase compatibility
    - NumPy-style docstrings added throughout
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """Custom policy head with three independent output heads.

    Attached to a T5 encoder's pooled output, this module produces the three
    outputs needed for actor-critic RL in the quiz bowl POMDP: a binary
    wait/answer-now decision, a K-way answer selection, and a scalar value
    estimate.

    All three heads are fully independent (no shared hidden layers beyond the
    encoder), using the same pattern: Linear -> ReLU -> Dropout -> Linear.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the input from the T5 encoder's pooled output.
        Default 1024 matches T5-large (``d_model``). Use 512 for t5-small,
        768 for t5-base.
    num_choices : int
        Number of answer options (K). Default 4 for quiz bowl MC questions.

    Attributes
    ----------
    wait_head : nn.Sequential
        Binary head producing [wait, answer_now] logits.
    answer_head : nn.Sequential
        Multi-class head producing logits over K answer choices.
    value_head : nn.Sequential
        Scalar head producing state value estimate.
    """

    def __init__(self, hidden_size: int = 1024, num_choices: int = 4) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_choices = num_choices

        # Wait/continue decision head (binary: wait vs answer_now)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),  # [wait, answer_now]
        )

        # Answer selection head (over K choices)
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_choices),
        )

        # Value head (state value estimate for PPO)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(
        self, encoder_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all three heads.

        Parameters
        ----------
        encoder_hidden_state : torch.Tensor
            Pooled encoder output of shape ``[batch_size, hidden_size]``.

        Returns
        -------
        wait_logits : torch.Tensor
            Shape ``[batch_size, 2]`` -- logits for [wait, answer_now].
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer options.
        values : torch.Tensor
            Shape ``[batch_size, 1]`` -- state value estimates.
        """
        wait_logits = self.wait_head(encoder_hidden_state)
        answer_logits = self.answer_head(encoder_hidden_state)
        values = self.value_head(encoder_hidden_state)

        return wait_logits, answer_logits, values


class T5PolicyModel(nn.Module):
    """T5 encoder with custom policy head for end-to-end RL.

    Combines a pre-trained T5 encoder with a ``PolicyHead`` to produce policy
    outputs directly from text observations. This is the alternative approach
    to Phase 4's MLP policy, which operates on numeric belief features.

    The model processes text in three stages:

    1. **Tokenization**: Text is tokenized with ``T5TokenizerFast`` (Rust-backed
       for speed) with padding and truncation.
    2. **Encoding**: ``T5EncoderModel`` produces contextual hidden states
       ``[B, seq_len, d_model]``.
    3. **Pooling + Heads**: Attention-masked mean pooling reduces to
       ``[B, d_model]``, then PolicyHead produces wait/answer/value outputs.

    Action space follows TossupMCEnv convention:
        - 0 = WAIT
        - 1..K = SELECT answer (i-1)

    Combined actions are decomposed into two independent decisions for log
    probability computation:
        - ``wait_action``: 0 (wait) or 1 (answer now)
        - ``answer_action``: 0..K-1 (which answer to select)

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with the following keys:

        - ``model_name`` (str): HuggingFace T5 model identifier.
          Default ``"t5-large"``. Options: ``"t5-small"``, ``"t5-base"``,
          ``"t5-large"``.
        - ``device`` (str): Torch device. Default auto-detects
          (cuda > mps > cpu).
        - ``max_input_length`` (int): Maximum token sequence length.
          Default 512.
        - ``num_choices`` (int): Number of answer options (K). Default 4.

    Attributes
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    device : torch.device
        Computation device.
    encoder : T5EncoderModel
        Pre-trained T5 encoder.
    tokenizer : T5TokenizerFast
        Fast T5 tokenizer.
    policy_head : PolicyHead
        Custom three-head policy module.
    max_input_length : int
        Maximum token sequence length for tokenization.

    Examples
    --------
    >>> config = {"model_name": "t5-small", "device": "cpu", "num_choices": 4}
    >>> model = T5PolicyModel(config)
    >>> texts = ["CLUES: first president | CHOICES: (1) Washington (2) Jefferson"]
    >>> wait_logits, answer_logits, values = model(texts)
    >>> wait_logits.shape
    torch.Size([1, 2])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        from transformers import T5EncoderModel, T5TokenizerFast

        self.config = config
        model_name = config.get("model_name", "t5-large")
        self.max_input_length = config.get("max_input_length", 512)
        num_choices = config.get("num_choices", 4)

        # Auto-detect device
        default_device = "cpu"
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            default_device = "mps"
        self.device = torch.device(config.get("device", default_device))

        # Load T5 encoder only (not full T5ForConditionalGeneration)
        # This is 2x faster and uses 50% less memory since the decoder is unused
        print(f"Loading T5 encoder: {model_name}")
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)

        # Get hidden size from T5 config (512 for small, 768 for base, 1024 for large)
        hidden_size = self.encoder.config.d_model

        # Custom policy head
        self.policy_head = PolicyHead(
            hidden_size=hidden_size,
            num_choices=num_choices,
        )

        # Move to device
        self.to(self.device)

        # Print model info
        self._print_model_info()

    def _print_model_info(self) -> None:
        """Print model architecture summary and parameter counts."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        total_params = encoder_params + policy_params

        print(f"Model Architecture:")
        print(f"  T5 encoder parameters: {encoder_params:,}")
        print(f"  Policy head parameters: {policy_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {self.device}")

    def encode_input(
        self,
        text_inputs: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs using T5TokenizerFast.

        Parameters
        ----------
        text_inputs : list[str]
            List of input text strings to tokenize.
        max_length : int or None
            Maximum sequence length. If None, uses ``self.max_input_length``.

        Returns
```

```bash
sed -n '1,120p' models/answer_profiles.py

```

```output
"""qb-rl compatibility re-export for answer profile building."""

from qb_data.answer_profiles import AnswerProfileBuilder

__all__ = ["AnswerProfileBuilder"]
```

## 5. Environment layer

`qb_env/` turns the scoring stack into a reinforcement-learning problem. The environment exposes a `(K+6)` observation vector for the belief-feature path, and `TextObservationWrapper` converts the same underlying state into text for the T5 policy path.

This is where the repo encodes the action space, reward modes, belief update modes, and the distinction between visible clues and latent answer correctness.

```bash
sed -n '1,220p' qb_env/tossup_env.py

```

```output
"""
Gymnasium-compliant POMDP Environment for Quiz Bowl

Implements a tossup question environment where clues are revealed incrementally.
At each step the agent observes a belief-based feature vector and chooses either
to WAIT (action 0, reveals next clue) or to BUZZ with a specific answer option
(actions 1..K, ends the episode).

The environment computes beliefs over K answer options using a pluggable
LikelihoodModel and converts them to observations via extract_belief_features.

Ported from qb-rl reference implementation (qb_env/tossup_env.py) and adapted
for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models.features import extract_belief_features
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion


class TossupMCEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment for quiz bowl tossup questions with MC options.

    Models quiz bowl as a POMDP where clues are revealed incrementally.
    The agent maintains a belief distribution over K answer options, updated
    at each step by a likelihood model. The agent decides when to buzz and
    which answer to select.

    Action Space
    ------------
    Discrete(K + 1):
        - 0: WAIT -- reveal the next clue and update belief
        - 1..K: BUZZ with answer option (i-1), ending the episode

    Observation Space
    -----------------
    Box(K + 6,):
        Belief features: [belief[0..K-1], top_p, margin, entropy,
        stability, progress, clue_idx_norm].
        See ``models.features.extract_belief_features`` for details.

    Reward Modes
    ------------
    ``time_penalty`` (default):
        -wait_penalty per WAIT step; +buzz_correct for correct buzz,
        +buzz_incorrect (negative) for wrong buzz.
    ``simple``:
        +1.0 for correct buzz, -1.0 for incorrect buzz, no WAIT penalty.
    ``human_grounded``:
        0.0 if the agent buzzes after the sampled human buzz position;
        otherwise +buzz_correct/-buzz_incorrect for correct/incorrect.

    Belief Modes
    ------------
    ``from_scratch``:
        Recompute belief from all clues seen so far via cumulative_prefixes.
    ``sequential_bayes``:
        Bayesian update: multiply prior belief by likelihood of new clue
        fragment, then normalize.

    Parameters
    ----------
    questions : list[MCQuestion]
        Pool of questions to sample from. Must be non-empty.
    likelihood_model : LikelihoodModel
        Model that scores clue text against answer option profiles.
    K : int
        Number of answer options per question. Must be >= 2.
    reward_mode : str
        One of ``"time_penalty"``, ``"simple"``, ``"human_grounded"``.
    wait_penalty : float
        Per-step penalty when reward_mode is ``"time_penalty"``.
    buzz_correct : float
        Reward for buzzing with the correct answer.
    buzz_incorrect : float
        Reward (typically negative) for buzzing with an incorrect answer.
    belief_mode : str
        One of ``"from_scratch"``, ``"sequential_bayes"``.
    beta : float
        Softmax temperature for converting raw scores to probabilities.
        Higher values produce sharper distributions.
    seed : int
        Random seed for question sampling and human buzz simulation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        questions: list[MCQuestion],
        likelihood_model: LikelihoodModel,
        K: int = 4,
        reward_mode: str = "time_penalty",
        wait_penalty: float = 0.01,
        buzz_correct: float = 1.0,
        buzz_incorrect: float = -0.5,
        belief_mode: str = "from_scratch",
        beta: float = 5.0,
        seed: int = 13,
    ) -> None:
        if not questions:
            raise ValueError("questions cannot be empty")
        if K < 2:
            raise ValueError("K must be >= 2")

        self.questions = questions
        self.likelihood_model = likelihood_model
        self.K = K
        self.reward_mode = reward_mode
        self.wait_penalty = wait_penalty
        self.buzz_correct = buzz_correct
        self.buzz_incorrect = buzz_incorrect
        self.belief_mode = belief_mode
        self.beta = beta
        self.rng = random.Random(seed)

        self.action_space = spaces.Discrete(self.K + 1)
        # belief[K] + (top_p, margin, entropy, stability, progress, clue_idx)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.K + 6,), dtype=np.float32
        )

        self.question: MCQuestion | None = None
        self.step_idx: int = 0
        self.prev_belief: np.ndarray | None = None
        self.belief: np.ndarray = np.ones(self.K, dtype=np.float32) / self.K
        self.terminated: bool = False
        self.truncated: bool = False
        self._sampled_human_buzz_pos: int | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_steps(self) -> int:
        """Total number of incremental clue steps for the current question.

        Returns
        -------
        int
            Length of ``question.run_indices`` if a question is loaded, else 1.
        """
        if self.question is None:
            return 1
        return len(self.question.run_indices)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _sample_question(self) -> MCQuestion:
        """Sample a random question from the question pool.

        Returns
        -------
        MCQuestion
            A randomly selected question.
        """
        return self.rng.choice(self.questions)

    def _sample_human_buzz(self, question: MCQuestion) -> int | None:
        """Sample a human buzz position from the question's distribution.

        Uses weighted random sampling based on the number of humans who
        buzzed at each position. Returns None if no human buzz data exists.

        Parameters
        ----------
        question : MCQuestion
            The question to sample a human buzz position for.

        Returns
        -------
        int or None
            Sampled token position, or None if no human buzz data.
        """
        if not question.human_buzz_positions:
            return None
        positions = []
        weights = []
        for pos, count in question.human_buzz_positions:
            positions.append(int(pos))
            weights.append(max(1, int(count)))
        if not positions:
            return None
        return self.rng.choices(positions, weights=weights, k=1)[0]

    def _softmax_scores(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw likelihood scores to a probability distribution.

        Applies a temperature-scaled softmax with numerical stability
        (subtract max before exponentiation). Falls back to uniform
        distribution if the sum of exponentiated scores is non-positive.

        Parameters
        ----------
        scores : np.ndarray
            Raw similarity scores of shape (K,).

        Returns
        -------
        np.ndarray
            Probability distribution of shape (K,), dtype float32.
        """
        stable = scores - np.max(scores)
        probs = np.exp(self.beta * stable)
        probs_sum = np.sum(probs)
        if probs_sum <= 0:
            return np.ones_like(scores, dtype=np.float32) / len(scores)
        return (probs / probs_sum).astype(np.float32)
```

```bash
sed -n '222,420p' qb_env/tossup_env.py

```

```output
    def _compute_belief(self, question: MCQuestion, step_idx: int) -> np.ndarray:
        """Compute belief distribution over answer options at a given step.

        Two modes are supported:

        ``from_scratch``
            Score the cumulative clue prefix against all option profiles,
            then apply softmax. Each step is independent of the previous
            belief.

        ``sequential_bayes``
            Extract only the new clue fragment since the last step, score
            it, and perform a Bayesian update: posterior = prior * likelihood,
            then normalize. This is cheaper per step but may accumulate
            approximation errors.

        Parameters
        ----------
        question : MCQuestion
            Current question being played.
        step_idx : int
            Current step index (0-based, indexes into run_indices).

        Returns
        -------
        np.ndarray
            Updated belief distribution of shape (K,), dtype float32.

        Raises
        ------
        ValueError
            If ``self.belief_mode`` is not a recognized mode.
        """
        if self.belief_mode == "from_scratch":
            prefix = question.cumulative_prefixes[step_idx]
            scores = self.likelihood_model.score(prefix, question.option_profiles)
            return self._softmax_scores(scores)

        if self.belief_mode == "sequential_bayes":
            idx = question.run_indices[step_idx]
            prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
            frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])
            scores = self.likelihood_model.score(frag, question.option_profiles)
            likelihood = self._softmax_scores(scores)
            posterior = self.belief * likelihood
            denom = posterior.sum()
            if denom <= 0:
                posterior = np.ones(self.K, dtype=np.float32) / self.K
            else:
                posterior = posterior / denom
            return posterior.astype(np.float32)

        raise ValueError(f"Unknown belief_mode: {self.belief_mode}")

    def _obs(self) -> np.ndarray:
        """Build the observation vector from current belief state.

        Delegates to ``extract_belief_features`` which concatenates the raw
        belief vector with 6 derived scalar features.

        Returns
        -------
        np.ndarray
            Feature vector of shape (K + 6,), dtype float32.
        """
        return extract_belief_features(
            belief=self.belief,
            prev_belief=self.prev_belief,
            step_idx=self.step_idx,
            total_steps=self.total_steps,
        )

    def _step_to_token_pos(self, step_idx: int) -> int:
        """Convert a step index to the corresponding token position.

        Used by the ``human_grounded`` reward mode to compare the agent's
        buzz position against the sampled human buzz position.

        Parameters
        ----------
        step_idx : int
            Step index (0-based, indexes into run_indices).

        Returns
        -------
        int
            Token position in the original question text.
        """
        if self.question is None or not self.question.run_indices:
            return step_idx
        if step_idx >= len(self.question.run_indices):
            return self.question.run_indices[-1]
        if step_idx < 0:
            return self.question.run_indices[0]
        return self.question.run_indices[step_idx]

    def _buzz_reward(self, question: MCQuestion, chosen_idx: int, last_seen_step: int) -> float:
        """Compute the reward for buzzing with a given answer.

        Dispatches on ``self.reward_mode``:

        ``simple``
            +1.0 for correct, -1.0 for incorrect.
        ``human_grounded``
            0.0 if the agent buzzes after the sampled human would have;
            otherwise +buzz_correct / +buzz_incorrect.
        ``time_penalty`` (default)
            +buzz_correct / +buzz_incorrect. The per-step wait penalty
            is applied separately in ``step()``.

        Parameters
        ----------
        question : MCQuestion
            Current question.
        chosen_idx : int
            Index of the chosen answer option (0-based).
        last_seen_step : int
            Step index of the last clue seen before buzzing.

        Returns
        -------
        float
            Reward value.
        """
        correct = chosen_idx == question.gold_index
        if self.reward_mode == "simple":
            return 1.0 if correct else -1.0
        if self.reward_mode == "human_grounded":
            token_pos = self._step_to_token_pos(last_seen_step)
            if self._sampled_human_buzz_pos is not None and token_pos > self._sampled_human_buzz_pos:
                return 0.0
            return self.buzz_correct if correct else self.buzz_incorrect
        # default: time_penalty
        return self.buzz_correct if correct else self.buzz_incorrect

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and start a new episode.

        Samples a random question from the pool, initializes belief to a
        uniform distribution, and returns the initial observation.

        Parameters
        ----------
        seed : int or None
            If provided, reseeds both the internal RNG and numpy's global
            RNG for reproducibility.
        options : dict or None
            Unused. Included for Gymnasium API compatibility.

        Returns
        -------
        observation : np.ndarray
            Initial observation of shape (K + 6,), dtype float32.
            Belief is uniform, so top_p = 1/K, margin = 0, entropy = max.
        info : dict[str, Any]
            Episode metadata. Contains ``"qid"`` (the sampled question ID).
        """
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)

        self.question = self._sample_question()
        self.step_idx = 0
        self.prev_belief = None
        self.belief = np.ones(self.K, dtype=np.float32) / self.K
        self.terminated = False
        self.truncated = False
        self._sampled_human_buzz_pos = self._sample_human_buzz(self.question)
        return self._obs(), {"qid": self.question.qid}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        If ``action == 0`` (WAIT):
            - Saves previous belief, computes new belief from current clue.
            - Applies wait_penalty if reward_mode is ``"time_penalty"``.
            - Advances step counter.
            - If all clues exhausted: forced termination with best-guess
              answer (``truncated=True``).

        If ``action in 1..K`` (BUZZ):
            - Computes buzz reward for chosen answer option ``action - 1``.
            - Episode ends (``terminated=True``).

        Parameters
        ----------
        action : int
            Action to take. 0 = WAIT, 1..K = buzz with option (action-1).

        Returns
        -------
```

```bash
venv/bin/python - <<'PY'
from qb_data.data_loader import QANTADatasetLoader
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.mc_builder import MCBuilder
from models.likelihoods import TfIdfLikelihood
from qb_env.tossup_env import TossupMCEnv
from qb_env.text_wrapper import TextObservationWrapper
qs = QANTADatasetLoader().load_from_csv('data/test_questions.csv')
pb = AnswerProfileBuilder().fit(qs)
mc = MCBuilder(K=4, strategy='category_random').build(qs, pb)
corpus = [q.question for q in mc] + [p for q in mc for p in q.option_profiles]
lm = TfIdfLikelihood(corpus_texts=corpus)
env = TossupMCEnv(questions=mc, likelihood_model=lm, K=4)
obs, info = env.reset(seed=13)
wrapped = TextObservationWrapper(env)
text_obs, _ = wrapped.reset(seed=13)
print('action_space =', env.action_space)
print('observation_space =', env.observation_space)
print('obs_shape =', obs.shape)
print('qid =', info['qid'])
print('wrapped_prefix =', text_obs[:160])
PY

```

```output
action_space = Discrete(5)
observation_space = Box(-inf, inf, (10,), float32)
obs_shape = (10,)
qid = hist_002
wrapped_prefix = CLUES: This | CHOICES: (1) Hamlet (2) Lighthouse of Alexandria (3) Python (4) Laissez-faire
```

```bash
sed -n '1,220p' qb_env/text_wrapper.py

```

```output
"""
TextObservationWrapper for converting belief features to text observations.

Wraps TossupMCEnv to provide text-formatted observations (clues + choices)
instead of numeric belief feature vectors. This bridges the gap between
the environment's native observation space (Box(K+6,)) and T5PolicyModel's
text input requirement.

The underlying environment still operates on beliefs internally for reward
computation -- the wrapper only transforms what the agent SEES, not how the
environment computes rewards or transitions.

Text format matches T5PolicyModel's expected input:
    "CLUES: clue1 clue2 ... | CHOICES: (1) ans1 (2) ans2 (3) ans3 (4) ans4"

Ported from qanta-buzzer's environment.py get_text_representation() method,
adapted for the unified codebase's Gymnasium wrapper pattern.
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np

from qb_data.mc_builder import MCQuestion


class TextObservationWrapper(gym.ObservationWrapper):
    """Wrap TossupMCEnv to provide text observations instead of belief features.

    The underlying env still operates on beliefs internally (for reward
    computation), but the agent sees text-formatted observations for T5 input.
    This is a Gymnasium ObservationWrapper that intercepts the observation
    returned by reset() and step() and converts it to a text string.

    The observation space is set to a placeholder Box(1,) since Gymnasium
    requires a defined space, but text observations are variable-length
    strings. Downstream code (T5PolicyModel) handles tokenization.

    Parameters
    ----------
    env : gym.Env
        The underlying TossupMCEnv instance. Must have ``question``
        (MCQuestion) and ``step_idx`` (int) attributes.

    Examples
    --------
    >>> from qb_env.tossup_env import TossupMCEnv
    >>> env = TossupMCEnv(questions=qs, likelihood_model=lm, K=4)
    >>> wrapped = TextObservationWrapper(env)
    >>> obs, info = wrapped.reset()
    >>> assert isinstance(obs, str)
    >>> assert "CLUES:" in obs and "CHOICES:" in obs
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Override observation space with a placeholder.
        # Text observations are variable-length strings; Gymnasium requires
        # a Space object, so we use a minimal Box as a sentinel.
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> str:
        """Convert numeric belief observation to formatted text string.

        Reconstructs visible clues from the underlying environment's current
        question and step index, then formats them with answer choices in the
        standard T5PolicyModel input format.

        Parameters
        ----------
        obs : np.ndarray
            Numeric belief features from the underlying environment.
            Shape ``(K+6,)``. Not used directly -- the text is reconstructed
            from ``env.question`` and ``env.step_idx``.

        Returns
        -------
        str
            Formatted text observation:
            ``"CLUES: <visible clue tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
        """
        question: MCQuestion = self.env.question
        step_idx: int = self.env.step_idx

        # Build visible clue text from cumulative prefixes.
        #
        # TossupMCEnv step semantics:
        #   - reset() sets step_idx=0, belief is uniform (no clues processed).
        #   - step(WAIT) calls _compute_belief(step_idx), THEN increments step_idx.
        #   - The observation returned after step() has step_idx ALREADY incremented.
        #
        # So step_idx tells us how many WAIT actions have been taken:
        #   step_idx=0: No WAITs yet; no clues processed; show minimal context
        #   step_idx=N: N WAITs taken; beliefs from cumulative_prefixes[0..N-1]
        #
        # cumulative_prefixes[i] = text of tokens[0..run_indices[i]].
        # After N WAITs, the agent has seen information up to
        # cumulative_prefixes[N-1], so that is what the text obs shows.
        if step_idx == 0:
            # No clues processed yet; show question start as minimal context
            # (matches initial observation having some textual content for T5)
            clues_text = question.tokens[0] if question.tokens else ""
        elif step_idx <= len(question.cumulative_prefixes):
            clues_text = question.cumulative_prefixes[step_idx - 1]
        else:
            # Past all clues (truncated episode); show all text
            clues_text = question.cumulative_prefixes[-1]

        # Format answer choices
        choices_parts = [
            f"({i + 1}) {opt}" for i, opt in enumerate(question.options)
        ]
        choices_text = " ".join(choices_parts)

        return f"CLUES: {clues_text} | CHOICES: {choices_text}"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and return a text observation.

        Parameters
        ----------
        seed : int or None
            Random seed passed to underlying environment.
        options : dict or None
            Options passed to underlying environment.

        Returns
        -------
        observation : str
            Text-formatted initial observation.
        info : dict[str, Any]
            Episode metadata from underlying environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: int
    ) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        """Execute one step and return text observation.

        Parameters
        ----------
        action : int
            Action to take. 0 = WAIT, 1..K = BUZZ with answer (action-1).

        Returns
        -------
        observation : str
            Text-formatted observation after the step.
        reward : float
            Scalar reward for this step.
        terminated : bool
            True if the agent buzzed (natural episode end).
        truncated : bool
            True if all clues exhausted (forced termination).
        info : dict[str, Any]
            Step metadata from underlying environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    @property
    def unwrapped_env(self):
        """Access the underlying TossupMCEnv directly.

        Returns
        -------
        TossupMCEnv
            The unwrapped environment instance.
        """
        return self.env
```

```bash
for path in qb_env/data_loader.py qb_env/mc_builder.py qb_env/text_utils.py qb_env/__init__.py; do
  printf '\n=== %s ===\n' "$path"
  sed -n '1,120p' "$path"
done

```

```output

=== qb_env/data_loader.py ===
"""qb-rl compatibility re-exports for tossup data loading."""

from qb_data.data_loader import (
    QANTADatasetLoader,
    TossupQuestion,
    load_tossup_questions,
    load_tossup_questions_from_config,
    parse_row,
)

__all__ = [
    "TossupQuestion",
    "QANTADatasetLoader",
    "parse_row",
    "load_tossup_questions",
    "load_tossup_questions_from_config",
]

=== qb_env/mc_builder.py ===
"""qb-rl compatibility re-exports for MC question building."""

from qb_data.mc_builder import MCBuilder, MCQuestion, _token_overlap

__all__ = ["MCQuestion", "MCBuilder", "_token_overlap"]

=== qb_env/text_utils.py ===
"""qb-rl compatibility re-exports for text utilities."""

from qb_data.text_utils import normalize_answer, tokenize_text

__all__ = ["normalize_answer", "tokenize_text"]

=== qb_env/__init__.py ===
"""Quiz Bowl Environment Package.

Gymnasium-compliant POMDP environment for quiz bowl question answering,
plus thin qb-rl compatibility exports for the old `qb_env.*` import paths.
"""

from qb_env.data_loader import (
    QANTADatasetLoader,
    TossupQuestion,
    load_tossup_questions,
    load_tossup_questions_from_config,
    parse_row,
)
from qb_env.mc_builder import MCBuilder, MCQuestion
from qb_env.text_utils import normalize_answer, tokenize_text
from qb_env.tossup_env import TossupMCEnv, make_env_from_config
from qb_env.text_wrapper import TextObservationWrapper

__all__ = [
    "TossupMCEnv",
    "make_env_from_config",
    "TextObservationWrapper",
    "TossupQuestion",
    "QANTADatasetLoader",
    "parse_row",
    "load_tossup_questions",
    "load_tossup_questions_from_config",
    "MCQuestion",
    "MCBuilder",
    "normalize_answer",
    "tokenize_text",
]
```

## 6. Agent layer

The agent layer sits on top of the environment and defines actual buzzing behavior. The repo keeps a ladder of increasingly capable behaviors:

- threshold-based heuristics
- Bayesian update baselines
- an always-buzz-final floor
- an SB3 PPO policy over belief features

All of them emit per-step traces so evaluation can compare not just final accuracy but also when and how the agent decided to act.

```bash
sed -n '1,240p' agents/threshold_buzzer.py

```

```output
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class EpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    reward_like: float
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


class ThresholdBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        self.belief: np.ndarray | None = None

    def _belief_from_prefix(self, prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(prefix, option_profiles)
        scores = scores - np.max(scores)
        probs = np.exp(self.beta * scores)
        probs = probs / max(1e-12, probs.sum())
        return probs.astype(np.float32)

    def _confidence_proxy(self, top_p: float) -> float:
        return _sigmoid(self.alpha * (top_p - self.threshold))

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        chosen_step = len(question.cumulative_prefixes) - 1
        chosen_idx = 0

        for step_idx, prefix in enumerate(question.cumulative_prefixes):
            belief = self._belief_from_prefix(prefix, question.option_profiles)
            self.belief = belief
            top_p = float(np.max(belief))
            top_idx = int(np.argmax(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
            c_t = self._confidence_proxy(top_p)
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        correct = chosen_idx == question.gold_index
        reward_like = 1.0 if correct else -0.5
        return EpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=correct,
            reward_like=reward_like,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


class AlwaysBuzzFinalBuzzer:
    def __init__(self, likelihood_model: LikelihoodModel, beta: float = 5.0):
        self.likelihood_model = likelihood_model
        self.beta = beta

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        final_step = len(question.cumulative_prefixes) - 1
        final_belief = np.ones(len(question.options), dtype=np.float32) / len(question.options)
        for prefix in question.cumulative_prefixes:
            scores = self.likelihood_model.score(prefix, question.option_profiles)
            scores = scores - np.max(scores)
            probs = np.exp(self.beta * scores)
            probs = probs / max(1e-12, probs.sum())
            final_belief = probs
            top_idx = int(np.argmax(probs))
            top_p = float(np.max(probs))
            entropy = float(-(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))).sum())
            c_trace.append(0.0)
            g_trace.append(1.0 if top_idx == question.gold_index else 0.0)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

        c_trace[-1] = 1.0
        buzz_idx = int(np.argmax(final_belief))
        correct = buzz_idx == question.gold_index
        reward_like = 1.0 if correct else -0.5
        return EpisodeResult(
            qid=question.qid,
            buzz_step=final_step,
            buzz_index=buzz_idx,
            gold_index=question.gold_index,
            correct=correct,
            reward_like=reward_like,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


def sweep_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
) -> dict[float, list[EpisodeResult]]:
    out: dict[float, list[EpisodeResult]] = {}
    for threshold in thresholds:
        agent = ThresholdBuzzer(
            likelihood_model=likelihood_model,
            threshold=float(threshold),
            beta=beta,
            alpha=alpha,
        )
        out[float(threshold)] = [agent.run_episode(q) for q in questions]
    return out


def result_to_dict(result: EpisodeResult) -> dict[str, Any]:
    return {
        "qid": result.qid,
        "buzz_step": result.buzz_step,
        "buzz_index": result.buzz_index,
        "gold_index": result.gold_index,
        "correct": result.correct,
        "reward_like": result.reward_like,
        "c_trace": result.c_trace,
        "g_trace": result.g_trace,
        "top_p_trace": result.top_p_trace,
        "entropy_trace": result.entropy_trace,
    }
```

```bash
sed -n '1,220p' agents/bayesian_buzzer.py

```

```output
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class SoftmaxEpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


class SoftmaxProfileBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        self.belief: np.ndarray | None = None

    def _belief_from_scratch(self, cumulative_prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(cumulative_prefix, option_profiles)
        scores = scores - np.max(scores)
        probs = np.exp(self.beta * scores)
        probs = probs / max(1e-12, probs.sum())
        return probs.astype(np.float32)

    def confidence_proxy(self, top_p: float) -> float:
        return _sigmoid(self.alpha * (top_p - self.threshold))

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        chosen_idx = 0
        chosen_step = len(question.cumulative_prefixes) - 1

        for step_idx, prefix in enumerate(question.cumulative_prefixes):
            belief = self._belief_from_scratch(prefix, question.option_profiles)
            self.belief = belief
            top_idx = int(np.argmax(belief))
            top_p = float(np.max(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
            c_t = self.confidence_proxy(top_p)
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        return SoftmaxEpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=(chosen_idx == question.gold_index),
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


class SequentialBayesBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha

    def _step_update(self, prior: np.ndarray, fragment: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(fragment, option_profiles)
        scores = scores - np.max(scores)
        likelihood = np.exp(self.beta * scores)
        posterior = prior * likelihood
        denom = posterior.sum()
        if denom <= 0:
            return np.ones_like(prior) / len(prior)
        return (posterior / denom).astype(np.float32)

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        K = len(question.options)
        belief = np.ones(K, dtype=np.float32) / K
        chosen_idx = 0
        chosen_step = len(question.cumulative_prefixes) - 1

        for step_idx, token_idx in enumerate(question.run_indices):
            prev_token_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
            fragment = " ".join(question.tokens[prev_token_idx + 1 : token_idx + 1])
            belief = self._step_update(belief, fragment, question.option_profiles)
            top_idx = int(np.argmax(belief))
            top_p = float(np.max(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
            c_t = _sigmoid(self.alpha * (top_p - self.threshold))
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        return SoftmaxEpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=(chosen_idx == question.gold_index),
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )
```

```bash
sed -n '1,260p' agents/ppo_buzzer.py

```

```output
"""PPO Buzzer agent wrapping Stable-Baselines3's PPO.

Provides the PPOBuzzer class for training an MLP policy on belief-feature
observations from TossupMCEnv, and PPOEpisodeTrace for recording per-step
action probabilities needed to compute the S_q scoring metric.

The key design rationale: SB3's ``learn()`` does not expose per-step action
distributions, so ``run_episode()`` implements custom episode execution that
records c_trace (buzz probability) and g_trace (correctness probability)
at each step for downstream S_q computation.

Ported from qb-rl reference implementation (agents/ppo_buzzer.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO

from qb_env.tossup_env import TossupMCEnv


@dataclass
class PPOEpisodeTrace:
    """Record of a single episode with per-step action probability traces.

    Used to compute the S_q scoring metric: S_q = sum(c_t * g_t) over steps.

    Attributes
    ----------
    qid : str
        Question identifier.
    buzz_step : int
        Step at which the agent buzzed (-1 if never buzzed voluntarily).
    buzz_index : int
        Index of the chosen answer option (0-based, -1 if forced).
    gold_index : int
        Index of the correct answer option (0-based).
    correct : bool
        Whether the agent selected the correct answer.
    episode_reward : float
        Total accumulated reward over the episode.
    c_trace : list[float]
        Per-step buzz probability: 1 - P(wait) at each timestep.
    g_trace : list[float]
        Per-step correctness probability: P(gold_option) / P(buzz).
    entropy_trace : list[float]
        Per-step policy entropy over the full action distribution.
    """

    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    episode_reward: float
    c_trace: list[float]
    g_trace: list[float]
    entropy_trace: list[float]


class PPOBuzzer:
    """PPO-trained buzzer agent wrapping Stable-Baselines3's PPO.

    Trains an MLP policy on belief-feature observations (Box(K+6,)) from
    TossupMCEnv. The policy maps observation vectors to a Discrete(K+1)
    action space: WAIT (0) or BUZZ with option i (1..K).

    Parameters
    ----------
    env : TossupMCEnv
        Gymnasium environment with belief-feature observations.
    learning_rate : float
        Learning rate for the Adam optimizer.
    n_steps : int
        Number of steps per rollout buffer collection.
    batch_size : int
        Minibatch size for PPO updates.
    n_epochs : int
        Number of optimization epochs per rollout.
    gamma : float
        Discount factor for return computation.
    policy_kwargs : dict or None
        Additional keyword arguments for the MLP policy. Defaults to
        ``{"net_arch": [64, 64]}`` (two hidden layers of 64 units).
    verbose : int
        SB3 verbosity level (0=silent, 1=info, 2=debug).
    """

    def __init__(
        self,
        env: TossupMCEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 128,
        batch_size: int = 32,
        n_epochs: int = 10,
        gamma: float = 0.99,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
    ):
        if policy_kwargs is None:
            policy_kwargs = {"net_arch": [64, 64]}

        self.env = env
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=verbose,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
        )

    def train(self, total_timesteps: int = 100_000) -> None:
        """Train the PPO policy for the specified number of timesteps.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to collect during training.
        """
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str | Path) -> None:
        """Save the trained PPO model to disk.

        Parameters
        ----------
        path : str or Path
            File path for the saved model (SB3 appends .zip if needed).
        """
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, env: TossupMCEnv) -> "PPOBuzzer":
        """Load a previously saved PPO model.

        Parameters
        ----------
        path : str or Path
            Path to the saved model file.
        env : TossupMCEnv
            Environment to attach to the loaded model.

        Returns
        -------
        PPOBuzzer
            A PPOBuzzer with the loaded model weights.
        """
        agent = cls(env=env)
        agent.model = PPO.load(str(path), env=env)
        return agent

    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Extract action probabilities from the policy for a given observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).

        Returns
        -------
        np.ndarray
            Action probability vector of shape (K + 1,), dtype float32.
            Index 0 = P(wait), indices 1..K = P(buzz with option i).
        """
        obs_tensor = th.as_tensor(
            obs, dtype=th.float32, device=self.model.device
        ).unsqueeze(0)
        dist = self.model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs[0].detach().cpu().numpy()
        return probs.astype(np.float32)

    def c_t(self, obs: np.ndarray) -> float:
        """Compute buzz probability at the current step.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).

        Returns
        -------
        float
            Probability of buzzing: 1 - P(wait). Range [0, 1].
        """
        probs = self.action_probabilities(obs)
        return float(1.0 - probs[0])

    def g_t(self, obs: np.ndarray, gold_index: int) -> float:
        """Compute correctness probability at the current step.

        Given that the agent buzzes, what is the probability it selects
        the correct answer? Formally: P(gold_action) / P(buzz).

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).
        gold_index : int
            Index of the correct answer option (0-based).

        Returns
        -------
        float
            Conditional correctness probability. Returns 0.0 if buzz
            probability is near zero (< 1e-12).
        """
        probs = self.action_probabilities(obs)
        c_t = float(1.0 - probs[0])
        if c_t <= 1e-12:
            return 0.0
        return float(probs[gold_index + 1] / c_t)

    def run_episode(
        self, deterministic: bool = False, seed: int | None = None
    ) -> PPOEpisodeTrace:
        """Run a full episode and record per-step action probability traces.

        Executes the policy in the environment, computing c_trace (buzz
        probability), g_trace (correctness probability), and entropy_trace
        at each step. These traces are needed to compute the S_q metric.

        Parameters
        ----------
        deterministic : bool
            If True, select actions by argmax instead of sampling.
        seed : int or None
            If provided, seeds the environment reset for reproducibility.

        Returns
        -------
        PPOEpisodeTrace
            Complete episode record with action traces and outcome.
        """
        obs, info = self.env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        c_trace: list[float] = []
        g_trace: list[float] = []
        entropy_trace: list[float] = []

        buzz_step = -1
        buzz_index = -1
        gold_index = (
            self.env.question.gold_index if self.env.question is not None else -1
        )

        while not (terminated or truncated):
```

```bash
sed -n '1,120p' agents/softmax_profile_buzzer.py

```

```output
"""qb-rl compatibility re-exports for Bayesian-family buzzers."""

from agents.bayesian_buzzer import (
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
    SoftmaxProfileBuzzer,
)

__all__ = [
    "SoftmaxEpisodeResult",
    "SoftmaxProfileBuzzer",
    "SequentialBayesBuzzer",
]
```

```bash
venv/bin/python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('artifacts/smoke/baseline_summary.json').read_text())
print('baseline_groups =', sorted(summary.keys()))
for name, payload in summary.items():
    if isinstance(payload, dict) and 'buzz_accuracy' in payload:
        print(name, 'accuracy=', payload['buzz_accuracy'], 'mean_sq=', payload.get('mean_sq'))
    elif isinstance(payload, dict):
        first_key = sorted(payload.keys())[0]
        print(name, 'sample_threshold=', first_key, 'metrics=', payload[first_key])
PY

```

```output
baseline_groups = ['always_final', 'sequential_bayes', 'softmax_profile', 'threshold']
threshold sample_threshold= 0.5 metrics= {'n': 44.0, 'buzz_accuracy': 0.38636363636363635, 'mean_buzz_step': 3.5, 'mean_sq': 0.24329479467724396, 'mean_reward_like': 0.07954545454545454, 'ece': 0.0, 'brier': 0.0, 'n_calibration': 44.0}
softmax_profile sample_threshold= 0.5 metrics= {'n': 44.0, 'buzz_accuracy': 0.38636363636363635, 'mean_buzz_step': 3.5, 'mean_sq': 0.24329479467724396, 'mean_reward_like': 0.0, 'ece': 0.0, 'brier': 0.0, 'n_calibration': 44.0}
sequential_bayes sample_threshold= 0.5 metrics= {'n': 44.0, 'buzz_accuracy': 0.38636363636363635, 'mean_buzz_step': 3.0454545454545454, 'mean_sq': 0.26743703225887505, 'mean_reward_like': 0.0, 'ece': 0.0, 'brier': 0.0, 'n_calibration': 44.0}
always_final accuracy= 0.38636363636363635 mean_sq= 0.38636363636363635
```

## 7. Evaluation layer

The evaluation stack is where the repo becomes a research artifact instead of just a training script. It computes `S_q`, calibration metrics, per-category breakdowns, and the control experiments that try to detect choice-space artifacts.

The core pattern is consistent across all agent types: collect traces, summarize them into comparable metrics, then rerun evaluation on controlled variants of the same multiple-choice dataset.

```bash
sed -n '1,220p' evaluation/metrics.py

```

```output
"""
Evaluation Metrics for Quiz Bowl Buzzer Agents

Computes buzz accuracy, S_q scoring, calibration metrics (ECE, Brier score),
and buzz timing statistics from episode trace data.

Ported from qb-rl reference implementation (evaluation/metrics.py).
Accepts both raw dicts and dataclass instances (EpisodeResult,
SoftmaxEpisodeResult, PPOEpisodeTrace) via the _to_dict helper.

Functions
---------
system_score(c_trace, g_trace)
    Compute S_q = sum_t b_t * g_t where b_t = c_t * prod_{i<t} (1 - c_i).
expected_calibration_error(confidences, outcomes, n_bins)
    Binned ECE over confidence-outcome pairs.
brier_score(confidences, outcomes)
    Mean squared error between confidence and binary outcome.
summarize_buzz_metrics(results)
    Aggregate accuracy, buzz step, S_q, and reward across episodes.
calibration_at_buzz(results)
    Extract buzz-time confidence and compute ECE + Brier score.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def _to_dict(item: Any) -> dict[str, Any]:
    """Convert dataclass or object to dict for uniform access.

    Parameters
    ----------
    item : Any
        A dict, dataclass instance, or object with __dict__.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the item.
    """
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    return item.__dict__


def system_score(c_trace: list[float], g_trace: list[float]) -> float:
    """Compute S_q scoring metric for a single episode.

    S_q = sum_t b_t * g_t, where b_t = c_t * prod_{i<t} (1 - c_i).
    This is the expected correctness under the agent's buzz policy,
    accounting for the survival probability of not having buzzed earlier.

    Parameters
    ----------
    c_trace : list[float]
        Buzz probability at each time step (confidence proxy).
    g_trace : list[float]
        Correctness indicator at each time step (1.0 if top answer is
        correct, 0.0 otherwise).

    Returns
    -------
    float
        S_q score for the episode, in [0, 1].
    """
    c = np.array(c_trace, dtype=np.float64)
    g = np.array(g_trace, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    b = np.zeros_like(c)
    survival = 1.0
    for t in range(len(c)):
        b[t] = c[t] * survival
        survival *= (1.0 - c[t])
    return float(np.sum(b * g))


def expected_calibration_error(
    confidences: list[float], outcomes: list[int], n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) with uniform binning.

    ECE measures the gap between predicted confidence and actual accuracy
    across confidence bins. Lower ECE indicates better-calibrated predictions.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).
    n_bins : int
        Number of uniform bins for confidence bucketing.

    Returns
    -------
    float
        Expected calibration error in [0, 1]. Returns 0.0 if no data.
    """
    if not confidences:
        return 0.0
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        bin_acc = y[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    """Compute Brier score (mean squared calibration error).

    Brier score measures the mean squared difference between predicted
    confidence and binary outcome. Lower is better; 0 is perfect.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).

    Returns
    -------
    float
        Brier score in [0, 1]. Returns 0.0 if no data.
    """
    if not confidences:
        return 0.0
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    return float(np.mean((conf - y) ** 2))


def summarize_buzz_metrics(results: list[Any]) -> dict[str, float]:
    """Aggregate buzz metrics across a list of episode results.

    Computes accuracy, mean buzz step, mean S_q score, and mean reward
    from episode trace data. Accepts dicts or dataclass instances.

    Parameters
    ----------
    results : list[Any]
        List of episode results (dicts, EpisodeResult, SoftmaxEpisodeResult,
        or PPOEpisodeTrace instances). Each must have: correct, buzz_step,
        c_trace, g_trace. Optionally: reward_like or episode_reward.

    Returns
    -------
    dict[str, float]
        Summary metrics: n, buzz_accuracy, mean_buzz_step, mean_sq,
        mean_reward_like.
    """
    rows = [_to_dict(r) for r in results]
    if not rows:
        return {
            "n": 0.0,
            "buzz_accuracy": 0.0,
            "mean_buzz_step": 0.0,
            "mean_sq": 0.0,
            "mean_reward_like": 0.0,
        }

    correct = np.array(
        [1 if bool(r.get("correct", False)) else 0 for r in rows],
        dtype=np.float64,
    )
    buzz_steps = np.array(
        [int(r.get("buzz_step", 0)) for r in rows], dtype=np.float64
    )
    sq_scores = np.array(
        [
            system_score(
                list(r.get("c_trace", [])),
                list(r.get("g_trace", [])),
            )
            for r in rows
        ],
        dtype=np.float64,
    )
    reward_like = np.array(
        [
            float(r.get("reward_like", r.get("episode_reward", 0.0)))
            for r in rows
        ],
        dtype=np.float64,
    )

    return {
        "n": float(len(rows)),
        "buzz_accuracy": float(correct.mean()),
        "mean_buzz_step": float(buzz_steps.mean()),
        "mean_sq": float(sq_scores.mean()),
        "mean_reward_like": float(reward_like.mean()),
    }


def per_category_accuracy(
    results: list[Any],
    questions: list[Any],
) -> dict[str, dict[str, float]]:
    """Compute accuracy and S_q metrics grouped by question category.

    Joins results with questions to extract category field, then groups
    and computes summarize_buzz_metrics per category.

```

```bash
sed -n '260,320p' evaluation/metrics.py

```

```output
def calibration_at_buzz(results: list[Any]) -> dict[str, float]:
    """Compute calibration metrics at the buzz decision point.

    Extracts buzz-time confidence (from g_trace at buzz_step) and
    correctness outcome, then computes ECE and Brier score.

    Parameters
    ----------
    results : list[Any]
        List of episode results (dicts or dataclass instances). Each must
        have: c_trace, g_trace, buzz_step, correct.

    Returns
    -------
    dict[str, float]
        Calibration metrics: ece, brier, n_calibration.
    """
    rows = [_to_dict(r) for r in results]
    confidences: list[float] = []
    outcomes: list[int] = []
    for row in rows:
        c_trace = list(row.get("c_trace", []))
        g_trace = list(row.get("g_trace", []))
        buzz_step = int(row.get("buzz_step", max(0, len(g_trace) - 1)))
        if not g_trace:
            continue
        idx = min(max(0, buzz_step), len(g_trace) - 1)
        confidences.append(float(g_trace[idx]))
        outcomes.append(1 if bool(row.get("correct", False)) else 0)

    return {
        "ece": expected_calibration_error(confidences, outcomes),
        "brier": brier_score(confidences, outcomes),
        "n_calibration": float(len(confidences)),
    }
```

```bash
sed -n '1,260p' evaluation/controls.py

```

```output
"""
Control Experiments for Quiz Bowl Buzzer Evaluation

Implements three control experiments to validate that the buzzer agent
genuinely uses question clues rather than exploiting surface-form artifacts:

1. **Choices-only control**: Strips all clues, trains a logistic regression
   on option surface features (char n-grams, length, capitalization). Expected
   accuracy ~25% (1/K) if options have no exploitable artifacts.

2. **Shuffle control**: Randomizes option ordering to verify the agent has
   no position bias. Performance should be unchanged.

3. **Alias substitution control**: Swaps answer text with aliases to verify
   robustness to surface-form changes.

Ported from qb-rl reference implementation (evaluation/controls.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from qb_data.mc_builder import MCQuestion


def _option_scalar_features(option: str) -> list[float]:
    """Extract scalar surface features from a single option string.

    Parameters
    ----------
    option : str
        Answer option text.

    Returns
    -------
    list[float]
        Six scalar features: char length, token count, has_parens,
        has_comma, is_title, is_lower.
    """
    tokens = option.split()
    has_parens = 1.0 if "(" in option or ")" in option else 0.0
    has_comma = 1.0 if "," in option else 0.0
    is_title = 1.0 if option.istitle() else 0.0
    is_lower = 1.0 if option.islower() else 0.0
    return [
        float(len(option)),
        float(len(tokens)),
        has_parens,
        has_comma,
        is_title,
        is_lower,
    ]


def _cross_option_features(options: list[str]) -> list[float]:
    """Extract cross-option comparative features.

    Parameters
    ----------
    options : list[str]
        All answer options for a question.

    Returns
    -------
    list[float]
        Three features: max/min length ratio, length std, number of
        distinct capitalization patterns.
    """
    lengths = np.array(
        [max(1, len(o.split())) for o in options], dtype=np.float32
    )
    cap_patterns = len(
        set(
            ("title" if o.istitle() else "lower" if o.islower() else "mixed")
            for o in options
        )
    )
    return [
        float(lengths.max() / lengths.min()),
        float(lengths.std()),
        float(cap_patterns),
    ]


def run_choices_only_control(
    questions: list[MCQuestion],
    random_seed: int = 13,
    test_fraction: float = 0.25,
) -> dict[str, float]:
    """Run choices-only control: predict answer from surface features only.

    Strips all question clues and trains a logistic regression on option
    surface features (char n-grams, length, capitalization patterns).
    Expected accuracy ~25% (1/K) if options are well-constructed.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    random_seed : int
        Seed for reproducible train/test split.
    test_fraction : float
        Fraction of questions held out for testing.

    Returns
    -------
    dict[str, float]
        Control results: accuracy, chance baseline, and test set size.
    """
    if not questions:
        return {"accuracy": 0.0, "chance": 0.0, "n_test": 0.0}

    rng = random.Random(random_seed)
    shuffled = questions[:]
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - test_fraction)))
    train_q = shuffled[:split_idx]
    test_q = shuffled[split_idx:]
    if not test_q:
        test_q = train_q

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
    vec.fit([opt for q in train_q for opt in q.options])

    def build_matrix(
        rows: list[MCQuestion],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        X = []
        y = []
        group_sizes: list[int] = []
        for q in rows:
            cross = _cross_option_features(q.options)
            group_sizes.append(len(q.options))
            tfidf = vec.transform(q.options).toarray()
            for i, option in enumerate(q.options):
                feat = np.array(
                    _option_scalar_features(option) + cross, dtype=np.float32
                )
                row = np.concatenate([feat, tfidf[i]], axis=0)
                X.append(row)
                y.append(1 if i == q.gold_index else 0)
        return np.array(X), np.array(y), group_sizes

    X_train, y_train, _ = build_matrix(train_q)
    X_test, y_test, test_group_sizes = build_matrix(test_q)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    offset = 0
    correct = 0
    total = 0
    for q, group_size in zip(test_q, test_group_sizes):
        group_probs = probs[offset : offset + group_size]
        pred_idx = int(np.argmax(group_probs))
        if pred_idx == q.gold_index:
            correct += 1
        total += 1
        offset += group_size

    accuracy = correct / max(1, total)
    chance = 1.0 / max(1, len(questions[0].options))
    return {
        "accuracy": float(accuracy),
        "chance": float(chance),
        "n_test": float(total),
    }


def shuffled_option_copy(
    question: MCQuestion, rng: random.Random
) -> MCQuestion:
    """Create a copy of an MCQuestion with shuffled option ordering.

    Parameters
    ----------
    question : MCQuestion
        Original question.
    rng : random.Random
        Random number generator for shuffling.

    Returns
    -------
    MCQuestion
        Copy with permuted options, profiles, answer_primary, and
        updated gold_index.
    """
    perm = list(range(len(question.options)))
    rng.shuffle(perm)
    new_options = [question.options[i] for i in perm]
    new_profiles = [question.option_profiles[i] for i in perm]
    new_answer_primary = [question.option_answer_primary[i] for i in perm]
    new_gold = perm.index(question.gold_index)
    return replace(
        question,
        options=new_options,
        option_profiles=new_profiles,
        option_answer_primary=new_answer_primary,
        gold_index=new_gold,
    )


def run_shuffle_control(
    questions: list[MCQuestion],
    evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run shuffle control: randomize option ordering and evaluate.

    Permutes the answer options for each question and runs the evaluator.
    If the agent has no position bias, performance should be unchanged.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    evaluator : callable
        Function that takes a list of MCQuestion and returns a metrics dict.
    random_seed : int
        Seed for reproducible shuffling.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics on shuffled questions.
    """
    rng = random.Random(random_seed)
    shuffled = [shuffled_option_copy(q, rng) for q in questions]
    return evaluator(shuffled)


def alias_substitution_copy(
    question: MCQuestion,
    alias_lookup: dict[str, list[str]],
    rng: random.Random,
) -> MCQuestion:
    """Create a copy of an MCQuestion with alias-substituted options.

    Parameters
    ----------
    question : MCQuestion
        Original question.
    alias_lookup : dict[str, list[str]]
        Mapping from canonical answer to list of known aliases.
    rng : random.Random
        Random number generator for alias selection.

    Returns
    -------
    MCQuestion
        Copy with alias-substituted option text and profiles.
    """
```

```bash
sed -n '300,360p' evaluation/controls.py

```

```output
    random_seed : int
        Seed for reproducible alias selection.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics on alias-substituted questions.
    """
    rng = random.Random(random_seed)
    swapped = [
        alias_substitution_copy(q, alias_lookup=alias_lookup, rng=rng)
        for q in questions
    ]
    return evaluator(swapped)


def bootstrap_ci(
    values: list[float],
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : list[float]
        Observed values.
    n_samples : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (0.05 = 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    samples = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(arr), size=len(arr))
        samples.append(float(arr[idx].mean()))
    lo = np.quantile(samples, alpha / 2.0)
    hi = np.quantile(samples, 1.0 - alpha / 2.0)
    return float(lo), float(hi)
```

```bash
rg -n "^def " evaluation/plotting.py

```

```output
25:def _ensure_parent(path: str | Path) -> Path:
43:def plot_learning_curve(
76:def plot_entropy_vs_clue_index(
111:def plot_calibration_curve(
166:def save_comparison_table(
```

```bash
venv/bin/python - <<'PY'
import csv, json
from pathlib import Path
report = json.loads(Path('artifacts/smoke/evaluation_report.json').read_text())
print('full_eval_keys =', sorted(report['full_eval'].keys()))
print('controls =', sorted(report['controls'].keys()))
print('full_eval_mean_sq =', report['full_eval'].get('mean_sq'))
print('choices_only =', report['controls']['choices_only'])
print('per_category_count =', len(report['per_category']))
with open('artifacts/smoke/plots/comparison.csv', newline='') as fh:
    rows = list(csv.reader(fh))
print('comparison_csv_head =')
for row in rows[:5]:
    print(row)
PY

```

```output
full_eval_keys = ['brier', 'buzz_accuracy', 'ece', 'mean_buzz_step', 'mean_reward_like', 'mean_sq', 'n', 'n_calibration']
controls = ['alias_substitution', 'choices_only', 'shuffle']
full_eval_mean_sq = 0.24329479467724396
choices_only = {'accuracy': 0.09090909090909091, 'chance': 0.25, 'n_test': 11.0}
per_category_count = 11
comparison_csv_head =
['agent', 'n', 'buzz_accuracy', 'mean_buzz_step', 'mean_sq', 'mean_reward_like', 'ece', 'brier', 'n_calibration']
['threshold_0.5', '44.0', '0.38636363636363635', '3.5', '0.24329479467724396', '0.07954545454545454', '0.0', '0.0', '44.0']
['threshold_0.7', '44.0', '0.38636363636363635', '3.9545454545454546', '0.12951727467749788', '0.07954545454545454', '0.0', '0.0', '44.0']
['threshold_0.9', '44.0', '0.38636363636363635', '4.045454545454546', '0.05277608956128622', '0.07954545454545454', '0.0', '0.0', '44.0']
['softmax_0.5', '44.0', '0.38636363636363635', '3.5', '0.24329479467724396', '0.0', '0.0', '0.0', '44.0']
```

## 8. T5 training track

The T5 track is the heavier alternative to the belief-feature PPO path. Instead of giving the policy a numeric summary of belief state, it feeds text observations directly into a T5 encoder with separate wait, answer, and value heads.

The code is split into a supervised warm-start stage, then a custom PPO trainer with rollout buffering and GAE. The scripts in `scripts/` are thin orchestration layers over the implementations in `training/` and `models/`.

```bash
sed -n '1,240p' training/train_supervised_t5.py

```

```output
"""
Supervised warm-start training for T5PolicyModel.

Trains answer selection on complete questions using cross-entropy loss. All
clues are shown at once (not incremental), providing a strong initialization
before PPO fine-tuning on partial observations.

The training loop uses gradient accumulation (default 4 steps, effective
batch = 32) for stable training without exceeding GPU memory. Best model
is saved by validation accuracy to checkpoints/supervised/best_model/.

Ported from qanta-buzzer reference implementation (train_supervised.py)
with these changes:
    - Accepts list of MCQuestion objects instead of QuizBowlDataset class
    - Config dict interface instead of qanta-buzzer's Config class
    - Direct text formatting from MCQuestion (no QuizBowlEnvironment needed)
    - NumPy-style docstrings added throughout

Usage
-----
From Python::

    from training.train_supervised_t5 import SupervisedTrainer, run_supervised_training
    from models.t5_policy import T5PolicyModel
    from qb_data.mc_builder import MCQuestion

    model = T5PolicyModel({"model_name": "t5-small", "device": "cpu"})
    trainer = SupervisedTrainer(model, train_qs, val_qs, config)
    trainer.train()

From command line::

    python -m training.train_supervised_t5 --config configs/t5_policy.yaml
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion


def format_question_text(question: MCQuestion) -> str:
    """Format a complete question as text for supervised training.

    Shows ALL clues (complete question) since supervised training is the
    easier task of answer selection on full information. PPO later trains
    on incremental clues.

    Parameters
    ----------
    question : MCQuestion
        Question with tokens, options, and gold_index.

    Returns
    -------
    str
        Formatted text: ``"CLUES: <all tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
    """
    clues_text = " ".join(question.tokens)
    choices_parts = [f"({i + 1}) {opt}" for i, opt in enumerate(question.options)]
    choices_text = " ".join(choices_parts)
    return f"CLUES: {clues_text} | CHOICES: {choices_text}"


class SupervisedTrainer:
    """Trainer for supervised warm-start of T5PolicyModel.

    Trains the answer head using cross-entropy loss on complete questions
    (all clues shown at once). Uses gradient accumulation for stable training
    with large effective batch sizes without exceeding GPU memory.

    The training loop:
    1. Shuffles training data each epoch
    2. Iterates over mini-batches
    3. Computes cross-entropy loss on answer logits
    4. Accumulates gradients for ``grad_accum_steps`` batches
    5. Clips gradients and updates optimizer
    6. Validates after each epoch
    7. Saves best model by validation accuracy

    Parameters
    ----------
    model : T5PolicyModel
        Model to train. Must have ``predict_answer`` and ``tokenizer``.
    train_questions : list[MCQuestion]
        Training set questions.
    val_questions : list[MCQuestion]
        Validation set questions.
    config : dict[str, Any]
        Configuration dictionary with keys:

        - ``supervised_lr`` (float): Learning rate. Default 3e-4.
        - ``supervised_epochs`` (int): Number of epochs. Default 10.
        - ``supervised_batch_size`` (int): Batch size. Default 8.
        - ``supervised_grad_accum_steps`` (int): Gradient accumulation. Default 4.
        - ``checkpoint_dir`` (str): Base checkpoint directory. Default "checkpoints".
        - ``max_input_length`` (int): Max token length. Default 512.
        - ``max_grad_norm`` (float): Gradient clip norm. Default 1.0.
        - ``weight_decay`` (float): AdamW weight decay. Default 0.01.

    Attributes
    ----------
    model : T5PolicyModel
        The model being trained.
    optimizer : torch.optim.AdamW
        Optimizer with weight decay.
    criterion : nn.CrossEntropyLoss
        Loss function for answer classification.
    best_val_acc : float
        Best validation accuracy seen so far.
    train_history : list[dict]
        Per-epoch training metrics.
    val_history : list[dict]
        Per-epoch validation metrics.
    checkpoint_dir : Path
        Directory for saving checkpoints.
    """

    def __init__(
        self,
        model: T5PolicyModel,
        train_questions: List[MCQuestion],
        val_questions: List[MCQuestion],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_questions = list(train_questions)
        self.val_questions = list(val_questions)
        self.config = config

        self.device = model.device

        # Hyperparameters with defaults
        self.lr = float(config.get("supervised_lr", 3e-4))
        self.epochs = int(config.get("supervised_epochs", 10))
        self.batch_size = int(config.get("supervised_batch_size", 8))
        self.grad_accum_steps = int(config.get("supervised_grad_accum_steps", 4))
        self.max_input_length = int(config.get("max_input_length", 512))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.weight_decay = float(config.get("weight_decay", 0.01))

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history: List[Dict[str, Any]] = []
        self.val_history: List[Dict[str, Any]] = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints")) / "supervised"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def prepare_batch(
        self, questions: List[MCQuestion]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Format a batch of complete questions as tokenized tensors.

        Each question is formatted with ALL clues visible (supervised training
        shows complete information). Text is tokenized using the model's
        T5TokenizerFast.

        Parameters
        ----------
        questions : list[MCQuestion]
            Batch of questions to format.

        Returns
        -------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``, on device.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``, on device.
        labels : torch.Tensor
            Gold answer indices of shape ``[batch_size]``, on device.
        """
        texts = [format_question_text(q) for q in questions]
        labels = [q.gold_index for q in questions]

        # Tokenize
        inputs = self.model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

        return input_ids, attention_mask, labels_tensor

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with gradient accumulation.

        Shuffles training data, iterates over mini-batches, and updates
        the optimizer every ``grad_accum_steps`` batches. Gradients are
        clipped to ``max_grad_norm`` before each optimizer step.

        Returns
        -------
        epoch_loss : float
            Average loss over all batches in the epoch.
        epoch_acc : float
            Average accuracy over all batches in the epoch.
        """
        self.model.train()

        # Shuffle training data
        shuffled = self.train_questions[:]
        random.shuffle(shuffled)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = max(1, len(shuffled) // self.batch_size)

        # Zero gradients at start
        self.optimizer.zero_grad()

        for batch_idx in range(num_batches):
            # Get batch
```

```bash
rg -n "class RolloutStep|class RolloutBuffer|class PPOTrainer|def run_ppo_training" training/train_ppo_t5.py

```

```output
59:class RolloutStep:
102:class RolloutBuffer:
197:class PPOTrainer:
829:def run_ppo_training(
```

```bash
sed -n '1,260p' training/train_ppo_t5.py

```

```output
"""
Custom PPO Training for T5 Policy Model

Implements PPOTrainer with RolloutBuffer for end-to-end PPO fine-tuning of
T5PolicyModel on incremental quiz bowl episodes. Uses Generalized Advantage
Estimation (GAE) for variance reduction and dynamic batch padding to minimize
memory footprint.

Key design decisions:
    - Rollout tensors (input_ids, attention_mask) are immediately detached and
      moved to CPU after collection to prevent GPU memory accumulation.
    - Dynamic padding: each mini-batch is padded to the max length within that
      batch, not a global 512-token maximum, saving ~50%+ memory.
    - Config-dict interface for compatibility with the unified codebase YAML
      config pattern (see configs/t5_policy.yaml).

Ported from qanta-buzzer reference implementation (train_ppo.py) with:
    - TextObservationWrapper for text-based rollout collection
    - Memory-safe tensor management (detach + CPU storage)
    - Dynamic padding per mini-batch
    - Config dict interface replacing Config class
    - NumPy-style docstrings

Usage
-----
From Python::

    from training.train_ppo_t5 import PPOTrainer, run_ppo_training
    from models.t5_policy import T5PolicyModel
    from qb_data.mc_builder import MCQuestion

    model = T5PolicyModel({"model_name": "t5-small", "device": "cpu"})
    trainer = PPOTrainer(model, train_qs, val_qs, config)
    trainer.train()

From command line::

    python scripts/train_t5_policy.py --config configs/t5_policy.yaml
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion


@dataclass
class RolloutStep:
    """Single step in an episode rollout.

    Stores observation text, action, reward, value estimate, and log probability
    for a single environment step. Tokenized tensors (input_ids, attention_mask)
    are stored on CPU to prevent GPU memory accumulation during rollout collection.

    Attributes
    ----------
    observation_text : str
        Text observation at this step (CLUES: ... | CHOICES: ...).
    action : int
        Combined action taken (0=WAIT, 1..K=SELECT).
    reward : float
        Scalar reward received.
    done : bool
        Whether this step ended the episode.
    value : float
        Value estimate from the critic at this step.
    log_prob : float
        Log probability of the action under the policy at collection time.
    input_ids : torch.Tensor or None
        Tokenized input IDs stored on CPU. Shape ``[1, seq_len]``.
    attention_mask : torch.Tensor or None
        Attention mask stored on CPU. Shape ``[1, seq_len]``.
    return_ : float
        Discounted return (filled by ``compute_returns_and_advantages``).
    advantage : float
        GAE advantage (filled by ``compute_returns_and_advantages``).
    """

    observation_text: str
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    return_: float = 0.0
    advantage: float = 0.0


class RolloutBuffer:
    """Buffer to store and process episode rollouts for PPO updates.

    Accumulates complete episode rollouts (lists of RolloutStep), then computes
    discounted returns and GAE advantages across all episodes. Provides a flat
    view of all steps for mini-batch iteration during PPO updates.

    Attributes
    ----------
    rollouts : list[list[RolloutStep]]
        List of episode rollouts, each a list of steps.
    """

    def __init__(self) -> None:
        self.rollouts: List[List[RolloutStep]] = []

    def reset(self) -> None:
        """Clear all stored rollouts."""
        self.rollouts = []

    def add_rollout(self, steps: List[RolloutStep]) -> None:
        """Add a complete episode rollout to the buffer.

        Parameters
        ----------
        steps : list[RolloutStep]
            Complete episode rollout (ordered list of steps from reset to done).
        """
        self.rollouts.append(steps)

    def get_all_steps(self) -> List[RolloutStep]:
        """Get a flat list of all steps from all rollouts.

        Returns
        -------
        list[RolloutStep]
            All steps concatenated in order (rollout 0 steps, then rollout 1, ...).
        """
        all_steps: List[RolloutStep] = []
        for rollout in self.rollouts:
            all_steps.extend(rollout)
        return all_steps

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float
    ) -> None:
        """Compute discounted returns and GAE advantages for all rollouts.

        Uses Generalized Advantage Estimation (GAE) to compute per-step
        advantages. For each rollout, iterates backward from the terminal
        step computing:

            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = delta_t + gamma * lambda * A_{t+1}
            G_t = A_t + V(s_t)

        Terminal states reset next_value and gae to 0.

        Parameters
        ----------
        gamma : float
            Discount factor in [0, 1]. Higher values weight future rewards more.
        gae_lambda : float
            GAE lambda in [0, 1]. Trades off bias (low) vs variance (high).
        """
        for rollout in self.rollouts:
            rewards = [step.reward for step in rollout]
            values = [step.value for step in rollout]
            dones = [step.done for step in rollout]

            # GAE computation (backward pass)
            gae = 0.0
            next_value = 0.0  # Terminal state value

            for t in reversed(range(len(rollout))):
                if dones[t]:
                    next_value = 0.0
                    gae = 0.0

                # TD error
                delta = rewards[t] + gamma * next_value - values[t]

                # GAE accumulation
                gae = delta + gamma * gae_lambda * gae

                # Store return and advantage
                rollout[t].return_ = gae + values[t]
                rollout[t].advantage = gae

                next_value = values[t]

    def __len__(self) -> int:
        return len(self.rollouts)


class PPOTrainer:
    """Custom PPO trainer for T5PolicyModel on quiz bowl episodes.

    Collects rollouts by running T5PolicyModel in text-observation episodes
    (via TextObservationWrapper), then updates the policy using clipped
    surrogate PPO loss with value function and entropy regularization.

    The trainer handles the complete training loop:
    1. Collect rollouts (episodes) using the current policy
    2. Compute GAE advantages
    3. Update policy with mini-batch PPO for multiple epochs
    4. Periodically validate and save checkpoints

    Parameters
    ----------
    model : T5PolicyModel
        T5 policy model to train. Should be pre-trained via supervised
        warm-start for faster convergence.
    train_questions : list[MCQuestion]
        Training set questions for rollout collection.
    val_questions : list[MCQuestion]
        Validation set questions for periodic evaluation.
    config : dict[str, Any]
        Configuration dictionary with PPO hyperparameters:

        - ``ppo_lr`` (float): Learning rate. Default 1e-5.
        - ``ppo_iterations`` (int): Number of collect-update cycles. Default 100.
        - ``ppo_batch_size`` (int): Mini-batch size for PPO updates. Default 8.
        - ``ppo_epochs_per_iter`` (int): PPO epochs per iteration. Default 4.
        - ``ppo_gamma`` (float): Discount factor. Default 0.99.
        - ``ppo_gae_lambda`` (float): GAE lambda. Default 0.95.
        - ``ppo_clip_ratio`` (float): PPO clip ratio. Default 0.2.
        - ``ppo_value_coef`` (float): Value loss coefficient. Default 0.5.
        - ``ppo_entropy_coef`` (float): Entropy bonus coefficient. Default 0.01.
        - ``ppo_max_grad_norm`` (float): Gradient clip norm. Default 0.5.
        - ``ppo_episodes_per_iter`` (int): Episodes per rollout. Default 16.
        - ``eval_interval`` (int): Validate every N iterations. Default 10.
        - ``save_interval`` (int): Save checkpoint every N iterations. Default 20.
        - ``checkpoint_dir`` (str): Base checkpoint directory. Default "checkpoints".
        - ``reward_time_penalty`` (float): Time penalty for env. Default 0.1.

    Attributes
    ----------
    model : T5PolicyModel
        The model being trained.
    optimizer : torch.optim.AdamW
        Optimizer with weight decay.
    best_val_reward : float
        Best validation reward seen so far.
    history : list[dict]
        Per-iteration training metrics.
    checkpoint_dir : Path
        Directory for saving PPO checkpoints.
    """

    def __init__(
        self,
        model: T5PolicyModel,
        train_questions: List[MCQuestion],
        val_questions: List[MCQuestion],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_questions = list(train_questions)
```

```bash
sed -n '1,220p' scripts/train_t5_policy.py

```

```output
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

from scripts._common import ARTIFACT_DIR, load_mc_questions


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
```

```bash
sed -n '1,260p' scripts/compare_policies.py

```

```output
#!/usr/bin/env python3
"""
Compare T5-as-likelihood (MLP policy) vs T5-as-policy (end-to-end).

Evaluates both approaches on the same test set with identical metrics
(accuracy, S_q, ECE, Brier score, buzz position) for a fair comparison.

MLP Policy (Phase 4):
    T5 computes likelihood scores -> belief features -> MLP policy decides.
    Uses SB3 PPO with belief-feature observations from TossupMCEnv.

T5 Policy (Phase 6):
    T5 encoder processes text directly -> PolicyHead decides.
    Uses custom PPO with text observations via TextObservationWrapper.

Usage:
    # With trained checkpoints
    python scripts/compare_policies.py \\
        --mlp-checkpoint checkpoints/ppo/best_model \\
        --t5-checkpoint checkpoints/ppo_t5/best_model \\
        --output results/t5_comparison.json

    # Smoke test with subset
    python scripts/compare_policies.py \\
        --mlp-checkpoint checkpoints/ppo/best_model \\
        --t5-checkpoint checkpoints/ppo_t5/best_model \\
        --smoke

    # Only evaluate T5 policy (skip MLP if not trained)
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

    # Extract confidences and outcomes for calibration
    from dataclasses import asdict

    rows = [asdict(r) for r in results]
    confidences = []
    outcomes = []
    buzz_positions = []
    for row in rows:
        c_trace = list(row.get("c_trace", []))
        g_trace = list(row.get("g_trace", []))
        buzz_step = int(row.get("buzz_step", max(0, len(g_trace) - 1)))
        if g_trace:
            idx = min(max(0, buzz_step), len(g_trace) - 1)
            confidences.append(float(g_trace[idx]))
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
                wait_penalty=0.01,
                belief_mode="from_scratch",
            )
            wrapped_env = TextObservationWrapper(env)

            obs, info = wrapped_env.reset()
            done = False
            c_trace = []
            g_trace = []
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

                # Extract buzz probability from wait head
                wait_probs = act_info["wait_probs"]
                buzz_prob = wait_probs[0, 1].item()  # P(answer_now)
                c_trace.append(buzz_prob)

                # Correctness at this step: P(gold answer | answering)
                answer_probs = act_info["answer_probs"]
                gold_prob = answer_probs[0, question.gold_index].item()
                g_trace.append(gold_prob)

                obs, reward, terminated, truncated, step_info = (
                    wrapped_env.step(action)
                )
                done = terminated or truncated
                episode_reward += reward
                step_count += 1

            # Compute S_q for this episode
            sq = system_score(c_trace, g_trace)
            sq_scores.append(sq)

            # Check correctness
            is_correct = step_info.get("correct", False) or step_info.get(
                "forced_correct", False
            )
            if is_correct:
                correct_count += 1
            total_count += 1

            # Calibration data at buzz point
            if c_trace:
                buzz_step = step_count - 1
```

## 9. Legacy prototype path

The repo still contains the older qanta-buzzer prototype at the repo root. These files predate the modular refactor and bundle concepts that are now split across `qb_data/`, `qb_env/`, `models/`, `agents/`, `evaluation/`, and `scripts/`.

They still matter for understanding project history: the T5 policy idea, the older environment abstraction, and the earlier training entrypoints all originated here before the codebase was reorganized around the qb-rl-style modular pipeline.

```bash
sed -n '1,220p' main.py

```

```output
"""
Main training script for CS234 RL Question Answering project
"""

import argparse
import torch
from pathlib import Path

from config import Config
from dataset import setup_datasets
from train_supervised import run_supervised_training
from train_ppo import run_ppo_training
from metrics import evaluate_model, evaluate_choices_only
from model import T5PolicyModel


def parse_args():
    parser = argparse.ArgumentParser(description='CS234 RL Question Answering')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['supervised', 'ppo', 'full', 'eval'],
                       help='Training mode: supervised, ppo, full (both), or eval')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model (for ppo or eval mode)')
    
    parser.add_argument('--supervised_epochs', type=int, default=None,
                       help='Number of supervised epochs (overrides config)')
    
    parser.add_argument('--ppo_iterations', type=int, default=None,
                       help='Number of PPO iterations (overrides config)')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (overrides config)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--num_questions', type=int, default=None,
                       help='Number of questions in dataset (overrides config)')
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration with command-line overrides"""
    config = Config()
    
    # Override with command-line arguments
    if args.supervised_epochs is not None:
        config.SUPERVISED_EPOCHS = args.supervised_epochs
    
    if args.ppo_iterations is not None:
        config.PPO_ITERATIONS = args.ppo_iterations
    
    if args.batch_size is not None:
        config.PPO_BATCH_SIZE = args.batch_size
        config.SUPERVISED_BATCH_SIZE = args.batch_size
    
    if args.device is not None:
        config.DEVICE = args.device
    
    if args.num_questions is not None:
        config.NUM_QUESTIONS = args.num_questions
    
    config.SEED = args.seed
    
    return config


def main():
    args = parse_args()
    
    # Setup configuration
    config = setup_config(args)
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    import numpy as np
    import random
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    # Print configuration
    config.print_config()
    
    # Setup datasets
    print("\nSetting up datasets...")
    train_dataset, val_dataset, test_dataset = setup_datasets(config)
    
    # Mode-specific execution
    if args.mode == 'supervised':
        print("\n" + "=" * 60)
        print("Running supervised training only")
        print("=" * 60)
        run_supervised_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )
    
    elif args.mode == 'ppo':
        print("\n" + "=" * 60)
        print("Running PPO training only")
        print("=" * 60)
        
        # Determine pretrained model path
        if args.model_path:
            pretrained_path = args.model_path
        else:
            pretrained_path = Path(config.CHECKPOINT_DIR) / "supervised" / "best_model"
            if not pretrained_path.exists():
                print(f"\nWARNING: No pretrained model found at {pretrained_path}")
                print("Starting PPO without pretraining (not recommended)")
                pretrained_path = None
        
        run_ppo_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            pretrained_model_path=str(pretrained_path) if pretrained_path else None
        )
    
    elif args.mode == 'full':
        print("\n" + "=" * 60)
        print("Running full pipeline: supervised + PPO")
        print("=" * 60)
        
        # Phase 1: Supervised training
        print("\n### PHASE 1: SUPERVISED TRAINING ###\n")
        run_supervised_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=None  # Don't evaluate yet
        )
        
        # Phase 2: PPO training
        print("\n### PHASE 2: PPO TRAINING ###\n")
        supervised_path = Path(config.CHECKPOINT_DIR) / "supervised" / "best_model"
        run_ppo_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,  # Final evaluation after PPO
            pretrained_model_path=str(supervised_path)
        )
    
    elif args.mode == 'eval':
        print("\n" + "=" * 60)
        print("Running evaluation only")
        print("=" * 60)
        
        if not args.model_path:
            print("ERROR: --model_path required for eval mode")
            return
        
        # Load model
        print(f"Loading model from {args.model_path}")
        model = T5PolicyModel.load_pretrained(args.model_path, device=config.DEVICE)
        model.to(config.DEVICE)
        
        # Evaluate on test set
        print("\n### Full Question Evaluation ###")
        metrics = evaluate_model(model, test_dataset, device=config.DEVICE)
        metrics.print_summary()
        
        # Choices-only control
        print("\n### Choices-Only Control Experiment ###")
        choices_metrics = evaluate_choices_only(model, test_dataset, device=config.DEVICE)
        print(f"Accuracy (choices only): {choices_metrics.compute_accuracy():.4f}")
        print(f"Random baseline: 0.25 (1/4 choices)")
        print(f"ECE: {choices_metrics.compute_ece():.4f}")
        
        # Save results
        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(exist_ok=True)
        
        import json
        results = {
            'full_question': metrics.get_summary(),
            'choices_only': {
                'accuracy': choices_metrics.compute_accuracy(),
                'ece': choices_metrics.compute_ece()
            }
        }
        
        results_path = results_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

```bash
sed -n '1,220p' environment.py

```

```output
"""
POMDP Environment for Quiz Bowl Question Answering
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Question:
    """Represents a quiz bowl question with pyramidal clues"""
    question_id: str
    clues: List[str]  # Ordered from difficult to easy
    answer_choices: List[str]  # 4 choices: [correct, distractor1, distractor2, distractor3]
    correct_answer_idx: int  # Index of correct answer (0-3)
    category: str
    metadata: Optional[Dict] = None


class QuizBowlEnvironment:
    """
    POMDP Environment for incremental question answering.
    
    States: Complete questions with all clues
    Observations: Partial questions (clues revealed so far) + answer choices
    Actions: WAIT (0) or SELECT answer i (1-4)
    Rewards: Shaped reward based on correctness and timing
    """
    
    WAIT_ACTION = 0
    
    def __init__(self, question: Question, reward_time_penalty: float = 0.1):
        """
        Initialize environment with a question.
        
        Args:
            question: Question object containing clues and answers
            reward_time_penalty: Penalty coefficient for late answering
        """
        self.question = question
        self.reward_time_penalty = reward_time_penalty
        
        self.num_clues = len(question.clues)
        self.num_actions = 1 + len(question.answer_choices)  # WAIT + SELECT answer
        
        # Episode state
        self.current_clue_idx = 0
        self.done = False
        self.selected_answer = None
        
    def reset(self) -> Dict:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_clue_idx = 0
        self.done = False
        self.selected_answer = None
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: 0 for WAIT, 1-4 for SELECT answer choice
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            raise ValueError("Episode is already done. Call reset().")
        
        info = {
            'clue_position': self.current_clue_idx,
            'total_clues': self.num_clues
        }
        
        # Action is WAIT
        if action == self.WAIT_ACTION:
            self.current_clue_idx += 1
            
            # Check if we've run out of clues
            if self.current_clue_idx >= self.num_clues:
                # Forced to answer at last clue
                self.done = True
                info['forced_answer'] = True
                return self._get_observation(), 0.0, True, info
            
            # Continue episode
            return self._get_observation(), 0.0, False, info
        
        # Action is SELECT answer (1-4 maps to 0-3)
        else:
            answer_idx = action - 1
            
            if answer_idx < 0 or answer_idx >= len(self.question.answer_choices):
                raise ValueError(f"Invalid action: {action}. Must be 0-{self.num_actions-1}")
            
            self.selected_answer = answer_idx
            self.done = True
            
            # Compute reward
            is_correct = (answer_idx == self.question.correct_answer_idx)
            time_penalty = self.reward_time_penalty * (self.current_clue_idx / self.num_clues)
            
            if is_correct:
                reward = 1.0 - time_penalty
            else:
                reward = -time_penalty
            
            info['is_correct'] = is_correct
            info['answer_idx'] = answer_idx
            info['correct_idx'] = self.question.correct_answer_idx
            
            return self._get_observation(), reward, True, info
    
    def _get_observation(self) -> Dict:
        """
        Get current observation (partial question + answer choices).
        
        Returns:
            Dictionary containing visible clues and answer choices
        """
        visible_clues = self.question.clues[:self.current_clue_idx + 1]
        
        return {
            'clues': visible_clues,
            'answer_choices': self.question.answer_choices,
            'clue_position': self.current_clue_idx,
            'total_clues': self.num_clues,
            'category': self.question.category
        }
    
    def get_text_representation(self, observation: Optional[Dict] = None) -> str:
        """
        Convert observation to text string for model input.
        
        Args:
            observation: If None, use current observation
            
        Returns:
            Formatted text string
        """
        if observation is None:
            observation = self._get_observation()
        
        clues_text = " ".join(observation['clues'])
        choices_text = " | ".join([f"({i+1}) {choice}" 
                                   for i, choice in enumerate(observation['answer_choices'])])
        
        return f"CLUES: {clues_text} | CHOICES: {choices_text}"
    
    def get_choices_only_text(self) -> str:
        """Get text with only answer choices (for control experiment)"""
        choices_text = " | ".join([f"({i+1}) {choice}" 
                                   for i, choice in enumerate(self.question.answer_choices)])
        return f"CHOICES: {choices_text}"
    
    def render(self) -> str:
        """Render current state as string"""
        obs = self._get_observation()
        
        output = [
            f"Question ID: {self.question.question_id}",
            f"Category: {self.question.category}",
            f"Clue Position: {self.current_clue_idx + 1}/{self.num_clues}",
            "",
            "Visible Clues:"
        ]
        
        for i, clue in enumerate(obs['clues']):
            output.append(f"  {i+1}. {clue}")
        
        output.append("")
        output.append("Answer Choices:")
        for i, choice in enumerate(obs['answer_choices']):
            marker = " ✓" if i == self.question.correct_answer_idx else ""
            output.append(f"  ({i+1}) {choice}{marker}")
        
        if self.done and self.selected_answer is not None:
            output.append("")
            is_correct = self.selected_answer == self.question.correct_answer_idx
            output.append(f"Selected: ({self.selected_answer + 1}) {obs['answer_choices'][self.selected_answer]}")
            output.append(f"Result: {'CORRECT ✓' if is_correct else 'INCORRECT ✗'}")
        
        return "\n".join(output)


class BatchedEnvironment:
    """Manages multiple environments in parallel for efficient training"""
    
    def __init__(self, questions: List[Question], reward_time_penalty: float = 0.1):
        """
        Initialize batched environments.
        
        Args:
            questions: List of Question objects
            reward_time_penalty: Penalty coefficient for late answering
        """
        self.envs = [QuizBowlEnvironment(q, reward_time_penalty) for q in questions]
        self.num_envs = len(self.envs)
    
    def reset(self) -> List[Dict]:
        """Reset all environments"""
        return [env.reset() for env in self.envs]
    
    def step(self, actions: List[int]) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[Dict]]:
        """
        Take actions in all environments.
        
        Args:
            actions: List of actions, one per environment
            
```

```bash
sed -n '1,220p' dataset.py

```

```output
"""
Dataset handling for Quiz Bowl questions
"""

import json
import csv
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict

from environment import Question
from config import Config


class QuizBowlDataset:
    """Dataset class for quiz bowl questions with multiple-choice answers"""
    
    def __init__(self, questions: List[Question]):
        """
        Initialize dataset with questions.
        
        Args:
            questions: List of Question objects
        """
        self.questions = questions
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Question:
        return self.questions[idx]
    
    def shuffle(self):
        """Shuffle questions in place"""
        random.shuffle(self.questions)
    
    def get_batch(self, batch_size: int) -> List[Question]:
        """Get a random batch of questions"""
        return random.sample(self.questions, min(batch_size, len(self.questions)))
    
    def save(self, filepath: str):
        """Save dataset to JSON file"""
        data = [self._question_to_dict(q) for q in self.questions]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'QuizBowlDataset':
        """Load dataset from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        questions = [cls._dict_to_question(d) for d in data]
        return cls(questions)
    
    @staticmethod
    def _question_to_dict(question: Question) -> Dict:
        """Convert Question to dictionary"""
        return {
            'question_id': question.question_id,
            'clues': question.clues,
            'answer_choices': question.answer_choices,
            'correct_answer_idx': question.correct_answer_idx,
            'category': question.category,
            'metadata': question.metadata or {}
        }
    
    @staticmethod
    def _dict_to_question(data: Dict) -> Question:
        """Convert dictionary to Question"""
        return Question(
            question_id=data['question_id'],
            clues=data['clues'],
            answer_choices=data['answer_choices'],
            correct_answer_idx=data['correct_answer_idx'],
            category=data['category'],
            metadata=data.get('metadata', {})
        )


class QANTADatasetLoader:
    """
    Load Quiz Bowl questions from QANTA CSV format.
    Generates multiple-choice questions by selecting distractors from the same category.
    """
    
    @classmethod
    def load_from_csv(cls, 
                     csv_path: str,
                     num_questions: int = None,
                     num_choices: int = 4,
                     min_clues: int = 3,
                     max_clues: int = 6,
                     seed: int = 42) -> 'QuizBowlDataset':
        """
        Load questions from QANTA CSV file.
        
        Args:
            csv_path: Path to questions.csv file
            num_questions: Number of questions to load (None = all)
            num_choices: Number of answer choices (default: 4)
            min_clues: Minimum clues to include per question
            max_clues: Maximum clues to include per question
            seed: Random seed
            
        Returns:
            QuizBowlDataset object
        """
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"Loading questions from {csv_path}...")
        
        # Load all questions from CSV
        raw_questions = []
        category_answers = defaultdict(list)  # For generating distractors
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse clues (separated by "|||")
                text = row['Text']
                clues = [clue.strip() for clue in text.split('|||')]
                
                # Store question data
                question_data = {
                    'question_id': row['Question ID'],
                    'fold': row['Fold'],
                    'answer': row['Answer'],
                    'category': row['Category'],
                    'clues': clues
                }
                raw_questions.append(question_data)
                
                # Store answer by category for distractor selection
                if row['Answer'] not in category_answers[row['Category']]:
                    category_answers[row['Category']].append(row['Answer'])
        
        print(f"Loaded {len(raw_questions)} raw questions")
        
        # Shuffle and optionally limit
        random.shuffle(raw_questions)
        if num_questions is not None:
            raw_questions = raw_questions[:num_questions]
        
        # Convert to Question objects with multiple choice
        questions = []
        for idx, raw_q in enumerate(raw_questions):
            # Get available clues
            available_clues = len(raw_q['clues'])
            if available_clues < 1:
                # Skip questions with no clues
                continue
            
            # Select number of clues to use
            if available_clues < min_clues:
                # Use all available clues if less than minimum
                num_clues = available_clues
            else:
                # Randomly select between min and max (capped by available)
                num_clues = min(
                    random.randint(min_clues, max_clues),
                    available_clues
                )
            
            clues = raw_q['clues'][:num_clues]
            
            # Generate distractors from same category
            correct_answer = raw_q['answer']
            category = raw_q['category']
            
            # Get potential distractors (exclude correct answer)
            potential_distractors = [
                ans for ans in category_answers[category] 
                if ans != correct_answer
            ]
            
            # If not enough distractors in category, use from other categories
            if len(potential_distractors) < num_choices - 1:
                other_answers = []
                for cat, answers in category_answers.items():
                    if cat != category:
                        other_answers.extend(answers)
                potential_distractors.extend(
                    random.sample(other_answers, 
                                min(num_choices - 1 - len(potential_distractors), 
                                    len(other_answers)))
                )
            
            # Sample distractors
            distractors = random.sample(
                potential_distractors, 
                min(num_choices - 1, len(potential_distractors))
            )
            
            # Create answer choices
            answer_choices = [correct_answer] + distractors
            correct_idx = 0
            
            # Shuffle choices
            shuffle_indices = list(range(len(answer_choices)))
            random.shuffle(shuffle_indices)
            answer_choices = [answer_choices[i] for i in shuffle_indices]
            correct_idx = shuffle_indices.index(0)
            
            # Pad with empty choices if needed
            while len(answer_choices) < num_choices:
                answer_choices.append(f"[No answer {len(answer_choices)}]")
            
            # Create Question object
            question = Question(
                question_id=raw_q['question_id'],
                clues=clues,
                answer_choices=answer_choices,
                correct_answer_idx=correct_idx,
                category=category,
                metadata={
                    'source': 'qanta',
```

```bash
sed -n '1,220p' model.py

```

```output
"""
T5-based policy model for Quiz Bowl RL agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import Config


class PolicyHead(nn.Module):
    """
    Custom policy head for the T5 model.
    Outputs: wait probability, answer distribution over choices, value estimate.
    """
    
    def __init__(self, hidden_size: int = 1024, num_choices: int = 4):
        """
        Initialize policy head.
        
        Args:
            hidden_size: Size of T5 hidden states
            num_choices: Number of answer choices
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_choices = num_choices
        
        # Wait/continue decision head (binary)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # [wait, answer_now]
        )
        
        # Answer selection head (over choices)
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_choices)
        )
        
        # Value head (state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, encoder_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy head.
        
        Args:
            encoder_hidden_state: [batch_size, hidden_size] - pooled encoder output
            
        Returns:
            wait_logits: [batch_size, 2] - logits for wait/answer
            answer_logits: [batch_size, num_choices] - logits for answer selection
            value: [batch_size, 1] - value estimate
        """
        wait_logits = self.wait_head(encoder_hidden_state)
        answer_logits = self.answer_head(encoder_hidden_state)
        value = self.value_head(encoder_hidden_state)
        
        return wait_logits, answer_logits, value


class T5PolicyModel(nn.Module):
    """
    T5-based policy model that combines T5 encoder with custom policy head.
    """
    
    def __init__(self, config: Config):
        """
        Initialize T5 policy model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Load T5 model and tokenizer
        print(f"Loading T5 model: {config.MODEL_NAME}")
        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
        self.tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
        
        # Get hidden size from T5 config
        hidden_size = self.t5_model.config.d_model
        
        # Custom policy head
        self.policy_head = PolicyHead(
            hidden_size=hidden_size,
            num_choices=config.NUM_ANSWER_CHOICES
        )
        
        # Move to device
        self.to(self.device)
        
        # Print model size
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model architecture and parameter count"""
        t5_params = sum(p.numel() for p in self.t5_model.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        total_params = t5_params + policy_params
        
        print(f"Model Architecture:")
        print(f"  T5 parameters: {t5_params:,}")
        print(f"  Policy head parameters: {policy_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {self.device}")
    
    def encode_input(self, 
                     text_inputs: List[str],
                     max_length: int = None) -> Dict[str, torch.Tensor]:
        """
        Encode text inputs using T5 tokenizer.
        
        Args:
            text_inputs: List of input strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.config.MAX_INPUT_LENGTH
        
        encoding = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def get_encoder_output(self, 
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get T5 encoder output and pool to fixed-size representation.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        # Get encoder outputs
        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # encoder_outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        hidden_states = encoder_outputs.last_hidden_state
        
        # Mean pooling over sequence dimension (masked)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / sum_mask
        
        return pooled_output
    
    def forward(self,
                text_inputs: List[str],
                return_value: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            text_inputs: List of text inputs (observations)
            return_value: Whether to return value estimates
            
        Returns:
            wait_logits: [batch_size, 2]
            answer_logits: [batch_size, num_choices]
            values: [batch_size, 1] or None
        """
        # Encode inputs
        encoding = self.encode_input(text_inputs)
        
        # Get encoder output
        pooled_output = self.get_encoder_output(
            encoding['input_ids'],
            encoding['attention_mask']
        )
        
        # Pass through policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)
        
        if not return_value:
            values = None
        
        return wait_logits, answer_logits, values
    
    def predict_answer(self,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict answer choice (for supervised training).
        
```

```bash
sed -n '1,220p' metrics.py

```

```output
"""
Evaluation metrics for question answering
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any
from sklearn.metrics import accuracy_score
from collections import defaultdict


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types to JSON-serializable Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


class MetricsTracker:
    """Track and compute various metrics for QA evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked values"""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.rewards = []
        self.buzz_positions = []
        self.is_correct = []
        self.categories = []
        
    def update(self, 
               pred: int,
               target: int,
               confidence: float,
               reward: float = None,
               buzz_position: int = None,
               category: str = None):
        """Update metrics with new sample"""
        self.predictions.append(pred)
        self.targets.append(target)
        self.confidences.append(confidence)
        
        is_correct = (pred == target)
        self.is_correct.append(is_correct)
        
        if reward is not None:
            self.rewards.append(reward)
        if buzz_position is not None:
            self.buzz_positions.append(buzz_position)
        if category is not None:
            self.categories.append(category)
    
    def compute_accuracy(self) -> float:
        """Compute overall accuracy"""
        if len(self.predictions) == 0:
            return 0.0
        return accuracy_score(self.targets, self.predictions)
    
    def compute_average_reward(self) -> float:
        """Compute average reward"""
        if len(self.rewards) == 0:
            return 0.0
        return np.mean(self.rewards)
    
    def compute_average_buzz_position(self) -> float:
        """Compute average buzz position"""
        if len(self.buzz_positions) == 0:
            return 0.0
        return np.mean(self.buzz_positions)
    
    def compute_ece(self, num_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            num_bins: Number of bins for calibration
            
        Returns:
            ECE score
        """
        if len(self.confidences) == 0:
            return 0.0
        
        confidences = np.array(self.confidences)
        is_correct = np.array(self.is_correct, dtype=float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Average confidence in bin
                avg_confidence = np.mean(confidences[in_bin])
                # Average accuracy in bin
                avg_accuracy = np.mean(is_correct[in_bin])
                # Bin weight
                bin_weight = np.sum(in_bin) / len(confidences)
                
                # Add to ECE
                ece += bin_weight * np.abs(avg_confidence - avg_accuracy)
        
        return ece
    
    def compute_brier_score(self) -> float:
        """
        Compute Brier score (mean squared error between confidence and correctness).
        
        Returns:
            Brier score
        """
        if len(self.confidences) == 0:
            return 0.0
        
        confidences = np.array(self.confidences)
        is_correct = np.array(self.is_correct, dtype=float)
        
        return np.mean((confidences - is_correct) ** 2)
    
    def compute_category_accuracy(self) -> Dict[str, float]:
        """Compute accuracy per category"""
        if len(self.categories) == 0:
            return {}
        
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred, target, cat in zip(self.predictions, self.targets, self.categories):
            category_stats[cat]['total'] += 1
            if pred == target:
                category_stats[cat]['correct'] += 1
        
        return {cat: stats['correct'] / stats['total'] 
                for cat, stats in category_stats.items()}
    
    def compute_reliability_diagram_data(self, num_bins: int = 10) -> Dict:
        """
        Compute data for reliability diagram.
        
        Returns:
            Dictionary with bin information
        """
        if len(self.confidences) == 0:
            return {}
        
        confidences = np.array(self.confidences)
        is_correct = np.array(self.is_correct, dtype=float)
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_data = {
            'bin_centers': [],
            'accuracies': [],
            'confidences': [],
            'counts': []
        }
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_center = (bin_lower + bin_upper) / 2
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(is_correct[in_bin])
                count = np.sum(in_bin)
                
                bin_data['bin_centers'].append(bin_center)
                bin_data['accuracies'].append(avg_accuracy)
                bin_data['confidences'].append(avg_confidence)
                bin_data['counts'].append(count)
        
        return bin_data
    
    def compute_buzz_position_stats(self) -> Dict:
        """Compute statistics about buzzing positions"""
        if len(self.buzz_positions) == 0:
            return {}
        
        positions = np.array(self.buzz_positions)
        
        # Accuracy by position
        position_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        for pos, correct in zip(self.buzz_positions, self.is_correct):
            position_accuracy[pos]['total'] += 1
            if correct:
                position_accuracy[pos]['correct'] += 1
        
        return {
            'mean': np.mean(positions),
            'std': np.std(positions),
            'min': np.min(positions),
```

```bash
rg -n "^class |^def " train_supervised.py train_ppo.py visualize.py demo.py | sort

```

```output
demo.py:14:class InteractiveDemo:
demo.py:182:def demo_with_sample_questions(model_path: str, device: str = 'cpu'):
demo.py:239:def main():
train_ppo.py:107:class PPOTrainer:
train_ppo.py:23:class RolloutStep:
train_ppo.py:37:class RolloutBuffer:
train_ppo.py:461:def run_ppo_training(config: Config,
train_supervised.py:21:class SupervisedTrainer:
train_supervised.py:277:def run_supervised_training(config: Config,
visualize.py:110:def plot_buzzing_behavior(metrics_data, save_dir):
visualize.py:13:def plot_training_curves(history, save_dir):
visualize.py:157:def plot_category_performance(metrics_data, save_dir):
visualize.py:195:def create_summary_report(checkpoint_dir, output_dir):
visualize.py:248:def main():
visualize.py:67:def plot_reliability_diagram(metrics_data, save_dir):
```

```bash
sed -n '1,180p' train_supervised.py

```

```output
"""
Supervised training for T5 policy model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from model import T5PolicyModel
from dataset import QuizBowlDataset
from environment import QuizBowlEnvironment
from metrics import MetricsTracker, evaluate_model, evaluate_choices_only
from config import Config


class SupervisedTrainer:
    """Trainer for supervised learning phase"""
    
    def __init__(self,
                 model: T5PolicyModel,
                 train_dataset: QuizBowlDataset,
                 val_dataset: QuizBowlDataset,
                 config: Config):
        """
        Initialize supervised trainer.
        
        Args:
            model: T5PolicyModel to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration object
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.SUPERVISED_LR,
            weight_decay=0.01
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR) / "supervised"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_batch(self, questions):
        """
        Prepare batch of questions for supervised training.
        Uses complete questions (all clues).
        
        Args:
            questions: List of Question objects
            
        Returns:
            input_ids, attention_mask, labels (all on device)
        """
        texts = []
        labels = []
        
        for question in questions:
            # Create environment to get text representation
            env = QuizBowlEnvironment(question)
            # Set to last clue position (show all clues)
            env.current_clue_idx = len(question.clues) - 1
            text = env.get_text_representation()
            
            texts.append(text)
            labels.append(question.correct_answer_idx)
        
        # Tokenize
        inputs = self.model.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.MAX_INPUT_LENGTH
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        return input_ids, attention_mask, labels
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Shuffle dataset
        self.train_dataset.shuffle()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Training loop with mini-batches
        num_batches = len(self.train_dataset) // self.config.SUPERVISED_BATCH_SIZE
        
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx in progress_bar:
            # Get batch
            batch_questions = self.train_dataset.get_batch(self.config.SUPERVISED_BATCH_SIZE)
            input_ids, attention_mask, labels = self.prepare_batch(batch_questions)
            
            # Forward pass
            answer_logits, predictions = self.model.predict_answer(input_ids, attention_mask)
            
            # Compute loss
            loss = self.criterion(answer_logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.SUPERVISED_GRAD_ACCUM_STEPS == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_correct / total_samples
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.4f}'
            })
        
        # Compute epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_acc = total_correct / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate on validation set"""
        print("Validating...")
        metrics = evaluate_model(
            self.model,
            self.val_dataset,
            device=self.device,
            deterministic=True
        )
        
        return metrics.get_summary()
    
    def train(self):
        """Run full supervised training"""
        print(f"Starting supervised training for {self.config.SUPERVISED_EPOCHS} epochs")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Device: {self.device}")
```

```bash
sed -n '1,180p' train_ppo.py

```

```output
"""
PPO (Proximal Policy Optimization) training for T5 policy model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

from model import T5PolicyModel
from dataset import QuizBowlDataset
from environment import QuizBowlEnvironment
from metrics import MetricsTracker, evaluate_model
from config import Config


@dataclass
class RolloutStep:
    """Single step in an episode rollout"""
    observation_text: str
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    
    # For tokenization
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None


class RolloutBuffer:
    """Buffer to store episode rollouts for PPO"""
    
    def __init__(self):
        self.rollouts = []
        self.reset()
    
    def reset(self):
        """Clear buffer"""
        self.rollouts = []
    
    def add_rollout(self, steps: List[RolloutStep]):
        """Add a complete episode rollout"""
        self.rollouts.append(steps)
    
    def get_all_steps(self) -> List[RolloutStep]:
        """Get all steps from all rollouts"""
        all_steps = []
        for rollout in self.rollouts:
            all_steps.extend(rollout)
        return all_steps
    
    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        """
        Compute discounted returns and GAE advantages for all rollouts.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        for rollout in self.rollouts:
            # Extract rewards and values
            rewards = [step.reward for step in rollout]
            values = [step.value for step in rollout]
            dones = [step.done for step in rollout]
            
            # Compute returns and advantages
            returns = []
            advantages = []
            
            # GAE computation
            gae = 0
            next_value = 0  # Terminal state has value 0
            
            for t in reversed(range(len(rollout))):
                if dones[t]:
                    next_value = 0
                    gae = 0
                
                # TD error
                delta = rewards[t] + gamma * next_value - values[t]
                
                # GAE
                gae = delta + gamma * gae_lambda * gae
                
                # Return = advantage + value
                returns.insert(0, gae + values[t])
                advantages.insert(0, gae)
                
                next_value = values[t]
            
            # Attach returns and advantages to steps
            for step, ret, adv in zip(rollout, returns, advantages):
                step.return_ = ret
                step.advantage = adv
    
    def __len__(self):
        return len(self.rollouts)


class PPOTrainer:
    """Trainer for PPO"""
    
    def __init__(self,
                 model: T5PolicyModel,
                 train_dataset: QuizBowlDataset,
                 val_dataset: QuizBowlDataset,
                 config: Config):
        """
        Initialize PPO trainer.
        
        Args:
            model: T5PolicyModel to train (should be pre-trained with supervised learning)
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration object
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.PPO_LR,
            weight_decay=0.01
        )
        
        # Training state
        self.current_iteration = 0
        self.best_val_reward = -float('inf')
        self.history = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR) / "ppo"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_rollouts(self, num_episodes: int) -> RolloutBuffer:
        """
        Collect rollouts by running episodes in the environment.
        
       Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            RolloutBuffer with collected rollouts
        """
        self.model.eval()
        buffer = RolloutBuffer()
        
        # Sample questions
        questions = self.train_dataset.get_batch(num_episodes)
        
        with torch.no_grad():
            for question in questions:
                env = QuizBowlEnvironment(
                    question,
                    reward_time_penalty=self.config.REWARD_TIME_PENALTY
                )
                
                obs = env.reset()
                done = False
                rollout = []
                
                while not done:
                    # Get text representation
                    text = env.get_text_representation(obs)
                    
                    # Tokenize
                    inputs = self.model.tokenizer(
```

```bash
sed -n '1,180p' visualize.py

```

```output
"""
Visualization utilities for analyzing training results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def plot_training_curves(history, save_dir):
    """Plot training curves from history"""
    
    # Extract data
    iterations = [h['iteration'] for h in history]
    train_rewards = [h['train_reward'] for h in history]
    val_accuracies = [h['val']['accuracy'] for h in history]
    val_rewards = [h['val'].get('average_reward', 0) for h in history]
    val_ece = [h['val'].get('ece', 0) for h in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training reward
    axes[0, 0].plot(iterations, train_rewards, 'b-', linewidth=2, label='Train Reward')
    axes[0, 0].plot(iterations, val_rewards, 'r-', linewidth=2, label='Val Reward')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Reward Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy
    axes[0, 1].plot(iterations, val_accuracies, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ECE (calibration)
    axes[1, 0].plot(iterations, val_ece, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Expected Calibration Error')
    axes[1, 0].set_title('Calibration (ECE)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Policy and value loss
    policy_losses = [h.get('policy_loss', 0) for h in history]
    value_losses = [h.get('value_loss', 0) for h in history]
    
    axes[1, 1].plot(iterations, policy_losses, 'orange', linewidth=2, label='Policy Loss')
    axes[1, 1].plot(iterations, value_losses, 'cyan', linewidth=2, label='Value Loss')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_dir / 'training_curves.png'}")
    plt.close()


def plot_reliability_diagram(metrics_data, save_dir):
    """Plot reliability diagram for calibration"""
    
    # Get reliability data
    bin_data = metrics_data.get('reliability_data', {})
    
    if not bin_data:
        print("No reliability data available")
        return
    
    bin_centers = bin_data['bin_centers']
    accuracies = bin_data['accuracies']
    confidences = bin_data['confidences']
    counts = bin_data['counts']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Plot actual calibration
    ax.scatter(confidences, accuracies, s=np.array(counts)*5, 
              alpha=0.6, c='blue', edgecolors='black', linewidth=1.5,
              label='Model Calibration')
    
    # Plot bars
    for conf, acc, count in zip(confidences, accuracies, counts):
        ax.plot([conf, conf], [conf, acc], 'r-', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
    print(f"Saved reliability diagram to {save_dir / 'reliability_diagram.png'}")
    plt.close()


def plot_buzzing_behavior(metrics_data, save_dir):
    """Plot buzzing position distribution"""
    
    buzz_stats = metrics_data.get('buzz_stats', {})
    
    if not buzz_stats:
        print("No buzzing statistics available")
        return
    
    position_accuracy = buzz_stats.get('position_accuracy', {})
    
    if not position_accuracy:
        return
    
    positions = sorted(position_accuracy.keys())
    accuracies = [position_accuracy[p] for p in positions]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy by position
    axes[0].bar(positions, accuracies, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Buzz Position (Clue Number)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Buzz Position')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Distribution of buzz positions
    mean_pos = buzz_stats.get('mean', 0)
    std_pos = buzz_stats.get('std', 0)
    
    axes[1].axvline(mean_pos, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_pos:.2f}')
    axes[1].axvline(mean_pos - std_pos, color='orange', linestyle=':', linewidth=1.5,
                   label=f'±1 Std: {std_pos:.2f}')
    axes[1].axvline(mean_pos + std_pos, color='orange', linestyle=':', linewidth=1.5)
    axes[1].set_xlabel('Buzz Position')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Buzzing Position Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'buzzing_behavior.png', dpi=300, bbox_inches='tight')
    print(f"Saved buzzing behavior to {save_dir / 'buzzing_behavior.png'}")
    plt.close()


def plot_category_performance(metrics_data, save_dir):
    """Plot per-category performance"""
    
    category_acc = metrics_data.get('category_accuracy', {})
    
    if not category_acc:
        print("No category-specific data available")
        return
    
    categories = list(category_acc.keys())
    accuracies = [category_acc[c] for c in categories]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    categories = [categories[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
    bars = ax.barh(categories, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
```

```bash
sed -n '1,180p' demo.py

```

```output
"""
Interactive demo for testing the trained model
"""

import torch
import argparse
from pathlib import Path

from model import T5PolicyModel
from environment import QuizBowlEnvironment, Question
from config import Config


class InteractiveDemo:
    """Interactive demo for question answering"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize demo with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on
        """
        print(f"Loading model from {model_path}...")
        self.model = T5PolicyModel.load_pretrained(model_path, device=device)
        self.model.to(device)
        self.model.eval()
        self.device = device
        print("Model loaded successfully!")
    
    def run_episode(self, question: Question, verbose: bool = True):
        """
        Run a single episode with the given question.
        
        Args:
            question: Question object
            verbose: Whether to print step-by-step details
            
        Returns:
            Dictionary with episode results
        """
        env = QuizBowlEnvironment(question)
        obs = env.reset()
        done = False
        step_count = 0
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"Question ID: {question.question_id}")
            print(f"Category: {question.category}")
            print(f"Total Clues: {len(question.clues)}")
            print("=" * 70)
        
        with torch.no_grad():
            while not done:
                step_count += 1
                
                # Get current observation
                text = env.get_text_representation(obs)
                
                if verbose:
                    print(f"\n--- Step {step_count} (Clue {obs['clue_position'] + 1}/{obs['total_clues']}) ---")
                    print(f"Current clue: {obs['clues'][-1]}")
                
                # Tokenize
                inputs = self.model.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get model prediction
                outputs = self.model.forward(inputs['input_ids'], inputs['attention_mask'])
                
                # Get action probabilities
                action_probs = outputs['action_probs'][0].cpu().numpy()
                wait_prob = outputs['wait_prob'][0].item()
                answer_logits = outputs['answer_logits'][0].cpu()
                answer_probs = torch.softmax(answer_logits, dim=-1).numpy()
                
                if verbose:
                    print(f"\nModel predictions:")
                    print(f"  Wait probability: {wait_prob:.3f}")
                    print(f"  Answer probabilities:")
                    for i, (choice, prob) in enumerate(zip(obs['answer_choices'], answer_probs)):
                        marker = "✓" if i == question.correct_answer_idx else " "
                        print(f"    ({i+1}) {choice}: {prob:.3f} {marker}")
                
                # Select action (deterministic - argmax)
                action = action_probs.argmax()
                
                if action == 0:
                    if verbose:
                        print(f"\nAction: WAIT (continue to next clue)")
                else:
                    selected_idx = action - 1
                    if verbose:
                        print(f"\nAction: SELECT answer ({selected_idx + 1}) {obs['answer_choices'][selected_idx]}")
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                obs = next_obs
        
        # Episode complete
        if verbose:
            print("\n" + "=" * 70)
            print("EPISODE COMPLETE")
            print("=" * 70)
            
            if 'is_correct' in info:
                result = "CORRECT ✓" if info['is_correct'] else "INCORRECT ✗"
                print(f"Result: {result}")
                print(f"Selected: ({info['answer_idx'] + 1}) {question.answer_choices[info['answer_idx']]}")
                print(f"Correct: ({info['correct_idx'] + 1}) {question.answer_choices[info['correct_idx']]}")
                print(f"Buzz Position: {info['clue_position'] + 1}/{len(question.clues)}")
                print(f"Reward: {reward:.3f}")
            print("=" * 70)
        
        return {
            'is_correct': info.get('is_correct', False),
            'reward': reward,
            'buzz_position': info.get('clue_position', 0),
            'selected_answer': info.get('answer_idx', -1),
            'correct_answer': info.get('correct_idx', -1)
        }
    
    def interactive_mode(self):
        """Run interactive mode where user can input questions"""
        print("\n" + "=" * 70)
        print("INTERACTIVE QUESTION ANSWERING DEMO")
        print("=" * 70)
        print("\nEnter 'quit' to exit")
        
        while True:
            print("\n" + "-" * 70)
            
            # Get question from user
            question_id = input("\nQuestion ID (or 'quit'): ").strip()
            if question_id.lower() == 'quit':
                break
            
            category = input("Category: ").strip()
            
            # Get clues
            clues = []
            print("\nEnter clues (press Enter twice when done):")
            while True:
                clue = input(f"Clue {len(clues) + 1}: ").strip()
                if not clue:
                    break
                clues.append(clue)
            
            if not clues:
                print("No clues entered. Skipping question.")
                continue
            
            # Get answer choices
            choices = []
            print("\nEnter 4 answer choices:")
            for i in range(4):
                choice = input(f"Choice {i+1}: ").strip()
                choices.append(choice)
            
            correct_idx = int(input("\nCorrect answer index (1-4): ")) - 1
            
            # Create question
            question = Question(
                question_id=question_id,
                clues=clues,
                answer_choices=choices,
                correct_answer_idx=correct_idx,
                category=category
            )
            
            # Run episode
            self.run_episode(question, verbose=True)

```

## 10. Tests, packaging, and bridge surfaces

The remaining files explain how the repo is meant to be installed, tested, and used as a self-contained replacement for older qb-rl notebooks and scripts.

Two details matter here:

- the packaging metadata now treats `qanta-buzzer` as the canonical installable project
- the compatibility shims keep old import paths working from inside this repo instead of depending on a sibling checkout

```bash
rg --files tests | sort

```

```output
tests/__init__.py
tests/__pycache__/__init__.cpython-311.pyc
tests/__pycache__/__init__.cpython-312.pyc
tests/__pycache__/conftest.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/conftest.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/conftest.cpython-312.pyc
tests/__pycache__/test_agents.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_agents.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_environment.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_environment.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_factories.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_factories.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_features.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_features.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_likelihoods.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_likelihoods.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_metrics.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_metrics.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_ppo_buzzer.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_ppo_buzzer.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_ppo_t5.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_ppo_t5.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_qb_rl_bridge.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_supervised_t5.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_supervised_t5.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_t5_policy.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_t5_policy.cpython-312-pytest-9.0.2.pyc
tests/__pycache__/test_text_wrapper.cpython-311-pytest-9.0.2.pyc
tests/__pycache__/test_text_wrapper.cpython-312-pytest-9.0.2.pyc
tests/conftest.py
tests/test_agents.py
tests/test_environment.py
tests/test_factories.py
tests/test_features.py
tests/test_likelihoods.py
tests/test_metrics.py
tests/test_ppo_buzzer.py
tests/test_ppo_t5.py
tests/test_qb_rl_bridge.py
tests/test_supervised_t5.py
tests/test_t5_policy.py
tests/test_text_wrapper.py
```

```bash
venv/bin/python - <<'PY'
from pathlib import Path
files = sorted(Path('tests').glob('test_*.py'))
print('test_files =', len(files))
count = 0
for path in files:
    text = path.read_text()
    tests = text.count('\ndef test_') + (1 if text.startswith('def test_') else 0)
    print(path.name, 'test_functions=', tests)
    count += tests
print('approx_test_functions =', count)
PY

```

```output
test_files = 12
test_agents.py test_functions= 0
test_environment.py test_functions= 0
test_factories.py test_functions= 0
test_features.py test_functions= 0
test_likelihoods.py test_functions= 0
test_metrics.py test_functions= 17
test_ppo_buzzer.py test_functions= 0
test_ppo_t5.py test_functions= 0
test_qb_rl_bridge.py test_functions= 0
test_supervised_t5.py test_functions= 0
test_t5_policy.py test_functions= 0
test_text_wrapper.py test_functions= 0
approx_test_functions = 17
```

```bash
sed -n '1,220p' tests/test_qb_rl_bridge.py

```

```output
"""Compatibility bridge tests for qb-rl surfaces ported into qanta-buzzer."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import agents.bayesian_buzzer as bayesian_buzzer
import models.answer_profiles as compat_answer_profiles
import models.likelihoods as likelihoods
import qb_data.answer_profiles as qb_answer_profiles
import qb_data.data_loader as qb_data_loader
import qb_env.data_loader as compat_data_loader
import qb_env.mc_builder as compat_mc_builder
import qb_env.text_utils as compat_text_utils
from agents.softmax_profile_buzzer import (
    SequentialBayesBuzzer as CompatSequentialBayesBuzzer,
)
from agents.softmax_profile_buzzer import (
    SoftmaxEpisodeResult as CompatSoftmaxEpisodeResult,
)
from agents.softmax_profile_buzzer import (
    SoftmaxProfileBuzzer as CompatSoftmaxProfileBuzzer,
)
from models.likelihoods import OpenAILikelihood, build_likelihood_from_config
from qb_data.mc_builder import MCBuilder


def _install_fake_openai(monkeypatch, vectors: dict[str, list[float]], calls: list[tuple[str, tuple[str, ...]]]) -> None:
    """Install a fake ``openai`` module that serves deterministic embeddings."""

    class FakeEmbeddingsClient:
        def create(self, model: str, input: list[str]):
            calls.append((model, tuple(input)))
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=vectors[text])
                    for text in input
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.embeddings = FakeEmbeddingsClient()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))


class TestOpenAILikelihood:
    """Tests for optional OpenAI embedding support."""

    def test_openai_likelihood_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAILikelihood()

    def test_openai_likelihood_scores_and_reuses_cache(self, monkeypatch) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []
        vectors = {
            "first president": [2.0, 0.0],
            "george washington": [3.0, 0.0],
            "albert einstein": [0.0, 4.0],
        }
        _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        model = OpenAILikelihood(model="fake-embedding-model")

        embeddings = model._embed_batch(["first president", "george washington"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)
        calls_before_score = len(calls)

        scores_1 = model.score(
            "first president",
            ["george washington", "albert einstein"],
        )
        assert scores_1[0] > scores_1[1]
        assert len(calls) == calls_before_score + 2, (
            "first score should call the embeddings API twice"
        )

        scores_2 = model.score(
            "first president",
            ["george washington", "albert einstein"],
        )
        np.testing.assert_allclose(scores_1, scores_2, atol=1e-6)
        assert len(calls) == calls_before_score + 2, "second score should be served from cache"

    def test_likelihood_factory_openai(self, monkeypatch) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []
        vectors = {"a": [1.0, 0.0]}
        _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        config = {"likelihood": {"model": "openai", "openai_model": "fake-openai"}}
        model = build_likelihood_from_config(config)

        assert isinstance(model, OpenAILikelihood)
        assert model.model == "fake-openai"


class TestOpenAIProfileStrategy:
    """Tests for OpenAI-backed distractor ranking."""

    def test_openai_profile_uses_openai_embeddings(self, monkeypatch) -> None:
        calls: list[str] = []
        embeddings = {
            "gold profile": np.array([1.0, 0.0], dtype=np.float32),
            "near distractor": np.array([0.9, 0.1], dtype=np.float32),
            "far distractor": np.array([0.0, 1.0], dtype=np.float32),
        }

        class FakeOpenAILikelihood:
            def __init__(self, model: str = "unused") -> None:
                calls.append(model)

            def embed_and_cache(self, texts: list[str]) -> np.ndarray:
                return np.stack([embeddings[text] for text in texts]).astype(np.float32)

        monkeypatch.setattr(likelihoods, "OpenAILikelihood", FakeOpenAILikelihood)

        builder = MCBuilder(strategy="openai_profile", openai_model="fake-openai")
        rankings = builder._compute_rankings(
            answers=["gold", "near", "far"],
            answer_profiles={
                "gold": "gold profile",
                "near": "near distractor",
                "far": "far distractor",
            },
            answer_to_category={},
        )

        assert calls == ["fake-openai"]
        assert rankings["gold"][0] == "near"
        assert rankings["gold"][1] == "far"


class TestQBRLCompatibilityModules:
    """Tests for qb-rl import-path shims."""

    def test_module_aliases_resolve_expected_symbols(self) -> None:
        assert compat_answer_profiles.AnswerProfileBuilder is qb_answer_profiles.AnswerProfileBuilder
        assert compat_data_loader.parse_row is qb_data_loader.parse_row
        assert compat_mc_builder.MCBuilder.__name__ == "MCBuilder"
        assert compat_text_utils.normalize_answer("The Answer") == "answer"
        assert CompatSoftmaxProfileBuzzer is bayesian_buzzer.SoftmaxProfileBuzzer
        assert CompatSequentialBayesBuzzer is bayesian_buzzer.SequentialBayesBuzzer
        assert CompatSoftmaxEpisodeResult is bayesian_buzzer.SoftmaxEpisodeResult

    def test_parse_row_supports_qb_rl_metadata(self) -> None:
        question = compat_data_loader.parse_row(
            {
                "qid": "q-1",
                "question": "alpha beta gamma",
                "answer_primary": "George Washington",
                "clean_answers": ["George Washington", "Washington"],
                "run_indices": [1, 2],
                "metadata": {
                    "category": "History",
                    "human_buzz_positions": [{"position": 4, "count": 2}],
                },
            }
        )

        assert question.qid == "q-1"
        assert question.category == "History"
        assert question.human_buzz_positions == [(4, 2)]
        assert question.cumulative_prefixes == ["alpha beta", "alpha beta gamma"]

    def test_load_tossup_questions_from_config_prefers_dataset_smoke(
        self, monkeypatch
    ) -> None:
        captured: dict[str, object] = {}
        sample_question = compat_data_loader.TossupQuestion(
            qid="hf-1",
            question="alpha beta",
            tokens=["alpha", "beta"],
            answer_primary="Answer",
            clean_answers=["Answer"],
            run_indices=[1],
            human_buzz_positions=None,
            category="History",
            cumulative_prefixes=["alpha beta"],
        )

        def fake_load_tossup_questions(
            dataset: str,
            dataset_config: str | None = None,
            split: str = "eval",
            limit: int | None = None,
        ):
            captured["dataset"] = dataset
            captured["dataset_config"] = dataset_config
            captured["split"] = split
            captured["limit"] = limit
            return [sample_question]

        monkeypatch.setattr(qb_data_loader, "load_tossup_questions", fake_load_tossup_questions)

        config = {
            "data": {
                "dataset": "main-dataset",
                "dataset_config": "main-config",
                "dataset_smoke": "smoke-dataset",
                "dataset_smoke_config": "smoke-config",
                "split": "train",
            }
        }

        questions = compat_data_loader.load_tossup_questions_from_config(config, smoke=True)

        assert len(questions) == 1
        assert captured == {
            "dataset": "smoke-dataset",
            "dataset_config": "smoke-config",
```

```bash
sed -n '1,220p' pyproject.toml

```

```output
[build-system]
requires = ["setuptools>=69.0"]
build-backend = "setuptools.build_meta"

[project]
name = "qanta-buzzer"
version = "1.0.0"
description = "Unified quiz bowl RL buzzer system for Stanford CS234"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "datasets>=2.14.0",
  "gymnasium>=1.1.0",
  "jsonlines>=3.1.0",
  "matplotlib>=3.7.0",
  "numpy>=1.24.0",
  "pandas>=2.0.0",
  "PyYAML>=6.0.0",
  "scikit-learn>=1.3.0",
  "seaborn>=0.12.0",
  "sentence-transformers>=2.2.0",
  "stable-baselines3>=2.6.0",
  "torch>=2.0.0",
  "tqdm>=4.65.0",
  "transformers>=4.30.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0.0"]

[tool.setuptools.packages.find]
include = ["agents", "evaluation", "models", "qb_data", "qb_env", "training"]
```

```bash
sed -n '1,220p' requirements.txt

```

```output
# Base runtime dependencies for the unified modular repo.
# Preferred development setup is: pip install -e .

torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
datasets>=2.14.0
gymnasium>=1.1.0
stable-baselines3>=2.6.0
PyYAML>=6.0.0
tqdm>=4.65.0
jsonlines>=3.1.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0

# Optional OpenAI bridge support:
# pip install -e .[openai]
```

```bash
rg -n "qb_env.data_loader|qb_env.mc_builder|qb_env.text_utils|models.answer_profiles|agents.softmax_profile_buzzer|OpenAI embedding|compatibility" README.md CLAUDE.md qb_env/__init__.py models/__init__.py agents/softmax_profile_buzzer.py qb_env/data_loader.py qb_env/mc_builder.py qb_env/text_utils.py models/answer_profiles.py | sort

```

```output
CLAUDE.md:12:`qanta-buzzer` is the canonical repo. qb-rl compatibility is preserved through additive shims rather than structural rewrites.
CLAUDE.md:75:- `qb_env/`: Gymnasium environment plus text wrapper and qb-rl compatibility shims
CLAUDE.md:76:- `models/`: likelihood models, belief features, T5 policy model, compatibility exports
CLAUDE.md:85:- Old qb-rl imports like `qb_env.data_loader` and `models.answer_profiles` are thin re-exports over the canonical modules.
CLAUDE.md:9:1. Belief-feature pipeline: build MC tossups, score answer profiles with TF-IDF / SBERT / T5 / optional OpenAI embeddings, train or compare buzzers, and evaluate with S_q plus calibration metrics.
README.md:10:- Optional OpenAI embedding support for `likelihood.model: openai` and `data.distractor_strategy: openai_profile`
README.md:5:This repo keeps `qanta-buzzer` as the canonical implementation while preserving a qb-rl compatibility bridge:
README.md:69:- `qb_env.data_loader`
README.md:70:- `qb_env.mc_builder`
README.md:71:- `qb_env.text_utils`
README.md:72:- `models.answer_profiles`
README.md:73:- `agents.softmax_profile_buzzer`
README.md:85:- qb-rl compatibility bridge and mocked OpenAI coverage
agents/softmax_profile_buzzer.py:1:"""qb-rl compatibility re-exports for Bayesian-family buzzers."""
models/answer_profiles.py:1:"""qb-rl compatibility re-export for answer profile building."""
qb_env/__init__.py:14:from qb_env.mc_builder import MCBuilder, MCQuestion
qb_env/__init__.py:15:from qb_env.text_utils import normalize_answer, tokenize_text
qb_env/__init__.py:4:plus thin qb-rl compatibility exports for the old `qb_env.*` import paths.
qb_env/__init__.py:7:from qb_env.data_loader import (
qb_env/data_loader.py:1:"""qb-rl compatibility re-exports for tossup data loading."""
qb_env/mc_builder.py:1:"""qb-rl compatibility re-exports for MC question building."""
qb_env/text_utils.py:1:"""qb-rl compatibility re-exports for text utilities."""
```

## 11. Closing mental model

At this point the full repo can be summarized as one loop.

1. Raw tossups are loaded and normalized.
2. Answer profiles and guarded distractors turn them into multiple-choice tossups.
3. A likelihood model scores each option as clues unfold.
4. The environment converts those scores into belief state or text observations.
5. Agents decide when to wait and when to buzz.
6. Evaluation compares accuracy, timing, calibration, and artifact sensitivity.

If you want to change the repo, the easiest starting points are:

- change data construction in `qb_data/` if you care about answer-space quality or distractor artifacts
- change `models/likelihoods.py` or `models/t5_policy.py` if you care about semantic representations
- change `qb_env/` if you care about reward shaping or observation design
- change `agents/` or `training/` if you care about decision policies
- change `evaluation/` if you care about what counts as a good buzzer

That division is the main payoff of the refactor: the canonical modular path makes each research question live in a smaller, easier-to-understand part of the codebase.

## 12. Smoke test runs in practice

This appendix records a real end-to-end smoke pass I ran on March 6, 2026 from the repo root.

There is one important gotcha before running it: `scripts/build_mc_dataset.py` defaults `--config` to `configs/default.yaml`, so bare `--smoke` is not enough to get the 50-question smoke build. In practice, use `--config configs/smoke.yaml --smoke --output-dir artifacts/smoke` for the build step so the later smoke scripts read the small dataset from `artifacts/smoke/mc_dataset.json`.

The intended four-stage smoke workflow I ran was:

```bash
venv/bin/python -u scripts/build_mc_dataset.py --config configs/smoke.yaml --smoke --output-dir artifacts/smoke
venv/bin/python -u scripts/run_baselines.py --config configs/smoke.yaml --smoke
venv/bin/python -u scripts/train_ppo.py --config configs/smoke.yaml --smoke
venv/bin/python -u scripts/evaluate_all.py --config configs/smoke.yaml --smoke
```

Observed wall-clock times on this machine:

- dataset build: `real 2.07s`
- baseline sweep: `real 2.77s`
- PPO smoke training: `real 4.26s`
- evaluation: `real 2.37s`

Those timings are for the intended smoke path. If you accidentally let the build step fall back to `configs/default.yaml`, the later smoke scripts can end up reading a much larger dataset and become dramatically slower.

```bash
sed -n '1,120p' configs/smoke.yaml; printf '\n'
```

```output
# Smoke test configuration - quick testing with reduced data
# Inherits from default.yaml and overrides key settings

# Data settings for quick testing
data:
  csv_path: "questions.csv"
  K: 4
  distractor_strategy: "category_random"  # Faster than sbert_profile
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  max_questions: 50  # Use only 50 questions for smoke test
  shuffle_seed: 42

answer_profiles:
  max_tokens_per_profile: 500  # Reduced for speed
  min_questions_per_answer: 1
  leave_one_out: false  # Skip for smoke test

likelihood:
  model: "tfidf"  # Use TF-IDF for fastest smoke testing (<5 seconds)
  embedding_model: "all-MiniLM-L6-v2"
  beta: 5.0  # Softmax temperature for belief distribution
  cache_embeddings: true
  cache_dir: "cache/embeddings"
  batch_size: 4  # Smaller batch for memory
  max_length: 256  # Shorter sequences

environment:
  reward_mode: "time_penalty"
  wait_penalty: 0.1
  buzz_correct: 1.0
  buzz_incorrect: -0.5
  max_steps: 10  # Fewer steps for quick testing

mc_guards:
  alias_edit_distance_threshold: 0.2
  duplicate_token_overlap_threshold: 0.8
  max_length_ratio: 3.0

bayesian:  # Reduced sweep for smoke testing
  threshold_sweep: [0.5, 0.7, 0.9]
  alpha: 10.0

ppo:  # Reduced for smoke testing
  total_timesteps: 1000  # Much fewer steps
  learning_rate: 3e-4
  n_steps: 32  # Smaller rollout
  batch_size: 8  # Smaller batch
  n_epochs: 2  # Fewer epochs
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  target_kl: 0.03
  policy_kwargs:
    net_arch: [32, 32]  # Smaller network

evaluation:
  metrics:
    - accuracy
    - reward
  compute_sq: false  # Skip expensive metrics
  run_choices_only: false  # Skip control experiments
  run_shuffle: false
  bootstrap_ci_samples: 0  # No bootstrap for smoke test
  save_predictions: false
  prediction_dir: "results/predictions"

# Supervised settings for smoke test
supervised:
  epochs: 2  # Very few epochs
  batch_size: 4
  gradient_accumulation_steps: 1  # No accumulation for speed
  learning_rate: 1e-4
  warmup_steps: 10
  eval_steps: 20
  save_steps: 100
  save_total_limit: 1
  checkpoint_dir: "checkpoints/supervised_smoke"
```

### Smoke build

The corrected build command finishes quickly and leaves a small, inspectable dataset under `artifacts/smoke/`.

Key console outcomes from the run:

- loaded `20407` raw questions from `questions.csv`
- applied the smoke cap and kept `50` questions
- built `42` answer profiles
- generated `44` MC questions after guard filtering
- wrote `mc_dataset.json`, train/val/test splits, and `answer_profiles.json`

```bash
find artifacts/smoke -maxdepth 1 -type f | sort
```

```output
artifacts/smoke/answer_profiles.json
artifacts/smoke/baseline_floor_runs.json
artifacts/smoke/baseline_sequential_bayes_runs.json
artifacts/smoke/baseline_softmax_profile_runs.json
artifacts/smoke/baseline_summary.json
artifacts/smoke/baseline_threshold_runs.json
artifacts/smoke/evaluation_report.json
artifacts/smoke/mc_dataset.json
artifacts/smoke/ppo_model.zip
artifacts/smoke/ppo_runs.json
artifacts/smoke/ppo_summary.json
artifacts/smoke/test_dataset.json
artifacts/smoke/train_dataset.json
artifacts/smoke/val_dataset.json
```

Recorded from the run:

- `{"mc_questions": 44, "train": 28, "val": 3, "test": 13}`
- `{"qid": "1704", "answer": "Seneca Falls Convention", "category": "History", "options": ["Seneca Falls Convention", "Charlemagne", "Inca Empire", "Netherlands"]}`

### Baseline sweep

With the smoke dataset in place, the non-RL baseline stage is also cheap. It loaded `44` MC questions, built a TF-IDF likelihood model, and evaluated four baseline families over the reduced threshold sweep `[0.5, 0.7, 0.9]`.

The notable result in this run is that all of the baseline families landed on the same smoke accuracy (`0.386`), but the stopping behavior differed. `AlwaysBuzzFinal` waited the longest and therefore had the best `mean_sq`, while `SequentialBayes` buzzed a bit earlier than the from-scratch softmax baselines at the same threshold.

```bash
venv/bin/python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('artifacts/smoke/baseline_summary.json').read_text())
rows = [
    ('threshold@0.5', summary['threshold']['0.5']),
    ('threshold@0.9', summary['threshold']['0.9']),
    ('sequential_bayes@0.5', summary['sequential_bayes']['0.5']),
    ('always_final', summary['always_final']),
]
for name, metrics in rows:
    print(name)
    print({
        'buzz_accuracy': round(metrics['buzz_accuracy'], 3),
        'mean_buzz_step': round(metrics['mean_buzz_step'], 3),
        'mean_sq': round(metrics['mean_sq'], 3),
        'mean_reward_like': round(metrics['mean_reward_like'], 3),
    })
PY
```

```output
threshold@0.5
{'buzz_accuracy': 0.386, 'mean_buzz_step': 3.5, 'mean_sq': 0.243, 'mean_reward_like': 0.08}
threshold@0.9
{'buzz_accuracy': 0.386, 'mean_buzz_step': 4.045, 'mean_sq': 0.053, 'mean_reward_like': 0.08}
sequential_bayes@0.5
{'buzz_accuracy': 0.386, 'mean_buzz_step': 3.045, 'mean_sq': 0.267, 'mean_reward_like': 0.0}
always_final
{'buzz_accuracy': 0.386, 'mean_buzz_step': 4.045, 'mean_sq': 0.386, 'mean_reward_like': 0.08}
```

### PPO smoke training

The PPO smoke run trains a tiny belief-state policy for only `1000` timesteps. The console trace shows the Stable Baselines3 rollout summaries updating quickly, and the stage leaves both `ppo_model.zip` and structured run summaries in `artifacts/smoke/`.

This particular smoke policy is clearly not meant to be a good buzzer. It learned a very early-buzzing behavior: `mean_buzz_step` ended up essentially at the first clue, which is why its reward-like metric is still negative even though `mean_sq` is not terrible.

Recorded summary from the run:

- `{"n": 44.0, "buzz_accuracy": 0.25, "mean_buzz_step": 0.0, "mean_sq": 0.271, "mean_reward_like": -0.136, "ece": 0.056, "brier": 0.19, "n_calibration": 44.0}`

### Final evaluation

The evaluation smoke run reads the just-written baseline and PPO artifacts, chooses the best softmax threshold (`0.5` in this run), computes the main report, then executes the three control experiments that are still enabled here: shuffle, alias substitution, and choices-only.

Two details are worth noticing in the output:

- `alias_lookup.json` was absent in `artifacts/smoke/`, so the alias control fell back to an empty lookup.
- the choices-only control dropped to `0.091` accuracy against `0.25` chance, which is the kind of sanity check you want the walkthrough reader to notice when validating that clues matter.

```bash
venv/bin/python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('artifacts/smoke/evaluation_report.json').read_text())
full = report['full_eval']
print({'softmax_profile_best_threshold': report['softmax_profile_best_threshold']})
print({
    'buzz_accuracy': round(full['buzz_accuracy'], 3),
    'mean_buzz_step': round(full['mean_buzz_step'], 3),
    'mean_sq': round(full['mean_sq'], 3),
    'mean_reward_like': round(full['mean_reward_like'], 3),
})
print(report['controls'])
PY
```

```output
{'softmax_profile_best_threshold': 0.5}
{'buzz_accuracy': 0.386, 'mean_buzz_step': 3.5, 'mean_sq': 0.243, 'mean_reward_like': 0.0}
{'choices_only': {'accuracy': 0.09090909090909091, 'chance': 0.25, 'n_test': 11.0}, 'shuffle': {'n': 44.0, 'buzz_accuracy': 0.38636363636363635, 'mean_buzz_step': 3.5, 'mean_sq': 0.23666016887085728, 'mean_reward_like': 0.0, 'ece': 0.0, 'brier': 0.0, 'n_calibration': 44.0}, 'alias_substitution': {'n': 44.0, 'buzz_accuracy': 0.38636363636363635, 'mean_buzz_step': 3.5, 'mean_sq': 0.24329479467724396, 'mean_reward_like': 0.0, 'ece': 0.0, 'brier': 0.0, 'n_calibration': 44.0}}
```

```bash
venv/bin/python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('artifacts/smoke/evaluation_report.json').read_text())
per = report['per_category']
for key in ['Science:Chemistry', 'Science:Physics', 'Fine_Arts', 'Literature', 'Social_Science']:
    if key in per:
        metrics = per[key]
        print(key, {
            'n': int(metrics['n']),
            'buzz_accuracy': round(metrics['buzz_accuracy'], 3),
            'mean_sq': round(metrics['mean_sq'], 3),
        })
PY
```

```output
Science:Chemistry {'n': 6, 'buzz_accuracy': 1.0, 'mean_sq': 0.683}
Science:Physics {'n': 4, 'buzz_accuracy': 1.0, 'mean_sq': 0.532}
Fine_Arts {'n': 7, 'buzz_accuracy': 0.143, 'mean_sq': 0.159}
Literature {'n': 6, 'buzz_accuracy': 0.0, 'mean_sq': 0.0}
Social_Science {'n': 9, 'buzz_accuracy': 0.222, 'mean_sq': 0.139}
```
