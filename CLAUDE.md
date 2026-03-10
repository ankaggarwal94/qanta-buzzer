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

<<<<<<< HEAD
=======
Bare `python scripts/build_mc_dataset.py --smoke` is now a valid contract: it selects the smoke config path and writes datasets to `artifacts/smoke/` unless `--config` or `--output-dir` are supplied explicitly.

>>>>>>> cda02951d4f40d4e7f14fbb2626d3740699830af
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
