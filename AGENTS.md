# AGENTS.md

Canonical repo contract for all coding agents (Claude, Copilot, Cursor, etc.).

## Project Overview

Stanford CS234 final project: a unified quiz bowl RL buzzer system with two tracks. The belief-feature pipeline builds MC tossups, scores answer profiles with TF-IDF / SBERT / T5 / optional OpenAI embeddings, trains or compares buzzers, and evaluates with S_q plus calibration metrics. The T5 policy pipeline provides supervised warm-start and PPO for an end-to-end text policy. `qanta-buzzer` is the canonical repo. qb-rl compatibility is preserved through additive shims rather than structural rewrites.

## Setup

Requires Python >= 3.11.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
```

Optional OpenAI support:

```bash
pip install -e '.[openai]'
export OPENAI_API_KEY=...
```

## Architecture

| Package | Purpose |
|---------|---------|
| `qb_data/` | Canonical data loading, answer profiles, stratified splits, MC construction |
| `qb_env/` | Gymnasium environment, text wrapper, qb-rl compatibility shims |
| `models/` | Likelihood models (TF-IDF, SBERT, T5, OpenAI), belief features, T5 policy model |
| `agents/` | Threshold, softmax-profile, sequential Bayes, PPO wrapper |
| `evaluation/` | S_q metric, calibration, control experiments, plotting |
| `scripts/` | Pipeline entrypoints and shared helpers |
| `training/` | T5 policy supervised + PPO trainers |
| `configs/` | YAML configuration files (default, smoke, t5_policy) |

## Testing

220 tests across 13 test files.

```bash
pytest                    # full suite
pytest tests/test_qb_rl_bridge.py tests/test_factories.py tests/test_ppo_buzzer.py  # focused bridge/runtime checks
scripts/ci.sh             # CI entry point (runs pytest, exits nonzero on failure)
```

## Smoke Pipeline

Four-stage belief-feature smoke workflow. `--smoke` selects `configs/smoke.yaml` and writes outputs to `artifacts/smoke/`.

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

Or run all four stages via the wrapper script:

```bash
scripts/manual-smoke.sh
```

## T5 Policy Pipeline

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/compare_policies.py --config configs/t5_policy.yaml
```

## Configuration

| Config | Purpose |
|--------|---------|
| `configs/default.yaml` | Full runs with T5-large likelihood and 100k PPO timesteps |
| `configs/smoke.yaml` | Quick tests: 50 questions, TF-IDF likelihood, 3k PPO timesteps |
| `configs/t5_policy.yaml` | T5 policy pipeline: model, supervised, PPO, and data sections |

qb-rl config aliases are supported (e.g., `data.dataset`, `likelihood.sbert_name`, `environment.reward` as alias for `reward_mode`).

## Compatibility Bridge

Old qb-rl import paths that still resolve:

- `qb_env.data_loader`, `qb_env.mc_builder`, `qb_env.text_utils`
- `models.answer_profiles`
- `agents.softmax_profile_buzzer`

OpenAI support is opt-in only. Default local workflows stay offline-friendly and do not require the `openai` package or `OPENAI_API_KEY`.

## Conventions

- NumPy-style docstrings with Parameters/Returns sections
- RL notation: `V` (value), `R` (reward), `T` (transition), `gamma` (discount), `s`/`a` (state/action)
- Prefer NumPy/PyTorch vectorized operations over loops in ML code
- Explicit seeds for reproducibility (use 1, 2, 3 for multi-seed runs)
