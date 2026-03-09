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

If smoke commands fail due to optional extras not being present, install:

```bash
python -m pip install sentence-transformers seaborn
```

## Main Workflows

Belief-feature / PPO pipeline:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

One-command smoke pipeline:

```bash
python scripts/run_smoke_pipeline.py
```

This writes a stage-by-stage runtime summary to `artifacts/smoke/smoke_pipeline_summary.json`.

Quick result snapshot is also available at `artifacts/smoke/RESULTS_SUMMARY.md`.

Reward-shaping sweep for PPO smoke tuning:

```bash
python scripts/sweep_reward_shaping.py
```

Multi-seed sweep (recommended for stable selection):

```bash
python scripts/sweep_reward_shaping.py --seeds 13,42,123
```

Longer-horizon sweeps:

```bash
python scripts/sweep_reward_shaping.py --seeds 13,42,123 --timesteps 5000
python scripts/sweep_reward_shaping.py --seeds 13,42,123 --timesteps 10000
```

This writes:
- `artifacts/smoke/reward_sweep_results.json`
- `artifacts/smoke/reward_sweep_results.csv`

The sweep ranks settings by a balanced objective:
- `objective = mean_accuracy + mean_S_q - 0.5 * mean_ECE`

Current candidate reward settings for non-smoke PPO runs:
- `environment.wait_penalty: 0.05`
- `environment.early_buzz_penalty: 0.2`

T5 policy track:

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/compare_policies.py --config configs/t5_policy.yaml
```

Drop `--smoke` for full runs.

### Verified Smoke Runbook (Windows)

The following sequence was validated end-to-end in this repo:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

Smoke outputs are written to `artifacts/smoke/` (dataset, baseline summaries, PPO model, and `evaluation_report.json`).

## Config Notes

- Current canonical config style is CSV-first with optional Hugging Face fallback.
- qb-rl aliases are also supported:
  - `data.dataset`, `data.dataset_config`, `data.dataset_smoke`, `data.dataset_smoke_config`, `data.split`
  - `likelihood.sbert_name`, `likelihood.openai_model`
  - `environment.reward` as an alias for `reward_mode`
- PPO behavior can be tuned via environment rewards in config:
  - `environment.wait_penalty`
  - `environment.early_buzz_penalty` (extra penalty for very early buzzing)
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
