# Quiz Bowl RL Buzzer (Unified)

Unified CS234 final project codebase for quiz bowl buzzing under incremental clues.

This repo keeps `qanta-buzzer` as the canonical implementation while preserving a qb-rl compatibility bridge:

- Modular belief-feature pipeline: `qb_data/` -> `models/` -> `qb_env/` -> `agents/` -> `evaluation/` -> `scripts/`
- T5 policy pipeline: supervised warm-start and PPO for end-to-end text-based buzzing
- qb-rl-compatible import/config shims for older notebooks and scripts
- Optional OpenAI embedding support (`likelihood.model: openai`, `data.distractor_strategy: openai_profile`)

## Setup

Requires Python >= 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional OpenAI support:

```bash
pip install -e '.[openai]'
export OPENAI_API_KEY=...
```

## Main Workflows

### Belief-feature / PPO pipeline

The canonical four-stage smoke pipeline:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

`--smoke` selects `configs/smoke.yaml` and writes outputs to `artifacts/smoke/`. Drop `--smoke` for full runs (uses `configs/default.yaml`, writes to `artifacts/main/`).

The smoke config uses tuned reward settings (`wait_penalty=0.05`, `early_buzz_penalty=0.2`, `ppo.seed=13`, `ppo.total_timesteps=3000`).

`train_ppo.py` also accepts `--seed` to override the PPO/environment seed, and `--stochastic-eval` / `--deterministic-eval` to control post-training evaluation mode.

### T5 policy pipeline

Trains a T5-based policy with supervised warm-start followed by PPO fine-tuning:

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke  # quick test with t5-small
```

The T5 pipeline uses its own config (`configs/t5_policy.yaml`) which defines `model`, `supervised`, `ppo`, and `data` sections. It does not inherit `environment` or `likelihood` settings from the belief-feature configs -- the T5 PPO trainer uses default reward settings (`wait_penalty=0.1`).

### Policy comparison

```bash
python scripts/compare_policies.py --t5-checkpoint checkpoints/ppo_t5/best_model
```

Compares the MLP belief-feature policy against the T5 end-to-end policy on the same test set. Both evaluation paths use TF-IDF likelihood internally. Note: the MLP side uses config-driven env settings while the T5 side hardcodes its own defaults, so numeric comparisons should be interpreted with care.

### Additional scripts

- `scripts/run_smoke_pipeline.py` -- runs all four smoke stages sequentially and writes a timing summary to `artifacts/smoke/smoke_pipeline_summary.json`
- `scripts/sweep_reward_shaping.py` -- grid sweep over `wait_penalty` and `early_buzz_penalty` with multi-seed evaluation
- `generate_presentation.py` -- generates the Marp presentation slides

## Configuration

Two primary YAML configs:

| Config | Purpose | Key reward settings |
|--------|---------|-------------------|
| `configs/default.yaml` | Full runs | `wait_penalty=0.05`, `early_buzz_penalty=0.2`, `buzz_incorrect=-0.5` |
| `configs/smoke.yaml` | Quick tests (50 questions) | Same as default except `buzz_incorrect=-1.0`, `total_timesteps=3000` |
| `configs/t5_policy.yaml` | T5 pipeline | Own `model`/`supervised`/`ppo`/`data` sections; no `environment` |

qb-rl config aliases are also supported: `data.dataset`, `data.dataset_config`, `likelihood.sbert_name`, `environment.reward` as an alias for `reward_mode`, etc.

## Testing

220 tests across 13 test files:

```bash
pytest                    # full suite
pytest tests/test_agents.py tests/test_environment.py tests/test_ppo_buzzer.py  # quick iteration
```

The test suite covers:

- Baseline agents (threshold, softmax-profile, sequential Bayes) and PPO wrapper
- Gymnasium environment behavior, reward modes, and belief computation
- Likelihood model factories (TF-IDF, SBERT with offline-safe stubs)
- T5 policy model, supervised trainer, and PPO trainer
- Evaluation metrics (S_q, ECE, Brier score, per-category accuracy)
- qb-rl compatibility bridge
- Text observation wrapper

## Architecture

```
qb_data/        Data loading, answer profiles, stratified splits, MC construction
qb_env/         Gymnasium environment, text wrapper, qb-rl compatibility shims
models/         Likelihood models (TF-IDF, SBERT, T5, OpenAI), belief features, T5 policy
agents/         Threshold, softmax-profile, sequential Bayes, PPO buzzer
evaluation/     S_q metric, calibration, control experiments, plotting
scripts/        Pipeline entrypoints and shared helpers
training/       T5 policy supervised + PPO trainers
configs/        YAML configuration files
artifacts/      Generated pipeline outputs (smoke/ and main/)
```

## Compatibility Bridge

These old qb-rl import paths resolve in this repo:

- `qb_env.data_loader`, `qb_env.mc_builder`, `qb_env.text_utils`
- `models.answer_profiles`
- `agents.softmax_profile_buzzer`

The bridge is additive. `qb_data/` remains the canonical home for data loading and MC construction. OpenAI support is opt-in only -- default local workflows stay offline-friendly.

## Documentation

- `walkthrough.md` -- showboat-generated end-to-end walkthrough exercising both pipelines
- `CLAUDE.md` -- agent guidance with setup, commands, and architecture reference
- `PRESENTATION.md` -- Marp presentation slides for the CS234 final project
- `.planning/` -- canonical project state, roadmap, and architectural decisions

## Legacy Prototype

The older root-level prototype (`main.py`, `environment.py`, `model.py`, `dataset.py`, `config.py`) is still present but is no longer the primary path. The modular `scripts/` pipeline above is the canonical workflow.
