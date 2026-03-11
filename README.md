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

### Belief-feature / PPO pipeline

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

Bare `python scripts/build_mc_dataset.py --smoke` is now a valid contract: it selects the smoke config path and writes datasets to `artifacts/smoke/` unless `--config` or `--output-dir` are supplied explicitly.

Drop `--smoke` for full runs.

### T5 policy track

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/compare_policies.py --config configs/t5_policy.yaml
```

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

The repo has a formal pytest suite under `tests/`, including:

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

## Legacy Prototype

The older root-level prototype (`main.py`, `environment.py`, `model.py`, `dataset.py`, `config.py`) is still present but is no longer the primary path. The modular `scripts/` pipeline above is the canonical workflow. See `walkthrough.md` for a guided code tour.
