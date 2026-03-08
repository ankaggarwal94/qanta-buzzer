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
