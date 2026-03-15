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

The T5 policy uses factorized action semantics: the wait head models `P(WAIT)` vs `P(BUZZ)`, the answer head models `P(answer | BUZZ)`, and the flat action distribution is `P(WAIT)` plus `P(BUZZ_i) = P(BUZZ) * P(answer_i | BUZZ)`.

The CLI also reserves `--hazard-pretrain`, `--beta-terminal`, and `--freeze-answer-head` for an experimental hazard-style warm-start bridge. Those flags are parsed, but `--hazard-pretrain` currently raises `NotImplementedError` until the training loop is wired.

### Policy comparison

```bash
python scripts/compare_policies.py --t5-checkpoint checkpoints/ppo_t5/best_model
```

Compares the MLP belief-feature policy against the T5 end-to-end policy on the same test set. Accuracy and buzz-position metrics are directly comparable. ECE and Brier are computed identically (top-answer probability at buzz time). S_q and reward comparisons are qualitative because the two architectures use different confidence semantics (belief-sigmoid vs wait-head probability) and different reward settings (config-driven vs T5-pipeline defaults).

### Additional scripts

- `scripts/run_smoke_pipeline.py` -- runs all four smoke stages sequentially and writes a timing summary to `artifacts/smoke/smoke_pipeline_summary.json`
- `scripts/sweep_reward_shaping.py` -- grid sweep over `wait_penalty` and `early_buzz_penalty` with multi-seed evaluation
- `scripts/train_ppo.py --policy-mode flat_kplus1|stop_only` -- optional stop-only PPO surface; default remains `flat_kplus1`
- `generate_presentation.py` -- generates the Marp presentation slides

## Configuration

Two primary YAML configs:

| Config | Purpose | Key reward settings |
|--------|---------|-------------------|
| `configs/default.yaml` | Full runs | `wait_penalty=0.05`, `early_buzz_penalty=0.2`, `buzz_incorrect=-0.5` |
| `configs/smoke.yaml` | Quick tests (50 questions) | Same as default except `buzz_incorrect=-1.0`, `total_timesteps=3000` |
| `configs/t5_policy.yaml` | T5 pipeline | Own `model`/`supervised`/`ppo`/`data` sections; no `environment` |

qb-rl config aliases are also supported: `data.dataset`, `data.dataset_config`, `likelihood.sbert_name`, `environment.reward` as an alias for `reward_mode`, etc.

For horizon behavior, `environment.end_mode` defaults to `force_commit` (legacy behavior). Set `environment.end_mode: no_buzz` with `environment.no_buzz_reward` to end the episode without forcing a terminal answer.

## Testing

342 tests across 24 test files (3 skipped when optional extras not installed):

```bash
pytest                    # full suite
pytest tests/test_agents.py tests/test_environment.py tests/test_ppo_buzzer.py  # quick iteration
```

The test suite covers:

- Baseline agents (threshold, softmax-profile, sequential Bayes) and PPO wrapper
- Gymnasium environment behavior, reward modes (including Expected Wins), and belief computation
- Likelihood model factories (TF-IDF, SBERT, DSPy with offline-safe stubs)
- T5 policy model, supervised trainer, and PPO trainer
- Evaluation metrics (S_q, Expected Wins, ECE, Brier score, calibration at buzz, per-category accuracy)
- Dataset split reproducibility (cross-process determinism)
- Variable-K dataset construction and mixed-K integration
- Opponent buzz models (logistic, empirical)
- qb-rl compatibility bridge
- Text observation wrapper

## Architecture

```
qb_data/        Data loading, answer profiles, stratified splits, MC construction, DSPy profiles
qb_env/         Gymnasium environment, text wrapper, opponent models, optional StopOnlyEnv wrapper, qb-rl shims
models/         Likelihood models (TF-IDF, SBERT, T5, OpenAI, DSPy), belief features, T5 policy
agents/         Threshold, softmax-profile, sequential Bayes, PPO buzzer
evaluation/     S_q metric, Expected Wins, calibration, control experiments, plotting
scripts/        Pipeline entrypoints, DSPy compile, shared helpers
training/       T5 policy supervised + PPO trainers, hazard bridge utilities
configs/        YAML configuration files
artifacts/      Generated pipeline outputs (smoke/ and main/)
_legacy/        Pre-modularization prototypes (not installed)
```

## Compatibility Bridge

These old qb-rl import paths resolve in this repo:

- `qb_env.data_loader`, `qb_env.mc_builder`, `qb_env.text_utils`
- `models.answer_profiles`
- `agents.softmax_profile_buzzer`

The bridge is additive. `qb_data/` remains the canonical home for data loading and MC construction. OpenAI support is opt-in only -- default local workflows stay offline-friendly.

## Documentation

- `AGENTS.md` -- canonical repo contract for all coding agents (setup, architecture, testing, configuration)
- `CLAUDE.md` -- thin shim pointing to AGENTS.md with Claude-specific notes
- `walkthrough.md` -- end-to-end walkthrough exercising both pipelines (pre-remediation snapshot)
- `PRESENTATION.md` -- Marp presentation slides for the CS234 final project
- `.planning/` -- canonical project state, roadmap, architectural decisions, and remediation log

## Extensions (opt-in)

Three opt-in extensions are available. All are disabled by default — the smoke pipeline and T5 smoke path work unchanged.

### Expected Wins reward mode

Set `environment.reward_mode: expected_wins` and configure `environment.opponent_buzz_model` in YAML. Supports logistic and empirical (from `human_buzz_positions`) opponent models. Offline `expected_wins_score()` in `evaluation/metrics.py` uses the continuous formula: `V_self = g * R_correct + (1-g) * R_incorrect`.

### Variable-K answer choices

Set `data.variable_K: true` and `data.min_K` / `data.max_K` in YAML. `MCBuilder` samples K per question. The env uses padded observations and `action_masks()`. Optional `MaskablePPO` via `pip install -e '.[maskable]'`.

### DSPy integration (experimental)

Set `likelihood.model: dspy` and configure the `dspy` section in YAML. Requires `pip install -e '.[dspy]'`. Offline compile via `python scripts/optimize_dspy.py`. Does NOT integrate prompt optimization into PPO rollouts.

## Legacy Prototype

The pre-modularization prototype (`main.py`, `environment.py`, `model.py`, `dataset.py`, `config.py`, etc.) has been moved to `_legacy/`. These files are not part of the installed package and are preserved only for reference. The modular `scripts/` pipeline above is the canonical workflow.
