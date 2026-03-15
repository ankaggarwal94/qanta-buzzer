# Architecture

## System Overview

Two-track quiz bowl buzzer system with three opt-in extensions:

1. **Belief-feature pipeline:** Build MC tossups → score with likelihood models → train/compare buzzers → evaluate with S_q, Expected Wins, and calibration metrics
2. **T5 policy pipeline:** Supervised warm-start → PPO fine-tuning for an end-to-end text policy

Both tracks share the same data layer (`qb_data/`) and environment (`qb_env/`).

**Opt-in extensions:** Expected Wins reward mode (opponent models), Variable-K answer choices (padded obs + action masks), DSPy LM-based scoring (offline compile).

## Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Scripts Layer (pipeline entrypoints)                    │
│  scripts/build_mc_dataset.py → run_baselines.py →       │
│  train_ppo.py → evaluate_all.py                         │
│  scripts/train_t5_policy.py → compare_policies.py       │
│  scripts/optimize_dspy.py  (offline DSPy compile)        │
├─────────────────────────────────────────────────────────┤
│  Agent Layer                                             │
│  agents/threshold_buzzer.py  (ThresholdBuzzer)           │
│  agents/bayesian_buzzer.py   (SoftmaxProfileBuzzer)      │
│  agents/ppo_buzzer.py        (PPOBuzzer via SB3)         │
├─────────────────────────────────────────────────────────┤
│  Evaluation Layer                                        │
│  evaluation/metrics.py   (S_q, EW, ECE, Brier, accuracy)  │
│  evaluation/controls.py  (shuffle, choices-only, alias)   │
│  evaluation/plotting.py  (calibration curves, entropy)    │
├─────────────────────────────────────────────────────────┤
│  Environment Layer                                       │
│  qb_env/tossup_env.py    (TossupMCEnv: EW, variable-K)   │
│  qb_env/stop_only_env.py (StopOnlyEnv: Discrete(2))      │
│  qb_env/opponent_models.py (opponent buzz model protocol) │
│  qb_env/text_wrapper.py  (TextObservationWrapper)        │
├─────────────────────────────────────────────────────────┤
│  Model Layer                                             │
│  models/likelihoods.py   (TfIdf, SBERT, T5, OpenAI)      │
│  models/dspy_likelihood.py (DSPyLikelihood, score cache)  │
│  models/features.py      (belief + padded features)       │
│  models/t5_policy.py     (T5PolicyModel + PolicyHead)     │
├─────────────────────────────────────────────────────────┤
│  Data Layer                                              │
│  qb_data/data_loader.py     (QANTA CSV + HF loading)     │
│  qb_data/mc_builder.py      (MCBuilder + anti-artifact)   │
│  qb_data/answer_profiles.py (answer profile generation)   │
│  qb_data/dataset_splits.py  (stratified train/val/test)   │
│  qb_data/config.py          (YAML config loading)         │
│  qb_data/text_utils.py      (normalization, tokenization) │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### Belief-Feature Pipeline

```
QANTA CSV / HuggingFace
    ↓ (qb_data/data_loader.py)
List[TossupQuestion]
    ↓ (qb_data/mc_builder.py)
List[MCQuestion]  (with K options, anti-artifact guards)
    ↓ (qb_data/dataset_splits.py)
train / val / test splits → mc_dataset.json
    ↓ (models/likelihoods.py)
LikelihoodModel.score() → raw similarity scores
    ↓ (softmax with beta temperature)
Belief distribution over K options
    ↓ (models/features.py)
[belief[0..K-1], top_p, margin, entropy, stability, progress, clue_idx_norm]
    ↓ (qb_env/tossup_env.py)
TossupMCEnv observation (Box(K+6,))
    ↓ (agents/)
Buzz decision → EpisodeResult / SoftmaxEpisodeResult / PPOEpisodeTrace
    ↓ (evaluation/)
S_q, ECE, Brier score, accuracy, per-category stats
```

### T5 Policy Pipeline

```
MCQuestion dataset
    ↓ (training/train_supervised_t5.py)
T5PolicyModel supervised warm-start
    ↓ (training/train_ppo_t5.py)
PPO fine-tuning on TossupMCEnv with TextObservationWrapper
    ↓ (scripts/compare_policies.py)
Policy comparison metrics
```

## Key Abstractions

### `TossupQuestion` (dataclass, `qb_data/data_loader.py`)
Core data structure: question text, tokens, answer, run_indices for clue boundaries, cumulative_prefixes for incremental reveal.

### `MCQuestion` (dataclass, extends TossupQuestion, `qb_data/mc_builder.py`)
Adds: options (K answer choices), gold_index, option_profiles, distractor_strategy. Four anti-artifact guards prevent spurious patterns.

### `LikelihoodModel` (ABC, `models/likelihoods.py`)
Pluggable scoring interface. Implementations: `TfIdfLikelihood`, `SBERTLikelihood`, `T5Likelihood`, `OpenAILikelihood`, `DSPyLikelihood`. Each implements `score(clue_prefix, option_profiles) → np.ndarray`. Embedding-based models also implement `_embed_batch()`; `DSPyLikelihood` raises `NotImplementedError` on embedding operations.

### `TossupMCEnv` (Gymnasium env, `qb_env/tossup_env.py`)
POMDP environment: Discrete(K+1) action space (WAIT + K buzz options), Box(K+6) observation space (belief features). Four reward modes: `time_penalty`, `simple`, `human_grounded`, `expected_wins`. Supports variable-K mode with padded observations and `action_masks()`. End-of-horizon behavior configurable via `end_mode` (`force_commit` | `no_buzz`).

### `StopOnlyEnv` (wrapper, `qb_env/stop_only_env.py`)
Discrete(2) wrapper (0=WAIT, 1=BUZZ) that maps BUZZ to argmax(belief). Selectable via `--policy-mode stop_only` in `train_ppo.py`.

### Agent hierarchy
- `ThresholdBuzzer`: simple confidence threshold
- `SoftmaxProfileBuzzer`: Bayesian belief updates with sigmoid confidence proxy
- `PPOBuzzer`: SB3 PPO wrapper with custom `run_episode()` for S_q trace recording; supports optional `MaskablePPO` and stop-only mode

## Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/build_mc_dataset.py` | Load questions, build MC dataset, save artifacts |
| `scripts/run_baselines.py` | Sweep threshold/Bayesian buzzers |
| `scripts/train_ppo.py` | Train PPO agent on belief features |
| `scripts/evaluate_all.py` | Full evaluation + controls + plots |
| `scripts/train_t5_policy.py` | T5 policy supervised + PPO training |
| `scripts/compare_policies.py` | Compare T5 vs belief-feature policies |
| `scripts/sweep_reward_shaping.py` | Multi-seed reward parameter sweep |
| `scripts/run_smoke_pipeline.py` | End-to-end smoke test |
| `scripts/optimize_dspy.py` | Offline DSPy compile/optimize |
| `scripts/run_full_pipeline.sh` | Full 19-phase pipeline (4-wave DAG, forces tfidf) |
| `scripts/manual-smoke.sh` | Four-stage smoke wrapper (venv-aware, python3) |

All pipeline scripts accept `--smoke` for fast testing and `--config` for custom YAML configs. `run_full_pipeline.sh` explicitly overrides `likelihood.model=tfidf` for all belief-feature phases. `compare_policies.py` auto-detects MPS/CUDA/CPU for T5 inference.

## qb-rl Compatibility Layer

The `qb_env/` package provides thin re-export shims that map old `qb_env.data_loader`, `qb_env.mc_builder`, and `qb_env.text_utils` import paths to their canonical `qb_data.*` counterparts. Similarly, `models/answer_profiles.py` re-exports from `qb_data/answer_profiles.py`. This preserves backward compatibility with the earlier qb-rl codebase.
