# Quiz Bowl RL Buzzer (Unified)

## What This Is

An RL-based quiz bowl buzzer that decides when to buzz in and which answer to select as clues are revealed incrementally. Rebuilds around the qb-rl modular architecture (Gymnasium env, YAML config, belief features, S_q scoring, baselines, controls) while integrating qanta-buzzer's T5 encoder as both a likelihood model and an optional policy encoder. CS234 final project.

## Core Value

A principled, modular RL system that produces rigorous experimental results — S_q scoring, baseline comparisons, calibration metrics, and ablation controls — for the CS234 writeup, while supporting both lightweight belief-feature policies and T5-based semantic policies.

## Requirements

### Validated

- ✓ POMDP environment for incremental quiz bowl questions — existing (qanta-buzzer `environment.py`)
- ✓ T5 encoder with custom policy head (wait/answer/value) — existing (qanta-buzzer `model.py`)
- ✓ Supervised warm-start training on complete questions — existing (qanta-buzzer `train_supervised.py`)
- ✓ PPO fine-tuning with GAE advantage estimation — existing (qanta-buzzer `train_ppo.py`)
- ✓ Basic evaluation (accuracy, ECE, Brier, per-category) — existing (qanta-buzzer `metrics.py`)
- ✓ CSV dataset loading with distractor generation — existing (qanta-buzzer `dataset.py`)
- ✓ Checkpoint save/load with training history — existing
- ✓ Choices-only control experiment — existing (qanta-buzzer `metrics.py:evaluate_choices_only`)

### Active

- [ ] Rebuild around qb-rl's modular architecture (separate modules for env, models, agents, evaluation)
- [ ] Gymnasium-compliant environment with configurable reward modes (time_penalty, human_grounded, simple)
- [ ] YAML configuration system replacing Python Config class
- [ ] Belief feature extraction (margin, entropy, stability, progress) for policy input
- [ ] LikelihoodModel ABC with TF-IDF, SBERT, and T5 implementations
- [ ] T5 as a LikelihoodModel (scores options via encoder similarity)
- [ ] T5 as optional policy encoder (text + belief features → policy heads)
- [ ] MLP policy option (belief features → SB3 PPO, lightweight)
- [ ] Supervised warm-start as configurable pipeline stage (toggle via YAML)
- [ ] S_q metric (system score: sum of buzz probability × correctness)
- [ ] Four baseline agents: ThresholdBuzzer, SoftmaxProfileBuzzer, SequentialBayesBuzzer, AlwaysBuzzFinalBuzzer
- [ ] Anti-artifact guards in MC construction (alias collision, token overlap, length ratio)
- [ ] Evaluation controls: shuffle, alias substitution, choices-only
- [ ] Episode traces with c_trace/g_trace for S_q computation
- [ ] Calibration plots, entropy vs clue index, comparison tables
- [ ] CSV as primary data source, Hugging Face as optional fallback
- [ ] Smoke test mode for fast validation (`--smoke`)
- [ ] Four-stage pipeline scripts (build_mc, run_baselines, train_ppo, evaluate_all)

### Out of Scope

- Web UI or interactive demo — not needed for writeup
- OpenAI embedding likelihood model — API cost, SBERT sufficient
- Real-time quiz bowl game integration — academic project only
- Multi-GPU distributed training — single GPU/MPS sufficient for dataset size

## Context

**Two existing codebases being unified:**

- **qb-rl** (`../qb-rl/`): Modular pipeline with Gymnasium env, SB3 PPO, belief features, S_q scoring, 4 baselines, anti-artifact guards, YAML config. Uses lightweight MLP policy on numeric features. Production-quality architecture.

- **qanta-buzzer** (this repo): T5-large encoder with custom PPO, supervised warm-start, text-based observations. Monolithic structure but strong semantic understanding via pre-trained encoder.

**Key insight**: qb-rl's architecture is the better backbone. T5 adds value as a likelihood scorer (converting text to belief features) and optionally as a policy encoder (for end-to-end learning). The unified system should support both approaches and compare them in the writeup.

**CS234 final project** — deadline is this week. Must prioritize working results over perfect architecture.

## Constraints

- **Timeline**: This week — must produce runnable experiments and writeup-ready results
- **Hardware**: Single GPU (MPS on Mac) or CPU fallback. 16GB RAM minimum for T5-large.
- **Data**: QANTA CSV dataset (14.9MB, already available locally). HF loading as optional.
- **Dependencies**: PyTorch, Transformers, Stable-Baselines3, Gymnasium, sentence-transformers, scikit-learn
- **Python**: 3.11+
- **Compatibility**: Must support `--smoke` flag for fast iteration

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Rebuild around qb-rl architecture | Cleaner modularity, better eval framework, S_q metric | — Pending |
| T5 as both likelihood model and policy encoder | Maximize flexibility, compare approaches in writeup | — Pending |
| Supervised warm-start as config toggle | Useful for T5 policy, unnecessary for MLP policy | — Pending |
| CSV primary, HF optional | Data already local, minimize external dependencies | — Pending |
| Keep anti-artifact guards | Ensures fair MC construction, strengthens writeup rigor | — Pending |

---
*Last updated: 2026-02-24 after initialization*
