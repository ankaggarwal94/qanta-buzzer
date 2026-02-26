# Quiz Bowl RL Buzzer (Unified)

## What This Is

A unified RL-based quiz bowl buzzer system that decides when to buzz in and which answer to select as clues are revealed incrementally. Built on qb-rl's modular architecture (Gymnasium env, YAML config, belief features, S_q scoring) with qanta-buzzer's T5 encoder integrated as both a likelihood model and an optional policy encoder. Supports MLP policy on belief features (SB3 PPO) and T5 end-to-end policy with custom heads. CS234 final project, shipped v1.0.

## Core Value

A principled, modular RL system that produces rigorous experimental results — S_q scoring, baseline comparisons, calibration metrics, and ablation controls — for the CS234 writeup, while supporting both lightweight belief-feature policies and T5-based semantic policies.

## Requirements

### Validated

- ✓ Modular architecture with separate packages for data, models, env, agents, evaluation — v1.0
- ✓ Gymnasium-compliant TossupMCEnv with configurable reward modes (time_penalty, simple, human_grounded) — v1.0
- ✓ YAML configuration system with CLI override support — v1.0
- ✓ Belief feature extraction (margin, entropy, stability, progress) as (K+6) observation vector — v1.0
- ✓ LikelihoodModel ABC with TF-IDF, SBERT, and T5 implementations + factory — v1.0
- ✓ T5 as LikelihoodModel (semantic similarity scoring via encoder mean pooling) — v1.0
- ✓ T5 as policy encoder with custom heads (wait/answer/value) + supervised warm-start — v1.0
- ✓ MLP policy trained with SB3 PPO on belief features — v1.0
- ✓ Four baseline agents: Threshold, SoftmaxProfile, SequentialBayes, AlwaysBuzzFinal — v1.0
- ✓ Anti-artifact guards in MC construction (alias collision, token overlap, length ratio, question overlap) — v1.0
- ✓ S_q metric with episode traces (c_trace, g_trace) — v1.0
- ✓ Calibration metrics (ECE, Brier) and per-category accuracy breakdown — v1.0
- ✓ Control experiments: choices-only, shuffle, alias substitution — v1.0
- ✓ Comparison plots: calibration curves, entropy vs clue index, agent comparison tables — v1.0
- ✓ Four-stage pipeline: build_mc_dataset, run_baselines, train_ppo, evaluate_all — v1.0
- ✓ Smoke test mode (`--smoke`) completing full pipeline in <15 seconds — v1.0
- ✓ CSV primary data source with HuggingFace fallback — v1.0
- ✓ Comparison experiment: T5-as-likelihood vs T5-as-policy — v1.0

### Active

(None — v1.0 complete. See v2 requirements for future work.)

### Out of Scope

- Web UI or interactive demo — not needed for writeup
- OpenAI embedding likelihood model — API cost, SBERT sufficient
- Real-time quiz bowl game integration — academic project only
- Multi-GPU distributed training — single GPU/MPS sufficient for dataset size
- Custom PPO for MLP policy — SB3 is battle-tested (custom PPO only for T5 policy)
- Ensemble models — time constraint
- Bootstrap confidence intervals — deferred to v2

## Context

**v1.0 shipped** with 16,675 lines of Python across 61 files, 204 pytest tests, and a complete four-stage pipeline.

**Architecture:** `qb_data/` (data pipeline) → `models/` (likelihood + belief features) → `qb_env/` (Gymnasium env) → `agents/` (baselines + PPO) → `evaluation/` (metrics + controls + plots) → `scripts/` (pipeline orchestration) → `training/` (T5 policy training).

**Dual policy support:** MLP policy trains on (K+6) belief features via SB3 PPO. T5 policy trains end-to-end on text via custom PPO with GAE and supervised warm-start.

**Tech stack:** Python 3.12, PyTorch 2.3+, Transformers 4.45+, Stable-Baselines3 2.6+, Gymnasium 1.1+, sentence-transformers 3.3+, scikit-learn 1.3+.

## Constraints

- **Hardware**: Single GPU (MPS on Mac) or CPU fallback. 16GB RAM minimum for T5-large.
- **Data**: QANTA CSV dataset (14.9MB, locally available).
- **Dependencies**: PyTorch, Transformers, Stable-Baselines3, Gymnasium, sentence-transformers, scikit-learn
- **Python**: 3.12
- **Compatibility**: All scripts support `--smoke` flag for fast iteration

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Rebuild around qb-rl architecture | Cleaner modularity, better eval framework, S_q metric | ✓ Good — clean separation across 7 packages |
| T5 as both likelihood model and policy encoder | Maximize flexibility, compare approaches in writeup | ✓ Good — both approaches implemented and comparable |
| Supervised warm-start as config toggle | Useful for T5 policy, unnecessary for MLP policy | ✓ Good — 3-5x faster convergence for T5 |
| CSV primary, HF optional | Data already local, minimize external dependencies | ✓ Good — avoids network dependency |
| Keep anti-artifact guards | Ensures fair MC construction, strengthens writeup rigor | ✓ Good — 4-layer guard system prevents shortcuts |
| T5EncoderModel over T5ForConditionalGeneration | 2x faster, 50% less memory | ✓ Good — decoder unused for encoder-only tasks |
| SB3 PPO for MLP, custom PPO for T5 | SB3 battle-tested for numeric obs; T5 needs text handling | ✓ Good — each approach uses optimal framework |
| TF-IDF for fast agent tests, SBERT/T5 for semantic | Keeps test suite fast (<30s) | ✓ Good — 204 tests in ~10s |

---
*Last updated: 2026-02-26 after v1.0 milestone*
