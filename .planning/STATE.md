---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_plan: Plan 3 of 3 (06-01, 06-02 complete)
status: in_progress
last_updated: "2026-02-26T06:29:43Z"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 20
  completed_plans: 19
  percent: 95
---

# Project State: Quiz Bowl RL Buzzer (Unified)

**Project:** Quiz Bowl RL Buzzer (Unified System)
**Last Updated:** 2026-02-25
**Current Sprint:** Milestone 1

## Project Reference

### Core Value
A principled, modular RL system that produces rigorous experimental results — S_q scoring, baseline comparisons, calibration metrics, and ablation controls — for the CS234 writeup, while supporting both lightweight belief-feature policies and T5-based semantic policies.

### Current Focus
Building unified system by merging qb-rl's modular architecture with qanta-buzzer's T5 integration. Critical path: data pipeline → environment → T5 likelihood → MLP policy → evaluation.

## Current Position

**Phase:** 6 - T5 Policy Integration
**Current Plan:** Plan 3 of 3 (06-01, 06-02 complete)
**Status:** In progress
**Progress:** [█████████▌] 95%

### Active Work
- Completed: Plan 06-01 (T5PolicyModel architecture with 3 custom heads, 18 tests)
- Completed: Plan 06-02 (TextObservationWrapper and supervised training, 20 tests)
- Next: Plan 06-03 (Custom PPO and comparison experiment)

### Completed Phases
1. Phase 01 - Data Pipeline Foundation (5/5 plans complete)
2. Phase 02 - Environment and Core Likelihood Models (4/4 plans complete)
3. Phase 03 - Baseline Agents and T5 Likelihood (3/3 plans complete)
4. Phase 04 - PPO Training Pipeline (3/3 plans complete)
5. Phase 05 - Evaluation Framework (2/2 plans complete)

### Upcoming Plans
- 06-03: Custom PPO and comparison experiment

## Performance Metrics

### Velocity
- **Plans Completed (24h):** 1
- **Plans Completed (7d):** 1
- **Average Plan Duration:** 6 minutes

### Quality
- **First-Try Success Rate:** N/A
- **Revision Count:** 0
- **Test Coverage:** N/A

### Time Distribution
- **Planning:** 100% (roadmap creation)
- **Implementation:** 0%
- **Debugging:** 0%
- **Documentation:** 0%

## Accumulated Context

### Key Decisions
| Decision | Rationale | Date |
|----------|-----------|------|
| Derive 6 phases from requirements | Natural groupings: data, environment, baselines+T5, training, evaluation, optional T5 policy | 2026-02-25 |
| Phase 1-5 critical path | Delivers MLP policy with T5 likelihood - core contribution | 2026-02-25 |
| Phase 6 optional | T5 policy is stretch goal given tight deadline | 2026-02-25 |
| Support T5-base fallback | Memory constraints may require smaller model | 2026-02-25 |
| Unified column name support | Accept both QANTA (Text/Answer) and generic formats | 2026-02-25 |
| Hash-based ID generation | Use MD5 for deterministic unique IDs when not provided | 2026-02-25 |
| Cumulative prefix pre-computation | Build all prefixes during loading to avoid repeated operations | 2026-02-25 |
| Use YAML for configuration | Human-readable, standard in ML projects, supports comments | 2026-02-25 |
| Dot notation for CLI overrides | Easy experimentation without editing files (e.g., data.K=5) | 2026-02-25 |
| Category-based stratification | Preserve category distribution across train/val/test splits | 2026-02-25 |
| HuggingFace as optional fallback | Provide alternative data source when CSV unavailable | 2026-02-25 |
| Match existing dataclass structure | Fixed field name inconsistencies between plans and implementation | 2026-02-25 |
| Fix CSV paths in configs | Updated configs to point to root directory where questions.csv exists | 2026-02-25 |
| Use _grouped attribute | AnswerProfileBuilder stores data in _grouped, not profiles | 2026-02-25 |
| Add .gitignore for generated data | Prevent large JSON files from being committed | 2026-02-25 |
| Port qb-rl features.py exactly | Maintain compatibility with downstream environment and agent plans | 2026-02-25 |
| LikelihoodModel returns raw scores | Environment applies softmax with temperature (separation of concerns) | 2026-02-25 |
| Factory supports dual config keys | Both sbert_name and embedding_model keys for cross-project compat | 2026-02-25 |
| Lazy imports for optional deps | sklearn and sentence_transformers imported inside class constructors | 2026-02-25 |
| Port qb-rl TossupMCEnv exactly | Maintain downstream compatibility with agent and training plans | 2026-02-26 |
| Adapt MCQuestion import path | Use qb_data.mc_builder (this codebase) not qb_env.mc_builder (qb-rl) | 2026-02-26 |
| Dual reward config key support | Factory checks 'reward' then falls back to 'reward_mode' for cross-project compat | 2026-02-26 |
| TF-IDF for fast tests | Most tests use TF-IDF (fast), SBERT only for pluggability and semantic tests | 2026-02-26 |
| Shared conftest fixtures | Centralized test data avoids duplication across 4 test modules | 2026-02-26 |
| Direct port from qb-rl agents | Only import path changes (qb_env -> qb_data) to preserve exact agent logic | 2026-02-26 |
| Consolidate bayesian buzzers | Merged softmax_profile_buzzer.py into bayesian_buzzer.py (both Bayesian-family) | 2026-02-26 |
| T5EncoderModel over full T5 | 2x faster inference and half memory vs T5ForConditionalGeneration | 2026-02-26 |
| T5TokenizerFast over T5Tokenizer | Faster tokenization via Rust-backed fast tokenizer | 2026-02-26 |
| TF-IDF for agent tests | 0.19s execution vs 5+ seconds with neural models for testing agent logic | 2026-02-26 |
| Module-scoped T5 fixture | Load t5-small once per test file, not per function, for efficiency | 2026-02-26 |
| Lazy import for PPOBuzzer | agents/__init__.py uses __getattr__ to avoid requiring SB3 for baseline-only runs | 2026-02-26 |
| Direct port from qb-rl PPOBuzzer | Only import path changes to preserve exact logic and SB3 integration | 2026-02-26 |
| TF-IDF for PPO agent tests | sample_tfidf_env fixture enables 2.4s test execution for 19 PPO tests | 2026-02-26 |
| TF-IDF for smoke mode baselines | 0.9s execution vs 30s+ with T5-small for baseline sweep | 2026-02-26 |
| Lazy import PPOBuzzer in agents/__init__.py | Avoid hard stable_baselines3 dependency for baseline-only runs | 2026-02-26 |
| Fallback MC dataset path | run_baselines.py checks data/processed/ when artifacts/ not found | 2026-02-26 |
| 3 thresholds in smoke config | Reduced sweep (0.5, 0.7, 0.9) vs 5 in default for quick validation | 2026-02-26 |
| Matplotlib Agg backend | Non-interactive backend for headless environments and CI | 2026-02-26 |
| Graceful alias_lookup fallback | Empty dict when alias_lookup.json missing, controls still run | 2026-02-26 |
| MC dataset path fallback | Check data/processed/ when artifacts/ not found for portability | 2026-02-26 |
| Port qb-rl controls exactly | choices-only, shuffle, alias substitution controls from reference | 2026-02-26 |
| T5EncoderModel for T5PolicyModel | 2x faster, 50% less memory vs T5ForConditionalGeneration (decoder unused) | 2026-02-26 |
| T5TokenizerFast for T5PolicyModel | 3-5x faster tokenization via Rust backend, critical for PPO rollouts | 2026-02-26 |
| Lazy import T5PolicyModel | models/__init__.py uses __getattr__ to avoid loading transformers for belief-only usage | 2026-02-26 |
| Config-dict interface for T5PolicyModel | Accept dict instead of qanta-buzzer Config class for unified codebase compat | 2026-02-26 |
| TextObservationWrapper via cumulative_prefixes | step_idx maps directly to visible prefix index for accurate clue visibility | 2026-02-26 |
| Loss scaled by 1/grad_accum_steps | Correct gradient magnitude when accumulating over multiple batches | 2026-02-26 |
| Nested smoke section in config YAML | Clean override pattern without separate config file | 2026-02-26 |
| Best model by validation accuracy | checkpoints/supervised/best_model/ tracks highest val_acc across epochs | 2026-02-26 |
| Phase 05 P01 | 2min | 2 tasks | 3 files |
| Phase 05 P02 | 3min | 3 tasks | 1 files |
| Phase 06 P01 | 5min | 3 tasks | 3 files |
| Phase 06 P02 | 7min | 3 tasks | 7 files |

### Architecture Decisions
- Four-layer modular architecture: Pipeline → Agent → Environment → Model
- Dual policy support: MLP on belief features vs T5 end-to-end
- T5 serves dual purpose: likelihood model and optional policy
- YAML configuration with factory methods
- Gymnasium-compliant environment
- SB3 PPO for MLP policy, custom for T5 policy

### Known Issues
None yet

### Technical Debt
- Two existing codebases to merge (qb-rl and qanta-buzzer)
- Different observation spaces (numeric beliefs vs text)
- Memory requirements for T5-large (may need T5-base)

### Performance Bottlenecks
None identified yet

## Session Continuity

### Last Session Summary
- Executed Plan 06-02: TextObservationWrapper and supervised warm-start training
- TextObservationWrapper bridges TossupMCEnv beliefs to text observations via cumulative_prefixes
- SupervisedTrainer with gradient accumulation (4 steps, effective batch=32)
- T5 policy config YAML with t5-large defaults and t5-small smoke overrides
- 20 new tests passing (8 wrapper + 12 supervised), total project tests ~38+
- 7 files created/modified

### Next Session Priority
1. Plan 06-03: Custom PPO and T5-as-likelihood vs T5-as-policy comparison
2. CS234 writeup preparation

### Context for Next Claude
This is a CS234 final project due this week. We're merging two existing codebases:
- qb-rl: Has the modular architecture we want (Gymnasium env, belief features, S_q metric, baselines)
- qanta-buzzer: Has T5 integration we need (encoder, policy heads, supervised warm-start)

The novel contribution is using T5 as a likelihood model to compute beliefs for an MLP policy, then comparing with T5 as an end-to-end policy. Phase 1-5 is the critical path for the deadline. Phase 6 (T5 policy) is optional if time permits.

Key risks to watch:
- Scope explosion (stick to critical path)
- Memory issues with T5-large (have T5-base ready)
- Observation space incompatibility (keep interfaces clean)
- Belief state collapse (pre-compute answer profiles)

### Open Questions
1. Should we start with existing qanta-buzzer data loading or rebuild from qb-rl?
2. Is supervised warm-start necessary for T5 policy or just helpful?
3. What's the optimal time penalty coefficient for reward shaping?
4. Should we implement all 4 baselines or just threshold for MVP?

### Environment State
- Working directory: `/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer`
- Python environment: Not yet configured (needs 3.11+)
- Git status: Roadmap files created, not yet committed
- Dependencies: Not yet installed

---
*State file initialized: 2026-02-25*
*Last update: 2026-02-26 (Plan 06-01 completed, Phase 6 in progress)*