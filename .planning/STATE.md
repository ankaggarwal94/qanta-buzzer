---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_plan: Not started
status: completed
last_updated: "2026-03-08T18:00:00.000Z"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 20
  completed_plans: 19
  percent: 100
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

**Phase:** All phases complete
**Current Plan:** Not started
**Status:** v1.0 milestone complete
**Progress:** [██████████] 100%

### Active Work
- Completed: Plan 06-01 (T5PolicyModel architecture with 3 custom heads, 18 tests)
- Completed: Plan 06-02 (TextObservationWrapper and supervised training, 20 tests)
- Completed: Plan 06-03 (Custom PPO and comparison experiment, 14 tests)

### Completed Phases
1. Phase 01 - Data Pipeline Foundation (5/5 plans complete)
2. Phase 02 - Environment and Core Likelihood Models (4/4 plans complete)
3. Phase 03 - Baseline Agents and T5 Likelihood (3/3 plans complete)
4. Phase 04 - PPO Training Pipeline (3/3 plans complete)
5. Phase 05 - Evaluation Framework (2/2 plans complete)

### Upcoming Plans
- None - all plans complete

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
| Keep qanta-buzzer canonical | Bridge qb-rl into the unified repo instead of restoring qb-rl layout | 2026-03-06 |
| Add qb-rl compatibility shims | Preserve old imports/config keys with thin re-exports and aliases | 2026-03-06 |
| OpenAI support is optional | Default workflows remain offline-friendly; OpenAI activates only when selected | 2026-03-06 |
| Rewrite stale root docs | `.planning/` plus codebase are the source of truth over stale CLAUDE guidance | 2026-03-06 |
| Make bare `build_mc_dataset.py --smoke` a real workflow contract | Fix the code/docs mismatch by selecting smoke config and `artifacts/smoke/` defaults in code unless explicit overrides are passed | 2026-03-08 |
| Consolidate review remediation into PR #1 | Avoid stacked/noise follow-up history and keep one review surface for smoke + agent fixes | 2026-03-08 |
| Shared sigmoid helper lives in `agents/_math.py` | Confidence math belongs with agents; the stable implementation avoids overflow warnings in extreme cases | 2026-03-08 |
| Phase 05 P01 | 2min | 2 tasks | 3 files |
| Phase 05 P02 | 3min | 3 tasks | 1 files |
| Phase 06 P01 | 5min | 3 tasks | 3 files |
| Phase 06 P02 | 7min | 3 tasks | 7 files |
| Phase 06 P03 | 6min | 3 tasks | 4 files |

### Architecture Decisions
- Four-layer modular architecture: Pipeline → Agent → Environment → Model
- Dual policy support: MLP on belief features vs T5 end-to-end
- T5 serves dual purpose: likelihood model and optional policy
- YAML configuration with factory methods
- Gymnasium-compliant environment
- SB3 PPO for MLP policy, custom for T5 policy

### Quick Tasks Completed

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 1 | Repo-contract scaffolding: AGENTS.md, thin CLAUDE.md shim, .agentic.yml, ci.sh, manual-smoke.sh | 2026-03-13 | f478d1b3 | Verified | [1-repo-contract-scaffolding-agents-md-thin](./quick/1-repo-contract-scaffolding-agents-md-thin/) |
| 2 | Precompute belief-observation trajectories for PPO training speedup | 2026-03-13 | c3a69552, 0e8a60fa | Verified | [2-precompute-belief-observation-trajectori](./quick/2-precompute-belief-observation-trajectori/) |
| 3 | Persist embedding cache across subprocesses via .npz | 2026-03-13 | 553c78a1, d17f0f3b | Verified | [3-persist-cache-artifacts-across-subproces](./quick/3-persist-cache-artifacts-across-subproces/) |
| 4 | Collapse duplicate baseline sweeps into one-pass precomputed evaluation | 2026-03-13 | cdb89290, a9fb6da6, e56f125c | Verified | [4-collapse-duplicate-baseline-sweeps-into-](./quick/4-collapse-duplicate-baseline-sweeps-into-/) |
| 5 | Cache answer profiles: memoize _profile_text with (answer, exclude_qid) dict | 2026-03-13 | 476a24de, dcce59d8 | Verified | [5-cache-answer-profiles-especially-leave-o](./quick/5-cache-answer-profiles-especially-leave-o/) |

### Known Issues
None yet

### Technical Debt
- Two existing codebases to merge (qb-rl and qanta-buzzer)
- Different observation spaces (numeric beliefs vs text)
- Memory requirements for T5-large (may need T5-base)

### Performance Bottlenecks
- likelihood_model.score() dominated PPO training wall time (mitigated by quick task 2: precomputed belief cache)

## Session Continuity

### Last Session Summary
- Executed Plan 06-03: Custom PPO trainer and comparison experiment
- PPOTrainer with RolloutBuffer, GAE, dynamic padding, memory-safe rollouts (933 lines)
- End-to-end supervised-to-PPO training script with smoke mode (338 lines)
- Comparison experiment: T5-as-likelihood vs T5-as-policy on same test set (468 lines)
- 14 new tests passing, total project tests ~52+
- 4 files created
- All 20 plans across 6 phases now complete (100%)

### Next Session Priority
1. CS234 writeup preparation
2. Run full training pipeline: `python scripts/train_t5_policy.py --config configs/t5_policy.yaml`
3. Run comparison experiment for paper results

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
Last activity: 2026-03-13 - Completed quick task 5: Cache answer profiles with _profile_text memoization
