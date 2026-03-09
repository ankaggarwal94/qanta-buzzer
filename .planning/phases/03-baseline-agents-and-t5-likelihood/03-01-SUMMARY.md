---
phase: 03-baseline-agents-and-t5-likelihood
plan: 01
subsystem: agents
tags: [baseline, threshold, bayesian, softmax, episode-trace, buzzer]

# Dependency graph
requires:
  - phase: 02-environment-and-core-likelihood-models
    provides: LikelihoodModel ABC, TfIdfLikelihood, SBERTLikelihood
  - phase: 01-data-pipeline-foundation
    provides: MCQuestion dataclass with cumulative_prefixes, tokens, run_indices
provides:
  - ThresholdBuzzer agent with confidence-based buzz decision
  - AlwaysBuzzFinalBuzzer agent that waits until last clue
  - SoftmaxProfileBuzzer with per-step belief recomputation
  - SequentialBayesBuzzer with incremental Bayesian updates
  - EpisodeResult and SoftmaxEpisodeResult dataclasses with c_trace, g_trace
  - sweep_thresholds utility for hyperparameter search
  - result_to_dict serialization utility
affects: [03-02, 03-03, 04-ppo-training, 05-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns: [episode-trace-pattern, belief-from-likelihood, confidence-proxy-sigmoid]

key-files:
  created:
    - agents/__init__.py
    - agents/threshold_buzzer.py
    - agents/bayesian_buzzer.py
  modified: []

key-decisions:
  - "Direct port from qb-rl with only import path changes (qb_env -> qb_data)"
  - "Consolidated softmax_profile_buzzer.py and bayesian_buzzer.py into single bayesian_buzzer.py"

patterns-established:
  - "Episode trace pattern: all agents return c_trace (buzz confidence) and g_trace (correctness) per step"
  - "Belief computation: softmax(beta * scores) with numerical stability (subtract max)"
  - "Confidence proxy: sigmoid(alpha * (top_p - threshold)) for smooth buzz decision"

requirements-completed: [AGT-02, AGT-03, AGT-04, AGT-05, AGT-06]

# Metrics
duration: 2min
completed: 2026-02-26
---

# Phase 3 Plan 1: Baseline Agents Summary

**Four baseline buzzer agents ported from qb-rl: ThresholdBuzzer, AlwaysBuzzFinal, SoftmaxProfile, and SequentialBayes with episode trace dataclasses**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-26T03:17:28Z
- **Completed:** 2026-02-26T03:19:35Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Ported all four baseline agents from qb-rl reference implementation
- All agents produce EpisodeResult/SoftmaxEpisodeResult with c_trace and g_trace for S_q evaluation
- Verified all agents run correctly on MCQuestion data with TF-IDF likelihood model
- sweep_thresholds and result_to_dict utilities ready for evaluation scripts

## Task Commits

Each task was committed atomically:

1. **Task 1: Port ThresholdBuzzer and AlwaysBuzzFinalBuzzer** - `b8f564c` (feat)
2. **Task 2: Port SoftmaxProfileBuzzer and SequentialBayesBuzzer** - `d05b685` (feat)
3. **Task 3: Create agents package exports** - `9e074ef` (feat)

## Files Created/Modified
- `agents/__init__.py` - Package exports for all 4 agents, 2 result types, 2 utilities
- `agents/threshold_buzzer.py` - ThresholdBuzzer, AlwaysBuzzFinalBuzzer, EpisodeResult, sweep_thresholds, result_to_dict
- `agents/bayesian_buzzer.py` - SoftmaxProfileBuzzer, SequentialBayesBuzzer, SoftmaxEpisodeResult

## Decisions Made
- Direct port from qb-rl with only import path changes (qb_env.mc_builder -> qb_data.mc_builder) to preserve exact agent logic
- Consolidated qb-rl's separate softmax_profile_buzzer.py into bayesian_buzzer.py since both buzzers are Bayesian-family agents

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 4 baseline agents ready for integration testing (Plan 03-02)
- Episode trace format (c_trace, g_trace) compatible with S_q evaluation metric
- Agents accept any LikelihoodModel subclass, ready for T5 likelihood integration (Plan 03-03)

## Self-Check: PASSED

- FOUND: agents/__init__.py
- FOUND: agents/threshold_buzzer.py
- FOUND: agents/bayesian_buzzer.py
- FOUND: commit b8f564c (Task 1)
- FOUND: commit d05b685 (Task 2)
- FOUND: commit 9e074ef (Task 3)

---
*Phase: 03-baseline-agents-and-t5-likelihood*
*Completed: 2026-02-26*
