---
phase: quick
plan: 4
subsystem: agents
tags: [numpy, precomputed-beliefs, sweep, optimization]

# Dependency graph
requires:
  - phase: quick-2
    provides: precompute_beliefs() and _PrecomputedQuestion for ThresholdBuzzer
provides:
  - _softmax_episode_from_precomputed for SoftmaxProfile sweep without model calls
  - _always_final_from_precomputed for AlwaysBuzzFinal without model calls
  - precompute_sequential_beliefs for one-pass Bayesian posterior computation
  - sweep_sequential_thresholds for multi-threshold SequentialBayes sweep
affects: [scripts/run_baselines.py, evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns: [precompute-then-sweep for all baseline agents]

key-files:
  created: []
  modified:
    - agents/threshold_buzzer.py
    - agents/bayesian_buzzer.py
    - agents/__init__.py
    - scripts/run_baselines.py
    - tests/test_agents.py

key-decisions:
  - "Reuse _PrecomputedQuestion dataclass for sequential beliefs (same shape: qid, gold_index, num_options, beliefs[])"
  - "Lazy imports to avoid circular dependency between threshold_buzzer and bayesian_buzzer"

patterns-established:
  - "Precompute-then-sweep: compute beliefs once, sweep thresholds with pure numpy"

requirements-completed: [OPT-03]

# Metrics
duration: 4min
completed: 2026-03-13
---

# Quick Task 4: Collapse Duplicate Baseline Sweeps Summary

**Eliminated all redundant likelihood_model.score() calls in run_baselines.py by making SoftmaxProfile, SequentialBayes, and AlwaysBuzzFinal reuse precomputed beliefs with pure numpy threshold sweeps**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-13T05:49:51Z
- **Completed:** 2026-03-13T05:54:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Added 5 new functions across agents/threshold_buzzer.py and agents/bayesian_buzzer.py for precomputed-path evaluation
- Wired run_baselines.py to use precomputed paths, removing all per-threshold agent instantiation and redundant model calls
- 4 equivalence tests prove numerical identity between precomputed and live paths
- All 40 agent tests pass (36 existing + 4 new)

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD RED): Failing equivalence tests** - `cdb89290` (test)
2. **Task 1 (TDD GREEN): Precomputed-path functions** - `a9fb6da6` (feat)
3. **Task 2: Wire run_baselines.py** - `e56f125c` (feat)

## Files Created/Modified
- `agents/threshold_buzzer.py` - Added _softmax_episode_from_precomputed and _always_final_from_precomputed
- `agents/bayesian_buzzer.py` - Added precompute_sequential_beliefs, _sequential_episode_from_precomputed, sweep_sequential_thresholds
- `agents/__init__.py` - Exported sweep_sequential_thresholds
- `scripts/run_baselines.py` - Replaced live agent loops with precomputed sweeps
- `tests/test_agents.py` - Added TestPrecomputedEquivalence class with 4 tests

## Decisions Made
- Reused _PrecomputedQuestion dataclass for sequential beliefs since it has the same shape (qid, gold_index, num_options, beliefs list)
- Used lazy imports to avoid circular dependency between threshold_buzzer and bayesian_buzzer modules

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All baseline agents now use precomputed beliefs exclusively
- run_baselines.py makes zero redundant model calls across all threshold sweeps
- Pre-existing test failures in T5/SBERT tests are unrelated (huggingface_hub version mismatch)

## Self-Check: PASSED

All created/modified files verified to exist. All 3 commit hashes verified in git log.

---
*Quick Task: 4*
*Completed: 2026-03-13*
