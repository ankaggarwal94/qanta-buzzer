---
phase: quick-8
plan: 1
subsystem: evaluation
tags: [precomputed-beliefs, shuffle-control, performance]

requires:
  - phase: quick-2
    provides: precompute_beliefs and _softmax_episode_from_precomputed in threshold_buzzer
provides:
  - run_shuffle_control_precomputed() that permutes precomputed beliefs with zero score() calls
  - evaluate_all.py uses precomputed beliefs for full eval and shuffle control
affects: [evaluation, scripts]

tech-stack:
  added: []
  patterns: [belief permutation for shuffle control without re-scoring]

key-files:
  created: []
  modified:
    - evaluation/controls.py
    - scripts/evaluate_all.py
    - tests/test_agents.py

key-decisions:
  - "Belief permutation via numpy indexing (belief[perm]) matches shuffled_option_copy semantics exactly"
  - "Alias control remains live-evaluator because alias substitution changes option text requiring re-scoring"

patterns-established:
  - "Precomputed shuffle: permute belief arrays instead of re-running likelihood model"

requirements-completed: [OPT-7]

duration: 5min
completed: 2026-03-13
---

# Quick Task 8: Stop Re-scoring Control Experiments Summary

**Shuffle control now permutes precomputed belief vectors instead of re-running likelihood_model.score(), eliminating all redundant scoring calls**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-13T07:20:00Z
- **Completed:** 2026-03-13T07:25:38Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `run_shuffle_control_precomputed()` to `evaluation/controls.py` that permutes belief vectors with zero model calls
- Wired `evaluate_all.py` to use precomputed beliefs for both full evaluation and shuffle control
- Equivalence test proves numerical identity between precomputed and live-rescore shuffle paths
- Alias control preserved as live evaluator since alias substitution genuinely changes option text

## Task Commits

Each task was committed atomically:

1. **Task 1: Add run_shuffle_control_precomputed (TDD)** - `a4603919` (test: RED), `01902552` (feat: GREEN)
2. **Task 2: Wire precomputed paths into evaluate_all.py** - `af199b8b` (feat)

## Files Created/Modified
- `evaluation/controls.py` - Added `run_shuffle_control_precomputed()` function
- `scripts/evaluate_all.py` - Replaced full eval and shuffle control with precomputed paths
- `tests/test_agents.py` - Added `TestShufflePrecomputedEquivalence` with 2 test methods

## Decisions Made
- Belief permutation via `belief[perm]` where `perm` is the same permutation used on options matches the "gather by new ordering" semantics of `shuffled_option_copy`
- Alias control keeps the live `SoftmaxProfileBuzzer` evaluator because alias substitution changes actual option text and profiles, requiring genuine re-scoring
- `evaluate_questions_live` closure retained only for alias control; `evaluate_questions_precomputed` used for full eval

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing `huggingface_hub` version incompatibility causes transformer-dependent tests (SBERT, T5) to fail with `ImportError: cannot import name 'is_offline_mode'`. This is unrelated to the changes and affects 5 test files. All 143 non-transformer tests pass.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Evaluation pipeline now makes one pass of `likelihood_model.score()` via `precompute_beliefs()`, then reuses cached beliefs for both full evaluation and shuffle control
- Alias control still requires live scoring (intentional)

## Self-Check: PASSED

- All 3 modified files exist on disk
- All 3 task commits (a4603919, 01902552, af199b8b) found in git log
- `run_shuffle_control_precomputed` importable from `evaluation.controls`
- Both equivalence tests pass (2/2)
- 143 non-transformer tests pass with zero regressions

---
*Phase: quick-8*
*Completed: 2026-03-13*
