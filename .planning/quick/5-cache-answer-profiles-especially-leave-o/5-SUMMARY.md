---
phase: quick-5
plan: 01
subsystem: data-pipeline
tags: [memoization, caching, answer-profiles, performance]

requires:
  - phase: 01-data-pipeline
    provides: AnswerProfileBuilder with _grouped and _profile_text
provides:
  - Transparent _cache dict on AnswerProfileBuilder._profile_text eliminating O(N*K) redundant string processing
affects: [mc_builder, build_mc_dataset, run_baselines]

tech-stack:
  added: []
  patterns: [dict-based memoization keyed by (answer_primary, exclude_qid) tuple]

key-files:
  created:
    - tests/test_answer_profile_cache.py
  modified:
    - qb_data/answer_profiles.py

key-decisions:
  - "Cache keyed by (answer_primary, exclude_qid) tuple -- covers both distractor (None) and leave-one-out cases"
  - "Cache invalidated in fit() to prevent stale data after re-fitting on new questions"
  - "Both early-return (fallback) and normal computation paths are cached"

patterns-established:
  - "Transparent memoization: callers see no API change, cache is internal to _profile_text"

requirements-completed: [OPT-4]

duration: 3min
completed: 2026-03-13
---

# Quick Task 5: Cache Answer Profiles Summary

**Dict-based memoization on AnswerProfileBuilder._profile_text elimininating repeated string join/split/truncate for identical (answer, exclude_qid) pairs**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-13T06:14:44Z
- **Completed:** 2026-03-13T06:18:09Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `_cache: Dict[Tuple[str, Optional[str]], str]` to AnswerProfileBuilder with lookup-before-compute and store-after-compute in `_profile_text()`
- Cache invalidated on `fit()` to prevent stale data when builder is re-fitted on new questions
- 6 new tests covering cache hits, invalidation, byte-equivalence, and efficiency
- Full test suite (228 passed, 13 pre-existing failures in SBERT/T5 env, 32 pre-existing errors) and smoke pipeline confirmed no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD RED): Add failing tests for cache** - `476a24de` (test)
2. **Task 1 (TDD GREEN): Implement _cache memoization** - `dcce59d8` (feat)
3. **Task 2: Full test suite and smoke integration** - No files modified, verification only

**Plan metadata:** (pending)

_Note: TDD task has RED + GREEN commits. No refactor needed -- diff was already minimal._

## Files Created/Modified
- `tests/test_answer_profile_cache.py` - 6 tests: cache hits (distractor + leave-one-out), invalidation on fit(), byte-equivalence, efficiency
- `qb_data/answer_profiles.py` - Added `_cache` dict init, `fit()` invalidation, cache lookup/store wrapping `_profile_text()` body (+10 lines, -1 line)

## Decisions Made
- Cache key is `(answer_primary, exclude_qid)` tuple -- the exact signature of `_profile_text`
- Both early-return path (min_questions fallback) and normal path cache their results
- Cache is an instance attribute cleared on `fit()`, not a class-level or global cache

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `test_imports.py` at repo root causes pytest collection error (stale file importing incompatible transformers). Pre-existing, unrelated to cache change. Scoped test run to `tests/` directory.
- 13 pre-existing SBERT/T5 test failures and 32 T5-related collection errors from huggingface_hub version incompatibility. All unrelated to cache change.

## User Setup Required

None - no external service configuration required.

## Next Task Readiness
- Cache is transparent -- no downstream changes needed in mc_builder.py, scripts, or agents
- Performance benefit scales with number of distractor reuses across questions

## Self-Check: PASSED

- FOUND: qb_data/answer_profiles.py
- FOUND: tests/test_answer_profile_cache.py
- FOUND: 5-SUMMARY.md
- FOUND: 476a24de (RED commit)
- FOUND: dcce59d8 (GREEN commit)
