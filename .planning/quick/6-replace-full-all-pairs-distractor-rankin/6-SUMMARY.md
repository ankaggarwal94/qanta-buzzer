---
phase: quick-6
plan: 1
subsystem: data
tags: [numpy, argpartition, tfidf, distractor-ranking, optimization]

requires:
  - phase: 01-data-pipeline
    provides: MCBuilder._compute_rankings with full argsort
provides:
  - Top-M argpartition ranking in _compute_rankings
  - _rank_by_similarity helper eliminating code duplication
affects: [mc_builder, build_mc_dataset, sbert_profile, openai_profile]

tech-stack:
  added: []
  patterns: [argpartition-then-sort for top-M retrieval]

key-files:
  created:
    - tests/test_mc_builder_topk.py
  modified:
    - qb_data/mc_builder.py

key-decisions:
  - "M = min(max(5*K, 30), N-1) balances candidate pool size with memory savings"
  - "Extract _rank_by_similarity helper to eliminate argsort duplication across tfidf/sbert/openai blocks"
  - "Full argsort fallback for small N where argpartition has no benefit"

patterns-established:
  - "Top-M retrieval pattern: argpartition for O(N) selection, argsort on subset for O(M log M) ordering"

requirements-completed: [OPT-5]

duration: 4min
completed: 2026-03-13
---

# Quick Task 6: Replace Full All-Pairs Distractor Ranking Summary

**Top-M argpartition in _compute_rankings reduces per-answer ranking from O(N log N) to O(N + M log M) and memory from O(N^2) to O(N*M)**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-13T06:33:33Z
- **Completed:** 2026-03-13T06:37:25Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Replaced full np.argsort with np.argpartition + partial sort for all profile-based strategies (tfidf, sbert, openai)
- Extracted _rank_by_similarity helper to eliminate duplicated ranking loops
- Added 4 regression tests verifying truncation, order preservation, small-N degradation, and category_random isolation

## Task Commits

Each task was committed atomically:

1. **Task 1: Add top-M regression test** - `b0d5d21b` (test)
2. **Task 2: Replace full argsort with top-M argpartition** - `bc8b3b46` (feat)

## Files Created/Modified
- `tests/test_mc_builder_topk.py` - 4 regression tests for top-M ranking behavior
- `qb_data/mc_builder.py` - _rank_by_similarity helper with argpartition, updated _compute_rankings

## Decisions Made
- M = min(max(5*K, 30), N-1): With K=4, M=30 gives 29 usable candidates after self-exclusion, far more than the 3 needed by build() guards
- Extracted _rank_by_similarity to avoid maintaining identical loops in tfidf and sbert/openai blocks
- Small-N fallback (M >= N-1) uses full argsort since argpartition offers no benefit when all candidates fit in M

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Pre-existing `huggingface_hub` import error (`is_offline_mode`) causes 12 SBERT/T5-dependent tests to fail and 18 T5-dependent tests to error. These are caused by a version mismatch between `transformers` and `huggingface_hub` in the shared venv and are completely unrelated to this task. All 178 non-transformer tests pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All profile-based strategies now use top-M retrieval
- Random fallback in build() still handles the rare case of all top-M candidates being rejected by guards
- No API changes; downstream code (build(), scripts) unaffected

## Self-Check: PASSED

- [x] `tests/test_mc_builder_topk.py` exists
- [x] `qb_data/mc_builder.py` exists
- [x] `6-SUMMARY.md` exists
- [x] Commit `b0d5d21b` exists
- [x] Commit `bc8b3b46` exists

---
*Quick Task: 6 - Replace full all-pairs distractor ranking*
*Completed: 2026-03-13*
