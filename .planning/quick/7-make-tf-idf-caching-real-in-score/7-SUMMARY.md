---
phase: quick-7
plan: 01
subsystem: models
tags: [tfidf, caching, cosine-similarity, L2-normalization]

requires:
  - phase: 02-environment
    provides: LikelihoodModel base class with embed_and_cache infrastructure
provides:
  - TfIdfLikelihood.score() using embed_and_cache() for cached scoring
  - L2-normalized _embed_batch() matching SBERT/T5 convention
affects: [agents, evaluation, scripts]

tech-stack:
  added: []
  patterns: [dot-product-equals-cosine for all likelihood models]

key-files:
  created: []
  modified:
    - models/likelihoods.py
    - tests/test_likelihoods.py

key-decisions:
  - "L2-normalize in _embed_batch rather than score to match SBERT/T5 convention"
  - "Guard zero-norm rows to avoid NaN on empty documents"

patterns-established:
  - "All LikelihoodModel subclasses now use embed_and_cache in score(): TF-IDF, SBERT, T5, OpenAI"

requirements-completed: [OPT-6]

duration: 3min
completed: 2026-03-13
---

# Quick Task 7: Make TF-IDF Caching Real in score() Summary

**TF-IDF score() now uses embed_and_cache() with L2-normalized embeddings, eliminating redundant vectorizer.transform() calls on repeated option profiles**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-13T06:56:10Z
- **Completed:** 2026-03-13T06:59:20Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- TfIdfLikelihood.score() rewritten to use self.embed_and_cache(), matching the SBERT/T5/OpenAI pattern
- _embed_batch() now returns L2-normalized float32 vectors so dot product equals cosine similarity
- Removed sklearn.metrics.pairwise.cosine_similarity import from score()
- 4 new regression tests confirm: normalization, cache population, cache hit, cosine equivalence
- All 203 non-transformers tests pass; 13 pre-existing SBERT/T5 failures due to huggingface_hub version mismatch are unrelated

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for TF-IDF caching** - `e65b5cab` (test)
2. **Task 1 GREEN: L2-normalize _embed_batch, rewrite score() to use embed_and_cache** - `ba773e72` (feat)
3. **Task 2: Full test suite verification** - no commit (verification only, no code changes)

## Files Created/Modified
- `models/likelihoods.py` - TfIdfLikelihood.score() uses embed_and_cache(); _embed_batch() L2-normalizes
- `tests/test_likelihoods.py` - 4 new tests: normalization, cache population, cache hit, cosine reference equivalence

## Decisions Made
- L2 normalization applied in _embed_batch() (not score()) so embed_and_cache stores normalized vectors; this matches the convention used by SBERT, T5, and OpenAI implementations
- Zero-norm guard (`norms[norms == 0] = 1.0`) prevents NaN for empty documents, matching OpenAI's pattern

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing huggingface_hub version incompatibility causes 13 SBERT/T5 test failures and 32 errors in the full suite. These are unrelated to our changes (the `is_offline_mode` import was removed from `huggingface_hub`). Documented for awareness but not fixed (out of scope).

## User Setup Required

None - no external service configuration required.

## Next Task Readiness
- All 4 likelihood model subclasses now use identical score() pattern through embed_and_cache()
- Embedding cache is populated on first call and reused on subsequent calls for any likelihood model

## Self-Check: PASSED

- FOUND: models/likelihoods.py
- FOUND: tests/test_likelihoods.py
- FOUND: 7-SUMMARY.md
- FOUND: commit e65b5cab (test RED)
- FOUND: commit ba773e72 (feat GREEN)
- embed_and_cache references in likelihoods.py: 13
- cosine_similarity references in likelihoods.py: 0

---
*Quick Task: 7-make-tf-idf-caching-real-in-score*
*Completed: 2026-03-13*
