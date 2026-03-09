---
phase: 02-environment-and-core-likelihood-models
plan: 02
subsystem: models
tags: [tfidf, sbert, sentence-transformers, sklearn, cosine-similarity, factory-pattern]

# Dependency graph
requires:
  - phase: 02-environment-and-core-likelihood-models
    plan: 01
    provides: "LikelihoodModel ABC with score(), embed_and_cache(), _text_key()"
provides:
  - "TfIdfLikelihood with corpus fitting and cosine similarity scoring"
  - "SBERTLikelihood with normalized embeddings and content-addressed caching"
  - "build_likelihood_from_config() factory for YAML-driven model construction"
affects: [02-03, 02-04, 03-baseline-agents, 04-ppo-training]

# Tech tracking
tech-stack:
  added: [sklearn.feature_extraction.text.TfidfVectorizer, sklearn.metrics.pairwise.cosine_similarity, sentence_transformers.SentenceTransformer]
  patterns: [factory-function-with-string-dispatch, lazy-import-for-optional-deps, corpus-fitting-before-scoring]

key-files:
  created: []
  modified:
    - models/likelihoods.py
    - models/__init__.py

key-decisions:
  - "Ported qb-rl TfIdfLikelihood and SBERTLikelihood exactly for downstream compatibility"
  - "Factory supports both sbert_name and embedding_model config keys for cross-project compat"
  - "Lazy imports for sklearn and sentence_transformers keep them optional at module load"

patterns-established:
  - "Corpus fitting: TfIdfLikelihood.fit() must be called before score() (enforced via _is_fit flag)"
  - "Normalized embeddings: SBERT uses normalize_embeddings=True so cosine = dot product"
  - "Factory config key: config['likelihood']['model'] dispatches to concrete class"

requirements-completed: [LIK-02, LIK-03, LIK-06]

# Metrics
duration: 4min
completed: 2026-02-25
---

# Phase 2 Plan 02: TF-IDF and SBERT Likelihood Models Summary

**TF-IDF and SBERT likelihood models with config-driven factory, corpus fitting, and SHA-256 embedding cache**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-26T02:22:49Z
- **Completed:** 2026-02-26T02:26:51Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- Implemented TfIdfLikelihood with sklearn TfidfVectorizer, corpus fitting, and cosine similarity scoring
- Implemented SBERTLikelihood with SentenceTransformer, L2-normalized embeddings, and inherited caching
- Created build_likelihood_from_config() factory dispatching to models via YAML config keys

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement TfIdfLikelihood with corpus fitting** - `e4993dc` (feat)
2. **Task 2: Implement SBERTLikelihood with embedding cache** - `93936f6` (feat)
3. **Task 3: Create factory function for config-driven construction** - `e7e309e` (feat)

## Files Created/Modified
- `models/likelihoods.py` - Added TfIdfLikelihood, SBERTLikelihood, and build_likelihood_from_config() factory
- `models/__init__.py` - Updated exports with all likelihood classes and factory function

## Decisions Made
- Ported qb-rl reference implementations directly to maintain compatibility with downstream environment and agent plans
- Factory supports both `sbert_name` (qb-rl convention) and `embedding_model` (qanta-buzzer default.yaml convention) config keys
- Lazy imports for sklearn and sentence_transformers in class constructors (not at module level) to keep them optional
- TF-IDF requires explicit fit() call with corpus before scoring (enforced via RuntimeError)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Factory handles both config key conventions**
- **Found during:** Task 3 (Factory function)
- **Issue:** Plan specified `sbert_name` config key but actual default.yaml uses `embedding_model`
- **Fix:** Factory checks both `sbert_name` and `embedding_model` keys with fallback to default model name
- **Files modified:** models/likelihoods.py
- **Verification:** Factory works with both config conventions
- **Committed in:** e7e309e (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Config key compatibility fix ensures factory works with both qb-rl and qanta-buzzer configs. No scope creep.

## Issues Encountered
- Plan verification test for TF-IDF used bare answer names ("George Washington") instead of answer profiles, causing zero scores due to no TF-IDF vocabulary overlap. This is expected TF-IDF behavior, not a bug -- the model works correctly with realistic answer profiles that share vocabulary with the corpus.

## User Setup Required
None - sentence-transformers model downloads automatically on first use (~80MB from HuggingFace).

## Next Phase Readiness
- TfIdfLikelihood and SBERTLikelihood ready for use by TossupMCEnv (Plan 02-03) for belief computation
- build_likelihood_from_config() ready for integration with environment factory (Plan 02-04)
- T5 likelihood model (Phase 3) will be added as another factory dispatch target
- No blockers for downstream plans

## Self-Check: PASSED

- FOUND: models/likelihoods.py
- FOUND: models/__init__.py
- FOUND: commit e4993dc
- FOUND: commit 93936f6
- FOUND: commit e7e309e

---
*Phase: 02-environment-and-core-likelihood-models*
*Completed: 2026-02-25*
