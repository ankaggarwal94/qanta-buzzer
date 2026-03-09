---
phase: 03-baseline-agents-and-t5-likelihood
plan: 02
subsystem: models
tags: [t5, transformers, embeddings, cosine-similarity, mean-pooling]

# Dependency graph
requires:
  - phase: 02-environment-and-core-likelihood-models
    provides: LikelihoodModel ABC, SBERTLikelihood pattern, build_likelihood_from_config factory
provides:
  - T5Likelihood class with mean-pooled T5 encoder embeddings
  - Factory support for model="t5" in build_likelihood_from_config
  - Package-level export of T5Likelihood from models
affects: [04-ppo-training-pipeline, 05-evaluation-framework, 06-t5-policy-integration]

# Tech tracking
tech-stack:
  added: [sentencepiece, protobuf]
  patterns: [T5EncoderModel mean-pooling with attention mask, GPU tensor detach for memory safety]

key-files:
  created: []
  modified:
    - models/likelihoods.py
    - models/__init__.py

key-decisions:
  - "Used T5EncoderModel (encoder-only) instead of T5ForConditionalGeneration for 2x faster inference and half memory"
  - "Used T5TokenizerFast instead of T5Tokenizer for faster tokenization"
  - "Installed sentencepiece and protobuf as required T5 tokenizer dependencies"

patterns-established:
  - "T5 mean pooling: attention_mask.unsqueeze(-1) * last_hidden_state, sum/mask_sum, L2 normalize"
  - "Lazy import pattern: torch and transformers imported inside __init__ and _embed_batch"

requirements-completed: [LIK-04, LIK-05]

# Metrics
duration: 2min
completed: 2026-02-26
---

# Phase 03 Plan 02: T5 Likelihood Model Summary

**T5Likelihood class using T5EncoderModel with attention-masked mean pooling for semantic similarity scoring, integrated into factory and package exports**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-26T03:17:28Z
- **Completed:** 2026-02-26T03:20:03Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- T5Likelihood class implements LikelihoodModel ABC with mean-pooled T5 encoder embeddings
- Factory function supports model="t5" with configurable t5_name (defaults to t5-base)
- Semantic discrimination verified: T5 scores "Washington" 0.62 vs "Einstein" 0.45 for "first president" clue
- Embedding caching works correctly via inherited embed_and_cache()

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement T5Likelihood class with mean-pooled embeddings** - `0b31090` (feat)
2. **Task 2: Update build_likelihood_from_config factory to support T5** - `6812961` (feat)
3. **Task 3: Update models/__init__.py to export T5Likelihood** - `46fa749` (chore)

## Files Created/Modified
- `models/likelihoods.py` - Added T5Likelihood class (140 lines) and T5 factory branch; file now 553 lines
- `models/__init__.py` - Added T5Likelihood to imports and __all__ list

## Decisions Made
- Used T5EncoderModel (encoder-only) instead of T5ForConditionalGeneration for 2x faster inference and half memory usage
- Used T5TokenizerFast instead of T5Tokenizer for faster tokenization
- Followed SBERTLikelihood pattern exactly: lazy imports, embed_and_cache in score(), raw similarity return

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing sentencepiece and protobuf dependencies**
- **Found during:** Task 1 (T5Likelihood implementation)
- **Issue:** T5Tokenizer requires sentencepiece which was not installed
- **Fix:** Ran `pip install sentencepiece protobuf`
- **Files modified:** None (runtime dependency only)
- **Verification:** T5TokenizerFast imports and tokenizes successfully
- **Committed in:** Not committed (pip install, not a code change)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary dependency installation. No scope creep.

## Issues Encountered
None - all tasks executed successfully after dependency installation.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- T5Likelihood is ready for use in TossupMCEnv via factory configuration
- Config key: `likelihood.model: t5` with optional `likelihood.t5_name: t5-base`
- Ready for Plan 03-03 (baseline agents) which may use T5 likelihood for belief computation
- Phase 4 PPO training can use T5 likelihood for semantic belief features

## Self-Check: PASSED

All files exist. All commits verified.

---
*Phase: 03-baseline-agents-and-t5-likelihood*
*Completed: 2026-02-26*
