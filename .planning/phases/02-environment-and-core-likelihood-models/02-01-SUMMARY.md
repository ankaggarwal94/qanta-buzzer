---
phase: 02-environment-and-core-likelihood-models
plan: 01
subsystem: models
tags: [numpy, belief-features, entropy, abc, embedding-cache, sha256]

# Dependency graph
requires:
  - phase: 01-data-pipeline-foundation
    provides: "MCQuestion dataclass with options, gold_index, option_profiles"
provides:
  - "extract_belief_features() producing (K+6) observation vectors"
  - "entropy_of_distribution() with numerical stability"
  - "LikelihoodModel ABC with score() and embed_and_cache()"
  - "_text_key() SHA-256 hashing for embedding cache"
affects: [02-02, 02-03, 02-04, 03-baseline-agents, 04-ppo-training]

# Tech tracking
tech-stack:
  added: []
  patterns: [abstract-base-class, embedding-cache-with-content-hashing]

key-files:
  created:
    - models/__init__.py
    - models/features.py
    - models/likelihoods.py
  modified: []

key-decisions:
  - "Ported qb-rl features.py exactly to maintain compatibility"
  - "LikelihoodModel ABC returns raw scores (environment applies softmax)"

patterns-established:
  - "Belief feature layout: [belief[K], top_p, margin, entropy, stability, progress, clue_idx_norm]"
  - "Embedding cache: SHA-256 content hash keys, float32 values"
  - "Python 3.11+ type hints: list[str], dict[str, np.ndarray] not List/Dict"

requirements-completed: [LIK-01]

# Metrics
duration: 2min
completed: 2026-02-25
---

# Phase 2 Plan 01: Belief Features and LikelihoodModel ABC Summary

**Belief feature extraction (K+6 vector) and abstract LikelihoodModel with SHA-256 embedding cache**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-26T02:18:29Z
- **Completed:** 2026-02-26T02:20:02Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Ported belief feature extraction producing (K+6)-dimensional observation vectors from probability distributions
- Created LikelihoodModel ABC with pluggable score/embed interface and content-addressed cache
- Established models/ package as the foundation for all likelihood and feature modules

## Task Commits

Each task was committed atomically:

1. **Task 1: Create models/ package and port belief features** - `65d5800` (feat)
2. **Task 2: Port LikelihoodModel ABC with embedding cache** - `508459a` (feat)

## Files Created/Modified
- `models/__init__.py` - Package init for models module
- `models/features.py` - entropy_of_distribution() and extract_belief_features() with 6 derived features
- `models/likelihoods.py` - LikelihoodModel ABC with score(), embed_and_cache(), _embed_batch(), and _text_key() helper

## Decisions Made
- Ported qb-rl reference implementations exactly to maintain compatibility with downstream plans
- LikelihoodModel.score() returns raw scores; softmax with temperature is applied by the environment (separation of concerns)
- Used Python 3.11+ native type hints (list, dict) per project conventions

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- models/features.py ready for use by TossupMCEnv (Plan 02-03) for observation construction
- models/likelihoods.py ABC ready for TfIdfLikelihood and SBERTLikelihood implementations (Plan 02-02)
- No blockers for downstream plans

## Self-Check: PASSED

- FOUND: models/__init__.py
- FOUND: models/features.py
- FOUND: models/likelihoods.py
- FOUND: commit 65d5800
- FOUND: commit 508459a

---
*Phase: 02-environment-and-core-likelihood-models*
*Completed: 2026-02-25*
