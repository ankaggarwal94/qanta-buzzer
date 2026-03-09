---
phase: 06-t5-policy-integration
plan: 01
subsystem: model
tags: [t5, transformers, policy-head, actor-critic, mean-pooling, action-decomposition]

# Dependency graph
requires:
  - phase: 02-environment-and-core-likelihood-models
    provides: "T5EncoderModel pattern, mean pooling, LikelihoodModel interface"
provides:
  - "T5PolicyModel with 3 custom heads (wait/answer/value)"
  - "PolicyHead class for actor-critic architecture"
  - "Action decomposition (0=WAIT, 1-K=SELECT) with log prob computation"
  - "Checkpoint save/load with T5 weights + policy_head.pt"
  - "18-test unit test suite for T5 policy"
affects: [06-02 supervised-training, 06-03 custom-ppo-comparison]

# Tech tracking
tech-stack:
  added: [T5EncoderModel, T5TokenizerFast, PolicyHead]
  patterns: [three-head-actor-critic, attention-masked-mean-pooling, action-decomposition, lazy-import]

key-files:
  created:
    - models/t5_policy.py
    - tests/test_t5_policy.py
  modified:
    - models/__init__.py

key-decisions:
  - "T5EncoderModel over T5ForConditionalGeneration (2x faster, 50% less memory)"
  - "T5TokenizerFast over T5Tokenizer (3-5x faster via Rust backend)"
  - "Lazy import in models/__init__.py to avoid loading transformers for belief-only usage"
  - "Module-scoped t5-small fixture for test efficiency (load once per file)"

patterns-established:
  - "Three-head actor-critic: independent wait, answer, and value heads on shared encoder"
  - "Action decomposition: combined actions 0=WAIT, 1-K=SELECT decomposed to wait+answer for independent log probs"
  - "Config-dict interface: T5PolicyModel accepts dict with model_name, device, max_input_length, num_choices"

requirements-completed: [STR-01]

# Metrics
duration: 5min
completed: 2026-02-26
---

# Phase 6 Plan 01: T5 Policy Architecture Summary

**T5EncoderModel with 3-head PolicyHead (wait/answer/value), action decomposition for PPO, and 18-test suite verified on t5-small**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-26T06:13:14Z
- **Completed:** 2026-02-26T06:19:08Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Ported PolicyHead with 3 independent heads from qanta-buzzer, using ReLU+Dropout(0.1) architecture
- Built T5PolicyModel using T5EncoderModel (not full T5) with attention-masked mean pooling
- Implemented complete action decomposition: select_action (sampling), get_action_log_probs (PPO), predict_answer (supervised)
- Save/load checkpoint I/O with T5 encoder weights + policy_head.pt
- 18 unit tests pass in <5 seconds using t5-small

## Task Commits

Each task was committed atomically:

1. **Task 1: Port PolicyHead class with 3 independent heads** - `9b7f11a` (feat)
2. **Task 2: Port T5PolicyModel with encoder + action decomposition** - `95564ab` (feat)
3. **Task 3: Create test suite for T5 policy architecture** - `9b7f11a` (test, created with Task 1)

## Files Created/Modified
- `models/t5_policy.py` (678 lines) - T5PolicyModel and PolicyHead classes with full docstrings
- `tests/test_t5_policy.py` (380 lines) - 18 unit tests covering all public methods
- `models/__init__.py` - Added lazy import for T5PolicyModel and PolicyHead

## Decisions Made
- **T5EncoderModel over T5ForConditionalGeneration**: Decoder is unused for policy learning; encoder-only model is 2x faster and uses 50% less memory
- **T5TokenizerFast over T5Tokenizer**: Rust-backed tokenizer is 3-5x faster, critical for PPO rollout collection with thousands of tokenizations
- **Lazy import in models/__init__.py**: Follows same pattern as PPOBuzzer in agents/__init__.py to keep package lightweight when only using belief features
- **Module-scoped t5-small fixture**: Load 60M-param model once per test file instead of per function, reducing test time from ~30s to <5s

## Deviations from Plan

None - plan executed exactly as written. The test file was co-created with the implementation (Task 1) rather than as a separate task, since both files needed to exist for verification.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required. T5 models are automatically downloaded from HuggingFace on first use.

## Next Phase Readiness
- T5PolicyModel ready for supervised warm-start training (Plan 06-02)
- All interfaces match TossupMCEnv action space (0=WAIT, 1-K=SELECT)
- Config dict interface compatible with YAML configuration system
- predict_answer method ready for cross-entropy supervised loss
- get_action_log_probs method ready for PPO updates (Plan 06-03)

## Self-Check: PASSED

- FOUND: models/t5_policy.py (678 lines, min 400)
- FOUND: tests/test_t5_policy.py (380 lines, min 150)
- FOUND: .planning/phases/06-t5-policy-integration/06-01-SUMMARY.md
- FOUND: commit 9b7f11a (Task 1 + Task 3)
- FOUND: commit 95564ab (Task 2)
- 18/18 tests passing in <5 seconds

---
*Phase: 06-t5-policy-integration*
*Completed: 2026-02-26*
