---
phase: 06-t5-policy-integration
plan: 02
subsystem: training
tags: [gymnasium-wrapper, text-observations, supervised-training, gradient-accumulation, t5-policy, cross-entropy]

# Dependency graph
requires:
  - phase: 06-t5-policy-integration
    provides: "T5PolicyModel with predict_answer method, PolicyHead, save/load"
  - phase: 02-environment-and-core-likelihood-models
    provides: "TossupMCEnv with belief features, MCQuestion dataclass"
provides:
  - "TextObservationWrapper converting beliefs to text for T5 input"
  - "SupervisedTrainer with gradient accumulation (4 steps, effective batch=32)"
  - "T5 policy configuration with t5-large defaults and smoke overrides"
  - "20-test suite covering wrapper and supervised training"
affects: [06-03 custom-ppo-comparison]

# Tech tracking
tech-stack:
  added: [TextObservationWrapper, SupervisedTrainer]
  patterns: [gymnasium-observation-wrapper, gradient-accumulation, best-model-selection, format-question-text]

key-files:
  created:
    - qb_env/text_wrapper.py
    - training/train_supervised_t5.py
    - training/__init__.py
    - configs/t5_policy.yaml
    - tests/test_text_wrapper.py
    - tests/test_supervised_t5.py
  modified:
    - qb_env/__init__.py

key-decisions:
  - "TextObservationWrapper uses cumulative_prefixes indexed by step_idx for accurate clue visibility"
  - "Supervised trainer scales loss by 1/grad_accum_steps for correct gradient magnitude"
  - "Best model saved by validation accuracy to checkpoints/supervised/best_model/"
  - "Config YAML separates model, supervised, ppo, and smoke sections for clear override"

patterns-established:
  - "Text format: 'CLUES: <tokens> | CHOICES: (1) opt1 (2) opt2 (3) opt3 (4) opt4'"
  - "format_question_text() shows ALL clues for supervised training (complete information)"
  - "Gradient accumulation: loss/N backward, clip, step every N batches, flush remainder"

requirements-completed: [STR-02]

# Metrics
duration: 7min
completed: 2026-02-26
---

# Phase 6 Plan 02: TextObservationWrapper and Supervised Training Summary

**Gymnasium text wrapper bridging belief-to-text observations, plus supervised warm-start trainer with 4-step gradient accumulation and 20-test verification suite**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-26T06:22:38Z
- **Completed:** 2026-02-26T06:29:43Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- Created TextObservationWrapper that converts TossupMCEnv numeric beliefs to text observations using cumulative_prefixes indexed by step_idx
- Ported SupervisedTrainer from qanta-buzzer with MCQuestion interface, gradient accumulation (loss/N scaling), and best-model tracking
- 20 tests passing: 8 for text wrapper (format, incremental clues, Gymnasium API, reward preservation) + 12 for supervised trainer (batch prep, training epoch, gradient accum, checkpoints, history)
- T5 policy config YAML with t5-large production defaults and t5-small smoke overrides

## Task Commits

Each task was committed atomically:

1. **Task 1: Create TextObservationWrapper for belief-to-text conversion** - `8a39446` (feat)
2. **Task 2: Port supervised trainer with gradient accumulation** - `2446719` (feat)
3. **Task 3: Create T5 policy config and test suite** - `aa8046f` (test)

## Files Created/Modified
- `qb_env/text_wrapper.py` (179 lines) - Gymnasium ObservationWrapper for text observations
- `training/train_supervised_t5.py` (626 lines) - SupervisedTrainer with gradient accumulation
- `training/__init__.py` (6 lines) - Training package init
- `configs/t5_policy.yaml` (56 lines) - T5 policy hyperparameters with smoke overrides
- `tests/test_text_wrapper.py` (247 lines) - 8 tests for text wrapper
- `tests/test_supervised_t5.py` (371 lines) - 12 tests for supervised trainer
- `qb_env/__init__.py` - Added TextObservationWrapper export

## Decisions Made
- **TextObservationWrapper uses cumulative_prefixes[step_idx-1]**: After N WAITs, beliefs are computed through cumulative_prefixes[0..N-1], so the text observation shows the prefix corresponding to the last processed clue
- **Loss scaled by 1/grad_accum_steps before backward**: Ensures correct gradient magnitude when accumulating over multiple batches (matching PyTorch's gradient accumulation best practice)
- **Remainder gradient flush**: After the main training loop, any accumulated but un-stepped gradients are flushed to avoid losing the last batches of an epoch
- **Config YAML with nested smoke section**: Clean override pattern where smoke.supervised overrides supervised defaults, rather than separate smoke config file

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required. T5 models are automatically downloaded from HuggingFace on first use.

## Next Phase Readiness
- TextObservationWrapper ready for PPO rollout collection with T5PolicyModel (Plan 06-03)
- SupervisedTrainer ready for warm-start before PPO fine-tuning
- Config system ready for both production (t5-large) and smoke (t5-small) runs
- All interfaces verified: wrapper wraps TossupMCEnv, trainer uses T5PolicyModel.predict_answer

## Self-Check: PASSED

- FOUND: qb_env/text_wrapper.py (179 lines, min 80)
- FOUND: training/train_supervised_t5.py (626 lines, min 250)
- FOUND: configs/t5_policy.yaml (contains model_name: t5-large)
- FOUND: tests/test_text_wrapper.py (247 lines, 8 tests, min 100)
- FOUND: tests/test_supervised_t5.py (371 lines, 12 tests)
- FOUND: commit 8a39446 (Task 1)
- FOUND: commit 2446719 (Task 2)
- FOUND: commit aa8046f (Task 3)
- 20/20 tests passing in <10 seconds

---
*Phase: 06-t5-policy-integration*
*Completed: 2026-02-26*
