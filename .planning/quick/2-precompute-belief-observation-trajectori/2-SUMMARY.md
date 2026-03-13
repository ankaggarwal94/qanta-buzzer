---
phase: quick
plan: 2
subsystem: environment
tags: [performance, precomputation, belief-cache]
dependency_graph:
  requires: []
  provides: [precomputed_beliefs, belief_cache_bypass]
  affects: [qb_env/tossup_env.py, scripts/train_ppo.py]
tech_stack:
  added: []
  patterns: [precomputation-cache, dict-keyed-bypass]
key_files:
  created: []
  modified:
    - qb_env/tossup_env.py
    - scripts/train_ppo.py
    - tests/test_environment.py
decisions:
  - Shared _softmax module-level helper to avoid code duplication between precompute_beliefs and _softmax_scores
  - Dict keyed by (question_index, step_idx) tuple for O(1) belief lookup
  - qid-to-index map in __init__ for random-sample reset path
  - .copy() on cached beliefs to prevent mutation of shared cache
metrics:
  duration: 4min
  completed: "2026-03-13T00:57:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 3
  tests_added: 5
  tests_total_after: 167
---

# Quick Task 2: Precompute Belief-Observation Trajectories Summary

Precomputed belief trajectories for all questions once before PPO training, eliminating redundant likelihood_model.score() calls during SB3 rollout collection via O(1) dict lookups.

## Changes

### qb_env/tossup_env.py

- Added module-level `_softmax(scores, beta)` helper extracted from `_softmax_scores`
- Added `precompute_beliefs()` function that iterates over all questions and steps, computing beliefs using the same logic as `_compute_belief`, and returns a `dict[(q_idx, step_idx), np.ndarray]` cache
- `TossupMCEnv.__init__` accepts optional `precomputed_beliefs` parameter (default `None`)
- `_compute_belief` checks `self.precomputed_beliefs` first; if present, returns a copy from cache without calling `likelihood_model.score()`
- `reset()` tracks `_current_question_idx` for both explicit `question_idx` option and random sampling paths
- `_softmax_scores` now delegates to shared `_softmax`
- `make_env_from_config` accepts and passes through `precomputed_beliefs`

### scripts/train_ppo.py

- After building the likelihood model, calls `precompute_beliefs()` to build belief cache
- Passes `precomputed_beliefs=belief_cache` to `make_env_from_config`
- Prints precomputation progress and cache size for training log visibility

### tests/test_environment.py

- Added `TestPrecomputedBeliefs` class with 5 tests:
  - `test_precomputed_matches_live_from_scratch`: Belief equivalence for from_scratch mode
  - `test_precomputed_matches_live_sequential_bayes`: Belief equivalence for sequential_bayes mode
  - `test_precomputed_skips_scoring`: Mock verifies `score()` is never called
  - `test_no_precomputed_backward_compat`: `precomputed_beliefs=None` behaves identically to default
  - `test_precompute_beliefs_helper_shape`: Cache keys, shapes, dtypes, and sum-to-one validation

## Deviations from Plan

### Auto-fixed Issues

None -- plan executed exactly as written.

## Pre-existing Issues (Out of Scope)

- SBERT/T5/OpenAI tests fail due to `huggingface_hub` version incompatibility (`is_offline_mode` import error). This is a pre-existing environment issue affecting 63 tests in the suite and is unrelated to this change.

## Verification

- 162 tests pass in `tests/` (excluding 63 pre-existing SBERT/T5/OpenAI import failures)
- 35 environment tests pass (30 existing + 5 new), 2 deselected (pre-existing SBERT)
- `from scripts.train_ppo import main` imports cleanly
- Zero regressions

## Self-Check: PASSED
