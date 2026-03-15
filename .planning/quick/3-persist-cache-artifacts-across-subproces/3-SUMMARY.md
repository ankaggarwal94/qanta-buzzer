---
phase: quick-3
plan: 01
subsystem: models/likelihoods + scripts pipeline
tags: [cache, persistence, embedding, performance]
dependency_graph:
  requires: []
  provides: [embedding-cache-persistence]
  affects: [run_baselines, train_ppo, evaluate_all]
tech_stack:
  added: []
  patterns: [numpy-npz-persistence, sha256-keyed-cache]
key_files:
  created: []
  modified:
    - models/likelihoods.py
    - scripts/_common.py
    - scripts/run_baselines.py
    - scripts/train_ppo.py
    - scripts/evaluate_all.py
    - tests/test_likelihoods.py
decisions:
  - "TfIdfLikelihood.save_cache is a no-op because TF-IDF vectors depend on fitted vocabulary"
  - "load_cache merges without overwriting existing keys (existing keys win)"
  - "save_cache creates parent directories and uses np.savez_compressed for disk efficiency"
metrics:
  duration: 7min
  completed: "2026-03-13T05:36:00Z"
  tasks_completed: 2
  tasks_total: 2
  tests_added: 5
  tests_total: 230
  files_modified: 6
---

# Quick Task 3: Persist Embedding Cache Across Subprocesses Summary

Disk persistence for LikelihoodModel embedding cache so transformer forward passes computed in stage 2 are reused by stages 3 and 4 via numpy .npz files.

## What Was Done

### Task 1: save_cache/load_cache on LikelihoodModel (TDD)

Added two methods to the `LikelihoodModel` base class:

- `save_cache(path) -> int`: Persists `embedding_cache` dict to disk as compressed `.npz`. Keys are SHA-256 hex strings, values are float32 arrays. Creates parent directories.
- `load_cache(path) -> int`: Restores entries from `.npz` without overwriting existing keys. Returns 0 silently when file does not exist (cold start).

`TfIdfLikelihood` overrides `save_cache()` as a no-op returning 0, because TF-IDF embeddings are vocabulary-specific and not portable across separate `fit()` calls.

5 tests added in `TestEmbeddingCachePersistence`:
1. `test_save_load_cache_round_trip` - SBERT round-trip fidelity
2. `test_load_cache_missing_file` - cold start behavior
3. `test_save_cache_empty` - valid .npz with zero arrays
4. `test_tfidf_save_cache_noop` - TF-IDF returns 0, no file written
5. `test_load_cache_does_not_overwrite` - existing keys preserved

### Task 2: Pipeline integration

Three helpers added to `scripts/_common.py`:
- `embedding_cache_path(config)` - resolves path from `config['likelihood']['cache_dir']`
- `load_embedding_cache(model, config)` - load if file exists
- `save_embedding_cache(model, config)` - persist to disk

Wiring:
- `run_baselines.py`: load before precompute, save after
- `train_ppo.py`: load before belief computation, save after
- `evaluate_all.py`: load on startup (no save needed in final stage)

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 (RED) | `d201f3c5` | Failing tests for cache persistence |
| 1 (GREEN) | `553c78a1` | Implement save_cache/load_cache on LikelihoodModel |
| 2 | `d17f0f3b` | Wire cache persistence into pipeline scripts |

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- `scripts/ci.sh`: 230 tests passed (5 new + 225 existing)
- Cold start: `load_cache` returns 0 when no file exists, pipeline proceeds normally
- TF-IDF safety: `save_cache` is a no-op, no spurious files created
- Round-trip fidelity: `np.testing.assert_array_equal` confirms bitwise identical arrays
- No new dependencies: uses only `numpy.savez_compressed` / `numpy.load` (built-in)

## Self-Check: PASSED

All 6 modified files exist, all 3 commits verified, all key functions present in expected locations.
