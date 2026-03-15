# Concerns

## Legacy Root-Level Files

**Severity:** Resolved
**Files:** `_legacy/config.py`, `_legacy/dataset.py`, etc.

Legacy root-level files have been moved to `_legacy/`. They are not part of the installed package and `pyproject.toml` sets `testpaths = ["tests"]` to prevent pytest from collecting them.

## Dual Data Paths

**Severity:** Low
**Files:** `qb_data/data_loader.py`, `qb_data/huggingface_loader.py`, `qb_env/data_loader.py`

Two data loading strategies (local CSV and HuggingFace) with different field-name mappings and parsing logic. The `qb_env/data_loader.py` re-export shim adds a third import path. While the shims are thin, three ways to load the same data increases cognitive overhead.

## Anti-Artifact Guard Complexity

**Severity:** Low
**Files:** `qb_data/mc_builder.py`

`MCBuilder` implements four guard layers (alias collision, duplicate overlap, length ratio, question overlap) with configurable thresholds. These guards are critical for dataset quality but add complexity. If guard thresholds are misconfigured, questions may be silently dropped or distractor pools exhausted, falling back to random selection.

## Embedding Model Downloads

**Severity:** Low
**Files:** `models/likelihoods.py`, `tests/conftest.py`

Tests and pipeline scripts download models from HuggingFace on first run:
- `t5-small` (~240MB), `t5-base` (~890MB), `t5-large` (~2.8GB)
- `all-MiniLM-L6-v2` (~90MB)

No offline fallback exists. First-run tests require network access and may be slow. The `sample_t5_model` test fixture mitigates by using `t5-small` and module-scoped loading.

## In-Memory Embedding Cache

**Severity:** Low
**Files:** `models/likelihoods.py`

The `LikelihoodModel` base class caches embeddings in an in-memory dict keyed by SHA-256 hash. `save_cache()` / `load_cache()` persist SBERT/T5 caches to `.npz` files across pipeline stages. `TfIdfLikelihood` intentionally no-ops on `save_cache()` because its dense vectors are vocabulary-specific. The `cache_memory_bytes` property reports current cache size. Measured: ~1.9 MB for 44 questions, projected ~42 MB for 1000 questions.

## PPO Trace Recording Workaround

**Severity:** Low
**Files:** `agents/ppo_buzzer.py`

SB3's `learn()` does not expose per-step action distributions. `PPOBuzzer.run_episode()` implements a custom episode loop to record `c_trace`, `g_trace`, and `top_p_trace` for S_q and calibration computation. This duplicates some environment-stepping logic and must be kept in sync with any environment changes.

## Hardcoded Path Patterns

**Severity:** Low
**Files:** `scripts/_common.py`, `scripts/build_mc_dataset.py`

`PROJECT_ROOT` is computed via `Path(__file__).resolve().parents[1]` and scripts add it to `sys.path`. This works for the current directory structure but assumes scripts are exactly one level deep. The `ARTIFACT_DIR` path is relative to project root.

## No CI / Linting Configuration

**Severity:** Low (partially resolved)

No `.github/workflows/`, `tox.ini`, or pre-commit hooks. However, `scripts/ci.sh` provides a local CI entry point that auto-activates the project venv and runs the full test suite. `pyproject.toml` sets `testpaths = ["tests"]` to scope pytest correctly.

## Test Coverage Gaps

**Severity:** Low
**Files:** `tests/`

342 tests (3 skipped for optional extras) cover core abstractions, extensions, and optimizations. Remaining gaps:
- `evaluation/plotting.py` (plot generation — visual output only)
- Pipeline scripts end-to-end (partially covered by `--smoke` flag)
- Config validation edge cases in `qb_data/config.py`
- `evaluation/controls.py` choices-only and alias substitution controls (shuffle precomputed control has equivalence tests)
- DSPy live compile with real LM backend (only stubbed in tests)

## DSPy Integration Caveats

**Severity:** Low
**Files:** `models/dspy_likelihood.py`, `models/likelihoods.py`

`DSPyLikelihood` inherits `LikelihoodModel` but raises `NotImplementedError` from `embed_and_cache()` and `_embed_batch()`. Callers that use the generic `LikelihoodModel` interface for embedding-based operations (e.g. `precompute_embeddings()`) will fail at runtime if given a DSPy model. The factory returns a placeholder scorer by default which produces uniform scores — real scoring requires a compiled DSPy program. The `dspy` optional extra must be installed for the compile workflow.

## __pycache__ in Git Status

**Severity:** Cosmetic

Multiple `__pycache__/*.pyc` files appear in git status as modified. These should be in `.gitignore` to prevent noise.

**Recommendation:** Add `__pycache__/` and `*.pyc` to `.gitignore` if not already present.
