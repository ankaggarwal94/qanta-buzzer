# Concerns

## Legacy Root-Level Files

**Severity:** Medium
**Files:** `config.py`, `dataset.py`, `environment.py`, `model.py`, `main.py`, `train_supervised.py`, `train_ppo.py`, `metrics.py`, `visualize.py`, `demo.py`

The project has both a modular package structure (`qb_data/`, `qb_env/`, `models/`, `agents/`, `evaluation/`, `training/`, `scripts/`) and legacy root-level files from a pre-modularization phase. These legacy files are not part of the installed package (not listed in `pyproject.toml` packages) but still exist in the repo. They may cause confusion about which code is canonical.

**Recommendation:** Archive or remove root-level duplicates once the modular pipeline is fully validated.

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

The `LikelihoodModel` base class caches embeddings in an in-memory dict keyed by SHA-256 hash. For large datasets, this cache can consume significant memory. There is a `cache_dir` config option but no persistent disk-cache implementation in the base class — each process re-computes embeddings from scratch.

## PPO Trace Recording Workaround

**Severity:** Low
**Files:** `agents/ppo_buzzer.py`

SB3's `learn()` does not expose per-step action distributions. `PPOBuzzer.run_episode()` implements a custom episode loop to record `c_trace` and `g_trace` for S_q computation. This duplicates some environment-stepping logic and must be kept in sync with any environment changes.

## Hardcoded Path Patterns

**Severity:** Low
**Files:** `scripts/_common.py`, `scripts/build_mc_dataset.py`

`PROJECT_ROOT` is computed via `Path(__file__).resolve().parents[1]` and scripts add it to `sys.path`. This works for the current directory structure but assumes scripts are exactly one level deep. The `ARTIFACT_DIR` path is relative to project root.

## No CI / Linting Configuration

**Severity:** Low

No `.github/workflows/`, `tox.ini`, `pyproject.toml [tool.ruff]`, or pre-commit hooks. Code quality relies on manual review and CLAUDE.md conventions. This is expected for a course project but means no automated enforcement of coding standards.

## Test Coverage Gaps

**Severity:** Low
**Files:** `tests/`

Test coverage focuses on core abstractions (environment, likelihoods, features, agents) but does not cover:
- `evaluation/controls.py` (shuffle, choices-only, alias substitution experiments)
- `evaluation/plotting.py` (plot generation)
- Pipeline scripts end-to-end (partially covered by `--smoke` flag)
- Config validation edge cases in `qb_data/config.py`

## __pycache__ in Git Status

**Severity:** Cosmetic

Multiple `__pycache__/*.pyc` files appear in git status as modified. These should be in `.gitignore` to prevent noise.

**Recommendation:** Add `__pycache__/` and `*.pyc` to `.gitignore` if not already present.
