# Audit Remediation Checklist — COMPLETE (evidence-verified)

**Date:** 2026-03-14
**Branch:** main
**Scope:** Post-optimization truthfulness and consistency pass

---

## P0 — Correctness and Truthfulness

### P0-1: Calibration Bug — FIXED
- **Issue:** `calibration_at_buzz()` used `g_trace[buzz_step]` as "confidence". In baseline agents, `g_trace` is binary (0 or 1: whether argmax == gold), not a probability. Using binary values for ECE/Brier produced meaningless metrics.
- **Evidence:** `agents/threshold_buzzer.py:118`: `g_t = 1.0 if top_idx == question.gold_index else 0.0`. Same in bayesian_buzzer.py:66,134.
- **Fix:** Changed `calibration_at_buzz()` to use `top_p_trace` (belief model's probability for top answer) as confidence proxy. Falls back to `c_trace` when `top_p_trace` unavailable. Also fixed `evaluate_all.py` calibration plot and `compare_policies.py` both paths.
- **Files changed:** evaluation/metrics.py, scripts/evaluate_all.py, scripts/compare_policies.py, agents/ppo_buzzer.py
- **Tests added:** 4 tests in test_metrics.py, 1 in test_ppo_buzzer.py (test_ppo_calibration_uses_top_p_trace). PPOEpisodeTrace now includes top_p_trace = max(env.belief) at each step, making calibration consistent across all agent types.
- **Docs updated:** Module docstring in metrics.py, function docstring in calibration_at_buzz(), PPOEpisodeTrace docstring
- **Remaining risk:** None — the fix is backward-compatible (falls back to c_trace for trace formats without top_p_trace)

### P0-2: Split Reproducibility — FIXED
- **Issue:** `dataset_splits.py:70` used `hash(category)` which is randomized across Python processes (PYTHONHASHSEED).
- **Evidence:** `category_seed = seed + hash(category) % 1000000`
- **Fix:** Replaced `hash(category)` with `hashlib.md5(category.encode("utf-8")).hexdigest()` for deterministic cross-process behavior.
- **Files changed:** qb_data/dataset_splits.py
- **Tests added:** 4 tests in test_dataset_splits.py (new file) including cross-process determinism test with different PYTHONHASHSEED values
- **Remaining risk:** None — the MD5-based hash is platform-independent

### P0-3: Compare Policies Numerical Honesty — FIXED
- **Issue:** MLP path used config-driven env settings; T5 path hardcoded `wait_penalty=0.01`. MLP used g_trace for calibration; T5 used c_trace. Docstring overclaimed "identical metrics for a fair comparison."
- **Fix:** (a) Aligned calibration to use top_p_trace on both sides (from P0-1). (b) Fixed T5 wait_penalty to 0.1 (matching T5 pipeline's actual default). (c) Rewrote module docstring with honest caveats about comparison limitations. (d) Updated README comparison section.
- **Files changed:** scripts/compare_policies.py, README.md
- **Tests:** Comparison script is integration-level; calibration fix tested via P0-1 tests
- **Remaining risk:** S_q and reward comparisons remain qualitative (inherent to architecture differences)

### P0-4: CI Truthful and Robust — FIXED
- **Issue:** ci.sh needed robustness; potential root-level test collection.
- **Evidence:** ci.sh already scoped to `pytest tests/` (QT-10). All 260 tests pass. No `huggingface_hub` failures in current environment.
- **Fix:** (a) ci.sh now auto-activates .venv if present, or fails with actionable message. (b) Added `[tool.pytest.ini_options] testpaths = ["tests"]` to pyproject.toml so bare `pytest` also scopes correctly.
- **Files changed:** scripts/ci.sh, pyproject.toml
- **Remaining risk:** None — 260/260 tests pass

---

## P1 — Functional / Performance

### P1-5: Alias Control Re-scoring — VERIFIED CORRECT
- **Claimed issue:** alias control re-scores live instead of reusing precomputed
- **Verification:** alias_substitution_copy() changes option text and profiles. Since likelihood.score() depends on profile text, precomputed beliefs are invalid after substitution. Live re-scoring is genuinely necessary.
- **Code already documents this** via `evaluate_questions_live` vs `evaluate_questions_precomputed` pattern in evaluate_all.py.
- **No code change needed.**

### P1-6: SBERT Distractor Cache Bypass — VERIFIED, DOCUMENTED
- **Claimed issue:** mc_builder.py SBERT path calls SentenceTransformer directly, not through LikelihoodModel cache.
- **Verification:** This is a one-shot build step during MC dataset construction (Stage 1). It runs once and produces different embeddings (answer profiles for distractor ranking) than the runtime cache (clue-profile scoring). Sharing caches would add complexity for negligible benefit.
- **Fix:** Added clarifying comment in mc_builder.py.
- **Files changed:** qb_data/mc_builder.py (comment only)

### P1-7: Memory-Risk Cleanup — ADDRESSED
- **Issue:** TF-IDF embedding cache stores dense vocab-sized vectors; belief caches are unbounded.
- **Analysis:** (a) TF-IDF: ~120KB per entry at 30k vocab. For smoke (50q, ~250 texts): <30MB. For default (1000q): ~60MB. Manageable. (b) Precomputed beliefs: 10k entries × 16 bytes = ~160KB. Negligible. (c) SBERT/T5 cache: ~1.5-3MB for 1000 texts. Fine.
- **Fix:** Added `cache_memory_bytes` property to LikelihoodModel base class for runtime monitoring.
- **Files changed:** models/likelihoods.py
- **Tests added:** 2 tests in test_likelihoods.py::TestCacheMemory
- **Remaining risk:** For datasets >10k questions with TF-IDF, cache could exceed 500MB. Document this limit.

### P1-8: Legacy Root-Level Files — CLEANED
- **Issue:** 13 legacy .py files at repo root (pre-modularization prototypes).
- **Fix:** Moved to `_legacy/` directory. Updated README. pyproject.toml `testpaths` prevents collection.
- **Files moved:** config.py, dataset.py, demo.py, environment.py, main.py, metrics.py, model.py, test_csv_loader.py, test_imports.py, train_ppo.py, train_supervised.py, verify_data_loader.py, visualize.py
- **Kept at root:** generate_presentation.py, generate_poster.py, generate_dataflow_animation.py (active presentation scripts)

---

## P2 — Verification Gaps and Planning Drift

### P2-9: Post-Remediation Verification — COMPLETE
- **scripts/ci.sh:** 261/261 passed (75s)
- **scripts/manual-smoke.sh:** All 4 stages complete, 44 MC questions, evaluation report generated with corrected calibration (ECE=0.12, Brier=0.20)
- **T5 smoke:** supervised warm-start + 5 PPO iterations, test accuracy 62.5% (21s)
- **Reduced-scale default.yaml preflight:** 44 questions, default reward settings (time_penalty, wait_penalty=0.05, buzz_incorrect=-0.5), default MLP architecture [64,64], 500 PPO timesteps. top_p_trace confirmed present and used for calibration. Wall time: 4.5s.
- **Measured memory:** TF-IDF cache 1.87MB (44 questions), precomputed beliefs 3.5KB. Projected 42MB for 1000 questions.
- **Not verified:** Full 100k PPO training, SBERT/T5-large likelihood paths, sbert_profile distractor strategy — all require large model downloads
- **Timing:** CI ~75s, smoke pipeline ~10s, T5 smoke ~21s, preflight ~4.5s

### P2-10: Docs/Planning Sync — COMPLETE
- Updated AGENTS.md: test count 220→260, 13→15 test files
- Updated README.md: test count, test categories, comparison caveats, legacy section
- Updated STATE.md: known issues, last activity
- Downgraded claims: compare_policies is no longer "identical metrics for fair comparison"
- Added known issues: comparison limitations, TF-IDF memory scaling, unverified full-scale run

---

## Summary

| # | Issue | Status | Tests Added |
|---|-------|--------|-------------|
| P0-1 | Calibration bug | **FIXED** | 4 |
| P0-2 | Split reproducibility | **FIXED** | 4 |
| P0-3 | Compare policies honesty | **FIXED** | 0 (integration) |
| P0-4 | CI robustness | **FIXED** | 0 (config) |
| P1-5 | Alias re-scoring | **VERIFIED CORRECT** | 0 |
| P1-6 | SBERT cache bypass | **DOCUMENTED** | 0 |
| P1-7 | Memory risk | **ADDRESSED** | 2 |
| P1-8 | Legacy files | **CLEANED** | 0 |
| P2-9 | Verification | **COMPLETE** | 0 |
| P2-10 | Docs sync | **COMPLETE** | 0 |

**Total new tests: 11**
**Total tests in repo: 261**

### Files Changed (production)
- evaluation/metrics.py — calibration_at_buzz uses top_p_trace
- qb_data/dataset_splits.py — deterministic hash for category seeding
- scripts/evaluate_all.py — calibration plot uses top_p_trace
- scripts/compare_policies.py — aligned calibration, honest docstring, fixed wait_penalty
- scripts/ci.sh — auto-activate venv, actionable error message
- models/likelihoods.py — cache_memory_bytes property
- qb_data/mc_builder.py — documentation comment for SBERT path
- pyproject.toml — testpaths = ["tests"]
- README.md — test count, comparison caveats, legacy section
- AGENTS.md — test count
- .planning/STATE.md — known issues, last activity

### Files Changed (tests)
- tests/test_metrics.py — 4 new calibration tests
- tests/test_dataset_splits.py — 4 new split reproducibility tests (new file)
- tests/test_likelihoods.py — 2 new cache memory tests
- tests/test_ppo_buzzer.py — 1 new PPO calibration test + updated trace assertions

### Files Moved
- 13 legacy .py files from repo root to `_legacy/`

### Remaining Risks
1. Full 100k PPO training run (default.yaml) has not been verified end-to-end
2. SBERT/T5-large likelihood and sbert_profile distractor strategy require large model downloads — not exercised locally
3. compare_policies S_q/reward comparisons are qualitative across architectures
4. TF-IDF cache: measured 1.87MB for 44 questions, projected ~42MB for 1000. Not bounded in code.
5. Pre-existing config bug: `parse_overrides` in build_mc_dataset.py creates nested dicts that clobber parent config sections when merged (not introduced by this remediation)
