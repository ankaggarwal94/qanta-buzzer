# PR: Optimization Campaign, Three Extensions, PR #1 Reconciliation, and Three Review Rounds

**Branch:** `pr/final-sync-and-extensions` → `main`
**Squashed from:** 83 local commits
**422 files changed, +380,006 / -5,997 lines**
**365 tests pass, 4 skipped (3 optional extras + 1 E2E pipeline)**

---

## Motivation

This PR brings the `main` branch from a working-but-unoptimized v1.0 modular pipeline to a production-quality codebase with performance optimizations, three opt-in research extensions, factored action semantics from PR #1, and three rounds of adversarial code review fixes. Every change is backward-compatible: the existing smoke pipeline and T5 smoke path work identically when extensions are disabled.

---

## What's in this PR

### 1. Performance Optimization Campaign (8 quick tasks)

Seven ranked optimizations plus repo-contract scaffolding, each with equivalence tests proving behavior preservation:

| # | Optimization | Key technique |
|---|-------------|---------------|
| QT-1 | Repo-contract scaffolding | `AGENTS.md`, `scripts/ci.sh`, `scripts/manual-smoke.sh` |
| QT-2 | Precompute belief-observation trajectories | `qb_env/tossup_env.py:precompute_beliefs()` bypasses `likelihood_model.score()` during PPO training |
| QT-3 | Persist embedding cache across stages | `LikelihoodModel.save_cache()`/`load_cache()` via `.npz` |
| QT-4 | Collapse duplicate baseline sweeps | `_softmax_episode_from_precomputed()` — one belief pass, N threshold sweeps |
| QT-5 | Cache answer profiles | `AnswerProfileBuilder._cache` dict memoization with `(answer, exclude_qid)` key |
| QT-6 | Top-M argpartition distractor ranking | `np.argpartition` replacing full `np.argsort` in `mc_builder.py` |
| QT-7 | TF-IDF score() via embed_and_cache | L2-normalized dense vectors, dot product = cosine similarity |
| QT-8 | Precomputed shuffle control | Permute belief vectors instead of re-scoring |

### 2. Audit Remediation (10 issues, all closed)

Evidence-verified fixes for correctness, reproducibility, and truthfulness:

| Issue | Fix |
|-------|-----|
| **Calibration bug** | `calibration_at_buzz()` now uses `top_p_trace` (max belief probability), not binary `g_trace`. `PPOEpisodeTrace` gained `top_p_trace` field. |
| **Split reproducibility** | `dataset_splits.py` uses `hashlib.md5(category)` instead of `hash(category)` (immune to PYTHONHASHSEED). Cross-process determinism test added. |
| **Compare policies honesty** | Docstring no longer claims "identical metrics". T5 `wait_penalty` corrected from 0.01 to 0.1. |
| **CI robustness** | `ci.sh` auto-activates `.venv/`. `pyproject.toml` sets `testpaths = ["tests"]`. |
| **Config override clobbering** | `parse_overrides()` returns flat dotted keys (`{"data.K": 5}`) not nested dicts that replace sibling sections. |
| **Legacy root workflow** | Root-level prototype files still exist and remain non-canonical; cleanup deferred to a separate decision. |
| **Memory monitoring** | `LikelihoodModel.cache_memory_bytes` property added. Measured: 1.87 MB for 44 questions. |

### 3. Extension A: Expected Wins Reward Mode

Implements the QANTA Expected Wins scoring model where reward depends on beating an opponent's buzz timing.

**New files:**
- `qb_env/opponent_models.py` — `OpponentBuzzModel` protocol, `LogisticOpponentModel`, `EmpiricalHistogramOpponentModel`, `build_opponent_model_from_config()` factory
- `tests/test_opponent_models.py` — 11 tests (monotonicity, range, fallback, config factory)

**Modified files:**
- `qb_env/tossup_env.py` — new `expected_wins` reward mode: `R_t = S_t * V_self + (1 - S_t) * V_opp`
- `evaluation/metrics.py` — `expected_wins_score()` offline metric using continuous formula: `V_self_t = g_t * R_correct + (1 - g_t) * R_incorrect` (not binary branching)
- `scripts/evaluate_all.py` — EW summary in reports only when `reward_mode == expected_wins`
- `configs/default.yaml`, `configs/smoke.yaml` — `opponent_buzz_model` config section (disabled by default)

### 4. Extension B: Variable-K Answer Choices

Supports arbitrary numbers of answer options (2 to N) per question instead of fixed K=4.

**New files:**
- `tests/test_mc_builder_variable_k.py` — 5 tests (fixed-K unchanged, mixed-K yields variety, gold index valid)
- `tests/test_variable_k_integration.py` — 2 integration tests (build→env→baseline, text wrapper formatting)

**Modified files:**
- `qb_data/mc_builder.py` — `MCBuilder` gains `variable_K`/`min_K`/`max_K` with `_target_k()` per-question sampling
- `models/features.py` — `extract_padded_belief_features()` zero-pads belief to `max_K`
- `qb_env/tossup_env.py` — `variable_K`/`max_K` params, `action_masks()` method, padded-action rejection
- `agents/ppo_buzzer.py` — `use_maskable_ppo` flag for optional `MaskablePPO` (via `sb3-contrib`)
- `qb_env/text_wrapper.py` — already K-agnostic (verified with K=3 test)
- `configs/default.yaml` — `variable_K: false`, `min_K: 2`, `max_K: null`

### 5. Extension C: DSPy Integration

Optional LM-based scoring with offline prompt compilation via [DSPy](https://github.com/stanfordnlp/dspy).

**New files:**
- `models/dspy_likelihood.py` — `DSPyLikelihood(LikelihoodModel)` with real score-level cache (keyed by clue + options + program fingerprint), `.npz` persistence, `NotImplementedError` on embedding operations
- `qb_data/dspy_answer_profiles.py` — `build_dspy_profiles()` LM-augmented answer profiles with per-answer failure logging
- `scripts/optimize_dspy.py` — Offline compile workflow: `build_dspy_trainset()` (from train split, not combined dataset), `compile_dspy_scorer()` with argmax-based metric (not constant `lambda: 1.0`)
- `tests/test_dspy_likelihood.py` — 8 tests (score shape, cache hit, fingerprint-keyed invalidation, persistence roundtrip, isinstance check)
- `tests/test_dspy_optimize.py` — 5 tests (trainset structure, metric logic, mid-prefix selection)
- `tests/test_dspy_answer_profiles.py` — 3 tests (module importability, runtime ImportError, dspy-installed path)

**Modified files:**
- `models/likelihoods.py` — `build_likelihood_from_config()` dispatches `model_name == "dspy"` to `DSPyLikelihood`
- `pyproject.toml` — `[project.optional-dependencies]` gains `dspy = ["dspy>=2.5.0"]` and `maskable = ["sb3-contrib>=2.6.0"]`

### 6. PR #1 Reconciliation (Factored Action Semantics)

Selectively integrates the valuable parts of [PR #1](https://github.com/ankaggarwal94/qanta-buzzer/pull/1) while rejecting changes that would regress the calibration fix or break downstream consumers.

**Taken from PR #1:**
- `models/t5_policy.py` — `_joint_action_log_prob()` and `_joint_entropy()` (chain-rule: `H_wait + p_buzz * H_answer`). `select_action()` now only samples the answer distribution when `wait_action == 1`.
- `qb_env/stop_only_env.py` — `StopOnlyEnv` Discrete(2) wrapper mapping BUZZ to `argmax(belief)`
- `scripts/train_ppo.py` — `--policy-mode stop_only|flat_kplus1` flag (defaults to `flat_kplus1` to avoid compare_policies incompatibility)
- `qb_env/tossup_env.py` — `end_mode` (`force_commit`|`no_buzz`) and `no_buzz_reward` constructor args
- `training/hazard_pretrain.py` — `compute_survival_terms()` and `hazard_expected_nll_loss()` scaffold
- `tests/test_action_space_alignment.py` — 7 integration guards for factored semantics
- `tests/test_hazard_pretrain.py` — 3 hazard bridge tests

**Rejected from PR #1 (with rationale):**
- `g_trace` → `p_correct_trace` field rename — `@property` shim breaks `dataclasses.asdict()`, which breaks `compare_policies.py` serialization
- `calibration_at_buzz` rewrite — would revert the verified `top_p_trace` fix
- `stop_only` as default `--policy-mode` — produces 2-action checkpoints incompatible with compare_policies K+1 env
- `CanonicalEpisodeTrace` — unnecessary; existing `_to_dict()` handles all trace types
- `system_score()` param rename — breaks 9 call sites across the codebase

### 7. Three Rounds of Adversarial Code Review

Used ChatGPT 5.4 Pro with anti-hallucination prompts (requiring `file.py:LINE` citations for every claim). 17 verified issues found and fixed:

**Round 1 (7 issues):**
1. `DSPyLikelihood` didn't inherit `LikelihoodModel` — fixed
2. `score()` had no shape validation — added `ndim == 1` and `len == K` check
3. `dspy.enabled` config key existed but was never read — removed
4. Config comment listed non-existent `embedding_based` strategy — corrected
5. Module docstring falsely claimed dspy import required — rewritten
6. `test_changed_fingerprint_invalidates` was too weak — rewritten to test keys directly
7. `test_fallback_to_existing` name misleading — split into two accurate tests

**Round 2 (5 issues):**
1. Optimizer metric was constant `lambda: 1.0` — replaced with argmax-based `_score_metric`
2. Trainset loaded combined `mc_dataset.json` — now uses `train_dataset.json` to prevent data leakage
3. `build_dspy_profiles` docstring claimed leave-one-out it couldn't enforce — corrected
4. Silent `except Exception` in profile augmentation — added `logging.warning` per failure
5. `test_compile_requires_dspy` never called the function — added `test_score_metric_logic` and `test_trainset_uses_mid_prefix`

**Round 3 (5 issues from PR #1 reconciliation):**
DSPy compile path reconciled with train-split-aware workflow, hazard bridge integrated with guards.

---

## New Production Files (7)

| File | Purpose |
|------|---------|
| `qb_env/opponent_models.py` | OpponentBuzzModel protocol + logistic/empirical implementations |
| `qb_env/stop_only_env.py` | StopOnlyEnv: Discrete(2) WAIT/BUZZ wrapper |
| `models/dspy_likelihood.py` | DSPyLikelihood with score-level cache and persistence |
| `qb_data/dspy_answer_profiles.py` | Optional DSPy LM-augmented answer profiles |
| `scripts/optimize_dspy.py` | Offline DSPy compile/optimize workflow |
| `scripts/ci.sh` | Local CI entry point with venv auto-activation |
| `training/hazard_pretrain.py` | Hazard bridge loss utilities (scaffold) |

## New Test Files (11)

| File | Tests | Coverage |
|------|-------|----------|
| `test_opponent_models.py` | 11 | Logistic monotonicity, empirical CDF, config factory |
| `test_mc_builder_variable_k.py` | 5 | Fixed-K unchanged, mixed-K build, gold index validity |
| `test_variable_k_integration.py` | 2 | Build→env→baseline, text wrapper K=3 |
| `test_dspy_likelihood.py` | 8 | Score shape, cache hit/miss, fingerprint keys, persistence, isinstance |
| `test_dspy_optimize.py` | 5 | Trainset structure, metric logic, mid-prefix, cap |
| `test_dspy_answer_profiles.py` | 3 | Module importability, runtime ImportError, dspy path |
| `test_dataset_splits.py` | 4 | Same-process determinism, cross-process determinism, different seeds, all assigned |
| `test_answer_profile_cache.py` | 6 | Cache correctness for memoized profile builder |
| `test_mc_builder_topk.py` | 4 | Top-M ranking equivalence with full sort |
| `test_action_space_alignment.py` | 7 | Factored T5 semantics, StopOnlyEnv, flat K+1 ablation |
| `test_hazard_pretrain.py` | 3 | Survival terms sum-to-one, NLL loss behavior |

## Modified Test Files (10)

`test_agents.py`, `test_build_mc_dataset.py`, `test_environment.py`, `test_factories.py`, `test_features.py`, `test_likelihoods.py`, `test_metrics.py`, `test_ppo_buzzer.py`, `test_t5_policy.py`, `test_text_wrapper.py`

## Configuration Changes

New keys in `configs/default.yaml` (all opt-in, disabled by default):

```yaml
data:
  variable_K: false       # Enable variable-K per question
  min_K: 2
  max_K: null             # Defaults to K when null

environment:
  reward_mode: "time_penalty"   # Now also supports: expected_wins
  opponent_buzz_model:
    type: "none"                # none | logistic | empirical
  end_mode: "force_commit"      # force_commit | no_buzz
  no_buzz_reward: 0.0

dspy:
  model: "openai/gpt-4o-mini"
  optimizer: "BootstrapFewShot"
  cache_dir: "cache/dspy"
  max_examples: 50
```

New optional dependency extras in `pyproject.toml`:
```toml
[project.optional-dependencies]
openai = ["openai>=1.0.0"]        # existing
maskable = ["sb3-contrib>=2.6.0"] # new — MaskablePPO for variable-K
dspy = ["dspy>=2.5.0"]            # new — DSPy LM-based scoring
```

---

## Invariants Preserved

These were verified across every commit and review round:

- Smoke pipeline works unchanged: `build_mc_dataset --smoke` → `run_baselines --smoke` → `train_ppo --smoke` → `evaluate_all --smoke`
- T5 smoke path works unchanged: `train_t5_policy --smoke`
- Calibration uses `top_p_trace`, not binary `g_trace`
- Dataset splits use deterministic `hashlib.md5`, not Python `hash()`
- Alias control re-scores live (substitution changes option text)
- TF-IDF `save_cache()` is an intentional no-op (vocabulary-specific vectors)
- Config override merges are leaf-only (flat dotted keys)
- `g_trace` field preserved in all dataclasses (no `@property` shim)

## Test Plan

- [x] `pytest tests/` — **365 passed, 4 skipped** (current branch)
- [x] `bash scripts/manual-smoke.sh` — 4/4 stages complete (10.6s)
- [x] `python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke` — supervised 75% val acc → PPO 5 iters → test acc 62.5% (23.9s)
- [x] Reduced-scale default.yaml preflight — default reward settings, [64,64] MLP, 500 PPO timesteps
- [x] Default behavior unchanged when all extensions disabled

### Skipped (optional extras not installed locally)
- [ ] MaskablePPO integration (requires `sb3-contrib`)
- [ ] DSPy live compile (requires `dspy` + LM backend)
- [ ] Full 100k PPO training (intentionally out of scope)
- [ ] SBERT/T5-large likelihood paths (require large model downloads)

---

## Known Remaining Risks

1. Full 100k PPO training run not verified end-to-end
2. SBERT/T5-large likelihood paths not exercised locally
3. MaskablePPO path now has focused unit coverage (wrapper masks, masked inference, config plumbing), but no long-running integration training pass with `sb3-contrib`
4. DSPy compile/optimize requires live LM backend
5. `compare_policies` S_q/reward comparisons remain qualitative across architectures
6. TF-IDF cache memory grows with corpus size (~42 MB projected for 1000 questions)
7. `--hazard-pretrain` flag is scaffolded but the training loop is not wired up

## Supersedes

This PR supersedes [PR #1](https://github.com/ankaggarwal94/qanta-buzzer/pull/1) (factored action semantics). The useful parts of PR #1 have been reconciled into this branch with bug fixes applied. PR #1 can be closed.
