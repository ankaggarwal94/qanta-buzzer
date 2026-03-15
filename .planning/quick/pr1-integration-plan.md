# PR #1 Integration Plan: Factored Action Semantics

**PR:** https://github.com/ankaggarwal94/qanta-buzzer/pull/1
**PR branch:** `personal/codex/align-action-space-with-revised-report`
**PR commits:** `4f3e3009` (initial) + `f459f246` (fix)
**PR base:** `personal/main` (164 commits, ~72 commits behind local HEAD)
**Local HEAD:** `cbaa6f41` (220 commits)
**Status:** PR is Open with unresolved merge conflicts

---

## What PR #1 introduces

Seven features across 23 files (+530/-153 lines against its base):

### 1. Factored T5 action semantics
**Files:** `models/t5_policy.py`
**What:** Replaces independent wait+answer log-prob sum with mathematically correct joint factorization:
- `_joint_action_log_prob()`: P(WAIT)=log(p_wait), P(BUZZ_i)=log(p_buzz)+log(p_ans_i)
- `_joint_entropy()`: H(wait) + p_buzz * H(answer) (chain-rule entropy)
- `select_action()`: Only samples answer distribution when wait_action==1 (BUZZ)

### 2. StopOnlyEnv wrapper
**Files:** `qb_env/stop_only_env.py` (NEW), `qb_env/__init__.py`
**What:** Discrete(2) wrapper (0=WAIT, 1=BUZZ) that maps BUZZ to argmax(belief). Decouples the buzzing decision from answer selection.

### 3. `--policy-mode` CLI flag
**Files:** `scripts/train_ppo.py`
**What:** `--policy-mode stop_only|flat_kplus1` flag, defaults to `stop_only`. Wraps env with StopOnlyEnv when stop_only selected.

### 4. `end_mode` / `no_buzz_reward` env semantics
**Files:** `qb_env/tossup_env.py`, `configs/default.yaml`
**What:** New constructor args `end_mode` ("force_commit"|"no_buzz") and `no_buzz_reward`. In `no_buzz` mode, agent receives `no_buzz_reward` instead of being forced to answer at horizon.

### 5. `p_correct_trace` rename (from `g_trace`)
**Files:** `agents/ppo_buzzer.py`, `agents/threshold_buzzer.py`, `agents/bayesian_buzzer.py`, `evaluation/metrics.py`
**What:** Renames `g_trace` field to `p_correct_trace` with `@property` compatibility shim. Updates `system_score()` param name, `summarize_buzz_metrics()` lookup, `calibration_at_buzz()`.

### 6. Hazard pretraining bridge
**Files:** `training/hazard_pretrain.py` (NEW), `scripts/train_t5_policy.py`
**What:** `compute_survival_terms()` and `hazard_expected_nll_loss()` utilities. CLI flags `--hazard-pretrain`, `--beta-terminal`, `--freeze-answer-head` in train_t5_policy.py. **BUT**: the flag only prints a banner — no actual training loop is wired up.

### 7. Tests and docs
**Files:** `tests/test_action_space_alignment.py` (NEW), `tests/test_hazard_pretrain.py` (NEW), updates to `test_t5_policy.py`, `test_environment.py`, `test_metrics.py`, `test_ppo_buzzer.py`, `README.md`, `CLAUDE.md`, `walkthrough.md`

---

## Known bugs in PR #1 (from Codex/Copilot review)

These were found during review and are **unresolved** in the PR:

| Bug | Severity | Status |
|-----|----------|--------|
| `p_correct_trace` as `@property` breaks `asdict()` → downstream `compare_policies.py` reads empty `g_trace` → ECE/Brier silently become 0.0 | P1 | **Unfixed** |
| `--policy-mode stop_only` default produces 2-action checkpoints incompatible with `compare_policies.py` K+1 env | P1 | **Unfixed** |
| `no_buzz` end mode doesn't emit `forced_choice` → PPOBuzzer truncation handler falls back to argmax → synthetic correctness labels | P2 | Partially fixed in `f459f246` |
| `--hazard-pretrain` flag is a no-op (prints banner, no training loop) | P2 | **Unfixed** |
| `calibration_at_buzz` docstring still references `g_trace` while implementation uses `confidence_key` | Low | **Unfixed** |

---

## Conflict analysis with current local codebase

### Files that will conflict heavily

| PR file | Local state | Conflict type |
|---------|-------------|---------------|
| `agents/ppo_buzzer.py` | Has `top_p_trace` field (remediation), `use_maskable_ppo` (VK-05) | Field renames collide with field additions |
| `evaluation/metrics.py` | Has `expected_wins_score()`, corrected `calibration_at_buzz()` using `top_p_trace` | PR rewrites calibration to use `p_correct_trace`; local uses `top_p_trace`. Different confidence proxy semantics. |
| `qb_env/tossup_env.py` | Has `expected_wins` reward mode, `variable_K`, `max_K`, `action_masks()`, `opponent_buzz_model` | PR adds `end_mode`/`no_buzz_reward` to a constructor that has since grown 6+ new params |
| `models/t5_policy.py` | Largely unchanged from base | **Cleanest merge** — T5 math fixes apply directly |
| `scripts/train_ppo.py` | Has `precompute_beliefs` call, embedding cache load | PR adds `--policy-mode` flag + StopOnlyEnv wrapping |
| `configs/default.yaml` | Has EW, variable-K, DSPy sections | PR adds `end_mode`/`no_buzz_reward` — additive, low conflict |

### Files that merge cleanly

| PR file | Why |
|---------|-----|
| `qb_env/stop_only_env.py` | New file, no conflict |
| `training/hazard_pretrain.py` | New file, no conflict |
| `tests/test_action_space_alignment.py` | New file |
| `tests/test_hazard_pretrain.py` | New file |

### Files superseded by local work

| PR change | Local state | Decision |
|-----------|-------------|----------|
| `g_trace` → `p_correct_trace` rename | Local kept `g_trace` but added `top_p_trace` for calibration. Calibration uses `top_p_trace` (verified correct, evidence-tested). | **Do not adopt PR rename.** The `p_correct_trace` name is more descriptive, but the `@property` shim breaks `asdict()` and the PR's calibration change reverts the `top_p_trace` fix. |
| `calibration_at_buzz` rewrite | Local version uses `top_p_trace` with `c_trace` fallback (3 review rounds verified). PR version uses `p_correct_trace` with `g_trace` fallback. | **Keep local version.** PR's approach was flagged as breaking `compare_policies.py`. |

---

## Recommended integration patches

Break the PR into **4 independent, cherry-pickable patches**, ordered by value and conflict risk:

### PATCH A: T5 joint action semantics (HIGH VALUE, LOW CONFLICT)

**Source:** `models/t5_policy.py` changes from PR
**What to take:**
- `_joint_action_log_prob()` method
- `_joint_entropy()` method (chain-rule: H_wait + p_buzz * H_answer)
- Updated `select_action()` that only samples answer when buzzing
- Updated `get_action_log_probs()` using the joint methods

**Local conflicts:** Minimal — `models/t5_policy.py` hasn't changed much since the PR base.
**Tests:** Port `test_t5_policy.py` additions (joint log-prob tests, chain-rule entropy tests).
**Risk:** Low. This is a pure math fix that makes T5 policy training more principled.

### PATCH B: StopOnlyEnv + `--policy-mode` flag (MEDIUM VALUE, LOW CONFLICT)

**Source:** `qb_env/stop_only_env.py` (new file), `scripts/train_ppo.py` changes
**What to take:**
- `StopOnlyEnv` wrapper (clean new file)
- `--policy-mode` CLI arg in `train_ppo.py`
- Export from `qb_env/__init__.py`

**What to change vs PR:**
- Default should be `flat_kplus1` (not `stop_only`) to avoid breaking `compare_policies.py`
- Must handle `use_maskable_ppo` interaction (StopOnlyEnv is Discrete(2), masking is irrelevant)
- `run_episode()` needs the stop-only branch but must keep `top_p_trace` (not adopt `p_correct_trace`)

**Tests:** Port `test_environment.py` StopOnlyEnv tests, `test_ppo_buzzer.py` policy-mode test.
**Risk:** Medium. The P1 bug (compare_policies incompatibility) must be addressed by NOT defaulting to stop_only.

### PATCH C: `end_mode` / `no_buzz_reward` (LOW VALUE, MEDIUM CONFLICT)

**Source:** `qb_env/tossup_env.py` constructor + step changes, `configs/default.yaml`
**What to take:**
- `end_mode` and `no_buzz_reward` constructor args
- `no_buzz` terminal branch in step()
- Config defaults (`end_mode: force_commit`, `no_buzz_reward: 0.0`)

**What to change vs PR:**
- Constructor signature must merge with the 6+ new params (EW, variable-K, etc.)
- `no_buzz` branch must emit `no_buzz: True` in info AND set `forced_choice: -1` (PR bug fix)
- `make_env_from_config()` factory must read new config keys

**Tests:** Port `test_environment.py` no_buzz tests.
**Risk:** Medium. The P2 bug (no forced_choice marker) was partially fixed but needs further hardening.

### PATCH D: Hazard pretraining bridge (LOW VALUE, LOW CONFLICT)

**Source:** `training/hazard_pretrain.py` (new file), `scripts/train_t5_policy.py` CLI flags
**What to take:**
- `compute_survival_terms()` and `hazard_expected_nll_loss()` (the math is sound)
- CLI flag definitions

**What NOT to take:**
- The no-op banner-only implementation. Either wire up a real training loop or mark the flag as `--hazard-pretrain` with a `NotImplementedError` until the loop exists.

**Tests:** Port `test_hazard_pretrain.py` (3 tests).
**Risk:** Low. New file with no downstream callers until the training loop is wired.

### Cross-cutting: `test_action_space_alignment.py`

The PR adds a 7-test integration file (`tests/test_action_space_alignment.py`) that spans patches A, B, and C. These should be selectively ported after the relevant patches land:
- T5 log-prob/entropy tests → after PATCH A
- StopOnlyEnv discrete-2 and flat-kplus1 tests → after PATCH B
- no_buzz end-mode test → after PATCH C
- `p_correct_trace` metric test → **skip** (we keep `g_trace` + `top_p_trace`)
- `stop_only` default test → **skip** (we default to `flat_kplus1`)

---

## What to explicitly NOT take from PR #1

| Item | Reason |
|------|--------|
| `g_trace` → `p_correct_trace` field rename | Breaks `asdict()`, regresses calibration fix, conflicts with `top_p_trace` |
| `CanonicalEpisodeTrace` dataclass in `metrics.py` | Unnecessary; existing trace dataclasses work fine with `_to_dict()` |
| `calibration_at_buzz` rewrite with `confidence_key` param | Reverts the `top_p_trace` fix that was verified across 3 review rounds |
| `--policy-mode stop_only` as default | Breaks `compare_policies.py` (P1 bug, acknowledged by PR author) |
| `system_score()` param rename to `p_correct_trace` | Breaks all existing callers; the name `g_trace` is used consistently across the codebase and in S_q literature |
| Softened controls language | Subjective wording changes with no behavioral impact |

---

## Implementation order

1. **PATCH A** first (pure T5 math, no dependencies)
2. **PATCH B** second (needs PATCH A for consistent T5 factorization context)
3. **PATCH C** third (env changes, independent of A/B but benefits from B's StopOnlyEnv)
4. **PATCH D** last (standalone, can be deferred indefinitely)

## Estimated effort

| Patch | Conflict resolution | New code | Tests to port | Total |
|-------|-------------------|----------|---------------|-------|
| A | ~30 min | 0 | 3-5 tests | ~1 hr |
| B | ~1 hr | StopOnlyEnv (clean) | 4-6 tests | ~2 hr |
| C | ~1 hr | end_mode plumbing | 3-4 tests | ~2 hr |
| D | ~15 min | 0 | 3 tests | ~30 min |
| **Total** | | | | **~5.5 hr** |

---

## Pre-integration checklist

Before starting any patch:
- [ ] Fetch latest remote: `git fetch personal`
- [ ] Verify local tests green: `pytest tests/ -q` (320 passed, 3 skipped)
- [ ] Verify smoke green: `bash scripts/manual-smoke.sh`
- [ ] Create integration branch: `git checkout -b integrate-pr1-factored-actions`

After each patch:
- [ ] Run `pytest tests/ -q`
- [ ] Run `bash scripts/manual-smoke.sh`
- [ ] Verify `top_p_trace` calibration invariant still holds
- [ ] Commit atomically
