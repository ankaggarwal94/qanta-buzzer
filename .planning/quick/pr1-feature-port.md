# PR1-FEATURE-PORT — Substantive Upstream PR #1 Features

**Created:** 2026-03-15
**Branch:** main
**Upstream PR:** https://github.com/ankaggarwal94/qanta-buzzer/pull/1
**Upstream branch:** personal/codex/align-action-space-with-revised-report
**Prerequisite:** PR #1 DSPy reconciliation (WP1-4) closed — see pr1-reconciliation.md
**Source recommendation:** ChatGPT Pro 2026-03-15, CoVe-validated by Cursor/Claude

---

## CoVe validation summary

The ChatGPT Pro recommendation was validated against the actual repo state.
Corrections applied per checkpoint:

| Claim | Verdict | Correction |
|-------|---------|------------|
| Patch A targets `training/train_supervised_t5.py` | **FALSE** — supervised trainer does not call `select_action` or `get_action_log_probs` | Removed from Patch A target list |
| Patch A targets `tests/test_train_ppo_t5.py` | **FALSE** — file does not exist | Corrected to `tests/test_ppo_t5.py` |
| Patch B targets `training/train_supervised_t5.py` | **FALSE** — supervised training does not use envs | Removed from Patch B target list |
| Patch B targets `training/train_ppo_t5.py` | **FALSE** — StopOnlyEnv is wired in `scripts/train_ppo.py`, not the trainer | Corrected to `scripts/train_ppo.py` |
| Patch D targets `training/train_supervised_t5.py`, `training/train_ppo_t5.py` | **FALSE** — upstream adds a new file and CLI flags only; does not modify either trainer | Corrected to `training/hazard_pretrain.py` (new) + `scripts/train_t5_policy.py` |
| `_joint_action_log_prob` / `_joint_entropy` don't exist yet | **VERIFIED** — current `get_action_log_probs` uses independent sums | — |
| `qb_env/stop_only_env.py` doesn't exist yet | **VERIFIED** | — |
| `end_mode` / `no_buzz_reward` not in env or config | **VERIFIED** | — |
| `training/hazard_pretrain.py` doesn't exist yet | **VERIFIED** | — |

---

## Explicit exclusions

| Item | Reason |
|------|--------|
| `g_trace` → `p_correct_trace` field rename | Breaks `asdict()`, regresses `top_p_trace` calibration fix |
| Upstream calibration rewrite | Reverts verified `top_p_trace` fix (3 review rounds) |
| `--policy-mode stop_only` as default | Breaks `compare_policies.py` (P1 bug acknowledged in PR review) |
| `system_score()` param rename | Breaks all existing callers; `g_trace` is standard in S_q literature |
| Root-level upstream structure | Local modular layout is canonical |
| `_legacy/` changes | Remains non-canonical |

---

## Patch A — T5 Joint Action Factorization

| Field | Value |
|-------|-------|
| Objective | Port factored `_joint_action_log_prob()`, `_joint_entropy()`, updated `select_action()` and `get_action_log_probs()` from upstream PR into local `models/t5_policy.py` |
| Upstream feature | Mathematically correct joint factorization: P(WAIT)=log(p_wait), P(BUZZ_i)=log(p_buzz)+log(p_ans_i), chain-rule entropy H(wait)+p_buzz*H(answer), answer sampled only when buzzing |
| Current state | `get_action_log_probs` formerly used independent sum of wait and answer log-probs. Patch A adds `_joint_action_log_prob` and `_joint_entropy` and makes answer sampling conditional on buzzing. |
| Target files | `models/t5_policy.py`, `training/train_ppo_t5.py` (calls `select_action` at L362/651, `get_action_log_probs` at L543), `tests/test_t5_policy.py`, `tests/test_ppo_t5.py` |
| NOT touched | `training/train_supervised_t5.py` (does not call select_action/get_action_log_probs), MLP/belief-feature PPO path, `p_correct_trace` |
| Acceptance criteria | Factorized semantics in code; T5 smoke still passes; action/log-prob/value internally consistent; checkpoint loading backward-compatible or fails clearly |
| Verification | `pytest tests/test_t5_policy.py tests/test_ppo_t5.py -q` then `python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke` |
| Result | completed |
| Commit hash | not committed |
| Rollback command | — |
| Notes | Highest-value deferred item per reconciliation transcript. Added test-first coverage for joint WAIT/BUZZ log-probs, chain-rule entropy, and skipping answer sampling on all-WAIT batches. Verified with `pytest tests/test_t5_policy.py tests/test_ppo_t5.py -q` (35 passed) and `python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke` (passed). |

---

## Patch B — StopOnlyEnv + `--policy-mode` Flag

| Field | Value |
|-------|-------|
| Objective | Add `StopOnlyEnv` wrapper (Discrete(2) buzz/wait mapping BUZZ to argmax(belief)), `--policy-mode` CLI flag in `scripts/train_ppo.py` defaulting to `flat_kplus1` |
| Upstream feature | 37-line `StopOnlyEnv` gym.Wrapper in `qb_env/stop_only_env.py`, `--policy-mode` arg in train_ppo.py |
| Current state | `qb_env/stop_only_env.py` did not exist and `scripts/train_ppo.py` had no `--policy-mode` flag. `agents/ppo_buzzer.py` also assumed flat K+1 action traces and crashed/mis-labeled stop-only episodes. |
| Target files | `qb_env/stop_only_env.py` (new), `qb_env/__init__.py` (export), `scripts/train_ppo.py` (CLI flag + wrapping), `agents/ppo_buzzer.py` (2-action trace compatibility), `tests/test_environment.py`, `tests/test_ppo_buzzer.py` |
| NOT touched | `training/train_supervised_t5.py` (no env usage), `training/train_ppo_t5.py` (T5 PPO uses its own rollout loop, not SB3) |
| Local adaptation | Default MUST be `flat_kplus1` (not `stop_only`) to preserve `compare_policies.py` compatibility. Must handle `use_maskable_ppo` interaction (StopOnlyEnv is Discrete(2), masking irrelevant). Must keep `top_p_trace` (not adopt `p_correct_trace`). |
| Acceptance criteria | StopOnlyEnv constructible from config; existing full-policy path unchanged by default; tests prove stop-only does not mutate answer-head semantics |
| Verification | `pytest tests/test_environment.py tests/test_ppo_buzzer.py -q` then `bash scripts/manual-smoke.sh` (default path unchanged) |
| Result | completed |
| Commit hash | not committed |
| Rollback command | — |
| Notes | Depends on Patch A for consistent T5 factorization context. Added the minimal local adaptation in `agents/ppo_buzzer.py` so explicit stop-only runs use env-selected `chosen_idx` and belief-based `g_trace` instead of treating every buzz as `action - 1`. Verified with `pytest tests/test_environment.py tests/test_ppo_buzzer.py -q` (70 passed, 1 skipped) and default-path smoke pipeline (passed). |

---

## Patch C — `end_mode` / `no_buzz_reward` Env Semantics

| Field | Value |
|-------|-------|
| Objective | Add `end_mode` ("force_commit" / "no_buzz") and `no_buzz_reward` constructor args to `TossupMCEnv`, wire into `step()` terminal branch and config factory |
| Upstream feature | Two new constructor params, branching `if/elif/else` in terminal handler, `make_env_from_config()` reads new config keys, defaults to current behavior (`force_commit`) |
| Current state | `end_mode` and `no_buzz_reward` were absent from `tossup_env.py`, `make_env_from_config()`, and `configs/default.yaml`. Terminal handler unconditionally forced a final argmax commit. |
| Target files | `qb_env/tossup_env.py` (constructor + step + factory), `configs/default.yaml` (new keys with safe defaults), `tests/test_environment.py`, `tests/test_factories.py` |
| NOT touched | `configs/smoke.yaml` (unless explicitly needed), existing reward modes, Expected Wins extension |
| Local adaptation | Constructor must merge with existing 6+ params (EW, variable-K, etc.). `no_buzz` branch must emit `forced_choice: -1` and `forced_correct: False` in info dict (upstream P2 bug was partially fixed in commit f459f246). Must verify interaction with Expected Wins — either support or block with clear error. |
| Acceptance criteria | `end_mode` and `no_buzz_reward` are opt-in only; default reward behavior unchanged; no-buzz terminal semantics explicit and tested |
| Verification | `pytest tests/test_environment.py tests/test_factories.py -q` then `bash scripts/manual-smoke.sh` |
| Result | completed |
| Commit hash | not committed |
| Rollback command | — |
| Notes | Independent of A/B but benefits from B's StopOnlyEnv context for integration testing. Added opt-in `end_mode`/`no_buzz_reward` with safe defaults, `forced_choice = -1` / `forced_correct = False` markers for no-buzz episodes, and factory/config plumbing. Verified with `pytest tests/test_environment.py tests/test_factories.py -q` (68 passed) and default-path smoke pipeline (passed). |

---

## Patch D — Hazard Pretraining Bridge

| Field | Value |
|-------|-------|
| Objective | Add `training/hazard_pretrain.py` utilities (`compute_survival_terms`, `hazard_expected_nll_loss`), CLI flags in `scripts/train_t5_policy.py`, with `NotImplementedError` for the unimplemented training loop |
| Upstream feature | 40-line `training/hazard_pretrain.py` with `HazardBatchOutput` dataclass and two functions. CLI flags `--hazard-pretrain`, `--beta-terminal`, `--freeze-answer-head` in train_t5_policy.py (no-op banner in upstream). |
| Current state | `training/hazard_pretrain.py` did not exist and `scripts/train_t5_policy.py` had no hazard flags or fail-fast guard. |
| Target files | `training/hazard_pretrain.py` (new), `scripts/train_t5_policy.py` (CLI flags), `tests/test_hazard_pretrain.py` (new) |
| NOT touched | `training/train_supervised_t5.py`, `training/train_ppo_t5.py` — upstream PR does not modify these for hazard pretrain |
| Local adaptation | Must raise `NotImplementedError` instead of silently printing a banner (upstream P2 bug: no-op flag). Math utilities are sound and can be ported directly. |
| Acceptance criteria | Hazard utilities exist and are testable; CLI flag raises NotImplementedError clearly; existing T5 smoke unchanged |
| Verification | `pytest tests/test_hazard_pretrain.py -q` then `python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke` |
| Result | completed |
| Commit hash | not committed |
| Rollback command | — |
| Notes | Standalone new files with no downstream callers until training loop is wired. Added the upstream math utilities directly and a local `validate_args()` guard that raises `NotImplementedError` for `--hazard-pretrain` instead of silently printing a no-op banner. Verified with `pytest tests/test_hazard_pretrain.py -q` (3 passed) and default T5 smoke (passed). |

---

## Patch X — Integration Tests + Exclusion Docs

| Field | Value |
|-------|-------|
| Objective | Add integration tests proving the ported subset works; document exclusions in `.planning/` |
| Target files | Integration test file(s) spanning A/B/C, `.planning/quick/pr1-feature-port.md` (this file), `AGENTS.md` or `README.md` if public contract changed |
| Subset from upstream | Port applicable tests from `tests/test_action_space_alignment.py` (7 tests in upstream). Skip `p_correct_trace` metric test and `stop_only` default test. |
| Must document | `p_correct_trace` rename NOT ported; upstream calibration rewrite NOT ported; local confidence field remains `top_p_trace` |
| Result | completed |
| Commit hash | not committed |
| Rollback command | — |
| Notes | Added `tests/test_action_space_alignment.py` for the ported subset only (factorized T5 WAIT semantics, chain-rule entropy, StopOnlyEnv surface, flat-K+1 availability, no-buzz markers). Updated `README.md` and `AGENTS.md` to document factorized T5 semantics, optional stop-only PPO, `end_mode` / `no_buzz_reward`, and the hazard bridge’s current fail-fast status. Final review also surfaced two real edge cases that were fixed before closure: `StopOnlyEnv` now rejects invalid actions, and `PPOBuzzer.run_episode()` preserves `buzz_step == -1` for true `no_buzz` truncations. |

---

## Execution order

1. **Patch A** — T5 joint action factorization (no dependencies)
2. **Patch B** — StopOnlyEnv (benefits from A)
3. **Patch C** — end_mode / no_buzz_reward (independent of A/B, but benefits from stability)
4. **Patch D** — Hazard pretraining bridge (standalone)
5. **Patch X** — Integration tests + docs closure (depends on A/B/C)

## Series-level validation gates

After every patch: targeted tests for that patch.
Final gate:
- `pytest -q` → 342 passed, 3 skipped
- `bash scripts/manual-smoke.sh` → passed
- `python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke` → passed

## Invariants (must hold after every patch)

1. `pytest -q` → 342 passed, 3 skipped
2. `bash scripts/manual-smoke.sh` → 4/4 stages complete
3. `top_p_trace` calibration invariant preserved
4. Default smoke behavior unchanged when extensions are off
5. `_legacy/` remains non-canonical (untouched)
6. T5 smoke path: supervised + PPO + test evaluation green

---

## Final series summary

- Status: **completed**
- Commits made: none (user did not request commits)
- New files: `qb_env/stop_only_env.py`, `training/hazard_pretrain.py`, `tests/test_hazard_pretrain.py`, `tests/test_action_space_alignment.py`
- Existing files updated: `models/t5_policy.py`, `scripts/train_ppo.py`, `agents/ppo_buzzer.py`, `qb_env/tossup_env.py`, `qb_env/__init__.py`, `scripts/train_t5_policy.py`, `tests/test_t5_policy.py`, `tests/test_environment.py`, `tests/test_ppo_buzzer.py`, `tests/test_factories.py`, `configs/default.yaml`, `README.md`, `AGENTS.md`
- Full test suite delta: **320 → 342 passed** (22 new passing tests)
- Explicit exclusions still preserved: no `p_correct_trace` rename, no calibration rewrite, no `stop_only` default, no `_legacy/` changes
