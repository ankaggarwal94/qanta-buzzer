# PR #1 Reconciliation Ledger

**Created:** 2026-03-15
**Branch:** main
**Local HEAD:** 9664c487
**PR branch:** personal/codex/align-action-space-with-revised-report (commits 4f3e3009, f459f246)
**Reference:** .planning/quick/pr1-integration-plan.md
**Mode:** Semantic reconciliation (not literal cherry-pick)

---

## Work Packages

### WP-1: DSPy Likelihood / Factory Contract Surface

| Field | Value |
|-------|-------|
| Objective | Verify DSPyLikelihood subclass contract, score() (K,) enforcement, stale `dspy.enabled` removal, importability/runtime docs accuracy, fingerprint/cache test coverage, factory dispatch consistency |
| Repo evidence checked | `models/dspy_likelihood.py`, `models/likelihoods.py` (factory), `models/__init__.py`, `tests/test_dspy_likelihood.py`, `tests/test_factories.py`, `configs/default.yaml`, `configs/smoke.yaml`, `README.md`, `AGENTS.md` |
| Files changed | none |
| Tests run | `pytest tests/test_dspy_likelihood.py tests/test_factories.py -q` → 24 passed in 0.06s |
| Result | **verified_closed** |
| Commit hash | — |
| Rollback command | — |
| Notes | All 6 checkpoints verified without edits. (1) `DSPyLikelihood(LikelihoodModel)` with `super().__init__()`. (2) `score()` validates `ndim==1` and `len==K`. (3) No `dspy.enabled` in any config; comment on default.yaml:84 says "no separate enable flag"; stale key removal tracked to REVIEW commit fd34e25a. (4) Module docstring says importable without dspy extra — true (no dspy import at module level); README/AGENTS correctly describe opt-in via `pip install -e '.[dspy]'`. (5) `test_changed_fingerprint_invalidates` proves distinct keys per fingerprint + correct cache population; `test_repeated_call_hits_cache` proves cache hit. (6) Factory reads `config.get("dspy", {})` for `cache_dir`/`program_fingerprint` with "default" fallback; test coverage in `TestDSPyFactoryIntegration`. |

---

### WP-2: DSPy Offline Compile / Optimize Path

| Field | Value |
|-------|-------|
| Objective | Verify optimizer metric is real, trainset uses train split, compile path is offline/testable, tests validate real code |
| Repo evidence checked | `scripts/optimize_dspy.py`, `tests/test_dspy_optimize.py`, `scripts/build_mc_dataset.py` (line 332 confirms train_dataset.json output), `chatgpt_final_review_prompt.md` (REVIEW-2 context) |
| Files changed | `scripts/optimize_dspy.py`, `tests/test_dspy_optimize.py` |
| Tests run | `pytest tests/test_dspy_optimize.py -q` → 5 passed, 1 skipped in 0.02s |
| Result | **completed** |
| Commit hash | (pending) |
| Rollback command | `git revert <hash>` |
| Notes | (1) Metric is real argmax-based comparison, fixed in REVIEW-2. (2) `main()` looks for `train_dataset.json` first with fallback + warning; `build_mc_dataset.py` produces it at line 332. (3) `compile_dspy_scorer()` requires dspy only at runtime; helpers work without it. (4) FIX: extracted `_score_metric` from closure inside `compile_dspy_scorer()` to module level so test imports the real function instead of duplicating it. |

---

### WP-3: DSPy Answer-Profile Fallback / Doc / Test Behavior

| Field | Value |
|-------|-------|
| Objective | Verify docstrings don't overclaim leave-one-out, fallback logs truthfully, test names match bodies, no misleading naming drift |
| Repo evidence checked | `qb_data/dspy_answer_profiles.py`, `tests/test_dspy_answer_profiles.py` |
| Files changed | none |
| Tests run | `pytest tests/test_dspy_answer_profiles.py -q` → 2 passed, 1 skipped in 0.01s |
| Result | **verified_closed** |
| Commit hash | — |
| Rollback command | — |
| Notes | (1) Docstring explicitly disclaims leave-one-out enforcement: "This function itself does not receive per-question exclusion context — it augments whatever profiles it is given." ✅ (2) Fallback logs at WARNING per-answer and INFO summary with augmented/fallback counts (fixed in REVIEW-2 c912c814). ✅ (3) All three test names match their bodies: `test_module_importable_without_dspy` asserts callable, `test_runtime_call_without_dspy_raises` expects ImportError, `test_with_dspy_installed` guards with importorskip and asserts callable. ✅ (4) No naming drift found; REVIEW-2 resolved the prior "silent except" issue. ✅ |

---

### WP-4: Durable Docs — PR #1 Absorption Statement

| Field | Value |
|-------|-------|
| Objective | Ensure durable docs reflect that PR #1 review-remediation content is absorbed locally; no misleading upstream references |
| Repo evidence checked | `README.md`, `AGENTS.md`, `CLAUDE.md`, `.planning/STATE.md`, `.planning/quick/extensions-master-run.md` |
| Files changed | `.planning/STATE.md`, `.planning/quick/extensions-master-run.md` |
| Tests run | `pytest tests/test_dspy_likelihood.py tests/test_dspy_optimize.py tests/test_dspy_answer_profiles.py tests/test_factories.py -q` → (pending) |
| Result | **completed** |
| Commit hash | (pending) |
| Rollback command | `git revert <hash>` |
| Notes | (1) README/AGENTS/CLAUDE already reflect local reality — modular pipeline canonical, extensions opt-in, test counts accurate, no misleading upstream references. (2) STATE.md session summary updated to mention PR #1 reconciliation status. (3) extensions-master-run.md cross-references the PR #1 reconciliation ledger. (4) No changes needed to README/AGENTS/CLAUDE — they don't mention PR #1 and shouldn't. |

---

### WP-A: T5 Joint Action Semantics

| Field | Value |
|-------|-------|
| Objective | Port factored `_joint_action_log_prob()`, `_joint_entropy()`, updated `select_action()` / `get_action_log_probs()` from PR #1 to local `models/t5_policy.py` |
| Repo evidence checked | — |
| Files changed | — |
| Tests run | — |
| Result | pending |
| Commit hash | — |
| Rollback command | — |
| Notes | Pure T5 math fix. Lowest conflict risk per integration plan. |

### WP-B: StopOnlyEnv + `--policy-mode` Flag

| Field | Value |
|-------|-------|
| Objective | Add `StopOnlyEnv` wrapper (Discrete(2) buzz/wait), `--policy-mode` CLI flag to `train_ppo.py`. Default must be `flat_kplus1` (not `stop_only`) to preserve `compare_policies.py` compatibility. |
| Repo evidence checked | — |
| Files changed | — |
| Tests run | — |
| Result | pending |
| Commit hash | — |
| Rollback command | — |
| Notes | Must NOT adopt `p_correct_trace` rename. Must handle `use_maskable_ppo` interaction. |

### WP-C: `end_mode` / `no_buzz_reward` Env Semantics

| Field | Value |
|-------|-------|
| Objective | Add `end_mode` ("force_commit" | "no_buzz") and `no_buzz_reward` constructor args to `TossupMCEnv`, wire into `step()` terminal branch and config factory. |
| Repo evidence checked | — |
| Files changed | — |
| Tests run | — |
| Result | pending |
| Commit hash | — |
| Rollback command | — |
| Notes | Must merge with existing 6+ constructor params (EW, variable-K). `no_buzz` branch must emit `forced_choice: -1` in info dict (PR P2 bug fix). |

### WP-D: Hazard Pretraining Bridge

| Field | Value |
|-------|-------|
| Objective | Add `training/hazard_pretrain.py` utilities (`compute_survival_terms`, `hazard_expected_nll_loss`), CLI flags in `train_t5_policy.py`. Mark no-op loop with `NotImplementedError`. |
| Repo evidence checked | — |
| Files changed | — |
| Tests run | — |
| Result | pending |
| Commit hash | — |
| Rollback command | — |
| Notes | Standalone new files with no downstream callers. Can be deferred indefinitely. |

### WP-X: Cross-cutting Tests

| Field | Value |
|-------|-------|
| Objective | Port `test_action_space_alignment.py` tests that apply to landed patches. Skip `p_correct_trace` metric test and `stop_only` default test. |
| Repo evidence checked | — |
| Files changed | — |
| Tests run | — |
| Result | pending |
| Commit hash | — |
| Rollback command | — |
| Notes | Depends on WP-A/B/C. Subset selection per integration plan §Cross-cutting. |

---

## Explicit Exclusions (from pr1-integration-plan.md)

| Item | Reason |
|------|--------|
| `g_trace` → `p_correct_trace` field rename | Breaks `asdict()`, regresses `top_p_trace` calibration fix |
| `CanonicalEpisodeTrace` dataclass | Unnecessary; existing trace dataclasses work with `_to_dict()` |
| `calibration_at_buzz` rewrite | Reverts verified `top_p_trace` fix (3 review rounds) |
| `--policy-mode stop_only` as default | Breaks `compare_policies.py` (P1 bug) |
| `system_score()` param rename | Breaks all existing callers; `g_trace` is standard in S_q literature |
| Softened controls language | Subjective wording, no behavioral impact |

---

## Invariants (must hold after every WP)

1. `pytest tests/ -q` → 320+ passed, ≤3 skipped
2. `bash scripts/manual-smoke.sh` → 4/4 stages complete
3. `top_p_trace` calibration invariant preserved
4. Default smoke behavior unchanged when extensions are off
5. `_legacy/` remains non-canonical (untouched)
