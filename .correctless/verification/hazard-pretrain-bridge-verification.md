# Verification — hazard-pretrain-bridge

- Spec: `.correctless/specs/hazard-pretrain-bridge.md`
- Branch: `feature/hazard-pretrain-bridge`
- Date: 2026-08-17
- Verifier: parent (independent re-run + semantic read; subagent-implemented)

## Rule coverage (all COVERED)

| Rule | Test(s) | Status |
|---|---|---|
| R-001 (`--hazard-pretrain` no longer raises) | `test_hazard_pretrain_flag_no_longer_raises` | COVERED |
| R-002 (main() runs hazard between supervised & PPO; absent flag → not called) | `test_main_wires_hazard_between_supervised_and_ppo`, `test_main_skips_hazard_when_flag_absent` | COVERED |
| R-003 (load → loop → save `checkpoints/hazard/best_model` → reloadable) | `test_run_hazard_pretrain_smoke_finite_and_loadable` | COVERED |
| R-004 (`stop_probs`/`nll` shapes `[1,T]`, finite scalar loss) | `test_hazard_loss_shapes_and_finite`, `test_run_hazard_pretrain_smoke_finite_and_loadable` | COVERED |
| R-005 (`--freeze-answer-head` freezes answer head; without it wait-head moves) | `test_run_hazard_pretrain_freeze_answer_head`, `test_run_hazard_pretrain_no_freeze_moves_wait_head` | COVERED |
| R-006 (`--beta-terminal` threaded; monotone in never-buzz mass) | `test_beta_terminal_monotonic_increases_loss` | COVERED |
| R-007 (reads only `MCQuestion.{cumulative_prefixes,options,gold_index}`, AP-031 pinned; T=1 finite) | smoke/shapes tests exercise the real `MCQuestion` fields; `_format_choices` AP-031 comment pins the format | COVERED |
| R-008 (fail-loud on missing ckpt; empty = no-op; skip T=0) | `test_run_hazard_pretrain_missing_path_fails_loud`, `test_run_hazard_pretrain_empty_questions_noop`, `test_run_hazard_pretrain_skips_zero_prefix_questions` | COVERED |

## Evidence (independent, on canonical `.venv`)
- `tests/test_hazard_pretrain.py` — **13 passed** (re-run by verifier, not just the implementer).
- `tests/test_train_t5_policy_script.py` — **9 passed** (regression for the `validate_args` guard deletion + `main()` wiring).
- Semantic read of `training/hazard_pretrain.py::run_hazard_pretrain`: stop-prob index (col 1 = BUZZ), NLL broadcast over T prefixes, existing loss reused (not reimplemented), literal answer-head freeze, fail-loud/empty/T0 handling — all match the spec.
- Diff scope: only `training/hazard_pretrain.py` (+155), `scripts/train_t5_policy.py` (+18/−6), `tests/test_hazard_pretrain.py` (+570) — plus a separate `.correctless/hooks/workflow-advance.sh` infra fix (macOS `wc -l`).

## Caveats / not-validated
- **Smoke-validated (plumbing only).** Training efficacy (convergence / S_q / calibration) is NOT verified — needs full-scale CUDA (Device 2 / RTX 5090). Out of scope per spec.
- **Gate test-scoping:** the repo's `commands.test` is bare `pytest`, which on PATH resolves to the broken homebrew interpreter and would run the full suite; the TDD gates were therefore scoped to the hazard file on `.venv`. Verifier separately confirmed the hazard + train-script tests on `.venv`.
- **Full suite not run** (isolated change; the closest regression surface — the train-script tests — passes).

## Verdict
All 8 rules COVERED and passing on the canonical interpreter; implementation semantics verified against spec. Ready for docs + merge, subject to the documented smoke-only efficacy caveat.
