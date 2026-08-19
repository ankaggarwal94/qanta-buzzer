# Verification: Hazard-Pretrain Efficacy Eval Harness

- Spec: `.correctless/specs/hazard-efficacy-eval.md`
- Branch: `feature/hazard-efficacy-eval`
- Date: 2026-08-18
- Effective intensity: **standard** (feature: standard, project: absent → standard; mutation-survivor analysis is critical-only, skipped)
- Verifier: /cverify (independent agent; did not participate in implementation)

## Rule Coverage

Convention: every rule has rule-ID-prefixed tests (`test_rNNN_*`), so coverage tracing is mechanical. Counts verified by `grep "def test_rNNN"`.

| Rule | Test(s) | Status | Notes |
|------|---------|--------|-------|
| R-001 [integration] (E-1 seed) | `test_r001_seed_reseeds_before_each_phase_and_reproduces`, `test_r001_r003_config_used_records_seed_and_hazard_block`, `test_r001_default_no_seed_leaves_rng_untouched` — `tests/test_train_seed_e1.py` | covered | Strong: scrambles RNGs from OS entropy between fake phases so replay is only possible via per-phase re-seeding; asserts per-library (torch/random/numpy) divergence across seeds; split manifest byte-identical; real `main()`/`parse_args`/split loading. |
| R-002 [integration] (E-2 runs) | `test_r002_return_runs_is_keyword_only_defaulting_false`, `test_r002_return_runs_records_one_per_question_and_consistent`, `test_r002_default_omits_runs_and_payload_unchanged` — `tests/test_evaluate_t5_runs_e2.py` | covered | Signature verified in source: `*, return_runs: bool = False` (`scripts/compare_policies.py:423`). Real env episode loop. |
| R-003 [integration] (arm control) | `test_r003_clean_sidecars_pass`, `test_r003_model_name_mismatch_names_offending_arm`, `test_r003_split_qid_mismatch_names_offending_arm`, `test_r003_unexpected_key_difference_names_arm_and_key` — `tests/test_hazard_efficacy_orchestration.py` | covered | Doctored-sidecar exits per spec; identity gates further hardened by MA-001 tests. |
| R-004 [integration] (shuffled_nll ablation) | 7 tests `test_r004_*` — `tests/test_hazard_ablation_dynamics.py` | covered | Step-matched + loss-divergence proof, singleton-prefix identity, CLI rejection paths, unknown-ablation fail-loud. Real t5-small CPU loop. |
| R-005 [integration] (identical eval path) | `test_r005_one_call_per_run_identical_kwargs`, `test_r005_eval_result_enrichment_and_persistence` — orchestration | covered | Identical qids/kwargs across calls; no Expected Wins key (also `test_r009_no_expected_wins_key_anywhere`). Manifest-scoped question selection per QA-004 fix. |
| R-006 [unit] (primary endpoint) | 8 tests `test_r006_*` — `tests/test_hazard_efficacy_harness.py` | covered | Edge cases per spec: exact-threshold boundary = success, 0.99-gain = non-success, 2-of-3 replication, zero-correct-buzz → `undefined_position: true`, empty input raises. |
| R-007 [unit] (paired bootstrap, scale-gated) | 8 tests `test_r007_*` — harness | covered | Scale-gate boundary at n=50, seed-averaged pooling (never 3×n), unpaired/missing-qid fail-loud, determinism, plus identity assertion that `bootstrap_ci` comes from `evaluation/controls.py`. |
| R-008 [integration] (distinct dirs + provenance) | 6 tests `test_r008_*` — orchestration | covered | Distinct dir per arm×seed, provenance complete-or-fail-loud, supervised-manifest persisted + test-disjointness, variant path safety. |
| R-009 [unit] (report schema + caveat) | 5 tests `test_r009_*` — harness | covered | Verified against BOTH committed real reports (see Evidence): `schema_version: 1`, `endpoint_definition`, scale block incl. `disk_usage_bytes`, Device-2 caveat verbatim, structured `{verdict, scope, evidence}`, plot path, Agg/savefig. |
| R-010 [integration] (hazard_history + stop-prob probe) | `test_r010a_*` (2, ablation_dynamics) + `test_r010b_*` (6, `tests/test_hazard_efficacy_probe.py`) | covered | Pinned history schema incl. `wall_clock_seconds` (QA-006 semantic), real-checkpoint probe test, deterministic probe, first-min-32-in-order selection. |
| R-011 [integration] (smoke overrides, shell-safety, tee) | 9 tests `test_r011_*` — orchestration | covered | Smoke argv injects `ppo.eval_interval=1` and not `save_interval`; source-level no-`shell=True` assertion; REAL child tee test; nonzero-exit names run + log tail. See DRIFT-001 for a spec-wording nuance (shared supervised child's QA-007/MA-017 flags). |
| R-012 [unit] (reuse, no reimplementation) | `test_r012_shared_eval_entrypoint_and_no_env_reimplementation`, `test_r012_eval_stage_calls_entrypoint_exactly_once` — harness | covered | `is`-identity with `scripts.compare_policies.evaluate_t5_policy` + source-level absence of `TossupMCEnv`/`TextObservationWrapper` (grep-confirmed in `scripts/run_hazard_efficacy.py`: zero matches). |
| R-013 [integration] (preflight + resume) | 11 tests `test_r013_*` — orchestration | covered | Preflight-before-any-child, dry-run zero children, complete/partial/force classification, doctored resumed dir fails arm control, random-split fallback rejected, git-SHA mismatch recorded-not-fatal. Hardened further by MA-001/002/003/013/015 tests. |
| R-014 [integration] (per-run eval persistence, report-only, prune) | 4 tests `test_r014_*` — orchestration | covered | Assembly reads only per-run files; eval artifacts survive downstream failure; prune keeps regenerable state and refuses without `eval_result.json`. `--report-only × --dry-run` destructive combo fixed per QA-005. |
| R-015 [unit] (AGENTS.md harness docs) | asserted at /cverify per spec's Test line (re-verified post-fix `50ba19ba`) | covered | "### Hazard efficacy harness" subsection present with all 4 spec-required elements; every documented factual claim verified against ground truth, not just prose presence — see "Re-verification" evidence below. |

Mini-audit structural coverage: all 18 MA findings have dedicated `test_maNNN_*` tests (43 tests in `tests/test_hazard_efficacy_mini_audit.py`).

## Evidence (fresh, this verification run)

- **Full suite on canonical `.venv`** (`.venv/bin/python -m pytest`, log: `.correctless/artifacts/cverify-fullsuite-hazard-efficacy-eval.log`): **3 failed, 1879 passed, 4 skipped in 336.87s (exit 1)**.
  - The 3 failures are EXACTLY the pre-existing accounted standing failures, all in StopDFF files untouched by this feature: `test_stopdff_value_model.py::test_seed_determinism_within_one_seed` + `::test_smoke_training_runs_on_cpu_in_under_60s` (fail-closed dirty-tree provenance guard; tree has 34 untracked entries — the committed-results workflow guarantees this) and `test_stopdff_v5_package.py::test_checksum_inventory_rejects_special_entries_without_hashing[unix_socket]` (macOS AF_UNIX sun_path ~104-byte cap vs pytest tmp path). Root-caused + fixed on unmerged worktree branch `claude/mystifying-albattani-383fa8`; this matches the `--deselect` set used at this branch's earlier gate transitions.
  - **Every feature test passed** (all `test_r0*`/`test_ma0*` files green within the 1879).
  - **Test-success sentinel NOT written**: the bare full suite exited 1, so `.correctless/artifacts/test-success.sha` was deliberately not recorded (fail-closed). The `done` gate this sentinel serves has already passed on this branch.
- **Real end-to-end executions** (factory-boundary check — the documented command itself was run, twice, results committed): `results/hazard_efficacy_smoke/hazard_efficacy_report.json` (t5-small, 18 run rows, `significance: "not_evaluable_at_this_scale"` at n_test=13 — R-007 gate correct) and `results/hazard_efficacy_base/hazard_efficacy_report.json` (t5-base, 12 run rows, n_test=110 → `significance_evaluable: true`, paired bootstrap CI). Both carry `schema_version: 1`, `endpoint_definition`, structured verdict, Device-2 caveat, `hazard_dynamics.index_convention` (MA-011), plot files present.
- Structural greps: `--seed` flag (`scripts/train_t5_policy.py:165`); `ablation` keyword-only + fail-loud (`training/hazard_pretrain.py:91-191`); `hazard_history.json` pinned (R-010a); `DEVICE2_CAVEAT` constant (`scripts/run_hazard_efficacy.py:78`); `subprocess.Popen` context-manager with fixed argv, `shell=False` only (QA-009).

## Re-verification (2026-08-18, post-R-015 fix — fresh evidence this run)

Scope: targeted re-verify of the single blocking finding. Delta since the prior run is exactly one commit, `50ba19ba` ("docs(hazard-efficacy-eval): AGENTS.md harness subsection (R-015)"), touching only `AGENTS.md` (+41 lines) — docs-only, so all prior code-rule evidence (R-001..R-014, full-suite run, real end-to-end executions) stands unchanged.

Per-element verification of the subsection's claims:

1. **Smoke + t5-base invocations**: both documented command lines parsed through the harness's REAL `parse_args` in-process (canonical `.venv`) — clean, with expected namespace values (`--seeds 1 2 3`, `--arms A B C`, `--prune-checkpoints`, `--variant "beta_hi:--beta-terminal=2.0"`); both referenced configs exist (`configs/t5_policy.yaml`, `configs/t5_policy_base_prelim.yaml`).
2. **Wall-clock + disk footprint**: recomputed from committed artifacts — smoke: 18 `RUN_COMPLETE.json` markers, `wall_clock_seconds` sum = 10.3 min, report `scale.disk_usage_bytes` = 5.11 GB (doc: "~10 min … 18 runs, ~5 GB"); base: 12 markers, 108.9 min, 10.65 GB (doc: "~109 min … 12 runs, ~10.7 GB"). All four numbers match.
3. **Artifact layout**: checked against real run dirs — `results/hazard_efficacy_smoke/A_seed1/` has `RUN_COMPLETE.json`, `eval_result.json`, `train.log`, `ppo_t5/{best_model, config_used.json, split_manifest.json, …}`; `results/hazard_efficacy_base/B_seed1/` additionally has `hazard/{best_model, hazard_history.json}` + root `hazard_dynamics.json` (hazard-arm-only, as documented); both `hazard_efficacy_plot.png` files exist. Prune claim confirmed live: zero `iter_*`/`epoch_*`/`training_state.pt` anywhere in either tree while `best_model` + sidecars kept.
4. **Resume / `--report-only` / `--prune-checkpoints` semantics**: doc text matches the R-013/R-014-tested behavior; `--stall-timeout-minutes` default 120 confirmed from the live parser; "non-smoke split resolution prefers `artifacts/main/`, falls back to `artifacts/smoke/`" confirmed at `scripts/train_t5_policy.py:318-320` (candidate order `[main, smoke]` when non-smoke).

Delta-scoped regression check: the only two test files that read `AGENTS.md` (`tests/test_agents_doc_counts.py`, `tests/test_v5_cli_entrypoints.py`) were run fresh on the canonical `.venv` — **10 passed, exit 0**. No other file changed since the prior full-suite evidence, so no further suite run was warranted (narrow-verification rule).

## Dependencies

- **None.** `git diff main...HEAD -- pyproject.toml requirements.txt setup.py setup.cfg` is empty. No new imports outside the existing stack (torch/transformers/numpy/matplotlib already project deps).

## Architecture Adherence

- PAT-001 (likelihood factory dispatch): valid — entry's `Enforced at` paths (`models/likelihoods.py`, `qb_env/tossup_env.py`) exist and are untouched by this feature; the invariant is not contradicted (harness reaches scoring only through `evaluate_t5_policy`, enforced by R-012's identity + no-env-import tests). No ABS/TB/ENV entries exist (dormant).
- Advisory for /cdocs (LOW): the harness introduces documentable patterns (pinned sidecar formats, resume identity gates/fingerprints, plan-reconciled aggregation) that may merit ARCHITECTURE.md / AGENT_CONTEXT.md entries; and R-015's missing AGENTS.md subsection overlaps /cdocs scope but is spec-mandated NOW (see blocking finding).

### Drift Debt
- DRIFT-001 (new, logged this run): R-011 wording says smoke injects "ONLY `ppo.eval_interval=1` … into every child", but the shared supervised child also gets `--ppo-iterations 1` (QA-007) + `--skip-test-eval` (MA-017) — recorded, tested fixes; spec sentence never amended. Wording-only drift.

1 entry checked (PAT-001), 0 stale, 1 drift-debt item (open).

## Compliance Checks

- None configured (`workflow.compliance_checks` absent).

## QA Class Fixes Verified

- 41 findings across QA round 1 (QA-001..013), round 2 (QA-R2-1..4), mini-audit (MA-001..018), re-verification (F1..F6). All `fixed` except QA-R2-4 (`accepted`, record amended — shipped `validate_resumed_run` scope is model_name + qid membership, deliberately narrower; paper-trailed).
- All 18 MA class fixes have dedicated structural tests (`test_ma001_*`..`test_ma018_*`, 43 tests) ✓. QA class fixes are asserted in the R-011/R-013/R-014 orchestration tests and mini-audit file (QA IDs referenced across 6 test files) ✓. Spot-verified: QA-003's class fix (argv round-trip through the child's real parser at preflight) → `test_ma013_config_typo_dies_in_preflight`; QA-005 → `test_r014`-family + dry-run tests; QA-013 (save-best `-inf` init) → covered in `tests/test_ppo_t5.py`/supervised trainer changes.

## Antipattern Scan

`bash .correctless/scripts/antipattern-scan.sh main` → valid JSON, 130 findings (capped 20/file), artifacts: `.correctless/artifacts/cverify-antipattern-scan.json`.

| Category | Where | Disposition |
|---|---|---|
| `debug-print` (medium) ×~110 | `run_hazard_efficacy.py`, `train_*.py`, `compare_policies.py`, `_common.py`, tests | Not actionable: `print()` is the intended CLI/progress surface for these research scripts — R-011 explicitly REQUIRES per-run progress banners; scanner has no CLI-vs-debug distinction. |
| `error-suppression` (high) ×9, `debug-echo` (low) ×11 | `.correctless/hooks/workflow-advance.sh` | Pre-existing vendored-framework findings (upstream concern per the documented vendored-framework convention); the branch's only touch to this file is the macOS `wc` fix from the prior task. |
| Scanner `errors` array | 2 entries: "Failed to scan results/…/hazard_efficacy_plot.png: binary file" ×2 | Benign — binary plot artifacts; reported per protocol. |

Semantic checklist (`.correctless/checklists/ai-antipatterns.md`) + manual smell grep: **zero** TODO/FIXME/HACK/XXX in feature source files; no commented-out code blocks found; no broad `except:` in the harness (fail-loud typed errors used throughout).

## Drift

- DRIFT-001 logged (see Architecture Adherence). No other drift: spec `implemented_in`-style references all resolve (producers in the Format-pinning section exist where stated); code paths match declared abstractions (bootstrap reuse pinned by identity test).

## Spec Updates

- Adopted at `4e63a5e3`; amended twice during TDD with inline provenance notes in the Format-pinning section: QA fix round 1 (`697b7e8b` — QA-006 `wall_clock_seconds` semantics) and mini-audit round (`8676b1c8` — MA-001/003/010/11/015 additive sidecar fields). Amendments are additive/pinning-only; no rule semantics were weakened.

## Calibration

- Calibration entry assembled but NOT persisted: `.correctless/scripts/meta-record.sh` is absent in this install (writer script not found; direct Write/Edit to the SFG-protected `intensity-calibration.json` is prohibited). Mechanical token: `meta-record: FAILED .correctless/meta/intensity-calibration.json: writer script not found — run /csetup to install meta-record.sh`. Entry values for the record: recommended=standard, actual=standard, qa_rounds=2, blocking_findings=5, tokens=0 (no token log), spec_updates=2, harness_version=1, fix_rounds_triggered=2.
- Re-verification run (post-R-015): full entry re-assembled (30 files touched vs main, cost artifact absent → `actual_cost_usd` omitted) and the sanctioned writer re-attempted via stdin — same rc=127, same verbatim token as above. Not a blocker; unblocks via `/csetup`.

## Overall: **PASS — 0 blocking findings** (re-verified 2026-08-18 after the R-015 fix)

- All 15 rules covered: R-001..R-014 (code; unchanged since the prior run — delta is docs-only commit `50ba19ba`) + R-015 (AGENTS.md subsection, verified element-by-element against ground truth this run, with the two AGENTS.md-reading test files re-run green).
- Prior status for the audit trail: the initial verification FAILed solely on R-015 (subsection absent); the fix landed in `50ba19ba` and this targeted re-verification cleared it.
- Non-blocking, carried forward: DRIFT-001 (spec wording, logged), calibration writer missing (infra; `/csetup` to fix), 3 pre-existing accounted suite failures (out-of-feature StopDFF files, fix pending merge elsewhere).

Next step: run `/cdocs` (final step before merge; not ready to merge until it has run and `workflow-advance.sh documented` has been called).
