# AGENTS.md

Canonical repo contract for all coding agents (Claude, Copilot, Cursor, etc.).

## Project Overview

Stanford CS234 final project: a unified quiz bowl RL buzzer system with three tracks. The belief-feature pipeline builds MC tossups, scores answer profiles with TF-IDF / SBERT / T5 / optional OpenAI / optional DSPy, trains or compares buzzers, and evaluates with S_q, Expected Wins, and calibration metrics. The T5 policy pipeline provides supervised warm-start and PPO for an end-to-end text policy using factorized stop/answer semantics (`P(WAIT)` and `P(BUZZ_i) = P(BUZZ) * P(answer_i | BUZZ)`). The StopDFF v5 evidentiary pipeline (`scripts/stopdff_v5/`, CS321M) produces identity-bound, fail-closed audit runs; its normative contracts are `ACCEPTANCE_CONTRACT.md`, `SCIENTIFIC_CONTRACT.md`, and `IDENTITY_AND_ARTIFACT_CONTRACT.md` at the repo root, with reproduction steps in `docs/stopdff_v5/REPRODUCTION.md`. Three opt-in extensions: Expected Wins reward mode, variable-K answer choices, and DSPy integration. Additional opt-in feature-port surfaces are available for stop-only PPO (`scripts/train_ppo.py --policy-mode stop_only`) and no-buzz horizon behavior (`environment.end_mode: no_buzz`). `qanta-buzzer` is the canonical repo. qb-rl compatibility is preserved through additive shims rather than structural rewrites.

## Setup

Requires Python >= 3.11.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
```

Repo wrapper scripts intentionally use the repo-local `.venv` directly and do
not fall back to ambient `python3` or `pytest`, which avoids global package
skew during verification.

Optional extras:

```bash
pip install -e '.[openai]'    # OpenAI embedding support
pip install -e '.[maskable]'  # MaskablePPO for variable-K
pip install -e '.[dspy]'      # DSPy LM-based scoring
pip install -e '.[modal]'     # Modal cloud execution for the StopDFF v5 runners
```

## Architecture

| Package | Purpose |
|---------|---------|
| `qb_data/` | Data loading, answer profiles, stratified splits, MC construction, DSPy profiles |
| `qb_env/` | Gymnasium environment, text wrapper, opponent models, StopOnlyEnv wrapper (with action_masks), qb-rl shims |
| `models/` | Likelihood models (TF-IDF, SBERT, T5, OpenAI, DSPy), belief features, T5 policy |
| `agents/` | Threshold, softmax-profile, sequential Bayes, PPO wrapper |
| `evaluation/` | S_q metric, Expected Wins, calibration, control experiments, plotting |
| `scripts/` | Pipeline entrypoints, DSPy compile, shared helpers |
| `scripts/stopdff_v5/` | StopDFF v5 fail-closed pipeline (identity, manifests, checker, sweep, writers) |
| `training/` | T5 policy supervised + PPO trainers, hazard bridge utilities |
| `configs/` | YAML configuration files (default, smoke, t5_policy) |
| `schemas/` | JSON Schemas for StopDFF v5 profile/run-spec/calibrator/continuation/gate |

## Testing

The test suite spans the full v5 pipeline; counts drift with every PR, so no number is pinned here — count them live with `pytest tests/ --collect-only -q | tail -1`. Tests requiring optional extras (including `modal`) skip when those are not installed.

```bash
pytest                    # full suite
pytest tests/test_qb_rl_bridge.py tests/test_factories.py tests/test_ppo_buzzer.py  # focused bridge/runtime checks
bash scripts/ci.sh        # CI entry point via repo-local .venv
```

## Documented Solutions

`docs/solutions/` contains documented solutions to past problems, organized by category with YAML frontmatter (`module`, `tags`, `problem_type`). Relevant when implementing or debugging in documented areas.

## Smoke Pipeline

Four-stage belief-feature smoke workflow. `--smoke` selects `configs/smoke.yaml` and writes outputs to `artifacts/smoke/`.

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

`build_mc_dataset.py` writes `train_dataset.json`, `val_dataset.json`,
`test_dataset.json`, and retained-split `split_metadata.json` as canonical
downstream inputs. `mc_dataset.json` remains as a combined legacy/debug
artifact. By default, `run_baselines.py`
selects thresholds on validation, `train_ppo.py` trains on train and writes
validation metrics to `ppo_summary.json`, and `evaluate_all.py` writes the
canonical final test report on the test split (`evaluation_report.json`).

Or run all four stages via the wrapper script:

```bash
bash scripts/manual-smoke.sh
```

## Full Pipeline

For the core pipeline and scripted extensions at full scale with 4-wave parallel execution:

```bash
bash scripts/run_full_pipeline.sh --t5-model t5-base
```

The script forces `likelihood.model=tfidf` for all belief-feature phases. Phases 7, 8, 10, 11 (EW PPO), 12, 18, 19 require manual execution. See `docs/full-pipeline-runbook.md` for phase-by-phase details.

All pipeline scripts accept positional config overrides (e.g. `likelihood.model=tfidf`).

## T5 Policy Pipeline

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/compare_policies.py --config configs/t5_policy.yaml
```

Notes:
`scripts/train_t5_policy.py` parses `--hazard-pretrain`, `--beta-terminal`, and `--freeze-answer-head` for the hazard warm-start bridge (`training/hazard_pretrain.py::run_hazard_pretrain`). When `--hazard-pretrain` is set, the bridge slots between the supervised warm-start and PPO: it minimizes the survival/hazard expected-NLL loss over each question's `cumulative_prefixes` to teach the buzz/stop head *when to buzz*, then hands its checkpoint (`checkpoints/hazard/best_model`) to PPO. `--beta-terminal` weights the never-buzz survival penalty; `--freeze-answer-head` freezes the answer head only (answer-NLL gradient still reaches the shared encoder). `--hazard-ablation shuffled_nll` (requires `--hazard-pretrain`) runs a step-matched null-signal control that permutes each question's per-prefix NLL with a dedicated seeded RNG; `--seed <int>` re-seeds Python/NumPy/torch immediately before each training phase (supervised/hazard/PPO; separate from `data.seed`); the bridge also writes per-step losses to `checkpoints/hazard/hazard_history.json`. The bridge is **smoke-validated (plumbing only)** on CPU; training efficacy needs full-scale CUDA runs.

### Hazard efficacy harness

`scripts/run_hazard_efficacy.py` runs the controlled WITH/WITHOUT/compute-control
comparison for the hazard bridge: one shared supervised checkpoint branches into
arm A (control), arm B (`--hazard-pretrain`), arm C (`--hazard-ablation
shuffled_nll`, step-matched null signal) × `--seeds`, plus optional
`--variant NAME:FLAGS` knob arms, all evaluated through the same
`evaluate_t5_policy` path on the same test split.

```bash
# Smoke (t5-small; measured: ~10 min child wall-clock across 18 runs, ~5 GB tree)
python scripts/run_hazard_efficacy.py --smoke --config configs/t5_policy.yaml \
    --out-dir results/hazard_efficacy_smoke --seeds 1 2 3 --arms A B C --prune-checkpoints

# t5-base preliminary (measured: ~109 min child wall-clock across 12 runs, ~10.7 GB tree)
python scripts/run_hazard_efficacy.py --config configs/t5_policy_base_prelim.yaml \
    --out-dir results/hazard_efficacy_base --seeds 1 2 3 --arms A B C --prune-checkpoints \
    --variant "beta_hi:--beta-terminal=2.0"
```

Output layout: `<out-dir>/hazard_efficacy_report.json` (schema_version 1: scale
block, endpoint, scale-gated paired-bootstrap significance, arm deltas, hazard
dynamics, per-run provenance) + `hazard_efficacy_plot.png`; per run dir:
`ppo_t5/best_model` + sidecars (`config_used.json`, `split_manifest.json`),
`eval_result.json` (per-question records, written immediately after each eval),
`hazard/hazard_history.json` + `hazard_dynamics.json` (hazard arms),
`train.log`, and the completion marker `RUN_COMPLETE.json`. Non-smoke split
resolution prefers `artifacts/main/` — populate it (or run
`build_mc_dataset.py`) or children silently fall back to `artifacts/smoke/`.

Semantics: re-running the same command RESUMES — complete runs (marker +
checkpoints + sidecars, identity-validated against the current invocation) are
skipped, partial dirs fail loud with delete/`--force` instructions, `--force`
re-runs everything including the shared checkpoint. `--report-only` reassembles
report + plot from existing run dirs (no training/eval; same validation gates).
`--prune-checkpoints` deletes `iter_*`/`epoch_*`/`training_state.pt` per run
once its `eval_result.json` exists (report stays regenerable; `best_model` +
sidecars kept). `--dry-run` prints the validated plan (including the shared
supervised child and a disk estimate) and launches nothing. A stalled child is
killed after `--stall-timeout-minutes` (default 120) without output.

## StopDFF v5 Pipeline (CS321M)

Identity-bound, fail-closed StopDFF audit pipeline in `scripts/stopdff_v5/` with JSON Schemas in `schemas/`. Runs are create-once and content-addressed; the normative contracts are `ACCEPTANCE_CONTRACT.md` (acceptance gate), `SCIENTIFIC_CONTRACT.md` (scientific protocol), and `IDENTITY_AND_ARTIFACT_CONTRACT.md` (identity and artifact rules). Full local (CPU) and Modal reproduction steps: `docs/stopdff_v5/REPRODUCTION.md`.

```bash
# local end-to-end run (smoke variant)
python scripts/run_stopdff_v5_local.py --data-dir data/processed \
    --paper-exports paper_exports --out-dir stopdff_v5_smoke_out --variant smoke

# standalone acceptance-gate validation of a run package
python scripts/validate_stopdff_bucketed_sweep.py validate RUN_ROOT \
    --backend local --adapter-bundle ADAPTER_BUNDLE --require-final-profile --require-package
```

Modal execution uses `scripts/modal_stopdff_v5_runner.py` (requires the `modal` extra and explicit compute authorization).

## Configuration

| Config | Purpose |
|--------|---------|
| `configs/default.yaml` | Full runs with T5-large likelihood and 100k PPO timesteps |
| `configs/smoke.yaml` | Quick tests: 50 questions, TF-IDF likelihood, 3k PPO timesteps |
| `configs/t5_policy.yaml` | T5 policy pipeline: model, supervised, PPO, and data sections |

qb-rl config aliases are supported (e.g., `data.dataset`, `likelihood.sbert_name`, `environment.reward` as alias for `reward_mode`).

Additional environment options:
- `environment.end_mode: force_commit|no_buzz` controls horizon behavior
- `environment.no_buzz_reward` is only used when `end_mode: no_buzz`

## Compatibility Bridge

Old qb-rl import paths that still resolve:

- `qb_env.data_loader`, `qb_env.mc_builder`, `qb_env.text_utils`
- `models.answer_profiles`
- `agents.softmax_profile_buzzer`

OpenAI support is opt-in only. Default local workflows stay offline-friendly and do not require the `openai` package or `OPENAI_API_KEY`.

## Correctless (structured-development tooling)

`.correctless/` vendors the Correctless dev-workflow framework (spec → review → TDD → verify gates). Its phase gates are **advisory, best-effort tooling — bypassable by construction — not an integrity or security boundary.** It is layered under GSD, which remains **canonical for planning and phase state** (`.planning/`). The owner's monorepo-level agent policy (a parent-directory `CLAUDE.md` maintained *outside* this repository) forbids running a second orchestration engine inside active GSD-managed execution; that rule is honored here: Correctless is used for structured review/TDD, not as a competing planning engine, and `.correctless/ARCHITECTURE.md` records the GSD-canonical-for-planning carve-out (adopted as a deliberate, consented decision). Its hook wiring (`.claude/settings.json`) is **gitignored**, so a plain checkout does not auto-run any gate. Update/remove: re-run `/csetup` to regenerate `.correctless/` from `.correctless/.install-manifest.json`, or `rm -rf .correctless/` (and delete the gitignored `.claude/` hook wiring) to uninstall. Correctless's hooks and scripts require **bash ≥ 4** (macOS `/bin/bash` is 3.2; the hooks run under `#!/usr/bin/env bash`, so a Homebrew bash 4+/5+ on PATH is required — the auto-run PreToolUse hooks degrade with a clear message on older bash, the helper scripts assume bash 4+). See `.correctless/AGENT_CONTEXT.md`.

## Conventions

- NumPy-style docstrings with Parameters/Returns sections
- RL notation: `V` (value), `R` (reward), `T` (transition), `gamma` (discount), `s`/`a` (state/action)
- Prefer NumPy/PyTorch vectorized operations over loops in ML code
- Explicit seeds for reproducibility (use 1, 2, 3 for multi-seed runs)
