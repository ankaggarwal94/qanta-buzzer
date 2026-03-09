---
status: complete
phase: 04-ppo-training-pipeline
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md]
started: 2026-02-26T03:30:00Z
updated: 2026-02-26T03:35:00Z
---

## Current Test

[testing complete]

## Tests

### 1. PPOBuzzer unit tests pass
expected: All 19 tests in test_ppo_buzzer.py pass (config loading, JSON roundtrip, episode traces, checkpoints)
result: pass

### 2. Stage 1: build_mc_dataset --smoke
expected: Constructs MC questions from QANTA CSV, prints statistics, completes without error
result: pass

### 3. Stage 2: run_baselines --smoke
expected: Runs 4 baseline agents with threshold sweep, prints per-agent accuracy/S_q, saves baseline_summary.json in <30s
result: pass

### 4. Stage 3: train_ppo --smoke
expected: Trains PPO with 1000 timesteps, shows SB3 training metrics, saves ppo_model.zip and ppo_summary.json
result: pass

### 5. Stage 4: evaluate_all --smoke
expected: Runs full evaluation + control experiments (shuffle, alias, choices-only), generates plots, saves evaluation_report.json
result: pass

### 6. Output artifacts complete
expected: artifacts/smoke/ contains baseline_summary.json, ppo_model.zip, ppo_runs.json, ppo_summary.json, evaluation_report.json, plots/ with calibration.png, entropy_vs_clue.png, comparison.csv
result: pass

### 7. Evaluation report structure
expected: evaluation_report.json has sections: full_eval, controls, baseline_summary, ppo_summary with accuracy/S_q metrics
result: pass

## Summary

total: 7
passed: 7
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
