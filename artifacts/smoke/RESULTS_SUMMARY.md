# Smoke Pipeline Results Summary

Generated: 2026-03-09 (latest run)

## Pipeline Status

- Status: **ok**
- Full smoke pipeline passes end-to-end via:
  - `python scripts/run_smoke_pipeline.py`
- Runtime and per-stage timings are in:
  - `artifacts/smoke/smoke_pipeline_summary.json`

## Current Headline Metrics

### Best non-RL baseline (SoftmaxProfile @ threshold 0.5)
- Accuracy: **0.3864**
- Mean S_q: **0.2433**
- Mean buzz step: **3.50**

### PPO smoke (latest tuned reward + deterministic full-dataset eval)
Configuration knobs currently set in `configs/smoke.yaml`:
- `environment.wait_penalty: 0.05`
- `environment.early_buzz_penalty: 0.2`
- `environment.buzz_incorrect: -1.0`
- `ppo.total_timesteps: 3000`

Metrics:
- Accuracy: **0.3864**
- Mean S_q: **0.2734**
- Mean buzz step: **4.0455**
- ECE: **0.1445**
- Brier: **0.2783**

Source: `artifacts/smoke/ppo_summary.json`

## Reward Sweep Results (3x3)

Sweep script: `python scripts/sweep_reward_shaping.py --seeds 13,42,123`

Best config by balanced objective (`mean_accuracy + mean_S_q - 0.5 * mean_ECE`):
- `wait_penalty=0.05`
- `early_buzz_penalty=0.2`
- Mean accuracy (3 seeds): **0.3409**
- Mean S_q (3 seeds): **0.3169**
- Mean buzz step (3 seeds): **0.0000**
- Mean ECE (3 seeds): **0.1220**
- Objective: **0.5968**

### Follow-up multi-seed sweeps

- 5k timesteps (`--timesteps 5000`), best config `0.05/0.2`:
  - Mean accuracy: **0.3409**
  - Mean S_q: **0.3338**
  - Mean ECE: **0.0350**
  - Objective: **0.6572**
- 10k timesteps (`--timesteps 10000`), best config `0.05/0.2`:
  - Mean accuracy: **0.3409**
  - Mean S_q: **0.3396**
  - Mean ECE: **0.0060**
  - Objective: **0.6775**

Recommendation from current smoke sweeps:
- Keep `wait_penalty=0.05` and `early_buzz_penalty=0.2` as the candidate pair for non-smoke PPO runs.

Artifacts:
- `artifacts/smoke/reward_sweep_results.csv`
- `artifacts/smoke/reward_sweep_results.json`

## Comparison Notes

- The new recommended smoke reward config is selected for cross-seed stability rather than a single-seed peak.
- PPO performance still varies by seed and reward settings on this tiny smoke slice, so multi-seed means are more reliable for config decisions.
- Increasing timesteps from 5k to 10k improved objective mainly through better calibration (lower ECE) and slightly higher mean `S_q`.

## Controls Snapshot (evaluation report)

- Choices-only accuracy: **0.0909** (chance = 0.25 on this small smoke subset)
- Shuffle control S_q: **0.2367** (close to full-eval baseline S_q)

Source: `artifacts/smoke/evaluation_report.json`

## Recommended Next Steps

1. Increase PPO timesteps in smoke (e.g., 5k to 10k) before judging policy quality.
2. Keep using multi-seed sweeps and rank by the balanced objective (includes ECE penalty).
3. Promote the selected reward config to non-smoke runs and compare against baseline on larger splits.
