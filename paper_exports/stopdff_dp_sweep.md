# DP StopDFF Sweep

Generated: `2026-05-27T23:23:48.644030+00:00`

## paper-safe interpretation

**Verdict:** WARN (small_stopdff_but_coverage_or_ceiling_weak)

- PASS only if DP StopDFF is small and coverage/calibration gates pass.
- WARN if DP StopDFF is small but coverage is weak or a ceiling effect persists.
- FAIL if DP StopDFF is materially shifted under MC.
- Oracle continuation cells are diagnostic upper bounds and are excluded from confirmatory interpretation.

## Cell Status

| Status | Count |
|---|---:|
| completed | 1 |
| skipped | 0 |
| failed | 0 |

## Completed Cells

| Cell | Reward | Continuation | Calibrator | Format | Signed mean | Gate |
|---|---|---|---|---|---:|---|
| 92d7e8cdd9422cbd_acf_flat_empirical_bucket_uncalibrated | acf_flat | empirical_bucket | uncalibrated | MC-fixed | 0.000 | warn |