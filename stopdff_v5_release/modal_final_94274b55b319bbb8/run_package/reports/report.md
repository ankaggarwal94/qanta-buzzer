# StopDFF bucketed-DP paired audit (stopdff_bucketed_dp_paired_v2, protocol v5)

- Profile name / version: `stopdff_bucketed_dp_paired_v2` schema 2
- Profile variant: final
- Backend: modal

## Paired-format definition
paired_qa_prefix_vs_mc_fixed: each item contributes a QA prefix trajectory and an MC fixed-option trajectory; the signed index metric is tau_MC - tau_QA.

## Reward table
| schedule | correct_early | correct_late | wrong | split | wait_cost |
|---|---|---|---|---|---|
| acf_flat | 10 | 10 | -5 | 1.0 | 0 |
| power_mark | 15 | 10 | -5 | 0.5 | 0 |
| wait_cost_small | 15 | 10 | -5 | 0.5 | 0.05 |
| strict_wrong | 15 | 10 | -10 | 0.5 | 0 |

## Calibrator definition
Calibrators (platt-logistic, similarity-temperature, isotonic) are fit on validation MC rows only and the shared phase map is applied to MC and QA. Phase boundaries: ['0.33', '0.66'].

## Continuation definition
Empirical-bucket / pooled-empirical continuation with the canonical fallback ladders and coverage tags (primary/fallback/missing). FVI is damped (0.5), float64, with two-consecutive-iteration convergence and cycle detection.

## FVI settings
- damping: 0.5
- tolerance: 1e-6
- max_iterations: 100

## Cell counts
- requested: 96
- completed: 96
- skipped: 0
- failed: 0
- cell verdicts: PASS=82 WARN=14 FAIL=0

## Family maximum statistic and CI
- family statistic M (max cell median |index shift|): 1.0
- family 95% CI: [1.0, 1.0]
- family verdict: WARN

## MC gate evidence and overrides
- allow_low_mc_retention: False
- allow_incomplete_mc_coverage: False

## Never-buzz asymmetry
Per-cell never_buzz_MC and never_buzz_QA are preserved in each cell record.

## Release validity
- release_status: VALID

## Resource and cost summary
```json
{
  "backend": "modal"
}
```
