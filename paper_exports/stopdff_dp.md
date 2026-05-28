# Finite-Horizon DP StopDFF

**Metric type:** `finite_horizon_dp` - confirmatory: `True`

| Field | Value |
|-------|-------|
| Reward schedule | power_mark |
| Continuation estimator | empirical_bucket |
| Fit split | val |
| Eval split | test |
| n_items | 30 |
| StopDFF signed median | 0.0000 |
| StopDFF signed mean | -0.1333 |
| Gate verdict | warn |

## Coverage

- exact=0.710, pooled=0.290, missing=0.000; verdict=warn (fraction_pooled=0.290 > 0.05)

## Ceiling diagnostics

- all_stop_at_first_prefix: False
- all_stop_at_final_prefix: False
- no_cross_format_stopping_variance: False
- n_items: 30
- n_stopped_cells: 57
- n_never_stopped_cells: 3
- empty: False
