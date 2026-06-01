# Learned-Value StopDFF Robustness Sweep

This diagnostic sweep checks whether the learned-value StopDFF conclusion changes when the continuation-value model is trained harder than the canonical ensemble.

| Run | Seeds | Checkpoints | Hidden | Train wall seconds | Signed median | Signed mean | Abs median | DP delta abs median | DP signed mean diff | Myopic signed mean diff | Gate | Resolution |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| canonical | 1,2,3 | 3 | 128,128 |  | 0.000000 | 0.471656 | 0.000000 | 0.000000 | 0.604990 | 0.471656 | warn (low_mc_retention) | reduces |
| robust_same_capacity | 1,2,3,4,5 | 5 | 128,128 | None | 0.000000 | 0.332595 | 0.000000 | 0.000000 | 0.465929 | 0.332595 | warn (low_mc_retention) | reduces |
| robust_wide | 1,2,3,4,5 | 5 | 256,256 | None | 0.000000 | 0.371568 | 0.000000 | 0.000000 | 0.504901 | 0.371568 | warn (low_mc_retention) | reduces |

## Interpretation

The headline learned-value StopDFF conclusion is robust under both harder training settings: the signed median remains 0.0 and the DP comparison delta_abs_median remains 0.0. The signed mean shifts lower in the robustness runs, so the tail behavior is somewhat training-sensitive, but this does not reverse the median-based headline conclusion.

## Caveats

- Robustness outputs are diagnostic and do not replace the canonical learned-value audit artifact.
- Checkpoints and logs are local ignored artifacts under `artifacts/device2/learned_value_robustness/`.
- The audit card was not refreshed from robustness runs.
- The low-retention warning remains present for these learned-value evaluations.

