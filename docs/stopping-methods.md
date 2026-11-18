# Comparing stopping policies

Use the following command from the repository root in PowerShell or a POSIX shell. The checked-in smoke splits contain 28 training, three validation and 13 test questions.

```sh
python scripts/compare_stopping_methods.py --data-dir artifacts/smoke --output results/stopping-seed-1 --seed 1
```

Supply another directory containing `train_dataset.json`, `val_dataset.json` and `test_dataset.json` for a larger run. Select a subset with `--methods threshold,fitted_value,double_dqn`. Change the model or horizon with the usual dotted overrides, for example `environment.end_mode=no_buzz`. Use separate output directories for seeds 1, 2 and 3. The default configuration is `configs/smoke.yaml`, with a CPU TF-IDF scorer. The comparison uses undiscounted total reward. `--epochs` sets value sweeps or training passes and `--ppo-steps` sets the PPO interaction budget.

All stopping methods receive the same belief features and commit to the current most probable answer. The scorer and any population-level opponent statistics are fitted on train. Each question contributes the live simulator's observations, BUZZ reward and WAIT reward at every actionable step. This includes the initial uniform belief under the standard environment and its terminal forced answer or abstention. The standard belief features expose normalized progress, which assumes a known reveal horizon. The last scheduled reveal can settle the game automatically. A correct forced answer is counted separately from a voluntary buzz.

| Method | What it learns or measures |
|---|---|
| `threshold` | Selects a confidence cutoff on validation by mean return. |
| `final_buzz`, `wait_to_horizon` | Buzz at the last actionable state, or wait through terminal settlement. |
| `fitted_value` | Regresses both action values on observed belief features using repeated finite-horizon Bellman targets. |
| `stop_imitation` | Fits two teachers on separate training-question folds, then learns from the other fold's teacher at each state. |
| `double_dqn` | Uses offline replay covering both actions, with online next-action selection and a delayed network for target values. [Double DQN](https://arxiv.org/abs/1509.06461) |
| `ppo` | Trains the repository's PPO implementation on the same finite episodes and rewards. |
| `dpo` | Ranks legal stop trajectories within the same question and menu by simulated return and fits the policy relative to a fixed starting actor. [Direct preference optimization](https://arxiv.org/abs/2305.18290) |
| `grpo` | Samples eight trajectories per question, standardizes returns within each group and applies two clipped action-probability updates. The reference KL weight is zero. [Group relative policy optimization](https://arxiv.org/abs/2402.03300) |

`run.json` records the effective settings, split hashes, per-method validation and test results, and paired return differences from the threshold when included. `decisions.json` records each question's first stop, correctness and reward. Learned policies are saved and reloaded for evaluation. The DPO and GRPO training summaries report how many groups supplied a learning signal. Preference labels come from the configured simulator. The smoke split is sized for exercising training, scoring and checkpoint replay.

## Train the text answerer on visible clues

```sh
python scripts/train_t5_policy.py --config configs/t5_policy.yaml --supervised-only supervised.input_mode=prefixes
```

This command trains answer selection on each distinct scheduled cumulative reveal with the displayed menu and gold answer. Each source question stays in its original split. The supervised schedule can include the full question. The stopping benchmark above holds answer selection fixed, so its rewards isolate the decision to wait or buzz. The existing T5 default uses complete questions.

The stopping report weights each dataset row equally. The split audit checks question IDs and normalized text across partitions. Rows within a partition can share clue text. The PPO training receipt lists actual SB3 rollout steps and effective PPO coefficients.
