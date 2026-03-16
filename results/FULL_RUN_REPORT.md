# FULL RUN REPORT

- Commit history:
  - Initial full-scale run: `efe6697`
  - Remaining-phase completion pass: `3256e3c`
- Date(s): `2026-03-15`
- Repo: `/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer`
- Machine: Apple M3 Max, 64 GB unified memory, MPS available
- Python: `3.13.5` in `.venv`
- Preflight:
  - `python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"` -> `MPS: True`
  - `pytest tests/ -q --tb=short` -> `342 passed, 3 skipped`
- Live run windows:
  - Initial run: approximately `13:49` to `16:34` Pacific
  - Remaining-phase completion pass: approximately `19:01` to `19:06` Pacific

## Scope

Executed live at full scale:

- Core: Phases `1, 2, 3, 4, 5, 6`
- Scripted extensions recovered after wrapper abort: `11, 13, 14, 15, 16, 17`
- Manual extensions requested in the prompt: `7, 9, 10`
- Remaining phases completed in a follow-up pass: `8`, `11` EW-trained PPO, `11` empirical EW evaluation, `13` explicit `K=4`

Skipped by instruction or scope:

- `12` DSPy (not wired end-to-end / API-dependent)
- `18` OpenAI embeddings
- `19` DSPy MIPROv2

## Commands Run

```bash
# Verification
source .venv/bin/activate
pip install -e .
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
pytest tests/ -q --tb=short

# Wrapper start
bash scripts/run_full_pipeline.sh --t5-model t5-base

# Manual recovery after wrapper failed at Phase 6
python scripts/compare_policies.py \
    --config configs/t5_policy.yaml \
    --mlp-checkpoint artifacts/main/ppo_model \
    --t5-checkpoint checkpoints/ppo_t5/best_model \
    --mc-path artifacts/main/mc_dataset.json

python scripts/evaluate_all.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=tfidf \
    environment.reward_mode=expected_wins \
    environment.opponent_buzz_model.type=logistic
cp artifacts/main/evaluation_report.json results/eval_ew_logistic.json

python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    environment.belief_mode=sequential_bayes \
    likelihood.model=tfidf
cp artifacts/main/baseline_summary.json results/baselines_seqbayes.json

python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 \
    --deterministic-eval \
    likelihood.model=tfidf \
    environment.reward_mode=simple
cp artifacts/main/ppo_summary.json results/ppo_simple.json

python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 \
    --deterministic-eval \
    likelihood.model=tfidf \
    environment.reward_mode=human_grounded
cp artifacts/main/ppo_summary.json results/ppo_human_grounded.json

python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 \
    --deterministic-eval \
    --policy-mode stop_only \
    likelihood.model=tfidf
cp artifacts/main/ppo_summary.json results/ppo_stop_only.json

python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 \
    --deterministic-eval \
    likelihood.model=tfidf \
    environment.end_mode=no_buzz \
    environment.no_buzz_reward=-0.25
cp artifacts/main/ppo_summary.json results/ppo_no_buzz.json

for K in 2 3 5 6; do
    python scripts/build_mc_dataset.py \
        --config configs/default.yaml \
        --output-dir "artifacts/k$K" \
        data.K="$K" \
        data.distractor_strategy=category_random
    python scripts/run_baselines.py \
        --config configs/default.yaml \
        --mc-path "artifacts/k$K/mc_dataset.json" \
        likelihood.model=tfidf
    cp artifacts/main/baseline_summary.json "results/baselines_k$K.json"
done

# Manual extensions requested after wrapper
for SEED in 1 2 3; do
    python scripts/train_ppo.py \
        --config configs/default.yaml \
        --mc-path artifacts/main/mc_dataset.json \
        --seed "$SEED" \
        --deterministic-eval \
        likelihood.model=tfidf
    cp artifacts/main/ppo_summary.json "results/ppo_seed${SEED}.json"
    cp artifacts/main/ppo_model.zip "results/ppo_model_seed${SEED}.zip"
done

mkdir -p artifacts/distractor_comparison
cp artifacts/main/mc_dataset.json artifacts/distractor_comparison/mc_sbert.json
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/distractor_comparison/tfidf \
    data.distractor_strategy=tfidf_profile
cp artifacts/distractor_comparison/tfidf/mc_dataset.json artifacts/distractor_comparison/mc_tfidf.json
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/distractor_comparison/catrandom \
    data.distractor_strategy=category_random
cp artifacts/distractor_comparison/catrandom/mc_dataset.json artifacts/distractor_comparison/mc_catrandom.json
for STRATEGY in sbert tfidf catrandom; do
    python scripts/run_baselines.py \
        --config configs/default.yaml \
        --mc-path "artifacts/distractor_comparison/mc_${STRATEGY}.json" \
        likelihood.model=tfidf
    cp artifacts/main/baseline_summary.json "results/baselines_distractor_${STRATEGY}.json"
done

mkdir -p artifacts/variable_k
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/variable_k \
    data.variable_K=true \
    data.min_K=2 \
    data.max_K=6 \
    data.K=6 \
    data.distractor_strategy=category_random
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/variable_k/mc_dataset.json \
    likelihood.model=tfidf
cp artifacts/main/baseline_summary.json results/baselines_variable_k.json

# Remaining phases completed in follow-up pass
python scripts/sweep_reward_shaping.py --seeds 13,42,123 --timesteps 3000 \
    | tee results/phase_8_sweep.txt

cp results/baselines_tfidf.json artifacts/main/baseline_summary.json
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 \
    --deterministic-eval \
    likelihood.model=tfidf \
    environment.reward_mode=expected_wins \
    environment.opponent_buzz_model.type=logistic
cp artifacts/main/ppo_summary.json results/ppo_expected_wins.json
cp artifacts/main/ppo_model.zip results/ppo_model_expected_wins.zip

python scripts/evaluate_all.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=tfidf \
    environment.reward_mode=expected_wins \
    environment.opponent_buzz_model.type=empirical
cp artifacts/main/evaluation_report.json results/eval_expected_wins_empirical.json

python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/k4 \
    data.K=4 data.distractor_strategy=category_random
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/k4/mc_dataset.json \
    likelihood.model=tfidf
cp artifacts/main/baseline_summary.json results/baselines_k4.json
```

## Per-Phase Results

| Phase | Name | Approx wall-time | Key metrics | MPS | Status | Notes |
|---|---|---:|---|---|---|---|
| 1 | Build MC dataset | `~22s` | `14,961` MC questions; train/val/test=`10,453 / 2,226 / 2,282` | No | Pass | Wrapper Phase 1 |
| 2 | TF-IDF baselines | `~1m39s` | best `SequentialBayes@0.5`: acc=`0.9697`, `S_q=0.7413`, `ECE=0.3470`, `Brier=0.1535` | No | Pass | Archived to `results/baselines_tfidf.json` |
| 3 | PPO default | `~2m50s` | acc=`0.2525`, `S_q=0.2525`, reward-like=`-0.2777` | No | Pass | Degenerate default policy |
| 4 | Evaluate all | `~1m48s` | `SoftmaxProfile@0.5`: acc=`0.9748`, `S_q=0.6858`, `ECE=0.4555`, `Brier=0.2326` | No | Pass | Peak observed RSS during eval was ~32 GB |
| 5 | T5 policy supervised + PPO | `~1h58m` | supervised best val acc=`0.2652`; test acc=`0.94`; avg reward=`0.416`; avg episode length=`4.94` | Yes | Pass | Long pole of the run; logs update only at epoch boundaries during supervised warm-start |
| 6 | Compare policies | initial wrapper attempt failed; rerun `~4m50s` | MLP(T5-likelihood): acc=`0.2546`, `S_q=0.2546`; T5 policy: acc=`0.9334`, `S_q=0.2394` | Yes | Pass after fix | Wrapper failed on Apple Silicon MPS; recovered manually after patch |
| 7 | PPO multi-seed | `~6m` total | seed1 acc=`0.2445`, `S_q=0.2445`; seed2 acc=`0.2445`, `S_q=0.2445`; seed3 acc=`0.9755`, `S_q=0.000069` | No | Pass | Very high variance / multimodal collapse |
| 8 | Reward sweep (smoke) | `~2m` | best config `wait_penalty=0.0`, `early_buzz_penalty=0.8`; acc=`0.3712`, `S_q=0.2914`, objective=`0.6219` | No | Pass | Captured to `results/phase_8_sweep.txt`; script also wrote `artifacts/smoke/reward_sweep_results.{json,csv}` |
| 9 | Distractor comparison | `~7m` total | `SBERT`: best seqbayes `S_q=0.7413`; `tfidf_profile`: `0.6651`; `category_random`: `0.7962` | No | Pass | Category-random performed best on this run |
| 10 | Variable-K baselines | `~1m25s` | mixed-K distribution `{2:3875,3:3492,4:3232,5:2859,6:2585}`; best seqbayes `S_q=0.7804 @ 0.6` | No | Pass | Baselines only; PPO/MaskablePPO still not wired |
| 11 | Expected Wins eval | `~1m42s` | `mean_ew=4.4929` vs logistic opponent; `full_eval` metrics unchanged from Phase 4 report | No | Pass | Evaluates baseline agents, not PPO |
| 11b | EW-trained PPO | `~1m25s` | acc=`0.2525`, `S_q=0.2525`, reward-like=`-0.2777` | No | Pass | Converged to the same degenerate regime as default PPO |
| 11c | Expected Wins eval (empirical opponent) | `~1m00s` | `mean_ew=4.4929`; `full_eval` acc=`0.9748`, `S_q=0.6858` | No | Pass | Empirical-opponent EW matched logistic-opponent EW in this run |
| 13 | K-sensitivity | `~4m` total | best seqbayes `S_q`: `K=2 0.7765`, `K=3 0.7946`, `K=4 0.7413`, `K=5 0.8000`, `K=6 0.7983` | No | Pass | Original `K=4` point comes from default Phase 2 SBERT-distractor baseline |
| 13b | K=4 explicit category-random run | `~1m29s` | best seqbayes `S_q=0.7962 @ 0.6`; acc=`0.9942` | No | Pass | Controlled `K=4` comparison aligned with the category-random `K=2,3,5,6` runs |
| 14 | Reward-mode PPO ablations | `~3m` total | `simple` same as default (`S_q=0.2525`); `human_grounded` same as default (`S_q=0.2525`) | No | Pass | No improvement over default PPO on this configuration |
| 15 | Belief-mode comparison | `~1m06s` | `sequential_bayes` best `S_q=0.7413 @ 0.5` | No | Pass | Strongest belief-feature baseline in the run |
| 16 | Stop-only PPO | initial run failed; rerun `~1m25s` | acc=`0.9755`, `S_q=0.000029`, reward-like=`0.7174` | No | Pass after fix | Training succeeds, but policy is unusable by `S_q` despite high accuracy |
| 17 | No-buzz horizon PPO | `~1m20s` | acc=`0.2515`, `S_q=0.2515`, reward-like=`-0.2791` | No | Pass | No meaningful improvement over default |

## Runbook Issues Found

| Severity | Section | What went wrong | What I did instead | Suggested fix |
|---|---|---|---|---|
| High | `Phase 6: Compare policies` / wrapper path | Following the wrapper literally failed on Apple Silicon with `RuntimeError: Placeholder storage has not been allocated on MPS device!` because `compare_policies.py` tokenized observations onto CPU and then called the MPS-loaded T5 policy. | Patched `scripts/compare_policies.py` to use `model.encode_input([obs])`, then reran Phase 6 manually. | Keep the patch; add an explicit regression test around device placement in Phase 6. |
| High | `Phase 16: Stop-only PPO` | The stop-only ablation crashed at evaluation time with `AttributeError: 'StopOnlyEnv' object has no attribute 'question'` in `agents/ppo_buzzer.py`. | Patched `PPOBuzzer.run_episode()` to read `gold_index` from the unwrapped base env and strengthened `tests/test_ppo_buzzer.py` to mimic an actual wrapper without `.question` on the wrapper itself. | Keep the patch; ensure the test remains in the main repo and not only a worktree. |
| Medium | `Phase 5` wall-time / monitoring guidance | `t5-base` full-scale training was much slower than the optimistic table suggests. The run took nearly two hours, and the log remained quiet between epoch boundaries during supervised warm-start. | Let the run continue and used `ps`/process liveness instead of log churn as the health signal. | Update the Phase 5 time estimate upward for Apple Silicon full-scale runs and note that log output during supervised warm-start is sparse even with unbuffered Python. |
| Medium | `Phase 4` evaluation expectations | Full evaluation was memory-heavy on the full dataset, peaking around 32 GB RSS while running controls after the main evaluation. | Waited for the process to complete and recorded the observed memory footprint. | Add a memory note for `evaluate_all.py` on full-scale runs. |
| Low | `Phase 8` reward sweep documentation | The follow-up prompt treated Phase 8 as stdout-only, but the script also wrote `artifacts/smoke/reward_sweep_results.json` and `.csv` in addition to the captured stdout. | Captured stdout to `results/phase_8_sweep.txt` and used the JSON artifact in `artifacts/smoke/` to verify the best configuration. | Document the extra smoke-scale artifacts so follow-up analysis does not rely on parsing raw stdout alone. |

## Final Results Summary

### Baseline agents (TF-IDF belief-feature run)

| Agent | Best threshold | Accuracy | Mean buzz step | `S_q` | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| ThresholdBuzzer | `0.5` | `0.9748` | `3.1767` | `0.6858` | `0.4555` | `0.2326` |
| SoftmaxProfileBuzzer | `0.5` | `0.9748` | `3.1767` | `0.6858` | `0.4555` | `0.2326` |
| SequentialBayesBuzzer | `0.5` | `0.9697` | `2.3793` | `0.7413` | `0.3470` | `0.1535` |

Best baseline by `S_q`: `SequentialBayesBuzzer` (`0.7413`).

### PPO vs baseline

| Policy | Accuracy | `S_q` | Reward-like | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| PPO default | `0.2525` | `0.2525` | `-0.2777` | `0.0025` | `0.1887` |
| Best TF-IDF baseline (SequentialBayes) | `0.9697` | `0.7413` | `0.0` | `0.3470` | `0.1535` |

Conclusion: default PPO is not competitive with the baseline suite on this full-scale run.

### T5 policy vs MLP policy (Phase 6)

| Model | Accuracy | `S_q` | ECE | Brier | Avg buzz pos |
|---|---:|---:|---:|---:|---:|
| MLP policy (T5-as-likelihood) | `0.2546` | `0.2546` | `0.0046` | `0.1898` | `0.0000` |
| T5 policy (end-to-end) | `0.9334` | `0.2394` | `0.6816` | `0.5268` | `3.8966` |

Takeaway: the T5 policy is much better at eventual correctness, but its calibration and `S_q` are poor relative to that raw accuracy.

### Expected Wins

- Logistic-opponent Expected Wins evaluation: `mean_ew = 4.4929`
- Empirical-opponent Expected Wins evaluation: `mean_ew = 4.4929`
- The `full_eval` metrics in `results/eval_ew_logistic.json` and `results/eval_expected_wins_empirical.json` are otherwise the same SoftmaxProfile baseline metrics reported in Phase 4.
- EW-trained PPO converged to the same degenerate solution as the default PPO run: acc=`0.2525`, `S_q=0.2525`, reward-like=`-0.2777`.

### Reward-mode ablations (Phase 14)

| Mode | Accuracy | `S_q` | Reward-like |
|---|---:|---:|---:|
| Default / time-penalty PPO | `0.2525` | `0.2525` | `-0.2777` |
| `simple` | `0.2525` | `0.2525` | `-0.2777` |
| `human_grounded` | `0.2525` | `0.2525` | `-0.2777` |

No improvement observed from the reward-mode switch under this training budget.

### Belief-mode ablation (Phase 15)

| Belief mode | Best threshold | Accuracy | `S_q` |
|---|---:|---:|---:|
| `from_scratch` / default SoftmaxProfile | `0.5` | `0.9748` | `0.6858` |
| `sequential_bayes` | `0.5` | `0.9697` | `0.7413` |

Sequential Bayes belief tracking materially improves `S_q`.

### Policy-mode ablation (Phase 16)

| Mode | Accuracy | `S_q` | Reward-like |
|---|---:|---:|---:|
| Default PPO | `0.2525` | `0.2525` | `-0.2777` |
| `stop_only` | `0.9755` | `0.000029` | `0.7174` |

Stop-only PPO learns to delay and then commit correctly enough to maximize reward-like accuracy, but collapses under the `S_q` objective.

### Horizon ablation (Phase 17)

| Horizon mode | Accuracy | `S_q` | Reward-like |
|---|---:|---:|---:|
| `force_commit` (default) | `0.2525` | `0.2525` | `-0.2777` |
| `no_buzz` | `0.2515` | `0.2515` | `-0.2791` |

No useful gain from the no-buzz horizon on this run.

### K-sensitivity (Phase 13)

`K=4` appeared in two forms in this project:

- the default Phase 2 dataset with SBERT distractors, and
- a follow-up explicit `K=4` run with category-random distractors to match the `K=2,3,5,6` follow-up runs.

| Fixed K | Best baseline | Best threshold | Accuracy | `S_q` |
|---|---|---:|---:|---:|
| 2 | SequentialBayes | `0.7` | `0.9971` | `0.7765` |
| 3 | SequentialBayes | `0.6` | `0.9939` | `0.7946` |
| 4 (default SBERT) | SequentialBayes | `0.5` | `0.9697` | `0.7413` |
| 4 (explicit category-random) | SequentialBayes | `0.6` | `0.9942` | `0.7962` |
| 5 | SequentialBayes | `0.5` | `0.9918` | `0.8000` |
| 6 | SequentialBayes | `0.5` | `0.9908` | `0.7983` |

Observed pattern: the original default `K=4` point looked artificially weak because it used SBERT distractors. Under the controlled category-random comparison, `K=4` improved to `S_q=0.7962`, much closer to the `K=3/5/6` points.

### Multi-seed PPO variance (Phase 7)

| Seed | Accuracy | `S_q` | Reward-like |
|---|---:|---:|---:|
| 1 | `0.2445` | `0.2445` | `-0.2896` |
| 2 | `0.2445` | `0.2445` | `-0.2896` |
| 3 | `0.9755` | `0.000069` | `0.7174` |

Aggregate:

- Mean accuracy: `0.4882`
- Accuracy std: `0.3446`
- Mean `S_q`: `0.1630`
- `S_q` population std: `0.1152` (n=3)
- Mean reward-like: `0.0461`
- Reward-like std: `0.4747`

Takeaway: PPO is highly unstable; it converged to at least two qualitatively different degenerate modes across seeds.

### Distractor strategy comparison (Phase 9)

| Distractor strategy | Best baseline | Best threshold | Accuracy | `S_q` |
|---|---|---:|---:|---:|
| SBERT semantic ranking | SequentialBayes | `0.5` | `0.9697` | `0.7413` |
| TF-IDF profile ranking | SequentialBayes | `0.5` | `0.9274` | `0.6651` |
| Category-random | SequentialBayes | `0.6` | `0.9942` | `0.7962` |

Unexpectedly, category-random distractors were easiest for the TF-IDF baseline family in this run, while TF-IDF-profile distractors were the hardest.

### Variable-K mixed dataset (Phase 10)

- K distribution: `{2: 3875, 3: 3492, 4: 3232, 5: 2859, 6: 2585}`
- Best result: `SequentialBayes @ 0.6` with accuracy `0.9937` and `S_q = 0.7804`

## Artifact Inventory

```text
artifacts/main:
total 2077280
answer_profiles.json                  361K
baseline_floor_runs.json             9.2M
baseline_sequential_bayes_runs.json   44M
baseline_softmax_profile_runs.json    49M
baseline_summary.json                4.8K
baseline_threshold_runs.json          51M
evaluation_report.json                15K
mc_dataset.json                      422M
plots/                               160B
ppo_model.zip                        150K
ppo_runs.json                         11M
ppo_summary.json                     263B
test_dataset.json                     64M
train_dataset.json                   295M
val_dataset.json                      63M

artifacts/main/plots:
total 96
-rw-r--r--@ 1 ankit.aggarwal  staff    20K Mar 15 16:04 calibration.png
-rw-r--r--@ 1 ankit.aggarwal  staff   1.9K Mar 15 16:04 comparison.csv
-rw-r--r--@ 1 ankit.aggarwal  staff    23K Mar 15 16:04 entropy_vs_clue.png

checkpoints:
total 0
drwxr-xr-x@ 10 ankit.aggarwal  staff   320B Mar 15 05:34 ppo_t5
drwxr-xr-x@  4 ankit.aggarwal  staff   128B Mar 15 05:27 supervised

checkpoints/ppo_t5:
total 16
drwxr-xr-x@ 8 ankit.aggarwal  staff   256B Mar 15 05:28 best_model
-rw-r--r--@ 1 ankit.aggarwal  staff   3.4K Mar 15 15:48 history.json
drwxr-xr-x@ 8 ankit.aggarwal  staff   256B Mar 15 05:33 iter_100
drwxr-xr-x@ 8 ankit.aggarwal  staff   256B Mar 15 05:29 iter_20
drwxr-xr-x@ 8 ankit.aggarwal  staff   256B Mar 15 05:30 iter_40
drwxr-xr-x@ 8 ankit.aggarwal  staff   256B Mar 15 05:31 iter_60
drwxr-xr-x@ 8 ankit.aggarwal  staff   256B Mar 15 05:32 iter_80
-rw-r--r--@ 1 ankit.aggarwal  staff   223B Mar 15 15:48 test_results.json

checkpoints/ppo_t5/best_model:
total 2592912
-rw-r--r--@ 1 ankit.aggarwal  staff   1.5K Mar 15 15:40 config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   418M Mar 15 15:40 model.safetensors
-rw-r--r--@ 1 ankit.aggarwal  staff   3.0M Mar 15 15:40 policy_head.pt
-rw-r--r--@ 1 ankit.aggarwal  staff   2.3M Mar 15 15:40 tokenizer.json
-rw-r--r--@ 1 ankit.aggarwal  staff   2.5K Mar 15 15:40 tokenizer_config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   843M Mar 15 15:40 training_state.pt

checkpoints/ppo_t5/iter_100:
total 2592912
-rw-r--r--@ 1 ankit.aggarwal  staff   1.5K Mar 15 15:48 config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   418M Mar 15 15:48 model.safetensors
-rw-r--r--@ 1 ankit.aggarwal  staff   3.0M Mar 15 15:48 policy_head.pt
-rw-r--r--@ 1 ankit.aggarwal  staff   2.3M Mar 15 15:48 tokenizer.json
-rw-r--r--@ 1 ankit.aggarwal  staff   2.5K Mar 15 15:48 tokenizer_config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   843M Mar 15 15:48 training_state.pt

checkpoints/ppo_t5/iter_20:
total 2592912
-rw-r--r--@ 1 ankit.aggarwal  staff   1.5K Mar 15 15:41 config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   418M Mar 15 15:41 model.safetensors
-rw-r--r--@ 1 ankit.aggarwal  staff   3.0M Mar 15 15:41 policy_head.pt
-rw-r--r--@ 1 ankit.aggarwal  staff   2.3M Mar 15 15:41 tokenizer.json
-rw-r--r--@ 1 ankit.aggarwal  staff   2.5K Mar 15 15:41 tokenizer_config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   843M Mar 15 15:41 training_state.pt

checkpoints/ppo_t5/iter_40:
total 2592912
-rw-r--r--@ 1 ankit.aggarwal  staff   1.5K Mar 15 15:44 config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   418M Mar 15 15:44 model.safetensors
-rw-r--r--@ 1 ankit.aggarwal  staff   3.0M Mar 15 15:44 policy_head.pt
-rw-r--r--@ 1 ankit.aggarwal  staff   2.3M Mar 15 15:44 tokenizer.json
-rw-r--r--@ 1 ankit.aggarwal  staff   2.5K Mar 15 15:44 tokenizer_config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   843M Mar 15 15:44 training_state.pt

checkpoints/ppo_t5/iter_60:
total 2592912
-rw-r--r--@ 1 ankit.aggarwal  staff   1.5K Mar 15 15:45 config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   418M Mar 15 15:45 model.safetensors
-rw-r--r--@ 1 ankit.aggarwal  staff   3.0M Mar 15 15:45 policy_head.pt
-rw-r--r--@ 1 ankit.aggarwal  staff   2.3M Mar 15 15:45 tokenizer.json
-rw-r--r--@ 1 ankit.aggarwal  staff   2.5K Mar 15 15:45 tokenizer_config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   843M Mar 15 15:45 training_state.pt

checkpoints/ppo_t5/iter_80:
total 2592912
-rw-r--r--@ 1 ankit.aggarwal  staff   1.5K Mar 15 15:46 config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   418M Mar 15 15:46 model.safetensors
-rw-r--r--@ 1 ankit.aggarwal  staff   3.0M Mar 15 15:46 policy_head.pt
-rw-r--r--@ 1 ankit.aggarwal  staff   2.3M Mar 15 15:46 tokenizer.json
-rw-r--r--@ 1 ankit.aggarwal  staff   2.5K Mar 15 15:46 tokenizer_config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   843M Mar 15 15:46 training_state.pt

checkpoints/supervised:
total 8
drwxr-xr-x@ 8 ankit.aggarwal  staff   256B Mar 15 04:40 best_model
-rw-r--r--@ 1 ankit.aggarwal  staff   2.1K Mar 15 15:23 history.json

checkpoints/supervised/best_model:
total 2586736
-rw-r--r--@ 1 ankit.aggarwal  staff   1.5K Mar 15 15:09 config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   418M Mar 15 15:09 model.safetensors
-rw-r--r--@ 1 ankit.aggarwal  staff   3.0M Mar 15 15:09 policy_head.pt
-rw-r--r--@ 1 ankit.aggarwal  staff   2.3M Mar 15 15:09 tokenizer.json
-rw-r--r--@ 1 ankit.aggarwal  staff   2.3K Mar 15 15:09 tokenizer_config.json
-rw-r--r--@ 1 ankit.aggarwal  staff   840M Mar 15 15:09 training_state.pt

results:
total 3032
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 16:32 baselines_distractor_catrandom.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 16:30 baselines_distractor_sbert.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 16:31 baselines_distractor_tfidf.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.7K Mar 15 16:18 baselines_k2.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 16:19 baselines_k3.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 19:06 baselines_k4.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 16:21 baselines_k5.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 16:22 baselines_k6.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 16:05 baselines_seqbayes.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 13:51 baselines_tfidf.json
-rw-r--r--@ 1 ankit.aggarwal  staff   4.8K Mar 15 16:34 baselines_variable_k.json
-rw-r--r--@ 1 ankit.aggarwal  staff    15K Mar 15 15:50 eval_default.json
-rw-r--r--@ 1 ankit.aggarwal  staff    15K Mar 15 16:04 eval_ew_logistic.json
-rw-r--r--@ 1 ankit.aggarwal  staff    15K Mar 15 19:04 eval_expected_wins_empirical.json
-rw-r--r--@ 1 ankit.aggarwal  staff    37K Mar 15 13:51 phase_2.log
-rw-r--r--@ 1 ankit.aggarwal  staff   657K Mar 15 13:52 phase_3.log
-rw-r--r--@ 1 ankit.aggarwal  staff    24K Mar 15 15:50 phase_4.log
-rw-r--r--@ 1 ankit.aggarwal  staff    26K Mar 15 15:48 phase_5.log
-rw-r--r--@ 1 ankit.aggarwal  staff   4.9K Mar 15 15:50 phase_6.log
-rw-r--r--@ 1 ankit.aggarwal  staff   2.2M Mar 15 19:01 phase_8_sweep.txt
-rw-r--r--@ 1 ankit.aggarwal  staff   252B Mar 15 13:52 ppo_default.json
-rw-r--r--@ 1 ankit.aggarwal  staff   252B Mar 15 19:02 ppo_expected_wins.json
-rw-r--r--@ 1 ankit.aggarwal  staff   252B Mar 15 16:10 ppo_human_grounded.json
-rw-r--r--@ 1 ankit.aggarwal  staff   150K Mar 15 13:52 ppo_model_default.zip
-rw-r--r--@ 1 ankit.aggarwal  staff   150K Mar 15 19:02 ppo_model_expected_wins.zip
-rw-r--r--@ 1 ankit.aggarwal  staff   150K Mar 15 16:25 ppo_model_seed1.zip
-rw-r--r--@ 1 ankit.aggarwal  staff   150K Mar 15 16:27 ppo_model_seed2.zip
-rw-r--r--@ 1 ankit.aggarwal  staff   150K Mar 15 16:28 ppo_model_seed3.zip
-rw-r--r--@ 1 ankit.aggarwal  staff   249B Mar 15 16:16 ppo_no_buzz.json
-rw-r--r--@ 1 ankit.aggarwal  staff   251B Mar 15 16:25 ppo_seed1.json
-rw-r--r--@ 1 ankit.aggarwal  staff   251B Mar 15 16:27 ppo_seed2.json
-rw-r--r--@ 1 ankit.aggarwal  staff   263B Mar 15 16:28 ppo_seed3.json
-rw-r--r--@ 1 ankit.aggarwal  staff   252B Mar 15 16:08 ppo_simple.json
-rw-r--r--@ 1 ankit.aggarwal  staff   263B Mar 15 16:14 ppo_stop_only.json
-rw-r--r--@ 1 ankit.aggarwal  staff   717B Mar 15 16:02 t5_comparison.json

artifacts/smoke:
total 96
-rw-r--r--@ 1 ankit.aggarwal  staff    14K Mar 15 19:00 reward_sweep_results.csv
-rw-r--r--@ 1 ankit.aggarwal  staff    25K Mar 15 19:00 reward_sweep_results.json
```

## Post-run validation

- `pytest tests/ -q --tb=short` -> `342 passed, 3 skipped`
- Targeted tests after the two live fixes -> `44 passed, 1 skipped`
- Remaining-phase summary snippet run after completion and new artifacts verified in `results/`
- No mixed likelihood regime was used in the executed comparisons:
  - TF-IDF for all belief-feature phases
  - `t5-base` only for the end-to-end T5 policy pipeline
