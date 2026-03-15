# CoVe Validation for qanta-buzzer E2E data-flow animation

This bundle was generated from verified source paths in the local repo.
The scene grouping is an inference over entrypoints, but each animated
claim below is backed by source evidence.

## Verdict

- All constituent claims used in the animation were validated from source.
- Scene grouping into eight flows is an inference from the verified scripts and trainers.
- Known caveat carried into the animation: alias substitution currently falls back to an empty lookup in this end-to-end path.

## Verified claims

- `C1` The build_mc_dataset entrypoint loads questions from CSV when present, falls back to HuggingFace only if configured, builds answer profiles, constructs MC questions, creates stratified splits, and saves dataset JSONs.
  Verdict: `Verified`
  Evidence:
  - `scripts/build_mc_dataset.py:267-355`
  - `qb_data/answer_profiles.py:11-142`
  - `qb_data/dataset_splits.py:create_stratified_splits`

- `C2` MCBuilder supports category_random, tfidf_profile, sbert_profile, and openai_profile distractor strategies and applies four anti-artifact guards.
  Verdict: `Verified`
  Evidence:
  - `qb_data/mc_builder.py:63-97`
  - `qb_data/mc_builder.py:205-249`
  - `qb_data/mc_builder.py:251-367`

- `C3` Question history is represented as run_indices plus cumulative_prefixes, where each prefix is the text revealed up to a clue boundary.
  Verdict: `Verified`
  Evidence:
  - `qb_data/data_loader.py:65-74`
  - `qb_data/data_loader.py:156-168`

- `C4` The belief-feature path builds a configurable likelihood model over MC questions; supported backends are tfidf, sbert, openai, and t5 variants.
  Verdict: `Verified`
  Evidence:
  - `scripts/_common.py:44-53`
  - `models/likelihoods.py:374-731`

- `C5` TossupMCEnv supports from_scratch belief recomputation from cumulative prefixes and sequential_bayes updates from newly revealed fragments, then exposes belief features [belief..., top_p, margin, entropy, stability, progress, clue_idx_norm].
  Verdict: `Verified`
  Evidence:
  - `qb_env/tossup_env.py:93-114`
  - `qb_env/tossup_env.py:151-158`
  - `qb_env/tossup_env.py:351-388`
  - `models/features.py:50-108`

- `C6` run_baselines executes ThresholdBuzzer, SoftmaxProfileBuzzer, SequentialBayesBuzzer, and AlwaysBuzzFinal, then saves per-agent runs and a baseline_summary artifact.
  Verdict: `Verified`
  Evidence:
  - `scripts/run_baselines.py:5-12`
  - `scripts/run_baselines.py:141-225`

- `C7` train_ppo loads MC questions, builds a likelihood model, precomputes belief trajectories, creates TossupMCEnv from config, trains PPOBuzzer with SB3 MlpPolicy, and writes ppo_model, ppo_runs, and ppo_summary.
  Verdict: `Verified`
  Evidence:
  - `scripts/train_ppo.py:82-176`
  - `agents/ppo_buzzer.py:68-120`
  - `qb_env/tossup_env.py:598-660`

- `C8` evaluate_all loads MC questions and baseline outputs, picks the best softmax threshold from baseline_summary, runs full evaluation plus choices-only, shuffle, and alias controls, then writes evaluation_report and plotting artifacts.
  Verdict: `Verified`
  Evidence:
  - `scripts/evaluate_all.py:87-115`
  - `scripts/evaluate_all.py:152-255`
  - `scripts/evaluate_all.py:258-328`

- `C9` In the current end-to-end pipeline, alias substitution is effectively fed by an empty lookup when alias_lookup.json is absent, and build_mc_dataset.py does not write that file.
  Verdict: `Verified`
  Evidence:
  - `scripts/evaluate_all.py:158-163`
  - `scripts/build_mc_dataset.py:338-355`

- `C10` train_t5_policy loads an MC dataset, splits questions into train/val/test, runs supervised warm-start on complete-question text, then PPO fine-tunes a T5PolicyModel on incremental text observations.
  Verdict: `Verified`
  Evidence:
  - `scripts/train_t5_policy.py:1-236`
  - `training/train_supervised_t5.py:52-72`
  - `training/train_ppo_t5.py:299-380`

- `C11` TextObservationWrapper exposes visible history as 'CLUES: <prefix> | CHOICES: ...', and T5PolicyModel consumes that text to produce wait logits, answer logits, and values.
  Verdict: `Verified`
  Evidence:
  - `qb_env/text_wrapper.py:70-120`
  - `models/t5_policy.py:132-216`
  - `models/t5_policy.py:321-396`

- `C12` T5 PPO rollouts still instantiate TossupMCEnv with a TF-IDF likelihood for environment scoring/reward computation, while the T5 policy itself reads text observations through TextObservationWrapper.
  Verdict: `Verified`
  Evidence:
  - `training/train_ppo_t5.py:317-345`
  - `training/train_ppo_t5.py:351-380`

- `C13` compare_policies evaluates an MLP PPO policy on belief features and a T5 policy on text observations, then saves a comparison JSON.
  Verdict: `Verified`
  Evidence:
  - `scripts/compare_policies.py:58-142`
  - `scripts/compare_policies.py:145-260`
  - `scripts/compare_policies.py:393-464`

## Scene mapping

- `Build MC dataset` -> C1, C2, C3
- `Belief construction branch` -> C3, C4, C5
- `Baseline sweep` -> C4, C5, C6
- `Belief-feature PPO` -> C4, C5, C7
- `Comprehensive evaluation` -> C8, C9
- `T5 supervised warm-start` -> C10, C11
- `T5 PPO fine-tuning` -> C10, C11, C12
- `Comparison experiment` -> C12, C13