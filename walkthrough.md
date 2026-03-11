# Quiz Bowl RL Buzzer - End-to-End Walkthrough

*2026-03-11T02:08:30Z*

## Repo orientation

This walkthrough exercises both the modular belief-feature pipeline and the T5 policy pipeline in smoke mode. All commands are run from the repo root with the project venv activated.

```bash
ls -1 *.py *.md *.yaml *.toml 2>/dev/null | head -20
```

```output
CLAUDE.md
config.py
dataset.py
demo.py
environment.py
IMPLEMENTATION_README.md
main.py
metrics.py
model.py
PRESENTATION.md
PROJECT_OVERVIEW.md
pyproject.toml
README.md
test_csv_loader.py
test_imports.py
train_ppo.py
train_supervised.py
verify_data_loader.py
visualize.py
walkthrough.md
```

```bash
head -12 CLAUDE.md
```

```output
# CLAUDE.md

This file provides repo-local guidance for Claude Code and other coding agents.

## Project Overview

Stanford CS234 final project: a unified quiz bowl RL buzzer system with two tracks:

1. Belief-feature pipeline: build MC tossups, score answer profiles with TF-IDF / SBERT / T5 / optional OpenAI embeddings, train or compare buzzers, and evaluate with S_q plus calibration metrics.
2. T5 policy pipeline: supervised warm-start and PPO for an end-to-end text policy.

`qanta-buzzer` is the canonical repo. qb-rl compatibility is preserved through additive shims rather than structural rewrites.
```

## Belief-feature smoke pipeline

Four stages: build MC dataset, run baselines, train PPO, evaluate all. Uses TF-IDF likelihood for speed.

```bash
source .venv/bin/activate && python scripts/build_mc_dataset.py --smoke
```

```output
Loading configuration...

Loading questions...
Loading from CSV: questions.csv
Loaded 20407 questions from CSV
Limiting dataset to 50 questions

Building answer profiles...
Built 42 answer profiles

Constructing MC questions...
Generated 44 MC questions
Note: 6 questions filtered by guards

Creating stratified splits...
Dataset split complete:
  Train: 28 questions (63.6%)
  Val:   3 questions (6.8%)
  Test:  13 questions (29.5%)

Category distribution (11 categories):
  Fine_Arts: 4/1/2 (orig: 7)
  Fine_Arts:Music: 1/0/0 (orig: 1)
  History: 2/0/2 (orig: 4)
  Literature: 4/0/2 (orig: 6)
  Literature:Europe: 1/1/0 (orig: 2)
  ... and 6 more categories

Saving datasets...
Saved 44 items to artifacts/smoke/mc_dataset.json
Saved 28 items to artifacts/smoke/train_dataset.json
Saved 3 items to artifacts/smoke/val_dataset.json
Saved 13 items to artifacts/smoke/test_dataset.json
Saved answer profiles to artifacts/smoke/answer_profiles.json

============================================================
Dataset Construction Complete
============================================================

Total MC questions: 44
  Train: 28 (63.6%)
  Val:   3 (6.8%)
  Test:  13 (29.5%)

Categories: 11
Sample categories: Fine_Arts, Fine_Arts:Music, History, Literature, Literature:Europe

Answer profiles: 42
Average questions per answer: 1.2

Sample MC question:
  Question: A Frost diagram plots oxidation state against the relative value of this quantity, which can be writ...
  Correct answer: Gibbs free energy
  Options: Tyr, Josephson effect, Gibbs free energy...
  Category: Science:Chemistry

Total time: 0.6 seconds

============================================================
Sample MC Questions (Smoke Test)
============================================================

Question 1:
  First clue: A Frost diagram plots oxidation state against the relative value of this quantity, which can be writ...
  Category: Science:Chemistry
  Correct: Gibbs free energy
  Options: Tyr, Josephson effect, Gibbs free energy...

Question 2:
  First clue: A carbon alpha to two carbons with this functionality is alkylated and then decarboxylated in a reac...
  Category: Science:Chemistry
  Correct: Ester
  Options: Shiva, Ester, Maria Theresa...

Question 3:
  First clue: Setting the partial derivative of this quantity equal to zero will allow one to arrive at the standa...
  Category: Science:Chemistry
  Correct: Gibbs free energy
  Options: Gibbs free energy, Tyr, Josephson effect...

Dataset construction complete!
```

```bash
source .venv/bin/activate && python scripts/run_baselines.py --smoke --mc-path artifacts/smoke/mc_dataset.json
```

```output
Loading MC questions from: artifacts/smoke/mc_dataset.json
Loaded 44 MC questions
Building likelihood model: tfidf
Beta: 5.0, Alpha: 10.0
Thresholds: [0.5, 0.7, 0.9]

Running ThresholdBuzzer sweep...
Running SoftmaxProfile and SequentialBayes at threshold=0.5...
Running SoftmaxProfile and SequentialBayes at threshold=0.7...
Running SoftmaxProfile and SequentialBayes at threshold=0.9...
Running AlwaysBuzzFinal baseline...

Saving artifacts to: /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/artifacts/smoke

Wrote baseline outputs to: /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/artifacts/smoke
Total time: 1.4 seconds

--- Summary ---
  threshold[0.5]: accuracy=0.386, mean_sq=0.243
  threshold[0.7]: accuracy=0.386, mean_sq=0.130
  threshold[0.9]: accuracy=0.386, mean_sq=0.053
  softmax_profile[0.5]: accuracy=0.386, mean_sq=0.243
  softmax_profile[0.7]: accuracy=0.386, mean_sq=0.130
  softmax_profile[0.9]: accuracy=0.386, mean_sq=0.053
  sequential_bayes[0.5]: accuracy=0.386, mean_sq=0.267
  sequential_bayes[0.7]: accuracy=0.386, mean_sq=0.212
  sequential_bayes[0.9]: accuracy=0.386, mean_sq=0.141
  always_final: accuracy=0.386, mean_sq=0.386
```

```bash
source .venv/bin/activate && python scripts/train_ppo.py --smoke --mc-path artifacts/smoke/mc_dataset.json
```

```output
Loading MC questions from: artifacts/smoke/mc_dataset.json
Loaded 44 MC questions
Building likelihood model: tfidf
Training PPO for 3000 timesteps...
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1.19     |
|    ep_rew_mean     | -0.799   |
| time/              |          |
|    fps             | 948      |
|    iterations      | 1        |
|    time_elapsed    | 0        |
|    total_timesteps | 32       |
---------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.21         |
|    ep_rew_mean          | -0.627       |
| time/                   |              |
|    fps                  | 508          |
|    iterations           | 2            |
|    time_elapsed         | 0            |
|    total_timesteps      | 64           |
| train/                  |              |
|    approx_kl            | 9.135157e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.61        |
|    explained_variance   | -0.0164      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.683        |
|    n_updates            | 2            |
|    policy_gradient_loss | -0.00215     |
|    value_loss           | 1.6          |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.2           |
|    ep_rew_mean          | -0.686        |
| time/                   |               |
|    fps                  | 648           |
|    iterations           | 3             |
|    time_elapsed         | 0             |
|    total_timesteps      | 96            |
| train/                  |               |
|    approx_kl            | 4.4781715e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.61         |
|    explained_variance   | 0.00233       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.588         |
|    n_updates            | 4             |
|    policy_gradient_loss | -0.00024      |
|    value_loss           | 1.09          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.22          |
|    ep_rew_mean          | -0.627        |
| time/                   |               |
|    fps                  | 766           |
|    iterations           | 4             |
|    time_elapsed         | 0             |
|    total_timesteps      | 128           |
| train/                  |               |
|    approx_kl            | 2.5834888e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.61         |
|    explained_variance   | -0.0448       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.475         |
|    n_updates            | 6             |
|    policy_gradient_loss | -0.00084      |
|    value_loss           | 1.03          |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.23         |
|    ep_rew_mean          | -0.587       |
| time/                   |              |
|    fps                  | 850          |
|    iterations           | 5            |
|    time_elapsed         | 0            |
|    total_timesteps      | 160          |
| train/                  |              |
|    approx_kl            | 3.953837e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.61        |
|    explained_variance   | -0.033       |
|    learning_rate        | 0.0003       |
|    loss                 | 0.388        |
|    n_updates            | 8            |
|    policy_gradient_loss | -0.000781    |
|    value_loss           | 0.945        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.18          |
|    ep_rew_mean          | -0.766        |
| time/                   |               |
|    fps                  | 924           |
|    iterations           | 6             |
|    time_elapsed         | 0             |
|    total_timesteps      | 192           |
| train/                  |               |
|    approx_kl            | 1.4370307e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.61         |
|    explained_variance   | -0.0283       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.401         |
|    n_updates            | 10            |
|    policy_gradient_loss | -0.000218     |
|    value_loss           | 0.847         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.26          |
|    ep_rew_mean          | -0.709        |
| time/                   |               |
|    fps                  | 959           |
|    iterations           | 7             |
|    time_elapsed         | 0             |
|    total_timesteps      | 224           |
| train/                  |               |
|    approx_kl            | 4.8203394e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.61         |
|    explained_variance   | -0.0165       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.378         |
|    n_updates            | 12            |
|    policy_gradient_loss | -0.00101      |
|    value_loss           | 0.546         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.27         |
|    ep_rew_mean          | -0.789       |
| time/                   |              |
|    fps                  | 1008         |
|    iterations           | 8            |
|    time_elapsed         | 0            |
|    total_timesteps      | 256          |
| train/                  |              |
|    approx_kl            | 6.548315e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.61        |
|    explained_variance   | -0.0261      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.376        |
|    n_updates            | 14           |
|    policy_gradient_loss | -0.00148     |
|    value_loss           | 0.752        |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.28         |
|    ep_rew_mean          | -0.769       |
| time/                   |              |
|    fps                  | 1053         |
|    iterations           | 9            |
|    time_elapsed         | 0            |
|    total_timesteps      | 288          |
| train/                  |              |
|    approx_kl            | 7.542223e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.61        |
|    explained_variance   | -0.0309      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.0918       |
|    n_updates            | 16           |
|    policy_gradient_loss | -0.00197     |
|    value_loss           | 0.562        |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1.35        |
|    ep_rew_mean          | -0.751      |
| time/                   |             |
|    fps                  | 1082        |
|    iterations           | 10          |
|    time_elapsed         | 0           |
|    total_timesteps      | 320         |
| train/                  |             |
|    approx_kl            | 9.83458e-05 |
|    clip_fraction        | 0           |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.61       |
|    explained_variance   | -0.0252     |
|    learning_rate        | 0.0003      |
|    loss                 | 0.216       |
|    n_updates            | 18          |
|    policy_gradient_loss | -0.00299    |
|    value_loss           | 0.622       |
-----------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.32          |
|    ep_rew_mean          | -0.71         |
| time/                   |               |
|    fps                  | 1116          |
|    iterations           | 11            |
|    time_elapsed         | 0             |
|    total_timesteps      | 352           |
| train/                  |               |
|    approx_kl            | 5.6406483e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.61         |
|    explained_variance   | -0.0578       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.237         |
|    n_updates            | 20            |
|    policy_gradient_loss | -0.000723     |
|    value_loss           | 0.538         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.34          |
|    ep_rew_mean          | -0.709        |
| time/                   |               |
|    fps                  | 1143          |
|    iterations           | 12            |
|    time_elapsed         | 0             |
|    total_timesteps      | 384           |
| train/                  |               |
|    approx_kl            | 3.7783757e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.6          |
|    explained_variance   | -0.024        |
|    learning_rate        | 0.0003        |
|    loss                 | 0.734         |
|    n_updates            | 22            |
|    policy_gradient_loss | -0.000706     |
|    value_loss           | 0.962         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.42          |
|    ep_rew_mean          | -0.673        |
| time/                   |               |
|    fps                  | 1164          |
|    iterations           | 13            |
|    time_elapsed         | 0             |
|    total_timesteps      | 416           |
| train/                  |               |
|    approx_kl            | 1.9911677e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.6          |
|    explained_variance   | -0.000575     |
|    learning_rate        | 0.0003        |
|    loss                 | 0.484         |
|    n_updates            | 24            |
|    policy_gradient_loss | -0.000466     |
|    value_loss           | 0.866         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.32         |
|    ep_rew_mean          | -0.651       |
| time/                   |              |
|    fps                  | 1183         |
|    iterations           | 14           |
|    time_elapsed         | 0            |
|    total_timesteps      | 448          |
| train/                  |              |
|    approx_kl            | 3.568828e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.6         |
|    explained_variance   | -0.0412      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.51         |
|    n_updates            | 26           |
|    policy_gradient_loss | 7.42e-05     |
|    value_loss           | 0.7          |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1.29        |
|    ep_rew_mean          | -0.651      |
| time/                   |             |
|    fps                  | 1199        |
|    iterations           | 15          |
|    time_elapsed         | 0           |
|    total_timesteps      | 480         |
| train/                  |             |
|    approx_kl            | 3.72082e-05 |
|    clip_fraction        | 0           |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.6        |
|    explained_variance   | 0.0184      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.493       |
|    n_updates            | 28          |
|    policy_gradient_loss | -0.000661   |
|    value_loss           | 0.841       |
-----------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.19          |
|    ep_rew_mean          | -0.689        |
| time/                   |               |
|    fps                  | 1214          |
|    iterations           | 16            |
|    time_elapsed         | 0             |
|    total_timesteps      | 512           |
| train/                  |               |
|    approx_kl            | 1.8157065e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.6          |
|    explained_variance   | -0.0198       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.264         |
|    n_updates            | 30            |
|    policy_gradient_loss | 0.000366      |
|    value_loss           | 0.764         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.17          |
|    ep_rew_mean          | -0.687        |
| time/                   |               |
|    fps                  | 1226          |
|    iterations           | 17            |
|    time_elapsed         | 0             |
|    total_timesteps      | 544           |
| train/                  |               |
|    approx_kl            | 1.0963529e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.6          |
|    explained_variance   | -0.0094       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.252         |
|    n_updates            | 32            |
|    policy_gradient_loss | -0.000415     |
|    value_loss           | 0.578         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.22          |
|    ep_rew_mean          | -0.708        |
| time/                   |               |
|    fps                  | 1233          |
|    iterations           | 18            |
|    time_elapsed         | 0             |
|    total_timesteps      | 576           |
| train/                  |               |
|    approx_kl            | 1.4541671e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.6          |
|    explained_variance   | 0.00155       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.109         |
|    n_updates            | 34            |
|    policy_gradient_loss | 7.66e-05      |
|    value_loss           | 0.672         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.15          |
|    ep_rew_mean          | -0.704        |
| time/                   |               |
|    fps                  | 1254          |
|    iterations           | 19            |
|    time_elapsed         | 0             |
|    total_timesteps      | 608           |
| train/                  |               |
|    approx_kl            | 1.3895333e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.6          |
|    explained_variance   | -0.0152       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.243         |
|    n_updates            | 36            |
|    policy_gradient_loss | 0.000421      |
|    value_loss           | 0.747         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.14         |
|    ep_rew_mean          | -0.542       |
| time/                   |              |
|    fps                  | 1280         |
|    iterations           | 20           |
|    time_elapsed         | 0            |
|    total_timesteps      | 640          |
| train/                  |              |
|    approx_kl            | 8.404814e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.6         |
|    explained_variance   | -0.00183     |
|    learning_rate        | 0.0003       |
|    loss                 | 0.234        |
|    n_updates            | 38           |
|    policy_gradient_loss | -0.002       |
|    value_loss           | 0.679        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.17          |
|    ep_rew_mean          | -0.504        |
| time/                   |               |
|    fps                  | 1289          |
|    iterations           | 21            |
|    time_elapsed         | 0             |
|    total_timesteps      | 672           |
| train/                  |               |
|    approx_kl            | 0.00020119175 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.6          |
|    explained_variance   | -0.00395      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.625         |
|    n_updates            | 40            |
|    policy_gradient_loss | -0.00331      |
|    value_loss           | 1.02          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.16          |
|    ep_rew_mean          | -0.485        |
| time/                   |               |
|    fps                  | 1296          |
|    iterations           | 22            |
|    time_elapsed         | 0             |
|    total_timesteps      | 704           |
| train/                  |               |
|    approx_kl            | 7.9449266e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.59         |
|    explained_variance   | -0.00754      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.237         |
|    n_updates            | 42            |
|    policy_gradient_loss | -0.000344     |
|    value_loss           | 0.962         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.22          |
|    ep_rew_mean          | -0.467        |
| time/                   |               |
|    fps                  | 1305          |
|    iterations           | 23            |
|    time_elapsed         | 0             |
|    total_timesteps      | 736           |
| train/                  |               |
|    approx_kl            | 0.00013889559 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.59         |
|    explained_variance   | 0.000925      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.363         |
|    n_updates            | 44            |
|    policy_gradient_loss | -0.00375      |
|    value_loss           | 0.731         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.21         |
|    ep_rew_mean          | -0.587       |
| time/                   |              |
|    fps                  | 1316         |
|    iterations           | 24           |
|    time_elapsed         | 0            |
|    total_timesteps      | 768          |
| train/                  |              |
|    approx_kl            | 9.755418e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.59        |
|    explained_variance   | -0.0179      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.358        |
|    n_updates            | 46           |
|    policy_gradient_loss | -0.000278    |
|    value_loss           | 0.843        |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.18         |
|    ep_rew_mean          | -0.506       |
| time/                   |              |
|    fps                  | 1324         |
|    iterations           | 25           |
|    time_elapsed         | 0            |
|    total_timesteps      | 800          |
| train/                  |              |
|    approx_kl            | 8.545071e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.58        |
|    explained_variance   | 0.00642      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.492        |
|    n_updates            | 48           |
|    policy_gradient_loss | -0.00112     |
|    value_loss           | 0.608        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.15          |
|    ep_rew_mean          | -0.486        |
| time/                   |               |
|    fps                  | 1340          |
|    iterations           | 26            |
|    time_elapsed         | 0             |
|    total_timesteps      | 832           |
| train/                  |               |
|    approx_kl            | 0.00012436137 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.58         |
|    explained_variance   | 0.00488       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.479         |
|    n_updates            | 50            |
|    policy_gradient_loss | -0.00368      |
|    value_loss           | 1.13          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.12          |
|    ep_rew_mean          | -0.485        |
| time/                   |               |
|    fps                  | 1360          |
|    iterations           | 27            |
|    time_elapsed         | 0             |
|    total_timesteps      | 864           |
| train/                  |               |
|    approx_kl            | 0.00028509833 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.57         |
|    explained_variance   | -0.0171       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.277         |
|    n_updates            | 52            |
|    policy_gradient_loss | -0.00219      |
|    value_loss           | 0.943         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.15          |
|    ep_rew_mean          | -0.506        |
| time/                   |               |
|    fps                  | 1369          |
|    iterations           | 28            |
|    time_elapsed         | 0             |
|    total_timesteps      | 896           |
| train/                  |               |
|    approx_kl            | 0.00024232827 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.56         |
|    explained_variance   | 0.00453       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.475         |
|    n_updates            | 54            |
|    policy_gradient_loss | -0.00306      |
|    value_loss           | 0.675         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.13         |
|    ep_rew_mean          | -0.565       |
| time/                   |              |
|    fps                  | 1372         |
|    iterations           | 29           |
|    time_elapsed         | 0            |
|    total_timesteps      | 928          |
| train/                  |              |
|    approx_kl            | 9.635277e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.55        |
|    explained_variance   | -0.0101      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.585        |
|    n_updates            | 56           |
|    policy_gradient_loss | 0.000351     |
|    value_loss           | 1.01         |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.11         |
|    ep_rew_mean          | -0.585       |
| time/                   |              |
|    fps                  | 1364         |
|    iterations           | 30           |
|    time_elapsed         | 0            |
|    total_timesteps      | 960          |
| train/                  |              |
|    approx_kl            | 0.0001163315 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.55        |
|    explained_variance   | -0.00176     |
|    learning_rate        | 0.0003       |
|    loss                 | 0.305        |
|    n_updates            | 58           |
|    policy_gradient_loss | -0.00277     |
|    value_loss           | 0.814        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.09          |
|    ep_rew_mean          | -0.565        |
| time/                   |               |
|    fps                  | 1375          |
|    iterations           | 31            |
|    time_elapsed         | 0             |
|    total_timesteps      | 992           |
| train/                  |               |
|    approx_kl            | 0.00014591776 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.54         |
|    explained_variance   | -0.000269     |
|    learning_rate        | 0.0003        |
|    loss                 | 0.386         |
|    n_updates            | 60            |
|    policy_gradient_loss | -0.00225      |
|    value_loss           | 0.741         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.1           |
|    ep_rew_mean          | -0.663        |
| time/                   |               |
|    fps                  | 1380          |
|    iterations           | 32            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1024          |
| train/                  |               |
|    approx_kl            | 0.00018922612 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.53         |
|    explained_variance   | 0.000864      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.565         |
|    n_updates            | 62            |
|    policy_gradient_loss | -0.000738     |
|    value_loss           | 0.896         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.11         |
|    ep_rew_mean          | -0.623       |
| time/                   |              |
|    fps                  | 1384         |
|    iterations           | 33           |
|    time_elapsed         | 0            |
|    total_timesteps      | 1056         |
| train/                  |              |
|    approx_kl            | 0.0001565367 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.52        |
|    explained_variance   | -0.00529     |
|    learning_rate        | 0.0003       |
|    loss                 | 0.137        |
|    n_updates            | 64           |
|    policy_gradient_loss | -0.00196     |
|    value_loss           | 0.597        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.16          |
|    ep_rew_mean          | -0.625        |
| time/                   |               |
|    fps                  | 1386          |
|    iterations           | 34            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1088          |
| train/                  |               |
|    approx_kl            | 0.00027815253 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.51         |
|    explained_variance   | 0.00224       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.355         |
|    n_updates            | 66            |
|    policy_gradient_loss | -0.00334      |
|    value_loss           | 0.904         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.15          |
|    ep_rew_mean          | -0.645        |
| time/                   |               |
|    fps                  | 1390          |
|    iterations           | 35            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1120          |
| train/                  |               |
|    approx_kl            | 0.00044065714 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.5          |
|    explained_variance   | 0.00207       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.375         |
|    n_updates            | 68            |
|    policy_gradient_loss | -0.00417      |
|    value_loss           | 0.802         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.12          |
|    ep_rew_mean          | -0.542        |
| time/                   |               |
|    fps                  | 1394          |
|    iterations           | 36            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1152          |
| train/                  |               |
|    approx_kl            | 0.00021803193 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.48         |
|    explained_variance   | -0.00107      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.488         |
|    n_updates            | 70            |
|    policy_gradient_loss | -0.00358      |
|    value_loss           | 0.565         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.04          |
|    ep_rew_mean          | -0.519        |
| time/                   |               |
|    fps                  | 1403          |
|    iterations           | 37            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1184          |
| train/                  |               |
|    approx_kl            | 0.00012468174 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.46         |
|    explained_variance   | 0.0128        |
|    learning_rate        | 0.0003        |
|    loss                 | 0.713         |
|    n_updates            | 72            |
|    policy_gradient_loss | -0.000703     |
|    value_loss           | 1.23          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.07          |
|    ep_rew_mean          | -0.562        |
| time/                   |               |
|    fps                  | 1406          |
|    iterations           | 38            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1216          |
| train/                  |               |
|    approx_kl            | 0.00014370121 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.45         |
|    explained_variance   | 0             |
|    learning_rate        | 0.0003        |
|    loss                 | 0.482         |
|    n_updates            | 74            |
|    policy_gradient_loss | -0.00329      |
|    value_loss           | 0.813         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.05         |
|    ep_rew_mean          | -0.502       |
| time/                   |              |
|    fps                  | 1407         |
|    iterations           | 39           |
|    time_elapsed         | 0            |
|    total_timesteps      | 1248         |
| train/                  |              |
|    approx_kl            | 0.0004562121 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.44        |
|    explained_variance   | -0.000321    |
|    learning_rate        | 0.0003       |
|    loss                 | 0.371        |
|    n_updates            | 76           |
|    policy_gradient_loss | -0.00283     |
|    value_loss           | 0.6          |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.06          |
|    ep_rew_mean          | -0.501        |
| time/                   |               |
|    fps                  | 1407          |
|    iterations           | 40            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1280          |
| train/                  |               |
|    approx_kl            | 0.00024477392 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.41         |
|    explained_variance   | 0.0014        |
|    learning_rate        | 0.0003        |
|    loss                 | 0.688         |
|    n_updates            | 78            |
|    policy_gradient_loss | -0.00517      |
|    value_loss           | 1.19          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.07          |
|    ep_rew_mean          | -0.481        |
| time/                   |               |
|    fps                  | 1408          |
|    iterations           | 41            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1312          |
| train/                  |               |
|    approx_kl            | 0.00044154748 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.39         |
|    explained_variance   | 0.00403       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.365         |
|    n_updates            | 80            |
|    policy_gradient_loss | -0.00595      |
|    value_loss           | 0.852         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.04          |
|    ep_rew_mean          | -0.56         |
| time/                   |               |
|    fps                  | 1412          |
|    iterations           | 42            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1344          |
| train/                  |               |
|    approx_kl            | 0.00070066005 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.37         |
|    explained_variance   | -0.00366      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.485         |
|    n_updates            | 82            |
|    policy_gradient_loss | -0.00476      |
|    value_loss           | 0.777         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.04          |
|    ep_rew_mean          | -0.439        |
| time/                   |               |
|    fps                  | 1419          |
|    iterations           | 43            |
|    time_elapsed         | 0             |
|    total_timesteps      | 1376          |
| train/                  |               |
|    approx_kl            | 0.00039308518 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.34         |
|    explained_variance   | 0             |
|    learning_rate        | 0.0003        |
|    loss                 | 0.27          |
|    n_updates            | 84            |
|    policy_gradient_loss | -0.00231      |
|    value_loss           | 0.81          |
-------------------------------------------
--------------------------------------------
| rollout/                |                |
|    ep_len_mean          | 1.02           |
|    ep_rew_mean          | -0.377         |
| time/                   |                |
|    fps                  | 1422           |
|    iterations           | 44             |
|    time_elapsed         | 0              |
|    total_timesteps      | 1408           |
| train/                  |                |
|    approx_kl            | 0.000104792416 |
|    clip_fraction        | 0              |
|    clip_range           | 0.2            |
|    entropy_loss         | -1.32          |
|    explained_variance   | -0.00185       |
|    learning_rate        | 0.0003         |
|    loss                 | 0.471          |
|    n_updates            | 86             |
|    policy_gradient_loss | 0.000187       |
|    value_loss           | 1.21           |
--------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.04         |
|    ep_rew_mean          | -0.457       |
| time/                   |              |
|    fps                  | 1425         |
|    iterations           | 45           |
|    time_elapsed         | 1            |
|    total_timesteps      | 1440         |
| train/                  |              |
|    approx_kl            | 0.0003661234 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.3         |
|    explained_variance   | -0.00233     |
|    learning_rate        | 0.0003       |
|    loss                 | 0.326        |
|    n_updates            | 88           |
|    policy_gradient_loss | -0.00442     |
|    value_loss           | 0.855        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.07          |
|    ep_rew_mean          | -0.519        |
| time/                   |               |
|    fps                  | 1425          |
|    iterations           | 46            |
|    time_elapsed         | 1             |
|    total_timesteps      | 1472          |
| train/                  |               |
|    approx_kl            | 0.00027570315 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.28         |
|    explained_variance   | -0.00228      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.463         |
|    n_updates            | 90            |
|    policy_gradient_loss | -0.00248      |
|    value_loss           | 0.771         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.06          |
|    ep_rew_mean          | -0.559        |
| time/                   |               |
|    fps                  | 1433          |
|    iterations           | 47            |
|    time_elapsed         | 1             |
|    total_timesteps      | 1504          |
| train/                  |               |
|    approx_kl            | 0.00017585978 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.25         |
|    explained_variance   | -0.00159      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.545         |
|    n_updates            | 92            |
|    policy_gradient_loss | 0.00211       |
|    value_loss           | 0.889         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.06          |
|    ep_rew_mean          | -0.501        |
| time/                   |               |
|    fps                  | 1432          |
|    iterations           | 48            |
|    time_elapsed         | 1             |
|    total_timesteps      | 1536          |
| train/                  |               |
|    approx_kl            | 0.00011192821 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.24         |
|    explained_variance   | 0             |
|    learning_rate        | 0.0003        |
|    loss                 | 0.459         |
|    n_updates            | 94            |
|    policy_gradient_loss | -0.00356      |
|    value_loss           | 0.95          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.07          |
|    ep_rew_mean          | -0.381        |
| time/                   |               |
|    fps                  | 1431          |
|    iterations           | 49            |
|    time_elapsed         | 1             |
|    total_timesteps      | 1568          |
| train/                  |               |
|    approx_kl            | 0.00013080053 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.23         |
|    explained_variance   | -0.00417      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.252         |
|    n_updates            | 96            |
|    policy_gradient_loss | 0.00197       |
|    value_loss           | 0.892         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.11          |
|    ep_rew_mean          | -0.362        |
| time/                   |               |
|    fps                  | 1429          |
|    iterations           | 50            |
|    time_elapsed         | 1             |
|    total_timesteps      | 1600          |
| train/                  |               |
|    approx_kl            | 3.0012801e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.22         |
|    explained_variance   | -0.00499      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.531         |
|    n_updates            | 98            |
|    policy_gradient_loss | -5.23e-05     |
|    value_loss           | 0.985         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.12          |
|    ep_rew_mean          | -0.425        |
| time/                   |               |
|    fps                  | 1434          |
|    iterations           | 51            |
|    time_elapsed         | 1             |
|    total_timesteps      | 1632          |
| train/                  |               |
|    approx_kl            | 5.6494027e-06 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.22         |
|    explained_variance   | 0.00954       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.434         |
|    n_updates            | 100           |
|    policy_gradient_loss | 0.000489      |
|    value_loss           | 0.943         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.09          |
|    ep_rew_mean          | -0.522        |
| time/                   |               |
|    fps                  | 1439          |
|    iterations           | 52            |
|    time_elapsed         | 1             |
|    total_timesteps      | 1664          |
| train/                  |               |
|    approx_kl            | 5.1956624e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.22         |
|    explained_variance   | 0.00443       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.501         |
|    n_updates            | 102           |
|    policy_gradient_loss | -0.00126      |
|    value_loss           | 0.855         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.06         |
|    ep_rew_mean          | -0.622       |
| time/                   |              |
|    fps                  | 1442         |
|    iterations           | 53           |
|    time_elapsed         | 1            |
|    total_timesteps      | 1696         |
| train/                  |              |
|    approx_kl            | 0.0001221504 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.23        |
|    explained_variance   | 5.96e-08     |
|    learning_rate        | 0.0003       |
|    loss                 | 0.394        |
|    n_updates            | 104          |
|    policy_gradient_loss | -0.000861    |
|    value_loss           | 0.876        |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.06         |
|    ep_rew_mean          | -0.582       |
| time/                   |              |
|    fps                  | 1444         |
|    iterations           | 54           |
|    time_elapsed         | 1            |
|    total_timesteps      | 1728         |
| train/                  |              |
|    approx_kl            | 4.599616e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.23        |
|    explained_variance   | -0.00183     |
|    learning_rate        | 0.0003       |
|    loss                 | 0.224        |
|    n_updates            | 106          |
|    policy_gradient_loss | -0.00097     |
|    value_loss           | 0.795        |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.07         |
|    ep_rew_mean          | -0.583       |
| time/                   |              |
|    fps                  | 1436         |
|    iterations           | 55           |
|    time_elapsed         | 1            |
|    total_timesteps      | 1760         |
| train/                  |              |
|    approx_kl            | 0.0001438111 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.22        |
|    explained_variance   | -0.000618    |
|    learning_rate        | 0.0003       |
|    loss                 | 0.574        |
|    n_updates            | 108          |
|    policy_gradient_loss | -0.00101     |
|    value_loss           | 0.852        |
------------------------------------------
--------------------------------------------
| rollout/                |                |
|    ep_len_mean          | 1.07           |
|    ep_rew_mean          | -0.583         |
| time/                   |                |
|    fps                  | 1440           |
|    iterations           | 56             |
|    time_elapsed         | 1              |
|    total_timesteps      | 1792           |
| train/                  |                |
|    approx_kl            | 0.000120086595 |
|    clip_fraction        | 0              |
|    clip_range           | 0.2            |
|    entropy_loss         | -1.2           |
|    explained_variance   | -0.00258       |
|    learning_rate        | 0.0003         |
|    loss                 | 0.281          |
|    n_updates            | 110            |
|    policy_gradient_loss | -0.00133       |
|    value_loss           | 0.767          |
--------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1.06        |
|    ep_rew_mean          | -0.702      |
| time/                   |             |
|    fps                  | 1441        |
|    iterations           | 57          |
|    time_elapsed         | 1           |
|    total_timesteps      | 1824        |
| train/                  |             |
|    approx_kl            | 6.14617e-05 |
|    clip_fraction        | 0           |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.19       |
|    explained_variance   | 0.0017      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.147       |
|    n_updates            | 112         |
|    policy_gradient_loss | -0.00176    |
|    value_loss           | 0.762       |
-----------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.03          |
|    ep_rew_mean          | -0.761        |
| time/                   |               |
|    fps                  | 1447          |
|    iterations           | 58            |
|    time_elapsed         | 1             |
|    total_timesteps      | 1856          |
| train/                  |               |
|    approx_kl            | 7.6962635e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.17         |
|    explained_variance   | 0.00147       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.239         |
|    n_updates            | 114           |
|    policy_gradient_loss | 5.69e-05      |
|    value_loss           | 0.452         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.02         |
|    ep_rew_mean          | -0.72        |
| time/                   |              |
|    fps                  | 1452         |
|    iterations           | 59           |
|    time_elapsed         | 1            |
|    total_timesteps      | 1888         |
| train/                  |              |
|    approx_kl            | 7.905066e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.17        |
|    explained_variance   | 0.00146      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.231        |
|    n_updates            | 116          |
|    policy_gradient_loss | -0.00123     |
|    value_loss           | 0.682        |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.04         |
|    ep_rew_mean          | -0.66        |
| time/                   |              |
|    fps                  | 1456         |
|    iterations           | 60           |
|    time_elapsed         | 1            |
|    total_timesteps      | 1920         |
| train/                  |              |
|    approx_kl            | 0.0001288876 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.15        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 0.382        |
|    n_updates            | 118          |
|    policy_gradient_loss | -0.00103     |
|    value_loss           | 0.833        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.05          |
|    ep_rew_mean          | -0.601        |
| time/                   |               |
|    fps                  | 1454          |
|    iterations           | 61            |
|    time_elapsed         | 1             |
|    total_timesteps      | 1952          |
| train/                  |               |
|    approx_kl            | 3.9795414e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.13         |
|    explained_variance   | 0.00535       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.236         |
|    n_updates            | 120           |
|    policy_gradient_loss | 7.68e-05      |
|    value_loss           | 0.813         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.07         |
|    ep_rew_mean          | -0.642       |
| time/                   |              |
|    fps                  | 1455         |
|    iterations           | 62           |
|    time_elapsed         | 1            |
|    total_timesteps      | 1984         |
| train/                  |              |
|    approx_kl            | 6.055273e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.12        |
|    explained_variance   | 0.00207      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.128        |
|    n_updates            | 122          |
|    policy_gradient_loss | -0.000402    |
|    value_loss           | 0.815        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.09          |
|    ep_rew_mean          | -0.623        |
| time/                   |               |
|    fps                  | 1452          |
|    iterations           | 63            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2016          |
| train/                  |               |
|    approx_kl            | 0.00011927262 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.11         |
|    explained_variance   | 0.00494       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.247         |
|    n_updates            | 124           |
|    policy_gradient_loss | -0.00271      |
|    value_loss           | 0.681         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.06          |
|    ep_rew_mean          | -0.521        |
| time/                   |               |
|    fps                  | 1448          |
|    iterations           | 64            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2048          |
| train/                  |               |
|    approx_kl            | 0.00015577488 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.1          |
|    explained_variance   | 0.00597       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.612         |
|    n_updates            | 126           |
|    policy_gradient_loss | -0.00398      |
|    value_loss           | 0.815         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.06          |
|    ep_rew_mean          | -0.38         |
| time/                   |               |
|    fps                  | 1455          |
|    iterations           | 65            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2080          |
| train/                  |               |
|    approx_kl            | 0.00049224496 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.07         |
|    explained_variance   | -0.000813     |
|    learning_rate        | 0.0003        |
|    loss                 | 0.387         |
|    n_updates            | 128           |
|    policy_gradient_loss | -0.00168      |
|    value_loss           | 1.06          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.07          |
|    ep_rew_mean          | -0.461        |
| time/                   |               |
|    fps                  | 1452          |
|    iterations           | 66            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2112          |
| train/                  |               |
|    approx_kl            | 0.00017843954 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -1.05         |
|    explained_variance   | -0.0017       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.559         |
|    n_updates            | 130           |
|    policy_gradient_loss | -0.00393      |
|    value_loss           | 1.13          |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.05         |
|    ep_rew_mean          | -0.619       |
| time/                   |              |
|    fps                  | 1456         |
|    iterations           | 67           |
|    time_elapsed         | 1            |
|    total_timesteps      | 2144         |
| train/                  |              |
|    approx_kl            | 0.0004255753 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.02        |
|    explained_variance   | -0.0034      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.191        |
|    n_updates            | 132          |
|    policy_gradient_loss | -0.00525     |
|    value_loss           | 0.64         |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.06          |
|    ep_rew_mean          | -0.52         |
| time/                   |               |
|    fps                  | 1457          |
|    iterations           | 68            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2176          |
| train/                  |               |
|    approx_kl            | 0.00016361661 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.992        |
|    explained_variance   | -0.00373      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.465         |
|    n_updates            | 134           |
|    policy_gradient_loss | 0.000333      |
|    value_loss           | 0.665         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.06          |
|    ep_rew_mean          | -0.519        |
| time/                   |               |
|    fps                  | 1461          |
|    iterations           | 69            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2208          |
| train/                  |               |
|    approx_kl            | 0.00025469624 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.972        |
|    explained_variance   | 0.00354       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.555         |
|    n_updates            | 136           |
|    policy_gradient_loss | -0.00458      |
|    value_loss           | 1.2           |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.05          |
|    ep_rew_mean          | -0.5          |
| time/                   |               |
|    fps                  | 1465          |
|    iterations           | 70            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2240          |
| train/                  |               |
|    approx_kl            | 0.00020360202 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.951        |
|    explained_variance   | -0.000284     |
|    learning_rate        | 0.0003        |
|    loss                 | 0.394         |
|    n_updates            | 138           |
|    policy_gradient_loss | -0.000815     |
|    value_loss           | 0.897         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.04          |
|    ep_rew_mean          | -0.56         |
| time/                   |               |
|    fps                  | 1468          |
|    iterations           | 71            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2272          |
| train/                  |               |
|    approx_kl            | 9.1385096e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.934        |
|    explained_variance   | -0.000282     |
|    learning_rate        | 0.0003        |
|    loss                 | 0.209         |
|    n_updates            | 140           |
|    policy_gradient_loss | -0.000487     |
|    value_loss           | 0.734         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.03         |
|    ep_rew_mean          | -0.64        |
| time/                   |              |
|    fps                  | 1472         |
|    iterations           | 72           |
|    time_elapsed         | 1            |
|    total_timesteps      | 2304         |
| train/                  |              |
|    approx_kl            | 0.0001559928 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.915       |
|    explained_variance   | -0.00843     |
|    learning_rate        | 0.0003       |
|    loss                 | 0.564        |
|    n_updates            | 142          |
|    policy_gradient_loss | -0.00134     |
|    value_loss           | 0.907        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.04          |
|    ep_rew_mean          | -0.52         |
| time/                   |               |
|    fps                  | 1472          |
|    iterations           | 73            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2336          |
| train/                  |               |
|    approx_kl            | 7.8033656e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.899        |
|    explained_variance   | -0.00121      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.277         |
|    n_updates            | 144           |
|    policy_gradient_loss | 0.000112      |
|    value_loss           | 0.763         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.03          |
|    ep_rew_mean          | -0.519        |
| time/                   |               |
|    fps                  | 1471          |
|    iterations           | 74            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2368          |
| train/                  |               |
|    approx_kl            | 2.9746443e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.894        |
|    explained_variance   | 0.0085        |
|    learning_rate        | 0.0003        |
|    loss                 | 0.452         |
|    n_updates            | 146           |
|    policy_gradient_loss | -0.000967     |
|    value_loss           | 0.94          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.05          |
|    ep_rew_mean          | -0.439        |
| time/                   |               |
|    fps                  | 1475          |
|    iterations           | 75            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2400          |
| train/                  |               |
|    approx_kl            | 1.5571713e-06 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.897        |
|    explained_variance   | 0             |
|    learning_rate        | 0.0003        |
|    loss                 | 0.387         |
|    n_updates            | 148           |
|    policy_gradient_loss | 0.000197      |
|    value_loss           | 1.01          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.02          |
|    ep_rew_mean          | -0.478        |
| time/                   |               |
|    fps                  | 1480          |
|    iterations           | 76            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2432          |
| train/                  |               |
|    approx_kl            | 0.00021473318 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.906        |
|    explained_variance   | -0.00565      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.297         |
|    n_updates            | 150           |
|    policy_gradient_loss | -0.00702      |
|    value_loss           | 0.866         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.04          |
|    ep_rew_mean          | -0.599        |
| time/                   |               |
|    fps                  | 1485          |
|    iterations           | 77            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2464          |
| train/                  |               |
|    approx_kl            | 0.00024438463 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.923        |
|    explained_variance   | 0             |
|    learning_rate        | 0.0003        |
|    loss                 | 0.546         |
|    n_updates            | 152           |
|    policy_gradient_loss | 4.93e-05      |
|    value_loss           | 0.83          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.03          |
|    ep_rew_mean          | -0.579        |
| time/                   |               |
|    fps                  | 1488          |
|    iterations           | 78            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2496          |
| train/                  |               |
|    approx_kl            | 3.3564866e-06 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.93         |
|    explained_variance   | 0.0104        |
|    learning_rate        | 0.0003        |
|    loss                 | 0.545         |
|    n_updates            | 154           |
|    policy_gradient_loss | 0.000267      |
|    value_loss           | 0.721         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.07          |
|    ep_rew_mean          | -0.52         |
| time/                   |               |
|    fps                  | 1492          |
|    iterations           | 79            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2528          |
| train/                  |               |
|    approx_kl            | 2.7287751e-06 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.927        |
|    explained_variance   | -0.00138      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.362         |
|    n_updates            | 156           |
|    policy_gradient_loss | 0.000138      |
|    value_loss           | 0.961         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.05         |
|    ep_rew_mean          | -0.48        |
| time/                   |              |
|    fps                  | 1493         |
|    iterations           | 80           |
|    time_elapsed         | 1            |
|    total_timesteps      | 2560         |
| train/                  |              |
|    approx_kl            | 6.712042e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.926       |
|    explained_variance   | -0.00759     |
|    learning_rate        | 0.0003       |
|    loss                 | 0.586        |
|    n_updates            | 158          |
|    policy_gradient_loss | -0.00118     |
|    value_loss           | 0.865        |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.05         |
|    ep_rew_mean          | -0.44        |
| time/                   |              |
|    fps                  | 1495         |
|    iterations           | 81           |
|    time_elapsed         | 1            |
|    total_timesteps      | 2592         |
| train/                  |              |
|    approx_kl            | 6.834045e-05 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.915       |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 0.386        |
|    n_updates            | 160          |
|    policy_gradient_loss | -0.000116    |
|    value_loss           | 0.861        |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.02         |
|    ep_rew_mean          | -0.438       |
| time/                   |              |
|    fps                  | 1500         |
|    iterations           | 82           |
|    time_elapsed         | 1            |
|    total_timesteps      | 2624         |
| train/                  |              |
|    approx_kl            | 0.0003871806 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.894       |
|    explained_variance   | 0.00274      |
|    learning_rate        | 0.0003       |
|    loss                 | 0.381        |
|    n_updates            | 162          |
|    policy_gradient_loss | -0.00541     |
|    value_loss           | 0.961        |
------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.02          |
|    ep_rew_mean          | -0.359        |
| time/                   |               |
|    fps                  | 1502          |
|    iterations           | 83            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2656          |
| train/                  |               |
|    approx_kl            | 7.1793795e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.87         |
|    explained_variance   | 0             |
|    learning_rate        | 0.0003        |
|    loss                 | 0.476         |
|    n_updates            | 164           |
|    policy_gradient_loss | 0.00133       |
|    value_loss           | 0.956         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.02          |
|    ep_rew_mean          | -0.318        |
| time/                   |               |
|    fps                  | 1506          |
|    iterations           | 84            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2688          |
| train/                  |               |
|    approx_kl            | 4.5645982e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.864        |
|    explained_variance   | -0.0018       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.47          |
|    n_updates            | 166           |
|    policy_gradient_loss | -0.00101      |
|    value_loss           | 1.01          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.03          |
|    ep_rew_mean          | -0.338        |
| time/                   |               |
|    fps                  | 1509          |
|    iterations           | 85            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2720          |
| train/                  |               |
|    approx_kl            | 0.00020501018 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.847        |
|    explained_variance   | -0.00145      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.575         |
|    n_updates            | 168           |
|    policy_gradient_loss | -0.00366      |
|    value_loss           | 0.974         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.03          |
|    ep_rew_mean          | -0.398        |
| time/                   |               |
|    fps                  | 1511          |
|    iterations           | 86            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2752          |
| train/                  |               |
|    approx_kl            | 9.0356916e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.822        |
|    explained_variance   | -0.00465      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.568         |
|    n_updates            | 170           |
|    policy_gradient_loss | 0.000297      |
|    value_loss           | 0.948         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.04          |
|    ep_rew_mean          | -0.418        |
| time/                   |               |
|    fps                  | 1515          |
|    iterations           | 87            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2784          |
| train/                  |               |
|    approx_kl            | 1.2971461e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.814        |
|    explained_variance   | 0             |
|    learning_rate        | 0.0003        |
|    loss                 | 0.42          |
|    n_updates            | 172           |
|    policy_gradient_loss | 2.21e-05      |
|    value_loss           | 0.946         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.04          |
|    ep_rew_mean          | -0.519        |
| time/                   |               |
|    fps                  | 1517          |
|    iterations           | 88            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2816          |
| train/                  |               |
|    approx_kl            | 0.00024075061 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.797        |
|    explained_variance   | -0.00097      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.413         |
|    n_updates            | 174           |
|    policy_gradient_loss | -0.0052       |
|    value_loss           | 0.942         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.05          |
|    ep_rew_mean          | -0.379        |
| time/                   |               |
|    fps                  | 1516          |
|    iterations           | 89            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2848          |
| train/                  |               |
|    approx_kl            | 0.00040026382 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.766        |
|    explained_variance   | 0.000941      |
|    learning_rate        | 0.0003        |
|    loss                 | 0.407         |
|    n_updates            | 176           |
|    policy_gradient_loss | -0.00305      |
|    value_loss           | 0.777         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.04          |
|    ep_rew_mean          | -0.44         |
| time/                   |               |
|    fps                  | 1515          |
|    iterations           | 90            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2880          |
| train/                  |               |
|    approx_kl            | 4.2077154e-05 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.742        |
|    explained_variance   | 0.00454       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.411         |
|    n_updates            | 178           |
|    policy_gradient_loss | 0.0007        |
|    value_loss           | 1.11          |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.02          |
|    ep_rew_mean          | -0.439        |
| time/                   |               |
|    fps                  | 1518          |
|    iterations           | 91            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2912          |
| train/                  |               |
|    approx_kl            | 0.00018515438 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.729        |
|    explained_variance   | 0.00173       |
|    learning_rate        | 0.0003        |
|    loss                 | 0.319         |
|    n_updates            | 180           |
|    policy_gradient_loss | -0.00329      |
|    value_loss           | 0.869         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1.01          |
|    ep_rew_mean          | -0.519        |
| time/                   |               |
|    fps                  | 1522          |
|    iterations           | 92            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2944          |
| train/                  |               |
|    approx_kl            | 0.00024752133 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.703        |
|    explained_variance   | 0             |
|    learning_rate        | 0.0003        |
|    loss                 | 0.315         |
|    n_updates            | 182           |
|    policy_gradient_loss | -0.000969     |
|    value_loss           | 0.834         |
-------------------------------------------
-------------------------------------------
| rollout/                |               |
|    ep_len_mean          | 1             |
|    ep_rew_mean          | -0.64         |
| time/                   |               |
|    fps                  | 1526          |
|    iterations           | 93            |
|    time_elapsed         | 1             |
|    total_timesteps      | 2976          |
| train/                  |               |
|    approx_kl            | 0.00043149665 |
|    clip_fraction        | 0             |
|    clip_range           | 0.2           |
|    entropy_loss         | -0.679        |
|    explained_variance   | 0             |
|    learning_rate        | 0.0003        |
|    loss                 | 0.378         |
|    n_updates            | 184           |
|    policy_gradient_loss | -0.0035       |
|    value_loss           | 0.867         |
-------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1            |
|    ep_rew_mean          | -0.68        |
| time/                   |              |
|    fps                  | 1511         |
|    iterations           | 94           |
|    time_elapsed         | 1            |
|    total_timesteps      | 3008         |
| train/                  |              |
|    approx_kl            | 0.0001708474 |
|    clip_fraction        | 0            |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.649       |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 0.269        |
|    n_updates            | 186          |
|    policy_gradient_loss | -0.000836    |
|    value_loss           | 0.664        |
------------------------------------------
Evaluating PPO agent on 44 questions (deterministic=True)...
Saved PPO model to: /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/artifacts/smoke/ppo_model.zip
Saved PPO summaries to: /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/artifacts/smoke
```

```bash
source .venv/bin/activate && python scripts/evaluate_all.py --smoke --mc-path artifacts/smoke/mc_dataset.json
```

```output
Loading MC questions from: artifacts/smoke/mc_dataset.json
Loaded 44 MC questions
Warning: alias_lookup.json not found at /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/artifacts/smoke/alias_lookup.json, using empty lookup
Building likelihood model: tfidf
Using best softmax threshold: 0.5
Running full evaluation...

Computing per-category breakdown...

Per-category accuracy:
  Fine_Arts            (n=  7): acc=0.143, S_q=0.159
  Fine_Arts:Music      (n=  1): acc=1.000, S_q=0.620
  History              (n=  4): acc=0.250, S_q=0.064
  Literature           (n=  6): acc=0.000, S_q=0.000
  Literature:Europe    (n=  2): acc=0.000, S_q=0.000
  Literature:World     (n=  1): acc=1.000, S_q=0.871
  Science              (n=  3): acc=0.000, S_q=0.000
  Science:Chemistry    (n=  6): acc=1.000, S_q=0.683
  Science:Physics      (n=  4): acc=1.000, S_q=0.532
  Social_Science       (n=  9): acc=0.222, S_q=0.139
  Social_Science:Religion (n=  1): acc=1.000, S_q=0.362

Running shuffle control...
Running alias substitution control...
Running choices-only control...
Generating plots...
Wrote evaluation report to: /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/artifacts/smoke/evaluation_report.json
```

### Smoke pipeline outputs

Note: the alias substitution control in the evaluation report is currently a no-op. build_mc_dataset.py does not generate alias_lookup.json, so evaluate_all.py falls back to an empty lookup and the alias control returns results identical to the full evaluation. This is a known limitation, not a walkthrough error.

```bash
cat artifacts/smoke/ppo_summary.json
```

```output
{
  "n": 44.0,
  "buzz_accuracy": 0.3409090909090909,
  "mean_buzz_step": 0.0,
  "mean_sq": 0.32561864877729346,
  "mean_reward_like": -0.47600649350649354,
  "ece": 0.0990614827976308,
  "brier": 0.013474968274838784,
  "n_calibration": 44.0
}```
```

```bash
source .venv/bin/activate && python -c "import json; r=json.load(open('artifacts/smoke/evaluation_report.json')); print(json.dumps({k:r[k] for k in ['full_eval','controls']}, indent=2))"
```

```output
{
  "full_eval": {
    "n": 44.0,
    "buzz_accuracy": 0.38636363636363635,
    "mean_buzz_step": 3.5,
    "mean_sq": 0.24329479467724402,
    "mean_reward_like": 0.0,
    "ece": 0.0,
    "brier": 0.0,
    "n_calibration": 44.0
  },
  "controls": {
    "choices_only": {
      "accuracy": 0.09090909090909091,
      "chance": 0.25,
      "n_test": 11.0
    },
    "shuffle": {
      "n": 44.0,
      "buzz_accuracy": 0.38636363636363635,
      "mean_buzz_step": 3.5,
      "mean_sq": 0.23666016887085728,
      "mean_reward_like": 0.0,
      "ece": 0.0,
      "brier": 0.0,
      "n_calibration": 44.0
    },
    "alias_substitution": {
      "n": 44.0,
      "buzz_accuracy": 0.38636363636363635,
      "mean_buzz_step": 3.5,
      "mean_sq": 0.24329479467724402,
      "mean_reward_like": 0.0,
      "ece": 0.0,
      "brier": 0.0,
      "n_calibration": 44.0
    }
  }
}
```

## T5 policy smoke pipeline

Trains T5PolicyModel with supervised warm-start and PPO fine-tuning using t5-small. Note: the T5 PPO trainer uses its own default reward settings (wait_penalty=0.1, no early_buzz_penalty) which differ from the belief-feature smoke config. This is a limitation of the current T5 config flattener, not a walkthrough error.

The compare_policies step is skipped because it evaluates the MLP side with default-config env settings (wait_penalty=0.05, buzz_incorrect=-0.5) which differ from both the smoke training config (buzz_incorrect=-1.0) and the T5 side's hardcoded settings (wait_penalty=0.01). This three-way mismatch makes numeric comparison semantically invalid.

```bash
source .venv/bin/activate && python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke --mc-path artifacts/smoke/mc_dataset.json
```

```output
Loading MC questions from: artifacts/smoke/mc_dataset.json
Loaded 44 questions
Split: 30 train, 6 val, 8 test

============================================================
PHASE 1: SUPERVISED WARM-START
============================================================
============================================================
SUPERVISED TRAINING PHASE
============================================================
Loading T5 encoder: t5-small
Loading weights:   0%|          | 0/51 [00:00<?, ?it/s]Loading weights: 100%|██████████| 51/51 [00:00<00:00, 4448.76it/s]
Model Architecture:
  T5 encoder parameters: 35,330,816
  Policy head parameters: 528,135
  Total parameters: 35,858,951
  Device: mps
Starting supervised training for 2 epochs
  Training samples: 30
  Validation samples: 6
  Batch size: 4
  Gradient accumulation: 1 (effective batch = 4)
  Learning rate: 0.0003
  Device: mps

Epoch 1/2 - Train Loss: 1.3950, Train Acc: 0.1786 - Val Loss: 1.3837, Val Acc: 0.2500
Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]Writing model shards: 100%|██████████| 1/1 [00:00<00:00,  5.77it/s]Writing model shards: 100%|██████████| 1/1 [00:00<00:00,  5.77it/s]
Model saved to checkpoints/supervised/best_model
  -> New best validation accuracy: 0.2500
Epoch 2/2 - Train Loss: 1.3662, Train Acc: 0.3571 - Val Loss: 1.3714, Val Acc: 0.5000
Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]Writing model shards: 100%|██████████| 1/1 [00:00<00:00,  4.23it/s]Writing model shards: 100%|██████████| 1/1 [00:00<00:00,  4.22it/s]
Model saved to checkpoints/supervised/best_model
  -> New best validation accuracy: 0.5000

Supervised training completed!
  Best validation accuracy: 0.5000
Training history saved to checkpoints/supervised/history.json
Supervised model saved to: checkpoints/supervised/best_model

============================================================
PHASE 2: PPO FINE-TUNING (T5 Policy)
============================================================
============================================================
PPO TRAINING PHASE (T5 Policy)
============================================================
Loading pretrained model from checkpoints/supervised/best_model
Loading weights:   0%|          | 0/51 [00:00<?, ?it/s]Loading weights: 100%|██████████| 51/51 [00:00<00:00, 17466.28it/s]
Loading T5 encoder: checkpoints/supervised/best_model
Loading weights:   0%|          | 0/51 [00:00<?, ?it/s]Loading weights: 100%|██████████| 51/51 [00:00<00:00, 10167.77it/s]
Model Architecture:
  T5 encoder parameters: 35,330,816
  Policy head parameters: 528,135
  Total parameters: 35,858,951
  Device: mps
Loading weights:   0%|          | 0/51 [00:00<?, ?it/s]Loading weights: 100%|██████████| 51/51 [00:00<00:00, 11697.35it/s]
Model loaded from checkpoints/supervised/best_model
Starting PPO training for 5 iterations
  Training questions: 30
  Validation questions: 6
  Batch size: 4
  Episodes per iteration: 16
  Device: mps


Iteration 1/5
  Collecting rollouts...
  Avg episode reward: -0.4625
  Avg episode length: 2.31
  Updating policy...
  Policy loss: -0.1051
  Value loss: 0.4772
  Entropy: 2.0787

Iteration 2/5
  Collecting rollouts...
  Avg episode reward: -0.1875
  Avg episode length: 1.62
  Updating policy...
  Policy loss: -0.0869
  Value loss: 0.4658
  Entropy: 2.0789

Iteration 3/5
  Collecting rollouts...
  Avg episode reward: -0.2125
  Avg episode length: 1.81
  Updating policy...
  Policy loss: 0.0690
  Value loss: 0.4053
  Entropy: 2.0788

Iteration 4/5
  Collecting rollouts...
  Avg episode reward: -0.4750
  Avg episode length: 2.50
  Updating policy...
  Policy loss: 0.0044
  Value loss: 0.4179
  Entropy: 2.0787

Iteration 5/5
  Collecting rollouts...
  Avg episode reward: -0.3000
  Avg episode length: 2.69
  Updating policy...
  Policy loss: -0.0204
  Value loss: 0.4441
  Entropy: 2.0787

============================================================
PPO training completed!
Best validation reward: -inf
============================================================

============================================================
FINAL EVALUATION ON TEST SET
============================================================
Test Accuracy: 0.3750
Test Avg Reward: -0.0125
Test results saved to checkpoints/ppo_t5/test_results.json

============================================================
TRAINING COMPLETE
============================================================
Best PPO model saved to: checkpoints/ppo_t5/best_model
Training history: checkpoints/ppo_t5/history.json
```

## Test verification

Runs the belief-feature test suite (132 tests) plus T5-specific tests (44 tests).

```bash
source .venv/bin/activate && python -m pytest tests/test_agents.py tests/test_environment.py tests/test_ppo_buzzer.py tests/test_factories.py tests/test_build_mc_dataset.py tests/test_text_wrapper.py tests/test_metrics.py tests/test_t5_policy.py tests/test_supervised_t5.py tests/test_ppo_t5.py --tb=line -q
```

```output
........................................................................ [ 40%]
........................................................................ [ 81%]
................................                                         [100%]
176 passed in 16.75s
```

## Summary

Both pipelines completed successfully:

**Belief-feature smoke pipeline** (TF-IDF + PPO on 44 MC questions):
- Baseline accuracy: 38.6% across all threshold/profile/Bayesian agents
- PPO accuracy: 34.1%, mean S_q: 0.326
- Choices-only control: 9.1% (below 25% chance -- no exploitable surface artifacts)
- Alias substitution control: identical to full eval (no-op -- alias_lookup.json not generated)

**T5 policy smoke pipeline** (t5-small, 2 supervised epochs + 5 PPO iterations):
- Supervised best val accuracy: 50.0%
- PPO test accuracy: 37.5%, avg reward: -0.0125
- Note: T5 PPO uses default reward settings (wait_penalty=0.1) which differ from the belief-feature smoke config (wait_penalty=0.05, early_buzz_penalty=0.2)

**Test verification**: 176/176 tests passed (132 belief-feature + 44 T5-specific)

Training outputs contain nondeterministic elements (timings, SB3 verbose logs, gradient values) so this walkthrough is a demonstration document, not an exact-output reproducible proof.
