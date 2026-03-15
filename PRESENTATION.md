---
marp: true
theme: default
paginate: true
---

<!-- _class: lead -->

# Quiz Bowl RL Buzzer
## Multiple-Choice Strategic Buzzing Under Incremental Clues

CS234 Final Project

Kathleen Weng
Imran Hassan
Ankit Aggarwal

March 2026

---

## Problem

- Quiz bowl questions reveal evidence incrementally
- A good system must decide **when** to buzz, not just **what** answer to pick
- Buzz too early:
  - higher risk of a wrong answer
- Buzz too late:
  - lower strategic value, less chance to beat an opponent
- We study this in a **multiple-choice** setting so the answer space is controlled and evaluation is reproducible

---

## Background And Setup

- We model quiz bowl as a sequential decision problem over partial clues
- Each example is converted into:
  - question prefixes over time
  - `K=4` answer options
  - one correct option + three distractors
- The system supports two policy families:
  - belief-feature buzzers
  - end-to-end T5 text-policy buzzers

---

## Why Multiple Choice?

- Open-ended answering adds aliasing and grading complexity
- Multiple choice makes it easier to isolate the buzzing decision itself
- But naive distractors create artifacts, so answer generation matters

Our design goal:

- keep the answer space constrained
- make the options hard enough that the agent must use the clues

---

## Method Overview

End-to-end flow in the current codebase:

1. Load tossup questions
2. Build answer profiles from historical text
3. Construct artifact-resistant multiple-choice questions
4. Score options with a likelihood model
5. Convert beliefs into observations
6. Run baseline or learned buzzers
7. Evaluate timing, correctness, calibration, and controls

---

## Data And MC Construction

Key implementation pieces:

- `qb_data/data_loader.py`
  - CSV-first loading
  - optional Hugging Face fallback
- `qb_data/answer_profiles.py`
  - leave-one-out answer profiles
- `qb_data/mc_builder.py`
  - distractor selection strategies
  - anti-artifact guards

Distractor strategies in code:

- `sbert_profile`
- `tfidf_profile`
- `openai_profile`
- `category_random`

---

## Anti-Artifact Guards

The MC builder explicitly filters bad distractors using:

- alias collision checks
- duplicate token overlap checks
- length ratio checks
- overlap-with-question-text checks

Why this matters:

- otherwise a model can exploit answer-surface quirks
- we want the buzzer to rely on clue semantics and timing

---

## Likelihood Models

`models/likelihoods.py` supports:

- TF-IDF
- SBERT
- T5 embeddings
- optional OpenAI embeddings

These models are used to score clue text against answer profiles and produce beliefs over answer options.

This separation lets us compare:

- a fixed semantic scorer + simple buzzer
- a fixed semantic scorer + PPO buzzer
- a T5 policy that reads the text directly

---

## Belief-Feature Pipeline

The modular RL path uses:

- `qb_env/tossup_env.py`
- `models/features.py`
- `agents/threshold_buzzer.py`
- `agents/bayesian_buzzer.py`
- `agents/ppo_buzzer.py`

Observation vector:

- belief over answer options
- top probability
- margin
- entropy
- stability
- progress
- clue index

Action space:

- WAIT
- BUZZ with one of the `K` answer options

---

## T5 Policy Pipeline

The second path is an end-to-end text policy:

- `qb_env/text_wrapper.py`
- `models/t5_policy.py`
- `training/train_supervised_t5.py`
- `training/train_ppo_t5.py`

Workflow:

1. build a text observation from visible clues + answer choices
2. warm-start with supervised learning
3. fine-tune with PPO

This gives us a direct comparison between structured belief-based decisions and a heavier neural policy.

---

## Experimental Setup

Canonical scripts in the repo:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

Additional T5 path:

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/compare_policies.py --config configs/t5_policy.yaml
```

Configs:

- `configs/default.yaml`
- `configs/smoke.yaml`
- `configs/t5_policy.yaml`

---

## Training (Smoke)

Practical training path used in this repo:

1. `python scripts/build_mc_dataset.py --smoke`
2. `python scripts/run_baselines.py --smoke`
3. `python scripts/train_ppo.py --smoke`

What this gives us:

- reproducible MC episodes
- baseline timing/accuracy reference points
- PPO fine-tuning focused on buzz timing decisions

---

## Evaluation (Smoke)

Evaluation source:

- `artifacts/smoke/evaluation_report.json`

Main metrics:

- `S_q`
- buzz accuracy
- mean buzz step
- reward-like score
- calibration at buzz (ECE, Brier)

Controls:

- choices-only
- shuffle
- alias substitution

---

## Results Snapshot (Smoke)

Baseline from `artifacts/smoke/evaluation_report.json`; PPO from best aggregate in `artifacts/smoke/reward_sweep_results.json`:

| Model | mean `S_q` | buzz acc | mean step | reward-like | ECE / Brier |
|---|---:|---:|---:|---:|---:|
| `always_final` (best baseline) | `0.386` | `38.6%` | `4.05` | `0.080` | `0.000 / 0.000` |
| PPO (best sweep aggregate) | `0.340` | `34.1%` | `0.00` | `n/a` | `0.006 / 0.000` |

---

## Results Interpretation (Smoke)

Interpretation:

- In this smoke run, `always_final` is the strongest baseline on `S_q`.
- PPO currently trails the strongest baseline on both `S_q` and buzz accuracy.
- Likely cause: short smoke budget and reward-shaping sensitivity.
- Next run knobs: `ppo.total_timesteps`, `wait_penalty`, `early_buzz_penalty`.
- These are smoke diagnostics for pipeline validation, not final quality claims.

---

## Baselines And Learned Agents

Implemented agents:

- `ThresholdBuzzer`
- `AlwaysBuzzFinalBuzzer`
- `SoftmaxProfileBuzzer`
- `SequentialBayesBuzzer`
- `PPOBuzzer`

Why this baseline set is useful:

- confidence-threshold policy (`ThresholdBuzzer`)
- belief-update policy (`SequentialBayesBuzzer`)
- learned timing policy (PPO), compared against `AlwaysBuzzFinalBuzzer`

---

## Evaluation

`evaluation/` measures more than final accuracy:

- `S_q`, buzz accuracy, and mean buzz step
- calibration-at-buzz (`ECE`, `Brier`)
- per-category accuracy

The code also includes explicit controls:

- choices-only, shuffle, alias substitution

This is important because a buzzer can look strong while exploiting answer artifacts instead of clue content.

---

## Analysis Focus

The codebase is designed to answer questions like:

- Do sequential Bayesian updates change buzzing behavior relative to from-scratch softmax?
- Do learned buzzers trade confidence for earlier timing?
- Do models still perform when clue order or answer aliases are perturbed?
- Is confidence calibrated at the time the system decides to buzz?

The checked-in evaluation outputs include:

- JSON summaries
- comparison tables
- calibration plots
- entropy-vs-clue-index plots

---

## What The Current System Demonstrates

- a full pipeline from raw tossups to multiple-choice episodes
- swappable likelihood models for semantic scoring
- both simple and learned buzzer policies
- experiment-oriented evaluation with artifact controls
- an implementation that is modular enough to compare policy families, not just train one model

---

## Conclusions

- Multiple-choice quiz bowl is a useful controlled setting for studying strategic buzzing
- Answer construction quality is as important as the buzzer itself
- The repository now supports:
  - belief-feature baselines
  - PPO on structured observations
  - T5-based text-policy experiments
  - calibration and artifact-aware evaluation
- The most important contribution of the codebase is the **experiment platform**, not just a single model checkpoint

---

## References

- Rodriguez et al. — QANTA / quiz bowl question answering
- Schulman et al. — Proximal Policy Optimization
- Raffel et al. — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

---

<!-- _class: lead -->

# Thank You
