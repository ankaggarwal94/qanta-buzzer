---
marp: true
theme: default
paginate: true
---

<!-- _class: lead -->

# Quiz Bowl RL Buzzer
## Current `main` Branch Codebase

CS234 Final Project

Kathleen Weng
Imran Hassan
Ankit Aggarwal

March 2026

---

## What This Repository Is Now

- The canonical implementation is the modular pipeline under:
  - `qb_data/`
  - `qb_env/`
  - `models/`
  - `agents/`
  - `evaluation/`
  - `scripts/`
  - `training/`
- The older root-level prototype (`main.py`, `environment.py`, `model.py`, etc.) is still present, but it is no longer the primary path to understand the repo.
- The repo also includes a qb-rl compatibility bridge so older imports and configs can still resolve inside `qanta-buzzer`.

---

## Core Problem Framing

- Task: learn **when to buzz** on a quiz bowl tossup with **multiple-choice** answers
- Questions reveal information incrementally
- The agent must trade off:
  - waiting for more evidence
  - buzzing early enough to gain score advantage
- The codebase supports both:
  - belief-feature decision making
  - end-to-end text-policy experiments

---

## End-to-End Pipeline

1. Load tossup questions from CSV or optional Hugging Face fallback
2. Build answer profiles from historical question text
3. Construct multiple-choice questions with anti-artifact guards
4. Score answer options with a likelihood model
5. Convert beliefs into:
   - feature vectors for MLP/PPO agents, or
   - text observations for the T5 policy track
6. Run baseline or learned buzzers
7. Evaluate with accuracy, `S_q`, calibration, and control experiments

---

## Data Layer

Key modules:

- `qb_data/data_loader.py`
  - `TossupQuestion`
  - CSV-first loading
  - qb-rl-compatible row parsing
  - optional Hugging Face fallback
- `qb_data/answer_profiles.py`
  - leave-one-out answer profiles
- `qb_data/dataset_splits.py`
  - stratified train / val / test creation

The data layer is designed to feed both the modular RL pipeline and the legacy / compatibility surfaces.

---

## Multiple-Choice Construction

`qb_data/mc_builder.py` builds `MCQuestion` objects with:

- `K=4` options by default
- distractor strategies:
  - `sbert_profile`
  - `tfidf_profile`
  - `openai_profile`
  - `category_random`
- anti-artifact guards:
  - alias collision
  - duplicate token overlap
  - length ratio
  - overlap with question text

This is one of the repo's main research contributions: the answer space is constrained, but deliberately guarded against easy artifacts.

---

## Likelihood Models

`models/likelihoods.py` provides pluggable scoring backends:

- TF-IDF
- SBERT
- T5 embeddings
- optional OpenAI embeddings

These likelihoods drive two separate parts of the system:

- distractor ranking during MC construction
- belief updates during buzzing / environment interaction

The repo keeps the likelihood model separate from the buzzer policy, which makes comparisons easier.

---

## Environment

`qb_env/tossup_env.py` models the game as a Gymnasium environment:

- **Action space**: `Discrete(K + 1)`
  - `0` = WAIT
  - `1..K` = buzz with answer option
- **Observation space**: `K + 6` belief-feature vector
  - belief over options
  - top probability
  - margin
  - entropy
  - stability
  - progress
  - clue index

Configurable behavior:

- reward modes: `time_penalty`, `simple`, `human_grounded`
- belief modes: `from_scratch`, `sequential_bayes`

---

## Baseline Agents

Implemented baseline buzzers:

- `ThresholdBuzzer`
- `AlwaysBuzzFinalBuzzer`
- `SoftmaxProfileBuzzer`
- `SequentialBayesBuzzer`

These agents produce:

- `c_trace`: buzz-confidence style trace
- `g_trace`: correctness trace
- entropy / top-probability traces

Those traces are used downstream for `S_q` and calibration-style evaluation, not just raw accuracy.

---

## PPO Agent

`agents/ppo_buzzer.py` wraps Stable-Baselines3 PPO around the belief-feature environment.

Current design:

- MLP policy over belief features
- SB3 handles training
- custom episode execution records per-step traces for evaluation

Canonical training entrypoint:

```bash
python scripts/train_ppo.py --smoke
```

This is the main learned policy path in the modular codebase.

---

## T5 Policy Track

The repo also contains a second, heavier path:

- `models/t5_policy.py`
- `qb_env/text_wrapper.py`
- `training/train_supervised_t5.py`
- `training/train_ppo_t5.py`
- `scripts/train_t5_policy.py`

This track:

- converts the current visible clue text plus choices into a text observation
- uses a T5-based policy/value architecture
- supports supervised warm-start plus PPO fine-tuning

It is an alternative experimental path, not the default smoke workflow.

---

## Evaluation Layer

`evaluation/` contains:

- `metrics.py`
  - `S_q`
  - calibration-at-buzz
  - per-category accuracy
- `controls.py`
  - choices-only
  - shuffle
  - alias substitution
- `plotting.py`
  - calibration curves
  - entropy vs clue index
  - comparison tables / CSV

This makes the repo more than a training script collection: it is built as an experiment pipeline.

---

## Canonical Workflows

Smoke pipeline:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

T5 policy path:

```bash
python scripts/train_t5_policy.py --config configs/t5_policy.yaml
python scripts/compare_policies.py --config configs/t5_policy.yaml
```

Legacy root-level scripts still exist, but the modular `scripts/` path is the one the repo docs and walkthrough now center.

---

## Testing And Validation

The current repo is documented around three validation layers:

1. targeted `pytest` runs
2. the smoke pipeline above
3. `walkthrough.md` as a generated, verified code tour

Key reference docs:

- `README.md`
- `CLAUDE.md`
- `walkthrough.md`
- `.planning/`

There are also local repomix packs for model ingestion, but those are generated artifacts rather than tracked source files.

---

## Current Repo Caveat

This presentation is about the **architecture and intended operation** of the current `main` branch.

Two practical caveats:

- the repo contains both the modular path and the older prototype path
- some files on current `main` still need post-merge cleanup, so this deck avoids overclaiming polished benchmark numbers from the checked-in branch state

So the safest interpretation is:

- the modular pipeline is the canonical design
- the repo is a research codebase, not a polished package release

---

## What The Codebase Demonstrates

- quiz bowl buzzing framed over a multiple-choice answer space
- modular separation of:
  - data
  - likelihoods
  - environment
  - agents
  - evaluation
- both belief-feature RL and text-policy experimentation
- artifact-aware evaluation rather than raw accuracy alone
- compatibility bridging from qb-rl into a single canonical repo

---

<!-- _class: lead -->

# Takeaway

The current repository is best understood as a **modular experiment platform**
for multiple-choice quiz bowl buzzing:

data -> MC construction -> beliefs/text -> buzzer -> evaluation

not as the original single-path T5-only prototype.

---

<!-- _class: lead -->

# Thank You
