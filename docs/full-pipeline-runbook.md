# Full End-to-End Pipeline Runbook

Deterministic instruction set for running the complete qanta-buzzer pipeline
at full scale on the QANTA dataset (~20,407 questions).

---

## Prerequisites

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16 GB | 32+ GB |
| GPU VRAM | — (MPS/CPU ok for t5-base) | 8+ GB CUDA (for t5-large) |
| Disk | 10 GB free | 20 GB free |
| Time (total) | ~6–12 hours (MPS/CPU) | ~2–4 hours (CUDA) |

### Software

```bash
cd /path/to/qanta-buzzer
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Data

The file `questions.csv` must exist at the repo root. It contains ~20,407
QANTA quiz bowl questions with `|||`-separated clue tokens.

### Verify baseline

```bash
pytest tests/ -q --tb=no      # expect: 342 passed, 3 skipped
bash scripts/manual-smoke.sh   # expect: 4/4 stages complete
```

---

## Phase 0: Clean state

```bash
# Remove previous full-run artifacts (keep smoke artifacts intact)
rm -rf artifacts/main/
rm -rf cache/embeddings/
rm -rf checkpoints/supervised/ checkpoints/ppo/ checkpoints/ppo_t5/
rm -rf results/
mkdir -p artifacts/main
```

---

## Phase 1: Build MC dataset

**Config:** `configs/default.yaml`
**Distractor strategy:** `sbert_profile` (SBERT-based semantic ranking)
**K:** 4 fixed answer choices
**Expected output:** `artifacts/main/mc_dataset.json`, split files, answer profiles

```bash
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/main
```

**Expected behavior:**
- Loads ~20,407 questions from `questions.csv`
- Downloads `all-MiniLM-L6-v2` SBERT model (~90 MB) on first run
- Builds answer profiles with leave-one-out
- Ranks distractors by SBERT profile similarity (top-M argpartition)
- Applies 4 anti-artifact guards
- Creates stratified train/val/test splits (70/15/15)
- Writes `mc_dataset.json`, `train_dataset.json`, `val_dataset.json`, `test_dataset.json`, `answer_profiles.json`

**Estimated time:** 5–15 minutes (SBERT encoding is the bottleneck)

**Checkpoint:** Verify `artifacts/main/mc_dataset.json` exists and has >10,000 entries:
```bash
python -c "import json; d=json.load(open('artifacts/main/mc_dataset.json')); print(f'{len(d)} MC questions')"
```

---

## Phase 2: Run baseline sweeps

**Config:** `configs/default.yaml`
**Thresholds:** [0.5, 0.6, 0.7, 0.8, 0.9]
**Agents:** ThresholdBuzzer, SoftmaxProfileBuzzer, SequentialBayesBuzzer, AlwaysBuzzFinal
**Likelihood:** t5-large (from config); override to tfidf for speed

For a **fast first pass** using TF-IDF (minutes, not hours):

```bash
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=tfidf
```

For the **full T5-large baseline** (requires ~3 GB download, hours of compute):

```bash
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json
```

**Expected output:** `artifacts/main/baseline_summary.json`, per-agent run files

**Checkpoint:**
```bash
python -c "
import json
s = json.load(open('artifacts/main/baseline_summary.json'))
for agent, thresholds in s.items():
    best = max(thresholds.items(), key=lambda x: x[1].get('mean_sq', 0))
    print(f'{agent}: best threshold={best[0]}, S_q={best[1][\"mean_sq\"]:.3f}, acc={best[1][\"buzz_accuracy\"]:.3f}')
"
```

---

## Phase 3: Train PPO (MLP on belief features)

**Config:** `configs/default.yaml`
**Timesteps:** 100,000
**Network:** [64, 64] MLP
**Reward:** time_penalty (wait_penalty=0.05, early_buzz_penalty=0.2, buzz_incorrect=-0.5)

```bash
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 \
    --deterministic-eval
```

**Expected behavior:**
- Precomputes belief trajectories for all questions (one-time, ~minutes)
- Trains SB3 PPO for 100k timesteps
- Evaluates with deterministic policy
- Saves model to `artifacts/main/ppo_model.zip`

**Estimated time:** 30–90 minutes (CPU-bound: env stepping, not GPU)

**Checkpoint:**
```bash
ls -lh artifacts/main/ppo_model.zip
python -c "import json; s=json.load(open('artifacts/main/ppo_summary.json')); print(f'PPO: acc={s[\"buzz_accuracy\"]:.3f}, S_q={s[\"mean_sq\"]:.3f}')"
```

---

## Phase 4: Evaluate all (belief-feature pipeline)

**Config:** `configs/default.yaml`
**Controls:** choices-only, shuffle, alias substitution
**Metrics:** S_q, ECE, Brier, per-category accuracy, bootstrap CIs

```bash
python scripts/evaluate_all.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json
```

**Expected output:** `artifacts/main/evaluation_report.json`, `artifacts/main/plots/`

**Checkpoint:**
```bash
python -c "
import json
r = json.load(open('artifacts/main/evaluation_report.json'))
fe = r['full_eval']
print(f'Full eval: acc={fe[\"buzz_accuracy\"]:.3f}, S_q={fe[\"mean_sq\"]:.3f}, ECE={fe[\"ece\"]:.3f}, Brier={fe[\"brier\"]:.3f}')
for name, ctrl in r['controls'].items():
    print(f'  {name}: acc={ctrl.get(\"accuracy\", ctrl.get(\"buzz_accuracy\", \"N/A\"))}')
"
```

---

## Phase 5: Train T5 policy (end-to-end)

**Config:** `configs/t5_policy.yaml`
**Model:** t5-large (770M params, ~3 GB)
**Supervised:** 10 epochs, effective batch 32
**PPO:** 100 iterations

For **t5-large** (requires CUDA or long MPS run):

```bash
python scripts/train_t5_policy.py \
    --config configs/t5_policy.yaml
```

For **t5-base** (220M params, fits in 8 GB, ~2x faster):

```bash
python scripts/train_t5_policy.py \
    --config configs/t5_policy.yaml \
    model.model_name=t5-base
```

**Expected behavior:**
1. Supervised warm-start: trains answer selection on complete questions (10 epochs)
2. PPO fine-tuning: optimizes wait/answer policy on incremental episodes (100 iterations)
3. Saves best model to `checkpoints/ppo_t5/best_model/`

**Estimated time:** 2–8 hours depending on GPU/model size

**Checkpoint:**
```bash
ls checkpoints/ppo_t5/best_model/
cat checkpoints/ppo_t5/test_results.json
```

---

## Phase 6: Compare policies

**Requires:** Phase 3 PPO model + Phase 5 T5 model

```bash
python scripts/compare_policies.py \
    --mlp-checkpoint artifacts/main/ppo_model \
    --t5-checkpoint checkpoints/ppo_t5/best_model \
    --mc-path artifacts/main/mc_dataset.json \
    --output results/t5_comparison.json
```

**Expected output:** Side-by-side metrics table + `results/t5_comparison.json`

**Checkpoint:**
```bash
python -c "
import json
c = json.load(open('results/t5_comparison.json'))
for policy in ['mlp_policy', 't5_policy']:
    if policy in c:
        p = c[policy]
        print(f'{policy}: acc={p[\"accuracy\"]:.3f}, S_q={p[\"mean_sq\"]:.3f}, ECE={p[\"ece\"]:.3f}')
if 'difference' in c:
    d = c['difference']
    print(f'Δ accuracy: {d[\"accuracy\"]:+.3f}, Δ S_q: {d[\"mean_sq\"]:+.3f}')
"
```

---

## Phase 7: Multi-seed validation (optional)

Run PPO training with 3 seeds to assess variance:

```bash
for SEED in 1 2 3; do
    echo "=== Seed $SEED ==="
    python scripts/train_ppo.py \
        --config configs/default.yaml \
        --mc-path artifacts/main/mc_dataset.json \
        --seed $SEED \
        --deterministic-eval
    cp artifacts/main/ppo_summary.json "results/ppo_seed${SEED}.json"
    cp artifacts/main/ppo_model.zip "results/ppo_model_seed${SEED}.zip"
done

python -c "
import json
for seed in [1, 2, 3]:
    s = json.load(open(f'results/ppo_seed{seed}.json'))
    print(f'Seed {seed}: acc={s[\"buzz_accuracy\"]:.3f}, S_q={s[\"mean_sq\"]:.3f}, reward={s[\"mean_reward_like\"]:.3f}')
"
```

---

## Phase 8: Reward sweep (optional)

Grid search over wait_penalty and early_buzz_penalty:

```bash
python scripts/sweep_reward_shaping.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json
```

---

## Full pipeline single-script execution

Run all phases sequentially (Phases 1–4 of the belief-feature pipeline):

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Phase 1: Build MC dataset ==="
python scripts/build_mc_dataset.py --config configs/default.yaml --output-dir artifacts/main

echo "=== Phase 2: Run baselines ==="
python scripts/run_baselines.py --config configs/default.yaml --mc-path artifacts/main/mc_dataset.json

echo "=== Phase 3: Train PPO ==="
python scripts/train_ppo.py --config configs/default.yaml --mc-path artifacts/main/mc_dataset.json --seed 13 --deterministic-eval

echo "=== Phase 4: Evaluate all ==="
python scripts/evaluate_all.py --config configs/default.yaml --mc-path artifacts/main/mc_dataset.json

echo "=== Phase 5: Train T5 policy ==="
python scripts/train_t5_policy.py --config configs/t5_policy.yaml

echo "=== Phase 6: Compare policies ==="
python scripts/compare_policies.py \
    --mlp-checkpoint artifacts/main/ppo_model \
    --t5-checkpoint checkpoints/ppo_t5/best_model \
    --mc-path artifacts/main/mc_dataset.json \
    --output results/t5_comparison.json

echo "=== Pipeline complete ==="
```

---

## Expected artifact tree after full run

```
artifacts/main/
├── mc_dataset.json           # All MC questions (~15k+)
├── train_dataset.json        # 70% train split
├── val_dataset.json          # 15% validation split
├── test_dataset.json         # 15% test split
├── answer_profiles.json      # Profile metadata
├── alias_lookup.json         # Answer alias map
├── baseline_summary.json     # All baseline sweep results
├── baseline_threshold_runs.json
├── baseline_softmax_profile_runs.json
├── baseline_sequential_bayes_runs.json
├── baseline_floor_runs.json
├── ppo_model.zip             # Trained PPO model
├── ppo_runs.json             # PPO evaluation traces
├── ppo_summary.json          # PPO summary metrics
├── evaluation_report.json    # Full eval + controls + per-category
└── plots/
    ├── entropy_vs_clue.png
    ├── calibration.png
    └── comparison.csv

checkpoints/
├── supervised/best_model/    # T5 supervised checkpoint
└── ppo_t5/best_model/        # T5 PPO checkpoint

results/
└── t5_comparison.json        # MLP vs T5 comparison
```

---

## Extension Experiments

These phases exercise the three opt-in extensions. Each is independent
and can be run after the core pipeline (Phases 1–6) completes.

### Phase 9: Distractor strategy comparison

Build three MC datasets with different distractor selection strategies
and compare baseline performance across them.

```bash
mkdir -p artifacts/distractor_comparison

# Strategy A: SBERT semantic ranking (default — already built in Phase 1)
cp artifacts/main/mc_dataset.json artifacts/distractor_comparison/mc_sbert.json

# Strategy B: TF-IDF profile ranking
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/distractor_comparison/tfidf \
    data.distractor_strategy=tfidf_profile
cp artifacts/distractor_comparison/tfidf/mc_dataset.json artifacts/distractor_comparison/mc_tfidf.json

# Strategy C: Category-random (no semantic ranking)
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/distractor_comparison/catrandom \
    data.distractor_strategy=category_random
cp artifacts/distractor_comparison/catrandom/mc_dataset.json artifacts/distractor_comparison/mc_catrandom.json

# Run baselines on each (TF-IDF likelihood for speed)
for STRATEGY in sbert tfidf catrandom; do
    echo "=== Baselines on $STRATEGY distractors ==="
    python scripts/run_baselines.py \
        --config configs/default.yaml \
        --mc-path "artifacts/distractor_comparison/mc_${STRATEGY}.json" \
        likelihood.model=tfidf
    cp artifacts/main/baseline_summary.json "results/baselines_${STRATEGY}.json"
done
```

**Checkpoint:**
```bash
for STRATEGY in sbert tfidf catrandom; do
    python -c "
import json
s = json.load(open('results/baselines_${STRATEGY}.json'))
best = max(s.get('softmax_profile', {}).items(), key=lambda x: x[1].get('mean_sq', 0), default=('N/A', {}))
print(f'${STRATEGY}: best_threshold={best[0]}, S_q={best[1].get(\"mean_sq\", 0):.3f}')
"
done
```

---

### Phase 10: Variable-K experiment

Build a mixed-K dataset and train PPO with action masking to evaluate
how varying the number of answer options affects buzzer performance.

```bash
mkdir -p artifacts/variable_k

# Build mixed-K dataset (K sampled uniformly from 2 to 6 per question)
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/variable_k \
    data.variable_K=true data.min_K=2 data.max_K=6 data.K=6 \
    data.distractor_strategy=category_random

# Verify mixed K
python -c "
import json
qs = json.load(open('artifacts/variable_k/mc_dataset.json'))
from collections import Counter
k_counts = Counter(len(q['options']) for q in qs)
print(f'{len(qs)} questions, K distribution: {dict(sorted(k_counts.items()))}')
"

# Run baselines (agents are K-agnostic)
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/variable_k/mc_dataset.json \
    likelihood.model=tfidf
cp artifacts/main/baseline_summary.json results/baselines_variable_k.json
```

**Note:** PPO with variable-K requires `MaskablePPO` from `sb3-contrib`:
```bash
pip install -e '.[maskable]'
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/variable_k/mc_dataset.json \
    --seed 13 \
    --deterministic-eval \
    environment.variable_K=true environment.max_K=6 environment.use_action_masking=true \
    ppo.algorithm=maskable_ppo
```

---

### Phase 11: Expected Wins evaluation

Re-evaluate the trained PPO model from Phase 3 using the Expected Wins
metric with a logistic opponent model.

```bash
# Evaluate with Expected Wins reward mode and logistic opponent
python scripts/evaluate_all.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    environment.reward_mode=expected_wins \
    environment.opponent_buzz_model.type=logistic \
    environment.opponent_buzz_model.midpoint=0.6 \
    environment.opponent_buzz_model.steepness=6.0
cp artifacts/main/evaluation_report.json results/eval_expected_wins_logistic.json

# Also try empirical opponent (uses human_buzz_positions from QANTA data)
python scripts/evaluate_all.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    environment.reward_mode=expected_wins \
    environment.opponent_buzz_model.type=empirical
cp artifacts/main/evaluation_report.json results/eval_expected_wins_empirical.json
```

**Checkpoint:**
```bash
for MODEL in logistic empirical; do
    python -c "
import json
r = json.load(open('results/eval_expected_wins_${MODEL}.json'))
ew = r.get('expected_wins', {})
fe = r['full_eval']
print(f'EW (${MODEL}): mean_ew={ew.get(\"mean_ew\", \"N/A\")}, S_q={fe[\"mean_sq\"]:.3f}, acc={fe[\"buzz_accuracy\"]:.3f}')
"
done
```

**Train PPO with Expected Wins reward** (trains a new model optimizing for EW):
```bash
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 \
    --deterministic-eval \
    environment.reward_mode=expected_wins \
    environment.opponent_buzz_model.type=logistic
cp artifacts/main/ppo_summary.json results/ppo_expected_wins.json
cp artifacts/main/ppo_model.zip results/ppo_model_expected_wins.zip
```

---

### Phase 12: DSPy offline compile (experimental)

Compile a DSPy-optimized scorer using the training split, then evaluate.
Requires the `dspy` extra and an LM API key.

```bash
pip install -e '.[dspy]'
export OPENAI_API_KEY=...  # or configure another LM backend

# Compile scorer against training split
python scripts/optimize_dspy.py \
    --config configs/default.yaml \
    --max-examples 100

# Evaluate with DSPy scorer
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=dspy
cp artifacts/main/baseline_summary.json results/baselines_dspy.json
```

---

### Phase 13: K-sensitivity analysis (fixed K = 2, 3, 4, 5, 6)

Build 5 separate datasets with different fixed K values and compare baseline
performance to measure how answer-set size affects difficulty.

```bash
mkdir -p results/k_sensitivity

for K in 2 3 4 5 6; do
    echo "=== K=$K ==="
    python scripts/build_mc_dataset.py \
        --config configs/default.yaml \
        --output-dir "artifacts/k${K}" \
        data.K=$K data.distractor_strategy=category_random

    python scripts/run_baselines.py \
        --config configs/default.yaml \
        --mc-path "artifacts/k${K}/mc_dataset.json" \
        likelihood.model=tfidf

    cp artifacts/main/baseline_summary.json "results/k_sensitivity/baselines_k${K}.json"
done

# Summarize
python -c "
import json
for k in [2, 3, 4, 5, 6]:
    s = json.load(open(f'results/k_sensitivity/baselines_k{k}.json'))
    sp = s.get('softmax_profile', {})
    best = max(sp.items(), key=lambda x: x[1].get('mean_sq', 0), default=('N/A', {}))
    n = json.load(open(f'artifacts/k{k}/mc_dataset.json'))
    print(f'K={k}: {len(n)} questions, best S_q={best[1].get(\"mean_sq\", 0):.3f}, acc={best[1].get(\"buzz_accuracy\", 0):.3f}')
"
```

---

### Phase 14: Reward mode comparison

Train PPO under each reward mode and compare final metrics.

```bash
mkdir -p results/reward_modes

# time_penalty (default — already done in Phase 3)
cp artifacts/main/ppo_summary.json results/reward_modes/ppo_time_penalty.json

# simple (+1/-1, no wait penalty)
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 --deterministic-eval \
    environment.reward_mode=simple
cp artifacts/main/ppo_summary.json results/reward_modes/ppo_simple.json

# human_grounded (0 reward if agent buzzes after sampled human position)
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 --deterministic-eval \
    environment.reward_mode=human_grounded
cp artifacts/main/ppo_summary.json results/reward_modes/ppo_human_grounded.json

# Summarize
python -c "
import json
for mode in ['time_penalty', 'simple', 'human_grounded']:
    s = json.load(open(f'results/reward_modes/ppo_{mode}.json'))
    print(f'{mode}: acc={s[\"buzz_accuracy\"]:.3f}, S_q={s[\"mean_sq\"]:.3f}, reward={s[\"mean_reward_like\"]:.3f}')
"
```

---

### Phase 15: Belief mode comparison

Compare from-scratch vs sequential-Bayes belief computation for baselines.

```bash
mkdir -p results/belief_modes

# from_scratch (default — already done in Phase 2)
cp artifacts/main/baseline_summary.json results/belief_modes/baselines_from_scratch.json

# sequential_bayes (Bayesian update: posterior = prior * likelihood)
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    environment.belief_mode=sequential_bayes \
    likelihood.model=tfidf
cp artifacts/main/baseline_summary.json results/belief_modes/baselines_sequential_bayes.json

python -c "
import json
for mode in ['from_scratch', 'sequential_bayes']:
    s = json.load(open(f'results/belief_modes/baselines_{mode}.json'))
    sp = s.get('softmax_profile', {})
    best = max(sp.items(), key=lambda x: x[1].get('mean_sq', 0), default=('N/A', {}))
    print(f'{mode}: best S_q={best[1].get(\"mean_sq\", 0):.3f}, acc={best[1].get(\"buzz_accuracy\", 0):.3f}')
"
```

---

### Phase 16: Stop-only PPO (factored action space)

Train PPO with the Discrete(2) stop-only wrapper where the agent only
decides WAIT/BUZZ and the answer is selected by argmax(belief).

```bash
mkdir -p results/policy_modes

# flat_kplus1 (default — already done in Phase 3)
cp artifacts/main/ppo_summary.json results/policy_modes/ppo_flat_kplus1.json

# stop_only (Discrete(2), answer = argmax belief)
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 --deterministic-eval \
    --policy-mode stop_only
cp artifacts/main/ppo_summary.json results/policy_modes/ppo_stop_only.json

python -c "
import json
for mode in ['flat_kplus1', 'stop_only']:
    s = json.load(open(f'results/policy_modes/ppo_{mode}.json'))
    print(f'{mode}: acc={s[\"buzz_accuracy\"]:.3f}, S_q={s[\"mean_sq\"]:.3f}')
"
```

---

### Phase 17: No-buzz horizon mode

Evaluate with `end_mode=no_buzz` where the agent receives `no_buzz_reward`
instead of being forced to answer at the end of the question.

```bash
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 --deterministic-eval \
    environment.end_mode=no_buzz environment.no_buzz_reward=-0.25
cp artifacts/main/ppo_summary.json results/ppo_no_buzz.json

python -c "
import json
s = json.load(open('results/ppo_no_buzz.json'))
print(f'no_buzz: acc={s[\"buzz_accuracy\"]:.3f}, S_q={s[\"mean_sq\"]:.3f}, reward={s[\"mean_reward_like\"]:.3f}')
"
```

---

### Phase 18: OpenAI embedding pipeline (requires API key)

Run the full pipeline with OpenAI embeddings for both likelihood scoring
and distractor generation.

```bash
pip install -e '.[openai]'
export OPENAI_API_KEY=...

# Build dataset with OpenAI-profile distractors
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/openai \
    data.distractor_strategy=openai_profile

# Run baselines with OpenAI likelihood
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/openai/mc_dataset.json \
    likelihood.model=openai
cp artifacts/main/baseline_summary.json results/baselines_openai.json
```

---

### Phase 19: DSPy MIPROv2 optimizer (experimental)

Compare BootstrapFewShot vs MIPROv2 optimizers for DSPy scorer compilation.

```bash
pip install -e '.[dspy]'
export OPENAI_API_KEY=...

# BootstrapFewShot (default — already done in Phase 12 if run)
python scripts/optimize_dspy.py --config configs/default.yaml --optimizer BootstrapFewShot

# MIPROv2
python scripts/optimize_dspy.py --config configs/default.yaml --optimizer MIPROv2
```

---

## Reproducibility notes

- All random seeds are explicit: `data.shuffle_seed=42`, `environment.seed=13`, `ppo.seed=13`
- Dataset splits use `hashlib.md5` (immune to PYTHONHASHSEED)
- Same seed + same data + same code = identical results
- To reproduce exactly: pin the git commit hash and Python version
- Current: commit `$(git rev-parse --short HEAD)`, Python 3.13.5

## Known scale risks

- **T5-large likelihood** requires ~3 GB VRAM and is slow on CPU. Use `likelihood.model=tfidf` or `likelihood.model=t5-base` for faster iteration.
- **100k PPO timesteps** takes 30–90 minutes on CPU. Reduce with `--timesteps 10000` for quick validation.
- **SBERT distractor ranking** downloads `all-MiniLM-L6-v2` (~90 MB) on first run. Use `data.distractor_strategy=category_random` to skip.
- **Embedding cache** grows to ~42 MB for ~1000 questions with TF-IDF. Monitor via `model.cache_memory_bytes`.
