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
