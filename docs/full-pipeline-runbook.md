# Full End-to-End Pipeline Runbook

Deterministic instruction set for running the complete qanta-buzzer pipeline
at full scale on the QANTA dataset (~20,407 questions).

---

## Prerequisites

### Hardware

| Resource | Minimum | This machine (Apple M3 Max) |
|----------|---------|---------------------------|
| CPU | 8 cores | 16 cores |
| RAM | 32 GB (t5-base on MPS) | 64 GB |
| GPU | — (CPU ok for tfidf) | MPS (Apple Silicon) |
| Disk | 10 GB free | 38 GB free |

### Local wall-time estimates (Apple M3 Max, 64 GB, MPS)

All estimates assume the full QANTA dataset (~20,407 questions). Phases
marked ★ are the core pipeline; others are optional extensions/ablations.

| Phase | Description | Likelihood | Estimated time |
|-------|------------|------------|----------------|
| **★ 1** | Build MC dataset (SBERT distractors) | — | 5–10 min |
| **★ 2** | Baseline sweeps (TF-IDF) | tfidf | 5–10 min |
| **★ 2** | Baseline sweeps (T5-large) | t5-large | 2–4 hrs |
| **★ 2** | Baseline sweeps (T5-base) | t5-base | 45–90 min |
| **★ 3** | PPO 100k steps (TF-IDF beliefs) | tfidf | 30–60 min |
| **★ 4** | Evaluate all + controls | tfidf | 5–15 min |
| **★ 5** | T5 policy: supervised + PPO (t5-large) | — | 4–8 hrs |
| **★ 5** | T5 policy: supervised + PPO (t5-base) | — | 1.5–3 hrs |
| **★ 5** | T5 policy: supervised + PPO (t5-small) | — | 15–30 min |
| **★ 6** | Compare policies | tfidf | 10–20 min |
| 7 | Multi-seed PPO (3 seeds) | tfidf | 1.5–3 hrs |
| 8 | Reward sweep | tfidf | varies |
| 9 | Distractor comparison (3 strategies) | tfidf | 15–30 min |
| 10 | Variable-K baselines (MaskablePPO not wired) | tfidf | 15–30 min |
| 11 | Expected Wins eval (EW-trained PPO is manual only) | tfidf | 5–15 min |
| 12 | DSPy compile | API-bound | 5–10 min |
| 13 | K-sensitivity (5 values) | tfidf | 30–60 min |
| 14 | Reward mode comparison (3 modes) | tfidf | 1.5–3 hrs |
| 15 | Belief mode comparison | tfidf | 5–10 min |
| 16 | Stop-only PPO | tfidf | 30–60 min |
| 17 | No-buzz horizon | tfidf | 30–60 min |
| 18 | OpenAI embeddings | API-bound | 10–30 min |
| 19 | DSPy MIPROv2 | API-bound | 5–10 min |

**Totals (wall-clock, sequential):**

| Scope | T5-large | T5-base | T5-small / TF-IDF only |
|-------|----------|---------|------------------------|
| Core pipeline (★ Phases 1–6) | 7–13 hrs | 3–5.5 hrs | 1–2 hrs |
| Core + all extensions (1–19) | 12–22 hrs | 7–13 hrs | 5–9 hrs |

**Parallelism opportunities:** After Phase 1 completes, Phases 2/3/5 are
independent and can run in parallel (Wave 1 of `run_full_pipeline.sh`).
Phase 4 must follow Phase 2 (reads `baseline_summary.json`). Phase 11
must follow Phase 4 and run before Phase 15 (it reads `baseline_summary.json`
which Phase 15 overwrites). Phases 9/13/15 each overwrite
`baseline_summary.json` and must run sequentially after Phase 11.

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

## Quick start: automated parallel execution

For an automated run of the core pipeline and scripted extensions, use
`run_full_pipeline.sh`. Parallel mode runs Phases 1–6, 11 (eval only),
13–17; sequential mode (`--sequential`) also includes Phase 9.
Phases 7, 8, 10, 11 (EW-trained PPO), 12, 18, 19 require manual
execution. This is the **recommended path** for
local agents and unattended runs.

```bash
# 1. Clean previous artifacts
rm -rf artifacts/main/ artifacts/k* artifacts/distractor_*
rm -rf cache/embeddings/
rm -rf checkpoints/supervised/ checkpoints/ppo/ checkpoints/ppo_t5/
rm -rf results/

# 2. Run the full pipeline (pick ONE)

# Fastest — t5-small for T5 policy, TF-IDF for baselines (~2 hrs on M3 Max)
bash scripts/run_full_pipeline.sh --t5-model t5-small

# Balanced — t5-base for T5 policy (~3–4 hrs on M3 Max)
bash scripts/run_full_pipeline.sh --t5-model t5-base

# Full quality — t5-large (requires CUDA with 8+ GB VRAM; will likely OOM on Apple Silicon)
# bash scripts/run_full_pipeline.sh --t5-model t5-large

# Sequential (no background jobs, safe for debugging)
bash scripts/run_full_pipeline.sh --sequential --t5-model t5-base
```

The script executes the **core pipeline and extensions** in a 4-wave DAG.
Phases 7 (multi-seed), 8 (reward sweep), 10 (variable-K baselines),
11 EW-trained PPO, 12 (DSPy compile), 18 (OpenAI), and 19 (MIPROv2)
require manual execution — see the individual phase sections below.

```
Phase 1 (sequential — builds the shared MC dataset)
  │
  ├─ Wave 1 (3 parallel tracks):
  │    Track A: Phase 2  — baseline sweeps (→ artifacts/main/baseline_summary.json)
  │    Track B: Phase 3  — PPO training (→ artifacts/main/ppo_model.zip)
  │    Track C: Phase 5  — T5 policy (→ checkpoints/)
  │
  ├─ Wave 2 (sequential — all read/write artifacts/main/):
  │    Phase 4  — full evaluation + controls
  │    Phase 6  — MLP vs T5 comparison
  │    Phase 11 — Expected Wins evaluation
  │    Phase 15 — belief mode comparison
  │
  ├─ Wave 3 (sequential — PPO ablations):
  │    Phase 14 — reward modes (simple, human_grounded)
  │    Phase 16 — stop-only PPO
  │    Phase 17 — no-buzz horizon
  │
  └─ Wave 4 (sequential — K-sensitivity, clobbers baseline_summary.json):
       Phase 13 — K=2,3,5,6 builds + baselines (each result copied to results/)

Not in parallel mode (sequential only or run manually):
  Phase 9  — distractor comparison (sequential mode only)
  Phase 10 — variable-K baselines (MaskablePPO not wired through train_ppo.py)
  Phase 12 — DSPy compile (requires API key)
  Phase 18 — OpenAI embeddings (requires API key)
  Phase 19 — DSPy MIPROv2 (requires API key)
```

**Output:** All JSON results in `results/`. In parallel mode, Waves 1, 2,
and 4 write per-phase logs to `results/phase_*.log` (via `run_phase()`);
Wave 3 (PPO ablations) prints directly to stdout.
In `--sequential` mode, all output goes to stdout (no per-phase logs).

**Monitoring:** Phase logs are written via stdout redirection, so output is
buffered — `tail -f` may show no updates for extended periods (Phase 5 can
appear stuck for 40+ minutes during supervised warm-start). This is normal;
check the process is still running with `ps aux | grep train_`.
```bash
tail -f results/phase_3.log   # PPO training (updates every ~30s)
tail -f results/phase_5.log   # T5 policy (may buffer for long periods)
```

**After completion:** Generate summary table:
```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('results/*.json')):
    s = json.load(open(f))
    name = f.split('/')[-1].replace('.json', '')
    if 'full_eval' in s:
        fe = s['full_eval']
        print(f'{name}: acc={fe.get(\"buzz_accuracy\", \"N/A\")}, S_q={fe.get(\"mean_sq\", \"N/A\")}')
    elif 't5_policy' in s:
        for k in ('mlp_policy', 't5_policy'):
            if k in s:
                m = s[k]
                print(f'{name}/{k}: acc={m.get(\"accuracy\", \"N/A\")}, S_q={m.get(\"mean_sq\", \"N/A\")}')
    elif 'softmax_profile' in s:
        sp = s['softmax_profile']
        best = max(sp.items(), key=lambda x: x[1].get('mean_sq', 0), default=('N/A', {}))
        print(f'{name}: best_threshold={best[0]}, S_q={best[1].get(\"mean_sq\", \"N/A\")}')
    else:
        acc = s.get('buzz_accuracy', s.get('accuracy', 'N/A'))
        sq = s.get('mean_sq', 'N/A')
        print(f'{name}: acc={acc}, S_q={sq}')
"
```

**If a phase fails:** Check the phase log (if available) or stdout, fix the
issue, and re-run just that phase manually (see individual phase sections
below). The script exits on first failure — completed phases don't need
re-running since their outputs are already written.

---

## Agent instructions

If you are an AI coding agent executing this runbook:

1. **Preferred path:** Run `bash scripts/run_full_pipeline.sh --t5-model t5-base` and monitor the logs. This handles dependency ordering for the core pipeline and extensions. Only Wave 1 (baselines, PPO, T5) runs in parallel — all subsequent waves are sequential to avoid artifact races on `artifacts/main/`.

2. **If the script fails:** Check `results/phase_*.log` if available (Waves 1, 2, 4 in parallel mode); for Wave 3 and `--sequential` mode, check stdout. Diagnose, fix, then re-run only the failed phase using the manual commands below.

3. **Do not run phases 2/3/5 sequentially** if the parallel script is available — they are independent and running them in parallel saves 2–3x wall time. Phase 11 must run after Phase 4 and before Phase 15 (it reads `baseline_summary.json` which Phase 15 overwrites). Phases 9/13/15 each overwrite `baseline_summary.json` and must run sequentially after Phase 11.

4. **Do not modify `artifacts/main/mc_dataset.json`** after Phase 1 — every subsequent phase reads it.

5. **Phases 7, 8, 10, 11 (EW PPO), 12, 18, 19** are not in the automated script. Run them manually if needed.

---

## Manual phase-by-phase instructions

The sections below document each phase individually for debugging,
selective re-runs, or environments where the parallel script cannot be used.

---

## Phase 0: Clean state

```bash
rm -rf artifacts/main/ artifacts/k* artifacts/distractor_*
rm -rf cache/embeddings/
rm -rf checkpoints/supervised/ checkpoints/ppo/ checkpoints/ppo_t5/
rm -rf results/
mkdir -p artifacts/main results
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

For **T5-base baseline** (balanced quality/speed, ~45–90 min):

```bash
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=t5-base
```

For **T5-large baseline** (requires CUDA or 64+ GB RAM on MPS, hours of compute):

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
for agent, data in s.items():
    if isinstance(list(data.values())[0], dict):
        best = max(data.items(), key=lambda x: x[1].get('mean_sq', 0))
        print(f'{agent}: best_t={best[0]}, S_q={best[1][\"mean_sq\"]:.3f}, acc={best[1][\"buzz_accuracy\"]:.3f}')
    else:
        print(f'{agent}: S_q={data.get(\"mean_sq\", 0):.3f}, acc={data.get(\"buzz_accuracy\", 0):.3f}')
"
```

**Archive default baselines** (later phases clobber `artifacts/main/baseline_summary.json`):
```bash
# If you ran the TF-IDF command above (recommended):
cp artifacts/main/baseline_summary.json results/baselines_tfidf.json
# If you ran T5-base instead:
# cp artifacts/main/baseline_summary.json results/baselines_t5base.json
# If you ran T5-large instead:
# cp artifacts/main/baseline_summary.json results/baselines_t5large.json
```

> Phases 14–16 assume `results/baselines_tfidf.json` exists for apples-to-apples
> comparisons. If you used T5-large here, those comparisons will mix regimes.

---

## Phase 3: Train PPO (MLP on belief features)

**Config:** `configs/default.yaml`
**Timesteps:** 100,000
**Network:** [64, 64] MLP
**Reward:** time_penalty (wait_penalty=0.05, early_buzz_penalty=0.2, buzz_incorrect=-0.5)
**Likelihood:** add `likelihood.model=tfidf` to match the wrapper and extension phases

```bash
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 \
    --deterministic-eval \
    likelihood.model=tfidf
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

**Archive default PPO** (later phases clobber `artifacts/main/ppo_summary.json`):
```bash
cp artifacts/main/ppo_summary.json results/ppo_default.json
cp artifacts/main/ppo_model.zip results/ppo_model_default.zip
```

---

## Phase 4: Evaluate all (belief-feature pipeline)

**Config:** `configs/default.yaml`
**Controls:** choices-only, shuffle, alias substitution (alias control is a
no-op unless `alias_lookup.json` is provided externally — `build_mc_dataset.py`
does not generate it)
**Metrics:** S_q, ECE, Brier, per-category accuracy

```bash
python scripts/evaluate_all.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=tfidf
```

> **Selective re-run note:** `evaluate_all.py` reads
> `artifacts/main/baseline_summary.json` for the softmax threshold. If later
> phases (13, 15) have overwritten it, restore the TF-IDF archive first
> (this command uses `likelihood.model=tfidf`, so the baseline must match):
> `cp results/baselines_tfidf.json artifacts/main/baseline_summary.json`

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
**Model:** t5-base recommended (220M params); t5-large (770M) is slower and needs more memory
**Supervised:** 10 epochs, effective batch 32
**PPO:** 100 iterations

> **Memory warning (Apple Silicon / MPS):** A full-scale t5-base run on
> 20,407 questions reached ~41 GB physical memory footprint on an M3 Max.
> The "fits in 8 GB" claim from earlier docs was based on model weight size,
> not the actual working set with 20k-question tokenization, gradient buffers,
> and MPS allocations. **Minimum 32 GB RAM is recommended for t5-base at full
> scale.** t5-large will exceed 64 GB and likely OOM on most Apple Silicon Macs.

For **t5-base** (recommended path, matches `run_full_pipeline.sh --t5-model t5-base`):

```bash
python scripts/train_t5_policy.py \
    --config configs/t5_policy.yaml \
    model.model_name=t5-base
```

For **t5-large** (requires CUDA with 8+ GB VRAM, or 64+ GB system RAM on MPS):

```bash
python scripts/train_t5_policy.py \
    --config configs/t5_policy.yaml
```

**Expected behavior:**
1. Supervised warm-start: trains answer selection on complete questions (10 epochs)
2. PPO fine-tuning: optimizes wait/answer policy on incremental episodes (100 iterations)
3. Saves best model to `checkpoints/ppo_t5/best_model/`

**Estimated time:** t5-base ~2–3 hrs on M3 Max MPS; t5-large ~6–8 hrs on CUDA

**Checkpoint:**
```bash
ls checkpoints/ppo_t5/best_model/
cat checkpoints/ppo_t5/test_results.json
```

---

## Phase 6: Compare policies

**Requires:** Phase 3 PPO model + Phase 5 T5 model

> **Comparison caveats:** The MLP and T5 policies use different confidence
> semantics (belief-sigmoid vs wait-head probability) and different reward
> settings (config-driven vs T5-pipeline defaults). Accuracy and buzz-position
> are directly comparable; S_q, ECE, Brier, and reward comparisons are
> qualitative. See the docstring in `compare_policies.py` for details.
>
> `compare_policies.py` auto-detects the device for T5 inference (MPS on
> Apple Silicon, CUDA if available, otherwise CPU).
>
> **Selective re-run note:** Phases 11 (EW-trained PPO), 14, 16, and 17 all
> overwrite `artifacts/main/ppo_model`. If re-running after ablations, restore:
> `cp results/ppo_model_default.zip artifacts/main/ppo_model.zip`

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
        --deterministic-eval \
        likelihood.model=tfidf
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

Grid search over wait_penalty and early_buzz_penalty. Note: this script
is hardwired to use `configs/smoke.yaml` and `artifacts/smoke/` — it does
not accept `--config` or `--mc-path`.

```bash
python scripts/sweep_reward_shaping.py --seeds 13,42,123 --timesteps 3000
```

---

## Full pipeline single-script execution

Run the core pipeline sequentially (Phases 1–6) with TF-IDF beliefs and
t5-base for the T5 policy. Includes archive steps so extension phases
can reference Phase 2/3 defaults from `results/`.

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p results

echo "=== Phase 1: Build MC dataset ==="
python scripts/build_mc_dataset.py --config configs/default.yaml --output-dir artifacts/main

echo "=== Phase 2: Run baselines ==="
python scripts/run_baselines.py --config configs/default.yaml --mc-path artifacts/main/mc_dataset.json likelihood.model=tfidf
cp artifacts/main/baseline_summary.json results/baselines_tfidf.json

echo "=== Phase 3: Train PPO ==="
python scripts/train_ppo.py --config configs/default.yaml --mc-path artifacts/main/mc_dataset.json --seed 13 --deterministic-eval likelihood.model=tfidf
cp artifacts/main/ppo_summary.json results/ppo_default.json
cp artifacts/main/ppo_model.zip results/ppo_model_default.zip

echo "=== Phase 4: Evaluate all ==="
python scripts/evaluate_all.py --config configs/default.yaml --mc-path artifacts/main/mc_dataset.json likelihood.model=tfidf
cp artifacts/main/evaluation_report.json results/eval_default.json

echo "=== Phase 5: Train T5 policy (t5-base) ==="
python scripts/train_t5_policy.py --config configs/t5_policy.yaml model.model_name=t5-base

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

**`artifacts/main/` is a working directory** — files are overwritten by later
phases (e.g. K-sensitivity clobbers `baseline_summary.json`, PPO ablations
clobber `ppo_summary.json`). The **stable outputs** are in `results/*.json`,
which are copied after each phase completes.

The wrapper (`run_full_pipeline.sh`) writes all outputs to top-level
`results/*.json`. The manual extension sections write to subdirectories
(`results/k_sensitivity/`, `results/reward_modes/`, `results/belief_modes/`,
`results/policy_modes/`). Both are valid — the tree below shows the wrapper
layout; see manual phase sections for subdirectory paths.

```
results/                         # Stable per-phase outputs (wrapper layout)
├── baselines_tfidf.json         # Phase 2 baseline summary
├── ppo_default.json             # Phase 3 PPO summary
├── ppo_model_default.zip        # Phase 3 PPO model
├── eval_default.json            # Phase 4 evaluation report
├── t5_comparison.json           # Phase 6 policy comparison
├── eval_ew_logistic.json        # Phase 11 Expected Wins eval
├── baselines_seqbayes.json      # Phase 15 belief mode
├── ppo_simple.json              # Phase 14 reward ablation
├── ppo_human_grounded.json      # Phase 14 reward ablation
├── ppo_stop_only.json           # Phase 16 stop-only PPO
├── ppo_no_buzz.json             # Phase 17 no-buzz horizon
├── baselines_k{2,3,5,6}.json   # Phase 13 K-sensitivity (k4 is default)
├── baselines_tfidf_profile.json # Phase 9 (sequential mode only)
├── baselines_category_random.json # Phase 9 (sequential mode only)
│
│  # Manual extension subdirectories (not created by wrapper):
├── k_sensitivity/               # Phase 13 manual
├── reward_modes/                # Phase 14 manual
├── belief_modes/                # Phase 15 manual
└── policy_modes/                # Phase 16 manual

artifacts/main/                  # Working directory (overwritten by later phases)
├── mc_dataset.json              # Stable — built in Phase 1, never overwritten
├── train_dataset.json
├── val_dataset.json
├── test_dataset.json
├── answer_profiles.json
└── (baseline/ppo/eval files)    # Overwritten by later phases

checkpoints/
├── supervised/best_model/       # T5 supervised checkpoint
└── ppo_t5/best_model/           # T5 PPO checkpoint
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
    cp artifacts/main/baseline_summary.json "results/baselines_distractor_${STRATEGY}.json"
done
```

**Checkpoint:**
```bash
for STRATEGY in sbert tfidf catrandom; do
    python -c "
import json
s = json.load(open('results/baselines_distractor_${STRATEGY}.json'))
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

**Note:** PPO with variable-K and MaskablePPO is not yet wired end-to-end
through `train_ppo.py`. The `PPOBuzzer` class supports `use_maskable_ppo=True`
and the env supports `action_masks()`, but `train_ppo.py` does not read
`ppo.algorithm` from config or pass `use_maskable_ppo` to the constructor.
This requires a code change to `train_ppo.py` before it will work.
For now, variable-K baselines (which don't need MaskablePPO) work correctly.

---

### Phase 11: Expected Wins evaluation

Evaluate the SoftmaxProfile baseline (from `evaluate_all.py`) using the
Expected Wins metric with a logistic opponent model. Note: this evaluates
the baseline agents, not the PPO model — to train PPO with Expected Wins
reward, see the separate command below.

> **Selective re-run note:** Like Phase 4, this reads
> `artifacts/main/baseline_summary.json`. If later phases have overwritten it,
> restore the TF-IDF archive (this command uses `likelihood.model=tfidf`):
> `cp results/baselines_tfidf.json artifacts/main/baseline_summary.json`

```bash
# Evaluate with Expected Wins reward mode and logistic opponent
python scripts/evaluate_all.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=tfidf \
    environment.reward_mode=expected_wins \
    environment.opponent_buzz_model.type=logistic \
    environment.opponent_buzz_model.midpoint=0.6 \
    environment.opponent_buzz_model.steepness=6.0
cp artifacts/main/evaluation_report.json results/eval_expected_wins_logistic.json

# Also try empirical opponent (uses human_buzz_positions from QANTA data)
python scripts/evaluate_all.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=tfidf \
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
    likelihood.model=tfidf \
    environment.reward_mode=expected_wins \
    environment.opponent_buzz_model.type=logistic
cp artifacts/main/ppo_summary.json results/ppo_expected_wins.json
cp artifacts/main/ppo_model.zip results/ppo_model_expected_wins.zip
```

---

### Phase 12: DSPy offline compile (experimental — not wired end-to-end)

Compile a DSPy-optimized scorer using the training split.
Requires the `dspy` extra and an LM API key.

> **Limitation:** `optimize_dspy.py` compiles and reports metrics, but does
> not persist the compiled program in a way that `build_likelihood_from_config()`
> can load it. Setting `likelihood.model=dspy` constructs `DSPyLikelihood`
> with a placeholder uniform scorer, not the compiled program. This phase
> is useful for validating the DSPy pipeline contract, but the evaluated
> baselines below will use uniform scores — not the compiled model's.
>
> **Data path caveat:** `optimize_dspy.py` prefers
> `artifacts/smoke/train_dataset.json` over `artifacts/main/train_dataset.json`.
> If you have run the smoke pipeline, it will silently compile on the 50-question
> smoke split. To force the full training split, remove or rename the smoke
> artifact first: `rm artifacts/smoke/train_dataset.json`

```bash
pip install -e '.[dspy]'
export OPENAI_API_KEY=...  # or configure another LM backend

# Compile scorer against training split (reports metrics but does not persist)
python scripts/optimize_dspy.py \
    --config configs/default.yaml \
    --max-examples 100

# Evaluate with DSPy scorer (NOTE: uses placeholder uniform scorer, not compiled)
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

# time_penalty (default — already done in Phase 3, archived to results/ppo_default.json)
cp results/ppo_default.json results/reward_modes/ppo_time_penalty.json

# simple (+1/-1, no wait penalty)
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 --deterministic-eval \
    likelihood.model=tfidf environment.reward_mode=simple
cp artifacts/main/ppo_summary.json results/reward_modes/ppo_simple.json

# human_grounded (0 reward if agent buzzes after sampled human position)
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 --deterministic-eval \
    likelihood.model=tfidf environment.reward_mode=human_grounded
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

# from_scratch (default — already done in Phase 2, archived to results/baselines_tfidf.json)
cp results/baselines_tfidf.json results/belief_modes/baselines_from_scratch.json

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

# flat_kplus1 (default — already done in Phase 3, archived to results/ppo_default.json)
cp results/ppo_default.json results/policy_modes/ppo_flat_kplus1.json

# stop_only (Discrete(2), answer = argmax belief)
python scripts/train_ppo.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --seed 13 --deterministic-eval \
    --policy-mode stop_only likelihood.model=tfidf
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
    likelihood.model=tfidf environment.end_mode=no_buzz environment.no_buzz_reward=-0.25
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
- Current: commit `40cb9a3`, Python 3.13.5

## Known scale risks

- **T5-large likelihood** requires ~3 GB VRAM and is slow on CPU. Use `likelihood.model=tfidf` or `likelihood.model=t5-base` for faster iteration.
- **100k PPO timesteps** takes 30–90 minutes on CPU. Reduce with `--timesteps 10000` for quick validation.
- **SBERT distractor ranking** downloads `all-MiniLM-L6-v2` (~90 MB) on first run. Use `data.distractor_strategy=category_random` to skip.
- **Embedding cache** grows to ~42 MB for ~1000 questions with TF-IDF. Monitor via `model.cache_memory_bytes`.

---

## Wall-time summary (Apple M3 Max, parallel vs sequential)

| Mode | t5-small | t5-base | t5-large (CUDA only) |
|------|----------|---------|---------------------|
| `run_full_pipeline.sh` (parallel) | ~2 hrs | ~3–4 hrs | ~6–8 hrs |
| `run_full_pipeline.sh --sequential` | ~5 hrs | ~7–10 hrs | ~12–18 hrs |

t5-large will likely OOM on Apple Silicon Macs (64 GB) at full scale. Use t5-base on MPS.
