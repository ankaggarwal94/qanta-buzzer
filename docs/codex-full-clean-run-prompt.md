# Codex: Full Clean Pipeline Run (All Non-API Phases)

**Generated:** 2026-03-16 | **Repo commit:** `6304875` (review-fixes branch)

## Critical: Use the Main Repo

```bash
cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer
git rev-parse --short HEAD
```

Do **NOT** use any Codex worktree at `~/.codex/worktrees/*/`.

## Objective

Run the complete qanta-buzzer pipeline from scratch — all phases except those
requiring OpenAI/DSPy API keys. This is a clean run on the codebase that
includes all runtime correctness fixes from PR #13: opponent model wiring,
variable-K belief shapes, model-variant-specific embedding cache, no-buzz
calibration, padded action guards, and full MaskablePPO train/load/eval wiring.

Produce `results/FULL_RUN_REPORT.md` as the canonical handoff artifact.

## Machine

- Apple M3 Max, 16 cores, 64 GB RAM, MPS available
- Python 3.13.5 in `.venv/`
- Memory warning: t5-base at full scale reaches ~41 GB on MPS

## Setup

```bash
cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer
git checkout review-fixes
source .venv/bin/activate
pip install -e .
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
pytest tests/ -q --tb=short    # expect: 344 passed, 3 skipped
```

## Phase 0: Clean state

```bash
rm -rf artifacts/main/ artifacts/k* artifacts/distractor_* artifacts/variable_k/
rm -rf cache/embeddings/
rm -rf checkpoints/supervised/ checkpoints/ppo/ checkpoints/ppo_t5/
rm -rf results/
mkdir -p artifacts/main results
```

## Execution Plan

### Step 1: Core pipeline via wrapper (Phases 1–6, 11, 13–17)

```bash
bash scripts/run_full_pipeline.sh --t5-model t5-base 2>&1 | tee results/wrapper_stdout.log
```

The wrapper runs a 4-wave DAG:
- **Wave 1 (parallel):** Phase 2 (TF-IDF baselines), Phase 3 (PPO), Phase 5 (T5 policy)
- **Wave 2 (sequential):** Phase 4 (evaluate), Phase 6 (compare), Phase 11 (EW eval), Phase 15 (belief mode)
- **Wave 3 (sequential):** Phase 14 (reward modes), Phase 16 (stop-only), Phase 17 (no-buzz)
- **Wave 4 (sequential):** Phase 13 (K-sensitivity: K=2,3,5,6)

All belief-feature phases use `likelihood.model=tfidf`. Phase 5 uses t5-base on MPS.
Logs for Waves 1/2/4 are in `results/phase_*.log` (unbuffered via `PYTHONUNBUFFERED=1`).
Wave 3 prints to stdout.

**Monitoring:**
```bash
tail -f results/phase_5.log       # T5 training (may be sparse during supervised warm-start)
ps aux | grep train_t5_policy     # verify process is running
```

**Estimated time:** ~3–4 hours.

### Step 2: Phase 7 — Multi-seed PPO

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
```

**Estimated time:** ~6–10 minutes.

### Step 3: Phase 8 — Reward sweep (smoke-scale)

```bash
python scripts/sweep_reward_shaping.py --seeds 13,42,123 --timesteps 3000 \
    | tee results/phase_8_sweep.txt
```

Note: this script uses `configs/smoke.yaml` and `artifacts/smoke/` — it does
not run on the full dataset. It also writes `artifacts/smoke/reward_sweep_results.json`.

**Estimated time:** ~5–15 minutes.

### Step 4: Phase 9 — Distractor comparison

```bash
mkdir -p artifacts/distractor_comparison

# SBERT distractors (default — reuse Phase 1 dataset)
cp artifacts/main/mc_dataset.json artifacts/distractor_comparison/mc_sbert.json

# TF-IDF profile distractors
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/distractor_comparison/tfidf \
    data.distractor_strategy=tfidf_profile
cp artifacts/distractor_comparison/tfidf/mc_dataset.json artifacts/distractor_comparison/mc_tfidf.json

# Category-random distractors
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/distractor_comparison/catrandom \
    data.distractor_strategy=category_random
cp artifacts/distractor_comparison/catrandom/mc_dataset.json artifacts/distractor_comparison/mc_catrandom.json

for STRATEGY in sbert tfidf catrandom; do
    echo "=== Baselines on $STRATEGY distractors ==="
    python scripts/run_baselines.py \
        --config configs/default.yaml \
        --mc-path "artifacts/distractor_comparison/mc_${STRATEGY}.json" \
        likelihood.model=tfidf
    cp artifacts/main/baseline_summary.json "results/baselines_distractor_${STRATEGY}.json"
done
```

**Estimated time:** ~7–10 minutes.

### Step 5: Phase 10 — Variable-K baselines

```bash
mkdir -p artifacts/variable_k
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/variable_k \
    data.variable_K=true data.min_K=2 data.max_K=6 data.K=6 \
    data.distractor_strategy=category_random

python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/variable_k/mc_dataset.json \
    likelihood.model=tfidf
cp artifacts/main/baseline_summary.json results/baselines_variable_k.json
```

**Estimated time:** ~2–3 minutes.

### Step 6: Phase 11 extended — EW-trained PPO + empirical eval

```bash
# Restore baseline_summary.json (clobbered by Phase 13/15)
cp results/baselines_tfidf.json artifacts/main/baseline_summary.json

# Train PPO with Expected Wins reward
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

# Empirical opponent eval
python scripts/evaluate_all.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=tfidf \
    environment.reward_mode=expected_wins \
    environment.opponent_buzz_model.type=empirical
cp artifacts/main/evaluation_report.json results/eval_expected_wins_empirical.json
```

**Estimated time:** ~5 minutes.

### Step 7: Phase 13 supplement — explicit K=4

```bash
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir "artifacts/k4" \
    data.K=4 data.distractor_strategy=category_random
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path "artifacts/k4/mc_dataset.json" \
    likelihood.model=tfidf
cp artifacts/main/baseline_summary.json "results/baselines_k4.json"
```

**Estimated time:** ~2 minutes.

## Phases Skipped (require API keys)

| Phase | Reason |
|-------|--------|
| 12 | DSPy compile — not wired end-to-end, requires LM API key |
| 18 | OpenAI embeddings — requires OPENAI_API_KEY |
| 19 | DSPy MIPROv2 — requires LM API key |

## After All Phases Complete

### Generate summary table

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

### Create FULL_RUN_REPORT.md

Write `results/FULL_RUN_REPORT.md` containing:

1. **Per-phase results table:** phase number, exact command, wall-time, key metrics (accuracy, S_q, ECE, Brier), MPS usage, pass/fail, deviations from this prompt
2. **Runbook issues found:** severity, section, what went wrong, what you did, suggested fix
3. **Final results summary:**
   - Baseline comparison table (Threshold, SoftmaxProfile, SequentialBayes)
   - PPO vs baseline S_q
   - T5 vs MLP policy comparison
   - Ablation summaries: reward modes (14), belief modes (15), policy modes (16), horizon (17)
   - K-sensitivity curve (K=2,3,4,5,6 including controlled K=4)
   - Multi-seed variance (7)
   - Distractor comparison (9)
   - Variable-K (10)
   - Expected Wins eval + EW PPO (11)
   - Reward sweep best config (8)
4. **Artifact inventory:** `ls -lhR results/ artifacts/main/ checkpoints/`

### Verify

```bash
pytest tests/ -q --tb=short
bash -n scripts/run_full_pipeline.sh
```

## Decision-Making Guidelines

- If a command fails: diagnose, fix if obvious, document, continue
- If a phase takes longer than estimated: note actual time, don't kill unless hung (no output 10+ min)
- If MPS causes issues: set `PYTORCH_MPS_FALLBACK=1`, document the error
- If `artifacts/main/` is clobbered: restore from `results/` archives
- Phase ordering: Wave 1 is parallel; everything else is sequential
- Correctness fixes in this codebase: opponent models wired in EW PPO via `make_env_from_config`, variable-K belief shapes use question-local K, embedding cache keyed by model variant (not family), no-buzz calibration skips `buzz_step<0`, padded actions rejected in `step()`, MaskablePPO fully wired through train/load/eval, TF-IDF cache load is a no-op

## Success Criteria

1. All core phases (1–6) complete with valid outputs
2. All extension phases (7–11, 13–17) complete
3. `results/FULL_RUN_REPORT.md` exists with per-phase metrics and comparison tables
4. No mixed likelihood regimes
5. No silent shape mismatches in variable-K phases
6. `pytest tests/ -q --tb=short` passes after the run

## Estimated Total Time

~4–5 hours (wrapper ~3–4 hrs, manual extensions ~1 hr).
