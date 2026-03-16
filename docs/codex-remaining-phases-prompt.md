# Codex: Complete Remaining Pipeline Phases

**Date:** 2026-03-16
**Prerequisite:** The first full-scale run already completed Phases 1–7, 9–11 (eval), 13–17.

## Critical: Use the Main Repo

```bash
cd <repo-root>
git rev-parse --short HEAD   # should be on main or pr-to-origin branch
source .venv/bin/activate
```

Do **NOT** use any Codex worktree at `~/.codex/worktrees/*/`.

## What Was Already Run

These phases completed in the first run and their results exist in `results/`:

| Phase | Status | Output |
|-------|--------|--------|
| 1 | Done | `artifacts/main/mc_dataset.json` (14,961 questions) |
| 2 | Done | `results/baselines_tfidf.json` |
| 3 | Done | `results/ppo_default.json`, `results/ppo_model_default.zip` |
| 4 | Done | `results/eval_default.json` |
| 5 | Done | `checkpoints/ppo_t5/best_model/` (t5-base) |
| 6 | Done | `results/t5_comparison.json` |
| 7 | Done | `results/ppo_seed{1,2,3}.json` |
| 9 | Done | `results/baselines_distractor_{sbert,tfidf,catrandom}.json` |
| 10 | Done | `results/baselines_variable_k.json` |
| 11 eval | Done | `results/eval_ew_logistic.json` |
| 13 | Done | `results/baselines_k{2,3,5,6}.json` |
| 14 | Done | `results/ppo_simple.json`, `results/ppo_human_grounded.json` |
| 15 | Done | `results/baselines_seqbayes.json` |
| 16 | Done | `results/ppo_stop_only.json` |
| 17 | Done | `results/ppo_no_buzz.json` |

**Do NOT re-run any of the above.** Their outputs are authoritative.

## What Still Needs to Run

### Phase 8: Reward sweep

This script is hardwired to use `configs/smoke.yaml` and `artifacts/smoke/`.
It does not accept `--config` or `--mc-path`. It runs on the smoke dataset,
not the full dataset.

```bash
python scripts/sweep_reward_shaping.py --seeds 13,42,123 --timesteps 3000
```

**Expected output:** printed sweep table to stdout, plus
`artifacts/smoke/reward_sweep_results.json` and `.csv`. Capture stdout manually:

```bash
python scripts/sweep_reward_shaping.py --seeds 13,42,123 --timesteps 3000 | tee results/phase_8_sweep.txt
```

**Estimated time:** ~5–15 minutes (smoke-scale, not full-scale).

### Phase 11: EW-trained PPO

The previous run only did the EW *evaluation* (logistic opponent). This phase
trains a new PPO model optimizing for Expected Wins reward.

> **Important:** This overwrites `artifacts/main/ppo_model.zip`. The default
> PPO model is already archived at `results/ppo_model_default.zip`.

```bash
# Restore baseline_summary.json if it was clobbered by Phase 13/15:
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
```

**Checkpoint:**
```bash
python3 -c "
import json
s = json.load(open('results/ppo_expected_wins.json'))
print(f'EW PPO: acc={s[\"buzz_accuracy\"]:.4f}, S_q={s[\"mean_sq\"]:.4f}, reward={s[\"mean_reward_like\"]:.4f}')
"
```

**Estimated time:** ~2–3 minutes (TF-IDF beliefs, 100k timesteps).

### Phase 11: EW empirical evaluation

The first run only evaluated with a logistic opponent. Also run with empirical:

```bash
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
python3 -c "
import json
r = json.load(open('results/eval_expected_wins_empirical.json'))
ew = r.get('expected_wins', {})
fe = r['full_eval']
print(f'EW (empirical): mean_ew={ew.get(\"mean_ew\", \"N/A\")}, S_q={fe[\"mean_sq\"]:.3f}')
"
```

**Estimated time:** ~2 minutes.

### Phase 13: K=4 explicit run (optional)

The first run used Phase 2's default K=4 baseline. For a complete K-sensitivity
comparison with all 5 values run under identical conditions, you can explicitly
run K=4:

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

**Note:** This uses `category_random` distractors (like K=2,3,5,6) instead
of the default `sbert_profile` that Phase 2 used. The K=4 point from Phase 2
(`S_q=0.7413`) used SBERT distractors, so this gives a more controlled comparison.

## Phases to Skip

| Phase | Reason |
|-------|--------|
| 12 (DSPy) | Not wired end-to-end; compiled scorer is not consumed by the factory. Requires API key. |
| 18 (OpenAI embeddings) | Requires `OPENAI_API_KEY`. |
| 19 (DSPy MIPROv2) | Requires API key. |

If you have an OpenAI API key available, you may run Phase 18:

```bash
pip install -e '.[openai]'
export OPENAI_API_KEY=...
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    likelihood.model=openai
cp artifacts/main/baseline_summary.json results/baselines_openai.json
```

## After All Phases Complete

### Update FULL_RUN_REPORT.md

Add the new phase results to the per-phase table and final summary in
`results/FULL_RUN_REPORT.md`. Include:

- Phase 8 sweep results (from captured stdout)
- Phase 11 EW-trained PPO metrics
- Phase 11 empirical EW evaluation metrics
- Phase 13 K=4 explicit (if run)

### Run the summary snippet

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

## Success Criteria

1. Phase 8 sweep completes and output is captured
2. `results/ppo_expected_wins.json` exists with valid metrics
3. `results/eval_expected_wins_empirical.json` exists with `mean_ew`
4. `results/FULL_RUN_REPORT.md` is updated with all new results
5. No mixed likelihood regimes (all belief-feature phases use TF-IDF)

## Estimated Total Time

~15–25 minutes for all remaining phases. This is a short session compared
to the original 3+ hour run.

## Machine & Environment

- Apple M3 Max, 64 GB RAM, MPS available
- Python 3.13.5 in `.venv/`
- All remaining phases use TF-IDF (CPU-only, no MPS needed)
- Phase 8 uses smoke config (50 questions, fast)
