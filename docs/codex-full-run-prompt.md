# Codex Full-Scale Pipeline Run — Prompt

**Commit:** `efe6697` | **Generated:** 2026-03-15

## Critical: Use the Main Repo, Not a Worktree

You MUST operate on your primary **qanta-buzzer** clone, referred to here as `$REPO_ROOT` (the directory that contains this `docs/` folder).

Do **NOT** use any Codex worktree (e.g., `$HOME/.codex/worktrees/*/qanta-buzzer`). Those are stale snapshots. If your current working directory is a worktree, `cd` to `$REPO_ROOT` before doing anything.

Verify you're in the right place:
```bash
cd "$REPO_ROOT"
git rev-parse --short HEAD   # must be efe6697 or later
```

## Objective

Execute the complete qanta-buzzer pipeline end-to-end at full scale. Two purposes:

1. **Produce real experimental results** for the CS234 final project
2. **Create `results/FULL_RUN_REPORT.md`** as a handoff artifact with per-phase metrics, runbook issues, and a final comparison table

## Machine & Environment

- **Hardware:** Apple M3 Max, 16 cores, 64 GB RAM, Apple MPS GPU
- **MPS:** auto-detected by `_best_torch_device()` in `models/likelihoods.py` and `models/t5_policy.py`
- **Python:** 3.13.5 in `.venv/`
- **Memory warning:** t5-base at full scale reaches ~41 GB on MPS. t5-large will OOM on this machine.

## Setup

```bash
cd "$REPO_ROOT"   # your main qanta-buzzer repo
source .venv/bin/activate
pip install -e .
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
pytest tests/ -q --tb=short    # expect: 357 passed, 3 skipped
```

## Execution Plan

### Step 1: Core pipeline via wrapper (Phases 1–6, 11, 13–17)

```bash
bash scripts/run_full_pipeline.sh --t5-model t5-base
```

This runs the 4-wave DAG with `likelihood.model=tfidf` forced for all belief-feature phases. Logs for Waves 1/2/4 are in `results/phase_*.log`; Wave 3 prints to stdout. Logs are now unbuffered (`PYTHONUNBUFFERED=1`), but Phase 5 may still appear slow during supervised warm-start (normal).

Monitor in another terminal:
```bash
tail -f results/phase_5.log       # T5 training
ps aux | grep train_t5_policy     # verify process is running
```

**Estimated time:** ~3–4 hours for the full wrapper.

### Step 2: Manual extension phases

After the wrapper completes, run these manually from the runbook's individual phase sections. All commands already include `likelihood.model=tfidf`.

| Phase | Description | Est. time |
|-------|-------------|-----------|
| 7 | Multi-seed PPO (seeds 1,2,3) | 1.5–3 hrs |
| 9 | Distractor comparison (sbert/tfidf/catrandom) | 15–30 min |
| 10 | Variable-K baselines | 15–30 min |
| 13 | K-sensitivity (K=2,3,4,5,6) | 30–60 min |

Skip Phases 8, 11 (EW PPO), 12, 18, 19 unless specifically needed.

### Step 3: Generate the summary table

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

## What to Document in `results/FULL_RUN_REPORT.md`

1. **Per-phase results table:** phase number, exact command, wall-time, key metrics, MPS usage, pass/fail, deviations
2. **Runbook issues found:** severity, location, what was wrong, what you did, suggested fix
3. **Final results summary:** baseline comparison table, PPO vs baseline S_q, T5 vs MLP policy comparison, ablation summaries (reward/belief/policy/horizon modes), K-sensitivity data
4. **Artifact inventory:** `ls -lhR results/ artifacts/main/ checkpoints/` output

## Decision-Making Guidelines

- If a command fails: diagnose, fix if obvious, document, continue
- If a phase takes longer than estimated: note actual time, don't kill unless hung (no output 10+ min)
- If MPS causes issues: set `PYTORCH_MPS_FALLBACK=1` or pass `device=cpu`, document the error
- If `artifacts/main/` is clobbered: restore from `results/` archives (documented in runbook re-run notes)
- Phase ordering: 4 after 2, 11 before 15, 9/13/15 sequential after 11

## Success Criteria

1. All core phases (1–6) complete with valid outputs
2. At least 4 extension phases complete
3. `results/FULL_RUN_REPORT.md` exists with metrics and comparison tables
4. No mixed likelihood regimes in comparisons
