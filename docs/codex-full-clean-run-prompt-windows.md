# Codex: Full Clean Pipeline Run — Windows/WSL2 + CUDA Variant

**Generated:** 2026-03-17 | **Repo branch:** `review-fixes` (verify with `git rev-parse --short HEAD`)
**Base prompt:** `docs/codex-full-clean-run-prompt.md` (macOS/MPS variant)

## Critical: Use the Main Repo

```bash
cd <repo-root>
git rev-parse --short HEAD
```

Do **NOT** use any Codex worktree at `~/.codex/worktrees/*/`.

## Objective

Run the complete qanta-buzzer pipeline from scratch — all phases except those
requiring OpenAI/DSPy API keys. This is a clean run on the codebase that
includes all runtime correctness fixes from PR #13: opponent model wiring,
variable-K belief shapes, model-variant-specific embedding cache, no-buzz
calibration, padded action guards, and full MaskablePPO train/load/eval wiring.

Use one of two execution modes:

- **Mode A — Safe current repo:** runnable today. Uses the built-in wrapper
  parallelism plus limited extra concurrency where artifact paths do not clash.
  This is the default mode for this prompt.
- **Mode B — Max throughput with lane-local output dirs:** now technically
  available because the stage scripts support lane-local `--output-dir`
  usage. It should still be treated as the advanced path until it has been
  validated on a real full run on this machine.

Produce `results/FULL_RUN_REPORT.md` as the canonical handoff artifact, and
state which mode was used.

## Machine

- Eluktronics HYDROC-16 G2 Ultra Pro Edition
- CPU: Intel Core Ultra 9 275HX
- GPU: NVIDIA GeForce RTX 5090, 24 GB GDDR7 VRAM
- RAM: 128 GB DDR5-5600 (Crucial)
- Storage: 4 TB NVMe (2x WD Black SN850X)
- OS: Windows 11 Pro 64-bit, running **Ubuntu via WSL2**
- Python 3.12+ in `.venv/`
- CUDA available via WSL2 GPU passthrough

### WSL2 + CUDA prerequisites

The NVIDIA GPU driver is installed on the **Windows host** (not inside WSL).
WSL2 automatically exposes `/usr/lib/wsl/lib/libcuda.so` to the guest.
Inside WSL Ubuntu, install the CUDA toolkit (without the driver):

```bash
# One-time WSL2 setup (skip if already done)
sudo apt update && sudo apt install -y build-essential
# Install CUDA toolkit for WSL — see https://developer.nvidia.com/cuda-downloads
# Select: Linux > x86_64 > WSL-Ubuntu > deb (network)
nvidia-smi          # should show RTX 5090, driver version, CUDA version
python3 -c "import torch; avail = torch.cuda.is_available(); print(f'CUDA: {avail}' + (f', Device: {torch.cuda.get_device_name(0)}' if avail else ' — check driver/toolkit'))"
```

### VRAM budget

| Workload | Est. VRAM | Fits in 24 GB? |
|----------|-----------|----------------|
| TF-IDF baselines + PPO (MLP policy) | < 1 GB | Yes |
| t5-base supervised + PPO-T5 | ~4–6 GB | Yes |
| t5-large supervised + PPO-T5 | ~12–16 GB | Yes (recommended) |
| t5-3b supervised + PPO-T5 | ~24+ GB | Tight — not recommended |

With 24 GB VRAM, **t5-large is the recommended default** for this machine.
It provides substantially better semantic representations than t5-base while
fitting comfortably in VRAM. The macOS/MPS variant uses t5-base due to
unified memory pressure (~41 GB); that constraint does not apply here.

## Setup

```bash
cd <repo-root>
git checkout review-fixes
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python3 -c "import torch; avail = torch.cuda.is_available(); print(f'CUDA: {avail}' + (f', Device: {torch.cuda.get_device_name(0)}' if avail else ' — check driver/toolkit'))"
pytest tests/ -q --tb=short    # expect: 365 passed, 4 skipped
```

## Phase 0: Clean state

```bash
rm -rf artifacts/main/ artifacts/k* artifacts/distractor_* artifacts/variable_k/
rm -rf cache/embeddings/
rm -rf checkpoints/supervised/ checkpoints/ppo/ checkpoints/ppo_t5/
rm -rf results/
mkdir -p artifacts/main results
```

## Concurrency Model

### Mode A — Safe current repo (use this now)

This repo already has one safe concurrency point: the wrapper overlaps the
default baseline, default PPO, and T5 policy tracks. After that, many manual
phases still reuse `artifacts/main/`, so they remain intentionally serial.

This means Mode A is **artifact-safe concurrency**, not full machine
saturation. On this machine, that is still the right default for correctness.

One extra lane is safe today:

- `scripts/sweep_reward_shaping.py` writes to `artifacts/smoke/`, not
  `artifacts/main/`, so it can be run in parallel with the main wrapper if CPU
  headroom remains.

### Mode B — Max throughput with lane-local output dirs

The stage scripts now support lane-local `--output-dir`, so this machine can be
used with one GPU lane and multiple CPU lanes. Treat this as an advanced path
until the first real full run has been timed and validated end to end:

- **GPU lane:** one T5 job at a time. Do not run multiple `t5-large` jobs on
  the RTX 5090 simultaneously.
- **CPU lane A:** default baselines/evals using lane-local artifact dirs.
- **CPU lane B:** PPO family jobs (default PPO, multi-seed PPO, EW PPO,
  stop-only, no-buzz), each with its own output dir.
- **CPU lane C:** distractor and K-sensitivity dataset builds/baselines in
  separate output dirs.
- **Optional CPU lane D:** reward sweep on `artifacts/smoke` or another
  isolated smoke output tree.

Mode B should archive results directly from lane-local output directories rather
than copying back through `artifacts/main/`.

### Thread caps for multi-process CPU work

When you run multiple CPU-heavy Python jobs concurrently, cap BLAS/OpenMP
threads per process to avoid oversubscription:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

Use these caps for optional extra lanes in Mode A and as the default starting
point for Mode B. Re-measure if the machine shows idle cores or heavy context
switching.

## Execution Plan (Mode A — runnable today)

### Step 1: Core pipeline via wrapper (Phases 1–6, 11, 13–17)

```bash
bash scripts/run_full_pipeline.sh --t5-model t5-large 2>&1 | tee results/wrapper_stdout.log
```

The wrapper runs a 4-wave DAG:
- **Wave 1 (parallel):** Phase 2 (TF-IDF baselines), Phase 3 (PPO), Phase 5 (T5 policy)
- **Wave 2 (sequential):** Phase 4 (evaluate), Phase 6 (compare), Phase 11 (EW eval), Phase 15 (belief mode)
- **Wave 3 (sequential):** Phase 14 (reward modes), Phase 16 (stop-only), Phase 17 (no-buzz)
- **Wave 4 (sequential):** Phase 13 (K-sensitivity: K=2,3,5,6)

All belief-feature phases use `likelihood.model=tfidf`. Phase 5 uses t5-large on CUDA.
Logs for Waves 1/2/4 are in `results/phase_*.log` (unbuffered via `PYTHONUNBUFFERED=1`).
Wave 3 prints to stdout.

**Monitoring:**
```bash
tail -f results/phase_5.log       # T5 training (may be sparse during supervised warm-start)
ps aux | grep train_t5_policy     # verify process is running
nvidia-smi                        # monitor GPU utilization and VRAM
```

**Estimated time:** ~1.5–2.5 hours (RTX 5090 CUDA is significantly faster than M3 Max MPS; t5-large is ~2.5x the parameters of t5-base but the GPU throughput more than compensates).

### Step 1b (optional): Run reward sweep in a separate shell while Step 1 runs

Only do this if the machine still has comfortable CPU headroom during the
wrapper run. This is safe because the sweep uses `artifacts/smoke/`, not
`artifacts/main/`.

Before launching it, apply the thread caps from the previous section in the new
shell.

```bash
python scripts/sweep_reward_shaping.py --seeds 13,42,123 --timesteps 3000 \
    | tee results/phase_8_sweep.txt
```

If you launch Step 1b here, skip Step 3 later.

### Step 2: Phase 7 — Multi-seed PPO

Current repo note: keep this loop serial. These runs all reuse
`artifacts/main/`, so backgrounding them against each other is not safe until
output-dir isolation is implemented.

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

**Estimated time:** ~3–5 minutes.

### Step 3: Phase 8 — Reward sweep (smoke-scale)

```bash
python scripts/sweep_reward_shaping.py --seeds 13,42,123 --timesteps 3000 \
    | tee results/phase_8_sweep.txt
```

Note: this script uses `configs/smoke.yaml` and `artifacts/smoke/` — it does
not run on the full dataset. It also writes
`artifacts/smoke/reward_sweep_results.json`. If Step 1b already ran this in a
second shell, do not rerun it here.

**Estimated time:** ~3–8 minutes.

### Step 4: Phase 9 — Distractor comparison

Current repo note: the two alternate dataset builds could be parallelized on a
machine like this, but the subsequent baseline runs still reuse
`artifacts/main/`, so treat this whole step as serial until output isolation is
available.

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

**Estimated time:** ~5–7 minutes.

### Step 5: Phase 10 — Variable-K baselines

Current repo note: this is short enough that there is little reason to overlap
it with another `artifacts/main/` writer in the current codebase.

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

**Estimated time:** ~1–2 minutes.

### Step 6: Phase 11 extended — EW-trained PPO + empirical eval

Current repo note: keep these two commands serial because they intentionally
reuse `artifacts/main/` and restore `baseline_summary.json`.

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

**Estimated time:** ~3 minutes.

### Step 7: Phase 13 supplement — explicit K=4

Current repo note: this is another natural candidate for CPU fan-out after
output-dir isolation, but not before.

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

**Estimated time:** ~1 minute.

## Mode B Template — Max Throughput with Lane-Local Output Dirs

This mode is now mechanically available because the stage scripts support
lane-local `--output-dir` values end to end. Switch from the serial manual tail
above to a lane-based layout like this only when you are intentionally doing an
advanced throughput-focused run and are prepared to record the first full-run
measurements carefully:

1. Build the shared MC dataset once.
2. Start exactly one T5 GPU lane.
3. Fan out CPU-only baseline/PPO/dataset jobs into separate output dirs.
4. Archive results from those lane-local dirs directly.

Illustrative lane plan:

```bash
# GPU lane (one at a time)
python scripts/train_t5_policy.py --config configs/t5_policy.yaml \
    model.model_name=t5-large

# CPU lane A: default eval family
python scripts/run_baselines.py --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --output-dir artifacts/default_baselines \
    likelihood.model=tfidf

# CPU lane B: PPO family (example seed fan-out)
python scripts/train_ppo.py --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --output-dir artifacts/seed1 --seed 1 --deterministic-eval \
    likelihood.model=tfidf

python scripts/train_ppo.py --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --output-dir artifacts/seed2 --seed 2 --deterministic-eval \
    likelihood.model=tfidf

python scripts/train_ppo.py --config configs/default.yaml \
    --mc-path artifacts/main/mc_dataset.json \
    --output-dir artifacts/seed3 --seed 3 --deterministic-eval \
    likelihood.model=tfidf
```

Mode B priorities:

- Keep the GPU busy with one T5 job, not several.
- Use CPU fan-out for PPO seeds, distractor comparisons, K-sensitivity, and
  reward sweep.
- Never route independent jobs back through `artifacts/main/`.
- Re-measure thread caps and wall time on the first real run; do not assume a
  final SLA until the isolated-output version has been timed.

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

1. **Per-phase results table:** phase number, exact command, wall-time, key metrics (accuracy, S_q, ECE, Brier), GPU utilization, pass/fail, deviations from this prompt
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
- If CUDA causes issues: set `CUDA_LAUNCH_BLOCKING=1` for synchronous error reporting, document the error
- If CUDA OOM on t5-large: first fall back to `--t5-model t5-base`. For
  standalone T5 runs, reduce `supervised.batch_size`,
  `supervised.grad_accum_steps`, or `ppo.batch_size` instead of using a
  non-existent `training.batch_size` key
- If `artifacts/main/` is clobbered: restore from `results/` archives
- In Mode A, Wave 1 is parallel and the rest is mostly serial because of shared
  `artifacts/main/`
- In Mode B, parallelize only across lane-local output dirs and keep exactly one
  CUDA-heavy T5 job active at a time
- Correctness fixes in this codebase: opponent models wired in EW PPO via `make_env_from_config`, variable-K belief shapes use question-local K, embedding cache keyed by model variant (not family), no-buzz calibration skips `buzz_step<0`, padded actions rejected in `step()`, MaskablePPO fully wired through train/load/eval, TF-IDF cache load is a no-op

## WSL2-Specific Notes

- All commands in this prompt assume a **bash shell inside WSL2 Ubuntu**. Do not run them in PowerShell or cmd.exe.
- The repo should be cloned inside the WSL2 filesystem (`~/` or `/home/user/`), not on a Windows mount (`/mnt/c/`). Windows mount paths have poor I/O performance and may cause permission issues.
- `nvidia-smi` works inside WSL2 when the Windows host has the NVIDIA GPU driver installed (version 535+ recommended for RTX 50-series).
- If `torch.cuda.is_available()` returns `False`, verify: (1) NVIDIA driver is installed on Windows host, (2) WSL2 is using the Linux kernel 5.15+, (3) the CUDA toolkit is installed inside WSL.
- File watchers (e.g. `tail -f`) work normally inside WSL2.

## Success Criteria

1. All core phases (1–6) complete with valid outputs
2. All extension phases (7–11, 13–17) complete
3. `results/FULL_RUN_REPORT.md` exists with per-phase metrics, comparison tables, and the declared execution mode (`Mode A` or `Mode B`)
4. No mixed likelihood regimes
5. No silent shape mismatches in variable-K phases
6. `pytest tests/ -q --tb=short` passes after the run
7. If Mode B is used, all concurrent jobs write to lane-local
   output dirs rather than `artifacts/main/`
8. If extra CPU lanes are used, the report notes the thread caps and any
   observed contention or idle hardware

## Estimated Total Time

- **Mode A (current repo):** ~2.5–3.5 hours (wrapper ~1.5–2.5 hrs with RTX 5090
  CUDA + t5-large, manual extensions ~30 min, reward sweep optionally overlapped)
- **Mode B (advanced high-throughput path):** expected to reduce the manual
  extension tail materially by parallelizing PPO seeds, distractor runs,
  K-sensitivity, and reward sweep, but it must be re-measured on the first real
  lane-local-output run before treating it as a stable estimate

The RTX 5090 provides roughly 2–3x speedup over M3 Max MPS for T5 training,
which more than offsets the increase from t5-base to t5-large. CPU-bound phases
benefit from the larger machine most when they can be split into lane-local
jobs without fighting over `artifacts/main/`.
