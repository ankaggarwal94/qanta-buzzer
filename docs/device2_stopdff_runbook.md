# Device 2 StopDFF DP Sweep Runbook

Run these commands from a bash shell on Device 2. The harness uses the repo-local
`.venv` and writes durable logs under `artifacts/device2_stopdff/`.

## Setup

```bash
REPO="${REPO:-/mnt/c/Users/ankag/Dropbox/Stanford/CS234/final_project/qanta-buzzer}"
[ -d "$REPO" ] || REPO="$HOME/qanta-buzzer"
cd "$REPO"

git config --local core.autocrlf false
git config --local core.eol lf
git ls-files --eol scripts/device2_stopdff_run.sh threshold_manifest.json threshold_manifest.json.sha256

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .

python scripts/device2_cuda_preflight.py --help
bash -n scripts/device2_stopdff_run.sh
```

## Preflight-only validation

Use a fresh preflight directory. Do not pre-create the directory; the preflight
script writes the JSON report after validating that the requested output path is
safe for a new run.

```bash
REPO="${REPO:-/mnt/c/Users/ankag/Dropbox/Stanford/CS234/final_project/qanta-buzzer}"
[ -d "$REPO" ] || REPO="$HOME/qanta-buzzer"
cd "$REPO"
source .venv/bin/activate

PREFLIGHT_OUT="artifacts/device2_stopdff/preflight_$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_VISIBLE_DEVICES=0 python scripts/device2_cuda_preflight.py \
  --out-dir "$PREFLIGHT_OUT" \
  --data-dir data/processed \
  --calibration paper_exports/calibration.json \
  --fit-split val \
  --eval-split test \
  --output-json "$PREFLIGHT_OUT/preflight.json" \
  --device-index 0 \
  --min-free-gib 0
```

## Start tmux

```bash
tmux new -s cs321m-stopdff
```

Inside tmux:

```bash
REPO="${REPO:-/mnt/c/Users/ankag/Dropbox/Stanford/CS234/final_project/qanta-buzzer}"
[ -d "$REPO" ] || REPO="$HOME/qanta-buzzer"
cd "$REPO"
source .venv/bin/activate
```

Detach with `Ctrl-b d`. Reattach with:

```bash
tmux attach -t cs321m-stopdff
```

## Smoke run under 10 minutes

Smoke mode is bounded by the harness's `SMOKE_MAX_CELLS` value, which defaults
to `2`; pass `SMOKE_MAX_CELLS=2` explicitly so the cap is visible in the run log.

`tee` writes inside the run directory, so the directory is created first and
`--resume` is passed non-destructively. On an empty directory, this behaves like
a fresh run; on a rerun, completed sweep cells are skipped from the cache.

```bash
REPO="${REPO:-/mnt/c/Users/ankag/Dropbox/Stanford/CS234/final_project/qanta-buzzer}"
[ -d "$REPO" ] || REPO="$HOME/qanta-buzzer"
cd "$REPO"
source .venv/bin/activate

SMOKE_OUT="artifacts/device2_stopdff/smoke"
mkdir -p "$SMOKE_OUT"

SMOKE_MAX_CELLS=2 bash scripts/device2_stopdff_run.sh \
  --experiment dp_sweep \
  --artifact-dir paper_exports \
  --out-dir "$SMOKE_OUT" \
  --max-wall-hours 0.15 \
  --num-bootstrap 100 \
  --n-jobs 2 \
  --calibrators platt,temperature,isotonic \
  --reward-schedules acf_flat,power_mark,wait_cost_small,strict_wrong \
  --continuations empirical_bucket,pooled_empirical \
  --fit-split val \
  --eval-split test \
  --smoke \
  --resume \
  2>&1 | tee "$SMOKE_OUT/live.log"
```

## Full 6-hour run

This launch should require no interactive input after it starts. It verifies CUDA
readiness, records `command_manifest.json`, and runs the requested sweep axes.

```bash
REPO="${REPO:-/mnt/c/Users/ankag/Dropbox/Stanford/CS234/final_project/qanta-buzzer}"
[ -d "$REPO" ] || REPO="$HOME/qanta-buzzer"
cd "$REPO"
source .venv/bin/activate

mkdir -p artifacts/device2_stopdff/full_6h

bash scripts/device2_stopdff_run.sh \
  --experiment dp_sweep \
  --artifact-dir paper_exports \
  --out-dir artifacts/device2_stopdff/full_6h \
  --max-wall-hours 6 \
  --num-bootstrap 1000 \
  --n-jobs 12 \
  --calibrators platt,temperature,isotonic \
  --reward-schedules acf_flat,power_mark,wait_cost_small,strict_wrong \
  --continuations empirical_bucket,pooled_empirical \
  --fit-split val \
  --eval-split test \
  --resume \
  2>&1 | tee artifacts/device2_stopdff/full_6h/live.log
```

## Monitoring

```bash
tail -f artifacts/device2_stopdff/full_6h/live.log
```

```bash
tail -f artifacts/device2_stopdff/full_6h/stdout.log
tail -f artifacts/device2_stopdff/full_6h/stderr.log
```

```bash
watch -n 30 nvidia-smi
```

Paper export inventory checks:

```bash
ls -lh \
  artifacts/device2_stopdff/full_6h/paper_exports/stopdff_dp_sweep.json \
  artifacts/device2_stopdff/full_6h/paper_exports/stopdff_dp_sweep_table.tex

find artifacts/device2_stopdff/full_6h/paper_exports/figures \
  -maxdepth 1 -type f -print | sort

find artifacts/device2_stopdff/full_6h/paper_exports/stopdff_dp_sweep_cells \
  -maxdepth 1 -type f -name '*.json' | wc -l
```

Optional JSON sanity check:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/device2_stopdff/full_6h/paper_exports/stopdff_dp_sweep.json")
payload = json.loads(path.read_text())
print("metric_type:", payload.get("metadata", {}).get("metric_type"))
print("verdict:", payload.get("paper_safe_interpretation", {}).get("verdict"))
print("cells:", len(payload.get("cells", [])))
PY
```

## Stopping

In the tmux pane running the job, press `Ctrl-C`. Do not delete the run directory
or any cell cache files. Interrupted runs are expected to leave useful partial
artifacts under:

```bash
artifacts/device2_stopdff/full_6h/
```

## Resuming

Resume with the same `--out-dir` plus `--resume`. The sweep cache skips completed
cells and rebuilds aggregate outputs from cached and newly completed cells.

```bash
REPO="${REPO:-/mnt/c/Users/ankag/Dropbox/Stanford/CS234/final_project/qanta-buzzer}"
[ -d "$REPO" ] || REPO="$HOME/qanta-buzzer"
cd "$REPO"
source .venv/bin/activate

bash scripts/device2_stopdff_run.sh \
  --experiment dp_sweep \
  --artifact-dir paper_exports \
  --out-dir artifacts/device2_stopdff/full_6h \
  --max-wall-hours 6 \
  --num-bootstrap 1000 \
  --n-jobs 12 \
  --calibrators platt,temperature,isotonic \
  --reward-schedules acf_flat,power_mark,wait_cost_small,strict_wrong \
  --continuations empirical_bucket,pooled_empirical \
  --fit-split val \
  --eval-split test \
  --resume \
  2>&1 | tee -a artifacts/device2_stopdff/full_6h/live.log
```

## Import into cs321m-paper

Set `PAPER_REPO` to the local `cs321m-paper` checkout if it is not a sibling of
this repo. These commands copy only the expected export files and do not delete
anything in the paper repo.

```bash
REPO="${REPO:-/mnt/c/Users/ankag/Dropbox/Stanford/CS234/final_project/qanta-buzzer}"
[ -d "$REPO" ] || REPO="$HOME/qanta-buzzer"
cd "$REPO"

PAPER_REPO="${PAPER_REPO:-../cs321m-paper}"
SRC="artifacts/device2_stopdff/full_6h/paper_exports"

test -d "$PAPER_REPO"
test -f "$SRC/stopdff_dp_sweep.json"
test -f "$SRC/stopdff_dp_sweep_table.tex"
test -d "$SRC/figures"

mkdir -p "$PAPER_REPO/paper_exports/figures"
cp -p "$SRC/stopdff_dp_sweep.json" "$PAPER_REPO/paper_exports/"
cp -p "$SRC/stopdff_dp_sweep_table.tex" "$PAPER_REPO/paper_exports/"
rsync -av "$SRC/figures/" "$PAPER_REPO/paper_exports/figures/"

git -C "$PAPER_REPO" status --short -- paper_exports
```

## Acceptance and safety notes

- Expected final artifacts are
  `paper_exports/stopdff_dp_sweep.json`, `paper_exports/figures/`, and
  `paper_exports/stopdff_dp_sweep_table.tex` under
  `artifacts/device2_stopdff/full_6h/`.
- The full run should need no interactive input after launch.
- Resume skips completed cells via the sweep cache. Keep prior runs and cache
  files; do not delete them.
- The GPU is verified during preflight, but bucketed DP itself is not assumed to
  need GPU. CUDA helps only if downstream learned continuation or PyTorch
  calibration is added.
- Do not use test data for calibration or continuation fitting. Keep
  `--fit-split val` and `--eval-split test`.
- Current PR #15 caveat: before using sweep artifacts as paper evidence,
  unresolved sweep review issues should be fixed: MC coverage/retention gating
  in the sweep, dataset hashes in the cache fingerprint, and non-resume
  aggregation ignoring stale cells. This is a caution for evidentiary use, not a
  blocker for harness mechanics.
