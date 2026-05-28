# Modal StopDFF Runner Operator Guide

`scripts/modal_stopdff_runner.py` is the canonical Modal wrapper for
StopDFF DP and (future) learned-value StopDFF experiments. It
supersedes `scripts/modal_cs321m.py` for StopDFF-specific work;
`modal_cs321m.py` remains canonical for the seven-stage qanta-buzzer
pipeline (`build_mc_dataset` → … → `compute_stopdff`).

## Prerequisites

- Modal account configured locally (`modal token set`).
- A100-80GB (or chosen GPU tier) allocation validated for this
  account. The minimal A100-80GB probe was passed on 2026-05-26; see
  `.planning/STATE.md` for the canonical decision record.
- The repo's working tree is clean (`git status --short` shows only
  untracked `.claude/`), OR `--allow-dirty` is passed explicitly. The
  wrapper refuses to dispatch on a dirty tracked-tree by default so
  every artifact is auditable against a real commit SHA.
- `paper_exports/calibration.json` is present. The DP solver loads it
  in-container; the wrapper pre-flights for it before paying for a
  container spin-up.
- Optional: `modal volume create cs321m-stopdff-artifacts` (the
  wrapper will create-if-missing when it dispatches, but creating
  manually surfaces permission errors earlier).

## Common invocations

CPU smoke (no GPU credits, finishes under an hour):

```bash
modal run --detach scripts/modal_stopdff_runner.py \
    --experiment smoke \
    --gpu none \
    --artifact-subdir "dp_smoke_$(date -u +%Y%m%d_%H%M%S)" \
    --num-bootstrap 100 \
    --smoke
```

Full DP sensitivity sweep on L40S (long-running, use `--detach`):

```bash
modal run --detach scripts/modal_stopdff_runner.py \
    --experiment dp_sweep \
    --gpu L40S \
    --artifact-subdir "dp_sweep_$(date -u +%Y%m%d_%H%M%S)" \
    --max-wall-hours 6 \
    --num-bootstrap 1000 \
    --n-jobs 8
```

Single full DP run on A100-80GB (no sweep, no smoke trim):

```bash
modal run --detach scripts/modal_stopdff_runner.py \
    --experiment single \
    --gpu A100-80GB \
    --artifact-subdir "dp_single_$(date -u +%Y%m%d_%H%M%S)" \
    --num-bootstrap 500
```

## GPU selection

`--gpu` is a modern Modal string spec, not the deprecated
`modal.gpu.X()` object API:

| `--gpu` value                      | Container provisioned     |
|------------------------------------|---------------------------|
| `none`, `cpu`, `""`, `null`        | CPU container             |
| `L40S`                             | NVIDIA L40S               |
| `A100`                             | NVIDIA A100 (40 GB)       |
| `A100-80GB`                        | NVIDIA A100 (80 GB)       |
| `H100`                             | NVIDIA H100               |

CPU-only runs require no GPU credits and are appropriate for `smoke`
and small `single` runs. The full `dp_sweep` is GPU-bound in practice.

## Artifact retrieval

The wrapper writes everything under the `cs321m-stopdff-artifacts`
Modal Volume, namespaced by the `--artifact-subdir` you passed. The
curated repo `paper_exports/` is never overwritten by the wrapper:

```bash
modal volume ls cs321m-stopdff-artifacts
modal volume ls cs321m-stopdff-artifacts <subdir>
modal volume get cs321m-stopdff-artifacts <subdir> ./downloads/
```

Each subdir contains:

- `paper_exports/` -- DP / sweep outputs (including the resumable
  cell cache `stopdff_dp_sweep_cells/`)
- `run_manifest.json` -- git SHA, env vars, full CLI invocation
- `run.log` -- tee'd subprocess output (also streamed to your CLI
  during the run)

## Resume vs overwrite

`--resume` re-uses the existing `paper_exports/stopdff_dp_sweep_cells/`
directory under the subdir, skipping cells already written. This is
the supported path for picking up after a Modal container timeout.

`--overwrite` clears the subdir before dispatching (refusing to run if
the subdir is non-empty without it). The two flags are mutually
exclusive; the wrapper enforces this before dispatch.

## Distinction from `modal_cs321m.py`

| Concern             | `modal_cs321m.py`                       | `modal_stopdff_runner.py`                     |
|---------------------|-----------------------------------------|-----------------------------------------------|
| Use case            | Full 7-stage qanta-buzzer pipeline      | StopDFF DP / learned-value experiments        |
| GPU selection       | Hardcoded A100-80GB (deprecated SDK)    | Operator-selectable via `--gpu` (modern spec) |
| Volume support      | None (base64 round-trip)                | Yes (`cs321m-stopdff-artifacts`)              |
| Resume              | No                                      | Yes (cell cache survives container restarts)  |
| Secrets             | Always attached                         | Opt-in via `--with-openai-key`                |
| Dirty-tree refusal  | No                                      | Yes (override with `--allow-dirty`)           |
| Status              | Legacy; preserve for canonical-pipeline | Canonical for new StopDFF work                |

The split is intentional. Unification of `modal_cs321m.py` to the
modern Modal SDK (string GPU spec, Volume support, operator-selectable
GPU) is tracked separately and out of scope for this runbook.

## Troubleshooting

- "Repo working tree has uncommitted TRACKED changes" -- commit/stash
  first, or re-run with `--allow-dirty` (the manifest records the
  porcelain so paper provenance is preserved).
- "could not capture host git state" -- happens when the wrapper is
  invoked from a non-git checkout. Re-run with `--allow-dirty`; the
  manifest records `git_present_local: false`.
- "--max-wall-hours exceeds Modal's 24h container ceiling" -- split
  the sweep into multiple `--resume` runs, each with a budget under
  ~20.8h.
- Container OOMs on dp_sweep -- bump `--gpu` from L40S to A100 /
  A100-80GB and re-run with `--resume`; the cell cache survives the
  restart.
