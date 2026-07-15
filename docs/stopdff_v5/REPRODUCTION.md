# StopDFF v5 bucketed-DP paired audit — implementation & reproduction

This documents the **v5 "Does the Proxy Preserve the Decision?" StopDFF audit**
(`profile_name: stopdff_bucketed_dp_paired_v2`) implemented under `scripts/stopdff_v5/`,
and how to reproduce the published 96-cell evidentiary release **on any machine (CPU)** or
**on Modal**. It implements the normative v5 contracts (scientific / identity / acceptance).

## What this audits

Given quizbowl tossups revealed prefix-by-prefix, we compare two answer-elicitation formats
per item — a fixed multiple-choice condition (**MC**) and a gold-answer question condition
(**QA**) — under a finite-horizon **three-action Bellman stopping policy** (ANSWER / WAIT /
ABSTAIN). The paired signed index metric is `D_i = tau_MC - tau_QA` (difference in the
prefix index at which the policy commits). The preregistered gate is the **median absolute
prefix-index shift** with material threshold **1**, controlled family-wide across a
**96-cell sensitivity grid** (4 rewards × 2 continuations × 3 calibrators × 2 prefix-bucketings
× 2 category-poolings) via a common-bootstrap **maximum statistic**.

## Frozen identities of the published run

| Artifact | ID / value |
|---|---|
| Execution Git SHA (this branch) | `e3ca01830ed67ae81e3f9683db847249e3fbec14` |
| source manifest ID | `c80393ba2694a17e440c75a782af061efa138c35f9774e8ab12eb81c4fe93077` |
| raw-input bundle ID | `63ca235f7e359afe1e541a2637482eb7ed3c70d765c36e311b6b9ae0eb880224` |
| model snapshot | `sentence-transformers/all-MiniLM-L6-v2` @ rev `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, `trust_remote_code=false` |
| model snapshot ID | `33b48dc6daf60b6e0a2190964bf3faafc249732269fed1f717f197940cd6f893` |
| adapter bundle ID | `13559ea71e5573721da627434fb4cbf2e6ba499006a1b3f48300a59b89a1c355` |
| FVI-study ID (selected: tol `1e-6`, max_iter `100`) | `bfc74931b72b264b37e8a09fb90e4e446433060a118e8a92f9ec2f9275c872fe` |
| bootstrap-plan ID (final, 1000 reps, PCG64 seed 1) | `076ac23359ad4c7dd0c9ba108c5a41a4bb43ebd9fff3629837991ba459a426b0` |
| run-spec ID (final) | `94274b55b319bbb8c236b242ec436fad5756e8a37da5cdd608fc7690b20fa887` |

**Reproducing the exact `source_manifest_id`** requires the exact executed tree:
`git checkout e3ca01830ed67ae81e3f9683db847249e3fbec14` (this branch's history) before
building the source snapshot. Later commits on this branch (these docs + the artifacts)
are additive and change the tree hash.

## Published result (Modal backend, `backend=modal`)

`release_status = VALID` · requested/completed/skipped/failed = **96 / 96 / 0 / 0** ·
cell verdicts **82 PASS / 14 WARN** · family maximum statistic **M = 1.0**, 95% CI **[1.0, 1.0]**,
**family verdict = WARN** · gate overrides: none. The standalone Modal-side checker
independently recomputed every statistic (PASS, 0 errors); the negative mutation suite
rejected all 26 mutations.

Interpretation: the release is valid but the family verdict is a borderline **WARN** — the MC
reformulation shifts the stopping decision by a median absolute of ~1 prefix index at the
material threshold. This is an honest, non-clean-PASS audit outcome.

Artifacts live under [`stopdff_v5_release/modal_final_94274b55b319bbb8/`](../../stopdff_v5_release/modal_final_94274b55b319bbb8/)
(`run_package/` = the validated evidentiary package; `retrieval_index.json` = Modal-Volume
retrieval paths for the large artifacts that are not committed to Git; `PUBLICATION_MANIFEST.json`).

## Code layout

```
scripts/stopdff_v5/
  identity.py        canonical JSON identity bytes + content-addressed IDs
  profile.py         96-cell grid, smoke cells, FVI representative-24, canonical constants
  rewards.py         4 reward schedules
  policy.py          3-action Bellman DP (ANSWER/WAIT/ABSTAIN, tie policy, stop encoding)
  calibrators.py     platt-logistic / similarity-temperature / isotonic (val-MC fit, shared apply)
  continuation.py    bins, fallback ladders, coverage tags
  fvi.py             damped convergence-controlled FVI (float64, math.fsum, cycle detection)
  fvi_study.py       preregistered candidate study + deterministic selector
  bootstrap.py       common paired PCG64 bootstrap + cell/family statistics
  verdicts.py        cell / family verdicts + release validity
  cellcompute.py     shared per-cell compute (calibrate -> continuation -> FVI -> solve)
  sweep.py           sweep orchestrator (per-cell atomic writes, verdicts, manifests)
  manifests.py       source/raw/model/adapter/FVI/bootstrap/run-spec/cell identities
  producers.py       source snapshot + raw-input stager (+ semantic checks) + env contract
  adapter_build.py   model-snapshot freeze + deterministic adapter builder (real-data scoring)
  rowio.py           deterministic (byte-stable) gzip row I/O
  writers.py         Markdown / LaTeX / PNG report writers + package builder
  checker.py         standalone validator: independent recomputation of all statistics
  selftest.py        negative mutation suite + valid-package builder
scripts/validate_stopdff_bucketed_sweep.py   standalone checker CLI (acceptance contract)
scripts/run_stopdff_v5_local.py              CPU/local end-to-end reproduction driver
scripts/modal_stopdff_v5_runner.py           Modal remote functions (source-only image)
schemas/stopdff_*.schema.json                JSON schemas for profile/run-spec/calibrator/continuation/gate
tests/test_stopdff_v5_{core,pipeline,checker}.py   37 unit/integration tests + mutation suite
```

## Environment

Python 3.11. Install the runtime dependencies:

```bash
python3.11 -m venv .venv-stopdff-v5 && source .venv-stopdff-v5/bin/activate
pip install -U pip
pip install "numpy>=1.26,<3" "scipy>=1.11" "scikit-learn>=1.3" "pandas>=2.1" \
            "matplotlib>=3.7" "sentence-transformers>=2.7" "huggingface_hub>=0.23"
```

The published run used: python 3.11.12, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0,
pandas 3.0.3, sentence-transformers 5.6.0, transformers 5.13.1, huggingface_hub 1.23.0
(recorded in each run's `environment.json`).

## Local (CPU) reproduction — no Modal required

`scripts/run_stopdff_v5_local.py` runs the whole pipeline in-process on CPU:
stage raw inputs → freeze model snapshot → build adapter → FVI study → bootstrap plans →
2-cell smoke → 96-cell final → standalone validation (`backend=local`).

You need the nine raw inputs (the full-scale `mc_dataset.json` / `val_dataset.json` /
`test_dataset.json` / `build_metadata.json` / `split_metadata.json` live under
`data/processed/`; `calibration.json` and `stopdff.json` under `paper_exports/`;
`threshold_manifest.json{,.sha256}` at the repo root). These large inputs are reproducible
from the belief-feature pipeline (`scripts/build_mc_dataset.py`, `scripts/fresh_split.py`,
`scripts/compute_prefix_calibration.py`, `scripts/compute_stopdff.py`) and are synced via
Dropbox rather than committed (see `.gitignore` and `DATA.md`).

```bash
# smoke (fast: 2 cells, 100 bootstrap replicates) — sanity check the whole path
python scripts/run_stopdff_v5_local.py --data-dir data/processed \
    --paper-exports paper_exports --out-dir stopdff_v5_local_out --variant smoke

# full 96-cell final (1000 replicates) — ~1-2h on CPU (adapter build + FVI study dominate)
python scripts/run_stopdff_v5_local.py --data-dir data/processed \
    --paper-exports paper_exports --out-dir stopdff_v5_local_out --variant final
```

The driver prints each stage's identities and asserts `release_status == VALID`. Output run
directory: `stopdff_v5_local_out/runs/<run_id>/` with `aggregate.json`, `cells/`, `reports/`,
`figures/`, `command_manifest.json` (local backend), and `SHA256SUMS`.

> Note on `all-MiniLM-L6-v2` numerics: raw cosine similarities are rounded to 6 decimals so
> adapter rows are byte-stable across builds. Across *different* hardware the calibrated
> probabilities can differ negligibly, so the exact family CI is guaranteed only on matching
> hardware; the qualitative verdict is stable. The published run's adapter passed a
> two-build byte-identical determinism pilot on Modal L40S.

## Modal reproduction

`scripts/modal_stopdff_v5_runner.py` defines the stage functions with a **source-only**
image (`git archive`), the `/stopdff/` Volume layout on `cs321m-stopdff-artifacts`,
one-writer-per-run + per-cell Volume commits + reload-on-resume, and **L40S only for the
adapter build** (CPU elsewhere). Orchestration (upload → verify → adapter determinism pilot
→ FVI study → bootstrap → smoke → mutation gate → 96-cell final → validate → package) is
driven from a control machine; see the control-plane driver pattern in the PR description.
A Modal payment method is required for L40S GPU functions.

## Standalone validation (acceptance contract)

```bash
# validate the run spec is a well-formed final profile
python scripts/validate_stopdff_bucketed_sweep.py validate-spec \
    stopdff_v5_release/modal_final_94274b55b319bbb8/run_package/run_spec.json --require-final-profile

# validate a downloaded/local run package (recomputes every statistic from adapter rows)
python scripts/validate_stopdff_bucketed_sweep.py validate RUN_ROOT \
    --backend {local|modal} --adapter-bundle ADAPTER_BUNDLE \
    --require-final-profile --require-package

# negative mutation self-test (synthetic fixtures)
python scripts/validate_stopdff_bucketed_sweep.py self-test
```

The checker independently recomputes index/fractional metrics, never-buzz rates, continuation
coverage, ceiling flags, cell bootstrap intervals, cell verdicts, the family maximum-statistic
interval, the family verdict, gate-override effects, and release validity; it never trusts a
serialized verdict field. Device-1-style integrity checks (safe checksums, PNG validity, path
safety) are also enforced with `--require-package`.

## Tests

```bash
pip install pytest
python -m pytest tests/test_stopdff_v5_core.py tests/test_stopdff_v5_pipeline.py \
    tests/test_stopdff_v5_checker.py -q     # 37 passed + mutation suite
```

## Deviations from the v5 contract (documented)

1. Environments are pinned via explicit `pip` versions and recorded in each run's
   `environment.json` / environment-contract identity, rather than `uv sync --frozen` from a
   committed `uv.lock` (none exists in this repo).
2. `run_package/external_artifacts.json` is present (checker-required) but minimal; the full
   retrieval index for the large Volume-resident artifacts (raw inputs, model snapshot,
   adapter rows, bootstrap indices) is published alongside as `retrieval_index.json`.
