# StopDFF v5 bucketed-DP paired audit — corrected source & reproduction

This documents the **v5 "Does the Proxy Preserve the Decision?" StopDFF audit**
(`profile_name: stopdff_bucketed_dp_paired_v2`) implemented under `scripts/stopdff_v5/`,
and how to produce a new identity-bound run **on any machine (CPU)** or **on Modal**.
The corrected source is intentionally separate from generated scientific evidence.

## What this audits

Given quizbowl tossups revealed prefix-by-prefix, we compare two answer-elicitation formats
per item — a fixed multiple-choice condition (**MC**) and a gold-answer question condition
(**QA**) — under a finite-horizon **three-action Bellman stopping policy** (ANSWER / WAIT /
ABSTAIN). The paired signed index metric is `D_i = tau_MC - tau_QA` (difference in the
prefix index at which the policy commits). The preregistered gate is the **median absolute
prefix-index shift** with material threshold **1**, controlled family-wide across a
**96-cell sensitivity grid** (4 rewards × 2 continuations × 3 calibrators × 2 prefix-bucketings
× 2 category-poolings) via a common-bootstrap **maximum statistic**.

## Historical v5 evidence (not the corrected release)

Commit `e3ca01830ed67ae81e3f9683db847249e3fbec14` and the identities below describe
the earlier v5 execution. That commit is not an ancestor of this corrected source branch,
and its artifacts are not stored in this branch. Preserve it as historical evidence; do
not relabel, overwrite, or use it to certify a run produced by the corrected code.

| Artifact | ID / value |
|---|---|
| Historical execution Git SHA | `e3ca01830ed67ae81e3f9683db847249e3fbec14` |
| source manifest ID | `c80393ba2694a17e440c75a782af061efa138c35f9774e8ab12eb81c4fe93077` |
| raw-input bundle ID | `63ca235f7e359afe1e541a2637482eb7ed3c70d765c36e311b6b9ae0eb880224` |
| model snapshot | `sentence-transformers/all-MiniLM-L6-v2` @ rev `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, `trust_remote_code=false` |
| model snapshot ID | `33b48dc6daf60b6e0a2190964bf3faafc249732269fed1f717f197940cd6f893` |
| adapter bundle ID | `13559ea71e5573721da627434fb4cbf2e6ba499006a1b3f48300a59b89a1c355` |
| FVI-study ID (selected: tol `1e-6`, max_iter `100`) | `bfc74931b72b264b37e8a09fb90e4e446433060a118e8a92f9ec2f9275c872fe` |
| bootstrap-plan ID (final, 1000 reps, PCG64 seed 1) | `076ac23359ad4c7dd0c9ba108c5a41a4bb43ebd9fff3629837991ba459a426b0` |
| historical run-spec ID | `94274b55b319bbb8c236b242ec436fad5756e8a37da5cdd608fc7690b20fa887` |

The historical record reported 96/96 completed cells and a family verdict of `WARN`.
Those claims must be evaluated only with the historical package and validator. A corrected
release remains pending until artifacts are rebuilt from a frozen merged SHA and pass the
current checker and mutation suite.

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
tests/test_stopdff_v5_*.py                        unit/integration/identity/mutation tests
```

## Environment

Python 3.11. Install the runtime dependencies:

```bash
python3.11 -m venv .venv-stopdff-v5 && source .venv-stopdff-v5/bin/activate
pip install -U pip
pip install "numpy>=1.26,<3" "scipy>=1.11" "scikit-learn>=1.3" "pandas>=2.1" \
            "matplotlib>=3.7" "sentence-transformers>=2.7" "huggingface_hub>=0.23"
```

The historical run reported: python 3.11.12, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0,
pandas 3.0.3, sentence-transformers 5.6.0, transformers 5.13.1, huggingface_hub 1.23.0
(recorded in each run's `environment.json`).

## Local (CPU) reproduction — no Modal required

`scripts/run_stopdff_v5_local.py` runs the whole pipeline in-process on CPU:
stage raw inputs → freeze model snapshot → build adapter → FVI study → bootstrap plans →
2-cell smoke → 96-cell final → standalone validation (`backend=local`).

You need ten raw inputs (the full-scale `mc_dataset.json` / `train_dataset.json` /
`val_dataset.json` / `test_dataset.json` / `build_metadata.json` /
`split_metadata.json` live under
`data/processed/`; `calibration.json` and `stopdff.json` under `paper_exports/`;
`threshold_manifest.json{,.sha256}` at the repo root). These large inputs are reproducible
from the belief-feature pipeline (`scripts/build_mc_dataset.py`, `scripts/fresh_split.py`,
`scripts/compute_prefix_calibration.py`, `scripts/compute_stopdff.py`) and are synced via
Dropbox rather than committed (see `.gitignore` and `DATA.md`).

```bash
# smoke (fast: 2 cells, 100 bootstrap replicates) — sanity check the whole path
python scripts/run_stopdff_v5_local.py --data-dir data/processed \
    --paper-exports paper_exports --out-dir stopdff_v5_smoke_out --variant smoke

# full 96-cell final (1000 replicates) — ~1-2h on CPU (adapter build + FVI study dominate)
python scripts/run_stopdff_v5_local.py --data-dir data/processed \
    --paper-exports paper_exports --out-dir stopdff_v5_final_out --variant final
```

The driver requires a clean worktree, verifies that the executing repository is the
snapshotted repository, runs the synthetic mutation gate, and forbids a final run that
skips the FVI study. A final invocation also runs and independently validates a bounded
two-cell smoke and performs an independent second adapter build before starting the
96-cell sweep. The final run spec binds content-addressed successful receipts for the
smoke, mutation self-test, and deterministic two-build gates. Fresh mode refuses to reuse
an output directory. Before starting a sweep it also persists the exact FVI manifest needed
to reconstruct that run after interruption.

To resume an interrupted sweep, use the same data/repository arguments, variant, and output
directory with `--resume`:

```bash
python scripts/run_stopdff_v5_local.py --data-dir data/processed \
    --paper-exports paper_exports --out-dir stopdff_v5_final_out \
    --variant final --resume
```

Resume does not rebuild source, raw inputs, the model snapshot, adapters, FVI selection, or
prerequisite receipts. It rehashes those completed stages, requires their identities to match
the existing run spec and current executing source, reconstructs the bound bootstrap plan,
validates the append-only attempt history, and derives the next consecutive attempt. The
sweep's own preflight then compares every existing cell and run-level byte before it writes.
An ambiguous history, a missing durable FVI manifest, multiple matching run directories, or
any incompatible stage fails closed. A fully packaged, valid run is returned unchanged.

The driver prints each stage's identities and asserts `release_status == VALID`. Output run
directory: `<out-dir>/runs/<run_id>/` with `aggregate.json`, `cells/`, `reports/`,
`figures/`, `command_manifest.json` (local backend), and `SHA256SUMS`. Packaging is
create-once: a repeat accepts only byte-identical cached report/evidence bytes. Every
package contains exact FVI and environment manifests plus a nonempty retrieval ledger
for the source, raw-input, model, FVI, and environment evidence.

> Note on `all-MiniLM-L6-v2` numerics: raw cosine similarities are rounded to 6 decimals so
> adapter rows are byte-stable across builds. Across *different* hardware the calibrated
> probabilities can differ negligibly, so the exact family CI is guaranteed only on matching
> hardware; the qualitative verdict is stable. The historical run's adapter reported a
> two-build byte-identical determinism pilot on Modal L40S.

## Modal reproduction

Install the host-side Modal SDK explicitly before invoking the Modal runner:

```bash
pip install -e '.[modal]'
```

`scripts/modal_stopdff_v5_runner.py` defines the stage functions with a **source-only**
image (`git archive`), the `/stopdff/` Volume layout on `cs321m-stopdff-artifacts`,
one-writer-per-run + per-cell Volume commits + reload-on-resume, and **L40S only for the
adapter build** (CPU elsewhere). Orchestration (upload → verify → adapter determinism pilot
→ FVI study → bootstrap → smoke → mutation gate → 96-cell final → validate → package) is
driven from a control machine. Each checked-in remote stage now revalidates the exact
source/raw/model/adapter/bootstrap/run-spec graph before it writes evidence. The required
order is upload and verify inputs → two-build adapter determinism pilot → FVI study →
bootstrap → smoke → mutation gate → final sweep → validate → package. The final stage
must carry the three content-addressed receipt IDs in `run_spec.evidence_roots` and each
receipt must bind the exact source/raw/model/adapter/FVI/environment identities. The
remote sweep verifies the receipt bytes and IDs before creating its run directory, so
an orchestration claim or unbound success flag is insufficient. Sweep attempts append
a durable `state=started` record before any cell evidence; a completion/failure record
is stored separately, and resume preflights all existing evidence before repairing
anything. If a hard process exit leaves exactly the latest started attempt without a
terminal record, resume first creates a canonical `state=interrupted` result with reason
`terminal_result_missing_at_resume`, commits it, and only then appends the next attempt.
Attempt history is append-only, result files are create-once, and attempt numbers are
consecutive. A missing older result, multiple missing results, an unexpected result, or
a conflicting interruption record is ambiguous and fails closed. These recovery
semantics assume the documented one-writer-per-run invariant. A Modal payment method
and explicit compute authorization are required for L40S or full-release execution.
Set `STOPDFF_V5_SOURCE_DIR` to the clean extracted archive for the frozen source SHA;
the runner refuses to construct an image when this binding is absent.

After uploading the source and raw-input bundles to their documented Volume paths, create a
small control plan. The two adapter subdirectories must be distinct and create-once:

```json
{
  "source_id": "<64-hex source manifest ID>",
  "raw_id": "<64-hex raw-input manifest ID>",
  "adapter_subdirs": ["control_build_a", "control_build_b"],
  "gate_overrides": {},
  "resource_summary": {"backend": "modal"}
}
```

With explicit authorization for Modal compute, start the durable controller:

```bash
modal run scripts/modal_stopdff_v5_runner.py::control_main \
    --plan-path stopdff_v5_control.json \
    --state-path stopdff_v5_control_state.json
```

The controller verifies both uploaded bundles, freezes the model, performs and receipts two
adapter builds, promotes the bound adapter, runs the FVI study, smoke bootstrap/sweep,
mutation gate, final bootstrap/sweep, prepackage validation, packaging, and final validation
in that fixed order. Each stage intent and result is fsynced to the state file and adjacent
JSONL journal. To continue after an interrupted host process, use the same plan and state:

```bash
modal run scripts/modal_stopdff_v5_runner.py::control_main \
    --plan-path stopdff_v5_control.json \
    --state-path stopdff_v5_control_state.json --resume
```

Completed stages are reused from the bound journal. A sweep whose response was lost is
retried as the next explicit resume attempt; incompatible state or a changed plan fails
closed. `probe_main` remains available as a separate environment-only entry point.

## Standalone validation (acceptance contract)

```bash
# validate the run spec is a well-formed final profile
python scripts/validate_stopdff_bucketed_sweep.py validate-spec \
    RUN_ROOT/run_spec.json --require-final-profile

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
    tests/test_stopdff_v5_checker.py tests/test_stopdff_v5_identity_graph.py \
    tests/test_stopdff_v5_producers.py tests/test_dataset_splits.py -q
```

## Deviations from the v5 contract (documented)

1. Environments are pinned via explicit `pip` versions and recorded in each run's
   `environment.json` / environment-contract identity, rather than `uv sync --frozen` from a
   committed `uv.lock` (none exists in this repo).
2. Large raw inputs, model snapshots, adapter rows, and bootstrap arrays remain external
   artifacts. A future release must publish their exact IDs, hashes, byte sizes, and durable
   retrieval locations separately from the source PR.
