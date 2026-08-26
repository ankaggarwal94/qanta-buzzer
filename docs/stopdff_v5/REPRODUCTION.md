# StopDFF v5 bucketed-DP paired audit — corrected source & reproduction

> **Scope note (R-038):** stopping-shift evidence in this repository concerns
> constructed QA reference trajectories — a constructed-reference sensitivity
> diagnostic. It does not assert observed open-ended decision preservation;
> the authoritative claim ledger lives under `reproducibility/colm_aims_2026/`.


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

The corrected source changes science-affecting behavior: a uniquely optimal nonterminal
`ABSTAIN` now terminates as never-buzz, category labels must be nonempty and agree exactly
between split and MC rows, integer split targets use deterministic Hamilton apportionment,
and calibration ECE is recomputed from the rounded Platt parameters that are actually
serialized and used by the sweep. `torch` is also part of the environment identity. Therefore
all affected adapter, split, calibration, sweep, report, and package evidence must be
regenerated; none of the historical scientific bytes can certify the corrected release.

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
  checker_calibration.py focused producer/calibrator shape validation
  checker_package.py packaged evidence and manifest-graph validation
  selftest.py        negative mutation suite + valid-package builder
scripts/validate_stopdff_bucketed_sweep.py   standalone checker CLI (acceptance contract)
scripts/run_stopdff_v5_local.py              CPU/local end-to-end reproduction driver
scripts/modal_stopdff_v5_runner.py           Modal remote functions (source-only image)
scripts/modal_stopdff_v5_assurance.py        cross-process Modal recovery-assurance driver
scripts/verify_stopdff_v5_modal_assurance.py offline recovery-receipt verifier
schemas/stopdff_*.schema.json                JSON schemas for profile/run-spec/calibrator/continuation/gate
tests/test_stopdff_v5_*.py                        unit/integration/identity/mutation tests
```

## Environment

Python 3.11. Install the runtime dependencies:

```bash
python3.11 -m venv .venv-stopdff-v5 && source .venv-stopdff-v5/bin/activate
pip install -U pip
pip install "numpy>=1.26,<3" "scipy>=1.11" "scikit-learn>=1.3" "pandas>=2.1" \
            "matplotlib>=3.7" "sentence-transformers>=2.3.0" "torch>=2.6.0" \
            "huggingface_hub>=0.23"
```

The historical run reported: python 3.11.12, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0,
pandas 3.0.3, sentence-transformers 5.6.0, transformers 5.13.1, huggingface_hub 1.23.0
(recorded in each run's `environment.json`). That older package list omitted `torch`, so its
environment identity is incomplete and stale under the corrected contract. No historical
`torch` version is inferred here. New local and Modal environment identities include the
resolved `torch` version alongside every other evidentiary package.

## Local (CPU) reproduction — no Modal required

`scripts/run_stopdff_v5_local.py` runs the whole pipeline in-process on CPU:
stage raw inputs → freeze model snapshot → build adapter → FVI study → bootstrap plans →
2-cell smoke → 96-cell final → standalone validation (`backend=local`).

You need ten raw inputs (the full-scale `mc_dataset.json` / `train_dataset.json` /
`val_dataset.json` / `test_dataset.json` / `build_metadata.json` /
`split_metadata.json` live under
`data/processed/`; `calibration.json` and `stopdff.json` under `paper_exports/`;
`threshold_manifest.json{,.sha256}` at the repo root). `split_metadata.json` is emitted by
`scripts/build_mc_dataset.py` from the retained MC train/validation/test bytes and is
validated against those exact datasets during v5 staging. These large inputs are reproducible
from the belief-feature pipeline (`scripts/build_mc_dataset.py`,
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
for the source, raw-input, model, FVI, and environment evidence. Source, raw-input, and
model evidence is self-contained: packaging first validates each staged manifest against
its exhaustive content tree, copies every declared byte under `evidence/`, and final
validation repeats the exhaustive byte checks. The question-trajectory binding is then
recomputed from the packaged val/test/MC raw datasets and compared with the adapter, so
the raw manifest's semantic assertion is not trusted on its own.

> Note on `all-MiniLM-L6-v2` numerics: raw cosine similarities are rounded to 6 decimals so
> adapter rows are byte-stable across builds. Across *different* hardware the calibrated
> probabilities can differ negligibly, so the exact family CI is guaranteed only on matching
> hardware. No cross-hardware qualitative-verdict stability claim is made without a
> fresh validated run. The historical run's adapter reported a
> two-build byte-identical determinism pilot on Modal L40S.

## Modal reproduction

Install the host-side Modal SDK explicitly before invoking the Modal runner:

```bash
pip install -e '.[modal]'
```

`scripts/modal_stopdff_v5_runner.py` defines the stage functions with a **source-only**
image (`git archive`), the `/stopdff/` Volume layout on `cs321m-stopdff-artifacts`,
one-writer-per-run + per-cell Volume commits + reload-on-resume, and **L40S only for the
adapter build** (CPU elsewhere). Orchestration (create-once stage + verify inputs → adapter
determinism pilot → FVI study → bootstrap → smoke → mutation gate → 96-cell final →
package → validate the packaged output) is driven from a control machine. Each checked-in
remote stage now revalidates the exact source/raw/model/adapter/bootstrap/run-spec graph
before it writes evidence. The required order is create-once stage and verify inputs →
two-build adapter determinism pilot → FVI study → bootstrap → smoke → mutation gate →
final sweep → package (with fail-closed internal prevalidation) → validate the packaged
output. The final sweep
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
Set `STOPDFF_V5_SOURCE_DIR` to the canonical `source_snapshot/` bundle produced by
the local runner (the directory containing `source_manifest.json` and `source/`).
Before defining the Modal image, the runner validates the manifest, exact file set,
modes, sizes, and digests, copies that closed tree to a private staging directory,
revalidates the copy, and binds its manifest ID to the control plan. It refuses to
construct an image from a bare or unlisted source tree. The directly callable adapter
build, adapter-determinism, mutation, and sweep stages also reject any source identity
other than the one bound into that validated image. Each of those stages rehashes the
executing source tree before producing or consuming scientific evidence. Modal automatic
source inclusion is disabled, so the function-defining module is imported only from that
validated frozen tree rather than from an additional host-side source overlay. During host
deployment, `modal.is_local()` gates bundle materialization and `add_local_dir`; the validated
source-manifest ID and selected App name are baked into the image environment. A remote
container import never reads `STOPDFF_V5_SOURCE_DIR` or a host temporary path: it requires the
baked 64-hex identity and sets its execution source to `/root/src`.

Stage both input bundles with the checked-in create-once entry point; do not upload their
directories by hand. `STOPDFF_V5_SOURCE_DIR` must name the same closed source-snapshot bundle
used for `--source-bundle`, and `--raw-bundle` is the local runner's closed `raw_inputs/`
directory. Use a new local receipt path:

```bash
export STOPDFF_V5_SOURCE_DIR=/absolute/path/to/stopdff_v5_final_out/source_snapshot
modal run scripts/modal_stopdff_v5_runner.py::stage_inputs_main \
    --source-bundle "$STOPDFF_V5_SOURCE_DIR" \
    --raw-bundle /absolute/path/to/stopdff_v5_final_out/raw_inputs \
    --receipt-path stopdff_v5_input_staging_receipt.json
```

For each content-addressed destination, the receipt reports `status: created` only when the
remote destination had no entries and the runner uploaded it with
`Volume.batch_upload(force=False)`. An existing, exactly valid bundle reports
`status: cached`. Both paths require remote exhaustive
manifest verification and a host-side readback of the exact manifest bytes. A partial,
malformed, mismatched, or otherwise noncanonical existing path fails closed; the staging
command never repairs or overwrites it. Copy the `source.id` and `raw.id` from the receipt
into a small control plan. The two adapter subdirectories must be distinct and create-once:

```json
{
  "source_id": "<64-hex source manifest ID>",
  "raw_id": "<64-hex raw-input manifest ID>",
  "adapter_subdirs": ["control_build_a", "control_build_b"],
  "gate_overrides": {},
  "resource_summary": {"backend": "modal"}
}
```

`resource_summary` is content-addressed into the run spec. Its values must use the
canonical identity types (objects, arrays, strings, integers, booleans, or null);
record decimal costs as strings rather than binary floating-point values.

With explicit authorization for Modal compute, start the durable controller:

```bash
modal run scripts/modal_stopdff_v5_runner.py::control_main \
    --plan-path stopdff_v5_control.json \
    --state-path stopdff_v5_control_state.json
```

The controller verifies both uploaded bundles, freezes the model, and invokes one
determinism gate that owns exactly two fresh, path-distinct adapter producer calls. The
gate rejects pre-existing destinations and records the two distinct Modal function-call
IDs in schema-v2 evidence before issuing a receipt. The controller then promotes the
bound adapter, runs the FVI study, smoke bootstrap/sweep,
mutation gate, final bootstrap/sweep, package-internal prevalidation, packaging,
and final validation in that fixed order. Each stage intent and result is fsynced
to the state file and adjacent JSONL journal. Schema-v4 journal records have exact
event-specific shapes, canonical JSON bytes, and a hash link to the preceding record.
Every `stage_completed` event binds the complete completed-stage payload by a SHA-256
of finite canonical JSON. `control_completed` and `control_revalidated` likewise bind
the exact terminal result object (`run_id`, `run_spec_id`, `adapter_id`, `receipt_ids`,
and `validation`) rather than only its run ID. Replay requires the canonical stage set
and binds attempts, completed payloads, terminal status/result, and failure detail back
to the checkpoint; mutation of either a cached stage payload or the terminal validation
therefore fails replay. Projection replay also enforces the fixed stage order, requires
every predecessor before a stage can start, and forbids terminal completion until every
canonical stage is complete.

To continue after an interrupted host process, use the same plan and state:

```bash
modal run scripts/modal_stopdff_v5_runner.py::control_main \
    --plan-path stopdff_v5_control.json \
    --state-path stopdff_v5_control_state.json --resume
```

Completed stages are reused only after their validators and journal digests pass. If a
cached stage is invalid, the controller conservatively invalidates every completed stage
later in the fixed stage order (the full transitive downstream suffix), journals each
removal, and then invalidates and retries the upstream stage. Attempt counters are retained,
so each rerun receives the next attempt number. This suffix rule deliberately favors
fail-closed recomputation over trying to infer a narrower dependency graph.

If the host controller disappears after a durable `stage_started` event, resume first closes
that exact controller attempt with a `stage_failed` event of type
`HostControllerInterrupted`, before validating or invalidating any cached stage. The stage's
next invocation then receives its next controller attempt number; Modal sweep evidence still
derives its separate scientific attempt number from the durable run root.

Control-state schemas v1, v2, and v3 predate complete payload/terminal-result binding and
are rejected; restart with new adapter subdirectories. A lost
adapter-build response after its Volume commit likewise requires new subdirectories so
the determinism claim fails closed. A sweep retry derives its evidence
mode and number inside the remote call from the durable run directory: an absent root
starts fresh at evidence attempt 1, a canonical nonempty history resumes at its next
number, and a partial or malformed root fails closed without being repaired. The local
controller invocation count is never reused as a scientific attempt number. A changed
plan also fails closed. `probe_main` remains available as a separate environment-only
entry point.

### Bounded live recovery assurance

The checked-in recovery assurance is a CPU-only, zero-cell canary for the production sweep
attempt protocol. It deliberately hard-exits once, after committing attempt 1's `started`
record and a durable crash arm. The arm makes an automatic Modal reschedule return instead
of crashing again. A later call commits the canonical `interrupted` classification before
attempt 2 is appended; another call finishes attempt 2; a final call independently reads the
durable bytes back. Use a fresh 8–64 character lowercase hex/hyphen tag and a fresh local
receipt directory for every campaign.

Before any mutation, `classify` requires the exact initial state and `finish` requires the
exact classified state; `verify` requires the exact finished state. Repeating `classify` or
`finish` after its exact durable result exists performs an idempotent readback. Any other
out-of-order, partial, or noncanonical observed phase state fails before the sweep protocol
is invoked.

First deploy the exact frozen-source runner. Then run each driver command as a separate host
process so recovery cannot depend on the submitting process's memory.

`STOPDFF_V5_APP_NAME` deploys a second app against the shared
`cs321m-stopdff-artifacts` Volume, and `max_containers=1` serializes writers per *app* — so
a non-default app name weakens the single-writer invariant on shared slots. The runner
therefore refuses a non-default app name unless `STOPDFF_V5_ALLOW_APP_OVERRIDE=1` is also
set, and prints a warning to stderr while the override is active. The assurance campaign
below is the intended use: its uniquely tagged canary writes only under `pilots/<tag>/`,
never the pipeline's input/adapter/run slots. Do not run an overridden app concurrently
with the default pipeline app against the same slots.

```bash
export STOPDFF_V5_SOURCE_DIR=/absolute/path/to/stopdff_v5_final_out/source_snapshot
export STOPDFF_V5_APP_NAME=cs321m-stopdff-v5-assurance-45b7f81f
export STOPDFF_V5_ALLOW_APP_OVERRIDE=1
modal deploy scripts/modal_stopdff_v5_runner.py

export STOPDFF_ASSURANCE_DEPLOYMENT="$STOPDFF_V5_APP_NAME"
export STOPDFF_ASSURANCE_TAG=45b7f81f-acde
export STOPDFF_ASSURANCE_SOURCE_ID="replace-with-source.id-from-the-staging-receipt"
export STOPDFF_ASSURANCE_DIR=stopdff_v5_modal_assurance_45b7f81f-acde
mkdir "$STOPDFF_ASSURANCE_DIR"

python scripts/modal_stopdff_v5_assurance.py submit \
    --deployment "$STOPDFF_ASSURANCE_DEPLOYMENT" \
    --tag "$STOPDFF_ASSURANCE_TAG" \
    --receipt "$STOPDFF_ASSURANCE_DIR/submitted.json"

python scripts/modal_stopdff_v5_assurance.py recover \
    --call-receipt "$STOPDFF_ASSURANCE_DIR/submitted.json" \
    --timeout-seconds 300 \
    --receipt "$STOPDFF_ASSURANCE_DIR/recovered.json"

python scripts/modal_stopdff_v5_assurance.py classify \
    --deployment "$STOPDFF_ASSURANCE_DEPLOYMENT" \
    --tag "$STOPDFF_ASSURANCE_TAG" \
    --timeout-seconds 300 \
    --receipt "$STOPDFF_ASSURANCE_DIR/classified.json"

python scripts/modal_stopdff_v5_assurance.py finish \
    --deployment "$STOPDFF_ASSURANCE_DEPLOYMENT" \
    --tag "$STOPDFF_ASSURANCE_TAG" \
    --timeout-seconds 300 \
    --receipt "$STOPDFF_ASSURANCE_DIR/finished.json"

python scripts/modal_stopdff_v5_assurance.py verify \
    --deployment "$STOPDFF_ASSURANCE_DEPLOYMENT" \
    --tag "$STOPDFF_ASSURANCE_TAG" \
    --timeout-seconds 300 \
    --receipt "$STOPDFF_ASSURANCE_DIR/verified.json"

python scripts/verify_stopdff_v5_modal_assurance.py \
    --submitted "$STOPDFF_ASSURANCE_DIR/submitted.json" \
    --recovered "$STOPDFF_ASSURANCE_DIR/recovered.json" \
    --classified "$STOPDFF_ASSURANCE_DIR/classified.json" \
    --finished "$STOPDFF_ASSURANCE_DIR/finished.json" \
    --verified "$STOPDFF_ASSURANCE_DIR/verified.json" \
    --expected-source-manifest-id "$STOPDFF_ASSURANCE_SOURCE_ID"
```

`submit` persists the Modal FunctionCall ID before its process exits. `recover` reconstructs
that call with `FunctionCall.from_id`. Each later phase is also submitted as a FunctionCall
whose ID is bound into its receipt. Every wait is limited by `--timeout-seconds`; a timed-out
call and its containers are cancelled, and each remote phase also has the runner's 300-second
function timeout. All five local receipts are create-once. A timeout, cancellation, or failed
phase makes that tag inconclusive; diagnose it and start a new campaign with a fresh tag rather
than overwriting evidence.

The final verifier is offline and read-only. It requires exact receipt schemas and consistent
deployment/tag/call identities; the exact frozen source-manifest ID; canonical, digest-bound
run-spec and bootstrap manifests; a nonempty stable input ID across the hard-exit reschedule;
different container hostnames; the exact attempt/result transition; byte-stable interruption
evidence; and agreement between the finished and final-readback observations. It prints a
canonical PASS verdict JSON to standard output and exits nonzero on any violation.

The canary exercises real Volume commit/reload, hard-exit rescheduling, cross-process
FunctionCall recovery, missing-terminal classification, consecutive attempt numbering,
create-once terminal results, and final readback. It does **not** build an adapter, request an
L40S, run any sensitivity-grid cells, validate scientific outputs, or certify the full Modal
controller/release. Those remain separate full-reproduction and standalone-acceptance gates.
After the receipts and PASS verdict have been copied to durable storage, stop only the
uniquely named assurance App. Do not stop the default pipeline App or delete the shared
Volume.

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

The standalone checker has a **trusted-producer boundary**. It verifies canonical bytes,
identity bindings, two distinct logical call IDs and adapter destinations, and equality of the
two committed adapter outputs. The recorded Modal function-call IDs, `cached: false` values,
and source-execution fields are unsigned assertions made by the supported checked-in producer;
they are not Modal-signed receipts and the checker does not independently authenticate the
invocation or its executing source. Accordingly, standalone validation establishes internal
package consistency under the supported producer workflow, not provenance against a hostile
package author. Modal may retry a logical call, so distinct call IDs also do not prove exactly
two physical container attempts.

Pass `--json` to `validate-spec`, `validate-adapter`, or `validate` for a versioned object with
the command, pass status, errors, and recomputed identity/status fields. Without `--json`, the
existing human-readable `PASS`/`FAIL` output and exit-code contract are unchanged.

## Tests

```bash
pip install pytest
python -m pytest tests/test_stopdff_v5_core.py tests/test_stopdff_v5_pipeline.py \
    tests/test_stopdff_v5_checker.py tests/test_stopdff_v5_identity_graph.py \
    tests/test_stopdff_v5_producers.py tests/test_dataset_splits.py -q
```

## Deviations from the v5 contract (documented)

1. Environments are constrained by declared `pip` ranges; exact resolved versions are recorded in each run's
   `environment.json` / environment-contract identity, rather than `uv sync --frozen` from a
   committed `uv.lock` (none exists in this repo).
2. Large raw inputs, model snapshots, adapter rows, and bootstrap arrays remain external
   artifacts. A future release must publish their exact IDs, hashes, byte sizes, and durable
   retrieval locations separately from the source PR.
