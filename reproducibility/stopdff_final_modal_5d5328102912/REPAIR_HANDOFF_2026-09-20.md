# Repair handoff for Jane's StopDFF reproduction

This guide addresses the 19 September 2026 failures for
`final_modal_5d5328102912`. The historical source is
`0017b89da921e85a6960cd8a22f1969176aed079`; the repair is a separate source
commit. Record both. New results establish reproduction under their recorded
source and environment, not independent authentication of the historical
producer's execution.

The numerical reducer now treats an absent interval category as count zero.
It still rejects changed nonzero counts and unexpected nonzero categories.
The runner imports a verified external model bundle into a **new** output
directory. Do not pre-create or populate smoke/final outputs, manufacture a
resume checkpoint, edit historical manifests, or redownload a model and label
it as the archived snapshot.

## 1. Lock the two source checkouts and environment

Run these Bash commands after changing the first two absolute paths. The archive
root must contain `verified_export/` and `canonical_adapter/`. Use a fresh work
directory, on a filesystem that preserves executable permissions. The repair
branch is resolved once below; all later commands use that detached commit.

```bash
set -euo pipefail
ARCHIVE_ROOT=/absolute/path/to/extracted/final_modal_5d5328102912
WORK=/absolute/path/to/new/stopdff-repair-20260920
test -d "$ARCHIVE_ROOT/verified_export"
test -d "$ARCHIVE_ROOT/canonical_adapter"
test ! -e "$WORK"
mkdir "$WORK"
mkdir "$WORK/handoff"

git clone --branch fix/jane-stopdff-reproduction-20260920 \
  https://github.com/ankaggarwal94/qanta-buzzer.git "$WORK/code"
CODE="$WORK/code"
REPAIR_COMMIT=$(git -C "$CODE" rev-parse HEAD)
git -C "$CODE" switch --detach "$REPAIR_COMMIT"
git -C "$CODE" worktree add --detach "$WORK/historical" \
  0017b89da921e85a6960cd8a22f1969176aed079
HISTORICAL="$WORK/historical"
CAPSULE="$CODE/reproducibility/stopdff_final_modal_5d5328102912"
HANDOFF="$WORK/handoff"
PACKAGE_ORIGINAL="$ARCHIVE_ROOT/verified_export"
CANONICAL_ADAPTER="$ARCHIVE_ROOT/canonical_adapter"

git -C "$CODE" rev-parse HEAD > "$HANDOFF/repair_commit.txt"
git -C "$HISTORICAL" rev-parse HEAD > "$HANDOFF/historical_commit.txt"
git -C "$CODE" status --porcelain > "$HANDOFF/repair_worktree_status.txt"
test ! -s "$HANDOFF/repair_worktree_status.txt"

python3.11 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
python -c 'import sys; assert sys.version_info[:3] == (3, 11, 12), sys.version'
python -m pip install -r "$CAPSULE/requirements-stopdff-final.txt" \
  "jsonschema>=4.18,<5" PyYAML
python -m pip install -e "$CODE" --no-deps
python --version > "$HANDOFF/python_version.txt" 2>&1
python -m pip freeze > "$HANDOFF/pip_freeze.txt"
```

If Python 3.11.12 or a recorded package version is unavailable, retain that
failure and resolve the environment before claiming a matching environment.
Hardware and framework differences can produce numerical differences even
with identical model and input bytes. The runner records its resolved runtime.
`jsonschema` and `PyYAML` are validator/import dependencies outside the
historically recorded scientific package set; retain their resolved versions in
`pip_freeze.txt` too.

## 2. Check the historical archive and diagnose source permissions

First preserve checksum and numerical results on the original extraction:

```bash
(cd "$PACKAGE_ORIGINAL" && sha256sum -c SHA256SUMS) \
  2>&1 | tee "$HANDOFF/archive_sha256.log"
python "$CAPSULE/verify_expected_results.py" "$PACKAGE_ORIGINAL" \
  | tee "$HANDOFF/archive_numerical.json"
```

The reducer checks the exported statistics; it does not replace structural
validation or inference. A SHA256 check covers bytes, not executable modes.
The source inspection tool checks the complete declared inventory before any
permission restoration. Follow its diagnostic and working-copy commands below.
If it finds changed bytes, missing/extra files, unsafe paths or symlinks, stop
and retain the report. Mode restoration cannot repair those defects.

```bash
SOURCE_ID=83410b86a734be9fd4c43b7df7b05458d64c80ce898686e960be1860f5348197
SOURCE_STATUS=0
python "$CODE/scripts/inspect_stopdff_source_package.py" "$PACKAGE_ORIGINAL" \
  --expected-source-id "$SOURCE_ID" \
  | tee "$HANDOFF/source_original.json" || SOURCE_STATUS=$?
case "$SOURCE_STATUS" in
  0|2) ;;  # intact, or verified bytes with only executable-mode mismatches
  *) exit "$SOURCE_STATUS" ;;
esac

PACKAGE_WORK="$WORK/verified_export_working"
python "$CODE/scripts/inspect_stopdff_source_package.py" "$PACKAGE_ORIGINAL" \
  --expected-source-id "$SOURCE_ID" --repair-copy "$PACKAGE_WORK" \
  --receipt "$HANDOFF/source_mode_repair_receipt.json" \
  | tee "$HANDOFF/source_working_copy.json"
```

The tool requires fresh destination and receipt paths. Keep both package trees
private and free of concurrent writers while it runs. It verifies package bytes
before copying and after restoring only manifest-declared source modes. A failed
copy can leave a partial destination; preserve that failure and use a fresh path
after diagnosis. No success receipt is written for a failed repair.

After the source inventory passes, validate the working package with the
historical validator and retain its complete output:

```bash
python "$HISTORICAL/scripts/validate_stopdff_bucketed_sweep.py" \
  validate "$PACKAGE_WORK" --backend modal \
  --adapter-bundle "$CANONICAL_ADAPTER" \
  --require-final-profile --require-package --json \
  | tee "$HANDOFF/archive_structural.json"
```

An invalid source inventory previously caused the checker to discard the source
manifest and emit 15 additional producer-hash failures. These downstream errors
were not independent evidence of 15 different source files. The repair checker
reports the dependency as blocked. The original extraction's actual defect is
unresolved until its inspection report is available. The historical mode-repair
receipt records a permission-only change and zero changed bytes; preserve it.

## 3. Map and verify all ten frozen inputs

The archive stores the ten inputs by basename under `evidence/raw_inputs/raw/`.
The verifier expects their original repository-relative paths. Build that
layout outside both source checkouts, without regenerating any inputs:

```bash
INPUT_ROOT="$WORK/inputs"
python - "$PACKAGE_ORIGINAL/evidence/raw_inputs/raw" "$INPUT_ROOT" \
  "$CAPSULE/run_identity.json" <<'PY'
import json
from pathlib import Path
import shutil
import sys

raw, target, manifest = map(Path, sys.argv[1:])
target.mkdir()
for row in json.loads(manifest.read_text())["raw_inputs"]:
    destination = target / row["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw / destination.name, destination)
PY
python "$CAPSULE/verify_raw_inputs.py" --repo-root "$INPUT_ROOT" \
  | tee "$HANDOFF/raw_inputs.log"
cmp "$INPUT_ROOT/threshold_manifest.json" "$CODE/threshold_manifest.json"
cmp "$INPUT_ROOT/threshold_manifest.json.sha256" "$CODE/threshold_manifest.json.sha256"
```

The runner uses threshold files from its own clean checkout, so both `cmp`
commands must succeed. Do not copy raw inputs over tracked source files.

## 4. Import the archived model without creating run outputs

Keep the full snapshot, including its hidden `.cache` entries. The model import
verifies the manifest identity, complete file inventory, hashes and sizes;
it rejects symlinks and mismatches. It does not contact the Hugging Face Hub.

```bash
MODEL_INPUT="$WORK/model-input"
MODEL_ID=33b48dc6daf60b6e0a2190964bf3faafc249732269fed1f717f197940cd6f893
mkdir "$MODEL_INPUT"
cp -a "$PACKAGE_ORIGINAL/evidence/model_snapshot_manifest.json" \
  "$MODEL_INPUT/model_snapshot_manifest.json"
cp -a "$PACKAGE_ORIGINAL/evidence/model_snapshot/snapshot" \
  "$MODEL_INPUT/snapshot"

SMOKE_ROOT="$WORK/smoke"
FINAL_ROOT="$WORK/final"
test ! -e "$SMOKE_ROOT"
test ! -e "$FINAL_ROOT"
cd "$CODE"
python scripts/run_stopdff_v5_local.py \
  --repo-root "$CODE" --data-dir "$INPUT_ROOT/data/processed" \
  --paper-exports "$INPUT_ROOT/paper_exports" \
  --model-snapshot-dir "$MODEL_INPUT" --expected-model-snapshot-id "$MODEL_ID" \
  --out-dir "$SMOKE_ROOT" --variant smoke \
  2>&1 | tee "$HANDOFF/smoke.log"

smoke_runs=("$SMOKE_ROOT"/runs/smoke_local_*)
test "${#smoke_runs[@]}" -eq 1
test -d "${smoke_runs[0]}"
python scripts/validate_stopdff_bucketed_sweep.py \
  validate "${smoke_runs[0]}" --backend local \
  --adapter-bundle "$SMOKE_ROOT/adapter_bundle" --require-package --json \
  | tee "$HANDOFF/smoke_structural.json"
```

Start the full run only after the smoke command and smoke validator succeed:

```bash
python scripts/run_stopdff_v5_local.py \
  --repo-root "$CODE" --data-dir "$INPUT_ROOT/data/processed" \
  --paper-exports "$INPUT_ROOT/paper_exports" \
  --model-snapshot-dir "$MODEL_INPUT" --expected-model-snapshot-id "$MODEL_ID" \
  --out-dir "$FINAL_ROOT" --variant final \
  2>&1 | tee "$HANDOFF/final.log"

final_runs=("$FINAL_ROOT"/runs/final_local_*)
test "${#final_runs[@]}" -eq 1
test -d "${final_runs[0]}"
RUN_ROOT="${final_runs[0]}"
python scripts/validate_stopdff_bucketed_sweep.py \
  validate "$RUN_ROOT" --backend local \
  --adapter-bundle "$FINAL_ROOT/adapter_bundle" \
  --require-final-profile --require-package --json \
  | tee "$HANDOFF/final_structural.json"
python "$CAPSULE/verify_expected_results.py" "$RUN_ROOT" \
  | tee "$HANDOFF/final_numerical.json"
cp -a "$RUN_ROOT/aggregate.json" "$RUN_ROOT/run_manifest.json" "$HANDOFF/"
cp -a "$FINAL_ROOT/local_lifecycle.json" "$HANDOFF/"
```

For an interrupted run, repeat its exact command with `--resume` and a new log
filename. Keep both model arguments: the expected identity is bound into the
lifecycle. A changed identity, invalid completed stage or fabricated checkpoint
fails closed. The final runner performs its own two-cell prerequisite smoke,
independent second adapter build, mutation gate and final validation as well.

These commands run locally in the shell's environment; they do not submit a
Slurm job. On Nexus, use the appropriate allocation and record the actual
hardware and model device. Do not claim an executed smoke or 96-cell rerun from
successful unit tests alone.

## 5. Correct the leakage population and complete the handoff

The 20 September inspection downloaded the exact frozen train/validation/test
files and matched their sizes and SHA256 values against `run_identity.json`.
Across each pair of splits there were **zero shared QIDs, exact question texts,
or normalized question texts**. Normalization was Unicode NFKC, casefolding and
collapsed whitespace, preserving punctuation. All 96 exported cell item sets
matched the frozen test file's 3,037 QIDs.

| Frozen split | Rows | Unique normalized question texts | Excess repeated-text rows within split |
|---|---:|---:|---:|
| Train | 14,211 | 14,211 | 0 |
| Validation | 3,050 | 3,020 | 30 |
| Test | 3,037 | 3,007 | 30 |

The reported 627-duplicate count is not supported for this archive; the earlier
`full_qanta` overlap finding concerns a separate build. In any event,
627 / 3,037 is 20.645%, not 6.65%. The appropriate statement
for this frozen run is zero cross-split text overlap under the stated checks,
with the within-split repeats disclosed separately. This does not exclude near
duplicates, semantic overlap or pretraining contamination. Full-prompt accuracy,
Random-K and other calibration experiments remain unassessed until their input
hashes are mapped; do not clear those rows by analogy.

Return the `handoff/` directory with the original extraction's source inspection,
any working-copy repair receipt, complete hash/validator logs, both source
commits, environment, real smoke/full logs, aggregate, run manifest, lifecycle,
and all mismatches. Preserve the original archive and the new run directories.
Record the exact shared location for these files so follow-up can inspect the
actual failures. A private handoff does not resolve the public archive and
licensing items in `REPRODUCIBILITY_STATUS.md`.
