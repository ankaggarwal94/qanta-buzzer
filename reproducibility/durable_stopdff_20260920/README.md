# Durable StopDFF reproduction

This controller validates the existing repaired smoke package, then runs a fresh
96-cell final reproduction at commit `2ed304f6598b94c0c5dc15f77fb8b391ae2f03a3`.
It records a standalone smoke validation, not a successful historical runner
resume. It never edits old lifecycle records, attempts, input data, or source.

Run on a persistent POSIX job host with Git, Python **3.11.12**, the exact
`requirements.txt`, approximately 20 GiB RAM and at least 20 GiB available disk.
Work and final-output directories must use a regular POSIX filesystem supporting
advisory `flock`, hard links, atomic rename and `fsync`. Do not point the scientific
runner directly at an object-store mount or a provider volume without those
semantics. A provider-specific checkpoint/persistence adapter belongs outside
the unchanged scientific runner; this controller does not implement one.
The serial FVI studies can take hours. More allocated CPU cores do not parallelize
the Python FVI loops. Use a durable scheduler/job allocation with sufficient wall
time, rather than an interactive session that may disappear.

## Prepare inputs and code

Extract `StopDFF_Rerun_Inputs_2026-09-20.tar.gz` into a persistent directory,
preserving file modes. It contains:

```text
input/smoke_package/
input/adapter_bundle/
```

The smoke package includes all frozen raw/model/source evidence. Do not add files
inside it. The preflight rejects checksum, identity, inventory and executable-mode
mismatches; it does not silently repair inputs. Preserve the archive and its
transfer checksum. Install dependencies into a new Python 3.11.12 environment;
do not substitute nearby versions or copy a non-relocatable virtual environment.

Create a clean Git checkout outside inputs, work, outputs and receipts:

```bash
git clone https://github.com/ankaggarwal94/qanta-buzzer.git /persistent/stopdff-code
git -C /persistent/stopdff-code checkout --detach 2ed304f6598b94c0c5dc15f77fb8b391ae2f03a3
```

The code is obtained from GitHub, separately from the data and control artifacts.
The checkout must remain clean. Bytecode writes are disabled by the controller.

## Preflight, then execute

These five directories must be disjoint and may not traverse symlinks. The final
output directory must not exist. The work and receipt directories may already
exist; each invocation creates a unique receipt subdirectory.

```bash
export STOPDFF_PYTHON=/persistent/stopdff-env/bin/python
bash /persistent/controls/run_persistent.sh \
  --input-dir /persistent/inputs/input \
  --code-dir /persistent/stopdff-code \
  --work-dir /persistent/stopdff-work \
  --output-dir /persistent/final-2ed304f \
  --receipt-dir /persistent/stopdff-receipts \
  --preflight-only
```

Preflight reads and hashes approximately 1.8 GiB of inputs, checks the exact
runtime, source permissions, identities and adapter, and does not run FVI, copy
the model or create the final output. Its terminal status is `PREFLIGHT_PASSED`,
which explicitly does **not** mean scientific acceptance.

Submit the same command to the persistent host with `--preflight-only` removed.
The wrapper does not create a cloud job or detach itself. No credentials are
stored in this artifact. Supply platform-specific authentication separately.

Full execution performs:

1. Preflight and exact runtime checks.
2. Full standalone smoke validation using the embedded `abca5c6` validator.
3. Validated atomic model-stage copy into the work directory, preserving the
   archived model manifest and cache files.
4. Fresh final run at `2ed304f`, including two-build determinism, FVI selection,
   bounded smoke, all 96 cells, packaging and integrated package validation.
5. Independent final numerical comparison with the historical expected results.

The final runner already invokes the same full checker as the standalone CLI;
the controller does not duplicate that expensive validation.

## Results and interruptions

Each receipt subdirectory contains atomic `execution.json`, `heartbeat.json`,
and individual stdout/stderr logs. `execution.json` records stage return codes,
hashes, paths, exact commands, identities and acceptance evidence. Only terminal
`PASSED` with `scientific_acceptance: true` means all scientific gates completed.
`RUNNING`, an old heartbeat, `aggregate.json`, or a finished cell directory is
insufficient. SIGTERM/SIGINT terminate the child process group and record an
interruption when the host permits graceful shutdown. An uncatchable host loss
leaves an incomplete receipt rather than a false success.

Existing output is never overwritten or treated as an invented resume. After an
interruption with a final output already present, retain the directory and logs
and assess a genuine runner `--resume` separately. This controller intentionally
does not automate that recovery. A model stage already completed in the work
directory may be reused only after its full identity and bytes validate.

The original smoke failure and interrupted local validations remain historical
evidence. A successful new standalone validation closes package acceptance; it
does not retroactively convert those old process receipts into successful runs.
