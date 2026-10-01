# Frozen Jane Modal pilot

This is a new paired MC/OE pilot, not a reproduction of the unverified historical
200-question draft claims. The initial allocation is authorized up to $10. It
uses two pinned models sequentially in one L40S container, 2 physical CPU cores
and 32 GiB RAM. Both development-format gates must pass before any main jobs.

## Freeze and inspect before launching

The only remote inputs are `modal_pilot_data/public/{dev_jobs,main_jobs,
dev_choices_only,main_choices_only}.json`. Their exact allowlisted envelopes
exclude answerlines, accepted aliases, gold labels, evaluator datasets, and
source snapshots. The image includes only four files: the scripts package
initializer, the Modal wrapper, the GPU backend, and its exact JSON CPU parser.

```bash
python -m scripts.modal_jane_pilot plan \
  --public-dir modal_pilot_data/public \
  --source-commit EXACT_40_CHARACTER_REVIEWED_GIT_SHA --budget-usd 10
python -m pytest --noconftest tests/test_modal_jane_pilot.py -q
```

Neither command accesses Modal. Verify the source hashes, data identity, exact
model revisions, question-role separation, and frozen protocol before launch.

## Single paid launch

With normal configured Modal credentials, run `launch` instead of `plan` using
the same `python -m scripts.modal_jane_pilot` package invocation, and
add `--out results/jane_modal_gpu_20261001`. Install `modal==1.6.0` on the
submission host. The GPU image has independently pinned inference packages.
For a proxied submission environment, use `modal[api-proxy-support]==1.6.0` and
`truststore==0.10.4`. When installed, the wrapper injects the platform trust
store into its local SSL runtime before importing Modal. No trust-store
adaptation or submission credentials enter the GPU image. Launch verifies
that the four executing source files match the exact claimed Git commit.
Do not put credentials in arguments, image environment, artifacts, or output.

The Actions route restores the five-file public-only `public_inputs.zip`,
validates every byte against its included public manifest, and runs the three
pure boundary suites before exposing credentials. Its successful target is
6,866 model calls: 177 development, 2,856 main trajectory, and 400 main
choices-only calls per model. The 24 development choices-only jobs per model
are frozen as reserved controls and are not executed by this initial launch.

The Actions fallback uses existing `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` secrets
inside its submission step only. Stage all frozen source and public inputs on
`ops/jane-modal-pilot-20261001` first; add the narrow workflow last with the exact
message `ops: launch frozen Jane Modal pilot 20261001`. Adding that workflow is
the intentional paid launch. It checks the literal pushed commit, actor, branch,
and first run attempt, with read-only repository permissions and no checkout
credential persistence. It is not a PR trigger and is not merged into main.

The fresh named Volume `jane-mcq-pilot-20261001-initial` is created with
`allow_existing=False` before GPU work. Its existence forbids another initial
allocation, including a changed-source or manual repeat. A remote create-once
execution receipt committed before loading weights also refuses input replay.
No existing StopDFF data, volume, endpoint, or workflow is modified.

## Development gate and budget

Per model and per format, development responses require at least 95% exact
JSON schema compliance. All parsed MC answers must be legal A–D option IDs;
valid abstentions count as schema compliant. OE placeholder copies must be
zero. Answer confidence must be finite in [0,1]. The gate never uses gold
correctness, stopping coverage, or test outcomes. Failure stops without cloud
prompt repair or automatic scientific rerun.

Development choices-only menus are frozen and hash-verified for inspection,
but are reserved and not executed in this initial allocation. Successful
execution comprises 177 development, 2,856 main, and 400 main choices-only
calls per model, 6,866 calls across both models.

After both interface gates pass, the throughput gate projects all main and
choices-only calls at three times measured development inference time per job, plus
900 seconds of loading/setup reserve per model. It declines main inference
unless this fits the remaining allocation and shutdown reserve.
Inference time subtracts the separately recorded download/hash/model-load time
from the backend total; full phase wall time remains recorded for cost evidence.

Official pricing checked October 1, 2026:

| Resource | Maximum quantity | USD per second |
|---|---:|---:|
| L40S | 1 | 0.000542 |
| Physical CPU cores | 2 | 0.0000262 |
| RAM | 32 GiB | 0.00007104 |
| Total allocation | | 0.00063924 |

At most 12,000 seconds of attached session time gives $7.67088 of resource
allocation estimates. A further $2 is reserved for build, output storage,
network egress, and provider timing overhead, totaling $9.67088. Requests over
$10, nonfinite amounts, and insufficient reserve are rejected locally before
launch. CPU and RAM requests also specify limits, rather than permitting
unbounded resource bursting. Model caches are ephemeral and are not persisted
in the evidence volume. Downloaded evidence is limited to 256 MiB.

This is a conservative allocation estimate, not an account-wide provider spend
budget or an invoiced charge. Prices, actual builder allocation, cancellation
latency, and billing must be verified against the provider receipt. Existing
unrelated account jobs are outside this allocation. No new account-wide budget
settings are changed.

## Failure handling and evidence

The app runs attached with no warm containers, a two-second scaledown window,
`max_containers=1`, and `retries=0`. Modal can nevertheless reschedule crashed
containers independently of function retry configuration. The volume execution
claim, absolute deadline, POSIX watchdog, and host cancellation mitigate replay
and runaway allocation. These SDK behavior assumptions require a live canary;
pure unit tests do not prove provider durability or a final bill.

Before creating its allocation claim, the submission client verifies the token's
actual workspace is `ankaggarwal94`. Only the workspace name and ID enter the
receipt. The check uses the same low-level request as the pinned SDK's token
information command; this version-dependent interface fails closed if removed.

On errors, the host cancels the function with `terminate_containers=True`, exits
the ephemeral app context, and downloads committed partial checkpoints. The
backend flushes/fsyncs batches and the wrapper commits progress to the volume.
Checkpoint handles are closed before each commit, since the SDK can reload the
mounted volume during commit. The next batch opens the checkpoint for append.
Hard OOM, SIGKILL, or a provider timeout can interrupt a final commit, so a
partial checkpoint is not a completed trace. No failure is silently repaired.

Results contain source/input/model identities, raw responses and generated token
IDs in traces, checkpoints, development decisions, throughput projections,
completion records, and estimated allocation time. Actions uploads partial
results even on failure. Scientific grading runs locally from the separate
evaluator package after downloading evidence.

Primary provider documentation:

- [Pricing](https://modal.com/pricing)
- [Resource requests, limits, and billing](https://modal.com/docs/guide/resources)
- [Attached ephemeral apps](https://modal.com/docs/sdk/py/latest/App)
- [Timeout semantics](https://modal.com/docs/guide/timeouts)
- [Function and container retries](https://modal.com/docs/guide/retries)
- [Function-call cancellation](https://modal.com/docs/sdk/py/latest/FunctionCall)
- [Volume commits and create-once names](https://modal.com/docs/sdk/py/latest/Volume)
