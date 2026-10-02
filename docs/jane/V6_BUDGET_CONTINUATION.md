# v6 continuation of the verified constrained pilot

This attempt retains the exact v5 generation backend, output grammars, strict
parser, model revisions, public input bytes, seed, batch size, and 160-token
completion limit. The only runtime changes are the cumulative budget ledger
and fresh create-once execution identities. Scientific and analysis scope are
unchanged from `V5_CONSTRAINED_PILOT.md`.

## Evidence available before launch

The v5 run at source `cadc3643b00040ce3960592cb1f4c0a6ed684de4` completed
177 development responses per model. Both models passed the original
format-only gate: 118/118 MC and 59/59 OE responses were valid for each model,
with no illegal MC IDs or copied OE placeholders. Independent replay verified
354 responses, 155 aggregate checks and 7,788 token/parser/constraint checks.

Main generation never started. The original threefold throughput projection
was 7,910.339 seconds, exceeding the 5,107.916 seconds remaining in v5 before
its shutdown margin. All main response and evaluator labels remain unseen at
this freeze. The v5 Actions artifact is from run `36942800586`, artifact
`11201361246`, ZIP SHA256
`7a0a04444bea97e751adaa94976d1dd812de4eeb712da227a0dd59591141fb22`.

## Prospective budget accounting correction

The user's total authorization remains $10. Earlier attempts reserved $2
separately for each session. Those reservations were contingencies, not
observed charges. Each of the three apps is confirmed completed in its Actions
logs. Their host receipts account for 390.718294956, 260.846064845, and
363.81886268999995 seconds. Pricing the entire host interval at the configured
GPU plus capped CPU and RAM rate gives a cumulative estimate of
$0.649073571145, rounded upward to a $0.66 prior debit.

v6 retains one $2 contingency for the entire sequence, rather than repeatedly
reserving the same category of ancillary costs. A new $9.34 envelope includes
that shared contingency and at most 11,482 seconds at $0.00063924/second.
The maximum modeled cumulative total is therefore $9.99975368. This is
resource-based accounting with a contingency; no provider invoice has been
verified and no account-wide spending limit is asserted.

The ledger binds all three prior receipt hashes, source commits, completed
app log evidence, and host durations. It is validated before cloud actions
and again in the remote function. The threefold throughput multiplier, total
1,800-second loading allowance, 120-second shutdown margin, attached execution,
one-container cap, no retries, and host/remote deadlines are unchanged. Both
development gates are rerun before the full main and choices-only grid.

## Execution and interpretation

The new app and volume are `jane-mcq-pilot-20261001-v6`. The launch is restricted
to the isolated branch, a single workflow attempt, and the exact commit message
`ops: launch frozen Jane Modal v6 global-budget candidate 20261001`.
The separate inspection workflow only reads existing output counts and receipts.

No prompt or decoding choice was selected on main outcomes. The scratch
cached-logit-mask performance experiment was not incorporated. Analysis uses
the original 50 calibration / 50 selection / 100 test split and empirical
10% error-budget procedure. The convenience corpus, sentence-end proxies,
unreviewed distractors and answerline adjudication limitations remain. This is
an exploratory engineering pilot, not a recovered historical reproduction.
