# v7: retain capped failures and finish the 7B model

V6 completed both development gates, the 3B model's 2,856 main responses, and
its 400 choices-only controls. The 7B main phase stopped after its 150th batch
(1,200 rows): one open-ended response repeated digits until the unchanged
160-token cap, leaving an incomplete JSON string with no EOS. Its raw bytes,
tokens and invalid parse are preserved in the checkpoint. The other seven
batch members were valid. This is an observed generation failure, not a
correct answer or a successful abstention.

The v6 source is `6e3bf3d84cb141ede9f386d256bfd36eaff7d961`; its Actions run
is `36945283402`. Its failed run status remains unchanged. Independent replay
validated all 4,810 retained rows and the boundary of the failure, including
complete 3B main and control phases. No complete 7B main trace exists from v6.

## Narrow operational change

The backend now records and continues past a precisely identified capped
failure: output reaches the configured token limit without EOS, the strict
parser reports invalid, and the raw text is a valid incomplete prefix of the
same frozen grammar. The row retains null answer/confidence, invalid status,
exact raw output and tokens, and an explicit failure-kind field. Unexpected
grammar, parser, provenance or source errors still abort after checkpointing.
No response is repaired, retried until valid, shortened, or given a larger cap.

The next-token masks, grammars, prompts, model revisions, BF16 greedy generation,
seed, batch size, context cap and 160-token output limit are unchanged. Main
format failures remain unconditional errors for accuracy and provide no
confidence for a stopping decision under the original evaluation code.

V7 reruns the complete 7B development, main and choices-only phases. The
already completed 3B phases are reused from v6. This is an explicit model-level
operational retry, not a selective per-response retry; all original 7B partial
rows remain in the evidence. The repeated first 1,200 main responses will be
compared with the original records, excluding timing fields. Any differences
must be reported rather than silently choosing the better response.

## Scope and provenance

The paired analysis cohort will combine v6 3B with v7 7B if the latter completes.
Every phase retains its actual source, completion receipt, checkpoint, model
and input identity. A separate assembly certificate must bind this mixed-run
cohort; no provider receipt is rewritten to claim that v6 completed.

The public input identity remains
`d7d2168092ca6de6f89381f62581426a7405881bae9678a6348cc7334a4171bd`.
All 200 main questions, the 50/50/100 calibration/selection/test split, menus,
prefixes, controls and scoring rules remain frozen. The original development
interface thresholds and threefold inference throughput margin are retained.
The one-model retry reserves 900 seconds for its later loading calls.

3B answerline adjudication had begun before this operational recovery; the
single 7B capped response was inspected to diagnose the abort. The intervention
does not use correctness, confidence, selected thresholds or test performance
to alter generation. These facts and the exploratory engineering-sample
limitations must remain visible in reporting. AI adjudication is provisional
and has not been validated by a human.

## Budget

Four completed host sessions give a cumulative configured-resource estimate of
$1.684891692343, rounded upward to a $1.69 debit. One global $2 contingency
remains. The new envelope is $8.31, including this contingency, with at most
9,871 seconds of additional allocation. The user's total ceiling remains $10.
These are resource estimates, not verified provider invoices or an account-wide
spending limit. The v6 ledger records the observed uncaught-exception stop and
host finalization; it does not invent an `App completed` acknowledgement.

Execution uses a fresh create-once v7 app and volume, a single attached
container, no automatic retries, and the existing layered deadlines and
cancellation path. Launch requires the exact reviewed commit message
`ops: launch frozen Jane Modal v7 qwen7b recovery 20261001`.
