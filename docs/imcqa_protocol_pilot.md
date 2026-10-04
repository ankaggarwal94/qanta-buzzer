# Development diagnosis of the IMCQA action protocol

This follow-up reuses the completed WAIT pilot and separates answer elicitation
from stopping behavior. The preceding CPU decomposition is completed before
any new inference. It uses the WAIT prompt's own candidate stream, rather than
the forced prompt's different predictions, for fixed-round and hindsight
references. Hindsight is unavailable to a deployed policy. A one-feature
correctness calibrator is fitted only on the original 100 calibration questions
per model/menu; its positive-EV-versus-PASS policy is explicitly myopic and does
not estimate continuation value. All of these analyses remain exploratory.

## Frozen question and prompt design

Use exactly the original 40 diagnostic questions: 20 calibration and 20
selection. Both sets have already been inspected. Preserve their exact frozen
five word-fraction prefixes and two fixed menus. No test questions or newly
generated questions are introduced. More scored contexts do not increase the
number of independent questions.

| Block | Design | Contexts per model |
|---|---|---:|
| Main factorial | 40 questions × 2 menus × 5 rounds × 4 rotations × 3 prompts | 4,800 |
| Action-label control | Same 40 questions × 2 menus × 5 rounds; swap A and E | 400 |
| Synthetic comprehension | 16 certain-answer, 8 guaranteed-future-answer, 8 terminal-ignorance cases | 32 |
| Total | 5,232 contexts, of which 1,200 are reused | 5,232 |

The three main prompts are plain MCQA, the original game with answering required
now, and the original game with WAIT/PASS available. Plain MCQA omits the game,
reward, and prior-WAIT instructions while preserving the question, menu, chat
template, FP32 arithmetic, and supplied `{"action":"` scoring boundary.
Ordinary game prompts are byte-identical to the completed pilot. This contrast
isolates the instruction block, not each sentence or reward component inside it.

Four cyclic rotations put every candidate in every A-D position. They do not
exhaust the 24 possible candidate orders. Predictions and distributions are
mapped back to candidate identities before comparisons. The separate label
control exchanges A and E: the original A candidate is labeled E, and A means
WAIT/PASS; candidate content order remains unchanged. Candidate probabilities
must normalize B, C, D, E in that control, and the legal action probabilities
normalize A-E. This is a targeted relabeling control, not proof of complete
action-label invariance.

## Reuse and numerical evidence

Reuse the original forced rotation 0 and WAIT rotations 0 and 1. This supplies
1,200 contexts per model; 4,032 new contexts per model remain, 8,064 across both
models. Keep original score rows untouched and merge through an explicit new
job to old score identity map. Bind the complete prior public input and score
file hashes, exact prompts, token IDs, output boundary, pinned model files,
precision, and runtime metadata. Fresh overlap checks cover the reused strata,
both menus, rounds, and length extremes, within 16 contexts per model.

Retain exact promoted FP32 weights, eager attention, and disabled TF32. Check
cached versus uncached, unpadded single versus batch, exact replay, pair
permutation, prior-run overlap, and production versus diagnostic scores. The
combined logit tolerance is `atol=0.001, rtol=0.00001`; maximum action and
candidate probability differences are 0.001, with no action or candidate
argmax changes on the declared checks. Active-only replay validates the new
WAIT rotation and label-swapped trajectories within the declared 80-context
reserve. Never relax a failed gate after observing outcomes.

## Comprehension and interpretation

Synthetic prompts state exact correctness probabilities. A known correct answer
should be selected immediately; uniform uncertainty followed by guaranteed
correctness next round should produce WAIT; uniform uncertainty at the terminal
round should produce PASS. Known-answer cases cover every candidate and rounds
1 and 5 under both E and A as WAIT labels. Uniform cases cover all four rotations
under both labels. The expected-action fixture file remains evaluator-side; expected labels are
not included in model prompts, and real-question gold never enters inference. Synthetic failures are reported scientific
findings, not grounds to discard cases or rewrite prompts during the run.

At the final round, answering beats PASS only if calibrated correctness exceeds
5/6, because correct earns +0.2, wrong earns -1, and PASS earns zero. Earlier
decisions additionally require a continuation-value comparison. Neither legal
action softmax nor candidate softmax is presumed to be calibrated correctness.

Report paired effects and question-cluster descriptive intervals by split,
preserving all conditions, rounds, rotations, and menu pairs in each resample.
Report order-averaged accuracy alongside order-specific spread. Separate real
question trajectories from synthetic checks, and visited states from later
counterfactual states. Do not claim an optimal policy, an IRT ability estimate,
general model rankings, or complete-response sampling frequencies.

## Execution bound

Use two create-once cached-model Modal workers, with no automatic retries. The
new run uses volume `imcqa-protocol-dev-20261004`; prior evidence and original
model caches are read for validation. Timeout limits are 2,400 seconds for 3B
and 3,000 seconds for 7B, including conservative internal stopping deadlines.
At the rechecked rate of $0.00063924 per worker-second for an L40S, two physical
CPU cores, and 32 GiB memory, startup/scaledown, CPU allowance, and contingency
reserve $3.98674848. This is a resource reservation estimate under a $4 bound,
not a billing statement. A measured throughput gate must cover remaining
production and validation work. Preserve partial evidence on failure.

Do not expand to fresh confirmatory questions, alter reveal boundaries, or add
model families during this diagnostic run. Those are subsequent decisions based
on the measured causes of the current protocol's behavior.
