# Development pilot of explicit WAIT decisions

This protocol is frozen before new model outcomes are inspected. It adds an
explicit sequential action to the existing ACL expansion. It does not repeat
the 320,000 generated responses, use the 3,000-question test partition, or claim
that menu-only scores already measure incremental answering.

The original expansion provides ten independently prompted question prefixes
per question. The new pilot uses five of those exact prefixes: `p2`, `p4`,
`p6`, `p8`, and `p10`. Their nominal fractions are 20%, 40%, 60%, 80%, and 100%;
actual word fractions remain as recorded in the frozen inputs. These are word
endpoints, not newly annotated clue boundaries. More revealed text is not a
claim of empirically monotone difficulty.

The authoritative machine-readable design is
`configs/imcqa_wait_pilot.json`. A launch control must bind its hash, exact
source commit, public job package, question-selection manifest, pinned model
files, tokenizer, numerical settings, and budget. A design document by itself
is not an execution receipt.

## Questions and conditions

Select 100 questions from `calibration` and 100 from `selection`; exclude every
test question. Within each split, allocate category quotas proportionally by
largest remainder, resolving tied remainders by category name. Within each
category, order questions by the SHA256 of `1|split|category|qid`, followed by
qid for an exact hash tie. Small categories may receive zero quota. Category
stratification does not imply equal category representation.

The category map is prepared on CPU from the hash-verified evaluator metadata
and published without gold answers, answerlines, or correctness fields. Bind
its hash and the realized per-category counts in the selection manifest. Gold
labels remain on the evaluator side throughout inference. Select a nested
diagnostic subset of 20 questions per split using the same proportional quota
rule and a `diagnostic|` hash-domain prefix. Both models receive identical
questions, menus, and conditions.

| Condition | Questions | Contexts per model | Purpose |
|---|---:|---:|---|
| `wait` | 200 | 2,000 | Primary A-E sequential decision policy |
| `forced` | 200 | 2,000 | Matched A-D answer-only control |
| `questionless` | 40 | 400 | Same round, rewards, and menu with question text omitted |
| `rotation` | Same 40 | 400 | Primary prompt with one cyclic rotation of A-D candidates |
| Total | 200 unique | 4,800 | 9,600 production contexts across the two models |

Each question has five rounds and two existing menu regimes,
`independent_pool` and `same_category_pool`. Candidate identities are fixed
across an episode. In the rotation condition canonical A appears at displayed
B, B at C, C at D, and D at A. E is unchanged. Predictions are mapped back to
candidate identities before comparing conditions. One rotation detects one
form of sensitivity; it does not characterize all permutations.

## State, prompt, and reward

Every prompt is a fresh context. It states the current round, remaining
rounds, all five correct-answer rewards, the wrong-answer penalty, the current
cumulative question text, and the four fixed candidate answers. Previous
model outputs are excluded. The model is told that additional rounds reveal
more of the same question and retain the same candidates.
For a later-round state the prompt states that preceding decisions were WAIT.
This describes how that state was reached without reproducing a conversation
history. A scored later state remains counterfactual for an episode that would
already have answered.

The correct-answer reward is `[1.0, 0.8, 0.6, 0.4, 0.2]` by round. Any incorrect
answer receives `-1.0` and terminates the episode. There is no discount and no
additional waiting penalty. WAIT yields zero immediate reward and advances
one round; the final episode payoff is the eventual answer reward or zero
for terminal PASS. These literal reward units are part of the prompt and may
not be rescaled after the design is frozen.

| Action | Rounds 1-4 | Round 5 |
|---|---|---|
| A-D | Commit to the displayed candidate and terminate | Commit and terminate |
| E | WAIT for exactly the next prefix | PASS, terminate with reward zero |

The final prompt explicitly says that no further question text or sixth round
exists. An E at round 5 is recorded as terminal PASS, never as a WAIT. The
forced control retains the same question, menu, round, reward information, and
shared game description, including the meaning of WAIT/PASS. It replaces only
the current action rule to require A-D now and disallow E on this decision.
It is a separately scored prompt, not a renormalization of primary logits.
The questionless condition replaces only question text with a fixed omission
marker. Exact rendered prompts and their hashes are retained for inspection.

The policy selects the highest-logit legal action token after the fresh
assistant-turn boundary and fixed `{"action":"` prefix. It uses A-E in the primary and diagnostic conditions
and A-D in the forced condition. It is a **constrained single-token policy**,
not full-response generation, full-vocabulary argmax, or repeated sampling.
Verify that each letter is a distinct, exact single-token continuation under
each pinned tokenizer. Break exact ties by the first legal letter in ABCDE
order and report ties. The legal-action softmax is not assumed to be calibrated
correctness confidence.

Retain A-E raw logits, the legal-action softmax, full-vocabulary logsumexp,
and the unconstrained top token ID and logit. This permits reporting how much
probability mass the legal actions receive without storing a full vocabulary
vector for every context. The forced condition still records E diagnostically
but excludes it from its legal action set.

## Sequential execution and controls

The primary behavior is the first A-D action along the five-round trajectory,
or terminal PASS after four WAITs and E at round 5. Collecting every stateless
round permits both this replay and counterfactual round-wise diagnostics from
one set of scores. Scores for rounds after an early commitment are explicitly
counterfactual; they are not counted as rounds the policy actually played.

Independently validate this equivalence on eight preselected questions, four
per development split, by scoring only active episodes and stopping them at
their first answer. Use both menus. The maximum extra work is 80 contexts per
model. Active-only prompts must match their saved all-round counterparts
byte-for-byte. Compare scores under the numerical tolerances and require
identical actions, terminal states, rewards, and visited rounds. This checks
the scheduler and replay implementation; it does not establish equivalence
for a model that receives previous conversation history.

The existing frozen gold determines correctness only during CPU evaluation.
Reference policies answer at each fixed round, answer only at the final round,
or always PASS. The answer-now controls retain the payoff-aware game prompt;
they are not standard unconditional MCQA. The original experiment's
full-question MCQA results remain a separate retrospective baseline.
A hindsight best forced-answer round or PASS is an explicitly
unattainable oracle bound for policies selecting among forced-control answers.
It need not bound the primary A-E policy, whose prompt can change the predicted
candidate. No fitted threshold determines the primary policy.
Any optional threshold fitted on calibration data is evaluated on selection
data and clearly separated from the unfitted primary policy.

## Numerical and execution gates

Use the same pinned Qwen 3B and 7B weights, promoted exactly from their stored
BF16 values to FP32. Disable TF32 and retain eager attention. No precision
fallback is allowed. Select at most 24 numerical contexts spanning every
condition and menu, shortest and longest lengths, all rounds, and both WAIT
and terminal PASS semantics. Retain the underlying diagnostic vectors.

Before production, compare unpadded singles with production batches, exact
replay, and batch permutation. If shared-prefix caching is used, compare it
directly with uncached FP32. Require raw-logit `atol=1e-3, rtol=1e-5`, maximum
legal-action probability difference at most `1e-3`, and no action argmax flips
on diagnostics. A numerical failure stops the run without relaxing a gate.
Inputs exceeding the declared implementation limit cause a preflight failure;
never truncate or remove them silently.

The cumulative reservation for this pilot is capped at $4. GPU timeout limits
are 2,400 seconds for 3B and 3,000 seconds for 7B, with internal deadlines of
2,250 and 2,850 seconds. At the verified planning rate of $0.00063924 per
worker-second, those limits plus 90 seconds startup and two seconds scaledown
per worker reserve $3.56951616. A conservative CPU allowance of $0.01723232 and
$0.40 contingency bring the reservation to $3.98674848. This is an allocation
estimate, not an invoice or account-wide billing control. Recheck the rate
before launch. All numerical and active-only replay passes fit within the
same worker deadlines; there is no unbudgeted validation worker.

A measured throughput forecast must cover all remaining production,
diagnostic, and shutdown work. Stop if the reservation cannot cover completion.
Do not automatically retry, change precision, or choose a smaller favorable
sample after examining outcomes. A smaller pilot requires an explicitly
versioned design frozen before its outcomes are inspected. Preserve partial
evidence and the stopping reason for any incomplete attempt.

## Reporting and interpretation

Report mean episodic reward, commitment rate, wrong-answer rate among
commitments, correct commitments per episode, terminal PASS rate, and the
stopping-round distribution. At zero commitments, conditional risk is
undefined. Report both development partitions separately, with any pooled
description labeled development-only. Use 2,000 paired question-bootstrap
resamples with seed 1, retaining both menus, conditions, and models together.
If source grouping places multiple sampled questions in the same leakage
component, additionally report a group-cluster sensitivity analysis.

Report the questionless and rotation comparisons only on their matched
40-question subset, with the reduced denominator visible. Include legal-action
probability mass, exact ties, numerical discrepancies, coverage, source hashes,
runtime, and estimated cost. No population risk guarantee, confirmatory test
claim, or broad model-ranking conclusion follows from this development pilot.

This is a cumulative-prefix instantiation of IMCQA. Nishant's illustrated
proposal uses reformulated questions and changing distractors. This pilot
does not test those additional interventions. Changing candidate sets can
reveal the persistent correct answer through menu intersections, so a future
dynamic-menu experiment needs a menu-history-only control. Cross-benchmark
chains also need verified answer identity and an independent difficulty-order
audit. The reward schedule here is fixed by design, not derived from IRT.
Two related models do not provide a broad respondent panel for item-level IRT
claims. BenchMarker is an audit framework in the proposal, not a generator of
question chains.
