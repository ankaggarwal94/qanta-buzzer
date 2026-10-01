# Prospective v5 constrained-response pilot

The October 1, 2026 v3 and v4 development runs completed 531 responses but
stopped at the frozen interface gate. This v5 candidate changes the generation
interface before any main-set responses are observed. It is a new exploratory
pilot, not a reproduction of Jane's unrecovered historical run.

## Frozen inputs and scope

The v4 public inputs remain byte-identical, with public input identity
`d7d2168092ca6de6f89381f62581426a7405881bae9678a6348cc7334a4171bd`.
The transport archive SHA256 is
`43b5568bdbd927487e6850dcc75676f8d151618f07143aab31cba8195f220490`.
Questions, prefix texts, distractor menus, positions, prompts, job IDs, model
revisions, seed, batching, split roles, and choices-only controls are unchanged.
Main has 200 questions: 50 calibration, 50 selection, and 100 test. Development
has 12 disjoint questions and 177 trajectory jobs per model. Main has 2,856
trajectory and 400 choices-only jobs per model. Successful execution therefore
produces 6,866 responses across the two models.

The two checkpoints remain Qwen2.5-3B-Instruct at
`aa8e72537993ba99e69dfaafa59ed015b17504d1` and Qwen2.5-7B-Instruct at
`a09a35458c702b33eeacc393d103063234e8bc28`. Inference uses one L40S, BF16,
greedy decoding, batch eight, and fresh contexts for each job.

## Interface intervention

The backend masks inadmissible next tokens using pinned lm-format-enforcer
0.11.3 and interegular 0.3.3. Separate MC and OE grammars require exact JSON
keys, consistent answered/abstained fields, confidence within [0,1], and legal
MC option IDs. OE answer length is bounded by the 160-token completion cap,
with no additional character limit. A cached character-set adapter preserves
the regex language while avoiding repeated scans of long Unicode alphabets.
The exact regexes,
hashes, constraint source identity, package versions, and termination settings
are recorded with each trace. The output token ceiling increases from 96 to
160. These constraints can change model responses and confidence values and
must be reported as part of the evaluated system.

The strict parser is unchanged. There is no posthoc JSON repair or retry.
Responses that reach the token limit or violate the grammar are retained
in the batch checkpoint before execution stops without a complete trace. Format compliance is an execution measure, not correctness.
Confidence remains an uncalibrated model self-report, not normalized option
probability. No gold answerlines or evaluator labels enter the GPU image.

Both models must pass the original development interface criteria: at least
95% schema-valid responses in each format, zero illegal answered MC IDs, and
zero copied OE placeholders. Valid abstentions count as schema valid. The
gate does not use answer correctness or policy coverage. A subsequent measured
throughput gate requires the complete main grid and controls to fit the
remaining allocation, retaining the original threefold inference margin and
900-second loading reserve per model.

## Cumulative authorization and execution

The user's original total compute ceiling remains $10. The prior two host
durations at the configured resource rate, plus two full $2 contingency
reserves, total $4.41650600135919124 and are rounded up to a $4.43 debit.
The v5 ceiling is $5.57, including another $2 reserve, with at most 5,584
seconds at $0.00063924/second. These are bounded resource estimates, not
verified provider invoices. The runner binds both previous receipt hashes,
run IDs, and source commits and rejects an altered budget plan.

Execution uses the existing GitHub Actions secrets on the isolated branch
`ops/jane-modal-pilot-20261001`. The exact launch commit must have message
`ops: launch frozen Jane Modal v5 constrained candidate 20261001` and include the changed
workflow. The v5 app and volume are new, create-once identities. The app is
attached, has no automatic retry, runs one container, and has both host and
remote deadlines. Inputs and source identities are verified before inference.

## Analysis and interpretation

Prompt/backend selection stops before main generation. Calibration uses only
calibration questions; thresholds use only selection questions; test outcomes
are reported once. The existing empirical 10% error-budget procedure and
question-level bootstrap are retained. No positive coverage is guaranteed.
If an arm has no answered calibration prefixes, its calibration and transferred
policy are unestimable and must be reported explicitly.

Canonical OE matches can be accepted automatically. Other responses require
answerline adjudication, with reviewer identity and unresolved cases retained.
Any AI-assisted adjudication is provisional, and correctness conclusions must
preserve that limitation. Choices-only results are analyzed separately from
question-conditioned trajectories.

The convenience sample, unreviewed sentence-end prefix proxies, lack of science
development examples, and unreviewed distractor equivalence/hardness remain
limitations. In particular, only 96/200 main questions and 44/100 test questions
have a retained prefix at or below 20% of the question. These denominators must
not be silently expanded. The run's evidence scope remains engineering_smoke;
it supports exploratory conclusions about this frozen configuration.
