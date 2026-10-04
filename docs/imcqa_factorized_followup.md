# Numerical diagnosis and separated answer/stopping policies

This exploratory follow-up begins from the validated partial protocol run launched
at `dc02ad3fedee51eee63913a0bd9ddca0b6678348`. That run produced 4,032 new
Qwen 7B contexts and reused 1,200 prior contexts. Qwen 3B stopped before any new
production rows because two diagnostic contexts exceeded the unchanged raw-logit
tolerance. Its failed receipts and raw comparisons remain evidence of that run.

## CPU comparisons using saved scores

The fixed analysis plan is `configs/imcqa_factorized_cpu_analysis.json`, committed
before fitting at `a2f75098c777163f68bc7638a1d911a919848b7d`. It uses only the
validated Qwen 7B factorial scores: 40 previously inspected questions, two menus,
five prefixes, four cyclic answer orders, and three prompt conditions. The 20
calibration questions determine calibration fits and policy choices. The 20
selection questions provide descriptive paired evaluation.

For each prompt/menu, fit the declared regularized monotone logistic correctness
calibrator. Compare fixed rounds, always PASS, a myopic positive-value rule, and
a confidence threshold chosen on calibration questions. Replay each stopping
schedule with plain, forced-game, and WAIT-prompt answer candidates on the same
question/menu/order trajectory. Changing only the candidate stream under an
identical stopping schedule measures the answer component directly.

A separate fixed variant averages the four plain-prompt probability vectors
after restoring canonical candidate identities. It uses the same declared
calibration and policy procedures. Compare this complete ensemble pipeline with
the order-averaged plain pipeline after aggregating within question. Its invariance
to the four cyclic rotations follows from the closed ensemble construction; it is
not evidence of invariance to all 24 answer permutations.

All bootstrap intervals resample questions with their associated menus, rounds,
and rotations. Fits remain fixed in the bootstrap. Twenty questions per split,
previous inspection, fitting uncertainty, and multiple exploratory comparisons
limit inference. Positive value relative to PASS does not establish optimal
stopping or estimate the value of future information.

## Qwen 3B numerical shape diagnostic

Use the original ten diagnostic contexts, including both failing contexts, plus
ten deterministically selected prior-overlap contexts. Compare unpadded single
references, exact single replay and reversed order, uncached and cached batches
of 2, 4, and 8, and reproduction of the original batch-32 diagnostic configuration.
Record all candidate-path outcomes without converting failed checks into passes.

Acceptance remains raw-logit `atol=0.001, rtol=0.00001`, action and conditional
candidate probability difference at most `0.001`, and zero decision changes.
Fresh singles must reproduce their own replay and the prior overlap evidence.
A passing batch path is only diagnostic-compatible. Any production recovery
requires a separate frozen execution plan and full production validation.

The diagnostic has 200 logical context evaluations and 117 underlying forward
calls, with no production scores. One worker has a 360-second timeout and
300-second internal deadline. Its resource reservation is $0.40616880, including
startup, teardown, CPU staging, and a $0.10 contingency.

## Binary-controller pilot

The proposal at each real state is exactly the saved Qwen 7B plain-MCQA answer at
rotation zero. The controller receives that fixed proposal, the revealed question,
and the answer menu. It selects only SUBMIT or DEFER. DEFER means WAIT before the
last round and PASS at the last round. Future states can supply a new proposal.
No real gold answer or candidate softmax confidence enters the controller prompt.

Use two reversed X/Y action encodings, for 40 questions × two menus × five rounds
× two encodings = 800 real contexts. Defer on an exact score tie and report ties.
The 32 synthetic contexts comprise 16 certain-correct cases and four each of:
certain-incorrect now with a guaranteed-correct future proposal; certain-incorrect
terminal; uniform uncertainty now with a guaranteed-correct future proposal; and
uniform terminal uncertainty. Each underlying state has both encodings. All
synthetic outcomes remain scientific results, including incorrect actions.

Primary comparisons measure encoding sensitivity and episodic reward. Compare
the new controller with the original WAIT stopping schedule while holding the
saved plain answer proposals identical. This comparison changes the complete
controller interface and instructions, so it does not isolate a pure token effect.
Keep all numerical gates, source/weight/input identities, exact coverage, and
active-only replay checks before accepting scientific results.

One Qwen 7B worker has a 900-second timeout and 780-second internal deadline.
Its reservation is $0.75135840. Both initial GPU studies together reserve
$1.15752720 under the follow-up's $4 resource-estimate bound. Any additional
production recovery must fit the remaining bound and receive independent review
before launch. There are no automatic retries or precision fallbacks.

Rates were checked on 2026-10-04 UTC at https://modal.com/pricing/: L40S
$0.000542/second, two physical CPU cores at $0.0000131/core/second, and 32 GiB at
$0.00000222/GiB/second, totaling $0.00063924/worker-second. Reservations and
measured-window estimates are not verified invoices or account-wide spending limits.

No fresh confirmatory questions, full generated-response sampling, model-weight
changes, or reveal-boundary changes are included in this follow-up.
