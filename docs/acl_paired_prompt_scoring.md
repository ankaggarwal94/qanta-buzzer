# Paired menu-only prompt recomputation

Authorized October 3, 2026. Repeated full-response sampling is deferred.
Recompute the original abstention-permitted prompt and a minimally edited
forced-choice prompt for each frozen menu: 5,000 questions, two menu types,
two prompt conditions, and two pinned Qwen models, totaling 40,000 score rows.

The original prompt comes from the exact public choices-only package. Its
abstention instruction is replaced with an explicit requirement to choose an
option. The menu, remaining instructions, chat template and assistant answer
prefix stay fixed. Both conditions score the next A/B/C/D token after
`{"answer":"`. All four logits and their conditional softmax are retained.
This identifies instruction sensitivity within that conditional-token scoring
protocol. It does not measure full-response frequencies, abstention probability,
calibrated correctness confidence, or establish independence of irrelevant
alternatives. Gold labels do not reach inference.

Primary scores use FP32 because the earlier BF16 batch/single check showed
material sensitivity. Load the same hash-verified pinned weights in BF16 and
promote them exactly on the GPU. Disable TF32; retain eager attention and
explicit attention-mask-derived position IDs. Use one global padding width and
fixed batch size across both prompt conditions. No fallback changes precision.
Before production, retain numerical vectors for FP32 unpadded singles, fixed
batches, exact replay and permutation. Require the predeclared raw-logit
tolerance `atol=1e-3, rtol=1e-5`; near-tied rankings may differ inside this
tolerance. BF16 subset vectors are diagnostic only.

The first 256 score rows remain in the output and serve as the throughput
benchmark. Continue only if the projected remaining work, multiplied by 1.5,
fits the internal deadline with the shutdown reserve. A numerical failure or
budget stop preserves partial evidence and never silently starts another run.

Reuse the existing verified public-input/model cache. Allocate one L40S worker
per model, with two physical CPU cores and 32 GiB host memory each. External
timeouts are 850 seconds for 3B and 1,450 seconds for 7B; internal deadlines are
780 and 1,350 seconds. At $0.00063924 per worker-second, both maximum allocations,
90 seconds of startup and two seconds of scaledown per worker, and a $0.40
contingency reserve total $1.98787216. This is a reserved compute estimate under
the $2 ceiling, not a verified invoice or account-wide billing control.
Create-once run and model claims prohibit automatic retries and replayed work.

The guarded workflow binds the exact source commit. Original raw evidence is
preserved separately. Analysis checks hashes, coverage, pairing and numerical
evidence, then reports conditional probability changes by model and menu type.
Question-level resampling keeps both menus and both prompt conditions together.
Held-out test questions are reported separately from all-question descriptions.

## Reviewed efficiency recovery

The initial FP32 run passed its numerical gates but stopped at the throughput
gate after 256 rows per model. Measured batch32 throughput was 28.79 rows/s
(3B) and 12.92 rows/s (7B), with allocation receipts of 75.417 and 94.419
seconds. These partial results remain intact and are not complete-run findings.

The one reviewed recovery reuses the exact token prefix shared by each prompt
pair. Prefill the shared prefix once, duplicate its FP32 key/value cache, and
score the original and forced suffixes together. Use 128 score rows per batch,
explicit logical position IDs, and full prefix-plus-suffix attention masks.
No model weights, prompt text, answer prefix, arithmetic precision, or numerical
tolerance changes. Additional cached-versus-ordinary FP32 checks, cached replay,
and paired permutation checks must pass before production. Tensor memory peaks
and the exact prefix/suffix layout are retained.

The initial 128-menu sample shares 113.95 of 150.95/142.95 original/forced
tokens on average, suggesting approximately 39% less duplicated token work.
This is a work estimate, not a measured speedup. The new fixed-shape benchmark
must justify completion. Its forecast uses the larger of a 1.2-times-mean
projection and a p95 batch-time projection, with shutdown reserve. The reduced
forecast multiplier does not change hard allocation limits or numerical gates.

Recovery GPU timeouts are 650 seconds for 3B and 1,350 seconds for 7B, with
internal deadlines 50 seconds earlier. Including the two completed initial
allocations, 90-second startup and two-second scaledown allowances for all four
workers, and $0.35 contingency, the cumulative reserved estimate is $1.97228642.
Recovery requires the exact four initial receipt hashes and uses new atomic
run/model claims. There is no generic retry facility.
