# Frozen ACL 5000-question experiment

This branch contains the prospective expanded inference run requested on 2026-10-02. It does not establish completion or empirical results: those require provider launch IDs, per-model receipts, and hash-verified output coverage.

The two pinned Qwen2.5 Instruct models (3B and 7B) each receive 150,000 main jobs and 10,000 choices-only controls. The main dataset has 1,000 calibration,1,000 selection, and 3,000 test questions. The three main conditions are open-ended, independent-pool multiple choice, and same-category-pool multiple choice at ten frozen word endpoints. The optional type-matched menus and 2026 holdout are excluded.

Frozen identities:

* Evaluator dataset SHA256: d16d8e611965fba3829f01cda936145743b7d46187030e2138caf52068aa9b62
* Main public jobs SHA256: 9bfeaf2d86116ced0e8c55c3c390d3be12050ad38820b4e02c6ed684cc8786bf
* Choices-only jobs SHA256: 9db13301d928cd31dc54c97f0c5cfd88b9bc25774ceb56027c032c52e4afb043

Generation uses the previously tested backend without changes: L40S CUDA, bfloat16, eager attention, greedy decoding, batch size 8, seed 1, input cap 2048 tokens, output cap 160 tokens, and the frozen JSON grammar. Every prompt has a fresh context. Confidence is an uncalibrated model self-report. Recognized incomplete JSON at the token cap is retained as an invalid outcome; it is never repaired or retried for a better answer.

The new run has a separate $80 allocation-estimate ceiling. Each of the two model calls is bounded to 16 hours; maximum concurrency is two GPUs. Current published base rates checked 2026-10-02 give $0.00063924 per second for one L40S, two CPU cores, and 32 GiB memory. Two full 16-hour allocations plus a $6 contingency reserve total $79.640448. This is an operational estimate and reservation, not a verified invoice or a change to an account-level billing limit. No region premium is requested. The historical 200-question pilot ledger remains unchanged.

The first production shard supplies the throughput check and remains part of the experiment. Shards preserve original ordering and batch membership. The runner records public input, source, model, prompt, and generation identities, raw generations, and create-once completion receipts. Incomplete or ambiguous execution is surfaced explicitly. Completed shards must be verified before reuse. Automatic retry is disabled.

Only public prompts and execution code are sent to model containers. Gold answerlines, grading aliases, and evaluator data remain outside inference. Prompt compression is lossless transport only; restored bytes must match both frozen hashes before submission.

The source questions originate from the official QBReader 2026-07-18 backup, downloaded 2026-10-02. Question authors and tournament hosts retain ownership. Existing noncommercial source-use conditions and provenance accompany the separately frozen dataset release.

Analysis must preserve unmatched open-ended answers as unresolved until adjudicated. Deterministic exact-match analyses describe an explicit proxy target. Dataset semantic identity and distractor review remain incomplete; inference completion alone does not resolve those limitations.
