# Frozen-policy transfer after the factorized follow-up

The next question is whether the promising 7B plain-answer plus external-stop
policy transfers beyond the repeatedly inspected 20-question selection sample.
This study changes the questions and retains the exact model, prompts, scoring
grammar, policy coefficients, and reveal schedule. It does not train a controller.

Select 100 questions from the original selection split. Exclude every one of the
200 questions used in the first WAIT pilot and their answer-component group IDs.
Rank the remaining qids by SHA256 of `imcqa-frozen-transfer-20261004|<qid>`, breaking
hash ties lexically. Walk that order, excluding full-question near duplicates of
any prior question or previously accepted transfer question. The near-duplicate
test applies Unicode NFKC and casefold, extracts Unicode alphanumeric runs
(excluding underscores), forms distinct contiguous token 5-grams, and rejects
Jaccard similarity at least 0.8. Questions shorter than five tokens contribute
their complete token tuple. The inclusive cutoff uses exact integer comparison.
Record every skipped qid and matching reference. Selection reads no gold labels
or model outcomes. The original corpus was already evaluated in earlier studies,
so this is unused protocol-development validation, not a pristine final holdout.

Each question has the two frozen menus, prefixes p2/p4/p6/p8/p10, four cyclic
candidate rotations, and plain/WAIT prompts: 8,000 new Qwen 7B score contexts.
The prompt renderer and mapping are imported unchanged from the prior protocol.
WAIT remains E; there are no new reward instructions or action-label variants.
Correct rewards are 1.0/0.8/0.6/0.4/0.2, wrong answers -1, PASS zero. The
single-token action scores remain conditional on the fixed JSON answer prefix.

The correctness calibrator is sigmoid(intercept + slope × logit(max candidate
probability)), clipping its feature to [1e-6, 0.999999]. Independent-menu values
are intercept -0.703106162024301, slope 0.45313818462033534, threshold 0.60,
and fixed round 1. Same-category values are intercept -0.6355621259799796,
slope 0.42493943060428674, threshold 0.85, and fixed round 2. Submit the first
candidate crossing the frozen threshold; otherwise PASS. No refitting is allowed.
The threshold need not be a positive expected-value threshold at every round;
we preserve this existing policy to measure its transfer honestly. Always-PASS and final-round plain answering are descriptive baselines; neither
replaces the primary policy after outcomes.

Average rotations within question. Four prespecified primary reward differences
compare the frozen external policy with native WAIT and with the frozen fixed
baseline, separately for each menu. Use 20,000 question-cluster bootstrap samples,
seed 1, with two-sided 98.75% intervals for the four-comparison Bonferroni family.
PASS and other metrics use descriptive 95% intervals. Fits remain fixed. A positive
result against native WAIT demonstrates improvement under these protocols; a
separate positive result against the fixed baseline is needed to attribute value
to adaptive stopping beyond that baseline. Inconclusive intervals stay inconclusive.

Frozen input bytes, selection manifest, policies, and analysis plan are hashed
before inference. The inference package uses an explicit field allowlist and
contains no real gold labels. The selected evaluator is separate. Validation
rerenders all 8,000 prompts and checks identity, exact coverage, option mappings,
cumulative prefixes, exclusions, fitted-question separation, and source hashes.
Questions remain the statistical unit: rotations and prefixes are dependent.
