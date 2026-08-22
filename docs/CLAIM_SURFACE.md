# CS321M Claim Surface

> **Scope note (R-038):** stopping-shift evidence in this repository concerns
> constructed QA reference trajectories — a constructed-reference sensitivity
> diagnostic. It does not assert observed open-ended decision preservation;
> the authoritative claim ledger lives under `reproducibility/colm_aims_2026/`.


The final CS321M code package supports a narrow benchmark-translation audit.

## Supported headline claims

The current committed artifacts support:

1. **CSLI PASS**: choices-only excess is small under the implemented panel.
2. **Prefix calibration PASS**: prefix-wise ECE is below the frozen threshold
   under the implemented SBERT/Platt diagnostic.
3. **Diagnostic StopDFF WARN**: the implemented myopic-threshold StopDFF is a
   diagnostic null because of ceiling effect / unreachable buckets.
4. **Overall WARN**: the audit ran on a retained MC subset and StopDFF is not
   confirmatory.

## Claims that require stronger evidence

Do not claim these from the current artifacts:

- MC QANTA preserves open-ended quizbowl decision boundaries.
- The proxy is construct-equivalent to open-ended QA.
- StopDFF validates optimal stopping preservation.
- PPO solves quizbowl or beats Sequential Bayes.
- Model rankings are stable across open-ended and MC formats.
- Dynamic distractors are superior to fixed distractors.
- The audit covers the full raw fresh split without retention qualification.

## Required wording

Use:

> In a retained-subset pilot, the MC proxy did not show substantial
> choices-only leakage and was prefix-calibrated under our diagnostic, but
> the stopping-time audit was uninformative under the preregistered myopic
> threshold. The overall translation verdict is WARN.

Do not use:

> MC QANTA preserves open-ended quizbowl decision boundaries.
