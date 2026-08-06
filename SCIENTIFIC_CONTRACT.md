# StopDFF v5 scientific contract

This file is the stable prose index for the executable scientific contract.
Canonical constants and the complete profile identity are defined by
`scripts/stopdff_v5/profile.py`; the public JSON constraints are in
`schemas/stopdff_scientific_profile.schema.json`, `schemas/stopdff_calibrator.schema.json`,
`schemas/stopdff_continuation.schema.json`, and
`schemas/stopdff_gate_policy.schema.json`.

The normative algorithmic implementation is:

- `rewards.py` and `policy.py` for rewards and three-action Bellman decisions;
- `calibrators.py` and `continuation.py` for shared calibration and continuation;
- `fvi.py` and `fvi_study.py` for convergence and candidate selection;
- `bootstrap.py`, `verdicts.py`, and `cellcompute.py` for paired statistics and
  verdicts.

`docs/stopdff_v5/REPRODUCTION.md` defines the supported execution boundary. If
prose and an emitted identity disagree, the checked-in schema plus the canonical
identity produced by `profile_static_identity()` is authoritative.
