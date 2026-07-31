# StopDFF v5 acceptance contract

A run is accepted only when
`scripts/validate_stopdff_bucketed_sweep.py validate` succeeds against the
standalone recomputation in `scripts/stopdff_v5/checker.py`. Final packages must
also pass the safe-path, complete-checksum, report, evidence-ledger, and
prerequisite-binding checks selected by `--require-package`.

`scripts/validate_stopdff_bucketed_sweep.py self-test` is the normative
negative-mutation gate. The commands, supported backends, and required input
artifacts are documented in `docs/stopdff_v5/REPRODUCTION.md`.

Serialized verdict fields are never authoritative: acceptance recomputes cell,
family, gate-override, and release outcomes from identity-bound adapter rows.
