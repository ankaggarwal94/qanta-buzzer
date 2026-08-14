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
Packaged Markdown, LaTeX, and the canonical PNG must be byte-identical to a fresh
render from those validated aggregate fields and `resource_summary.json`, whose
content ID is part of the run-spec identity graph;
missing or additional report/figure paths are rejected even when `SHA256SUMS` was
regenerated. The figure uses a font-free, pure-Python indexed-PNG renderer with
canonical embedded plot data, so validation does not depend on Matplotlib state,
fonts, FreeType, or platform rendering.
Execution IDs, `cached` flags, and source-execution fields in prerequisite evidence
are unsigned trusted-producer assertions. The standalone checker establishes
internal package consistency; it does not authenticate Modal provenance against a
hostile package author. See the explicit threat boundary in
`docs/stopdff_v5/REPRODUCTION.md`.
