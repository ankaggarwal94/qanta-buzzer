# COLM AIMS 2026 evidence contract

Fail-closed, create-once evidence flow for the **constructed QA reference**
sensitivity diagnostic backing the COLM 2026 AIMS workshop paper. Everything
here verifies constructed-reference artifacts only: the strict profile's
semantic layer pins `trajectory_source: constructed_reference` and
`does_not_support: actual_decision_preservation_or_format_effect`, so no
verdict from this namespace asserts anything about an observed open-ended
stopping policy. If an observed paired claim is ever intended, the only
sanctioned output is `observed_paired_claim=OBSERVED_PAIRED_STUDY_REQUIRED`.

This verifier is separate from the legacy `scripts/verify_audit_release.py`
(the StopDFF v5 audit-release checker), which stays byte-identical and is not
redefined by anything in this namespace.

Spec: `.correctless/specs/camera-ready-aims-evidence.md`.

## Exact invocation (both modes)

Run from the repo root (documented module-run form):

```bash
# Source-contract mode (ceiling: PASS_SOURCE_ONLY)
python -m reproducibility.colm_aims_2026.verify \
    --mode source \
    --tree <package>/tree \
    --receipts-dir <package>/receipts

# Release mode (requires independently anchored expectations)
python -m reproducibility.colm_aims_2026.verify \
    --mode release \
    --tree <package>/tree \
    --expectations <package>/expectations.json \
    --receipts-dir <package>/receipts
```

Direct-path invocation (`python reproducibility/colm_aims_2026/verify.py …`)
bootstraps the repo root onto `sys.path` and behaves identically. Unknown
flags and abbreviated flag forms are usage errors (`allow_abbrev=False`);
no flag, environment variable, or config key disables a release gate.

## Input layout

```
<package>/
  tree/                          <- the verified artifact tree
    profile.json                 <- strict constructed QA reference profile
    records.jsonl                <- retained non-reversible per-item records
    presentation_manifest.json   <- declared artifacts + per-file allowlist
  expectations.json              <- OUTSIDE the verified tree (required for
                                    release mode; anchored to a reviewed
                                    source commit and the frozen ledger)
  ledger.json                    <- frozen claim ledger the anchor pins
  rights.json                    <- rights inventory covering every tree file
  receipts/                      <- receipt output dir, outside the tree
```

The expectations file must resolve (symlink-free) outside the verified tree;
an artifact plus its own generated manifest reaches at most source-level
status. Evidence packages publish create-once into run-scoped directories;
canonical selection happens only via the ledger pointer, and retiring a
defective run is a ledger status change plus a new run directory.

## Verdict semantics

| Verdict            | Meaning                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `PASS_SOURCE_ONLY` | Source-contract ceiling: profile validation, typed ingress, pair/censoring identities, receipt emission. Does NOT certify release bindings, rights, anchored expectations, or any observed-decision claim. |
| `PASS_RELEASE`     | Release mode: every binding leg, rights row, manifest reconciliation, and ledger recomputation passed against independently anchored expectations. |
| `FAIL`             | At least one gate failed; failing legs list leg id, expected vs observed, and a remediation class (`ARTIFACT_DEFECT | MISSING_EXPECTATION | AUTHOR_DECISION_REQUIRED | EXTERNAL`). |

Source mode's verdict vocabulary is the closed enum
`{PASS_SOURCE_ONLY, FAIL}`; no source-mode code path emits a release token.

## Exit codes

| Exit code | Meaning                                                  |
|-----------|----------------------------------------------------------|
| 0         | Mode-ceiling pass (`PASS_SOURCE_ONLY` / `PASS_RELEASE`)  |
| 1         | Gate FAIL (verdict `FAIL`)                               |
| 2         | Usage/config error (unknown flag/key, containment)       |
| 3         | Typed ingress error (malformed/vacuous/empty inputs)     |

## Receipts

Every verifier run emits a schema-versioned JSON receipt — mode, verdict,
per-leg outcomes, input-tree hash, expectations-anchor hash, verifier code
hash, timestamp — into `--receipts-dir` (outside the verified tree),
published create-once under run-scoped unique names. An interrupted publish
leaves no parseable partial receipt; staging debris is auto-reclaimed by the
`scripts/stopdff_v5/fileio.py` create-once primitives this namespace routes
all final-path publications through.
