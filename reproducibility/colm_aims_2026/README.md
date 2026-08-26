# COLM AIMS 2026 evidence verifier (v2)

Two-mode fail-closed verifier for the constructed QA reference sensitivity
evidence package — a constructed-reference sensitivity diagnostic, never
observed open-ended decision evidence. It is distinct from
`scripts/verify_audit_release.py` (the legacy audit-release checker, which
stays byte-identical and is not part of this contract).

## Invocation

From the repository root:

```bash
python -m reproducibility.colm_aims_2026.verify \
    --mode source --tree PATH/tree --receipts-dir PATH/receipts

python -m reproducibility.colm_aims_2026.verify \
    --mode release --tree PATH/tree \
    --expectations PATH/expectations.json --receipts-dir PATH/receipts

python -m reproducibility.colm_aims_2026.verify \
    --mode release --runs-root SITE/runs \
    --expectations SITE/expectations.json --receipts-dir SITE/receipts
```

Direct-path invocation (`python reproducibility/colm_aims_2026/verify.py`)
bootstraps `sys.path` repo-root-first and behaves identically.

## Input layout

```
pkg/
  tree/                          <- the verified artifact tree
    profile.json                 <- strict v2 ten-cell profile
    records/<cell_id>.jsonl      <- exactly one per declared cell
    presentation_manifest.json
  expectations.json              <- OUTSIDE the tree (release mode input)
  ledger.json                    <- frozen claim ledger (anchored)
  rights.json                    <- rights inventory
  receipts/                      <- receipt output directory
```

The `--runs-root` release entry selects the canonical package exclusively
through the ledger's `canonical_run_id` pointer over a
`runs/<run_id>/tree` layout; symlinked, escaping, empty, or dangling
pointers fail the release run, and newest-wins selection never happens.

## Verdicts

- `--mode source` ceiling: `PASS_SOURCE_ONLY` — in-package consistency only.
  It does NOT certify release bindings, anchored expectations, rights
  clearance, archival identity, or any observed-decision claim.
- `--mode release`: `PASS_RELEASE` requires the independently anchored
  expectations file (outside the tree), verified rights, manifest
  reconciliation, ledger recomputation, and the anchored grid/inference
  pins.
- Any failed leg yields `FAIL`.
- The `CAMERA_READY_CLOSURE` gate (`closure.py`) is a separate inventory
  gate over the D6 manuscript baseline; neither verifier verdict implies
  it, and it implies neither of them.

## Exit codes

- `0` — mode-ceiling pass (`PASS_SOURCE_ONLY` / `PASS_RELEASE`)
- `1` — gate FAIL
- `2` — usage/config error (unknown flags fail closed; `allow_abbrev` off)
- `3` — typed ingress error (malformed/oversized/non-finite/versioned
  surface defects)
- `4` — internal (non-ingress) error; no verdict was reached

## Receipts

Every run emits a schema-versioned JSON receipt (`receipt-<run_id>.json`)
into `--receipts-dir`, outside the verified tree, create-once: mode,
verdict, per-leg outcomes, input-tree hash, expectations-anchor hash,
verifier code hash, timestamp.

## Phase-4 process/host trust boundary

The certificate-bound Phase-4 launcher is an integrity and reproducibility
workflow, not a sandbox, privilege boundary, or hostile-process containment
mechanism. The certified producer and dependencies, host OS and filesystem,
and processes sharing the launcher's OS identity must be cooperative, and the
supported run leaves no surviving producer descendants. The launcher guards
ordinary crashes, malformed or drifting bytes, aliases, replacement races,
and create-once publication, but it cannot protect a path-addressable
candidate or promoted tree from a deliberately hostile process with the same
OS access. Its launch receipt binds
`trusted_same_os_identity_no_surviving_descendants_v1`; the D7(b) driver
rejects a missing or different process-trust token. If this operator
precondition is unavailable, do not run the ceremony: provision a separately
approved OS isolation boundary first. No verifier or closure verdict proves
hostile-process provenance or tamper resistance.
