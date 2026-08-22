# Dev Journal

Append-only maintainer notes: why things are shaped the way they are.

## 2026-08-22 — camera-ready-aims-evidence-v2

This feature is a greenfield v2 reimplementation, not an evolution of the v1
namespace. The v1 verifier tower (branch `feature/camera-ready-aims-spec`
plus two local-only repair commits, `2709624b` and `f8ba2042`) accumulated
enough adjudicated repairs that the external reviewer signed a contract
freeze (`contract_freeze_signoff_2026-08-20.md`) authorizing a successor
built fresh on canonical `main` — REIMPLEMENT, never cherry-pick. The v1
commits were review inputs only; every rule in the 71-rule spec
(`.correctless/specs/camera-ready-aims-evidence-2.md`) was re-derived from
the frozen contract and re-encoded as failing-first tests before any
implementation existed. That RED suite (11 test modules + 1 helpers module,
605 tests) *is* the API contract: module boundaries (`schema` / `pairing` /
`verifier` / `ledger` / `closure` / `qa012` / `receipt` / `render` /
`legacy` / `verify`), the exact CLI surface, and the exit-code map were all
fixed by tests before GREEN began, and two GREEN-phase disagreements were
escalated as TEST_BUG adjudications rather than test edits.

The D7(b) inference goldens deserve special note for future maintainers:
they are pure-procedure truths, not implementation snapshots. The seed
derivation, the shared PCG64 resample matrix, the `+1/1001`-corrected
p-values, and the m=10 Holm family are pinned by hand-checked golden
literals in `tests/_colm_aims_v2_helpers.py` computed from the written
procedure (sign-off §3) under NumPy exactly 2.4.6 — so any reimplementation
in any language must reproduce them bit-exactly, and a "passing" refactor
that shifts a single matrix byte or Holm tie-break is a real defect, not
tolerance noise. This is also why the suite hard-requires NumPy 2.4.6 and
why a release leg checks the runtime version: bit-exactness is part of the
estimand identity, not an environment nicety.

Two design choices shape most of `verifier.py`. First, guarded leg builders:
every check runs inside `_guarded`, so an unexpected exception becomes that
leg's FAIL and the run still reaches a verdict and a create-once receipt —
the failure mode "verifier crashed, no receipt, rerun and hope" is
structurally excluded. Second, the R-064 mode split for tree sidecars:
source mode *raises* a typed ingress error on a non-object sidecar (fast,
loud, exit 3 — appropriate while an author is iterating), while release mode
*collects* the same defect as a mandatory failing `sidecar_ingress:<path>`
leg, so a release report enumerates every defect in one pass instead of
revealing them one crash at a time. Relatedly, legacy ingest is one named
door — `legacy.load_legacy_v1_document` — with a per-family certify table:
only the three captured `paper_exports` aggregate families can back an
aggregate ledger row, and a v1-versioned profile parses but certifies
nothing. Strict surfaces never import the legacy module; laundering v1
bytes into a certification leg requires editing code, not crafting input.

Finally, the headline-estimand change that motivated much of §2.3: the v1
both-finite interval path was retired from the headline label because
conditioning on both arms stopping silently changes the population — it
drops every timeout-asymmetric pair, which is exactly where a reference
shift shows up, and lets a "mean over finite stops" masquerade as the
package-level effect. The v2 headline is the sentinel-coded, terminal-imputed
signed shift over ALL 2,249 complete pairs (R-048); the finite-only summary
survives only as a separately named secondary estimand with its own
denominator, digest, and ledger row that can never pool with the headline
(R-049, R-068). One consequence still pends externally: pinning the
bootstrap-seed input to the 2,249 complete-pair key set (zero in-package
exclusions) narrows the original handoff lineage term, and that narrowing is
flagged RESOLVED-pending-ack for the reviewer's next shuttle round.
