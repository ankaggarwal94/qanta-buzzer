# Decision record — Phase-4 Option A (six-field known-divergence carve-out) (2026-08-24)

**Recorded by:** Cursor agent (implementation worker), at the author's direction (full authorization). **Audience:** project record + Phase-4 reconciliation lane.
**This document (Dropbox connector):** `Stanford/CS234/final_project/qanta-buzzer/decision_record_phase4_option_A_2026-08-24.md`
**Reference:** the repair-option queue folded into `phase4_reconciliation_amendment_proposal_A_2026-08-24.md` (Option A) and its validated integration artifact `phase4_reconciliation_integration_diff_A_2026-08-24.md`.

| # | Decision | Author's disposition (settled rationale) |
|---|---|---|
| **Option A** | Phase-4 R-077 parity reconciliation — repair-option choice for the single divergent cell/policy `idealized+performat` · `dp` | **(A) CHOSEN.** Scoped **six-field known-divergence carve-out** for `idealized+performat` · `dp` (fields: `signed_mean`, `abs_mean`, `mc_earlier`, `qa_earlier`, `same_step`, `signed_mean_ci`). A **VALUE** mismatch on exactly those six routes to a new informational report `known_divergence_informational` instead of the blocking `failures` list, so a run diverging *only* on the six yields `verdict == "PASS"`. **STRUCTURE stays blocking** (routing gated on `observed_value is not _MISSING`); `checked == 194` preserved; frozen anchor preserved **byte-for-byte**. Mirrors R-077's existing Random-K informational treatment, **narrowed from whole-cell to a six-field allowlist**. **No model re-run.** |

## Consequences now in force

1. **Comparator (`reproducibility/colm_aims_2026/phase4.py`)** gains the amended-R-077 carve-out: module constants `KNOWN_DIVERGENCE_CELL/POLICY/FIELDS` + the `KNOWN_DIVERGENCES` frozenset; in `compare_parity`, the six triples are still compared and still counted in `checked` (194 unchanged), but a VALUE mismatch on any of the six is appended to `known_divergences` (and surfaced in a new `known_divergence_informational` report) instead of `failures`. Routing is gated on `observed_value is not _MISSING`, so a **missing** carve-out field is still a **blocking structural failure**. The verdict formula (`PASS iff not failures and checked == 194`) and the raise-on-missing-anchor-field guard are unchanged.
2. **Spec (`.correctless/specs/camera-ready-aims-evidence-2.md`, R-077)** records the same carve-out: the header marker is amended (`*(PRE-6; amended by Repair-A, 2026-08-24 — idealized+performat·dp six-field known-divergence)*`, preserving the original `*(PRE-6)*` tag and mirroring the R-043/R-072 amendment convention), and the R-077 body + `Enforcement:` clause add the four Option-A enforcement legs (six-field VALUE mutation → PASS with the mutation surfaced informationally; a 7th `dp` field, the `myopic` block, or any other cell still FAILs; a missing cell/field still FAILs structurally; `checked == 194` on PASS preserved).
3. **Blast radius is minimal by construction:** the other six fields of this `dp` block (`n`, `signed_median`, `abs_median`, `mc_never_buzz`, `qa_never_buzz`, `signed_median_ci`), this cell's `myopic` block, every other nonrandom cell/policy, and the Random-K policy all remain exactly as before. The frozen anchor is **not** re-frozen or overwritten.
4. **This commit is local only.** The integration patch (identical to the validated `phase4_reconciliation_integration_diff_A_2026-08-24.md`) is applied to the two governed files above and paired with this decision record; no push, no tracker write, no model/comparator execution.

## Mechanism (why these six diverge)

### H3 — accepted (tolerated-divergence attribution)
The per-format ("performat") QA calibrator for the idealized cell is **single-class degenerate**: `score_arms` labels every idealized QA row `correct:1`, so `fit_performat` collapses to the constant fallback `np.clip(1.0, 1e-4, 1 - 1e-4) = 0.9999` → QA confidence ≈ 1.0 → ≈ **96.8% (2178/2249)** of items buzz at step 0 → the DP stop-at-0-vs-1 partition sits on a **tie boundary** where ≈ **29 knife-edge items** reshuffle. The DP-input code chain is byte-identical (`4bf5e02d` → pin) and the numpy aggregation is platform-stable, so the residual divergence is **INFERRED** to enter upstream via the model/library stack (SBERT / torch / sklearn).

> **This is an INFERENCE, not a byte-proof.** Export-A's environment is absent from the cited evidence and cannot be re-run here; the attribution is the best-supported explanation, not a reproduced result.

### Ruled out / reframed
- **H2 (export-semantics mismatch): RULED OUT** by two independent recomputes.
- **H1 (stale anchor): remediation FRAMING only** — the frozen anchor is Export-A **byte-for-byte** (anchor sha256 `2efff657…973eee`, producer commit `4bf5e02d`, 2026-06-12); it is not stale, so "re-freeze" is framing, not a fix that Option A performs.

## The six values (frozen anchor → divergent run)

| Field | Anchor | Run | Δ |
|---|---|---|---|
| `signed_mean` | 1.6332 | 1.6083 | −0.0249 |
| `abs_mean` | 1.6394 | 1.6198 | −0.0196 |
| `mc_earlier` | 4 | 8 | +4 |
| `qa_earlier` | 1915 | 1886 | −29 |
| `same_step` | 330 | 355 | +25 |
| `signed_mean_ci` | [1.5785, 1.6914] | [1.5526, 1.6652] | — |

**Invariants:** count conservation `−29 + 25 + 4 = 0`; `n = 2249` invariant; both `STOP_REPORT`s carry `checked = 194`.

## Rationale / verdict — A ≫ C > B

- **A (CHOSEN):** smallest blast radius (6 fields), preserves the anchor byte-for-byte, no model run, and **settles the release now**. It *tolerates the symptom* and does **not** remove the amplifier (the degenerate calibrator).
- **C (held, complementary later hardening — NOT a release blocker):** de-degenerate the calibrator. This is a producer-semantics change with a ≈24-field blast radius that **breaks the currently-PASSING `idealized+performat` · `myopic` block** and needs an anchor **re-freeze** via a new authorized run. If C is ever executed, its re-freeze **supersedes** A's carve-out for this cell.
- **B (rejected — infeasible):** reproduce Export-A exactly. The Export-A environment is unrecoverable from the cited artifacts.

## Provenance

- **Proposal (design source of truth):** `phase4_reconciliation_amendment_proposal_A_2026-08-24.md`, committed `c128d5c`.
- **Validated integration diff:** `phase4_reconciliation_integration_diff_A_2026-08-24.md`, committed `d93a1c9`; doc-nits `2cd7eea`; follow-up handoff `9d5faa6a`.
- **Validation:** 29/29 assertions across 7 scenarios (S0–S4, incl. S2a/S2b/S2c) on the **real edited** `compare_parity` + `git apply --check` **OK** — authoritative, repo-grounded. An independent session lane re-ran a **49/49 superset** returning `COMMIT_SAFE` (session record only; not in committed bytes).
- **Applied in this commit** to `reproducibility/colm_aims_2026/phase4.py` + `.correctless/specs/camera-ready-aims-evidence-2.md` (R-077).

## Scope / limits

- Frozen anchor **untouched** (`parity_anchor_export_a.json`, sha256 `2efff657…973eee`).
- **No model / comparator / calibrator / estimator executed**; no code executed from Dropbox.
- **Option C held** (untracked); it remains available as later hardening, not a blocker.
- Any **retry #2** (to confirm a PASS after A, or to pursue C) still requires the full ceremony: a fresh `PRE_RUN_READY` certificate + fresh ledger / quarantine / promotion paths + a new digest-specific user activation. The retired certificate `8731ad00…` must not be reused.

— RECORD COMPLETE. Standing vocabulary rule: constructed-reference sensitivity evidence only; no observed open-ended decision-preservation claims. The H3 attribution above is an INFERENCE, explicitly not a byte-proof.
