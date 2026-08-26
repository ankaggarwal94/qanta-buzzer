# PROPOSAL — Phase-4 parity spec amendment (Repair Option A): scoped known-divergence for `idealized+performat` · `dp`

## **PROPOSAL — [REQUIRES AUTHORIZATION] — NOT APPLIED**

**Status:** PROPOSAL / DRAFT. **Nothing here has been applied.** No governed file
was edited to produce this document: not the frozen anchor, not
`.correctless/specs/camera-ready-aims-evidence-2.md`, not the comparator
(`reproducibility/colm_aims_2026/phase4.py`), not any certificate / ledger /
quarantine / exchange bundle / record, and **not**
`phase4_exchange_spec_amendment_pending.md`. This is a standalone proposal for
the author to review before any integration.

**What applying it would mean (still gated):** editing the actual R-077 parity
policy (spec text + comparator) to carve out six fields of one cell as an
informational known-divergence. That edit — and any subsequent retry #2 — is
governed by the full ceremony (see §8): a **fresh PRE_RUN_READY certificate**,
**fresh ledger / quarantine / promotion paths**, and a **new digest-specific
user activation**. The retired certificate `8731ad00…` must not be reused; there
is **no currently usable certificate**.

**Scope of the carve-out:** exactly SIX fields of ONE cell/policy —
`idealized+performat` · `dp` · `{signed_mean, abs_mean, mc_earlier, qa_earlier,
same_step, signed_mean_ci}`. Not a blanket tolerance; not a numeric epsilon on
any cell; not the whole cell (the other six fields of that same cell/policy stay
blocking); not any other cell; not the Random-K policy.

**Mode of the drafting work:** READ-ONLY, evidence-transfer authority only (not
execution authority). No model load, no inference, no calibrator/estimator fit,
no launcher, no retry, no git. Findings below are transcribed from the two
cross-verified persisted reconciliation docs (§6) and are **not re-derived here**.

---

## 0. One-sentence summary

Treat the six repeated `idealized+performat` · `dp` divergences as an explicit,
allowlisted **known / tolerated divergence** — compared and reported
informationally, never blocking, so parity no longer STOPs on them — expressed
the same way the project already treats **Random-K numerical divergences as
informational-only** (R-077), while **preserving the frozen anchor's historical
values byte-for-byte** and changing nothing else.

---

## 1. What this proposes (the carve-out)

Today the comparator classifies these six fields as blocking nonrandom
comparisons; each run they produce six STOP failures (identical across
`qanta_phase4_windows_v3` and `qanta_phase4_windows_retry1`):

| field | type | anchor (Export-A) | regenerated (both runs) | delta |
|---|---|---|---|---|
| `signed_mean` | stat | 1.6332 | 1.6083 | −0.0249 |
| `abs_mean` | stat | 1.6394 | 1.6198 | −0.0196 |
| `mc_earlier` | count | 4 | 8 | +4 |
| `qa_earlier` | count | 1915 | 1886 | −29 |
| `same_step` | count | 330 | 355 | +25 |
| `signed_mean_ci` | stat (boot) | [1.5785, 1.6914] | [1.5526, 1.6652] | shifted down |

(Count conservation: −29 `qa_earlier` +25 `same_step` +4 `mc_earlier` = 0; n = 2249 invariant.)

**Proposed behavior:** for exactly these six `(cell, policy, field)` triples, a
mismatch is recorded as an **informational divergence**, not a blocking failure.
The verdict is computed from blocking failures only, so these six can no longer
force `FAIL`/STOP. Every other field — including the other six fields of the same
`idealized+performat` · `dp` block (`n`, `signed_median`, `abs_median`,
`mc_never_buzz`, `qa_never_buzz`, `signed_median_ci`), which all currently MATCH
— stays blocking.

---

## 2. The policy this mirrors — Random-K informational-only (exact location + shape)

The project already expresses "a divergence that is real but tolerated and
non-blocking" for the two Random-K cells. This proposal mirrors that mechanism
precisely; the carve-out below is intentionally the same *shape*.

**2a. Comparator — `reproducibility/colm_aims_2026/phase4.py`.**

- `compare_parity` docstring (lines 2549–2559), verbatim: *"The two Random-K
  cells are exempt from historical parity and reported informationally."*
- Structural-vs-value split (lines 2625–2631), verbatim comment: *"the STRUCTURE
  is blocking … while the numeric VALUES stay exempt (never compared,
  informational only). Structural rows do NOT increment `checked`: PASS still
  implies exactly the 194 blocking value comparisons."*
- Values-never-blocking path (lines 2662–2701). Line 2662 verbatim: *"Random-K
  VALUES: NEVER blocking; informational report only (R-077)."* The path compares
  each Random-K field against `informational_archived_values`, collects
  mismatches into a `divergences` list, and returns them under a dedicated
  informational key (`random_k_informational`, lines 2694–2701) that carries
  `exempt_from_historical_parity: True`, `compared`, and `divergences`.
- Verdict (lines 2706–2710): `verdict = "PASS" if not failures and checked ==
  EXPECTED_PARITY_CHECKED else "FAIL"` — i.e. the Random-K `divergences` never
  enter `failures`, so they cannot flip the verdict. (`EXPECTED_PARITY_CHECKED`
  is defined at line 190 as `8 * 2 * (10 + 2) + len(IDENTITY_FIELDS)` = 194.)

**2b. Spec — `.correctless/specs/camera-ready-aims-evidence-2.md`, R-077 (lines 831–855).**

Verbatim (lines 846–852): *"The two Random-K cells are exempt from historical
parity and reported informationally with `archived_rng_pinned=false` /
`fresh_rng_pinned=true` — but their STRUCTURE is required: both Random-K cells
must be present in the regenerated export with the full point and CI field set
(a missing Random-K cell or field is a blocking structural failure; only the
numeric VALUES are exempt from historical parity — operational-rejection repair
2026-08-22)."*

Note the precedent: **R-077 was itself already amended once** ("operational-rejection
repair 2026-08-22") to declare a tolerated divergence. This proposal is a sibling
amendment to R-077 in exactly that mold.

**2c. Anchor data shape — `reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json`
(`random_k` block, lines 394–401+), sha256 `2efff657…973eee`.**

```json
"random_k": {
  "archived_rng_pinned": false,
  "cells": ["krandom+shared", "krandom+performat"],
  "exempt_from_historical_parity": true,
  "fresh_rng_pinned": true,
  "informational_archived_values": { "...": "per-cell/policy archived values, compared informationally" }
}
```

The carve-out mirrors this triad: an `exempt_from_historical_parity: true` flag +
an explicit list of what is exempt + the tolerated values retained for
informational comparison. **The one deliberate difference:** Random-K exempts
whole cells' *values*; this carve-out is **field-scoped** (six named fields of
one cell/policy), and it **does not re-freeze the anchor** (see §4 rationale).

---

## 3. Proposed amendment text (mirrors the pending-amendment style; folds into R-077)

> Drafted to match the format/tone of `phase4_exchange_spec_amendment_pending.md`
> (bold rule id, `[unit]` tag, dated parenthetical, prose body, `Enforcement:`
> clause). **Do not apply now.** Fold as an amendment into **R-077** at commit
> time, under the same ceremony as the pending R-083 note.

- **R-077** *(amended — scoped known-divergence, 2026-08-24)*: In addition to the
  Random-K value exemption, exactly SIX fields of the single cell/policy
  `idealized+performat` · `dp` — `signed_mean`, `abs_mean`, `mc_earlier`,
  `qa_earlier`, `same_step`, `signed_mean_ci` — are a **declared
  known-divergence**: they are still compared against the frozen anchor and are
  still counted in `checked` (the 194-field cardinality is unchanged), but a
  mismatch on any of these six is recorded as an **informational divergence**,
  never a blocking failure, and can never flip the verdict to `FAIL`. This is the
  same treatment R-077 already gives Random-K numeric values (never blocking;
  informational report only), narrowed from whole-cell to a named six-field
  allowlist. The historical anchor values for these six fields are **preserved
  byte-for-byte** in `expected[idealized+performat][dp]` — the frozen anchor
  (`parity_anchor_export_a.json`, sha256 `2efff657…973eee`) is **not** re-frozen
  or overwritten. STRUCTURE remains blocking for this cell exactly as before (the
  cell and its full point + CI field set must be present; a missing cell/field is
  still a blocking structural failure). Every other field of this cell/policy and
  every other cell/policy remain blocking; the Random-K policy is unchanged; the
  producer is unchanged. Basis and provenance: the cross-verified reconciliation
  in `phase4_reconciliation_diagnosis_2026-08-24.md` and
  `phase4_reconciliation_verification_2026-08-24.md`. Enforcement: comparator-unit
  tests that (a) a single-field mutation on each of the six allowlisted fields
  yields `verdict == "PASS"` with the mutation surfaced in the informational
  divergence report; (b) a mutation on any of the other six fields of the same
  `dp` block, or on that cell's `myopic` block, or on any other cell, still
  `FAIL`s; (c) a missing `idealized+performat` cell or any of its fields still
  `FAIL`s structurally; (d) `checked == 194` on PASS is preserved.

---

## 4. Precise mechanism (what an author would change to apply this)

Two touch points, both localized. **The frozen anchor JSON is not touched** (that
is the option-A-defining property: preserve historical values, do not overwrite a
frozen artifact — so the allowlist lives in the comparator + spec, not by
re-freezing the anchor).

**4a. Comparator (`phase4.py`, `compare_parity`).** In the nonrandom-cell loop
(lines 2592–2623), before appending a mismatch to `failures`, test whether
`(cell, policy, field)` is in a small module-level allowlist, e.g.:

```python
KNOWN_DIVERGENCES = {
    ("idealized+performat", "dp", "signed_mean"),
    ("idealized+performat", "dp", "abs_mean"),
    ("idealized+performat", "dp", "mc_earlier"),
    ("idealized+performat", "dp", "qa_earlier"),
    ("idealized+performat", "dp", "same_step"),
    ("idealized+performat", "dp", "signed_mean_ci"),
}
```

If it is **and** the field is present with a diverging value (`observed_value is
not _MISSING`), route the mismatch to a new informational report
(`known_divergence_informational`, shaped like `random_k_informational` at lines
2694–2701: `{"exempt_from_historical_parity": True, "scope": "field",
"compared": ..., "divergences": [{cell, policy, field, expected, observed}, ...]}`)
instead of `failures`. Structure is **never** carved out: a MISSING carve-out field
(or any structural failure) on one of the six still routes to `failures` and BLOCKS,
exactly as today. The field is **still counted in `checked`** (so the 194
cardinality and the `checked == EXPECTED_PARITY_CHECKED` verdict gate at line 2708
are untouched). The verdict formula (line 2706–2710) needs no change: because
these six no longer enter `failures` on a present-but-diverging value, `not failures`
can now be true for a run that diverges only on them.
*(corrected 2026-08-24: the original sketch implied unconditional informational
routing; per validated integration-diff probe S3, routing is gated on field presence
(`observed_value is not _MISSING`) and a missing-field/structural failure for any of
the six still BLOCKS — structural failures are never carved out, and `checked` still
counts all six (194 preserved).)*

**4b. Spec (`camera-ready-aims-evidence-2.md`, R-077).** Append the §3 amendment
prose and the four enforcement bullets.

**Cardinality note (the one place the Random-K analogy is not 1:1).** Random-K
values were *never* part of `checked` (structural rows do not increment it;
`checked == 194` counts only blocking nonrandom comparisons). These six fields
*currently are* in the 194. The recommended variant above keeps them in `checked`
(still compared, still counted; only their failure-routing changes) so the pinned
194 magic number, the R-077 cardinality pins, and the anchor cardinality
validator (`_validate_parity_anchor`, lines 2507–2526) are all unaffected — the
smallest blast radius. An alternative that instead *removes* them from `checked`
(dropping the pin to 188) is possible but touches `EXPECTED_PARITY_CHECKED`
(line 190), the R-077 "194"/"160+32" prose, and every test asserting 194; it is
**not** recommended.

**Author's-choice alternative (more data-driven).** If the author prefers the
allowlist to be data (like `random_k`) rather than a code constant, add a NEW,
additive frozen policy artifact (e.g. `parity_known_divergences.json`) with its
own hash gate and have the comparator consult it. This still leaves
`parity_anchor_export_a.json` byte-identical, but adds a new frozen file +
validator wiring — a larger surface than the code constant. **Rejected variant:**
adding the carve-out block *into* `parity_anchor_export_a.json` — that changes its
sha256 and thus overwrites/re-freezes a frozen artifact, violating option A.

---

## 5. What it CHANGES / does NOT change

**Changes (only this):**
- Parity treats the six `idealized+performat` · `dp` fields
  (`signed_mean`, `abs_mean`, `mc_earlier`, `qa_earlier`, `same_step`,
  `signed_mean_ci`) as informational-only / known-divergence → **no STOP** on
  them. A run that diverges *only* on these six can PASS; the divergence is
  surfaced in an informational report.

**Does NOT change:**
- **No anchor overwrite.** The historical anchor values (1.6332 / 1.6394 / 4 /
  1915 / 330 / [1.5785, 1.6914]) stay byte-for-byte in
  `expected[idealized+performat][dp]`; `parity_anchor_export_a.json` sha256
  `2efff657…973eee` is preserved.
- **All other fields/cells stay blocking** — the other six fields of this same
  `dp` block, this cell's `myopic` block, and all seven other nonrandom cells.
- **Structure stays blocking** for this cell (present with the full field set).
- **Random-K policy unchanged** (its own `random_k` block and code path).
- **Producer code unchanged** — `scripts/stopdff_fair_qa_retest.py`, the DP core
  (`scripts/stopdff_dp/*`, `compute_prefix_calibration.py`), and the freeze/records
  path are untouched. The degenerate calibrator is **not** repaired here (that is
  option C — §7).
- **194 cardinality preserved** (recommended variant): `checked == 194` still
  gates PASS.

---

## 6. PROVENANCE

**Persisted, cross-verified reconciliation docs (repo root) this proposal cites:**
- `phase4_reconciliation_diagnosis_2026-08-24.md` — consolidated diagnosis +
  smallest-safe-repair plan (Generator lane `5edea440`, corrected by the
  Verifier). Source of the six values (§2 table), the four-way ledger, and repair
  options A/B/C (§4).
- `phase4_reconciliation_verification_2026-08-24.md` — independent re-derivation
  (Verifier lane `2801d4b8`, repo `.venv` CPython 3.11.15, numpy 2.4.6). Reproduced
  the six values independently for both runs and confirmed the mechanism.

**Byte-level basis for treating these six as a tolerated divergence (transcribed,
not re-derived):**
- **DP-input chain identical.** The full DP-input chain is empty-diff
  `4bf5e02d`→pin (`dp_solver`, `continuation`, `compute_prefix_calibration`,
  adapter, `rewards`, `types`, `_common`, `diagnostics`, `_provenance`), and
  `fit_performat` / `apply_cal` / `summarize` are AST-identical. The value path
  for this cell is **behaviorally unchanged on the exercised path**.
- **Platform-stable aggregation.** The numpy summary/bootstrap stage is
  platform-stable — the Verifier reproduced the Windows-produced CI **exactly** on
  macOS numpy 2.4.6 → aggregation is not the divergence channel.
- **Degenerate-calibrator knife-edge.** `score_arms` labels every idealized QA row
  `"correct": 1`, so `fit_performat` hits its single-class fallback
  `np.clip(1.0, 1e-4, 1-1e-4) = 0.9999`; QA confidence pins at ~1.0, giving a
  step-0 buzz fraction of 2178/2249 = 96.8% and putting the DP stop-at-0-vs-1
  partition on a tie boundary where ~29 knife-edge items reshuffle.
- **Inference-not-proof caveat (preserved verbatim from the diagnosis):** *"This
  is an INFERENCE, not byte-proof"* — Export-A's environment is absent from the
  evidence and the model cannot be re-run here; the residual divergence is
  *inferred* to enter upstream via the model/library stack (SBERT / torch /
  sklearn). H2 is ruled out (records reproduce the regenerated six exactly);
  H1 is a remediation framing only (anchor == Export-A byte-for-byte on this cell).
- **Intake fully verified; no open blocker.** The sibling
  `…PUBLISH_RECEIPT.json` carries `manifest_sha256=dd6f3e6c…385f55`, which
  **matches** the live `RETURN_MANIFEST.json` (size 11916; `file_count=52`;
  `total_bytes=10,791,780`). (An earlier draft's "receipt carries no manifest
  digest" BLOCKER was VOID and must not be reintroduced.) Certificates observed:
  original `a9dd121c…0994b`, retry1 `cbf0e2aa…447ce`; both ledgers consumed,
  neither run promoted.

---

## 7. RISKS / TRADEOFFS (A vs B vs C)

**Option A (this proposal) — scoped known-divergence.** *Recommended: smallest
change; no model re-run; preserves the anchor's historical values.*
- **Risk accepted:** we are tolerating a **real cross-environment divergence**
  rather than resolving it. Parity for this cell/policy's six fields becomes a
  structural + informational check, not a value check; a *future, unrelated*
  regression that happened to move only these six fields would no longer STOP.
  This is bounded by the field-scoped allowlist (six fields, one cell/policy) and
  by keeping structure blocking, but it is a genuine reduction in blocking
  coverage and should be recorded as such in the ledger/provenance.
- **Residual uncertainty carried forward:** the env/library drift attribution is
  an inference, not byte-proof (§6). Option A tolerates the *symptom*; it does not
  establish *which* value (Export-A's timing vs the pinned producer's) is
  "correct" — that remains a policy decision, not a byte fact.

**Option B — environment reproduction.** Pin Export-A's model/library stack and
re-run to match the anchor.
- **Stronger guarantee** (would restore value parity rather than tolerate its
  loss), but **Export-A's environment is absent from the evidence**, and this
  needs a full retry ceremony (§8) plus authorized model execution — prohibited
  in the current read-only lane. Higher cost, higher assurance.

**Option C — de-degenerate the idealized QA calibrator (the root amplifier).**
- **Not the smallest change:** it alters producer semantics and would re-anchor
  additional cells. Flagged as a **substantive latent finding worth a separate
  research decision** regardless of whether A or B is chosen — the single-class
  degenerate `fit_performat` (const 0.9999) is the amplifier that turns sub-ULP
  upstream drift into a visible stop-timing shift. Option A explicitly does **not**
  touch it.

---

## 8. Authorization & retry ceremony (unchanged; applies to applying A and to any re-run)

- **This document is a proposal only** — evidence-transfer authority, not
  execution authority. Applying it means editing the R-077 spec text and the
  comparator; that edit itself requires the author's explicit authorization.
- **Any retry #2** (e.g. to confirm a PASS after applying A, or to pursue option
  B) requires a **fresh PRE_RUN_READY certificate**, **fresh ledger / quarantine /
  promotion paths**, and a **new digest-specific user activation**.
- The retired certificate **`8731ad00…` must not be reused**; no consumed ledger
  may be reused; **there is no currently usable certificate.**
- Tracked-file timing: per `phase4_exchange_spec_amendment_pending.md`, the spec
  is a TRACKED file synced across devices via Dropbox; no tracked file may change
  until the relevant run's outcome is confirmed settled. Apply the R-077 amendment
  under that same commit-time discipline.

---

## 9. Integration target (reference only — not edited)

The eventual home for this text is an **R-077 amendment**, integrated alongside
the existing pending note **`phase4_exchange_spec_amendment_pending.md`** (which
holds the R-083 exchange-transport amendment "apply at commit time only"). That
pending file is the model for how held-but-drafted spec amendments are carried in
this repo, and is the natural companion at integration time. **It was not edited
by this proposal and must not be edited to integrate this one** — fold R-077's
amendment into the spec directly at commit time, or add a sibling pending note;
do not modify the existing R-083 pending file.

---

## 10. Uncertainty + implicit decisions

**Uncertainty (carried, not resolved):**
- The env/library drift entry point is an **inference, not byte-proof** (Export-A's
  env absent; model not re-runnable here). Option A tolerates the divergence; it
  does not prove its channel or adjudicate which value is "correct."
- Whether the author wants the allowlist as a code constant (§4a, smallest) or a
  new additive frozen artifact (§4 alternative, data-driven) is left open.

**Implicit decisions I made in drafting (author may override):**
1. **Filename & placement:** created at repo root as
   `phase4_reconciliation_amendment_proposal_A_2026-08-24.md` per the suggested
   name and the `phase4_<desc>_<date>.md` convention.
2. **Integrate as an R-077 amendment** (not a new rule number) — because R-077 is
   the parity policy and was already amended once for a tolerated divergence
   (Random-K "operational-rejection repair 2026-08-22"); this is the same mold.
   The author may instead assign a new rule id.
3. **Allowlist lives in comparator + spec, not in the frozen anchor** — chosen to
   honor option A's "preserve historical values / do not overwrite a frozen
   artifact." A data-driven variant is offered but not recommended as smallest.
4. **194 cardinality preserved** (six fields stay in `checked`, only their
   failure-routing changes) — chosen for smallest blast radius over the
   drop-to-188 variant.
5. **Field-scoped, not cell-scoped:** carve out exactly the six *diverging*
   fields, leaving the other six fields of the same `dp` block blocking — tighter
   than Random-K's whole-cell exemption, matching the "scoped, not blanket"
   requirement.
6. **New informational key named `known_divergence_informational`** (shaped like
   `random_k_informational`) — a suggested name; the author may rename.
