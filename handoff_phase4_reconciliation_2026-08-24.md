# Phase-4 reconciliation handoff — continue in the qanta-buzzer repo (2026-08-24)

You are a fresh agent continuing **Phase-4** of the QANTA Buzzer camera-ready evidence task (Stanford
CS234) in this repository. You carry no memory of the prior conversation; this document is self-contained
and every load-bearing claim is paired with a spot-checkable artifact. The forensic reconciliation of the
six `idealized+performat`·DP parity failures is **complete and independently verified** — your job is to
decide and (only when authorized) execute the smallest safe repair, not to re-litigate the diagnosis.

## 0. Where you are

- **Repo:** `Stanford/CS234/final_project/qanta-buzzer` (this Dropbox-synced repository).
- **Branch:** `feature/camera-ready-aims-v2`.
- **Pinned commit:** `7a1d02203fb9e6b8fff5d0c2cf6abe3e2c40b372`.
- **Pinned tree:** `cea970fe19072d5c93846810b6361599f63a9bf6`.
- **Authority:** **evidence-transfer only, not execution authority.** Writing NEW analysis/handoff docs
  into the repo root is authorized; every governed evidence file is strictly read-only (see §5).

## 1. The two persisted reconciliation docs (read these first)

- `phase4_reconciliation_diagnosis_2026-08-24.md` — the consolidated, correction-applied forensic
  diagnosis + smallest-safe-repair plan (four-way ledger; A/B/C repair options).
- `phase4_reconciliation_verification_2026-08-24.md` — the independent Verifier's per-claim verdicts and
  independent recomputation of the six values.

Both were produced on Device 1 (macOS, no CUDA) by two independent lanes (Generator `5edea440`, Verifier
`2801d4b8`) plus a parent hash spot-check. The bulky verified-intake bundle (~10.8 MB) was **not** copied
into the repo; it lives in the Dropbox exchange bundle as transport.

## 2. Settled facts (verified; do not re-derive to act)

**Verdict (two independent lanes + parent hash spot-check agree):**
- **H3 is the mechanism** — the pinned producer does not reproduce Export-A's `idealized+performat`·DP
  stop-timing because that cell's per-format QA calibrator is single-class **degenerate** (idealized arms
  coded `correct=1` → `fit_performat` collapses to constant `0.9999` → 2178/2249 = 96.8% of QA arms buzz
  at step 0 → ~29 knife-edge items reshuffle on the DP tie boundary). The DP-input code chain is
  **byte-identical `4bf5e02d`→pin** and numpy aggregation is platform-stable, so the divergence is
  **inferred** to enter upstream in record generation via the model/library stack (SBERT / torch /
  sklearn). **This is an inference, not byte-proof** (Export-A's env is absent; the model cannot be
  re-run here).
- **H2 (export-semantics mismatch) is RULED OUT** — recomputing the six from the byte-identical records
  reproduces the REGENERATED numbers exactly, by two independent implementations.
- **H1 (stale/incorrect anchor) is a remediation FRAMING only** — the frozen anchor equals Export-A
  byte-for-byte on this cell (anchor sha `2efff657…973eee`; producer commit `4bf5e02d` added Export-A with
  a `dp` block equal to the anchor).

**The six values** (`idealized+performat·dp`; identical across `qanta_phase4_windows_v3` and
`qanta_phase4_windows_retry1`):

| field | anchor | regenerated |
|---|---|---|
| `signed_mean` | 1.6332 | 1.6083 |
| `abs_mean` | 1.6394 | 1.6198 |
| `mc_earlier` | 4 | 8 |
| `qa_earlier` | 1915 | 1886 |
| `same_step` | 330 | 355 |
| `signed_mean_ci` | [1.5785, 1.6914] | [1.5526, 1.6652] |

**Certificates (both runs):** original `a9dd121c…0994b`; retry1 `cbf0e2aa…447ce`. Both ledgers consumed;
neither run promoted. Both STOP_REPORTs `checked=194`, same six failures.

**Intake is FULLY VERIFIED — no open blocker.** `phase4_exchange.py` sha256 `93fd54d5…78feb8`;
`pull_verified` → 52 files / 10,791,780 B; live `HEAD`/`tree` == pin. The sibling
`…PUBLISH_RECEIPT.json` carries `manifest_sha256=dd6f3e6c…385f55` which **MATCHES** the live
`RETURN_MANIFEST.json` bytes (parent-confirmed via `shasum`, size 11916); `file_count=52` /
`total_bytes=10,791,780` match. **Do NOT repeat the earlier "no digest / blocker" error** — that
out-of-band authenticity check is available and PASSES.

**Supporting digests:** records `records/idealized__format_specific.jsonl` sha `4a179b68…24bf` (2249 rows,
byte-identical both runs); regenerated v3 `dd2333dd…9444`, retry1 `af224503…6bfc`.

## 3. Smallest safe repair options (all [REQUIRES AUTHORIZATION]; none executed)

- **(A) RECOMMENDED — scoped known-divergence / tolerance for `idealized+performat·dp`**, analogous to the
  existing frozen policy that treats Random-K numerical divergences as informational-only. Recorded via
  the pending spec amendment + a provenance note. Smallest change; **no model re-run; preserves the
  anchor's historical values** rather than overwriting a frozen artifact.
- **(B) Environment reproduction** — pin Export-A's model/library stack and re-run to match the anchor.
  Stronger guarantee, but Export-A's env is absent from the evidence and it needs a full retry ceremony.
- **(C) De-degenerate the idealized QA calibrator** (the root amplifier) — NOT the smallest change; it
  changes producer semantics and would re-anchor more cells. Flag as a substantive latent finding worth a
  research decision regardless of A/B.

**Any retry #2 requires the full ceremony:** a **fresh PRE_RUN_READY certificate**, **fresh
ledger / quarantine / promotion paths**, and a **new digest-specific user activation**. There is **no
currently usable certificate**: `8731ad00…` is retired, and both run ledgers are consumed. This handoff is
evidence-transfer authority only, not execution authority.

## 4. Expected untracked Phase-4 materials (do NOT touch; do NOT `git clean`)

Roughly **28** untracked entries pre-exist in the working tree (dev notes / diffs / transcripts predating
the runs). Expected Phase-4 materials among them include:
- `reproducibility/colm_aims_2026/phase4_exchange.py` (must hash to `93fd54d5…78feb8`),
- `tests/test_phase4_exchange.py`,
- `phase4_exchange_spec_amendment_pending.md` (do NOT edit — drafting the amendment text is a separate
  future step).

Treat `phase4_pre_repair_summary_2026-08-22.md` as **historical context only**: its awaiting-activation
instruction is superseded, certificate `8731ad00…` is retired, and no currently usable certificate exists.
Your two new reconciliation docs (§1) plus this handoff are also untracked additions — distinguish them
from the pre-existing entries when you inspect `git status`.

## 5. Hard prohibitions (carry these forward)

- **Read-only on ALL governed evidence:** the frozen anchor
  (`reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json`), every spec, both certificates,
  both ledgers, the Dropbox exchange bundle, all records, and `phase4_exchange.py`. Do not edit, replace,
  or recreate any of them.
- **No execution as part of diagnosis:** no model load, no inference, no calibrator / estimator / TF-IDF
  fit, no launcher invocation, and no retry #2.
- **Do not execute from Dropbox**, and do not read staged / model assets from Dropbox.
- **Do not run `git clean`**; do not delete/replace/recreate either ledger; do not change branch or
  checkout.
- **Do not rewrite historical absolute paths** embedded in signed evidence.
- **No git/GitHub writes without explicit user authorization** — no `git add` / commit / push, no PR or
  tracker writes. Leave new docs as untracked working-tree additions.
- The bulky raw verified-intake bundle is transport; **do not copy it into the repo**. If you believe raw
  evidence must be included, FLAG it for the author instead of copying.

## 6. Evidence index (repo-root relative unless noted)

| Artifact | Path |
|---|---|
| Consolidated diagnosis (this loop) | `phase4_reconciliation_diagnosis_2026-08-24.md` |
| Independent verification (this loop) | `phase4_reconciliation_verification_2026-08-24.md` |
| Original transport handoff | `../phase4_exchange/phase4_run_and_retry1_fail_closed_2026-08-24/DEVICE1_HANDOFF_PROMPT.md` |
| Historical PRE reconciliation | `phase4_pre_run_reconciliation_2026-08-22.md` |
| Historical PRE repair summary (superseded) | `phase4_pre_repair_summary_2026-08-22.md` |
| Pending spec amendment (do NOT edit) | `phase4_exchange_spec_amendment_pending.md` |
| Frozen anchor (read-only) | `reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json` (sha `2efff657…973eee`) |
| Exchange impl (read-only, untracked) | `reproducibility/colm_aims_2026/phase4_exchange.py` (sha `93fd54d5…78feb8`) |
| Producer commit that added Export-A | `4bf5e02d5447202a1a39f2e86c948ecb9a1614b8` (2026-06-12) |

## 7. Your immediate next step

Present options **A / B / C** to the author as a decision (A recommended), with the retry ceremony spelled
out for B. Take no repair action, no re-run, and no governed-file edit until the author records an explicit
decision — and, for any re-run, the full fresh-certificate + fresh-ledger + new-activation ceremony.

**Status line:** `RECONCILIATION: COMPLETE_AND_VERIFIED · VERDICT: H3_MECHANISM / H2_RULED_OUT /
H1_REMEDIATION_FRAMING · INTAKE: FULLY_VERIFIED · REPAIRS: A(rec)/B/C AWAITING_AUTHOR_DECISION ·
CERTIFICATE: NONE_USABLE · MODELS_EXECUTED: NONE · WRITE_GATES: GOVERNED EVIDENCE READ-ONLY`
