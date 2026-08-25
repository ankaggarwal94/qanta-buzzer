# Phase-4 reconciliation FOLLOW-UP handoff — qanta-buzzer (2026-08-24)

> **SUPERSEDES `handoff_phase4_reconciliation_2026-08-24.md`** (which presented repair
> options **A / B / C** as an open decision). **Option A has since been CHOSEN.** Read
> *this* file; the original is now a committed historical snapshot (do not edit it).

**What changed since the original handoff (delta).** The forensic reconciliation is
unchanged and still holds (H3 mechanism / H2 ruled out / H1 remediation-framing; the six
`idealized+performat`·`dp` divergences; intake fully verified). What is *new*: the author
**chose Option A** (scoped known-divergence / tolerance carve-out for
`idealized+performat`·`dp`); an **Option-A amendment proposal** and a **validated,
`git apply`-able integration diff** were drafted, then **committed + pushed** (proposal
`c128d5c`, diff `d93a1c9`), followed by **two doc-nit corrections** (`2cd7eea`, now
`HEAD == origin`). An **Option-C research ADR** (de-degenerate the idealized QA
calibrator) was drafted but is **HELD** (untracked, uncommitted). **Option B** (environment
reproduction) was **ruled infeasible** — Export-A's model/library environment is captured
nowhere in the evidence. Nothing has been *applied* to governed spec/comparator; that is
ceremony-gated (§4). **Verdict: A ≫ C > B.**

---

## 0. Where you are

- **Repo:** `Stanford/CS234/final_project/qanta-buzzer` (this Dropbox-synced repository).
- **Branch:** `feature/camera-ready-aims-v2`.
- **Current HEAD:** `2cd7eea3278850649304263e70a22d2eca3551e0` — **verified `== @{u}` (origin, up to date)**.
- **Pinned run/intake commit:** `7a1d02203fb9e6b8fff5d0c2cf6abe3e2c40b372`; **pinned tree**
  `cea970fe19072d5c93846810b6361599f63a9bf6` (verified this pass: `7a1d0220^{tree}` == that tree).
  HEAD is now 4 commits past the pin (intake was verified at the pin).
- **Authority: evidence-transfer + doc-authoring only — NOT execution authority, NOT
  git-write authority.** Writing NEW untracked analysis/handoff docs at repo root is
  authorized. Every governed evidence file is strictly read-only (§5). No `git add`/commit/
  push/clean/checkout/branch. No model/comparator execution. No applying the integration diff.

**Recent commit topology (verified `git log --oneline`):**

```
2cd7eea3  Correct Phase-4 reconciliation doc imprecisions (F5 env label, proposal 4a gating)   <- HEAD == origin
d93a1c97  Add Phase-4 Option-A integration diff (R-077 + comparator)
c128d5cc  Add Phase-4 Option-A repair proposal (scoped known-divergence carve-out)
b6860e9f  Add Phase-4 reconciliation diagnosis, verification, and handoff
7a1d0220  Harden Phase-4 certificate and Windows launcher                                       <- pinned run/intake commit
```

---

## 1. The persisted Phase-4 docs (read these for full detail; all verified to exist)

**Committed / tracked (do NOT edit — treat as immutable snapshots):**
- `phase4_reconciliation_diagnosis_2026-08-24.md` — consolidated diagnosis + smallest-safe-
  repair plan (four-way ledger; A/B/C). §F5 corrected to a neutral **"library stack"** label:
  the only in-repo successor cert (`pre_run_ready_certificate_2026-08-22.json`) records
  **macOS arm64 / Apple M3 Max** (Darwin, cpython-3.11.15) and hash-binds a pip-freeze whose
  versions match the cited stack; the **v3/retry1 runtime OS is NOT established from in-repo
  bytes** (those runs' certs live in the transport bundle, not inspected).
- `phase4_reconciliation_verification_2026-08-24.md` — independent Verifier per-claim verdicts +
  independent recompute of the six (v3 == retry1).
- `phase4_reconciliation_amendment_proposal_A_2026-08-24.md` — the Option-A proposal (folds R-077).
  §4a corrected: informational routing is **gated on field presence** (`observed_value is not
  _MISSING`), so a missing-field / structural failure on any of the six **still BLOCKS**; the six
  are **still counted in `checked` (= 194)**.
- `phase4_reconciliation_integration_diff_A_2026-08-24.md` — the validated `git apply`-able diff
  (committed `d93a1c9`); implements the proposal, adds no new policy.
- `handoff_phase4_reconciliation_2026-08-24.md` — the ORIGINAL handoff this file supersedes.

**HELD, must remain UNTRACKED (verified `?? ` in `git status --porcelain` and not in
`git ls-files`):**
- `phase4_reconciliation_research_adr_option_C_dedegenerate_calibrator_2026-08-24.md` — the
  Option-C research ADR (producer-semantics change + anchor re-freeze). Do **not** `git add` it.

---

## 2. Settled facts (verified; do not re-derive to act)

**Verdict (two independent lanes — Generator `5edea440`, Verifier `2801d4b8` — + parent hash
spot-check all agree):**

- **H3 is the mechanism.** The pinned successor producer does not reproduce Export-A's
  `idealized+performat`·**DP** stop-timing because that cell's per-format ("performat") QA
  calibrator is **single-class degenerate**: `score_arms` labels every idealized QA row
  `"correct": 1` → `fit_performat` collapses to the constant fallback
  `np.clip(1.0, 1e-4, 1-1e-4) = 0.9999` → QA confidence pins at ~1.0 → the raw QA `ref_stop`
  distribution is `{0:2178, 1:34, 2:28, 3:4, 4:2, 5:2, 6:1}`, i.e. **2178/2249 = 96.8%** buzz at
  step 0 → the DP stop-at-0-vs-1 partition sits on a **tie boundary** where **~29 knife-edge
  items reshuffle**. The DP-input code chain is **byte-identical `4bf5e02d`→pin** and numpy
  aggregation is **platform-stable**, so the residual divergence is **inferred** to enter
  upstream in record generation via the model/library stack (SBERT / torch / sklearn).
  **⚠ This is an INFERENCE, not byte-proof** — Export-A's environment is absent from the
  evidence and the model cannot be re-run here.
- **H2 (export-semantics mismatch) is RULED OUT** — recomputing the six from the byte-identical
  records reproduces the REGENERATED numbers **exactly**, by two independent implementations.
- **H1 (stale/incorrect anchor) is a remediation FRAMING only** — the frozen anchor equals
  Export-A byte-for-byte on this cell (anchor sha `2efff657…973eee`; producer commit `4bf5e02d`,
  2026-06-12, added Export-A with a `dp` block equal to the anchor).

**The six values** (`idealized+performat`·`dp`; identical across `qanta_phase4_windows_v3` and
`qanta_phase4_windows_retry1`; consistent across all four committed docs this pass):

| field | anchor (Export-A) | regenerated (both runs) | delta |
|---|---|---|---|
| `signed_mean` | 1.6332 | 1.6083 | −0.0249 |
| `abs_mean` | 1.6394 | 1.6198 | −0.0196 |
| `mc_earlier` | 4 | 8 | +4 |
| `qa_earlier` | 1915 | 1886 | −29 |
| `same_step` | 330 | 355 | +25 |
| `signed_mean_ci` | [1.5785, 1.6914] | [1.5526, 1.6652] | shifted down |

Count conservation: −29 (`qa_earlier`) +25 (`same_step`) +4 (`mc_earlier`) = 0; n = 2249
invariant. Both STOP_REPORTs `checked = 194`, same six failures, all `cell=idealized+performat,
policy=dp`; the other 188 fields pass.

**Certificates / ceremony.** Original run cert `a9dd121c…0994b` (v3); retry1 cert
`cbf0e2aa…447ce`; `8731ad00…` **retired**. Both ledgers **consumed**; **neither run promoted**.
There is **no currently usable certificate**: any retry #2 needs a fresh `PRE_RUN_READY`
certificate + fresh ledger / quarantine / promotion paths + a new digest-specific user
activation (§4).

**Intake is FULLY VERIFIED — no open blocker.** `phase4_exchange.py` sha256 `93fd54d5…78feb8`
(verified live this pass; untracked); `pull_verified` → 52 files / 10,791,780 B; live HEAD/tree
== pin at intake time. The sibling `…PUBLISH_RECEIPT.json` carries
`manifest_sha256 = dd6f3e6c…385f55` (+ `manifest_size = 11916`, `file_count = 52`,
`total_bytes = 10,791,780`) which **MATCHES** the live `RETURN_MANIFEST.json` bytes
(parent-confirmed). The out-of-band authenticity check is **available and PASSES** — do NOT
reintroduce the earlier (VOID) "receipt carries no manifest digest / blocker" claim.

**Supporting digests (transcribed from committed docs; transport bundle NOT re-inspected this
pass):** records `records/idealized__format_specific.jsonl` sha `4a179b68…24bf` (2249 rows,
byte-identical both runs); regenerated v3 `dd2333dd…9444`, retry1 `af224503…6bfc`.

---

## 3. Current decision state

**Option A — CHOSEN (release repair).** Treat exactly six fields of the single cell/policy
`idealized+performat`·`dp` — `{signed_mean, abs_mean, mc_earlier, qa_earlier, same_step,
signed_mean_ci}` — as a **declared known-divergence**: still compared against the frozen anchor
and **still counted in `checked` (194 preserved)**, but a **VALUE** mismatch routes to a new
informational report (`known_divergence_informational`) instead of `failures`, so a run that
diverges *only* on these six yields `verdict == "PASS"`. **STRUCTURE stays blocking** (a
missing carve-out field is still a blocking structural failure — routing is gated on
`observed_value is not _MISSING`). The **frozen anchor is preserved byte-for-byte** (not
re-frozen). Mirrors R-077's existing Random-K informational-only treatment, narrowed from
whole-cell to a six-field allowlist. **No model re-run.**

- **Proposal COMMITTED:** `c128d5c` (`phase4_reconciliation_amendment_proposal_A_2026-08-24.md`).
- **Validated integration diff COMMITTED:** `d93a1c9`
  (`phase4_reconciliation_integration_diff_A_2026-08-24.md`). Touches two tracked governed
  files: `reproducibility/colm_aims_2026/phase4.py` (5 hunks) + `.correctless/specs/
  camera-ready-aims-evidence-2.md` (R-077, 2 hunks). Frozen anchor **not** touched.
- **Diff validation (as recorded in the committed diff doc §4; NOT re-executed here):**
  `py_compile` + `ast.parse` OK; a behavioral probe on the **real edited `compare_parity`**
  (exercised from `/tmp` scratch copies of the pristine `phase4.py` under system Homebrew
  CPython 3.14.7 — never the Dropbox `.venv`; `/tmp` deleted after) passed **29/29 assertions
  across 7 scenarios S0–S4** (S1 = diverge only the six → PASS, `checked==194`, 6 informational;
  S2a/S2b/S2c = 7th field / same-name `myopic` field / other cell → FAIL; S3 = missing carve-out
  field → FAIL structurally; S4 = Random-K value → PASS/unchanged); and `git apply --check
  --verbose` on the combined patch → **both files OK** (`-p1`, no working-tree mutation).
  *(Note on validation figures: the committed diff doc's **29/29 assertions across 7 scenarios**
  (+ both files OK under `git apply --check`; phase4.py is 5 hunks, spec 2 hunks) is the
  **authoritative, repo-grounded** figure — trust it. Separately, an **independent verification
  lane** re-cut the diff from the committed bytes and re-ran its own **superset** battery
  (**49/49 assertions**) on the real edited `compare_parity`, returning **COMMIT_SAFE**; that
  49/49 result is a **session record only — NOT in any committed repo file** and cannot be
  spot-checked from tracked bytes. Both lanes agree: the six-field VALUE carve-out routes to
  informational while STRUCTURE still blocks and `checked == 194`.)*
- **NOT APPLIED.** The diff is committed as an *artifact*; it has **not** been applied to the
  governed comparator/spec. Verified live this pass: `phase4.py` still hashes to the diff's
  pristine base `c28e8f93…53fc8` and the spec still hashes to `9028c307…8bcfd` — i.e. both
  governed files are **byte-unchanged** and the diff base is **still current** (no Dropbox drift
  *yet*). Applying is **ceremony-gated** (§4).

**Option C — HELD (research ADR, untracked).**
`phase4_reconciliation_research_adr_option_C_dedegenerate_calibrator_2026-08-24.md`. Replace the
degenerate `"correct": 1` label with a graded/non-degenerate two-class label so `fit_performat`
fits a real Platt and the DP tie-boundary amplifier disappears. This is a **producer-semantics
change** with a larger blast radius (**~24 fields**: `idealized+performat` `dp` **+** `myopic`,
12 each; plus `qa_accuracy[idealized]` and the performat continuation/Platt digests), it **breaks
the currently-PASSING `idealized+performat`·`myopic` block**, and it **requires an anchor
re-freeze via a new authorized model run** — so it cannot be a spec amendment like A. A and C are
**complementary, not exclusive** (A settles the release now; C is later hardening). **Do not gate
the release on C.**

**Option B — INFEASIBLE (for now).** Pin Export-A's model/library stack and re-run to match the
anchor. Strongest guarantee, but **Export-A's environment is absent from the evidence and
unrecoverable from cited artifacts**; needs a full retry ceremony + authorized model execution
and still may not converge.

**Verdict: A ≫ C > B.**
- **A** — smallest blast radius (6 fields), preserves the anchor byte-for-byte, no model run;
  settles the release now. Tolerates the *symptom*, does not remove the amplifier.
- **C** — permanently removes the fragility at the source, but heaviest (producer change +
  re-freeze + headline-strawman perturbation); a research decision, not a release blocker.
- **B** — infeasible: the target environment simply isn't captured anywhere in the evidence.

---

## 4. Remaining decisions / next steps — ALL `[REQUIRES AUTHORIZATION]`

**(a) Record the Option-A author decision** in the appropriate spec-amendment / provenance
location. Natural homes (author's choice): fold the R-077 amendment into the spec at commit time,
or add a **sibling pending note** next to `phase4_exchange_spec_amendment_pending.md` (which holds
the R-083 "apply at commit time only" note), or a `decision_record_*_<date>.md` entry mirroring
`decision_record_D7_D8_A1_A2_2026-08-19.md`. **Do NOT edit the existing R-083 pending note.**

**(b) Apply the integration diff** to `reproducibility/colm_aims_2026/phase4.py` + R-077 in
`.correctless/specs/camera-ready-aims-evidence-2.md` — **only** if/when explicitly authorized, and
**only after re-running `git apply --check`** (the worktree may drift under Dropbox sync; the base
is current *as of this pass* but re-verify at apply time). Both are **tracked, Dropbox-synced**
files; per `phase4_exchange_spec_amendment_pending.md`, no tracked file may change until the
relevant run's outcome is confirmed settled — apply under that same commit-time discipline, and
honor the full ceremony below for any confirming retry #2.

**(c) Decide whether to commit or act on the Option-C ADR** — log C as a standalone, tracked
research decision (permanent-hardening follow-up) *independently* of the release repair, or leave
it held. If C is ever executed, its re-freeze **supersedes** A's carve-out for this cell.

**(d) Any retry #2 requires the full ceremony (none currently usable):**
1. a **fresh `PRE_RUN_READY` certificate** (HEAD-bound; the retired `8731ad00…` must not be
   reused; the consumed run-certs `a9dd121c…` / `cbf0e2aa…` may not be reused);
2. **fresh ledger / quarantine / promotion paths** (both existing ledgers are consumed; neither
   may be reused/deleted/recreated);
3. a **new digest-specific user activation**.
   For an Option-C run additionally: the producer-binding gate
   (`scripts/stopdff_fair_qa_retest.py:348–385`) requires the exact new graded-label producer to
   be **committed at HEAD** (rejects `git_dirty`, non-canonical `script_path`,
   `committed_sha256 != script_sha256`) before any run.

---

## 5. Expected untracked Phase-4 materials (do NOT touch; do NOT `git clean`) + hard prohibitions

**Untracked working-tree entries are EXPECTED and legitimate** (≈29 `?? ` entries: dev notes /
diffs / transcripts predating the runs, plus this loop's new docs). Do **not** `git clean`, add,
or delete them. Notable Phase-4 entries to leave exactly as-is:
- `reproducibility/colm_aims_2026/phase4_exchange.py` (must hash `93fd54d5…78feb8`; untracked),
  `tests/test_phase4_exchange.py`,
- `phase4_exchange_spec_amendment_pending.md` (holds the R-083 note — do NOT edit),
- `phase4_reconciliation_research_adr_option_C_dedegenerate_calibrator_2026-08-24.md` (held C-ADR),
- `pre_run_ready_certificate_2026-08-22.json`, `phase4_pre_receipts/`,
  `phase4_pre_repair_summary_2026-08-22.md` (**historical context only** — its awaiting-activation
  instruction is superseded; `8731ad00…` retired; no usable cert),
- and this file (`handoff_phase4_reconciliation_followup_2026-08-24.md`).

**Hard prohibitions (carry forward):**
- **Read-only on ALL governed evidence:** frozen anchor
  (`reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json`), every spec, the
  comparator `phase4.py`, both certificates, both ledgers, the Dropbox exchange bundle, all
  records, and `phase4_exchange.py`. Do not edit, replace, re-freeze, or recreate any of them.
- **No execution of any kind:** no model load, no inference, no calibrator / estimator / TF-IDF
  fit, no launcher, no retry, **no `compare_parity` run**, **no applying the integration diff**.
- **Do NOT execute from Dropbox**, and do NOT read staged / model assets from Dropbox.
- **Do NOT copy the bulky raw verified-intake bundle** (~10.8 MB) into the repo (it is transport).
- **No git / GitHub writes without explicit user authorization** — no `add`/commit/push/clean/
  checkout/branch, no PR/tracker writes. Leave new docs as untracked working-tree additions.
- Do not rewrite historical absolute paths embedded in signed evidence.

---

## 6. Evidence index (repo-root-relative unless noted)

`✔live` = digest/fact re-verified against live bytes this pass; `[doc]` = transcribed from the
committed reconciliation docs (transport bundle NOT re-inspected here).

| Artifact / fact | Path / value | Verify |
|---|---|---|
| Branch | `feature/camera-ready-aims-v2` | ✔live |
| HEAD `== @{u}` | `2cd7eea3278850649304263e70a22d2eca3551e0` | ✔live |
| Option-A proposal (committed) | `phase4_reconciliation_amendment_proposal_A_2026-08-24.md` (22,456 B @ HEAD; added `c128d5c` @ 21,801 B, §4a amended `2cd7eea`) | ✔live |
| Option-A integration diff (committed) | `phase4_reconciliation_integration_diff_A_2026-08-24.md` @ `d93a1c9` (19,865 B) | ✔live |
| Doc-nit corrections (== HEAD) | `2cd7eea` (edits proposal_A §4a + diagnosis §F5) | ✔live |
| Consolidated diagnosis (committed) | `phase4_reconciliation_diagnosis_2026-08-24.md` (16,314 B) | ✔live |
| Independent verification (committed) | `phase4_reconciliation_verification_2026-08-24.md` (8,995 B) | ✔live |
| Original handoff (superseded, committed) | `handoff_phase4_reconciliation_2026-08-24.md` (9,311 B) | ✔live |
| Option-C ADR (HELD, untracked) | `phase4_reconciliation_research_adr_option_C_dedegenerate_calibrator_2026-08-24.md` (23,271 B) | ✔live (untracked) |
| Frozen anchor (read-only, NOT touched) | `reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json` — sha256 `2efff657…973eee` | ✔live |
| Comparator (read-only; == diff base) | `reproducibility/colm_aims_2026/phase4.py` — sha256 `c28e8f93…53fc8` | ✔live |
| Spec R-077 (read-only; == diff base) | `.correctless/specs/camera-ready-aims-evidence-2.md` — sha256 `9028c307…8bcfd` | ✔live |
| Exchange impl (read-only, untracked) | `reproducibility/colm_aims_2026/phase4_exchange.py` — sha256 `93fd54d5…78feb8` | ✔live |
| Pending R-083 amendment (do NOT edit) | `phase4_exchange_spec_amendment_pending.md` (2,954 B, untracked) | ✔live |
| Pinned run/intake commit + tree | commit `7a1d0220…`; tree `cea970fe…` (`7a1d0220^{tree}`) | ✔live |
| Producer commit (added Export-A) | `4bf5e02d5447202a1a39f2e86c948ecb9a1614b8` (2026-06-12) | ✔live |
| Records (both runs, byte-identical) | `records/idealized__format_specific.jsonl` — sha `4a179b68…24bf` (2249 rows) | [doc] |
| Regenerated result JSONs | v3 `dd2333dd…9444`; retry1 `af224503…6bfc` | [doc] |
| Intake receipt vs live manifest | `manifest_sha256 = dd6f3e6c…385f55` == `RETURN_MANIFEST.json` (size 11916, 52 files, 10,791,780 B) | [doc] |
| Run certificates | v3 `a9dd121c…0994b`; retry1 `cbf0e2aa…447ce`; retired `8731ad00…`; none usable | [doc] |

---

**STATUS:** `RECONCILIATION: COMPLETE_AND_VERIFIED · VERDICT: H3_MECHANISM / H2_RULED_OUT /
H1_REMEDIATION_FRAMING (env-drift = INFERENCE, not byte-proof) · OPTION_A: CHOSEN + DRAFTED +
COMMITTED (proposal c128d5c, diff d93a1c9, doc-nits 2cd7eea==HEAD==origin) — NOT_APPLIED
(ceremony-gated) · OPTION_C: HELD (untracked ADR) · OPTION_B: INFEASIBLE · VERDICT_ORDER: A≫C>B ·
CERTIFICATE: NONE_USABLE · MODELS_EXECUTED: NONE · GOVERNED_EVIDENCE: READ-ONLY · THIS_DOC:
UNTRACKED`
