# Phase-4 forensic reconciliation — CONSOLIDATED diagnosis + smallest-safe-repair plan (2026-08-24)

**From:** Phase-4 forensic reconciliation, Device 1 (MacBook Pro M3 Max, macOS, no CUDA).
**To:** the author and the next Phase-4 agent in this repo.
**Re:** the six repeated `idealized+performat` **DP** parity failures (runs `qanta_phase4_windows_v3`
and `qanta_phase4_windows_retry1`) against the frozen anchor
`reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json`.
**Mode:** READ-ONLY, fail-closed, **evidence-transfer authority only** (not execution authority). No
model load, no inference, no calibrator/estimator fit, no launcher, no retry, no writes to any governed
evidence. This document is a NEW persisted analysis artifact; it consolidates and *corrects* the
machine-local Generator diagnosis using the independent Verifier's findings.
**Provenance of the underlying work:** Generator lane `5edea440`, independent Verifier lane `2801d4b8`,
plus a parent hash spot-check. The bulky verified-intake bundle stays in transport (the Dropbox
exchange bundle); only this diagnosis is persisted here.

## Status line

`VERDICT: H3_MECHANISM · H2_RULED_OUT · H1_REMEDIATION_FRAMING · INTAKE: FULLY_VERIFIED (incl. receipt
manifest digest match) · OPEN_BLOCKERS: NONE · SIX_VALUES: STABLE_ACROSS_v3_AND_retry1 · REPAIRS:
A/B/C ALL [REQUIRES AUTHORIZATION], NONE EXECUTED · MODELS_EXECUTED: NONE · WRITE_GATES: GOVERNED
EVIDENCE READ-ONLY`

---

## 0. Conclusion (one sentence)

The pinned successor producer does **not** reproduce Export-A's `idealized+performat`·**DP** stop-timing
because that cell's per-format ("performat") QA calibrator is **single-class degenerate** — the all-correct
idealized arm collapses `fit_performat` to a constant `0.9999`, so 96.8% of QA arms buzz at step 0 and the
DP stop-partition sits on a tie boundary where ~29 knife-edge items reshuffle; the value-producing code
chain for this cell is **behaviorally unchanged on the exercised path** (byte-identical DP inputs
`4bf5e02d`→pin) and numpy aggregation is platform-stable, so the residual divergence is **inferred** to
enter upstream in record generation via the model/library stack (SBERT / torch / sklearn). **H2** is
excluded by exact recomputation on the current side; **H1** is a remediation *framing* only (the anchor
equals Export-A byte-for-byte on this cell), not the mechanism.

---

## 1. Intake verification — FULLY VERIFIED, no open blocker

| Check | Result |
|---|---|
| `reproducibility/colm_aims_2026/phase4_exchange.py` sha256 == `93fd54d5…78feb8` | **PASS** |
| `pull_verified(bundle → machine-local intake)` per-member size + hash | **PASS** — 52 files / 10,791,780 B; no missing / extra / size / hash mismatch |
| Live repo `HEAD` == pin `7a1d0220…c40b372` and `HEAD^{tree}` == `cea970fe…3a9bf6` | **PASS** |
| Out-of-band receipt authenticity — sibling `…PUBLISH_RECEIPT.json` `manifest_sha256` vs live `RETURN_MANIFEST.json` bytes | **PASS** — `dd6f3e6c…385f55` **matches** live manifest (parent-confirmed via `shasum`), `manifest_size=11916`; receipt `file_count=52` / `total_bytes=10,791,780` match |

> **CORRECTION 1 (applied).** An earlier draft of the Generator diagnosis listed the receipt as
> carrying *no* manifest digest and treated "external manifest-SHA UNVERIFIED" as its sole BLOCKER. That
> is **VOID**: the receipt DOES carry `manifest_sha256=dd6f3e6c…` and it MATCHES the live manifest bytes.
> The strongest out-of-band authenticity check is **available and PASSES**; intake is fully verified.

Certificates observed: original `a9dd121c…0994b`, retry1 `cbf0e2aa…447ce`. Both ledgers consumed; neither
run promoted. No launcher invoked; no ledger deleted/recreated; nothing written to repo or Dropbox.

---

## 2. The six diverging values (cell `idealized+performat`, policy `dp`)

Identical across `qanta_phase4_windows_v3` and `qanta_phase4_windows_retry1`.

| field | type | anchor (Export-A) | regenerated (both runs) | delta |
|---|---|---|---|---|
| `signed_mean` | stat | 1.6332 | 1.6083 | −0.0249 |
| `abs_mean` | stat | 1.6394 | 1.6198 | −0.0196 |
| `mc_earlier` | count | 4 | 8 | +4 |
| `qa_earlier` | count | 1915 | 1886 | −29 |
| `same_step` | count | 330 | 355 | +25 |
| `signed_mean_ci` | stat (boot) | [1.5785, 1.6914] | [1.5526, 1.6652] | shifted down |

Count conservation: −29 (`qa_earlier`) +25 (`same_step`) +4 (`mc_earlier`) = 0; n = 2249 invariant.
Interpretation: on ~29 borderline items the regenerated QA arm buzzes marginally later (or MC marginally
earlier), moving them from `qa_earlier` (v>0) into `same_step` (v=0) and a few into `mc_earlier` (v<0),
where `v := mc_stop_step − ref_stop_step`.

---

## 3. Four-way ledger (keep these strictly separated)

### (a) Immutable observations from the two runs
- Both runs: STOP_REPORT `checked=194`; exactly the six failures above, all `cell=idealized+performat`,
  `policy=dp`. The other 188 fields pass (every other performat cell and both idealized+shared policies).
- Records `records/idealized__format_specific.jsonl` **byte-identical across both runs**
  (`4a179b68…24bf`), 2249 rows → not run variance.
- Regenerated result JSONs are semantically identical on all 194 parity fields; they byte-differ only in
  embedded run-identity/provenance metadata (v3 `dd2333dd…9444`, retry1 `af224503…6bfc`).
- Both ledgers consumed; neither run promoted.

### (b) Historical anchor claims
- Anchor `expected["idealized+performat"]["dp"]` = 1.6332 / 1.6394 / 4 / 1915 / 330 / [1.5785,1.6914],
  copied verbatim out of Export-A `stopdff_fair_qa.json`. Anchor sha256 `2efff657…973eee`.
- Producer commit `4bf5e02d` (2026-06-12) is the commit that *added* Export-A; its `dp` block equals the
  anchor. The anchor honestly records what Export-A produced (no transcription/edit error).

### (c) Verified causal findings (byte-grounded)
- **F1 — anchor is a faithful Export-A copy.** `phase4_generate_freeze.py` copies Export-A's per-cell
  `dp`/`myopic` blocks directly; the anchor's `idealized+performat.dp` equals Export-A exactly. → rules
  out H1-as-corruption.
- **F2 — regenerated == exact recompute of records.** Recomputing all six fields (plus medians,
  never-buzz, and the seed-1 / `num_boot=1000` bootstrap CI) from the 2249 records, mapping sentinel
  `null` stop steps to `trajectory_horizon`, reproduces the regenerated values **exactly** for both runs,
  by two independent implementations. → rules out H2 on the current side.
- **F3 — fault isolated to `performat × dp`.** `idealized+performat/myopic` ALL-MATCH; `…/dp` 6-DIFF;
  `idealized+shared/{myopic,dp}` ALL-MATCH. The fault requires two conditions simultaneously: (1) the
  per-`(format,bucket)` performat calibrator (shared mode pools formats and stays non-degenerate → immune)
  and (2) the DP policy (myopic has no continuation stage → immune).
- **F4 — mechanism of the degeneracy.** `score_arms` labels every idealized QA row `"correct": 1`, so
  `fit_performat` hits its single-class fallback `const = np.clip(1.0, 1e-4, 1-1e-4) = 0.9999` (exact,
  platform-independent) and `apply_cal` clips to the same constant. With QA confidence pinned at ~1.0 the
  raw QA `ref_stop` distribution is `{0:2178, 1:34, 2:28, 3:4, 4:2, 5:2, 6:1}` — **2178/2249 = 96.8% buzz
  at step 0** — putting the DP stop-at-0-vs-1 partition on a tie boundary hypersensitive to sub-ULP shifts
  in the continuation estimate. The well-conditioned nonrandom cells absorb the same drift via
  `round(·,4)` and integer-count stability → they pass.
- **F5 — value path behaviorally unchanged (⇒ environment); DP core byte-identical.** The full DP-input
  chain is empty-diff `4bf5e02d`→pin (`dp_solver`, `continuation`, `compute_prefix_calibration`, adapter,
  `rewards`, `types`, `_common`, `diagnostics`, `_provenance`), and `fit_performat` / `apply_cal` /
  `summarize` are AST-identical. Identical source logic on the exercised path + identical pinned
  config/seeds + identical eligible population (n=2249), yet different deterministic DP output ⇒ the
  residual difference enters through a **non-source input**, i.e. the execution environment / library
  stack (a bleeding-edge 2026 library stack: numpy 2.4.6, scipy 1.17.1,
  scikit-learn 1.8.0, sentence-transformers 5.5.1, transformers 5.9.0, torch 2.12.0 — the only in-repo
  successor certificate, `pre_run_ready_certificate_2026-08-22.json`, records env **macOS arm64 / Apple
  M3 Max** (Darwin, cpython-3.11.15-macos-aarch64) and binds via `environment_lock_sha256` a pip-freeze
  (`phase4_pre_receipts/environment_lock_pip_freeze.txt`) whose versions match this exact list; the
  actual v3/retry1 runtime OS is **not** established from in-repo bytes — those runs' certs `a9dd121c…` /
  `cbf0e2aa…` live in the transport bundle and were not inspected). numpy
  summary/bootstrap is platform-stable (the Verifier reproduced the runs' CI **exactly** on
  macOS numpy 2.4.6) → aggregation is not the channel; the drift is upstream in record generation. **This
  is an INFERENCE, not byte-proof** — Export-A's environment is absent from the evidence and the model
  cannot be re-run here (see §5). *(corrected 2026-08-24: prior "Windows stack" / "Windows-produced"
  wording was imprecise for persisted evidence — the only in-repo successor cert
  `pre_run_ready_certificate_2026-08-22.json` records macOS arm64 / Apple M3 Max with a version list
  matching the one cited (via hash-bound `environment_lock_pip_freeze.txt`); the v3/retry1 runtime OS is
  not established from in-repo bytes, as their transport-bundle certs were not inspected.)*
- **F6 — determinism within the pinned environment.** Byte-identical records across the two runs isolate
  the problem to the {records + current producer} vs {anchor} axis, not runtime randomness.

> **CORRECTION 3 (applied).** "byte-for-byte unchanged source" / "every hot-path function logically
> identical" is **overstated**. `bootstrap_ci` (added `num_boot<=0` guard), `score_arms` (per-item
> krandom RNG + validation guards; MC/idealized untouched), `signed_per_item` (additive collectors +
> label-only coverage tagger + `break`→`raise` guarded out upstream), and `main` **do** differ between
> `4bf5e02d` and the pin. **None** of those diffs touch `idealized+performat·dp` value production. The
> correct phrasing is **"behaviorally unchanged on the exercised path,"** used throughout F5.

### (d) Proposed repairs requiring authorization
See §4. None executed; all are proposals only.

---

## 4. Smallest safe repair plan (all [REQUIRES AUTHORIZATION]; NONE executed)

- **(A) RECOMMENDED — scoped known-divergence / tolerance for `idealized+performat·dp`.** Record a
  narrow, documented tolerance for this degenerate knife-edge cell, analogous to the existing frozen
  policy that treats Random-K numerical divergences as informational-only. Carried via the pending spec
  amendment plus a provenance note. **Smallest change; no model re-run; preserves the anchor's historical
  values** rather than overwriting a frozen artifact.
- **(B) Environment reproduction.** Pin Export-A's model/library stack and re-run to match the anchor.
  Stronger guarantee, but Export-A's environment is **absent from the evidence** and this needs a full
  retry ceremony (see below).
- **(C) De-degenerate the idealized QA calibrator (the root amplifier).** NOT the smallest change: it
  alters producer semantics and would re-anchor additional cells. Flag as a **substantive latent finding**
  worth a research decision regardless of A/B.

**Retry ceremony (applies to any option that re-runs).** Any retry #2 requires a **fresh PRE_RUN_READY
certificate**, **fresh ledger / quarantine / promotion paths**, and a **new digest-specific user
activation**. This reconciliation is evidence-transfer authority only, not execution authority. The
retired certificate `8731ad00…` must not be reused, and no consumed ledger may be reused; there is **no
currently usable certificate**.

---

## 5. Uncertainty (not provable from bytes)

- **Environment/library drift is inferred, not proven.** Export-A's environment is not in the evidence
  bundle or cited repo artifacts; the F5 attribution rests on source-invariance on the exercised path, not
  a direct env diff. Population invariants (n=2249; `mc_never_buzz=0.0209`; `qa_never_buzz=0.0`) all equal
  the anchor, indicating the input population is unchanged, but an unobserved data/checkpoint difference
  cannot be byte-excluded.
- **Exact drift channel is unlocalized.** Candidates: value-iteration init/order; MC-val-derived pooled
  fallback means feeding the QA arm at the ~1.34% pooled lookups; SBERT / cos-sim / Platt-LBFGS drift in
  the MC calibration seeding the continuation table. Confirming this needs an instrumented, authorized
  re-run — prohibited here.
- **Which value is "correct"** (Export-A's timing vs the pinned producer's) is a policy decision, not a
  byte fact; hence H3 (mechanism) vs H1 (remediation framing) cannot be collapsed from bytes alone.

---

## 6. Working-tree safety at diagnosis time

- `git status --porcelain`: **0 tracked add/mod/del/rename** (all entries `?? `). The frozen anchor
  (`2efff657…973eee`), spec, certificates, ledgers, exchange bundle, `phase4_exchange.py`, and records are
  unchanged.

> **CORRECTION 2 (applied).** "only three untracked files" is **wrong**: **~28 untracked entries** exist
> in the repo (pre-existing dev notes / diffs / transcripts predating the runs, e.g.
> `phase4_exchange.py`, `tests/test_phase4_exchange.py`, `phase4_exchange_spec_amendment_pending.md`, and
> ~25 others). The safety property (no governed tracked write) still holds; the untracked count has no
> safety impact.

---

## 7. Cited artifacts (spot-checkable)

Repo (read in place, == pin `7a1d0220…`, tree `cea970fe…`):
- `scripts/stopdff_fair_qa_retest.py` — `score_arms` idealized `"correct": 1` (:177); `fit_performat`
  const-0.9999 branch (:187); `apply_cal` (:202); `signed_per_item` (:216); `bootstrap_ci` (:283);
  `summarize` (:298).
- `scripts/stopdff_dp/{dp_solver.py,continuation.py}`, `scripts/compute_prefix_calibration.py` — no diff
  `4bf5e02d`↔pin.
- `reproducibility/colm_aims_2026/{phase4_generate_freeze.py,phase4_records.py,pairing.py}` — anchor =
  verbatim Export-A copy; `null`↔horizon sentinel mapping; `pairing.sentinel_coded_stop` (:233), shift
  vector (:589).
- `reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json` — sha256 `2efff657…973eee`.
- `reproducibility/colm_aims_2026/phase4_exchange.py` — sha256 `93fd54d5…78feb8` (verified).
- Producer commit `4bf5e02d5447202a1a39f2e86c948ecb9a1614b8` (added Export-A + retest script).

Machine-local verified intake (transport; not persisted here):
- both runs' `output/…/STOP_REPORT.json` — `checked=194`; six failures `idealized+performat/dp`.
- `records/idealized__format_specific.jsonl` — sha256 `4a179b68…24bf` (identical both runs), 2249 rows.
- regenerated `stopdff_fair_qa_regenerated.json` — v3 `dd2333dd…9444`, retry1 `af224503…6bfc`.
- sibling `…PUBLISH_RECEIPT.json` `manifest_sha256=dd6f3e6c…385f55` (== live `RETURN_MANIFEST.json`,
  size 11916); certificates original `a9dd121c…0994b` / retry1 `cbf0e2aa…447ce`.

---

## 8. Corrections summary (Verifier `2801d4b8` caught these; applied here, DIAGNOSIS.md not copied verbatim)

1. Receipt **does** carry `manifest_sha256=dd6f3e6c…` (== live manifest) → the Generator's sole BLOCKER
   is VOID; intake is fully verified incl. the strong out-of-band authenticity check.
2. "Only three untracked files" is wrong → **~28** untracked entries exist (pre-existing); safety intact.
3. "byte-for-byte unchanged source" is overstated → correct phrasing **"behaviorally unchanged on the
   exercised path"** (`bootstrap_ci` / `score_arms` / `signed_per_item` / `main` differ, none on this
   cell's value path).

Everything else in the Generator diagnosis reproduced exactly under independent recomputation (the six
values, all hashes, the QA distribution, localization, DP-core no-diff, anchor/cert digests). The
companion independent-verification report is persisted alongside this file at
`phase4_reconciliation_verification_2026-08-24.md`.
