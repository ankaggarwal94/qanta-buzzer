# Phase-4 forensic reconciliation — INDEPENDENT VERIFICATION report (2026-08-24)

**From:** independent read-only Verifier lane `2801d4b8`, Device 1 (MacBook Pro M3 Max, macOS). Repo env:
`.venv` CPython 3.11.15, numpy 2.4.6. Cross-checked by a parent hash spot-check.
**To:** the author and the next Phase-4 agent in this repo.
**Re:** independent re-derivation of every load-bearing claim in the Generator lane `5edea440`
reconciliation of the six `idealized+performat`·DP parity failures.
**Method:** re-derived each claim from bytes; did **not** trust the Generator's machine-local
`DIAGNOSIS.md`. READ-ONLY, fail-closed, evidence-transfer authority only. This is a NEW persisted
companion to `phase4_reconciliation_diagnosis_2026-08-24.md`; the bulky verified-intake bundle stays in
transport (Dropbox exchange bundle) and is not copied here.

## Status line

`OVERALL: DIAGNOSIS_SOUND · H2_RULED_OUT (CONFIRMED) · H3_MECHANISM (CONFIRMED, env-drift = INFERENCE) ·
H1_REMEDIATION_FRAMING (CONFIRMED) · 3 PRIOR-REPORT ERRORS FOUND & CORRECTED · SIX VALUES REPRODUCED
INDEPENDENTLY (v3 == retry1) · MODELS_EXECUTED: NONE · WRITE_GATES: GOVERNED EVIDENCE READ-ONLY`

---

## 1. Per-claim verdicts

| Claim | Verdict | One-line byte-grounded evidence |
|---|---|---|
| **E — intake** | **CONFIRMED (+ prior error found)** | `phase4_exchange.py` sha256 `93fd54d5…78feb8`; `pull_verified`→52 files / 10,791,780 B; HEAD `7a1d0220…`, tree `cea970fe…`. **Receipt DOES carry `manifest_sha256=dd6f3e6c…385f55` (== live manifest) + `manifest_size=11916`** — the prior report said it did not. |
| **F — safety** | **PARTIAL** | `git status --porcelain`: **0 tracked** add/mod/del/rename (all `?? `); anchor `2efff657…973eee` unchanged; bundle + manifest re-hash clean. But prior "only 3 untracked files" is wrong — **~28 untracked** entries exist (pre-existing). Safety property (no governed tracked write) holds. |
| **A — H2 ruled out** | **CONFIRMED** | Independent recompute from records reproduces the **regenerated** six EXACTLY, both runs (table below). Records byte-identical `4a179b68…24bf`. |
| **B — no code edit (this cell)** | **CONFIRMED core / prior wording REFUTED** | Full DP-input chain empty-diff `4bf5e02d`→pin (`dp_solver`, `continuation`, `compute_prefix_calibration`, adapter, `rewards`, `types`, `_common`, `diagnostics`, `_provenance`); `fit_performat` / `apply_cal` / `summarize` AST-identical. **But `bootstrap_ci` / `score_arms` / `signed_per_item` / `main` DO differ** — prior "byte-for-byte unchanged" overstated; none touch this cell's value path. |
| **C — degenerate calibrator** | **CONFIRMED** | `score_arms` hardcodes idealized `"correct": 1` (retest:177, both eras) → `fit_performat` const `np.clip(1.0,1e-4,1-1e-4)=0.9999`. QA step-0 fraction from records = 2178/2249 = **0.9684**; dist `{0:2178,1:34,2:28,3:4,4:2,5:2,6:1}`. |
| **D — localization** | **CONFIRMED** | Both runs: `idealized+performat/myopic` ALL-MATCH; `…/dp` 6-DIFF; `idealized+shared/{myopic,dp}` ALL-MATCH. STOP_REPORT `checked=194`, all six failures `cell=idealized+performat, policy=dp`. |

---

## 2. Recomputed six — independent vs regenerated vs anchor (identical for v3 and retry1)

| field | recomputed (independent) | regenerated | anchor |
|---|---|---|---|
| `signed_mean` | 1.6083 | 1.6083 | 1.6332 |
| `abs_mean` | 1.6198 | 1.6198 | 1.6394 |
| `mc_earlier` | 8 | 8 | 4 |
| `qa_earlier` | 1886 | 1886 | 1915 |
| `same_step` | 355 | 355 | 330 |
| `signed_mean_ci` | [1.5526, 1.6652] | [1.5526, 1.6652] | [1.5785, 1.6914] |

Signed value `d_i = sentinel_coded_stop(mc) − sentinel_coded_stop(ref)`, coded = `horizon` if
null/NEVER else `min(stop,horizon)` (`pairing.py:233,589`). Counts: `mc_earlier=#(d<0)`,
`qa_earlier=#(d>0)`, `same_step=#(d==0)`; `round(mean,4)`. The CI was reimplemented independently but
necessarily uses the repo method — `np.random.default_rng(seed=1)` + `integers(0,n,n)`, `num_boot=1000`,
`np.percentile` 2.5/97.5, `round(·,4)`. It reproduced the Windows-produced CI **EXACTLY on macOS numpy
2.4.6** → the numpy summary/bootstrap stage is platform-stable; any drift is upstream in record
generation, not in aggregation.

---

## 3. Overall verdict — the diagnosis is SOUND (agree with operative conclusions)

- **H2 correctly ruled out.** The regenerator is faithful to its records, and Export-A@`4bf5e02d` was
  produced by an AST-identical `summarize` → both sides share export semantics. **AGREE.**
- **H3 supported as mechanism**, refined: the value path *for this cell* is behaviorally unchanged (all
  DP deps byte-identical), so divergence enters via a non-source input (model/library stack feeding the
  degenerate-QA DP knife-edge). The environment/library float-drift attribution is an **INFERENCE, not
  byte-proof** (the prior report honestly flags this). **AGREE w/ caveat.**
- **H1 = remediation framing.** Anchor == Export-A byte-for-byte on this cell → not corruption.
  **AGREE.**

---

## 4. Discrepancies found vs the Generator's DIAGNOSIS.md (the three corrections)

1. **Receipt manifest digest.** Prior: receipt "carries no manifest digest" → strong out-of-band check
   "UNVERIFIED" and listed as the sole BLOCKER. **FALSE:** `manifest_sha256` / `manifest_size` are present
   AND match the live manifest. The check is available and **PASSES**; the prior's sole blocker is void.
2. **Untracked count.** Prior: "only expected untracked files present (three)." Actually **~28** untracked
   entries (dev notes / diffs / pre-receipts predating the runs). No safety impact.
3. **"byte-for-byte unchanged."** Overstated: `bootstrap_ci` (added `num_boot<=0` guard), `score_arms`
   (per-item krandom RNG + validation guards; MC/idealized unchanged), `signed_per_item` (additive
   collectors + label-only coverage tagger + `break`→`raise` guarded out by upstream eligibility), and
   `main` differ. **None** affect `idealized+performat·dp`. Correct statement: **"behaviorally unchanged
   on the exercised path."**

Everything else in the prior report reproduced exactly (six values, hashes, distribution, localization,
DP-core no-diff, anchor/cert digests).

---

## 5. Residual uncertainty / blockers (fail-closed)

- Environment/library drift is **inferred, not proven**: Export-A's env is not in evidence; the model
  cannot be re-run (prohibited + no CUDA). Population invariants (n=2249, `mc_never_buzz=0.0209`,
  `qa_never_buzz=0.0`, all == anchor) indicate an unchanged input population, but an unobserved
  data/checkpoint difference cannot be byte-excluded.
- Both Windows runs produced byte-identical records → deterministic within the environment (not runtime
  randomness).
- **No open blocker; no prohibition conflicts.** No writes to repo/Dropbox; evidence read only from the
  machine-local intake.

---

## 6. Cited artifacts (spot-checkable)

- Exchange impl: `reproducibility/colm_aims_2026/phase4_exchange.py` sha256 `93fd54d5…78feb8`.
- Receipt: `…/phase4_exchange/phase4_run_and_retry1_fail_closed_2026-08-24.PUBLISH_RECEIPT.json` →
  `manifest_sha256=dd6f3e6cd918463c21503dac44a3d9b00e021f05ea2d1a30c5afcb0a36385f55`,
  `manifest_size=11916`, `manifest_entry_count=52`, `manifest_entry_total_bytes=10791780`.
- Live manifest `RETURN_MANIFEST.json` sha256 `dd6f3e6c…385f55`, size 11916 (== receipt).
- Anchor `reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json` sha256 `2efff657…973eee`;
  `expected[idealized+performat][dp]` = the six anchor values.
- Records (both runs) `records/idealized__format_specific.jsonl` sha256 `4a179b68…24bf`, 2249 rows.
- Regenerated `stopdff_fair_qa_regenerated.json` v3 `dd2333dd…9444`, retry1 `af224503…6bfc`.
- STOP_REPORT v3 (cert `a9dd121c…0994b`) / retry1 (cert `cbf0e2aa…447ce`): `checked=194`, six failures
  `idealized+performat/dp`.
- Source: `scripts/stopdff_fair_qa_retest.py` :177 (`"correct": 1`), :187 `fit_performat`, :202
  `apply_cal`, :216 `signed_per_item`, :283 `bootstrap_ci`, :298 `summarize`;
  `reproducibility/colm_aims_2026/pairing.py` :233 `sentinel_coded_stop`, :589 shift vector.
- Producer commit `4bf5e02d5447202a1a39f2e86c948ecb9a1614b8` (2026-06-12) added Export-A (dp block ==
  anchor) + the retest script.

---

## 7. Implicit decisions (another verifier might differ)

- Treated null `stop_step` (NEVER) as coded→horizon per `pairing.sentinel_coded_stop`; this matched the
  regenerated aggregates exactly, validating the interpretation.
- Used AST-dump (docstring-stripped) as the behavioral-equality oracle (ignores comments / whitespace /
  formatting), plus git empty-diff for dependency modules.
- Accepted `4bf5e02d` as the anchor-era producer because it is the commit that ADDED Export-A itself.
- Graded claim B "CONFIRMED core" because the value path for this cell is unchanged, while explicitly
  refuting the prior report's absolute wording.

**Companion diagnosis:** `phase4_reconciliation_diagnosis_2026-08-24.md` (same repo root).
