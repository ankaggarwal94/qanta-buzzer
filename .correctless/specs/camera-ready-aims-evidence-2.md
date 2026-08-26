# Spec: Camera-Ready AIMS Evidence Flow v2 (frozen-contract successor)

## Metadata

- **Task**: camera-ready-aims-evidence-v2
- **Date**: 2026-08-20
- **Branch**: `feature/camera-ready-aims-v2`, cut from exact canonical `main` =
  `5cafae3c4fc3dfd1525975aa34fda77975b7f4cb` (sign-off Phase 1: never branch
  from #32, #41, #42, `2709624b`, or `f8ba2042`). The v1 implementation is NOT
  in this tree.
- **Governing contract — WINS on any conflict with any other source**:
  `contract_freeze_signoff_2026-08-20.md` (verbatim reviewer sign-off §§1–9 +
  probe postscript confirming A′-F1/F2). Dropbox-relative path:
  `Stanford/CS234/final_project/qanta-buzzer/contract_freeze_signoff_2026-08-20.md`.
  SHA-256 `40f02d2ad561a682e4011f55f0b083e1cfe6035af18e723c0da7f1b7a90a5749`.
  Status line issued: `CONTRACT_FREEZE: SIGNED_WITH_AMENDMENTS ·
  TRACK_A_PRIME: PARTIAL_PASS_REIMPLEMENT · A_2_SUCCESSOR: UNBLOCKED ·
  PR42_TRACKER_WRITES: DEFERRED_UNTIL_SUCCESSOR_EXACT_HEAD_GREEN · QA_012:
  UNVERIFIED_RELEASE_BLOCKER · CAMERA_READY_CLOSURE: NOT_YET_SATISFIED`.
- **Amendment chain (later amends earlier; the sign-off amends everything)**:
  1. `handoff_prompt_camera_ready_2026-08-18.md` — originating engagement
     contract (SHA-256
     `a9d69495423545324a75ebd9021e1151c9eebd60d072c6b818a6debe3f9335f6`);
     `EXECUTION_MODE=SOURCE_CONTRACT_ONLY`; §8 pinned semantic block; §6
     ledger row schema; rights posture; vocabulary rules.
  2. v1 spec (39 rules):
     `feature/camera-ready-aims-spec:.correctless/specs/camera-ready-aims-evidence.md`
     (git blob `af3962bbfa2a16b1137729e74c198ed97e07efdb`).
  3. `camera_ready_plan_v3_2026-08-19.md` §3 (contract freeze v1) + §5
     (successor amendment list) (SHA-256
     `5dc692e9a03d08b4be64c95e389ab398392e68f0efb1441db594d5fd86c04a1a`).
  4. `decision_record_D7_D8_A1_A2_2026-08-19.md` — D7(b), D8, A-1, A-2
     (SHA-256
     `5f49b441b2d5bdcbbb20fa4e572ce8be255dc404897f7d5b523da8a92d834ded`).
  5. The governing sign-off (above).
- **Implementation reference — REIMPLEMENT, never cherry-pick (sign-off §6 +
  Phase 1)**: `track_a_commit_2709624b.diff` (SHA-256
  `b74fc6d7f241b51ee959879ad240599f66519670feede91069b00124a6121361`),
  `track_a_prime_commit_f8ba2042.diff` (SHA-256
  `ba1cb2d2ee50085f3c37923eb9385b8565f5f20d5d833d455f9068d9d620ba4a`), and the
  v1 namespace at `feature/camera-ready-aims-spec:reproducibility/colm_aims_2026/`
  (`schema.py`, `verifier.py`, `pairing.py`, `ledger.py`, `receipt.py`,
  `render.py`, `verify.py`). The local commits are review inputs, not
  transplantable authority; successor tests are re-derived against this spec;
  the Track A / Track A′ exploit files are not ported wholesale.
- **D6 designated manuscript baseline** (two-party hash-verified): `main.tex`
  SHA-256 `79dccfb3fbdfafbd566a3fb239755ab35142bac510d629d513ed8b3c2c4cdd2f`;
  rendered `main.pdf` SHA-256
  `6de23119df59679befc356e3c916bc5a498b2cc2015b6cd8a516a5181dabf10a`; complete
  checksum closure `FINAL_CHECKSUMS.sha256`, all in
  `Stanford/CS321M/final_project/does_proxy_preserve_decision_aims_colm2026_tier1_revision_bundle/`.
  Closure duties bind the COMPLETE manifest (figures, bibliography), never
  `main.tex` alone (plan v3 §1 qualifications, adopted).
- **Numbering**: v1 rule IDs `R-001..R-039` are retained for carried/amended
  rules (Appendix A maps all 39); new v2 rules start at `R-040`. Appendix B
  maps the sign-off Phase-2 RED coverage list to rule IDs.

## What

The COLM 2026 AIMS workshop camera-ready (non-archival; deadline 2026-09-21)
is supported by a repo-side evidence contract whose defensible claim is a
**constructed-reference sensitivity audit** — never observed open-ended
decision preservation. v2 is the successor reconstruction authorized by A-2:
one coherent contract introduced fresh on canonical `main` (no stacking atop
the v1 tower), under the externally signed frozen contract. It reimplements
the v1 evidence flow with: the frozen 5×2 ten-cell single-package data model
(D1 + sign-off §2.1); the canonical `FINITE_STOP`/`NEVER_STOPPED` event
representation (§2.2); the sentinel-coded all-pair historical headline
estimand (§2.3); the exact D7(b) regenerated inference procedure (§3);
one-token schema v2 versioning with a single bool-safe checker (§4.1/4.2);
field-specific integer domains (§4.3, D8); closed-map discipline (§4.4); the
pinned R2 legacy-sidecar boundary (§5, ASK-2(a)); the approved Track A and
Track A′ repairs as v2-native rules (§6); R6 canonical-selection wiring into
the actual release path (Phase 3); successor CI evidence receipts (A′-F4);
and the `CAMERA_READY_CLOSURE` gate (D4, distinct from `PASS_RELEASE`) frozen
against the D6 baseline's complete checksum closure. Everything new lives
under `reproducibility/colm_aims_2026/`; source-contract mode still tops out
at `PASS_SOURCE_ONLY`; release mode still fails closed without independently
anchored expectations. `EXECUTION_MODE=SOURCE_CONTRACT_ONLY` stands: the
single deterministic Phase-4 regeneration (package, D7(b) inference, ledger,
expectations, rights inventory, manifest, renderer outputs) is resampling
analysis over retained per-item records — permitted; model execution is not.

## Rules

### Vocabulary and scope (sign-off §1 — immutable)

- **R-027** [unit] *(amended by sign-off §1 and §2.3)*: No rendered output,
  verdict string, ledger text, receipt, or doc produced by this feature may
  violate the immutable scope: (a) the historical and successor packages
  concern constructed-reference sensitivity evidence; (b) they do not contain
  observed open-ended responses or observed open-ended stopping actions;
  (c) they do not establish actual MC-versus-QA decision preservation;
  (d) they do not establish a causal response-format effect;
  (e) `PASS_SOURCE_ONLY` never implies camera-ready or release certification.
  No manuscript, report, verifier output, PR body, or comment may silently
  upgrade that scope. The v1 enforcement core carries: an enumerated
  banned-phrase and required-qualifier fixture asserted over every renderer
  output; unqualified "QA effect" and observed-preservation phrasings are
  banned; "would hide real shifts"-style causal wording narrows to
  insensitivity-to-sub-threshold-shifts. If the intended headline becomes
  actual preservation/change, the only sanctioned output is
  `observed_paired_claim=OBSERVED_PAIRED_STUDY_REQUIRED` (handoff §10).
  Enforcement: vocabulary fixture over all renderers + the R-048 label gate.

### Profile and semantics

- **R-001** [unit] *(amended by D1 and sign-off §4.1)*: The writer emits only
  the strict v2 profile: `schema_version: 2`; `profile_id:
  "colm_aims_constructed_reference_v2"` (spec-pinned successor of the v1 ID);
  the pinned semantic block verbatim from handoff §8 —
  `trajectory_source: constructed_reference`, `observed_open_ended: false`,
  `observed_open_ended_answers: false`,
  `observed_open_ended_stopping_actions: false`,
  `pairing_unit: matched_item_prefix_grid`,
  `pairing_is_observed_sessions: false`,
  `supports: reference_sensitivity_diagnostic`,
  `does_not_support: actual_decision_preservation_or_format_effect` — and the
  D1 multi-cell package shape: one package containing a `grid` block, a
  calibration-identity MAP (scalar→map: one entry per calibration ID), an
  in-profile `inference` block (R-050..R-056), and per-cell record files
  `records/<cell_id>.jsonl`. Top-level keys are a closed set (R-063) that
  includes `grid` and `inference`. The validator rejects a missing, renamed,
  altered, or unknown key inside the semantic block; historical
  `format="QA"` identifiers parse only in the legacy loader (R-060) and never
  substitute for the semantic layer. Enforcement: schema constants +
  mutation fixtures per field.
- **R-002** [unit] *(carried)*: A constructed-reference artifact asserting
  `observed_open_ended: true` fails validation. The reserved observed-study
  profile identifier remains distinct; the constructed-reference validator
  never accepts it and no code path converts one profile into the other.
- **R-003** [unit] *(amended by sign-off §2.2 / plan v3 §3.2)*: Every arm
  identifies its construction, scalar-vs-K-way status, selector/scorer,
  candidate-pool role, correctness assignment, calibration/continuation role,
  seed contract, and reporting eligibility. `idealized` is scalar
  prefix-to-gold cosine with oracle-assigned correctness; a payload declaring
  all arms K-way while containing an idealized arm fails. The
  constructed-reference, fixed-threshold, myopic, and learned-continuation
  families each carry a qualified CLOSED vocabulary — no overloaded global
  stop integer across families. Enforcement: per-family enum constants +
  cross-family confusion fixtures.
- **R-004** [unit] *(carried)*: encode→decode roundtrip of a strict-profile
  artifact is lossless (equal value); writers reject non-finite floats at
  write time (`allow_nan=False` semantics).
- **R-029** [unit] *(carried)*: The strict profile carries the
  `llm_involvement` block per contribution axis (reference construction,
  data/plot creation, evaluation); `none` is an explicit value, never an
  absent field; free-text tool/version note required when any axis is
  non-none; artifacts missing the block are rejected.
- **R-030** [unit] *(carried)*: Ledger `archival_doi` support: an
  Available-grade assertion without a DOI-class archival identifier fails
  validation; a GitHub URL does not qualify; absent any Available-grade
  assertion the field is optional.
- **R-031** [unit] *(amended by sign-off §2.2)*: Per-item records are
  non-reversible: opaque stable keys, enumerated categorical fields, numeric
  values only; the validator rejects any string field outside the enumerated
  identifier allowlist. v2 extends the allowed field set for the canonical
  event representation: per-arm `event_status` categoricals (closed enum
  R-045), nullable stop steps (`null` allowed exactly and only for
  `NEVER_STOPPED`, R-061), and `terminal_imputation` categoricals. No
  free-text field may enter records. Enforcement: allowlist constant +
  free-text fixture → FAIL.
- **R-032** [unit] *(carried)*: The schema pins a maximum admissible
  numerical tolerance; a declared tolerance above it fails validation; the
  declared tolerance is an estimand-defining digest field (R-011).

### Grid identity (sign-off §2.1)

- **R-040** [unit]: The strict package contains exactly the Cartesian product
  of the five approved constructed references and the two calibration maps
  `shared` and `format_specific`: exact-set equality on the reference-ID
  axis (cardinality 5), the calibration-ID axis (cardinality 2, exactly
  `{"shared", "format_specific"}`), and the ten derived cell IDs
  (cardinality 10). A missing, extra, duplicated, or renamed axis member or
  cell fails closed. Enforcement: exact-set comparisons (never subset), one
  mutation fixture per failure class. (Sign-off §2.1; cardinalities R-061.)
- **R-041** [unit]: Cell↔record-file bijection: exactly one
  `records/<cell_id>.jsonl` per declared cell; no missing, duplicate,
  undeclared, or orphaned record file. Present-but-undeclared and
  declared-but-absent both FAIL (two directions, R-035). Enforcement:
  directory↔grid reconciliation in both modes.
- **R-042** [unit]: Every cell contains exactly 2,249 complete paired item
  keys, and the item-key SET is byte-exact identical across all ten cells.
  Any exclusion inside the frozen package breaks this gate by construction —
  there is no partial-population escape hatch. Enforcement: per-cell count +
  cross-cell set-equality legs; ±1 and one-key-swapped fixtures FAIL.
  (Sign-off §2.1; carries the v1 R-008 key-set discipline.)
- **R-043** [unit] *(amended by PRE-2, 2026-08-22)*: Held-fixed identities:
  the same raw MC trajectory identity and horizon identity wherever the
  contract declares them held fixed; the horizon identity IS the canonical
  per-item horizon-map digest defined by R-073 (a scalar-horizon
  representation is defective — the real retained trajectories are
  heterogeneous, eligible horizons 2–10); `mc_stop_step` equality across the
  five references WITHIN each calibration ID only. There is NO requirement that MC stops match between
  `shared` and `format_specific` — those conditions use separately fitted MC
  calibrators, and a verifier that enforces cross-calibration MC-stop
  equality is itself defective. Enforcement: within-calibration equality
  legs; a nearest-true fixture with differing cross-calibration MC stops must
  PASS; a within-calibration mismatch must FAIL. (Sign-off §2.1; plan v3
  §3.1 — the plan-v2 global MC-stop-map leg is retracted.)
- **R-044** [integration]: Independent grid anchoring: the out-of-tree
  expectations contract pins SEMANTICALLY — not merely via a final package
  hash — the exact reference IDs, calibration IDs, ten cell IDs,
  cell→record-file mapping, item-key-set digest, and the held-fixed
  trajectory identities. The in-package `grid` block proves only internal
  source-mode consistency and is never its own release oracle. Enforcement:
  release legs compare package grid state against the expectations pins
  field-by-field; R-021's mutation test flips each pin. (Sign-off §2.1.)

### Event representation (sign-off §2.2)

- **R-045** [unit]: The canonical event vocabulary is the closed enum
  `event_status ∈ {FINITE_STOP, NEVER_STOPPED}`. `FINITE_STOP` carries a real
  finite integer `stop_step` (domain R-061); `NEVER_STOPPED` carries
  `stop_step = null` and `terminal_imputation = FINAL_PREFIX_IF_NEVER` where
  a reporting imputation applies. Each record set additionally binds:
  the PER-RECORD trajectory horizon (`trajectory_horizon`, a positive
  integer owned by each record; heterogeneity across records is legal and
  expected, and both arms of one record share that record's horizon —
  amended by PRE-2, 2026-08-22); zero-based index convention (`index_base = 0`);
  producer/profile identity; original encoded value where historical data are
  imported; the historical sentinel convention; and the terminal-imputation
  policy. Enforcement: schema validation of every combination —
  `FINITE_STOP` with null/missing stop, `NEVER_STOPPED` with numeric stop,
  and missing bound identities all FAIL.
- **R-046** [unit]: A genuine threshold crossing at the final prefix is
  `FINITE_STOP`. A threshold never crossed, followed by final-prefix or
  horizon-sentinel reporting imputation, remains `event_status =
  NEVER_STOPPED`, `stop_step = null`, `terminal_imputation =
  FINAL_PREFIX_IF_NEVER`. The scalar consumed by the historical headline
  estimator is a DERIVED reporting encoding; it must never replace or
  overwrite the canonical event. The derived transform is declared per
  package, and for the historical ten-cell package it must reproduce the
  preserved fair-QA producer's convention (v1 lineage:
  `timeout_coded_as_horizon`, per-side `min(stop, horizon)`), locked by
  R-010 golden parity. Enforcement: a record set whose canonical events were
  overwritten by the derived scalar cannot roundtrip (distinct fields, both
  validated); final-prefix-crossing-mislabeled fixture FAILs.
- **R-047** [unit]: An ambiguous legacy `T−1` value cannot be normalized
  merely because doing so makes a table convenient: absent an authenticated
  producer convention or an explicit crossing indicator, it is excluded with
  the named enumerated reason `AMBIGUOUS_TERMINAL_SENTINEL` (spec-pinned new
  member of the exclusion-reason enum) and stays in the legacy
  representation, never silently promoted. Inside the frozen package such an
  exclusion necessarily fails R-042 — by design. Enforcement: normalization
  fixtures with and without an authenticated convention/crossing flag; the
  without-case must refuse.

### Pair and censoring decomposition

- **R-005** [unit] *(carried; recomputed per cell over
  `records/<cell_id>.jsonl`)*: The checker recomputes per-cell counts from
  per-item records and enforces exactly:
  `n_complete = n_both_finite + n_mc_finite_ref_timeout + n_mc_timeout_ref_finite + n_both_timeout`;
  `n_pairing_population = n_complete + n_excluded_or_unpaired`;
  `n_excluded_or_unpaired = sum(exclusion_reason_counts.values())`;
  `n_mc_timeout = n_mc_timeout_ref_finite + n_both_timeout`;
  `n_ref_timeout = n_mc_finite_ref_timeout + n_both_timeout`.
  A ±1 mutation of any recorded count fails.
- **R-006** [unit] *(amended by sign-off §2.3)*: All four joint-class rates
  use `n_complete` as denominator, are `null` when it is zero, and otherwise
  sum to 1 within the declared tolerance. `n_pairing_population == 0` is a
  typed error, never a trivially-passing cell. Role reassignment under §2.3:
  the sentinel-coded all-pair statistic is the HEADLINE historical estimand
  (R-048); the finite-only summary is a SEPARATELY NAMED secondary estimand
  (R-049) that uses exactly `n_both_finite` and declares its conditional
  population. Neither is ever pooled with the other.
- **R-008** [unit] *(amended by R-042)*: Complete-pair and excluded/unpaired
  key sets are disjoint and duplicate-free; their union equals the declared
  pairing-population key set and hash. Duplicate pair keys fail closed. Each
  excluded unit carries exactly one primary exclusion reason; missing reasons
  are `UNKNOWN_NOT_INFERRED`, never guessed; secondary diagnostics are not
  counted in `exclusion_reason_counts`. The profile pins one of two exact,
  closed item-key schemes: the generic NFC/SHA-256 opaque-text derivation, or
  the Phase-4 canonical unsigned-decimal dataset-QID identity required by the
  frozen R-074 population. The verifier validates every record key against the
  declared scheme; keys compare byte-exact; generic-scheme duplicate fixtures
  include Unicode normalization-variant near-duplicates.
- **R-009** [unit] *(carried)*: Arm reversal leaves the diagonal joint
  classes unchanged, exchanges the two off-diagonal classes and the two
  timeout totals, and flips a timing-summary sign only when that summary's
  frozen definition is antisymmetric (property-based over generated tiny
  record sets).

### Estimands and historical parity (sign-off §2.3)

- **R-048** [unit]: The historical headline estimand is the mean
  horizon-capped / terminal-imputed signed shift over ALL 2,249 complete
  pairs, defined as `MC − QA-reference`; positive values mean the QA
  reference stops earlier; negative values mean the MC trajectory stops
  earlier. Prohibited labels (each a failing fixture): it must not be called
  (a) a finite-only mean; (b) a mean over mathematical stopping times
  containing infinity; (c) an observed MC-versus-open-ended timing effect;
  (d) evidence of actual decision preservation. Enforcement: closed estimand
  label + required `population` identity field (R-054 enum) validated
  against the canonical recompute (R-068); renderer vocabulary gate (R-027).
- **R-049** [unit]: Finite-only separation: the existing both-finite interval
  path is NOT the historical headline estimand and must not survive under
  the headline label. A finite-only statistic may be emitted only as a
  separately named secondary estimand with its own denominator
  (`n_both_finite`), its own cell identity/digest, its own intervals, its own
  ledger row, and its own manuscript wording. Enforcement: a package
  carrying a both-finite population under the headline label FAILS; the two
  estimands carry distinct digests (R-011) and cannot pool.
- **R-068** [unit] *(Track A F7A, reimplemented)*: Estimand-label binding:
  every recorded estimand label and population identity is validated against
  the canonical recompute for that summary — the finite-only summary's label
  must equal the canonical finite-only recompute identity; the headline
  all-pair summary's label/population must equal the canonical
  sentinel-coded recompute identity; and any truth-set of profile estimands
  the verifier consumes is built from VALIDATED fields, never from an
  unvalidated recorded string. Enforcement: label-swap fixtures
  (headline↔finite-only) FAIL; the F7A regression shape (trusted-label
  truth-set) is a RED fixture.
- **R-010** [unit] *(amended)*: Historical parity is fixture-locked against
  goldens produced by the pinned historical implementation, never by running
  the new implementation twice. Two parity targets: (a) the finite-only
  summary reproduces the historical paired summary exactly on all-finite
  record sets (v1 duty, incl. DP/myopic paths where defined); (b) the
  headline all-pair sentinel-coded estimator reproduces the preserved
  fair-QA producer's values on golden record sets. Goldens are regenerated
  once under NumPy 2.4.6 (D5) at reconstruction and never hand-edited.
- **R-011** [unit] *(amended by plan v3 §5.7 and §3.1–§3.3)*: Every analysis
  cell carries a digest over all estimand-defining fields — now explicitly
  including reference ID, calibration ID, pairing definition, timeout
  parameters, event-representation identities (index base, horizon identity,
  sentinel convention, terminal-imputation policy), estimand `population`,
  denominator policy, declared tolerance (R-032), calibration/continuation
  identities, and Random-K draw identity. `pairing_definition` and
  `timeout_parameters.rule` take values from closed legal-value vocabularies
  and must mutually reconcile (the 7B/M3 surface). Changing any digest field
  changes the digest; pooling or comparing cells with differing digests is
  refused; `check_comparable` is PRODUCTION-WIRED — every production
  comparison/pooling site calls it (a helper with zero production callers
  does not satisfy this rule; R6 lesson). Cell identifiers are unique per
  artifact; duplicates fail closed.

### D7(b) regenerated inference (sign-off §3 — verbatim fidelity; executed once at evidence-package production)

- **R-050** [unit]: Canonical item and difference vectors, per cell:
  (1) sort the 2,249 opaque item keys in ascending UTF-8 byte order;
  (2) apply the contract's declared terminal-imputation transform (R-046) to
  each MC and reference event; (3) per-item signed difference
  `d_{c,i} = s^MC_{c,i} − s^REF_{c,i}`; (4) cell statistic
  `mu_hat_c = (1/2249) * sum_i d_{c,i}`. All ten cells use the SAME
  canonical item order. Enforcement: order-permutation fixture changes the
  recorded item-order digest and FAILs; per-cell means recompute exactly.
- **R-051** [unit]: ONE collection-level paired-item resample matrix shared
  by all ten cells: NumPy version exactly `2.4.6`; bit generator
  `numpy.random.PCG64`; construction
  `numpy.random.Generator(numpy.random.PCG64(seed))`; exactly `B=1000`
  resamples; resampling unit item/tossup with all prefixes and both arms
  clustered inside the sampled item; sample size per draw exactly 2,249;
  with replacement; dtype `numpy.int64`; generated as
  ```python
  indices = rng.integers(
      0,
      2249,
      size=(1000, 2249),
      dtype=np.int64,
      endpoint=False,
  )
  ```
  No calibration, continuation, reference, checkpoint, model-selection, or
  Random-K refitting inside any bootstrap draw. Enforcement: recorded plan
  fields are exact-match validated; a per-cell (non-shared) matrix, wrong B,
  wrong dtype, or `endpoint=True` FAILs.
- **R-052** [unit]: Deterministic seed, no outcome-dependent author choice:
  ```text
  seed_material =
      b"colm_aims_2026/v2/bootstrap_holm\0"
      + bytes.fromhex(pairing_population_keyset_sha256)

  seed =
      int.from_bytes(
          sha256(seed_material).digest()[:8],
          byteorder="big",
          signed=False
      )
  ```
  Seed-input identity is pinned and triple-bound: (a) in the frozen v2
  package the pairing population IS exactly the 2,249 complete-pair key set
  — zero in-package exclusions; the 9 upstream-unpaired items (2,258
  retained − 2,249 paired) are pre-package retention documentation recorded
  in provenance, never in-package `excluded_keys`; (b) a mandatory assert
  leg requires the declared `pairing_population_keyset_sha256` to equal the
  digest of the shared complete-key set AND to equal the expectations-pinned
  item-key-set digest (R-044); (c) this narrows the lineage term (handoff §8
  / v1 R-008: pairing population = complete ∪ excluded) for the frozen v2
  package — flagged for reviewer acknowledgment (RESOLVED-pending-ack note
  under Open Questions). BOTH the derived integer seed and its derivation
  string are recorded. Exactly ONE seed (F8A lineage: real int, bool
  rejected, uint64 domain per R-061). Enforcement: the triple-bind leg plus
  a release leg re-deriving the seed from the expectations-pinned key-set
  digest and requiring equality with the recorded integer; a package
  recording a different seed, extra seeds, a missing derivation string, or
  nonempty in-package `excluded_keys` FAILs.
- **R-053** [unit]: The package records a SHA-256 digest over the exact
  resample-index bytes, together with the dtype, shape, byte order, and the
  canonical item-order digest it covers. Enforcement: the verifier
  regenerates the matrix from the recorded seed/plan and recomputes the
  digest; any mismatch FAILs; the digest record missing any of the four
  covering fields FAILs.
- **R-054** [unit]: Confidence intervals, per cell: compute the mean signed
  difference for every shared resample; uncentered percentile interval;
  endpoints 2.5% and 97.5%; quantile `method="linear"`; UNROUNDED endpoints
  stored in the package; display rounding only in the renderer. The interval
  identity names its population explicitly via the closed enum
  `population ∈ {"all_complete_pairs_terminal_imputed", "both_finite_only"}`
  (spec-pinned); the ten headline cells use
  `all_complete_pairs_terminal_imputed` — the both-finite conditioning of
  the v1 interval recompute is retracted for headline cells (plan v3 §3.3).
  Enforcement: deterministic recomputation equality; a rounded stored
  endpoint or a headline interval declaring `both_finite_only` FAILs.
- **R-055** [unit]: Raw hypothesis tests, per cell: `H0: mu_c = 0`,
  two-sided, via null-centered paired bootstrap — `z_{c,i} = d_{c,i} −
  mu_hat_c`; the SAME resample-index matrix applied to `z`; with
  `mu0_hat_{c,b}` the resampled null-centered means,
  `p_c = (1 + sum_{b=1..1000} 1(|mu0_hat_{c,b}| >= |mu_hat_c|)) / 1001`.
  The `+1`/`1001` finite-resample correction is MANDATORY. Enforcement:
  recomputation equality; an uncorrected `x/1000` p-value FAILs (off-by-one
  fixture).
- **R-056** [unit]: Holm family: exactly the ten-cell 5×2 grid INCLUDING both
  historical Random-K cells; `m=10`; familywise alpha `0.05`; Holm step-down;
  ascending raw-p ordering; ties resolved by ascending UTF-8 byte order of
  `cell_id`; adjusted p-values computed and stored. Per cell the package
  stores `raw_p_value`, `holm_rank`, `holm_adjusted_p_value`,
  `holm_rejected`. The inference block additionally stores the complete
  ordered family and the exact rejected-ID set; the ledger names the exact
  rejected cell IDs (R-023). No selective omission of non-headline or
  inconvenient cells. Enforcement: recompute-and-compare over all stored
  fields; an m=8 family, an omitted Random-K cell, or a tie broken against
  UTF-8 order FAILs.
- **R-057** [unit]: The D7(b) outputs are explicitly a NEW analysis: no
  output, doc, ledger row, or manuscript-facing text may claim they recover
  or authenticate the historical p-values, seed, rejection set, or Holm
  ordering. Enforcement: required `analysis_provenance =
  "d7b_regenerated_2026"`-class discriminator field (exact token pinned at
  schema.py) + vocabulary fixture banning "recovered"/"original analysis"
  phrasings for the inference block.

### Versioning and revision (sign-off §4.1/§4.2)

- **R-058** [unit]: One canonical revision constant set:
  ```python
  SCHEMA_VERSION = 2
  SUPPORTED_SCHEMA_VERSION_MIN = 2
  SUPPORTED_SCHEMA_VERSION_MAX = 2
  VERIFIER_REVISION = "reproducibility.colm_aims_2026:r2"
  ```
  `NAMESPACE_REVISION` is DELETED. If a temporary compatibility alias is
  genuinely needed during reconstruction it must be
  `NAMESPACE_REVISION = VERIFIER_REVISION`, marked deprecated, never
  independently maintained. No per-surface version-constant forks (the v1
  ledger/manifest constant triplets collapse into this set). Enforcement:
  single-source constant + a grep/AST test asserting no second literal
  revision token exists in the namespace.
- **R-059** [unit]: One bool-safe version checker across EVERY versioned
  surface — strict profile; records envelope, if versioned; ledger;
  presentation manifest; expectations; rights inventory, if versioned;
  inference/grid sidecars, if separately versioned; receipts, if versioned —
  with admissibility exactly
  `type(version) is int and min_supported <= version <= max_supported`
  (a JSON Boolean is not an integer version — closes A′-F2). Validation
  order on every surface: container shape → `schema_version`
  presence/type/range → all other key and semantic checks (Track A′ R3,
  version-first). Every version error names the artifact's observed version,
  the supported range, AND `VERIFIER_REVISION` (closes A′-F1 — the v1
  manifest error omitted the revision token and its test never noticed).
  Enforcement: shared helper + per-surface tests asserting the exact
  canonical revision token in the message; combined-defect precedence tests
  (version defect + co-present content defect → the VERSION error surfaces);
  `true`/`1.0`/`"2"`/`3` matrix per surface.
- **R-060** [unit]: Strict v2 loaders never silently accept v1 documents.
  Historical v1 artifacts enter only through an explicitly named legacy
  loader (name: OQ-V2-002) and remain historical/noncertifying unless
  independently migrated AND regenerated. Enforcement: v1-versioned fixture
  into every strict v2 loader → typed version error; the legacy loader
  accepts it but its output is barred from certification legs.

### Integer domains and ingress (sign-off §4.3; D8; Track A′ R4/R5; Track A F6)

- **R-061** [unit]: Field-specific integer domains (semantic validation, not
  parser policy): `index_base` exactly `0`; finite `stop_step` integer with
  `0 <= stop_step < horizon`; `NEVER_STOPPED.stop_step` exactly `null`;
  horizon a positive integer equal to the declared trajectory length; counts
  nonnegative integers bounded by and reconciled with the record population;
  bootstrap `draw_count` exactly `1000` for this profile; bootstrap seed an
  unsigned 64-bit integer; resample indices integers in `[0, 2248]`; schema
  version exactly `2`; cell/reference/calibration cardinalities exactly
  10/5/2 (the frozen grid); allocation and file-size ceilings are operational
  safeguards, never construct definitions. Bools never satisfy an integer
  domain. Enforcement: per-field boundary fixtures (both sides of each
  bound).
- **R-062** [unit]: Parse-level protection is NON-SEMANTIC only: a 100-digit
  JSON integer-token length guard applied lexically BEFORE `int()`
  conversion (closes the Track A′ R5 escape where CPython's 4,300-digit
  int-str limit raised a bare `ValueError`/exit-4 instead of a typed
  ingress error/exit-3), raising the typed ingress error. The global ±2^53
  parser ceiling is REMOVED and must not be revived (D8; Track A′ R4).
  Float-exactness checks live exactly at float-touching gates: the
  gate-local native-finite guard and the `OverflowError` leg-catch stay (so
  gate arithmetic can never abort a run; FD-001 lineage). Enforcement:
  >4,300-digit literal fixture → typed ingress error at every parse site;
  a legitimate float-untouched large int (e.g. uint64 seed, ns-timestamp
  legacy metadata) parses.
- **R-067** [unit] *(Track A F6, reimplemented)*: Interval `ci` is exactly 2
  NATIVE finite numbers (str/bool rejected), `lo <= hi`, gated BEFORE any
  `float()` conversion. JSON ingress at every parse site rejects
  NaN/Infinity/−Infinity tokens AND overflow-to-inf literals (`1e999`, via
  `parse_float` hook) AND overlong integer tokens (R-062, via `parse_int`
  hook). Enforcement: per-site ingress fixtures (every `json.loads` in the
  namespace routes through the hardened loader; an AST test enumerates parse
  sites).
- **R-064** [unit] *(Track A′ R2 as amended by sign-off §5, ASK-2(a))*:
  Legacy-sidecar boundary: a well-formed JSON OBJECT that does not match an
  enumerated legacy family remains tolerated as an unknown historical
  sidecar (subject to manifest, rights, and tree anchoring). A top-level
  JSON array, string, number, Boolean, or null is ingress-DEFECTIVE for this
  evidence-tree namespace: source mode raises a typed
  `ColmAimsError`-family error naming the file and top-level type; release
  mode records a mandatory failing ingress leg naming the file and top-level
  type. Malformed JSON, invalid UTF-8, non-finite tokens, and overlong
  integer tokens remain ingress defects. Enforcement: parameterized matrix
  over `[]`, `"string"`, `0`, `true`, `null` in BOTH modes, plus
  nearest-true unknown-OBJECT controls that must stay tolerated; end-to-end
  (not parser-only) fixtures in both modes — the direct parser test alone
  does not cover the classification path.

### Closed vs open maps (sign-off §4.4; Track A′ R1)

- **R-063** [unit]: Structural and trusted configuration objects use CLOSED
  key sets — including every nested trusted block (the v1 R1 lesson: the
  expectations `anchor` block was open and `anchor.get("ledger_path", ...)`
  silently defaulted on a typo). Unknown nested keys are a typed config
  error. Open-key maps are allowed only where keys are data identifiers
  (path→hash maps, cell-ID maps, package-name→version maps), and for every
  such map: every key and value has an explicit schema; all required
  semantic IDs are checked by exact-set equality; an unknown data key cannot
  disable a required gate; no unchecked `dict.get(..., default)` may turn a
  misspelling into omitted verification. Enforcement: substitution-negative
  tests per trusted block (typo'd key → typed error, never a default); an
  AST test flags defaulted `dict.get` on trusted-block keys.

### Provenance and certification

- **R-012** [unit] *(amended)*: Release mode fails closed per leg, one
  fixture per missing or `UNRESOLVED` binding — the v1 leg families carry
  (schema/profile + producer/helper hashes; semantic command; seeds;
  dirty-state identity; fit/eval split names/counts/key-set hashes/
  zero-overlap; calibration and continuation identities; input and
  `split_metadata.json` hashes; MC-build-after-split freshness, coverage,
  retention policy/counts; scoring/selector model repository namespace/id +
  immutable revision — a full-length commit SHA; short hashes, tags, branch
  names, and bare repo ids are rejected (repo ids are reassignable —
  observed live on the `datasets/qanta` handle) — or a complete canonical
  byte-digest manifest, plus a content-level hash of any loaded weights
  file; tokenizer/config; dtype/device/numerical settings; runtime package
  versions; rights status; presentation manifest; anchored expectations) —
  and v2 adds the grid legs (R-040..R-044), per-file ingress legs (R-064),
  inference legs (R-050..R-056), ledger↔anchor equality (R-065), and the
  fail-closed git-object leg (R-066). Empty evaluation datasets error before
  any report; release evaluates all legs (collect-don't-halt except
  unreadable inputs), each failure naming leg id, expected vs observed, and
  a remediation class (`ARTIFACT_DEFECT | MISSING_EXPECTATION |
  AUTHOR_DECISION_REQUIRED | EXTERNAL`); release mode recomputes every
  non-EXTERNAL claim-ledger row's status from current verification, failing
  on any row whose recorded status is stronger than the recomputed one
  (EXTERNAL rows stay R-024-immune).
- **R-013** [integration] *(amended by sign-off §2.1)*: No self-attestation:
  certification requires an expectations file outside the artifact tree,
  anchored to a reviewed source commit and the frozen claim ledger, with the
  anchor cross-checked before expectations are consumed; the expectations
  pin the R-044 grid semantics and the inference identities (key-set digest
  from which the seed derives), not merely a package hash. An artifact plus
  its own generated manifest reaches at most source-level status.
  Containment uses fully resolved, symlink-free paths; the anchor check is
  string-exact and works without a git checkout, with an object-existence
  check when a repository is available (R-066 governs its failure mode).
- **R-014** [integration] *(amended by §4.2 and §5)*: Verifier runs never
  mutate inputs (byte-identical tree before/after). Artifacts bound to a
  superseded, dirty, or unresolved estimand-defining dependency closure
  classify `HISTORICAL_NONCERTIFYING`; current metadata is never backfilled
  into old bytes; only a producer/closure change invalidates an artifact.
  Legacy v1/historical profiles parse ONLY via the named legacy loader
  (R-060) and are refused certification unless independently migrated and
  regenerated; the legacy-sidecar boundary is R-064. Aggregate-only files
  cannot certify per-item paired claims. The known legacy profile set is
  enumerated as a fixture corpus captured (sanitized) from the repo's actual
  historical artifact formats (the `paper_exports/csli.json`,
  `paper_exports/audit_card.json`, and `paper_exports/calibration.json`
  families) — the enumeration R-064's unknown-family boundary is defined
  against; legacy parse tests run against captured bytes, not
  reconstructions.
- **R-015** [unit] *(amended by §2.3 and §3)*: Every reported top-level
  aggregate and paired interval recomputes from retained per-item records
  within declared tolerance; mismatch → FAIL; absent records →
  non-certifying. Interval-bearing cells record the full interval identity
  (procedure, draw count, the single derived seed + derivation string,
  statistic, population); headline-cell interval recomputation binds to the
  sentinel-coded all-pair population (never both-finite; plan v3 §3.3
  implementation implication); recomputation re-runs the recorded D7(b)
  procedure deterministically; missing interval identity leaves the interval
  non-certifying.
- **R-016** [integration] *(carried)*: Evidence writers publish create-once
  via the no-replace primitives of `scripts/stopdff_v5/fileio.py` (or an
  equivalent with identical guarantees): interrupted publish leaves no
  parseable partial artifact at the final path; second publish to an
  existing path fails; kill-mid-publish/retry test asserts exactly one
  resulting artifact; staging-debris policy pinned.
- **R-065** [unit] *(Track A F2, reimplemented)*: Cross-document commit
  equality: the frozen ledger's `anchored_source_commit` must EQUAL the
  expectations anchor's `source_commit`; the failing leg names both commits.
  A ledger re-anchored to a different commit (digest rebuilt over the new
  bytes so the ledger-hash leg passes) still FAILs this leg. Enforcement:
  release leg + mutation fixture.
- **R-066** [unit] *(Track A F4, reimplemented; sign-off Phase 3)*: The
  anchor commit object-existence check is a separate release-mode leg bound
  to the source repository. Repo available + object missing → FAIL. Git
  unavailable / check cannot run → ALSO a FAILING leg: release verification
  must fail when the commit-object check is expected to be available but
  cannot run — `PASS_RELEASE` cannot be obtained by making `git` disappear.
  A source-only environment without a repository may RECORD the capability
  gap (receipt/source mode), but never converts it into a release pass. The
  string-exact anchor leg (R-013) stays independent and passes without a
  checkout. Enforcement: git-removed end-to-end fixture → release FAIL;
  source mode unaffected.

### Verifier modes, namespace, CLI

- **R-017** [unit] *(amended — v1 source-mode floor restated with v2 rule
  families; reaffirmed by sign-off §1)*: The source-mode verdict vocabulary
  is a closed enum whose strongest member is `PASS_SOURCE_ONLY`; no
  source-mode code path emits a release/camera-ready token; the rendered
  summary states what source-only verification does NOT certify and lists
  the checks performed; author-side verdicts never use ACM-v1.1 third-party
  terms "Reproduced"/"Replicated". Source mode's minimum positive check set
  is typed ingress (R-020/R-059/R-061/R-062 and the R-064 ingress legs),
  full in-package profile validation (R-001–R-011 as amended, including
  grid completeness R-040–R-043, event representation R-045–R-047, estimand
  rules R-048–R-049, and in-package inference recompute R-050–R-056 where
  the inference block is present), and receipt emission (R-036).
- **R-018** [integration] *(carried)*: Everything new lives under
  `reproducibility/colm_aims_2026/` (plus tests/docs);
  `scripts/verify_audit_release.py` stays byte-identical with its existing
  suite green; no doc redefines the legacy verifier as camera-ready
  certification; `scripts/stopdff_v5/` primitives are consumed, not forked.
- **R-020** [unit] *(amended)*: Typed ingress: artifact bytes validate into
  typed records at the load boundary; malformed, truncated, or unknown-keyed
  files produce a typed error naming the file (repo- or tree-relative path)
  and field — no partial semantic processing, silent key-dropping, or
  truthiness coercion downstream. Version-first ordering and version-error
  naming are governed by R-059 on every surface; ingress hook hardening by
  R-067/R-062; the sidecar boundary by R-064.
- **R-021** [integration] *(amended)*: Bindings demonstrably reach the
  verdict: an end-to-end test runs the real verifier CLI (subprocess, the
  documented command line — never an imported `main()`) over a complete tiny
  fixture package, then mutates one expectation at a time — model revision,
  split hash, producer hash, calibration identity, ledger anchor, and (v2)
  each grid pin (reference IDs, calibration IDs, cell IDs, record-file
  mapping, item-key-set digest) and each inference pin (seed, index digest)
  — asserting the verdict flips each time, with exit codes (R-037) and
  sentinel-free output (R-026) asserted alongside.
- **R-022** [unit] *(carried)*: The verifier CLI and its config fail closed
  on unknown flags or keys (error, not no-op); no flag, env var, or config
  door disables a release gate; `allow_abbrev=False`.
- **R-033** [unit] *(carried)*: No vacuous verdicts: both modes fail with a
  typed error naming the resolved path and expected layout when zero
  candidate artifacts are found; any PASS-class verdict requires ≥1
  validated artifact enumerated in summary and receipt; release additionally
  requires ≥1 retained claim row and ≥1 manifest-declared artifact.
- **R-034** [unit] *(carried)*: Deserialization safety: evidence ingestion is
  JSON/JSONL only; `pickle`, `marshal`, `torch.load`, non-safe YAML never
  appear in the namespace (AST/import scan). The D7(b) resample matrix is
  regenerated in-memory from the recorded seed — never deserialized from
  binary.
- **R-035** [unit] *(amended)*: Release reconciles the evidence package
  against the presentation manifest in both directions
  (declared-but-absent → FAIL; present-but-undeclared → FAIL absent an
  explicit per-file allowlist entry); the rights inventory covers every file
  found; and (v2) the `records/` directory reconciles against the grid block
  and expectations per R-041.
- **R-036** [integration] *(carried)*: Every verifier run emits a
  schema-versioned JSON receipt — mode, verdict, per-leg outcomes,
  input-tree hash, expectations-anchor hash, verifier code hash, timestamp —
  outside the verified tree, create-once under run-scoped unique names
  (R-016 primitives). If versioned, the receipt uses the R-058/R-059
  constants and checker.
- **R-037** [integration] *(amended)*: CLI contract: documented invocation
  `python -m reproducibility.colm_aims_2026.verify` from the repo root;
  direct-path invocation bootstraps `sys.path` repo-root-first (dedupe then
  insert-at-front) or errors naming the module-run form. Pinned exit codes
  (v1 implementation QA-019 promoted to contract): `0` mode-ceiling pass,
  `1` gate FAIL, `2` usage/config error, `3` typed ingress error, `4`
  internal (non-ingress) error. R-021 asserts them end-to-end; R-062's
  overlong-token case exits 3, never 4.

### Canonical selection and the release path (sign-off Phase 3; Track A′ R6)

- **R-039** [unit] *(amended by R-069)*: Supersede/retire: evidence packages
  publish into run-scoped (or content-addressed) directories; canonical
  selection happens ONLY via the ledger/expectations pointer; retiring a
  defective artifact is a ledger status change plus a new run directory —
  historical bytes retained, never republished over.
- **R-069** [integration]: Canonical package selection is WIRED INTO THE
  ACTUAL RELEASE PATH (a helper with zero production callers protects
  nothing — the v1 F3/R6 lesson). The release entry resolves the canonical
  package exclusively through the ledger pointer and must: reject symlinked
  run directories outright (even in-root targets); enforce resolved
  containment under `runs_root`; reject empty crash relics; reject dangling
  pointers; NEVER select newest-wins; NEVER fall back to any other package
  after an invalid canonical pointer. Enforcement: end-to-end release-CLI
  coverage of every refusal class (symlink, escape, empty, dangling,
  decoy-newer-run present) — each must FAIL the release run itself, not
  merely a unit helper.

### Ledger, Random-K, rights

- **R-023** [unit] *(amended)*: Each claim-ledger row carries `claim_id`,
  exact manuscript location and wording, estimand and allowed scope,
  producer entrypoint and dependency closure, input/split/model/calibration
  identities, artifact and renderer identities, independent verifier/oracle,
  rights status, status in the closed enum `PASS | FAIL | UNVERIFIED |
  EXTERNAL`, and blocking repo task. v2 additions: the ledger carries
  `anchored_source_commit` (R-065 twin); `schema_version` 2 via the single
  checker (R-058/R-059); the Holm/inference row names the exact rejected
  cell IDs from R-056; `blocking_task` names a genuine actionable remaining
  blocker — never a restatement of a resolved decision (the D3 lesson;
  golden-ledger fixture pins the Random-K row's task: bind archived +
  fresh-run identities with `rng_pinned=false`; publish the first pinned
  evidence run under the v2 data model). The ledger distinguishes manuscript
  identity (the D6 hashes), historical submission artifacts, historical
  Random-K/v5 results, current source, and future clean evidence; PR #41
  hazard reports appear only if the exact manuscript cites them.
- **R-024** [integration] *(carried)*: `EXTERNAL`-typed rows are immune to
  repo tooling (byte-identical across every tool run: writer, checker, both
  verifier modes, ledger validator); `EXTERNAL → PASS` without a
  human-attribution field fails ledger validation; repo green tests never
  substitute for an EXTERNAL item; venue-rule rows record only officially
  published facts with source and as-of date — self-imposed gates are never
  attributed to the venue.
- **R-025** [unit] *(amended by plan v3 §3.5 + Track A F1)*: Random-K
  binding: `author_decision` for this paper is recorded as
  `historical_nonconfirmatory` with `headline_eligible = false`; any
  recorded `author_decision` must be in the closed two-member enum;
  `headline_eligible: true` requires exactly `predeclared_multidraw_family`
  (F1 consistency — the pair is validated jointly, not independently). Both
  Random-K cells are present in the m=10 family and reproduced faithfully
  (R-056); no headline/substantive use; dagger disclosure retained.
  `rng_pinned = false` is recorded explicitly; the archived AND fresh-run
  draw identities are both bound; a differing fresh draw is never treated as
  confirmation; draw identity is estimand-defining (R-011), so a substituted
  favorable draw changes the digest and is refused.
  `predeclared_multidraw_family` exists as a DISCRIMINATED, INACTIVE
  contract — its row schema requires frozen protocol, complete seed/draw
  registry, no selective omission, aggregation rule, sensitivity analysis,
  and multiplicity treatment — and it is not active for this paper. This
  feature implements metadata/validation support only; it never runs the
  multi-draw experiment (SOURCE_CONTRACT_ONLY).
- **R-026** [unit] *(carried)*: Rights inventory: every included path is one
  of `VERIFIED_ALLOWED | VERIFIED_RESTRICTED | UNVERIFIED |
  AUTHOR_DECISION_REQUIRED`; release requires every included path
  `VERIFIED_ALLOWED` and fails on any other value or any uninventoried
  included path; each row names its upstream terms basis; fixtures are
  synthetic (no raw quizbowl text, no real qids); sentinel-leak tests prove
  error paths reference items by opaque keys and never emit restricted
  content, credentials, or local absolute paths.

### Receipts and suite evidence (A′-F4; plan v3 §5.12)

### Phase-4 PRE-run repairs (ChatGPT Phase-4 adjudication accepted 2026-08-22; source-only, no model execution anywhere in this section's implementation or tests)

- **R-073** [unit] *(PRE-2)*: Canonical per-item horizon map: the JSON object
  mapping each of the 2,249 item keys to that item's positive-integer
  trajectory horizon, serialized with keys sorted ascending by UTF-8 byte
  order, compact separators (`","`, `":"`), UTF-8 encoded with ASCII
  escaping (`ensure_ascii`; inert for the all-ASCII qid keys, pinned to kill
  reimplementation ambiguity); its digest is the lowercase-hex SHA-256 of
  those bytes. `estimand.timeout_parameters` is the
  closed key set `{horizon_map_sha256, rule}` (the scalar
  `trajectory_horizon` member is RETIRED). The verifier RECOMPUTES the map
  from the cell's records (never trusts the declaration) and requires:
  recomputed digest == `timeout_parameters.horizon_map_sha256` == the
  held-fixed `horizon_identity` (R-043), equal across all ten cells, and — in
  release mode — equal to the expectations pin (extends R-044's semantic
  pin set). Enforcement: heterogeneous fixtures (horizons spanning 2–10)
  PASS; a single-item horizon mutation changes the recomputed digest and
  FAILS all three comparison legs; an all-equal-horizons nearest-true
  fixture PASSes; a genuine `FINITE_STOP` at the final prefix of a
  short-horizon record (e.g. horizon 2) PASSes R-046 unchanged.
- **R-074** [unit] *(PRE-1)*: Frozen pairing eligibility: the committed
  artifact `reproducibility/colm_aims_2026/frozen/pairing_eligibility_v2.json`
  pins the 2,249 sorted eligible item keys; the 9 excluded qids
  (`103295, 119618, 190798, 191687, 196619, 197040, 206660, 207981, 209745`),
  each with enumerated reason `SINGLE_PREFIX_TRAJECTORY` (new member of the
  exclusion-reason enum); the eligible-keyset SHA-256 computed by EXACTLY
  `pairing.keyset_sha256` (sorted, newline-joined — this digest is the
  D7(b) seed input `pairing_population_keyset_sha256`); and derivation
  provenance (`test_dataset.json` SHA-256
  `638a4df978b77a12655ea72d56daad7fa70851ae486ddb4365d9b060549e34f1`,
  two-party pinned by the item-10 Appendix-B balance file). The producer
  consumes the frozen set explicitly; a cell-specific eligibility failure at
  production time raises fatally — silent exclusion and crash-by-default are
  both defects. Enforcement: artifact-schema validation; digest recompute
  test; producer-unit fatality test on a synthetic ineligible item.
- **R-075** [unit] *(PRE-3)*: Role-keyed model identity pins: the committed
  manifest's canonical artifact path and raw-byte SHA-256 are certificate
  bindings (a same-role/revision replacement manifest cannot authorize new
  snapshot-file hashes). The manifest binds `primary_scorer`
  (`all-MiniLM-L6-v2`) and
  `disjoint_selector` (`all-mpnet-base-v2`) each to an immutable local
  snapshot — HF revision hash plus the complete per-file SHA-256 manifest of
  weights/tokenizer/pooling/config — and binds the TF-IDF configuration
  (vectorizer parameters) separately. In pinned mode the producer loads ONLY
  from the snapshot paths with offline flags set (`HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`). Tests operate on manifests and path/flag
  plumbing exclusively — no model load, no network (R-028 carries).
  Enforcement: manifest schema + digest tests; a snapshot-file mutation or
  absence fails the pre-load gate.
- **R-076** [unit] *(PRE-4)*: Staged-input gates: every fit/eval input is
  hash-verified fail-closed BEFORE any model load — `calibration_train.json`
  expected `745bd67597278bd9d24d41c1dea53bf3a7c56cd6334cfc07ea62bccbdcf44259`
  (staged read-only from the archival copy; absent from the primary tree by
  historical accident), `test_dataset.json` expected `638a4df9…` (R-074),
  `val_dataset.json` expected `9b7a131b…`, `mc_dataset.json` expected
  `3dbebf8e…`, `answer_profiles.json` expected `63558639…`, and
  `build_metadata.json` expected `70871984…`; every entry records equal
  expected+observed SHA-256. The run receipt captures fitted Platt
  coefficients and the continuation-estimator digest per calibration mode.
  Enforcement: gate-ordering test (gates run before any loader is invoked;
  a mismatched or missing input aborts with a typed error naming the file
  and both hashes); receipt-field presence tests.
- **R-077** [unit] *(PRE-6; amended by Repair-A, 2026-08-24 — idealized+performat·dp six-field known-divergence)*: Materialized parity comparator: the committed
  anchor artifact is derived from Export A
  (`stopdff_fair_qa.json`, SHA-256
  `59e1c1a74e5fc0cf4f09f8befca87cfc81516684dca2e88dd275c952b28893ff`) and
  freezes the field allowlist — all 160 nonrandom point fields (8 nonrandom
  cells × {dp, myopic} × 10 point fields) and all 32 nonrandom CI arrays,
  plus the population identity fields (`n`, `n_eval`, `n_fit`) — with
  stored-precision comparison (the producer's own `round(x, 4)` path; parsed
  JSON value equality AT THE SAME JSON TYPE — an int drifting to a float,
  a bool, a string-encoded number, or a non-finite value all FAIL; no
  cross-type numeric laundering). The comparator refuses a truncated anchor:
  the loaded anchor must carry exactly 8 nonrandom cells × 2 policies × 10
  point fields × 2 CI fields, and a comparison that checks fewer than the
  full allowlist can never emit PASS. The comparator emits a machine-readable per-field
  PASS/FAIL report; ANY mismatch, including any CI-array element, is a
  blocking FAIL. The two Random-K cells are exempt from historical parity
  and reported informationally with `archived_rng_pinned=false` /
  `fresh_rng_pinned=true` — but their STRUCTURE is required: both Random-K
  cells must be present in the regenerated export with the full point and
  CI field set (a missing Random-K cell or field is a blocking structural
  failure; only the numeric VALUES are exempt from historical parity —
  operational-rejection repair 2026-08-22). In addition to the Random-K value
  exemption, exactly six fields of the single cell/policy
  `idealized+performat` · `dp` — `signed_mean`, `abs_mean`, `mc_earlier`,
  `qa_earlier`, `same_step`, `signed_mean_ci` — are a declared
  known-divergence: they are still compared against the frozen anchor and
  still counted in `checked` (the 194-field cardinality is unchanged), but a
  VALUE mismatch on any of these six is recorded as an informational
  divergence, never a blocking failure, and can never flip the verdict to
  FAIL — the same treatment R-077 already gives Random-K numeric values,
  narrowed from whole-cell to a named six-field allowlist. STRUCTURE stays
  blocking for this cell (the cell and its full point + CI field set must be
  present; a missing cell or field is still a blocking structural failure).
  The other six fields of this same `dp` block (`n`, `signed_median`,
  `abs_median`, `mc_never_buzz`, `qa_never_buzz`, `signed_median_ci`), this
  cell's `myopic` block, and every other nonrandom cell/policy remain fully
  blocking; the Random-K policy is unchanged. The frozen anchor's historical
  values for these six fields are preserved byte-for-byte — the anchor
  (`parity_anchor_export_a.json`, SHA-256 `2efff657…973eee`) is NOT re-frozen
  or overwritten. Basis and provenance: the cross-verified reconciliation in
  `phase4_reconciliation_diagnosis_2026-08-24.md` and
  `phase4_reconciliation_verification_2026-08-24.md`, folded from
  `phase4_reconciliation_amendment_proposal_A_2026-08-24.md` (scoped
  known-divergence repair 2026-08-24). Export B (`ba784741…`) is
  corroborative, never the anchor. Enforcement: comparator-unit tests with exact-match,
  single-field-mutated, and CI-element-mutated payloads; allowlist
  completeness test (160 + 32 + identity fields, no more, no fewer); a
  single-field VALUE mutation on each of the six known-divergence fields
  yields `verdict == "PASS"` with the mutation surfaced in the
  known-divergence informational report; a VALUE mutation on any of the other
  six `dp` fields, on this cell's `myopic` block, or on any other cell still
  FAILs; a missing `idealized+performat` cell or any of its fields still
  FAILs structurally; `checked == 194` on PASS is preserved.
- **R-078** [unit] *(PRE-7)*: QA-012 compatibility fixtures: committed
  exact-byte excerpt fixtures (first records, verbatim line bytes) for each
  of the four hit files, with each FULL file bound by its SHA-256
  (`32ecda09…`, `8f38ef3f…`, `c3aa6308…`, `f7dcb43b…`), plus a demonstration
  test that the v2 record ingestion REJECTS these shapes (they carry
  `{item_id, format, prefix_fractions, p_calibrated}` — no `item_key`, no
  event statuses, no stop steps, no horizon): they cannot substitute for the
  required ten-cell semantic block. Enforcement: fixture-bytes digest test +
  loader-rejection test naming the missing required fields.
- **R-079** [integration] *(PRE-5/§5)*: `PRE_RUN_READY` certificate: a
  checked-in generator emits one machine-readable JSON binding — repo commit
  AND tree with clean-state proof (both bind the repository's NATIVE git
  object ids, runner-sourced from `git rev-parse`; 40-hex SHA-1 and 64-hex
  SHA-256 object formats are both admissible, lowercase hex, fixed length —
  SPEC_ISSUE-1 adjudication 2026-08-22; the clean-state proof is the
  TRACKED tree — `git status --porcelain --untracked-files=no` empty —
  because untracked evidence artifacts (the certificate itself, receipts,
  staged inputs) are unavoidable by construction and cannot alter the
  commit/tree identity; they are DISCLOSED as a recorded list in the repo
  component — amendment 2026-08-22); producer/verifier/spec content hashes;
  the R-074 eligibility digest, canonical artifact path + raw-byte SHA-256,
  frozen source-test SHA-256, and the R-073 horizon-map digest; the R-075
  snapshot-manifest artifact path + raw-byte hash, both role revisions/file
  manifests, and offline flags; every R-076 staged
  input with expected+observed SHA-256; focused and full suite receipts
  (R-070 shape); the R-077 canonical comparator identity + anchor path/raw
  hash + embedded Export-A source hash; the canonical R-072 rev3 manifest
  path and raw hash (rev2 is retired); interpreter realpath, OS/arch/CPU, BLAS, thread settings,
  environment-lock hash, exact command, seeds, `PYTHONHASHSEED`, and the
  rng-pinned flags. The environment also signs the absolute external
  quarantine, promotion, and create-once exception-ledger paths. Both suite
  receipts must bind the SAME interpreter realpath and dependency-lock hash
  as that environment, in addition to the certified commit/tree/clean state.
  The command is not merely recorded: it must be the audited producer entry
  point with no unknown, abbreviated, underscore-alias, positional, or
  duplicate arguments; it pins `fit=val`, `eval=test`, seed 1, the full five
  reference arms × two calibration modes, 1,000 bootstraps, all rows, the
  frozen eligibility/snapshots, `--records-out phase4_run_output`,
  `--out phase4_run_output/stopdff_fair_qa_regenerated.json`, and contains no
  caller-supplied certificate digest. Every component check is fail-closed:
  any dirty tree,
  hash mismatch, missing snapshot, or suite failure yields a certificate
  with `ready: false` and the named failing checks — never a partial pass.
  The author's single-run exception activation references the certificate's
  SHA-256. Enforcement: generator-unit tests for field completeness and for
  fail-closed behavior on each induced defect class.
- **R-080** [unit] *(records export)*: The producer emits
  `records/<cell_id>.jsonl` for all ten cells in the v2 record schema, with
  the historical `performat` label mapped to `format_specific` at the export
  boundary (provenance-noted). Each record key is the canonical unsigned-
  decimal dataset QID and the emitted profile declares the exact Phase-4 QID
  scheme; noncanonical or non-QID keys fail at export, and the verifier checks
  record keys against that declaration. This preserves the R-074 keyset and
  D7(b) seed rather than falsely claiming the QIDs were outputs of the generic
  NFC/SHA-256 text-key scheme. The v2 record loader is the oracle: exported
  rows for a synthetic scored frame must load cleanly under
  `reproducibility.colm_aims_2026` ingestion, with canonical events
  (`FINITE_STOP` iff the DP stop is strictly below that item's horizon;
  `stop == horizon` is the sentinel and becomes `NEVER_STOPPED` with
  `stop_step = null`, `terminal_imputation = FINAL_PREFIX_IF_NEVER`;
  `stop > horizon` is unreachable from the DP and is REFUSED as frame
  corruption, never absorbed into the weaker bucket) and the derived
  reporting encoding kept distinct (R-046). The export stage is testable
  WITHOUT models (it consumes scored frames). Enforcement: producer-unit
  export tests on synthetic frames covering finite, never-stopped, and
  final-prefix-crossing items at heterogeneous horizons. `--records-out`
  receives the PARENT directory; the exporter owns the `records/` path
  segment (doubled-segment invocations were the P1-6 defect). Each
  `metadata.phase4.exported_records[*].path` is the exact artifact-relative
  `records/<cell_id>.jsonl`, never an absolute quarantine path, so it remains
  valid after the launcher's atomic directory promotion.
- **R-081** [integration] *(operational rejection repair, 2026-08-22)*: The
  single authorized run is executable ONLY through the checked-in
  single-use launcher. The module is an executable interface, not an
  import-only shim: `python -m
  reproducibility.colm_aims_2026.phase4_launcher --config <CONFIG.json>`
  hardened-parses exactly one strict, closed-key JSON config and exits
  nonzero on every config, preflight, run, comparison, or promotion defect.
  A successful import/no-op exit is a blocking defect. Before any model
  construction the launcher must:
  (1) read the certificate bytes in binary mode (including Windows
  `O_BINARY`) and require sha256 == the activation
  digest, hardened-parse it, and require `ready is True` (bool-safe);
  (2) require LIVE `git rev-parse HEAD` == certificate commit, LIVE
  `rev-parse HEAD^{tree}` == certificate tree, and a live tracked-clean
  check; (3) REJECT ambient provenance overrides (`MODAL_HOST_GIT_STATUS`,
  `MODAL_HOST_GIT_COMMIT` set at all = refusal) and launch the producer as
  a SUBPROCESS whose environment sets, before interpreter start:
  `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `VECLIB_MAXIMUM_THREADS=1`, and the offline flags; require the live OS,
  architecture, and resolved child
  interpreter to equal the certificate environment binding, and recompute
  the child interpreter's `pip freeze` bytes to require the recorded
  `environment_lock_sha256`; require the exact command shape, seed/thread
  cross-bindings, and audited producer entry point again at launch; (4)
  re-verify the snapshot manifest's raw bytes against its certified artifact
  path/hash, then both model snapshots against that frozen manifest; require the configured
  manifest and role snapshot directories to resolve to the corresponding
  certificate-command arguments, and require the configured parity anchor's
  SHA-256 to equal the certificate parity binding; rehash every certified
  staged input and require live == expected == certificate-observed, and
  raw-byte verify + hardened-load the frozen eligibility artifact against
  its artifact/keyset/horizon/source-test pins, BEFORE the ledger; (5)
  require the config's quarantine, promotion, and ledger strings to match
  the signed certificate values exactly; all three are absolute, outside
  the repository, mutually safe/disjoint (neither workspace nested in the
  other and the ledger neither inside nor an ancestor of either workspace),
  fresh where applicable, with existing writable parents and same-device
  promotion; then consume the exception
  via a CREATE-ONCE ledger file
  (`O_CREAT|O_EXCL`, plus binary mode where required) recording the
  certificate digest BEFORE launching. The launcher loops until every ledger
  byte is written, fsyncs the file and, where supported, its parent directory;
  a partial/zero write, close, fsync, or publication failure is an irrevocable
  consumed STOP with the partial ledger and quarantine preserved and NO child
  launch. A
  pre-existing ledger refuses (no second run without a new recorded
  amendment); (6) write all outputs into a FRESH quarantine directory
  (pre-existing quarantine or promote destination = refusal), compose the
  producer argv FROM the certificate's recorded command (remapping only
  output paths into quarantine, appending `--certificate-digest`);
  (7) after a zero exit, MANDATORILY run the parity comparator and the
  structure checks: the main export must bind the exact activation digest,
  metadata must declare exactly the ten expected record cells, and each
  declared record path/hash/count/historical-cell/policy must bind a real
  `records/<cell_id>.jsonl` whose strict rows equal the frozen eligible
  population and horizon map. Atomically promote with strict no-replace
  semantics only on PASS (a racing destination must refuse, never be replaced);
  if the rename commit point succeeds but the subsequent durability sync
  fails, report STOP at the destination that now owns the outputs and describe
  that committed-but-not-certified state truthfully — never recreate an empty
  quarantine or claim that promotion rolled back;
  on any crash, nonzero exit, comparator failure, or promotion failure leave
  the quarantine in place, write a STOP report, and exit nonzero. The
  producer records
  the certificate digest in its output metadata. Enforcement: launcher
  unit tests with injectable runners/launchers for every refusal class and
  the single-use ledger; no test loads a model.

  **Process/host trust boundary — author amendment, 2026-08-26.** The
  Phase-4 launcher is not a sandbox, privilege boundary, or hostile-process
  containment mechanism; it is an integrity and reproducibility workflow. It
  assumes the certified producer and dependency environment, operating
  system and filesystem, and processes running with the launcher's OS
  identity are cooperative. In particular, the supported producer workflow
  has no surviving producer descendants: no independently running descendant
  retains access to the launch workspace after the direct producer exits. The
  launcher detects ordinary
  path and byte drift and reconstructs comparator-approved bytes into a
  path-detached candidate, but it does not protect that candidate or the
  promoted directory from a process able to enumerate and mutate files under
  the same OS identity, or from privileged host compromise. "Private
  promotion" denotes launcher ownership and byte provenance, not ACL or
  process isolation. `PASS`, the launch receipt, `PASS_RELEASE`, and
  `CAMERA_READY_CLOSURE` certify the declared bytes and scientific bindings
  only under this boundary; they do not establish hostile-process provenance
  or tamper resistance. This amendment does not weaken any existing
  untrusted-artifact, symlink/reparse, checksum, TOCTOU, schema, no-replace,
  durability, or truthful-STOP check within the supported workflow. A
  conforming launch receipt binds the exact closed token
  `process_trust_model =
  "trusted_same_os_identity_no_surviving_descendants_v1"`; a missing or
  different token is an ingress defect. If this cooperative-host precondition
  cannot be established, the ceremony must not run; hostile same-principal
  containment requires a separately approved and tested OS isolation
  backend.
- **R-082** [unit] *(operational rejection repair, 2026-08-22)*: Staged
  inputs live OUTSIDE the repository tree — identity is carried by the
  hash gates (R-076), never by location — because the producer's
  committed-writer guard computes `git status` over a pathspec including
  the staged calibration path, and an in-repo untracked staged file aborts
  the run AFTER scoring (the P0-1 trap). The launcher and certificate
  orchestration must stage under an out-of-repo directory and the
  certificate records those absolute staged paths. This applies to ALL six
  consumed/provenance inputs, not calibration alone: calibration, fit, eval,
  MC dataset, answer profiles, and `build_metadata.json`. The command's
  `--data-dir`, `--calibration`, and every `--staged-input` path must resolve
  outside the repository. The certificate requires exactly one component for
  each of the six labels; pins `calibration_train` to the archival digest
  `745bd67597278bd9d24d41c1dea53bf3a7c56cd6334cfc07ea62bccbdcf44259`
  (a caller cannot substitute a self-consistent path+digest pair); binds
  `calibration_train` to `--calibration`; binds
  the other five paths to the explicit `--data-dir` plus the exactly-once
  `--fit-split`/`--eval-split` selectors; and requires exactly one matching
  path+digest `--staged-input` for `fit_split`, `mc_dataset`,
  `answer_profiles`, and `build_metadata`. The certificate gathering path
  evaluates the same
  containment policy and emits `ready:false` with a named path-policy
  failure; runtime enforcement alone is insufficient because an
  operationally impossible certificate must never advertise `ready:true`.
  The eligibility path is mandatory and binds exactly to the R-074 artifact;
  its raw-byte digest and embedded test-dataset provenance are independently
  certified. Suite receipts must be
  produced by EXECUTING the suites at the certified head (never ingesting
  pre-existing receipt files from earlier heads): each receipt binds
  `commit`, `tree_sha256`, and `dirty` matching the repo component, real
  captured exit codes, and `failures == 0`/`errors == 0`; the certificate
  checker refuses receipts missing or mismatching any of these.
  Enforcement: receipt-checker tests (head mismatch, nonzero failures,
  missing binding all flip ready:false).

- **R-070** [integration]: Any suite-execution claim made for this feature —
  and the successor PR's CI evidence in particular — is accompanied by a
  machine-readable receipt binding: dependency lockfile or
  environment-export HASH (not a metadata object); workflow-file hash;
  interpreter realpath; repository commit AND tree; dirty state; the exact
  command; exit code (plan v3 §5.12 — subsumed by, and a harmless superset
  of, the sign-off A′-F4 machine-readable-output item); JUnit or equivalent
  machine-readable test output; skip identities; artifact hashes. A receipt
  includes valid JUnit and transcript SHA-256 digests; its counts map is
  closed to `{tests, failures, errors, skipped}`, uses real nonnegative ints,
  has `tests > skipped`, and has zero failures/errors. The focused and full
  receipt commands must select the certificate's exact focused file set and
  `tests/`, respectively, under the certified interpreter; a zero-test,
  all-skipped, or arbitrary no-op command is never evidence. Local transcripts remain trust-required
  evidence; exact-head CI on the successor PR is the durable form.
  Enforcement: receipt-schema validation test + a fixture rejecting the
  A′-F4 defect shape (`environment_digest` as an object instead of a hash).

### CAMERA_READY_CLOSURE (D4; sign-off §8; handoff §9/L526)

- **R-071** [integration]: `CAMERA_READY_CLOSURE` is a distinct gate token
  from `PASS_RELEASE` (neither implies the other; source mode can emit
  neither). Its expected-claim inventory is FROZEN from the D6 baseline's
  COMPLETE checksum closure (`main.tex`
  `79dccfb3fbdfafbd566a3fb239755ab35142bac510d629d513ed8b3c2c4cdd2f`,
  `main.pdf`
  `6de23119df59679befc356e3c916bc5a498b2cc2015b6cd8a516a5181dabf10a`, and
  every entry of `FINAL_CHECKSUMS.sha256` — never `main.tex` alone). It
  absorbs handoff-L526: every displayed number, count, table, and figure
  maps to clean bound evidence or is removed/downgraded (rows for
  manuscript-side items stay `EXTERNAL` per R-024). The Holm/inference row
  is satisfied ONLY by the D7(b) regenerated outputs (R-050..R-057) — until
  they exist the gate fails on that row by construction. QA-012 (R-072) is
  blocking for closure. Current state per the sign-off:
  `NOT_YET_SATISFIED`. Enforcement: closed inventory file + per-row gate;
  fixtures for a satisfied-except-Holm inventory (FAIL) and a
  main.tex-only closure binding (FAIL).
- **R-072** [integration] *(amended by PRE-7, 2026-08-22)*: QA-012 state:
  rev1 inventory (46 files, zero hits) was VOID for scope omission; rev2 at
  corrected scope (67 files including the item-10
  `reachable_comparator_prototype` bundle, `data/processed/*.json`, and the
  successor suite transcripts) found **4,556 hits** in the four
  `per_prefix_scores*` JSONLs — manifest
  `qa012_inventory_2026-08-22_rev2.json`, SHA-256 `52ac2902…`. Rev2 was
  itself found defective at readback (operational rejection 2026-08-22):
  entries carried SHA-256 only — §8 requires BOTH a content hash (the
  Dropbox content hash, 4 MiB-block sha256-of-block-sha256s, for
  Dropbox-resident evidence) AND SHA-256 — and the 4,556 JSONL hit
  pointers were 0-based where 1-based line numbers are required. Rev3
  corrects both over the same 67-file scope (supersession chain recorded
  in-manifest). QA-012 is `HITS_PRESENT, non-blocking for source
  reconstruction, blocking for final CAMERA_READY_CLOSURE` until the
  R-078 fixtures land. Executing
  this rule's inventory REQUIRES a per-prong disposition: every enumerated
  scope prong is recorded as located/scanned or UNLOCATABLE-escalate — a
  zero-hit verdict over an incompletely-located scope is a false vacuity and
  is itself a defect. Closure requires
  the inventory procedure of sign-off §8 executed over: the complete D6
  checksum closure; historical `paper_exports` copies outside the GitHub
  repository; supplied source/export bundles used by the paper; any external
  expectations or ledger sidecars. For every `.json`/`.jsonl` file:
  (1) record path, size, content hash, and SHA-256; (2) strictly parse;
  (3) recursively detect any key named `format` whose value is exactly
  `"QA"`; (4) emit the matching JSON pointer for every hit; (5) store a
  signed or hashed zero-hit/match manifest. Zero hits → QA-012 recorded
  vacuous with that inventory hash. Any hit → its exact bytes become a
  compatibility fixture plus a demonstration that it cannot substitute for
  the required semantic block. Enforcement: inventory-manifest schema +
  detector unit tests (nested/array-embedded `format:"QA"` hits, JSON
  pointers exact); the closure gate consumes the manifest hash.

### Documentation (D4 widening)

- **R-038** [integration] *(amended)*: `reproducibility/colm_aims_2026/README.md`
  pins both modes' exact invocation, input layout including the
  expectations-file location, verdict enum semantics, exit codes and receipt
  location, and a one-line disambiguation from
  `scripts/verify_audit_release.py`. The doc-audit widens to the handoff §9
  seven-target set: `README.md`, `DATA.md`, `ARTIFACTS.md`,
  `docs/CLAIM_SURFACE.md`, `docs/stopdff-learned-value-fair-qa.md`,
  `docs/stopdff_v5/REPRODUCTION.md`, and `reproducibility/source_to_claim.md`
  — each audited for constructed-reference qualification and
  historical-vs-current scope, with `source_to_claim.md` carrying a
  historical-scope header naming the manuscript it maps and pointing at the
  ledger, and the ledger recording it as a historical-submission-artifact
  document. Enforcement: doc-content tests over the seven targets.

### No-network and fixtures

- **R-028** [unit] *(carried)*: No network, model downloads, or training:
  namespace test conftest installs a no-network guard; import scan rejects
  the deny-list (`requests`, `httpx`, `urllib.request`, `huggingface_hub`,
  `transformers`, `torch`). NumPy 2.4.6 is REQUIRED (D5/R-051) and is not on
  the deny-list. Fixtures are tiny and synthetic.
- **R-019** [unit] *(amended)*: The adversarial fixture corpus carries v1's
  thirteen classes (constructed-as-observed; denominator mismatch; timeout
  mismatch; hash/model/split mismatch; historical/current substitution;
  empty evaluation; unbound calibration; unverified-rights inclusion;
  stale-PASS ledger row; zero-artifact tree; empty ledger; empty
  presentation manifest; oversized tolerance) and adds the v2 classes:
  missing/duplicate/undeclared/orphaned record file; cross-cell key-set
  drift; within-calibration MC-stop mismatch; cross-calibration MC-stop
  equality wrongly enforced (nearest-true control); `FINITE_STOP`/null and
  `NEVER_STOPPED`/numeric confusions; final-prefix crossing mislabeled;
  unauthenticated legacy `T−1` normalization; headline label over a
  both-finite population; uncorrected p-value; wrong/multiple seeds; index
  digest mismatch; m≠10 or Random-K-omitted family; UTF-8 tie-order
  violation; per-surface version `true`/`1.0`/`"2"`/`3`; the non-object
  sidecar matrix; symlinked/dangling/empty/newest-wins canonical selection;
  git-disappeared release. Every negative fixture FAILs in its declared
  mode(s) and is paired with a nearest-true sibling that PASSes.

## Won't Do

- Run or regenerate any model-executed scientific result — no training, no
  inference, no new model runs, no multi-draw Random-K rerun
  (`EXECUTION_MODE=SOURCE_CONTRACT_ONLY`). The D7(b) resampling analysis
  over retained records is the sole sanctioned computation (decision D7(b)).
- The observed-paired study (separately authorized lane; the reserved
  observed profile identifier stays unsatisfiable by constructed-reference
  artifacts).
- Cherry-pick `2709624b` or `f8ba2042`, or port the Track A / Track A′
  exploit test files wholesale (sign-off §6: REIMPLEMENT; Phase 1: local
  commits are review inputs, not transplantable authority).
- Edit the D6 manuscript bundle, historical `paper_exports/`, or any
  committed result bytes; create/move tags or releases; reuse the historical
  `stopdff-fair-qa-retest` tag namespace; add a license.
- Wholesale adoption of PR #32 or PR #41; fixing issues #33–#40 beyond the
  narrow fail-closed/identity-bound/nonempty-evaluation principles already
  embedded in this spec's own paths.
- Tracker writes outside the sign-off §7 Phase-6 sequencing (thread replies,
  #42 body edit, close-as-superseded come only after the successor's exact
  head is green and reviewed; #42 is never retargeted; #32/#41 are not
  merged or closed by this transaction). The transaction operating procedure
  itself stays in `handoff_prompt_camera_ready_2026-08-18.md` and the
  sign-off — deliberately not restated as testable rules.
- Maintaining a second independent revision token or a v1-compat write path
  (writers emit strict v2 only).

## Risks

- Fail-open defect certifies bad evidence — mitigated (R-012/R-019/R-021/
  R-022/R-069: per-leg fixtures, widened adversarial corpus, end-to-end
  mutation wiring incl. grid/inference pins, no bypass doors, release-path
  canonical selection).
- Researcher degrees of freedom re-enter the inference — mitigated
  (R-050..R-057: everything frozen before outcomes are seen; derived seed;
  recorded index digest; new-analysis discriminator).
- Silent estimand relabeling (headline vs finite-only) — mitigated
  (R-048/R-049/R-054 population enum; R-068 label binding; R-011 digests).
- v1→v2 laundering (old artifacts or versions slipping into strict paths) —
  mitigated (R-059/R-060/R-064: bool-safe version-first checks, named
  legacy loader, pinned sidecar boundary).
- Author-decision and EXTERNAL items block closure regardless of code
  quality — accepted by design (`CAMERA_READY_CLOSURE` fails on the Holm row
  until D7(b) outputs exist; QA-012 blocks closure until inventoried).
- Single-shot Phase-4 regeneration produces goldens the suite then trusts —
  mitigated (R-010 fixture-locked parity against the pinned historical
  implementation; hand-editing barred; any later producer/estimand change
  invalidates the package and requires a new run identity).

## Open Questions

- **RESOLVED-PENDING-ACK — pairing-population narrowing (R-052; not an
  OQ)**: the seed input `pairing_population_keyset_sha256` is pinned to the
  2,249 complete-pair key set — zero in-package exclusions; the 9
  upstream-unpaired items (2,258 retained − 2,249 paired, per the preserved
  producer's retention counts) are pre-package retention documentation in
  provenance, never in-package `excluded_keys`. This narrows the lineage
  term (handoff §8 / v1 R-008: pairing population = complete ∪ excluded)
  for the frozen v2 package only; flagged for reviewer acknowledgment at
  the next shuttle round. Nothing blocks on it.
- **OQ-V2-001 — cell/reference ID spellings**: The sign-off freezes the
  calibration IDs (`shared`, `format_specific`) and the 5×2 shape, but not
  the exact v2 reference-ID spellings or the `cell_id` composition rule. The
  preserved producer's arm spellings are `idealized, krandom, khard,
  kdisjoint, klex` and its calibration flag spelled `performat` for the
  format-specific map. Proposal: keep the producer's reference spellings;
  `cell_id = "<reference_id>__<calibration_id>"` (ASCII lowercase, so UTF-8
  tie order is plain lexicographic); record the `performat`→`format_specific`
  identity mapping in provenance, never silently aliased. Exact IDs are
  recorded in the expectations contract at regeneration (R-044).
- **OQ-V2-002 — legacy-loader surface naming**: R-060 requires an
  "explicitly named" legacy loader; the name itself is free. Proposal:
  `legacy.py::load_legacy_v1_document` (one module, one entry point, no
  strict-path imports of it).
- **OQ-V2-003 — which auxiliary surfaces are separately versioned**: The
  sign-off lists "records envelope, if versioned" and "inference/grid
  sidecars, if separately versioned". D1 puts grid/inference in-profile
  (covered by the profile's version), so the open choice is only whether
  record JSONL lines carry an envelope version and whether receipts are
  versioned; whichever exist must use R-058/R-059. Decide at implementation;
  no contract text constrains the choice beyond that.

## Appendix A — carry-forward mapping (v1 R-001..R-039 → v2)

| v1 rule | Disposition | By what / why |
|---|---|---|
| R-001 | AMENDED | D1 multi-cell package shape + sign-off §4.1 (`schema_version: 2`, v2 profile ID); semantic block verbatim-carried |
| R-002 | CARRIED | unchanged |
| R-003 | AMENDED | sign-off §2.2 / plan v3 §3.2: closed per-family vocabularies, no overloaded global stop integer |
| R-004 | CARRIED | unchanged |
| R-005 | CARRIED | unchanged (recomputed per `records/<cell_id>.jsonl`) |
| R-006 | AMENDED | sign-off §2.3 via R-048/R-049: headline = sentinel-coded all-pairs; finite-only = separately named secondary |
| R-007 | SUPERSEDED | by R-045–R-047 (§2.2): canonical `FINITE_STOP`/`NEVER_STOPPED` events; the v1 boundary law survives only as the pinned historical sentinel convention inside R-045/R-046 |
| R-008 | AMENDED | R-042: exactly 2,249 complete keys, byte-exact identical across all ten cells |
| R-009 | CARRIED | unchanged |
| R-010 | AMENDED | parity target extended to the all-pair sentinel-coded headline estimator vs the preserved fair-QA producer; goldens under NumPy 2.4.6 (D5) |
| R-011 | AMENDED | digest fields extended (reference/calibration/event/population); 7B/M3 closed vocabularies reconciled; `check_comparable` production-wired (plan v3 §5.7) |
| R-012 | AMENDED | leg families extended: grid (R-040..R-044), ingress (R-064), inference (R-050..R-056), ledger↔anchor (R-065), git-object fail-closed (R-066) |
| R-013 | AMENDED | sign-off §2.1: expectations pin grid/inference identities semantically, not merely a package hash |
| R-014 | AMENDED | §4.2/§5: legacy entry only via named legacy loader (R-060); sidecar boundary per R-064 |
| R-015 | AMENDED | §2.3/§3: headline intervals bind to the sentinel-coded all-pair population; identity names its population; D7(b) governs the ten-cell intervals |
| R-016 | CARRIED | unchanged |
| R-017 | AMENDED | v1 source-mode minimum-positive-check-set floor restated with the v2 rule families (carried-with-restated-floor: ingress R-020/R-059/R-061/R-062/R-064, in-package validation R-001–R-011 incl. R-040–R-056, receipts R-036); reaffirmed by sign-off §1 |
| R-018 | CARRIED | unchanged |
| R-019 | AMENDED | corpus widened with the v2 adversarial classes |
| R-020 | AMENDED | version duties moved to R-059 (version-first on every surface; errors name `VERIFIER_REVISION` — closes A′-F1/F2); ingress hooks per R-067/R-062 |
| R-021 | AMENDED | mutation set extended to grid pins and inference pins |
| R-022 | CARRIED | unchanged |
| R-023 | AMENDED | `anchored_source_commit`; single version checker; Holm rejected-ID storage; genuine-blocker `blocking_task` discipline (D3) |
| R-024 | CARRIED | unchanged |
| R-025 | AMENDED | plan v3 §3.5 + Track A F1: recorded `historical_nonconfirmatory`; `rng_pinned=false`; archived+fresh identities; INACTIVE discriminated `predeclared_multidraw_family`; headline⇔disposition joint validation |
| R-026 | CARRIED | unchanged |
| R-027 | AMENDED | sign-off §1 immutable statements + §2.3 prohibited headline labels |
| R-028 | CARRIED | unchanged (NumPy 2.4.6 required, not denied) |
| R-029 | CARRIED | unchanged |
| R-030 | CARRIED | unchanged |
| R-031 | AMENDED | record allowlist extended for `event_status`/`terminal_imputation` categoricals and nullable stop steps |
| R-032 | CARRIED | unchanged |
| R-033 | CARRIED | unchanged |
| R-034 | CARRIED | unchanged (resample matrix regenerated, never deserialized) |
| R-035 | AMENDED | reconciliation extended with the grid↔records bijection (R-041) |
| R-036 | CARRIED | unchanged (versioned receipts use the single checker) |
| R-037 | AMENDED | exit-code set pinned `0/1/2/3/4` (v1 QA-019 promoted to contract) |
| R-038 | AMENDED | D4: doc-audit widened to the handoff seven-target set |
| R-039 | AMENDED | R6 wired into the actual release path with end-to-end coverage (R-069) |

Dropped: none — every v1 rule survives carried, amended, or (R-007)
superseded by a stronger representation.

## Appendix B — test plan (sign-off Phase-2 RED coverage → rule IDs)

Every rule above is encodable as a failing-first test. The RED suite must
cover, before implementation (sign-off Phase 2, mapped):

| # | RED coverage item (sign-off Phase 2) | Rule(s) |
|---|---|---|
| 1 | exact 5×2 grid completeness | R-040 |
| 2 | cell-to-record-file bijection | R-041, R-035 |
| 3 | 2,249-key exact-set equality | R-042, R-008 |
| 4 | held-fixed MC identity rules | R-043 |
| 5 | shared-versus-format_specific calibration distinction (no cross-calibration MC-stop equality) | R-043 |
| 6 | canonical finite/never event representation | R-045, R-031, R-061 |
| 7 | genuine final-prefix finite stops | R-046 |
| 8 | ambiguous legacy sentinel refusal | R-047 |
| 9 | historical all-pair estimator parity | R-048, R-010 |
| 10 | finite-only estimand separation | R-049, R-054, R-068 |
| 11 | exact D7(b) resampling and Holm procedure | R-050..R-057, R-015 |
| 12 | Random-K non-headline rules | R-025, R-023 |
| 13 | R1–R5 repairs | R-063 (R1), R-064 (R2), R-059 (R3), R-062 (R4/R5), R-067 (F6 ingress siblings) |
| 14 | R6 canonical release-path containment | R-069, R-039 |
| 15 | bool-safe versioning across every surface | R-059, R-058, R-061 |
| 16 | v1/v2 transition behavior | R-060, R-014 |
| 17 | unknown-object versus non-object legacy-sidecar matrix | R-064 |
| 18 | release-level independent grid and inference anchoring | R-044, R-013, R-021, R-052, R-053 |

Supplementary RED coverage required by this spec beyond the Phase-2 list:
Track A reimplementations (R-065 ledger↔anchor equality; R-066
git-disappeared release FAIL; R-067 native-finite ordered CI; R-068
estimand-label binding; R-025 headline⇔disposition joint gate); CI receipt
shape incl. the A′-F4 defect fixture (R-070); `CAMERA_READY_CLOSURE`
inventory and Holm-row blocking (R-071); QA-012 detector and manifest
(R-072); vocabulary prohibitions (R-027, R-048, R-057); the widened
adversarial corpus with nearest-true siblings (R-019); doc-audit content
tests (R-038); CLI exit codes end-to-end (R-037, R-021); no-network/import
scans (R-028, R-034); create-once publish (R-016); vacuous-input refusals
(R-033); rights sentinel-leak tests (R-026).
