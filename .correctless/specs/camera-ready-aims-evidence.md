# Spec: Camera-Ready AIMS Evidence Flow

## Metadata
- **Task**: camera-ready-aims-evidence
- **Recommended-intensity**: high
- **Intensity**: standard
- **Intensity reason**: user override. Detection recommended `high` (keyword signal: credential/token/secret appear in the rights/leak rules; TB-xxx, antipattern, and QA-history signals dormant; humility qualifier — only 1 completed feature in workflow history). User set standard; `allow_intensity_downgrade` is not disabled in workflow-config.
- **Override**: lowered
- **Branch**: feature/camera-ready-aims-spec
- **Research**: .correctless/artifacts/research/camera-ready-aims-evidence-research.md
- **Handoff-source SHA-256**: a9d69495423545324a75ebd9021e1151c9eebd60d072c6b818a6debe3f9335f6 (handoff_prompt_camera_ready_2026-08-18.md, untracked — identity recorded per review RF-24)

## What

The COLM AIMS camera-ready must be supported by the smallest independently
validated repository flow. The standing reviewer boundary: the existing analyses
are **constructed-reference sensitivity audits** — they do not establish that MC
preserves or changes an observed open-ended stopping policy; Random-K failed to
reproduce under a new draw; reproducibility is incomplete. This feature adds the
repo-side evidence contract that makes the defensible negative-result claim
independently checkable: a strict schema-versioned constructed-reference result
profile with an explicit semantic layer, exact pair/censoring count identities,
fail-closed provenance binding, and a claim-ledger + two-mode verifier namespace
(`reproducibility/colm_aims_2026/`) whose source-contract mode tops out at
`PASS_SOURCE_ONLY` and whose release mode fails closed without independently
anchored expectations and verified rights. Source document for every gate:
`handoff_prompt_camera_ready_2026-08-18.md` (§0, §6–§10). It composes with the
StopDFF v5 fail-closed/create-once contract family instead of reinventing it.
Governing venue facts (verified 2026-08-18): the AIMS workshop ("AI Measurement
Science: Toward Rigorous AI Evaluation", COLM 2026) camera-ready is
**2026-09-21** and the workshop is non-archival — the repo evidence package is
the artifact of record.

## Rules

### Profile and semantics

- **R-001** [unit]: The new writer emits only the strict schema-versioned
  profile, whose semantic block carries exactly these pinned fields/values:
  `trajectory_source: constructed_reference`, `observed_open_ended: false`,
  `observed_open_ended_answers: false`, `observed_open_ended_stopping_actions:
  false`, `pairing_unit: matched_item_prefix_grid`,
  `pairing_is_observed_sessions: false`, `supports:
  reference_sensitivity_diagnostic`, `does_not_support:
  actual_decision_preservation_or_format_effect`. The validator rejects a
  missing, renamed, altered, or unknown key inside the block; historical
  `format="QA"` identifiers parse for compatibility but never substitute for
  the semantic layer. (Format pinned per AP-031: field vocabulary is verbatim
  from handoff §8; on implementation the authoritative producer is
  `reproducibility/colm_aims_2026/schema.py`.)
- **R-002** [unit]: A constructed-reference artifact asserting
  `observed_open_ended: true` fails validation. A distinct future profile
  identifier is reserved for genuinely observed studies; the
  constructed-reference validator never accepts it and no code path converts
  one profile into the other.
- **R-003** [unit]: Every arm identifies its construction, scalar-vs-K-way
  status, selector/scorer, candidate-pool role, correctness assignment,
  calibration/continuation role, seed contract, and reporting eligibility.
  `idealized` is scalar prefix-to-gold cosine with oracle-assigned correctness;
  a payload declaring all arms K-way while containing an idealized arm fails.
- **R-004** [unit]: encode→decode roundtrip of a strict-profile artifact is
  lossless (equal value), and writers reject non-finite floats at write time
  (`allow_nan=False` semantics).

### Pair and censoring decomposition

- **R-005** [unit]: The checker recomputes per-cell counts from per-item
  records and enforces exactly:
  `n_complete = n_both_finite + n_mc_finite_ref_timeout + n_mc_timeout_ref_finite + n_both_timeout`;
  `n_pairing_population = n_complete + n_excluded_or_unpaired`;
  `n_excluded_or_unpaired = sum(exclusion_reason_counts.values())`;
  `n_mc_timeout = n_mc_timeout_ref_finite + n_both_timeout`;
  `n_ref_timeout = n_mc_finite_ref_timeout + n_both_timeout`.
  A ±1 mutation of any recorded count fails. (Field names verbatim from
  handoff §8 — AP-031.)
- **R-006** [unit]: All four joint-class rates use `n_complete` as denominator,
  are `null` when it is zero, and otherwise sum to 1 within the artifact's
  declared tolerance. The finite-only timing summary uses exactly
  `n_both_finite` and declares its conditional estimand. A retained
  sentinel-coded historical summary is separately named and never pooled. A
  cell declaring `n_pairing_population == 0` is a typed error (consistent with
  the empty-evaluation refusal), never a trivially-passing cell.
- **R-007** [unit]: Timeout rule is the existing one, exactly: zero-indexed
  `0 <= stop_step < trajectory_horizon` is finite, `stop_step >=
  trajectory_horizon` is timeout — red/green tests on both sides of the
  boundary. Missing/malformed stops and MC/reference grid mismatches become
  exclusions with reason, never imputed stops.
- **R-008** [unit]: Complete-pair and excluded/unpaired key sets are disjoint
  and duplicate-free; their union equals the declared pairing-population key
  set and hash. Duplicate pair keys fail closed. Each excluded unit carries
  exactly one primary exclusion reason; missing reasons are recorded
  `UNKNOWN_NOT_INFERRED`, never guessed; secondary diagnostics are not counted
  in `exclusion_reason_counts`. The stable item-key derivation (hash function
  and text normalization) is pinned in the profile so a third party with
  legitimate source access can re-derive and match keys. Keys are compared
  byte-exact after derivation; duplicate-detection fixtures include Unicode
  normalization-variant near-duplicates.
- **R-009** [unit]: Arm reversal leaves the diagonal joint classes unchanged,
  exchanges the two off-diagonal classes and the two timeout totals, and flips
  a timing-summary sign only when that summary's frozen definition is
  antisymmetric (property-based over generated tiny record sets).
- **R-010** [unit]: On an all-finite record set, the finite-only timing summary
  reproduces the historical paired summary exactly (same items, same
  estimator; includes DP/myopic paths where the historical summary defines
  them). Parity is asserted against golden values produced by the pinned
  historical implementation (fixture-locked), never by running the new
  implementation twice.
- **R-011** [unit]: Every analysis cell carries a digest over all
  estimand-defining fields (arm identities, pairing definition, timeout
  parameters, denominator policy, calibration/continuation identities,
  Random-K draw identity). Changing any such field changes the digest;
  pooling or comparing cells with differing digests is refused. The declared
  numerical tolerance (R-032) is an estimand-defining digest field. Cell
  identifiers are unique per artifact; duplicates fail closed.

### Provenance and certification

- **R-012** [unit]: Release mode fails closed per leg, one fixture per missing
  or `UNRESOLVED` binding: schema/profile + producer/helper hashes; semantic
  command; seeds; dirty-state identity; fit/eval split names, counts, key-set
  hashes, zero-overlap check; calibration and continuation identities; input
  and `split_metadata.json` hashes; MC-build-after-split freshness, coverage,
  retention policy/counts; scoring/selector model repository namespace/id +
  immutable revision — a full-length commit SHA; short hashes, tags, branch
  names, and bare repo ids are rejected (repo ids are reassignable — observed
  live on the `datasets/qanta` handle) — or a complete canonical byte-digest
  manifest, plus a content-level hash of any loaded weights file;
  tokenizer/config,
  dtype/device class/numerical settings; runtime package versions — plus
  missing rights status, missing presentation manifest, and missing anchored
  expectations. An explicitly empty evaluation dataset errors before any
  report is emitted; a genuine tiny nonempty dataset remains valid. Release
  mode evaluates and reports all legs (collect-don't-halt, except on
  unreadable inputs), each failure naming the leg id, expected vs observed
  binding, and a remediation class (`ARTIFACT_DEFECT | MISSING_EXPECTATION |
  AUTHOR_DECISION_REQUIRED | EXTERNAL`); and it recomputes every non-EXTERNAL
  claim-ledger row's status from current verification, failing on any row
  whose recorded status is stronger than the recomputed one.
- **R-013** [integration]: No self-attestation: certification requires an
  expectations file located outside the artifact tree it certifies, anchored
  to a reviewed source commit and the frozen claim ledger, and the verifier
  cross-checks that anchor before consuming expectations. An artifact plus its
  own generated manifest — timestamps and self-reported booleans included —
  reaches at most source-level status. Containment decisions (expectations
  outside the verified tree) use fully resolved, symlink-free paths; the
  anchor check is a string-exact comparison of recorded commit/ledger
  identities that works without a git checkout, with an additional
  object-existence check when a repository is available.
- **R-014** [integration]: Verifier runs never mutate inputs: the artifact
  tree's byte hashes are identical before and after every run. Artifacts bound
  to a superseded, dirty, or unresolved estimand-defining dependency closure
  classify `HISTORICAL_NONCERTIFYING`; current metadata is never backfilled
  into old bytes; only a producer/closure change invalidates an artifact.
  Legacy profiles parse and are refused only on a demonstrably missing named
  invariant or a failed exact binding check — never merely for predating the
  strict schema. Aggregate-only files cannot certify per-item paired claims.
  The known legacy profile set is enumerated as a fixture corpus captured
  (sanitized) from the repo's actual historical artifact formats (the
  `paper_exports/csli.json`, `paper_exports/audit_card.json`, and
  `paper_exports/calibration.json` families); legacy parse tests run against
  captured bytes, not reconstructions.
- **R-015** [unit]: Every reported top-level aggregate and paired interval
  recomputes from retained per-question records within declared tolerance;
  recomputation mismatch → FAIL; absent records → non-certifying.
  Interval-bearing cells additionally record the interval procedure, draw
  count, and resampling seed(s); interval recomputation re-runs the recorded
  procedure deterministically, and missing interval identity leaves the
  interval non-certifying.
- **R-016** [integration]: Evidence writers publish create-once via the
  no-replace primitives of `scripts/stopdff_v5/fileio.py` (or an equivalent
  with identical guarantees): an interrupted publish leaves no parseable
  partial artifact at the final path, and a second publish to an existing path
  fails rather than clobbering. A kill-mid-publish/retry test asserts that a
  retried publish of identical content succeeds with exactly one resulting
  artifact, and the staging-debris policy is pinned (auto-reclaim or an
  enumeration command).

### Verifier modes and namespace

- **R-017** [unit]: The source-contract mode's verdict vocabulary is a closed
  enum whose strongest member is `PASS_SOURCE_ONLY`; no code path in that mode
  emits a release/camera-ready token, and the rendered summary states what
  source-only verification does NOT certify. Author-side verdicts never use
  the ACM-v1.1 third-party terms "Reproduced"/"Replicated". Source mode's
  minimum positive check set is profile validation (R-001–R-011), typed
  ingress (R-020), and receipt emission (R-036); the rendered summary lists
  the checks performed.
- **R-018** [integration]: Everything new lives under
  `reproducibility/colm_aims_2026/` (plus tests/docs);
  `scripts/verify_audit_release.py` stays byte-identical and its existing
  suite green; no doc redefines the legacy verifier as camera-ready
  certification.
- **R-019** [unit]: The suite ships the adversarial fixture corpus — the
  handoff's eight (constructed-as-observed; denominator mismatch; timeout
  mismatch; hash/model/split mismatch; historical/current substitution; empty
  evaluation; unbound calibration; unverified-rights inclusion) plus the
  review-added five (stale-PASS ledger row; zero-artifact tree; empty ledger;
  empty presentation manifest; oversized declared tolerance) — each of which
  must FAIL, each paired with a nearest-true sibling (minimal compliant
  variant) that must PASS, and each mapped to the verifier mode(s) in which
  it must FAIL.
- **R-020** [unit]: Typed ingress: artifact bytes are validated into typed
  records at the load boundary; malformed, truncated, or unknown-keyed files
  produce a typed error naming the file and field — no partial semantic
  processing, silent key-dropping, or truthiness coercion downstream. The
  `schema_version` field is validated before any other check; an unsupported
  version produces a typed error naming the artifact's version, the supported
  range, and the matching verifier revision — never a generic unknown-key
  error. Error messages identify files by repo-relative or
  artifact-tree-relative path (the form the R-026 sentinel test accepts).
- **R-021** [integration]: Bindings demonstrably reach the verdict: an
  end-to-end test runs the real verifier CLI over a complete tiny fixture
  package, then mutates one expectation at a time (model revision, split hash,
  producer hash, calibration identity, ledger anchor) and asserts the verdict
  flips each time. The test invokes the documented command line (R-037) via
  subprocess — never an imported `main()` — and asserts exit codes (R-037)
  and sentinel-free output (R-026) in addition to the verdict flips.
- **R-022** [unit]: The verifier CLI and its config fail closed on unknown
  flags or keys (error, not no-op), and no flag, environment variable, or
  config door disables a release gate. The CLI parser sets
  `allow_abbrev=False` so abbreviated flag forms cannot smuggle past the
  unknown-flag check.

### Ledger, Random-K, rights

- **R-023** [unit]: Each claim-ledger row carries: `claim_id`, exact manuscript
  location and wording, estimand and allowed scope, producer entrypoint and
  dependency closure, input/split/model/calibration identities, artifact and
  renderer identities, independent verifier/oracle, rights status, status in
  the closed enum `PASS | FAIL | UNVERIFIED | EXTERNAL`, and blocking repo
  task. The ledger distinguishes manuscript identity (submission PDF SHA-256
  `6de23119…dabf10a`), historical submission artifacts, historical
  Random-K/v5 results, current source, and future clean evidence. PR #41
  hazard reports may appear only if the exact manuscript cites them.
- **R-024** [integration]: `EXTERNAL`-typed rows (manuscript wording
  qualifications, Figure 1 framing, causal-wording narrowing,
  bootstrap-interval description, Appendix D / page-11 layout inspection,
  venue final-format/anonymity/AI-disclosure/supplement rules, content
  rights) are immune to repo tooling: every tool run leaves them
  byte-identical, and an `EXTERNAL → PASS` edit without a human-attribution
  field fails ledger validation. Repository green tests never substitute for
  an EXTERNAL item. Venue-rule rows record only officially published facts
  with source and as-of date — the AIMS workshop page publishes no artifact,
  supplement, or AI-disclosure rules (as of 2026-08-18), so repo gates are
  recorded as self-imposed, never attributed to the venue. The tool list for
  the immunity test is enumerated: the writer, the checker, the verifier in
  both modes, and the ledger validator.
- **R-025** [unit]: Random-K disposition gate: a headline-eligible Random-K row
  requires an explicit `author_decision` naming one of exactly two
  dispositions — historical/nonconfirmatory (excluded from headlines) or
  predeclared multi-draw family (all seeds/draws retained, draw sensitivity
  and multiplicity scope reported). Draw identity is estimand-defining
  (R-011), so a substituted favorable draw changes the digest and is refused.
  This feature implements metadata/validation support only — it never runs
  the multi-draw experiment (SOURCE_CONTRACT_ONLY).
- **R-026** [unit]: The rights inventory assigns every included path one of
  `VERIFIED_ALLOWED | VERIFIED_RESTRICTED | UNVERIFIED |
  AUTHOR_DECISION_REQUIRED`; release mode requires every included path
  `VERIFIED_ALLOWED` and fails on any other value or any uninventoried
  included path. Each row names its upstream terms basis (e.g., NAQT
  proprietary/excluded; PACE-archive author-retained packet copyright; QANTA
  permissioned-aggregation-without-redistribution) rather than a bare enum
  tick. Fixtures are synthetic (no raw quizbowl text, no real qids);
  a sentinel-leak test proves error paths reference items by opaque stable
  keys and never emit restricted content, credentials, or local absolute
  paths.

### Never (prohibition rules)

- **R-027** [unit]: No rendered output, verdict string, or doc produced by this
  feature asserts observed decision preservation, uses unqualified "QA
  effect", or drops the "constructed QA reference" qualification;
  "would hide real shifts"-style causal wording is narrowed to
  insensitivity-to-sub-threshold-shifts. The enforceable core is an
  enumerated banned-phrase and required-qualifier list maintained as a
  fixture file; the vocabulary test asserts it over every renderer output
  (implication-level judgment stays with human review). If the intended headline becomes
  actual preservation/change, the only sanctioned output is
  `observed_paired_claim=OBSERVED_PAIRED_STUDY_REQUIRED`.
- **R-028** [unit]: No network, model downloads, or training in this feature:
  the namespace's test conftest installs a no-network guard (primary gate),
  and an import scan over `reproducibility/colm_aims_2026/` rejects an
  enumerated deny-list (`requests`, `httpx`, `urllib.request`,
  `huggingface_hub`, `transformers`, `torch`). Fixtures are tiny and
  synthetic.

### Venue-grounded additions (author-approved 2026-08-19; self-imposed, not venue-mandated)

- **R-029** [unit]: The strict profile carries an `llm_involvement` block
  declaring, per contribution axis (reference construction, data/plot
  creation, evaluation), whether LLMs contributed — `none` is an explicit
  value, never an absent field — plus a free-text tool/version note when any
  axis is non-none. The validator rejects artifacts missing the block.
  Rationale: COLM 2026 treats LLM use in data/plot creation as disclosable, so
  the paper's disclosure statement becomes mechanically derivable from its
  evidence (research brief, finding 2).
- **R-030** [unit]: The claim ledger supports an `archival_doi` field for the
  release snapshot. A ledger asserting "Artifacts Available"-grade status
  without a DOI-class archival identifier fails validation — a GitHub URL does
  not qualify (ACM v1.1). Absent any Available-grade assertion the field is
  optional. Rationale: the workshop is non-archival, so a DOI snapshot is the
  only Available-grade anchor for the artifact of record (research brief,
  findings 1 and 4).

### Review-hardened additions (all 26 /creview findings accepted 2026-08-19)

- **R-031** [unit]: Per-item records are non-reversible: they contain only
  opaque stable keys, enumerated categorical fields, and numeric values
  (scores, stop steps, timeouts); the validator rejects any string field
  outside an enumerated identifier allowlist. Fixture: a record with a
  free-text field → FAIL. (RF-01)
- **R-032** [unit]: The schema pins a maximum admissible numerical tolerance;
  a declared tolerance above it fails validation, and the declared tolerance
  is an estimand-defining digest field (R-011). Fixture: oversized tolerance
  → FAIL. (RF-02)
- **R-033** [unit]: No vacuous verdicts: both modes fail with a typed error
  naming the resolved path and expected layout when zero candidate artifacts
  are found; any PASS-class verdict requires ≥1 validated artifact enumerated
  in the rendered summary and receipt; release mode additionally requires ≥1
  retained claim row and ≥1 manifest-declared artifact. (RF-04; precedent:
  the MA-019 present-but-empty gate, commit ae0e2487.)
- **R-034** [unit]: Deserialization safety: evidence ingestion is JSON/JSONL
  only; `pickle`, `marshal`, `torch.load`, and non-safe YAML loading never
  appear in `reproducibility/colm_aims_2026/` (AST/import scan test, sibling
  of R-028's scan). (RF-05)
- **R-035** [unit]: Release mode reconciles the evidence package against the
  presentation manifest in both directions: manifest-declared-but-absent →
  FAIL; present-but-undeclared → FAIL absent an explicit per-file allowlist
  entry; the rights inventory covers every file found, not merely every file
  declared. (RF-06)
- **R-036** [integration]: Every verifier run emits a schema-versioned JSON
  receipt — mode, verdict, per-leg outcomes, input-tree hash,
  expectations-anchor hash, verifier code hash, timestamp — to a path outside
  the verified artifact tree, published create-once under run-scoped unique
  names (R-016 primitives). (RF-07; handoff Tranche B's "independent
  verification receipt".)
- **R-037** [integration]: CLI contract: the documented invocation is
  `python -m reproducibility.colm_aims_2026.verify` from the repo root;
  direct-path invocation either bootstraps sys.path (repo convention) or
  errors naming the module-run form. Distinct exit codes are pinned for
  mode-ceiling pass, gate FAIL, usage/config error (R-022), and typed ingress
  error (R-020); R-021's end-to-end test asserts them. (RF-09)
- **R-038** [integration]: Documentation: `reproducibility/colm_aims_2026/
  README.md` pins both modes' exact invocation, the input layout including
  the expectations-file location, verdict enum semantics, exit codes and
  receipt location, and a one-line disambiguation from
  `scripts/verify_audit_release.py`. `reproducibility/source_to_claim.md`
  gains a historical-scope header naming the manuscript it maps and pointing
  to the new ledger, and the ledger records it as a
  historical-submission-artifact document. (RF-11; restores the handoff
  Tranche C documentation-audit deliverable.)
- **R-039** [unit]: Supersede/retire: evidence packages publish into
  run-scoped (or content-addressed) directories; canonical selection happens
  only via the ledger/expectations pointer; retiring a defective artifact is
  a ledger status change plus a new run directory — historical bytes are
  retained, never republished over. (RF-12)

## Won't Do

- Run or regenerate any real scientific result — no training, no inference, no
  multi-draw Random-K rerun (EXECUTION_MODE=SOURCE_CONTRACT_ONLY).
- The observed-paired study (separately authorized design lane; this spec only
  reserves the observed profile identifier that constructed-reference
  artifacts can never satisfy).
- Edit manuscript TeX/PDF, historical `paper_exports/`, or any committed
  result bytes; create/move tags or releases; reuse the historical
  `stopdff-fair-qa-retest` tag namespace; add a license.
- Wholesale adoption of PR #32 or PR #41; fixing issues #33–#40 (only the
  narrow fail-closed/identity-bound/nonempty-evaluation principles of
  #35/#37/#38 apply, and only to this feature's own paths).
- Tracker writes, merges, force pushes, history rewriting — transaction
  operating procedure stays in `handoff_prompt_camera_ready_2026-08-18.md`.
- The root-controller transaction protocol (budgets, worktrees, scouts, push
  ceremony) — not repo-testable contract; deliberately not restated here.

## Risks

- Fail-open defect certifies bad evidence — mitigated (R-012, R-019, R-021,
  R-022: per-leg fixture matrix, adversarial corpus, wiring test, no bypass
  doors).
- Overclaim wording reaches docs/verdicts — mitigated (R-017, R-027 vocabulary
  tests).
- Restricted content or secrets leak via fixtures/errors — mitigated (R-026
  sentinel tests; synthetic fixtures; review of any diff touching
  result-bearing paths).
- Hand-edited results pass as evidence — mitigated (R-015 recomputation makes
  hand-edits fail; R-014 immutability harness).
- Estimand drift / silent pooling across analysis families — mitigated (R-011
  digest refusal; R-010 parity check).
- Verifier crash-loops on pathological local files — accepted (local CLI, no
  service exposure; R-020 typed errors bound the blast radius).
- [integration] rules lack formal Entry/Through/Exit contracts because
  ARCHITECTURE.md defines no entrypoints — deferred (OQ-003; optionally run
  /carchitect to define the verifier CLI as an entrypoint).
- Author-decision items (Random-K, rights, manuscript wording) block
  release-mode PASS regardless of code quality — accepted (by design:
  `AUTHOR_DECISION_REQUIRED` is a feature, not a defect).

## Open Questions

- **OQ-001**: Random-K author disposition — remove from headline
  (historical/nonconfirmatory) or predeclared multi-draw family? Schema
  supports both (R-025); release mode cannot PASS a Random-K-bearing
  presentation manifest until decided. AUTHOR_DECISION_REQUIRED. **Resolved
  2026-08-19: author explicitly deferred the decision —
  AUTHOR_DECISION_REQUIRED stands; nothing in this feature blocks on it.**
- **OQ-002**: Confirm scope interpretation: this spec governs the repo-side
  deliverable contract (schema/verifier/ledger); the transaction protocol
  stays in the handoff document as operating procedure. Alternative — speccing
  the transaction itself — would be far less pytest-able. **Resolved
  2026-08-19: confirmed — deliverable contract only.**
- **OQ-003**: No entrypoints in ARCHITECTURE.md, so the five [integration]
  rules carry no Entry/Through/Exit blocks. Define the verifier CLI as an
  entrypoint via /carchitect, or accept integration rules without formal
  contracts?
- **OQ-004**: When claim-reachability later confirms Tranche-A paper-path
  fixes (borrowing #35/#37/#38 principles on exact paper producers), should
  each adopted fix get its own minimal follow-up spec? (Recommended: yes —
  this spec pins the contract; per-fix specs stay small and red/green-driven.)
- **OQ-005**: Are editable manuscript sources available read-only? If not,
  `external_manuscript: EXTERNAL_MANUSCRIPT_REQUIRED` stands and several
  checklist rows can never leave EXTERNAL from inside this repo.
- **OQ-006**: Does any NAQT-authored question text exist in the repo's
  processed splits? NAQT text is affirmatively proprietary and excluded from
  the public QANTA release — worth an explicit content-level exclusion check
  feeding the rights inventory (research brief, finding 14).
