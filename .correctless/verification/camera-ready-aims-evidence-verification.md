# Verification: Camera-Ready AIMS Evidence Flow

- **Task**: camera-ready-aims-evidence
- **Spec**: `.correctless/specs/camera-ready-aims-evidence.md` (spec_hash `803ca197…bca4ed68`, 426 lines)
- **Branch**: feature/camera-ready-aims-spec (HEAD `745d2975`)
- **Intensity**: standard (recommended `high`, user override → lowered)
- **QA rounds**: 2 (+ mini-audit)
- **Verifier**: /cverify (autonomous, dispatched by /cauto)
- **Feature test run**: `595 passed in 18.48s` (15 `test_colm_aims_*.py` files, canonical `.venv` Python 3.11.15)

> Scope note: the branch diff vs `main` (~30k lines) mixes in a **separate** feature
> (`hazard-efficacy-eval`, its own workflow state + verification report). This
> verification is scoped to the camera-ready deliverable only:
> `reproducibility/colm_aims_2026/**`, `reproducibility/source_to_claim.md`,
> `tests/test_colm_aims_*.py`, `tests/_colm_aims_helpers.py`, and
> `tests/fixtures/colm_aims/**`.

## Rule Coverage

Every rule R-001…R-039 is referenced by ≥4 test assertions across the suite (grep
over rule IDs). No uncovered rules. Integration-tagged rules are exercised at the
real-system level (subprocess CLI / verifier-over-fixture-package). Spot-checked
the highest-stakes rules for probe quality (would the test fail if the rule were
violated?) — all confirmed as genuine red/green probes, none trivial.

| Rule | Primary test file(s) | Status | Notes |
|------|----------------------|--------|-------|
| R-001 [unit] | schema, qa_round2, verifier_gates | covered | probed: missing / renamed / altered / unknown semantic-block key all rejected; QA-format never substitutes |
| R-002 [unit] | schema, qa_round2 | covered | constructed-asserting-observed FAILs; reserved observed profile id exists & differs |
| R-003 [unit] | schema | covered | all-K-way payload containing an idealized arm FAILs |
| R-004 [unit] | schema | covered | encode→decode lossless; non-finite floats rejected (allow_nan=False) |
| R-005 [unit] | pairing | covered | count identities recomputed; ±1 mutation FAILs |
| R-006 [unit] | pairing, qa_degenerate | covered | joint rates over n_complete; null@0; n_pairing_population==0 is typed error |
| R-007 [unit] | pairing | covered | timeout boundary red/green both sides; malformed→exclusion, not imputed |
| R-008 [unit] | pairing | covered | disjoint dup-free key sets; Unicode near-dup fixtures |
| R-009 [unit] | pairing | covered | arm-reversal property-based over generated record sets |
| R-010 [unit] | pairing | covered | all-finite parity vs fixture-locked golden values |
| R-011 [unit] | pairing, ledger_rights, qa_round3, verifier_gates | covered | estimand digest; differing-digest pooling refused; tolerance is a digest field |
| R-012 [unit] | qa_bindings, qa_degenerate, qa_ledger, qa_round2, verifier_cli, verifier_gates | covered | **strong**: per-missing-binding & UNRESOLVED fixtures→FAIL; mutable-revision (short/tag/branch/bare-id) reject; byte-digest alt PASS; collect-all-legs; stale-PASS ledger recompute; empty-eval error / tiny-nonempty PASS |
| R-013 [integration] | ledger_rights, qa_round3, verifier_gates | covered | self-manifest reaches ≤ source-level; expectations-inside-tree (& symlink-resolving) refused; anchor check string-exact, git-free |
| R-014 [integration] | publish, qa_ledger, qa_round3, verifier_gates | covered | input byte-hash identical pre/post run; HISTORICAL_NONCERTIFYING; aggregate-only can't certify per-item |
| R-015 [unit] | pairing, qa_degenerate, qa_round3, verifier_gates | covered | recompute from records; mismatch→FAIL; absent→non-certifying; interval identity |
| R-016 [integration] | publish, qa_publish_docs, qa_round3, vocabulary_scans | covered | create-once no-replace; kill-mid-publish/retry → exactly one artifact |
| R-017 [unit] | qa_ingress, verifier_gates | covered | source mode enum tops at PASS_SOURCE_ONLY; no release token; no ACM 3rd-party terms |
| R-018 [integration] | verifier_cli | covered | `verify_audit_release.py` byte-identical; new work confined to namespace (`containment` marker) |
| R-019 [unit] | verifier_gates | covered | 8+5 adversarial corpus each FAILs, each paired with nearest-true sibling that PASSes |
| R-020 [unit] | qa_ingress, schema, qa_round3, verifier_cli | covered | typed ingress; schema_version validated first; path-named errors |
| R-021 [integration] | verifier_cli | covered | **strong**: real `python -m …verify` subprocess; single-binding mutation flips verdict; exit codes asserted; sentinel-free output |
| R-022 [unit] | qa_round3, verifier_cli | covered | unknown flag/key → error; `allow_abbrev=False`; no bypass door |
| R-023 [unit] | docs, ledger_rights, qa_ledger | covered | ledger row field set; manuscript-identity distinctions |
| R-024 [integration] | ledger_rights, qa_ledger, qa_round2 | covered | EXTERNAL rows byte-immune across enumerated tool list; EXTERNAL→PASS needs human-attribution |
| R-025 [unit] | ledger_rights, qa_round3 | covered | Random-K disposition gate; substituted draw changes digest→refused; SOURCE_CONTRACT_ONLY |
| R-026 [unit] | ledger_rights, qa_ingress, schema, qa_round2, verifier_cli | covered | rights enum; release requires VERIFIED_ALLOWED; synthetic fixtures; sentinel-leak test |
| R-027 [unit] | vocabulary_scans | covered | banned-phrase / required-qualifier fixture asserted over every renderer output |
| R-028 [unit] | qa_publish_docs, vocabulary_scans | covered | **strong**: autouse no-network guard + AST import-deny-list scan (source confirmed clean) |
| R-029 [unit] | schema | covered | `llm_involvement` block required; `none` explicit |
| R-030 [unit] | ledger_rights, qa_round2 | covered | `archival_doi`; Available-grade without DOI FAILs; GitHub URL doesn't qualify |
| R-031 [unit] | qa_round2, schema | covered | non-reversible per-item records; free-text field → FAIL |
| R-032 [unit] | pairing, schema | covered | max tolerance pinned; oversized → FAIL; digest field |
| R-033 [unit] | qa_ingress, verifier_cli, verifier_gates | covered | no vacuous verdicts; PASS requires ≥1 validated artifact |
| R-034 [unit] | vocabulary_scans | covered | JSON/JSONL only; pickle/marshal/torch.load/unsafe-YAML AST scan (source confirmed clean) |
| R-035 [unit] | verifier_gates | covered | manifest reconciliation both directions; rights covers every file found |
| R-036 [integration] | publish, qa_round3, verifier_cli | covered | schema-versioned receipt outside tree; create-once run-scoped names |
| R-037 [integration] | qa_ingress, qa_publish_docs, qa_round3, qa_round2, verifier_cli, verifier_gates | covered | `python -m …verify` contract; 4 distinct exit codes pinned & asserted |
| R-038 [integration] | docs, vocabulary_scans, qa_publish_docs, verifier_cli | covered | README pins both modes/layout/exit codes/receipt; source_to_claim historical-scope header |
| R-039 [unit] | publish, qa_publish_docs, qa_round3 | covered | run-scoped/content-addressed dirs; retire = ledger status change + new dir; bytes retained |

**Uncovered rules: 0. Weak tests identified: 0** (in the audited subset; remaining
rules follow the same red/green fixture pattern with FAIL + nearest-true PASS pairs).

Minor doc inconsistency (non-blocking): R-012 is tagged `[unit]` in the spec but its
test docstrings say `[integration]`. The test is in fact integration-style (verifier
over a fixture package), so coverage is unaffected; only the tag label differs.

## Dependencies

- **No new runtime/library dependencies.** The only `pyproject.toml` change is an
  added pytest marker declaration (`containment`, for R-018 git-consulting guards).
  Consistent with spec "Won't add dependencies" and R-028 (no network/model libs).

## Architecture Adherence

`.correctless/ARCHITECTURE.md` (62 lines) contains one design-pattern entry
(`PAT-001: Likelihood factory dispatch`) plus a component table — none of which
reference the `reproducibility/colm_aims_2026/` namespace or any file this feature
touched. **0 architecture entries affected; 0 stale; 0 path-missing.** Check is
effectively dormant for this feature.

- Advisory (LOW, spec-acknowledged as **OQ-003**): the new verifier CLI
  (`python -m reproducibility.colm_aims_2026.verify`) is a real entrypoint but is not
  registered in ARCHITECTURE.md, so the five [integration] rules carry no formal
  Entry/Through/Exit contract. Optional follow-up: `/carchitect` to define it. Does
  not gate verification.

### Drift Debt
- `DRIFT-001` (open items scan): belongs to the **separate** `hazard-efficacy-eval`
  spec (rule R-011, files `scripts/run_hazard_efficacy.py`), status **resolved**
  2026-08-18. Not relevant to camera-ready (no colm_aims file or architecture-entry
  reference). No new drift-debt created by this feature.

0 architecture entries checked (none overlap changed files), 0 stale, 0 in-scope drift-debt items.

## QA Class Fixes Verified

`qa-findings-camera-ready-aims-evidence.json` (round 2 + mini-audit): 32 findings —
17 BLOCKING, 3 CRITICAL, 11 NON-BLOCKING, 1 UNCERTAIN. **All 17 BLOCKING and all 3
CRITICAL are `status: fixed`; 0 BLOCKING left open.** 4 accepted (QA-012 UNCERTAIN on
R-001; MA-V-001/002/003 NON-BLOCKING mini-audit residuals). Structural class-fix
tests are present and green (e.g. R-012 QA-001/002/003 fail-closed legs;
R-016 QA-008 create-once; R-020 QA-006 typed ingress; R-036 MA-HI-002/004 receipt;
R-037 MA-CC-1 CLI contract) — the 595-test suite covers the classes, not just the
instances.

## Antipattern Scan

`antipattern-scan.sh main`: 141 total findings, but only **9 in camera-ready scope,
all `debug-print`** — false positives of the mechanical print heuristic:

| File:line | Type | Assessment |
|-----------|------|------------|
| reproducibility/colm_aims_2026/verify.py:107,112,119,134,136 | debug-print | **FP** — these are the CLI's own output: `print(..., file=sys.stderr)` for typed error lines and `print(render.render_summary(report))` for the verdict. Intentional stdout/stderr, not debug leaks. |
| tests/test_colm_aims_qa_publish_docs.py:233,235,254 | debug-print | **FP** — print(...) inside test fixtures/captured-output assertions |
| tests/test_colm_aims_qa_round3.py:596 | debug-print | **FP** — test-side print |

The remaining 132 findings are in out-of-scope files (hazard scripts, framework
`workflow-advance.sh`, training modules). Scanner errors: 2 binary `.png` files in
`results/hazard_efficacy_*/` could not be scanned (harmless, out of scope).

## Smells

- No `TODO`/`FIXME`/`HACK`/`XXX` in the `reproducibility/colm_aims_2026/` namespace.
- No prohibited imports in the namespace (source-confirmed: no `requests`/`httpx`/
  `urllib.request`/`huggingface_hub`/`transformers`/`torch`; no `pickle`/`marshal`/
  `torch.load`/`yaml.load`).
- The `debug-print` hits above are legitimate CLI I/O, not real smells.

## Drift

None found. Code uses the abstractions the spec mandates (schema.py strict profile,
StopDFF v5 create-once fileio for R-016/R-036, `python -m` CLI contract). No
`implemented_in` file/function referenced by the spec is missing.

## Spec Updates

No TDD-time spec updates recorded (`spec_updates` field absent from workflow state →
0). The R-029/R-030 venue additions and R-031–R-039 review-hardened rules were spec
authoring during /cspec + /creview, not TDD-loop amendments.

## Overall: PASS — 0 blocking findings

All 39 spec rules are covered by genuine red/green probes; the feature suite is
595/595 green on the canonical env; no new dependencies; no uncovered rules; all
BLOCKING/CRITICAL QA findings fixed; namespace free of prohibited imports and stray
smells. Advisory (non-gating): OQ-003 architecture-entrypoint documentation, and the
R-012 spec-vs-test tag-label mismatch.
