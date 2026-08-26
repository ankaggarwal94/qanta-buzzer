# Architecture — qanta-buzzer

> Canonical, full repo contract lives in **`AGENTS.md`** (setup, testing, smoke
> pipeline, StopDFF v5 contracts) and planning state lives in **`.planning/`**
> (GSD is canonical for phases/requirements/state). This file captures the
> Correctless-structured essentials that specs/reviews/verification reference —
> it does not replace AGENTS.md.

Three systems share this repo:
1. **Quiz-bowl RL buzzer** — RL agents decide *when to buzz* on incrementally
   revealed tossup questions, scoring answer options against the revealed clue.
2. **StopDFF v5 audit pipeline** (CS321M) — an identity-bound, fail-closed,
   create-once, content-addressed audit/reproduction pipeline.
3. **COLM AIMS v2 evidence verifier** — two-mode fail-closed verifier for the
   camera-ready evidence package (spec:
   `.correctless/specs/camera-ready-aims-evidence-2.md`).

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Likelihood models | `models/likelihoods.py`, `models/dspy_likelihood.py` | Pluggable answer-scoring backends behind `LikelihoodModel.score()`; built via `build_likelihood_from_config()` |
| Buzzer environment | `qb_env/tossup_env.py` | `TossupMCEnv`; converts likelihood scores → belief via `softmax(scores, beta)`; `make_env_from_config()` |
| Policies / value model | `models/t5_policy.py`, `models/stopdff_value_model.py` | T5 buzz policy; StopDFF value model |
| Answer profiles / distractors | `models/answer_profiles.py`, `qb_data/mc_builder.py` | Distractor ranking, MC question construction |
| Data pipeline | `qb_data/`, `scripts/build_mc_dataset.py` | Dataset splits, MC dataset build |
| StopDFF v5 pipeline | `scripts/stopdff_v5/`, `scripts/run_stopdff_v5_local.py`, `scripts/modal_stopdff_v5_runner.py` | Fail-closed, create-once audit pipeline (contracts in `ACCEPTANCE_CONTRACT.md`, `SCIENTIFIC_CONTRACT.md`, `IDENTITY_AND_ARTIFACT_CONTRACT.md`) |
| COLM AIMS v2 verifier | `reproducibility/colm_aims_2026/` | Camera-ready evidence verifier. CLI: `python -m reproducibility.colm_aims_2026.verify --mode {source,release} (--tree PATH \| --runs-root PATH) [--expectations PATH] --receipts-dir PATH`. Spec: `.correctless/specs/camera-ready-aims-evidence-2.md` |
| Entry points | `scripts/` (`train_t5_policy.py`, `run_baselines.py`, `train_ppo.py`, `evaluate_all.py`, `optimize_dspy.py`) | CLI training/eval; standalone entrypoints force repo root to `sys.path[0]` |
| Tests | `tests/` (~119 files) | `pytest`; canonical env is the primary clone `.venv` |

## Design Patterns

### PAT-001: Likelihood factory dispatch
- Every answer-scoring backend subclasses `LikelihoodModel` and is constructed
  **only** via `build_likelihood_from_config(config)`, keyed on
  `config["likelihood"]["model"]` (`tfidf` | `sbert` | `openai` | `t5*` | `dspy`).
- Enforced in `models/likelihoods.py` (`build_likelihood_from_config`, ~L676).
  Downstream, belief = `softmax(scores, beta)` in `qb_env/tossup_env.py`.
- **Invariant — never silently uniform:** a selectable model must return
  *discriminating* scores or **fail loud**. A backend that silently returns
  `[1/K]*K` produces a flat belief and no buzz signal, silently invalidating any
  experiment that selects it. (This is why the `dspy` branch now raises rather
  than substituting a uniform stub — see Known Limitations.)

### PAT-002: Two-mode fail-closed verifier (COLM AIMS v2)
- `run_verifier()` snapshots the candidate tree ONCE (bounded, symlink-refusing
  reads), then verifies only that snapshot — no mid-run re-reads. Enforced at
  `reproducibility/colm_aims_2026/verifier.py:299` (`_read_tree_snapshot`),
  `verifier.py:1617` (`run_verifier`).
- Per-leg collect-don't-halt: each check appends a PASS/FAIL leg
  (`_record_leg`, `verifier.py:205`); expected defects surface as the typed
  `ColmAimsError` families (`schema.py:24-52`), caught per leg via `_LEG_CATCH`
  (`verifier.py:229`).
- Guarded leg builders: any non-`ColmAimsError` raised inside a builder becomes
  THAT leg's FAIL — the run still reaches a verdict and a receipt, never an
  unreceipted crash (`_guarded`, `verifier.py:250`).
- **Invariant — source ceiling:** source mode emits at most `PASS_SOURCE_ONLY`;
  `PASS_RELEASE` exists only in release mode with anchored expectations (closed
  verdict vocabularies, `verifier.py:54-59`).
- Violated when: a leg builder halts the run, a defect escapes without a
  receipt, or any source-mode path emits a release token.
- Test: `tests/test_colm_aims_v2_verifier_source_cli.py`,
  `tests/test_colm_aims_v2_verifier_release.py`.

## Trust Boundaries

### TB-001: Untrusted artifact bytes → hardened ingress (COLM AIMS v2)
- Crosses: candidate evidence-package bytes on disk (untrusted) → typed
  in-memory records the legs consume.
- Every untrusted read goes through `schema.read_regular_file_bytes`
  (`reproducibility/colm_aims_2026/schema.py:665`): `O_NOFOLLOW` open, fstat
  `S_ISREG` gate (no symlinks/FIFOs/devices), size-capped.
- JSON parses carry three protective hooks (`parse_json_bytes_strict`,
  `schema.py:588`): non-finite constants rejected (`schema.py:516`), int
  tokens length-capped at 100 digits (`MAX_JSON_INT_TOKEN_DIGITS`,
  `schema.py:94`, `schema.py:534`), `RecursionError` converted to a typed
  ingress error (`schema.py:575`).
- Trusted nested config blocks use CLOSED key sets — unknown keys are typed
  errors, so `dict.get` defaults cannot engage on typos (`_validate_closed_map`,
  `schema.py:1059`; anchor keys `verifier.py:109-114`).
- ONE bool-safe version checker on every versioned surface:
  `check_schema_version` (`schema.py:479`) admits only `type(version) is int`
  within the supported range, else raises `SchemaVersionError` (`schema.py:41`).
- **Invariant:** no artifact byte reaches a semantic check except through this
  ingress; every rejection is a typed `ColmAimsError`.
- Violated when: a reader opens artifact paths directly, a `json.loads` call
  omits the three hooks, or a versioned surface skips `check_schema_version`.
- Test: `tests/test_colm_aims_v2_versioning_ingress.py` (incl. AST guards
  pinning the parse hooks at every call site).

### TB-002: Release certification anchors outside the artifact (COLM AIMS v2)
- Crosses: verified artifact tree (untrusted) → `PASS_RELEASE` verdict
  (trusted release claim).
- Release requires an independently anchored expectations file that must
  resolve — fully, symlink-free — OUTSIDE the verified tree
  (`reproducibility/colm_aims_2026/verifier.py:1645-1659`; typed load
  `_load_expectations`, `verifier.py:524`).
- The anchor block is closed with every key required (`_ANCHOR_KEYS`,
  `verifier.py:109-114`); it pins the semantic grid fields
  (`ANCHORED_GRID_FIELDS`, `verifier.py:121`) and inference fields
  (`ANCHORED_INFERENCE_FIELDS`, `verifier.py:130`) compared field-by-field,
  the ledger digest (`anchor["ledger_sha256"]`, `verifier.py:2115`), and
  cross-document source-commit equality + git-object existence bound to THIS
  repo under a `GIT_*` env denylist (`verifier.py:82-83`, `:2155-2193`,
  `:29-42`).
- **Invariant:** an artifact plus its own manifest can never certify itself —
  every release binding derives from the out-of-tree anchor.
- Violated when: an expectations path inside the tree is accepted, or a
  release leg reads its expected value from the artifact under test.
- Test: `tests/test_colm_aims_v2_verifier_release.py`,
  `tests/test_colm_aims_v2_ledger_rights.py`.

## Abstractions

### ABS-001: Create-once publication + ledger-pointer canonical selection
- What: ALL artifact publication routes through the no-replace primitives in
  `scripts/stopdff_v5/fileio.py` — `create_once_bytes` (`fileio.py:45`),
  `publish_dir_create_once` (`fileio.py:97`). v2 consumers: evidence-package
  staging/publish (`reproducibility/colm_aims_2026/schema.py:1423`,
  `schema.py:1470`) and receipts (`receipt.py:64`).
- Receipts are create-once: every verifier run — refusal paths included —
  emits exactly one new receipt (`receipt.emit_receipt`, `receipt.py:35`).
- Canonical run selection is ledger-pointer-ONLY with resolved containment
  under the runs root — never newest-wins; empty, symlink, or escaping run
  dirs FAIL the release run itself (`resolve_canonical_package`,
  `verifier.py:470`; `run_release_over_runs_root`, `verifier.py:1900`).
- **Invariant:** conforming writers publish artifacts with create-once,
  no-replace semantics; canonical selection has exactly one authority (the
  ledger pointer). This is a protocol invariant under the Phase-4 R-081
  cooperative process/host trust boundary, not filesystem ACL immutability or
  containment against a hostile process sharing the same OS identity.
- Violated when: a publish site uses bare `os.replace`/`open("w")` on a final
  path, or selection falls back to mtime/newest.
- Test: `tests/test_colm_aims_v2_receipts_closure.py`,
  `tests/test_colm_aims_v2_ledger_rights.py`.

### ABS-002: D7(b) frozen inference procedure
- What: the ONE pinned bootstrap/Holm procedure that regenerates all ten
  cells' intervals + Holm at evidence-package production
  (`reproducibility/colm_aims_2026/pairing.py`).
- Seed derived from the pinned pairing-population keyset digest under a domain
  prefix (`d7b_seed`, `pairing.py:464`; prefix `pairing.py:461`; recorded
  derivation string `schema.py:201`). ONE shared resample matrix from
  `numpy.random.Generator(PCG64(seed))`, B=1000 (`d7b_resample_matrix`,
  `pairing.py:484`; `BOOTSTRAP_DRAW_COUNT`, `schema.py:197`).
- p-values are `(1 + exceed) / (B + 1)` — the +1 is mandatory (`d7b_p_value`,
  `pairing.py:529`); Holm step-down m=10, ties by ascending UTF-8 byte order
  of cell_id (`d7b_holm`, `pairing.py:540`).
- **Invariant:** bit-exact against the authoring-time goldens — any drift in
  seed, matrix bytes, p-values, or Holm order is a failure, not a tolerance.
- Violated when: the procedure re-derives an input from unpinned state, or a
  second resample matrix appears anywhere.
- Test: `tests/test_colm_aims_v2_inference_d7b.py` (golden keyset/seed/matrix
  digests); golden literals in `tests/_colm_aims_v2_helpers.py` (~L249).

## Conventions

- NumPy-style docstrings; RL notation (`V`, `R`, `T`, `gamma`, `s`/`a`).
- Reproducibility: seeds set explicitly (numpy/torch/random); multi-seed runs use 1, 2, 3.
- StopDFF v5 artifacts are create-once + content-addressed; publish via the
  no-replace primitives in `scripts/stopdff_v5/fileio.py`.
- See `AGENTS.md` / `CLAUDE.md` for the full convention set.

## Known Limitations

- **DSPy likelihood backend is not production-wired.** `scripts/optimize_dspy.py`
  compiles a program but does not persist it, and the factory has no loader or
  `dspy.Predict → callable` adapter. `likelihood.model: dspy` therefore
  **fails loud** (`NotImplementedError`) unless `dspy.allow_uniform_placeholder:
  true`, which opts into an explicit, *warned* uniform stub (plumbing tests
  only). Real use: inject `DSPyLikelihood(scorer=...)` directly.
- **StopDFF v5 crash-window tradeoffs** (create-once/reclaim; adopt-orphan
  determinism-receipt provenance) are accepted, deliberate, fail-safe
  limitations with reopen-triggers — documented in
  `.planning/reviews/PR 30 v4 (ankaggarwal94__qanta-buzzer)/_RECLAIM_SCOPE_DECISION.md`
  and the `_materialize_adapter_stage` docstring.
