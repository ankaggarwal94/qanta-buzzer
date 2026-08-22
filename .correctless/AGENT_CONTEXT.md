# Agent Context — qanta-buzzer

> Last updated: 2026-08-22
> Companion to `AGENTS.md` (canonical full contract) and `.planning/` (GSD state).

## What This Project Does

RL agents decide *when to buzz* on incrementally revealed quiz-bowl tossup
questions, scoring candidate answers against the revealed clue. The repo also
hosts the **StopDFF v5** audit pipeline (CS321M) — an identity-bound,
fail-closed, create-once reproduction/audit system — and the **COLM AIMS
2026 evidence verifier (v2)**, a two-mode fail-closed verifier for the
constructed-reference evidence package. Python research sub-project; not a
deployed service.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Likelihood factory | `models/likelihoods.py` | `build_likelihood_from_config()` → `tfidf`/`sbert`/`openai`/`t5*`/`dspy` |
| DSPy likelihood | `models/dspy_likelihood.py` | Wraps a real `scorer(clue, options)->list[float]`; importable without the `dspy` extra |
| Buzzer env | `qb_env/tossup_env.py` | belief = `softmax(scores, beta)` |
| StopDFF v5 | `scripts/stopdff_v5/`, `scripts/run_stopdff_v5_local.py` | fail-closed, create-once audit pipeline |
| COLM AIMS v2 verifier | `reproducibility/colm_aims_2026/` | Two-mode fail-closed evidence verifier. CLI: `python -m reproducibility.colm_aims_2026.verify --mode {source,release} (--tree PATH \| --runs-root PATH) [--expectations PATH] --receipts-dir PATH`. Spec: `.correctless/specs/camera-ready-aims-evidence-2.md`; usage: `reproducibility/colm_aims_2026/README.md` |

## Design Patterns

- **Likelihood factory dispatch (PAT-001)**: scoring backends are built only via
  `build_likelihood_from_config`; a selectable model must return discriminating
  scores or fail loud — never silently uniform.
- **Two-mode fail-closed verifier (PAT-002)**: the COLM AIMS v2 verifier
  snapshots the tree once, runs per-leg collect-don't-halt checks through
  guarded leg builders (a defect becomes that leg's FAIL, never an unreceipted
  crash), and caps source mode at `PASS_SOURCE_ONLY`. See
  `.correctless/ARCHITECTURE.md` PAT-002 (+ TB-001/TB-002, ABS-001/ABS-002).

## Common Pitfalls

- **DSPy likelihood is NOT production-wired.** `model: dspy` fails loud unless
  `dspy.allow_uniform_placeholder: true` (an explicit, warned uniform stub).
  For real scoring, inject `DSPyLikelihood(scorer=...)` directly — the factory
  cannot build one from config (no persisted program).
- **Test env**: the canonical interpreter is the primary clone `.venv`. Homebrew
  `python3.11` has a torch/transformers version mismatch that breaks the
  torch/sbert test files — use `.venv/bin/python -m pytest`.
- **StopDFF v5** artifacts are create-once; never publish via a bare
  `os.replace`/`Path.replace` (use `scripts/stopdff_v5/fileio.py` no-replace
  primitives).
- **The COLM AIMS v2 verifier certifies constructed references ONLY — never
  observed decision preservation.** The pinned semantic block is verbatim:
  `trajectory_source: constructed_reference`, `observed_open_ended: false`,
  `supports: reference_sensitivity_diagnostic`, `does_not_support:
  actual_decision_preservation_or_format_effect`. No output, doc, or PR text
  may upgrade that scope (spec R-027/R-001).
- **`PASS_RELEASE` requires a live git checkout.** The anchor-commit
  object-existence leg (R-066) FAILS when `git` is unavailable or the check
  cannot run — release verification cannot pass by making git disappear. This
  is by design; source mode merely records the capability gap.
- **v1 documents are refused by every strict v2 surface.** Historical/legacy
  artifacts enter only via `legacy.load_legacy_v1_document` (the ONE named
  loader), whose output is always noncertifying (`certifying: False`); strict
  loaders raise a typed version error on `schema_version: 1` (R-059/R-060).
- **NumPy exactly 2.4.6 is required** for the v2 suite and the D7(b)
  bit-exact inference goldens (D5/R-051; enforced by a runtime release leg and
  a version test). Note: `requirements.txt` still pins `numpy==2.4.4` — the
  canonical `.venv` already runs 2.4.6; don't "fix" an env to match the stale
  pin.

## Quick Reference

| Need to... | Do this |
|------------|---------|
| Run tests | `.venv/bin/python -m pytest` (canonical env; not homebrew py3.11 for torch tests) |
| Verify an AIMS evidence package | `python -m reproducibility.colm_aims_2026.verify --mode source --tree PATH --receipts-dir PATH` (see namespace README for release mode) |
| Lint | `ruff check .` (requires `pip install ruff`; not in the default `.venv`) |
| Offline DSPy compile | `python scripts/optimize_dspy.py --config configs/default.yaml` (prints fingerprint; does not persist a program yet) |
| Find a spec | `.correctless/specs/{feature}.md` (created on first `/cspec`) |
| Check architecture | `.correctless/ARCHITECTURE.md` (essentials) + `AGENTS.md` (canonical) |
| Planning / phase state | `.planning/` (GSD is canonical) |
| See known bugs | `.correctless/antipatterns.md` |
