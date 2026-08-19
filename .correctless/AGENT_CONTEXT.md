# Agent Context — qanta-buzzer

> Last updated: 2026-08-14
> Companion to `AGENTS.md` (canonical full contract) and `.planning/` (GSD state).

## What This Project Does

RL agents decide *when to buzz* on incrementally revealed quiz-bowl tossup
questions, scoring candidate answers against the revealed clue. The repo also
hosts the **StopDFF v5** audit pipeline (CS321M) — an identity-bound,
fail-closed, create-once reproduction/audit system. Python research
sub-project; not a deployed service.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Likelihood factory | `models/likelihoods.py` | `build_likelihood_from_config()` → `tfidf`/`sbert`/`openai`/`t5*`/`dspy` |
| DSPy likelihood | `models/dspy_likelihood.py` | Wraps a real `scorer(clue, options)->list[float]`; importable without the `dspy` extra |
| Buzzer env | `qb_env/tossup_env.py` | belief = `softmax(scores, beta)` |
| StopDFF v5 | `scripts/stopdff_v5/`, `scripts/run_stopdff_v5_local.py` | fail-closed, create-once audit pipeline |
| Hazard efficacy harness | `scripts/run_hazard_efficacy.py` | Controlled A/B/C(+variants) comparison for the hazard bridge: shared supervised branch point, identity-validated resume, scale-gated paired-bootstrap significance, `hazard_efficacy_report.json` |

## Design Patterns

- **Likelihood factory dispatch (PAT-001)**: scoring backends are built only via
  `build_likelihood_from_config`; a selectable model must return discriminating
  scores or fail loud — never silently uniform.

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
- **Non-smoke split resolution prefers `artifacts/main/` and silently falls
  back to `artifacts/smoke/`** (then `data/processed/`). Populate
  `artifacts/main/` before any non-smoke T5 training/eval or children train on
  the 44-question smoke split while claiming full scale.
- **The hazard bridge's `--beta-terminal` sits on a threshold against the
  answer head's NLL scale** (~ln K at chance): β below it teaches never-buzz
  (S_q collapse PPO does not recover from), β above teaches immediate-buzz.
  Neither corner is timing. Efficacy verdict: `docs/hazard-efficacy-report.md`.

## Quick Reference

| Need to... | Do this |
|------------|---------|
| Run tests | `.venv/bin/python -m pytest` (canonical env; not homebrew py3.11 for torch tests) |
| Lint | `ruff check .` (requires `pip install ruff`; not in the default `.venv`) |
| Offline DSPy compile | `python scripts/optimize_dspy.py --config configs/default.yaml` (prints fingerprint; does not persist a program yet) |
| Find a spec | `.correctless/specs/{feature}.md` (created on first `/cspec`) |
| Check architecture | `.correctless/ARCHITECTURE.md` (essentials) + `AGENTS.md` (canonical) |
| Planning / phase state | `.planning/` (GSD is canonical) |
| See known bugs | `.correctless/antipatterns.md` |
