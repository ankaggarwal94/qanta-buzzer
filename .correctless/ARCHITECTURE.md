# Architecture — qanta-buzzer

> Canonical, full repo contract lives in **`AGENTS.md`** (setup, testing, smoke
> pipeline, StopDFF v5 contracts) and planning state lives in **`.planning/`**
> (GSD is canonical for phases/requirements/state). This file captures the
> Correctless-structured essentials that specs/reviews/verification reference —
> it does not replace AGENTS.md.

Two systems share this repo:
1. **Quiz-bowl RL buzzer** — RL agents decide *when to buzz* on incrementally
   revealed tossup questions, scoring answer options against the revealed clue.
2. **StopDFF v5 audit pipeline** (CS321M) — an identity-bound, fail-closed,
   create-once, content-addressed audit/reproduction pipeline.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Likelihood models | `models/likelihoods.py`, `models/dspy_likelihood.py` | Pluggable answer-scoring backends behind `LikelihoodModel.score()`; built via `build_likelihood_from_config()` |
| Buzzer environment | `qb_env/tossup_env.py` | `TossupMCEnv`; converts likelihood scores → belief via `softmax(scores, beta)`; `make_env_from_config()` |
| Policies / value model | `models/t5_policy.py`, `models/stopdff_value_model.py` | T5 buzz policy; StopDFF value model |
| Answer profiles / distractors | `models/answer_profiles.py`, `qb_data/mc_builder.py` | Distractor ranking, MC question construction |
| Data pipeline | `qb_data/`, `scripts/build_mc_dataset.py` | Dataset splits, MC dataset build |
| StopDFF v5 pipeline | `scripts/stopdff_v5/`, `scripts/run_stopdff_v5_local.py`, `scripts/modal_stopdff_v5_runner.py` | Fail-closed, create-once audit pipeline (contracts in `ACCEPTANCE_CONTRACT.md`, `SCIENTIFIC_CONTRACT.md`, `IDENTITY_AND_ARTIFACT_CONTRACT.md`) |
| Entry points | `scripts/` (`train_t5_policy.py`, `run_baselines.py`, `train_ppo.py`, `evaluate_all.py`, `optimize_dspy.py`) | CLI training/eval; standalone entrypoints force repo root to `sys.path[0]` |
| Tests | `tests/` (~96 files) | `pytest`; canonical env is the primary clone `.venv` |

## Design Patterns

### PAT-001: Likelihood factory dispatch
- Every answer-scoring backend subclasses `LikelihoodModel` and is constructed
  **only** via `build_likelihood_from_config(config)`, keyed on
  `config["likelihood"]["model"]` (`tfidf` | `sbert` | `openai` | `t5*` | `dspy`).
- Enforced in `models/likelihoods.py` (`build_likelihood_from_config`, ~L725).
  Downstream, belief = `softmax(scores, beta)` in `qb_env/tossup_env.py`.
- **Invariant — never silently uniform:** a selectable model must return
  *discriminating* scores or **fail loud**. A backend that silently returns
  `[1/K]*K` produces a flat belief and no buzz signal, silently invalidating any
  experiment that selects it. (This is why the `dspy` branch now raises rather
  than substituting a uniform stub — see Known Limitations.)

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
