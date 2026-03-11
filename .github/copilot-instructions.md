# Copilot Instructions for `qanta-buzzer`

Use these instructions as the repo-wide baseline for Copilot work in this repository. Keep them concise, and prefer branch-local source-of-truth docs when they exist.

## Source of truth

- If the checked-out branch contains `CLAUDE.md`, follow it.
- If the checked-out branch contains `.planning/`, treat `.planning/` as the durable project state and keep important workflow decisions aligned with it.
- Do not invent a second planning system in parallel with existing repo docs.

## Code paths

- This repository has an older root-level prototype path centered on files such as `main.py`, `environment.py`, `dataset.py`, `model.py`, `train_supervised.py`, `train_ppo.py`, and `metrics.py`.
- Some branches also contain a newer modular pipeline with packages such as `qb_data/`, `qb_env/`, `models/`, `agents/`, `evaluation/`, `scripts/`, and `training/`.
- Match the checked-out branch. Do not assume the modular pipeline exists on every branch, and do not force work back into the root-level prototype if the modular packages are already present.

## Change discipline

- Keep changes minimal and scoped to the request.
- Prefer editing existing modules over introducing new abstractions unless the request clearly needs them.
- Do not add dependencies unless they are required.
- Do not commit generated Python cache files, virtual environments, model artifacts, or local notebooks unless the task explicitly asks for tracked generated outputs.

## Validation

- Prefer the narrowest relevant verification for the files you changed.
- On older/root-prototype branches, the lightweight validation scripts are:
  - `python test_imports.py`
  - `python test_csv_loader.py`
- On branches with `tests/` and `pyproject.toml`, prefer targeted `pytest` first and run the full suite when the change is broad or touches shared infrastructure.
- If the branch exposes smoke workflows such as `python scripts/build_mc_dataset.py --smoke`, prefer those over heavyweight full training runs during routine iteration.

## Heavyweight ML workflows

- This repo uses heavyweight ML dependencies including PyTorch, Transformers, sentence-transformers, and Stable-Baselines3.
- Avoid expensive model downloads or long training runs unless the task actually requires them.
- If you are editing docs, config handling, tests, or small control-flow logic, do not trigger full T5 or PPO training just to prove the change.

## Practical repo guidance

- Respect the existing file organization and naming conventions on the active branch.
- When documentation and code disagree, trust the executable code first, then update docs to match.
- If a branch includes compatibility shims or bridge code, preserve backward-compatible imports and config aliases unless the task explicitly asks to remove them.
