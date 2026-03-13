---
phase: quick-1-repo-contract
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - AGENTS.md
  - CLAUDE.md
  - .agentic.yml
  - scripts/ci.sh
  - scripts/manual-smoke.sh
autonomous: true
requirements: [SCAFFOLD-01, SCAFFOLD-02, SCAFFOLD-03, SCAFFOLD-04]

must_haves:
  truths:
    - "AGENTS.md is the single canonical repo contract with setup, architecture, testing, and smoke commands"
    - "CLAUDE.md is a thin shim that points to AGENTS.md and adds only Claude-specific conventions"
    - ".agentic.yml truthfully describes this repo's packages, test commands, and smoke pipeline"
    - "scripts/ci.sh runs pytest and exits nonzero on failure"
    - "scripts/manual-smoke.sh documents the four-stage smoke pipeline with human-readable output"
  artifacts:
    - path: "AGENTS.md"
      provides: "Canonical repo contract for all coding agents"
      contains: "qanta-buzzer"
    - path: "CLAUDE.md"
      provides: "Thin shim pointing to AGENTS.md"
      contains: "AGENTS.md"
    - path: ".agentic.yml"
      provides: "Machine-readable repo metadata"
      contains: "qanta-buzzer"
    - path: "scripts/ci.sh"
      provides: "Automated CI entry point"
      contains: "pytest"
    - path: "scripts/manual-smoke.sh"
      provides: "Human-runnable smoke pipeline"
      contains: "build_mc_dataset"
  key_links:
    - from: "CLAUDE.md"
      to: "AGENTS.md"
      via: "markdown reference"
      pattern: "AGENTS\\.md"
    - from: "scripts/ci.sh"
      to: "tests/"
      via: "pytest invocation"
      pattern: "pytest"
    - from: "scripts/manual-smoke.sh"
      to: "scripts/build_mc_dataset.py"
      via: "python invocation"
      pattern: "python scripts/build_mc_dataset.py --smoke"
---

<objective>
Add the minimal repo-contract scaffolding: AGENTS.md as the canonical repo contract, reduce CLAUDE.md to a thin shim, add .agentic.yml, and create ci.sh and manual-smoke.sh scripts that invoke commands already existing in this repo.

Purpose: Give every coding agent (Claude, Copilot, Cursor, etc.) a single committed contract to read, and provide executable CI and smoke scripts that wrap the repo's actual test and pipeline commands.

Output: 5 files created or rewritten. No new dependencies, no lockfile changes, no infrastructure.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@./CLAUDE.md
@./README.md
@./pyproject.toml
@./configs/default.yaml
@./configs/smoke.yaml
@./tests/conftest.py
@./.github/workflows/python-app.yml
@./.github/copilot-instructions.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create AGENTS.md and reduce CLAUDE.md to a thin shim</name>
  <files>AGENTS.md, CLAUDE.md</files>
  <action>
Create AGENTS.md as the canonical repo contract. Move the substantive content from the current CLAUDE.md into AGENTS.md, and expand it to be the single source of truth for any coding agent. The content must be truthful and derived from what actually exists in the repo (pyproject.toml, README.md, configs/, tests/, scripts/).

AGENTS.md structure (all sections required):

1. **Project Overview** -- one paragraph: Stanford CS234 final project, unified quiz bowl RL buzzer, two tracks (belief-feature pipeline, T5 policy pipeline), qanta-buzzer is canonical repo.

2. **Setup** -- exact commands from README.md:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -U pip && pip install -e .`
   - Optional OpenAI: `pip install -e '.[openai]'`
   - Requires Python >= 3.11

3. **Architecture** -- the 7 packages verbatim from current CLAUDE.md Architecture section:
   - qb_data/, qb_env/, models/, agents/, evaluation/, scripts/, training/
   - Plus configs/ for YAML configuration

4. **Testing** -- truthful pytest coverage:
   - `pytest` for full suite (13 test files, ~220 tests)
   - `pytest tests/test_qb_rl_bridge.py tests/test_factories.py tests/test_ppo_buzzer.py` for focused bridge/runtime checks
   - `scripts/ci.sh` as the CI entry point

5. **Smoke Pipeline** -- the four canonical commands:
   - `python scripts/build_mc_dataset.py --smoke`
   - `python scripts/run_baselines.py --smoke`
   - `python scripts/train_ppo.py --smoke`
   - `python scripts/evaluate_all.py --smoke`
   - Note that `--smoke` selects configs/smoke.yaml and writes to artifacts/smoke/
   - `scripts/manual-smoke.sh` as the runnable wrapper

6. **T5 Policy Pipeline** -- the two commands:
   - `python scripts/train_t5_policy.py --config configs/t5_policy.yaml`
   - `python scripts/compare_policies.py --config configs/t5_policy.yaml`

7. **Configuration** -- three YAML configs (default.yaml, smoke.yaml, t5_policy.yaml) with one-line descriptions. Note qb-rl config alias support.

8. **Compatibility Bridge** -- old qb-rl import paths that still resolve (qb_env.data_loader, qb_env.mc_builder, qb_env.text_utils, models.answer_profiles, agents.softmax_profile_buzzer). OpenAI opt-in only.

9. **Conventions** -- NumPy-style docstrings, RL notation (V, R, T, gamma, s/a), vectorized operations preferred, explicit seeds for reproducibility.

Then rewrite CLAUDE.md to a thin shim (~15-25 lines total):
- First line: "# CLAUDE.md"
- Second line: reference to AGENTS.md as the canonical repo contract ("See AGENTS.md for the full repo contract: setup, architecture, testing, smoke pipeline, and configuration.")
- Then a short "Claude-specific notes" section with only these items:
  - `.planning/` is durable project memory; respect STATE.md decisions
  - Prefer narrow verification over broad cargo-cult test runs
  - Do not add dependencies unless required
  - Seeds: use 1, 2, 3 for multi-seed runs
  - NumPy/PyTorch vectorized operations over loops in ML code

Do NOT include any content in CLAUDE.md that duplicates AGENTS.md (no setup, no architecture, no test commands, no smoke commands).
  </action>
  <verify>
    <automated>grep -q "AGENTS.md" CLAUDE.md && grep -q "qanta-buzzer" AGENTS.md && test $(wc -l < CLAUDE.md) -lt 40 && echo "PASS" || echo "FAIL"</automated>
  </verify>
  <done>AGENTS.md exists as the full repo contract with all 9 sections. CLAUDE.md is under 40 lines and references AGENTS.md. No substantive duplication between the two files.</done>
</task>

<task type="auto">
  <name>Task 2: Create .agentic.yml, scripts/ci.sh, and scripts/manual-smoke.sh</name>
  <files>.agentic.yml, scripts/ci.sh, scripts/manual-smoke.sh</files>
  <action>
Create .agentic.yml at the repo root. This is a machine-readable metadata file for agentic tools. All values must be truthful (derived from pyproject.toml, README.md, and actual repo contents):

```yaml
# .agentic.yml -- machine-readable repo contract for coding agents
project: qanta-buzzer
description: "Unified quiz bowl RL buzzer for Stanford CS234"
python: ">=3.11"
install: "pip install -e ."
install_openai: "pip install -e '.[openai]'"

packages:
  - agents
  - evaluation
  - models
  - qb_data
  - qb_env
  - training
  - scripts

configs:
  default: configs/default.yaml
  smoke: configs/smoke.yaml
  t5_policy: configs/t5_policy.yaml

testing:
  framework: pytest
  command: pytest
  test_dir: tests
  test_files: 13
  ci_script: scripts/ci.sh

smoke:
  script: scripts/manual-smoke.sh
  steps:
    - python scripts/build_mc_dataset.py --smoke
    - python scripts/run_baselines.py --smoke
    - python scripts/train_ppo.py --smoke
    - python scripts/evaluate_all.py --smoke

repo_contract: AGENTS.md
agent_shims:
  claude: CLAUDE.md
  copilot: .github/copilot-instructions.md
```

Create scripts/ci.sh:
```bash
#!/usr/bin/env bash
# CI entry point -- runs the full pytest suite.
# Exit nonzero on any failure so CI gates catch regressions.
set -euo pipefail
pytest "$@"
```

This is intentionally minimal. It runs pytest, passes through any extra arguments (e.g., `-x`, `-k`, specific test files), and exits nonzero on failure. Do not add linting, type checking, or any other steps -- smallest truthful diff.

Create scripts/manual-smoke.sh:
```bash
#!/usr/bin/env bash
# Manual smoke pipeline -- runs the four-stage belief-feature smoke workflow.
# Intended for human verification, not CI (stages are heavyweight ML runs).
#
# Prereqs: pip install -e .  (see AGENTS.md for full setup)
# Outputs: artifacts/smoke/
set -euo pipefail

echo "=== Stage 1/4: Build MC dataset ==="
python scripts/build_mc_dataset.py --smoke

echo "=== Stage 2/4: Run baselines ==="
python scripts/run_baselines.py --smoke

echo "=== Stage 3/4: Train PPO ==="
python scripts/train_ppo.py --smoke

echo "=== Stage 4/4: Evaluate all ==="
python scripts/evaluate_all.py --smoke

echo "=== Smoke pipeline complete. Check artifacts/smoke/ ==="
```

After creating both shell scripts, make them executable with `chmod +x`.
  </action>
  <verify>
    <automated>test -f .agentic.yml && test -x scripts/ci.sh && test -x scripts/manual-smoke.sh && grep -q "pytest" scripts/ci.sh && grep -q "build_mc_dataset" scripts/manual-smoke.sh && echo "PASS" || echo "FAIL"</automated>
  </verify>
  <done>.agentic.yml exists with truthful metadata. scripts/ci.sh is executable and runs pytest. scripts/manual-smoke.sh is executable and runs the four smoke stages. No new dependencies added.</done>
</task>

</tasks>

<verification>
After both tasks complete:
1. `cat AGENTS.md` -- confirm all 9 sections present with truthful content
2. `wc -l CLAUDE.md` -- confirm thin shim (under 40 lines)
3. `grep "AGENTS.md" CLAUDE.md` -- confirm reference link
4. `cat .agentic.yml` -- confirm truthful metadata
5. `bash scripts/ci.sh --co` -- confirm pytest invocation works (--co = collect-only, no execution)
6. `head -5 scripts/manual-smoke.sh` -- confirm shebang and set -euo pipefail
</verification>

<success_criteria>
- AGENTS.md is the single canonical repo contract with setup, architecture, testing, smoke, T5, config, bridge, and conventions sections
- CLAUDE.md is a thin shim under 40 lines referencing AGENTS.md, with only Claude-specific notes
- .agentic.yml truthfully describes packages, configs, testing, and smoke pipeline
- scripts/ci.sh runs pytest and exits nonzero on failure
- scripts/manual-smoke.sh runs the four canonical smoke stages
- No new dependencies, lockfile changes, or unrelated refactors
- Total diff touches exactly 5 files (2 new, 2 rewritten, 1 new)
</success_criteria>

<output>
After completion, create `.planning/quick/1-repo-contract-scaffolding-agents-md-thin/1-SUMMARY.md`
</output>
