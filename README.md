# Does the Proxy Preserve the Decision? -- Code Package

> **Scope note (R-038):** stopping-shift evidence in this repository concerns
> constructed QA reference trajectories — a constructed-reference sensitivity
> diagnostic. It does not assert observed open-ended decision preservation;
> the authoritative claim ledger lives under `reproducibility/colm_aims_2026/`.


CS321M (AI Measurement Science) final project code repository. This project audits whether multiple-choice reformulations of incremental AI benchmarks (quizbowl tossups) preserve decision-relevant psychometric properties. The audit uses three metrics:

1. **CSLI** (Choice-Set Leakage Index) -- quantifies information leakage from answer choices
2. **Prefix-wise calibration** -- Platt-scaled ECE across early/mid/late question prefixes
3. **Diagnostic StopDFF** (Stopping-Decision Fairness) -- tests whether MC vs open-ended format changes optimal stopping behavior

Built on the `qanta-buzzer` infrastructure developed for CS234 (Reinforcement Learning).

## Environment Setup

**Requirements:** Python 3.11+

```bash
# Clone and enter the repository
git clone <repo-url>
cd qanta-buzzer

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -e .
# OR for exact reproducibility:
pip install -r requirements.txt && pip install -e . --no-deps
```

For paper reproduction, use `requirements.txt`.

For library development, use `pip install -e .[dev]`.

For minimum-supported CI, use `requirements-min.txt`; this is a compatibility
gate, not a paper-reproduction lock.

## Quickstart

Run the four-stage smoke pipeline (builds MC dataset, runs baselines, trains PPO, evaluates):

```bash
bash scripts/manual-smoke.sh
```

**Expected output:** Artifacts written to `artifacts/smoke/` including `mc_dataset.json`, `train_dataset.json`, `val_dataset.json`, `test_dataset.json`, `evaluation_report.json`, and PPO checkpoints.

**Runtime:** ~2-5 minutes on laptop CPU.

## CS321M Audit Scripts

These are the novel scripts implementing the three-metric audit framework:

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/fresh_split.py` | Execute the v10 fresh-split protocol (seed 789685) | `train_dataset.json`, `val_dataset.json`, `test_dataset.json` |
| `scripts/compute_csli.py` | Compute Choice-Set Leakage Index panel (TF-IDF, SBERT, T5-small) | `paper_exports/csli.json` |
| `scripts/compute_prefix_calibration.py` | Platt-scaled prefix-wise calibration (ECE per prefix bucket) | `paper_exports/calibration.json` |
| `scripts/compute_stopdff.py` | Myopic-threshold StopDFF diagnostic | `paper_exports/stopdff.json` |
| `scripts/make_audit_card.py` | Aggregate all metrics into Pilot Benchmark Translation Card | `paper_exports/audit_card.json` |
| `scripts/regenerate_figures.py` | Regenerate LaTeX tables and figures from cached JSONs | `paper_exports/audit_table.tex`, `paper_exports/csli_panel.png`, `paper_exports/reliability_*.png` |
| `scripts/run_stopdff_v5_local.py` | StopDFF v5 local (CPU) end-to-end audit driver (uses `scripts/stopdff_v5/`) | identity-bound run package under `<out-dir>/runs/<run_id>/` |
| `scripts/modal_stopdff_v5_runner.py` | StopDFF v5 Modal stage functions + durable controller | run package on the `cs321m-stopdff-artifacts` Volume |
| `scripts/modal_stopdff_v5_assurance.py` | StopDFF v5 cross-process Modal recovery-assurance canary | create-once assurance receipts |
| `scripts/verify_stopdff_v5_modal_assurance.py` | Offline verifier for the recovery-assurance receipts | PASS verdict JSON |
| `scripts/validate_stopdff_bucketed_sweep.py` | StopDFF v5 standalone acceptance-gate checker (see `ACCEPTANCE_CONTRACT.md`) | `PASS`/`FAIL` verdict (`--json` report) |

## Pre-computed Artifacts

The `paper_exports/` directory contains all pre-computed results referenced in the manuscript:

| File | Description |
|------|-------------|
| `audit_card.json` | Pilot Benchmark Translation Card (overall verdict + per-metric summaries) |
| `audit_card.md` | Human-readable markdown rendering of the audit card |
| `csli.json` | Panel CSLI results: per-model leakage estimates with confidence intervals |
| `calibration.json` | Prefix-wise calibration results: ECE by prefix bucket (early/mid/late) |
| `stopdff.json` | StopDFF diagnostic results: median absolute prefix shift |
| `audit_table.tex` | LaTeX 8-column audit table for manuscript inclusion |
| `csli_panel.png` | CSLI bar chart (panel across 3 models) |
| `reliability_early.png` | Reliability diagram -- early prefixes |
| `reliability_mid.png` | Reliability diagram -- mid prefixes |
| `reliability_late.png` | Reliability diagram -- late prefixes |

Additionally at repo root:

| File | Description |
|------|-------------|
| `threshold_manifest.json` | Frozen threshold parameters (pre-registered before test inspection) |
| `threshold_manifest.json.sha256` | SHA-256 integrity sidecar for manifest (verified at load time by `scripts/threshold_manifest.py`) |
| `stopdff_report.json` | StopDFF attestation report with timestamp and verdict |
| `ACCEPTANCE_CONTRACT.md` | StopDFF v5 acceptance gate: what `scripts/validate_stopdff_bucketed_sweep.py` must verify for a run to be accepted |
| `SCIENTIFIC_CONTRACT.md` | StopDFF v5 scientific contract (prose index for the executable profile constants) |
| `IDENTITY_AND_ARTIFACT_CONTRACT.md` | StopDFF v5 content-addressed identity, evidence, and create-once artifact rules |
| `SCIENTIFIC_PROFILE.template.json` | Template for the preregistered StopDFF v5 scientific profile |

## Repository Structure

```
qanta-buzzer/
+-- scripts/              Pipeline entrypoints and CS321M audit scripts
|   +-- build_mc_dataset.py       [CS234] MC dataset construction
|   +-- run_baselines.py          [CS234] Baseline agent evaluation
|   +-- train_ppo.py              [CS234] PPO training loop
|   +-- evaluate_all.py           [CS234] Final evaluation report
|   +-- manual-smoke.sh           [CS234] Four-stage smoke wrapper
|   +-- compute_csli.py           [CS321M] CSLI panel computation
|   +-- compute_prefix_calibration.py  [CS321M] Prefix calibration
|   +-- compute_stopdff.py        [CS321M] StopDFF diagnostic
|   +-- make_audit_card.py        [CS321M] Audit card aggregation
|   +-- regenerate_figures.py     [CS321M] Figure/table regeneration
|   +-- fresh_split.py            [CS321M] Fresh split protocol
|   +-- run_stopdff_v5_local.py   [CS321M] StopDFF v5 local reproduction driver
|   +-- validate_stopdff_bucketed_sweep.py  [CS321M] StopDFF v5 acceptance-gate checker
|   +-- stopdff_v5/               [CS321M] StopDFF v5 fail-closed audit pipeline package
+-- qb_data/              [CS234] Data loading, MC construction, stratified splits
+-- qb_env/               [CS234] Gymnasium environment, opponent models
+-- models/               [CS234] Likelihood models (TF-IDF, SBERT, T5)
+-- agents/               [CS234] Threshold, softmax-profile, PPO buzzer agents
+-- evaluation/           [CS234] S_q metric, calibration, plotting utilities
+-- training/             [CS234] T5 policy trainers (supervised + PPO)
+-- configs/              [CS234] YAML configuration files
+-- schemas/              [CS321M] JSON Schemas for StopDFF v5 artifacts
+-- tests/                [CS234] unit + regression test suite (run via `pytest`)
+-- paper_exports/        [CS321M] Pre-computed audit results and figures
+-- artifacts/            Generated pipeline outputs (smoke/ and main/)
+-- docs/                 Pipeline runbook and architecture docs
```

## Attribution

### Contributors

| Contributor | Role | Scope |
|-------------|------|-------|
| Imran Hassan | Original developer | qanta-buzzer core architecture: qb_data/, qb_env/, models/, agents/, evaluation/, training/, main pipeline scripts (build_mc_dataset.py, run_baselines.py, train_ppo.py, evaluate_all.py) |
| Kathleenkk23 | Collaborator | CS234 team contributions (T5 policy extensions, testing) |
| GitHub Copilot | AI assistant | Code suggestions during CS234 development phase |
| Ankit Aggarwal | Current owner, CS321M extensions | All CS321M audit scripts (compute_csli.py, compute_prefix_calibration.py, compute_stopdff.py, make_audit_card.py, regenerate_figures.py, fresh_split.py, modal_cs321m.py), paper_exports/, threshold_manifest, stopdff_report, this README |

### Novel vs Reused Modules

**Novel (CS321M, Ankit Aggarwal):**
- `scripts/compute_csli.py` -- Choice-Set Leakage Index panel computation
- `scripts/compute_prefix_calibration.py` -- Platt scaling + per-bucket ECE
- `scripts/compute_stopdff.py` -- Myopic-threshold StopDFF diagnostic
- `scripts/make_audit_card.py` -- Pilot Benchmark Translation Card aggregation
- `scripts/regenerate_figures.py` -- Figure/table regeneration from cached data
- `scripts/fresh_split.py` -- v10 section 0.3 fresh split protocol
- `modal_cs321m.py` -- Modal A100 compute orchestration wrapper
- `scripts/stopdff_v5/` -- StopDFF v5 fail-closed evidentiary audit pipeline (identity, manifests, sweep, checker, writers)
- `scripts/run_stopdff_v5_local.py`, `scripts/modal_stopdff_v5_runner.py`, `scripts/modal_stopdff_v5_assurance.py`, `scripts/verify_stopdff_v5_modal_assurance.py`, `scripts/validate_stopdff_bucketed_sweep.py` -- StopDFF v5 runners and acceptance-gate checker
- `schemas/` -- JSON Schemas for StopDFF v5 artifacts, plus root contracts (`ACCEPTANCE_CONTRACT.md`, `SCIENTIFIC_CONTRACT.md`, `IDENTITY_AND_ARTIFACT_CONTRACT.md`, `SCIENTIFIC_PROFILE.template.json`)

**Reused (CS234 team, attributed to original contributors):**
- `qb_data/` -- Data loading, MC construction, stratified splits
- `qb_env/` -- Gymnasium environment, opponent models
- `models/` -- Likelihood models (TF-IDF, SBERT, T5)
- `agents/` -- Threshold, softmax-profile, PPO buzzer agents
- `evaluation/` -- S_q metric, calibration, plotting utilities
- `scripts/build_mc_dataset.py`, `run_baselines.py`, `train_ppo.py`, `evaluate_all.py` -- Pipeline entrypoints

### AI Tool Disclosure

Claude Code (Anthropic) assisted with CS321M extension development. GitHub Copilot assisted during the CS234 development phase. See manuscript AI Disclosures for full details.

## Testing

Run the full pytest suite via:

```bash
pytest                    # full suite (use `pytest tests/ --collect-only -q | tail -1` for the live test count)
pytest tests/ -x -q       # quick with stop-on-first-failure
bash scripts/ci.sh        # full suite via repo-local .venv
```

## License

This is a student project submission for Stanford CS321M (AI Measurement Science). Not licensed for redistribution.
