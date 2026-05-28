# Device 2 Cross-Platform Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the Device 2 StopDFF harness against Windows CRLF checkout and checksum-provenance mistakes.

**Architecture:** Add a repo-level line-ending policy, document the local Git settings needed on Windows, and cover both with focused tests in the existing Device 2 harness test module.

**Tech Stack:** pytest, Git attributes, bash runbook commands.

---

## Task 1: Add Failing Tests

**Files:**
- Modify: `tests/test_device2_stopdff_run.py`

- [ ] **Step 1: Write failing tests for `.gitattributes` policy and runbook Windows setup**

Add tests that assert `.gitattributes` pins `*.sh`, `*.py`, `*.json`, `*.json.sha256`, and `*.md` to LF, that the runbook tells Windows users to set `core.autocrlf false` and `core.eol lf`, and that critical Device 2 files contain no CRLF bytes.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_device2_stopdff_run.py::test_gitattributes_pins_text_outputs_to_lf tests/test_device2_stopdff_run.py::test_runbook_documents_windows_lf_checkout_setup -q
```

Expected: fail because `.gitattributes` is absent and the runbook does not document the Windows LF setup.

## Task 2: Minimal Hardening

**Files:**
- Create: `.gitattributes`
- Modify: `docs/device2_stopdff_runbook.md`

- [ ] **Step 3: Add `.gitattributes` LF rules**

Create `.gitattributes` with exact rules:

```gitattributes
*.sh text eol=lf
*.py text eol=lf
*.json text eol=lf
*.json.sha256 text eol=lf
*.md text eol=lf
```

- [ ] **Step 4: Add runbook Windows setup commands**

Add a setup subsection before virtualenv creation:

```bash
git config --local core.autocrlf false
git config --local core.eol lf
git ls-files --eol scripts/device2_stopdff_run.sh threshold_manifest.json threshold_manifest.json.sha256
```

- [ ] **Step 5: Verify GREEN**

Run the failing-test command from Step 2, then run the full Device 2 focused test file and bash syntax check:

```bash
pytest tests/test_device2_stopdff_run.py -q
bash -n scripts/device2_stopdff_run.sh
python scripts/device2_cuda_preflight.py --help
```

## Task 3: Commit

**Files:**
- Stage only the Device 2 harness files, this hardening plan, `.gitattributes`, and related tests.

- [ ] **Step 6: Confirm diff scope**

Run:

```bash
git status --short
git diff -- .gitattributes docs/device2_stopdff_runbook.md tests/test_device2_stopdff_run.py docs/superpowers/plans/2026-05-27-device2-cross-platform-hardening.md
```

- [ ] **Step 7: Commit**

Run:

```bash
git add .gitattributes docs/device2_stopdff_runbook.md docs/superpowers/plans/2026-05-27-device2-cross-platform-hardening.md scripts/device2_cuda_preflight.py scripts/device2_stopdff_run.sh tests/test_device2_stopdff_run.py
git commit -m "chore: harden device2 harness line endings"
```
