# Device 2 StopDFF Run Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local Device 2 CUDA long-run harness for the expanded DP StopDFF sweep with preflight checks, durable logs, resume support, and paper-export artifacts.

**Architecture:** Keep the scientific sweep logic in `scripts/sweep_stopdff_dp.py`; add a small preflight script that validates the machine, inputs, split separation, and output directory contract before a long run starts. Add a bash wrapper that creates a run directory, records a command manifest, tees stdout/stderr to stable logs, exports `CUDA_VISIBLE_DEVICES=0`, and invokes the existing resumable sweep with the requested axes. Add a runbook with copy-paste commands for starting, monitoring, stopping, resuming, and importing artifacts into `cs321m-paper`.

**Tech Stack:** Python 3.11+, bash, PyTorch CUDA probes, `nvidia-smi`, git, pytest.

---

## Files

- Create: `scripts/device2_cuda_preflight.py`
- Create: `scripts/device2_stopdff_run.sh`
- Create: `docs/device2_stopdff_runbook.md`
- Create: `tests/test_device2_stopdff_run.py`

## Task 1: CUDA Preflight

**Files:**
- Create: `scripts/device2_cuda_preflight.py`
- Create/modify: `tests/test_device2_stopdff_run.py`

- [ ] **Step 1: Write preflight tests first**

Create tests that call preflight helpers without requiring a real GPU:

```python
def test_preflight_rejects_existing_output_without_resume_or_overwrite(tmp_path):
    from scripts import device2_cuda_preflight as preflight
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    result = preflight.check_output_directory(out_dir, resume=False, overwrite=False)
    assert result["ok"] is False
    assert "already exists" in result["message"]

def test_split_separation_detects_overlap(tmp_path):
    from scripts import device2_cuda_preflight as preflight
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "val_dataset.json").write_text('[{"qid": "q1"}]', encoding="utf-8")
    (data_dir / "test_dataset.json").write_text('[{"qid": "q1"}]', encoding="utf-8")
    result = preflight.check_split_separation(data_dir, "val", "test")
    assert result["ok"] is False
    assert result["overlap_count"] == 1

def test_preflight_writes_json_report_with_failed_checks(tmp_path, monkeypatch):
    from scripts import device2_cuda_preflight as preflight
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ("mc_dataset.json", "val_dataset.json", "test_dataset.json"):
        (data_dir / name).write_text("[]", encoding="utf-8")
    calibration = tmp_path / "missing_calibration.json"
    report_path = tmp_path / "preflight.json"
    rc = preflight.main([
        "--out-dir", str(tmp_path / "run"),
        "--data-dir", str(data_dir),
        "--calibration", str(calibration),
        "--output-json", str(report_path),
        "--skip-cuda-probe-for-tests",
    ])
    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["checks"]["required_artifacts"]["ok"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_device2_stopdff_run.py -q`

Expected: failures because `scripts.device2_cuda_preflight` does not exist.

- [ ] **Step 3: Implement preflight**

Implement `scripts/device2_cuda_preflight.py` with:

- argparse flags: `--out-dir`, `--data-dir`, `--calibration`, `--fit-split`, `--eval-split`, `--resume`, `--overwrite`, `--output-json`, `--device-index`, `--min-free-gib`, `--required-artifact`, `--skip-cuda-probe-for-tests`.
- checks for `nvidia-smi`, `torch.cuda.is_available()`, CUDA device name/memory total/free, Python version, repo commit/dirty status, disk free bytes/GiB, required artifact existence, validation/test split disjoint qids, and output directory existence rules.
- JSON report schema: top-level `ok`, `timestamp_utc`, `repo_root`, `checks`, and `errors`.
- no test-split fitting: report `fit_split`, `eval_split`, and fail if split names match or qids overlap.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_device2_stopdff_run.py -q`

Expected: pass for preflight tests.

## Task 2: Bash Harness

**Files:**
- Create: `scripts/device2_stopdff_run.sh`
- Modify: `tests/test_device2_stopdff_run.py`

- [ ] **Step 1: Write harness tests first**

Add tests that copy the wrapper into a temp repo and stub `.venv/bin/python` to verify:

```python
def test_device2_harness_uses_run_dir_logs_and_resume(tmp_path):
    repo = tmp_path / "repo"
    script = copy_script(repo, "device2_stopdff_run.sh")
    calls = repo / "calls.log"
    write_executable(repo / ".venv" / "bin" / "python", python_stub_that_logs_calls)
    write_executable(repo / "git", git_stub_returning_known_sha_and_clean_status)
    result = subprocess.run([
        bash, str(script),
        "--experiment", "dp_sweep",
        "--out-dir", str(repo / "artifacts" / "device2_stopdff" / "full_6h"),
        "--artifact-dir", "paper_exports",
        "--max-wall-hours", "6",
        "--num-bootstrap", "1000",
        "--n-jobs", "12",
        "--resume",
        "--calibrators", "platt,temperature,isotonic",
        "--reward-schedules", "acf_flat,power_mark",
        "--continuations", "empirical_bucket,pooled_empirical",
    ], cwd=repo, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert (repo / "artifacts" / "device2_stopdff" / "full_6h" / "stdout.log").exists()
    assert (repo / "artifacts" / "device2_stopdff" / "full_6h" / "stderr.log").exists()
    assert "--resume" in calls.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=0" in calls.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_device2_stopdff_run.py -q`

Expected: harness test fails because `scripts/device2_stopdff_run.sh` does not exist.

- [ ] **Step 3: Implement harness**

Implement `scripts/device2_stopdff_run.sh` with:

- `set -euo pipefail`.
- repo-local Python selection from `.venv/bin/python` first, then `.venv/Scripts/python.exe`.
- default timestamped run directory `artifacts/device2_stopdff/<timestamp>_<gitsha>/` when `--out-dir` is omitted; exact explicit `--out-dir` support for repeatable resume paths.
- run directory contents: `preflight.json`, `command_manifest.json`, `stdout.log`, `stderr.log`, and `paper_exports/`.
- logging via `exec > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)`.
- preflight invocation before sweep, passing resume/overwrite and input paths.
- sweep invocation:

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/sweep_stopdff_dp.py \
  --artifact-dir "$RUN_DIR/$ARTIFACT_DIR" \
  --out "$RUN_DIR/$ARTIFACT_DIR/stopdff_dp_sweep.json" \
  --max-wall-hours "$MAX_WALL_HOURS" \
  --num-bootstrap "$NUM_BOOTSTRAP" \
  --n-jobs "$N_JOBS" \
  --calibrators "$CALIBRATORS" \
  --reward-schedules "$REWARD_SCHEDULES" \
  --continuations "$CONTINUATIONS"
```

Append `--resume` when requested. In `--smoke` mode append `--smoke`, set `--max-cells` from `SMOKE_MAX_CELLS` defaulting to `2`, and keep writes under the run directory.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_device2_stopdff_run.py -q`

Expected: pass.

## Task 3: Runbook and Final Review

**Files:**
- Create: `docs/device2_stopdff_runbook.md`

- [ ] **Step 1: Write runbook**

The runbook must include copy-paste commands for:

- setup and preflight-only validation,
- tmux start: `tmux new -s cs321m-stopdff`,
- smoke run under 10 minutes,
- full 6-hour run with the requested calibrators, rewards, continuations, bootstrap count, and jobs,
- monitoring with `tail -f`, `nvidia-smi`, and paper export inventory checks,
- stopping with `Ctrl-C` and no deletion,
- resuming with the same `--out-dir --resume`,
- importing artifacts into `cs321m-paper`.

- [ ] **Step 2: Review against acceptance criteria**

Check that:

- smoke mode can be bounded by `SMOKE_MAX_CELLS=2`,
- full run needs no interactive input after launch,
- resume passes through to the sweep cache,
- the final directory contains `paper_exports/stopdff_dp_sweep.json`, `figures/`, and `stopdff_dp_sweep_table.tex`,
- the preflight does not assume GPU is needed for bucketed DP; it validates CUDA readiness for optional learned/PyTorch stages.

- [ ] **Step 3: Run verification**

Run:

```bash
pytest tests/test_device2_stopdff_run.py -q
bash -n scripts/device2_stopdff_run.sh
python scripts/device2_cuda_preflight.py --help
```

Expected: all commands exit 0.
