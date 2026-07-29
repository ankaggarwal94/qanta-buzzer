"""scripts/modal_stopdff_runner.py -- Modal execution wrapper for the StopDFF DP sweep.

Thin Modal wrapper that dispatches the CPU/GPU-bound finite-horizon DP StopDFF
computation (``scripts/compute_stopdff_dp.py``) or its resumable sensitivity
sweep (``scripts/sweep_stopdff_dp.py``) on managed compute. Artifacts are
written to a Modal Volume mounted at ``/artifacts`` under an operator-chosen
subdir so multiple runs can coexist; the local repo's curated
``paper_exports/`` is never touched. The wrapper streams live subprocess logs
both to the Modal CLI and to a tee'd log file on the Volume; it also writes a
``run_manifest.json`` that captures git commit, environment, and invocation so
downstream paper-repo tooling can cite the run.

Usage (recommended invocations):

    modal volume create cs321m-stopdff-artifacts

    # CPU-only smoke (under an hour)
    modal run --detach scripts/modal_stopdff_runner.py \\
        --experiment smoke \\
        --artifact-subdir dp_smoke_$(date +%Y%m%d_%H%M%S) \\
        --gpu none \\
        --max-wall-hours 1 \\
        --num-bootstrap 100 \\
        --smoke

    # Full DP sensitivity sweep on L40S
    modal run --detach scripts/modal_stopdff_runner.py \\
        --experiment dp_sweep \\
        --artifact-subdir dp_sweep_$(date +%Y%m%d_%H%M%S) \\
        --gpu L40S \\
        --max-wall-hours 6 \\
        --num-bootstrap 1000 \\
        --n-jobs 8

    # Single full DP run (no smoke trim) on CPU
    modal run scripts/modal_stopdff_runner.py \\
        --experiment single \\
        --artifact-subdir dp_single_$(date +%Y%m%d_%H%M%S) \\
        --gpu none \\
        --num-bootstrap 500

After completion, artifacts can be inspected and downloaded with::

    modal volume ls cs321m-stopdff-artifacts <artifact-subdir>
    modal volume get cs321m-stopdff-artifacts <artifact-subdir> ./downloads/

Design constraints (see PRIOR LESSONS L1-L16 in the implementation prompt):

* Modal SDK v1.x API only: ``modal.App``, string GPU spec, ``add_local_dir(copy=True)``,
  ``scaledown_window``, ``max_containers``. No deprecated ``modal.Stub`` /
  ``modal.gpu.X()`` objects / ``container_idle_timeout`` / ``Mount``.
* Default DP path requires no API keys. Secrets are opt-in via
  ``--with-openai-key`` and attached via ``with_options`` only at call time.
* ``paper_exports/calibration.json`` is a hard prerequisite -- the wrapper
  pre-flight checks for it inside the container and fails fast if missing.
* Artifacts are routed strictly under ``/artifacts/<subdir>/`` on the Volume;
  the curated repo ``paper_exports/`` is never overwritten.
* Subprocess logs stream live to both the Modal CLI and a tee'd log file via
  ``Popen`` + a small line loop.
* The cell-cache directory lives under ``/artifacts/<subdir>/paper_exports/``
  so ``--resume`` works across Modal container restarts; this is enforced by
  passing ``--artifact-dir`` explicitly to the sweep.
* The local entrypoint refuses to dispatch on a dirty repo unless
  ``--allow-dirty``; refuses to overwrite a non-empty existing subdir unless
  ``--overwrite`` (``--resume`` is the legitimate alternative).
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "cs321m-stopdff"
VOLUME_NAME = "cs321m-stopdff-artifacts"
ARTIFACTS_ROOT = PurePosixPath("/artifacts")
REPO_PATH = PurePosixPath("/root/qanta-buzzer")  # baked-in repo location

EXPERIMENTS = (
    "smoke",
    "single",
    "dp_sweep",
    "fair_qa",
    "learned_value_train",
    "learned_value_eval",
)
GPU_NONE_SYNONYMS = {"none", "cpu", "", "null"}
MODAL_MAX_TIMEOUT_SECONDS = 86400  # Modal's hard 24h ceiling on @app.function timeout
TIMEOUT_BUFFER = 1.15  # 15% buffer over sweep's --max-wall-hours for setup/teardown

# Local repo root (used only by @app.local_entrypoint host-side checks).
LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# Per L8: copy=True so `pip install -e .` sees the repo at build time.
# Per L9: PYTHONUNBUFFERED + MPLBACKEND=Agg are mandatory.
# Per L11: we do NOT rely on .git inside the container -- the host stamps
# the commit via run_stopdff(...) args, so we ignore .git to keep the image
# small and avoid the "modified during build" race that bit modal_cs321m.py.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("pip>=24.0", "setuptools>=69.0", "wheel")
    .pip_install_from_requirements(str(LOCAL_REPO_ROOT / "requirements.txt"))
    .add_local_dir(
        str(LOCAL_REPO_ROOT),
        remote_path=str(REPO_PATH),
        copy=True,
        ignore=[
            "__pycache__",
            "**/__pycache__",
            "**/__pycache__/**",
            "*.pyc",
            "*.pyo",
            ".pytest_cache",
            ".pytest_cache/**",
            ".mypy_cache",
            ".mypy_cache/**",
            ".ruff_cache",
            ".ruff_cache/**",
            ".venv",
            ".venv/**",
            ".claude",
            ".claude/**",
            ".git",
            ".git/**",
            "node_modules",
            "node_modules/**",
            "data/raw",
            "data/raw/**",
        ],
    )
    .run_commands(f"cd {REPO_PATH} && pip install -e . --no-deps")
    .workdir(str(REPO_PATH))
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "PIP_NO_CACHE_DIR": "1",
        }
    )
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_gpu(g: Optional[str]) -> Optional[str]:
    """Map CLI ``--gpu`` to Modal's accepted forms.

    Accepts ``none``, ``cpu``, ``""``, ``null`` as synonyms for CPU-only and
    returns ``None`` so Modal allocates a CPU container. Otherwise returns the
    trimmed string (e.g., ``L40S``, ``A100-80GB``, ``H100``) for Modal to
    validate.
    """
    if g is None:
        return None
    g_clean = g.strip()
    if g_clean.lower() in GPU_NONE_SYNONYMS:
        return None
    return g_clean


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _capture_local_git(
    repo_root: Path,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Capture ``git rev-parse HEAD`` plus two porcelain views on the host.

    Per L11/L12: this runs on the LOCAL side before dispatching to Modal, where
    git is reliable. The container itself does not need .git.

    Returns ``(commit_sha_or_None, tracked_porcelain_or_None, full_porcelain_or_None)``
    where:

    * ``commit_sha_or_None`` is ``git rev-parse HEAD`` output (or ``None`` if
      ``git`` is not installed or this is not a git checkout).
    * ``tracked_porcelain_or_None`` is ``git status --porcelain --untracked-files=no``
      -- used by the dirty-tree gate so that untracked-only worktrees (e.g., a
      fresh wrapper file not yet committed) do not block dispatch.
    * ``full_porcelain_or_None`` is ``git status --porcelain`` -- preserved in
      the manifest for complete provenance.

    Both porcelain values are ``None`` only when the git command itself failed
    (binary missing or not a repo); an empty string means a clean tree.
    """
    commit: Optional[str] = None
    tracked_porcelain: Optional[str] = None
    full_porcelain: Optional[str] = None
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            commit = r.stdout.strip()
    except OSError:
        pass
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            tracked_porcelain = r.stdout  # may be empty (clean) or have entries
    except OSError:
        pass
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            full_porcelain = r.stdout  # may be empty (clean) or have entries
    except OSError:
        pass
    return commit, tracked_porcelain, full_porcelain


def _producer_script_path(experiment: str) -> Path:
    mapping = {
        "smoke": "scripts/compute_stopdff_dp.py",
        "single": "scripts/compute_stopdff_dp.py",
        "dp_sweep": "scripts/sweep_stopdff_dp.py",
        "fair_qa": "scripts/stopdff_fair_qa_retest.py",
        "learned_value_train": "scripts/train_stopdff_value_model.py",
        "learned_value_eval": "scripts/compute_stopdff_learned_value.py",
    }
    try:
        return LOCAL_REPO_ROOT / mapping[experiment]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment: {experiment!r}") from exc


def _canonical_source_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _committed_file_sha256(commit: str, path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(LOCAL_REPO_ROOT).as_posix()
    except ValueError:
        return None
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=str(LOCAL_REPO_ROOT),
            check=False,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _build_command(
    experiment: str,
    artifact_subdir_abs: PurePosixPath,
    num_bootstrap: int,
    max_wall_hours: float,
    n_jobs: int,
    resume: bool,
    smoke: bool,
) -> list[str]:
    """Construct the subprocess command for the selected experiment.

    Routes outputs under ``artifact_subdir_abs/paper_exports/`` (L4) and passes
    ``--artifact-dir`` for the sweep so the cell cache survives container
    restarts (L7). ``compute_stopdff_dp.py`` does not accept ``--artifact-dir``
    or sweep-specific flags, so we only pass what each script understands (L6).
    """
    exports_dir = artifact_subdir_abs / "paper_exports"
    python = sys.executable

    if experiment == "smoke":
        # compute_stopdff_dp.py with --smoke + sensible smoke defaults.
        out_json = exports_dir / "stopdff_dp.json"
        out_md = exports_dir / "stopdff_dp.md"
        out_tex = exports_dir / "stopdff_dp_table.tex"
        cmd = [
            python,
            "scripts/compute_stopdff_dp.py",
            "--smoke",
            "--out",
            str(out_json),
            "--out-md",
            str(out_md),
            "--out-tex",
            str(out_tex),
            # The smoke fixture has tiny val/test slices; permit incomplete
            # MC coverage / low retention to mirror modal_cs321m.py's smoke
            # routing so the run completes end-to-end.
            "--allow-incomplete-mc-coverage",
            "--allow-low-mc-retention",
        ]
        return cmd

    if experiment == "single":
        # compute_stopdff_dp.py without --smoke (a single full DP run).
        out_json = exports_dir / "stopdff_dp.json"
        out_md = exports_dir / "stopdff_dp.md"
        out_tex = exports_dir / "stopdff_dp_table.tex"
        return [
            python,
            "scripts/compute_stopdff_dp.py",
            "--out",
            str(out_json),
            "--out-md",
            str(out_md),
            "--out-tex",
            str(out_tex),
        ]

    if experiment == "dp_sweep":
        # sweep_stopdff_dp.py with the full grid.
        out_json = exports_dir / "stopdff_dp_sweep.json"
        cmd = [
            python,
            "scripts/sweep_stopdff_dp.py",
            "--artifact-dir",
            str(exports_dir),
            "--num-bootstrap",
            str(int(num_bootstrap)),
            "--n-jobs",
            str(int(n_jobs)),
            "--out",
            str(out_json),
        ]
        if max_wall_hours and max_wall_hours > 0:
            cmd.extend(["--max-wall-hours", f"{float(max_wall_hours):.6f}"])
        if resume:
            cmd.append("--resume")
        if smoke:
            cmd.append("--smoke")
        return cmd

    if experiment == "learned_value_train":
        # Train a learned continuation-value model on train, using validation
        # for early stopping, for the learned-value DP evaluator.
        checkpoint_dir = artifact_subdir_abs / "value_model"
        cmd = [
            python,
            "scripts/train_stopdff_value_model.py",
            "--artifact-dir",
            str(exports_dir),
            "--train-split",
            "train",
            "--val-split",
            "val",
            "--device",
            "cuda",
            "--out",
            str(checkpoint_dir),
        ]
        if smoke:
            cmd.extend([
                "--epochs", "2",
                "--seeds", "1",
                "--hidden", "32",
            ])
        return cmd

    if experiment == "learned_value_eval":
        # Apply trained learned-value checkpoints to the test split and write
        # paper_exports/stopdff_learned_value.json.
        # See learned_value_train for the upstream checkpoint that this run
        # consumes.
        checkpoint_dir = artifact_subdir_abs / "value_model"
        out_json = exports_dir / "stopdff_learned_value.json"
        cmd = [
            python,
            "scripts/compute_stopdff_learned_value.py",
            "--artifact-dir",
            str(exports_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--eval-split",
            "test",
            "--out",
            str(out_json),
        ]
        if smoke:
            cmd.append("--smoke")
        return cmd

    if experiment == "fair_qa":
        # Difficulty-matched fair-QA StopDFF retest with per-format calibration
        # and item-bootstrap CIs (scripts/stopdff_fair_qa_retest.py). Reuses the
        # real stopdff_dp solver; runs the full eval/fit splits by default. The
        # --num-bootstrap value is forwarded; --smoke trims to 30/30 for a quick run.
        out_json = exports_dir / "stopdff_fair_qa.json"
        cmd = [
            python,
            "scripts/stopdff_fair_qa_retest.py",
            "--num-bootstrap",
            str(int(num_bootstrap)),
            "--reward-schedule",
            "power_mark",
            "--fit-split",
            "val",
            "--eval-split",
            "test",
            "--qa-arms",
            "idealized,krandom,khard,kdisjoint,klex",
            "--calibrations",
            "shared,performat",
            "--out",
            str(out_json),
        ]
        if smoke:
            cmd.extend(["--n-test", "30", "--n-val", "30"])
        return cmd

    raise ValueError(f"Unknown experiment: {experiment!r} (expected one of {EXPERIMENTS})")


# Credential-shaped env var prefixes/keys we scrub from the subprocess env by
# default. The default DP path does not need any of these; if a future
# experiment legitimately requires one, plumb a `keep=` whitelist through
# `_scrub_env`. Keeps OPENAI_API_KEY (and friends) out of subprocess debug
# dumps that would otherwise tee into the Volume log file (FIX-7).
SENSITIVE_ENV_PREFIXES: tuple[str, ...] = (
    "OPENAI_",
    "ANTHROPIC_",
    "HF_",
    "WANDB_",
    "AWS_",
    "GCP_",
    "AZURE_",
)
SENSITIVE_ENV_KEYS: frozenset[str] = frozenset(
    {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "WANDB_API_KEY"}
)


def _scrub_env(parent_env: dict, keep: tuple[str, ...] = ()) -> dict:
    """Return a copy of ``parent_env`` with credential-shaped vars removed.

    Variables listed in ``keep`` are passed through verbatim even if they match
    a sensitive prefix or key, for the rare case where a subprocess truly needs
    them (no current call site does).
    """
    keep_set = set(keep)
    out: dict = {}
    for k, v in parent_env.items():
        if k in keep_set:
            out[k] = v
            continue
        if k in SENSITIVE_ENV_KEYS:
            continue
        if any(k.startswith(p) for p in SENSITIVE_ENV_PREFIXES):
            continue
        out[k] = v
    return out


def _stream_subprocess(
    cmd: list[str],
    log_path: Path,
    cwd: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> int:
    """Run ``cmd`` and tee live stdout/stderr to both ``sys.stdout`` and ``log_path``.

    Per L5: ``Popen`` + line-streaming loop avoids capture-deadlock and surfaces
    logs to ``modal run`` in real time while also writing to the Volume. stderr
    is merged into stdout to preserve interleaving in the log file. Per FIX-7,
    credential-shaped env vars are scrubbed from the subprocess env by default.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[runner] command: {shlex.join(cmd)}", flush=True)
    print(f"[runner] cwd:     {cwd}", flush=True)
    print(f"[runner] log:     {log_path}", flush=True)

    env = _scrub_env(dict(os.environ))
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== subprocess start {_utcnow_iso()} =====\n")
        logf.write(f"cmd: {shlex.join(cmd)}\ncwd: {cwd}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            env=env,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
        finally:
            rc = proc.wait()
            logf.write(f"===== subprocess exit rc={rc} {_utcnow_iso()} =====\n")
    return rc


def _subprocess_provenance_env(
    *,
    commit: str | None,
    tracked_status: str | None,
    producer_sha256: str,
    trainer_sha256: str,
) -> dict[str, str]:
    """Build the exact source identity passed into the tool-poor child."""
    return {
        "MODAL_HOST_GIT_COMMIT": commit or "",
        "MODAL_HOST_GIT_STATUS": tracked_status or "",
        "MODAL_HOST_PRODUCER_SCRIPT_SHA256": producer_sha256,
        "MODAL_HOST_TRAINER_SCRIPT_SHA256": trainer_sha256,
    }


def _print_env_banner(artifact_subdir_abs: PurePosixPath) -> dict:
    """Print and return diagnostic info about the runtime environment."""
    banner: dict = {
        "timestamp": _utcnow_iso(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cwd": str(REPO_PATH),
        "artifact_subdir": str(artifact_subdir_abs),
    }
    print("=" * 60, flush=True)
    print(f"[runner] modal_stopdff_runner.py @ {banner['timestamp']}", flush=True)
    print(f"[runner] python:  {banner['python_version']} ({banner['python_executable']})", flush=True)
    print(f"[runner] cwd:     {banner['cwd']}", flush=True)
    print(f"[runner] subdir:  {banner['artifact_subdir']}", flush=True)

    # torch / CUDA probe.
    try:
        import torch  # noqa: WPS433 -- intentional runtime probe

        cuda_available = bool(torch.cuda.is_available())
        torch_version = str(torch.__version__)
        cuda_version = str(getattr(torch.version, "cuda", None))
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        device_name = (
            torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None
        )
        banner.update(
            {
                "torch_version": torch_version,
                "torch_cuda_build": cuda_version,
                "cuda_available": cuda_available,
                "cuda_device_count": device_count,
                "cuda_device_name": device_name,
            }
        )
        print(f"[runner] torch:   {torch_version} (cuda build {cuda_version})", flush=True)
        print(
            f"[runner] cuda:    available={cuda_available} count={device_count} "
            f"name={device_name!r}",
            flush=True,
        )
    except Exception as exc:  # torch import failure is non-fatal for the wrapper itself.
        banner["torch_error"] = repr(exc)
        print(f"[runner] torch:   import failed: {exc!r}", flush=True)

    # nvidia-smi if available.
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            out = subprocess.run(
                [nvsmi],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            print("[runner] nvidia-smi:", flush=True)
            if out.stdout:
                print(out.stdout, flush=True)
            if out.returncode != 0 and out.stderr:
                print(out.stderr, file=sys.stderr, flush=True)
            banner["nvidia_smi_returncode"] = out.returncode
        except Exception as exc:  # don't fail the run on probe error
            print(f"[runner] nvidia-smi probe failed: {exc!r}", flush=True)
            banner["nvidia_smi_error"] = repr(exc)
    else:
        print("[runner] nvidia-smi: not installed (CPU container or driver absent)", flush=True)
        banner["nvidia_smi_returncode"] = None

    # Disk usage at /artifacts and repo.
    for label, target in (("artifacts", str(ARTIFACTS_ROOT)), ("repo", str(REPO_PATH))):
        try:
            usage = shutil.disk_usage(target)
            print(
                f"[runner] disk[{label}]: total={usage.total // (1 << 20)}MB "
                f"used={usage.used // (1 << 20)}MB free={usage.free // (1 << 20)}MB",
                flush=True,
            )
            banner[f"disk_{label}_total_mb"] = usage.total // (1 << 20)
            banner[f"disk_{label}_used_mb"] = usage.used // (1 << 20)
            banner[f"disk_{label}_free_mb"] = usage.free // (1 << 20)
        except FileNotFoundError:
            banner[f"disk_{label}"] = "missing"
    print("=" * 60, flush=True)
    return banner


def _calibration_path() -> PurePosixPath:
    return PurePosixPath(str(REPO_PATH)) / "paper_exports" / "calibration.json"


def _list_existing_subdir(artifact_subdir_abs: PurePosixPath) -> list[str]:
    """Return a list of entries (file/dir names) inside ``artifact_subdir_abs``.

    Empty list means the dir doesn't exist or is empty. Used to enforce the
    overwrite policy (L13).
    """
    p = Path(str(artifact_subdir_abs))
    if not p.exists():
        return []
    try:
        return sorted(child.name for child in p.iterdir())
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------
# Per L1/L2: no module-level secrets, no deprecated knobs. Per L15: timeout is
# overridden per-call via with_options() in the local entrypoint; the default
# here is the Modal hard ceiling so detached long runs are not pre-empted by a
# tight default. scaledown_window is short because each invocation does its own
# work; we don't want a warm pool.

@app.function(
    volumes={str(ARTIFACTS_ROOT): vol},
    timeout=MODAL_MAX_TIMEOUT_SECONDS,
    scaledown_window=60,
    max_containers=1,
)
def run_stopdff(
    experiment: str,
    artifact_subdir: str,
    num_bootstrap: int,
    max_wall_hours: float,
    n_jobs: int,
    resume: bool,
    smoke: bool,
    overwrite: bool,
    git_ref_local: Optional[str],
    git_ref_declared: Optional[str],
    git_ref_actual: Optional[str],
    git_dirty_local: bool,
    git_tracked_porcelain_local: Optional[str],
    git_porcelain_local: Optional[str],
    git_present_local: bool,
    producer_script_sha256_local: str,
    trainer_script_sha256_local: str,
    cli_invocation: list[str],
) -> dict:
    """Execute the StopDFF DP run on a Modal container and persist artifacts.

    All artifacts (paper_exports, logs, manifest) are written under
    ``/artifacts/<artifact_subdir>/`` and committed to the Volume.

    Returns a small dict summarizing status, paths, and wall-clock so the local
    entrypoint can print a useful completion banner.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment!r} (expected one of {EXPERIMENTS})")

    run_start = time.time()
    artifact_subdir_abs = ARTIFACTS_ROOT / artifact_subdir
    exports_dir = artifact_subdir_abs / "paper_exports"
    logs_dir = artifact_subdir_abs / "logs"
    log_path = Path(str(logs_dir)) / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    manifest_path = Path(str(artifact_subdir_abs)) / "run_manifest.json"

    summary: dict = {
        "experiment": experiment,
        "artifact_subdir": str(artifact_subdir_abs),
        "exports_dir": str(exports_dir),
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
        "status": "pending",
        "smoke": bool(smoke),
        "resume": bool(resume),
        "overwrite": bool(overwrite),
        "started_at": _utcnow_iso(),
        "git_ref_local": git_ref_local,
        "git_ref_declared": git_ref_declared,
        "git_ref_actual": git_ref_actual,
        "git_dirty_local": bool(git_dirty_local),
        "git_tracked_porcelain_local": git_tracked_porcelain_local,
        "git_porcelain_local": git_porcelain_local,
        "git_present_local": bool(git_present_local),
    }

    # --- Pre-flight: env banner (L9) ------------------------------------
    env_banner = _print_env_banner(artifact_subdir_abs)

    # --- Pre-flight: overwrite policy (L13, FIX-6) ----------------------
    existing = _list_existing_subdir(artifact_subdir_abs)
    if existing and not overwrite and not resume:
        msg = (
            f"Refusing to run: artifact subdir already exists and is non-empty: "
            f"{artifact_subdir_abs} (entries: {existing[:10]}{'...' if len(existing) > 10 else ''}). "
            f"Pass --overwrite to replace it, or --resume to continue an interrupted run."
        )
        print(f"ERROR: {msg}", file=sys.stderr, flush=True)
        summary["status"] = "refused_existing_subdir"
        summary["error"] = msg
        summary["existing_entries"] = existing
        return summary

    # FIX-6: when --overwrite (and not --resume), actually clear the subdir so
    # the new run does not mix with stale artifacts. --resume legitimately
    # wants to keep prior cell-cache state and is the only case where the
    # existing tree should remain untouched.
    if existing and overwrite and not resume:
        shutil.rmtree(str(artifact_subdir_abs), ignore_errors=True)
        summary["overwritten_existing"] = True
        print(
            f"[runner] --overwrite: cleared {artifact_subdir_abs} ({len(existing)} entries)",
            flush=True,
        )

    # --- Pre-flight: calibration prerequisite (L3, FIX-5) ---------------
    cal_path = _calibration_path()
    cal_path_local = Path(str(cal_path))
    if not cal_path_local.exists():
        msg = (
            f"Hard prerequisite missing: {cal_path}. Generate it with "
            f"`python scripts/compute_prefix_calibration.py ...` locally and "
            f"rebuild the Modal image (it is baked in via add_local_dir)."
        )
        print(f"ERROR: {msg}", file=sys.stderr, flush=True)
        summary["status"] = "missing_calibration"
        summary["error"] = msg
        # Persist the failure manifest so paper-repo tooling has a record.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            manifest_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception as exc:
            print(f"[runner] WARNING: failed to write failure manifest: {exc!r}", flush=True)
        try:
            vol.commit()
        except Exception:
            pass
        return summary

    # FIX-5: not just .exists() -- parse + lightly validate the schema so a
    # corrupted/empty calibration file fails fast with a clear error, and
    # stamp a SHA256 in the manifest for paper provenance.
    try:
        calibration_text = cal_path_local.read_text(encoding="utf-8")
        calibration_obj = json.loads(calibration_text)
    except (OSError, json.JSONDecodeError) as exc:
        err = f"Calibration at {cal_path} could not be parsed: {exc!r}"
        print(f"ERROR: {err}", file=sys.stderr, flush=True)
        summary["status"] = "invalid_calibration"
        summary["error"] = err
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            manifest_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception as write_exc:
            print(
                f"[runner] WARNING: failed to write failure manifest: {write_exc!r}",
                flush=True,
            )
        try:
            vol.commit()
        except Exception:
            pass
        raise SystemExit(err)

    if not isinstance(calibration_obj, dict) or not calibration_obj:
        err = f"Calibration at {cal_path} is not a non-empty dict"
        print(f"ERROR: {err}", file=sys.stderr, flush=True)
        summary["status"] = "invalid_calibration"
        summary["error"] = err
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            manifest_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception as write_exc:
            print(
                f"[runner] WARNING: failed to write failure manifest: {write_exc!r}",
                flush=True,
            )
        try:
            vol.commit()
        except Exception:
            pass
        raise SystemExit(err)

    import hashlib

    summary["calibration_path"] = str(cal_path)
    summary["calibration_sha256"] = hashlib.sha256(calibration_text.encode("utf-8")).hexdigest()
    print(
        f"[runner] calibration: {cal_path} (sha256={summary['calibration_sha256'][:12]}...)",
        flush=True,
    )

    # Make the artifact subdir tree (idempotent on resume) and commit a
    # pre-flight stamp so the Volume reflects the run start even if the
    # subprocess crashes hard (L16).
    Path(str(exports_dir)).mkdir(parents=True, exist_ok=True)
    Path(str(logs_dir)).mkdir(parents=True, exist_ok=True)
    preflight_stamp = Path(str(artifact_subdir_abs)) / ".run_started.json"
    preflight_stamp.write_text(
        json.dumps(
            {
                "started_at": summary["started_at"],
                "experiment": experiment,
                "smoke": bool(smoke),
                "git_ref_local": git_ref_local,
                "git_ref_actual": git_ref_actual,
                "git_ref_declared": git_ref_declared,
                "git_dirty_local": bool(git_dirty_local),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Pre-flight commit is informational -- swallow-and-warn is fine here
    # because the load-bearing artifacts haven't been produced yet (FIX-4).
    try:
        vol.commit()
    except Exception as exc:
        print(f"[runner] WARNING: pre-flight vol.commit() failed: {exc!r}", flush=True)

    # --- Build subprocess command --------------------------------------
    cmd = _build_command(
        experiment=experiment,
        artifact_subdir_abs=artifact_subdir_abs,
        num_bootstrap=num_bootstrap,
        max_wall_hours=max_wall_hours,
        n_jobs=n_jobs,
        resume=resume,
        smoke=smoke,
    )
    summary["command"] = cmd

    # --- Run subprocess with live tee (L5, FIX-2) ----------------------
    # The subprocess can crash hard (OOM, SIGKILL, KeyboardInterrupt). The
    # try/except/finally below guarantees the manifest is written and the
    # Volume is committed even if BaseException bubbles up. After the
    # finally block we re-raise so Modal marks the run failed.
    sp_start = time.time()
    rc = -1
    caught_exc: Optional[BaseException] = None
    try:
        try:
            provenance_env = _subprocess_provenance_env(
                commit=git_ref_actual,
                tracked_status=git_tracked_porcelain_local,
                producer_sha256=producer_script_sha256_local,
                trainer_sha256=trainer_script_sha256_local,
            )
            rc = _stream_subprocess(
                cmd,
                log_path=log_path,
                cwd=str(REPO_PATH),
                extra_env=provenance_env,
            )
        except Exception as exc:  # OS-level failure (binary missing, etc.)
            rc = -1
            summary["status"] = "subprocess_launch_error"
            summary["error"] = repr(exc)
            print(f"ERROR: subprocess launch failed: {exc!r}", file=sys.stderr, flush=True)
    except BaseException as exc:  # e.g. KeyboardInterrupt, SystemExit, etc.
        caught_exc = exc
        # rc stays at the value set by _stream_subprocess (or -1 if interrupted
        # before completion); status is decided in the finally block.
    finally:
        sp_duration = time.time() - sp_start
        summary["finished_at"] = _utcnow_iso()
        summary["subprocess_returncode"] = rc
        summary["subprocess_duration_seconds"] = round(sp_duration, 2)
        if caught_exc is not None:
            summary["status"] = "interrupted"
            summary["error"] = repr(caught_exc)
        elif summary.get("status") in (None, "pending"):
            summary["status"] = "success" if rc == 0 else "subprocess_failed"

        # --- Write manifest (always) -----------------------------------
        manifest = {
            "schema_version": 1,
            "app_name": APP_NAME,
            "volume_name": VOLUME_NAME,
            "experiment": experiment,
            "smoke": bool(smoke),
            "artifact_subdir": artifact_subdir,
            "artifact_subdir_abs": str(artifact_subdir_abs),
            "exports_dir": str(exports_dir),
            "log_path": str(log_path),
            "started_at": summary["started_at"],
            "finished_at": summary["finished_at"],
            "subprocess_duration_seconds": summary["subprocess_duration_seconds"],
            "subprocess_returncode": rc,
            "status": summary["status"],
            "wrapper_invocation": cli_invocation,
            "subprocess_command": cmd,
            "git_ref_local": git_ref_local,
            "git_ref_declared": git_ref_declared,
            "git_ref_actual": git_ref_actual,
            "git_dirty_local": bool(git_dirty_local),
            "git_tracked_porcelain_local": git_tracked_porcelain_local,
            "git_porcelain_local": git_porcelain_local,
            "git_present_local": bool(git_present_local),
            "producer_script_sha256_local": producer_script_sha256_local,
            "trainer_script_sha256_local": trainer_script_sha256_local,
            "calibration_path": summary.get("calibration_path"),
            "calibration_sha256": summary.get("calibration_sha256"),
            "overwritten_existing": summary.get("overwritten_existing", False),
            "error": summary.get("error"),
            "params": {
                "num_bootstrap": int(num_bootstrap),
                "max_wall_hours": (
                    float(max_wall_hours) if max_wall_hours is not None else None
                ),
                "n_jobs": int(n_jobs),
                "resume": bool(resume),
                "overwrite": bool(overwrite),
            },
            "environment": env_banner,
        }
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(f"[runner] manifest: {manifest_path}", flush=True)
        except Exception as exc:
            # Last-ditch: print so it's at least in container logs.
            print(
                f"FATAL: failed to write run_manifest.json: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

        # --- Final disk usage report -----------------------------------
        try:
            usage = shutil.disk_usage(str(ARTIFACTS_ROOT))
            print(
                f"[runner] post-run disk[artifacts]: total={usage.total // (1 << 20)}MB "
                f"used={usage.used // (1 << 20)}MB free={usage.free // (1 << 20)}MB",
                flush=True,
            )
            summary["post_disk_artifacts_free_mb"] = usage.free // (1 << 20)
        except Exception:
            pass

        # --- vol.commit() with retry + escalation (FIX-4) --------------
        commit_succeeded = False
        last_commit_err: Optional[BaseException] = None
        for attempt in range(2):
            try:
                vol.commit()
                commit_succeeded = True
                print("[runner] vol.commit() OK", flush=True)
                break
            except Exception as exc:
                last_commit_err = exc
                if attempt == 0:
                    print(
                        f"[runner] WARNING: vol.commit() attempt {attempt + 1} failed: {exc!r}; retrying",
                        flush=True,
                    )
                    time.sleep(2)  # one quick retry
                    continue

        if not commit_succeeded:
            commit_msg = f"FATAL: vol.commit() failed after retry: {last_commit_err!r}"
            print(commit_msg, file=sys.stderr, flush=True)
            summary.setdefault("warnings", []).append(commit_msg)
            # If the subprocess succeeded but we can't durably persist its
            # work, escalate so the Modal job is marked failed. Re-write the
            # manifest with the warning and raise.
            if caught_exc is None and rc == 0:
                manifest["status"] = "vol_commit_failed"
                manifest.setdefault("warnings", []).append(commit_msg)
                try:
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
                    )
                except Exception:
                    pass
                # Defer the raise to outside the finally block so the
                # original-exception precedence is preserved when both fire.
                summary["status"] = "vol_commit_failed"

        summary["total_wall_seconds"] = round(time.time() - run_start, 2)

    # --- Post-finally re-raise / status escalation ---------------------
    if caught_exc is not None:
        # Re-raise the original exception so Modal marks the run failed and
        # the operator sees the true cause. SystemExit/KeyboardInterrupt are
        # both fine to re-raise here.
        raise caught_exc
    if summary.get("status") == "vol_commit_failed":
        raise SystemExit(
            f"vol.commit() failed after subprocess success; artifacts may not be persisted. "
            f"See warnings in {manifest_path}."
        )
    if rc != 0:
        # Raise so Modal marks the run as failed and `modal run` exits non-zero.
        raise SystemExit(
            f"StopDFF subprocess exited with code {rc}; see log at {log_path}"
        )

    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    experiment: str = "smoke",
    artifact_subdir: str = "",
    git_ref: str = "",  # informational override; if empty we capture locally
    gpu: str = "L40S",
    max_wall_hours: float = 1.0,
    num_bootstrap: int = 100,
    n_jobs: int = 1,
    resume: bool = False,
    smoke: bool = False,
    allow_dirty: bool = False,
    overwrite: bool = False,
    with_openai_key: bool = False,
) -> None:
    """Modal CLI entrypoint -- pre-flight host checks, then dispatch to the container.

    See the module docstring for usage examples. All flags are typed so Modal
    binds them as ``--<name>`` on the CLI (e.g., ``--max-wall-hours 6``).
    """
    # --- Validate experiment -------------------------------------------
    if experiment not in EXPERIMENTS:
        raise SystemExit(
            f"--experiment must be one of {EXPERIMENTS}; got {experiment!r}"
        )

    # --- Mutual-exclusion: --overwrite vs --resume (FIX-6) -------------
    if overwrite and resume:
        raise SystemExit("--overwrite and --resume are mutually exclusive.")

    # --- Host-side pre-flight: target script must exist (FIX-1) --------
    # Catch the absent-script class of failures before we pay for an image
    # build + container spin-up + Volume mount on Modal.
    if experiment == "dp_sweep":
        sweep_script = LOCAL_REPO_ROOT / "scripts" / "sweep_stopdff_dp.py"
        if not sweep_script.is_file():
            raise SystemExit(
                f"--experiment dp_sweep requires {sweep_script}, which is absent.\n"
                f"Ship sweep_stopdff_dp.py to the worktree first."
            )
    elif experiment in ("smoke", "single"):
        compute_script = LOCAL_REPO_ROOT / "scripts" / "compute_stopdff_dp.py"
        if not compute_script.is_file():
            raise SystemExit(
                f"--experiment {experiment} requires {compute_script}, which is absent."
            )
    elif experiment == "fair_qa":
        target = LOCAL_REPO_ROOT / "scripts" / "stopdff_fair_qa_retest.py"
        if not target.is_file():
            raise SystemExit(
                f"--experiment fair_qa requires {target}, which is absent."
            )
    elif experiment == "learned_value_train":
        target = LOCAL_REPO_ROOT / "scripts" / "train_stopdff_value_model.py"
        if not target.is_file():
            raise SystemExit(
                f"--experiment learned_value_train requires {target}, which is "
                f"absent."
            )
    elif experiment == "learned_value_eval":
        target = LOCAL_REPO_ROOT / "scripts" / "compute_stopdff_learned_value.py"
        if not target.is_file():
            raise SystemExit(
                f"--experiment learned_value_eval requires {target}, which is "
                f"absent."
            )

    # --- Capture host git state (L11, FIX-3, FIX-9) --------------------
    host_commit, tracked_porcelain, full_porcelain = _capture_local_git(LOCAL_REPO_ROOT)
    # Operator may override the recorded ref string (e.g., a tag name) for
    # provenance, but the dirty check still uses the live tracked porcelain.
    git_ref_declared = git_ref.strip() if git_ref else None
    git_ref_actual = host_commit
    git_ref_recorded = git_ref_declared if git_ref_declared else host_commit
    producer_script = _producer_script_path(experiment)
    producer_script_sha256 = _canonical_source_sha256(producer_script)
    committed_producer_sha256 = (
        _committed_file_sha256(host_commit, producer_script)
        if host_commit is not None
        else None
    )
    if committed_producer_sha256 != producer_script_sha256:
        raise SystemExit(
            "The selected producer is not exactly present in the current Git commit: "
            f"{producer_script.relative_to(LOCAL_REPO_ROOT)}. Commit the producer "
            "before dispatch so outputs cannot cite a parent that lacks their writer."
        )
    trainer_script = LOCAL_REPO_ROOT / "scripts" / "train_stopdff_value_model.py"
    trainer_script_sha256 = _canonical_source_sha256(trainer_script)
    committed_trainer_sha256 = (
        _committed_file_sha256(host_commit, trainer_script)
        if host_commit is not None
        else None
    )
    if (
        experiment in {"learned_value_train", "learned_value_eval"}
        and committed_trainer_sha256 != trainer_script_sha256
    ):
        raise SystemExit(
            "The learned-value trainer is not exactly present in the current Git "
            "commit. Commit the trainer before dispatch."
        )
    git_present_local = host_commit is not None or tracked_porcelain is not None

    # --- Enforce dirty-tree refusal (L12, FIX-3) -----------------------
    # Use the TRACKED-only porcelain for the gate: untracked files (e.g., a
    # fresh wrapper added but not yet committed) should not block dispatch.
    # The manifest still records the full porcelain for paper provenance.
    is_tracked_dirty = bool(tracked_porcelain and tracked_porcelain.strip())
    if is_tracked_dirty and not allow_dirty:
        raise SystemExit(
            "Repo working tree has uncommitted TRACKED changes; refusing to dispatch.\n"
            "Commit/stash your changes, or pass --allow-dirty to override.\n"
            f"git status --porcelain --untracked-files=no:\n{tracked_porcelain}"
        )

    # FIX-3 fallback (A10): if git is absent or this isn't a git checkout
    # (both porcelain outputs and commit are None), refuse unless allow_dirty
    # is also explicitly set. Provenance becomes unauditable otherwise.
    if not git_present_local and not allow_dirty:
        raise SystemExit(
            "Could not capture host git state (git not installed, or not a git checkout).\n"
            "Provenance cannot be stamped. Re-run with --allow-dirty to override "
            "(the manifest will record git_present_local=false)."
        )

    # --- Normalize GPU (L14) --------------------------------------------
    gpu_norm = _normalize_gpu(gpu)

    # --- Validate / cap max_wall_hours (L15) ----------------------------
    if max_wall_hours is None or max_wall_hours <= 0:
        raise SystemExit("--max-wall-hours must be > 0")
    if max_wall_hours * 3600 > MODAL_MAX_TIMEOUT_SECONDS / TIMEOUT_BUFFER:
        # User wants > ~20.87h of sweep budget; with the 15% buffer we'd
        # exceed Modal's 24h hard cap. Refuse with a helpful split suggestion.
        max_sweep_hours = (MODAL_MAX_TIMEOUT_SECONDS / TIMEOUT_BUFFER) / 3600
        raise SystemExit(
            f"--max-wall-hours {max_wall_hours} exceeds Modal's 24h container ceiling "
            f"after a {int((TIMEOUT_BUFFER - 1) * 100)}% setup/teardown buffer "
            f"(max allowed: ~{max_sweep_hours:.2f}h per invocation). "
            f"Split the sweep into multiple --resume runs."
        )
    container_timeout = min(
        int(max_wall_hours * 3600 * TIMEOUT_BUFFER),
        MODAL_MAX_TIMEOUT_SECONDS,
    )

    # --- FIX-8: warn when --max-wall-hours is moot for the chosen experiment.
    # Only dp_sweep honors a wall-clock budget internally; for smoke/single
    # the flag only sets the Modal container timeout, which can mislead
    # operators expecting in-process early termination.
    if experiment in ("smoke", "single") and max_wall_hours != 1.0:
        print(
            f"NOTE: --max-wall-hours only meaningfully bounds --experiment dp_sweep.\n"
            f"For --experiment {experiment} it only sets the Modal container timeout.\n"
            f"  Requested: {max_wall_hours}h -> container timeout = "
            f"{int(max_wall_hours * 3600 * TIMEOUT_BUFFER)}s",
            file=sys.stderr,
        )

    # --- Auto-generate artifact_subdir if not supplied -----------------
    subdir = artifact_subdir.strip()
    if not subdir:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        subdir = f"{experiment}_{ts}"

    # Validate subdir does not escape the volume root.
    if "/" in subdir or subdir.startswith(".") or subdir in {"", "."}:
        raise SystemExit(
            f"--artifact-subdir must be a single path segment (no '/'); got {subdir!r}"
        )

    # --- Build with_options kwargs (L1/L2/L15) -------------------------
    opts: dict = {
        "gpu": gpu_norm,
        "timeout": container_timeout,
    }
    if with_openai_key:
        # Opt-in only; the default DP path does not need it.
        opts["secrets"] = [modal.Secret.from_name("openai-key")]

    # Record the operator's literal sys.argv for the manifest -- pass through
    # to the remote so it lands in run_manifest.json verbatim.
    cli_invocation = list(sys.argv)

    # --- Banner --------------------------------------------------------
    print("=" * 60)
    print(f"[runner] {APP_NAME} -- dispatching to Modal")
    print(f"[runner] experiment:       {experiment}")
    print(f"[runner] artifact_subdir:  {subdir}")
    print(f"[runner] gpu:              {gpu_norm!r}")
    print(f"[runner] timeout (s):      {container_timeout}")
    print(f"[runner] max_wall_hours:   {max_wall_hours}")
    print(f"[runner] num_bootstrap:    {num_bootstrap}")
    print(f"[runner] n_jobs:           {n_jobs}")
    print(f"[runner] resume:           {resume}")
    print(f"[runner] smoke:            {smoke}")
    print(f"[runner] overwrite:        {overwrite}")
    print(f"[runner] with_openai_key:  {with_openai_key}")
    print(f"[runner] git_ref:          {git_ref_recorded}")
    print(f"[runner] git_ref_declared: {git_ref_declared!r}")
    print(f"[runner] git_ref_actual:   {git_ref_actual}")
    print(f"[runner] git_present:      {git_present_local}")
    print(f"[runner] git_dirty:        {is_tracked_dirty} (allow_dirty={allow_dirty})")
    print("=" * 60)

    # --- Dispatch ------------------------------------------------------
    result = run_stopdff.with_options(**opts).remote(
        experiment=experiment,
        artifact_subdir=subdir,
        num_bootstrap=int(num_bootstrap),
        max_wall_hours=float(max_wall_hours),
        n_jobs=int(n_jobs),
        resume=bool(resume),
        smoke=bool(smoke),
        overwrite=bool(overwrite),
        git_ref_local=git_ref_recorded,
        git_ref_declared=git_ref_declared,
        git_ref_actual=git_ref_actual,
        git_dirty_local=is_tracked_dirty,
        git_tracked_porcelain_local=tracked_porcelain,
        git_porcelain_local=full_porcelain,
        git_present_local=git_present_local,
        producer_script_sha256_local=producer_script_sha256,
        trainer_script_sha256_local=trainer_script_sha256,
        cli_invocation=cli_invocation,
    )

    # --- Completion banner ---------------------------------------------
    print("=" * 60)
    print(f"[runner] Modal run complete -- status: {result.get('status')!r}")
    print(f"[runner] subdir:    {result.get('artifact_subdir')}")
    print(f"[runner] manifest:  {result.get('manifest_path')}")
    print(f"[runner] log:       {result.get('log_path')}")
    print(f"[runner] wall (s):  {result.get('total_wall_seconds')}")
    print("=" * 60)
    print(
        f"Download artifacts with:\n"
        f"    modal volume ls {VOLUME_NAME} {subdir}\n"
        f"    modal volume get {VOLUME_NAME} {subdir} ./downloads/"
    )
