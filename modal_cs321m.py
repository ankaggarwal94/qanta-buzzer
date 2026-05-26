"""CS321M Final Project Modal compute orchestrator.

Thin wrapper dispatching existing qanta-buzzer and CS321M audit scripts on a
managed GPU. Dependent stages run in one remote invocation so file artifacts
written by early stages are visible to later stages.

Usage:
    python modal_cs321m.py --dry-run --config configs/cs321m_smoke.yaml
    modal run modal_cs321m.py -- --config configs/cs321m_smoke.yaml --smoke
    modal run modal_cs321m.py -- --config configs/cs321m_final.yaml
    modal run modal_cs321m.py -- --stages build_mc_dataset compute_csli
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Defer modal import so --dry-run works without modal installed.
# When invoked via `modal run`, modal is guaranteed available.
try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent
CONTAINER_ROOT = Path("/app")
DEFAULT_GPU_TYPE = "A100-80GB"
A100_RATE_PER_HOUR = 3.00  # Approximate Modal A100-80GB rate
DEFAULT_SPEND_LOG = Path("PROJECT_WIKI/TRANSCRIPTS/modal_spend.log")
DEFAULT_STAGES = [
    "build_mc_dataset",
    "run_baselines",
    "train_ppo",
    "evaluate_all",
    "compute_csli",
    "compute_prefix_calibration",
    "compute_stopdff",
]
STAGE_SCRIPTS = {
    "build_mc_dataset": "scripts/build_mc_dataset.py",
    "run_baselines": "scripts/run_baselines.py",
    "train_ppo": "scripts/train_ppo.py",
    "evaluate_all": "scripts/evaluate_all.py",
    "compute_csli": "scripts/compute_csli.py",
    "compute_prefix_calibration": "scripts/compute_prefix_calibration.py",
    "compute_stopdff": "scripts/compute_stopdff.py",
}
AUDIT_STAGES = {"compute_csli", "compute_prefix_calibration", "compute_stopdff"}
LEGACY_STAGES = set(STAGE_SCRIPTS) - AUDIT_STAGES


def _setup_modal():
    """Create Modal app and image. Only called when modal is available."""
    app = modal.App("cs321m-qanta-buzzer")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install_from_pyproject(str(REPO_ROOT / "pyproject.toml"))
        .pip_install("modal")
        .add_local_dir(str(REPO_ROOT), "/app")
    )
    return app, image


if _MODAL_AVAILABLE:
    app, image = _setup_modal()
else:
    app = None
    image = None


def _modal_gpu_config() -> Any:
    """Return a Modal GPU request for the configured A100-80GB lane."""
    if not _MODAL_AVAILABLE:
        return DEFAULT_GPU_TYPE
    try:
        return modal.gpu.A100(size="80GB")
    except Exception:
        return DEFAULT_GPU_TYPE


def load_cs321m_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config and return its dictionary contents."""
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml
    except ImportError:
        return _load_minimal_cs321m_config(path)
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must parse to a mapping: {path}")
    loaded["_config_validation"] = "full-yaml"
    return loaded


def _parse_scalar(value: str) -> Any:
    """Parse a small YAML scalar subset for dry-run metadata fallback."""
    value = value.strip().strip('"').strip("'")
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value


def _load_minimal_cs321m_config(path: Path) -> dict[str, Any]:
    """Parse only the cs321m metadata section when PyYAML is unavailable."""
    config: dict[str, Any] = {"cs321m": {}}
    in_cs321m = False
    list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            in_cs321m = line == "cs321m:"
            list_key = None
            continue
        if not in_cs321m:
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and list_key:
            config["cs321m"].setdefault(list_key, []).append(_parse_scalar(stripped[2:]))
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            if value.strip():
                config["cs321m"][key] = _parse_scalar(value)
                list_key = None
            else:
                config["cs321m"][key] = []
                list_key = key
    config["_config_validation"] = "metadata-only"
    return config


def cs321m_section(config: dict[str, Any]) -> dict[str, Any]:
    """Return the optional CS321M metadata section."""
    section = config.get("cs321m", {})
    return section if isinstance(section, dict) else {}


def resolve_gpu_type(config: dict[str, Any]) -> str:
    """Resolve the GPU label used consistently for Modal and spend logs."""
    return str(cs321m_section(config).get("gpu", DEFAULT_GPU_TYPE))


def resolve_budget(config: dict[str, Any]) -> float:
    """Resolve the configured spend budget in USD."""
    value = cs321m_section(config).get("budget_limit_usd", 300)
    return float(value)


def resolve_smoke_mode(config_path: str, config: dict[str, Any], explicit: bool) -> bool:
    """Infer smoke mode from CLI, config metadata, or config filename."""
    section = cs321m_section(config)
    return bool(explicit or section.get("smoke") or "smoke" in Path(config_path).stem)


def resolve_spend_log_path(config: dict[str, Any]) -> Path:
    """Resolve spend log path relative to the repo root."""
    raw = cs321m_section(config).get("spend_log_path", str(DEFAULT_SPEND_LOG))
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _path_for_container(path: str | Path | None) -> str | None:
    """Convert a repo-local path to its container equivalent."""
    if path is None:
        return None
    local_path = Path(path)
    if local_path.is_absolute():
        try:
            rel = local_path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            return str(local_path)
    else:
        rel = local_path
    return str(CONTAINER_ROOT / rel)


def _path_for_local(path: str | Path | None) -> Path | None:
    """Resolve a repo-local path for the local checkout."""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def data_dir_for_run(smoke: bool, output_dir: str | None) -> str:
    """Return the data directory the wrapper expects downstream stages to use."""
    if output_dir:
        return output_dir
    return "artifacts/smoke" if smoke else "data/processed"


def export_dir_for_run(output_dir: str | None) -> str:
    """Return the directory for CS321M metric JSON outputs."""
    if output_dir:
        return str(Path(output_dir) / "paper_exports")
    return "paper_exports"


def build_stage_command(
    stage: str,
    config_path: str,
    *,
    smoke: bool,
    output_dir: str | None = None,
    container: bool = False,
) -> list[str]:
    """Build the exact command for a supported pipeline stage."""
    if stage not in STAGE_SCRIPTS:
        raise ValueError(f"Unknown stage: {stage}")

    script = STAGE_SCRIPTS[stage]
    script_path = str(CONTAINER_ROOT / script) if container else script
    config_arg = _path_for_container(config_path) if container else config_path
    data_dir = data_dir_for_run(smoke, output_dir)
    export_dir = export_dir_for_run(output_dir)
    data_arg = _path_for_container(data_dir) if container else data_dir
    export_arg = _path_for_container(export_dir) if container else export_dir

    cmd = [sys.executable, script_path]
    if stage in LEGACY_STAGES:
        cmd.extend(["--config", str(config_arg)])
        if smoke:
            cmd.append("--smoke")
        if output_dir:
            cmd.extend(["--output-dir", str(_path_for_container(output_dir) if container else output_dir)])
        return cmd

    cmd.extend(["--data-dir", str(data_arg), "--smoke"] if smoke else ["--data-dir", str(data_arg)])
    if stage == "compute_csli":
        cmd.extend(["--output", str(Path(export_arg) / "csli.json")])
    elif stage == "compute_prefix_calibration":
        cmd.extend(["--output", str(Path(export_arg) / "calibration.json")])
    elif stage == "compute_stopdff":
        cmd.extend([
            "--output",
            str(Path(export_arg) / "stopdff.json"),
            "--report-output",
            str(Path(export_arg) / "stopdff_report.json"),
            "--calibration",
            str(Path(export_arg) / "calibration.json"),
        ])
    return cmd


def validate_stage_plan(stages: list[str], config_path: str, smoke: bool, output_dir: str | None) -> list[list[str]]:
    """Validate stage scripts and return local commands for dry-run display."""
    commands = []
    for stage in stages:
        script = REPO_ROOT / STAGE_SCRIPTS[stage]
        if not script.exists():
            raise FileNotFoundError(f"Stage script not found: {script}")
        commands.append(build_stage_command(stage, config_path, smoke=smoke, output_dir=output_dir))
    spend_path = resolve_spend_log_path(load_cs321m_config(config_path))
    spend_path.parent.mkdir(parents=True, exist_ok=True)
    return commands


def parse_existing_spend(log_path: Path) -> float:
    """Parse cumulative estimated cost from an existing spend log."""
    if not log_path.exists():
        return 0.0
    total = 0.0
    pattern = re.compile(r"estimated_cost:\s*\$([0-9]+(?:\.[0-9]+)?)")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            total += float(match.group(1))
    return total


def estimate_cost(duration_seconds: float) -> float:
    """Estimate Modal cost from duration using the configured A100-80GB rate."""
    return duration_seconds / 3600 * A100_RATE_PER_HOUR


def spend_tracker(
    stage_name: str,
    duration_seconds: float,
    *,
    gpu_type: str,
    log_path: Path,
    status: str = "success",
) -> None:
    """Append a spend log entry."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    estimate = estimate_cost(duration_seconds)
    entry = (
        f"{timestamp} | {stage_name} | {duration_seconds:.1f}s "
        f"| {gpu_type} | status: {status} | estimated_cost: ${estimate:.4f}\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[spend] {entry.strip()}")


def _modal_function_decorator(func):
    """Apply @app.function decorator only when modal is available."""
    if _MODAL_AVAILABLE and app is not None:
        return app.function(gpu=_modal_gpu_config(), timeout=3600, image=image)(func)
    return func


@_modal_function_decorator
def run_pipeline(
    stages: list[str],
    config_path: str,
    output_dir: str | None,
    smoke: bool,
    budget_limit_usd: float,
    initial_spend_usd: float,
) -> dict:
    """Run all requested stages sequentially inside one Modal container."""
    results = []
    cumulative_estimated_spend = initial_spend_usd
    for stage in stages:
        if cumulative_estimated_spend >= budget_limit_usd:
            results.append({
                "stage": stage,
                "exit_code": 99,
                "duration_seconds": 0.0,
                "status": "skipped_budget_exceeded",
            })
            break

        cmd = build_stage_command(
            stage,
            config_path,
            smoke=smoke,
            output_dir=output_dir,
            container=True,
        )
        start = time.time()
        result = subprocess.run(cmd, cwd=str(CONTAINER_ROOT), capture_output=True, text=True)
        duration = round(time.time() - start, 2)
        cumulative_estimated_spend += estimate_cost(duration)

        if result.stdout:
            print(result.stdout[-2000:])
        if result.returncode != 0 and result.stderr:
            print(f"STDERR: {result.stderr[-2000:]}", file=sys.stderr)

        results.append({
            "stage": stage,
            "command": cmd,
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "status": "success" if result.returncode == 0 else "failed",
        })
        if result.returncode != 0:
            break

    return {"results": results, "artifacts": collect_small_artifacts(output_dir, smoke)}


def collect_small_artifacts(output_dir: str | None, smoke: bool) -> dict[str, dict[str, str]]:
    """Collect small JSON/text outputs so Modal returns them to the local process."""
    export_dir = Path(_path_for_container(export_dir_for_run(output_dir)) or "/app/paper_exports")
    default_artifact_dir = "/app/artifacts/smoke" if smoke else "/app/artifacts/main"
    artifact_root = Path(_path_for_container(output_dir) or default_artifact_dir)
    candidates = []
    if export_dir.exists():
        candidates.extend(path for path in export_dir.rglob("*") if path.is_file())
    if artifact_root.exists():
        candidates.extend(
            path
            for path in artifact_root.rglob("*")
            if path.is_file()
            and path.name in {
                "baseline_summary.json",
                "ppo_summary.json",
                "evaluation_report.json",
                "evaluation_plots.png",
            }
        )

    artifacts: dict[str, dict[str, str]] = {}
    for path in candidates:
        if path.exists() and path.stat().st_size <= 1_000_000:
            try:
                rel = path.relative_to(CONTAINER_ROOT).as_posix()
            except ValueError:
                rel = str(path)
            data = path.read_bytes()
            try:
                artifacts[rel] = {"encoding": "text", "content": data.decode("utf-8")}
            except UnicodeDecodeError:
                artifacts[rel] = {
                    "encoding": "base64",
                    "content": base64.b64encode(data).decode("ascii"),
                }
    return artifacts


def write_returned_artifacts(artifacts: dict[str, dict[str, str]]) -> None:
    """Write small returned Modal artifacts into the local repository."""
    for rel_path, artifact in artifacts.items():
        if Path(rel_path).is_absolute():
            continue
        local_path = REPO_ROOT / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact.get("encoding") == "base64":
            local_path.write_bytes(base64.b64decode(artifact["content"]))
        else:
            local_path.write_text(artifact.get("content", ""), encoding="utf-8")
        print(f"[artifact] wrote {local_path.relative_to(REPO_ROOT)}")


def _main_impl():
    """CLI entrypoint for Modal dispatch of CS321M pipeline stages."""
    parser = argparse.ArgumentParser(
        description="CS321M Modal A100-80GB pipeline orchestrator"
    )
    parser.add_argument(
        "--config", default="configs/cs321m_smoke.yaml",
        help="Path to YAML config (default: cs321m_smoke.yaml)",
    )
    parser.add_argument(
        "--stages", nargs="+", default=None, choices=sorted(STAGE_SCRIPTS),
        help="Pipeline stages to run (default: config cs321m.stages or all stages)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Force smoke-mode routing and forward --smoke to stage scripts.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and print commands without launching Modal",
    )
    parser.add_argument(
        "--metadata-only-dry-run", action="store_true",
        help="Allow dry-run when PyYAML is unavailable; validates only cs321m metadata and scripts.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for all stages",
    )
    args = parser.parse_args()

    try:
        config = load_cs321m_config(args.config)
        stages = args.stages or list(cs321m_section(config).get("stages", DEFAULT_STAGES))
        unknown = [stage for stage in stages if stage not in STAGE_SCRIPTS]
        if unknown:
            raise ValueError(f"Unsupported stages in config/CLI: {unknown}")
        validation_mode = str(config.get("_config_validation", "unknown"))
        if (
            args.dry_run
            and validation_mode == "metadata-only"
            and not args.metadata_only_dry_run
        ):
            raise RuntimeError(
                "PyYAML is unavailable, so dry-run can only validate cs321m metadata. "
                "Install PyYAML/use the repo environment, or pass --metadata-only-dry-run "
                "to accept this weaker check."
            )
        smoke = resolve_smoke_mode(args.config, config, args.smoke)
        gpu_type = resolve_gpu_type(config)
        budget_limit = resolve_budget(config)
        spend_log_path = resolve_spend_log_path(config)
        commands = validate_stage_plan(stages, args.config, smoke, args.output_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        print("=" * 60)
        print("CS321M Modal Pipeline -- DRY RUN")
        print("=" * 60)
        print(f"Config:     {args.config}")
        print(f"Stages:     {', '.join(stages)}")
        print(f"Smoke:      {smoke}")
        print(f"GPU:        {gpu_type}")
        print(f"Output dir: {args.output_dir or '(default)'}")
        print(f"Data dir:   {data_dir_for_run(smoke, args.output_dir)}")
        print(f"Exports:    {export_dir_for_run(args.output_dir)}")
        print(f"Spend log:  {spend_log_path.relative_to(REPO_ROOT) if spend_log_path.is_relative_to(REPO_ROOT) else spend_log_path}")
        print(f"Budget:     ${budget_limit:.2f} max")
        print(f"Validation: {validation_mode}")
        print(f"Modal SDK:  {'available' if _MODAL_AVAILABLE else 'NOT INSTALLED'}")
        print("Commands:")
        for cmd in commands:
            print(f"  {' '.join(cmd)}")
        print("=" * 60)
        print("No Modal functions launched. Pass without --dry-run to execute.")
        return

    if not _MODAL_AVAILABLE:
        print("ERROR: modal package not installed. Install with: pip install modal",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("CS321M Modal Pipeline -- EXECUTING")
    print(f"Config: {args.config} | Stages: {', '.join(stages)} | GPU: {gpu_type}")
    print("=" * 60)

    initial_spend = parse_existing_spend(spend_log_path)
    if initial_spend >= budget_limit:
        print(
            f"ERROR: existing estimated spend ${initial_spend:.2f} exceeds "
            f"configured budget ${budget_limit:.2f}",
            file=sys.stderr,
        )
        sys.exit(3)

    total_start = time.time()

    try:
        remote_result = run_pipeline.remote(
            stages,
            args.config,
            args.output_dir,
            smoke,
            budget_limit,
            initial_spend,
        )
    except Exception:
        duration = time.time() - total_start
        spend_tracker(
            "pipeline_launch",
            duration,
            gpu_type=gpu_type,
            log_path=spend_log_path,
            status="failed",
        )
        raise

    results = remote_result.get("results", [])
    for result in results:
        spend_tracker(
            result["stage"],
            float(result["duration_seconds"]),
            gpu_type=gpu_type,
            log_path=spend_log_path,
            status=str(result.get("status", "unknown")),
        )
    write_returned_artifacts(remote_result.get("artifacts", {}))

    total_duration = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total_duration:.1f}s")
    print(f"Results: {json.dumps(results, indent=2)}")
    if any(result["exit_code"] != 0 for result in results):
        sys.exit(4)


# When modal is available, register as local_entrypoint for `modal run`
if _MODAL_AVAILABLE and app is not None:
    main = app.local_entrypoint()(_main_impl)
else:
    main = _main_impl


if __name__ == "__main__":
    _main_impl()
