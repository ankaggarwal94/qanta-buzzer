"""Preflight checks for Device 2 CUDA StopDFF long runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


_GIB = 1024.0 ** 3
_SPLIT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class CommandResult:
    """Small command result shape used for subprocess failures and successes."""

    args: Sequence[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _make_check(
    name: str,
    *,
    ok: bool,
    details: dict[str, Any] | None = None,
    error: str | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    check: dict[str, Any] = {
        "name": name,
        "ok": ok,
        "skipped": skipped,
        "details": details or {},
    }
    if error:
        check["error"] = error
    return check


def _round_gib(value_bytes: int | float) -> float:
    return round(float(value_bytes) / _GIB, 3)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _command_text(args: Sequence[str]) -> str:
    return " ".join(str(arg) for arg in args)


def _run_command(args: Sequence[str], *, cwd: Path | None = None) -> CommandResult:
    command = list(args)
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        timeout = exc.timeout
        stderr = (
            f"command timed out after {timeout} seconds: {_command_text(command)}"
        )
        extra_stderr = _as_text(exc.stderr).strip()
        if extra_stderr:
            stderr = f"{stderr}; stderr: {extra_stderr}"
        return CommandResult(
            args=command,
            returncode=124,
            stdout=_as_text(exc.stdout),
            stderr=stderr,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            args=command,
            returncode=127,
            stderr=f"command not found: {_command_text(command)} ({exc})",
        )
    except OSError as exc:
        return CommandResult(
            args=command,
            returncode=1,
            stderr=f"command failed: {_command_text(command)} ({exc})",
        )

    return CommandResult(
        args=command,
        returncode=result.returncode,
        stdout=_as_text(result.stdout),
        stderr=_as_text(result.stderr),
    )


def check_python_version() -> dict[str, Any]:
    version = sys.version_info
    details = {
        "version": sys.version,
        "major": version.major,
        "minor": version.minor,
        "micro": version.micro,
    }
    if version < (3, 11):
        return _make_check(
            "python_version",
            ok=False,
            details=details,
            error=f"Python >= 3.11 is required; found {version.major}.{version.minor}",
        )
    return _make_check("python_version", ok=True, details=details)


def check_repo_state(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or _repo_root()
    commit = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    if commit.returncode != 0:
        return _make_check(
            "repo_state",
            ok=False,
            details={"repo_root": str(root)},
            error=commit.stderr.strip() or "failed to read git commit",
        )

    status = _run_command(["git", "status", "--porcelain"], cwd=root)
    if status.returncode != 0:
        return _make_check(
            "repo_state",
            ok=False,
            details={"repo_root": str(root), "commit": commit.stdout.strip()},
            error=status.stderr.strip() or "failed to read git dirty status",
        )

    dirty_entries = [line for line in status.stdout.splitlines() if line.strip()]
    return _make_check(
        "repo_state",
        ok=True,
        details={
            "repo_root": str(root),
            "commit": commit.stdout.strip(),
            "dirty": bool(dirty_entries),
            "dirty_entry_count": len(dirty_entries),
        },
    )


def _nearest_existing_parent(path: Path) -> Path:
    current = path if path.exists() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def check_disk_space(out_dir: Path) -> dict[str, Any]:
    target = _nearest_existing_parent(out_dir)
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return _make_check(
            "disk_space",
            ok=False,
            details={"path": str(target)},
            error=f"failed to read disk usage for {target}: {exc}",
        )

    return _make_check(
        "disk_space",
        ok=True,
        details={
            "path": str(target),
            "total_gib": _round_gib(usage.total),
            "used_gib": _round_gib(usage.used),
            "free_gib": _round_gib(usage.free),
        },
    )


def check_output_directory(
    *,
    out_dir: Path,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    details = {
        "out_dir": str(out_dir),
        "exists": out_dir.exists(),
        "is_directory": out_dir.is_dir(),
        "resume": resume,
        "overwrite": overwrite,
    }
    if out_dir.exists() and not out_dir.is_dir():
        return _make_check(
            "output_directory",
            ok=False,
            details=details,
            error=f"output path exists but is not a directory: {out_dir}",
        )
    if out_dir.exists() and not (resume or overwrite):
        return _make_check(
            "output_directory",
            ok=False,
            details=details,
            error=(
                f"output directory already exists: {out_dir}; "
                "use --resume or --overwrite"
            ),
        )
    return _make_check("output_directory", ok=True, details=details)


def _qid_set(dataset_path: Path) -> set[str]:
    payload = json.loads(dataset_path.read_text())
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        rows = payload["questions"]
    else:
        raise ValueError(
            f"{dataset_path} must contain a JSON list or a questions list"
        )

    qids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "qid" not in row:
            raise ValueError(f"{dataset_path} row {index} is missing qid")
        qids.add(str(row["qid"]))
    return qids


def _validate_split_name(split: str, label: str) -> str | None:
    if not _SPLIT_NAME_RE.fullmatch(split):
        return (
            f"invalid {label} split name {split!r}; use only letters, "
            "numbers, underscore, and hyphen"
        )
    return None


def check_split_separation(
    *,
    data_dir: Path,
    fit_split: str,
    eval_split: str,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "fit_split": fit_split,
        "eval_split": eval_split,
    }

    for label, split in (("fit", fit_split), ("eval", eval_split)):
        error = _validate_split_name(split, label)
        if error:
            return _make_check(
                "split_separation",
                ok=False,
                details=details,
                error=error,
            )

    if fit_split == eval_split:
        return _make_check(
            "split_separation",
            ok=False,
            details=details,
            error="fit and eval split must differ",
        )
    if fit_split.lower() == "test":
        return _make_check(
            "split_separation",
            ok=False,
            details=details,
            error="test split cannot be used for fitting calibration or continuation",
        )

    fit_path = data_dir / f"{fit_split}_dataset.json"
    eval_path = data_dir / f"{eval_split}_dataset.json"
    details.update({
        "fit_path": str(fit_path),
        "eval_path": str(eval_path),
    })

    try:
        fit_qids = _qid_set(fit_path)
        eval_qids = _qid_set(eval_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _make_check(
            "split_separation",
            ok=False,
            details=details,
            error=f"failed to read split qids: {exc}",
        )

    overlap = sorted(fit_qids & eval_qids)
    details.update(
        {
            "fit_qid_count": len(fit_qids),
            "eval_qid_count": len(eval_qids),
            "overlap_count": len(overlap),
            "overlap_sample": overlap[:20],
        }
    )
    if overlap:
        return _make_check(
            "split_separation",
            ok=False,
            details=details,
            error=(
                f"{fit_split}/{eval_split} qid overlap detected: "
                f"{len(overlap)} overlapping qids"
            ),
        )
    return _make_check("split_separation", ok=True, details=details)


def default_required_artifacts(
    *,
    data_dir: Path,
    calibration: Path,
    fit_split: str,
    eval_split: str,
) -> list[Path]:
    return [
        data_dir / "mc_dataset.json",
        data_dir / f"{fit_split}_dataset.json",
        data_dir / f"{eval_split}_dataset.json",
        calibration,
    ]


def check_required_artifacts(artifacts: Sequence[Path]) -> dict[str, Any]:
    records = [
        {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
        for path in artifacts
    ]
    missing = [record["path"] for record in records if not record["is_file"]]
    if missing:
        return _make_check(
            "required_artifacts",
            ok=False,
            details={"artifacts": records, "missing": missing},
            error="missing required artifact(s): " + ", ".join(missing),
        )
    return _make_check("required_artifacts", ok=True, details={"artifacts": records})


def check_nvidia_smi(
    *,
    device_index: int,
    skip_cuda_probe_for_tests: bool,
) -> dict[str, Any]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    index_interpretation = (
        "torch_logical"
        if cuda_visible_devices is not None
        else "nvidia_smi_physical_row"
    )
    if skip_cuda_probe_for_tests:
        return _make_check(
            "nvidia_smi",
            ok=True,
            skipped=True,
            details={
                "reason": "--skip-cuda-probe-for-tests",
                "cuda_visible_devices": cuda_visible_devices,
                "device_index_interpretation": index_interpretation,
            },
        )

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return _make_check(
            "nvidia_smi",
            ok=False,
            details={
                "cuda_visible_devices": cuda_visible_devices,
                "device_index_interpretation": index_interpretation,
            },
            error="nvidia-smi was not found on PATH",
        )

    result = _run_command([
        executable,
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if result.returncode != 0:
        return _make_check(
            "nvidia_smi",
            ok=False,
            details={
                "executable": executable,
                "stderr": result.stderr.strip(),
                "cuda_visible_devices": cuda_visible_devices,
                "device_index_interpretation": index_interpretation,
            },
            error="nvidia-smi failed to print GPU info",
        )

    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return _make_check(
            "nvidia_smi",
            ok=False,
            details={
                "executable": executable,
                "cuda_visible_devices": cuda_visible_devices,
                "device_index_interpretation": index_interpretation,
            },
            error="nvidia-smi printed no GPU rows",
        )
    if cuda_visible_devices is None and (
        device_index < 0 or device_index >= len(rows)
    ):
        return _make_check(
            "nvidia_smi",
            ok=False,
            details={
                "executable": executable,
                "gpu_rows": rows,
                "cuda_visible_devices": cuda_visible_devices,
                "device_index_interpretation": index_interpretation,
            },
            error=f"device index {device_index} is out of range for {len(rows)} GPU(s)",
        )

    details: dict[str, Any] = {
        "executable": executable,
        "gpu_rows": rows,
        "device_index": device_index,
        "cuda_visible_devices": cuda_visible_devices,
        "device_index_interpretation": index_interpretation,
    }
    selected_index: int | None
    if cuda_visible_devices is None:
        selected_index = device_index
        details["physical_row_mapping_resolved"] = True
    else:
        tokens = [
            token.strip()
            for token in cuda_visible_devices.split(",")
            if token.strip()
        ]
        details["cuda_visible_device_tokens"] = tokens
        selected_index = None
        mapping_note: str | None = None
        if device_index < 0 or device_index >= len(tokens):
            mapping_note = (
                "could not resolve physical nvidia-smi row: torch logical "
                f"device index {device_index} is outside CUDA_VISIBLE_DEVICES"
            )
        else:
            token = tokens[device_index]
            try:
                physical_index = int(token)
            except ValueError:
                mapping_note = (
                    "could not resolve physical nvidia-smi row: "
                    f"CUDA_VISIBLE_DEVICES token {token!r} is not an integer"
                )
            else:
                details["mapped_physical_device_index"] = physical_index
                if 0 <= physical_index < len(rows):
                    selected_index = physical_index
                    details["physical_row_mapping_resolved"] = True
                else:
                    mapping_note = (
                        "could not resolve physical nvidia-smi row: mapped "
                        f"physical index {physical_index} is outside "
                        f"{len(rows)} printed GPU row(s)"
                    )
        if selected_index is None:
            details["physical_row_mapping_resolved"] = False
            details["physical_row_mapping_note"] = mapping_note or (
                "could not resolve physical nvidia-smi row"
            )

    if selected_index is not None:
        selected = rows[selected_index]
        fields = [field.strip() for field in selected.split(",")]
        details.update(
            {
                "selected_nvidia_smi_row_index": selected_index,
                "selected_raw": selected,
            }
        )
        if len(fields) == 3:
            details.update(
                {
                    "device_name": fields[0],
                    "memory_total_mib": fields[1],
                    "memory_free_mib": fields[2],
                }
            )
    return _make_check("nvidia_smi", ok=True, details=details)


def check_torch_cuda(
    *,
    device_index: int,
    min_free_gib: float,
    skip_cuda_probe_for_tests: bool,
) -> dict[str, Any]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if skip_cuda_probe_for_tests:
        return _make_check(
            "torch_cuda",
            ok=True,
            skipped=True,
            details={
                "reason": "--skip-cuda-probe-for-tests",
                "cuda_visible_devices": cuda_visible_devices,
                "device_index_interpretation": "torch_logical",
            },
        )

    try:
        import torch
    except ImportError as exc:
        return _make_check("torch_cuda", ok=False, error=f"failed to import torch: {exc}")

    details: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": cuda_visible_devices,
        "device_index_interpretation": "torch_logical",
    }
    if not torch.cuda.is_available():
        return _make_check(
            "torch_cuda",
            ok=False,
            details=details,
            error="torch.cuda.is_available() is false",
        )

    device_count = torch.cuda.device_count()
    details["device_count"] = device_count
    if device_index < 0 or device_index >= device_count:
        return _make_check(
            "torch_cuda",
            ok=False,
            details=details,
            error=f"device index {device_index} is out of range for {device_count} CUDA device(s)",
        )

    props = torch.cuda.get_device_properties(device_index)
    with torch.cuda.device(device_index):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = _round_gib(free_bytes)
    total_gib = _round_gib(total_bytes)
    details.update(
        {
            "device_index": device_index,
            "device_name": props.name,
            "memory_total_gib": total_gib,
            "memory_free_gib": free_gib,
            "min_free_gib": min_free_gib,
        }
    )
    if free_gib < min_free_gib:
        return _make_check(
            "torch_cuda",
            ok=False,
            details=details,
            error=(
                f"CUDA free memory {free_gib} GiB is below required "
                f"{min_free_gib} GiB"
            ),
        )
    return _make_check("torch_cuda", ok=True, details=details)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CUDA, repo, data, and output state for Device 2 runs."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=0.0,
        help="Minimum free CUDA memory required on --device-index.",
    )
    parser.add_argument("--required-artifact", type=Path, action="append", default=[])
    parser.add_argument(
        "--skip-cuda-probe-for-tests",
        action="store_true",
        help="Only for tests: report CUDA checks as skipped instead of probing.",
    )
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo_root()
    artifacts = [
        *default_required_artifacts(
            data_dir=args.data_dir,
            calibration=args.calibration,
            fit_split=args.fit_split,
            eval_split=args.eval_split,
        ),
        *args.required_artifact,
    ]
    checks = [
        check_python_version(),
        check_repo_state(repo_root),
        check_disk_space(args.out_dir),
        check_output_directory(
            out_dir=args.out_dir,
            resume=args.resume,
            overwrite=args.overwrite,
        ),
        check_required_artifacts(artifacts),
        check_split_separation(
            data_dir=args.data_dir,
            fit_split=args.fit_split,
            eval_split=args.eval_split,
        ),
        check_nvidia_smi(
            device_index=args.device_index,
            skip_cuda_probe_for_tests=args.skip_cuda_probe_for_tests,
        ),
        check_torch_cuda(
            device_index=args.device_index,
            min_free_gib=args.min_free_gib,
            skip_cuda_probe_for_tests=args.skip_cuda_probe_for_tests,
        ),
    ]
    errors = [
        f"{check['name']}: {check.get('error', 'failed')}"
        for check in checks
        if not check.get("skipped") and not check["ok"]
    ]
    return {
        "ok": not errors,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "checks": checks,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    payload = json.dumps(report, indent=2, sort_keys=True)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")

    print(payload)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
