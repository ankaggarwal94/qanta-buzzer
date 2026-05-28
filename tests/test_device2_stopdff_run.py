"""Focused tests for the Device 2 CUDA StopDFF preflight."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gitattributes_pins_text_outputs_to_lf() -> None:
    attrs = REPO_ROOT / ".gitattributes"
    lines = {
        line.strip()
        for line in attrs.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".gitattributes text eol=lf" in lines
    assert "threshold_manifest.json text eol=lf" in lines
    assert "threshold_manifest.json.sha256 text eol=lf" in lines
    assert "scripts/device2_stopdff_run.sh text eol=lf" in lines
    assert "scripts/device2_cuda_preflight.py text eol=lf" in lines
    assert "docs/device2_stopdff_runbook.md text eol=lf" in lines
    assert "tests/test_device2_stopdff_run.py text eol=lf" in lines
    assert (
        "docs/superpowers/plans/2026-05-27-device2-cross-platform-hardening.md "
        "text eol=lf"
    ) in lines
    assert (
        "docs/superpowers/plans/2026-05-27-device2-stopdff-run.md text eol=lf"
    ) in lines
    assert not any(line.startswith("*.") for line in lines)


def test_runbook_documents_windows_lf_checkout_setup() -> None:
    runbook = (REPO_ROOT / "docs" / "device2_stopdff_runbook.md").read_text()

    assert "git config --local core.autocrlf false" in runbook
    assert "git config --local core.eol lf" in runbook
    assert "git ls-files --eol" in runbook
    assert "threshold_manifest.json.sha256" in runbook


def test_device2_hardening_files_are_lf_only() -> None:
    paths = [
        REPO_ROOT / ".gitattributes",
        REPO_ROOT / "scripts" / "device2_stopdff_run.sh",
        REPO_ROOT / "scripts" / "device2_cuda_preflight.py",
        REPO_ROOT / "docs" / "device2_stopdff_runbook.md",
        REPO_ROOT / "tests" / "test_device2_stopdff_run.py",
        REPO_ROOT / "threshold_manifest.json",
        REPO_ROOT / "threshold_manifest.json.sha256",
    ]

    for path in paths:
        assert b"\r\n" not in path.read_bytes(), f"{path} contains CRLF line endings"


def _write_dataset(path: Path, qids: list[str]) -> None:
    path.write_text(json.dumps([{"qid": qid} for qid in qids]))


def _write_wrapped_dataset(path: Path, qids: list[str]) -> None:
    path.write_text(json.dumps({
        "metadata": {"source": "unit-test"},
        "questions": [{"qid": qid} for qid in qids],
    }))


def _write_preflight_inputs(
    tmp_path: Path,
    *,
    val_qids: list[str] | None = None,
    test_qids: list[str] | None = None,
) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val = ["v1", "v2"] if val_qids is None else val_qids
    test = ["t1", "t2"] if test_qids is None else test_qids
    _write_dataset(data_dir / "val_dataset.json", val)
    _write_dataset(data_dir / "test_dataset.json", test)
    _write_dataset(data_dir / "mc_dataset.json", [*val, *test])
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    return data_dir, calibration


def test_check_output_directory_rejects_existing_without_resume_or_overwrite(
    tmp_path: Path,
) -> None:
    from scripts import device2_cuda_preflight

    out_dir = tmp_path / "existing"
    out_dir.mkdir()

    rejected = device2_cuda_preflight.check_output_directory(
        out_dir=out_dir,
        resume=False,
        overwrite=False,
    )
    assert rejected["ok"] is False
    assert "already exists" in rejected["error"]

    resumed = device2_cuda_preflight.check_output_directory(
        out_dir=out_dir,
        resume=True,
        overwrite=False,
    )
    overwritten = device2_cuda_preflight.check_output_directory(
        out_dir=out_dir,
        resume=False,
        overwrite=True,
    )
    assert resumed["ok"] is True
    assert overwritten["ok"] is True


def test_check_output_directory_rejects_existing_file_even_with_resume_or_overwrite(
    tmp_path: Path,
) -> None:
    from scripts import device2_cuda_preflight

    out_dir = tmp_path / "not_a_directory"
    out_dir.write_text("existing file")

    resumed = device2_cuda_preflight.check_output_directory(
        out_dir=out_dir,
        resume=True,
        overwrite=False,
    )
    overwritten = device2_cuda_preflight.check_output_directory(
        out_dir=out_dir,
        resume=False,
        overwrite=True,
    )

    assert resumed["ok"] is False
    assert overwritten["ok"] is False
    assert "not a directory" in resumed["error"]
    assert "not a directory" in overwritten["error"]


def test_check_split_separation_detects_qid_overlap(tmp_path: Path) -> None:
    from scripts import device2_cuda_preflight

    data_dir, _calibration = _write_preflight_inputs(
        tmp_path,
        val_qids=["shared", "v2"],
        test_qids=["shared", "t2"],
    )

    result = device2_cuda_preflight.check_split_separation(
        data_dir=data_dir,
        fit_split="val",
        eval_split="test",
    )

    assert result["ok"] is False
    assert result["details"]["overlap_count"] == 1
    assert result["details"]["overlap_sample"] == ["shared"]


def test_check_split_separation_rejects_test_as_fit_split(tmp_path: Path) -> None:
    from scripts import device2_cuda_preflight

    data_dir, _calibration = _write_preflight_inputs(tmp_path)

    result = device2_cuda_preflight.check_split_separation(
        data_dir=data_dir,
        fit_split="test",
        eval_split="val",
    )

    assert result["ok"] is False
    assert "test split cannot be used for fitting" in result["error"]


def test_check_split_separation_rejects_path_fragment_fit_split(
    tmp_path: Path,
) -> None:
    from scripts import device2_cuda_preflight

    data_dir, _calibration = _write_preflight_inputs(tmp_path)

    result = device2_cuda_preflight.check_split_separation(
        data_dir=data_dir,
        fit_split="./test",
        eval_split="val",
    )

    assert result["ok"] is False
    assert "invalid fit split name" in result["error"]


def test_check_split_separation_accepts_wrapped_question_datasets(
    tmp_path: Path,
) -> None:
    from scripts import device2_cuda_preflight

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_wrapped_dataset(data_dir / "val_dataset.json", ["v1", "v2"])
    _write_wrapped_dataset(data_dir / "test_dataset.json", ["t1", "t2"])

    result = device2_cuda_preflight.check_split_separation(
        data_dir=data_dir,
        fit_split="val",
        eval_split="test",
    )

    assert result["ok"] is True
    assert result["details"]["fit_qid_count"] == 2
    assert result["details"]["eval_qid_count"] == 2
    assert result["details"]["overlap_count"] == 0


def test_preflight_qid_loader_reads_json_as_utf8(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import device2_cuda_preflight

    dataset = tmp_path / "val_dataset.json"
    dataset.write_text(json.dumps([{"qid": "q1"}]), encoding="utf-8")
    calls: list[tuple[Path, dict]] = []
    original_read_text = Path.read_text

    def tracking_read_text(self: Path, *args, **kwargs) -> str:
        calls.append((self, dict(kwargs)))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracking_read_text)

    assert device2_cuda_preflight._qid_set(dataset) == {"q1"}
    assert calls == [(dataset, {"encoding": "utf-8"})]


def test_main_writes_json_report_and_returns_1_when_artifact_missing(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts import device2_cuda_preflight

    data_dir, calibration = _write_preflight_inputs(tmp_path)
    missing = tmp_path / "missing.json"
    output_json = tmp_path / "preflight.json"

    rc = device2_cuda_preflight.main([
        "--out-dir", str(tmp_path / "run"),
        "--data-dir", str(data_dir),
        "--calibration", str(calibration),
        "--required-artifact", str(missing),
        "--output-json", str(output_json),
        "--skip-cuda-probe-for-tests",
    ])

    assert rc == 1
    report = json.loads(output_json.read_text())
    assert report["ok"] is False
    assert any(str(missing) in error for error in report["errors"])
    stdout_report = json.loads(capsys.readouterr().out)
    assert stdout_report == report


def test_main_writes_json_report_when_command_probe_times_out(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    from scripts import device2_cuda_preflight

    data_dir, calibration = _write_preflight_inputs(tmp_path)
    output_json = tmp_path / "preflight.json"

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=20)

    monkeypatch.setattr(device2_cuda_preflight.subprocess, "run", raise_timeout)

    rc = device2_cuda_preflight.main([
        "--out-dir", str(tmp_path / "run"),
        "--data-dir", str(data_dir),
        "--calibration", str(calibration),
        "--output-json", str(output_json),
        "--skip-cuda-probe-for-tests",
    ])

    assert rc == 1
    report = json.loads(output_json.read_text())
    assert report["ok"] is False
    assert any("timed out" in error for error in report["errors"])
    stdout_report = json.loads(capsys.readouterr().out)
    assert stdout_report == report


def test_nvidia_smi_maps_torch_logical_index_to_visible_physical_row(
    monkeypatch,
) -> None:
    from scripts import device2_cuda_preflight

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(device2_cuda_preflight.shutil, "which", lambda _name: "nvidia-smi")

    def fake_run_command(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout=(
                "GPU 0, 40960, 32768\n"
                "GPU 1, 40960, 32768\n"
                "GPU 2, 40960, 32768\n"
                "GPU 3, 40960, 32768\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(device2_cuda_preflight, "_run_command", fake_run_command)

    result = device2_cuda_preflight.check_nvidia_smi(
        device_index=1,
        skip_cuda_probe_for_tests=False,
    )

    assert result["ok"] is True
    assert result["details"]["cuda_visible_devices"] == "2,3"
    assert result["details"]["device_index_interpretation"] == "torch_logical"
    assert result["details"]["selected_nvidia_smi_row_index"] == 3
    assert result["details"]["selected_raw"].startswith("GPU 3")


def test_nvidia_smi_does_not_pretend_unresolved_visible_mapping_is_row_zero(
    monkeypatch,
) -> None:
    from scripts import device2_cuda_preflight

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")
    monkeypatch.setattr(device2_cuda_preflight.shutil, "which", lambda _name: "nvidia-smi")

    def fake_run_command(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="GPU 0, 40960, 32768\n",
            stderr="",
        )

    monkeypatch.setattr(device2_cuda_preflight, "_run_command", fake_run_command)

    result = device2_cuda_preflight.check_nvidia_smi(
        device_index=0,
        skip_cuda_probe_for_tests=False,
    )

    assert result["ok"] is True
    assert "selected_nvidia_smi_row_index" not in result["details"]
    assert "selected_raw" not in result["details"]
    assert result["details"]["physical_row_mapping_resolved"] is False
    assert "could not resolve" in result["details"]["physical_row_mapping_note"]


def test_main_returns_0_for_synthetic_clean_preflight(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    from scripts import device2_cuda_preflight

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    data_dir, calibration = _write_preflight_inputs(tmp_path)
    output_json = tmp_path / "preflight.json"

    rc = device2_cuda_preflight.main([
        "--out-dir", str(tmp_path / "run"),
        "--data-dir", str(data_dir),
        "--calibration", str(calibration),
        "--output-json", str(output_json),
        "--skip-cuda-probe-for-tests",
    ])

    assert rc == 0
    report = json.loads(output_json.read_text())
    assert report["ok"] is True
    assert report["errors"] == []
    assert {check["name"] for check in report["checks"]} >= {
        "nvidia_smi",
        "torch_cuda",
        "required_artifacts",
        "split_separation",
        "output_directory",
    }
    skipped = {check["name"] for check in report["checks"] if check.get("skipped")}
    assert {"nvidia_smi", "torch_cuda"}.issubset(skipped)
    cuda_checks = {
        check["name"]: check for check in report["checks"]
        if check["name"] in {"nvidia_smi", "torch_cuda"}
    }
    assert cuda_checks["nvidia_smi"]["details"]["cuda_visible_devices"] == "0"
    assert cuda_checks["torch_cuda"]["details"]["cuda_visible_devices"] == "0"
    stdout_report = json.loads(capsys.readouterr().out)
    assert stdout_report == report


def _bash() -> str:
    candidates = [
        "C:/Program Files/Git/bin/bash.exe",
        "C:/Program Files/Git/usr/bin/bash.exe",
        "C:/Program Files (x86)/Git/bin/bash.exe",
        "C:/msys64/usr/bin/bash.exe",
        shutil.which("bash"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    import pytest

    pytest.skip("bash is required for wrapper tests")


def test_bash_helper_prefers_git_bash_before_path_bash(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "C:/Windows/System32/bash.exe")

    def fake_exists(path: Path) -> bool:
        return path.as_posix() in {
            "C:/Program Files/Git/bin/bash.exe",
            "C:/Windows/System32/bash.exe",
        }

    monkeypatch.setattr(Path, "exists", fake_exists)

    assert _bash() == "C:/Program Files/Git/bin/bash.exe"


def _write_python_stub(repo: Path) -> Path:
    python_path = repo / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${STUB_LOG_DIR:?}"
mkdir -p "$LOG_DIR"
cmd="${1:-}"
printf '%s\n' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}" >> "$LOG_DIR/env.log"
printf '%s\n' "$*" >> "$LOG_DIR/args.log"

if [[ "$cmd" == "scripts/device2_cuda_preflight.py" ]]; then
  out_json=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output-json)
        out_json="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  mkdir -p "$(dirname "$out_json")"
  printf '{"ok": true, "errors": []}\n' > "$out_json"
  printf 'preflight stdout\n'
  printf 'preflight stderr\n' >&2
  exit 0
fi

if [[ "$cmd" == "scripts/sweep_stopdff_dp.py" ]]; then
  artifact_dir=""
  out_path=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --artifact-dir)
        artifact_dir="$2"
        shift 2
        ;;
      --out)
        out_path="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  mkdir -p "$artifact_dir/figures"
  printf '{"sweep": true}\n' > "$artifact_dir/stopdff_dp_sweep.json"
  printf '\\begin{tabular}{}\\end{tabular}\n' > "$artifact_dir/stopdff_dp_sweep_table.tex"
  printf 'png\n' > "$artifact_dir/figures/stopdff_dp_phase_diagram.png"
  if [[ -n "$out_path" && "$out_path" != "$artifact_dir/stopdff_dp_sweep.json" ]]; then
    mkdir -p "$(dirname "$out_path")"
    printf '{"sweep": true}\n' > "$out_path"
  fi
  printf 'sweep stdout\n'
  printf 'sweep stderr\n' >&2
  exit 0
fi

printf 'unexpected python invocation: %s\n' "$*" >&2
exit 99
"""
    )
    python_path.chmod(0o755)
    return python_path


def _copy_harness(temp_repo: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "scripts" / "device2_stopdff_run.sh"
    assert source.exists(), "scripts/device2_stopdff_run.sh must exist"
    target = temp_repo / "scripts" / "device2_stopdff_run.sh"
    target.parent.mkdir(parents=True)
    target.write_text(source.read_text())
    target.chmod(0o755)
    return target


def _run_harness(temp_repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    harness = _copy_harness(temp_repo)
    logs = temp_repo / "stub_logs"
    logs.mkdir()
    env = {
        **os.environ,
        "STUB_LOG_DIR": logs.as_posix(),
        "PATH": f"{temp_repo / '.venv' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    return subprocess.run(
        [_bash(), harness.as_posix(), *args],
        cwd=temp_repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_device2_stopdff_harness_creates_run_artifacts_with_explicit_out_dir(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_python_stub(repo)
    out_dir = repo / "run"

    result = _run_harness(
        repo,
        "--out-dir",
        str(out_dir),
        "--data-dir",
        str(repo / "data"),
        "--calibration",
        str(repo / "calibration.json"),
        "--max-wall-hours",
        "1.5",
        "--num-bootstrap",
        "7",
        "--n-jobs",
        "3",
        "--calibrators",
        "uncalibrated,temperature",
        "--reward-schedules",
        "acf_flat,strict_wrong",
        "--continuations",
        "empirical_bucket,oracle_trajectory",
        "--fit-split",
        "val",
        "--eval-split",
        "test",
    )

    assert result.returncode == 0, result.stderr
    artifact_dir = out_dir / "paper_exports"
    assert (out_dir / "stdout.log").exists()
    assert (out_dir / "stderr.log").exists()
    assert json.loads((out_dir / "preflight.json").read_text())["ok"] is True
    manifest = json.loads((out_dir / "command_manifest.json").read_text())
    assert manifest["run_dir"] == str(out_dir)
    assert manifest["artifact_path"] == str(artifact_dir)
    assert manifest["parsed_axes"] == {
        "calibrators": "uncalibrated,temperature",
        "reward_schedules": "acf_flat,strict_wrong",
        "continuations": "empirical_bucket,oracle_trajectory",
    }
    assert manifest["CUDA_VISIBLE_DEVICES"] == "0"
    assert (artifact_dir / "stopdff_dp_sweep.json").exists()
    assert "preflight stdout" in (out_dir / "stdout.log").read_text()
    assert "sweep stdout" in (out_dir / "stdout.log").read_text()
    assert "preflight stderr" in (out_dir / "stderr.log").read_text()
    assert "sweep stderr" in (out_dir / "stderr.log").read_text()


def test_device2_stopdff_harness_passes_resume_cuda_and_smoke_args(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_python_stub(repo)
    out_dir = repo / "run"

    result = _run_harness(
        repo,
        "--out-dir",
        str(out_dir),
        "--resume",
        "--smoke",
        "--data-dir",
        str(repo / "data"),
        "--calibration",
        str(repo / "calibration.json"),
    )

    assert result.returncode == 0, result.stderr
    args_log = (repo / "stub_logs" / "args.log").read_text().splitlines()
    preflight_args = next(line for line in args_log if "device2_cuda_preflight.py" in line)
    sweep_args = next(line for line in args_log if "sweep_stopdff_dp.py" in line)
    assert "--resume" in preflight_args
    assert "--resume" in sweep_args
    assert "--smoke" in sweep_args
    assert "--max-cells 2" in sweep_args
    env_log = (repo / "stub_logs" / "env.log").read_text().splitlines()
    assert env_log == ["CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=0"]


def test_device2_stopdff_harness_masks_requested_device_as_logical_zero(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_python_stub(repo)
    out_dir = repo / "run"

    result = _run_harness(
        repo,
        "--out-dir",
        str(out_dir),
        "--device-index",
        "2",
        "--data-dir",
        str(repo / "data"),
        "--calibration",
        str(repo / "calibration.json"),
    )

    assert result.returncode == 0, result.stderr
    env_log = (repo / "stub_logs" / "env.log").read_text().splitlines()
    assert env_log == ["CUDA_VISIBLE_DEVICES=2", "CUDA_VISIBLE_DEVICES=2"]

    args_log = (repo / "stub_logs" / "args.log").read_text().splitlines()
    preflight_args = next(line for line in args_log if "device2_cuda_preflight.py" in line)
    preflight_tokens = preflight_args.split()
    device_index_pos = preflight_tokens.index("--device-index")
    assert preflight_tokens[device_index_pos + 1] == "0"

    manifest = json.loads((out_dir / "command_manifest.json").read_text())
    assert manifest["CUDA_VISIBLE_DEVICES"] == "2"


def test_device2_stopdff_harness_rejects_unknown_experiment(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_python_stub(repo)

    result = _run_harness(
        repo,
        "--experiment",
        "other",
        "--out-dir",
        str(repo / "run"),
        "--data-dir",
        str(repo / "data"),
        "--calibration",
        str(repo / "calibration.json"),
    )

    assert result.returncode != 0
    assert "unsupported experiment" in result.stderr
    assert not (repo / "run").exists()


def test_device2_stopdff_harness_accepts_learned_value_experiments() -> None:
    """Source-level check: the new experiment cases are wired in.

    Avoids extending the bash stub harness (which would need to mock
    Prompt 5's not-yet-landed scripts). The dispatch is exercised at
    runtime by ``test_device2_stopdff_harness_dispatches_learned_value_train``
    below using a focused stub extension.
    """
    source = (REPO_ROOT / "scripts" / "device2_stopdff_run.sh").read_text()
    # Validator (rejects everything not in this list).
    assert "dp_sweep|learned_value_train|learned_value_eval" in source
    # Dispatch case statements (one per learned-value mode).
    assert "learned_value_train)" in source
    assert "learned_value_eval)" in source
    # Each branch dispatches the canonical Prompt 5 script path.
    assert "scripts/train_stopdff_value_model.py" in source
    assert "scripts/compute_stopdff_learned_value.py" in source


def _write_python_stub_with_learned_value(repo: Path) -> Path:
    """Variant of ``_write_python_stub`` that also handles Prompt 5 scripts.

    Returns 0 + tee'd args.log entries for
    ``train_stopdff_value_model.py`` and
    ``compute_stopdff_learned_value.py``. Used only by the
    learned_value_* dispatch tests; the existing dp_sweep stub remains
    untouched so unrelated tests are unaffected.
    """
    python_path = repo / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${STUB_LOG_DIR:?}"
mkdir -p "$LOG_DIR"
cmd="${1:-}"
printf '%s\n' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}" >> "$LOG_DIR/env.log"
printf '%s\n' "$*" >> "$LOG_DIR/args.log"

if [[ "$cmd" == "scripts/device2_cuda_preflight.py" ]]; then
  out_json=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output-json) out_json="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  mkdir -p "$(dirname "$out_json")"
  printf '{"ok": true, "errors": []}\n' > "$out_json"
  exit 0
fi

if [[ "$cmd" == "scripts/train_stopdff_value_model.py" ]]; then
  printf 'learned_value_train stub stdout\n'
  exit 0
fi

if [[ "$cmd" == "scripts/compute_stopdff_learned_value.py" ]]; then
  printf 'learned_value_eval stub stdout\n'
  exit 0
fi

printf 'unexpected python invocation: %s\n' "$*" >&2
exit 99
"""
    )
    python_path.chmod(0o755)
    return python_path


def test_device2_stopdff_harness_dispatches_learned_value_train(
    tmp_path: Path,
) -> None:
    """--experiment learned_value_train must dispatch the trainer script."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_python_stub_with_learned_value(repo)
    out_dir = repo / "run"

    result = _run_harness(
        repo,
        "--experiment", "learned_value_train",
        "--out-dir", str(out_dir),
        "--data-dir", str(repo / "data"),
        "--calibration", str(repo / "calibration.json"),
    )

    assert result.returncode == 0, result.stderr
    args_log = (repo / "stub_logs" / "args.log").read_text().splitlines()
    trainer_args = next(
        (line for line in args_log if "train_stopdff_value_model.py" in line),
        None,
    )
    assert trainer_args is not None, \
        f"trainer script not dispatched; args.log: {args_log}"
    assert "--train-split val" not in trainer_args  # uses train, not val
    assert "--train-split train" in trainer_args
    assert "--val-split val" in trainer_args
    assert "--device cuda" in trainer_args


def test_device2_stopdff_harness_dispatches_learned_value_eval(
    tmp_path: Path,
) -> None:
    """--experiment learned_value_eval must dispatch the eval script."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_python_stub_with_learned_value(repo)
    out_dir = repo / "run"

    result = _run_harness(
        repo,
        "--experiment", "learned_value_eval",
        "--out-dir", str(out_dir),
        "--data-dir", str(repo / "data"),
        "--calibration", str(repo / "calibration.json"),
    )

    assert result.returncode == 0, result.stderr
    args_log = (repo / "stub_logs" / "args.log").read_text().splitlines()
    eval_args = next(
        (line for line in args_log if "compute_stopdff_learned_value.py" in line),
        None,
    )
    assert eval_args is not None, \
        f"eval script not dispatched; args.log: {args_log}"
    assert "--checkpoint-dir" in eval_args
    assert "--eval-split test" in eval_args


def test_device2_stopdff_harness_default_experiment_remains_dp_sweep(
    tmp_path: Path,
) -> None:
    """No --experiment flag → backward compat: dispatches dp_sweep."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_python_stub(repo)
    out_dir = repo / "run"

    result = _run_harness(
        repo,
        "--out-dir", str(out_dir),
        "--data-dir", str(repo / "data"),
        "--calibration", str(repo / "calibration.json"),
    )

    assert result.returncode == 0, result.stderr
    args_log = (repo / "stub_logs" / "args.log").read_text().splitlines()
    sweep_args = next(
        (line for line in args_log if "sweep_stopdff_dp.py" in line),
        None,
    )
    assert sweep_args is not None, \
        "default experiment must dispatch sweep_stopdff_dp.py"


def test_device2_stopdff_harness_rejects_artifact_dir_escape(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_python_stub(repo)

    result = _run_harness(
        repo,
        "--out-dir",
        str(repo / "run"),
        "--artifact-dir",
        "../outside",
        "--data-dir",
        str(repo / "data"),
        "--calibration",
        str(repo / "calibration.json"),
    )

    assert result.returncode != 0
    assert "--artifact-dir" in result.stderr
    assert not (repo / "outside").exists()
