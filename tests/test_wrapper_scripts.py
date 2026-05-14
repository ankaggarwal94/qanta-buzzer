"""Tests for repo wrapper scripts that should prefer the local `.venv`."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
import subprocess

import pytest

_BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(_BASH is None, reason="bash not available")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, body: str) -> None:
    """Write a small executable script."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _copy_repo_script(repo_root: Path, name: str) -> Path:
    """Copy the current wrapper script under test into a temp repo."""

    script_path = repo_root / "scripts" / name
    script_path.parent.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "scripts" / name
    script_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _run_wrapper(
    script_path: Path,
    calls_log: Path,
    external_bin: Path,
) -> subprocess.CompletedProcess[str]:
    """Run a wrapper script with stubbed ambient executables."""

    env = os.environ.copy()
    env["CALLS_LOG"] = str(calls_log)
    env["PATH"] = f"{external_bin}:{env['PATH']}"
    return subprocess.run(
        [_BASH, str(script_path)],
        cwd=script_path.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_ci_sh_uses_repo_venv_pytest_without_activate_path_changes(tmp_path) -> None:
    """CI wrapper should run `.venv/bin/pytest` directly when `.venv` exists."""

    repo_root = tmp_path / "repo"
    script_path = _copy_repo_script(repo_root, "ci.sh")
    calls_log = repo_root / "calls.log"

    _write_executable(
        repo_root / ".venv" / "bin" / "activate",
        "#!/usr/bin/env bash\nexport ACTIVATE_SOURCED=1\n",
    )
    _write_executable(
        repo_root / ".venv" / "bin" / "pytest",
        "#!/usr/bin/env bash\necho \"venv-pytest $*\" >> \"$CALLS_LOG\"\n",
    )
    _write_executable(
        repo_root / "external-bin" / "pytest",
        "#!/usr/bin/env bash\necho \"external-pytest $*\" >> \"$CALLS_LOG\"\n",
    )

    result = _run_wrapper(script_path, calls_log, repo_root / "external-bin")

    assert result.returncode == 0, result.stderr
    assert calls_log.read_text(encoding="utf-8").splitlines() == [
        "venv-pytest tests/"
    ]


def test_ci_sh_requires_repo_venv_instead_of_ambient_pytest(tmp_path) -> None:
    """CI wrapper should fail fast instead of picking up ambient pytest."""

    repo_root = tmp_path / "repo"
    script_path = _copy_repo_script(repo_root, "ci.sh")
    calls_log = repo_root / "calls.log"

    _write_executable(
        repo_root / "external-bin" / "pytest",
        "#!/usr/bin/env bash\necho \"external-pytest $*\" >> \"$CALLS_LOG\"\n",
    )

    result = _run_wrapper(script_path, calls_log, repo_root / "external-bin")

    assert result.returncode != 0
    assert ".venv" in result.stderr
    assert not calls_log.exists()


def test_manual_smoke_uses_repo_venv_python_without_activate_path_changes(
    tmp_path,
) -> None:
    """Smoke wrapper should run `.venv/bin/python` directly when `.venv` exists."""

    repo_root = tmp_path / "repo"
    script_path = _copy_repo_script(repo_root, "manual-smoke.sh")
    calls_log = repo_root / "calls.log"

    _write_executable(
        repo_root / ".venv" / "bin" / "activate",
        "#!/usr/bin/env bash\nexport ACTIVATE_SOURCED=1\n",
    )
    _write_executable(
        repo_root / ".venv" / "bin" / "python",
        "#!/usr/bin/env bash\necho \"venv-python $*\" >> \"$CALLS_LOG\"\n",
    )
    _write_executable(
        repo_root / "external-bin" / "python3",
        "#!/usr/bin/env bash\necho \"external-python $*\" >> \"$CALLS_LOG\"\n",
    )

    result = _run_wrapper(script_path, calls_log, repo_root / "external-bin")

    assert result.returncode == 0, result.stderr
    assert calls_log.read_text(encoding="utf-8").splitlines() == [
        "venv-python scripts/build_mc_dataset.py --smoke",
        "venv-python scripts/run_baselines.py --smoke",
        "venv-python scripts/train_ppo.py --smoke",
        "venv-python scripts/evaluate_all.py --smoke",
    ]


def test_manual_smoke_requires_repo_venv_instead_of_ambient_python(
    tmp_path,
) -> None:
    """Smoke wrapper should fail fast instead of picking up ambient python."""

    repo_root = tmp_path / "repo"
    script_path = _copy_repo_script(repo_root, "manual-smoke.sh")
    calls_log = repo_root / "calls.log"

    _write_executable(
        repo_root / "external-bin" / "python3",
        "#!/usr/bin/env bash\necho \"external-python $*\" >> \"$CALLS_LOG\"\n",
    )

    result = _run_wrapper(script_path, calls_log, repo_root / "external-bin")

    assert result.returncode != 0
    assert ".venv" in result.stderr
    assert not calls_log.exists()
