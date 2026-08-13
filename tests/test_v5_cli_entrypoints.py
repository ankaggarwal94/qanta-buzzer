"""Import-bootstrap guard for the documented StopDFF v5 CLI entrypoints.

The ``scripts`` tree is excluded from packaging (``pyproject.toml``
``tool.setuptools.packages``), and the documented invocation form is
``python scripts/<X>.py`` (docs/stopdff_v5/REPRODUCTION.md). Launched that way,
Python puts the *script's own directory* (``scripts/``) on ``sys.path[0]`` — not
the repository root — so any module-level ``from scripts...`` import dies with
``ModuleNotFoundError: No module named 'scripts'`` before argparse can run
unless the script bootstraps the repo root onto ``sys.path`` itself.

This is the mechanical guard for that whole class (PR #30 H-V3-1 /
PROMOTED-V3-1). Each entrypoint's ``--help`` is run as a subprocess from the
repo root with the repo root deliberately kept off ``PYTHONPATH`` — prepending
it would mask the very bug this test exists to catch — and must exit 0 (the
argparse ``--help`` convention) with no ``No module named 'scripts'`` on
stderr. That stderr assertion is deliberately specific to the ``'scripts'``
package: a broad ``ModuleNotFoundError`` check would let an absent optional
extra (e.g. a missing ``modal``) masquerade as the packaging bug this guard
exists to catch (PR #30 H-V4-01 / M-V4-01).

Note: this relies on the scripts importing ``scripts.*`` at module load. For
``modal_stopdff_v5_assurance`` in particular the fix hoists its formerly
deferred ``scripts.*`` imports to module level, so ``--help`` now exercises
them at load and this guard covers it. That entrypoint additionally imports
``modal`` at module load, and ``modal`` is an opt-in extra
(``pip install -e '.[modal]'``); on the documented default install its
``--help`` aborts before argparse, so per AGENTS.md ("tests requiring optional
extras skip when not installed") that one parametrization is skipped when
``modal`` is absent. The other three entrypoints carry no ``modal`` import and
stay unconditional.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# Documented standalone ``python scripts/<X>.py`` entrypoints (REPRODUCTION.md).
_V5_CLI_ENTRYPOINTS = (
    "verify_stopdff_v5_modal_assurance",
    "modal_stopdff_v5_assurance",
    "run_stopdff_v5_local",
    "validate_stopdff_bucketed_sweep",
)

# Entrypoints whose module-level imports include ``import modal`` and so cannot
# reach argparse ``--help`` without the opt-in ``modal`` extra. Established by
# grepping each script for a top-level ``import modal``: only
# ``modal_stopdff_v5_assurance`` has one — the other three resolve their
# ``scripts.*`` imports without touching ``modal``.
_MODAL_REQUIRING_ENTRYPOINTS = frozenset({"modal_stopdff_v5_assurance"})


def _run_help_from_repo_root(name: str) -> subprocess.CompletedProcess[str]:
    """Run ``python scripts/<name>.py --help`` exactly as the docs prescribe.

    cwd is the repo root and the repo root is deliberately removed from
    ``PYTHONPATH`` so a missing in-script ``sys.path`` bootstrap is not masked.
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, os.path.join("scripts", f"{name}.py"), "--help"],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("name", _V5_CLI_ENTRYPOINTS)
def test_v5_cli_entrypoint_help_launches_from_repo_root(name: str) -> None:
    script = _REPO / "scripts" / f"{name}.py"
    assert script.is_file(), f"documented v5 entrypoint is missing: {script}"

    if name in _MODAL_REQUIRING_ENTRYPOINTS:
        # This entrypoint's module-level ``import modal`` runs before argparse,
        # so without the opt-in ``modal`` extra its ``--help`` cannot launch at
        # all. Skip rather than fail (AGENTS.md optional-extras contract);
        # ``importorskip`` is the repo's uniform gate for the ``modal`` extra
        # (cf. tests/test_modal_stopdff_runner.py).
        pytest.importorskip("modal")

    completed = _run_help_from_repo_root(name)

    assert "No module named 'scripts'" not in completed.stderr, (
        f"{name} --help raised ModuleNotFoundError for the packaging-excluded "
        f"'scripts' tree — it needs a sys.path bootstrap before its "
        f"module-level scripts.* imports:\n{completed.stderr}"
    )
    assert completed.returncode == 0, (
        f"{name} --help exited {completed.returncode}; argparse --help must "
        f"exit 0.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
