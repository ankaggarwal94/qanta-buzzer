"""Import-precedence guard for the StopDFF v5 standalone acceptance validator.

Companion to ``tests/test_v5_cli_entrypoints.py``. That guard covers the
*absent-root* case (the repo root off ``sys.path`` entirely -> the bootstrap
must ADD it). This guard covers the complementary *stale-precedence* case: a
second checkout (or installed tree) already on ``PYTHONPATH`` AHEAD of this repo
root, with this repo root also present but not first.

``scripts/validate_stopdff_bucketed_sweep.py`` recomputes every acceptance
statistic from ``scripts.stopdff_v5`` producer code, so which checkout that
``scripts.*`` import resolves from is a provenance guarantee, not a convenience.
The old bootstrap was a membership guard::

    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

When ``_REPO`` is already *in* ``sys.path`` but not at index 0, the insert is
skipped and an earlier (stale) entry keeps precedence, so
``scripts.stopdff_v5.checker`` resolves from STALE code. The fix force-fronts
the repo root unconditionally (dedupe + insert at 0), matching the three sibling
``python scripts/*.py`` entrypoints (``run_stopdff_v5_local``,
``modal_stopdff_v5_assurance``, ``verify_stopdff_v5_modal_assurance``). PR #30.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_VALIDATOR = _REPO / "scripts" / "validate_stopdff_bucketed_sweep.py"

_STALE_SENTINEL = "STOPDFF_STALE_SHADOW_CHECKER_IMPORTED"


def _write_stale_shadow(root: Path) -> Path:
    """Create a throwaway ``scripts.stopdff_v5`` checkout that shadows the real one.

    Mirrors the real *regular*-package chain (both levels carry ``__init__.py``)
    so, when ahead on ``sys.path``, it fully shadows ``scripts.stopdff_v5`` --
    exactly the stale-checkout hazard the fix defends against. ``checker`` marks
    itself (a stderr sentinel) so a load from here is observable; ``selftest``
    exists only so the validator's ``from scripts.stopdff_v5 import checker,
    selftest`` completes from the shadow instead of dying on a missing name
    (which would mask the precedence bug behind an unrelated ImportError).
    """
    package = root / "scripts"
    subpackage = package / "stopdff_v5"
    subpackage.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (subpackage / "__init__.py").write_text("", encoding="utf-8")
    (subpackage / "checker.py").write_text(
        "import sys\n"
        "STOPDFF_STALE_SHADOW = True\n"
        f'sys.stderr.write("{_STALE_SENTINEL}\\n")\n',
        encoding="utf-8",
    )
    (subpackage / "selftest.py").write_text(
        "def run_self_test(base_dir):\n"
        "    raise AssertionError('stale shadow selftest must never run')\n",
        encoding="utf-8",
    )
    return package


def test_validator_forces_repo_root_ahead_of_stale_checkout(tmp_path: Path) -> None:
    """``scripts.stopdff_v5.checker`` must resolve from THIS repo, never a stale one.

    Deterministic probe: seed ``sys.path`` with the stale checkout AHEAD of the
    repo root (repo root present but not first -- the precise state the old
    membership guard mishandled), load the validator by file path so its
    module-level bootstrap runs, then inspect where ``scripts.stopdff_v5.checker``
    actually resolved. Also asserts the fix's structural guarantee -- the repo
    root is ``sys.path[0]`` exactly once (no duplicate) -- which additionally
    rejects a non-deduping ``insert(0, ...)`` regression.
    """
    stale_root = tmp_path / "stale_checkout"
    _write_stale_shadow(stale_root)
    stale = str(stale_root.resolve())
    # str(_REPO) is exactly what the validator's bootstrap recomputes as
    # ``_REPO_IMPORT_ROOT`` (both from ``Path(...).resolve()``), so the dedupe's
    # string equality -- and the count()/[0] assertions below -- line up.
    repo = str(_REPO)
    validator = str(_VALIDATOR)

    probe = textwrap.dedent(
        """
        import importlib.util, pathlib, sys
        stale, repo, validator = sys.argv[1], sys.argv[2], sys.argv[3]
        # Poison precedence the way the membership guard mishandled it: the repo
        # root is already present (so ``str(_REPO) not in sys.path`` is False)
        # with a stale checkout ahead of it. These inserts sit at the very front,
        # ahead of any ambient site/.pth entries, so the scenario is exact.
        sys.path.insert(0, repo)
        sys.path.insert(0, stale)
        spec = importlib.util.spec_from_file_location("_validator_under_test", validator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # runs the sys.path bootstrap + scripts.* imports
        loaded = pathlib.Path(sys.modules["scripts.stopdff_v5.checker"].__file__).resolve()
        stale_dir = pathlib.Path(stale).resolve()
        repo_dir = pathlib.Path(repo).resolve()
        if stale_dir in loaded.parents:
            raise SystemExit("STALE_CHECKER_LOADED:%s" % loaded)
        if repo_dir not in loaded.parents:
            raise SystemExit("UNKNOWN_CHECKER_ORIGIN:%s" % loaded)
        if sys.path[0] != repo:
            raise SystemExit("REPO_ROOT_NOT_FRONTED:%r" % sys.path[:3])
        if sys.path.count(repo) != 1:
            raise SystemExit("REPO_ROOT_DUPLICATED:%d" % sys.path.count(repo))
        print("REAL_CHECKER_LOADED:%s" % loaded)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe, stale, repo, validator],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "validator resolved scripts.stopdff_v5.checker from the wrong checkout "
        "(or failed the force-front invariant).\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "REAL_CHECKER_LOADED" in completed.stdout, (
        completed.stdout + completed.stderr
    )
    assert _STALE_SENTINEL not in completed.stderr, completed.stderr


def test_validator_help_ignores_stale_checkout_on_pythonpath(tmp_path: Path) -> None:
    """Faithful launcher check: ``python scripts/validate_stopdff_bucketed_sweep.py --help``.

    Runs the exact documented entrypoint with a stale checkout ahead of the repo
    root on ``PYTHONPATH`` (repo root kept present so the membership guard's skip
    path is exercised). ``--help`` triggers the module-level ``from
    scripts.stopdff_v5 import checker, selftest`` before argparse, so a stale
    ``checker`` would announce itself on stderr. The fix keeps the real checker,
    so the sentinel never appears and ``--help`` exits 0.
    """
    stale_root = tmp_path / "stale_checkout"
    _write_stale_shadow(stale_root)

    env = os.environ.copy()
    # Stale checkout AHEAD of the repo root; both present. ``python script.py``
    # puts ``scripts/`` on sys.path[0], then these PYTHONPATH entries follow, so
    # the repo root is present-but-not-first -- the membership-guard trap. Set
    # exactly these two (venv site-packages still provide third-party deps).
    env["PYTHONPATH"] = os.pathsep.join([str(stale_root.resolve()), str(_REPO)])

    completed = subprocess.run(
        [
            sys.executable,
            os.path.join("scripts", "validate_stopdff_bucketed_sweep.py"),
            "--help",
        ],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"--help exited {completed.returncode}; argparse --help must exit 0.\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert _STALE_SENTINEL not in completed.stderr, (
        "validator imported scripts.stopdff_v5.checker from the stale checkout on "
        f"PYTHONPATH instead of this repo:\n{completed.stderr}"
    )
