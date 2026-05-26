"""DATA-05 symbol-collision guard enforcement (Phase 02 review WR-01).

The CSLI panel script (``scripts/compute_csli.py``) implements a local
local-model panel for choice-set leakage. It is FORBIDDEN by MASTER_PLAN_v10
DATA-05 from importing ``evaluation.controls.run_choices_only_control``,
which is a different experiment (surface-feature logistic regression on
char-trigram TF-IDF) and would produce a contaminated audit claim if
called from this code path.

The runtime guard inside ``compute_csli.py`` (``_assert_no_controls_import``)
catches violations at import time. This static AST test catches them at
CI time so they never reach import in the first place.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_COMPUTE_CSLI = Path(__file__).resolve().parents[1] / "scripts" / "compute_csli.py"


def _parse_compute_csli() -> ast.Module:
    """Parse compute_csli.py into an AST.

    Returns
    -------
    ast.Module
        Parsed AST of scripts/compute_csli.py.
    """
    return ast.parse(_COMPUTE_CSLI.read_text(encoding="utf-8"))


def test_compute_csli_file_exists() -> None:
    """Sanity check: the file we're protecting is at the expected path."""
    assert _COMPUTE_CSLI.exists(), (
        f"DATA-05 guard: expected scripts/compute_csli.py at {_COMPUTE_CSLI}, "
        "but it is missing. Has the script been renamed?"
    )


def test_compute_csli_does_not_import_evaluation_controls() -> None:
    """DATA-05: compute_csli.py must not ``from evaluation.controls import ...``."""
    tree = _parse_compute_csli()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "evaluation.controls", (
                "DATA-05 violation: compute_csli.py contains "
                "`from evaluation.controls import ...`. "
                "Use the local-model panel only. See MASTER_PLAN_v10 DATA-05."
            )
            if node.module == "evaluation":
                for alias in node.names:
                    assert alias.name != "controls", (
                        "DATA-05 violation: compute_csli.py contains "
                        "`from evaluation import controls`. "
                        "Use the local-model panel only."
                    )


def test_compute_csli_does_not_import_evaluation_controls_submodule() -> None:
    """DATA-05: compute_csli.py must not ``import evaluation.controls``."""
    tree = _parse_compute_csli()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("evaluation.controls"), (
                    "DATA-05 violation: compute_csli.py contains "
                    f"`import {alias.name}`. Use the local-model panel only."
                )


def test_runtime_guard_function_is_called_at_module_scope() -> None:
    """The runtime guard must execute at import time, not just be defined.

    Catches the regression where a maintainer adds the function but
    forgets to call it -- in that case ``_assert_no_controls_import``
    would silently pass and DATA-05 enforcement would collapse back
    to a comment-only convention.
    """
    tree = _parse_compute_csli()
    has_call_at_module_scope = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_assert_no_controls_import"
        for node in tree.body
    )
    assert has_call_at_module_scope, (
        "DATA-05 guard: compute_csli.py defines _assert_no_controls_import "
        "but does not invoke it at module scope. The guard MUST run at "
        "import time, otherwise it cannot fire before the forbidden symbol "
        "is reachable."
    )


def test_runtime_guard_fires_when_evaluation_controls_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """When evaluation.controls is already in sys.modules, the guard raises.

    Verifies behavior of the live guard function (not just its presence).
    """
    import sys

    # Stash and clear any real `scripts.compute_csli` so we re-trigger
    # _assert_no_controls_import on fresh import.
    for mod in list(sys.modules):
        if mod.startswith("scripts.compute_csli"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    # Plant a fake evaluation.controls into sys.modules so the guard
    # has something to detect.
    import types
    fake = types.ModuleType("evaluation.controls")
    monkeypatch.setitem(sys.modules, "evaluation.controls", fake)

    with pytest.raises(ImportError, match="DATA-05 violation"):
        import importlib
        # Re-import compute_csli with the planted forbidden module.
        importlib.import_module("scripts.compute_csli")
