"""Seam-binding coverage for the production ``_default_control_stage_api``.

Control-plane tests inject fake stage APIs whose callables accept ``*args``,
so drift between the real Modal stage functions and the argument tuples
``run_control_plane`` passes them would surface only on a paid live run.
This test constructs the REAL default stage API under the fake-modal seam
(the one ``tests/test_modal_runner_promotion.py`` uses, where
``app.function()`` attaches the plain function as ``.remote``) and binds it
against the driver's own dispatch sites, read from the AST of
``scripts/stopdff_v5_control_plane.py`` (the module that owns the
``run_control_plane`` driver; the runner re-exports it as a facade):

- name binding: the default API's key set equals the set of stages
  ``run_control_plane`` actually dispatches;
- arity binding: each real stage function accepts every positional argument
  tuple the control plane passes it.
"""

from __future__ import annotations

import ast
import inspect

from tests.harness_control_plane import MODAL_RUNNER, _load_modal_runner

CONTROL_PLANE = MODAL_RUNNER.with_name("stopdff_v5_control_plane.py")


def _control_plane_dispatch_arities() -> dict[str, set[int]]:
    """Return ``{stage: {positional-arg counts}}`` from ``api["…"](...)`` sites."""
    module = ast.parse(CONTROL_PLANE.read_text(encoding="utf-8"))
    driver = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_control_plane"
    )
    arities: dict[str, set[int]] = {}
    for node in ast.walk(driver):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Subscript)
            and isinstance(func.value, ast.Name)
            and func.value.id == "api"
            and isinstance(func.slice, ast.Constant)
            and isinstance(func.slice.value, str)
        ):
            continue
        assert not node.keywords, (
            "control plane stage dispatch is positional-only"
        )
        assert not any(isinstance(arg, ast.Starred) for arg in node.args), (
            "control plane stage dispatch must not splat arguments"
        )
        arities.setdefault(func.slice.value, set()).add(len(node.args))
    return arities


def test_default_stage_api_matches_control_plane_dispatch(monkeypatch):
    runner = _load_modal_runner(monkeypatch)
    api = runner._default_control_stage_api()
    dispatched = _control_plane_dispatch_arities()
    assert dispatched, "run_control_plane must dispatch through api[...]"

    # Name binding: ground truth is the driver's own dispatch sites.
    assert set(api) == set(dispatched), (
        "default stage API keys must match run_control_plane dispatch sites"
    )
    # Regression anchor for the canonical stage set (a rename shows up here
    # with a readable diff even if both sides drift together).
    assert set(api) == {
        "probe",
        "verify_volume_artifact",
        "freeze_model",
        "adapter_determinism_receipt",
        "promote_adapter",
        "fvi_study",
        "bootstrap_plan",
        "run_sweep",
        "mutation_gate",
        "validate",
        "package",
    }

    # Arity binding: under the fake modal, ``.remote`` is the real function,
    # so ``signature.bind`` proves each dispatch tuple is accepted.
    for stage, counts in sorted(dispatched.items()):
        target = api[stage]
        assert callable(target), stage
        signature = inspect.signature(target)
        for count in sorted(counts):
            try:
                signature.bind(*(object(),) * count)
            except TypeError as exc:
                raise AssertionError(
                    f"stage {stage!r} does not accept the control plane's "
                    f"{count}-positional-argument call: {exc}"
                ) from exc
