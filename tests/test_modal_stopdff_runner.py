"""Smoke tests for scripts/modal_stopdff_runner.py.

These tests verify the operator-facing surface (CLI parameters,
EXPERIMENTS tuple, GPU normalization, app/volume names) without
performing any actual Modal dispatch. Real remote execution requires a
Modal account, network access, and credits; that is out of scope for
the pytest suite.

The script uses ``@app.local_entrypoint()`` rather than ``argparse`` --
its CLI surface is the ``main()`` function signature, introspected by
Modal's CLI. We therefore validate the signature (and the
module-level constants Modal binds against) rather than running
``--help``, which would print nothing in a pure-Python invocation
(Modal only wires CLI when invoked via ``modal run``).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "modal_stopdff_runner.py"


# ---------------------------------------------------------------------------
# File-level smoke
# ---------------------------------------------------------------------------


def test_runner_file_exists_and_is_executable() -> None:
    assert RUNNER.exists(), f"{RUNNER} missing"
    # The runner is part of an operator workflow; check the +x bit so a
    # missing chmod doesn't silently regress operator UX.
    assert RUNNER.stat().st_mode & 0o111, f"{RUNNER} not executable"


def test_runner_file_parses_as_python() -> None:
    """Catches Dropbox-sync CRLF corruption / partial copies."""
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)


def test_runner_has_operator_docstring_with_usage_examples() -> None:
    """Module docstring must include the canonical ``modal run`` usage."""
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    assert "modal run" in docstring, \
        "Module docstring must include `modal run` usage examples"
    assert "artifact-subdir" in docstring, \
        "Module docstring must mention --artifact-subdir"


# ---------------------------------------------------------------------------
# Module surface (import-time)
# ---------------------------------------------------------------------------


def _load_module():
    """Import the runner module fresh against the real ``modal`` SDK.

    ``tests/test_modal_cs321m_stage_commands.py`` installs a permissive
    ``modal`` stub into ``sys.modules`` to test the legacy wrapper
    without requiring the SDK. That stub lacks ``modal.Volume``,
    ``modal.Secret``, etc., which the canonical runner uses. If pytest
    runs that file before this one, the stub poisons our import. Pop
    the stub (and any cached runner module that was built against it)
    before importing so we always exercise the real SDK surface. Skip
    this test module entirely if the real ``modal`` SDK is unavailable.
    """
    import importlib
    import sys

    cached_modal = sys.modules.get("modal")
    if cached_modal is not None and not hasattr(cached_modal, "Volume"):
        # Stub-poisoned. Evict and let the real package re-import.
        sys.modules.pop("modal", None)
        for cached in [k for k in sys.modules
                       if k == "modal" or k.startswith("modal.")]:
            sys.modules.pop(cached, None)
        sys.modules.pop("scripts.modal_stopdff_runner", None)
    pytest.importorskip("modal")
    real_modal = sys.modules["modal"]
    if not hasattr(real_modal, "Volume"):
        pytest.skip("real `modal` SDK with `modal.Volume` not available")
    # Allow direct ``scripts.modal_stopdff_runner`` resolution.
    sys.path.insert(0, str(PROJECT_ROOT))
    return importlib.import_module("scripts.modal_stopdff_runner")


def test_module_imports_cleanly() -> None:
    """End-to-end import (loads modal SDK + builds the App)."""
    m = _load_module()
    assert m is not None
    assert hasattr(m, "EXPERIMENTS")
    assert hasattr(m, "main")


def test_experiments_tuple_lists_canonical_baseline() -> None:
    """Baseline EXPERIMENTS must include smoke, single, dp_sweep."""
    m = _load_module()
    for required in ("smoke", "single", "dp_sweep"):
        assert required in m.EXPERIMENTS, \
            f"EXPERIMENTS missing baseline entry: {required!r}"


def test_experiments_tuple_includes_learned_value_pair() -> None:
    """Prompt 5 learned-value branches must be wired into EXPERIMENTS.

    The dispatched scripts (train_stopdff_value_model.py /
    compute_stopdff_learned_value.py) do NOT yet exist in this commit;
    invoking them produces a fail-fast subprocess error, which is the
    correct behavior. The runner must still know about them so future
    work can land the scripts without touching the runner.
    """
    m = _load_module()
    for required in ("learned_value_train", "learned_value_eval"):
        assert required in m.EXPERIMENTS, \
            f"EXPERIMENTS missing learned-value entry: {required!r}"


def test_build_command_dispatches_learned_value_train_to_trainer_script() -> None:
    """--experiment learned_value_train must invoke the trainer script."""
    m = _load_module()
    from pathlib import PurePosixPath
    cmd = m._build_command(
        experiment="learned_value_train",
        artifact_subdir_abs=PurePosixPath("/artifacts/test_subdir"),
        num_bootstrap=100,
        max_wall_hours=1.0,
        n_jobs=1,
        resume=False,
        smoke=False,
    )
    assert "scripts/train_stopdff_value_model.py" in cmd
    assert "--train-split" in cmd and "--val-split" in cmd
    assert "--device" in cmd and "cuda" in cmd


def test_build_command_dispatches_learned_value_eval_to_eval_script() -> None:
    """--experiment learned_value_eval must invoke the eval script."""
    m = _load_module()
    from pathlib import PurePosixPath
    cmd = m._build_command(
        experiment="learned_value_eval",
        artifact_subdir_abs=PurePosixPath("/artifacts/test_subdir"),
        num_bootstrap=100,
        max_wall_hours=1.0,
        n_jobs=1,
        resume=False,
        smoke=False,
    )
    assert "scripts/compute_stopdff_learned_value.py" in cmd
    assert "--checkpoint-dir" in cmd
    assert "--eval-split" in cmd and "test" in cmd


def test_build_command_smoke_trims_learned_value_train_hyperparams() -> None:
    """`smoke=True` must trim epochs/seeds/hidden for the trainer branch."""
    m = _load_module()
    from pathlib import PurePosixPath
    cmd = m._build_command(
        experiment="learned_value_train",
        artifact_subdir_abs=PurePosixPath("/artifacts/test_subdir"),
        num_bootstrap=100,
        max_wall_hours=1.0,
        n_jobs=1,
        resume=False,
        smoke=True,
    )
    assert "--epochs" in cmd and "2" in cmd
    assert "--seeds" in cmd and "1" in cmd
    assert "--hidden" in cmd and "32" in cmd


def test_build_command_still_dispatches_baseline_dp_sweep() -> None:
    """Backward compat: dp_sweep dispatch must be preserved unchanged."""
    m = _load_module()
    from pathlib import PurePosixPath
    cmd = m._build_command(
        experiment="dp_sweep",
        artifact_subdir_abs=PurePosixPath("/artifacts/test_subdir"),
        num_bootstrap=42,
        max_wall_hours=2.0,
        n_jobs=4,
        resume=False,
        smoke=False,
    )
    assert "scripts/sweep_stopdff_dp.py" in cmd
    assert "--num-bootstrap" in cmd and "42" in cmd
    assert "--n-jobs" in cmd and "4" in cmd


def test_app_and_volume_names_match_runbook() -> None:
    """APP_NAME and VOLUME_NAME are the operator-facing identifiers."""
    m = _load_module()
    assert m.APP_NAME == "cs321m-stopdff"
    assert m.VOLUME_NAME == "cs321m-stopdff-artifacts"


# ---------------------------------------------------------------------------
# GPU normalization (the only pure helper that does not touch Modal state)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias", ["none", "cpu", "", "null", "NONE", " None ", "CPU"])
def test_normalize_gpu_cpu_aliases_resolve_to_none(alias: str) -> None:
    """All accepted CPU aliases must collapse to None (Modal's CPU sentinel)."""
    m = _load_module()
    assert m._normalize_gpu(alias) is None, \
        f"_normalize_gpu({alias!r}) should return None"


@pytest.mark.parametrize(
    "gpu_spec", ["L40S", "A100", "A100-80GB", "H100", "  L40S  "],
)
def test_normalize_gpu_passes_through_gpu_strings(gpu_spec: str) -> None:
    """Modern Modal GPU string specs must pass through (trimmed)."""
    m = _load_module()
    result = m._normalize_gpu(gpu_spec)
    assert result is not None
    assert result == gpu_spec.strip()


def test_normalize_gpu_handles_none_input() -> None:
    m = _load_module()
    assert m._normalize_gpu(None) is None


# ---------------------------------------------------------------------------
# Local entrypoint signature (Modal binds these to CLI --<flag> arguments)
# ---------------------------------------------------------------------------


def _raw_main(m):
    """Return the underlying function the Modal CLI binds.

    ``@app.local_entrypoint()`` wraps ``main`` into a Modal
    ``LocalEntrypoint`` whose ``info.raw_f`` is the original Python
    function. We inspect the raw function so the test sees the real
    signature, not the generic ``*args, **kwargs`` wrapper.
    """
    return m.main.info.raw_f


def test_local_entrypoint_exposes_required_operator_flags() -> None:
    """main(...) signature must include every operator-facing parameter.

    Modal's CLI auto-generates --<param-name> flags from the signature,
    so the test pins the exposed surface. Adding a new flag is a
    deliberate signature change.
    """
    m = _load_module()
    sig = inspect.signature(_raw_main(m))
    expected = {
        "experiment",
        "artifact_subdir",
        "git_ref",
        "gpu",
        "max_wall_hours",
        "num_bootstrap",
        "n_jobs",
        "resume",
        "smoke",
        "allow_dirty",
        "overwrite",
        "with_openai_key",
    }
    missing = expected - set(sig.parameters)
    assert not missing, f"main() signature missing flags: {sorted(missing)}"


def test_local_entrypoint_defaults_to_smoke_experiment() -> None:
    """The default --experiment selects a cheap, CPU-runnable smoke target."""
    m = _load_module()
    sig = inspect.signature(_raw_main(m))
    assert sig.parameters["experiment"].default == "smoke"


def test_local_entrypoint_rejects_unknown_experiment() -> None:
    """Invalid --experiment must SystemExit before any Modal dispatch."""
    m = _load_module()
    raw_main = _raw_main(m)
    with pytest.raises(SystemExit, match="--experiment must be one of"):
        raw_main(experiment="unknown_xyz", artifact_subdir="test")
