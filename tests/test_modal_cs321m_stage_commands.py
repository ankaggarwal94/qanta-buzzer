"""Regression tests for modal_cs321m.build_stage_command.

PR #14 follow-up review (Codex 3308173797): smoke audit stages must
propagate `--allow-low-mc-retention` and `--allow-incomplete-mc-coverage`
so the committed smoke build metadata (val=0.0 / test=0.1429 retention
vs the smoke threshold 0.5) doesn't abort the smoke pipeline at the
calibration stage. Full mode must NOT receive these flags — the
full corpus is expected to meet the gate naturally.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_modal_module():
    """Import modal_cs321m.py as a module without requiring the `modal` package.

    The top-level `modal` import in modal_cs321m.py would otherwise force
    every test environment to install the modal dependency. We patch a
    stub into sys.modules before importing.
    """
    if "modal" not in sys.modules:
        import types

        class _PermissiveBuilder:
            """Catches any builder-pattern method call and returns self."""

            def __getattr__(self, _name):
                return lambda *a, **k: self

        class _AppStub:
            """Catches @app.function(...) decorator chains."""

            def function(self, *a, **k):
                return lambda f: f

            def __getattr__(self, _name):
                return lambda *a, **k: (lambda f: f)

        modal_stub = types.ModuleType("modal")
        modal_stub.App = lambda *a, **k: _AppStub()
        modal_stub.Image = type(
            "_Image",
            (),
            {
                "debian_slim": staticmethod(lambda *a, **k: _PermissiveBuilder()),
            },
        )
        modal_stub.gpu = type("_Gpu", (), {"A100": lambda *a, **k: None})
        sys.modules["modal"] = modal_stub

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "modal_cs321m_under_test",
        repo_root / "modal_cs321m.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_audit_stages_propagate_retained_subset_overrides() -> None:
    """Smoke audit stages must carry the retained-subset overrides.

    The committed smoke build metadata has retention well below the
    smoke threshold (val=0.0, test=0.1429 vs 0.5), so without the
    explicit overrides compute_prefix_calibration / compute_csli /
    compute_stopdff would abort. The smoke fixture is too small for
    the retention gate to be meaningful, so the overrides are the
    correct opt-in.
    """
    modal_cs321m = _load_modal_module()
    for stage in ("compute_csli", "compute_prefix_calibration", "compute_stopdff"):
        cmd = modal_cs321m.build_stage_command(
            stage, config_path="configs/cs321m_smoke.yaml", smoke=True
        )
        joined = " ".join(cmd)
        assert "--allow-low-mc-retention" in joined, (
            f"smoke {stage} command is missing --allow-low-mc-retention: {joined}"
        )
        assert "--allow-incomplete-mc-coverage" in joined, (
            f"smoke {stage} command is missing --allow-incomplete-mc-coverage: {joined}"
        )
        assert "--smoke" in joined


def test_full_audit_stages_do_not_propagate_retained_subset_overrides() -> None:
    """Full-mode audit stages must NOT auto-pass the overrides.

    The full corpus is expected to meet the retention gate naturally.
    Auto-passing the override would silently launder the gate's
    pre-registered constraint.
    """
    modal_cs321m = _load_modal_module()
    for stage in ("compute_csli", "compute_prefix_calibration", "compute_stopdff"):
        cmd = modal_cs321m.build_stage_command(
            stage, config_path="configs/cs321m_smoke.yaml", smoke=False
        )
        joined = " ".join(cmd)
        assert "--allow-low-mc-retention" not in joined, (
            f"full {stage} command unexpectedly carries --allow-low-mc-retention: {joined}"
        )
        assert "--allow-incomplete-mc-coverage" not in joined, (
            f"full {stage} command unexpectedly carries --allow-incomplete-mc-coverage: {joined}"
        )


def test_legacy_stages_unaffected_by_override_flags() -> None:
    """Legacy stages (build_mc_dataset, run_baselines, train_ppo, evaluate_all)
    use --config / --smoke / --output-dir and must not get the audit-only flags."""
    modal_cs321m = _load_modal_module()
    for stage in ("build_mc_dataset", "run_baselines", "train_ppo", "evaluate_all"):
        cmd = modal_cs321m.build_stage_command(
            stage, config_path="configs/cs321m_smoke.yaml", smoke=True
        )
        joined = " ".join(cmd)
        assert "--allow-low-mc-retention" not in joined
        assert "--allow-incomplete-mc-coverage" not in joined


def test_smoke_audit_outputs_isolated_from_production_paper_exports() -> None:
    """Smoke audit stages must NEVER write to the unqualified paper_exports/.

    PR #14 follow-up review (post-stale ChatGPT re-validation Lane E
    FN-1): commit 828d452 made the smoke audit pipeline succeed
    end-to-end. Without isolated output routing the smoke run can
    silently overwrite curated production paper_exports/*.json with
    smoke-only numbers (3 val rows, 14 test rows), corrupting the
    final paper evidence package. Smoke outputs must land under
    ``artifacts/smoke/paper_exports/`` whenever ``--output-dir`` is
    unspecified; full mode and explicit overrides preserve the
    operator-specified directory.
    """
    modal_cs321m = _load_modal_module()
    # Helper: default smoke routing must be artifacts/smoke/paper_exports/.
    assert (
        modal_cs321m.export_dir_for_run(smoke=True, output_dir=None)
        == "artifacts/smoke/paper_exports"
    )
    # Helper: full mode default unchanged (production paper_exports/).
    assert modal_cs321m.export_dir_for_run(smoke=False, output_dir=None) == "paper_exports"
    # Helper: explicit output_dir always wins (operator-controlled).
    assert modal_cs321m.export_dir_for_run(
        smoke=True, output_dir="/tmp/run42"
    ) == "/tmp/run42/paper_exports"
    assert modal_cs321m.export_dir_for_run(
        smoke=False, output_dir="/tmp/run42"
    ) == "/tmp/run42/paper_exports"

    # Stage-level: smoke commands must NOT name the bare production
    # paper_exports/ path; they must route through artifacts/smoke/.
    for stage in ("compute_csli", "compute_prefix_calibration", "compute_stopdff"):
        cmd = modal_cs321m.build_stage_command(
            stage, config_path="configs/cs321m_smoke.yaml", smoke=True
        )
        joined = " ".join(cmd)
        # The output path string for each audit stage must point at
        # artifacts/smoke/paper_exports/, never at the bare production
        # directory. We require the substring rather than scanning each
        # --output arg individually because the stopdff stage also
        # references calibration.json via --calibration which must
        # share the smoke routing.
        assert "artifacts/smoke/paper_exports" in joined, (
            f"smoke {stage} command does not route through artifacts/smoke/: "
            f"{joined}"
        )
        # Negative assertion: no bare production paper_exports/ path
        # may appear anywhere in the smoke command (would indicate a
        # forgotten caller that still uses the legacy default).
        bare_prod_tokens = [
            t for t in cmd
            if t.startswith("paper_exports/")
            or t == "paper_exports"
        ]
        assert not bare_prod_tokens, (
            f"smoke {stage} command leaks bare paper_exports/ tokens: "
            f"{bare_prod_tokens} in {joined}"
        )

    # Full mode: outputs continue to use bare production paper_exports/
    # (this is the path operators expect for full-corpus runs).
    for stage in ("compute_csli", "compute_prefix_calibration", "compute_stopdff"):
        cmd = modal_cs321m.build_stage_command(
            stage, config_path="configs/cs321m_smoke.yaml", smoke=False
        )
        joined = " ".join(cmd)
        assert "paper_exports/" in joined and "artifacts/smoke/" not in joined, (
            f"full {stage} command unexpectedly routes through smoke dir: {joined}"
        )


def test_absolute_output_dir_round_trip_preserves_artifacts(tmp_path) -> None:
    """PR #14 follow-up review (Codex 3308546548): when --output-dir is an
    absolute path outside REPO_ROOT (e.g., /tmp/run42), Modal writes
    artifacts at that absolute path inside the container. The collect ->
    return -> write round trip must not silently drop those files.

    Previously: collect_small_artifacts produced absolute keys for
    out-of-/app paths; write_returned_artifacts skipped absolute keys
    with `continue`. Net effect was data loss whenever the operator
    passed an absolute --output-dir outside the repo.

    Now: collector anchors absolute-outside-repo paths to the operator's
    output_dir, and writer accepts output_dir to re-anchor on the local
    side. Pin both directions of the round trip.
    """
    import json
    import tempfile

    modal_cs321m = _load_modal_module()

    # Anchor-resolver behavior.
    repo_root = modal_cs321m.REPO_ROOT
    inside_repo = str(repo_root / "artifacts" / "smoke")
    outside_repo = str(tmp_path / "run42")
    # No output_dir -> no anchor.
    assert modal_cs321m._absolute_output_anchor(None) is None
    # Relative output_dir -> no anchor (stays under REPO_ROOT semantics).
    assert modal_cs321m._absolute_output_anchor("mytest") is None
    # Absolute under REPO_ROOT -> no anchor (REPO_ROOT path still works).
    assert modal_cs321m._absolute_output_anchor(inside_repo) is None
    # Absolute outside REPO_ROOT -> returns the path as anchor.
    anchor = modal_cs321m._absolute_output_anchor(outside_repo)
    assert anchor is not None
    assert str(anchor) == outside_repo

    # Full round-trip on the writer side: synthesize an artifacts payload
    # mimicking what collect_small_artifacts would emit for an absolute
    # output_dir, then write it via write_returned_artifacts with the same
    # output_dir, and verify files land at the operator's absolute path.
    artifacts = {
        "paper_exports/csli.json": {
            "encoding": "text",
            "content": json.dumps({"test": "round_trip"}),
        },
        "paper_exports/audit_card.md": {
            "encoding": "text",
            "content": "# round-trip test card\n",
        },
    }
    modal_cs321m.write_returned_artifacts(
        artifacts, output_dir=outside_repo
    )
    landed_csli = tmp_path / "run42" / "paper_exports" / "csli.json"
    landed_md = tmp_path / "run42" / "paper_exports" / "audit_card.md"
    assert landed_csli.exists(), (
        f"Round-trip failed: {landed_csli} not written. "
        f"Absolute --output-dir would silently drop artifacts."
    )
    assert json.loads(landed_csli.read_text()) == {"test": "round_trip"}
    assert landed_md.exists()
    assert "round-trip test card" in landed_md.read_text()

    # And the default behavior (output_dir=None) still anchors to REPO_ROOT.
    # We don't actually exercise the write here (would clobber real artifacts);
    # we verify only the anchor resolution.
    default_anchor = modal_cs321m._absolute_output_anchor(None)
    assert default_anchor is None  # writer falls back to REPO_ROOT
