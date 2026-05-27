"""Regression test: committed smoke artifacts must not embed absolute home paths.

PR #14's contract (body L37-39) is: "Writes ``requested_mc_path`` /
``resolved_mc_path`` as repo-relative paths in baseline / evaluation
artifacts via a new ``scripts._common.project_relative`` helper, so
committed smoke artifacts no longer leak the author's home directory."

The first round of this guarantee covered ``requested_mc_path`` /
``resolved_mc_path`` only. A subsequent rerun surfaced two missed
producer sites that bypassed the helper (``likelihood_reference_path``
in ``scripts/train_ppo.py`` and a path interpolation in
``scripts/compute_csli.py`` -- now patched). This test prevents the
*class* of bug, not just those two instances: any future producer
field that smuggles an absolute user-directory path into a committed
smoke JSON will fail this scan.

Test parameterizes over every JSON under ``artifacts/smoke/`` (a single
parametrize call so each file gets its own pytest node id; one failure
does not mask the rest). The scan is intentionally broad -- it looks
for the three platform-specific user-directory roots, not just the
current developer's home -- so the test is portable across contributors
and CI hosts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SMOKE = _REPO_ROOT / "artifacts" / "smoke"

_JSONS = sorted(_SMOKE.rglob("*.json")) if _SMOKE.exists() else []

_LEAK_NEEDLES = (
    "/Users/",       # macOS
    "/home/",        # Linux
    "C:\\Users\\",   # Windows
    "C:/Users/",     # Windows forward-slash variant
)


@pytest.mark.parametrize(
    "json_path",
    _JSONS,
    ids=[str(p.relative_to(_SMOKE)) for p in _JSONS] or ["<no_smoke_artifacts>"],
)
def test_no_absolute_user_paths(json_path: Path) -> None:
    if not _JSONS:
        pytest.skip(
            "No smoke artifacts present; run scripts/build_mc_dataset.py "
            "(and the rest of the smoke pipeline) before exercising this guard."
        )
    text = json_path.read_text(encoding="utf-8")
    hits: list[tuple[str, int]] = []
    for needle in _LEAK_NEEDLES:
        idx = text.find(needle)
        if idx >= 0:
            hits.append((needle, idx))
    assert not hits, (
        f"{json_path.relative_to(_SMOKE)}: leaked user-directory path(s) "
        f"{hits}. Route the offending value through "
        f"``scripts._common.project_relative`` (or compute_csli's "
        f"``_display_path``) at the producer call site so committed smoke "
        f"JSONs stay machine-portable. Contract: PR #14 body L37-39."
    )


def test_smoke_artifact_inventory_nonempty_in_ci() -> None:
    """Tripwire: if a CI run produces zero smoke JSONs, the smoke pipeline
    silently degraded -- this catches that before it masks
    ``test_no_absolute_user_paths`` skipping silently."""
    if not _SMOKE.exists():
        pytest.skip(
            "artifacts/smoke/ does not exist; smoke pipeline has not run "
            "in this checkout. Run scripts/build_mc_dataset.py --smoke + "
            "the rest of the smoke pipeline first."
        )
    artifact_count = len(_JSONS)
    assert artifact_count > 0, (
        "artifacts/smoke/ exists but contains zero JSON files. The smoke "
        "pipeline appears to have produced no artifacts. Check the "
        "outputs of build_mc_dataset, run_baselines, train_ppo, "
        "evaluate_all in artifacts/smoke/."
    )


def test_committed_smoke_jsons_parse_cleanly() -> None:
    """Tripwire: every committed smoke JSON must round-trip through
    json.loads without error. Catches accidental truncation, encoding
    issues, or non-JSON content under artifacts/smoke/*.json."""
    if not _JSONS:
        pytest.skip("No smoke artifacts present")
    failures: list[tuple[Path, str]] = []
    for p in _JSONS:
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append((p.relative_to(_SMOKE), str(exc)))
    assert not failures, f"Smoke JSONs failed to parse: {failures}"
