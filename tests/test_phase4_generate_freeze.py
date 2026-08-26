"""Regression tests for the source-only Phase-4 freeze generator."""
from __future__ import annotations

import shutil
from pathlib import Path

from reproducibility.colm_aims_2026 import phase4_generate_freeze, qa012

from tests._colm_aims_v2_helpers import REPO_ROOT


def test_generate_qa012_fixtures_round_trips_committed_authority_set(
    tmp_path: Path,
) -> None:
    """Regeneration must remain directly admissible to the QA-012 verifier."""
    committed = REPO_ROOT / "tests" / "fixtures" / "qa012_item10"
    item10 = tmp_path / "item10_reachable_comparator_prototype"
    generated = tmp_path / "generated"
    item10.mkdir()
    for basename in phase4_generate_freeze.QA012_HIT_FILES:
        shutil.copyfile(committed / basename, item10 / basename)

    phase4_generate_freeze.generate_qa012_fixtures(item10, generated)

    expected_names = {path.name for path in committed.iterdir() if path.is_file()}
    assert {path.name for path in generated.iterdir() if path.is_file()} == expected_names
    for name in expected_names:
        assert (generated / name).read_bytes() == (committed / name).read_bytes()
    assert qa012.authority_hit_fixtures_verified(
        qa012.load_authority_manifest(), generated
    )
