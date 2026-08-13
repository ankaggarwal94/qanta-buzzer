"""Direct tests for the local adapter-stage adoption gate.

M-V3-01 (testing, PR #30 round 3): ``_load_valid_adapter_stage`` is the sole
build-time gate for adopting a crash-orphaned adapter stage (stage present, no
checkpoint), yet every test that reaches it monkeypatches it. These tests drive
the REAL gate against a real checker-valid bundle: it accepts a matching stage
and fail-closes on a checker-invalid stage and on each upstream-id binding
mismatch -- killing an inverted ``result.passed`` clause or a binding-key typo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_stopdff_v5_local as local_runner  # noqa: E402
from scripts.stopdff_v5 import checker, selftest  # noqa: E402


def _bundle_and_ids(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    bundle = Path(built["adapter_bundle"])
    manifest = checker.load_json(bundle / "manifest.json")
    identity = manifest["identity"]
    ids = {
        "source_id": identity["source_manifest_id"],
        "raw_id": identity["raw_input_bundle_id"],
        "model_id": identity["model_snapshot_id"],
    }
    return bundle, manifest, ids


def test_load_valid_adapter_stage_accepts_matching_stage(tmp_path):
    bundle, manifest, ids = _bundle_and_ids(tmp_path)
    assert local_runner._load_valid_adapter_stage(bundle, **ids) == manifest


def test_load_valid_adapter_stage_rejects_checker_invalid_stage(tmp_path):
    bundle, _manifest, ids = _bundle_and_ids(tmp_path)
    # A symlinked bundle root is checker-invalid (validate_adapter fails closed)
    # while manifest.json still reads through the link -- isolates the
    # ``not result.passed`` clause with the identity bindings all correct.
    link = tmp_path / "adapter_bundle_link"
    link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(ValueError, match="invalid or bound to different inputs"):
        local_runner._load_valid_adapter_stage(link, **ids)


@pytest.mark.parametrize("wrong_key", ["source_id", "raw_id", "model_id"])
def test_load_valid_adapter_stage_rejects_upstream_id_mismatch(tmp_path, wrong_key):
    bundle, _manifest, ids = _bundle_and_ids(tmp_path)
    tampered = {**ids, wrong_key: "0" * 64}
    with pytest.raises(ValueError, match="invalid or bound to different inputs"):
        local_runner._load_valid_adapter_stage(bundle, **tampered)
