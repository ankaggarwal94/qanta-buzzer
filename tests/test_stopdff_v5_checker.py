"""Checker + negative mutation suite (synthetic fixtures)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import checker, selftest  # noqa: E402


def test_valid_package_passes(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    res = checker.validate_run(
        built["run_root"], backend="modal", adapter_bundle=built["adapter_bundle"],
        require_final_profile=False, require_package=True,
    )
    assert res.passed, res.errors


def test_validate_adapter_ok(tmp_path):
    built = selftest.build_valid_package(tmp_path)
    res = checker.validate_adapter(built["adapter_bundle"])
    assert res.passed, res.errors


def test_negative_mutation_suite(tmp_path):
    ok, results = selftest.run_self_test(tmp_path)
    failures = [r for r in results if not r["ok"]]
    assert ok, f"mutations not rejected: {[(r['mutation'], r['errors']) for r in failures]}"
    # sanity: we exercised a broad battery
    assert len(results) >= 20


def test_validate_spec_placeholder_rejected(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text('{"id":"x","identity":{"kind":"run_spec","profile_name":"stopdff_bucketed_dp_paired_v2",'
                    '"identity":{"adapter_bundle_id":"<ADAPTER_ID>"}}}', encoding="utf-8")
    res = checker.validate_spec(spec, require_final_profile=False)
    assert not res.passed
