"""Codex PR-review fix-round-4 regression suite (CX-1..CX-6, all CONFIRMED).

CX-1: duplicate rights-inventory path rows fail closed (no last-row-wins).
CX-2: sentinel-coded summary codes each timeout stop AS its horizon.
CX-3: estimand horizon reconciliation covers per-arm-horizon records.
CX-4: canonical_run_id is a single path component inside the runs root.
CX-5: validate_profile gates schema_version before the create-once publish.
CX-6: anchored ledger/rights hash+parse share ONE bounded read.
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import ledger as ledger_mod
from reproducibility.colm_aims_2026 import pairing, schema, verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    build_package,
    colm_no_network,
    expected_estimand_digest,
    keyset_sha256,
    make_ledger,
    make_profile,
    make_record,
    make_rights,
    repo_head_commit,
    rewrite_json,
    standard_records,
)


def _run_release(pkg):
    return verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path,
    )


def _failing(report):
    return [leg for leg in report.legs if leg.get("outcome") == "FAIL"]


# ---------------------------------------------------------------------------
# CX-1 (P1): duplicate rights-inventory path rows fail closed
# ---------------------------------------------------------------------------


def _dup_rights(first_status: str, second_status: str) -> dict:
    rights = make_rights(paths=[])
    rights["paths"] = [
        {
            "path": "profile.json",
            "status": first_status,
            "upstream_terms_basis": "NAQT proprietary/excluded"
            if "RESTRICTED" in first_status
            else "synthetic test fixture generated in-repo",
        },
        {
            "path": "profile.json",
            "status": second_status,
            "upstream_terms_basis": "synthetic test fixture generated in-repo"
            if "ALLOWED" in second_status
            else "NAQT proprietary/excluded",
        },
    ]
    return rights


@pytest.mark.parametrize(
    "first,second",
    [
        ("VERIFIED_RESTRICTED", "VERIFIED_ALLOWED"),  # the laundering order
        ("VERIFIED_ALLOWED", "VERIFIED_RESTRICTED"),  # order-independent
    ],
    ids=["restricted_then_allowed", "allowed_then_restricted"],
)
def test_duplicate_rights_rows_fail_closed_naming_path(first, second):
    # CX-1 [R-026]: duplicate path rows are a typed error naming the path in
    # BOTH orders — a trailing ALLOWED row can never shadow RESTRICTED.
    rights = _dup_rights(first, second)
    with pytest.raises(ledger_mod.RightsError) as exc:
        ledger_mod.check_rights_release(rights, ["profile.json"])
    assert "profile.json" in str(exc.value)
    assert "duplicate" in str(exc.value).lower()
    # The inventory validator (the ingestion boundary) rejects it directly too.
    with pytest.raises(ledger_mod.RightsError):
        ledger_mod.validate_rights_inventory(rights)


def test_unique_rights_rows_still_pass():
    # CX-1 guard: the unique-row inventory is unaffected.
    rights = make_rights()
    ledger_mod.check_rights_release(
        rights, ["profile.json", "records.jsonl", "presentation_manifest.json"]
    )


def test_duplicate_rights_rows_fail_release_end_to_end(tmp_path: Path):
    # CX-1 [R-026]: end to end — a package whose rights inventory carries
    # RESTRICTED-then-ALLOWED duplicate rows for a tree file FAILs release
    # (previously the trailing ALLOWED row won and release PASSed).
    def add_dup(rights):
        rights["paths"].append(
            {
                "path": "profile.json",
                "status": "VERIFIED_RESTRICTED",
                "upstream_terms_basis": "NAQT proprietary/excluded",
            }
        )
        rights["paths"].append(
            {
                "path": "profile.json",
                "status": "VERIFIED_ALLOWED",
                "upstream_terms_basis": "synthetic test fixture generated in-repo",
            }
        )

    pkg = build_package(tmp_path, rights_mutator=add_dup)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg for leg in _failing(report) if leg.get("leg_id") == "rights_release"
    ]
    assert legs and "profile.json" in json.dumps(legs[0].get("observed"))


# ---------------------------------------------------------------------------
# CX-2 (P1): sentinel-coded summary codes timeouts AS the horizon
# ---------------------------------------------------------------------------


def _over_horizon_records() -> list[dict]:
    # stop>horizon on BOTH arms across the set: itm-a mc=9 (horizon 6),
    # itm-c ref=9. Complete pairs; horizon-coded shifts:
    #   a: min(9,6)-2 = 4;  b: 1-3 = -2;  c: 2-min(9,6) = -4.
    return [
        make_record("itm-a", 9, 2),
        make_record("itm-b", 1, 3),
        make_record("itm-c", 2, 9),
    ]


def _over_horizon_cell_fixup(sentinel_summary):
    counts = {
        "n_both_finite": 1,
        "n_mc_finite_ref_timeout": 1,
        "n_mc_timeout_ref_finite": 1,
        "n_both_timeout": 0,
        "n_complete": 3,
        "n_excluded_or_unpaired": 0,
        "exclusion_reason_counts": {},
        "n_pairing_population": 3,
        "n_mc_timeout": 1,
        "n_ref_timeout": 1,
    }

    def mut(profile):
        cell = profile["cells"][0]
        cell["counts"] = counts
        cell["rates"] = {
            "rate_both_finite": 1 / 3,
            "rate_mc_finite_ref_timeout": 1 / 3,
            "rate_mc_timeout_ref_finite": 1 / 3,
            "rate_both_timeout": 0.0,
        }
        cell["timing_summary_finite_only"] = {
            "conditional_on": "n_both_finite",
            "estimand": "signed_index_shift_mc_minus_ref",
            "n": 1,
            "signed_index_mean": -2.0,
            "signed_index_median": -2.0,
            "absolute_index_mean": 2.0,
            "absolute_index_median": 2.0,
        }
        cell["timing_summary_sentinel_coded_historical"] = sentinel_summary
        cell.pop("interval", None)
        keys = ["itm-a", "itm-b", "itm-c"]
        cell["complete_pair_keys"] = sorted(keys)
        cell["excluded_keys"] = []
        cell["pairing_population_keyset_sha256"] = keyset_sha256(keys)

    return mut


def test_horizon_coded_artifact_validates_with_over_horizon_stops():
    # CX-2 [R-006]: the artifact whose sentinel summary was computed WITH the
    # recorded timeout_coded_as_horizon convention validates — shifts
    # (4, -2, -4) -> mean -2/3, median -2.
    sentinel = {
        "convention": "timeout_coded_as_horizon",
        "n": 3,
        "signed_index_mean": -2.0 / 3.0,
        "signed_index_median": -2.0,
    }
    profile = make_profile(_over_horizon_records())
    _over_horizon_cell_fixup(sentinel)(profile)
    pairing.validate_cell(profile["cells"][0], _over_horizon_records())


def test_raw_coded_artifact_fails_with_over_horizon_stops():
    # CX-2 [R-006]: the RAW-stop summary (shifts 7, -2, -7 -> mean -2/3?? no:
    # raw shifts are (9-2, 1-3, 2-9) = (7, -2, -7) -> mean -2/3, median -2 —
    # deliberately choose a raw statistic that DIFFERS: use median of raw
    # sorted (-7, -2, 7) = -2 equals... so assert on the mean of a 2-record
    # variant instead, where raw and coded diverge decisively.
    records = [make_record("itm-a", 9, 2), make_record("itm-b", 1, 3)]
    raw_sentinel = {
        "convention": "timeout_coded_as_horizon",
        "n": 2,
        "signed_index_mean": 2.5,  # raw: (9-2, 1-3) -> (7, -2) -> 2.5
        "signed_index_median": 2.5,
    }
    profile = make_profile(records)
    cell = profile["cells"][0]
    # itm-a: mc=9 (timeout at horizon 6) / ref=2 (finite) -> mc_timeout_ref_finite.
    cell["counts"] = {
        "n_both_finite": 1,
        "n_mc_finite_ref_timeout": 0,
        "n_mc_timeout_ref_finite": 1,
        "n_both_timeout": 0,
        "n_complete": 2,
        "n_excluded_or_unpaired": 0,
        "exclusion_reason_counts": {},
        "n_pairing_population": 2,
        "n_mc_timeout": 1,
        "n_ref_timeout": 0,
    }
    cell["rates"] = {
        "rate_both_finite": 0.5,
        "rate_mc_finite_ref_timeout": 0.0,
        "rate_mc_timeout_ref_finite": 0.5,
        "rate_both_timeout": 0.0,
    }
    cell["timing_summary_finite_only"] = {
        "conditional_on": "n_both_finite",
        "estimand": "signed_index_shift_mc_minus_ref",
        "n": 1,
        "signed_index_mean": -2.0,
        "signed_index_median": -2.0,
        "absolute_index_mean": 2.0,
        "absolute_index_median": 2.0,
    }
    cell["timing_summary_sentinel_coded_historical"] = raw_sentinel
    cell.pop("interval", None)
    keys = ["itm-a", "itm-b"]
    cell["complete_pair_keys"] = sorted(keys)
    cell["excluded_keys"] = []
    cell["pairing_population_keyset_sha256"] = keyset_sha256(keys)
    with pytest.raises(pairing.RateError):
        pairing.validate_cell(cell, records)
    # The horizon-coded variant of the SAME cell validates:
    # coded shifts (min(9,6)-2, 1-3) = (4, -2) -> mean 1.0, median 1.0.
    cell["timing_summary_sentinel_coded_historical"] = {
        "convention": "timeout_coded_as_horizon",
        "n": 2,
        "signed_index_mean": 1.0,
        "signed_index_median": 1.0,
    }
    pairing.validate_cell(cell, records)


def test_finite_only_and_interval_paths_have_no_raw_stop_exposure():
    # CX-2 side-check: both-finite selection EXCLUDES any stop >= horizon, so
    # the finite-only summary and the interval recompute never see a raw
    # over-horizon stop — identical output whether the timeout stop is 9 or 6.
    records_raw = [make_record("itm-a", 9, 2), make_record("itm-b", 1, 3)]
    records_at_horizon = [make_record("itm-a", 6, 2), make_record("itm-b", 1, 3)]
    assert pairing.finite_only_timing_summary(
        records_raw
    ) == pairing.finite_only_timing_summary(records_at_horizon)
    spec = {
        "procedure": "percentile_bootstrap",
        "draw_count": 50,
        "resampling_seeds": [1],
        "statistic": "signed_index_mean",
    }
    assert (
        pairing.recompute_interval(records_raw, spec)["ci"]
        == pairing.recompute_interval(records_at_horizon, spec)["ci"]
    )


# ---------------------------------------------------------------------------
# CX-3 (P1): per-arm-horizon records reach the estimand horizon check
# ---------------------------------------------------------------------------


def _per_arm_record(key: str, mc: int, ref: int) -> dict:
    return {
        "item_key": key,
        "mc_trajectory_horizon": 6,
        "ref_trajectory_horizon": 6,
        "mc_stop_step": mc,
        "ref_stop_step": ref,
    }


def _per_arm_package(tmp_path: Path, estimand_horizon: int):
    records = [
        _per_arm_record("itm-0001", 1, 3),
        _per_arm_record("itm-0002", 2, 2),
        _per_arm_record("itm-0003", 5, 1),
    ]

    def fixup(profile):
        cell = profile["cells"][0]
        cell["counts"] = {
            "n_both_finite": 3,
            "n_mc_finite_ref_timeout": 0,
            "n_mc_timeout_ref_finite": 0,
            "n_both_timeout": 0,
            "n_complete": 3,
            "n_excluded_or_unpaired": 0,
            "exclusion_reason_counts": {},
            "n_pairing_population": 3,
            "n_mc_timeout": 0,
            "n_ref_timeout": 0,
        }
        cell["rates"] = {
            "rate_both_finite": 1.0,
            "rate_mc_finite_ref_timeout": 0.0,
            "rate_mc_timeout_ref_finite": 0.0,
            "rate_both_timeout": 0.0,
        }
        cell["timing_summary_finite_only"] = {
            "conditional_on": "n_both_finite",
            "estimand": "signed_index_shift_mc_minus_ref",
            "n": 3,
            "signed_index_mean": 0.6666666666666666,
            "signed_index_median": 0.0,
            "absolute_index_mean": 2.0,
            "absolute_index_median": 2.0,
        }
        cell["timing_summary_sentinel_coded_historical"] = {
            "convention": "timeout_coded_as_horizon",
            "n": 3,
            "signed_index_mean": 0.6666666666666666,
            "signed_index_median": 0.0,
        }
        cell["interval"] = {
            "procedure": "percentile_bootstrap",
            "draw_count": 100,
            "resampling_seeds": [1],
            "statistic": "signed_index_mean",
            "ci": [-1.6833333333333331, 4.0],
        }
        keys = ["itm-0001", "itm-0002", "itm-0003"]
        cell["complete_pair_keys"] = sorted(keys)
        cell["excluded_keys"] = []
        cell["pairing_population_keyset_sha256"] = keyset_sha256(keys)
        cell["estimand"]["timeout_parameters"]["trajectory_horizon"] = (
            estimand_horizon
        )
        cell["estimand_digest"] = expected_estimand_digest(cell["estimand"])

    return build_package(tmp_path, records=records, profile_mutator=fixup)


def test_per_arm_horizon_fabricated_estimand_horizon_fails(tmp_path: Path):
    # CX-3 [R-011]: records carrying only equal per-arm horizons no longer
    # yield an EMPTY authoritative set — the fabricated estimand horizon (99)
    # FAILs the estimand_reconciliation leg.
    pkg = _per_arm_package(tmp_path, estimand_horizon=99)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "estimand_reconciliation"
    ]
    assert legs, "fabricated horizon escaped reconciliation via per-arm records"
    assert "99" in json.dumps(legs[0].get("observed"))


def test_per_arm_horizon_honest_package_passes(tmp_path: Path):
    # CX-3 guard: the honest per-arm-horizon package (estimand horizon 6)
    # PASSes release — per-arm horizons are an explicitly allowed record form.
    pkg = _per_arm_package(tmp_path, estimand_horizon=6)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS


# ---------------------------------------------------------------------------
# CX-4 (P2): canonical_run_id path-component gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "escape_id", ["../other-package", "/etc", "a/b", ".."],
    ids=["dotdot_rel", "absolute", "nested", "bare_dotdot"],
)
def test_canonical_run_id_traversal_refused(tmp_path: Path, escape_id):
    # CX-4 [R-039]: traversal/absolute pointers are refused with a typed
    # error BEFORE any directory is consulted.
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    outside = tmp_path / "other-package"
    outside.mkdir()
    (outside / "x.json").write_text("{}", encoding="utf-8")
    doc = make_ledger(
        source_commit=repo_head_commit(), canonical_run_id=escape_id
    )
    with pytest.raises(schema.ColmAimsError) as exc:
        verifier.resolve_canonical_package(runs_root, doc)
    assert "single path component" in str(exc.value)


def test_canonical_run_id_honest_still_resolves(tmp_path: Path):
    # CX-4 guard: the honest single-component id resolves as before.
    runs_root = tmp_path / "runs"
    slot = runs_root / "run-0001"
    slot.mkdir(parents=True)
    (slot / "profile.json").write_text("{}", encoding="utf-8")
    doc = make_ledger(
        source_commit=repo_head_commit(), canonical_run_id="run-0001"
    )
    assert verifier.resolve_canonical_package(runs_root, doc) == slot


# ---------------------------------------------------------------------------
# CX-5 (P2): validate_profile gates schema_version before publish
# ---------------------------------------------------------------------------


def test_write_profile_refuses_unsupported_version_nothing_published(
    tmp_path: Path,
):
    # CX-5 [R-020/R-016]: an unsupported schema_version is refused BEFORE the
    # create-once publish — no immutable unloadable artifact.
    target = tmp_path / "profile.json"
    profile = make_profile()
    profile["schema_version"] = 99
    with pytest.raises(schema.SchemaValidationError) as exc:
        schema.write_profile(target, profile)
    assert "99" in str(exc.value) and "schema_version" in str(exc.value)
    assert not target.exists(), "nothing may be published on refusal"


def test_validate_profile_rejects_unsupported_version_directly():
    # CX-5: the validator surface itself gates the version (first check).
    profile = make_profile()
    profile["schema_version"] = 0
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


def test_write_profile_supported_version_still_publishes(tmp_path: Path):
    # CX-5 guard: the supported version publishes and round-trips the loader.
    target = tmp_path / "profile.json"
    profile = make_profile()
    schema.write_profile(target, profile)
    assert schema.load_artifact(target)["schema_version"] == 1


# ---------------------------------------------------------------------------
# CX-6 (P2): anchored ledger/rights hash+parse share one bounded read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sidecar", ["ledger.json", "rights.json"])
def test_oversized_sidecar_is_typed_leg_refusal_not_full_read(
    tmp_path: Path, sidecar
):
    # CX-6 [R-020]: a sparse regular file above MAX_ARTIFACT_BYTES at the
    # anchored ledger/rights path yields a typed size refusal via the bounded
    # reader's fstat gate (never a full unbounded read_bytes), collected as a
    # leg — the run still reaches a verdict.
    pkg = build_package(tmp_path)
    target = pkg.root / sidecar
    with open(target, "wb") as fh:
        fh.truncate(schema.MAX_ARTIFACT_BYTES + 1)  # sparse: instant, ~0 disk
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    surface = json.dumps(_failing(report))
    assert "exceeds the maximum admissible" in surface
    assert report.receipt_path is not None  # verdict reached, no abort/OOM


def test_sidecar_hashes_route_through_bounded_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # CX-6 class check: the ledger and rights hash computations consume the
    # bounded reader's bytes (single read; no Path.read_bytes side channel).
    pkg = build_package(tmp_path)
    seen: list[str] = []
    real = schema.read_regular_file_bytes

    def recorder(path, **kwargs):
        seen.append(Path(path).name)
        return real(path, **kwargs)

    monkeypatch.setattr(schema, "read_regular_file_bytes", recorder)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS
    assert "ledger.json" in seen
    assert "rights.json" in seen
    assert "expectations.json" in seen  # MA-HI-001 route, still bounded


def test_fifo_sidecar_is_leg_refusal_not_hang(tmp_path: Path):
    # CX-6/MA-HI-001: an irregular (FIFO) file at the anchored ledger path is
    # a collected leg refusal — the is_file() gate treats a non-regular file
    # as not-a-frozen-ledger (MISSING_EXPECTATION) and the bounded reader
    # backstops any path that slips past; either way the run reaches a
    # verdict with the anchor_ledger leg failing, never a blocking read.
    import os as _os

    if not hasattr(_os, "mkfifo"):
        pytest.skip("platform without os.mkfifo")
    pkg = build_package(tmp_path)
    pkg.ledger_path.unlink()
    _os.mkfifo(pkg.ledger_path)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    failing_ids = {leg.get("leg_id") for leg in _failing(report)}
    assert "anchor_ledger" in failing_ids
    surface = json.dumps(_failing(report))
    assert "regular file" in surface or "absent" in surface
    assert report.receipt_path is not None  # verdict reached, no hang
