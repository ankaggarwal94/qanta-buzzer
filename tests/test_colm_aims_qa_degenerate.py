"""QA fix-round-1 regression suite — degenerate-but-honest evaluations
(QA-007).

EmptyEvaluationError is reserved for the one condition R-006/R-012 name as a
typed error: ``n_pairing_population == 0``. Every other degenerate-but-honest
shape (all-timeout, all-excluded, zero both-finite, single-record) must
produce a VERDICT — null finite-only summaries validated as legs — never an
abort.
Spec: .correctless/specs/camera-ready-aims-evidence.md (R-006/R-012)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import schema, verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    EXIT_PASS,
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    VERDICT_SOURCE_PASS,
    build_package,
    cli_args_for,
    colm_no_network,
    keyset_sha256,
    make_record,
    run_cli,
)


def _null_finite_summary() -> dict:
    return {
        "conditional_on": "n_both_finite",
        "estimand": "signed_index_shift_mc_minus_ref",
        "n": 0,
        "signed_index_mean": None,
        "signed_index_median": None,
        "absolute_index_mean": None,
        "absolute_index_median": None,
    }


def _cell_fixup(counts, rates, finite, sentinel, complete, excluded):
    def mut(profile):
        cell = profile["cells"][0]
        cell["counts"] = counts
        cell["rates"] = rates
        cell["timing_summary_finite_only"] = finite
        cell["timing_summary_sentinel_coded_historical"] = sentinel
        cell.pop("interval", None)  # no interval over zero both-finite pairs
        cell["complete_pair_keys"] = sorted(complete)
        cell["excluded_keys"] = sorted(excluded)
        cell["pairing_population_keyset_sha256"] = keyset_sha256(
            list(complete) + list(excluded)
        )

    return mut


def _all_timeout_package(base: Path):
    records = [make_record("itm-0001", 6, 6), make_record("itm-0002", 6, 6)]
    counts = {
        "n_both_finite": 0,
        "n_mc_finite_ref_timeout": 0,
        "n_mc_timeout_ref_finite": 0,
        "n_both_timeout": 2,
        "n_complete": 2,
        "n_excluded_or_unpaired": 0,
        "exclusion_reason_counts": {},
        "n_pairing_population": 2,
        "n_mc_timeout": 2,
        "n_ref_timeout": 2,
    }
    rates = {
        "rate_both_finite": 0.0,
        "rate_mc_finite_ref_timeout": 0.0,
        "rate_mc_timeout_ref_finite": 0.0,
        "rate_both_timeout": 1.0,
    }
    sentinel = {
        "convention": "timeout_coded_as_horizon",
        "n": 2,
        "signed_index_mean": 0.0,
        "signed_index_median": 0.0,
    }
    return build_package(
        base,
        records=records,
        profile_mutator=_cell_fixup(
            counts,
            rates,
            _null_finite_summary(),
            sentinel,
            ["itm-0001", "itm-0002"],
            [],
        ),
    )


def _all_excluded_package(base: Path):
    records = [
        make_record(
            "itm-0001", None, None, excluded=True,
            exclusion_reason="MALFORMED_STOP",
        ),
        make_record(
            "itm-0002", None, None, excluded=True,
            exclusion_reason="MALFORMED_STOP",
        ),
    ]
    counts = {
        "n_both_finite": 0,
        "n_mc_finite_ref_timeout": 0,
        "n_mc_timeout_ref_finite": 0,
        "n_both_timeout": 0,
        "n_complete": 0,
        "n_excluded_or_unpaired": 2,
        "exclusion_reason_counts": {"MALFORMED_STOP": 2},
        "n_pairing_population": 2,
        "n_mc_timeout": 0,
        "n_ref_timeout": 0,
    }
    rates = {
        "rate_both_finite": None,
        "rate_mc_finite_ref_timeout": None,
        "rate_mc_timeout_ref_finite": None,
        "rate_both_timeout": None,
    }
    sentinel = {
        "convention": "timeout_coded_as_horizon",
        "n": 0,
        "signed_index_mean": None,
        "signed_index_median": None,
    }
    return build_package(
        base,
        records=records,
        profile_mutator=_cell_fixup(
            counts,
            rates,
            _null_finite_summary(),
            sentinel,
            [],
            ["itm-0001", "itm-0002"],
        ),
    )


def _zero_finite_mixed_package(base: Path):
    records = [make_record("itm-0001", 2, 6), make_record("itm-0002", 6, 3)]
    counts = {
        "n_both_finite": 0,
        "n_mc_finite_ref_timeout": 1,
        "n_mc_timeout_ref_finite": 1,
        "n_both_timeout": 0,
        "n_complete": 2,
        "n_excluded_or_unpaired": 0,
        "exclusion_reason_counts": {},
        "n_pairing_population": 2,
        "n_mc_timeout": 1,
        "n_ref_timeout": 1,
    }
    rates = {
        "rate_both_finite": 0.0,
        "rate_mc_finite_ref_timeout": 0.5,
        "rate_mc_timeout_ref_finite": 0.5,
        "rate_both_timeout": 0.0,
    }
    # Timeout-coded shifts: (2-6, 6-3) = (-4, 3) -> mean -0.5, median -0.5.
    sentinel = {
        "convention": "timeout_coded_as_horizon",
        "n": 2,
        "signed_index_mean": -0.5,
        "signed_index_median": -0.5,
    }
    return build_package(
        base,
        records=records,
        profile_mutator=_cell_fixup(
            counts,
            rates,
            _null_finite_summary(),
            sentinel,
            ["itm-0001", "itm-0002"],
            [],
        ),
    )


DEGENERATE_FAMILY = [
    ("all_timeout", _all_timeout_package),
    ("all_excluded", _all_excluded_package),
    ("zero_finite_mixed", _zero_finite_mixed_package),
]


@pytest.mark.parametrize(
    "name,builder", DEGENERATE_FAMILY, ids=[n for n, _ in DEGENERATE_FAMILY]
)
def test_degenerate_honest_source_passes(tmp_path: Path, name, builder):
    # QA-007 [R-006]: honest degenerate artifacts VERIFY — null finite-only
    # summaries are validated as legs, never aborted.
    pkg = builder(tmp_path)
    report = verifier.run_verifier(
        pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
    )
    assert report.verdict == VERDICT_SOURCE_PASS, (name, report.legs)


@pytest.mark.parametrize(
    "name,builder", DEGENERATE_FAMILY, ids=[n for n, _ in DEGENERATE_FAMILY]
)
def test_degenerate_honest_release_reaches_a_verdict(
    tmp_path: Path, name, builder
):
    # QA-007 [R-012]: collect-don't-halt holds in release mode too — a
    # verdict and a receipt are produced (the honest packages pass).
    pkg = builder(tmp_path)
    report = verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path,
    )
    assert report.verdict == VERDICT_RELEASE_PASS, (name, report.legs)
    assert report.receipt_path is not None


def test_all_timeout_cli_exits_zero(tmp_path: Path):
    # QA-007: the audit's headline case end-to-end — exit code EQUALITY.
    pkg = _all_timeout_package(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "source"))
    assert proc.returncode == EXIT_PASS, proc.stderr[-300:]
    assert VERDICT_SOURCE_PASS in proc.stdout


def test_only_population_zero_escapes_collect_dont_halt(tmp_path: Path):
    # QA-007 class meta-test: across the degenerate family only the
    # explicitly empty population (n_pairing_population == 0) raises the
    # typed EmptyEvaluationError; every sibling produces a report.
    for index, (name, builder) in enumerate(DEGENERATE_FAMILY):
        pkg = builder(tmp_path / f"deg-{index}")
        report = verifier.run_verifier(
            pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
        )
        assert report.verdict in verifier.SOURCE_MODE_VERDICTS, name

    from tests.test_colm_aims_verifier_gates import _build_empty_eval_package

    empty_pkg = _build_empty_eval_package(tmp_path / "empty")
    with pytest.raises(schema.EmptyEvaluationError):
        verifier.run_verifier(
            empty_pkg.tree,
            mode="release",
            receipts_dir=empty_pkg.receipts_dir,
            expectations=empty_pkg.expectations_path,
        )
    assert list(empty_pkg.receipts_dir.iterdir()) == []


def test_zero_finite_summary_with_nonnull_statistics_fails_as_leg(
    tmp_path: Path,
):
    # QA-007 [R-006]: a zero-both-finite cell claiming numeric finite-only
    # statistics is a LEG failure (dishonest degenerate summary), and the
    # run still reaches a verdict rather than aborting.
    def lie(profile):
        summary = profile["cells"][0]["timing_summary_finite_only"]
        summary["signed_index_mean"] = 0.25
        summary["signed_index_median"] = 0.25
        summary["absolute_index_mean"] = 0.25
        summary["absolute_index_median"] = 0.25

    records = [make_record("itm-0001", 6, 6), make_record("itm-0002", 6, 6)]
    pkg = _all_timeout_package(tmp_path)
    # Rewrite the profile with the lie while keeping expectations consistent:
    # rebuild the package from scratch with both mutators composed.
    counts_pkg = build_package(
        tmp_path / "lie",
        records=records,
        profile_mutator=lambda p: (
            _cell_fixup(
                {
                    "n_both_finite": 0,
                    "n_mc_finite_ref_timeout": 0,
                    "n_mc_timeout_ref_finite": 0,
                    "n_both_timeout": 2,
                    "n_complete": 2,
                    "n_excluded_or_unpaired": 0,
                    "exclusion_reason_counts": {},
                    "n_pairing_population": 2,
                    "n_mc_timeout": 2,
                    "n_ref_timeout": 2,
                },
                {
                    "rate_both_finite": 0.0,
                    "rate_mc_finite_ref_timeout": 0.0,
                    "rate_mc_timeout_ref_finite": 0.0,
                    "rate_both_timeout": 1.0,
                },
                _null_finite_summary(),
                {
                    "convention": "timeout_coded_as_horizon",
                    "n": 2,
                    "signed_index_mean": 0.0,
                    "signed_index_median": 0.0,
                },
                ["itm-0001", "itm-0002"],
                [],
            )(p),
            lie(p),
        )[-1],
    )
    report = verifier.run_verifier(
        counts_pkg.tree, mode="source", receipts_dir=counts_pkg.receipts_dir
    )
    assert report.verdict == VERDICT_FAIL
    failing = [leg for leg in report.legs if leg.get("outcome") == "FAIL"]
    assert any("null statistics" in str(leg.get("observed")) for leg in failing)


def test_zero_finite_cell_carrying_interval_fails_as_leg(tmp_path: Path):
    # QA-007 [R-015]: an interval over zero both-finite pairs cannot
    # recompute — leg failure, not an abort.
    records = [make_record("itm-0001", 6, 6), make_record("itm-0002", 6, 6)]

    def keep_interval(profile):
        cell = profile["cells"][0]
        _cell_fixup(
            {
                "n_both_finite": 0,
                "n_mc_finite_ref_timeout": 0,
                "n_mc_timeout_ref_finite": 0,
                "n_both_timeout": 2,
                "n_complete": 2,
                "n_excluded_or_unpaired": 0,
                "exclusion_reason_counts": {},
                "n_pairing_population": 2,
                "n_mc_timeout": 2,
                "n_ref_timeout": 2,
            },
            {
                "rate_both_finite": 0.0,
                "rate_mc_finite_ref_timeout": 0.0,
                "rate_mc_timeout_ref_finite": 0.0,
                "rate_both_timeout": 1.0,
            },
            _null_finite_summary(),
            {
                "convention": "timeout_coded_as_horizon",
                "n": 2,
                "signed_index_mean": 0.0,
                "signed_index_median": 0.0,
            },
            ["itm-0001", "itm-0002"],
            [],
        )(profile)
        cell["interval"] = {
            "procedure": "percentile_bootstrap",
            "draw_count": 100,
            "resampling_seeds": [1],
            "statistic": "signed_index_mean",
            "ci": [0.0, 0.0],
        }

    pkg = build_package(tmp_path, records=records, profile_mutator=keep_interval)
    report = verifier.run_verifier(
        pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
    )
    assert report.verdict == VERDICT_FAIL
    failing = [leg for leg in report.legs if leg.get("outcome") == "FAIL"]
    assert any(
        "zero both-finite" in str(leg.get("observed")) for leg in failing
    )
    assert report.receipt_path is not None  # verdict reached, no abort


def test_population_zero_keyset_hash_helper_consistency(tmp_path: Path):
    # Guard for the fixture algebra itself: the empty-population package the
    # meta-test relies on hashes an empty key set.
    assert keyset_sha256([]) == hashlib.sha256(b"").hexdigest()
