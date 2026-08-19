"""RED suite — pair/censoring decomposition algebra.

Covers: R-005, R-006, R-007, R-008, R-009, R-010, R-011 (+ the
recompute_interval unit surface of R-015).
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import random

import pytest

from reproducibility.colm_aims_2026 import pairing, schema
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    STANDARD_COUNTS,
    STANDARD_RATES,
    STANDARD_TIMING_SUMMARY,
    TRAJECTORY_HORIZON,
    colm_no_network,
    expected_estimand_digest,
    expected_item_key,
    keyset_sha256,
    load_parity_golden,
    make_cell,
    make_estimand,
    make_profile,
    make_record,
    standard_records,
)


# ---------------------------------------------------------------------------
# R-005: exact count identities recomputed from per-item records
# ---------------------------------------------------------------------------


def test_recompute_counts_matches_hand_computed_oracle():
    # Tests R-005 [unit]: per-cell counts recompute from per-item records.
    # STANDARD_COUNTS is hand-computed in the helper, never derived from the
    # implementation under test.
    assert pairing.recompute_counts(standard_records()) == STANDARD_COUNTS


def test_recompute_counts_with_exclusions_matches_oracle():
    # Tests R-005 [unit]: exclusions enter n_excluded_or_unpaired and the
    # reason histogram, never any joint class.
    records = standard_records() + [
        make_record("itm-0007", None, 3, excluded=True, exclusion_reason="MALFORMED_STOP"),
        make_record("itm-0008", 1, 2, excluded=True, exclusion_reason="GRID_MISMATCH"),
    ]
    counts = pairing.recompute_counts(records)
    assert counts["n_complete"] == 6
    assert counts["n_excluded_or_unpaired"] == 2
    assert counts["n_pairing_population"] == 8
    assert counts["exclusion_reason_counts"] == {
        "MALFORMED_STOP": 1,
        "GRID_MISMATCH": 1,
    }


COUNT_FIELDS = sorted(k for k in STANDARD_COUNTS if k != "exclusion_reason_counts")


@pytest.mark.parametrize("field", COUNT_FIELDS)
@pytest.mark.parametrize("delta", [1, -1])
def test_plus_minus_one_mutation_of_any_count_fails(field, delta):
    # Tests R-005 [unit]: a +/-1 mutation of any recorded count fails the
    # exact identity check against records.
    counts = dict(STANDARD_COUNTS)
    counts["exclusion_reason_counts"] = dict(counts["exclusion_reason_counts"])
    counts[field] = counts[field] + delta
    with pytest.raises(pairing.CountIdentityError):
        pairing.check_count_identities(counts, standard_records())


def test_exclusion_reason_count_mutation_fails():
    # Tests R-005 [unit]: n_excluded_or_unpaired == sum(exclusion_reason_counts)
    # is enforced; inflating a reason count by 1 fails.
    records = standard_records() + [
        make_record("itm-0007", None, 3, excluded=True, exclusion_reason="MALFORMED_STOP"),
    ]
    counts = pairing.recompute_counts(records)
    counts["exclusion_reason_counts"] = dict(counts["exclusion_reason_counts"])
    counts["exclusion_reason_counts"]["MALFORMED_STOP"] += 1
    with pytest.raises(pairing.CountIdentityError):
        pairing.check_count_identities(counts, records)


def test_unmutated_counts_pass_identity_check():
    # Tests R-005 [unit]: the honest count block passes.
    pairing.check_count_identities(dict(STANDARD_COUNTS), standard_records())


# ---------------------------------------------------------------------------
# R-006: denominators, null-at-zero, sum-to-1, zero-population typed error
# ---------------------------------------------------------------------------


def test_rates_use_n_complete_denominator():
    # Tests R-006 [unit]: all four joint-class rates use n_complete.
    rates = pairing.compute_rates(STANDARD_COUNTS)
    assert rates["rate_both_finite"] == pytest.approx(3 / 6, abs=1e-12)
    assert rates["rate_mc_finite_ref_timeout"] == pytest.approx(1 / 6, abs=1e-12)
    assert rates["rate_mc_timeout_ref_finite"] == pytest.approx(1 / 6, abs=1e-12)
    assert rates["rate_both_timeout"] == pytest.approx(1 / 6, abs=1e-12)


def test_rates_null_when_n_complete_zero():
    # Tests R-006 [unit]: rates are null when n_complete is zero (population
    # nonzero: all excluded).
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
    rates = pairing.compute_rates(counts)
    assert all(rates[k] is None for k in (
        "rate_both_finite",
        "rate_mc_finite_ref_timeout",
        "rate_mc_timeout_ref_finite",
        "rate_both_timeout",
    ))


def test_rates_sum_to_one_within_declared_tolerance():
    # Tests R-006 [unit]: sum-to-1 enforced within the artifact's declared
    # tolerance — a 3e-9 violation against a 1e-9 tolerance fails.
    cell = make_cell()
    pairing.check_rates(cell)  # honest cell passes
    bad = make_cell()
    bad["rates"] = dict(STANDARD_RATES)
    bad["rates"]["rate_both_timeout"] = STANDARD_RATES["rate_both_timeout"] + 3e-9
    with pytest.raises(pairing.RateError):
        pairing.check_rates(bad)


def test_rates_with_wrong_denominator_fail():
    # Tests R-006 [unit]: rates computed over n_pairing_population instead of
    # n_complete are rejected (here they differ because of exclusions).
    records = standard_records() + [
        make_record("itm-0007", None, 3, excluded=True, exclusion_reason="MALFORMED_STOP"),
        make_record("itm-0008", 1, 2, excluded=True, exclusion_reason="GRID_MISMATCH"),
    ]
    counts = pairing.recompute_counts(records)
    keys = [r["item_key"] for r in records]
    cell = make_cell()
    cell["counts"] = counts
    cell["complete_pair_keys"] = sorted(r["item_key"] for r in standard_records())
    cell["excluded_keys"] = ["itm-0007", "itm-0008"]
    cell["pairing_population_keyset_sha256"] = keyset_sha256(keys)
    # Denominator mistake: population (8) instead of n_complete (6).
    cell["rates"] = {
        "rate_both_finite": 3 / 8,
        "rate_mc_finite_ref_timeout": 1 / 8,
        "rate_mc_timeout_ref_finite": 1 / 8,
        "rate_both_timeout": 1 / 8,
    }
    with pytest.raises(pairing.RateError):
        pairing.check_rates(cell)


def test_zero_pairing_population_cell_is_typed_error():
    # Tests R-006 [unit]: a cell declaring n_pairing_population == 0 is a
    # typed error (consistent with the empty-evaluation refusal), never a
    # trivially-passing cell.
    cell = make_cell()
    cell["counts"] = {k: (0 if not isinstance(v, dict) else {}) for k, v in STANDARD_COUNTS.items()}
    cell["rates"] = {k: None for k in STANDARD_RATES}
    cell["complete_pair_keys"] = []
    cell["excluded_keys"] = []
    cell["pairing_population_keyset_sha256"] = keyset_sha256([])
    with pytest.raises(schema.EmptyEvaluationError):
        pairing.validate_cell(cell, [])


def test_finite_only_summary_uses_exactly_n_both_finite():
    # Tests R-006 [unit]: the finite-only timing summary uses exactly
    # n_both_finite items and declares its conditional estimand.
    summary = pairing.finite_only_timing_summary(standard_records())
    assert summary["n"] == STANDARD_COUNTS["n_both_finite"]
    assert summary["conditional_on"] == "n_both_finite"
    assert summary["signed_index_mean"] == STANDARD_TIMING_SUMMARY["signed_index_mean"]
    assert summary["absolute_index_median"] == STANDARD_TIMING_SUMMARY["absolute_index_median"]


def test_sentinel_coded_summary_is_separately_named_and_never_pooled():
    # Tests R-006 [unit]: the retained sentinel-coded historical summary is
    # separately named; a finite-only summary carrying the sentinel-coded
    # convention (pooling the two) is rejected.
    good = make_cell()
    assert "timing_summary_sentinel_coded_historical" in good
    pairing.validate_cell(good, standard_records())
    bad = make_cell()
    bad["timing_summary_finite_only"] = dict(bad["timing_summary_finite_only"])
    bad["timing_summary_finite_only"]["convention"] = "timeout_coded_as_horizon"
    with pytest.raises((pairing.RateError, schema.SchemaValidationError, pairing.EstimandMismatchError)):
        pairing.validate_cell(bad, standard_records())


# ---------------------------------------------------------------------------
# R-007: zero-indexed timeout boundary; exclusions never imputed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stop,horizon,expected",
    [
        (0, 6, "finite"),               # lower boundary: 0 is finite
        (5, 6, "finite"),               # horizon - 1: finite (green side)
        (6, 6, "timeout"),              # stop == horizon: timeout (red side)
        (7, 6, "timeout"),              # stop > horizon: timeout
        (0, 1, "finite"),
        (1, 1, "timeout"),
    ],
)
def test_timeout_boundary_both_sides(stop, horizon, expected):
    # Tests R-007 [unit]: zero-indexed `0 <= stop_step < trajectory_horizon`
    # is finite; `stop_step >= trajectory_horizon` is timeout.
    assert pairing.classify_stop(stop, horizon) == expected


@pytest.mark.parametrize("bad_stop", [None, "3", -1, 3.5])
def test_malformed_stop_becomes_exclusion_never_imputed(bad_stop):
    # Tests R-007 [unit]: missing/malformed stops become exclusions with a
    # reason — never imputed stops.
    record = make_record("itm-0001", bad_stop, 3)
    result = pairing.classify_record(record)
    assert result["status"] == "excluded"
    assert result["exclusion_reason"] == "MALFORMED_STOP"
    assert "joint_class" not in result or result["joint_class"] is None


def test_grid_mismatch_becomes_exclusion_with_reason():
    # Tests R-007 [unit]: MC/reference grid mismatches become exclusions with
    # reason, never imputed.
    record = make_record("itm-0002", 1, 2)
    record["mc_trajectory_horizon"] = 6
    record["ref_trajectory_horizon"] = 8
    result = pairing.classify_record(record)
    assert result["status"] == "excluded"
    assert result["exclusion_reason"] == "GRID_MISMATCH"


def test_complete_record_joint_classes():
    # Tests R-007 [unit]: joint classification of well-formed records.
    assert pairing.classify_record(make_record("a", 1, 3))["joint_class"] == "both_finite"
    assert pairing.classify_record(make_record("b", 2, 6))["joint_class"] == "mc_finite_ref_timeout"
    assert pairing.classify_record(make_record("c", 6, 3))["joint_class"] == "mc_timeout_ref_finite"
    assert pairing.classify_record(make_record("d", 6, 6))["joint_class"] == "both_timeout"


def test_exclusions_never_enter_joint_classes():
    # Tests R-007 [unit]: an excluded record never contributes to a joint
    # class count (no imputation).
    records = standard_records() + [
        make_record("itm-0009", None, 0, excluded=True, exclusion_reason="MALFORMED_STOP"),
    ]
    counts = pairing.recompute_counts(records)
    assert counts["n_both_finite"] == STANDARD_COUNTS["n_both_finite"]
    assert counts["n_complete"] == STANDARD_COUNTS["n_complete"]
    assert counts["n_excluded_or_unpaired"] == 1


# ---------------------------------------------------------------------------
# R-008: key-set discipline
# ---------------------------------------------------------------------------


def test_key_sets_disjoint_union_and_hash_pass_on_honest_cell():
    # Tests R-008 [unit]: disjoint, duplicate-free, union == declared
    # population key set and hash.
    pairing.check_key_sets(make_cell(), standard_records())


def test_overlapping_complete_and_excluded_keys_fail():
    # Tests R-008 [unit]: complete and excluded key sets must be disjoint.
    cell = make_cell()
    cell["excluded_keys"] = ["itm-0001"]  # also a complete key
    with pytest.raises(pairing.KeySetError):
        pairing.check_key_sets(cell, standard_records())


def test_union_hash_mismatch_fails():
    # Tests R-008 [unit]: union must equal the declared population key-set hash.
    cell = make_cell()
    cell["pairing_population_keyset_sha256"] = keyset_sha256(["itm-9999"])
    with pytest.raises(pairing.KeySetError):
        pairing.check_key_sets(cell, standard_records())


def test_duplicate_pair_keys_fail_closed():
    # Tests R-008 [unit]: duplicate pair keys fail closed.
    records = standard_records() + [make_record("itm-0001", 2, 2)]
    with pytest.raises(pairing.KeySetError):
        pairing.recompute_counts(records)


def test_missing_exclusion_reason_recorded_unknown_not_inferred():
    # Tests R-008 [unit]: missing reasons are recorded UNKNOWN_NOT_INFERRED,
    # never guessed.
    record = make_record("itm-0010", None, None, excluded=True)
    result = pairing.classify_record(record)
    assert result["status"] == "excluded"
    assert result["exclusion_reason"] == "UNKNOWN_NOT_INFERRED"


def test_secondary_diagnostics_not_counted_in_reason_counts():
    # Tests R-008 [unit]: each excluded unit carries exactly one primary
    # reason; secondary diagnostics are not counted.
    records = standard_records() + [
        make_record(
            "itm-0011",
            None,
            3,
            excluded=True,
            exclusion_reason="MALFORMED_STOP",
            secondary_diagnostics=["GRID_MISMATCH"],
        ),
    ]
    counts = pairing.recompute_counts(records)
    assert counts["exclusion_reason_counts"] == {"MALFORMED_STOP": 1}


def test_item_key_derivation_is_pinned_and_rederivable():
    # Tests R-008 [unit]: the stable item-key derivation (hash function and
    # text normalization) is pinned so a third party can re-derive and match.
    # DECISION: itm-<first 16 hex of sha256(NFC(text))>.
    assert pairing.derive_item_key("synthetic item zero one") == expected_item_key(
        "synthetic item zero one"
    )
    assert pairing.derive_item_key("a") != pairing.derive_item_key("b")


def test_unicode_normalization_variant_near_duplicates_collide_and_fail():
    # Tests R-008 [unit]: keys are compared byte-exact after derivation;
    # NFC vs NFD normalization variants of the same text derive the SAME key,
    # so records built from both fail closed as duplicates.
    import unicodedata

    nfc_text = unicodedata.normalize("NFC", "caf\u00e9 synthetic")  # precomposed
    nfd_text = unicodedata.normalize("NFD", nfc_text)  # e + combining acute
    assert nfc_text != nfd_text
    key_nfc = pairing.derive_item_key(nfc_text)
    key_nfd = pairing.derive_item_key(nfd_text)
    assert key_nfc == key_nfd
    records = [
        make_record(key_nfc, 1, 2),
        make_record(key_nfd, 2, 3),
    ]
    with pytest.raises(pairing.KeySetError):
        pairing.recompute_counts(records)


# ---------------------------------------------------------------------------
# R-009: arm-reversal symmetry (seeded generation, plain pytest)
# ---------------------------------------------------------------------------


def _generated_record_sets(n_sets: int = 25) -> list[list[dict]]:
    # Property-based over generated tiny record sets, without hypothesis
    # (not installed): seeded random generation, plain pytest.
    rng = random.Random(20260819)
    sets = []
    for s in range(n_sets):
        n = rng.randint(2, 8)
        records = []
        for i in range(n):
            if i == 0:
                mc, ref = rng.randint(0, TRAJECTORY_HORIZON - 1), rng.randint(
                    0, TRAJECTORY_HORIZON - 1
                )  # force >=1 both-finite pair per set
            else:
                mc = rng.randint(0, TRAJECTORY_HORIZON)
                ref = rng.randint(0, TRAJECTORY_HORIZON)
            records.append(make_record(f"itm-{s:02d}{i:02d}", mc, ref))
        sets.append(records)
    return sets


def _reversed(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        swapped = dict(r)
        swapped["mc_stop_step"], swapped["ref_stop_step"] = (
            r["ref_stop_step"],
            r["mc_stop_step"],
        )
        out.append(swapped)
    return out


@pytest.mark.parametrize("idx", range(25))
def test_arm_reversal_count_symmetry(idx):
    # Tests R-009 [unit]: arm reversal leaves diagonal joint classes
    # unchanged, exchanges the two off-diagonal classes and the two timeout
    # totals.
    records = _generated_record_sets()[idx]
    fwd = pairing.recompute_counts(records)
    rev = pairing.recompute_counts(_reversed(records))
    assert rev["n_both_finite"] == fwd["n_both_finite"]
    assert rev["n_both_timeout"] == fwd["n_both_timeout"]
    assert rev["n_mc_finite_ref_timeout"] == fwd["n_mc_timeout_ref_finite"]
    assert rev["n_mc_timeout_ref_finite"] == fwd["n_mc_finite_ref_timeout"]
    assert rev["n_mc_timeout"] == fwd["n_ref_timeout"]
    assert rev["n_ref_timeout"] == fwd["n_mc_timeout"]
    assert rev["n_complete"] == fwd["n_complete"]
    assert rev["n_pairing_population"] == fwd["n_pairing_population"]


@pytest.mark.parametrize("idx", range(25))
def test_arm_reversal_timing_summary_antisymmetry(idx):
    # Tests R-009 [unit]: a timing-summary sign flips under reversal only
    # when that summary's frozen definition is antisymmetric — signed
    # mean/median flip; absolute mean/median are invariant.
    records = _generated_record_sets()[idx]
    fwd = pairing.finite_only_timing_summary(records)
    rev = pairing.finite_only_timing_summary(_reversed(records))
    assert rev["signed_index_mean"] == pytest.approx(-fwd["signed_index_mean"], abs=1e-12)
    assert rev["signed_index_median"] == pytest.approx(-fwd["signed_index_median"], abs=1e-12)
    assert rev["absolute_index_mean"] == pytest.approx(fwd["absolute_index_mean"], abs=1e-12)
    assert rev["absolute_index_median"] == pytest.approx(fwd["absolute_index_median"], abs=1e-12)


# ---------------------------------------------------------------------------
# R-010: parity with the historical paired summary (fixture-locked golden)
# ---------------------------------------------------------------------------


def _parity_records() -> list[dict]:
    golden = load_parity_golden()["all_finite_12"]
    horizon = golden["trajectory_horizon"]
    return [
        make_record(key, stops["mc"], stops["ref"], trajectory_horizon=horizon)
        for key, stops in sorted(golden["stops"].items())
    ]


def test_finite_only_summary_reproduces_historical_paired_summary_exactly():
    # Tests R-010 [unit]: on an all-finite record set the finite-only timing
    # summary reproduces the historical paired summary EXACTLY. Golden values
    # were produced by the pinned historical implementation
    # (scripts/stopdff_v5/bootstrap.py::cell_bootstrap_stats), fixture-locked
    # in tests/fixtures/colm_aims/parity_golden.json — never by running the
    # new implementation twice.
    # Source: tests/fixtures/colm_aims/parity_golden.json
    golden = load_parity_golden()["all_finite_12"]
    summary = pairing.finite_only_timing_summary(_parity_records())
    assert summary["signed_index_mean"] == golden["point"]["signed_index_mean"]
    assert summary["signed_index_median"] == golden["point"]["signed_index_median"]
    assert summary["absolute_index_mean"] == golden["point"]["absolute_index_mean"]
    assert summary["absolute_index_median"] == golden["point"]["absolute_index_median"]


def test_parity_golden_matches_live_historical_estimator():
    # Tests R-010 [unit]: guard against golden-file drift — the fixture-locked
    # values still match the live pinned historical implementation (same
    # items, same estimator). This anchors the golden to the oracle, not to
    # the new code.
    from scripts.stopdff_v5.bootstrap import build_bootstrap_plan, cell_bootstrap_stats

    golden = load_parity_golden()["all_finite_12"]
    shifts = {
        key: stops["mc"] - stops["ref"] for key, stops in golden["stops"].items()
    }
    plan = build_bootstrap_plan(
        sorted(shifts),
        replicates=golden["bootstrap"]["replicates"],
        seed=golden["bootstrap"]["seed"],
    )
    stats = cell_bootstrap_stats(shifts, plan)
    assert stats["point"] == golden["point"]
    assert stats["ci"] == golden["ci"]


def test_interval_recomputation_reruns_recorded_procedure(  # R-015 unit surface
):
    # Tests R-015 [unit]: interval recomputation re-runs the recorded
    # procedure (procedure, draw count, resampling seed) deterministically and
    # reproduces the historical CI on the golden record set.
    # Source: tests/fixtures/colm_aims/parity_golden.json
    golden = load_parity_golden()["all_finite_12"]
    spec = {
        "procedure": "percentile_bootstrap",
        "draw_count": golden["bootstrap"]["replicates"],
        "resampling_seeds": [golden["bootstrap"]["seed"]],
        "statistic": "signed_index_mean",
    }
    result = pairing.recompute_interval(_parity_records(), spec)
    assert list(result["ci"]) == golden["ci"]["signed_index_mean"]
    # Determinism: a second run reproduces the same interval.
    again = pairing.recompute_interval(_parity_records(), spec)
    assert list(again["ci"]) == list(result["ci"])


# ---------------------------------------------------------------------------
# R-011: estimand digest discipline
# ---------------------------------------------------------------------------


def test_estimand_digest_matches_pinned_formula():
    # Tests R-011 [unit]: DECISION — digest = sha256 over canonical compact
    # JSON (sort_keys, separators (",",":")) of the estimand block, so the
    # package builder and a third party can re-derive it.
    est = make_estimand()
    assert pairing.estimand_digest(est) == expected_estimand_digest(est)


ESTIMAND_FIELDS = [
    ("arm_mc", "arm-other"),
    ("arm_ref", "arm-other"),
    ("pairing_definition", "loose_grid"),
    ("timeout_parameters", {"trajectory_horizon": 9, "rule": "zero_indexed_stop_ge_horizon_is_timeout"}),
    ("denominator_policy", "n_pairing_population"),
    ("calibration_identity", "cal-9999"),
    ("continuation_identity", "cont-9999"),
    ("random_k_draw_id", "draw-0042"),
    ("numerical_tolerance", 5e-10),
]


@pytest.mark.parametrize("field,new_value", ESTIMAND_FIELDS)
def test_changing_any_estimand_field_changes_digest(field, new_value):
    # Tests R-011 [unit]: changing any estimand-defining field — including
    # the Random-K draw identity and the declared tolerance (R-032) —
    # changes the digest.
    base = make_estimand()
    changed = make_estimand(**{field: new_value})
    assert base[field] != changed[field]
    assert pairing.estimand_digest(base) != pairing.estimand_digest(changed)


def test_pooling_cells_with_differing_digests_is_refused():
    # Tests R-011 [unit]: pooling or comparing cells with differing digests
    # is refused.
    cell_a = make_cell()
    est_b = make_estimand(random_k_draw_id="draw-0042")
    cell_b = make_cell(estimand=est_b)
    cell_b["cell_id"] = "cell-0002"
    cell_b["estimand_digest"] = expected_estimand_digest(est_b)
    with pytest.raises(pairing.EstimandMismatchError):
        pairing.check_poolable(cell_a, cell_b)


def test_pooling_cells_with_equal_digests_is_allowed():
    # Tests R-011 [unit]: equal-digest cells pool without refusal.
    cell_a = make_cell()
    cell_b = make_cell()
    cell_b["cell_id"] = "cell-0002"
    pairing.check_poolable(cell_a, cell_b)


def test_duplicate_cell_identifiers_fail_closed():
    # Tests R-011 [unit]: cell identifiers are unique per artifact.
    profile = make_profile()
    dup = make_cell()
    profile["cells"] = [make_cell(), dup]  # same cell_id twice
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


def test_recorded_digest_must_match_recomputed_digest():
    # Tests R-011 [unit]: a cell whose recorded digest does not match its own
    # estimand fields is refused (digest is over ALL estimand-defining fields).
    cell = make_cell()
    cell["estimand_digest"] = "0" * 64
    with pytest.raises((pairing.EstimandMismatchError, schema.SchemaValidationError)):
        pairing.validate_cell(cell, standard_records())
