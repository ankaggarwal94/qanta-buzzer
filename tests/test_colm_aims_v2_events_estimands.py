"""Event representation + estimand rules: R-045..R-049, R-068, R-010 parity,
R-011 digests/comparability (sign-off SS2.2/SS2.3).

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import ast
import json

import pytest

from reproducibility.colm_aims_2026 import pairing, schema

from tests._colm_aims_v2_helpers import (
    AMBIGUOUS_TERMINAL_SENTINEL,
    EVENT_FINITE,
    EVENT_NEVER,
    FINITE_ONLY_ESTIMAND_LABEL,
    HEADLINE_ESTIMAND_LABEL,
    IMPUTATION_FINAL_PREFIX,
    IMPUTATION_NONE,
    LEG_ESTIMAND_LABELS,
    N_ITEMS,
    POPULATION_ALL,
    POPULATION_FINITE,
    SENTINEL_CONVENTION,
    TRAJECTORY_HORIZON,
    VERDICT_SOURCE_PASS,
    assert_failing_leg,
    build_package_v2,
    colm_no_network,  # noqa: F401 - autouse fixture
    expected_estimand_digest,
    make_estimand,
    make_record_v2,
    namespace_py_files,
    source_report,
)


# ---------------------------------------------------------------------------
# R-045: canonical event vocabulary — every combination validated
# ---------------------------------------------------------------------------


class TestEventCombinations:
    def test_finite_with_null_stop_rejected(self):
        rec = make_record_v2("itm-0001", 2, 3)
        rec["mc_stop_step"] = None
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_finite_with_missing_stop_rejected(self):
        rec = make_record_v2("itm-0001", 2, 3)
        del rec["mc_stop_step"]
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_never_stopped_with_numeric_stop_rejected(self):
        # This is also the "canonical event overwritten by the derived
        # scalar" signature (R-046): NEVER_STOPPED + stop_step == horizon.
        rec = make_record_v2("itm-0001", None, 3)
        rec["mc_stop_step"] = TRAJECTORY_HORIZON
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_never_stopped_requires_explicit_null_not_absent(self):
        rec = make_record_v2("itm-0001", None, 3)
        del rec["mc_stop_step"]
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_never_stopped_requires_final_prefix_imputation(self):
        rec = make_record_v2("itm-0001", None, 3)
        rec["mc_terminal_imputation"] = IMPUTATION_NONE
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_finite_with_imputation_marker_rejected(self):
        rec = make_record_v2("itm-0001", 2, 3)
        rec["mc_terminal_imputation"] = IMPUTATION_FINAL_PREFIX
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    @pytest.mark.parametrize(
        "missing_key",
        [
            "index_base",
            "horizon_identity",
            "mc_trajectory_identity",
            "historical_sentinel_convention",
            "terminal_imputation_policy",
            "producer_profile_identity",
        ],
    )
    def test_missing_bound_identity_fails(self, tmp_path, missing_key):
        # R-045: each record set binds horizon/index-base/producer identity/
        # sentinel convention/imputation policy — a missing binding FAILs.
        def mutate(profile):
            est = profile["cells"][0]["estimand"]
            del est["event_representation"][missing_key]
            profile["cells"][0]["estimand_digest"] = (
                expected_estimand_digest(est)
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_bool_stop_step_rejected(self):
        # R-061: bools never satisfy an integer domain.
        rec = make_record_v2("itm-0001", 2, 3)
        rec["mc_stop_step"] = True
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)


# ---------------------------------------------------------------------------
# R-046: final-prefix crossings; derived scalar never replaces the event
# ---------------------------------------------------------------------------


class TestFinalPrefixCrossing:
    def test_genuine_final_prefix_crossing_is_finite_stop(self):
        # stop_step == horizon-1 is a real finite stop (positive control).
        schema.validate_record(
            make_record_v2("itm-0001", TRAJECTORY_HORIZON - 1, 2)
        )

    def test_crossing_indicator_contradicting_never_stopped_rejected(self):
        # Mislabeled fixture: an explicit crossing indicator with
        # NEVER_STOPPED is a contradiction — the crossing happened.
        rec = make_record_v2("itm-0001", None, 2)
        rec["mc_crossing_indicator"] = True
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_crossing_indicator_with_finite_stop_accepted(self):
        rec = make_record_v2("itm-0001", TRAJECTORY_HORIZON - 1, 2)
        rec["mc_crossing_indicator"] = True
        schema.validate_record(rec)

    @pytest.mark.parametrize("prefix", ["mc", "ref"])
    def test_false_crossing_indicator_with_finite_stop_rejected(self, prefix):
        rec = make_record_v2("itm-0001", TRAJECTORY_HORIZON - 1, 2)
        rec[f"{prefix}_crossing_indicator"] = False
        with pytest.raises(schema.RecordValidationError, match="no crossing"):
            schema.validate_record(rec)

    def test_distinct_fields_roundtrip_both_validated(self):
        # The canonical event and any imported original encoding are
        # DISTINCT fields; both survive a roundtrip unchanged.
        rec = make_record_v2("itm-0001", None, 2)
        rec["mc_original_encoded_stop"] = TRAJECTORY_HORIZON
        schema.validate_record(rec)
        blob = json.dumps(rec, sort_keys=True).encode("utf-8")
        back = schema.parse_json_bytes_strict(blob)
        assert back["mc_event_status"] == EVENT_NEVER
        assert back["mc_stop_step"] is None
        assert back["mc_original_encoded_stop"] == TRAJECTORY_HORIZON

    def test_sentinel_convention_token_is_closed(self, tmp_path):
        def mutate(profile):
            est = profile["cells"][0]["estimand"]
            est["event_representation"]["historical_sentinel_convention"] = (
                "timeout_coded_as_null"
            )
            profile["cells"][0]["estimand_digest"] = (
                expected_estimand_digest(est)
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS


# ---------------------------------------------------------------------------
# R-047: ambiguous legacy T-1 sentinel refusal
# ---------------------------------------------------------------------------


class TestAmbiguousLegacySentinel:
    def test_without_authentication_refuses_with_named_reason(self):
        out = pairing.normalize_legacy_terminal(
            TRAJECTORY_HORIZON - 1,
            horizon=TRAJECTORY_HORIZON,
            authenticated_convention=None,
            crossing_indicator=None,
        )
        assert out["excluded"] is True
        assert out["exclusion_reason"] == AMBIGUOUS_TERMINAL_SENTINEL
        # Stays in the legacy representation, never silently promoted:
        assert out.get("legacy_value") == TRAJECTORY_HORIZON - 1
        assert "event_status" not in out

    def test_with_crossing_indicator_is_finite_stop(self):
        out = pairing.normalize_legacy_terminal(
            TRAJECTORY_HORIZON - 1,
            horizon=TRAJECTORY_HORIZON,
            authenticated_convention=None,
            crossing_indicator=True,
        )
        assert out["event_status"] == EVENT_FINITE
        assert out["stop_step"] == TRAJECTORY_HORIZON - 1

    def test_with_authenticated_terminal_convention_is_never_stopped(self):
        out = pairing.normalize_legacy_terminal(
            TRAJECTORY_HORIZON - 1,
            horizon=TRAJECTORY_HORIZON,
            authenticated_convention="timeout_coded_as_final_index",
            crossing_indicator=None,
        )
        assert out["event_status"] == EVENT_NEVER
        assert out["stop_step"] is None
        assert out["terminal_imputation"] == IMPUTATION_FINAL_PREFIX

    def test_horizon_convention_treats_t_minus_1_as_ordinary_finite(self):
        # Under timeout_coded_as_horizon, T-1 is an ordinary in-range stop.
        out = pairing.normalize_legacy_terminal(
            TRAJECTORY_HORIZON - 1,
            horizon=TRAJECTORY_HORIZON,
            authenticated_convention=SENTINEL_CONVENTION,
            crossing_indicator=None,
        )
        assert out["event_status"] == EVENT_FINITE
        assert out["stop_step"] == TRAJECTORY_HORIZON - 1


# ---------------------------------------------------------------------------
# R-048: headline estimand identity + prohibited labels
# ---------------------------------------------------------------------------


class TestHeadlineEstimand:
    def test_headline_population_must_be_all_pairs(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["headline_summary"]["population"] = (
                POPULATION_FINITE
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_ESTIMAND_LABELS)

    def test_headline_label_outside_closed_set_rejected(self, tmp_path):
        # Prohibited labels (finite-only mean / infinite stopping times /
        # observed timing effect / preservation evidence) are unreachable:
        # the label vocabulary is CLOSED.
        def mutate(profile):
            profile["cells"][0]["headline_summary"]["estimand_label"] = (
                "observed_mc_versus_open_ended_timing_effect"
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_headline_n_must_be_2249(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["headline_summary"]["n"] = (
                profile["cells"][0]["counts"]["n_both_finite"]
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_headline_recompute_on_tiny_known_records(self):
        # d = MC - REF, timeouts coded as horizon; positive => QA reference
        # stops earlier. Hand values: shifts (-2, 0, +4, -4, +3, 0), mean 1/6.
        records = [
            make_record_v2("itm-0001", 1, 3),
            make_record_v2("itm-0002", 2, 2),
            make_record_v2("itm-0003", 5, 1),
            make_record_v2("itm-0004", 2, None),
            make_record_v2("itm-0005", None, 3),
            make_record_v2("itm-0006", None, None),
        ]
        summary = pairing.sentinel_coded_headline_summary(records)
        assert summary["n"] == 6
        assert summary["mean_signed_shift"] == pytest.approx(
            0.16666666666666666
        )


# ---------------------------------------------------------------------------
# R-049 / R-068: finite-only separation and label binding
# ---------------------------------------------------------------------------


class TestEstimandSeparation:
    def test_finite_only_summary_must_declare_conditional_population(
        self, tmp_path
    ):
        def mutate(profile):
            profile["cells"][0]["finite_only_summary"]["population"] = (
                POPULATION_ALL
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_ESTIMAND_LABELS)

    def test_finite_only_n_must_be_n_both_finite(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["finite_only_summary"]["n"] = N_ITEMS

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_label_swap_headline_finite_fails(self, tmp_path):
        # R-068 label-swap fixture: swapping the two labels must FAIL even
        # though both labels are individually legal.
        def mutate(profile):
            cell = profile["cells"][0]
            cell["headline_summary"]["estimand_label"] = (
                FINITE_ONLY_ESTIMAND_LABEL
            )
            cell["finite_only_summary"]["estimand_label"] = (
                HEADLINE_ESTIMAND_LABEL
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_ESTIMAND_LABELS)

    def test_trusted_label_truth_set_regression_shape(self, tmp_path):
        # F7A RED fixture: a consistently misspelled recorded label must
        # fail against the CANONICAL recompute identity — a verifier that
        # builds its truth-set from the recorded strings would pass it.
        def mutate(profile):
            for cell in profile["cells"]:
                cell["headline_summary"]["estimand_label"] = (
                    HEADLINE_ESTIMAND_LABEL + "_x"
                )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_two_estimands_have_distinct_digests(self):
        est_headline = make_estimand("idealized__shared")
        est_finite = make_estimand(
            "idealized__shared", population=POPULATION_FINITE
        )
        assert expected_estimand_digest(est_headline) != (
            expected_estimand_digest(est_finite)
        )

    def test_check_comparable_refuses_population_mix(self):
        cell_a = {
            "cell_id": "a",
            "estimand": make_estimand("idealized__shared"),
        }
        cell_a["estimand_digest"] = expected_estimand_digest(
            cell_a["estimand"]
        )
        cell_b = {
            "cell_id": "b",
            "estimand": make_estimand(
                "idealized__shared", population=POPULATION_FINITE
            ),
        }
        cell_b["estimand_digest"] = expected_estimand_digest(
            cell_b["estimand"]
        )
        with pytest.raises(pairing.EstimandMismatchError):
            pairing.check_comparable(cell_a, cell_b)

    def test_check_comparable_passes_on_equal_digests(self):
        est = make_estimand("idealized__shared")
        cell = {"cell_id": "a", "estimand": est}
        cell["estimand_digest"] = expected_estimand_digest(est)
        other = json.loads(json.dumps(cell))
        other["cell_id"] = "a"
        pairing.check_comparable(cell, other)


# ---------------------------------------------------------------------------
# R-010: historical parity, fixture-locked goldens
# ---------------------------------------------------------------------------

# Golden values produced by the PINNED HISTORICAL implementation
# (scripts/stopdff_v5/bootstrap.py::cell_bootstrap_stats over
# policy.py::signed_index_shift), regenerated once under NumPy 2.4.6 (D5) at
# v1 RED fixture creation and carried verbatim — NEVER produced by running
# the new implementation twice.
ALL_FINITE_12_STOPS = {
    "itm-0001": (1, 3),
    "itm-0002": (2, 2),
    "itm-0003": (5, 1),
    "itm-0004": (0, 7),
    "itm-0005": (4, 4),
    "itm-0006": (6, 2),
    "itm-0007": (3, 5),
    "itm-0008": (2, 6),
    "itm-0009": (7, 0),
    "itm-0010": (1, 1),
    "itm-0011": (5, 4),
    "itm-0012": (0, 2),
}
ALL_FINITE_12_GOLDEN_POINT = {
    "signed_index_mean": -0.08333333333333333,
    "signed_index_median": 0.0,
    "absolute_index_mean": 2.75,
    "absolute_index_median": 2.0,
}


class TestHistoricalParity:
    def test_finite_only_summary_reproduces_historical_estimator(self):
        records = [
            make_record_v2(key, mc, ref, trajectory_horizon=8)
            for key, (mc, ref) in sorted(ALL_FINITE_12_STOPS.items())
        ]
        summary = pairing.finite_only_timing_summary(records)
        assert summary["n"] == 12
        for stat, golden in ALL_FINITE_12_GOLDEN_POINT.items():
            assert summary[stat] == pytest.approx(golden, abs=1e-12), stat

    def test_finite_only_parity_cross_checked_by_pure_python(self):
        # Second independent oracle for the same golden (truth-tracking
        # triangle: golden literal, pure-python recompute, implementation).
        shifts = sorted(
            mc - ref for mc, ref in ALL_FINITE_12_STOPS.values()
        )
        assert sum(shifts) / 12 == pytest.approx(
            ALL_FINITE_12_GOLDEN_POINT["signed_index_mean"]
        )
        assert (shifts[5] + shifts[6]) / 2 == pytest.approx(
            ALL_FINITE_12_GOLDEN_POINT["signed_index_median"]
        )

    def test_headline_estimator_reproduces_fair_qa_producer_convention(self):
        # Golden record set from the preserved fair-QA producer lineage
        # (min(stop, horizon) per side, timeout coded as horizon): the
        # six-pack sentinel-coded mean is exactly 1/6.
        records = [
            make_record_v2("itm-0001", 1, 3),
            make_record_v2("itm-0002", 2, 2),
            make_record_v2("itm-0003", 5, 1),
            make_record_v2("itm-0004", 2, None),
            make_record_v2("itm-0005", None, 3),
            make_record_v2("itm-0006", None, None),
        ]
        summary = pairing.sentinel_coded_headline_summary(records)
        assert summary["mean_signed_shift"] == pytest.approx(
            0.16666666666666666, abs=1e-15
        )
        assert summary["convention"] == SENTINEL_CONVENTION


# ---------------------------------------------------------------------------
# R-011: estimand digests over ALL estimand-defining fields
# ---------------------------------------------------------------------------

DIGEST_FIELD_MUTATIONS = [
    ("reference_id", "khard"),
    ("calibration_id", "shared_other"),
    ("pairing_definition", "matched_item_grid_alt"),
    ("population", POPULATION_FINITE),
    ("denominator_policy", "n_both_finite"),
    ("numerical_tolerance", 1e-6),
    ("calibration_identity", "cal-other-9999"),
    ("continuation_identity", "cont-9999"),
    ("random_k_draw_id", "draw-favorable-7"),
]


class TestEstimandDigest:
    @pytest.mark.parametrize(("field", "new_value"), DIGEST_FIELD_MUTATIONS)
    def test_changing_any_digest_field_changes_the_digest(
        self, field, new_value
    ):
        est = make_estimand("idealized__format_specific")
        mutated = json.loads(json.dumps(est))
        mutated[field] = new_value
        assert pairing.estimand_digest(est) != pairing.estimand_digest(
            mutated
        )

    @pytest.mark.parametrize(
        ("subblock", "field", "new_value"),
        [
            ("timeout_parameters", "trajectory_horizon", 8),
            ("timeout_parameters", "rule", "one_indexed_stop_gt_horizon"),
            ("event_representation", "index_base", 1),
            ("event_representation", "horizon_identity", "hz-0008"),
            (
                "event_representation",
                "historical_sentinel_convention",
                "timeout_coded_as_null",
            ),
            (
                "event_representation",
                "terminal_imputation_policy",
                "HORIZON_SENTINEL",
            ),
        ],
    )
    def test_nested_event_identity_fields_are_digest_fields(
        self, subblock, field, new_value
    ):
        est = make_estimand("idealized__format_specific")
        mutated = json.loads(json.dumps(est))
        mutated[subblock][field] = new_value
        assert pairing.estimand_digest(est) != pairing.estimand_digest(
            mutated
        )

    def test_digest_matches_pinned_convention(self):
        est = make_estimand("idealized__shared")
        assert pairing.estimand_digest(est) == expected_estimand_digest(est)

    def test_recorded_digest_mismatch_fails(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["estimand_digest"] = "0" * 64

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_pairing_definition_vocabulary_closed(self, tmp_path):
        def mutate(profile):
            est = profile["cells"][0]["estimand"]
            est["pairing_definition"] = "freeform_pairing"
            profile["cells"][0]["estimand_digest"] = (
                expected_estimand_digest(est)
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_timeout_rule_must_reconcile_with_pairing_definition(
        self, tmp_path
    ):
        # The 7B/M3 surface: grid pairing with a wall-clock timeout rule is
        # an illegal combination even though each token might exist
        # somewhere in a legal-value list.
        def mutate(profile):
            est = profile["cells"][0]["estimand"]
            est["timeout_parameters"]["rule"] = "wall_clock_seconds_cap"
            profile["cells"][0]["estimand_digest"] = (
                expected_estimand_digest(est)
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_check_comparable_is_production_wired(self):
        # R6 lesson: a helper with zero production callers protects nothing.
        # At least one namespace module OTHER than its defining module must
        # call check_comparable.
        callers: list[str] = []
        for path in namespace_py_files():
            tree = ast.parse(path.read_text("utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = (
                        func.attr
                        if isinstance(func, ast.Attribute)
                        else getattr(func, "id", None)
                    )
                    if name == "check_comparable":
                        callers.append(path.name)
        assert any(c != "pairing.py" for c in callers), (
            "check_comparable has no production caller outside its defining"
            f" module (callers: {callers}) — R-011/R6"
        )
