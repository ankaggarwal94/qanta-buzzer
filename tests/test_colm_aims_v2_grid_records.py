"""Grid identity + pair/censoring rules: R-040..R-043, R-005, R-006, R-008,
R-009 (spec sign-off SS2.1). End-to-end through source-mode verifier runs
over the FULL-SIZE canonical ten-cell package (exact leg-id oracles), plus
unit-level pairing checks.

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import json
import random

import pytest

from reproducibility.colm_aims_2026 import pairing, schema, verifier

from tests._colm_aims_v2_helpers import (
    AMBIGUOUS_TERMINAL_SENTINEL,
    CELL_IDS,
    LEG_COUNTS,
    LEG_GRID_COMPLETENESS,
    LEG_HELD_FIXED,
    LEG_ITEM_KEY_SET,
    LEG_MC_STOP_WITHIN_CAL,
    LEG_RATES,
    LEG_RECORD_FILE_BIJECTION,
    N_ITEMS,
    TRAJECTORY_HORIZON,
    VERDICT_SOURCE_PASS,
    assert_failing_leg,
    assert_passing_report,
    build_package_v2,
    canonical_data,
    canonical_item_keys,
    colm_no_network,  # noqa: F401 - autouse fixture
    expected_item_key,
    expected_estimand_digest,
    horizon_map_sha256,
    make_record_v2,
    release_report,
    source_report,
)


# ---------------------------------------------------------------------------
# Nearest-true baseline: the canonical package PASSES source mode. This is
# ALSO the R-043 control: canonical MC stops differ across calibrations, so
# a verifier that wrongly enforces cross-calibration MC-stop equality is
# itself defective and fails here.
# ---------------------------------------------------------------------------


def test_canonical_ten_cell_package_passes_source_mode(tmp_path):
    pkg = build_package_v2(tmp_path)
    report = source_report(pkg)
    assert_passing_report(report, VERDICT_SOURCE_PASS)


# ---------------------------------------------------------------------------
# R-040: exact 5x2 Cartesian grid, exact-set equality (never subset)
# ---------------------------------------------------------------------------


class TestGridCompleteness:
    def _run_fail(self, tmp_path, profile_mutator):
        pkg = build_package_v2(tmp_path, profile_mutator=profile_mutator)
        report = source_report(pkg)
        return assert_failing_leg(report, LEG_GRID_COMPLETENESS)

    def test_missing_reference_axis_member_fails(self, tmp_path):
        def mutate(profile):
            profile["grid"]["reference_ids"] = [
                r for r in profile["grid"]["reference_ids"] if r != "klex"
            ]

        self._run_fail(tmp_path, mutate)

    def test_extra_reference_axis_member_fails(self, tmp_path):
        def mutate(profile):
            profile["grid"]["reference_ids"] = sorted(
                profile["grid"]["reference_ids"] + ["kextra"]
            )

        self._run_fail(tmp_path, mutate)

    def test_renamed_calibration_axis_member_fails(self, tmp_path):
        def mutate(profile):
            profile["grid"]["calibration_ids"] = ["pooled", "shared"]

        self._run_fail(tmp_path, mutate)

    def test_calibration_axis_must_be_exactly_shared_and_format_specific(
        self, tmp_path
    ):
        def mutate(profile):
            profile["grid"]["calibration_ids"] = ["per_format", "shared"]

        self._run_fail(tmp_path, mutate)

    def test_missing_cell_fails(self, tmp_path):
        def mutate(profile):
            victim = "khard__shared"
            profile["grid"]["cell_ids"] = [
                c for c in profile["grid"]["cell_ids"] if c != victim
            ]
            del profile["grid"]["record_files"][victim]
            profile["cells"] = [
                c for c in profile["cells"] if c["cell_id"] != victim
            ]

        self._run_fail(tmp_path, mutate)

    def test_duplicated_cell_fails(self, tmp_path):
        def mutate(profile):
            dup = json.loads(json.dumps(profile["cells"][0]))
            profile["cells"].append(dup)

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        # Duplicate cell identifiers fail closed (R-011/R-040) — grid leg or
        # profile validation, but the run must FAIL.
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_extra_undeclared_cell_object_fails(self, tmp_path):
        def mutate(profile):
            extra = json.loads(json.dumps(profile["cells"][0]))
            extra["cell_id"] = "idealized__pooled"
            extra["calibration_id"] = "pooled"
            profile["cells"].append(extra)

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_subset_match_is_not_enough(self, tmp_path):
        # Exact-set comparisons, never subset: grid axes listing only 4 of
        # the 5 present references must fail even though every listed member
        # exists.
        def mutate(profile):
            profile["grid"]["reference_ids"] = sorted(
                profile["grid"]["reference_ids"]
            )[:4]

        self._run_fail(tmp_path, mutate)


# ---------------------------------------------------------------------------
# R-041 / R-035: cell <-> record-file bijection, both directions
# ---------------------------------------------------------------------------


class TestRecordFileBijection:
    def test_declared_but_absent_record_file_fails(self, tmp_path):
        pkg = build_package_v2(
            tmp_path, omit_record_files=("klex__shared",)
        )
        report = source_report(pkg)
        assert_failing_leg(report, LEG_RECORD_FILE_BIJECTION)

    def test_present_but_undeclared_record_file_fails(self, tmp_path):
        rogue = json.dumps(make_record_v2("itm-9999", 1, 2)) + "\n"
        pkg = build_package_v2(
            tmp_path,
            extra_tree_files={"records/rogue.jsonl": rogue.encode("utf-8")},
        )
        report = source_report(pkg)
        assert_failing_leg(report, LEG_RECORD_FILE_BIJECTION)

    def test_orphaned_mapping_entry_fails(self, tmp_path):
        def mutate(profile):
            profile["grid"]["record_files"]["idealized__pooled"] = (
                "records/idealized__pooled.jsonl"
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_two_cells_mapped_to_same_file_fails(self, tmp_path):
        def mutate(profile):
            profile["grid"]["record_files"]["klex__shared"] = (
                "records/khard__shared.jsonl"
            )
            for cell in profile["cells"]:
                if cell["cell_id"] == "klex__shared":
                    cell["records_file"] = "records/khard__shared.jsonl"

        pkg = build_package_v2(
            tmp_path,
            profile_mutator=mutate,
            omit_record_files=("klex__shared",),
        )
        report = source_report(pkg)
        assert_failing_leg(report, LEG_RECORD_FILE_BIJECTION)


# ---------------------------------------------------------------------------
# R-042 / R-008: exactly 2,249 complete pairs; byte-exact key set everywhere
# ---------------------------------------------------------------------------


class TestItemKeySetEquality:
    def test_one_record_short_fails(self, tmp_path):
        def drop_one(cell_id, records):
            if cell_id == "khard__shared":
                return records[:-1]
            return None

        pkg = build_package_v2(tmp_path, records_mutator=drop_one)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_ITEM_KEY_SET)

    def test_one_record_extra_fails(self, tmp_path):
        def add_one(cell_id, records):
            if cell_id == "khard__shared":
                return records + [
                    make_record_v2(expected_item_key("extra-item"), 1, 2)
                ]
            return None

        pkg = build_package_v2(tmp_path, records_mutator=add_one)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_ITEM_KEY_SET)

    def test_one_key_swapped_same_count_fails(self, tmp_path):
        # Same cardinality, different key SET in one cell: the cross-cell
        # byte-exact set-equality gate must fire.
        def swap_key(cell_id, records):
            if cell_id == "kdisjoint__format_specific":
                records[0] = dict(
                    records[0], item_key=expected_item_key("swapped-item")
                )
                return records
            return None

        pkg = build_package_v2(tmp_path, records_mutator=swap_key)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_ITEM_KEY_SET)

    def test_in_package_exclusion_breaks_the_gate_by_construction(
        self, tmp_path
    ):
        # R-042/R-047: there is no partial-population escape hatch inside
        # the frozen package — an excluded record cannot satisfy 2,249
        # complete pairs.
        def exclude_one(cell_id, records):
            if cell_id == "klex__shared":
                records[0] = dict(
                    records[0],
                    excluded=True,
                    exclusion_reason=AMBIGUOUS_TERMINAL_SENTINEL,
                )
                return records
            return None

        pkg = build_package_v2(tmp_path, records_mutator=exclude_one)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    @pytest.mark.parametrize("reporter", [source_report, release_report])
    def test_extra_excluded_rows_fail_with_complete_population_intact(
        self, tmp_path, reporter
    ):
        extra_key = expected_item_key("extra-excluded-item")
        horizon_identity = horizon_map_sha256(
            {
                **{
                    key: TRAJECTORY_HORIZON
                    for key in canonical_item_keys()
                },
                extra_key: TRAJECTORY_HORIZON,
            }
        )

        def add_excluded(_cell_id, records):
            return records + [
                dict(
                    make_record_v2(extra_key, 0, 0),
                    excluded=True,
                    exclusion_reason=AMBIGUOUS_TERMINAL_SENTINEL,
                )
            ]

        def bind_matching_declarations(profile):
            profile["grid"]["held_fixed"]["horizon_identity"] = (
                horizon_identity
            )
            for cell in profile["cells"]:
                counts = cell["counts"]
                counts["n_excluded_or_unpaired"] = 1
                counts["exclusion_reason_counts"] = {
                    AMBIGUOUS_TERMINAL_SENTINEL: 1
                }
                counts["n_pairing_population"] = N_ITEMS + 1
                estimand = cell["estimand"]
                estimand["timeout_parameters"]["horizon_map_sha256"] = (
                    horizon_identity
                )
                estimand["event_representation"]["horizon_identity"] = (
                    horizon_identity
                )
                cell["estimand_digest"] = expected_estimand_digest(estimand)

        pkg = build_package_v2(
            tmp_path,
            records_mutator=add_excluded,
            profile_mutator=bind_matching_declarations,
        )
        assert all(not cell["excluded_keys"] for cell in pkg.profile["cells"])
        report = reporter(pkg)
        assert_failing_leg(report, "record_validation")

    def test_duplicate_pair_keys_fail_closed(self, tmp_path):
        def duplicate_key(cell_id, records):
            if cell_id == "klex__shared":
                records[1] = dict(records[1], item_key=records[0]["item_key"])
                return records
            return None

        pkg = build_package_v2(tmp_path, records_mutator=duplicate_key)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_unicode_normalization_variants_collide_as_duplicates(self):
        # R-008 unit: NFC normalization makes normalization-variant
        # near-duplicates derive the SAME key (and thus fail closed as
        # duplicates upstream). Built programmatically so no editor can
        # silently re-normalize the fixture.
        import unicodedata

        composed = unicodedata.normalize("NFC", "A\u030angstro\u0308m item")
        decomposed = unicodedata.normalize("NFD", composed)
        assert composed != decomposed  # genuinely different byte sequences
        assert pairing.derive_item_key(composed) == pairing.derive_item_key(
            decomposed
        )

    def test_item_key_derivation_pinned(self):
        assert pairing.derive_item_key("synthetic-item-0000") == (
            expected_item_key("synthetic-item-0000")
        )


# ---------------------------------------------------------------------------
# R-043: held-fixed identities; MC-stop equality WITHIN calibration only
# ---------------------------------------------------------------------------


class TestHeldFixedIdentities:
    def test_within_calibration_mc_stop_mismatch_fails(self, tmp_path):
        # Perturb one cell's MC side: the five references within a
        # calibration must share the same raw MC trajectory stops.
        def perturb_mc(cell_id, records):
            if cell_id == "khard__shared":
                for i, rec in enumerate(records[:50]):
                    if rec["mc_event_status"] == "FINITE_STOP":
                        records[i] = dict(
                            rec,
                            mc_stop_step=(rec["mc_stop_step"] + 1) % 6,
                        )
                return records
            return None

        pkg = build_package_v2(tmp_path, records_mutator=perturb_mc)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_MC_STOP_WITHIN_CAL)

    def test_cross_calibration_difference_is_not_a_defect(self, tmp_path):
        # The retracted plan-v2 global MC-stop-map leg must NOT exist: the
        # canonical package's MC stops differ between shared and
        # format_specific (fixture property asserted in the d7b module) and
        # source mode PASSES.
        pkg = build_package_v2(tmp_path)
        report = source_report(pkg)
        assert_passing_report(report, VERDICT_SOURCE_PASS)

    def test_held_fixed_trajectory_identity_mismatch_fails(self, tmp_path):
        from tests._colm_aims_v2_helpers import expected_estimand_digest

        def mutate(profile):
            est = profile["cells"][3]["estimand"]
            est["event_representation"]["mc_trajectory_identity"] = (
                "traj-mc-v2-OTHER"
            )
            # Keep the recorded digest honest so the held-fixed leg is the
            # ONLY defect (substitution-negative discipline).
            profile["cells"][3]["estimand_digest"] = expected_estimand_digest(
                est
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_HELD_FIXED)

    def test_held_fixed_horizon_identity_mismatch_fails(self, tmp_path):
        def mutate(profile):
            profile["grid"]["held_fixed"]["horizon_identity"] = "hz-0008"

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_HELD_FIXED)


# ---------------------------------------------------------------------------
# R-005: count identities recomputed per cell from records
# ---------------------------------------------------------------------------

COUNT_KEYS = (
    "n_both_finite",
    "n_mc_finite_ref_timeout",
    "n_mc_timeout_ref_finite",
    "n_both_timeout",
    "n_complete",
    "n_excluded_or_unpaired",
    "n_pairing_population",
    "n_mc_timeout",
    "n_ref_timeout",
)


class TestCountIdentities:
    def test_recompute_counts_matches_independent_oracle(self):
        data = canonical_data()
        cell = data["cells"]["idealized__shared"]
        records = [
            json.loads(line)
            for line in cell.records_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        recomputed = pairing.recompute_counts(records)
        for key in COUNT_KEYS:
            assert recomputed[key] == cell.counts[key], key

    @pytest.mark.parametrize("count_key", COUNT_KEYS)
    def test_plus_one_mutation_of_any_recorded_count_fails(
        self, tmp_path, count_key
    ):
        def mutate(profile):
            profile["cells"][0]["counts"][count_key] += 1

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_COUNTS)


# ---------------------------------------------------------------------------
# R-006: joint-class rates (denominator n_complete); typed empty evaluation
# ---------------------------------------------------------------------------


class TestRates:
    def test_rates_null_when_n_complete_zero(self):
        counts = {
            "n_both_finite": 0,
            "n_mc_finite_ref_timeout": 0,
            "n_mc_timeout_ref_finite": 0,
            "n_both_timeout": 0,
            "n_complete": 0,
            "n_excluded_or_unpaired": 5,
            "exclusion_reason_counts": {"UNKNOWN_NOT_INFERRED": 5},
            "n_pairing_population": 5,
            "n_mc_timeout": 0,
            "n_ref_timeout": 0,
        }
        rates = pairing.compute_rates(counts)
        assert all(v is None for v in rates.values())

    def test_rate_mutation_beyond_tolerance_fails(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["rates"]["rate_both_finite"] += 0.01

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert_failing_leg(report, LEG_RATES)

    def test_empty_pairing_population_is_typed_error_not_trivial_pass(
        self, tmp_path
    ):
        def mutate(profile):
            cell = profile["cells"][0]
            cell["counts"] = dict(
                cell["counts"],
                n_both_finite=0,
                n_mc_finite_ref_timeout=0,
                n_mc_timeout_ref_finite=0,
                n_both_timeout=0,
                n_complete=0,
                n_pairing_population=0,
                n_mc_timeout=0,
                n_ref_timeout=0,
            )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        with pytest.raises(schema.EmptyEvaluationError):
            source_report(pkg)


# ---------------------------------------------------------------------------
# R-009: arm-reversal properties over generated tiny record sets
# ---------------------------------------------------------------------------


def _random_tiny_records(seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for i in range(40):
        mc = rng.choice([None, 0, 1, 2, 3, 4, 5])
        ref = rng.choice([None, 0, 1, 2, 3, 4, 5])
        records.append(make_record_v2(f"itm-{i:04d}", mc, ref))
    return records


def _swap_arms(records: list[dict]) -> list[dict]:
    swapped = []
    for rec in records:
        out = dict(rec)
        for field in ("event_status", "stop_step", "terminal_imputation"):
            out[f"mc_{field}"], out[f"ref_{field}"] = (
                rec[f"ref_{field}"],
                rec[f"mc_{field}"],
            )
        swapped.append(out)
    return swapped


class TestArmReversal:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_reversal_exchanges_off_diagonal_and_preserves_diagonal(
        self, seed
    ):
        records = _random_tiny_records(seed)
        fwd = pairing.recompute_counts(records)
        rev = pairing.recompute_counts(_swap_arms(records))
        assert rev["n_both_finite"] == fwd["n_both_finite"]
        assert rev["n_both_timeout"] == fwd["n_both_timeout"]
        assert rev["n_mc_finite_ref_timeout"] == fwd["n_mc_timeout_ref_finite"]
        assert rev["n_mc_timeout_ref_finite"] == fwd["n_mc_finite_ref_timeout"]
        assert rev["n_mc_timeout"] == fwd["n_ref_timeout"]
        assert rev["n_ref_timeout"] == fwd["n_mc_timeout"]

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_antisymmetric_summaries_flip_sign_absolute_do_not(self, seed):
        records = _random_tiny_records(seed)
        fwd = pairing.sentinel_coded_headline_summary(records)
        rev = pairing.sentinel_coded_headline_summary(_swap_arms(records))
        assert rev["mean_signed_shift"] == pytest.approx(
            -fwd["mean_signed_shift"]
        )
        fwd_f = pairing.finite_only_timing_summary(records)
        rev_f = pairing.finite_only_timing_summary(_swap_arms(records))
        assert rev_f["signed_index_mean"] == pytest.approx(
            -fwd_f["signed_index_mean"]
        )
        assert rev_f["absolute_index_mean"] == pytest.approx(
            fwd_f["absolute_index_mean"]
        )
        assert rev_f["absolute_index_median"] == pytest.approx(
            fwd_f["absolute_index_median"]
        )


# ---------------------------------------------------------------------------
# R-008: exclusion bookkeeping units
# ---------------------------------------------------------------------------


class TestExclusionBookkeeping:
    def test_missing_reason_is_unknown_not_inferred(self):
        records = [
            make_record_v2("itm-0001", 1, 2),
            dict(make_record_v2("itm-0002", None, None), excluded=True),
        ]
        counts = pairing.recompute_counts(records)
        assert counts["n_excluded_or_unpaired"] == 1
        assert counts["exclusion_reason_counts"] == {
            "UNKNOWN_NOT_INFERRED": 1
        }

    def test_secondary_diagnostics_not_counted_as_reasons(self):
        records = [
            make_record_v2("itm-0001", 1, 2),
            dict(
                make_record_v2("itm-0002", None, None),
                excluded=True,
                exclusion_reason="UNKNOWN_NOT_INFERRED",
                secondary_diagnostics=[AMBIGUOUS_TERMINAL_SENTINEL],
            ),
        ]
        counts = pairing.recompute_counts(records)
        assert sum(counts["exclusion_reason_counts"].values()) == 1

    def test_verifier_exposes_ten_validated_cells(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        report = source_report(pkg)
        assert report.verdict == VERDICT_SOURCE_PASS
        assert len(CELL_IDS) == 10 and N_ITEMS == 2249
        assert verifier is not None
