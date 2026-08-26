"""Implementation-facing D7(b) inference-block rules: R-050..R-057, R-015
interval identity, R-052 triple-bind + zero-exclusions FAIL enumeration.
(The pure-procedure early-signal core lives in
``test_colm_aims_v2_inference_d7b.py`` and passes at head.)

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import pytest

from reproducibility.colm_aims_2026 import schema  # noqa: F401 - RED import

from tests._colm_aims_v2_helpers import (
    CELL_IDS,
    LEG_INFERENCE_HOLM,
    LEG_INFERENCE_MATRIX,
    LEG_INFERENCE_RECOMPUTE,
    LEG_INFERENCE_SEED,
    VERDICT_SOURCE_PASS,
    assert_failing_leg,
    build_package_v2,
    canonical_data,
    colm_no_network,  # noqa: F401 - autouse fixture
    item_order_sha256,
    source_report,
)


def _fail_with(tmp_path, leg_id, profile_mutator):
    pkg = build_package_v2(tmp_path, profile_mutator=profile_mutator)
    report = source_report(pkg)
    return assert_failing_leg(report, leg_id)


def _fails(tmp_path, profile_mutator):
    pkg = build_package_v2(tmp_path, profile_mutator=profile_mutator)
    report = source_report(pkg)
    assert report.verdict != VERDICT_SOURCE_PASS
    return report


# ---------------------------------------------------------------------------
# R-050: canonical item order + per-cell means recompute exactly
# ---------------------------------------------------------------------------


class TestCanonicalVectors:
    def test_permuted_item_order_digest_fails(self, tmp_path):
        data = canonical_data()
        permuted = list(reversed(data["keys"]))

        def mutate(profile):
            profile["inference"]["canonical_item_order_digest"] = (
                item_order_sha256(permuted)
            )

        _fails(tmp_path, mutate)

    def test_mutated_cell_mean_fails_recompute(self, tmp_path):
        def mutate(profile):
            profile["cells"][2]["headline_summary"]["mean_signed_shift"] += (
                0.001
            )

        _fail_with(tmp_path, LEG_INFERENCE_RECOMPUTE, mutate)


# ---------------------------------------------------------------------------
# R-051: shared resample plan fields are exact-match validated
# ---------------------------------------------------------------------------


class TestResamplePlanFields:
    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("numpy_version", "2.4.5"),
            ("bit_generator", "MT19937"),
            ("draw_count", 999),
            ("sample_size", 2248),
            ("dtype", "int32"),
            ("endpoint", True),
            ("with_replacement", False),
        ],
    )
    def test_wrong_plan_field_fails(self, tmp_path, field, bad_value):
        def mutate(profile):
            profile["inference"][field] = bad_value

        _fails(tmp_path, mutate)

    def test_per_cell_matrix_declaration_rejected(self, tmp_path):
        # ONE collection-level matrix shared by all ten cells: a per-cell
        # seed table is an unknown key in the CLOSED inference block.
        def mutate(profile):
            profile["inference"]["per_cell_seeds"] = {
                cid: 1 + i for i, cid in enumerate(CELL_IDS)
            }

        _fails(tmp_path, mutate)


# ---------------------------------------------------------------------------
# R-052: seed triple-bind + zero-exclusions FAIL enumeration
# ---------------------------------------------------------------------------


class TestSeedBinding:
    def test_different_seed_fails(self, tmp_path):
        def mutate(profile):
            profile["inference"]["seed"] = profile["inference"]["seed"] + 1
            for cell in profile["cells"]:
                cell["interval"]["seed"] = profile["inference"]["seed"]

        _fail_with(tmp_path, LEG_INFERENCE_SEED, mutate)

    def test_extra_seeds_fail(self, tmp_path):
        # Exactly ONE seed: an additional seed list is refused (closed keys).
        def mutate(profile):
            profile["inference"]["resampling_seeds"] = [1, 2, 3]

        _fails(tmp_path, mutate)

    def test_missing_derivation_string_fails(self, tmp_path):
        def mutate(profile):
            del profile["inference"]["seed_derivation"]

        _fails(tmp_path, mutate)

    def test_nonempty_in_package_excluded_keys_fails(self, tmp_path):
        # R-052(a): the frozen v2 package has ZERO in-package exclusions;
        # the 9 upstream-unpaired items live in provenance documentation.
        def mutate(profile):
            profile["cells"][0]["excluded_keys"] = ["itm-0000000000000000"]

        _fails(tmp_path, mutate)

    def test_keyset_digest_not_matching_records_fails(self, tmp_path):
        # Triple-bind (b): the declared pairing_population_keyset_sha256
        # must equal the digest of the shared complete-key set.
        def mutate(profile):
            profile["inference"]["pairing_population_keyset_sha256"] = (
                "1" * 64
            )
            for cell in profile["cells"]:
                cell["pairing_population_keyset_sha256"] = "1" * 64

        _fails(tmp_path, mutate)

    def test_seed_bool_rejected(self, tmp_path):
        def mutate(profile):
            profile["inference"]["seed"] = True

        _fails(tmp_path, mutate)

    def test_seed_beyond_uint64_rejected(self, tmp_path):
        # R-061: bootstrap seed is an unsigned 64-bit integer. Both the
        # recorded seed and its per-cell copies move together so the DOMAIN
        # rule (not merely the derivation mismatch) is exercised; the
        # message names the uint64 domain.
        def mutate(profile):
            profile["inference"]["seed"] = 2**64
            for cell in profile["cells"]:
                cell["interval"]["seed"] = 2**64

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_pre_package_retention_documentation_required(self, tmp_path):
        # The 9 upstream-unpaired items are provenance documentation with
        # exact arithmetic: retained - paired == unpaired.
        def mutate(profile):
            profile["provenance"]["pre_package_retention"][
                "upstream_unpaired_count"
            ] = 8

        _fails(tmp_path, mutate)


# ---------------------------------------------------------------------------
# R-053: resample-matrix digest record (verifier regenerates and compares)
# ---------------------------------------------------------------------------


class TestMatrixDigest:
    def test_wrong_matrix_sha_fails(self, tmp_path):
        def mutate(profile):
            profile["inference"]["resample_matrix_digest"]["sha256"] = (
                "2" * 64
            )

        _fail_with(tmp_path, LEG_INFERENCE_MATRIX, mutate)

    @pytest.mark.parametrize(
        "covering_field",
        ["dtype", "shape", "byte_order", "canonical_item_order_digest"],
    )
    def test_digest_record_missing_covering_field_fails(
        self, tmp_path, covering_field
    ):
        def mutate(profile):
            del profile["inference"]["resample_matrix_digest"][covering_field]

        _fails(tmp_path, mutate)

    def test_wrong_shape_in_digest_record_fails(self, tmp_path):
        def mutate(profile):
            profile["inference"]["resample_matrix_digest"]["shape"] = [
                1000,
                2248,
            ]

        _fails(tmp_path, mutate)


# ---------------------------------------------------------------------------
# R-054 / R-015: interval identity and unrounded endpoints
# ---------------------------------------------------------------------------


class TestIntervals:
    def test_rounded_stored_endpoint_fails(self, tmp_path):
        def mutate(profile):
            ci = profile["cells"][0]["interval"]["ci"]
            profile["cells"][0]["interval"]["ci"] = [
                round(ci[0], 3),
                ci[1],
            ]

        _fail_with(tmp_path, LEG_INFERENCE_RECOMPUTE, mutate)

    def test_headline_interval_declaring_both_finite_fails(self, tmp_path):
        # Plan v3 SS3.3: the both-finite conditioning of the v1 interval
        # recompute is RETRACTED for headline cells.
        def mutate(profile):
            profile["cells"][0]["interval"]["population"] = "both_finite_only"

        _fails(tmp_path, mutate)

    def test_interval_population_outside_enum_rejected(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["interval"]["population"] = "everything"

        _fails(tmp_path, mutate)

    def test_wrong_quantile_method_fails(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["interval"]["quantile_method"] = "nearest"

        _fails(tmp_path, mutate)

    @pytest.mark.parametrize(
        "identity_field",
        [
            "procedure",
            "draw_count",
            "seed",
            "seed_derivation",
            "statistic",
            "population",
        ],
    )
    def test_missing_interval_identity_field_noncertifying(
        self, tmp_path, identity_field
    ):
        # R-015: missing interval identity leaves the interval
        # non-certifying — the run cannot PASS.
        def mutate(profile):
            del profile["cells"][0]["interval"][identity_field]

        _fails(tmp_path, mutate)


# ---------------------------------------------------------------------------
# R-055: mandatory +1/1001 correction
# ---------------------------------------------------------------------------


class TestPValues:
    def test_uncorrected_p_value_fails(self, tmp_path):
        # Off-by-one fixture: x/1000 instead of (x+1)/1001. The canonical
        # cells all have p = 1/1001; an uncorrected recompute yields 0.0.
        def mutate(profile):
            for cell in profile["cells"]:
                cell["raw_p_value"] = 0.0

        _fail_with(tmp_path, LEG_INFERENCE_RECOMPUTE, mutate)

    def test_single_cell_p_mutation_fails(self, tmp_path):
        def mutate(profile):
            profile["cells"][4]["raw_p_value"] = 0.5

        _fail_with(tmp_path, LEG_INFERENCE_RECOMPUTE, mutate)


# ---------------------------------------------------------------------------
# R-056: Holm family completeness and storage
# ---------------------------------------------------------------------------


class TestHolmFamily:
    def test_family_size_eight_fails(self, tmp_path):
        def mutate(profile):
            profile["inference"]["family_size"] = 8
            profile["inference"]["ordered_family"] = (
                profile["inference"]["ordered_family"][:8]
            )

        _fail_with(tmp_path, LEG_INFERENCE_HOLM, mutate)

    def test_omitted_random_k_cells_fail(self, tmp_path):
        # No selective omission of non-headline cells: dropping both
        # krandom cells from the ordered family (backfilled with duplicates
        # to keep length 10) must FAIL.
        def mutate(profile):
            fam = [
                c
                for c in profile["inference"]["ordered_family"]
                if not c.startswith("krandom__")
            ]
            profile["inference"]["ordered_family"] = fam + fam[:2]

        _fail_with(tmp_path, LEG_INFERENCE_HOLM, mutate)

    def test_tie_broken_against_utf8_order_fails(self, tmp_path):
        # All canonical raw p are tied at 1/1001, so the ordered family IS
        # the UTF-8 ascending cell-id order; swapping two adjacent entries
        # violates the tie rule.
        def mutate(profile):
            fam = list(profile["inference"]["ordered_family"])
            fam[0], fam[1] = fam[1], fam[0]
            profile["inference"]["ordered_family"] = fam

        _fail_with(tmp_path, LEG_INFERENCE_HOLM, mutate)

    def test_wrong_familywise_alpha_fails(self, tmp_path):
        def mutate(profile):
            profile["inference"]["familywise_alpha"] = 0.01

        _fails(tmp_path, mutate)

    def test_mutated_holm_rank_fails(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["holm_rank"] = 10
            profile["cells"][9]["holm_rank"] = 1

        _fail_with(tmp_path, LEG_INFERENCE_HOLM, mutate)

    def test_mutated_adjusted_p_fails(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["holm_adjusted_p_value"] = 0.9

        _fail_with(tmp_path, LEG_INFERENCE_HOLM, mutate)

    def test_mutated_rejected_set_fails(self, tmp_path):
        def mutate(profile):
            profile["inference"]["rejected_cell_ids"] = (
                profile["inference"]["rejected_cell_ids"][:-1]
            )
            for cell in profile["cells"]:
                if cell["cell_id"] == sorted(CELL_IDS)[-1]:
                    cell["holm_rejected"] = False

        _fail_with(tmp_path, LEG_INFERENCE_HOLM, mutate)

    def test_holm_rejected_flag_flip_fails(self, tmp_path):
        def mutate(profile):
            profile["cells"][3]["holm_rejected"] = False

        _fail_with(tmp_path, LEG_INFERENCE_HOLM, mutate)
