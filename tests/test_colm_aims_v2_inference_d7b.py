"""D7(b) pure-procedure tests (spec R-050..R-056 arithmetic; sign-off SS3).

EARLY-SIGNAL CORE: this module imports NO production namespace — every test
here encodes the frozen D7(b) inference procedure with hashlib/NumPy only and
MUST PASS at the greenfield head (the rest of the v2 suite fails collection
until GREEN exists). Golden literals were computed once at RED authoring time
(Python 3.11.15, NumPy 2.4.6, little-endian arm64) so any drift in the
environment or in the derivation itself is loud before implementation starts.

Implementation-facing D7(b) negatives live in
``test_colm_aims_v2_inference_block.py``.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pytest

from tests._colm_aims_v2_helpers import (
    CANONICAL_KEYSET_SHA256,
    CANONICAL_MATRIX_SHA256,
    CANONICAL_SEED,
    CELL_IDS,
    FIXTURE_KEYSET_SHA256,
    FIXTURE_MATRIX_SHA256,
    FIXTURE_SEED,
    N_ITEMS,
    canonical_data,
    canonical_item_keys,
    colm_no_network,  # noqa: F401 - autouse fixture
    d7b_holm,
    d7b_interval,
    d7b_matrix_digest_record,
    d7b_p_value,
    d7b_p_value_from_null_means,
    d7b_resample_matrix,
    d7b_seed,
    item_order_sha256,
    keyset_sha256,
)


# ---------------------------------------------------------------------------
# Environment pins (D5 / R-051 / R-028)
# ---------------------------------------------------------------------------


class TestEnvironmentPins:
    def test_numpy_version_exactly_2_4_6(self):
        # R-051/D5: NumPy version exactly 2.4.6 is REQUIRED, never skipped.
        assert np.__version__ == "2.4.6"

    def test_little_endian_platform_assumption(self):
        # The R-053 digest record covers byte order; the golden literals
        # below were produced on a little-endian machine.
        assert sys.byteorder == "little"


# ---------------------------------------------------------------------------
# R-052: deterministic seed derivation (no outcome-dependent author choice)
# ---------------------------------------------------------------------------


class TestSeedDerivation:
    def test_fixture_digest_seed_golden(self):
        assert d7b_seed(FIXTURE_KEYSET_SHA256) == FIXTURE_SEED

    def test_fixture_digest_value_is_its_own_derivation(self):
        # The fixture digest itself re-derives (provenance of the literal).
        assert (
            hashlib.sha256(b"colm-aims-v2-fixture-keyset").hexdigest()
            == FIXTURE_KEYSET_SHA256
        )

    def test_seed_two_independent_computations_agree(self):
        # int.from_bytes(digest[:8], "big") == int(hexdigest[:16], 16):
        # two arithmetically independent readings of the same 8 bytes.
        material = b"colm_aims_2026/v2/bootstrap_holm\0" + bytes.fromhex(
            FIXTURE_KEYSET_SHA256
        )
        hexdigest = hashlib.sha256(material).hexdigest()
        assert int(hexdigest[:16], 16) == FIXTURE_SEED

    def test_canonical_keyset_digest_golden(self):
        keys = canonical_item_keys()
        assert len(keys) == N_ITEMS
        assert keyset_sha256(keys) == CANONICAL_KEYSET_SHA256

    def test_canonical_seed_golden(self):
        assert d7b_seed(CANONICAL_KEYSET_SHA256) == CANONICAL_SEED

    def test_seed_domain_uint64_real_int(self):
        # R-052/R-061: exactly ONE seed; a real int in the uint64 domain,
        # never a bool.
        for seed in (FIXTURE_SEED, CANONICAL_SEED):
            assert type(seed) is int
            assert not isinstance(seed, bool)
            assert 0 <= seed < 2**64

    def test_prefix_material_is_null_terminated_domain_string(self):
        # The domain-separation prefix is pinned byte-for-byte: changing it
        # (or dropping the NUL) changes the seed.
        material = b"colm_aims_2026/v2/bootstrap_holm\0" + bytes.fromhex(
            FIXTURE_KEYSET_SHA256
        )
        wrong = b"colm_aims_2026/v2/bootstrap_holm" + bytes.fromhex(
            FIXTURE_KEYSET_SHA256
        )
        assert (
            int.from_bytes(hashlib.sha256(wrong).digest()[:8], "big")
            != int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        )

    def test_different_keyset_digest_different_seed(self):
        # Substitution-negative: the seed is BOUND to the key-set digest.
        other = hashlib.sha256(b"some-other-keyset").hexdigest()
        assert d7b_seed(other) != d7b_seed(FIXTURE_KEYSET_SHA256)


# ---------------------------------------------------------------------------
# R-051/R-053: shared resample matrix + digest record
# ---------------------------------------------------------------------------


class TestResampleMatrix:
    def test_canonical_matrix_sha256_golden(self):
        indices = d7b_resample_matrix(CANONICAL_SEED)
        assert (
            hashlib.sha256(indices.tobytes()).hexdigest()
            == CANONICAL_MATRIX_SHA256
        )

    def test_fixture_matrix_sha256_golden(self):
        indices = d7b_resample_matrix(FIXTURE_SEED)
        assert (
            hashlib.sha256(indices.tobytes()).hexdigest()
            == FIXTURE_MATRIX_SHA256
        )

    def test_matrix_plan_fields_exact(self):
        indices = d7b_resample_matrix(FIXTURE_SEED)
        assert indices.dtype == np.int64
        assert indices.shape == (1000, N_ITEMS)
        assert int(indices.min()) >= 0
        # endpoint=False: indices lie in [0, 2248] (R-061).
        assert int(indices.max()) <= N_ITEMS - 1

    def test_matrix_regeneration_is_deterministic(self):
        # R-034: the matrix is regenerated in-memory from the recorded seed —
        # regeneration must be byte-identical, never deserialized.
        a = d7b_resample_matrix(FIXTURE_SEED)
        b = d7b_resample_matrix(FIXTURE_SEED)
        assert a.tobytes() == b.tobytes()

    def test_digest_record_carries_all_four_covering_fields(self):
        indices = d7b_resample_matrix(FIXTURE_SEED)
        order_digest = item_order_sha256(canonical_item_keys())
        rec = d7b_matrix_digest_record(indices, order_digest)
        assert rec["sha256"] == FIXTURE_MATRIX_SHA256
        assert rec["dtype"] == "int64"
        assert rec["shape"] == [1000, N_ITEMS]
        assert rec["byte_order"] == "little"
        assert rec["canonical_item_order_digest"] == order_digest

    def test_endpoint_true_changes_the_bytes(self):
        # Nearest-false control: flipping endpoint=True is a DIFFERENT plan
        # and produces different bytes (so the digest gate must fire).
        rng = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
        wrong = rng.integers(
            0, N_ITEMS, size=(1000, N_ITEMS), dtype=np.int64, endpoint=True
        )
        assert (
            hashlib.sha256(wrong.tobytes()).hexdigest()
            != FIXTURE_MATRIX_SHA256
        )

    def test_wrong_dtype_changes_the_bytes(self):
        rng = np.random.Generator(np.random.PCG64(FIXTURE_SEED))
        wrong = rng.integers(
            0, N_ITEMS, size=(1000, N_ITEMS), dtype=np.int32, endpoint=False
        )
        assert (
            hashlib.sha256(wrong.tobytes()).hexdigest()
            != FIXTURE_MATRIX_SHA256
        )

    def test_wrong_bit_generator_changes_the_bytes(self):
        rng = np.random.Generator(np.random.MT19937(FIXTURE_SEED))
        wrong = rng.integers(
            0, N_ITEMS, size=(1000, N_ITEMS), dtype=np.int64, endpoint=False
        )
        assert (
            hashlib.sha256(wrong.tobytes()).hexdigest()
            != FIXTURE_MATRIX_SHA256
        )

    def test_wrong_seed_changes_the_bytes(self):
        wrong = d7b_resample_matrix(FIXTURE_SEED + 1)
        assert (
            hashlib.sha256(wrong.tobytes()).hexdigest()
            != FIXTURE_MATRIX_SHA256
        )


# ---------------------------------------------------------------------------
# R-054: uncentered percentile interval, method="linear"
# ---------------------------------------------------------------------------


class TestPercentileInterval:
    def test_linear_quantile_hand_checked_1000(self):
        # v = 1..1000: 2.5% linear = 25.975; 97.5% linear = 975.025.
        v = np.arange(1, 1001, dtype=np.float64)
        lo, hi = np.quantile(v, [0.025, 0.975], method="linear")
        assert lo == pytest.approx(25.975, abs=1e-12)
        assert hi == pytest.approx(975.025, abs=1e-12)

    def test_linear_quantile_hand_checked_10(self):
        # v = 1..10: pos 0.025*9=0.225 -> 1.225; pos 0.975*9=8.775 -> 9.775.
        v = np.arange(1, 11, dtype=np.float64)
        lo, hi = np.quantile(v, [0.025, 0.975], method="linear")
        assert lo == pytest.approx(1.225, abs=1e-12)
        assert hi == pytest.approx(9.775, abs=1e-12)

    def test_interval_helper_matches_numpy_linear(self):
        # Degenerate resamples: row b repeats index b everywhere, so the ten
        # bootstrap means are exactly the item values 1..10 -> hand-checkable
        # against the linear-quantile arithmetic above.
        d = np.arange(1, 11, dtype=np.float64)
        indices = np.tile(np.arange(10, dtype=np.int64).reshape(-1, 1), (1, 10))
        lo, hi = d7b_interval(d, indices)
        assert lo == pytest.approx(1.225, abs=1e-12)
        assert hi == pytest.approx(9.775, abs=1e-12)

    def test_interval_is_uncentered_percentile(self):
        # The uncentered percentile of a shifted-constant bootstrap is the
        # constant itself — a basic-bootstrap (2*mu - q) implementation
        # would differ. boot means all equal 7 -> interval [7, 7].
        d = np.full(4, 7.0)
        indices = np.zeros((1000, 4), dtype=np.int64)
        lo, hi = d7b_interval(d, indices)
        assert (lo, hi) == (7.0, 7.0)

    def test_interval_endpoints_ordered(self):
        d = np.array([1.0, -3.0, 2.5, 0.0, 4.0])
        indices = d7b_resample_matrix(FIXTURE_SEED, n=5, b=1000)
        lo, hi = d7b_interval(d, indices)
        assert lo <= hi


# ---------------------------------------------------------------------------
# R-055: null-centered paired bootstrap p-value with +1/1001 correction
# ---------------------------------------------------------------------------


class TestPValue:
    def test_plus_one_correction_hand_checked(self):
        # 7 null means exceed |mu|=0.3 -> p = (1+7)/1001, NOT 7/1000.
        null = np.array([0.5] * 7 + [0.1] * 993)
        p = d7b_p_value_from_null_means(null, 0.3)
        assert p == pytest.approx(8 / 1001, abs=1e-15)
        assert p != pytest.approx(7 / 1000, abs=1e-15)

    def test_uncorrected_zero_is_impossible(self):
        # Even with zero exceedances the corrected p is 1/1001, never 0.
        null = np.zeros(1000)
        p = d7b_p_value_from_null_means(null, 5.0)
        assert p == pytest.approx(1 / 1001, abs=1e-15)
        assert p > 0.0

    def test_exceedance_is_greater_or_equal(self):
        # |mu0_b| >= |mu_hat| counts EQUALITY as exceedance.
        null = np.array([0.3, -0.3] + [0.0] * 998)
        p = d7b_p_value_from_null_means(null, 0.3)
        assert p == pytest.approx(3 / 1001, abs=1e-15)

    def test_two_sided_via_absolute_values(self):
        # Sign-symmetric null means count symmetrically (two-sided test).
        null = np.array([0.4, -0.4, 0.2, -0.2] + [0.0] * 996)
        p = d7b_p_value_from_null_means(null, 0.3)
        assert p == pytest.approx(3 / 1001, abs=1e-15)

    def test_null_centering_zeroes_the_mean(self):
        d = np.array([1.0, 2.0, 6.0, -3.0, 4.0])
        z = d - d.mean()
        assert z.mean() == pytest.approx(0.0, abs=1e-12)

    def test_full_p_value_on_identity_resamples(self):
        # Identity resamples: null means all exactly 0 (rows resample the
        # whole set), so p = 1/1001 when mu != 0.
        d = np.array([2.0, 2.0, 2.0, 2.0])
        indices = np.tile(np.arange(4, dtype=np.int64), (1000, 1))
        p = d7b_p_value(d, indices)
        assert p == pytest.approx(1 / 1001, abs=1e-15)


# ---------------------------------------------------------------------------
# R-056: Holm step-down, m=10, UTF-8 tie ordering
# ---------------------------------------------------------------------------

HAND_RAW_P = {
    "a__1": 0.001,
    "B__x": 0.002,
    "b__x": 0.002,
    "c__1": 0.02,
    "d__1": 0.03,
    "e__1": 0.04,
    "f__1": 0.2,
    "g__1": 0.5,
    "h__1": 0.7,
    "i__1": 0.9,
}


class TestHolm:
    def test_hand_checked_m10_family(self):
        out = d7b_holm(dict(HAND_RAW_P))
        # UTF-8 byte order: "B" (0x42) sorts before "b" (0x62) on the tie.
        assert out["ordered_family"] == [
            "a__1", "B__x", "b__x", "c__1", "d__1",
            "e__1", "f__1", "g__1", "h__1", "i__1",
        ]
        assert out["rejected_cell_ids"] == sorted(["a__1", "B__x", "b__x"])
        per = out["per_cell"]
        assert per["a__1"]["holm_rank"] == 1
        assert per["B__x"]["holm_rank"] == 2
        assert per["b__x"]["holm_rank"] == 3
        # Hand-computed adjusted p (step-down running max, capped at 1):
        assert per["a__1"]["holm_adjusted_p_value"] == pytest.approx(0.01)
        assert per["B__x"]["holm_adjusted_p_value"] == pytest.approx(0.018)
        assert per["b__x"]["holm_adjusted_p_value"] == pytest.approx(0.018)
        assert per["c__1"]["holm_adjusted_p_value"] == pytest.approx(0.14)
        assert per["d__1"]["holm_adjusted_p_value"] == pytest.approx(0.18)
        assert per["e__1"]["holm_adjusted_p_value"] == pytest.approx(0.20)
        assert per["f__1"]["holm_adjusted_p_value"] == pytest.approx(0.8)
        assert per["g__1"]["holm_adjusted_p_value"] == pytest.approx(1.0)
        assert per["h__1"]["holm_adjusted_p_value"] == pytest.approx(1.0)
        assert per["i__1"]["holm_adjusted_p_value"] == pytest.approx(1.0)
        for cid in ("a__1", "B__x", "b__x"):
            assert per[cid]["holm_rejected"] is True
        for cid in ("c__1", "d__1", "e__1", "f__1", "g__1", "h__1", "i__1"):
            assert per[cid]["holm_rejected"] is False
        assert out["familywise_alpha"] == 0.05
        assert out["family_size"] == 10

    def test_tie_order_is_utf8_bytes_not_case_insensitive(self):
        raw = dict(HAND_RAW_P)
        raw.pop("B__x")
        raw.pop("b__x")
        raw["Z__t"] = 0.002
        raw["a__t"] = 0.002
        out = d7b_holm(raw)
        # "Z" (0x5A) < "a" (0x61) in UTF-8 bytes; a case-insensitive sort
        # would put "a__t" first.
        assert out["ordered_family"][1] == "Z__t"
        assert out["ordered_family"][2] == "a__t"

    def test_step_down_stops_at_first_acceptance(self):
        # Holm step-down: once p_(k) > alpha/(m-k+1), NOTHING later rejects
        # even if a later raw p would pass its own threshold in isolation.
        raw = {f"c{i:02d}": 0.9 for i in range(10)}
        raw["c00"] = 0.001  # rejects: 0.001 <= 0.05/10
        raw["c01"] = 0.03   # accepts: 0.03 > 0.05/9 -> step-down stops here
        raw["c02"] = 0.05   # sorts after the acceptance; must NOT reject
        out = d7b_holm(raw)
        assert out["rejected_cell_ids"] == ["c00"]

    def test_all_tied_minimum_p_rejects_everything(self):
        # The canonical package's shape: all ten raw p equal 1/1001.
        raw = {cid: 1 / 1001 for cid in CELL_IDS}
        out = d7b_holm(raw)
        assert out["ordered_family"] == sorted(CELL_IDS)
        assert out["rejected_cell_ids"] == sorted(CELL_IDS)
        # adjusted p = max running (1/1001 * 10) = 10/1001 everywhere.
        for cid in CELL_IDS:
            assert out["per_cell"][cid]["holm_adjusted_p_value"] == (
                pytest.approx(10 / 1001)
            )

    def test_family_must_be_exactly_ten_cells(self):
        # m=10 is frozen: an 8-cell family is refused by the reference
        # procedure itself (the implementation gate is R-056).
        raw = {f"c{i}": 0.01 for i in range(8)}
        with pytest.raises(ValueError):
            d7b_holm(raw)

    def test_adjusted_p_monotone_and_capped(self):
        out = d7b_holm(dict(HAND_RAW_P))
        ordered = out["ordered_family"]
        adj = [out["per_cell"][c]["holm_adjusted_p_value"] for c in ordered]
        assert adj == sorted(adj)
        assert all(0.0 <= a <= 1.0 for a in adj)


# ---------------------------------------------------------------------------
# Canonical ten-cell fixture data: oracle self-consistency (no namespace)
# ---------------------------------------------------------------------------


class TestCanonicalFixtureData:
    def test_counts_cross_check_one_cell_pure_python(self):
        # Two independent oracles: the numpy generative arithmetic in the
        # helpers vs a pure-python recount here.
        data = canonical_data()
        cell = data["cells"]["idealized__shared"]
        h = 6
        n_bf = n_mf_rt = n_mt_rf = n_bt = 0
        for i in range(N_ITEMS):
            mc = (7 * i + 3 + 11 * 1) % 9   # "shared" is calibration index 1
            rf = (5 * i + 1 + 3 * 0 + 13 * 1) % 8  # idealized is ref index 0
            mc_fin, rf_fin = mc < h, rf < h
            if mc_fin and rf_fin:
                n_bf += 1
            elif mc_fin:
                n_mf_rt += 1
            elif rf_fin:
                n_mt_rf += 1
            else:
                n_bt += 1
        assert cell.counts["n_both_finite"] == n_bf
        assert cell.counts["n_mc_finite_ref_timeout"] == n_mf_rt
        assert cell.counts["n_mc_timeout_ref_finite"] == n_mt_rf
        assert cell.counts["n_both_timeout"] == n_bt
        assert n_bf + n_mf_rt + n_mt_rf + n_bt == N_ITEMS

    def test_mc_stops_identical_within_calibration(self):
        # R-043 fixture property: MC raw stops shared across the five
        # references WITHIN each calibration...
        data = canonical_data()
        for cal in ("shared", "format_specific"):
            base = data["cells"][f"idealized__{cal}"].mc_raw
            for ref in ("kdisjoint", "khard", "klex", "krandom"):
                other = data["cells"][f"{ref}__{cal}"].mc_raw
                assert np.array_equal(base, other)

    def test_mc_stops_differ_across_calibrations(self):
        # ...and DIFFER between shared and format_specific (the nearest-true
        # control for the retracted cross-calibration equality leg).
        data = canonical_data()
        a = data["cells"]["idealized__shared"].mc_raw
        b = data["cells"]["idealized__format_specific"].mc_raw
        assert not np.array_equal(a, b)

    def test_references_pairwise_distinct_within_calibration(self):
        data = canonical_data()
        for cal in ("shared", "format_specific"):
            sigs = {
                tuple(data["cells"][f"{ref}__{cal}"].ref_raw[:16])
                for ref in ("idealized", "kdisjoint", "khard", "klex", "krandom")
            }
            assert len(sigs) == 5

    def test_all_ten_cells_p_is_min_and_all_rejected(self):
        data = canonical_data()
        for cid in CELL_IDS:
            assert data["cells"][cid].raw_p_value == pytest.approx(1 / 1001)
        assert data["holm"]["rejected_cell_ids"] == sorted(CELL_IDS)

    def test_intervals_contain_their_point_estimates(self):
        data = canonical_data()
        for cid in CELL_IDS:
            cell = data["cells"][cid]
            lo, hi = cell.ci
            assert lo <= cell.headline_mean <= hi
            assert lo <= hi

    def test_headline_sign_convention_positive_means_ref_earlier(self):
        # d = MC - REF; positive => the QA reference stops earlier (R-048).
        mc = np.array([5.0, 5.0])
        ref = np.array([1.0, 3.0])
        d = mc - ref
        assert float(d.mean()) == pytest.approx(3.0)
        # The canonical cells all have positive means: refs stop earlier.
        data = canonical_data()
        for cid in CELL_IDS:
            assert data["cells"][cid].headline_mean > 0
