"""Pair/censoring decomposition + the D7(b) inference arithmetic.

Spec rules owned here: R-005..R-011, R-045..R-050 (event classification and
the two named estimands), R-047 (ambiguous legacy terminal refusal),
R-050..R-056 (the exact regenerated inference procedure), R-015 (recompute
helpers).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import functools
import hashlib
import sys
import unicodedata
from typing import Any

import numpy as np

from .schema import (
    AMBIGUOUS_TERMINAL_SENTINEL,
    BOOTSTRAP_DRAW_COUNT,
    EVENT_FINITE,
    EVENT_NEVER,
    EXCLUSION_REASONS,
    EXPECTED_COMPLETE_PAIRS,
    FINITE_ONLY_ESTIMAND_LABEL,
    HEADLINE_ESTIMAND_LABEL,
    IMPUTATION_FINAL_PREFIX,
    MAX_BOOTSTRAP_CELLS,
    MAX_BOOTSTRAP_DRAWS,
    POPULATION_ALL,
    POPULATION_FINITE,
    SENTINEL_CONVENTION,
    ColmAimsError,
    canonical_estimand_digest,
    is_real_int,
    is_uint64,
)

JOINT_CLASSES = (
    "both_finite",
    "mc_finite_ref_timeout",
    "mc_timeout_ref_finite",
    "both_timeout",
)

_JOINT_CLASS_BY_KINDS = {
    ("finite", "finite"): "both_finite",
    ("finite", "timeout"): "mc_finite_ref_timeout",
    ("timeout", "finite"): "mc_timeout_ref_finite",
    ("timeout", "timeout"): "both_timeout",
}

RATE_KEYS = (
    "rate_both_finite",
    "rate_mc_finite_ref_timeout",
    "rate_mc_timeout_ref_finite",
    "rate_both_timeout",
)

FINITE_ONLY_STATISTICS = (
    "signed_index_mean",
    "signed_index_median",
    "absolute_index_mean",
    "absolute_index_median",
)


class CountIdentityError(ColmAimsError):
    """A recorded count violates an exact count identity (R-005)."""


class RateError(ColmAimsError):
    """Joint-class rate denominator/sum-to-1/tolerance violation (R-006)."""


class KeySetError(ColmAimsError):
    """Key-set discipline violation: duplicates/overlap/hash mismatch (R-008)."""


class EstimandMismatchError(ColmAimsError):
    """Pooling or comparing cells with differing estimand digests (R-011)."""


def _is_valid_horizon(value: Any) -> bool:
    return is_real_int(value) and value > 0


# ---------------------------------------------------------------------------
# Canonical event classification (R-045)
# ---------------------------------------------------------------------------


def _arm_kind(record: dict[str, Any], prefix: str, horizon: Any) -> str | None:
    """``finite`` / ``timeout`` for one arm, ``None`` when malformed."""
    status = record.get(f"{prefix}_event_status")
    stop = record.get(f"{prefix}_stop_step")
    if status == EVENT_FINITE:
        if is_real_int(stop) and 0 <= stop < horizon:
            return "finite"
        return None
    if status == EVENT_NEVER:
        if stop is None:
            return "timeout"
        return None
    return None


def _excluded(item_key: Any, reason: str) -> dict[str, Any]:
    return {
        "item_key": item_key,
        "status": "excluded",
        "exclusion_reason": reason,
        "joint_class": None,
    }


def classify_record(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one canonical-event record into joint class or exclusion."""
    key = record.get("item_key")
    if record.get("excluded") is True:
        declared = record.get("exclusion_reason")
        if isinstance(declared, str) and declared in EXCLUSION_REASONS:
            return _excluded(key, declared)
        # Missing/undeclared reasons are recorded, never guessed (R-008).
        return _excluded(key, "UNKNOWN_NOT_INFERRED")
    horizon = record.get("trajectory_horizon")
    if not _is_valid_horizon(horizon):
        return _excluded(key, "MALFORMED_STOP")
    mc_kind = _arm_kind(record, "mc", horizon)
    ref_kind = _arm_kind(record, "ref", horizon)
    if mc_kind is None or ref_kind is None:
        return _excluded(key, "MALFORMED_STOP")
    return {
        "item_key": key,
        "status": "complete",
        "joint_class": _JOINT_CLASS_BY_KINDS[(mc_kind, ref_kind)],
    }


def _check_duplicate_keys(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        key = record.get("item_key")
        if not isinstance(key, str) or not key:
            raise KeySetError("record without an opaque item_key (R-008)")
        if key in seen:
            raise KeySetError(
                f"duplicate pair key {key!r} — duplicate pair keys fail"
                " closed (R-008)"
            )
        seen.add(key)


def recompute_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the per-cell count block from per-item records (R-005)."""
    _check_duplicate_keys(records)
    joint_counts = {joint: 0 for joint in JOINT_CLASSES}
    reason_counts: dict[str, int] = {}
    n_excluded = 0
    for record in records:
        outcome = classify_record(record)
        if outcome["status"] == "excluded":
            n_excluded += 1
            reason = outcome["exclusion_reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            joint_counts[outcome["joint_class"]] += 1
    n_complete = sum(joint_counts.values())
    return {
        "n_both_finite": joint_counts["both_finite"],
        "n_mc_finite_ref_timeout": joint_counts["mc_finite_ref_timeout"],
        "n_mc_timeout_ref_finite": joint_counts["mc_timeout_ref_finite"],
        "n_both_timeout": joint_counts["both_timeout"],
        "n_complete": n_complete,
        "n_excluded_or_unpaired": n_excluded,
        "exclusion_reason_counts": reason_counts,
        "n_pairing_population": n_complete + n_excluded,
        "n_mc_timeout": (
            joint_counts["mc_timeout_ref_finite"] + joint_counts["both_timeout"]
        ),
        "n_ref_timeout": (
            joint_counts["mc_finite_ref_timeout"] + joint_counts["both_timeout"]
        ),
    }


def check_count_identities(
    counts: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    """Enforce the exact count identities against records (R-005)."""
    recomputed = recompute_counts(records)
    mismatches = []
    for field, value in recomputed.items():
        if field not in counts:
            mismatches.append(f"{field} missing from recorded counts")
        elif counts[field] != value:
            mismatches.append(
                f"{field} recorded {counts[field]!r} != recomputed {value!r}"
            )
    extras = sorted(set(counts) - set(recomputed))
    if extras:
        mismatches.append(f"unknown recorded count field(s) {extras}")
    if mismatches:
        raise CountIdentityError(
            "count identity violation against per-item records (R-005): "
            + "; ".join(mismatches)
        )


def compute_rates(counts: dict[str, Any]) -> dict[str, Any]:
    """Joint-class rates over ``n_complete``; ``None`` at zero (R-006)."""
    n_complete = counts["n_complete"]
    if n_complete == 0:
        return {key: None for key in RATE_KEYS}
    return {
        "rate_both_finite": counts["n_both_finite"] / n_complete,
        "rate_mc_finite_ref_timeout": (
            counts["n_mc_finite_ref_timeout"] / n_complete
        ),
        "rate_mc_timeout_ref_finite": (
            counts["n_mc_timeout_ref_finite"] / n_complete
        ),
        "rate_both_timeout": counts["n_both_timeout"] / n_complete,
    }


# ---------------------------------------------------------------------------
# Derived reporting encodings (R-046) and the two named estimands (R-048/49)
# ---------------------------------------------------------------------------


def sentinel_coded_stop(record: dict[str, Any], prefix: str) -> int:
    """The DERIVED reporting scalar for one arm: the preserved fair-QA
    producer's ``timeout_coded_as_horizon`` convention — per-side
    ``min(stop, horizon)``, with NEVER_STOPPED coded as the horizon. It never
    replaces or overwrites the canonical event (R-046)."""
    horizon = record["trajectory_horizon"]
    stop = record[f"{prefix}_stop_step"]
    if record[f"{prefix}_event_status"] == EVENT_NEVER or stop is None:
        return int(horizon)
    return int(min(stop, horizon))


def sentinel_coded_headline_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """The HEADLINE historical estimand (R-048): mean sentinel-coded signed
    shift ``MC − REF`` over ALL complete pairs. Positive values mean the QA
    reference stops earlier."""
    _check_duplicate_keys(records)
    shifts: list[float] = []
    for record in records:
        outcome = classify_record(record)
        if outcome["status"] != "complete":
            continue
        shifts.append(
            float(
                sentinel_coded_stop(record, "mc")
                - sentinel_coded_stop(record, "ref")
            )
        )
    d = np.array(shifts, dtype=np.float64)
    return {
        "estimand_label": HEADLINE_ESTIMAND_LABEL,
        "population": POPULATION_ALL,
        "n": int(d.size),
        "mean_signed_shift": float(np.mean(d)) if d.size else None,
        "convention": SENTINEL_CONVENTION,
    }


def finite_only_timing_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """The separately named SECONDARY estimand (R-049): finite-only timing
    summary over exactly ``n_both_finite`` items; reproduces the historical
    paired summary on all-finite record sets (R-010)."""
    _check_duplicate_keys(records)
    shifts: dict[str, int] = {}
    for record in records:
        outcome = classify_record(record)
        if (
            outcome["status"] == "complete"
            and outcome["joint_class"] == "both_finite"
        ):
            shifts[record["item_key"]] = (
                record["mc_stop_step"] - record["ref_stop_step"]
            )
    d = np.array([shifts[key] for key in sorted(shifts)], dtype=np.float64)
    if d.size == 0:
        return {
            "estimand_label": FINITE_ONLY_ESTIMAND_LABEL,
            "population": POPULATION_FINITE,
            "n": 0,
            "signed_index_mean": None,
            "signed_index_median": None,
            "absolute_index_mean": None,
            "absolute_index_median": None,
        }
    absd = np.abs(d)
    return {
        "estimand_label": FINITE_ONLY_ESTIMAND_LABEL,
        "population": POPULATION_FINITE,
        "n": int(d.size),
        "signed_index_mean": float(np.mean(d)),
        "signed_index_median": float(np.median(d)),
        "absolute_index_mean": float(np.mean(absd)),
        "absolute_index_median": float(np.median(absd)),
    }


# ---------------------------------------------------------------------------
# R-047: ambiguous legacy terminal sentinel
# ---------------------------------------------------------------------------


def _normalized_finite(value: int) -> dict[str, Any]:
    """One normalized FINITE_STOP outcome (fresh dict per call)."""
    return {
        "event_status": EVENT_FINITE,
        "stop_step": value,
        "terminal_imputation": "NONE",
    }


def _normalized_never() -> dict[str, Any]:
    """One normalized NEVER_STOPPED outcome (fresh dict per call)."""
    return {
        "event_status": EVENT_NEVER,
        "stop_step": None,
        "terminal_imputation": IMPUTATION_FINAL_PREFIX,
    }


def normalize_legacy_terminal(
    value: int,
    *,
    horizon: int,
    authenticated_convention: str | None,
    crossing_indicator: bool | None,
) -> dict[str, Any]:
    """Normalize one legacy stop value into the canonical event vocabulary.

    An ambiguous legacy ``T−1`` value cannot be normalized merely because
    doing so makes a table convenient: absent an authenticated producer
    convention or an explicit crossing indicator, it is EXCLUDED with the
    named reason ``AMBIGUOUS_TERMINAL_SENTINEL`` and stays in the legacy
    representation, never silently promoted (R-047).
    """
    if not _is_valid_horizon(horizon):
        raise ColmAimsError(
            f"legacy normalization requires a positive integer horizon, got"
            f" {horizon!r} (R-061)"
        )
    if not is_real_int(value):
        raise ColmAimsError(
            f"legacy normalization requires an integer legacy value, got"
            f" {value!r} (R-047)"
        )
    if crossing_indicator is not None:
        return (
            _normalized_finite(value)
            if crossing_indicator
            else _normalized_never()
        )
    if authenticated_convention == SENTINEL_CONVENTION:
        # Under timeout_coded_as_horizon, only the horizon value encodes a
        # timeout; T−1 is an ordinary in-range finite stop.
        if value >= horizon:
            return _normalized_never()
        return _normalized_finite(value)
    if authenticated_convention == "timeout_coded_as_final_index":
        if value == horizon - 1:
            return _normalized_never()
        if 0 <= value < horizon:
            return _normalized_finite(value)
    if authenticated_convention is None and 0 <= value < horizon - 1:
        # Unambiguously in-range: an ordinary finite stop.
        return _normalized_finite(value)
    # Ambiguous terminal (T−1 or beyond) without an authenticated convention
    # or crossing indicator: refuse to promote; exclude with the named
    # reason; keep the legacy representation.
    return {
        "excluded": True,
        "exclusion_reason": AMBIGUOUS_TERMINAL_SENTINEL,
        "legacy_value": value,
    }


# ---------------------------------------------------------------------------
# Item keys, estimand digests, comparability (R-008/R-011)
# ---------------------------------------------------------------------------


def derive_item_key(source_text: str) -> str:
    """Stable opaque item-key derivation pinned by the profile (R-008):
    itm-<first 16 hex of sha256(NFC-normalized text, utf-8)>."""
    if not isinstance(source_text, str):
        raise ColmAimsError("item-key derivation requires text (R-008)")
    normalized = unicodedata.normalize("NFC", source_text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return "itm-" + digest[:16]


def keyset_sha256(keys: list[str]) -> str:
    """Pinned key-set hash: sha256 over SORTED keys joined by newlines."""
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def item_order_sha256(ordered_keys: list[str]) -> str:
    """R-050: digest over keys IN THE ORDER USED for the vectors."""
    return hashlib.sha256("\n".join(ordered_keys).encode("utf-8")).hexdigest()


def canonical_item_order(keys: list[str]) -> list[str]:
    """R-050: ascending UTF-8 byte order — the ONE canonical item order."""
    return sorted(keys, key=lambda key: key.encode("utf-8"))


def estimand_digest(estimand: dict[str, Any]) -> str:
    """Digest over all estimand-defining fields (R-011, R-032)."""
    if not isinstance(estimand, dict):
        raise ColmAimsError("estimand must be an object (R-011)")
    return canonical_estimand_digest(estimand)


def _cell_digest_checked(cell: dict[str, Any]) -> str:
    recorded = cell.get("estimand_digest")
    recomputed = estimand_digest(cell.get("estimand") or {})
    if recorded != recomputed:
        raise EstimandMismatchError(
            f"cell {cell.get('cell_id')!r} recorded estimand_digest does not"
            " match the digest recomputed over its estimand-defining fields"
            " (R-011)"
        )
    return recomputed


def check_comparable(cell_a: dict[str, Any], cell_b: dict[str, Any]) -> None:
    """Refuse pooling/comparison across differing estimand digests (R-011).

    PRODUCTION-WIRED: every production comparison/pooling site calls this
    (the verifier's Holm-family pooling gate compares the axis-stripped
    residual estimands of all ten cells through it).
    """
    digest_a = _cell_digest_checked(cell_a)
    digest_b = _cell_digest_checked(cell_b)
    if digest_a != digest_b:
        raise EstimandMismatchError(
            f"cells {cell_a.get('cell_id')!r} and {cell_b.get('cell_id')!r}"
            " carry differing estimand digests; pooling or comparing them is"
            " refused (R-011)"
        )


# ---------------------------------------------------------------------------
# D7(b) regenerated inference procedure (sign-off §3, exact; R-050..R-056)
# ---------------------------------------------------------------------------

_SEED_DOMAIN_PREFIX = b"colm_aims_2026/v2/bootstrap_holm\0"


def d7b_seed(pairing_population_keyset_sha256: str) -> int:
    """R-052: deterministic seed, no outcome-dependent author choice."""
    seed_material = _SEED_DOMAIN_PREFIX + bytes.fromhex(
        pairing_population_keyset_sha256
    )
    return int.from_bytes(
        hashlib.sha256(seed_material).digest()[:8],
        byteorder="big",
        signed=False,
    )


@functools.lru_cache(maxsize=4)
def _cached_resample_matrix(seed: int, n: int, b: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, n, size=(b, n), dtype=np.int64, endpoint=False)
    indices.setflags(write=False)
    return indices


def d7b_resample_matrix(
    seed: int,
    *,
    n: int = EXPECTED_COMPLETE_PAIRS,
    b: int = BOOTSTRAP_DRAW_COUNT,
) -> np.ndarray:
    """R-051: the ONE collection-level paired-item resample matrix, shared by
    all ten cells; regenerated in-memory from the recorded seed — never
    deserialized from binary (R-034). Returned read-only."""
    if not is_uint64(seed):
        raise ColmAimsError(
            "resample seed must be a real integer in the unsigned 64-bit"
            " domain (R-052/R-061)"
        )
    if not is_real_int(n) or not is_real_int(b) or n <= 0 or b <= 0:
        raise ColmAimsError("resample plan dimensions must be positive ints")
    if b > MAX_BOOTSTRAP_DRAWS or b * n > MAX_BOOTSTRAP_CELLS:
        raise ColmAimsError(
            "resample plan exceeds the operational allocation safeguards"
            " (R-061)"
        )
    return _cached_resample_matrix(seed, n, b)


def d7b_matrix_digest_record(
    indices: np.ndarray, canonical_item_order_digest: str
) -> dict[str, Any]:
    """R-053: digest over the exact resample-index bytes + the four covering
    fields (dtype, shape, byte order, item-order digest)."""
    return {
        "sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "dtype": str(indices.dtype),
        "shape": list(indices.shape),
        "byte_order": sys.byteorder,
        "canonical_item_order_digest": canonical_item_order_digest,
    }


def d7b_interval(d: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    """R-054: uncentered percentile interval (2.5/97.5, method='linear')."""
    boot_means = d[indices].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def d7b_p_value(d: np.ndarray, indices: np.ndarray) -> float:
    """R-055: null-centered paired bootstrap p over the SAME index matrix;
    ``p = (1 + #{|mu0_b| >= |mu_hat|}) / (B + 1)`` — the +1 is MANDATORY."""
    b = int(indices.shape[0])
    mu_hat = float(np.mean(d))
    z = d - mu_hat
    null_means = z[indices].mean(axis=1)
    exceed = int(np.sum(np.abs(null_means) >= abs(mu_hat)))
    return (1 + exceed) / (b + 1)


def d7b_holm(raw_p_by_cell: dict[str, float]) -> dict[str, Any]:
    """R-056: Holm step-down, m=10, alpha 0.05, ascending raw p, ties by
    ascending UTF-8 byte order of cell_id; adjusted p capped at 1 with
    step-down monotonicity."""
    if len(raw_p_by_cell) != 10:
        raise ColmAimsError(
            f"Holm family must be exactly the ten-cell 5x2 grid (m=10);"
            f" got {len(raw_p_by_cell)} cells (R-056)"
        )
    m = 10
    alpha = 0.05
    ordered = sorted(
        raw_p_by_cell.items(),
        key=lambda kv: (kv[1], kv[0].encode("utf-8")),
    )
    per_cell: dict[str, dict[str, Any]] = {}
    adjusted_running = 0.0
    still_rejecting = True
    rejected: list[str] = []
    for rank0, (cell_id, p) in enumerate(ordered):
        mult = m - rank0
        adjusted_running = max(adjusted_running, min(1.0, p * mult))
        if still_rejecting and p <= alpha / mult:
            rejected.append(cell_id)
        else:
            still_rejecting = False
        per_cell[cell_id] = {
            "holm_rank": rank0 + 1,
            "holm_adjusted_p_value": adjusted_running,
            "holm_rejected": cell_id in rejected,
        }
    return {
        "ordered_family": [cell_id for cell_id, _ in ordered],
        "rejected_cell_ids": sorted(rejected),
        "per_cell": per_cell,
        "familywise_alpha": alpha,
        "family_size": m,
    }


def sentinel_coded_shift_vector(
    records_by_key: dict[str, dict[str, Any]], ordered_keys: list[str]
) -> np.ndarray:
    """R-050: the canonical per-item signed difference vector
    ``d_i = s^MC_i − s^REF_i`` (sentinel-coded, terminal-imputed), over the
    canonical item order."""
    values = np.empty(len(ordered_keys), dtype=np.int64)
    for index, key in enumerate(ordered_keys):
        record = records_by_key[key]
        values[index] = sentinel_coded_stop(record, "mc") - sentinel_coded_stop(
            record, "ref"
        )
    return values.astype(np.float64)
