"""Common paired bootstrap (indexed by SCIENTIFIC_CONTRACT.md).

One bootstrap plan for the complete evaluation item set:
    sorted item IDs; numpy Generator with PCG64; seed = 1;
    replicates = 1000 (final) / 100 (smoke);
    sample N item indices with replacement per replicate.
The same resample indices are used for all cells; MC/QA observations for an item stay paired.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

RESAMPLE_DTYPE = np.int64


@dataclass
class BootstrapPlan:
    item_ids: list[str]  # sorted
    resample_indices: np.ndarray  # shape (replicates, N), dtype int64
    replicates: int
    seed: int
    numpy_version: str
    bit_generator: str
    item_id_list_sha256: str
    resample_index_sha256: str

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def _item_id_list_sha(item_ids: Sequence[str]) -> str:
    payload = "\n".join(item_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resample_sha(indices: np.ndarray) -> str:
    arr = np.ascontiguousarray(indices, dtype=RESAMPLE_DTYPE)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def build_bootstrap_plan(item_ids: Sequence[str], *, replicates: int, seed: int = 1) -> BootstrapPlan:
    sorted_ids = sorted(str(i) for i in item_ids)
    n = len(sorted_ids)
    if n == 0:
        raise ValueError("cannot build a bootstrap plan over an empty item set")
    rng = np.random.Generator(np.random.PCG64(seed))
    resample = rng.integers(low=0, high=n, size=(replicates, n), dtype=RESAMPLE_DTYPE)
    return BootstrapPlan(
        item_ids=sorted_ids,
        resample_indices=resample,
        replicates=replicates,
        seed=seed,
        numpy_version=np.__version__,
        bit_generator="PCG64",
        item_id_list_sha256=_item_id_list_sha(sorted_ids),
        resample_index_sha256=_resample_sha(resample),
    )


def plan_identity(plan: BootstrapPlan) -> dict:
    """Identity block for the bootstrap-plan manifest (IDENTITY 8)."""
    return {
        "evaluation_item_id_list_sha256": plan.item_id_list_sha256,
        "bit_generator": plan.bit_generator,
        "seed": plan.seed,
        "replicate_count": plan.replicates,
        "resample_index_sha256": plan.resample_index_sha256,
        "numpy_version_contract": plan.numpy_version,
        "resample_dtype": "int64",
    }


def _percentile_ci(samples: np.ndarray) -> list[float]:
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return [lo, hi]


def cell_bootstrap_stats(index_shift_by_item: dict[str, int], plan: BootstrapPlan) -> dict:
    """Compute point estimates + 95% percentile CIs for one cell.

    ``index_shift_by_item`` maps item_id -> signed D_i (tau_MC - tau_QA). Items are ordered
    by plan.item_ids so the common resample indices apply.
    """
    d = np.array([float(index_shift_by_item[item]) for item in plan.item_ids], dtype=np.float64)
    absd = np.abs(d)

    point = {
        "signed_index_mean": float(np.mean(d)),
        "signed_index_median": float(np.median(d)),
        "absolute_index_mean": float(np.mean(absd)),
        "absolute_index_median": float(np.median(absd)),
    }

    resampled_d = d[plan.resample_indices]  # (R, N)
    resampled_abs = absd[plan.resample_indices]
    per_rep = {
        "signed_index_mean": np.mean(resampled_d, axis=1),
        "signed_index_median": np.median(resampled_d, axis=1),
        "absolute_index_mean": np.mean(resampled_abs, axis=1),
        "absolute_index_median": np.median(resampled_abs, axis=1),
    }
    ci = {name: _percentile_ci(arr) for name, arr in per_rep.items()}

    for name, (lo, hi) in ci.items():
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"non-finite CI for {name}: [{lo}, {hi}]")
        if lo > hi:
            raise ValueError(f"CI lower exceeds upper for {name}: [{lo}, {hi}]")

    return {
        "point": point,
        "ci": ci,
        "abs_median_point": point["absolute_index_median"],
        "abs_median_ci": ci["absolute_index_median"],
        "abs_median_replicates": per_rep["absolute_index_median"],
    }


def family_statistic(
    abs_median_replicates_by_cell: dict[str, np.ndarray],
    abs_median_point_by_cell: dict[str, float],
) -> dict:
    """Family maximum-statistic (indexed by SCIENTIFIC_CONTRACT.md).

    M_b = max_c median_i |D_{i,c}| for each replicate b; M = max_c m_c.
    """
    if not abs_median_replicates_by_cell:
        raise ValueError("no cells for family statistic")
    stacked = np.vstack([abs_median_replicates_by_cell[c] for c in sorted(abs_median_replicates_by_cell)])
    m_b = np.max(stacked, axis=0)  # (R,)
    ci = _percentile_ci(m_b)
    m_point = max(abs_median_point_by_cell.values())
    return {"M": float(m_point), "ci": [float(ci[0]), float(ci[1])], "m_b": m_b}
