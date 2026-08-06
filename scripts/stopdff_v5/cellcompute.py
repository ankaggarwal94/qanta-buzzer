"""Shared per-cell computation (calibrate -> continuation -> FVI -> solve -> index shifts).

Used by both the sweep (metric_split='test') and the FVI study (metric_split='val',
fit-only). Operates on in-memory adapter rows so it is unit-testable on synthetic data;
real-data invocation happens on Modal.

Adapter row schema (indexed by SCIENTIFIC_CONTRACT.md), minimum required here:
    item_id, prefix_idx, prefix_fraction, format ("MC"|"QA"), split,
    raw_similarity, correct, category
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .calibrators import Calibrator, CalibratorFitError, fit_calibrator
from .continuation import ContinuationEstimator, build_counts, make_observation
from .fvi import FitTrajectory, FVIResult, run_fvi
from .policy import DPTrace, solve_trajectory
from .rewards import get_schedule
from .verdicts import TraceStop


@dataclass
class _Traj:
    item_id: str
    fmt: str
    category: str
    prefix_idx: list[int]
    prefix_fractions: list[float]
    raw_similarity: list[float]
    correct: list[int]


@dataclass
class CellResult:
    status: str  # completed | calibrator_failed | fvi_failed
    reason: str = ""
    fvi: FVIResult | None = None
    calibrator_parameters: dict[str, Any] = field(default_factory=dict)
    index_shift_by_item: dict[str, int] = field(default_factory=dict)
    stop_index_by_key: dict[str, int] = field(default_factory=dict)  # "item|fmt" -> stop_index
    coverage: dict[str, float] = field(default_factory=dict)
    mc_stops: list[TraceStop] = field(default_factory=list)
    qa_stops: list[TraceStop] = field(default_factory=list)
    never_buzz_mc: int = 0
    never_buzz_qa: int = 0
    descriptive: dict[str, Any] = field(default_factory=dict)


def _group_trajectories(rows: Sequence[dict], split: str) -> list[_Traj]:
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["split"] != split:
            continue
        buckets.setdefault((str(r["item_id"]), str(r["format"])), []).append(r)
    trajs: list[_Traj] = []
    for (item_id, fmt), group in buckets.items():
        group.sort(key=lambda x: int(x["prefix_idx"]))
        trajs.append(
            _Traj(
                item_id=item_id,
                fmt=fmt,
                category=str(group[0].get("category", "")),
                prefix_idx=[int(x["prefix_idx"]) for x in group],
                prefix_fractions=[float(x["prefix_fraction"]) for x in group],
                raw_similarity=[float(x["raw_similarity"]) for x in group],
                correct=[int(x["correct"]) for x in group],
            )
        )
    trajs.sort(key=lambda t: (t.item_id, t.fmt))
    return trajs


def _calibrated_p(cal: Calibrator, traj: _Traj) -> list[float]:
    return [cal.apply(traj.raw_similarity[t], traj.prefix_fractions[t]) for t in range(len(traj.prefix_idx))]


def _obs_for(traj: _Traj, p_traj: list[float], prefix_bucketing: str) -> list[dict]:
    # observation per nonterminal t (len T-1)
    return [
        make_observation(
            prefix_bucketing=prefix_bucketing,
            prefix_idx=traj.prefix_idx[t],
            prefix_fraction=traj.prefix_fractions[t],
            fmt=traj.fmt,
            category=traj.category,
            p_calibrated=p_traj[t],
        )
        for t in range(len(p_traj) - 1)
    ]


def compute_cell(
    *,
    rows: Sequence[dict],
    cell: dict[str, str],
    calibration_json: dict | None,
    tolerance: float,
    max_iterations: int,
    tolerance_label: str = "",
    metric_split: str = "test",
) -> CellResult:
    schedule = get_schedule(cell["reward_schedule"])
    prefix_bucketing = cell["prefix_bucketing"]

    val_mc_rows = [r for r in rows if r["split"] == "val" and r["format"] == "MC"]

    # 1) calibrator (validation MC only)
    try:
        cal = fit_calibrator(
            cell["calibrator"], mc_val_rows=val_mc_rows, calibration_json=calibration_json
        )
    except CalibratorFitError as exc:
        return CellResult(status="calibrator_failed", reason=str(exc))

    # 2) fit trajectories (val, both formats) + continuation estimator
    fit_trajs = _group_trajectories(rows, "val")
    fvi_fit: list[FitTrajectory] = []
    for tr in fit_trajs:
        p = _calibrated_p(cal, tr)
        obs = _obs_for(tr, p, prefix_bucketing)
        fvi_fit.append(
            FitTrajectory(
                item_id=tr.item_id, fmt=tr.fmt, category=tr.category,
                p_trajectory=p, prefix_fractions=tr.prefix_fractions, obs_at_t=obs,
            )
        )
    estimator = ContinuationEstimator(cell["continuation"], cell["category_pooling"])
    build_counts(estimator, [t.obs_at_t for t in fvi_fit])

    # 3) FVI
    fvi_res = run_fvi(
        estimator, fvi_fit, schedule,
        tolerance=tolerance, max_iterations=max_iterations, tolerance_label=tolerance_label,
    )
    if not fvi_res.converged:
        return CellResult(status="fvi_failed", reason=f"fvi {fvi_res.status}", fvi=fvi_res,
                          calibrator_parameters=cal.parameters())

    # 4) solve metric split (both formats), collect stop indices per item
    metric_trajs = _group_trajectories(rows, metric_split)
    mc_stop: dict[str, DPTrace] = {}
    qa_stop: dict[str, DPTrace] = {}
    cov_primary = cov_fallback = cov_missing = 0

    for tr in metric_trajs:
        p = _calibrated_p(cal, tr)
        obs = _obs_for(tr, p, prefix_bucketing)

        def _cont(t, p, prefix_fraction, _obs=obs):
            return estimator.estimate(_obs[t])

        def _cov(t, p, prefix_fraction, _obs=obs):
            # terminal has no continuation lookup; tag only nonterminal
            if t >= len(_obs):
                return "primary"
            return estimator.coverage_tag(_obs[t])

        trace = solve_trajectory(
            p_trajectory=p, prefix_fractions=tr.prefix_fractions, schedule=schedule,
            continuation_fn=_cont, item_id=tr.item_id, fmt=tr.fmt, coverage_tagger=_cov,
        )
        # coverage over nonterminal decision points only
        for t in range(len(obs)):
            tag = estimator.coverage_tag(obs[t])
            if tag == "primary":
                cov_primary += 1
            elif tag == "fallback":
                cov_fallback += 1
            else:
                cov_missing += 1
        if tr.fmt == "MC":
            mc_stop[tr.item_id] = trace
        else:
            qa_stop[tr.item_id] = trace

    # 5) paired index shift per item present in both formats
    index_shift: dict[str, int] = {}
    for item_id in sorted(set(mc_stop) & set(qa_stop)):
        index_shift[item_id] = mc_stop[item_id].stop_index - qa_stop[item_id].stop_index

    stop_index_by_key: dict[str, int] = {}
    for item_id, trace in mc_stop.items():
        stop_index_by_key[f"{item_id}|MC"] = trace.stop_index
    for item_id, trace in qa_stop.items():
        stop_index_by_key[f"{item_id}|QA"] = trace.stop_index

    total_cov = cov_primary + cov_fallback + cov_missing
    coverage = {
        "primary_fraction": (cov_primary / total_cov) if total_cov else 0.0,
        "fallback_fraction": (cov_fallback / total_cov) if total_cov else 0.0,
        "missing_fraction": (cov_missing / total_cov) if total_cov else 0.0,
        "decision_points": total_cov,
    }

    mc_stops = [TraceStop(mc_stop[i].stop_index, mc_stop[i].never_buzz, mc_stop[i].T) for i in sorted(mc_stop)]
    qa_stops = [TraceStop(qa_stop[i].stop_index, qa_stop[i].never_buzz, qa_stop[i].T) for i in sorted(qa_stop)]
    never_mc = sum(1 for t in mc_stop.values() if t.never_buzz)
    never_qa = sum(1 for t in qa_stop.values() if t.never_buzz)

    return CellResult(
        status="completed",
        fvi=fvi_res,
        calibrator_parameters=cal.parameters(),
        index_shift_by_item=index_shift,
        stop_index_by_key=stop_index_by_key,
        coverage=coverage,
        mc_stops=mc_stops,
        qa_stops=qa_stops,
        never_buzz_mc=never_mc,
        never_buzz_qa=never_qa,
        descriptive={
            "metric_split": metric_split,
            "n_paired_items": len(index_shift),
            "never_buzz_mc": never_mc,
            "never_buzz_qa": never_qa,
        },
    )
