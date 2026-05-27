#!/usr/bin/env python3
"""Compute finite-horizon DP StopDFF.

Supplementary metric to scripts/compute_stopdff.py: replaces the myopic
threshold stopping rule with an explicit backward-induction policy over
calibrated prefix trajectories. Writes JSON/MD/LaTeX exports under
``paper_exports/``. See docs/superpowers/plans/2026-05-27-stopdff-dp.md
for the design rationale.

Usage:
    python scripts/compute_stopdff_dp.py --help
    python scripts/compute_stopdff_dp.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CALIBRATION = PROJECT_ROOT / "paper_exports" / "calibration.json"
DEFAULT_OUT_JSON = PROJECT_ROOT / "paper_exports" / "stopdff_dp.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "paper_exports" / "stopdff_dp.md"
DEFAULT_OUT_TEX = PROJECT_ROOT / "paper_exports" / "stopdff_dp_table.tex"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute finite-horizon DP StopDFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--responses", default=None,
                        help="(unused; reserved for future responses.json input)")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--split", default="test")
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--reward-schedule", default="power_mark")
    parser.add_argument("--continuation", default="empirical_bucket",
                        choices=[
                            "oracle_trajectory", "empirical_bucket",
                            "pooled_empirical",
                        ])
    parser.add_argument("--out", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--out-tex", default=str(DEFAULT_OUT_TEX))
    parser.add_argument("--smoke", action="store_true",
                        help="Trim to 30 val + 30 test questions for a <5min run.")
    parser.add_argument("--identity-calibration", action="store_true",
                        help="Skip SBERT and use deterministic synthetic signal (test only).")
    parser.add_argument("--allow-incomplete-mc-coverage", action="store_true")
    parser.add_argument("--allow-low-mc-retention", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    effective_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = _parse_args(argv)

    if args.allow_incomplete_mc_coverage or args.allow_low_mc_retention:
        print(
            "WARNING: --allow-incomplete-mc-coverage and "
            "--allow-low-mc-retention are accepted for CLI symmetry with "
            "scripts/compute_stopdff.py but currently no-op in the DP "
            "StopDFF script. Coverage and retention gates may be added in a "
            "future revision; for now these flags do not affect any check.",
            file=sys.stderr,
        )

    data_dir = Path(args.data_dir)
    out_json = Path(args.out)
    out_md = Path(args.out_md)
    out_tex = Path(args.out_tex)

    from scripts.stopdff_dp import adapter as adapter_module
    from scripts.stopdff_dp import diagnostics as diag_module
    from scripts.stopdff_dp import dp_solver as dp_module
    from scripts.stopdff_dp import rewards as rewards_module
    from scripts.stopdff_dp import writers as writers_module
    from scripts.stopdff_dp.continuation import (
        EmpiricalBucketEstimator,
        OracleTrajectoryEstimator,
        PooledEmpiricalEstimator,
        _assign_entropy_bin,
        _assign_p_bin,
        _assign_prefix_bucket,
    )

    try:
        adapter_module.validate_split_separation(
            fit_split=args.fit_split, eval_split=args.split
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.identity_calibration:
        if not Path(args.calibration).exists():
            print(
                f"ERROR: calibration JSON not found: {args.calibration}",
                file=sys.stderr,
            )
            return 1

    # Load split datasets.
    mc_path = data_dir / "mc_dataset.json"
    fit_path = data_dir / f"{args.fit_split}_dataset.json"
    eval_path = data_dir / f"{args.split}_dataset.json"
    for p in (mc_path, fit_path, eval_path):
        if not p.exists():
            print(f"ERROR: missing dataset {p}", file=sys.stderr)
            return 1

    from scripts._common import load_json, iter_split_questions
    mc_questions = iter_split_questions(load_json(mc_path), source_path=mc_path)
    fit_questions = iter_split_questions(load_json(fit_path), source_path=fit_path)
    eval_questions = iter_split_questions(load_json(eval_path), source_path=eval_path)
    fit_qids = {str(q["qid"]) for q in fit_questions}
    eval_qids = {str(q["qid"]) for q in eval_questions}

    if args.smoke:
        # Subsample mc_questions to first 30 qids of each split.
        # sorted() ensures deterministic selection across PYTHONHASHSEED;
        # list(set) iteration order is salt-dependent.
        keep_fit = sorted(fit_qids)[:30]
        keep_eval = sorted(eval_qids)[:30]
        fit_qids = set(keep_fit)
        eval_qids = set(keep_eval)
        kept_qids = fit_qids | eval_qids
        mc_questions = [
            q for q in mc_questions if str(q["qid"]) in kept_qids
        ]

    calibration_path = Path(args.calibration) if not args.identity_calibration else None
    fit_df = adapter_module.build_dataframe(
        mc_questions=mc_questions,
        target_qids=fit_qids,
        split_name=args.fit_split,
        calibration_path=calibration_path,
        identity_calibration=args.identity_calibration,
    )
    eval_df = adapter_module.build_dataframe(
        mc_questions=mc_questions,
        target_qids=eval_qids,
        split_name=args.split,
        calibration_path=calibration_path,
        identity_calibration=args.identity_calibration,
    )

    schedule = rewards_module.get_schedule(args.reward_schedule)

    # Build continuation estimator. All three .fit() signatures now accept
    # the same keyword arguments after Task 4's review fix unifying them.
    if args.continuation == "oracle_trajectory":
        estimator: object = OracleTrajectoryEstimator.fit(
            fit_df=fit_df, fit_split_name=args.fit_split,
        )
    elif args.continuation == "pooled_empirical":
        estimator = PooledEmpiricalEstimator.fit(
            fit_df=fit_df, fit_split_name=args.fit_split,
        )
    else:  # empirical_bucket
        estimator = EmpiricalBucketEstimator.fit(
            fit_df=fit_df, fit_split_name=args.fit_split,
        )

    # Run DP per (item, format) over the eval split.
    mc_traces: list = []
    qa_traces: list = []
    per_item_stopdff: list[tuple[str, int]] = []
    for item_id, group in eval_df.groupby("item_id"):
        group = group.sort_values("prefix_idx")
        mc_rows = group[group["format"] == "MC"]
        qa_rows = group[group["format"] == "QA"]
        if mc_rows.empty or qa_rows.empty:
            continue

        def _run(rows: pd.DataFrame, fmt: str):
            ps = rows["p_calibrated"].tolist()
            T = len(ps)
            prefix_fractions = [(i + 1) / T for i in range(T)]

            # Tag-capture pattern: solve_trajectory calls _continuation
            # once per backward step (t = T-2 .. 0). We record the
            # estimator's per-step tag immediately, then replay it from
            # the dict in _coverage_tagger after the backward loop ends.
            # Without this, _last_tag would be overwritten by every
            # successive call and the trace would record a single tag
            # for every step (the bug from the v1 draft of this plan).
            tags_per_step: dict[int, str] = {(T - 1): "exact"}

            def _continuation(t, p, prefix_fraction, _fmt=fmt, _ps=ps):
                if isinstance(estimator, OracleTrajectoryEstimator):
                    tags_per_step[t] = "exact"
                    return estimator.estimate(item_trajectory=_ps, t=t)
                v = estimator.estimate(
                    prefix_bucket=_assign_prefix_bucket(prefix_fraction),
                    fmt=_fmt,
                    subject_bucket=rows["subject"].iloc[0],
                    p_bin=_assign_p_bin(p),
                    entropy_bin=_assign_entropy_bin(p),
                )
                tags_per_step[t] = getattr(estimator, "_last_tag", "exact")
                return v

            def _coverage_tagger(t):
                return tags_per_step.get(t, "exact")

            return dp_module.solve_trajectory(
                p_trajectory=ps,
                prefix_fractions=prefix_fractions,
                schedule=schedule,
                continuation_fn=_continuation,
                item_id=str(item_id),
                fmt=fmt,
                coverage_tagger=_coverage_tagger,
            )

        mc_trace = _run(mc_rows, "MC")
        qa_trace = _run(qa_rows, "QA")
        mc_traces.append(mc_trace)
        qa_traces.append(qa_trace)
        per_item_stopdff.append(
            (str(item_id), dp_module.stopdff_for_item(
                mc_trace=mc_trace, qa_trace=qa_trace
            ))
        )

    coverage_summary = diag_module.summarize_coverage(mc_traces + qa_traces)
    ceiling_flags = diag_module.detect_ceiling_effects(mc_traces, qa_traces)

    confirmatory = not isinstance(estimator, OracleTrajectoryEstimator)
    if not confirmatory:
        print(
            "WARNING: oracle_trajectory continuation is upper-bound diagnostic only; "
            "output is flagged confirmatory=false.",
            file=sys.stderr,
        )

    if coverage_summary["verdict"] == "warn":
        gate_verdict = "warn"
        gate_verdict_reason = f"coverage:{coverage_summary['reason']}"
    elif any(ceiling_flags[k] for k in (
        "all_stop_at_first_prefix",
        "all_stop_at_final_prefix",
        "no_cross_format_stopping_variance",
    )):
        gate_verdict = "warn"
        gate_verdict_reason = "ceiling_effect"
    else:
        gate_verdict = "pass"
        gate_verdict_reason = "all_clean"

    # Provenance.
    try:
        from scripts._common import build_generation_provenance
        generation = build_generation_provenance(
            __file__, effective_argv,
            output_path=out_json,
            extra_paths=[calibration_path] if calibration_path else [],
        )
    except Exception:  # noqa: BLE001 — provenance is best-effort
        generation = None

    payload = writers_module.assemble_payload(
        mc_traces=mc_traces,
        qa_traces=qa_traces,
        reward_schedule_name=args.reward_schedule,
        continuation_estimator_name=args.continuation,
        fit_split=args.fit_split,
        eval_split=args.split,
        coverage_summary=coverage_summary,
        ceiling_flags=ceiling_flags,
        per_item_stopdff=per_item_stopdff,
        gate_verdict=gate_verdict,
        gate_verdict_reason=gate_verdict_reason,
        confirmatory=confirmatory,
        generation=generation,
    )

    writers_module.write_json(out_json, payload)
    writers_module.write_markdown(out_md, payload)
    writers_module.write_latex(out_tex, payload)

    print(
        f"[STOPDFF-DP] Wrote {out_json} (verdict={gate_verdict}, "
        f"n_items={len(per_item_stopdff)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
