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
from typing import Optional

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

    # PR #15 review (chatgpt-codex-connector P2 3313638262): the DP StopDFF
    # row is audit-card eligible, so it must enforce the same MC coverage
    # (>=98%) and retention gates that scripts/compute_stopdff.py enforces.
    # Missing MC rows are not random (items where good distractors could not
    # be built), so a partial subset would silently bias the DP metric.
    from scripts._audit_gates import (
        build_coverage_metadata,
        build_retention_metadata,
        filter_mc_questions_to_split,
        load_mc_build_metadata,
    )

    # Filter MC pool to eval split and compute coverage.
    _mc_eval_rows, eval_coverage = filter_mc_questions_to_split(
        mc_questions, eval_qids
    )
    print(
        f"[STOPDFF-DP] MC eval: matched {eval_coverage['matched_qids']} / "
        f"{eval_coverage['target_qids']} qids "
        f"({eval_coverage['coverage_rate']:.1%})",
        file=sys.stderr,
    )
    MIN_MC_COVERAGE = 0.98
    if (
        eval_coverage["coverage_rate"] < MIN_MC_COVERAGE
        and not args.allow_incomplete_mc_coverage
    ):
        print(
            f"ERROR: MC eval coverage is {eval_coverage['coverage_rate']:.1%} "
            f"(threshold: {MIN_MC_COVERAGE:.1%}). The DP StopDFF row is "
            f"audit-card eligible; missing MC qids are not random (selected "
            f"against 'hard to find distractors'). Pass "
            f"--allow-incomplete-mc-coverage to override.",
            file=sys.stderr,
        )
        return 1

    # Same coverage check for the fit split -- empirical bucket fit needs
    # representative val data.
    _mc_fit_rows, fit_coverage = filter_mc_questions_to_split(
        mc_questions, fit_qids
    )
    if (
        fit_coverage["coverage_rate"] < MIN_MC_COVERAGE
        and not args.allow_incomplete_mc_coverage
    ):
        print(
            f"ERROR: MC fit coverage is {fit_coverage['coverage_rate']:.1%} "
            f"(threshold: {MIN_MC_COVERAGE:.1%}). Pass "
            f"--allow-incomplete-mc-coverage to override.",
            file=sys.stderr,
        )
        return 1

    # Retention gate from build_metadata.json. Mirrors compute_stopdff.py.
    try:
        build_metadata = load_mc_build_metadata(data_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    eval_retention_meta = build_retention_metadata(
        build_metadata,
        split=args.split,
        smoke=args.smoke,
        explicit_threshold=None,
        override=args.allow_low_mc_retention,
    )
    if (
        eval_retention_meta["applies"]
        and eval_retention_meta["passed"] is False
        and not args.allow_low_mc_retention
    ):
        print(
            f"ERROR: raw-{args.split} MC retention is "
            f"{eval_retention_meta['retention_rate']:.1%} (threshold: "
            f"{eval_retention_meta['threshold']:.1%}). Pass "
            f"--allow-low-mc-retention only if you intend the DP StopDFF "
            f"artifact to qualify as a retained-MC-subset metric.",
            file=sys.stderr,
        )
        return 1

    fit_retention_meta = build_retention_metadata(
        build_metadata,
        split=args.fit_split,
        smoke=args.smoke,
        explicit_threshold=None,
        override=args.allow_low_mc_retention,
    )
    if (
        fit_retention_meta["applies"]
        and fit_retention_meta["passed"] is False
        and not args.allow_low_mc_retention
    ):
        print(
            f"ERROR: raw-{args.fit_split} MC retention is "
            f"{fit_retention_meta['retention_rate']:.1%} (threshold: "
            f"{fit_retention_meta['threshold']:.1%}). Pass "
            f"--allow-low-mc-retention only if you intend the DP StopDFF "
            f"artifact to qualify as a retained-MC-subset metric.",
            file=sys.stderr,
        )
        return 1

    # Build audit-card-ready metadata blocks for the writer.
    eval_coverage_metadata = build_coverage_metadata(
        eval_coverage,
        threshold=MIN_MC_COVERAGE,
        override=args.allow_incomplete_mc_coverage,
    )
    eval_coverage_metadata["split"] = args.split

    fit_coverage_metadata = build_coverage_metadata(
        fit_coverage,
        threshold=MIN_MC_COVERAGE,
        override=args.allow_incomplete_mc_coverage,
    )
    fit_coverage_metadata["split"] = args.fit_split

    mc_coverage_block = {
        args.split: eval_coverage_metadata,
        args.fit_split: fit_coverage_metadata,
    }
    mc_retention_block = {
        args.split: eval_retention_meta,
        args.fit_split: fit_retention_meta,
    }
    mc_build_metadata_block = {
        "status": build_metadata["status"],
        "source_path": build_metadata["source_path"],
        "source_sha256": build_metadata["source_sha256"],
    }

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
            fit_df=fit_df, schedule=schedule, fit_split_name=args.fit_split,
        )
    elif args.continuation == "pooled_empirical":
        estimator = PooledEmpiricalEstimator.fit(
            fit_df=fit_df, schedule=schedule, fit_split_name=args.fit_split,
        )
    else:  # empirical_bucket
        estimator = EmpiricalBucketEstimator.fit(
            fit_df=fit_df, schedule=schedule, fit_split_name=args.fit_split,
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
            # PR #15 review (Copilot 3313507021/3313507055): read the
            # adapter-provided prefix_fraction (len(prefix)/len(full_question))
            # instead of rank/T even-spacing so the DP early/late split and
            # bucket assignment agree with scripts/compute_prefix_calibration.py
            # and scripts/compute_stopdff.py.
            prefix_fractions = rows["prefix_fraction"].tolist()

            # Tag-capture pattern: solve_trajectory calls _continuation
            # once per backward step (t = T-2 .. 0). We record the
            # estimator's per-step tag immediately, then replay it from
            # the dict in _coverage_tagger after the backward loop ends.
            # Without this, _last_tag would be overwritten by every
            # successive call and the trace would record a single tag
            # for every step (the bug from the v1 draft of this plan).
            tags_per_step: dict[int, str] = {(T - 1): "exact"}

            def _continuation(t, p, prefix_fraction, _fmt=fmt, _ps=ps, _pfs=prefix_fractions):
                if isinstance(estimator, OracleTrajectoryEstimator):
                    tags_per_step[t] = "exact"
                    return estimator.estimate(
                        item_trajectory=_ps,
                        item_prefix_fractions=_pfs,
                        t=t,
                        schedule=schedule,
                    )
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
        from scripts.stopdff_dp._provenance import helper_sha256s
        generation = build_generation_provenance(
            __file__, effective_argv,
            output_path=out_json,
            extra_paths=[calibration_path] if calibration_path else [],
        )
        # PR #15 review (chatgpt-codex-connector 3314086941): the DP
        # producer's behavior is mostly delegated to imported helpers
        # under scripts/stopdff_dp/ + scripts/_audit_gates.py +
        # scripts/_common.py. Embed those module SHAs so the audit-card
        # consumer (make_audit_card._build_artifact_provenance) can
        # cross-check them and force the WARN downgrade when any helper
        # drifts after the JSON was committed.
        generation["helper_sha256s"] = helper_sha256s()
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
        mc_coverage=mc_coverage_block,
        mc_retention_gate=mc_retention_block,
        mc_build_metadata=mc_build_metadata_block,
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
