"""Fair-QA StopDFF retest: difficulty-matched QA arm + per-format calibration + bootstrap CIs.

Audits whether the MC<->QA reformulation changes optimal buzz timing (StopDFF) once the
QA arm is made difficulty-comparable to MC, rather than the idealized cos-to-gold/correct==1
arm (which is trivially easy and manufactures a "QA buzzes earlier" artifact).

QA arms (all K-way, K = the item's MC option count):
  - idealized : cos(prefix, answer_primary), correct==1            (production anchor; too easy)
  - krandom   : candidate set [gold] + (K-1) random pool answers   (easier than MC)
  - khard     : candidate set [gold] + (K-1) nearest-neighbour pool answers (difficulty-matched)
                NOTE: "hard" distractors are SBERT nearest neighbours in the same space used to
                score, so they are adversarially close to gold -- a circularity caveat. khard is
                the best available difficulty-matched proxy, not a clean open-ended measurement.

Scoring replicates the adapter formulas (MC = max cos over options, correct=argmax==gold_index;
QA arms scored over their candidate set) but uses BATCHED SBERT encoding so the full splits
score in minutes (the adapter's per-prefix encoding is too slow at full scale).

Calibration: shared (one per-bucket Platt) or performat (separate Platt per (format,bucket) fit
on the fit split). StopDFF = stop_step(MC) - stop_step(QA); >0 => QA earlier. Reuses the real
stopdff_dp solver + EmpiricalBucketEstimator (DP) and continuation==0 (myopic). Bootstrap CIs
are percentile intervals over an item resample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts._common import (  # noqa: E402
    build_generation_provenance,
    iter_split_questions,
    load_json,
    sha256_file,
)
from scripts.stopdff_dp._provenance import (  # noqa: E402
    helper_paths as coverage_helper_paths,
    helper_sha256s as coverage_helper_sha256s,
)
from scripts.stopdff_dp.adapter import (  # noqa: E402
    _get_sbert_model, _assign_bucket, _load_platt_params, _platt,
)
from scripts.stopdff_dp.rewards import get_schedule  # noqa: E402
from scripts.stopdff_dp.dp_solver import solve_trajectory, stopdff_for_item  # noqa: E402
from scripts.stopdff_dp.diagnostics import summarize_coverage  # noqa: E402
from scripts.stopdff_dp.continuation import (  # noqa: E402
    EmpiricalBucketEstimator, _assign_prefix_bucket, _assign_p_bin, _assign_entropy_bin,
)
from scripts.stopdff_dp.types import ADAPTER_COLUMNS  # noqa: E402
from scripts.compute_prefix_calibration import fit_platt  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

BUCKETS = ["early", "mid", "late"]
PRODUCER_SCRIPT_PATH = "scripts/stopdff_fair_qa_retest.py"


def _qids(path: Path) -> list[str]:
    return [str(q["qid"]) for q in iter_split_questions(load_json(path))]


def _encode(model, strings, batch_size=256):
    if not strings:
        return np.zeros((0, 384), dtype=float)
    return model.encode(list(strings), convert_to_numpy=True, batch_size=batch_size)


def _item_rng(seed: int, qid: str, arm: str) -> random.Random:
    """Return an RNG whose stream is stable for one item/arm.

    A process-wide seeded stream makes candidate assignment depend on question
    iteration order. Deriving the seed from stable serialized fields makes the
    result invariant to list order and ``PYTHONHASHSEED``.
    """
    material = f"{int(seed)}\0{qid}\0{arm}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(derived)


def score_arms(questions, split, model, pool, pool_embs, gold_to_idx, seed, arms, reps=None):
    """Batched scoring. Returns {arm: DataFrame[ADAPTER_COLUMNS]} for arms in
    {"MC","idealized","krandom","khard","kdisjoint","klex"}."""
    # Flattened batched encodes (deterministic; batch-invariant vs per-string).
    flat_pre, pre_slices = [], []
    flat_opt, opt_slices = [], []
    for q in questions:
        prefs = q["cumulative_prefixes"]
        pre_slices.append((len(flat_pre), len(flat_pre) + len(prefs))); flat_pre.extend(prefs)
        opts = q["options"]
        opt_slices.append((len(flat_opt), len(flat_opt) + len(opts))); flat_opt.extend(opts)
    pre_embs = _encode(model, flat_pre)
    opt_embs = _encode(model, flat_opt)
    ans_embs = _encode(model, [q["answer_primary"] for q in questions])

    out = {a: [] for a in arms}
    for qi, q in enumerate(questions):
        p0, p1 = pre_slices[qi]; qpre = pre_embs[p0:p1]
        o0, o1 = opt_slices[qi]; qopt = opt_embs[o0:o1]
        qans = ans_embs[qi:qi + 1]
        prefixes = q["cumulative_prefixes"]; full = max(1, len(q["question"]))
        gold = q["answer_primary"]; gold_index = int(q["gold_index"]); K = max(2, len(q["options"]))
        cat = q.get("category", ""); subj = f"sbert:{cat}"
        gi = gold_to_idx.get(gold)
        cand = {}
        candidate_arms = {"krandom", "khard", "kdisjoint", "klex"} & set(arms)
        qid = str(q["qid"])
        if candidate_arms and gi is None:
            raise ValueError(
                f"{qid}: gold answer {gold!r} is absent from the answer-profile pool"
            )
        if candidate_arms and len(pool) - 1 < K - 1:
            raise ValueError(
                f"{qid}: candidate pool has {len(pool) - 1} non-gold answers, "
                f"but K={K} requires {K - 1}"
            )
        if gi is not None and "krandom" in arms:
            rng = _item_rng(seed, qid, "krandom")
            cand["krandom"] = [gi] + rng.sample(
                [x for x in range(len(pool)) if x != gi],
                K - 1,
            )
        # Distractors selected by nearest-neighbour in a chosen rep space; ALWAYS
        # scored below with pool_embs (MiniLM). khard's rep IS pool_embs (circular,
        # by design); kdisjoint/klex use disjoint spaces to remove the circularity.
        for m in ("khard", "kdisjoint", "klex"):
            if m not in arms:
                continue
            if gi is None:
                raise AssertionError("gold membership was validated above")
            if reps is None or m not in reps:
                raise ValueError(f"{qid}: requested arm {m!r} has no selector representation")
            rep = reps[m]
            gsims = cosine_similarity(rep[gi:gi + 1], rep)[0]
            cand[m] = [gi] + [int(x) for x in np.argsort(-gsims) if x != gi][: K - 1]
        mc_sims = cosine_similarity(qpre, qopt) if len(qopt) else None
        qa_sims = cosine_similarity(qpre, qans)[:, 0]
        cand_sims = {m: cosine_similarity(qpre, pool_embs[idx]) for m, idx in cand.items()}
        for t, pre in enumerate(prefixes):
            base = dict(subject=subj, item_id=str(q["qid"]), prefix_idx=t,
                        prefix_fraction=len(pre) / full, split=split,
                        p_second_best=0.0, top2_margin=0.0,
                        gold=q["options"][gold_index] if q.get("options") else gold,
                        category=cat, K=K, option_set_id=str(q["qid"]),
                        distractor_strategy=q.get("distractor_strategy", "unknown"))
            if "MC" in arms and mc_sims is not None:
                r = mc_sims[t]; j = int(r.argmax())
                out["MC"].append({**base, "format": "MC", "p_raw": float(r.max()), "p_calibrated": 0.0,
                                  "correct": int(j == gold_index), "top_answer": q["options"][j]})
            if "idealized" in arms:
                out["idealized"].append({**base, "format": "QA", "p_raw": float(qa_sims[t]), "p_calibrated": 0.0,
                                         "correct": 1, "top_answer": gold, "distractor_strategy": "qa_idealized"})
            for m in ("khard", "krandom", "kdisjoint", "klex"):
                if m in arms and m in cand_sims:
                    r = cand_sims[m][t]; j = int(r.argmax())
                    out[m].append({**base, "format": "QA", "p_raw": float(r.max()), "p_calibrated": 0.0,
                                   "correct": int(j == 0), "top_answer": pool[cand[m][j]],
                                   "distractor_strategy": f"qa_{m}"})
    return {a: pd.DataFrame(rows, columns=ADAPTER_COLUMNS) for a, rows in out.items()}


def fit_performat(fit_df):
    cals = {}
    fit_df = fit_df.copy()
    fit_df["bucket"] = fit_df["prefix_fraction"].map(_assign_bucket)
    for fmt in ("MC", "QA"):
        for b in BUCKETS:
            m = fit_df[(fit_df["format"] == fmt) & (fit_df["bucket"] == b)]
            y = m["correct"].to_numpy(); x = m["p_raw"].to_numpy(dtype=float)
            if len(y) >= 3 and len(set(y.tolist())) >= 2:
                cals[(fmt, b)] = ("platt", fit_platt(x, y))
            else:
                cals[(fmt, b)] = ("const", float(np.clip(y.mean() if len(y) else 0.5, 1e-4, 1 - 1e-4)))
    return cals


def apply_cal(df, cals=None, shared=None):
    df = df.copy()
    buckets = df["prefix_fraction"].map(_assign_bucket)
    out = np.zeros(len(df))
    for i, (fmt, b, z) in enumerate(zip(df["format"], buckets, df["p_raw"].astype(float))):
        if shared is not None:
            coef, intercept = shared.get(b, (1.0, 0.0)); out[i] = _platt(float(z), coef, intercept)
        else:
            kind, val = cals[(fmt, b)]
            out[i] = float(val.predict_proba(np.array([[z]]))[:, 1][0]) if kind == "platt" else val
    df["p_calibrated"] = np.clip(out, 0.0, 1.0)
    return df


def signed_per_item(test_df, estimator, schedule, myopic, collect_traces=None):
    if test_df.empty:
        raise ValueError("fair-QA evaluation contains no rows")
    signed = {}; never = {"mc": 0, "qa": 0}
    for _id, g in test_df.groupby("item_id"):
        traces = {}; horizon = {}
        for fmt in ("MC", "QA"):
            rows = g[g["format"] == fmt].sort_values("prefix_idx").reset_index(drop=True)
            if len(rows) < 2:
                raise ValueError(
                    f"{_id} {fmt} requires at least two prefixes; found {len(rows)}"
                )
            ps = rows["p_calibrated"].astype(float).clip(0, 1).tolist()
            fr = rows["prefix_fraction"].astype(float).tolist()
            subj = str(rows["subject"].iloc[0]); horizon[fmt] = len(ps)
            # Coverage-tag capture (additive, numerically inert): record the
            # estimator's per-step fallback tag during each backward
            # continuation lookup, then replay it via coverage_tagger after
            # the backward loop finishes. Mirrors scripts/compute_stopdff_dp.py.
            # The terminal prefix has no continuation lookup -> "exact" by
            # DP convention. The tagger only populates DPTrace.coverage_tags
            # and never affects values/stop_step, so StopDFF is unchanged.
            tags_per_step = {len(ps) - 1: "exact"}
            if myopic:
                cont = lambda *a, **k: 0.0  # noqa: E731
                tagger = None
            else:
                def cont(t, p, prefix_fraction, _subj=subj, _fmt=fmt, _tags=tags_per_step):
                    v = estimator.estimate(
                        prefix_bucket=_assign_prefix_bucket(prefix_fraction), fmt=_fmt,
                        subject_bucket=_subj, p_bin=_assign_p_bin(p), entropy_bin=_assign_entropy_bin(p))
                    _tags[t] = getattr(estimator, "_last_tag", "exact")
                    return v

                def tagger(t, _tags=tags_per_step):
                    return _tags.get(t, "exact")
            traces[fmt] = solve_trajectory(p_trajectory=ps, prefix_fractions=fr, schedule=schedule,
                                           continuation_fn=cont, item_id=str(_id), fmt=fmt,
                                           coverage_tagger=tagger)
        if len(traces) == 2:
            signed[str(_id)] = stopdff_for_item(mc_trace=traces["MC"], qa_trace=traces["QA"])
            never["mc"] += int(traces["MC"].stop_step >= horizon["MC"])
            never["qa"] += int(traces["QA"].stop_step >= horizon["QA"])
            if collect_traces is not None:
                collect_traces.extend((traces["MC"], traces["QA"]))
    if not signed:
        raise ValueError("fair-QA evaluation produced no paired MC/QA items")
    return signed, never


def bootstrap_ci(values, num_boot, seed):
    if num_boot <= 0:
        raise ValueError("num_bootstrap must be positive")
    if not values:
        return {"signed_mean_ci": [None, None], "signed_median_ci": [None, None]}
    arr = np.asarray(values, dtype=float); rng = np.random.default_rng(seed); n = len(arr)
    means = np.empty(num_boot); meds = np.empty(num_boot)
    for b in range(num_boot):
        s = arr[rng.integers(0, n, n)]; means[b] = s.mean(); meds[b] = np.median(s)
    return {
        "signed_mean_ci": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
        "signed_median_ci": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
    }


def summarize(signed_map, never, num_boot, seed):
    vals = list(signed_map.values()); n = len(vals)
    out = {
        "n": n,
        "signed_mean": round(mean(vals), 4) if n else None,
        "signed_median": float(median(vals)) if n else None,
        "abs_mean": round(mean(abs(v) for v in vals), 4) if n else None,
        "abs_median": float(median([abs(v) for v in vals])) if n else None,
        "mc_earlier": sum(1 for v in vals if v < 0),
        "qa_earlier": sum(1 for v in vals if v > 0),
        "same_step": sum(1 for v in vals if v == 0),
        "mc_never_buzz": round(never["mc"] / n, 4) if n else None,
        "qa_never_buzz": round(never["qa"] / n, 4) if n else None,
    }
    out.update(bootstrap_ci(vals, num_boot, seed))
    return out


def qa_accuracy(df):
    d = df.copy(); d["bucket"] = d["prefix_fraction"].map(_assign_bucket)
    return {
        "overall": round(float(d["correct"].mean()), 4) if len(d) else None,
        "by_bucket": d.groupby("bucket")["correct"].mean().reindex(BUCKETS).round(4).to_dict(),
    }


def _committed_script_sha256(commit: str, script_path: str) -> str | None:
    """Hash the producer blob at ``commit`` or use the host-injected hash.

    Modal images exclude ``.git``. The local runner therefore injects the
    already-verified host producer hash for that runtime.
    """
    injected = os.environ.get("MODAL_HOST_PRODUCER_SCRIPT_SHA256")
    if injected:
        return injected
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{script_path}"],
            cwd=str(_REPO),
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _require_exact_producer_binding(generation: dict) -> None:
    """Fail closed unless the output is bound to its exact committed writer."""
    script_sha = generation.get("script_sha256")
    if (
        not isinstance(script_sha, str)
        or len(script_sha) != 64
        or any(char not in "0123456789abcdef" for char in script_sha.lower())
    ):
        raise RuntimeError("invalid producer script SHA-256")
    commit = generation.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit.lower())
    ):
        raise RuntimeError("missing producer git commit")
    if generation.get("git_dirty"):
        raise RuntimeError(
            "uncommitted producer or dependency changes; commit the exact writer before running"
        )
    script_path = generation.get("script_path")
    if script_path != PRODUCER_SCRIPT_PATH:
        raise RuntimeError(
            f"canonical producer script path must be {PRODUCER_SCRIPT_PATH!r}; "
            f"got {script_path!r}"
        )
    committed_sha = _committed_script_sha256(commit, script_path)
    generation["commit_script_sha256"] = committed_sha
    generation["commit_contains_exact_script"] = committed_sha == script_sha
    if committed_sha is None:
        raise RuntimeError(
            f"producer commit {commit} does not contain the producer at {script_path}"
        )
    if committed_sha != script_sha:
        raise RuntimeError(
            f"producer commit {commit} does not match the producer at {script_path}"
        )


def _build_output_provenance(
    *,
    out: Path,
    effective_argv: list[str],
    calibration_path: Path,
    data_inputs: list[Path],
) -> dict:
    """Build and validate fail-closed provenance for the fair-QA output."""
    calibration_helper = _REPO / "scripts" / "compute_prefix_calibration.py"
    extras = [
        calibration_path,
        *data_inputs,
        *coverage_helper_paths(),
        calibration_helper,
    ]
    generation = build_generation_provenance(
        __file__,
        effective_argv,
        output_path=out,
        extra_paths=extras,
    )
    generation["helper_sha256s"] = coverage_helper_sha256s()
    generation["fair_qa_helper_sha256s"] = {
        _display_path(calibration_helper): sha256_file(calibration_helper),
    }
    generation["input_sha256s"] = _input_sha256s(
        [calibration_path, *data_inputs]
    )
    _require_exact_producer_binding(generation)
    return generation


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return str(path)


def _input_sha256s(paths: list[Path]) -> dict[str, str | None]:
    """Hash every declared input, recording optional missing inputs explicitly."""
    return {
        _display_path(path): sha256_file(path) if path.is_file() else None
        for path in paths
    }


def _require_disjoint_split_qids(
    *,
    fit_ids: list[str],
    eval_ids: list[str],
    fit_split: str,
    eval_split: str,
) -> None:
    overlap = sorted(set(fit_ids) & set(eval_ids))
    if overlap:
        raise ValueError(
            "fair-QA fit/evaluation QID overlap is forbidden: "
            f"{fit_split}/{eval_split} share {len(overlap)} QID(s), "
            f"including {overlap[:10]}"
        )


def main():
    effective_argv = list(sys.argv[1:])
    ap = argparse.ArgumentParser(description="Fair-QA StopDFF retest with per-format calibration + bootstrap CIs.")
    ap.add_argument("--data-dir", default=str(_REPO / "data" / "processed"))
    ap.add_argument("--calibration", default=str(_REPO / "paper_exports" / "calibration_train.json"))
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--reward-schedule", default="power_mark")
    ap.add_argument(
        "--qa-arms",
        default="idealized,krandom,khard,kdisjoint,klex",
    )
    ap.add_argument("--disjoint-model", default="all-mpnet-base-v2",
                    help="embedding model for non-circular kdisjoint distractor selection")
    ap.add_argument("--calibrations", default="shared,performat")
    ap.add_argument("--num-bootstrap", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=0, help="0 = full eval split")
    ap.add_argument("--n-val", type=int, default=0, help="0 = full fit split")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(_REPO / "paper_exports" / "stopdff_fair_qa.json"))
    args = ap.parse_args()
    if args.num_bootstrap <= 0:
        ap.error("--num-bootstrap must be positive")

    data = Path(args.data_dir)
    arms_req = [a.strip() for a in args.qa_arms.split(",") if a.strip()]
    cals_req = [c.strip() for c in args.calibrations.split(",") if c.strip()]
    valid_arms = {"idealized", "krandom", "khard", "kdisjoint", "klex"}
    unknown_arms = sorted(set(arms_req) - valid_arms)
    if not arms_req or unknown_arms:
        ap.error(
            "--qa-arms must contain one or more of "
            f"{sorted(valid_arms)}; unknown={unknown_arms}"
        )
    unknown_cals = sorted(set(cals_req) - {"shared", "performat"})
    if not cals_req or unknown_cals:
        ap.error(
            "--calibrations must contain shared and/or performat; "
            f"unknown={unknown_cals}"
        )
    calib_path = Path(args.calibration)
    want = {"MC", *arms_req}

    rng = random.Random(args.seed)
    test_ids = _qids(data / f"{args.eval_split}_dataset.json"); rng.shuffle(test_ids)
    val_ids = _qids(data / f"{args.fit_split}_dataset.json"); rng.shuffle(val_ids)
    if args.n_test > 0:
        test_ids = test_ids[: args.n_test]
    if args.n_val > 0:
        val_ids = val_ids[: args.n_val]
    _require_disjoint_split_qids(
        fit_ids=val_ids,
        eval_ids=test_ids,
        fit_split=args.fit_split,
        eval_split=args.eval_split,
    )
    S_test, S_val = set(test_ids), set(val_ids)
    print(f"[setup] {len(S_test)} {args.eval_split} / {len(S_val)} {args.fit_split} qids (seed {args.seed})", flush=True)

    print("[load] mc_dataset.json ...", flush=True)
    by_id = {str(q["qid"]): q for q in load_json(data / "mc_dataset.json") if str(q["qid"]) in (S_test | S_val)}
    missing_test = [qid for qid in test_ids if qid not in by_id]
    missing_val = [qid for qid in val_ids if qid not in by_id]
    if missing_test or missing_val:
        raise ValueError(
            "MC dataset is missing requested split questions: "
            f"{args.eval_split}={missing_test[:10]}, {args.fit_split}={missing_val[:10]}"
        )
    mc_test = [by_id[qid] for qid in test_ids]
    mc_val = [by_id[qid] for qid in val_ids]
    if not mc_test or not mc_val:
        raise ValueError(
            "fair-QA requires non-empty fit and evaluation question sets; "
            f"got n_eval={len(mc_test)}, n_fit={len(mc_val)}"
        )
    print(f"[load] matched {len(mc_test)} eval, {len(mc_val)} fit questions", flush=True)

    model = _get_sbert_model()
    pool = list(load_json(data / "answer_profiles.json").keys())
    print(f"[score] encode pool ({len(pool)}) + batched arms ...", flush=True)
    pool_embs = _encode(model, pool)
    gold_to_idx = {a: i for i, a in enumerate(pool)}

    reps = {}
    if "khard" in want:
        reps["khard"] = pool_embs  # NN in the SAME space used to score (circular, by design)
    if "kdisjoint" in want:
        from sentence_transformers import SentenceTransformer
        print(f"[score] disjoint selector {args.disjoint_model} (encode pool) ...", flush=True)
        reps["kdisjoint"] = SentenceTransformer(args.disjoint_model).encode(
            pool, convert_to_numpy=True, batch_size=256)
    if "klex" in want:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("[score] lexical selector char_wb tfidf (fit pool) ...", flush=True)
        reps["klex"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(pool)

    test_arms = score_arms(mc_test, args.eval_split, model, pool, pool_embs, gold_to_idx, args.seed, want, reps=reps)
    val_arms = score_arms(mc_val, args.fit_split, model, pool, pool_embs, gold_to_idx, args.seed + 7, want, reps=reps)
    print("[score] done; running DP + bootstrap ...", flush=True)
    mc_test_rows, mc_val_rows = test_arms["MC"], val_arms["MC"]

    schedule = get_schedule(args.reward_schedule)
    shared = _load_platt_params(calib_path)
    results, qa_acc = {}, {}
    coverage_by_cell, coverage_by_reference = {}, {}
    all_dp_traces = []
    for name in arms_req:
        qa_test, qa_val = test_arms[name], val_arms[name]
        qa_acc[name] = qa_accuracy(qa_test)
        ref_traces = []
        for calib in cals_req:
            val_df = pd.concat([mc_val_rows, qa_val], ignore_index=True)
            test_df = pd.concat([mc_test_rows, qa_test], ignore_index=True)
            if calib == "shared":
                val_df = apply_cal(val_df, shared=shared); test_df = apply_cal(test_df, shared=shared)
            else:
                cals = fit_performat(val_df)
                val_df = apply_cal(val_df, cals=cals); test_df = apply_cal(test_df, cals=cals)
            val_df["split"] = args.fit_split; test_df["split"] = args.eval_split
            est = EmpiricalBucketEstimator.fit(fit_df=val_df, schedule=schedule, fit_split_name=args.fit_split)
            cell_traces = []
            dp_signed, dp_never = signed_per_item(test_df, est, schedule, myopic=False, collect_traces=cell_traces)
            myo_signed, myo_never = signed_per_item(test_df, est, schedule, myopic=True)
            results[f"{name}+{calib}"] = {
                "dp": summarize(dp_signed, dp_never, args.num_bootstrap, args.seed),
                "myopic": summarize(myo_signed, myo_never, args.num_bootstrap, args.seed + 1),
            }
            coverage_by_cell[f"{name}+{calib}"] = summarize_coverage(cell_traces)
            ref_traces.extend(cell_traces); all_dp_traces.extend(cell_traces)
            print(f"[result] {name}+{calib} DP: {results[f'{name}+{calib}']['dp']}", flush=True)
            print(f"[coverage] {name}+{calib}: {coverage_by_cell[f'{name}+{calib}']}", flush=True)
        coverage_by_reference[name] = summarize_coverage(ref_traces)

    continuation_coverage = {
        "estimator": "empirical_bucket",
        "scope": "dp_non_myopic",
        "note": (
            "Per-prefix continuation-lookup coverage for the DP (empirical-bucket, "
            "non-myopic) StopDFF arm, aggregated over MC+QA traces via "
            "scripts/stopdff_dp/diagnostics.summarize_coverage. Tags label the "
            "fallback-ladder rung used at each decision step: 'exact' = "
            "full-specificity bucket hit (also the terminal prefix, which has no "
            "continuation lookup by DP convention); 'pooled' = a coarser fallback "
            "rung; 'missing' = no bucket met min size. The myopic arm uses "
            "continuation==0 and is excluded. Emitted additively; does NOT alter "
            "any StopDFF value."
        ),
        "overall": summarize_coverage(all_dp_traces),
        "by_reference": coverage_by_reference,
        "by_cell": coverage_by_cell,
    }

    out = Path(args.out)
    data_inputs = [
        data / "mc_dataset.json",
        data / "answer_profiles.json",
        data / f"{args.fit_split}_dataset.json",
        data / f"{args.eval_split}_dataset.json",
        data / "build_metadata.json",
    ]
    generation = _build_output_provenance(
        out=out,
        effective_argv=effective_argv,
        calibration_path=calib_path,
        data_inputs=data_inputs,
    )
    payload = {
        "metadata": {
            "metric_type": "stopdff_fair_qa_retest",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reward_schedule": args.reward_schedule,
            "fit_split": args.fit_split, "eval_split": args.eval_split,
            "n_eval": len(mc_test), "n_fit": len(mc_val),
            "num_bootstrap": args.num_bootstrap, "seed": args.seed,
            "qa_arms": arms_req, "calibrations": cals_req,
            "calibration_path": _display_path(calib_path),
            "git_commit": generation["git_commit"],
            "generation": generation,
            "khard_circularity_caveat": (
                "khard distractors are SBERT nearest-neighbours in the same space used to score, "
                "so they are adversarially close to gold; khard is a difficulty-matched proxy, not "
                "a clean open-ended measurement."
            ),
        },
        "mc_accuracy": qa_accuracy(mc_test_rows.assign(format="MC")),
        "qa_accuracy": qa_acc,
        "results": results,
        "continuation_coverage": continuation_coverage,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[done] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
