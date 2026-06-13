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
import json
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

from scripts._common import load_json, iter_split_questions  # noqa: E402
from scripts.stopdff_dp.adapter import (  # noqa: E402
    _get_sbert_model, _assign_bucket, _load_platt_params, _platt,
)
from scripts.stopdff_dp.rewards import get_schedule  # noqa: E402
from scripts.stopdff_dp.dp_solver import solve_trajectory, stopdff_for_item  # noqa: E402
from scripts.stopdff_dp.continuation import (  # noqa: E402
    EmpiricalBucketEstimator, _assign_prefix_bucket, _assign_p_bin, _assign_entropy_bin,
)
from scripts.stopdff_dp.types import ADAPTER_COLUMNS  # noqa: E402
from scripts.compute_prefix_calibration import fit_platt  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

BUCKETS = ["early", "mid", "late"]


def _qids(path: Path) -> list[str]:
    return [str(q["qid"]) for q in iter_split_questions(load_json(path))]


def _encode(model, strings, batch_size=256):
    if not strings:
        return np.zeros((0, 384), dtype=float)
    return model.encode(list(strings), convert_to_numpy=True, batch_size=batch_size)


def score_arms(questions, split, model, pool, pool_embs, gold_to_idx, seed, arms, reps=None):
    """Batched scoring. Returns {arm: DataFrame[ADAPTER_COLUMNS]} for arms in
    {"MC","idealized","krandom","khard"}."""
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

    rng = random.Random(seed)
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
        if gi is not None:
            if "krandom" in arms:
                cand["krandom"] = [gi] + rng.sample([x for x in range(len(pool)) if x != gi], K - 1)
            # Distractors selected by nearest-neighbour in a chosen rep space; ALWAYS
            # scored below with pool_embs (MiniLM). khard's rep IS pool_embs (circular,
            # by design); kdisjoint/klex use disjoint spaces to remove the circularity.
            for m in ("khard", "kdisjoint", "klex"):
                if m in arms and reps is not None and m in reps:
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


def signed_per_item(test_df, estimator, schedule, myopic):
    signed = {}; never = {"mc": 0, "qa": 0}
    for _id, g in test_df.groupby("item_id"):
        traces = {}; horizon = {}
        for fmt in ("MC", "QA"):
            rows = g[g["format"] == fmt].sort_values("prefix_idx").reset_index(drop=True)
            if len(rows) < 2:
                break
            ps = rows["p_calibrated"].astype(float).clip(0, 1).tolist()
            fr = rows["prefix_fraction"].astype(float).tolist()
            subj = str(rows["subject"].iloc[0]); horizon[fmt] = len(ps)
            if myopic:
                cont = lambda *a, **k: 0.0  # noqa: E731
            else:
                def cont(t, p, prefix_fraction, _subj=subj, _fmt=fmt):
                    return estimator.estimate(
                        prefix_bucket=_assign_prefix_bucket(prefix_fraction), fmt=_fmt,
                        subject_bucket=_subj, p_bin=_assign_p_bin(p), entropy_bin=_assign_entropy_bin(p))
            traces[fmt] = solve_trajectory(p_trajectory=ps, prefix_fractions=fr, schedule=schedule,
                                           continuation_fn=cont, item_id=str(_id), fmt=fmt)
        if len(traces) == 2:
            signed[str(_id)] = stopdff_for_item(mc_trace=traces["MC"], qa_trace=traces["QA"])
            never["mc"] += int(traces["MC"].stop_step >= horizon["MC"])
            never["qa"] += int(traces["QA"].stop_step >= horizon["QA"])
    return signed, never


def bootstrap_ci(values, num_boot, seed):
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


def _git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_REPO), capture_output=True,
                           text=True, timeout=10, check=False)
        return r.stdout.strip() or None
    except Exception:
        return None


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return str(path)


def main():
    ap = argparse.ArgumentParser(description="Fair-QA StopDFF retest with per-format calibration + bootstrap CIs.")
    ap.add_argument("--data-dir", default=str(_REPO / "data" / "processed"))
    ap.add_argument("--calibration", default=str(_REPO / "paper_exports" / "calibration_train.json"))
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--reward-schedule", default="power_mark")
    ap.add_argument("--qa-arms", default="idealized,krandom,khard")
    ap.add_argument("--disjoint-model", default="all-mpnet-base-v2",
                    help="embedding model for non-circular kdisjoint distractor selection")
    ap.add_argument("--calibrations", default="shared,performat")
    ap.add_argument("--num-bootstrap", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=0, help="0 = full eval split")
    ap.add_argument("--n-val", type=int, default=0, help="0 = full fit split")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(_REPO / "paper_exports" / "stopdff_fair_qa.json"))
    args = ap.parse_args()

    data = Path(args.data_dir)
    arms_req = [a.strip() for a in args.qa_arms.split(",") if a.strip()]
    cals_req = [c.strip() for c in args.calibrations.split(",") if c.strip()]
    calib_path = Path(args.calibration)
    want = {"MC", *arms_req}

    rng = random.Random(args.seed)
    test_ids = _qids(data / f"{args.eval_split}_dataset.json"); rng.shuffle(test_ids)
    val_ids = _qids(data / f"{args.fit_split}_dataset.json"); rng.shuffle(val_ids)
    if args.n_test > 0:
        test_ids = test_ids[: args.n_test]
    if args.n_val > 0:
        val_ids = val_ids[: args.n_val]
    S_test, S_val = set(test_ids), set(val_ids)
    print(f"[setup] {len(S_test)} {args.eval_split} / {len(S_val)} {args.fit_split} qids (seed {args.seed})", flush=True)

    print("[load] mc_dataset.json ...", flush=True)
    by_id = {str(q["qid"]): q for q in load_json(data / "mc_dataset.json") if str(q["qid"]) in (S_test | S_val)}
    mc_test = [by_id[i] for i in S_test if i in by_id]
    mc_val = [by_id[i] for i in S_val if i in by_id]
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
    for name in arms_req:
        qa_test, qa_val = test_arms[name], val_arms[name]
        qa_acc[name] = qa_accuracy(qa_test)
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
            dp_signed, dp_never = signed_per_item(test_df, est, schedule, myopic=False)
            myo_signed, myo_never = signed_per_item(test_df, est, schedule, myopic=True)
            results[f"{name}+{calib}"] = {
                "dp": summarize(dp_signed, dp_never, args.num_bootstrap, args.seed),
                "myopic": summarize(myo_signed, myo_never, args.num_bootstrap, args.seed + 1),
            }
            print(f"[result] {name}+{calib} DP: {results[f'{name}+{calib}']['dp']}", flush=True)

    payload = {
        "metadata": {
            "metric_type": "stopdff_fair_qa_retest",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reward_schedule": args.reward_schedule,
            "fit_split": args.fit_split, "eval_split": args.eval_split,
            "n_eval": len(mc_test), "n_fit": len(mc_val),
            "num_bootstrap": args.num_bootstrap, "seed": args.seed,
            "qa_arms": arms_req, "calibrations": cals_req,
            "calibration_path": _display_path(calib_path), "git_commit": _git_commit(),
            "khard_circularity_caveat": (
                "khard distractors are SBERT nearest-neighbours in the same space used to score, "
                "so they are adversarially close to gold; khard is a difficulty-matched proxy, not "
                "a clean open-ended measurement."
            ),
        },
        "mc_accuracy": qa_accuracy(mc_test_rows.assign(format="MC")),
        "qa_accuracy": qa_acc,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[done] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
