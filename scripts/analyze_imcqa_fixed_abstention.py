#!/usr/bin/env python3
"""CPU-only 2x2 decomposition of frozen timing and terminal abstention rules.

The original model outputs, logistic maps, thresholds, fixed rounds, candidate
ties, and reward schedule are unchanged. The added fixed-selective policy has
one answer opportunity. Failing its threshold leads to terminal PASS, without
another answer opportunity. This is exploratory reuse of inspected development
data; the fixed-selective baseline was not optimized on these results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import analyze_imcqa_transfer as transfer

old = transfer.old
MENUS, REWARDS = transfer.MENUS, transfer.REWARDS
POLICIES = ("fixed_forced", "fixed_selective", "adaptive_forced", "adaptive_selective")
PLAN_SHA256 = "4a455bb6bc3d46eb7b2a547130b69008d67ed41f3391e97f50dfd3e87ae7d195"
PUBLIC_SHA256 = "4172b83e16f753f51442ff314d60ac2c1d75b12b7faa6574d5918319936ad2b9"
METRICS = ("mean_reward", "coverage", "conditional_error", "mean_observed_round")
SIMPLE_EFFECTS = (
    ("timing_forced", "adaptive_forced", "fixed_forced"),
    ("timing_selective", "adaptive_selective", "fixed_selective"),
    ("abstention_fixed", "fixed_selective", "fixed_forced"),
    ("abstention_adaptive", "adaptive_selective", "adaptive_forced"),
)


def first_crossing(trajectory, parameters):
    """Use the same inclusive frozen threshold; apply no additional EV gate."""
    threshold = parameters["selected_threshold"]
    if threshold is None:
        return None
    return next((row["round"] for row in trajectory if transfer.calibrated_probability(
        max(row["canonical_answer_probabilities"].values()), parameters) >= threshold), None)


def policy_rounds(trajectory, parameters):
    """Return each policy's commitment round, or None for terminal PASS.

    Parameters
    ----------
    trajectory : list of dict
        The five ordered plain-MCQA candidate states for one question/order.
    parameters : dict
        The unchanged per-menu fit, threshold, and fixed-round selection.
    """
    if [row["round"] for row in trajectory] != [1, 2, 3, 4, 5]:
        raise ValueError("five ordered rounds required")
    if any(row["arm"] != "plain" for row in trajectory):
        raise ValueError("only original plain candidate states permitted")
    fixed = int(parameters["selected_fixed_policy"].split("_")[1])
    if fixed not in range(1, 6):
        raise ValueError("invalid frozen fixed round")
    crossed = first_crossing(trajectory, parameters)
    threshold = parameters["selected_threshold"]
    fixed_p = transfer.calibrated_probability(max(trajectory[fixed-1]["canonical_answer_probabilities"].values()), parameters)
    return {"fixed_forced": fixed,
        "fixed_selective": fixed if threshold is not None and fixed_p >= threshold else None,
        "adaptive_forced": crossed if crossed is not None else 5,
        "adaptive_selective": crossed}


def build_episodes(views, fits):
    """Replay the four rules on the same frozen candidates; retain trajectories."""
    groups = defaultdict(list)
    qids = set()
    for row in views:
        if row["arm"] != "plain":
            continue
        groups[row["qid"], row["condition"], row["rotation"]].append(row)
        qids.add(row["qid"])
    expected = {(qid, menu, rotation) for qid in qids for menu in MENUS for rotation in range(4)}
    if not qids or set(groups) != expected:
        raise ValueError("missing or extra question/menu/rotation trajectory")
    episodes, states = [], []
    for (qid, menu, rotation), trajectory in sorted(groups.items()):
        trajectory.sort(key=lambda row: row["round"])
        if len({row["group_id"] for row in trajectory}) != 1:
            raise ValueError("question group changes across rounds")
        parameters = fits[menu]
        rounds = policy_rounds(trajectory, parameters)
        fixed = int(parameters["selected_fixed_policy"].split("_")[1])
        crossing = first_crossing(trajectory, parameters)
        for row in trajectory:
            confidence = max(row["canonical_answer_probabilities"].values())
            calibrated = transfer.calibrated_probability(confidence, parameters)
            states.append({key: row[key] for key in ("qid", "group_id", "condition", "rotation", "round", "score_id",
                "candidate_choice", "candidate_correct", "canonical_gold_option_id")} | {
                "candidate_confidence": confidence, "calibrated_probability": calibrated,
                "threshold": parameters["selected_threshold"],
                "threshold_met": parameters["selected_threshold"] is not None and calibrated >= parameters["selected_threshold"],
                "fixed_round": fixed, "first_crossing_round": crossing,
                "canonical_answer_probabilities": row["canonical_answer_probabilities"]})
        for policy, round_number in rounds.items():
            chosen = trajectory[-1] if round_number is None else trajectory[round_number-1]
            committed = round_number is not None
            correct = committed and bool(chosen["candidate_correct"])
            episodes.append({"qid": qid, "group_id": chosen["group_id"], "condition": menu,
                "rotation": rotation, "policy": policy, "committed": committed, "correct": correct,
                "wrong": committed and not correct, "terminal_pass": not committed,
                "round": round_number, "observed_round": 5 if round_number is None else round_number,
                "canonical_choice": chosen["candidate_choice"] if committed else None,
                "reward": (REWARDS[round_number-1] if correct else -1.) if committed else 0.,
                "fixed_round": fixed, "first_crossing_round": crossing,
                "answer_score_id": chosen["score_id"] if committed else None,
                "observed_score_id": chosen["score_id"],
                "fixed_opportunity_rejected": policy == "fixed_selective" and not committed,
                "forced_final_fallback": policy == "adaptive_forced" and crossing is None})
    return episodes, states


def metric_samples(values, indices):
    """Compute pooled conditional error in each paired question resample.

    Parameters
    ----------
    values : numpy.ndarray
        Question means in columns reward, committed, wrong, observed_round.
    indices : numpy.ndarray
        One shared question-resampling index matrix for all policies/menus.
    """
    means = values.mean(axis=0)
    draws = values[indices].mean(axis=1)
    risk = float(means[2]/means[1]) if means[1] else None
    risk_draws = np.divide(draws[:,2], draws[:,1], out=np.full(len(draws),np.nan), where=draws[:,1]>0)
    return ({"mean_reward":float(means[0]), "coverage":float(means[1]),
             "conditional_error":risk, "mean_observed_round":float(means[3])},
            {"mean_reward":draws[:,0], "coverage":draws[:,1],
             "conditional_error":risk_draws, "mean_observed_round":draws[:,3]})


def summarize(episodes, *, samples=20000, seed=1):
    """Average rotations within each question, then resample paired questions."""
    qids = sorted({row["qid"] for row in episodes})
    if not qids:
        raise ValueError("cannot summarize empty episodes")
    grouped = defaultdict(list)
    for row in episodes:
        grouped[row["qid"],row["condition"],row["policy"]].append(row)
    expected = {(qid, menu, policy) for qid in qids for menu in MENUS for policy in POLICIES}
    if set(grouped) != expected:
        raise ValueError("incomplete policy question pairing")
    per_question = []
    for (qid, menu, policy), rows in sorted(grouped.items()):
        if sorted(row["rotation"] for row in rows) != [0,1,2,3]:
            raise ValueError("four distinct rotations required within question")
        per_question.append({"qid":qid, "condition":menu, "policy":policy,
            **{field:float(np.mean([row[field] for row in rows])) for field in
               ("reward","committed","correct","wrong","terminal_pass","observed_round")}})
    indices = np.random.default_rng(seed).integers(0,len(qids),(samples,len(qids)))
    summaries, primary, simple, metric_deltas = [], [], [], []
    point, bootstrap = {}, {}
    for menu in MENUS:
        for policy in POLICIES:
            rows = [row for row in per_question if row["condition"]==menu and row["policy"]==policy]
            if [row["qid"] for row in rows] != qids:
                raise ValueError("question order differs across paired policies")
            values = np.asarray([[row[k] for k in ("reward","committed","wrong","observed_round")] for row in rows])
            point[menu,policy], bootstrap[menu,policy] = metric_samples(values,indices)
            original = [row for row in episodes if row["condition"]==menu and row["policy"]==policy]
            summaries.append({"condition":menu,"policy":policy,"n_questions":len(qids),"n_episodes":len(original),
                **{label:sum(row[field] for row in original) for label,field in
                   (("committed_count","committed"),("correct_count","correct"),("wrong_count","wrong"),("terminal_pass_count","terminal_pass"))},
                **{metric:transfer.estimate(point[menu,policy][metric],bootstrap[menu,policy][metric]) for metric in METRICS}})
        for name,left,right in SIMPLE_EFFECTS:
            left_point,right_point = point[menu,left],point[menu,right]
            for metric in ("mean_reward","coverage","conditional_error"):
                delta = left_point[metric]-right_point[metric] if left_point[metric] is not None and right_point[metric] is not None else None
                draws = bootstrap[menu,left][metric]-bootstrap[menu,right][metric]
                result = {"condition":menu,"contrast":name,"left":left,"right":right,"metric":metric,
                    "n_questions":len(qids),"mean_delta":delta,
                    **{k:v for k,v in transfer.estimate(delta,draws).items() if k!="mean"}}
                if metric=="mean_reward":
                    simple.append(result)
                    if name=="timing_selective":
                        primary.append({**result,"ci97_5":np.quantile(draws,[.0125,.9875]).tolist(),"family_size":2,
                            "interval_scope":"Exploratory two-menu Bonferroni family; additions were fixed before this analysis, after earlier results were inspected."})
                else:
                    metric_deltas.append(result)
        weights={"adaptive_selective":1,"fixed_selective":-1,"adaptive_forced":-1,"fixed_forced":1}
        interaction=sum(weight*point[menu,policy]["mean_reward"] for policy,weight in weights.items())
        draws=sum(weight*bootstrap[menu,policy]["mean_reward"] for policy,weight in weights.items())
        simple.append({"condition":menu,"contrast":"timing_by_abstention_interaction","metric":"mean_reward",
            "formula":"(adaptive_selective - fixed_selective) - (adaptive_forced - fixed_forced)",
            "n_questions":len(qids),"mean_delta":interaction,
            **{k:v for k,v in transfer.estimate(interaction,draws).items() if k!="mean"}})
    return {"schema_version":"imcqa-fixed-abstention-analysis-v1","n_questions":len(qids),
        "n_plain_score_states":len(qids)*2*4*5,"n_policy_episodes":len(episodes),
        "model":"qwen7b","bootstrap_samples":samples,"bootstrap_seed":seed,
        "unit":"Question; average four cyclic rotations before paired resampling; both menus and all policies share the resampling indices.",
        "policy_summaries":summaries,"primary_contrasts":primary,"simple_effects":simple,"metric_deltas":metric_deltas},per_question


def verify_manifest(root):
    """Require the retained artifact bytes to match the original manifest."""
    manifest=old.load_json(root/"artifact_manifest.json")
    paths=set()
    for row in manifest["files"]:
        relative=Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in paths:
            raise ValueError("unsafe or duplicate original manifest path")
        paths.add(row["path"])
        path=root/relative
        if path.stat().st_size!=row["bytes"] or old.sha256(path)!=row["sha256"]:
            raise ValueError("original artifact file differs: "+row["path"])
    required={"inputs/main_jobs.json","inputs/main_dataset.json","inputs/fitted_parameters.json",
              "inputs/prior_wait_public.json","analysis/episodes.csv","analysis/analysis_receipt.json",
              "run/output/qwen7b/scores.jsonl","run/output/qwen7b/receipt.json"}
    if not required<=paths:
        raise ValueError("original manifest does not bind required inputs")
    return {"passed":True,"n_files":len(paths),"manifest_sha256":old.sha256(root/"artifact_manifest.json")}


def load_validated_inputs(evidence, archive, plan_path):
    """Revalidate source/gold joins and numerical evidence before new analysis."""
    if old.sha256(plan_path)!=PLAN_SHA256:
        raise ValueError("pre-analysis frozen addition plan hash differs")
    plan=old.load_json(plan_path)
    if old.sha256(archive)!=plan["input_archive_sha256"]:
        raise ValueError("original input archive hash differs")
    artifact=verify_manifest(evidence)
    original_receipt=old.load_json(evidence/"analysis/analysis_receipt.json")
    if old.sha256(Path(transfer.__file__))!=original_receipt["analyzer_sha256"]:
        raise ValueError("original validation/analyzer code differs from retained evidence")
    for name,expected in original_receipt["outputs_sha256"].items():
        if old.sha256(evidence/"analysis"/name)!=expected:
            raise ValueError("prior analytical output hash differs: "+name)
    root=Path(__file__).resolve().parents[1]
    config_path=root/"configs/imcqa_frozen_transfer.json"
    config=old.load_json(config_path)
    inputs=evidence/"inputs"
    mapping={"main_jobs":"main_jobs.json","main_dataset":"main_dataset.json",
             "prior_public":"prior_wait_public.json","fitted_parameters":"fitted_parameters.json"}
    for key,name in mapping.items():
        if old.sha256(inputs/name)!=config["frozen_source_sha256"][key]:
            raise ValueError("original frozen input hash differs: "+key)
    public_path=evidence/"run/pilot.json"
    if old.sha256(public_path)!=PUBLIC_SHA256:
        raise ValueError("original public context hash differs")
    package=old.load_json(public_path)
    dataset=old.load_json(inputs/"main_dataset.json")
    source=old.load_json(inputs/"main_jobs.json")
    fitted=old.load_json(inputs/"fitted_parameters.json")
    previous=old.load_json(inputs/"prior_wait_public.json")
    jobs,fits=transfer.validate_public(package,dataset,source,previous,fitted,config)
    views,numerics=transfer.validate_model(evidence/"run/output/qwen7b",package,jobs,config,PUBLIC_SHA256,
                                         inputs/"prior_qwen7b",inputs/"cache_prepare_receipt.json")
    if len(views)!=8000 or len({row["qid"] for row in views})!=100:
        raise ValueError("original validated cohort count differs")
    return views,fits,plan,{"passed":True,"addition_plan_sha256":PLAN_SHA256,
        "input_archive_sha256":plan["input_archive_sha256"],"original_artifact":artifact,
        "original_analyzer_sha256":original_receipt["analyzer_sha256"],"model_source_numerics":numerics,
        "fitting_performed":False,"model_inference_performed":False,"new_gpu_cost_usd":0.0}


def match_prior_episodes(episodes, path):
    """Require exact reproduction of the old fixed and adaptive-selective rows."""
    with path.open(newline="") as stream:
        previous=list(csv.DictReader(stream))
    policy_mapping={"fixed_forced":"frozen_plain_fixed","adaptive_selective":"frozen_plain_threshold"}
    old_rows={}
    for row in previous:
        if row["policy"] not in policy_mapping.values():
            continue
        key=(row["qid"],row["condition"],int(row["rotation"]),row["policy"])
        if key in old_rows:
            raise ValueError("duplicate previous policy episode")
        old_rows[key]=row
    seen=set()
    for row in episodes:
        if row["policy"] not in policy_mapping:
            continue
        key=(row["qid"],row["condition"],row["rotation"],policy_mapping[row["policy"]])
        if key not in old_rows or key in seen:
            raise ValueError("new/old policy episode identity differs")
        seen.add(key)
        for field,value in old_rows[key].items():
            if field=="policy":
                continue
            if field not in row or ("" if row[field] is None else str(row[field]))!=value:
                raise ValueError("frozen old policy episode differs: "+field)
    if seen!=set(old_rows) or len(seen)!=1600:
        raise ValueError("old FF/AS policy coverage differs")
    return {"passed":True,"exact_episode_matches":len(seen),"policies":policy_mapping,
            "prior_episodes_sha256":old.sha256(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("evidence","archive","plan","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError("create-once output path already exists")
    views,fits,plan,validation=load_validated_inputs(args.evidence,args.archive,args.plan)
    episodes,states=build_episodes(views,fits)
    validation["old_policy_reproduction"]=match_prior_episodes(episodes,args.evidence/"analysis/episodes.csv")
    summary,per_question=summarize(episodes,samples=plan["inference"]["bootstrap_samples"],seed=plan["inference"]["seed"])
    summary.update(analysis_id=plan["analysis_id"],status=plan["status"],limitations=plan["limitations"],
                   plan_sha256=PLAN_SHA256,policy_definitions=plan["policies"],frozen_parameters=fits,
                   model_inference_performed=False,fitting_performed=False)
    args.out.mkdir(parents=True,exist_ok=False)
    old.write_json(args.out/"summary.json",summary)
    old.write_json(args.out/"validation.json",validation)
    for name,rows in (("episodes",episodes),("per_question",per_question),("trajectories",states)):
        transfer.write_csv(args.out/(name+".csv"),rows)
    for name in ("policy_summaries","primary_contrasts","simple_effects","metric_deltas"):
        transfer.write_csv(args.out/(name+".csv"),summary[name])
    old.write_json(args.out/"analysis_receipt.json",{"status":"complete","analyzer_sha256":old.sha256(Path(__file__)),
        "plan_sha256":PLAN_SHA256,"input_archive_sha256":plan["input_archive_sha256"],
        "fitting_performed":False,"model_inference_performed":False,
        "outputs_sha256":{path.name:old.sha256(path) for path in sorted(args.out.iterdir())}})
    print(json.dumps({"status":"complete","n_questions":summary["n_questions"],"n_policy_episodes":len(episodes),
                      "exact_old_episode_matches":validation["old_policy_reproduction"]["exact_episode_matches"],"out":str(args.out)}))


if __name__=="__main__":
    main()
