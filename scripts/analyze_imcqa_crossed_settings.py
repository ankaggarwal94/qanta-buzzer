#!/usr/bin/env python3
"""Replay crossed frozen thresholds and rounds without fitting or inference.

The same question/order trajectory appears in several setting cells. This is
intentional; it does not increase the independent sample size. Every statistic
uses question averages over four rotations and one shared bootstrap index
matrix across all menus, policies, and settings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from scripts import analyze_imcqa_fixed_abstention as base

old, MENUS, POLICIES = base.old, base.MENUS, base.POLICIES
PLAN_SHA256 = "d678b4bc4a28bb9570d77e1ad79f388d5ad7c7b9080a2b3d1b801a8b5a23ebd1"
SETTINGS = [
    {"setting_id":"t060_r1","threshold":.6,"fixed_round":1,"requested_bundle":True},
    {"setting_id":"t060_r2","threshold":.6,"fixed_round":2,"requested_bundle":False},
    {"setting_id":"t085_r1","threshold":.85,"fixed_round":1,"requested_bundle":False},
    {"setting_id":"t085_r2","threshold":.85,"fixed_round":2,"requested_bundle":True},
]
OUTCOME_FIELDS = ("committed","correct","wrong","terminal_pass","round","observed_round",
                  "canonical_choice","reward","answer_score_id","observed_score_id")


def build_crossed_episodes(views, fits, settings):
    """Copy per-menu fits, replacing only threshold and fixed-round selection.

    Parameters
    ----------
    views : list of dict
        Validated original raw-score views with CPU-only answer-key joins.
    fits : dict
        Each menu's own previously fitted logistic calibration parameters.
    settings : list of dict
        The exact four frozen threshold/round cells.
    """
    if settings!=SETTINGS or set(fits)!=set(MENUS):
        raise ValueError("crossed settings or menu calibrators differ from frozen design")
    original=deepcopy(fits)
    episodes,states,used=[],[],[]
    for setting in settings:
        parameters=deepcopy(fits)
        for menu in MENUS:
            parameters[menu]["selected_threshold"]=setting["threshold"]
            parameters[menu]["selected_fixed_policy"]=f'fixed_{setting["fixed_round"]}'
            for name,value in original[menu].items():
                if name not in {"selected_threshold","selected_fixed_policy"} and parameters[menu][name]!=value:
                    raise ValueError("own menu calibrator changed")
            used.append({**setting,"condition":menu,"parameters":deepcopy(parameters[menu])})
        cell_episodes,cell_states=base.build_episodes(views,parameters)
        episodes.extend({**row,**setting} for row in cell_episodes)
        states.extend({**row,**setting} for row in cell_states)
    if fits!=original:
        raise ValueError("original frozen fit object was mutated")
    return episodes,states,used


def check_invariants(episodes):
    """Check behavior repeats across settings irrelevant to the given policy."""
    lookup={}
    for row in episodes:
        key=(row["qid"],row["condition"],row["rotation"],row["policy"],row["setting_id"])
        if key in lookup:
            raise ValueError("duplicate crossed episode invariant key")
        lookup[key]=row
    identities={(r["qid"],r["condition"],r["rotation"]) for r in episodes}
    expected={(*identity,policy,setting["setting_id"]) for identity in identities for policy in POLICIES for setting in SETTINGS}
    if set(lookup)!=expected:
        raise ValueError("crossed episode invariant coverage missing")
    adaptive=fixed=0
    def same(left,right):
        if any(left[field]!=right[field] for field in OUTCOME_FIELDS):
            raise ValueError("behavioral setting invariant failed")
    for identity in identities:
        for policy in ("adaptive_forced","adaptive_selective"):
            for threshold in ("060","085"):
                same(lookup[*identity,policy,f"t{threshold}_r1"],lookup[*identity,policy,f"t{threshold}_r2"])
                adaptive+=1
        for round_number in (1,2):
            same(lookup[*identity,"fixed_forced",f"t060_r{round_number}"],
                 lookup[*identity,"fixed_forced",f"t085_r{round_number}"])
            fixed+=1
    return {"passed":True,"adaptive_fixed_round_pairs_checked":adaptive,"fixed_forced_threshold_pairs_checked":fixed,
            "behavioral_fields":list(OUTCOME_FIELDS),"diagnostic_setting_metadata_excluded_from_equality":True}


def match_original(episodes,path):
    """Exactly reproduce all earlier policies in their original menu settings."""
    with path.open(newline="") as stream:
        previous=list(csv.DictReader(stream))
    old_lookup={}
    for row in previous:
        key=(row["qid"],row["condition"],int(row["rotation"]),row["policy"])
        if key in old_lookup:
            raise ValueError("duplicate original episode")
        old_lookup[key]=row
    selected={MENUS[0]:"t060_r1",MENUS[1]:"t085_r2"}
    seen=set()
    for row in episodes:
        if row["setting_id"]!=selected[row["condition"]]:
            continue
        key=(row["qid"],row["condition"],row["rotation"],row["policy"])
        if key not in old_lookup or key in seen:
            raise ValueError("original episode identity differs")
        seen.add(key)
        for field,value in old_lookup[key].items():
            if field not in row or ("" if row[field] is None else str(row[field]))!=value:
                raise ValueError("original episode outcome or diagnostic differs: "+field)
    if seen!=set(old_lookup) or not seen:
        raise ValueError("original episode coverage differs")
    return {"passed":True,"exact_episode_matches":len(seen),"original_settings":selected,
            "original_episodes_sha256":old.sha256(path)}


def difference(left,right,left_boot,right_boot):
    point=None if left is None or right is None else left-right
    interval=base.transfer.estimate(point,left_boot-right_boot)
    return {"mean_delta":point,**{name:value for name,value in interval.items() if name!="mean"}}


def summarize(episodes, settings, *, samples=20000, seed=1):
    """Use one question-bootstrap matrix for all thirty-two policy cells."""
    qids=sorted({row["qid"] for row in episodes})
    if not qids or settings!=SETTINGS:
        raise ValueError("empty questions or altered frozen settings")
    groups=defaultdict(list)
    for row in episodes:
        groups[row["qid"],row["condition"],row["setting_id"],row["policy"]].append(row)
    expected={(qid,menu,setting["setting_id"],policy) for qid in qids for menu in MENUS for setting in settings for policy in POLICIES}
    if set(groups)!=expected:
        raise ValueError("incomplete paired question/setting/policy cells")
    per_question=[]
    for (qid,menu,setting_id,policy),rows in sorted(groups.items()):
        if sorted(row["rotation"] for row in rows)!=[0,1,2,3]:
            raise ValueError("four distinct rotations required within every question cell")
        setting=next(value for value in settings if value["setting_id"]==setting_id)
        per_question.append({"qid":qid,"condition":menu,"policy":policy,**setting,
            **{field:float(np.mean([row[field] for row in rows])) for field in
               ("reward","committed","correct","wrong","terminal_pass","observed_round")}})
    indices=np.random.default_rng(seed).integers(0,len(qids),(samples,len(qids)))
    points,draws,values={},{},{}
    summaries,focal,menu_gains,menu_policies,menu_metrics=[],[],[],[],[]
    for setting in settings:
        setting_id=setting["setting_id"]
        for menu in MENUS:
            for policy in POLICIES:
                rows=[r for r in per_question if r["condition"]==menu and r["setting_id"]==setting_id and r["policy"]==policy]
                if [r["qid"] for r in rows]!=qids:
                    raise ValueError("question ordering differs across setting cells")
                key=(menu,setting_id,policy)
                array=np.asarray([[row[name] for name in ("reward","committed","wrong","observed_round")] for row in rows])
                values[key]=array
                points[key],draws[key]=base.metric_samples(array,indices)
                original=[r for r in episodes if r["condition"]==menu and r["setting_id"]==setting_id and r["policy"]==policy]
                summaries.append({**setting,"condition":menu,"policy":policy,"n_questions":len(qids),"n_episodes":len(original),
                    **{label:sum(row[field] for row in original) for label,field in
                       (("committed_count","committed"),("correct_count","correct"),("wrong_count","wrong"),("terminal_pass_count","terminal_pass"))},
                    **{metric:base.transfer.estimate(points[key][metric],draws[key][metric]) for metric in base.METRICS}})
            left=(menu,setting_id,"adaptive_selective");right=(menu,setting_id,"fixed_selective")
            delta=(values[left][:,0]-values[right][:,0])
            boot=delta[indices].mean(axis=1)
            focal.append({**setting,"condition":menu,"left":"adaptive_selective","right":"fixed_selective","metric":"mean_reward",
                "n_questions":len(qids),"family_size":8,"mean_delta":float(delta.mean()),
                "ci95":np.quantile(boot,[.025,.975]).tolist(),"ci99_375":np.quantile(boot,[.003125,.996875]).tolist(),
                "defined_resamples":samples,"total_resamples":samples})
        by_menu=[]
        for menu in MENUS:
            by_menu.append(values[menu,setting_id,"adaptive_selective"][:,0]-values[menu,setting_id,"fixed_selective"][:,0])
        gain_delta=by_menu[1]-by_menu[0]
        gain_boot=gain_delta[indices].mean(axis=1)
        gain_estimate=base.transfer.estimate(float(gain_delta.mean()),gain_boot)
        menu_gains.append({**setting,"metric":"mean_reward","direction":"same_category minus independent",
            "contrast":"difference_in_adaptive_selective_minus_fixed_selective_gains","n_questions":len(qids),
            "mean_delta":gain_estimate.pop("mean"),**gain_estimate})
        for policy in POLICIES:
            left=(MENUS[1],setting_id,policy);right=(MENUS[0],setting_id,policy)
            for metric in base.METRICS:
                result={**setting,"policy":policy,"metric":metric,"left_condition":MENUS[1],"right_condition":MENUS[0],
                    "n_questions":len(qids),"direction":"same_category minus independent",
                    **difference(points[left][metric],points[right][metric],draws[left][metric],draws[right][metric])}
                if metric=="mean_reward":
                    menu_policies.append(result)
                else:
                    menu_metrics.append(result)
    return {"schema_version":"imcqa-crossed-settings-analysis-v1","n_questions":len(qids),
        "n_unique_plain_score_states":len(qids)*2*4*5,"n_policy_episode_cell_rows":len(episodes),
        "model":"qwen7b","bootstrap_samples":samples,"bootstrap_seed":seed,
        "unit":"Question; four rotations averaged within question; one shared bootstrap index matrix across all settings, menus and policies.",
        "repeated_setting_cells_are_not_independent":True,"policy_summaries":summaries,"primary_contrasts":focal,
        "menu_gain_differences":menu_gains,"menu_policy_differences":menu_policies,"menu_metric_differences":menu_metrics},per_question


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("evidence","archive","prior-analysis","plan","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError("create-once output path already exists")
    if old.sha256(args.plan)!=PLAN_SHA256:
        raise ValueError("frozen crossed-settings plan hash differs")
    plan=old.load_json(args.plan)
    root=Path(__file__).resolve().parents[1]
    prior_receipt=old.load_json(args.prior_analysis/"analysis_receipt.json")
    if old.sha256(Path(base.__file__))!=prior_receipt["analyzer_sha256"]:
        raise ValueError("original fixed-abstention analyzer source differs")
    for name,expected in prior_receipt["outputs_sha256"].items():
        if old.sha256(args.prior_analysis/name)!=expected:
            raise ValueError("original fixed-abstention output differs: "+name)
    views,fits,_,validation=base.load_validated_inputs(args.evidence,args.archive,root/"configs/imcqa_fixed_abstention.json")
    if old.sha256(args.archive)!=plan["input_archive_sha256"]:
        raise ValueError("crossed-settings input archive differs")
    episodes,states,used=build_crossed_episodes(views,fits,plan["settings"])
    invariants=check_invariants(episodes)
    original=match_original(episodes,args.prior_analysis/"episodes.csv")
    if len(episodes)!=12800 or len(states)!=16000 or original["exact_episode_matches"]!=3200:
        raise ValueError("crossed or original episode coverage differs")
    summary,per_question=summarize(episodes,plan["settings"],samples=plan["inference"]["bootstrap_samples"],seed=plan["inference"]["seed"])
    summary.update(analysis_id=plan["analysis_id"],status=plan["status"],plan_sha256=PLAN_SHA256,
        policy_definitions=plan["policies"],limitations=plan["limitations"],frozen_calibrators=fits,
        used_parameter_cells=used,fitting_performed=False,model_inference_performed=False)
    validation.update(crossed_settings_plan_sha256=PLAN_SHA256,parameter_immutability_passed=True,
        crossed_setting_invariants=invariants,original_fixed_abstention_reproduction=original,
        original_fixed_abstention_receipt_sha256=old.sha256(args.prior_analysis/"analysis_receipt.json"))
    args.out.mkdir(parents=True,exist_ok=False)
    old.write_json(args.out/"summary.json",summary)
    old.write_json(args.out/"validation.json",validation)
    for name,rows in (("episodes",episodes),("per_question",per_question),("trajectories",states)):
        base.transfer.write_csv(args.out/(name+".csv"),rows)
    for name in ("policy_summaries","primary_contrasts","menu_gain_differences","menu_policy_differences","menu_metric_differences"):
        base.transfer.write_csv(args.out/(name+".csv"),summary[name])
    old.write_json(args.out/"analysis_receipt.json",{"status":"complete","analyzer_sha256":old.sha256(Path(__file__)),
        "plan_sha256":PLAN_SHA256,"input_archive_sha256":plan["input_archive_sha256"],
        "original_fixed_abstention_receipt_sha256":old.sha256(args.prior_analysis/"analysis_receipt.json"),
        "fitting_performed":False,"model_inference_performed":False,
        "outputs_sha256":{path.name:old.sha256(path) for path in sorted(args.out.iterdir())}})
    print(json.dumps({"status":"complete","n_questions":summary["n_questions"],"n_policy_episode_cell_rows":len(episodes),
        "exact_old_episode_matches":original["exact_episode_matches"],"out":str(args.out)}))


if __name__=="__main__":
    main()
