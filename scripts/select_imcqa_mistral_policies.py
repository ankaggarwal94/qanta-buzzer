#!/usr/bin/env python3
"""Fit Mistral on 20 calibration questions, select on 120, and lock evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit

from scripts import imcqa_mistral_design as design
from scripts import analyze_imcqa_protocol_pilot as protocol
from scripts import select_imcqa_tuned_policies as selection


def validate_policy_lock(lock,evaluation_package=None):
    """Validate the model-specific lock, then project to the legacy replay schema."""
    required={"schema_version","status","model","protocol","qwen_coefficients_reused","evaluation_outcomes_used","n_model_specific_calibration_questions","n_policy_selection_questions","config_sha256","development_public_sha256","development_evaluator_sha256","development_scores_sha256","development_receipt_sha256","evaluation_template_sha256","source_manifest_sha256","selection_script_sha256","calibration_qids","calibration_group_ids","selection_qids","selection_group_ids","calibrators","policies","development_numeric_audit","development_token_audit"}
    if not isinstance(lock,dict) or not required<=set(lock):raise ValueError("complete scientific policy lock required")
    if (lock.get("schema_version")!="imcqa-mistral-policies-v1" or lock.get("status")!="frozen_for_evaluation" or lock.get("model")!=design.MODEL or lock.get("protocol")!=design.PROTOCOL
        or lock.get("qwen_coefficients_reused") is not False or lock.get("evaluation_outcomes_used") is not False
        or lock.get("n_model_specific_calibration_questions")!=20 or lock.get("n_policy_selection_questions")!=120
        or lock.get("config_sha256")!=design.file_hash(design.config_path())):raise ValueError("Mistral policy-lock scientific identity differs")
    for key in ("development_public_sha256","development_evaluator_sha256","development_scores_sha256","development_receipt_sha256","evaluation_template_sha256","source_manifest_sha256","selection_script_sha256"):
        if not design._hash(lock.get(key)):raise ValueError("missing policy-lock input binding")
    for qkey,gkey,n in (("calibration_qids","calibration_group_ids",20),("selection_qids","selection_group_ids",120)):
        if len(lock[qkey])!=n or len(set(lock[qkey]))!=n or len(lock[gkey])!=n or len(set(lock[gkey]))!=n:raise ValueError("policy-lock role identities differ")
    if set(lock["calibration_qids"])&set(lock["selection_qids"]) or set(lock["calibration_group_ids"])&set(lock["selection_group_ids"]):raise ValueError("policy-lock role leakage")
    fits=lock.get("calibrators",{})
    if set(fits)!=set(selection.MENUS):raise ValueError("calibration menu coverage differs")
    for menu,fit in fits.items():
        if fit.get("condition")!=menu or fit.get("fit_split")!="calibration" or fit.get("n_fit_questions")!=20 or fit.get("n_fit_states")!=400 or set(fit.get("fit_qids",[]))!=set(lock["calibration_qids"]) or fit.get("feature_clip")!=[1e-6,1-1e-6] or fit.get("l2")!=.01:
            raise ValueError("model-specific calibrator provenance differs")
        if not all(isinstance(fit.get(k),(int,float)) and np.isfinite(fit[k]) for k in ("intercept","slope")) or fit["slope"]<0:raise ValueError("invalid monotone logistic fit")
    audit=lock.get("development_numeric_audit",{})
    if audit.get("passed") is not True or audit.get("validated_rows")!=5600 or audit.get("scores_sha256")!=lock["development_scores_sha256"] or audit.get("public_input_sha256")!=lock["development_public_sha256"]:raise ValueError("lock lacks complete development numeric audit")
    from scripts import imcqa_mistral_scoring as scoring
    token_audit=lock["development_token_audit"]
    if token_audit.get("passed") is not True or token_audit.get("reconstructed_contexts")!=5600 or token_audit.get("chat_template_sha256")!=scoring.CHAT_TEMPLATE_SHA256 or token_audit.get("action_token_ids")!=scoring.ACTION_TOKEN_IDS:
        raise ValueError("lock lacks complete native re-tokenization audit")
    if evaluation_package is not None:
        design.validate_public_package(evaluation_package)
        if evaluation_package["stage"]!="evaluation" or evaluation_package["policy_lock_sha256"]!=design.value_hash(lock) or evaluation_package["development_receipt_sha256"]!=lock["development_receipt_sha256"]:raise ValueError("evaluation public/lock binding differs")
        template={**evaluation_package,"policy_lock_sha256":None,"development_receipt_sha256":None}
        if design.value_hash(template)!=lock["evaluation_template_sha256"] or evaluation_package["source_manifest_sha256"]!=lock["source_manifest_sha256"]:raise ValueError("evaluation template/source changed after development")
    generic={**lock,"schema_version":"imcqa-independently-tuned-policies-v1","status":"frozen_for_fresh_evaluation"}
    policykeys={"condition","family","threshold","fixed_round","always_pass"}
    if not isinstance(lock["policies"],list) or len(lock["policies"])!=4 or any(not isinstance(p,dict) or not policykeys<=set(p) for p in lock["policies"]):raise ValueError("four complete selected policies required")
    from scripts.analyze_imcqa_tuned import policy_map
    policy_map(generic)
    return generic


def fit_calibrator(rows,settings):
    """Preserve the original monotone, question-balanced logistic objective."""
    if not rows or any(r["split"]!="calibration" for r in rows): raise ValueError("fit accepts calibration rows only")
    if len({r["condition"] for r in rows})!=1: raise ValueError("fit cannot mix menus")
    qids=sorted({r["qid"] for r in rows})
    if len(qids)!=20 or set(Counter(r["qid"] for r in rows).values())!={20}: raise ValueError("20 questions with 20 states each required")
    identities={(r["qid"],r["round"],r["rotation"]) for r in rows}
    if len(identities)!=400 or identities!={(q,t,r) for q in qids for t in range(1,6) for r in range(4)}: raise ValueError("duplicate/missing calibration state")
    raw=np.asarray([r["candidate_confidence"] for r in rows],float);y=np.asarray([r["candidate_correct"] for r in rows],float)
    if not np.all(np.isfinite(raw)) or np.any((raw<0)|(raw>1)) or np.any((y!=0)&(y!=1)): raise ValueError("invalid fit probability or correctness")
    x=logit(np.clip(raw,*settings["feature_clip"]));initial=np.array([logit((float(y.sum())+.5)/(len(y)+1)),0.])
    penalty=settings["l2"]
    def objective(theta):
        z=theta[0]+theta[1]*x;residual=expit(z)-y
        return float(np.mean(np.logaddexp(0,z)-y*z)+penalty*theta[1]**2/2),np.array([np.mean(residual),np.mean(residual*x)+penalty*theta[1]])
    if len(set(y))==1:theta,method=initial,"smoothed_intercept_only_degenerate_labels"
    else:
        fitted=minimize(objective,initial,jac=True,method="L-BFGS-B",bounds=[(None,None),(0,None)],options={k:settings[k] for k in ("maxiter","ftol","gtol")})
        if not fitted.success or not np.all(np.isfinite(fitted.x)): raise ValueError("original calibration optimizer failed: "+fitted.message)
        theta,method=fitted.x,"monotone_regularized_logistic_correctness"
    return {"answer_source":"plain","condition":rows[0]["condition"],"fit_split":"calibration","fit_qids":qids,"n_fit_questions":20,"n_fit_states":400,"method":method,"intercept":float(theta[0]),"slope":float(theta[1]),"l2":penalty,"feature_clip":settings["feature_clip"],"objective_value":objective(theta)[0]}


def reconstruct_states(package,evaluator,rows):
    """Join CPU-only labels after complete score identity/numeric validation."""
    jobs=design.validate_public_package(package)
    if package["stage"]!="development" or evaluator.get("stage")!="development" or design.value_hash(evaluator)!=package["main_dataset_sha256"]: raise ValueError("development evaluator identity differs")
    questions={q["qid"]:q for q in evaluator["questions"]}
    if len(questions)!=140 or len(questions)!=len(evaluator["questions"]) or set(questions)!=set(package["selection"]["selected_qids"]): raise ValueError("evaluator question coverage differs")
    lookup={j["score_id"]:j for j in jobs}
    if len(rows)!=len(jobs) or {r["score_id"] for r in rows}!=set(lookup): raise ValueError("complete distinct development scores required")
    result=[]
    for row in rows:
        job=lookup[row["score_id"]];question=questions[job["qid"]]
        if row.get("model_tag")!=design.MODEL["tag"]: raise ValueError("scores are not from the replication model")
        if any(row.get(k)!=job[k] for k in design.PUBLIC_JOB_KEYS-{"prompt"}): raise ValueError("score/public metadata differ")
        if question["split"]!=job["split"] or question["group_id"]!=job["group_id"]: raise ValueError("evaluator group or role differs")
        menu=next(m for m in question["menus"] if m["condition"]==job["condition"])
        if menu["menu_id"]!=job["menu_id"]: raise ValueError("evaluator menu identity differs")
        # Verify gold is interpreted against exactly the displayed options/text.
        prefix=next(p for p in question["prefixes"] if p["prefix_id"]==job["prefix_id"])
        opts={o["id"]:o["text"] for o in menu["options"]}
        displayed=[{"id":label,"text":opts[identity]} for label,identity in job["option_source_ids"].items()]
        if protocol.expected_prompt(prefix["text"],displayed,job["round"],"plain","E")!=job["prompt"]: raise ValueError("CPU evaluator options/prefix differ from scored prompt")
        view=protocol.score_view(row,{**job,"canonical_gold_option_id":menu["gold_option_id"]})
        result.append({k:view[k] for k in ("qid","group_id","condition","rotation","round","score_id","candidate_choice","candidate_correct","canonical_gold_option_id","split")} | {"candidate_confidence":max(view["canonical_answer_probabilities"].values()),"cohort":"mistral_development"})
    return result


def select_from_states(states,config):
    """Use calibration rows for fitting and selection rows for policy search."""
    if any(r["split"] not in ("calibration","selection") for r in states): raise ValueError("evaluation data cannot enter fitting/selection")
    qroles={}
    for row in states:
        if qroles.setdefault(row["qid"],row["split"])!=row["split"]: raise ValueError("question appears in two roles")
    counts=Counter(qroles.values())
    if counts!={"calibration":20,"selection":120}: raise ValueError("20/120 development roles required")
    cal=[r for r in states if r["split"]=="calibration"];dev=[r for r in states if r["split"]=="selection"]
    if {r["group_id"] for r in cal}&{r["group_id"] for r in dev}: raise ValueError("calibration/selection group leakage")
    fits={menu:fit_calibrator([r for r in cal if r["condition"]==menu],config["calibration"]) for menu in selection.MENUS}
    # Generic role validation also rejects repeated or incomplete state grids.
    questions={r["qid"]:{"group_id":r["group_id"]} for r in states}
    validation=selection.validate_roles(dev,fits,questions,{"mistral_development":120})
    qids,grid,chosen,values,episodes=selection.select_policies(dev,fits)
    return fits,grid,list(chosen.values()),episodes,validation


def lock_and_prepare(public_path,evaluator_path,run_dir,eval_template_path,source_manifest_path,out,tokenizer_dir):
    """Require complete numerical evidence before creating a Mistral policy lock."""
    from scripts import imcqa_mistral_scoring as scoring
    config=design.read_config();package=json.loads(Path(public_path).read_text());evaluator=json.loads(Path(evaluator_path).read_text())
    run_dir=Path(run_dir);out=Path(out)
    numeric_audit=scoring.validate_completed_run(Path(public_path),run_dir)
    receipt_path=run_dir/"receipt.json";receipt=json.loads(receipt_path.read_text())
    # Canonical byte identity is needed to pass this object unchanged to the launcher.
    if design.value_hash(receipt)!=design.file_hash(receipt_path): raise ValueError("development receipt is not canonical; preserve original bytes")
    if receipt.get("status")!="complete" or receipt.get("model_tag")!=design.MODEL["tag"] or receipt.get("public_input_sha256")!=design.file_hash(public_path): raise ValueError("complete model-specific development receipt required")
    scores=run_dir/"scores.jsonl"
    if receipt.get("scores_sha256")!=design.file_hash(scores): raise ValueError("development scores checksum differs")
    rows=[json.loads(line) for line in scores.read_text().splitlines()]
    from scripts.analyze_imcqa_mistral import load_tokenizer,validate_token_contexts
    token_audit=validate_token_contexts(package["jobs"],rows,load_tokenizer(tokenizer_dir))
    states=reconstruct_states(package,evaluator,rows)
    fits,grid,policies,episodes,validation=select_from_states(states,config)
    template=json.loads(Path(eval_template_path).read_text());design.validate_public_package(template,allow_unlocked_evaluation=True)
    if template["stage"]!="evaluation" or template["policy_lock_sha256"] is not None: raise ValueError("an untouched unlocked evaluation template is required")
    if design.file_hash(source_manifest_path)!=package["source_manifest_sha256"] or package["source_manifest_sha256"]!=template["source_manifest_sha256"] or package["config_sha256"]!=template["config_sha256"]: raise ValueError("development/evaluation source/config binding differs")
    if set(package["selection"]["selected_qids"])&set(template["selection"]["selected_qids"]) or set(package["selection"]["qid_group"].values())&set(template["selection"]["qid_group"].values()) or set(package["selection"]["qid_text_sha256"].values())&set(template["selection"]["qid_text_sha256"].values()): raise ValueError("evaluation leakage into development")
    lock={"schema_version":"imcqa-mistral-policies-v1","status":"frozen_for_evaluation","protocol":design.PROTOCOL,"model":design.MODEL,"frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "calibrators":fits,"policies":policies,"calibration_qids":validation["calibration_qids"],"selection_qids":validation["selection_qids"],"calibration_group_ids":validation["calibration_group_ids"],"selection_group_ids":validation["selection_group_ids"],
        "development_public_sha256":design.file_hash(public_path),"development_evaluator_sha256":design.file_hash(evaluator_path),"development_scores_sha256":design.file_hash(scores),"development_receipt_sha256":design.file_hash(receipt_path),"development_numeric_audit":numeric_audit,"development_token_audit":token_audit,
        "evaluation_template_sha256":design.file_hash(eval_template_path),"source_manifest_sha256":design.file_hash(source_manifest_path),"config_sha256":design.file_hash(design.config_path()),"selection_script_sha256":design.file_hash(__file__),"inference_source_commit":receipt.get("source_commit"),
        "qwen_coefficients_reused":False,"evaluation_outcomes_used":False,"n_model_specific_calibration_questions":20,"n_policy_selection_questions":120,"validation":validation}
    evaluation={**template,"policy_lock_sha256":design.value_hash(lock),"development_receipt_sha256":design.file_hash(receipt_path)}
    validate_policy_lock(lock,evaluation)
    manifest=design.stage_manifest(evaluation);design.validate_stage_manifest(manifest,evaluation,policy_lock=lock,development_receipt=receipt)
    design.write_new(out/"policy_lock.json",lock)
    design.write_new(out/"evaluation_public.json",evaluation)
    design.write_new(out/"evaluation_stage_manifest.json",manifest)
    design.write_new(out/"development_receipt.json",receipt)
    design.write_new(out/"selection_summary.json",{"status":"complete","scope":"model-specific development fitting and policy selection; no evaluation outcomes","model":design.MODEL,"policies":policies,"calibrators":fits,"validation":validation})
    for name,data in (("candidate_grid.csv",grid),("selected_development_episodes.csv",episodes)):
        with (out/name).open("x") as h:
            w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    return lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("public","evaluator","run","evaluation-template","source-manifest","out","tokenizer-dir"):parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args();lock=lock_and_prepare(a.public,a.evaluator,a.run,a.evaluation_template,a.source_manifest,a.out,a.tokenizer_dir)
    print(json.dumps({"status":"frozen_for_evaluation","policy_lock_sha256":design.value_hash(lock),"policies":lock["policies"]}))


if __name__=="__main__":main()
