"""Leakage, complete-grid, role-selection and lock-before-evaluation tests."""
from copy import deepcopy
import math

import numpy as np
import pytest

from scripts import imcqa_mistral_design as d
from scripts import select_imcqa_mistral_policies as selector
from scripts import acl_option_scoring as base


def synthetic_package(stage="development",locked=False):
    n=140 if stage=="development" else 850
    qids=[f"q{i}" for i in range(n)]
    s={"selected_qids":qids,"qid_split":{},"qid_group":{},"qid_text_sha256":{},"qid_category":{},"source_job_ids":{},"original_score_ids":{},"prompt_sha256":{},"outcomes_used_for_selection":False}
    jobs=[]
    for i,q in enumerate(qids):
        split=("calibration" if i<20 else "selection") if stage=="development" else "test"
        s["qid_split"][q]=split;s["qid_group"][q]="g"+q;s["qid_category"][q]="synthetic"
        full=" ".join([q]+["word"+str(x) for x in range(99)])
        s["qid_text_sha256"][q]=base.sha(base.canonical(d.tuned.normalized_tokens(full)))
        for c in d.CONDITIONS:
            for rnd,p in enumerate(d.PREFIX_IDS,1):
                prefix=" ".join(full.split()[:rnd*20])
                source={"source_job_id":base.sha(f"{q}|{c}|{p}".encode()),"source_prompt_sha256":base.sha(prefix.encode()),"qid":q,"group_id":"g"+q,"split":split,"condition":c,"menu_id":"m"+c,"prefix_id":p,"fraction":rnd/5,"round":rnd,"reward":d.REWARDS[rnd-1]}
                for r in range(4):
                    old=d.old._real_job(source,{"question_prefix":prefix,"options":[{"id":k,"text":"option"+k} for k in "ABCD"]},"plain",r)
                    job={**old,"score_index":len(jobs),"score_id":source["source_job_id"]+f":mistral:plain:r{r}","source_score_id":old["score_id"],"execution":"new"}
                    jobs.append(job)
                    for name,k in (("source_job_ids","source_job_id"),("original_score_ids","source_score_id"),("prompt_sha256","prompt_sha256")):s[name][job["score_id"]]=job[k]
    return {"schema_version":d.SCHEMA,"protocol":d.PROTOCOL,"model":d.MODEL,"stage":stage,"n_questions":n,"jobs":jobs,"rewards":list(d.REWARDS),"wrong_reward":-1,"pass_reward":0,"prefix_ids":list(d.PREFIX_IDS),"selection":s,"selection_id":d.value_hash(s),"main_dataset_sha256":"1"*64,"source_manifest_sha256":"2"*64,"config_sha256":d.file_hash(d.config_path()),"policy_lock_sha256":"3"*64 if locked else None,"development_receipt_sha256":"4"*64 if locked else None}


@pytest.fixture(scope="module")
def development(): return synthetic_package()


def test_gold_free_complete_development(development):
    assert len(d.validate_public_package(development))==5600
    assert d.validate_stage_manifest(d.stage_manifest(development),development)


@pytest.mark.parametrize("where",["package","job"])
def test_gold_field_rejected(development,where):
    p=deepcopy(development)
    (p if where=="package" else p["jobs"][0])["gold_option_id"]="A"
    with pytest.raises(ValueError):d.validate_public_package(p)


@pytest.mark.parametrize("mutation",["drop","duplicate","prompt","rotation","reuse","model","role","group"])
def test_mutations_rejected(development,mutation):
    p=deepcopy(development)
    if mutation=="drop":p["jobs"].pop()
    elif mutation=="duplicate":p["jobs"][1]=deepcopy(p["jobs"][0])
    elif mutation=="prompt":p["jobs"][0]["prompt"]+="leaked text"
    elif mutation=="rotation":p["jobs"][0]["option_source_ids"]={"A":"B","B":"A","C":"C","D":"D"}
    elif mutation=="reuse":p["jobs"][0]["execution"]="reuse"
    elif mutation=="model":p["model"]={**d.MODEL,"name":"Qwen/Qwen2.5-7B-Instruct"}
    elif mutation=="role":p["selection"]["qid_split"]["q0"]="test";p["selection_id"]=d.value_hash(p["selection"])
    elif mutation=="group":p["selection"]["qid_group"]["q1"]="gq0";p["selection_id"]=d.value_hash(p["selection"])
    with pytest.raises(ValueError):d.validate_public_package(p)


def test_evaluation_rejects_missing_lock():
    p=synthetic_package("evaluation")
    with pytest.raises(ValueError,match="requires frozen"):d.validate_public_package(p)
    assert len(d.validate_public_package(p,allow_unlocked_evaluation=True))==34000


def eval_boundary():
    p=synthetic_package("evaluation",True)
    receipt={"status":"complete","stage":"development","model_tag":d.MODEL["tag"],"model":d.MODEL["name"],"revision":d.MODEL["revision"],"expected_rows":5600,"completed_rows":5600,"total_contexts":5600,"automatic_retries":0,"reused_rows":0,"generation":False,"sampling":False,"public_input_sha256":"5"*64,"scores_sha256":"6"*64}
    lock={"status":"frozen_for_evaluation","protocol":d.PROTOCOL,"model":d.MODEL,"development_public_sha256":"5"*64,"development_scores_sha256":"6"*64,"development_receipt_sha256":d.value_hash(receipt),"config_sha256":p["config_sha256"],"source_manifest_sha256":p["source_manifest_sha256"],"development_numeric_audit":{"passed":True,"validated_rows":5600,"public_input_sha256":"5"*64,"scores_sha256":"6"*64}}
    return p,receipt,lock


def test_evaluation_complete_prerequisites():
    p,r,l=eval_boundary();p["policy_lock_sha256"]=d.value_hash(l);p["development_receipt_sha256"]=d.value_hash(r)
    assert d.validate_stage_manifest(d.stage_manifest(p),p,policy_lock=l,development_receipt=r)


@pytest.mark.parametrize("bad",["partial","wrong_model","count","retry","numeric"])
def test_self_consistent_bad_prerequisite_rejected(bad):
    p,r,l=eval_boundary()
    if bad=="partial":r["status"]="budget_stop"
    elif bad=="wrong_model":r["model_tag"]="qwen7b"
    elif bad=="count":r["completed_rows"]=5599
    elif bad=="retry":r["automatic_retries"]=1
    else:l["development_numeric_audit"]["passed"]=False
    l["development_receipt_sha256"]=d.value_hash(r);p["policy_lock_sha256"]=d.value_hash(l);p["development_receipt_sha256"]=d.value_hash(r)
    with pytest.raises(ValueError):d.validate_stage_manifest(d.stage_manifest(p),p,policy_lock=l,development_receipt=r)


def fit_rows(correct=True):
    return [{"qid":f"q{q}","split":"calibration","condition":"independent_pool","round":t,"rotation":r,"candidate_confidence":.2+.1*t,"candidate_correct":correct} for q in range(20) for t in range(1,6) for r in range(4)]


def test_fit_degenerate_labels():
    f=selector.fit_calibrator(fit_rows(False),d.read_config()["calibration"])
    assert f["slope"]==0
    assert f["method"]=="smoothed_intercept_only_degenerate_labels"
    assert math.isclose(1/(1+math.exp(-f["intercept"])),.5/401)


@pytest.mark.parametrize("bad",["selection","test","duplicate","nonfinite"])
def test_fit_leakage_or_malformed_rejected(bad):
    rows=fit_rows()
    if bad in ("selection","test"):rows[0]["split"]=bad
    elif bad=="duplicate":rows[0]=rows[1]
    else:rows[0]["candidate_confidence"]=float("nan")
    with pytest.raises(ValueError):selector.fit_calibrator(rows,d.read_config()["calibration"])


def test_monotone_fit_with_inverse_relation():
    rows=fit_rows()
    for r in rows:r["candidate_correct"]=r["round"]<3
    f=selector.fit_calibrator(rows,d.read_config()["calibration"])
    assert f["slope"]==pytest.approx(0,abs=1e-8)


def test_near_duplicate_across_roles_rejected():
    qs=[{"qid":"a","group_id":"g1","split":"calibration","prefixes":[{"prefix_id":"p10","text":" ".join('word'+str(i) for i in range(100))}]},{"qid":"b","group_id":"g2","split":"test","prefixes":[{"prefix_id":"p10","text":" ".join('word'+str(i) for i in range(99))+" changed"}]}]
    with pytest.raises(ValueError,match="near-duplicate"):d.validate_role_separation(qs)


def test_all_correct_selection_uses_calibration_only():
    states=[]
    for i in range(140):
        for menu in d.CONDITIONS:
            for t in range(1,6):
                for r in range(4):
                    states.append({"qid":f"q{i}","group_id":f"g{i}","split":"calibration" if i<20 else "selection","condition":menu,"round":t,"rotation":r,"candidate_confidence":.8,"candidate_correct":True,"candidate_choice":"A","score_id":f"q{i}:{menu}:{t}:{r}","cohort":"mistral_development"})
    fits,grid,chosen,episodes,validation=selector.select_from_states(states,d.read_config())
    assert len(grid)==256 and len(episodes)==1920
    assert all(len(f["fit_qids"])==20 for f in fits.values())
    assert all(p["threshold"]==.95 and p["mean_reward"]==1 for p in chosen)
    assert all(p["fixed_round"]==1 for p in chosen if p["family"]=="fixed_selective")
    states[0]["split"]="test"
    with pytest.raises(ValueError):selector.select_from_states(states,d.read_config())


def scientific_lock():
    from scripts import imcqa_mistral_scoring as scoring
    _,receipt,lock=eval_boundary()
    lock.update(schema_version="imcqa-mistral-policies-v1",qwen_coefficients_reused=False,evaluation_outcomes_used=False,n_model_specific_calibration_questions=20,n_policy_selection_questions=120,
        calibration_qids=[f"q{i}" for i in range(20)],calibration_group_ids=[f"g{i}" for i in range(20)],selection_qids=[f"q{i}" for i in range(20,140)],selection_group_ids=[f"g{i}" for i in range(20,140)],
        development_evaluator_sha256="7"*64,evaluation_template_sha256="8"*64,selection_script_sha256="9"*64)
    fit=selector.fit_calibrator(fit_rows(),d.read_config()["calibration"])
    lock["calibrators"]={m:{**fit,"condition":m} for m in d.CONDITIONS}
    lock["policies"]=[{"condition":m,"family":f,"threshold":.65,"fixed_round":1 if f=="fixed_selective" else None,"always_pass":False} for m in d.CONDITIONS for f in ("fixed_selective","adaptive_selective")]
    lock["development_token_audit"]={"passed":True,"reconstructed_contexts":5600,"chat_template_sha256":scoring.CHAT_TEMPLATE_SHA256,"action_token_ids":scoring.ACTION_TOKEN_IDS}
    return lock


def test_full_policy_lock_projection():
    projected=selector.validate_policy_lock(scientific_lock())
    assert projected["schema_version"]=="imcqa-independently-tuned-policies-v1"
    assert projected["status"]=="frozen_for_fresh_evaluation"


@pytest.mark.parametrize("bad",["missing_coefficients","qwen_coefficients","eval_fit","negative_slope","out_of_grid","role_overlap","missing_policy","token_audit"])
def test_scientific_lock_mutations_rejected(bad):
    lock=scientific_lock()
    if bad=="missing_coefficients":del lock["calibrators"]
    elif bad=="qwen_coefficients":lock["qwen_coefficients_reused"]=True
    elif bad=="eval_fit":lock["calibrators"][d.CONDITIONS[0]]["fit_split"]="test"
    elif bad=="negative_slope":lock["calibrators"][d.CONDITIONS[0]]["slope"]=-.1
    elif bad=="out_of_grid":lock["policies"][0]["threshold"]=.123
    elif bad=="role_overlap":lock["selection_group_ids"][0]=lock["calibration_group_ids"][0]
    elif bad=="missing_policy":lock["policies"].pop()
    else:lock["development_token_audit"]["reconstructed_contexts"]=5599
    with pytest.raises(ValueError):selector.validate_policy_lock(lock)
