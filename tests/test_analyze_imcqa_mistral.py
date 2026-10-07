"""CPU-only false-case tests for frozen Mistral evaluation and pairing."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_imcqa_mistral as analysis
from scripts import analyze_imcqa_tuned as tuned
from scripts import imcqa_mistral_design as design
from scripts import imcqa_mistral_scoring as scoring
from scripts import select_imcqa_mistral_policies as selector


def frozen_lock():
    cal=[f"cal{i}" for i in range(20)]
    dev=[f"dev{i}" for i in range(120)]
    fits={m:{"condition":m,"answer_source":"plain","fit_split":"calibration",
        "fit_qids":cal,"n_fit_questions":20,"n_fit_states":400,"feature_clip":[1e-6,1-1e-6],
        "l2":.01,"intercept":0.,"slope":1.} for m in tuned.MENUS}
    lock={"schema_version":"imcqa-mistral-policies-v1","status":"frozen_for_evaluation",
        "protocol":design.PROTOCOL,"model":deepcopy(design.MODEL),"qwen_coefficients_reused":False,
        "evaluation_outcomes_used":False,"n_model_specific_calibration_questions":20,
        "n_policy_selection_questions":120,"config_sha256":design.file_hash(design.config_path()),
        "calibration_qids":cal,"selection_qids":dev,"calibration_group_ids":["g"+q for q in cal],
        "selection_group_ids":["g"+q for q in dev],"calibrators":fits,
        "policies":[{"condition":m,"family":f,"always_pass":False,"threshold":.5,
            "fixed_round":2 if f=="fixed_selective" else None}
            for m in tuned.MENUS for f in ("fixed_selective","adaptive_selective")]}
    for key in ("development_public_sha256","development_evaluator_sha256","development_scores_sha256",
                "development_receipt_sha256","evaluation_template_sha256","source_manifest_sha256","selection_script_sha256"):
        lock[key]="1"*64
    lock["development_numeric_audit"]={"passed":True,"validated_rows":5600,
        "scores_sha256":lock["development_scores_sha256"],"public_input_sha256":lock["development_public_sha256"]}
    lock["development_token_audit"]={"passed":True,"reconstructed_contexts":5600,
        "chat_template_sha256":scoring.CHAT_TEMPLATE_SHA256,"action_token_ids":scoring.ACTION_TOKEN_IDS}
    return lock


def views(n=2):
    return [{"qid":f"eval{i}","group_id":f"eg{i}","condition":menu,"rotation":rotation,
        "round":rnd,"arm":"plain","split":"test","score_id":f"{i}:{menu}:{rotation}:{rnd}",
        "canonical_answer_probabilities":{"A":.5,"B":.2,"C":.2,"D":.1},
        "candidate_choice":"A","candidate_correct":rnd%2==1}
        for i in range(n) for menu in tuned.MENUS for rotation in range(4) for rnd in range(1,6)]


def replay(n=2):
    return tuned.build_episodes(views(n),selector.validate_policy_lock(frozen_lock()))


@pytest.mark.parametrize("mutation",["model","revision","reuse","refit","fit_split","fit_qids","numeric"])
def test_invalid_model_or_refit_lock_rejected(mutation):
    lock=frozen_lock()
    if mutation=="model":lock["model"]["name"]="Qwen/Qwen2.5-7B-Instruct"
    elif mutation=="revision":lock["model"]["revision"]="0"*40
    elif mutation=="reuse":lock["qwen_coefficients_reused"]=True
    elif mutation=="refit":lock["evaluation_outcomes_used"]=True
    elif mutation=="fit_split":lock["calibrators"][tuned.MENUS[0]]["fit_split"]="test"
    elif mutation=="fit_qids":lock["calibrators"][tuned.MENUS[0]]["fit_qids"]=["eval0"]*20
    else:lock["development_numeric_audit"]["passed"]=False
    with pytest.raises(ValueError):selector.validate_policy_lock(lock)


def test_projection_is_explicit_and_does_not_mutate_lock():
    lock=frozen_lock();before=deepcopy(lock)
    projected=selector.validate_policy_lock(lock)
    assert lock==before
    assert projected["schema_version"]=="imcqa-independently-tuned-policies-v1"
    assert projected["model"]==design.MODEL
    assert projected["policies"]==lock["policies"]


@pytest.mark.parametrize("mutation",["missing_round","duplicate_round","missing_rotation","calibration_question","selection_group"])
def test_missing_grid_or_leakage_rejected(mutation):
    states=views();lock=selector.validate_policy_lock(frozen_lock())
    if mutation=="missing_round":states.pop()
    elif mutation=="duplicate_round":states[0]=deepcopy(states[1])
    elif mutation=="missing_rotation":states=[r for r in states if r["rotation"]!=3]
    elif mutation=="calibration_question":
        for r in states:
            if r["qid"]=="eval0":r["qid"]="cal0"
    else:
        for r in states:
            if r["qid"]=="eval0":r["group_id"]="gdev0"
    with pytest.raises(ValueError):tuned.build_episodes(states,lock)


def test_inclusive_threshold_and_pass_forced_controls():
    lock=frozen_lock()
    # A fitted constant exactly .5 exercises the >= boundary without logit roundoff.
    for fit in lock["calibrators"].values():fit["slope"]=0
    episodes=tuned.build_episodes(views(),selector.validate_policy_lock(lock))
    assert all(r["round"]==1 for r in episodes if r["policy"]=="adaptive_selective")
    assert all(r["round"]==2 for r in episodes if r["policy"]=="fixed_selective")
    for policy in lock["policies"]:policy.update(always_pass=True,threshold=None,fixed_round=None)
    passed=tuned.build_episodes(views(),selector.validate_policy_lock(lock))
    assert all(r["round"] is None and r["reward"]==0 for r in passed if r["policy"].endswith("selective"))
    assert all(r["round"]==5 for r in passed if r["policy"].endswith("forced"))
    timing=analysis.supplemental_summaries(passed,views())["answer_timing"]
    assert all(r["answer_only_mean_round"] is None and r["answer_only_mean_round_ci95"] is None
               for r in timing if r["policy"].endswith("selective"))


def test_label_ties_follow_display_order_before_canonical_rotation():
    job={"option_source_ids":{"A":"B","B":"C","C":"D","D":"A"},
         "allowed_actions":"ABCD","wait_label":"E","round":1,"canonical_gold_option_id":"B"}
    logits=[1.,1.,1.,1.,0.]
    row={"raw_action_logits":dict(zip("ABCDE",logits)),
         **scoring.action_statistics(logits,"ABCD",job["option_source_ids"])}
    view=analysis.protocol.score_view(row,job)
    assert view["candidate_choice"]=="B" and view["candidate_correct"] is True
    row["chosen_action"]="B"
    with pytest.raises(ValueError,match="argmax or tie"):analysis.protocol.score_view(row,job)


@pytest.mark.parametrize("mutation",["model","missing","action"])
def test_model_aware_raw_score_validator(mutation):
    job={k:None for k in design.PUBLIC_JOB_KEYS}
    job.update(score_id="state",allowed_actions="ABCD",option_source_ids=dict(zip("ABCD","ABCD")))
    row={k:v for k,v in job.items() if k!="prompt"}
    row.update(model_tag=scoring.MODEL_TAG,schema_version=scoring.SCORE_SCHEMA,
        **scoring.action_statistics([1.,0.,0.,0.,0.],"ABCD",job["option_source_ids"]))
    scoring.validate_rows([job],[row],complete=True)
    if mutation=="model":row["model_tag"]="qwen7b"
    elif mutation=="action":row["chosen_action"]="B"
    if mutation=="missing":
        with pytest.raises(ValueError):scoring.validate_rows([job],[],complete=True)
    else:
        with pytest.raises(ValueError):scoring.validate_rows([job],[row],complete=True)


def test_token_context_mutation_rejected(monkeypatch):
    # Isolates the wrapper gate; worker integration tests cover native tokenizer.
    tokenizer=SimpleNamespace(chat_template="fake")
    monkeypatch.setattr(scoring,"CHAT_TEMPLATE_SHA256",scoring.base.sha(b"fake"))
    context={"scored_input_token_ids":[1,2,3],"scored_context_sha256":"x",
        "rendered_prompt_sha256":"y","option_token_ids":scoring.ACTION_TOKEN_IDS}
    monkeypatch.setattr(scoring,"prepare_context",lambda tokenizer,job:deepcopy(context))
    row={"score_id":"s",**context}
    assert analysis.validate_token_contexts([{"score_id":"s"}],[row],tokenizer)["passed"]
    row["scored_input_token_ids"]=[1,2,4]
    with pytest.raises(ValueError,match="tokens/context"):analysis.validate_token_contexts([{"score_id":"s"}],[row],tokenizer)


@pytest.mark.parametrize("mutation",["source_commit","development_receipt_hash","policy_lock_hash"])
def test_receipt_source_mismatch_prevents_gold_join(tmp_path,monkeypatch,mutation):
    # Test the analyzer's outer source-commit gate; shared numeric validator has its own suite.
    lock=frozen_lock();evaluator={};dev={}
    for name,value in (("lock",lock),("evaluator",evaluator),("dev",dev),("manifest",{})):
        design.write_new(tmp_path/(name+".json"),value)
    public={"policy_lock_sha256":design.file_hash(tmp_path/"lock.json"),
        "main_dataset_sha256":design.file_hash(tmp_path/"evaluator.json"),
        "development_receipt_sha256":design.file_hash(tmp_path/"dev.json")}
    if mutation=="development_receipt_hash":public["development_receipt_sha256"]="0"*64
    elif mutation=="policy_lock_hash":public["policy_lock_sha256"]="0"*64
    design.write_new(tmp_path/"public.json",public)
    design.write_new(tmp_path/"run/receipt.json",{"source_commit":"b"*40})
    monkeypatch.setattr(design,"validate_stage_manifest",lambda *a,**k:True)
    monkeypatch.setattr(selector,"validate_policy_lock",lambda *a,**k:{})
    monkeypatch.setattr(analysis,"reconstruct_views",lambda *a:pytest.fail("gold joined before source gate"))
    with pytest.raises(ValueError,match="committed source|binding differs"):
        analysis.load_views(tmp_path/"public.json",tmp_path/"evaluator.json",tmp_path/"run",tmp_path/"lock.json",
            tmp_path/"manifest.json",tmp_path/"dev.json",tmp_path,"a"*40)


def test_qwen_file_hash_is_mandatory(tmp_path):
    path=tmp_path/"episodes.csv";path.write_text("qid,policy\nwrong,wrong\n")
    with pytest.raises(ValueError,match="frozen validated"):analysis.load_qwen_episodes(path)


def test_cross_model_pairing_and_zero_interaction():
    episodes=replay(3)
    result=analysis.cross_model_interactions(episodes,list(reversed(episodes)),samples=123)
    assert result["bootstrap_seed"]==20261007
    assert all(r["mean_delta"]==0 and r["ci95"]==[0,0] for r in result["contrasts"])
    bad=deepcopy(episodes);bad.pop()
    with pytest.raises(ValueError):analysis.cross_model_interactions(episodes,bad,samples=12)


def test_cross_model_effect_matches_hand_calculation():
    qwen=replay(2);mistral=deepcopy(qwen)
    for row in mistral:
        if row["policy"]=="fixed_selective":
            row.update(round=1,observed_round=1,correct=True,wrong=False,reward=1.)
    # Qwen AS - FS = 1 - (-1) = 2; altered Mistral AS - FS = 1 - 1 = 0.
    result=analysis.cross_model_interactions(mistral,qwen,samples=99)
    assert result["contrasts"][0]["mean_delta"]==pytest.approx(-2)
    assert result["contrasts"][1]["mean_delta"]==pytest.approx(-2)
    assert result["contrasts"][2]["mean_delta"]==pytest.approx(0)


def test_primary_summary_keeps_original_seed_and_adjustment():
    result,_=tuned.summarize(replay())
    assert result["bootstrap_samples"]==20000 and result["bootstrap_seed"]==1
    assert len(result["primary_contrasts"])==2
    assert all(r["family_size"]==2 and "ci97_5" in r for r in result["primary_contrasts"])


@pytest.mark.parametrize("mutation",["falsepass","wrongreward","duplicate","group"])
def test_episode_invariants_reject_corruption(mutation):
    episodes=replay()
    if mutation=="falsepass":episodes[0]["terminal_pass"]=True
    elif mutation=="wrongreward":episodes[0]["reward"]=99
    elif mutation=="duplicate":episodes[1]=deepcopy(episodes[0])
    else:episodes[0]["group_id"]="different"
    with pytest.raises(ValueError):analysis.validate_episode_grid(episodes)


def load_sibling_design_helpers():
    """Load the exact sibling fixture in either pytest package/import mode."""
    path=Path(__file__).with_name("test_imcqa_mistral_design.py")
    spec=importlib.util.spec_from_file_location("_imcqa_local_design_test_helpers",path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the local design-test fixture")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sibling_fixture_load_is_path_bound(tmp_path,monkeypatch):
    # A package-style test checkout must not resolve an ambient top-level module.
    monkeypatch.setitem(sys.modules,"test_imcqa_mistral_design",None)
    monkeypatch.chdir(tmp_path)
    helper=load_sibling_design_helpers()
    assert Path(helper.__file__).resolve()==Path(__file__).with_name("test_imcqa_mistral_design.py").resolve()
    assert callable(helper.synthetic_package)


@pytest.fixture(scope="module")
def full_synthetic_evaluation():
    package=load_sibling_design_helpers().synthetic_package("evaluation",locked=True)
    questions={}
    rows=[]
    for job in package["jobs"]:
        q=questions.setdefault(job["qid"],{"qid":job["qid"],"group_id":job["group_id"],
            "split":"test","menus":[],"prefixes":[]})
        payload=design.old._payload(job["prompt"])
        if job["rotation"]==0:
            if job["round"]==1:
                q["menus"].append({"condition":job["condition"],"menu_id":job["menu_id"],
                    "gold_option_id":"A","options":payload["options"]})
            if job["condition"]==design.CONDITIONS[0]:
                q["prefixes"].append({"prefix_id":job["prefix_id"],"text":payload["question_prefix"],"fraction":job["fraction"]})
        rows.append({**{k:v for k,v in job.items() if k!="prompt"},"model_tag":scoring.MODEL_TAG,
            "schema_version":scoring.SCORE_SCHEMA,
            **scoring.action_statistics([1.,0.,0.,0.,0.],"ABCD",job["option_source_ids"])})
    evaluator={"schema_version":"imcqa-mistral-evaluator-v1","stage":"evaluation","questions":list(questions.values())}
    package["main_dataset_sha256"]=design.value_hash(evaluator)
    return package,evaluator,rows


def test_full_850_gold_join_and_rotations(full_synthetic_evaluation):
    package,evaluator,rows=full_synthetic_evaluation
    result=analysis.reconstruct_views(package,evaluator,list(reversed(rows)))
    assert len(result)==34000 and len({r["qid"] for r in result})==850
    assert sum(r["candidate_correct"] for r in result)==8500


@pytest.mark.parametrize("mutation",["wrong_model","missing_grid","evaluator_hash","duplicate_prefix"])
def test_full_reconstruction_false_cases(full_synthetic_evaluation,mutation):
    package,evaluator,rows=deepcopy(full_synthetic_evaluation)
    if mutation=="wrong_model":rows[0]["model_tag"]="qwen7b"
    elif mutation=="missing_grid":rows.pop()
    elif mutation=="evaluator_hash":evaluator["questions"][0]["menus"][0]["gold_option_id"]="B"
    else:
        evaluator["questions"][0]["prefixes"].append(deepcopy(evaluator["questions"][0]["prefixes"][0]))
        package["main_dataset_sha256"]=design.value_hash(evaluator)
    with pytest.raises(ValueError):analysis.reconstruct_views(package,evaluator,rows)
