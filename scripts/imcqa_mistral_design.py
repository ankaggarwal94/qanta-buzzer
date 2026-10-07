"""Identity-bound, gold-free Mistral replication of the original saved prompts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import gzip
import json
from pathlib import Path
import math
import re

from scripts import acl_option_scoring as base
from scripts import imcqa_protocol_design as old
from scripts import imcqa_tuned_design as tuned

PROTOCOL = "imcqa-mistral-replication-v1"
SCHEMA = "imcqa-mistral-public-v1"
MODEL = {"tag":"mistral7b","name":"mistralai/Mistral-7B-Instruct-v0.3","revision":"c170c708c41dac9275d15a8fff4eca08d52bab71"}
PUBLIC_JOB_KEYS = old.PUBLIC_JOB_KEYS
PREFIX_IDS, REWARDS, CONDITIONS = old.PREFIX_IDS, old.REWARDS, old.CONDITIONS
PACKAGE_KEYS = {"schema_version","protocol","model","stage","n_questions","jobs","rewards","wrong_reward","pass_reward","prefix_ids","selection","selection_id","main_dataset_sha256","source_manifest_sha256","config_sha256","policy_lock_sha256","development_receipt_sha256"}
SELECTION_KEYS = {"selected_qids","qid_split","qid_group","qid_text_sha256","qid_category","source_job_ids","original_score_ids","prompt_sha256","outcomes_used_for_selection"}
STAGE_KEYS = {"schema_version","protocol","model","stage","n_questions","expected_rows","public_input_sha256","evaluator_sha256","source_manifest_sha256","config_sha256","policy_lock_sha256","development_receipt_sha256"}
SOURCE_INPUTS = {
    "protocol_public": ("factorized/inputs/protocol/public.json","3c84125a4891f276565f127436b1e2fa7aa3e30ffbab75c55549e742c145fdbe"),
    "transfer_public": ("transfer/run/pilot.json","4172b83e16f753f51442ff314d60ac2c1d75b12b7faa6574d5918319936ad2b9"),
    "development_dataset": ("factorized/inputs/frozen/main_dataset.json","d16d8e611965fba3829f01cda936145743b7d46187030e2138caf52068aa9b62"),
    "evaluation_public": ("frozen/fresh/public/pilot.json","250d3fd3aff11cd9566e470eb79da057f2444ce2429654e8b27034760dddc430"),
    "evaluation_dataset": ("frozen/fresh/evaluator/main_dataset.json","bb8aa81b67f41a280df809c10f74f016d998b56debd32f71eb592e22bd2d32e0"),
    "original_policy_lock": ("frozen/fresh/policy_lock.json","68445eb706531bfbf78dd4ff18f88d260c6391de3b38e29dd469ae5d6b3fe58a"),
}


def file_hash(path):
    """Compute a streaming SHA256 identity."""
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def json_bytes(value):
    """Canonical transport bytes, also used for package and lock hashes."""
    return base.canonical(value)


def value_hash(value): return hashlib.sha256(json_bytes(value)).hexdigest()


def write_new(path,value):
    """Create-only persistence; never silently overwrite an existing identity."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as handle: handle.write(json_bytes(value))


def _hash(value): return isinstance(value,str) and re.fullmatch(r"[a-f0-9]{64}",value) is not None


def config_path(): return Path(__file__).resolve().parents[1]/"configs/imcqa_mistral_replication.json"


def read_config():
    config=json.loads(config_path().read_text())
    if config["model"] != MODEL or config["protocol"] != PROTOCOL or config["roles"] != {"calibration":20,"selection":120,"test":850}:
        raise ValueError("replication configuration identity differs")
    if config["calibration"] != {"feature_clip":[1e-6,1-1e-6],"l2":.01,"maxiter":1000,"ftol":1e-12,"gtol":1e-8}:
        raise ValueError("original calibration settings changed")
    if config["policy_selection"]["threshold_grid"] != [i/20 for i in range(21)] or config["policy_selection"]["fixed_rounds"] != list(range(1,6)):
        raise ValueError("original policy search space changed")
    return config


def validate_public_package(package, *, allow_unlocked_evaluation=False):
    """Reject gold fields, changed prompt semantics and incomplete stage grids."""
    if not isinstance(package,dict) or set(package)!=PACKAGE_KEYS: raise ValueError("unexpected public package fields")
    stage=package["stage"]; n=package["n_questions"]
    if (package["schema_version"]!=SCHEMA or package["protocol"]!=PROTOCOL or package["model"]!=MODEL
        or stage not in ("development","evaluation") or type(n) is not int or n!={"development":140,"evaluation":850}[stage]
        or package["rewards"]!=list(REWARDS) or package["wrong_reward"]!=-1 or package["pass_reward"]!=0 or package["prefix_ids"]!=list(PREFIX_IDS)):
        raise ValueError("stage/model/scientific identity differs")
    for k in ("main_dataset_sha256","source_manifest_sha256","config_sha256","selection_id"):
        if not _hash(package[k]): raise ValueError("missing source binding")
    if package["config_sha256"]!=file_hash(config_path()): raise ValueError("configuration binding differs")
    if stage=="development":
        if package["policy_lock_sha256"] is not None or package["development_receipt_sha256"] is not None: raise ValueError("development cannot carry selected policy")
    elif not (allow_unlocked_evaluation and package["policy_lock_sha256"] is None and package["development_receipt_sha256"] is None):
        if not _hash(package["policy_lock_sha256"]) or not _hash(package["development_receipt_sha256"]): raise ValueError("evaluation requires frozen model-specific policy and development receipt")
    s=package["selection"]
    if not isinstance(s,dict) or set(s)!=SELECTION_KEYS or value_hash(s)!=package["selection_id"] or s["outcomes_used_for_selection"] is not False:
        raise ValueError("selection identity differs")
    qids=s["selected_qids"]
    if not isinstance(qids,list) or len(qids)!=n or len(set(qids))!=n: raise ValueError("question identities differ")
    for k in ("qid_split","qid_group","qid_text_sha256","qid_category"):
        if not isinstance(s[k],dict) or set(s[k])!=set(qids): raise ValueError("question role metadata differs")
    if len(set(s["qid_group"].values()))!=n or len(set(s["qid_text_sha256"].values()))!=n: raise ValueError("repeated question group or text")
    counts={split:sum(v==split for v in s["qid_split"].values()) for split in set(s["qid_split"].values())}
    if counts!=({"calibration":20,"selection":120} if stage=="development" else {"test":850}): raise ValueError("stage split counts differ")
    jobs=package["jobs"]
    if not isinstance(jobs,list) or len(jobs)!=40*n: raise ValueError("incomplete stage contexts")
    expected=[(q,c,p,r) for q in qids for c in CONDITIONS for p in PREFIX_IDS for r in range(4)]
    menus={};prefixes={};identities={};seen=set()
    for i,(j,key) in enumerate(zip(jobs,expected)):
        if not isinstance(j,dict) or set(j)!=PUBLIC_JOB_KEYS or j["score_index"]!=i: raise ValueError("public job fields/index differ")
        if (j["qid"],j["condition"],j["prefix_id"],j["rotation"])!=key or j["score_id"] in seen: raise ValueError("job coverage/order differs")
        seen.add(j["score_id"])
        if (j["split"]!=s["qid_split"][j["qid"]] or j["group_id"]!=s["qid_group"][j["qid"]]
            or j["arm"]!="plain" or j["block"]!="factorial" or j["wait_label"]!="E" or j["execution"]!="new"
            or j["synthetic_case"] is not None or j["allowed_actions"]!="ABCD"
            or type(j["round"]) is not int or j["round"]!=PREFIX_IDS.index(j["prefix_id"])+1 or j["reward"]!=REWARDS[j["round"]-1]
            or type(j["rotation"]) is not int or j["option_source_ids"]!=old.mapping(j["rotation"])
            or type(j["fraction"]) not in (int,float) or not math.isfinite(j["fraction"]) or not 0<j["fraction"]<=1
            or not _hash(j["source_job_id"]) or not _hash(j["source_prompt_sha256"])
            or j["score_id"]!=j["source_job_id"]+f':mistral:plain:r{j["rotation"]}'
            or hashlib.sha256(j["prompt"].encode()).hexdigest()!=j["prompt_sha256"]
            or s["source_job_ids"].get(j["score_id"])!=j["source_job_id"]
            or s["original_score_ids"].get(j["score_id"])!=j["source_score_id"]
            or s["prompt_sha256"].get(j["score_id"])!=j["prompt_sha256"]): raise ValueError("job provenance/scoring semantics differ")
        payload=old._payload(j["prompt"]);old._validate_displayed_options(payload["options"],"E")
        opts={j["option_source_ids"][o["id"]]:o["text"] for o in payload["options"]}
        mk=(j["qid"],j["condition"]);pk=(j["qid"],j["prefix_id"])
        if menus.setdefault(mk,opts)!=opts or prefixes.setdefault(pk,payload["question_prefix"])!=payload["question_prefix"]: raise ValueError("menu/prefix altered across factors")
        sk=(j["qid"],j["condition"],j["prefix_id"]);ident=tuple(j[k] for k in ("source_job_id","source_prompt_sha256","group_id","menu_id","fraction"))
        if identities.setdefault(sk,ident)!=ident: raise ValueError("source identity altered across rotations")
        rebuilt=old._real_job(j,{"question_prefix":payload["question_prefix"],"options":[{"id":k,"text":opts[k]} for k in "ABCD"]},"plain",j["rotation"])
        if rebuilt["prompt"]!=j["prompt"]: raise ValueError("original plain prompt semantics changed")
    for name in ("source_job_ids","original_score_ids","prompt_sha256"):
        if set(s[name])!=seen: raise ValueError("source mapping coverage differs")
    for q in qids:
        previous=""
        for prefix in PREFIX_IDS:
            now=prefixes[q,prefix]
            if not now.startswith(previous): raise ValueError("prefixes are not cumulative")
            previous=now
        if base.sha(base.canonical(tuned.normalized_tokens(previous)))!=s["qid_text_sha256"][q]: raise ValueError("full text binding differs")
    return jobs


def validate_role_separation(questions):
    """Check cross-role identity, exact text and 0.8 five-gram near duplicates."""
    if len({q["qid"] for q in questions})!=len(questions) or len({q["group_id"] for q in questions})!=len(questions): raise ValueError("question/group leakage")
    tokens={q["qid"]:tuned.normalized_tokens(next(p["text"] for p in q["prefixes"] if p["prefix_id"]=="p10")) for q in questions}
    if len(set(tokens.values()))!=len(questions): raise ValueError("exact normalized text leakage")
    shingles={q["qid"]:tuned.shingles(" ".join(tokens[q["qid"]])) for q in questions}
    pairs=0
    for i,left in enumerate(questions):
        for right in questions[i+1:]:
            if left["split"]==right["split"]: continue
            pairs+=1
            a,b=shingles[left["qid"]],shingles[right["qid"]]
            if min(len(a),len(b))*5<max(len(a),len(b))*4: continue
            if tuned.near_duplicate(a,b): raise ValueError("cross-role near-duplicate text: "+left["qid"]+" / "+right["qid"])
    return {"question_overlap":0,"group_overlap":0,"normalized_text_overlap":0,"cross_role_near_duplicate_pairs":0,"cross_role_text_pairs_checked":pairs,"jaccard_cutoff":.8}


def stage_manifest(package):
    """Produce a minimal, gold-free transport and launch binding."""
    validate_public_package(package)
    return {"schema_version":"imcqa-mistral-stage-manifest-v1","protocol":PROTOCOL,"model":MODEL,"stage":package["stage"],"n_questions":package["n_questions"],"expected_rows":len(package["jobs"]),"public_input_sha256":value_hash(package),"evaluator_sha256":package["main_dataset_sha256"],**{k:package[k] for k in ("source_manifest_sha256","config_sha256","policy_lock_sha256","development_receipt_sha256")}}


def validate_stage_manifest(manifest, package, *, policy_lock=None, development_receipt=None):
    """Validate transport hashes and complete development-run prerequisites.

    The CPU evaluation launcher must additionally call
    ``select_imcqa_mistral_policies.validate_policy_lock`` to validate coefficient
    values, role identities and selected policy choices. This public-only helper
    intentionally does not import the CPU-only fitting/selection implementation.
    """
    if set(manifest)!=STAGE_KEYS or manifest!=stage_manifest(package): raise ValueError("stage manifest differs from exact public package")
    if package["stage"]=="evaluation":
        if not isinstance(policy_lock,dict) or not isinstance(development_receipt,dict): raise ValueError("evaluation launch requires policy lock and development receipt")
        if value_hash(policy_lock)!=package["policy_lock_sha256"] or value_hash(development_receipt)!=package["development_receipt_sha256"]: raise ValueError("evaluation prerequisite binding differs")
        if policy_lock.get("development_receipt_sha256")!=package["development_receipt_sha256"] or policy_lock.get("status")!="frozen_for_evaluation" or policy_lock.get("model")!=MODEL: raise ValueError("policy lock provenance differs")
        required={"status":"complete","stage":"development","model_tag":MODEL["tag"],"model":MODEL["name"],"revision":MODEL["revision"],"expected_rows":5600,"completed_rows":5600,"total_contexts":5600,"automatic_retries":0,"reused_rows":0,"generation":False,"sampling":False}
        if any(development_receipt.get(k)!=v for k,v in required.items()): raise ValueError("evaluation requires complete model-specific, no-retry development execution")
        if (development_receipt.get("public_input_sha256")!=policy_lock.get("development_public_sha256")
            or development_receipt.get("scores_sha256")!=policy_lock.get("development_scores_sha256")
            or policy_lock.get("protocol")!=PROTOCOL or policy_lock.get("config_sha256")!=package["config_sha256"]
            or policy_lock.get("source_manifest_sha256")!=package["source_manifest_sha256"]): raise ValueError("evaluation prerequisite run identity differs")
        audit=policy_lock.get("development_numeric_audit",{})
        if audit.get("passed") is not True or audit.get("validated_rows")!=5600 or audit.get("scores_sha256")!=development_receipt["scores_sha256"] or audit.get("public_input_sha256")!=development_receipt["public_input_sha256"]:
            raise ValueError("evaluation requires successful complete numerical audit")
    return True


def prepare(root,out):
    """Convert exact source prompts; this function never reads model scores."""
    root=Path(root);out=Path(out);read_config()
    values={};manifest={"schema_version":"imcqa-mistral-source-manifest-v1","model":MODEL,"source_inputs":{}}
    for name,(rel,expected) in SOURCE_INPUTS.items():
        path=root/rel
        if file_hash(path)!=expected: raise ValueError("original input hash differs: "+rel)
        values[name]=json.loads(path.read_text());manifest["source_inputs"][name]={"path":rel,"sha256":expected}
    original=values["original_policy_lock"]
    cal=set(original["calibration_qids"]);dev=set(original["selection_qids"])
    prior={q["qid"]:q for q in values["development_dataset"]["questions"]}
    evalq=values["evaluation_dataset"]["questions"]
    questions=[prior[q] for q in sorted(cal|dev)]+evalq
    roles={q["qid"]:q["split"] for q in questions}
    if {q for q,s in roles.items() if s=="calibration"}!=cal or {q for q,s in roles.items() if s=="selection"}!=dev: raise ValueError("original question split differs")
    manifest["role_separation"]=validate_role_separation(questions)
    manifest["roles"]={k:sorted(q for q,s in roles.items() if s==k) for k in ("calibration","selection","test")}
    if {k:len(v) for k,v in manifest["roles"].items()}!={"calibration":20,"selection":120,"test":850}: raise ValueError("role cardinality differs")
    sourcejobs={}
    for name in ("protocol_public","transfer_public","evaluation_public"):
        for job in values[name]["jobs"]:
            if job["qid"] not in roles or job["arm"]!="plain" or job["block"]!="factorial": continue
            key=(job["qid"],job["condition"],job["prefix_id"],job["rotation"])
            if key in sourcejobs: raise ValueError("duplicate original source state")
            if set(job)!=PUBLIC_JOB_KEYS or base.sha(job["prompt"].encode())!=job["prompt_sha256"]: raise ValueError("original prompt fields/hash differ")
            sourcejobs[key]=job
    byq={q["qid"]:q for q in questions}
    for stage,splits in (("development",{"calibration","selection"}),("evaluation",{"test"})):
        qids=sorted(q for q in roles if roles[q] in splits)
        evaluator={"schema_version":"imcqa-mistral-evaluator-v1","stage":stage,"questions":[byq[q] for q in qids]}
        jobs=[];sel={"selected_qids":qids,"qid_split":{},"qid_group":{},"qid_text_sha256":{},"qid_category":{},"source_job_ids":{},"original_score_ids":{},"prompt_sha256":{},"outcomes_used_for_selection":False}
        for q in qids:
            question=byq[q];sel["qid_split"][q]=roles[q];sel["qid_group"][q]=question["group_id"]
            sel["qid_category"][q]=question.get("source",{}).get("category","unknown")
            text=next(p["text"] for p in question["prefixes"] if p["prefix_id"]=="p10")
            sel["qid_text_sha256"][q]=base.sha(base.canonical(tuned.normalized_tokens(text)))
            for c in CONDITIONS:
                for p in PREFIX_IDS:
                    for r in range(4):
                        job=sourcejobs[q,c,p,r]
                        converted={**job,"score_index":len(jobs),"score_id":job["source_job_id"]+f":mistral:plain:r{r}","source_score_id":job["score_id"],"execution":"new"}
                        jobs.append(converted)
                        for field,source in (("source_job_ids","source_job_id"),("original_score_ids","source_score_id"),("prompt_sha256","prompt_sha256")): sel[field][converted["score_id"]]=converted[source]
        package={"schema_version":SCHEMA,"protocol":PROTOCOL,"model":MODEL,"stage":stage,"n_questions":len(qids),"jobs":jobs,"rewards":list(REWARDS),"wrong_reward":-1,"pass_reward":0,"prefix_ids":list(PREFIX_IDS),"selection":sel,"selection_id":value_hash(sel),"main_dataset_sha256":value_hash(evaluator),"source_manifest_sha256":value_hash(manifest),"config_sha256":file_hash(config_path()),"policy_lock_sha256":None,"development_receipt_sha256":None}
        validate_public_package(package,allow_unlocked_evaluation=True)
        write_new(out/f"{stage}_evaluator.json",evaluator)
        write_new(out/("development_public.json" if stage=="development" else "evaluation_public_unlocked.json"),package)
        if stage=="development":write_new(out/"development_stage_manifest.json",stage_manifest(package))
    write_new(out/"source_manifest.json",manifest)
    write_new(out/"public_bindings.json",{"schema_version":"imcqa-mistral-public-bindings-v1","model":MODEL,"source_manifest":manifest,
        "stages":{stage:{"public_sha256":file_hash(out/("development_public.json" if stage=="development" else "evaluation_public_unlocked.json")),"evaluator_sha256":file_hash(out/f"{stage}_evaluator.json")} for stage in ("development","evaluation")}})
    return manifest


def prepare_public_from_transports(transport_root,bindings_path,out):
    """Build inference packages from the three existing gold-free GitHub transports.

    No evaluator file or model score is opened. CPU-generated evaluator hashes
    arrive as opaque source bindings; the serialized package must exactly match
    the package previously produced and checked against local evaluator data.
    """
    read_config();transport_root=Path(transport_root);out=Path(out)
    bindings=json.loads(Path(bindings_path).read_text())
    if set(bindings)!={"schema_version","model","source_manifest","stages"} or bindings["schema_version"]!="imcqa-mistral-public-bindings-v1" or bindings["model"]!=MODEL: raise ValueError("public transport bindings differ")
    manifest=bindings["source_manifest"]
    if manifest["model"]!=MODEL or manifest["source_inputs"]!={k:{"path":v[0],"sha256":v[1]} for k,v in SOURCE_INPUTS.items()}: raise ValueError("original source bindings differ")
    sources={};categories={};roles={};groups={};texts={}
    transports={"protocol_public":"imcqa_protocol_public/public.json.gz","transfer_public":"imcqa_transfer_public/public.json.gz","evaluation_public":"imcqa_tuned_public/public.json.gz"}
    for key,relative in transports.items():
        raw=gzip.decompress((transport_root/relative).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=SOURCE_INPUTS[key][1]: raise ValueError("original public transport hash differs: "+relative)
        public=json.loads(raw)
        categories.update(public["selection"]["qid_category"])
        for j in public["jobs"]:
            if j["arm"]!="plain" or j["block"]!="factorial":continue
            sourcekey=(j["qid"],j["condition"],j["prefix_id"],j["rotation"])
            if sourcekey in sources:raise ValueError("duplicate public source state")
            sources[sourcekey]=j
            if roles.setdefault(j["qid"],j["split"])!=j["split"] or groups.setdefault(j["qid"],j["group_id"])!=j["group_id"]: raise ValueError("inconsistent source roles/groups")
            if j["prefix_id"]=="p10":
                text=old._payload(j["prompt"])["question_prefix"]
                if texts.setdefault(j["qid"],text)!=text:raise ValueError("full text differs across menus/rotations")
    observed_roles={k:sorted(q for q,s in roles.items() if s==k) for k in ("calibration","selection","test")}
    if observed_roles!=manifest["roles"]:raise ValueError("source question identities differ")
    pseudo=[{"qid":q,"split":roles[q],"group_id":groups[q],"prefixes":[{"prefix_id":"p10","text":texts[q]}]} for q in sorted(roles)]
    if validate_role_separation(pseudo)!=manifest["role_separation"]:raise ValueError("transport role/text audit differs")
    for stage,allowed in (("development",{"calibration","selection"}),("evaluation",{"test"})):
        qids=sorted(q for q in roles if roles[q] in allowed);jobs=[]
        sel={"selected_qids":qids,"qid_split":{q:roles[q] for q in qids},"qid_group":{q:groups[q] for q in qids},"qid_text_sha256":{q:base.sha(base.canonical(tuned.normalized_tokens(texts[q]))) for q in qids},"qid_category":{q:categories[q] for q in qids},"source_job_ids":{},"original_score_ids":{},"prompt_sha256":{},"outcomes_used_for_selection":False}
        for q in qids:
            for c in CONDITIONS:
                for p in PREFIX_IDS:
                    for r in range(4):
                        source=sources[q,c,p,r];j={**source,"score_index":len(jobs),"score_id":source["source_job_id"]+f":mistral:plain:r{r}","source_score_id":source["score_id"],"execution":"new"};jobs.append(j)
                        for name,field in (("source_job_ids","source_job_id"),("original_score_ids","source_score_id"),("prompt_sha256","prompt_sha256")):sel[name][j["score_id"]]=j[field]
        package={"schema_version":SCHEMA,"protocol":PROTOCOL,"model":MODEL,"stage":stage,"n_questions":len(qids),"jobs":jobs,"rewards":list(REWARDS),"wrong_reward":-1,"pass_reward":0,"prefix_ids":list(PREFIX_IDS),"selection":sel,"selection_id":value_hash(sel),"main_dataset_sha256":bindings["stages"][stage]["evaluator_sha256"],"source_manifest_sha256":value_hash(manifest),"config_sha256":file_hash(config_path()),"policy_lock_sha256":None,"development_receipt_sha256":None}
        validate_public_package(package,allow_unlocked_evaluation=True)
        if value_hash(package)!=bindings["stages"][stage]["public_sha256"]:raise ValueError("gold-free conversion differs from locally checked exact package")
        write_new(out/("development_public.json" if stage=="development" else "evaluation_public_unlocked.json"),package)
        if stage=="development":write_new(out/"development_stage_manifest.json",stage_manifest(package))
    write_new(out/"source_manifest.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True);group.add_argument("--root",type=Path);group.add_argument("--transport-root",type=Path);parser.add_argument("--public-bindings",type=Path);parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    if a.transport_root and not a.public_bindings:parser.error("--public-bindings required with --transport-root")
    m=prepare(a.root,a.out) if a.root else prepare_public_from_transports(a.transport_root,a.public_bindings,a.out)
    print(json.dumps({"status":"prepared","roles":{k:len(v) for k,v in m["roles"].items()},"out":str(a.out)}))


if __name__=="__main__":main()
