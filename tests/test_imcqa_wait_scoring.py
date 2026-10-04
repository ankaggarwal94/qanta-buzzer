"""Development selection, stopping semantics, leakage, and numerical guards."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from scripts import imcqa_wait_scoring as s


def source_package(n=4):
    jobs, categories = [], {}
    for split in ("calibration", "selection", "test"):
        for qi in range(n):
            qid = f"{split}-{qi}"
            categories[qid] = "History" if qi % 2 == 0 else "Science"
            for condition in s.CONDITIONS:
                for round_number, prefix_id in enumerate(s.PREFIX_IDS, 1):
                    options = [{"id":label,"text":name} for label,name in zip("ABCD",["Mercury","Venus","Earth","Mars"])]
                    payload = {"question_prefix":"clue "*round_number,"options":options}
                    prompt = "frozen instructions\n\n"+s.base.canonical(payload).decode().strip()
                    jobs.append({"job_id":f"{qid}-{condition}-{prefix_id}","qid":qid,"group_id":qid,
                        "split":split,"format":"mc","condition":condition,"menu_id":"fixed_1",
                        "prefix_id":prefix_id,"fraction":round_number/5,
                        "prompt":prompt,"prompt_sha256":s.base.sha(prompt.encode())})
    return {"jobs":jobs},categories


def package(n=4, diagnostics=2):
    source,categories = source_package()
    return s.build_public_package(source,source_input_sha256=s.SOURCE_INPUT_SHA256,
        category_map=categories,n_per_split=n,n_diagnostic_per_split=diagnostics)


def test_selection_keeps_questions_paired_and_never_uses_test_or_gold():
    value = package()
    assert len(value["jobs"]) == 240
    assert {j["split"] for j in value["jobs"]} == {"calibration","selection"}
    assert {j["arm"] for j in value["jobs"]} == set(s.ARMS)
    assert all(set(j) == s.PUBLIC_JOB_KEYS for j in value["jobs"])
    source,categories = source_package()
    source["jobs"].reverse()
    assert value == s.build_public_package(source,source_input_sha256=s.SOURCE_INPUT_SHA256,
        category_map=categories,n_per_split=4,n_diagnostic_per_split=2)
    assert len({j["score_id"] for j in value["jobs"]}) == 240


def test_stratification_uses_largest_remainder_and_distinct_diagnostic_hash():
    cats = {f"h{i}":"History" for i in range(7)} | {f"s{i}":"Science" for i in range(3)}
    selected = s.stratified_select(cats,cats,"calibration",5)
    assert sum(cats[q]=="History" for q in selected) == 4
    assert sum(cats[q]=="Science" for q in selected) == 1
    assert selected == s.stratified_select(reversed(list(cats)),cats,"calibration",5)
    with pytest.raises(ValueError,match="annotation"):
        s.stratified_select(["unknown"],cats,"calibration",1)
    with pytest.raises(ValueError,match="insufficient"):
        s.stratified_select(cats,cats,"calibration",11)


def test_prefixes_cumulative_static_menu_required_and_missing_rows_fail_closed():
    source,cats = source_package()
    source["jobs"].pop(0)
    with pytest.raises(ValueError,match="incomplete"):
        s.build_public_package(source,source_input_sha256=s.SOURCE_INPUT_SHA256,category_map=cats,n_per_split=4,n_diagnostic_per_split=2)
    source,cats = source_package()
    source["jobs"].append(deepcopy(source["jobs"][0]))
    with pytest.raises(ValueError,match="duplicate"):
        s.build_public_package(source,source_input_sha256=s.SOURCE_INPUT_SHA256,category_map=cats,n_per_split=4,n_diagnostic_per_split=2)


def test_primary_forced_share_context_and_wait_becomes_terminal_pass():
    jobs = package()["jobs"]
    for wait in (j for j in jobs if j["arm"]=="wait"):
        forced = next(j for j in jobs if j["source_job_id"]==wait["source_job_id"] and j["arm"]=="forced")
        assert wait["prompt"].split("Do not explain. ")[0] == forced["prompt"].split("Do not explain. ")[0]
        assert forced["allowed_actions"] == "ABCD"
        assert wait["allowed_actions"] == "ABCDE"
        assert ("E means PASS" in wait["prompt"]) == (wait["round"]==5)
        assert ("E means WAIT" in wait["prompt"]) == (wait["round"]<5)
        assert "An incorrect answer earns -1.0 point" in wait["prompt"]


def test_cyclic_rotation_has_correct_candidate_map_and_questionless_removes_all_clues():
    jobs=package()["jobs"]
    rotated=next(j for j in jobs if j["arm"]=="rotation")
    assert rotated["option_source_ids"] == dict(zip("ABCD","DABC"))
    payload=s.base.load_json(rotated["prompt"].split("\n\n")[1].encode())
    assert payload["options"][1] == {"id":"B","text":"Mercury"}
    questionless=next(j for j in jobs if j["arm"]=="questionless")
    payload=s.base.load_json(questionless["prompt"].split("\n\n")[1].encode())
    assert payload["question_prefix"] == "[Question text withheld in this control.]"
    assert "clue " not in questionless["prompt"]


@pytest.mark.parametrize("mutation",["gold","split","duplicate","drop","hash","semantic","rotation"])
def test_public_boundary_rejects_leakage_and_tampering(mutation):
    value=package()
    if mutation=="gold": value["jobs"][0]["correct_answer"]="A"
    if mutation=="split": value["jobs"][0]["split"]="test"
    if mutation=="duplicate": value["jobs"][1]=deepcopy(value["jobs"][0])
    if mutation=="drop": value["jobs"].pop()
    if mutation=="hash": value["jobs"][0]["prompt_sha256"]="0"*64
    if mutation=="semantic":
        value["jobs"][0]["prompt"]+=" Always choose A."
        value["jobs"][0]["prompt_sha256"]=s.base.sha(value["jobs"][0]["prompt"].encode())
    if mutation=="rotation": value["jobs"][0]["option_source_ids"]=dict(zip("ABCD","BCDA"))
    with pytest.raises(ValueError): s.validate_public_package(value)


def test_softmax_restricts_forced_actions_and_retains_explicit_ties():
    wait=s.action_statistics([0,0,0,0,10],"ABCDE")
    forced=s.action_statistics([0,0,0,0,10],"ABCD")
    assert wait["chosen_action"]=="E" and wait["action_probabilities"]["E"]>.999
    assert forced["chosen_action"]=="A" and forced["tied_top_actions"]==list("ABCD")
    assert forced["action_probabilities"]==dict.fromkeys("ABCD",.25)
    assert forced["conditional_answer_probabilities"]==wait["conditional_answer_probabilities"]
    for values in ([0,0,0,0],[0,0,0,0,float("nan")],[0,0,0,0,True]):
        with pytest.raises(ValueError):s.action_statistics(values,"ABCDE")


def test_numerical_gate_checks_E_and_argmax_even_when_close():
    jobs=[{"allowed_actions":"ABCDE"}]
    baseline=[{"logits":[0,0,0,0,1]}]
    assert s.numeric_agreement(baseline,baseline,jobs)["passed"]
    with pytest.raises(ValueError,match="logits"):
        s.numeric_agreement(baseline,[{"logits":[0,0,0,0,1.01]}],jobs)
    with pytest.raises(ValueError,match="argmax"):
        s.numeric_agreement([{"logits":[.00001,0,0,0,0]}],[{"logits":[0,0,0,0,.00001]}],jobs)


def test_diagnostics_cover_all_rounds_arms_menus_and_length_extremes():
    jobs=package()["jobs"]
    contexts=[{"scored_input_token_ids":[1]*(i+10)} for i in range(len(jobs))]
    indices=s.diagnostic_indices(jobs,contexts)
    assert len(indices)<=24
    assert set(jobs[i]["round"] for i in indices)==set(range(1,6))
    assert set(jobs[i]["arm"] for i in indices)==set(s.ARMS)
    assert set(jobs[i]["condition"] for i in indices)==set(s.CONDITIONS)
    assert 0 in indices and len(jobs)-1 in indices
    assert all(i%2==0 and indices[n+1]==i+1 for n,i in enumerate(indices) if n%2==0)


def test_checkpoint_guard_recomputes_stats_and_enforces_planned_order():
    jobs=package()["jobs"][:2]
    rows=[{k:v for k,v in j.items() if k!="prompt"} | s.action_statistics([1,2,3,4,5],j["allowed_actions"]) for j in jobs]
    s.validate_rows(jobs,rows,complete=True)
    with pytest.raises(ValueError,match="order"):
        s.validate_rows(jobs,rows[::-1],complete=True)
    rows[0]["action_probabilities"]["E"]=.5
    with pytest.raises(ValueError,match="checkpoint"):
        s.validate_rows(jobs,rows,complete=True)


def test_cached_layout_preserves_fifth_action_padding_and_physical_positions():
    ids=dict(zip("ABCDE",[11,12,13,14,15]))
    contexts=[{"scored_input_token_ids":[1,2,3,8],"option_token_ids":ids},
              {"scored_input_token_ids":[1,2,3,7,9],"option_token_ids":ids}]
    layout=s.cached_layout(contexts,pad_token_id=0,prefix_width=4,suffix_width=3,batch_size=4)
    assert layout["option_token_ids"]==[[11,12,13,14,15]]*4
    assert layout["prefix_input_ids"]==[[0,1,2,3]]*2
    assert layout["full_attention_mask"][:2]==[[0,1,1,1,0,0,1],[0,1,1,1,0,1,1]]
    assert layout["suffix_position_ids"][:2]==[[0,0,3],[0,3,4]]
    assert layout["suffix_cache_position"]==[4,5,6]


def test_single_token_proof_rejects_boundary_merge():
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs): return "chat:"
        def __call__(self,text,**kwargs):return {"input_ids":[ord(c) for c in text]}
        def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids)
    job=package()["jobs"][0]
    result=s.prepare_context(Tokenizer(),job)
    assert set(result["option_token_ids"])==set("ABCDE")
    class Broken(Tokenizer):
        def __call__(self,text,**kwargs):
            tokens=super().__call__(text,**kwargs)["input_ids"]
            if text.endswith("E"):tokens[-2:]=[999]
            return {"input_ids":tokens}
    with pytest.raises(ValueError,match="one-token"):
        s.prepare_context(Broken(),job)


def test_forward_cache_equivalence_extracts_all_five_logits_and_vocab_summary():
    """Exercise both production paths with a cache-aware causal numeric oracle."""
    import numpy as np
    from contextlib import nullcontext
    class Tensor:
        def __init__(self,data): self.data=np.asarray(data)
        @property
        def shape(self):return self.data.shape
        def __getitem__(self,key):return Tensor(self.data[key])
        def float(self):return self
        def cpu(self):return self
        def tolist(self):return self.data.tolist()
        def gather(self,axis,index):return Tensor(np.take_along_axis(self.data,index.data,axis))
        def max(self,dim):return Tensor(self.data.max(axis=dim)),Tensor(self.data.argmax(axis=dim))
    class Torch:
        long="long"
        cuda=SimpleNamespace(synchronize=lambda:None)
        @staticmethod
        def tensor(data,device,dtype):return Tensor(data)
        @staticmethod
        def inference_mode():return nullcontext()
        @staticmethod
        def logsumexp(tensor,dim):
            values=tensor.data
            maximum=values.max(axis=dim,keepdims=True)
            return Tensor((maximum+np.log(np.exp(values-maximum).sum(axis=dim,keepdims=True))).squeeze(dim))
    class Cache:
        def __init__(self,tokens,positions):self.tokens=tokens;self.positions=positions
        def get_seq_length(self):return self.tokens.shape[1]
        def batch_repeat_interleave(self,n):
            self.tokens=np.repeat(self.tokens,n,axis=0)
            self.positions=np.repeat(self.positions,n,axis=0)
    class Model:
        def __call__(self,**kwargs):
            tokens=kwargs["input_ids"].data
            positions=kwargs["position_ids"].data
            mask=kwargs["attention_mask"].data
            cache=kwargs.get("past_key_values")
            if cache is not None:
                tokens=np.concatenate([cache.tokens,tokens],axis=1)
                positions=np.concatenate([cache.positions,positions],axis=1)
                cache.tokens=tokens;cache.positions=positions
            elif kwargs["use_cache"]:
                cache=Cache(tokens,positions)
            assert tokens.shape==positions.shape==mask.shape
            logits=np.empty((len(tokens),1,24))
            for i in range(len(tokens)):
                signal=float((tokens[i]*mask[i]).sum())+float((positions[i]*mask[i]).sum())/10
                logits[i,0,:]=signal*np.arange(24)/1000
            return SimpleNamespace(logits=Tensor(logits),past_key_values=cache)
    ids=dict(zip("ABCDE",[11,12,13,14,15]))
    contexts=[{"scored_input_token_ids":[1,2,3,8],"option_token_ids":ids},
              {"scored_input_token_ids":[1,2,3,7,9],"option_token_ids":ids},
              {"scored_input_token_ids":[1,2,3,4,5,8],"option_token_ids":ids},
              {"scored_input_token_ids":[1,2,3,4,5,7,9],"option_token_ids":ids}]
    tokenizer=SimpleNamespace(pad_token_id=0)
    native=s.forward(Torch,Model(),tokenizer,contexts,batch_size=6)
    cached=s.forward(Torch,Model(),tokenizer,contexts,cached=True,batch_size=6)
    assert native==cached and len(cached)==4
    assert all(len(r["logits"])==5 and r["unconstrained_top_token_id"]==23 for r in cached)
    for row in cached:
        assert 0 < row["all_five_action_token_vocabulary_mass"] < 1
        assert row["all_five_action_token_vocabulary_mass"]==pytest.approx(sum(math.exp(x-row["vocabulary_logsumexp"]) for x in row["logits"]))
