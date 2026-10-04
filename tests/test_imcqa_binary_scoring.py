"""Binary semantics, exact token boundaries, and independent causal cache checks."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from scripts import imcqa_binary_scoring as s


def test_tie_defers_under_both_opposite_encodings():
    for submit, defer in (("X", "Y"), ("Y", "X")):
        stats = s.action_statistics([2., 2.], submit, defer)
        assert stats["chosen_action"] == defer
        assert stats["chosen_semantic_action"] == "DEFER"
        assert stats["exact_tie"] and stats["tied_top_actions"] == ["X", "Y"]
        assert stats["semantic_probabilities"] == {"SUBMIT": .5, "DEFER": .5}


def test_reversing_tokens_reverses_semantics_without_changing_raw_distribution():
    left = s.action_statistics([1000., 999.], "X", "Y")
    right = s.action_statistics([1000., 999.], "Y", "X")
    assert left["action_probabilities"] == right["action_probabilities"]
    assert left["chosen_semantic_action"] == "SUBMIT"
    assert right["chosen_semantic_action"] == "DEFER"
    assert sum(left["action_probabilities"].values()) == pytest.approx(1.)


@pytest.mark.parametrize("logits,submit,defer", [([0],"X","Y"),([True,0],"X","Y"),
    ([math.nan,0],"X","Y"),([math.inf,0],"X","Y"),([0,0],"X","X"),([0,0],"A","B")])
def test_invalid_statistics_fail_closed(logits, submit, defer):
    with pytest.raises(ValueError):
        s.action_statistics(logits, submit, defer)


def test_numeric_guard_forbids_tiny_decision_changes_and_material_raw_drift():
    job = {"submit_label":"X", "defer_label":"Y"}
    a = [{"logits":[.00001, 0]}]
    with pytest.raises(ValueError, match="argmax"):
        s.numeric_agreement(a,[{"logits":[0,.00001]}],[job])
    with pytest.raises(ValueError, match="logits"):
        s.numeric_agreement(a,[{"logits":[.1,0]}],[job])
    assert s.numeric_agreement(a,a,[job])["argmax_changes"] == 0


def test_numeric_guard_catches_tie_to_submit_even_inside_tolerance():
    job = {"submit_label":"Y", "defer_label":"X"}
    with pytest.raises(ValueError, match="argmax"):
        s.numeric_agreement([{"logits":[0,0]}],[{"logits":[0,.000001]}],[job])


def test_exact_XY_boundary_proof_rejects_merged_action_token():
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs): return "chat:"
        def __call__(self,text,**kwargs): return {"input_ids":[ord(c) for c in text]}
        def decode(self,ids,**kwargs): return ''.join(chr(i) for i in ids)
    result = s.prepare_context(Tokenizer(), {"prompt":"question"})
    assert result["option_token_ids"] == {"X":88,"Y":89}
    class Broken(Tokenizer):
        def __call__(self,text,**kwargs):
            tokens = super().__call__(text,**kwargs)["input_ids"]
            if text.endswith("Y"): tokens[-2:] = [999]
            return {"input_ids":tokens}
    with pytest.raises(ValueError, match="one-token"):
        s.prepare_context(Broken(), {"prompt":"question"})


def test_cache_layout_preserves_binary_ids_and_exact_logical_positions():
    ids = {"X":19,"Y":7}
    contexts = [{"scored_input_token_ids":[1,2,3,8], "option_token_ids":ids},
                {"scored_input_token_ids":[1,2,3,7,9], "option_token_ids":ids}]
    layout = s.cached_batch_layout(contexts,pad_token_id=0,prefix_width=4,suffix_width=3,batch_size=4)
    assert layout["option_token_ids"] == [[19,7]]*4
    assert layout["prefix_input_ids"] == [[0,1,2,3]]*2
    assert layout["full_attention_mask"][:2] == [[0,1,1,1,0,0,1],[0,1,1,1,0,1,1]]
    assert layout["suffix_position_ids"][:2] == [[0,0,3],[0,3,4]]
    assert layout["suffix_cache_position"] == [4,5,6]
    with pytest.raises(ValueError):
        s.cached_batch_layout(contexts[:1],pad_token_id=0,prefix_width=4,suffix_width=3,batch_size=4)


def test_checkpoint_guard_recomputes_semantics_and_rejects_identity_reorder():
    from scripts import imcqa_binary_design as design
    jobs, _ = design.comprehension_cases()
    jobs = jobs[:2]
    rows = [{k:v for k,v in job.items() if k != "prompt"} | s.action_statistics([1,2],job["submit_label"],job["defer_label"]) for job in jobs]
    s.validate_rows(jobs,rows,complete=True)
    with pytest.raises(ValueError,match="order"):
        s.validate_rows(jobs,rows[::-1],complete=True)
    rows[0]["chosen_semantic_action"] = "INVALID"
    with pytest.raises(ValueError,match="checkpoint"):
        s.validate_rows(jobs,rows,complete=True)


def test_diagnostic_coverage_retains_every_factor_and_length_extremes():
    jobs=[]; contexts=[]
    for condition in s.CONDITIONS:
        for round_number in range(1,6):
            for mapping in ("submit_x","submit_y"):
                jobs.append({"condition":condition,"round":round_number,"mapping":mapping,"block":"real","synthetic_case":None})
                contexts.append({"scored_input_token_ids":[1]*len(jobs)})
    for i, case in enumerate(("known_correct","known_incorrect","uniform")):
        for mapping in ("submit_x","submit_y"):
            jobs.append({"condition":None,"round":1,"mapping":mapping,"block":"comprehension","synthetic_case":case})
            contexts.append({"scored_input_token_ids":[1]*len(jobs)})
    indices=s.diagnostic_indices(jobs,contexts)
    assert len(indices) <= 32 and len(indices)%2==0
    for key in ("condition","round","mapping","block","synthetic_case"):
        assert {jobs[i][key] for i in indices} == {job[key] for job in jobs}
    assert 0 in indices and len(jobs)-1 in indices


def test_forward_cache_equivalence_extracts_binary_logits_and_vocab_summary():
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
    ids={"X":19,"Y":7}
    contexts=[{"scored_input_token_ids":[1,2,3,8],"option_token_ids":ids},
              {"scored_input_token_ids":[1,2,3,7,9],"option_token_ids":ids},
              {"scored_input_token_ids":[1,2,3,4,5,8],"option_token_ids":ids},
              {"scored_input_token_ids":[1,2,3,4,5,7,9],"option_token_ids":ids}]
    tokenizer=SimpleNamespace(pad_token_id=0)
    native=s.forward(Torch,Model(),tokenizer,contexts,batch_size=6)
    cached=s.forward(Torch,Model(),tokenizer,contexts,cached=True,batch_size=6)
    assert native==cached and len(cached)==4
    assert all(len(r["logits"])==2 and r["unconstrained_top_token_id"]==23 for r in cached)
    for row in cached:
        assert 0 < row["legal_action_vocabulary_mass"] < 1
        assert row["legal_action_vocabulary_mass"]==pytest.approx(sum(math.exp(x-row["vocabulary_logsumexp"]) for x in row["logits"]))

    assert all(r["logits"][0] > r["logits"][1] for r in cached)


def test_full_synthetic_design_uses_family_coverage_within_single_batch():
    from scripts import imcqa_binary_design as design
    jobs, _ = design.comprehension_cases()
    contexts = [{"scored_input_token_ids": list(job["prompt"].encode()), "option_token_ids":{"X":88,"Y":89}} for job in jobs]
    indices = s.diagnostic_indices(jobs,contexts)
    assert len(indices) <= 32
    family = lambda job: job["synthetic_case"].split("-")[0]
    assert {family(jobs[i]) for i in indices} == {family(job) for job in jobs}
    assert {jobs[i]["proposal_id"] for i in indices} == set("ABCD")
