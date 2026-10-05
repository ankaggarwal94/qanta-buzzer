"""Policy semantics and paired uncertainty for the missing selective baseline."""
from copy import deepcopy
import numpy as np
import pytest

from scripts import analyze_imcqa_fixed_abstention as addition


def fit(condition="independent_pool", threshold=.6):
    return {"condition":condition,"intercept":0.,"slope":1.,"feature_clip":[1e-6,1-1e-6],
            "selected_threshold":threshold,"selected_fixed_policy":"fixed_1" if condition=="independent_pool" else "fixed_2"}


def trajectory(confidences, *, correct=None):
    correct=[True]*5 if correct is None else correct
    return [{"qid":"q0","group_id":"g0","condition":"independent_pool","arm":"plain","round":index+1,
        "rotation":0,"score_id":f"score{index}","candidate_choice":"A","candidate_correct":correct[index],
        "canonical_gold_option_id":"A" if correct[index] else "B",
        "canonical_answer_probabilities":{"A":value,"B":(1-value)/3,"C":(1-value)/3,"D":(1-value)/3}}
        for index,value in enumerate(confidences)]


def factorial():
    rows=[]
    for qi in range(3):
        for menu in addition.MENUS:
            for rotation in range(4):
                values=trajectory([.4,.5,.7,.8,.9],correct=[rotation!=0]*5)
                for row in values:
                    row.update(qid=f"q{qi}",group_id=f"g{qi}",condition=menu,rotation=rotation)
                rows.extend(values)
    return rows


def test_fixed_selective_does_not_take_later_crossing():
    values=trajectory([.4,.5,.8,.9,.9])
    rounds=addition.policy_rounds(values,fit())
    assert rounds=={"fixed_forced":1,"fixed_selective":None,"adaptive_forced":3,"adaptive_selective":3}


def test_forced_fallback_is_only_when_no_crossing():
    values=trajectory([.4]*5)
    assert addition.policy_rounds(values,fit())=={
        "fixed_forced":1,"fixed_selective":None,"adaptive_forced":5,"adaptive_selective":None}


def test_inclusive_threshold_and_no_added_ev_gate():
    # .75 at final reveal crosses frozen threshold, despite final EV below PASS.
    values=trajectory([.4,.4,.4,.4,.75])
    assert addition.policy_rounds(values,fit(threshold=.75))["adaptive_selective"]==5
    values[0]["canonical_answer_probabilities"]={"A":.75,"B":.1,"C":.1,"D":.05}
    assert addition.policy_rounds(values,fit(threshold=.75))["fixed_selective"]==1


def test_wrong_crossing_ends_episode_and_pass_observed_at_five():
    rows=factorial()
    episodes,states=addition.build_episodes(rows,{m:fit(m) for m in addition.MENUS})
    fs=[r for r in episodes if r["policy"]=="fixed_selective"]
    assert all(r["observed_round"]==5 and r["round"] is None and r["reward"]==0 for r in fs)
    adaptive=[r for r in episodes if r["policy"]=="adaptive_selective" and r["rotation"]==0]
    assert all(r["round"]==3 and r["wrong"] and r["reward"]==-1 for r in adaptive)
    assert len(states)==3*2*4*5


def test_missing_or_duplicate_rounds_fail_closed():
    with pytest.raises(ValueError,match="five ordered"):
        addition.policy_rounds(trajectory([.9]*5)[1:],fit())
    rows=factorial()
    with pytest.raises(ValueError,match="round"):
        addition.build_episodes(rows+[rows[0]],{m:fit(m) for m in addition.MENUS})


def test_conditional_error_bootstraps_pooled_ratio_not_mean_of_question_errors():
    # Each record is already a question-level average over four rotations.
    values=np.array([[0.,1.,.5,1.],[0.,.25,0.,4.],[0.,0.,0.,5.]])
    indices=np.array([[0,0,1],[0,1,2],[1,1,2],[2,2,2]])
    point,boot=addition.metric_samples(values,indices)
    assert point["conditional_error"]==pytest.approx(.5/1.25)
    assert boot["conditional_error"][:3]==pytest.approx([1/2.25,.5/1.25,0])
    assert np.isnan(boot["conditional_error"][3])


def test_four_rotations_are_question_unit_and_primary_family_two():
    records,_=addition.build_episodes(factorial(),{m:fit(m) for m in addition.MENUS})
    summary,per_question=addition.summarize(records,samples=1000,seed=1)
    assert len(per_question)==3*2*4
    assert len(summary["primary_contrasts"])==2
    assert all(r["family_size"]==2 for r in summary["primary_contrasts"])
    for result in summary["primary_contrasts"]:
        # all three questions have the same 4-order average reward at round3:
        # 3*.6 -1, divided4 =.2; fixed-selective alwaysPASS=0.
        assert result["mean_delta"]==pytest.approx(.2)
        assert result["ci97_5"]==pytest.approx([.2,.2])
    assert len(summary["simple_effects"])==10
    assert len(summary["metric_deltas"])==16


def test_zero_coverage_error_remains_undefined():
    records,_=addition.build_episodes(factorial(),{m:fit(m) for m in addition.MENUS})
    summary,_=addition.summarize(records,samples=100,seed=1)
    fs=[r for r in summary["policy_summaries"] if r["policy"]=="fixed_selective"]
    assert all(r["conditional_error"]["mean"] is None and r["conditional_error"]["ci95"] is None for r in fs)
    assert all(r["committed_count"]==0 and r["terminal_pass_count"]==12 for r in fs)


def test_analysis_does_not_mutate_or_refit_inputs():
    rows=factorial(); before=deepcopy(rows)
    fits={m:fit(m) for m in addition.MENUS}; oldfits=deepcopy(fits)
    left,_=addition.build_episodes(rows,fits)
    right,_=addition.build_episodes(rows,fits)
    assert left==right and rows==before and fits==oldfits
