"""Scientific edge cases for fixed-policy transfer, independent of inference."""
from copy import deepcopy
import math

import numpy as np
import pytest

from scripts import analyze_imcqa_transfer as analysis


def fit(condition="independent_pool", threshold=.6):
    return {"condition":condition, "intercept":0., "slope":1., "feature_clip":[1e-6,1-1e-6],
            "selected_threshold":threshold, "selected_fixed_policy":"fixed_1" if condition == "independent_pool" else "fixed_2"}


def trajectory(probabilities, *, arm="plain", actions=None, correct=None):
    actions = actions or ["A"]*5
    correct = correct or [True]*5
    return [{"qid":"q0", "group_id":"g0", "condition":"independent_pool", "arm":arm,
             "round":i+1, "rotation":0, "chosen_action":actions[i],
             "canonical_answer_probabilities":{"A":p,"B":(1-p)/3,"C":(1-p)/3,"D":(1-p)/3},
             "candidate_choice":"A", "candidate_correct":correct[i],
             "native_semantic_action":actions[i] if actions[i] != "E" else ("PASS" if i==4 else "WAIT")}
            for i,p in enumerate(probabilities)]


def factorial(n=3):
    rows=[]
    for qi in range(n):
        for menu in analysis.MENUS:
            for rotation in range(4):
                for arm in analysis.ARMS:
                    values=trajectory([.9]*5,arm=arm,actions=["E"]*5 if arm=="wait" else ["A"]*5)
                    for row in values:
                        row.update(qid=f"q{qi}",group_id=f"g{qi}",condition=menu,rotation=rotation)
                        # Rotation effects must be averaged inside each question.
                        row["candidate_correct"] = qi != 0 or rotation != 0
                    rows.extend(values)
    return rows


def test_frozen_threshold_has_no_added_myopic_positive_ev_gate():
    # A final-round .70 probability is below .833 required to beat PASS, but
    # the frozen threshold .60 still commits. Adding an EV gate changes policy.
    rows=trajectory([.4,.4,.4,.4,.7])
    assert analysis.stopping_round(rows,"frozen_plain_threshold",fit()) == 5


def test_frozen_threshold_inclusive_and_first_crossing():
    params=fit(threshold=.75)
    rows=trajectory([.4,.5,.75,.9,.9])
    assert analysis.stopping_round(rows,"frozen_plain_threshold",params) == 3
    params["selected_threshold"]=None
    assert analysis.stopping_round(rows,"frozen_plain_threshold",params) is None


def test_native_wait_terminates_on_first_answer_even_when_wrong():
    rows=trajectory([.9]*5,arm="wait",actions=["E","C","A","A","A"],correct=[True,False,True,True,True])
    assert analysis.stopping_round(rows,"native_wait",fit()) == 2
    with pytest.raises(ValueError,match="WAIT candidates"):
        analysis.stopping_round(trajectory([.9]*5),"native_wait",fit())


def test_terminal_pass_and_no_answer_risk_are_undefined():
    rows=factorial()
    result,records=analysis.evaluate(rows,{menu:fit(menu) for menu in analysis.MENUS},samples=200,seed=1)
    for policy in ("native_wait","always_pass"):
        values=[r for r in result["policy_summaries"] if r["policy"]==policy]
        assert all(r["mean_reward"]["mean"] == 0 for r in values)
        assert all(r["risk"]["mean"] is None and r["risk"]["ci95"] is None for r in values)
        assert all(r["risk"]["defined_resamples"] == 0 for r in values)
        assert all(r["mean_observed_round"]["mean"] == 5 for r in values)
    assert len(records["episodes"]) == 3*2*4*5


def test_frozen_fixed_rounds_differ_by_menu():
    rows=trajectory([.9]*5)
    assert analysis.stopping_round(rows,"frozen_plain_fixed",fit()) == 1
    assert analysis.stopping_round(rows,"frozen_plain_fixed",fit("same_category_pool")) == 2
    assert analysis.stopping_round(rows,"plain_final_round",fit()) == 5


def test_sigmoid_clipping_and_fixed_parameters():
    parameters={**fit(),"intercept":-.7,"slope":.45}
    expected=1/(1+math.exp(-(-.7+.45*math.log(.9/.1))))
    assert analysis.calibrated_probability(.9,parameters) == pytest.approx(expected,abs=1e-15)
    assert analysis.calibrated_probability(1.,parameters) == analysis.calibrated_probability(1-1e-6,parameters)


def test_question_bootstrap_averages_rotations_before_resampling():
    rows=factorial(3)
    summary,records=analysis.evaluate(rows,{m:fit(m) for m in analysis.MENUS},samples=5000,seed=1)
    cell=next(r for r in summary["primary_contrasts"] if r["condition"]=="independent_pool" and r["right"]=="native_wait")
    # q0 has 3 correct and1wrong first answers: .5; q1,q2 each1.0.
    values=np.array([.5,1.,1.])
    indices=np.random.default_rng(1).integers(0,3,(5000,3))
    expected=values[indices].mean(axis=1)
    assert cell["mean_delta"] == pytest.approx(values.mean())
    assert cell["ci95"] == pytest.approx(np.quantile(expected,[.025,.975]))
    assert cell["ci98_75"] == pytest.approx(np.quantile(expected,[.00625,.99375]))
    assert cell["bonferroni_family_size"] == 4
    assert len(summary["primary_contrasts"]) == 4
    own=next(r for r in records["per_question"] if r["qid"]=="q0" and r["condition"]=="independent_pool" and r["policy"]=="frozen_plain_threshold")
    assert own["reward"] == .5


def test_screening_does_not_claim_adaptive_superiority_on_tie_with_fixed():
    summary,_=analysis.evaluate(factorial(),{m:fit(m) for m in analysis.MENUS},samples=500,seed=1)
    screen=next(r for r in summary["screening"] if r["condition"]=="independent_pool")
    assert screen["continue_development_screen"] is True
    assert screen["beats_fixed_family_interval"] is False
    assert screen["evidence_beyond_frozen_fixed_baseline"] is False


def test_science_rejects_missing_rounds_and_rotations():
    rows=factorial()
    with pytest.raises(ValueError,match="missing or duplicate"):
        analysis.evaluate(rows[1:],{m:fit(m) for m in analysis.MENUS},samples=20)
    with pytest.raises(ValueError,match="five ordered"):
        analysis.stopping_round(trajectory([.9]*5)[1:],"frozen_plain_threshold",fit())


def test_results_are_reproducible_and_inputs_unchanged():
    rows=factorial(); original=deepcopy(rows)
    fits={m:fit(m) for m in analysis.MENUS}; frozen=deepcopy(fits)
    one=analysis.evaluate(rows,fits,samples=100)
    two=analysis.evaluate(rows,fits,samples=100)
    assert one == two
    assert fits == frozen and rows == original


def test_frozen_fit_rejects_training_selection_overlap():
    parameters=[]
    for condition in analysis.MENUS:
        parameters.append({**fit(condition),"answer_source":"plain","fit_split":"calibration",
                           "fit_qids":["q0"],"n_fit_questions":1,"selection_uses_only_calibration":True,
                           "method":"monotone_regularized_logistic_correctness"})
    fixture={"fit_split":"calibration","selection_outcomes_used_for_fitting":False,"parameters":parameters}
    with pytest.raises(ValueError,match="overlap"):
        analysis.frozen_parameters(fixture,["q0"])
    value=analysis.frozen_parameters(fixture,["q1"])
    assert value["independent_pool"]["selected_threshold"] == .6


def test_display_label_ties_restore_identity_before_grading():
    # Rotation1: displayedA denotes canonicalD. Lexicographic tie resolution
    # must choose displayedA, even when canonicalA would be graded correct.
    job={"option_source_ids":{"A":"D","B":"A","C":"B","D":"C"},
         "allowed_actions":"ABCD","wait_label":"E","round":1,"canonical_gold_option_id":"A"}
    row={"raw_action_logits":dict(zip("ABCDE",[0,0,0,0,100])),
         "action_probabilities":dict.fromkeys("ABCD",.25),"conditional_answer_probabilities":dict.fromkeys("ABCD",.25),
         "chosen_action":"A","tied_top_actions":list("ABCD")}
    viewed=analysis.prior.score_view(row,job)
    assert viewed["candidate_choice"] == "D" and viewed["candidate_correct"] is False


def test_score_softmax_inconsistency_fails_closed():
    job={"option_source_ids":dict(zip("ABCD","ABCD")),"allowed_actions":"ABCDE","wait_label":"E","round":5,"canonical_gold_option_id":"A"}
    row={"raw_action_logits":dict.fromkeys("ABCDE",0.),"action_probabilities":dict.fromkeys("ABCDE",.2),
         "conditional_answer_probabilities":dict.fromkeys("ABCD",.25),"chosen_action":"A","tied_top_actions":list("ABCDE")}
    analysis.prior.score_view(row,job)
    row["action_probabilities"]["A"] = .21
    with pytest.raises(ValueError,match="softmax"):
        analysis.prior.score_view(row,job)
