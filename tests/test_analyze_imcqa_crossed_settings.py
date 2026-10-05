"""Crossed threshold/round analysis preserves pairing and original parameters."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import analyze_imcqa_crossed_settings as crossed


def fits():
    return {menu:{"condition":menu,"intercept":0.,"slope":1.,"feature_clip":[1e-6,1-1e-6],
        "selected_threshold":.6 if menu==crossed.MENUS[0] else .85,
        "selected_fixed_policy":"fixed_1" if menu==crossed.MENUS[0] else "fixed_2"}
        for menu in crossed.MENUS}


def views(n=3):
    rows=[]
    for qi in range(n):
        for menu in crossed.MENUS:
            for rotation in range(4):
                for ri,p in enumerate((.55,.7,.9,.95,.95)):
                    correct=(qi+rotation+ri+(menu==crossed.MENUS[1]))%3!=0
                    rows.append({"qid":f"q{qi}","group_id":f"g{qi}","condition":menu,"arm":"plain",
                        "round":ri+1,"rotation":rotation,"score_id":f"{qi}-{menu}-{rotation}-{ri}",
                        "candidate_choice":"A","candidate_correct":correct,"canonical_gold_option_id":"A" if correct else "B",
                        "canonical_answer_probabilities":{"A":p,"B":(1-p)/3,"C":(1-p)/3,"D":(1-p)/3}})
    return rows


def test_crossed_parameters_preserve_own_calibrator_and_all_original_inputs():
    parameters=fits();parameters[crossed.MENUS[1]]["intercept"]=-.3
    before=deepcopy(parameters); data=views(); beforedata=deepcopy(data)
    episodes,states,used=crossed.build_crossed_episodes(data,parameters,crossed.SETTINGS)
    assert parameters==before and data==beforedata
    assert len(episodes)==3*2*4*4*4 and len(states)==3*2*4*5*4
    assert len(used)==8
    for row in used:
        assert row["parameters"]["intercept"]==before[row["condition"]]["intercept"]
        assert row["parameters"]["slope"]==before[row["condition"]]["slope"]
        assert row["parameters"]["selected_threshold"]==row["threshold"]


def test_expected_invariants_and_original_policy_reproduction(tmp_path):
    parameters=fits();data=views()
    episodes,_,_=crossed.build_crossed_episodes(data,parameters,crossed.SETTINGS)
    checked=crossed.check_invariants(episodes)
    assert checked["passed"] is True
    original,_=crossed.base.build_episodes(data,parameters)
    path=tmp_path/"old.csv"
    crossed.base.transfer.write_csv(path,original)
    assert crossed.match_original(episodes,path)["exact_episode_matches"]==len(original)


@pytest.mark.parametrize("policy",["adaptive_forced","adaptive_selective","fixed_forced"])
def test_irrelevant_setting_change_cannot_change_outcomes(policy):
    episodes,_,_=crossed.build_crossed_episodes(views(),fits(),crossed.SETTINGS)
    target=next(row for row in episodes if row["policy"]==policy)
    target["reward"]+=.2
    with pytest.raises(ValueError,match="invariant"):
        crossed.check_invariants(episodes)


def test_original_outcome_mutation_is_rejected(tmp_path):
    data=views();parameters=fits()
    original,_=crossed.base.build_episodes(data,parameters)
    path=tmp_path/"old.csv";crossed.base.transfer.write_csv(path,original)
    episodes,_,_=crossed.build_crossed_episodes(data,parameters,crossed.SETTINGS)
    target=next(row for row in episodes if row["condition"]==crossed.MENUS[0] and row["setting_id"]=="t060_r1")
    target["committed"]=not target["committed"]
    with pytest.raises(ValueError,match="original episode"):
        crossed.match_original(episodes,path)


def test_one_paired_question_bootstrap_and_eight_contrast_adjustment():
    episodes,_,_=crossed.build_crossed_episodes(views(),fits(),crossed.SETTINGS)
    summary,perquestion=crossed.summarize(episodes,crossed.SETTINGS,samples=1000,seed=1)
    assert len(summary["policy_summaries"])==32
    assert len(summary["primary_contrasts"])==8
    assert len(summary["menu_gain_differences"])==4
    assert len(summary["menu_policy_differences"])==16
    indices=np.random.default_rng(1).integers(0,3,(1000,3))
    def values(menu,policy,setting):
        return np.asarray([r["reward"] for r in perquestion if r["condition"]==menu and r["policy"]==policy and r["setting_id"]==setting])
    for cell in summary["primary_contrasts"]:
        delta=values(cell["condition"],"adaptive_selective",cell["setting_id"])-values(cell["condition"],"fixed_selective",cell["setting_id"])
        boot=delta[indices].mean(axis=1)
        assert cell["family_size"]==8
        assert cell["n_questions"]==3
        assert cell["mean_delta"]==pytest.approx(delta.mean())
        assert cell["ci99_375"]==pytest.approx(np.quantile(boot,[.003125,.996875]))
    for cell in summary["menu_gain_differences"]:
        gains=[]
        for menu in crossed.MENUS:
            gains.append(values(menu,"adaptive_selective",cell["setting_id"])-values(menu,"fixed_selective",cell["setting_id"]))
        delta=gains[1]-gains[0]
        assert cell["ci95"]==pytest.approx(np.quantile(delta[indices].mean(axis=1),[.025,.975]))


def test_duplicate_or_missing_rotation_does_not_inflate_sample_size():
    episodes,_,_=crossed.build_crossed_episodes(views(),fits(),crossed.SETTINGS)
    with pytest.raises(ValueError,match="four distinct rotations"):
        crossed.summarize(episodes+[episodes[0]],crossed.SETTINGS,samples=10)
    with pytest.raises(ValueError,match="four distinct rotations"):
        crossed.summarize(episodes[1:],crossed.SETTINGS,samples=10)


def test_fixed_forced_statistics_repeat_across_thresholds():
    episodes,_,_=crossed.build_crossed_episodes(views(),fits(),crossed.SETTINGS)
    summary,_=crossed.summarize(episodes,crossed.SETTINGS,samples=100,seed=1)
    lookup={(row["condition"],row["setting_id"],row["policy"]):row for row in summary["policy_summaries"]}
    for menu in crossed.MENUS:
        for fixed in (1,2):
            left,right=(lookup[menu,f"t{threshold}_r{fixed}","fixed_forced"] for threshold in ("060","085"))
            for metric in crossed.base.METRICS:
                assert left[metric]==right[metric]
