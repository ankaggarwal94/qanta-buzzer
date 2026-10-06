"""Focused semantics and leakage checks for the independently tuned policies."""
from copy import deepcopy
import math
from statistics import NormalDist

import numpy as np
import pytest

from scripts import select_imcqa_tuned_policies as selector


def test_complete_independent_candidate_grids():
    adaptive = selector.candidates("adaptive_selective")
    fixed = selector.candidates("fixed_selective")
    assert len(adaptive) == 22 and len(fixed) == 106
    assert {(r["fixed_round"], r["threshold"]) for r in fixed if not r["always_pass"]} == {
        (rnd, tau) for rnd in range(1, 6) for tau in selector.GRID}
    assert sum(r["always_pass"] for r in adaptive) == 1


def test_inclusive_crossing_fixed_rejection_and_nonmonotone_confidence():
    p = np.tile([.4, .8, .3, .9, .95], (2, 4, 1))
    correct = np.zeros_like(p, dtype=bool)
    correct[:, :, 1] = True
    base = {"always_pass": False, "threshold": .8}
    adaptive = selector.evaluate_candidate(p, correct, {**base, "family": "adaptive_selective", "fixed_round": None})
    fixed = selector.evaluate_candidate(p, correct, {**base, "family": "fixed_selective", "fixed_round": 3})
    assert np.all(adaptive["round"] == 2)
    assert np.all(adaptive["reward"] == .8)
    # Fixed refusal does not get another chance when confidence rises at round4.
    assert not fixed["committed"].any()
    assert not fixed["reward"].any()


def test_no_crossing_passes_and_wrong_answer_cost_is_not_round_discounted():
    p = np.full((1, 4, 5), .7)
    correct = np.zeros_like(p, dtype=bool)
    passed = selector.evaluate_candidate(p, correct, {"family": "adaptive_selective", "always_pass": False, "threshold": .8})
    wrong = selector.evaluate_candidate(p, correct, {"family": "fixed_selective", "always_pass": False, "threshold": .7, "fixed_round": 5})
    assert not passed["committed"].any() and np.all(passed["round"] == 0)
    assert np.all(wrong["reward"] == -1) and wrong["wrong"].all()


def candidate(name, *, reward=0., coverage=0., threshold=.5, rnd=1, passed=False):
    return {"candidate_id": name, "mean_reward": reward, "coverage": coverage,
        "threshold": None if passed else threshold, "fixed_round": rnd, "always_pass": passed}


def test_conservative_ties_are_order_independent():
    rows = [candidate("answer", reward=5e-13, coverage=1.), candidate("pass", passed=True)]
    assert selector.choose_candidate(rows)["candidate_id"] == "pass"
    rows = [candidate("highcoverage", reward=.3, coverage=.9, threshold=.95),
            candidate("lowthreshold", reward=.3, coverage=.5, threshold=.6),
            candidate("laterr", reward=.3, coverage=.5, threshold=.7, rnd=3),
            candidate("earlierr", reward=.3, coverage=.5, threshold=.7, rnd=2)]
    assert selector.choose_candidate(rows)["candidate_id"] == "earlierr"
    assert selector.choose_candidate(rows[::-1])["candidate_id"] == "earlierr"


def role_fixture():
    fit_qids = [f"cal{i}" for i in range(20)]
    questions = {q: {"group_id": "g:" + q} for q in fit_qids + ["dev0", "dev1"]}
    fits = {m: {"fit_qids": fit_qids} for m in selector.MENUS}
    states = [{"qid": q, "group_id": questions[q]["group_id"], "condition": m,
               "rotation": rot, "round": rnd, "cohort": "development"}
              for q in ["dev0", "dev1"] for m in selector.MENUS for rot in range(4) for rnd in range(1, 6)]
    return states, fits, questions


def test_roles_reject_group_leakage_and_missing_or_duplicate_states():
    states, fits, questions = role_fixture()
    assert selector.validate_roles(states, fits, questions)["unique_development_groups"] == 2
    with pytest.raises(ValueError, match="incomplete"):
        selector.validate_roles(states[:-1], fits, questions)
    with pytest.raises(ValueError, match="duplicate"):
        selector.validate_roles(states + [states[0]], fits, questions)
    leaked = deepcopy(questions)
    leaked["dev0"]["group_id"] = leaked["cal0"]["group_id"]
    with pytest.raises(ValueError, match="calibration group"):
        selector.validate_roles(states, fits, leaked)


def test_planning_uses_selected_pair_not_largest_grid_variance():
    n = 20
    delta = np.array([-.1, .1] * 10)
    bad = np.array([-1., 1.] * 10)
    values = {}
    selected = {}
    for menu in selector.MENUS:
        values[menu, "adaptive_selective"] = np.stack([delta, bad], axis=1)
        values[menu, "fixed_selective"] = np.zeros((n, 1))
        selected[menu, "adaptive_selective"] = {"candidate_id": "adaptive_selective:000"}
        selected[menu, "fixed_selective"] = {"candidate_id": "fixed_selective:000"}
    actual = selector.plan_sample_size(values, selected, samples=400, seed=1)
    indices = np.random.default_rng(1).integers(0, n, (400, n))
    expected_sd = float(np.quantile(delta[indices].std(axis=1, ddof=1), .95))
    assert actual["used_planning_sd"] == expected_sd
    assert actual["full_grid_largest_point_sd"] > 9 * expected_sd
    raw = math.ceil((NormalDist().inv_cdf(.9875) + NormalDist().inv_cdf(.9)) ** 2 * expected_sd ** 2 / .05 ** 2)
    assert actual["planned_questions_unrounded"] == raw
    assert actual["planned_questions"] == max(200, 50 * math.ceil(raw / 50))
