"""Named-curve finite refinement beyond schedule monotonicity."""
from fractions import Fraction as F
import math

import pytest

from qb_env.acl_certified_curve import certify_payout

pytest.importorskip("flint", reason="verified curve tests require the pinned python-flint extra")


def test_wrong_monotone_interior_is_rejected():
    p = ["p"]
    q = lambda n: ["q", n, 1]
    quadratic = ["pow", ["sub", q(1), p], q(2)]
    # Both the quadratic and the linear replacement satisfy the same endpoints/order.
    with pytest.raises(ValueError, match="certified payout"):
        certify_payout(quadratic,F(1,2),lambda *args: None,claimed=.5)
    value, cert = certify_payout(quadratic,F(1,2),lambda *args: None)
    assert value == .25 and cert["lower"] == cert["upper"] == [1,4]


def test_exact_branch_at_nondyadic_progress():
    tree = ["if_lt",["p"],["q",1,3],["q",1,1],["q",0,1]]
    assert certify_payout(tree,F(1,3),lambda *a: None)[0] == 0.
    assert certify_payout(tree,F(1,3)-F(1,10**100),lambda *a: None)[0] == 1.


def test_native_enclosure_endpoints_are_supplied():
    pytest.importorskip("flint")
    calls = []
    tree = ["cospi",["p"]]
    value, cert = certify_payout(tree,F(1,4),lambda *args: calls.append(args))
    assert value == math.sqrt(.5)
    low, high = (F(*cert[k]) for k in ("lower","upper"))
    assert low < high and low*low <= F(1,2) <= high*high
    assert calls == [(low,0.,0.,0.,value),(high,0.,0.,0.,value)]

@pytest.mark.parametrize("arm", ["baseline","constant","linear","exponential","power",
                                  "cosine","piecewise","step"])
def test_native_program_drives_real_environment(arm, sample_mc_question, monkeypatch):
    import numpy as np
    from formal.acl_expansion.runtime import LeanRuntime
    from qb_data.acl_protocol import digest
    from qb_env.acl_rewards import RewardSpec
    from qb_env.tossup_env import TossupMCEnv
    try:
        native = LeanRuntime()
    except FileNotFoundError:
        pytest.skip("build the pinned native verifier")
    spec = RewardSpec(arm=arm, shape=2., knots=[(0.,1.),(.5,.2),(1.,0.)])
    with native:
        def monotone_mutant(self, p, baseline):
            return baseline if self.arm == "baseline" else (1. if self.arm == "constant" else 1-p)
        monkeypatch.setattr(RewardSpec,"correct_reward",monotone_mutant)
        env = TossupMCEnv([sample_mc_question], None, acl_reward=spec,
                         precomputed_beliefs={(0,i):np.full(4,.25) for i in range(6)},
                         acl_schedule_validator=native.validate_schedule,
                         acl_transition_validator=native.validate_transition, acl_formula=native.formula)
        env.reset()
        first = env.step(0)[4]["acl_transition"]
        second = env.step(1)[4]["acl_transition"]
        formula = first["curve_program"]
        assert digest(formula) == second["curve_program_sha256"]
        correct, cert = certify_payout(formula, F(1,5), native.validate_transition)
        assert correct == second["outcome_reward"] and second["curve_certificate"] == cert
        if arm == "power":
            assert correct == float(F(16,25))
            with pytest.raises(ValueError,match="certified payout"):
                certify_payout(formula,F(1,5),native.validate_transition,claimed=.8)


@pytest.mark.parametrize("rate", [math.ulp(0.), 0., 1., 1e308])
def test_extreme_rates_round_to_one_certified_value(rate):
    from formal.acl_expansion.runtime import LeanRuntime
    try:
        native = LeanRuntime()
    except FileNotFoundError:
        pytest.skip("build the pinned native verifier")
    with native:
        tree = native.formula("exponential",1.,0.,rate,.5,((0.,1.),(1.,0.)),1.)
        value, certificate = certify_payout(tree,F(1,2),native.validate_transition)
        if rate < 1e-10:
            assert value == .5
        elif rate > 1e300:
            assert value == 0.
        else:
            assert .37 < value < .38
        assert float(F(*certificate["lower"])) == value == float(F(*certificate["upper"]))


@pytest.mark.parametrize("tree", [
    ["q",True,1], ["q",3,0], ["exp",["p"],["p"]], ["invalid",1],
    ["if_lt",["cospi",["p"]],["p"],["p"],["p"]],
])
def test_malformed_or_inexact_decision_opcode_fails_closed(tree):
    with pytest.raises(ValueError):
        certify_payout(tree,F(1,4),lambda *args: None)

def test_threshold_distinguishes_exact_prefix_from_rounded_diagnostic():
    from formal.acl_expansion.runtime import LeanRuntime
    try:
        native = LeanRuntime()
    except FileNotFoundError:
        pytest.skip("build the pinned native verifier")
    with native:
        threshold = .1
        assert F(1,10) < F(threshold)
        tree = native.formula("step", 1., 0., 1., threshold, ((0.,1.),(1.,0.)), 1.)
        assert certify_payout(tree,F(1,10),native.validate_transition)[0] == 1.
        assert certify_payout(tree,F(threshold),native.validate_transition)[0] == 0.

def test_huge_exact_denominator_has_portable_json_certificate():
    import json

    program = ["pow",["q",1,3],["q",30000,1]]
    value, record = certify_payout(program,F(0),lambda *args: None)
    assert value == 0.
    assert F(*record["lower"]) <= F(1,3)**30000 <= F(*record["upper"])
    assert json.loads(json.dumps(record)) == record
