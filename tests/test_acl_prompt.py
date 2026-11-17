"""Public payoff text for ACL training and inference."""
from dataclasses import replace
import pytest

from qb_data.acl_protocol import ArmSpec
from qb_env.acl_prompt import reward_instructions
from qb_env.acl_rewards import RewardSpec


def arm(name, **kwargs):
    return ArmSpec(version=1, menu="private-menu-label", k=4, seed=987654321,
                   reward=RewardSpec(arm=name, start=3., end=.5, gamma=.875, **kwargs),
                   wait_cost=-.125, wrong_reward=-2.5, abstain_reward=.25, correct_reward=7.)


@pytest.mark.parametrize("name,expected", [
    ("baseline", "Correct 7."), ("constant", "Correct 3."),
    ("linear", "1-p"), ("exponential", "exp(-2*p)"),
    ("power", "(1-p)^2"), ("cosine", "cos(pi*p)"),
    ("piecewise", "(0,1), (0.25,0.75), (1,0)"), ("step", "p<0.4"),
])
def test_rules_match_each_public_schedule(name, expected):
    spec = arm(name, shape=2., knots=[(0,1),(.25,.75),(1,0)], threshold=.4)
    text = reward_instructions(spec)
    assert expected in text
    assert "Wrong -2.5" in text and "every WAIT -0.125" in text
    assert "final WAIT also adds 0.25" in text and "discount 0.875" in text
    if name not in ("baseline","constant"):
        assert "prefix number-1" in text and "p=1 for a single prefix" in text
        assert "0.5+(3-0.5)" in text
    assert text == reward_instructions(replace(spec, menu="other", k=7, seed=2))
    assert "private-menu-label" not in text and "987654321" not in text


def test_zero_rate_shaping_and_exact_decimal_precision():
    spec = arm("exponential", shape=0., potential_scale=-.3333333333333333)
    text = reward_instructions(spec)
    assert "1-p" in text and "exp(" not in text
    assert "0.875*P(next)-P(now)" in text
    assert "-0.3333333333333333" in text and "terminal P=0" in text
