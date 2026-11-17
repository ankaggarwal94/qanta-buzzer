"""Executable contract for the opt-in ACL reward and prefix state machine."""

from dataclasses import replace
import math

import gymnasium as gym
import numpy as np
import pytest

from qb_env.acl_rewards import RewardSpec
from qb_env.tossup_env import TossupMCEnv


@pytest.mark.parametrize("arm", ["linear", "exponential", "power", "cosine", "piecewise", "step"])
def test_schedule_contract(arm):
    for shape in (1e-310, .5, 1., 1000.):
        spec = RewardSpec(arm=arm, shape=shape, start=1.5, end=.25)
        xs = np.unique(np.r_[0., math.nextafter(0., 1.), np.linspace(0, 1, 101),
                              math.nextafter(1., 0.), 1.])
        ys = [spec.correct_reward(x, 2.) for x in xs]
        assert ys[0] == 1.5 and ys[-1] == .25
        assert all(.25 <= b <= a <= 1.5 for a, b in zip(ys, ys[1:]))
        if arm == "exponential" and shape < 1e-10:
            np.testing.assert_allclose(ys, .25 + 1.25 * (1-xs), atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"shape": float("nan")}, {"start": float("inf")}, {"arm": "bogus"},
    {"arm":"power", "shape":0}, {"arm":"exponential","shape":-1},
    {"gamma":1.001}, {"threshold":0}, {"start":0, "end":1},
    {"knots":[[0,1],[.4,.2],[.6,.4],[1,0]]}, {"knots":[[0,1],[0,0],[1,0]]},
])
def test_invalid_specs(kwargs):
    with pytest.raises(ValueError):
        RewardSpec(**kwargs)


def make_env(question, h, k, mode, spec, **kwargs):
    q = replace(question, run_indices=question.run_indices[:h],
                cumulative_prefixes=question.cumulative_prefixes[:h],
                options=question.options[:k], option_profiles=question.option_profiles[:k],
                gold_index=k-1)
    cache = {(0,i): np.full(k, 1/k, dtype=np.float32) for i in range(h)}
    return TossupMCEnv([q], None, K=k, max_K=4, variable_K=True,
                       end_mode=mode, precomputed_beliefs=cache, acl_reward=spec,
                       buzz_incorrect=-.5, no_buzz_reward=-.2, wait_penalty=.1, **kwargs)


@pytest.mark.parametrize("h",[1,2,3])
@pytest.mark.parametrize("k",[2,3,4])
@pytest.mark.parametrize("mode",["force_commit","no_buzz"])
def test_all_reachable_action_paths(sample_mc_question, h, k, mode):
    spec = RewardSpec(arm="linear", gamma=.8, potential_scale=.75)
    env = make_env(sample_mc_question, h, k, mode, spec)
    paths = [[0]*i+[a] for i in range(h) for a in range(1,k+1)] + [[0]*h]
    for path in paths:
        env.reset(options={"question_idx":0})
        actual = []
        base = []
        for i, action in enumerate(path):
            assert env.step_idx == i+1
            assert list(env.action_masks()) == [True]*(k+1)+[False]*(4-k)
            with pytest.raises(ValueError):
                env.step(k+1)
            _, reward, terminated, truncated, info = env.step(action)
            p = i/(h-1) if h>1 else 1.
            expected = (1-p if action == k else -.5) if action else -.1
            if not action and i == h-1:
                expected += -.2 if mode == "no_buzz" else -.5
            terminal = bool(action) or i == h-1
            assert (terminated,truncated) == (terminal,False)
            row = info["acl_transition"]
            assert row["base_reward"] == pytest.approx(expected)
            assert reward == pytest.approx(expected + (.8*(0 if terminal else .75/k) - .75/k))
            actual.append(.8**i*reward)
            base.append(.8**i*expected)
        assert sum(actual) == pytest.approx(sum(base)-.75/k)
        with pytest.raises(RuntimeError):
            env.step(0)


def test_external_time_limit_keeps_bootstrap(sample_mc_question):
    spec = RewardSpec(gamma=.8,potential_scale=1.)
    env = gym.wrappers.TimeLimit(make_env(sample_mc_question,3,2,"no_buzz",spec),1)
    env.reset()
    _,r,term,trunc,info = env.step(0)
    assert (term,trunc) == (False,True)
    assert r == pytest.approx(-.1 + .8*.5 - .5)
    assert info["acl_transition"]["next_potential"] == .5


def test_suffix_length_and_gold_are_invisible(sample_mc_question):
    a = make_env(sample_mc_question,2,2,"no_buzz",RewardSpec())
    b = make_env(sample_mc_question,3,2,"no_buzz",RewardSpec())
    b.questions[0].gold_index = 0
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
    np.testing.assert_array_equal(a.step(0)[0], b.step(0)[0])

@pytest.mark.parametrize("arm, kwargs, want", [
    ("baseline", {}, 7.),
    ("constant", {}, 2.),
    ("linear", {}, 1.75),
    ("exponential", {"shape": 0.}, 1.75),
    ("exponential", {"shape": math.log(16.)}, 1. + 7/15),
    ("power", {"shape": 2.}, 1. + .75**2),
    ("cosine", {}, 1.5 + .5*math.cos(math.pi/4)),
    ("piecewise", {"knots": [[0.,1.],[.5,.2],[1.,0.]]}, 1.6),
    ("step", {"threshold": .25}, 1.),
    ("step", {"threshold": .5}, 2.),
])
def test_hand_computed_interior_values(arm, kwargs, want):
    spec = RewardSpec(arm=arm, start=2., end=1., **kwargs)
    assert spec.correct_reward(.25, 7.) == pytest.approx(want, abs=2e-15)


@pytest.mark.parametrize("p", [float("inf"),float("nan"),-.1,1.1,True])
def test_schedule_rejects_invalid_progress(p):
    with pytest.raises(ValueError, match="progress"):
        RewardSpec().correct_reward(p,1.)


def test_runtime_monitor_rejects_bad_belief(sample_mc_question):
    env = make_env(sample_mc_question, 2, 2, "no_buzz", RewardSpec())
    env.reset()
    env.precomputed_beliefs[(0,1)][:] = float("nan")
    with pytest.raises(ValueError, match="question_idx=0"):
        env.step(0)


def test_forced_correct_and_replay(sample_mc_question):
    env = make_env(sample_mc_question, 2, 2, "force_commit", RewardSpec(
        arm="constant", gamma=.8, potential_scale=2.))
    env.precomputed_beliefs[(0,1)][:] = np.array([.25,.75])
    def trajectory():
        env.reset(options={"question_idx":0})
        return [env.step(0)[4]["acl_transition"] for _ in range(2)]
    first = trajectory()
    assert first == trajectory()
    assert first[0]["next_potential"] == 1.5
    assert first[1]["potential"] == 1.5
    assert first[1]["base_reward"] == .9
    assert first[1]["reward"] == pytest.approx(-.6)


def test_finite_reward_monitor(sample_mc_question):
    env = make_env(sample_mc_question, 2, 2, "no_buzz", RewardSpec(
        arm="constant", start=1.7e308, potential_scale=-1.7e308))
    env.reset()
    with pytest.raises(RuntimeError, match="ACL transition violation"):
        env.step(2)

def test_checked_schedule_is_cached_and_consumed(sample_mc_question):
    env = make_env(sample_mc_question, 3, 2, "no_buzz", RewardSpec(arm="linear"))
    approved = []
    env._acl_schedule_validator = lambda values, first, last: approved.append((values, first, last))
    env.reset()
    env.step(0)
    assert env.step(2)[1] == approved[0][0][1]
    env.reset()
    assert approved == [((1., .5, 0.), 1., 0.)]


def test_failed_schedule_check_blocks_before_first_action(sample_mc_question):
    env = make_env(sample_mc_question, 1, 2, "no_buzz", RewardSpec(arm="linear"))
    def reject(values, first, last):
        assert values == (0.,) and first == last == 0.
        raise ValueError("rejected schedule")
    env._acl_schedule_validator = reject
    with pytest.raises(ValueError, match="rejected schedule"):
        env.reset()
    assert not env._acl_schedules

def test_transition_monitor_stops_corrupted_visible_count(sample_mc_question, monkeypatch):
    env = make_env(sample_mc_question, 3, 2, "no_buzz", RewardSpec())
    env.reset()
    compute = env._compute_belief
    def corrupted_update(question, position):
        env.step_idx += 1
        return compute(question, position)
    monkeypatch.setattr(env, "_compute_belief", corrupted_update)
    with pytest.raises(RuntimeError, match=r"ACL transition violation: .*'visible': 1, 'next_visible': 3"):
        env.step(0)
    with pytest.raises(RuntimeError, match="successful reset"):
        env.step(0)


def test_randomized_schedule_parameters():
    rng = np.random.default_rng(321)
    for _ in range(100):
        start, end = sorted(rng.uniform(0.,1e6,size=2), reverse=True)
        xs = np.sort(np.r_[0., 1., rng.random(30)])
        rate = 10. ** rng.uniform(-307.,307.)
        for arm in ("exponential","power","linear","cosine","step","piecewise"):
            spec = RewardSpec(arm=arm, start=start, end=end, shape=rate,
                              threshold=rng.uniform(.001,1.))
            ys = [spec.correct_reward(x,0.) for x in xs]
            assert ys[0] == start and ys[-1] == end
            assert all(end <= b <= a <= start for a,b in zip(ys,ys[1:]))

def test_correct_rounding_recovers_cancelled_small_payoff(sample_mc_question):
    # (16 + 2^-53) - 16 loses the payoff if the first addition is binary64.
    from fractions import Fraction

    env = make_env(sample_mc_question, 1, 2, "no_buzz", RewardSpec(potential_scale=32.))
    env.wait_penalty = 2.**-53
    env.no_buzz_reward = 16.
    approved = []
    env._acl_transition_validator = lambda *args: approved.append(args)
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == -2.**-53
    evidence = info["acl_transition"]["exact"]
    assert Fraction(*evidence["base"]) == Fraction(16) - Fraction(2.**-53)
    assert Fraction(*evidence["total"]) == Fraction(reward)
    assert approved == [(Fraction(*evidence["base"]), .99, 16., 0., reward)]


def test_discounted_exact_reference_and_learner_rounding_bound(sample_mc_question):
    from fractions import Fraction

    env = make_env(sample_mc_question, 3, 3, "no_buzz", RewardSpec(potential_scale=.7))
    env.reset()
    rows = [env.step(a)[4]["acl_transition"] for a in (0,0,3)]
    scale = Fraction(env.acl_reward.gamma)
    exact_base = exact_total = abi_return = signed_error = Fraction(0)
    for i, row in enumerate(rows):
        exact_base += scale**i * Fraction(*row["exact"]["base"])
        exact_total += scale**i * Fraction(*row["exact"]["total"])
        abi_return += scale**i * Fraction(row["learner_reward_float32"])
        signed_error += scale**i * Fraction(*row["exact"]["learner_rounding_error"])
    assert exact_total == exact_base - Fraction(rows[0]["potential"])
    assert abi_return == exact_total + signed_error

def test_gym_halfway_ties_and_double_rounding():
    from fractions import Fraction
    from qb_env.acl_settlement import settle_reward

    # Exactly half an ULP above an even and an odd binary64 significand.
    for outcome, expected in ((1., 1.), (math.nextafter(1.,2.), 1. + 2.**-51)):
        row = settle_reward(2.**-53, outcome, 1., 0., 0.)
        assert row["reward"] == expected
        assert abs(Fraction(*row["exact"]["gym_rounding_error"])) == Fraction(2.**-53)
    # Halfway in binary32 plus a tiny positive epsilon: binary64 conversion
    # drops epsilon and the subsequent float32 cast selects the even lower endpoint.
    row = settle_reward(2.**-55, 1.+2.**-24, 1., 0., 0.)
    assert row["reward"] == 1.+2.**-24
    assert row["learner_reward_float32"] == 1.
    assert Fraction(*row["exact"]["total"]) > Fraction(row["reward"])


def test_positive_subnormal_underflow_tie():
    from fractions import Fraction
    from qb_env.acl_settlement import settle_reward

    smallest = math.ulp(0.)
    row = settle_reward(0., 0., .5, 0., smallest)
    assert row["reward"] == 0.
    assert Fraction(*row["exact"]["total"]) == Fraction(1, 2**1075)


def test_rejected_live_transition_disarms_env(sample_mc_question):
    env = make_env(sample_mc_question, 3, 3, "no_buzz", RewardSpec())
    def reject(*args):
        raise ValueError("wrong rounding cell")
    env._acl_transition_validator = reject
    env.reset()
    with pytest.raises(RuntimeError, match="wrong rounding cell"):
        env.step(0)
    with pytest.raises(RuntimeError, match="successful reset"):
        env.step(0)

def test_native_validator_settles_live_environment(sample_mc_question):
    from formal.acl_expansion.runtime import LeanRuntime

    try:
        verifier = LeanRuntime()
    except FileNotFoundError:
        pytest.skip("build the pinned Lean runtime for native conformance")
    with verifier:
        env = make_env(sample_mc_question, 3, 3, "no_buzz",
                       RewardSpec(arm="exponential", potential_scale=.7),
                       acl_schedule_validator=verifier.validate_schedule,
                       acl_transition_validator=verifier.validate_transition)
        env.reset()
        for action in (0,0,3):
            env.step(action)
        assert env.terminated
        def corrupted(base, gamma, current, successor, rounded):
            verifier.validate_transition(base, gamma, current, successor,
                                         math.nextafter(rounded, math.inf))
        env._acl_transition_validator = corrupted
        env.reset()
        with pytest.raises(RuntimeError, match="ACL transition violation"):
            env.step(0)

def test_python_schedule_uses_exact_step_threshold_at_prefix_boundary():
    from fractions import Fraction

    spec = RewardSpec(arm="step", threshold=.1)
    assert spec.correct_reward(Fraction(1,10),1.) == 1.
    assert spec.correct_reward(Fraction(.1),1.) == 0.
