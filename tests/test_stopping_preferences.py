"""Trajectory probability and optimizer regressions for the optional preference arms."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.stopping_preferences import (
    _dpo_loss, _grpo_loss, _group_advantages, _trajectory_log_probs,
    _trajectory_returns, build_preference_policy, fit_preferences,
)


def test_all_trajectories_normalize_with_padding_and_terminal_wait():
    probs = torch.tensor([[[.8, .2], [.6, .4]], [[.9, .1], [.01, .99]]])
    mask = torch.tensor([[True, True], [True, False]])
    logp = _trajectory_log_probs(probs.log(), mask)
    torch.testing.assert_close(logp.exp(), torch.tensor([[.2, .32, .48], [.1, 0., .9]]))
    torch.testing.assert_close(logp.logsumexp(dim=-1), torch.zeros(2), atol=1e-7, rtol=0)
    returns = _trajectory_returns(torch.tensor([.2, -.3]), torch.tensor([1., 2.]))
    # BUZZ now, WAIT then BUZZ, and WAIT twice.
    torch.testing.assert_close(returns, torch.tensor([1., 2.2, -.1]))


def test_dpo_uses_frozen_reference_and_strict_preferences_only():
    policy = torch.tensor([[.1, .3, .6], [.8, .2, 0.]]).log().requires_grad_()
    reference = policy.detach().clone()
    rewards = torch.tensor([[2., 1., 1.], [0., 0., float("-inf")]])
    loss, pairs = _dpo_loss(policy, reference, rewards, beta=1.)
    assert pairs == 2
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert policy.grad[0, 0] < 0 and torch.all(policy.grad[0, 1:] > 0)
    assert torch.equal(policy.grad[1], torch.zeros(3))


def test_grpo_clips_each_visited_action_and_stops_gradients_at_terminal():
    mask = torch.tensor([[True, True]])
    old = torch.tensor([[[.5, .5], [.5, .5]]]).log()
    # BUZZ immediately wins, WAIT then BUZZ loses. The second decision of
    # the immediate-stop trajectory does not enter its ratio or length.
    stops = torch.tensor([[0, 1]])
    advantages = _group_advantages(torch.tensor([[1., 0.]]))
    new = old.clone().detach().requires_grad_()
    loss = _grpo_loss(new, old, mask, stops, advantages, clip=.2)
    loss.backward()
    torch.testing.assert_close(new.grad, torch.tensor([[[.25, -.5], [0., .25]]]))
    # At horizon one, both favorable moves exceed the clip range.
    new = torch.tensor([[[.1, .9]]]).log().requires_grad_()
    loss = _grpo_loss(new, old[:, :1], mask[:, :1], torch.tensor([[0, 1]]), advantages, clip=.2)
    assert loss.item() == pytest.approx(-.2, abs=1e-6)
    loss.backward()
    torch.testing.assert_close(new.grad, torch.zeros_like(new))


def one_state(buzz):
    return SimpleNamespace(observations=np.array([[1., 0.]], dtype=np.float32),
                           buzz_rewards=np.array([buzz]), wait_rewards=np.array([0.]))


@pytest.mark.parametrize("method", ["dpo", "grpo"])
def test_fit_learns_a_winning_action_reloads_and_repeats(method):
    episodes = [one_state(1.) for _ in range(16)]
    model = fit_preferences(episodes, method, seed=3, epochs=3)
    x = torch.tensor(episodes[0].observations)
    assert model(x).argmax(dim=-1).item() == 1
    other = build_preference_policy(2)
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x), other(x))
    repeat = fit_preferences(episodes, method, seed=3, epochs=3)
    torch.testing.assert_close(repeat(x), other(x))


@pytest.mark.parametrize("method", ["dpo", "grpo"])
def test_flat_rewards_supply_no_gradient_and_do_not_trigger_extra_draws(method):
    model = fit_preferences([one_state(0.)] * 5, method, seed=3, epochs=2)
    diagnostics = model.training_diagnostics
    assert diagnostics["groups_seen"] == diagnostics["uninformative_groups"] == 10
    assert diagnostics["optimizer_steps"] == 0
    assert torch.isfinite(model(torch.zeros((1, 2)))).all()
