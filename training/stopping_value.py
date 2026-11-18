"""Offline learners for finite, undiscounted binary stopping episodes.

Columns are always WAIT, BUZZ. BUZZ terminates at any prefix. WAIT earns the
recorded incremental reward and reveals the next observation, except at the
last decision, where the recorded WAIT reward resolves the configured horizon.
Only training targets use future rewards or subsequent observations.
"""
from __future__ import annotations

import copy
import math
import random
from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from training.stopping_data import StoppingEpisode


class _Replay(NamedTuple):
    observations: torch.Tensor
    rewards: torch.Tensor
    successors: torch.Tensor
    continues: torch.Tensor


def make_value_model(n_features: int) -> nn.Sequential:
    """Construct the two-output CPU MLP used by all stopping value checkpoints."""
    if n_features < 1:
        raise ValueError("expected at least one observation feature")
    return nn.Sequential(nn.Linear(n_features, 32), nn.Tanh(), nn.Linear(32, 2))


def _pack(episodes: Sequence[StoppingEpisode]) -> _Replay:
    observations, rewards, boundaries = [], [], []
    width = None
    for ep in episodes:
        obs = np.asarray(ep.observations)
        wait = np.asarray(ep.wait_rewards)
        buzz = np.asarray(ep.buzz_rewards)
        if obs.ndim != 2 or min(obs.shape) < 1 or (width is not None and width != obs.shape[1]):
            raise ValueError("expected nonempty observations with a consistent feature width")
        if wait.shape != (len(obs),) or buzz.shape != (len(obs),):
            raise ValueError("expected one WAIT and BUZZ reward per decision")
        width = obs.shape[1]
        observations.append(obs)
        rewards.append(np.column_stack((wait, buzz)))
        boundaries.append(len(obs))

    if not observations:
        raise ValueError("expected at least one training episode")
    states = torch.from_numpy(np.concatenate(observations).astype(np.float32, copy=False))
    returns = torch.from_numpy(np.concatenate(rewards).astype(np.float32, copy=False))
    if not torch.isfinite(states).all() or not torch.isfinite(returns).all():
        raise ValueError("training observations and rewards must be finite")
    continues = torch.ones(len(states), dtype=torch.bool)
    continues[np.cumsum(boundaries) - 1] = False
    # Terminal rows point to themselves solely to keep indexing in range.
    successors = torch.arange(len(states)) + continues.long()
    return _Replay(states, returns, successors, continues)


@torch.no_grad()
def _bellman_targets(rewards, continues, next_online, next_target):
    """Select the continuation action online and evaluate it on the target."""
    actions = next_online.argmax(dim=1, keepdim=True)
    continuation = next_target.gather(1, actions).squeeze(1)
    targets = rewards.clone()
    targets[:, 0] += torch.where(continues, continuation, 0.0)
    return targets


@torch.no_grad()
def _fitted_value(model: nn.Sequential, replay: _Replay, epochs: int) -> None:
    """Iterate full-data ridge Bellman regressions on fixed random tanh features."""
    hidden = model[:2](replay.observations).double()
    design = torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=1)
    gram = design.T @ design
    gram.diagonal()[:-1].add_(1e-4)  # Do not penalize the intercept.
    factor = torch.linalg.cholesky(gram)
    rewards = replay.rewards.double()
    estimates = torch.zeros_like(rewards)
    for _ in range(epochs):
        following = estimates[replay.successors]
        targets = _bellman_targets(rewards, replay.continues, following, following)
        coefficients = torch.cholesky_solve(design.T @ targets, factor)
        estimates = design @ coefficients
    model[-1].weight.copy_(coefficients[:-1].T)
    model[-1].bias.copy_(coefficients[-1])


def _teacher_actions(episodes: Sequence[StoppingEpisode], *, seed: int, epochs: int) -> torch.Tensor:
    """Label each training episode with a teacher fitted on other question groups."""
    keys = [ep.group_id if ep.group_id is not None else ep.qid for ep in episodes]
    groups = list(dict.fromkeys(keys))
    if len(groups) < 2:
        raise ValueError("stop_imitation requires at least two distinct training question groups")
    random.Random(seed).shuffle(groups)
    folds = {key: i % 2 for i, key in enumerate(groups)}
    labels = [None] * len(episodes)
    with torch.no_grad():
        for fold in range(2):
            train = [ep for key, ep in zip(keys, episodes) if folds[key] != fold]
            teacher = fit_value(train, "fitted_value", seed=seed + fold, epochs=epochs)
            for i, (key, ep) in enumerate(zip(keys, episodes)):
                if folds[key] == fold:
                    obs = torch.as_tensor(ep.observations, dtype=torch.float32)
                    labels[i] = teacher(obs).argmax(dim=1)
    return torch.cat(labels)


def fit_value(
    episodes: Sequence[StoppingEpisode], method: str, *, seed: int, epochs: int
) -> nn.Sequential:
    """Fit a stopping rule using only the supplied training questions.

    Parameters
    ----------
    episodes : sequence of StoppingEpisode
        Prefix observations and realized simulator rewards for both actions.
    method : str
        fitted_value uses fixed-feature least squares Bellman iteration.
        double_dqn uses uniformly sampled offline replay with full action
        coverage, a Huber loss, and a target network refreshed every 20 updates.
        stop_imitation learns hard actions from two out-of-fold value teachers.
    seed : int
        Seed for initialization, replay sampling and teacher folds.
    epochs : int
        Fitted Bellman sweeps or replay passes. The gradient-based learners use
        at least eight minibatches per pass for small datasets.

    Returns
    -------
    torch.nn.Sequential
        A CPU model in evaluation mode. Its last dimension contains two action
        values (or imitation logits). Argmax selects WAIT on an exact tie.
    """
    if method not in {"fitted_value", "double_dqn", "stop_imitation"}:
        raise ValueError(f"unknown stopping value method {method!r}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    replay = _pack(episodes)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = make_value_model(replay.observations.shape[1])
    if method == "fitted_value":
        _fitted_value(model, replay, epochs)
        return model.eval()

    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    labels = None
    if method == "stop_imitation":
        labels = _teacher_actions(episodes, seed=seed, epochs=epochs)
    else:
        target_model = copy.deepcopy(model).eval().requires_grad_(False)

    n = len(replay.observations)
    batch_size = min(n, 128)
    updates = epochs * max(8, math.ceil(n / batch_size))
    for step in range(1, updates + 1):
        indices = torch.randint(n, (batch_size,), generator=generator)
        values = model(replay.observations[indices])
        if labels is not None:
            loss = F.cross_entropy(values, labels[indices])
        else:
            next_obs = replay.observations[replay.successors[indices]]
            with torch.no_grad():
                targets = _bellman_targets(
                    replay.rewards[indices], replay.continues[indices],
                    model(next_obs), target_model(next_obs),
                )
            loss = F.smooth_l1_loss(values, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if labels is None and step % 20 == 0:
            target_model.load_state_dict(model.state_dict())
    return model.eval()
