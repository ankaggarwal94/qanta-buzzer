"""Opt-in DPO and GRPO for WAIT/BUZZ policies with a fixed answer selector.

Each question has T possible voluntary stops plus the result of waiting through
the last decision. Rewards label training trajectories; the actor sees only the
current observation when selecting an action.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from training.stopping_data import StoppingEpisode


def build_preference_policy(n_features: int) -> nn.Module:
    """Construct the CPU actor used by both preference trainers and their checkpoints."""
    actor = nn.Sequential(nn.Linear(n_features, 32), nn.Tanh(), nn.Linear(32, 2))
    # Start both algorithms from equal WAIT/BUZZ probabilities at every prefix.
    nn.init.zeros_(actor[-1].weight)
    nn.init.zeros_(actor[-1].bias)
    return actor


def _trajectory_returns(wait_rewards, buzz_rewards):
    """Return the undiscounted simulation payoff at each stop, then at terminal WAIT."""
    cumulative_wait = wait_rewards.cumsum(0)
    before = torch.cat((wait_rewards.new_zeros(1), cumulative_wait[:-1]))
    return torch.cat((before + buzz_rewards, cumulative_wait[-1:]))


def _trajectory_log_probs(step_logp, mask):
    """Joint log P(WAIT until t, BUZZ at t), followed by log P(always WAIT)."""
    wait = step_logp[:, :, 0].masked_fill(~mask, 0.)
    cumulative_wait = wait.cumsum(1)
    before = F.pad(cumulative_wait[:, :-1], (1, 0))
    stop = (before + step_logp[:, :, 1]).masked_fill(~mask, -torch.inf)
    return torch.cat((stop, cumulative_wait[:, -1:]), dim=1)


def _dpo_loss(logp, reference_logp, rewards, *, beta):
    """Weight questions equally and use every strictly ordered legal trajectory pair."""
    valid = torch.isfinite(rewards)
    pairs = valid[:, :, None] & valid[:, None, :] & (rewards[:, :, None] > rewards[:, None, :])
    question, winner, loser = pairs.nonzero(as_tuple=True)
    if not len(winner):
        return logp.masked_fill(~valid, 0.).sum() * 0., 0

    gain = ((logp[question, winner] - reference_logp[question, winner]) -
            (logp[question, loser] - reference_logp[question, loser]))
    pair_counts = pairs.sum(dim=(1, 2))
    per_pair = F.softplus(-beta * gain) / pair_counts[question]
    return per_pair.sum() / pair_counts.count_nonzero(), len(winner)


def _group_advantages(rewards):
    """Center and scale episode returns within each question's independent rollouts."""
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, keepdim=True, correction=0)
    variable = rewards.amax(dim=1, keepdim=True) > rewards.amin(dim=1, keepdim=True)
    return torch.where(variable, centered / scale.clamp_min(1e-12), 0.)


def _grpo_loss(step_logp, old_step_logp, mask, stops, advantages, *, clip):
    """Clipped action-level ratios, averaged over visited actions, then trajectories."""
    t = torch.arange(step_logp.shape[1], device=step_logp.device)
    buzz = t == stops.unsqueeze(-1)
    visited = mask.unsqueeze(1) & (t <= stops.unsqueeze(-1))
    delta = step_logp - old_step_logp
    chosen_delta = torch.where(buzz, delta[:, None, :, 1], delta[:, None, :, 0])
    ratio = chosen_delta.masked_fill(~visited, 0.).exp()
    advantage = advantages.unsqueeze(-1)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    per_rollout = surrogate.masked_fill(~visited, 0.).sum(-1) / visited.sum(-1)
    return -per_rollout.mean()


def fit_preferences(episodes: Sequence[StoppingEpisode], method: str, *, seed: int, epochs: int) -> nn.Module:
    """Fit an actor on within-question simulated returns for the fixed answer selector.

    Parameters
    ----------
    episodes : sequence of StoppingEpisode
        Train-only prefix observations and realized one-step payoffs for both actions.
        The last WAIT is terminal, including its configured horizon payoff.
    method : {"dpo", "grpo"}
        DPO compares complete trajectories to the frozen initial actor. GRPO samples
        eight independent stopping trajectories per question, then takes two clipped
        updates against the frozen rollout policy. No reference KL term is used.
    seed : int
        CPU model initialization, rollout sampling and minibatch permutation seed.
    epochs : int
        Number of visits to each question. Constant-return groups are retained in
        the diagnostics and are never resampled to obtain a favorable group.

    Returns
    -------
    torch.nn.Module
        Evaluation-mode CPU actor with two logits in WAIT, BUZZ order and
        a training_diagnostics dictionary describing the attempted updates.
    """
    if method not in ("dpo", "grpo"):
        raise ValueError("method must be dpo or grpo")
    if type(epochs) is not int or epochs < 1 or not len(episodes):
        raise ValueError("positive integer epochs and training episodes are required")

    training = []
    n_features = None
    for ep in episodes:
        obs = torch.tensor(ep.observations, dtype=torch.float32)
        wait = torch.tensor(ep.wait_rewards, dtype=torch.float64)
        buzz = torch.tensor(ep.buzz_rewards, dtype=torch.float64)
        if obs.ndim != 2 or not obs.shape[0] or not obs.shape[1]:
            raise ValueError("each episode must contain a nonempty [steps, features] observation matrix")
        if n_features is None:
            n_features = obs.shape[1]
        if obs.shape[1] != n_features or wait.shape != (len(obs),) or buzz.shape != wait.shape:
            raise ValueError("trajectory rewards, observations and feature widths must align")
        returns = _trajectory_returns(wait, buzz)
        if not (obs.isfinite().all() and returns.isfinite().all()):
            raise ValueError("non-finite preference training observation or return")
        training.append((obs, returns))

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        actor = build_preference_policy(n_features)
    reference = copy.deepcopy(actor).requires_grad_(False).eval() if method == "dpo" else None
    optimizer = torch.optim.Adam(actor.parameters(), lr=.003)
    beta, clip, group_size, batch_size = .5, .2, 8, 64
    groups_seen = uninformative = steps = pairs_seen = 0

    for _ in range(epochs):
        order = torch.randperm(len(training), generator=generator)
        for indices in order.split(batch_size):
            batch = [training[i] for i in indices.tolist()]
            obs = pad_sequence([x for x, _ in batch], batch_first=True)
            lengths = torch.tensor([len(x) for x, _ in batch])
            mask = torch.arange(obs.shape[1]) < lengths[:, None]
            voluntary = pad_sequence([r[:-1] for _, r in batch], batch_first=True, padding_value=-torch.inf)
            returns = torch.cat((voluntary, torch.stack([r[-1] for _, r in batch])[:, None]), dim=1)
            groups_seen += len(batch)
            with torch.no_grad():
                if method == "dpo":
                    ref_logp = _trajectory_log_probs(F.log_softmax(reference(obs), dim=-1), mask)
                    worst = returns.masked_fill(~returns.isfinite(), torch.inf).amin(dim=1)
                    active = returns.amax(dim=1) > worst
                else:
                    old = F.log_softmax(actor(obs), dim=-1)
                    # Sampling the distribution of first BUZZ (or terminal WAIT)
                    # is exactly sequential stochastic rollout in this fixed-prefix environment.
                    probs = _trajectory_log_probs(old, mask).exp()
                    stops = torch.multinomial(probs, group_size, replacement=True, generator=generator)
                    sampled = returns.gather(1, stops)
                    advantages = _group_advantages(sampled).to(obs.dtype)
                    active = sampled.amax(dim=1) > sampled.amin(dim=1)
                uninformative += int((~active).sum())
            if not active.any():
                continue
            for _update in range(1 if method == "dpo" else 2):
                current = F.log_softmax(actor(obs), dim=-1)
                if method == "dpo":
                    loss, pairs = _dpo_loss(_trajectory_log_probs(current, mask), ref_logp, returns, beta=beta)
                    pairs_seen += pairs
                else:
                    loss = _grpo_loss(current, old, mask, stops, advantages, clip=clip)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                steps += 1
    actor.training_diagnostics = {
        "method": method, "groups_seen": groups_seen, "uninformative_groups": uninformative,
        "optimizer_steps": steps, "batch_size": batch_size, "learning_rate": .003,
        "preference_pairs": pairs_seen if method == "dpo" else None,
        "beta": beta if method == "dpo" else None,
        "group_size": group_size if method == "grpo" else None,
        "ratio_clip": clip if method == "grpo" else None,
        "updates_per_rollout_batch": 2 if method == "grpo" else None,
        "reference_kl_weight": 0. if method == "grpo" else None,
    }
    return actor.eval()
