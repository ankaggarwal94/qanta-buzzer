"""
Custom PPO Training for T5 Policy Model

Implements PPOTrainer with RolloutBuffer for end-to-end PPO fine-tuning of
T5PolicyModel on incremental quiz bowl episodes. Uses Generalized Advantage
Estimation (GAE) for variance reduction and dynamic batch padding to minimize
memory footprint.

Key design decisions:
    - Rollout tensors (input_ids, attention_mask) are immediately detached and
      moved to CPU after collection to prevent GPU memory accumulation.
    - Dynamic padding: each mini-batch is padded to the max length within that
      batch, not a global 512-token maximum, saving ~50%+ memory.
    - Config-dict interface for compatibility with the unified codebase YAML
      config pattern (see configs/t5_policy.yaml).

Ported from qanta-buzzer reference implementation (train_ppo.py) with:
    - TextObservationWrapper for text-based rollout collection
    - Memory-safe tensor management (detach + CPU storage)
    - Dynamic padding per mini-batch
    - Config dict interface replacing Config class
    - NumPy-style docstrings

Usage
-----
From Python::

    from training.train_ppo_t5 import PPOTrainer, run_ppo_training
    from models.t5_policy import T5PolicyModel
    from qb_data.mc_builder import MCQuestion

    model = T5PolicyModel({"model_name": "t5-small", "device": "cpu"})
    trainer = PPOTrainer(model, train_qs, val_qs, config)
    trainer.train()

From command line::

    python scripts/train_t5_policy.py --config configs/t5_policy.yaml
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion


@dataclass
class RolloutStep:
    """Single step in an episode rollout.

    Stores observation text, action, reward, value estimate, and log probability
    for a single environment step. Tokenized tensors (input_ids, attention_mask)
    are stored on CPU to prevent GPU memory accumulation during rollout collection.

    Attributes
    ----------
    observation_text : str
        Text observation at this step (CLUES: ... | CHOICES: ...).
    action : int
        Combined action taken (0=WAIT, 1..K=SELECT).
    reward : float
        Scalar reward received.
    done : bool
        Whether this step ended the episode.
    value : float
        Value estimate from the critic at this step.
    log_prob : float
        Log probability of the action under the policy at collection time.
    input_ids : torch.Tensor or None
        Tokenized input IDs stored on CPU. Shape ``[1, seq_len]``.
    attention_mask : torch.Tensor or None
        Attention mask stored on CPU. Shape ``[1, seq_len]``.
    return_ : float
        Discounted return (filled by ``compute_returns_and_advantages``).
    advantage : float
        GAE advantage (filled by ``compute_returns_and_advantages``).
    """

    observation_text: str
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    return_: float = 0.0
    advantage: float = 0.0


class RolloutBuffer:
    """Buffer to store and process episode rollouts for PPO updates.

    Accumulates complete episode rollouts (lists of RolloutStep), then computes
    discounted returns and GAE advantages across all episodes. Provides a flat
    view of all steps for mini-batch iteration during PPO updates.

    Attributes
    ----------
    rollouts : list[list[RolloutStep]]
        List of episode rollouts, each a list of steps.
    """

    def __init__(self) -> None:
        self.rollouts: List[List[RolloutStep]] = []

    def reset(self) -> None:
        """Clear all stored rollouts."""
        self.rollouts = []

    def add_rollout(self, steps: List[RolloutStep]) -> None:
        """Add a complete episode rollout to the buffer.

        Parameters
        ----------
        steps : list[RolloutStep]
            Complete episode rollout (ordered list of steps from reset to done).
        """
        self.rollouts.append(steps)

    def get_all_steps(self) -> List[RolloutStep]:
        """Get a flat list of all steps from all rollouts.

        Returns
        -------
        list[RolloutStep]
            All steps concatenated in order (rollout 0 steps, then rollout 1, ...).
        """
        all_steps: List[RolloutStep] = []
        for rollout in self.rollouts:
            all_steps.extend(rollout)
        return all_steps

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float
    ) -> None:
        """Compute discounted returns and GAE advantages for all rollouts.

        Uses Generalized Advantage Estimation (GAE) to compute per-step
        advantages. For each rollout, iterates backward from the terminal
        step computing:

            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = delta_t + gamma * lambda * A_{t+1}
            G_t = A_t + V(s_t)

        Terminal states reset next_value and gae to 0.

        Parameters
        ----------
        gamma : float
            Discount factor in [0, 1]. Higher values weight future rewards more.
        gae_lambda : float
            GAE lambda in [0, 1]. Trades off bias (low) vs variance (high).
        """
        for rollout in self.rollouts:
            rewards = [step.reward for step in rollout]
            values = [step.value for step in rollout]
            dones = [step.done for step in rollout]

            # GAE computation (backward pass)
            gae = 0.0
            next_value = 0.0  # Terminal state value

            for t in reversed(range(len(rollout))):
                if dones[t]:
                    next_value = 0.0
                    gae = 0.0

                # TD error
                delta = rewards[t] + gamma * next_value - values[t]

                # GAE accumulation
                gae = delta + gamma * gae_lambda * gae

                # Store return and advantage
                rollout[t].return_ = gae + values[t]
                rollout[t].advantage = gae

                next_value = values[t]

    def __len__(self) -> int:
        return len(self.rollouts)


class PPOTrainer:
    """Custom PPO trainer for T5PolicyModel on quiz bowl episodes.

    Collects rollouts by running T5PolicyModel in text-observation episodes
    (via TextObservationWrapper), then updates the policy using clipped
    surrogate PPO loss with value function and entropy regularization.

    The trainer handles the complete training loop:
    1. Collect rollouts (episodes) using the current policy
    2. Compute GAE advantages
    3. Update policy with mini-batch PPO for multiple epochs
    4. Periodically validate and save checkpoints

    Parameters
    ----------
    model : T5PolicyModel
        T5 policy model to train. Should be pre-trained via supervised
        warm-start for faster convergence.
    train_questions : list[MCQuestion]
        Training set questions for rollout collection.
    val_questions : list[MCQuestion]
        Validation set questions for periodic evaluation.
    config : dict[str, Any]
        Configuration dictionary with PPO hyperparameters:

        - ``ppo_lr`` (float): Learning rate. Default 1e-5.
        - ``ppo_iterations`` (int): Number of collect-update cycles. Default 100.
        - ``ppo_batch_size`` (int): Mini-batch size for PPO updates. Default 8.
        - ``ppo_epochs_per_iter`` (int): PPO epochs per iteration. Default 4.
        - ``ppo_gamma`` (float): Discount factor. Default 0.99.
        - ``ppo_gae_lambda`` (float): GAE lambda. Default 0.95.
        - ``ppo_clip_ratio`` (float): PPO clip ratio. Default 0.2.
        - ``ppo_value_coef`` (float): Value loss coefficient. Default 0.5.
        - ``ppo_entropy_coef`` (float): Entropy bonus coefficient. Default 0.01.
        - ``ppo_max_grad_norm`` (float): Gradient clip norm. Default 0.5.
        - ``ppo_episodes_per_iter`` (int): Episodes per rollout. Default 16.
        - ``eval_interval`` (int): Validate every N iterations. Default 10.
        - ``save_interval`` (int): Save checkpoint every N iterations. Default 20.
        - ``checkpoint_dir`` (str): Base checkpoint directory. Default "checkpoints".
        - ``reward_time_penalty`` (float): Time penalty for env. Default 0.1.

    Attributes
    ----------
    model : T5PolicyModel
        The model being trained.
    optimizer : torch.optim.AdamW
        Optimizer with weight decay.
    best_val_reward : float
        Best validation reward seen so far.
    history : list[dict]
        Per-iteration training metrics.
    checkpoint_dir : Path
        Directory for saving PPO checkpoints.
    """

    def __init__(
        self,
        model: T5PolicyModel,
        train_questions: List[MCQuestion],
        val_questions: List[MCQuestion],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_questions = list(train_questions)
        self.val_questions = list(val_questions)
        self.config = config

        self.device = model.device

        # PPO hyperparameters
        self.lr = float(config.get("ppo_lr", 1e-5))
        self.iterations = int(config.get("ppo_iterations", 100))
        self.batch_size = int(config.get("ppo_batch_size", 8))
        self.epochs_per_iter = int(config.get("ppo_epochs_per_iter", 4))
        self.gamma = float(config.get("ppo_gamma", 0.99))
        self.gae_lambda = float(config.get("ppo_gae_lambda", 0.95))
        self.clip_ratio = float(config.get("ppo_clip_ratio", 0.2))
        self.value_coef = float(config.get("ppo_value_coef", 0.5))
        self.entropy_coef = float(config.get("ppo_entropy_coef", 0.01))
        self.max_grad_norm = float(config.get("ppo_max_grad_norm", 0.5))
        self.episodes_per_iter = int(config.get("ppo_episodes_per_iter", 16))
        self.eval_interval = int(config.get("eval_interval", 10))
        self.save_interval = int(config.get("save_interval", 20))
        self.reward_time_penalty = float(config.get("reward_time_penalty", 0.1))
        self.max_input_length = int(config.get("max_input_length", 512))

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=0.01
        )

        # Training state
        self.current_iteration = 0
        self.best_val_reward = -float("inf")
        self.history: List[Dict[str, Any]] = []

        # Checkpoint directory
        self.checkpoint_dir = (
            Path(config.get("checkpoint_dir", "checkpoints")) / "ppo_t5"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def collect_rollouts(self, num_episodes: int) -> RolloutBuffer:
        """Collect rollouts by running episodes with the current policy.

        Creates a TossupMCEnv + TextObservationWrapper for each sampled
        question, runs the policy until episode termination, and stores
        all steps in a RolloutBuffer. Tokenized tensors are detached and
        moved to CPU immediately to prevent GPU memory accumulation.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to collect.

        Returns
        -------
        RolloutBuffer
            Buffer containing all collected episode rollouts.
        """
        from qb_env.text_wrapper import TextObservationWrapper
        from qb_env.tossup_env import TossupMCEnv
        from models.likelihoods import TfIdfLikelihood

        self.model.eval()
        buffer = RolloutBuffer()

        # Sample questions for this iteration
        questions = random.choices(self.train_questions, k=num_episodes)

        # Build a simple TF-IDF likelihood for environment scoring
        # (The T5 policy reads text directly; likelihood is only used for
        # environment reward computation via belief updates)
        corpus = []
        for q in self.train_questions[:100]:  # Use subset for speed
            corpus.extend(q.option_profiles)
        likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

        with torch.no_grad():
            for question in questions:
                env = TossupMCEnv(
                    questions=[question],
                    likelihood_model=likelihood_model,
                    K=len(question.options),
                    reward_mode="time_penalty",
                    wait_penalty=self.reward_time_penalty,
                    belief_mode="from_scratch",
                )
                wrapped_env = TextObservationWrapper(env)

                obs, info = wrapped_env.reset()
                done = False
                rollout: List[RolloutStep] = []

                while not done:
                    # Tokenize text observation
                    inputs = self.model.tokenizer(
                        obs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_length,
                    ).to(self.device)

                    # Get action from policy
                    actions, act_info = self.model.select_action(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        deterministic=False,
                    )

                    action = actions.item()
                    value = act_info["values"].squeeze().item()
                    log_prob = act_info["log_probs"].item()

                    # Take environment step
                    next_obs, reward, terminated, truncated, step_info = (
                        wrapped_env.step(action)
                    )
                    done = terminated or truncated

                    # CRITICAL: Detach and move tensors to CPU immediately
                    # to prevent GPU memory accumulation during rollout collection
                    step = RolloutStep(
                        observation_text=obs,
                        action=action,
                        reward=reward,
                        done=done,
                        value=value,
                        log_prob=log_prob,
                        input_ids=inputs["input_ids"].detach().cpu(),
                        attention_mask=inputs["attention_mask"].detach().cpu(),
                    )
                    rollout.append(step)

                    obs = next_obs

                buffer.add_rollout(rollout)

        return buffer

    def _pad_batch(
        self, batch_steps: List[RolloutStep]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamically pad a mini-batch of steps to the max length in the batch.

        Instead of padding all sequences to the global max (512 tokens), pads
        only to the longest sequence in the current mini-batch. This typically
        saves 50%+ memory since most quiz bowl observations are 100-200 tokens.

        Parameters
        ----------
        batch_steps : list[RolloutStep]
            Mini-batch of rollout steps with stored input_ids and attention_mask.

        Returns
        -------
        input_ids : torch.Tensor
            Padded input IDs of shape ``[batch_size, max_len]``, on device.
        attention_mask : torch.Tensor
            Padded attention mask of shape ``[batch_size, max_len]``, on device.
        """
        max_len = max(step.input_ids.shape[1] for step in batch_steps)
        pad_token_id = self.model.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []

        for step in batch_steps:
            seq_len = step.input_ids.shape[1]
            if seq_len < max_len:
                pad_len = max_len - seq_len
                input_ids_padded = torch.cat(
                    [
                        step.input_ids,
                        torch.full(
                            (1, pad_len),
                            pad_token_id,
                            dtype=step.input_ids.dtype,
                        ),
                    ],
                    dim=1,
                )
                attention_mask_padded = torch.cat(
                    [
                        step.attention_mask,
                        torch.zeros(
                            (1, pad_len), dtype=step.attention_mask.dtype
                        ),
                    ],
                    dim=1,
                )
            else:
                input_ids_padded = step.input_ids
                attention_mask_padded = step.attention_mask

            padded_input_ids.append(input_ids_padded)
            padded_attention_mask.append(attention_mask_padded)

        input_ids = torch.cat(padded_input_ids).to(self.device)
        attention_mask = torch.cat(padded_attention_mask).to(self.device)

        return input_ids, attention_mask

    def update_policy(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update the policy using PPO with clipped surrogate loss.

        Computes GAE advantages, normalizes them, then runs multiple epochs
        of mini-batch PPO updates. Each update computes the clipped surrogate
        policy loss, value function MSE loss, and entropy bonus.

        Parameters
        ----------
        buffer : RolloutBuffer
            Buffer with collected rollouts (compute_returns_and_advantages
            will be called internally).

        Returns
        -------
        dict[str, float]
            Training metrics: policy_loss, value_loss, entropy, num_updates.
        """
        self.model.train()

        # Compute returns and advantages
        buffer.compute_returns_and_advantages(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        # Get all steps
        all_steps = buffer.get_all_steps()
        if not all_steps:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "num_updates": 0,
            }

        # Normalize advantages
        advantages = torch.tensor(
            [step.advantage for step in all_steps], dtype=torch.float32
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # PPO epochs
        for epoch in range(self.epochs_per_iter):
            # Shuffle step indices
            indices = np.random.permutation(len(all_steps))

            # Mini-batch updates
            for start_idx in range(0, len(all_steps), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(all_steps))
                batch_indices = indices[start_idx:end_idx]

                # Get batch steps
                batch_steps = [all_steps[i] for i in batch_indices]

                # Dynamic padding to max length in THIS batch
                input_ids, attention_mask = self._pad_batch(batch_steps)

                # Prepare batch tensors
                actions = torch.tensor(
                    [step.action for step in batch_steps],
                    dtype=torch.long,
                ).to(self.device)
                old_log_probs = torch.tensor(
                    [step.log_prob for step in batch_steps],
                    dtype=torch.float32,
                ).to(self.device)
                returns = torch.tensor(
                    [step.return_ for step in batch_steps],
                    dtype=torch.float32,
                ).to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)

                # Get new log probs, entropy, and values from current policy
                new_log_probs, entropy, values = (
                    self.model.get_action_log_probs(
                        input_ids, attention_mask, actions
                    )
                )

                # PPO clipped surrogate policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE)
                value_loss = nn.MSELoss()(values, returns)

                # Entropy bonus (negative because we maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backward pass and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "num_updates": num_updates,
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set by running deterministic episodes.

        Runs one episode per validation question with deterministic action
        selection (argmax) and computes accuracy and average reward.

        Returns
        -------
        dict[str, float]
            Validation metrics: accuracy, average_reward, avg_episode_length.
        """
        from qb_env.text_wrapper import TextObservationWrapper
        from qb_env.tossup_env import TossupMCEnv
        from models.likelihoods import TfIdfLikelihood

        self.model.eval()

        corpus = []
        for q in self.train_questions[:100]:
            corpus.extend(q.option_profiles)
        likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

        correct = 0
        total = 0
        total_reward = 0.0
        total_length = 0

        # Limit validation size for speed
        val_questions = self.val_questions[:50]

        with torch.no_grad():
            for question in val_questions:
                env = TossupMCEnv(
                    questions=[question],
                    likelihood_model=likelihood_model,
                    K=len(question.options),
                    reward_mode="time_penalty",
                    wait_penalty=self.reward_time_penalty,
                    belief_mode="from_scratch",
                )
                wrapped_env = TextObservationWrapper(env)

                obs, info = wrapped_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done:
                    inputs = self.model.tokenizer(
                        obs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_length,
                    ).to(self.device)

                    actions, act_info = self.model.select_action(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        deterministic=True,
                    )

                    action = actions.item()
                    obs, reward, terminated, truncated, step_info = (
                        wrapped_env.step(action)
                    )
                    done = terminated or truncated
                    episode_reward += reward
                    episode_length += 1

                total_reward += episode_reward
                total_length += episode_length
                total += 1

                # Check if answer was correct
                if step_info.get("correct", False) or step_info.get(
                    "forced_correct", False
                ):
                    correct += 1

        return {
            "accuracy": correct / max(1, total),
            "average_reward": total_reward / max(1, total),
            "avg_episode_length": total_length / max(1, total),
        }

    def train(self) -> Dict[str, Any]:
        """Run the full PPO training loop.

        Alternates between rollout collection and policy updates for
        ``self.iterations`` cycles. Periodically validates and saves
        checkpoints.

        Returns
        -------
        dict[str, Any]
            Training summary: best_val_reward, total_iterations.
        """
        print(f"Starting PPO training for {self.iterations} iterations")
        print(f"  Training questions: {len(self.train_questions)}")
        print(f"  Validation questions: {len(self.val_questions)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Episodes per iteration: {self.episodes_per_iter}")
        print(f"  Device: {self.device}")
        print()

        for iteration in range(self.iterations):
            self.current_iteration = iteration

            # Collect rollouts
            print(f"\nIteration {iteration + 1}/{self.iterations}")
            print("  Collecting rollouts...")
            buffer = self.collect_rollouts(self.episodes_per_iter)

            # Compute episode statistics
            episode_rewards = []
            episode_lengths = []
            for rollout in buffer.rollouts:
                episode_reward = sum(step.reward for step in rollout)
                episode_rewards.append(episode_reward)
                episode_lengths.append(len(rollout))

            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0.0

            print(f"  Avg episode reward: {avg_reward:.4f}")
            print(f"  Avg episode length: {avg_length:.2f}")

            # Update policy
            print("  Updating policy...")
            update_metrics = self.update_policy(buffer)

            print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {update_metrics['value_loss']:.4f}")
            print(f"  Entropy: {update_metrics['entropy']:.4f}")

            # Validate periodically
            if (iteration + 1) % self.eval_interval == 0:
                print("\n  Validating...")
                val_summary = self.validate()
                val_reward = val_summary.get("average_reward", 0.0)

                print(f"  Val Accuracy: {val_summary['accuracy']:.4f}")
                print(f"  Val Reward: {val_reward:.4f}")
                print(
                    f"  Val Avg Length: {val_summary['avg_episode_length']:.2f}"
                )

                # Save history
                self.history.append(
                    {
                        "iteration": iteration + 1,
                        "train_reward": float(avg_reward),
                        "train_length": float(avg_length),
                        **update_metrics,
                        "val": val_summary,
                    }
                )

                # Save best model
                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self.save_checkpoint(is_best=True)
                    print(
                        f"  -> New best validation reward: {val_reward:.4f}"
                    )

            # Save regular checkpoint
            if (iteration + 1) % self.save_interval == 0:
                self.save_checkpoint(is_best=False)
                self.save_history()

        print("\n" + "=" * 60)
        print("PPO training completed!")
        print(f"Best validation reward: {self.best_val_reward:.4f}")
        print("=" * 60)

        # Save final history
        self.save_history()

        return {
            "best_val_reward": self.best_val_reward,
            "total_iterations": self.iterations,
        }

    def save_checkpoint(self, is_best: bool = False) -> Path:
        """Save model checkpoint to disk.

        Parameters
        ----------
        is_best : bool
            If True, save to ``best_model/`` directory.

        Returns
        -------
        Path
            Path to the saved checkpoint directory.
        """
        if is_best:
            save_path = self.checkpoint_dir / "best_model"
        else:
            save_path = (
                self.checkpoint_dir
                / f"iter_{self.current_iteration + 1}"
            )

        # Use T5PolicyModel's save() method
        self.model.save(str(save_path))

        # Save training state
        state = {
            "iteration": self.current_iteration + 1,
            "best_val_reward": self.best_val_reward,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, save_path / "training_state.pt")

        print(f"  Checkpoint saved to {save_path}")
        return save_path

    def save_history(self) -> Path:
        """Save training history to JSON.

        Returns
        -------
        Path
            Path to the saved history file.
        """
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=float)
        return history_path


def run_ppo_training(
    config: Dict[str, Any],
    train_questions: List[MCQuestion],
    val_questions: List[MCQuestion],
    test_questions: Optional[List[MCQuestion]] = None,
    pretrained_model_path: Optional[str] = None,
) -> Tuple[T5PolicyModel, PPOTrainer]:
    """Run the PPO training pipeline with optional pretrained model.

    Creates or loads a T5PolicyModel, trains it with PPO on quiz bowl
    episodes, and optionally evaluates on a test set.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with model and PPO hyperparameters.
    train_questions : list[MCQuestion]
        Training set questions.
    val_questions : list[MCQuestion]
        Validation set questions.
    test_questions : list[MCQuestion] or None
        Optional test set for final evaluation.
    pretrained_model_path : str or None
        Path to a supervised pretrained checkpoint. If provided, loads the
        model from this path. Otherwise creates a new model.

    Returns
    -------
    model : T5PolicyModel
        The trained model.
    trainer : PPOTrainer
        The trainer instance with training history.
    """
    print("=" * 60)
    print("PPO TRAINING PHASE (T5 Policy)")
    print("=" * 60)

    # Load or create model
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}")
        device = config.get("device", "cpu")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        model = T5PolicyModel.load_pretrained(
            pretrained_model_path, device=device
        )
    else:
        print("Initializing new model (no pretraining)")
        model_config = {
            "model_name": config.get("model_name", "t5-large"),
            "device": config.get("device", "cpu"),
            "max_input_length": config.get("max_input_length", 512),
            "num_choices": config.get("num_choices", 4),
        }
        model = T5PolicyModel(model_config)

    # Create trainer
    trainer = PPOTrainer(
        model=model,
        train_questions=train_questions,
        val_questions=val_questions,
        config=config,
    )

    # Train
    summary = trainer.train()

    # Evaluate on test set if provided
    if test_questions is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        # Load best model if it exists
        best_model_path = trainer.checkpoint_dir / "best_model"
        if best_model_path.exists():
            print(f"Loading best model from {best_model_path}")
            model.load(str(best_model_path))

        # Run validation on test set
        # Temporarily swap val questions with test questions
        original_val = trainer.val_questions
        trainer.val_questions = list(test_questions)
        test_metrics = trainer.validate()
        trainer.val_questions = original_val

        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Avg Reward: {test_metrics['average_reward']:.4f}")

        # Save test results
        test_results = {
            "test_metrics": test_metrics,
            "training_summary": summary,
        }
        results_path = trainer.checkpoint_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2, default=float)
        print(f"Test results saved to {results_path}")

    return model, trainer
