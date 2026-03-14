"""
PPO (Proximal Policy Optimization) training for T5 policy model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

from model import T5PolicyModel
from dataset import QuizBowlDataset
from environment import QuizBowlEnvironment
from metrics import MetricsTracker, evaluate_model
from config import Config


@dataclass
class RolloutStep:
    """Single step in an episode rollout"""
    observation_text: str
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    
    # For tokenization
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None


class RolloutBuffer:
    """Buffer to store episode rollouts for PPO"""
    
    def __init__(self):
        self.rollouts = []
        self.reset()
    
    def reset(self):
        """Clear buffer"""
        self.rollouts = []
    
    def add_rollout(self, steps: List[RolloutStep]):
        """Add a complete episode rollout"""
        self.rollouts.append(steps)
    
    def get_all_steps(self) -> List[RolloutStep]:
        """Get all steps from all rollouts"""
        all_steps = []
        for rollout in self.rollouts:
            all_steps.extend(rollout)
        return all_steps
    
    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        """
        Compute discounted returns and GAE advantages for all rollouts.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        for rollout in self.rollouts:
            # Extract rewards and values
            rewards = [step.reward for step in rollout]
            values = [step.value for step in rollout]
            dones = [step.done for step in rollout]
            
            # Compute returns and advantages
            returns = []
            advantages = []
            
            # GAE computation
            gae = 0
            next_value = 0  # Terminal state has value 0
            
            for t in reversed(range(len(rollout))):
                if dones[t]:
                    next_value = 0
                    gae = 0
                
                # TD error
                delta = rewards[t] + gamma * next_value - values[t]
                
                # GAE
                gae = delta + gamma * gae_lambda * gae
                
                # Return = advantage + value
                returns.insert(0, gae + values[t])
                advantages.insert(0, gae)
                
                next_value = values[t]
            
            # Attach returns and advantages to steps
            for step, ret, adv in zip(rollout, returns, advantages):
                step.return_ = ret
                step.advantage = adv
    
    def __len__(self):
        return len(self.rollouts)


class PPOTrainer:
    """Trainer for PPO"""
    
    def __init__(self,
                 model: T5PolicyModel,
                 train_dataset: QuizBowlDataset,
                 val_dataset: QuizBowlDataset,
                 config: Config):
        """
        Initialize PPO trainer.
        
        Args:
            model: T5PolicyModel to train (should be pre-trained with supervised learning)
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration object
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.PPO_LR,
            weight_decay=0.01
        )
        
        # Training state
        self.current_iteration = 0
        self.best_val_reward = -float('inf')
        self.history = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR) / "ppo"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_rollouts(self, num_episodes: int) -> RolloutBuffer:
        """
        Collect rollouts by running episodes in the environment.
        
       Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            RolloutBuffer with collected rollouts
        """
        self.model.eval()
        buffer = RolloutBuffer()
        
        # Sample questions
        questions = self.train_dataset.get_batch(num_episodes)
        
        with torch.no_grad():
            for question in questions:
                env = QuizBowlEnvironment(
                    question,
                    reward_time_penalty=self.config.REWARD_TIME_PENALTY
                )
                
                obs = env.reset()
                done = False
                rollout = []
                
                while not done:
                    # Get text representation
                    text = env.get_text_representation(obs)
                    
                    # Tokenize
                    inputs = self.model.tokenizer(
                        text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.config.MAX_INPUT_LENGTH
                    ).to(self.device)
                    
                    # Get action and log prob
                    actions, info = self.model.select_action(
                        inputs['input_ids'],
                        inputs['attention_mask'],
                        deterministic=False
                    )
                    
                    action = actions.item()
                    value = info['values'].item()
                    
                    # Get log prob of selected action
                    log_prob = info['log_probs'].item()
                    
                    # Take step
                    next_obs, reward, done, step_info = env.step(action)
                    
                    # Store step
                    step = RolloutStep(
                        observation_text=text,
                        action=action,
                        reward=reward,
                        done=done,
                        value=value,
                        log_prob=log_prob,
                        input_ids=inputs['input_ids'].cpu(),
                        attention_mask=inputs['attention_mask'].cpu()
                    )
                    rollout.append(step)
                    
                    obs = next_obs
                
                buffer.add_rollout(rollout)
        
        return buffer
    
    def update_policy(self, buffer: RolloutBuffer) -> Dict:
        """
        Update policy using PPO.
        
        Args:
            buffer: RolloutBuffer with collected rollouts
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Compute returns and advantages
        buffer.compute_returns_and_advantages(
            gamma=self.config.PPO_GAMMA,
            gae_lambda=self.config.PPO_GAE_LAMBDA
        )
        
        # Get all steps
        all_steps = buffer.get_all_steps()
        
        # Normalize advantages
        advantages = torch.tensor([step.advantage for step in all_steps])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # PPO epochs
        for epoch in range(self.config.PPO_EPOCHS_PER_ITER):
            # Shuffle steps
            indices = np.random.permutation(len(all_steps))
            
            # Mini-batch updates
            for start_idx in range(0, len(all_steps), self.config.PPO_BATCH_SIZE):
                end_idx = min(start_idx + self.config.PPO_BATCH_SIZE, len(all_steps))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_steps = [all_steps[i] for i in batch_indices]
                
                # Prepare batch tensors with padding
                # Find max sequence length in batch
                max_len = max(step.input_ids.shape[1] for step in batch_steps)
                
                # Pad sequences
                padded_input_ids = []
                padded_attention_mask = []
                for step in batch_steps:
                    seq_len = step.input_ids.shape[1]
                    if seq_len < max_len:
                        # Pad with tokenizer's pad_token_id
                        pad_len = max_len - seq_len
                        input_ids_padded = torch.cat([
                            step.input_ids,
                            torch.full((1, pad_len), self.model.tokenizer.pad_token_id, dtype=step.input_ids.dtype)
                        ], dim=1)
                        attention_mask_padded = torch.cat([
                            step.attention_mask,
                            torch.zeros((1, pad_len), dtype=step.attention_mask.dtype)
                        ], dim=1)
                    else:
                        input_ids_padded = step.input_ids
                        attention_mask_padded = step.attention_mask
                    padded_input_ids.append(input_ids_padded)
                    padded_attention_mask.append(attention_mask_padded)
                
                input_ids = torch.cat(padded_input_ids).to(self.device)
                attention_mask = torch.cat(padded_attention_mask).to(self.device)
                actions = torch.tensor([step.action for step in batch_steps], dtype=torch.long).to(self.device)
                old_log_probs = torch.tensor([step.log_prob for step in batch_steps]).to(self.device)
                returns = torch.tensor([step.return_ for step in batch_steps]).to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)
                
                # Get new log probs and values
                new_log_probs, values, entropy = self.model.get_action_log_probs(
                    input_ids, attention_mask, actions
                )
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.PPO_CLIP_RATIO,
                    1.0 + self.config.PPO_CLIP_RATIO
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.PPO_VALUE_COEF * value_loss +
                       self.config.PPO_ENTROPY_COEF * entropy_loss)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.PPO_MAX_GRAD_NORM)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Return average metrics
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'num_updates': num_updates
        }
    
    def validate(self) -> Dict:
        """Validate on validation set"""
        metrics = evaluate_model(
            self.model,
            self.val_dataset,
            device=self.device,
            deterministic=True
        )
        return metrics.get_summary()
    
    def train(self):
        """Run full PPO training"""
        print(f"Starting PPO training for {self.config.PPO_ITERATIONS} iterations")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.config.PPO_BATCH_SIZE}")
        print(f"Device: {self.device}")
        print()
        
        for iteration in range(self.config.PPO_ITERATIONS):
            self.current_iteration = iteration
            
            # Collect rollouts
            print(f"\nIteration {iteration + 1}/{self.config.PPO_ITERATIONS}")
            print("Collecting rollouts...")
            buffer = self.collect_rollouts(self.config.PPO_BATCH_SIZE)
            
            # Compute episode statistics
            episode_rewards = []
            episode_lengths = []
            for rollout in buffer.rollouts:
                episode_reward = sum(step.reward for step in rollout)
                episode_rewards.append(episode_reward)
                episode_lengths.append(len(rollout))
            
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            
            print(f"Avg episode reward: {avg_reward:.4f}")
            print(f"Avg episode length: {avg_length:.2f}")
            
            # Update policy
            print("Updating policy...")
            update_metrics = self.update_policy(buffer)
            
            print(f"Policy loss: {update_metrics['policy_loss']:.4f}")
            print(f"Value loss: {update_metrics['value_loss']:.4f}")
            print(f"Entropy: {update_metrics['entropy']:.4f}")
            
            # Validate periodically
            if (iteration + 1) % self.config.EVAL_INTERVAL == 0:
                print("\nValidating...")
                val_summary = self.validate()
                val_reward = val_summary.get('average_reward', 0)
                
                print(f"Val Accuracy: {val_summary['accuracy']:.4f}")
                print(f"Val Reward: {val_reward:.4f}")
                print(f"Val ECE: {val_summary.get('ece', 0):.4f}")
                print(f"Val Buzz Position: {val_summary.get('average_buzz_position', 0):.2f}")
                
                # Save history
                self.history.append({
                    'iteration': iteration + 1,
                    'train_reward': avg_reward,
                    'train_length': avg_length,
                    **update_metrics,
                    'val': val_summary
                })
                
                # Save best model
                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self.save_checkpoint(is_best=True)
                    print(f"New best validation reward: {val_reward:.4f}")
            
            # Save regular checkpoint
            if (iteration + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(is_best=False)
                self.save_history()
        
        print("\n" + "=" * 60)
        print("PPO training completed!")
        print(f"Best validation reward: {self.best_val_reward:.4f}")
        print("=" * 60)
        
        # Save final history
        self.save_history()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if is_best:
            save_path = self.checkpoint_dir / "best_model"
        else:
            save_path = self.checkpoint_dir / f"iter_{self.current_iteration + 1}"
        
        # Use T5PolicyModel's save() method
        self.model.save(str(save_path))
        
        # Save training state
        state = {
            'iteration': self.current_iteration + 1,
            'best_val_reward': self.best_val_reward,
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, save_path / "training_state.pt")
        
        print(f"Checkpoint saved to {save_path}")
    
    def save_history(self):
        """Save training history"""
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def run_ppo_training(config: Config,
                    train_dataset: QuizBowlDataset,
                    val_dataset: QuizBowlDataset,
                    test_dataset: QuizBowlDataset = None,
                    pretrained_model_path: str = None):
    """
    Run PPO training pipeline.
    
    Args:
        config: Configuration object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset for final evaluation
        pretrained_model_path: Path to pretrained supervised model
    """
    print("=" * 60)
    print("PPO TRAINING PHASE")
    print("=" * 60)
    
    # Load model
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}")
        model = T5PolicyModel.load_pretrained(pretrained_model_path, device=config.DEVICE)
    else:
        print("Initializing new model (no pretraining)")
        model = T5PolicyModel(config)
    
    model.to(config.DEVICE)
    
    # Create trainer
    trainer = PPOTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    trainer.train()
    
    # Evaluate on test set if provided
    if test_dataset is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)
        
        # Load best model if it exists, otherwise use current model
        best_model_path = trainer.checkpoint_dir / "best_model"
        if best_model_path.exists():
            print(f"\nLoading best model from {best_model_path}")
            model = T5PolicyModel.load_pretrained(str(best_model_path), device=config.DEVICE)
            model.to(config.DEVICE)
        else:
            print("\nNo best model found, using current model for evaluation")
            model = trainer.model
        
        # Evaluate
        print("\nRunning full evaluation...")
        metrics = evaluate_model(model, test_dataset, device=config.DEVICE)
        metrics.print_summary()
        
        # Save test results
        test_results = metrics.get_summary()
        results_path = trainer.checkpoint_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\nTest results saved to {results_path}")
    
    return model, trainer


if __name__ == "__main__":
    from dataset import setup_datasets
    
    # Load config
    config = Config()
    config.print_config()
    
    # Setup datasets
    train_dataset, val_dataset, test_dataset = setup_datasets(config)
    
    # Path to supervised pretrained model
    supervised_model_path = Path(config.CHECKPOINT_DIR) / "supervised" / "best_model"
    
    if supervised_model_path.exists():
        print(f"\nFound supervised pretrained model at {supervised_model_path}")
    else:
        print(f"\nWARNING: Supervised model not found at {supervised_model_path}")
        print("Consider running supervised training first!")
        supervised_model_path = None
    
    # Run PPO training
    model, trainer = run_ppo_training(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        pretrained_model_path=str(supervised_model_path) if supervised_model_path else None
    )
