"""
T5-based policy model for Quiz Bowl RL agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import Config


class PolicyHead(nn.Module):
    """
    Custom policy head for the T5 model.
    Outputs: wait probability, answer distribution over choices, value estimate.
    """
    
    def __init__(self, hidden_size: int = 1024, num_choices: int = 4):
        """
        Initialize policy head.
        
        Args:
            hidden_size: Size of T5 hidden states
            num_choices: Number of answer choices
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_choices = num_choices
        
        # Wait/continue decision head (binary)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # [wait, answer_now]
        )
        
        # Answer selection head (over choices)
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_choices)
        )
        
        # Value head (state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, encoder_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy head.
        
        Args:
            encoder_hidden_state: [batch_size, hidden_size] - pooled encoder output
            
        Returns:
            wait_logits: [batch_size, 2] - logits for wait/answer
            answer_logits: [batch_size, num_choices] - logits for answer selection
            value: [batch_size, 1] - value estimate
        """
        wait_logits = self.wait_head(encoder_hidden_state)
        answer_logits = self.answer_head(encoder_hidden_state)
        value = self.value_head(encoder_hidden_state)
        
        return wait_logits, answer_logits, value


class T5PolicyModel(nn.Module):
    """
    T5-based policy model that combines T5 encoder with custom policy head.
    """
    
    def __init__(self, config: Config):
        """
        Initialize T5 policy model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Load T5 model and tokenizer
        print(f"Loading T5 model: {config.MODEL_NAME}")
        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
        self.tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
        
        # Get hidden size from T5 config
        hidden_size = self.t5_model.config.d_model
        
        # Custom policy head
        self.policy_head = PolicyHead(
            hidden_size=hidden_size,
            num_choices=config.NUM_ANSWER_CHOICES
        )
        
        # Move to device
        self.to(self.device)
        
        # Print model size
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model architecture and parameter count"""
        t5_params = sum(p.numel() for p in self.t5_model.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        total_params = t5_params + policy_params
        
        print(f"Model Architecture:")
        print(f"  T5 parameters: {t5_params:,}")
        print(f"  Policy head parameters: {policy_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {self.device}")
    
    def encode_input(self, 
                     text_inputs: List[str],
                     max_length: int = None) -> Dict[str, torch.Tensor]:
        """
        Encode text inputs using T5 tokenizer.
        
        Args:
            text_inputs: List of input strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.config.MAX_INPUT_LENGTH
        
        encoding = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def get_encoder_output(self, 
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get T5 encoder output and pool to fixed-size representation.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        # Get encoder outputs
        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # encoder_outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        hidden_states = encoder_outputs.last_hidden_state
        
        # Mean pooling over sequence dimension (masked)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / sum_mask
        
        return pooled_output
    
    def forward(self,
                text_inputs: List[str],
                return_value: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            text_inputs: List of text inputs (observations)
            return_value: Whether to return value estimates
            
        Returns:
            wait_logits: [batch_size, 2]
            answer_logits: [batch_size, num_choices]
            values: [batch_size, 1] or None
        """
        # Encode inputs
        encoding = self.encode_input(text_inputs)
        
        # Get encoder output
        pooled_output = self.get_encoder_output(
            encoding['input_ids'],
            encoding['attention_mask']
        )
        
        # Pass through policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)
        
        if not return_value:
            values = None
        
        return wait_logits, answer_logits, values
    
    def predict_answer(self,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict answer choice (for supervised training).
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            answer_logits: [batch_size, num_choices] - logits over answer choices
            predictions: [batch_size] - predicted answer indices (argmax)
        """
        # Get encoder output
        pooled_output = self.get_encoder_output(input_ids, attention_mask)
        
        # Get answer logits from policy head
        _, answer_logits, _ = self.policy_head(pooled_output)
        
        # Get predictions
        predictions = torch.argmax(answer_logits, dim=-1)
        
        return answer_logits, predictions
    
    def select_action(self,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     deterministic: bool = False,
                     temperature: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Select actions based on current policy.
        
        Args:
            input_ids: [batch_size, seq_len] - tokenized inputs
            attention_mask: [batch_size, seq_len] - attention mask
            deterministic: Use argmax instead of sampling
            temperature: Temperature for sampling
            
        Returns:
            actions: [batch_size] - combined actions (0=WAIT, 1-4=SELECT answer 0-3)
            info: Dictionary with action details including logits, probabilities, etc.
        """
        with torch.no_grad():
            # Get encoder output
            pooled_output = self.get_encoder_output(input_ids, attention_mask)
            
            # Get logits from policy head
            wait_logits, answer_logits, values = self.policy_head(pooled_output)
            
            # Apply temperature
            wait_logits = wait_logits / temperature
            answer_logits = answer_logits / temperature
            
            # Get probabilities
            wait_probs = F.softmax(wait_logits, dim=-1)
            answer_probs = F.softmax(answer_logits, dim=-1)
            
            if deterministic:
                # Take argmax
                wait_actions = torch.argmax(wait_probs, dim=-1)
                answer_actions = torch.argmax(answer_probs, dim=-1)
            else:
                # Sample from distribution
                wait_dist = torch.distributions.Categorical(wait_probs)
                answer_dist = torch.distributions.Categorical(answer_probs)
                
                wait_actions = wait_dist.sample()
                answer_actions = answer_dist.sample()
            
            # Compute log probabilities
            wait_log_probs = F.log_softmax(wait_logits, dim=-1)
            answer_log_probs = F.log_softmax(answer_logits, dim=-1)
            
            selected_wait_log_probs = wait_log_probs.gather(1, wait_actions.unsqueeze(-1)).squeeze(-1)
            selected_answer_log_probs = answer_log_probs.gather(1, answer_actions.unsqueeze(-1)).squeeze(-1)
            
            # Total log prob is sum (since actions are independent)
            log_probs = selected_wait_log_probs + selected_answer_log_probs
            
            # Combine wait and answer into single action
            # If wait_action == 0: action = 0 (WAIT)
            # If wait_action == 1: action = 1 + answer_action (SELECT answer 0-3)
            combined_actions = torch.where(
                wait_actions == 0,
                torch.zeros_like(wait_actions),
                1 + answer_actions
            )
            
            # Create info dict
            info = {
                'wait_logits': wait_logits,
                'answer_logits': answer_logits,
                'wait_probs': wait_probs,
                'answer_probs': answer_probs,
                'wait_actions': wait_actions,
                'answer_actions': answer_actions,
                'values': values,
                'log_probs': log_probs
            }
            
            return combined_actions, info
    
    def get_action_log_probs(self,
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and values for given actions.
        Used during PPO training.
        
        Args:
            input_ids: [batch_size, seq_len] - tokenized inputs
            attention_mask: [batch_size, seq_len] - attention mask
            actions: [batch_size] - combined actions (0=WAIT, 1-4=SELECT answer 0-3)
            
        Returns:
            log_probs: [batch_size] - log probs of actions
            entropy: [batch_size] - action entropy
            values: [batch_size] - value estimates
        """
        # Decompose combined actions into wait and answer actions
        # action 0 -> wait=0 (WAIT)
        # action 1-4 -> wait=1, answer=0-3 (SELECT answer)
        wait_actions = (actions > 0).long()
        answer_actions = torch.clamp(actions - 1, min=0)  # Map 1-4 to 0-3, keep 0 as 0
        
        # Get encoder output
        pooled_output = self.get_encoder_output(input_ids, attention_mask)
        
        # Get logits from policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)
        
        # Compute log probabilities
        wait_log_probs = F.log_softmax(wait_logits, dim=-1)
        answer_log_probs = F.log_softmax(answer_logits, dim=-1)
        
        # Get log probs for selected actions
        selected_wait_log_probs = wait_log_probs.gather(1, wait_actions.unsqueeze(-1)).squeeze(-1)
        selected_answer_log_probs = answer_log_probs.gather(1, answer_actions.unsqueeze(-1)).squeeze(-1)
        
        # Total log prob
        log_probs = selected_wait_log_probs + selected_answer_log_probs
        
        # Compute entropy
        wait_probs = F.softmax(wait_logits, dim=-1)
        answer_probs = F.softmax(answer_logits, dim=-1)
        
        wait_entropy = -(wait_probs * wait_log_probs).sum(dim=-1)
        answer_entropy = -(answer_probs * answer_log_probs).sum(dim=-1)
        
        entropy = wait_entropy + answer_entropy
        
        return log_probs, entropy, values.squeeze(-1)
    
    def save(self, save_dir: str):
        """Save model checkpoint"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save T5 model
        self.t5_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save policy head
        policy_head_path = os.path.join(save_dir, 'policy_head.pt')
        torch.save(self.policy_head.state_dict(), policy_head_path)
        
        print(f"Model saved to {save_dir}")
    
    def load(self, load_dir: str):
        """Load model checkpoint"""
        import os
        
        # Load T5 model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(load_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(load_dir)
        
        # Load policy head
        policy_head_path = os.path.join(load_dir, 'policy_head.pt')
        self.policy_head.load_state_dict(torch.load(policy_head_path, map_location=self.device))
        
        self.to(self.device)
        print(f"Model loaded from {load_dir}")
    
    @classmethod
    def load_pretrained(cls, load_dir: str, device: str = None):
        """
        Load a pretrained model from a directory.
        
        Args:
            load_dir: Directory containing saved model
            device: Device to load model on (e.g., 'cpu', 'cuda', 'mps')
            
        Returns:
            Loaded T5PolicyModel instance
        """
        import os
        from config import Config
        
        # Create a temporary config with the appropriate device
        config = Config()
        if device:
            config.DEVICE = device
        
        # Load T5 model and tokenizer from the directory
        # This will determine the model name from the saved config
        t5_model = T5ForConditionalGeneration.from_pretrained(load_dir)
        tokenizer = T5Tokenizer.from_pretrained(load_dir)
        
        # Create new model instance
        model = cls(config)
        model.t5_model = t5_model
        model.tokenizer = tokenizer
        
        # Load policy head
        policy_head_path = os.path.join(load_dir, 'policy_head.pt')
        if os.path.exists(policy_head_path):
            model.policy_head.load_state_dict(
                torch.load(policy_head_path, map_location=torch.device(config.DEVICE))
            )
        
        model.to(config.DEVICE)
        print(f"Model loaded from {load_dir}")
        
        return model
