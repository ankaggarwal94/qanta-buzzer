"""
T5-based Policy Model for Quiz Bowl RL Agent

Implements T5PolicyModel with a custom PolicyHead containing three independent
heads (wait/answer/value) for end-to-end text-based policy learning. This
provides an alternative to the MLP policy trained on belief features
(Phase 4 approach).

Architecture overview:

    Text input  -->  T5 Encoder  -->  Mean Pooling  -->  PolicyHead
                                                          |-- Wait head (2)
                                                          |-- Answer head (K)
                                                          |-- Value head (1)

The T5 encoder produces contextual embeddings from tokenized text. Mean pooling
(attention-masked) reduces the variable-length sequence to a fixed-size vector.
The PolicyHead then produces three independent outputs:

- **Wait logits** [B, 2]: probability of waiting vs answering now
- **Answer logits** [B, K]: probability of selecting each answer option
- **Value estimate** [B, 1]: state value for PPO advantage computation

Action space maps to the TossupMCEnv convention:
    0 = WAIT (wait head selects "wait")
    1..K = SELECT answer i-1 (wait head selects "answer now", answer head picks i-1)

Ported from qanta-buzzer reference implementation (model.py) with these changes:
    - T5EncoderModel replaces T5ForConditionalGeneration (2x faster, 50% less memory)
    - T5TokenizerFast replaces T5Tokenizer (3-5x faster tokenization via Rust backend)
    - Config dict replaces qanta-buzzer's Config class for unified codebase compatibility
    - NumPy-style docstrings added throughout
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """Custom policy head with three independent output heads.

    Attached to a T5 encoder's pooled output, this module produces the three
    outputs needed for actor-critic RL in the quiz bowl POMDP: a binary
    wait/answer-now decision, a K-way answer selection, and a scalar value
    estimate.

    All three heads are fully independent (no shared hidden layers beyond the
    encoder), using the same pattern: Linear -> ReLU -> Dropout -> Linear.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the input from the T5 encoder's pooled output.
        Default 1024 matches T5-large (``d_model``). Use 512 for t5-small,
        768 for t5-base.
    num_choices : int
        Number of answer options (K). Default 4 for quiz bowl MC questions.

    Attributes
    ----------
    wait_head : nn.Sequential
        Binary head producing [wait, answer_now] logits.
    answer_head : nn.Sequential
        Multi-class head producing logits over K answer choices.
    value_head : nn.Sequential
        Scalar head producing state value estimate.
    """

    def __init__(self, hidden_size: int = 1024, num_choices: int = 4) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_choices = num_choices

        # Wait/continue decision head (binary: wait vs answer_now)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),  # [wait, answer_now]
        )

        # Answer selection head (over K choices)
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_choices),
        )

        # Value head (state value estimate for PPO)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(
        self, encoder_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all three heads.

        Parameters
        ----------
        encoder_hidden_state : torch.Tensor
            Pooled encoder output of shape ``[batch_size, hidden_size]``.

        Returns
        -------
        wait_logits : torch.Tensor
            Shape ``[batch_size, 2]`` -- logits for [wait, answer_now].
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer options.
        values : torch.Tensor
            Shape ``[batch_size, 1]`` -- state value estimates.
        """
        wait_logits = self.wait_head(encoder_hidden_state)
        answer_logits = self.answer_head(encoder_hidden_state)
        values = self.value_head(encoder_hidden_state)

        return wait_logits, answer_logits, values


class T5PolicyModel(nn.Module):
    """T5 encoder with custom policy head for end-to-end RL.

    Combines a pre-trained T5 encoder with a ``PolicyHead`` to produce policy
    outputs directly from text observations. This is the alternative approach
    to Phase 4's MLP policy, which operates on numeric belief features.

    The model processes text in three stages:

    1. **Tokenization**: Text is tokenized with ``T5TokenizerFast`` (Rust-backed
       for speed) with padding and truncation.
    2. **Encoding**: ``T5EncoderModel`` produces contextual hidden states
       ``[B, seq_len, d_model]``.
    3. **Pooling + Heads**: Attention-masked mean pooling reduces to
       ``[B, d_model]``, then PolicyHead produces wait/answer/value outputs.

    Action space follows TossupMCEnv convention:
        - 0 = WAIT
        - 1..K = SELECT answer (i-1)

    Combined actions are treated as a factorized policy:
        - ``P(WAIT) = p_wait``
        - ``P(BUZZ_i) = p_buzz * p_ans(i)``

    The answer distribution only contributes when the policy chooses to buzz.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with the following keys:

        - ``model_name`` (str): HuggingFace T5 model identifier.
          Default ``"t5-large"``. Options: ``"t5-small"``, ``"t5-base"``,
          ``"t5-large"``.
        - ``device`` (str): Torch device. Default auto-detects
          (cuda > mps > cpu).
        - ``max_input_length`` (int): Maximum token sequence length.
          Default 512.
        - ``num_choices`` (int): Number of answer options (K). Default 4.

    Attributes
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    device : torch.device
        Computation device.
    encoder : T5EncoderModel
        Pre-trained T5 encoder.
    tokenizer : T5TokenizerFast
        Fast T5 tokenizer.
    policy_head : PolicyHead
        Custom three-head policy module.
    max_input_length : int
        Maximum token sequence length for tokenization.

    Examples
    --------
    >>> config = {"model_name": "t5-small", "device": "cpu", "num_choices": 4}
    >>> model = T5PolicyModel(config)
    >>> texts = ["CLUES: first president | CHOICES: (1) Washington (2) Jefferson"]
    >>> wait_logits, answer_logits, values = model(texts)
    >>> wait_logits.shape
    torch.Size([1, 2])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        from transformers import T5EncoderModel, T5TokenizerFast

        self.config = config
        model_name = config.get("model_name", "t5-large")
        self.max_input_length = config.get("max_input_length", 512)
        num_choices = config.get("num_choices", 4)

        # Auto-detect device
        default_device = "cpu"
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            default_device = "mps"
        self.device = torch.device(config.get("device", default_device))

        # Load T5 encoder only (not full T5ForConditionalGeneration)
        # This is 2x faster and uses 50% less memory since the decoder is unused
        print(f"Loading T5 encoder: {model_name}")
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)

        # Get hidden size from T5 config (512 for small, 768 for base, 1024 for large)
        hidden_size = self.encoder.config.d_model

        # Custom policy head
        self.policy_head = PolicyHead(
            hidden_size=hidden_size,
            num_choices=num_choices,
        )

        # Move to device
        self.to(self.device)

        # Print model info
        self._print_model_info()

    def _print_model_info(self) -> None:
        """Print model architecture summary and parameter counts."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        total_params = encoder_params + policy_params

        print("Model Architecture:")
        print(f"  T5 encoder parameters: {encoder_params:,}")
        print(f"  Policy head parameters: {policy_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {self.device}")

    def encode_input(
        self,
        text_inputs: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs using T5TokenizerFast.

        Parameters
        ----------
        text_inputs : list[str]
            List of input text strings to tokenize.
        max_length : int or None
            Maximum sequence length. If None, uses ``self.max_input_length``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with ``"input_ids"`` and ``"attention_mask"`` tensors,
            both of shape ``[batch_size, seq_len]``, moved to ``self.device``.
        """
        if max_length is None:
            max_length = self.max_input_length

        encoding = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {k: v.to(self.device) for k, v in encoding.items()}

    def get_encoder_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute T5 encoder output and pool to a fixed-size vector.

        Uses attention-masked mean pooling: sum hidden states where attention
        mask is 1, divide by number of non-padding tokens. This ensures
        padding tokens contribute zero to the pooled representation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]`` (1 for real
            tokens, 0 for padding).

        Returns
        -------
        torch.Tensor
            Pooled encoder output of shape ``[batch_size, hidden_size]``.
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # encoder_outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        hidden_states = encoder_outputs.last_hidden_state

        # Attention-masked mean pooling over sequence dimension
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / sum_mask

        return pooled_output

    def forward(
        self,
        text_inputs: List[str],
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: tokenize, encode, pool, then apply policy head.

        Parameters
        ----------
        text_inputs : list[str]
            List of text observations (e.g.,
            ``"CLUES: clue1 clue2 | CHOICES: (1) ans1 (2) ans2"``).
        return_value : bool
            If True, return value estimates. If False, values is None.

        Returns
        -------
        wait_logits : torch.Tensor
            Shape ``[batch_size, 2]`` -- logits for [wait, answer_now].
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer options.
        values : torch.Tensor or None
            Shape ``[batch_size, 1]`` if return_value is True, else None.
        """
        # Encode inputs
        encoding = self.encode_input(text_inputs)

        # Get pooled encoder output
        pooled_output = self.get_encoder_output(
            encoding["input_ids"],
            encoding["attention_mask"],
        )

        # Pass through policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)

        if not return_value:
            values = None

        return wait_logits, answer_logits, values

    def predict_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict answer choice for supervised training.

        Only uses the answer head (wait and value heads are ignored). This is
        the interface for supervised warm-start training where the model learns
        to select the correct answer from complete questions.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``.

        Returns
        -------
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer choices.
        predictions : torch.Tensor
            Shape ``[batch_size]`` -- predicted answer indices (argmax).
        """
        # Get encoder output
        pooled_output = self.get_encoder_output(input_ids, attention_mask)

        # Get answer logits from policy head
        _, answer_logits, _ = self.policy_head(pooled_output)

        # Get predictions
        predictions = torch.argmax(answer_logits, dim=-1)

        return answer_logits, predictions

    def _joint_action_log_prob(
        self,
        wait_logits: torch.Tensor,
        answer_logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute factorized log-probabilities for flat WAIT/BUZZ actions.

        Parameters
        ----------
        wait_logits : torch.Tensor
            Binary logits of shape ``[batch_size, 2]`` for [WAIT, BUZZ].
        answer_logits : torch.Tensor
            Answer logits of shape ``[batch_size, K]``.
        actions : torch.Tensor
            Flat actions of shape ``[batch_size]`` where 0 = WAIT and
            1..K = BUZZ with answer index action-1.

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape ``[batch_size]``.
        """
        wait_log_probs = F.log_softmax(wait_logits, dim=-1)
        answer_log_probs = F.log_softmax(answer_logits, dim=-1)

        wait_actions = (actions > 0).long()
        answer_actions = torch.clamp(actions - 1, min=0)

        selected_wait = wait_log_probs.gather(1, wait_actions.unsqueeze(-1)).squeeze(-1)
        selected_answer = answer_log_probs.gather(
            1, answer_actions.unsqueeze(-1)
        ).squeeze(-1)

        return torch.where(actions == 0, selected_wait, selected_wait + selected_answer)

    def _joint_entropy(
        self,
        wait_logits: torch.Tensor,
        answer_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chain-rule entropy for the factorized wait/answer policy.

        Returns ``H(wait) + p_buzz * H(answer)`` for each example.
        """
        wait_probs = F.softmax(wait_logits, dim=-1)
        wait_log_probs = F.log_softmax(wait_logits, dim=-1)
        answer_probs = F.softmax(answer_logits, dim=-1)
        answer_log_probs = F.log_softmax(answer_logits, dim=-1)

        wait_entropy = -(wait_probs * wait_log_probs).sum(dim=-1)
        answer_entropy = -(answer_probs * answer_log_probs).sum(dim=-1)
        return wait_entropy + wait_probs[:, 1] * answer_entropy

    def select_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Select flat WAIT/BUZZ actions from the factorized policy.

        Produces combined actions following TossupMCEnv convention:
        0 = WAIT, 1..K = SELECT answer 0..K-1. Under the factorized policy:

        - ``P(WAIT) = p_wait``
        - ``P(BUZZ_i) = p_buzz * p_ans(i)``

        Answer sampling only occurs for examples that actually buzz.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``.
        deterministic : bool
            If True, use argmax instead of sampling.
        temperature : float
            Temperature for softmax. Higher values increase randomness.
            Default 1.0 (no scaling).

        Returns
        -------
        combined_actions : torch.Tensor
            Shape ``[batch_size]`` -- combined actions (0 = WAIT, 1..K = SELECT).
        info : dict[str, Any]
            Dictionary with keys:

            - ``wait_logits``: raw wait head output
            - ``answer_logits``: raw answer head output
            - ``wait_probs``: softmax of wait logits
            - ``answer_probs``: softmax of answer logits
            - ``wait_actions``: sampled wait decisions (0 or 1)
            - ``answer_actions``: sampled answer indices (0..K-1)
            - ``values``: value estimates
            - ``log_probs``: total log probability of the combined action
        """
        with torch.no_grad():
            pooled_output = self.get_encoder_output(input_ids, attention_mask)
            wait_logits, answer_logits, values = self.policy_head(pooled_output)

            wait_logits_scaled = wait_logits / temperature
            answer_logits_scaled = answer_logits / temperature

            wait_probs = F.softmax(wait_logits_scaled, dim=-1)
            answer_probs = F.softmax(answer_logits_scaled, dim=-1)
            flat_action_probs = torch.cat(
                [wait_probs[:, :1], wait_probs[:, 1:2] * answer_probs],
                dim=-1,
            )

            if deterministic:
                combined_actions = torch.argmax(flat_action_probs, dim=-1)
                wait_actions = (combined_actions > 0).long()
                answer_actions = torch.clamp(combined_actions - 1, min=0)
            else:
                wait_actions = torch.distributions.Categorical(wait_probs).sample()
                answer_actions = torch.argmax(answer_probs, dim=-1)
                buzz_mask = wait_actions == 1
                if buzz_mask.any():
                    buzz_answers = torch.distributions.Categorical(
                        answer_probs[buzz_mask]
                    ).sample()
                    answer_actions = answer_actions.clone()
                    answer_actions[buzz_mask] = buzz_answers
                combined_actions = torch.where(
                    wait_actions == 0,
                    torch.zeros_like(wait_actions),
                    1 + answer_actions,
                )

            log_probs = self._joint_action_log_prob(
                wait_logits_scaled, answer_logits_scaled, combined_actions
            )

            combined_actions = torch.where(
                wait_actions == 0,
                torch.zeros_like(wait_actions),
                1 + answer_actions,
            )

            info = {
                "wait_logits": wait_logits,
                "answer_logits": answer_logits,
                "wait_probs": wait_probs,
                "answer_probs": answer_probs,
                "wait_actions": wait_actions,
                "answer_actions": answer_actions,
                "values": values,
                "log_probs": log_probs,
            }

            return combined_actions, info

    def get_action_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy for given actions.

        Used during PPO training to evaluate old actions under the current
        policy. Combined actions follow the factorized semantics:

        - ``P(WAIT) = p_wait``
        - ``P(BUZZ_i) = p_buzz * p_ans(i)``

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``.
        actions : torch.Tensor
            Combined actions of shape ``[batch_size]``. Values in {0, 1, ..., K}.

        Returns
        -------
        log_probs : torch.Tensor
            Shape ``[batch_size]`` -- total log probability of each action.
        entropy : torch.Tensor
            Shape ``[batch_size]`` -- chain-rule entropy for the factorized policy.
        values : torch.Tensor
            Shape ``[batch_size]`` -- value estimates (squeezed).
        """
        pooled_output = self.get_encoder_output(input_ids, attention_mask)
        wait_logits, answer_logits, values = self.policy_head(pooled_output)

        log_probs = self._joint_action_log_prob(wait_logits, answer_logits, actions)
        entropy = self._joint_entropy(wait_logits, answer_logits)

        return log_probs, entropy, values.squeeze(-1)

    def save(self, save_dir: str) -> None:
        """Save model checkpoint to disk.

        Saves three components:
        1. T5 encoder weights and config (HuggingFace format)
        2. Tokenizer files (HuggingFace format)
        3. Policy head state dict (PyTorch format as ``policy_head.pt``)

        Parameters
        ----------
        save_dir : str
            Directory path to save the checkpoint. Created if it doesn't exist.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save T5 encoder
        self.encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save policy head
        policy_head_path = os.path.join(save_dir, "policy_head.pt")
        torch.save(self.policy_head.state_dict(), policy_head_path)

        print(f"Model saved to {save_dir}")

    def load(self, load_dir: str) -> None:
        """Load model checkpoint from disk.

        Loads T5 encoder weights, tokenizer, and policy head state dict from
        the specified directory. The model is moved to ``self.device`` after
        loading.

        Parameters
        ----------
        load_dir : str
            Directory containing a previously saved checkpoint.

        Raises
        ------
        FileNotFoundError
            If ``policy_head.pt`` is not found in ``load_dir``.
        """
        from transformers import T5EncoderModel, T5TokenizerFast

        # Load T5 encoder
        self.encoder = T5EncoderModel.from_pretrained(load_dir)
        self.tokenizer = T5TokenizerFast.from_pretrained(load_dir)

        # Load policy head
        policy_head_path = os.path.join(load_dir, "policy_head.pt")
        self.policy_head.load_state_dict(
            torch.load(policy_head_path, map_location=self.device, weights_only=True)
        )

        self.to(self.device)
        print(f"Model loaded from {load_dir}")

    @classmethod
    def load_pretrained(
        cls,
        load_dir: str,
        device: Optional[str] = None,
    ) -> "T5PolicyModel":
        """Load a pretrained model from a directory.

        Class method that creates a new T5PolicyModel instance and loads
        weights from a saved checkpoint.

        Parameters
        ----------
        load_dir : str
            Directory containing a previously saved checkpoint.
        device : str or None
            Device to load model on (e.g., ``"cpu"``, ``"cuda"``, ``"mps"``).
            If None, auto-detects.

        Returns
        -------
        T5PolicyModel
            A loaded model instance ready for inference.
        """
        from transformers import T5Config

        # Validate checkpoint integrity (lightweight — config JSON only)
        T5Config.from_pretrained(load_dir, local_files_only=True)

        # Infer num_choices from policy head state dict
        policy_head_path = os.path.join(load_dir, "policy_head.pt")
        policy_head_state = torch.load(
            policy_head_path, map_location="cpu", weights_only=True
        )
        # answer_head final linear layer weight shape is [num_choices, hidden_dim]
        num_choices = policy_head_state["answer_head.3.weight"].shape[0]

        config = {
            "model_name": load_dir,
            "num_choices": num_choices,
        }
        if device is not None:
            config["device"] = device

        model = cls(config)
        model.load(load_dir)
        return model
