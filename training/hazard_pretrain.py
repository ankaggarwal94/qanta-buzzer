"""Hazard pretraining bridge utilities for stopping-aware warm starts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.nn.functional as F

from models.t5_policy import T5PolicyModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qb_data.mc_builder import MCQuestion


@dataclass
class HazardBatchOutput:
    """Container for hazard-bridge intermediate tensors."""

    stop_probs: torch.Tensor
    survival: torch.Tensor
    stop_mass: torch.Tensor
    nll_per_prefix: torch.Tensor
    loss: torch.Tensor


def compute_survival_terms(stop_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute survival and stop-mass terms from per-prefix stop probabilities."""
    stay_probs = 1.0 - stop_probs
    batch_size, steps = stop_probs.shape
    survival = torch.ones(
        (batch_size, steps + 1), dtype=stop_probs.dtype, device=stop_probs.device
    )
    if steps > 0:
        survival[:, 1:] = torch.cumprod(stay_probs, dim=1)
    stop_mass = survival[:, :-1] * stop_probs
    return survival, stop_mass


def hazard_expected_nll_loss(
    stop_probs: torch.Tensor,
    nll_per_prefix: torch.Tensor,
    beta_terminal: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the hazard-bridge expected NLL loss prior to PPO."""
    survival, stop_mass = compute_survival_terms(stop_probs)
    weighted_nll = stop_mass * nll_per_prefix
    if mask is not None:
        weighted_nll = weighted_nll * mask
    seq_loss = weighted_nll.sum(dim=1) + beta_terminal * survival[:, -1]
    return seq_loss.mean()


def _format_choices(options: List[str]) -> str:
    """Format answer options as ``"(1) opt1 (2) opt2 ..."``.

    Mirrors the canonical option formatting used by the PPO text observation
    (``qb_env/text_wrapper.py``) and the supervised trainer
    (``training/train_supervised_t5.py::format_question_text``) so the
    hazard-bridge inputs match the PPO observation distribution exactly
    (AP-031 format-pinning: keep this in sync with those producers).
    """
    return " ".join(f"({i + 1}) {opt}" for i, opt in enumerate(options))


def run_hazard_pretrain(
    config: Dict[str, Any],
    train_questions: List["MCQuestion"],
    *,
    pretrained_model_path: str,
    beta_terminal: float = 1.0,
    freeze_answer_head: bool = False,
) -> str:
    """Warm-start the buzz/stop head with the hazard survival loss before PPO.

    Loads the supervised checkpoint, then for each training question forwards
    the T5 policy over that question's ``cumulative_prefixes`` to obtain
    per-prefix stop-probabilities (from the wait head) and per-prefix answer-NLL
    (from the answer head vs. ``gold_index``), and minimizes
    :func:`hazard_expected_nll_loss`. This teaches the stop head *when to buzz*
    before PPO fine-tuning consumes the resulting checkpoint. MVP is ``B=1`` per
    question (no padding/mask path).

    Parameters
    ----------
    config : dict[str, Any]
        Flat config. Reads ``checkpoint_dir`` (required), ``device`` (optional;
        CPU is fine), ``hazard_lr`` (falls back to ``supervised_lr`` then
        ``5e-5``), ``hazard_epochs`` (default 1) and ``max_grad_norm``
        (default 1.0).
    train_questions : list of MCQuestion
        Training questions. Only ``cumulative_prefixes``, ``options`` and
        ``gold_index`` are read (field names pinned to ``qb_data/mc_builder.py``
        as the authoritative producer; AP-031). Questions with zero prefixes
        (``T == 0``) are skipped. An empty list is a no-op that still saves a
        copy of the loaded model.
    pretrained_model_path : str, keyword-only
        Directory of the supervised warm-start checkpoint (contains
        ``policy_head.pt``). Must exist and be a loadable policy directory, or a
        clear error is raised (never a silent, untrained checkpoint).
    beta_terminal : float, keyword-only
        Terminal survival penalty threaded into
        :func:`hazard_expected_nll_loss` (weights the never-buzz survival mass).
    freeze_answer_head : bool, keyword-only
        When True, ``model.policy_head.answer_head`` parameters are frozen
        (``requires_grad = False``) so their weights do not move. Documented
        caveat (intended, not a bug): the answer-NLL gradient still flows
        through the frozen head into the **shared T5 encoder**, so this freezes
        only the answer head, matching the flag's literal name. A strict
        encoder-freeze variant is deliberately deferred.

    Returns
    -------
    str
        Path to the saved hazard checkpoint
        (``<checkpoint_dir>/hazard/best_model``), re-loadable via
        :meth:`T5PolicyModel.load_pretrained` (contains ``policy_head.pt``).

    Raises
    ------
    FileNotFoundError
        If ``pretrained_model_path`` is not an existing directory. A directory
        that exists but is not a valid policy checkpoint fails loud from
        :meth:`T5PolicyModel.load_pretrained` (not caught here).

    Notes
    -----
    Ships **smoke-validated (plumbing only)**: this confirms the flag runs, the
    loss is finite, the freeze works, the checkpoint is loadable and PPO
    consumes it. It does NOT establish training efficacy (convergence / S_q /
    calibration), which requires full-scale CUDA runs (Device 2 / RTX 5090) and
    is out of scope here.
    """
    # Fail loud on a missing checkpoint directory (R-008): never silently
    # proceed to save an untrained / freshly-initialized model.
    ckpt_dir = Path(pretrained_model_path)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            "run_hazard_pretrain: pretrained_model_path is not a directory: "
            f"{pretrained_model_path!r}. The hazard bridge requires a saved "
            "supervised T5 policy checkpoint to warm-start from."
        )

    device = config.get("device")
    # A directory that is not a valid policy checkpoint (missing config or
    # policy_head.pt) raises here — we deliberately let it propagate (R-008).
    model = T5PolicyModel.load_pretrained(str(ckpt_dir), device=device)

    if freeze_answer_head:
        model.policy_head.answer_head.requires_grad_(False)

    lr = float(config.get("hazard_lr", config.get("supervised_lr", 5e-5)))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    epochs = int(config.get("hazard_epochs", 1))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    model.train()
    for _epoch in range(epochs):
        for question in train_questions:
            prefixes = list(question.cumulative_prefixes)
            steps = len(prefixes)
            if steps == 0:
                # R-008: skip degenerate zero-prefix questions rather than crash.
                continue

            choices_text = _format_choices(question.options)
            texts = [
                f"CLUES: {prefix} | CHOICES: {choices_text}" for prefix in prefixes
            ]

            wait_logits, answer_logits, _ = model(texts)  # [T, 2], [T, K]
            # index 1 = BUZZ, so column 1 is P(stop) at each prefix; shape [1, T].
            stop_probs = torch.softmax(wait_logits, dim=-1)[:, 1].unsqueeze(0)
            gold = torch.full(
                (steps,),
                int(question.gold_index),
                dtype=torch.long,
                device=answer_logits.device,
            )
            nll_per_prefix = F.cross_entropy(
                answer_logits, gold, reduction="none"
            ).unsqueeze(0)  # [1, T]

            loss = hazard_expected_nll_loss(
                stop_probs, nll_per_prefix, beta_terminal=beta_terminal
            )

            optimizer.zero_grad()
            loss.backward()
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()

    save_dir = Path(config["checkpoint_dir"]) / "hazard" / "best_model"
    model.save(str(save_dir))
    return str(save_dir)
