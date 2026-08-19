"""Hazard pretraining bridge utilities for stopping-aware warm starts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.nn.functional as F

from models.t5_policy import T5PolicyModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qb_data.mc_builder import MCQuestion


# Known hazard-phase ablations (R-004). ``None`` means the real hazard loss.
_VALID_ABLATIONS = ("shuffled_nll",)

# Mini-audit-verify F2: the hazard loop prints one terse progress line every
# this many optimizer steps. The efficacy harness's MA-006 output-staleness
# watchdog kills children that go silent, and a full-scale hazard phase can
# otherwise run for hours with zero output between phase banners; the print
# is stdout-only — hazard_history.json stays exactly as pinned (R-010).
_PROGRESS_PRINT_EVERY_STEPS = 25

# Fixed seed for the DEDICATED shuffled_nll permutation generator. The
# ablation must never draw from the global torch/numpy/random streams (that
# would desync otherwise-identical runs), so permutations come from a private
# ``torch.Generator`` seeded with this constant. Empirically verified
# (torch 2.x CPU): the first draws are non-identity for T>=3
# (``randperm(4) -> [1, 3, 2, 0]``, ``randperm(3) -> [2, 0, 1]``), which the
# R-004 first-step loss-divergence contract requires.
_ABLATION_RNG_SEED = 1


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
    ablation: str | None = None,
) -> str:
    """Warm-start the buzz/stop head with the hazard survival loss before PPO.

    Loads the supervised checkpoint, then for each training question forwards
    the T5 policy over that question's ``cumulative_prefixes`` to obtain
    per-prefix stop-probabilities (from the wait head) and per-prefix answer-NLL
    (from the answer head vs. ``gold_index``), and minimizes
    :func:`hazard_expected_nll_loss`. This teaches the stop head *when to buzz*
    before PPO fine-tuning consumes the resulting checkpoint. MVP is ``B=1`` per
    question (no padding/mask path).

    Each optimizer step is additionally recorded to
    ``<checkpoint_dir>/hazard/hazard_history.json`` (R-010) with the pinned
    schema ``{"steps": [{"epoch": int, "question_index": int, "loss": float}],
    "config": {"beta_terminal": float, "freeze_answer_head": bool,
    "ablation": str|null, "lr": float, "epochs": int},
    "wall_clock_seconds": float}`` (the top-level ``wall_clock_seconds`` —
    the HAZARD-PHASE wall clock covering checkpoint load, training loop, and
    checkpoint save — was added in QA fix round 1, QA-006; the whole-child
    elapsed time is PPO-dominated and lives in the harness's run marker, not
    here); the returned checkpoint path and format are unchanged. The loop
    also prints one terse progress line every 25 optimizer steps
    (mini-audit-verify F2: the efficacy harness's MA-006 output-staleness
    watchdog needs periodic child output; stdout-only —
    ``hazard_history.json`` is unchanged).

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
    ablation : str or None, keyword-only
        ``None`` (default) runs the real hazard loss. ``"shuffled_nll"`` runs
        the identical loop — same question set, epochs, and optimizer-step
        count — but permutes each question's per-prefix NLL vector across its
        ``T`` positions before the loss, destroying the temporal signal while
        preserving compute (the R-004 step-matched null-signal control).
        Permutations come from a dedicated ``torch.Generator`` seeded with
        ``_ABLATION_RNG_SEED`` so the global RNG streams are untouched; a
        ``T == 1`` question therefore reproduces the non-ablated losses
        exactly (the only permutation is the identity). Any other value
        raises ``ValueError`` before any artifact is written.

    Returns
    -------
    str
        Path to the saved hazard checkpoint
        (``<checkpoint_dir>/hazard/best_model``), re-loadable via
        :meth:`T5PolicyModel.load_pretrained` (contains ``policy_head.pt``).
        The parent dir additionally carries ``hazard_history.json`` (R-010).

    Raises
    ------
    ValueError
        If ``ablation`` is neither ``None`` nor ``"shuffled_nll"``. Raised
        before any checkpoint/history artifact is written (no partial
        ``hazard/`` dir is left behind).
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
    # Fail loud on an unknown ablation (R-004) BEFORE any I/O or artifact
    # write: a rejected value must leave no partial hazard checkpoint/history.
    if ablation is not None and ablation not in _VALID_ABLATIONS:
        raise ValueError(
            f"run_hazard_pretrain: unknown ablation {ablation!r}; expected "
            f"None or one of {_VALID_ABLATIONS}."
        )

    # Fail loud on a missing checkpoint directory (R-008): never silently
    # proceed to save an untrained / freshly-initialized model.
    ckpt_dir = Path(pretrained_model_path)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            "run_hazard_pretrain: pretrained_model_path is not a directory: "
            f"{pretrained_model_path!r}. The hazard bridge requires a saved "
            "supervised T5 policy checkpoint to warm-start from."
        )

    # QA-006: the hazard-PHASE wall clock (load + train + save), measured at
    # the phase boundary so the report can describe the hazard phase itself
    # rather than the PPO-dominated child total.
    phase_start = time.monotonic()

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

    # Dedicated permutation generator for the shuffled_nll ablation (R-004):
    # never the global torch/numpy/random streams, so a run with the ablation
    # enabled consumes exactly the same global-RNG sequence as one without.
    ablation_rng: torch.Generator | None = None
    if ablation == "shuffled_nll":
        ablation_rng = torch.Generator()
        ablation_rng.manual_seed(_ABLATION_RNG_SEED)

    # R-010: one record per optimizer step, written to hazard_history.json.
    history_steps: List[Dict[str, Any]] = []

    model.train()
    for epoch in range(epochs):
        for question_index, question in enumerate(train_questions):
            prefixes = list(question.cumulative_prefixes)
            steps = len(prefixes)
            if steps == 0:
                # R-008: skip degenerate zero-prefix questions rather than
                # crash. No optimizer step -> no history record either.
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

            if ablation_rng is not None:
                # shuffled_nll (R-004): permute the per-prefix NLL vector
                # across its T positions before the loss. Pure index gather —
                # same tensor values, same optimizer-step count — so compute
                # is preserved while the temporal alignment between stop mass
                # and answer difficulty is destroyed. For T == 1 the only
                # permutation is the identity, reproducing the non-ablated
                # loss bitwise.
                perm = torch.randperm(steps, generator=ablation_rng).to(
                    nll_per_prefix.device
                )
                nll_per_prefix = nll_per_prefix[:, perm]

            loss = hazard_expected_nll_loss(
                stop_probs, nll_per_prefix, beta_terminal=beta_terminal
            )

            optimizer.zero_grad()
            loss.backward()
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()

            loss_value = float(loss.item())
            history_steps.append(
                {
                    "epoch": int(epoch),
                    "question_index": int(question_index),
                    "loss": loss_value,
                }
            )
            # Mini-audit-verify F2: terse periodic progress so the harness's
            # MA-006 output-staleness watchdog sees a live child (flush so
            # the line crosses the pipe immediately even if buffered).
            step_count = len(history_steps)
            if step_count % _PROGRESS_PRINT_EVERY_STEPS == 0:
                print(
                    f"[hazard] step {step_count} epoch {epoch} "
                    f"question {question_index} loss {loss_value:.4f}",
                    flush=True,
                )

    save_dir = Path(config["checkpoint_dir"]) / "hazard" / "best_model"
    model.save(str(save_dir))

    # R-010: persist per-step training dynamics next to the checkpoint with
    # the pinned schema (see the Format-pinning section of the hazard spec).
    # Written even for the empty-questions no-op (steps: []) so the harness
    # read path never crashes on a degenerate run.
    history = {
        "steps": history_steps,
        "config": {
            "beta_terminal": float(beta_terminal),
            "freeze_answer_head": bool(freeze_answer_head),
            "ablation": ablation,
            "lr": float(lr),
            "epochs": int(epochs),
        },
        # QA-006 (spec amendment, QA fix round 1): hazard-phase wall clock.
        "wall_clock_seconds": float(time.monotonic() - phase_start),
    }
    history_path = save_dir.parent / "hazard_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    # MA-017 parity (PR #41 round-2 P3 [2]): strict JSON — a non-finite loss
    # (NaN/inf training blow-up) must fail loud at the writer, never land as
    # a strict-invalid token that every downstream reader (harness dynamics,
    # step parity, report compute block) then chokes on or misparses.
    history_path.write_text(
        json.dumps(history, indent=2, allow_nan=False), encoding="utf-8"
    )

    return str(save_dir)
