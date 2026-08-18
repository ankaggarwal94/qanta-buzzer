"""Tests for the hazard pretraining bridge: loss math, training loop, CLI wiring.

Covers the hazard-pretrain warm-start bridge spec (R-001..R-008):

- R-001: ``validate_args`` no longer raises for ``--hazard-pretrain``.
- R-002: ``main()`` wires ``run_hazard_pretrain`` between the supervised phase
  and PPO, and feeds its checkpoint into PPO; absent flag ⇒ never called.
- R-003: ``run_hazard_pretrain`` loads the warm start, saves to
  ``checkpoints/hazard/best_model`` and returns a re-loadable path.
- R-004: per-prefix ``stop_probs``/``nll`` are shape ``[1, T]`` and the loss is
  a finite scalar.
- R-005: ``--freeze-answer-head`` freezes the answer head (requires_grad False
  and weights unchanged); without it the wait head moves.
- R-006: a larger ``--beta-terminal`` yields a strictly larger loss when
  residual survival mass remains.
- R-007: only ``cumulative_prefixes``/``options``/``gold_index`` are read; a
  single-prefix (T=1) question still produces a finite loss.
- R-008: fail-loud on a bad checkpoint path; empty ``train_questions`` is a
  no-op that still saves; zero-prefix (T=0) questions are skipped.

Model-touching tests use t5-small on CPU and skip when transformers is absent.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_DEFAULT_PREFIXES = [
    "Who",
    "Who was the",
    "Who was the first president",
]


def _make_mc_question(
    qid: str = "q",
    gold_index: int = 0,
    prefixes: list[str] | None = None,
) -> MCQuestion:
    """Build an MCQuestion with a controllable number of cumulative prefixes.

    Only ``cumulative_prefixes``, ``options`` and ``gold_index`` are read by
    ``run_hazard_pretrain`` (R-007); the remaining fields mirror the producer
    schema in ``qb_data/mc_builder.py`` so construction succeeds.
    """
    if prefixes is None:
        prefixes = list(_DEFAULT_PREFIXES)
    return MCQuestion(
        qid=qid,
        question="Who was the first president",
        tokens=["Who", "was", "the", "first", "president"],
        answer_primary="George Washington",
        clean_answers=["George Washington"],
        run_indices=[0, 2, 4],
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=list(prefixes),
        options=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        gold_index=gold_index,
        option_profiles=[
            "George Washington first president",
            "Thomas Jefferson third president",
            "John Adams second president",
            "Benjamin Franklin inventor diplomat",
        ],
        option_answer_primary=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        distractor_strategy="test",
    )


def _hazard_config(tmp_path: Path) -> dict:
    """Minimal config for ``run_hazard_pretrain`` on CPU / t5-small."""
    return {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "device": "cpu",
        "num_choices": 4,
        "max_input_length": 64,
        "hazard_epochs": 1,
        "hazard_lr": 1e-3,
    }


def _hazard_module():
    """Import the hazard module fresh (module import always succeeds).

    ``run_hazard_pretrain`` is accessed as an attribute inside each test so a
    not-yet-implemented symbol surfaces as a test-time ``AttributeError`` rather
    than a collection-time ``ImportError`` (which the RED gate treats as a build
    error, not a genuine test failure).
    """
    return importlib.import_module("training.hazard_pretrain")


@pytest.fixture(scope="module")
def supervised_ckpt(tmp_path_factory) -> str:
    """Create and save a t5-small policy checkpoint to act as the warm start."""
    pytest.importorskip("transformers")
    model = T5PolicyModel(
        {
            "model_name": "t5-small",
            "device": "cpu",
            "max_input_length": 64,
            "num_choices": 4,
        }
    )
    ckpt_dir = tmp_path_factory.mktemp("supervised_best_model")
    model.save(str(ckpt_dir))
    return str(ckpt_dir)


# ---------------------------------------------------------------------------
# Existing loss-math unit tests (unchanged)
# ---------------------------------------------------------------------------


def test_compute_survival_terms_simple_case() -> None:
    """compute_survival_terms returns expected survival and stop masses."""
    compute_survival_terms = _hazard_module().compute_survival_terms

    stop_probs = torch.tensor([[0.2, 0.5]], dtype=torch.float32)
    survival, stop_mass = compute_survival_terms(stop_probs)

    expected_survival = torch.tensor([[1.0, 0.8, 0.4]], dtype=torch.float32)
    expected_stop_mass = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
    assert torch.allclose(survival, expected_survival, atol=1e-6)
    assert torch.allclose(stop_mass, expected_stop_mass, atol=1e-6)


def test_hazard_expected_nll_loss_uses_terminal_penalty() -> None:
    """hazard_expected_nll_loss returns a scalar with beta_terminal applied."""
    hazard_expected_nll_loss = _hazard_module().hazard_expected_nll_loss

    stop_probs = torch.tensor([[0.2, 0.5]], dtype=torch.float32)
    nll_per_prefix = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    loss = hazard_expected_nll_loss(
        stop_probs=stop_probs,
        nll_per_prefix=nll_per_prefix,
        beta_terminal=1.5,
    )

    assert loss.ndim == 0
    assert loss.item() == pytest.approx(1.6)


# ---------------------------------------------------------------------------
# R-001: CLI guard flipped — validate_args must accept --hazard-pretrain
# ---------------------------------------------------------------------------


def test_hazard_pretrain_flag_no_longer_raises() -> None:
    """validate_args accepts --hazard-pretrain once the loop is wired (R-001).

    Flipped from the original guard test, which asserted ``validate_args``
    raised ``NotImplementedError``. The bridge is now implemented, so the guard
    must be gone and ``validate_args`` should return normally.
    """
    validate_args = importlib.import_module("scripts.train_t5_policy").validate_args

    args = argparse.Namespace(
        config="configs/t5_policy.yaml",
        smoke=False,
        skip_supervised=False,
        model_path=None,
        mc_path=None,
        ppo_iterations=None,
        hazard_pretrain=True,
        beta_terminal=1.0,
        freeze_answer_head=False,
    )

    assert validate_args(args) is None


# ---------------------------------------------------------------------------
# R-006: beta_terminal monotonicity (loss-function contract)
# ---------------------------------------------------------------------------


def test_beta_terminal_monotonic_increases_loss() -> None:
    """Larger beta_terminal strictly increases loss with residual survival (R-006)."""
    hazard_expected_nll_loss = _hazard_module().hazard_expected_nll_loss

    # survival[:, -1] = (1-0.2)*(1-0.3) = 0.56 > 0, so the terminal term bites.
    stop_probs = torch.tensor([[0.2, 0.3]], dtype=torch.float32)
    nll_per_prefix = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    loss_small = hazard_expected_nll_loss(
        stop_probs, nll_per_prefix, beta_terminal=0.5
    )
    loss_large = hazard_expected_nll_loss(
        stop_probs, nll_per_prefix, beta_terminal=2.0
    )

    assert loss_large.item() > loss_small.item()


# ---------------------------------------------------------------------------
# R-004: per-prefix stop_probs / nll shapes and finite loss (design contract)
# ---------------------------------------------------------------------------


def test_hazard_loss_shapes_and_finite(supervised_ckpt: str) -> None:
    """Per-prefix stop_probs and nll are [1, T] and the loss is finite (R-004)."""
    pytest.importorskip("transformers")
    hazard_expected_nll_loss = _hazard_module().hazard_expected_nll_loss

    model = T5PolicyModel.load_pretrained(supervised_ckpt, device="cpu")
    q = _make_mc_question(prefixes=list(_DEFAULT_PREFIXES))
    choices = " ".join(f"({i + 1}) {opt}" for i, opt in enumerate(q.options))
    texts = [f"CLUES: {p} | CHOICES: {choices}" for p in q.cumulative_prefixes]
    T = len(q.cumulative_prefixes)

    with torch.no_grad():
        wait_logits, answer_logits, _ = model(texts)

    stop_probs = torch.softmax(wait_logits, dim=-1)[:, 1].unsqueeze(0)
    gold = torch.full((T,), q.gold_index, dtype=torch.long)
    nll_per_prefix = F.cross_entropy(answer_logits, gold, reduction="none").unsqueeze(0)

    assert stop_probs.shape == (1, T)
    assert nll_per_prefix.shape == (1, T)

    loss = hazard_expected_nll_loss(stop_probs, nll_per_prefix, beta_terminal=1.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# R-003 / R-004 / R-007: end-to-end smoke — trains, saves, re-loads, finite
# ---------------------------------------------------------------------------


def test_run_hazard_pretrain_smoke_finite_and_loadable(
    supervised_ckpt: str, tmp_path: Path
) -> None:
    """run_hazard_pretrain over mixed T>=2 and T==1 questions saves a re-loadable
    checkpoint whose logits stay finite (R-003, R-004, R-007)."""
    pytest.importorskip("transformers")
    hz = _hazard_module()
    config = _hazard_config(tmp_path)
    questions = [
        _make_mc_question(
            "q_multi",
            gold_index=0,
            prefixes=["Who", "Who was the", "Who was the first president"],
        ),
        _make_mc_question(
            "q_single",
            gold_index=1,
            prefixes=["Who was the first president"],  # T == 1
        ),
    ]

    out_path = hz.run_hazard_pretrain(
        config=config,
        train_questions=questions,
        pretrained_model_path=supervised_ckpt,
    )

    out = Path(out_path)
    assert out.parts[-2:] == ("hazard", "best_model")
    assert out.exists()
    assert (out / "policy_head.pt").exists()

    reloaded = T5PolicyModel.load_pretrained(str(out), device="cpu")
    with torch.no_grad():
        wait_logits, answer_logits, _ = reloaded(
            ["CLUES: Who | CHOICES: (1) a (2) b (3) c (4) d"]
        )
    assert torch.isfinite(wait_logits).all()
    assert torch.isfinite(answer_logits).all()


# ---------------------------------------------------------------------------
# R-005: --freeze-answer-head
# ---------------------------------------------------------------------------


def test_run_hazard_pretrain_freeze_answer_head(
    supervised_ckpt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """freeze_answer_head=True: answer-head params are frozen (requires_grad
    False) and unchanged after the pass (R-005)."""
    pytest.importorskip("transformers")
    hz = _hazard_module()

    real_load = T5PolicyModel.load_pretrained
    captured: dict = {}

    def spy_load(load_dir, device=None):
        model = real_load(load_dir, device=device)
        captured["model"] = model
        captured["answer_before"] = [
            p.detach().clone() for p in model.policy_head.answer_head.parameters()
        ]
        return model

    monkeypatch.setattr(T5PolicyModel, "load_pretrained", spy_load)

    q = _make_mc_question(prefixes=list(_DEFAULT_PREFIXES))
    hz.run_hazard_pretrain(
        config=_hazard_config(tmp_path),
        train_questions=[q],
        pretrained_model_path=supervised_ckpt,
        freeze_answer_head=True,
    )

    answer_params = list(captured["model"].policy_head.answer_head.parameters())
    assert answer_params  # sanity: the head has parameters
    assert all(p.requires_grad is False for p in answer_params)
    for before, after in zip(captured["answer_before"], answer_params):
        assert torch.allclose(before, after)


def test_run_hazard_pretrain_no_freeze_moves_wait_head(
    supervised_ckpt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without freezing, the wait-head parameters change after the pass (R-005)."""
    pytest.importorskip("transformers")
    hz = _hazard_module()

    real_load = T5PolicyModel.load_pretrained
    captured: dict = {}

    def spy_load(load_dir, device=None):
        model = real_load(load_dir, device=device)
        captured["model"] = model
        captured["wait_before"] = [
            p.detach().clone() for p in model.policy_head.wait_head.parameters()
        ]
        return model

    monkeypatch.setattr(T5PolicyModel, "load_pretrained", spy_load)

    q = _make_mc_question(prefixes=list(_DEFAULT_PREFIXES))
    hz.run_hazard_pretrain(
        config=_hazard_config(tmp_path),
        train_questions=[q],
        pretrained_model_path=supervised_ckpt,
        freeze_answer_head=False,
    )

    wait_after = list(captured["model"].policy_head.wait_head.parameters())
    changed = any(
        not torch.allclose(b, a) for b, a in zip(captured["wait_before"], wait_after)
    )
    assert changed, "wait-head parameters should move during the hazard pass"


# ---------------------------------------------------------------------------
# R-008: fail-loud / empty no-op / T=0 skip
# ---------------------------------------------------------------------------


def test_run_hazard_pretrain_missing_path_fails_loud(tmp_path: Path) -> None:
    """A missing or unloadable checkpoint path raises rather than silently
    returning a checkpoint (R-008)."""
    pytest.importorskip("transformers")
    hz = _hazard_module()
    config = _hazard_config(tmp_path)
    q = _make_mc_question()

    missing = tmp_path / "does_not_exist"
    with pytest.raises((FileNotFoundError, OSError, ValueError, RuntimeError)):
        hz.run_hazard_pretrain(
            config=config,
            train_questions=[q],
            pretrained_model_path=str(missing),
        )

    # An empty directory exists but is not a loadable policy checkpoint.
    empty_dir = tmp_path / "empty_ckpt"
    empty_dir.mkdir()
    with pytest.raises((FileNotFoundError, OSError, ValueError, RuntimeError)):
        hz.run_hazard_pretrain(
            config=config,
            train_questions=[q],
            pretrained_model_path=str(empty_dir),
        )


def test_run_hazard_pretrain_empty_questions_noop(
    supervised_ckpt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty train_questions is a no-op that still saves a re-loadable copy of
    the loaded model unchanged (R-008)."""
    pytest.importorskip("transformers")
    hz = _hazard_module()

    real_load = T5PolicyModel.load_pretrained
    captured: dict = {}

    def spy_load(load_dir, device=None):
        model = real_load(load_dir, device=device)
        captured["model"] = model
        captured["wait_before"] = [
            p.detach().clone() for p in model.policy_head.wait_head.parameters()
        ]
        return model

    monkeypatch.setattr(T5PolicyModel, "load_pretrained", spy_load)

    out_path = hz.run_hazard_pretrain(
        config=_hazard_config(tmp_path),
        train_questions=[],
        pretrained_model_path=supervised_ckpt,
    )

    out = Path(out_path)
    assert out.exists()
    assert (out / "policy_head.pt").exists()

    wait_after = list(captured["model"].policy_head.wait_head.parameters())
    for before, after in zip(captured["wait_before"], wait_after):
        assert torch.allclose(before, after)


def test_run_hazard_pretrain_skips_zero_prefix_questions(
    supervised_ckpt: str, tmp_path: Path
) -> None:
    """Zero-prefix (T=0) questions are skipped, not crashed on (R-008)."""
    pytest.importorskip("transformers")
    hz = _hazard_module()
    questions = [
        _make_mc_question("q_zero", prefixes=[]),  # T == 0 -> skipped
        _make_mc_question("q_ok", prefixes=["Who", "Who was the"]),  # T == 2
    ]

    out_path = hz.run_hazard_pretrain(
        config=_hazard_config(tmp_path),
        train_questions=questions,
        pretrained_model_path=supervised_ckpt,
    )

    out = Path(out_path)
    assert out.exists()
    assert (out / "policy_head.pt").exists()


# ---------------------------------------------------------------------------
# R-002: main() wiring — run_hazard_pretrain between supervised and PPO
# ---------------------------------------------------------------------------


def _fake_manifest() -> dict:
    return {
        "source": "persisted_artifacts",
        "mc_path": None,
        "train_path": None,
        "val_path": None,
        "test_path": None,
        "train_qids": ["q1"],
        "val_qids": ["q1"],
        "test_qids": ["q1"],
    }


class _FakeTrainer:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir


def test_main_wires_hazard_between_supervised_and_ppo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--hazard-pretrain: main() calls run_hazard_pretrain after the supervised
    phase, threading its args, and PPO consumes the returned checkpoint (R-002)."""
    import sys

    import scripts.train_t5_policy as train_mod
    import training.hazard_pretrain as hz_mod
    import training.train_ppo_t5 as ppo_mod

    fake_q = type("Q", (), {"qid": "q1"})()
    ckpt_dir = tmp_path / "ppo_t5"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hazard_path = str(tmp_path / "hazard" / "best_model")
    supervised_path = str(tmp_path / "supervised_pretrained")

    calls: dict = {}

    def fake_hazard(
        config=None,
        train_questions=None,
        *,
        pretrained_model_path,
        beta_terminal=1.0,
        freeze_answer_head=False,
    ):
        calls["hazard"] = {
            "pretrained_model_path": pretrained_model_path,
            "beta_terminal": beta_terminal,
            "freeze_answer_head": freeze_answer_head,
        }
        return hazard_path

    def fake_ppo(**kwargs):
        calls["ppo"] = {"pretrained_model_path": kwargs.get("pretrained_model_path")}
        return object(), _FakeTrainer(ckpt_dir)

    monkeypatch.setattr(
        train_mod,
        "load_question_splits_with_metadata",
        lambda _args, _config: ([fake_q], [fake_q], [fake_q], _fake_manifest()),
    )
    monkeypatch.setattr(ppo_mod, "run_ppo_training", fake_ppo)
    monkeypatch.setattr(hz_mod, "run_hazard_pretrain", fake_hazard, raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_t5_policy.py",
            "--config",
            str(train_mod.PROJECT_ROOT / "configs" / "t5_policy.yaml"),
            "--skip-supervised",
            "--model-path",
            supervised_path,
            "--hazard-pretrain",
            "--beta-terminal",
            "2.0",
            "--freeze-answer-head",
        ],
    )

    train_mod.main()

    assert "hazard" in calls, "run_hazard_pretrain must be called with --hazard-pretrain"
    assert calls["hazard"]["pretrained_model_path"] == supervised_path
    assert calls["hazard"]["beta_terminal"] == 2.0
    assert calls["hazard"]["freeze_answer_head"] is True
    # PPO consumes the hazard checkpoint, not the raw supervised one.
    assert calls["ppo"]["pretrained_model_path"] == hazard_path


def test_main_skips_hazard_when_flag_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --hazard-pretrain, run_hazard_pretrain is never called and PPO
    consumes the supervised checkpoint unchanged (R-002)."""
    import sys

    import scripts.train_t5_policy as train_mod
    import training.hazard_pretrain as hz_mod
    import training.train_ppo_t5 as ppo_mod

    fake_q = type("Q", (), {"qid": "q1"})()
    ckpt_dir = tmp_path / "ppo_t5"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    supervised_path = str(tmp_path / "supervised_pretrained")

    calls: dict = {}

    def boom(*_args, **_kwargs):
        raise AssertionError(
            "run_hazard_pretrain must not be called without --hazard-pretrain"
        )

    def fake_ppo(**kwargs):
        calls["ppo"] = {"pretrained_model_path": kwargs.get("pretrained_model_path")}
        return object(), _FakeTrainer(ckpt_dir)

    monkeypatch.setattr(
        train_mod,
        "load_question_splits_with_metadata",
        lambda _args, _config: ([fake_q], [fake_q], [fake_q], _fake_manifest()),
    )
    monkeypatch.setattr(ppo_mod, "run_ppo_training", fake_ppo)
    monkeypatch.setattr(hz_mod, "run_hazard_pretrain", boom, raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_t5_policy.py",
            "--config",
            str(train_mod.PROJECT_ROOT / "configs" / "t5_policy.yaml"),
            "--skip-supervised",
            "--model-path",
            supervised_path,
        ],
    )

    train_mod.main()

    assert calls["ppo"]["pretrained_model_path"] == supervised_path


# ---------------------------------------------------------------------------
# QA-013: supervised save-best gate (the shared warm-start every hazard/PPO
# arm branches from is produced by this trainer)
# ---------------------------------------------------------------------------


def test_qa013_supervised_best_gate_saves_on_zero_val_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA-013: save-best gates initialize to -inf, not a reachable value.

    With the old ``best_val_acc = 0.0`` init and the strict ``>`` gate, a
    run whose validation accuracy is exactly 0.0 (plausible for fresh
    t5-small heads on a tiny smoke split) never wrote
    ``supervised/best_model`` and the efficacy harness died loud on the
    missing shared checkpoint. Epoch 1 must ALWAYS save (mirrors
    ``PPOTrainer.best_val_reward = -inf``).
    """
    pytest.importorskip("transformers")
    import training.train_supervised_t5 as sup_mod

    class _StubModel:
        device = "cpu"

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    trainer = sup_mod.SupervisedTrainer(
        model=_StubModel(),
        train_questions=[],
        val_questions=[],
        config={
            "supervised_epochs": 1,
            "checkpoint_dir": str(tmp_path / "ckpt"),
        },
    )
    assert trainer.best_val_acc == -float("inf"), (
        "the save-best gate must initialize to -inf, never to a reachable "
        "metric value (QA-013)"
    )

    saves: list[bool] = []
    monkeypatch.setattr(trainer, "train_epoch", lambda: (1.0, 0.0))
    monkeypatch.setattr(trainer, "validate", lambda: (1.0, 0.0))  # 0.0 acc
    monkeypatch.setattr(
        trainer,
        "save_checkpoint",
        lambda is_best=False: saves.append(is_best) or (tmp_path / "ckpt"),
    )
    monkeypatch.setattr(trainer, "save_history", lambda: tmp_path / "h.json")

    summary = trainer.train()

    assert True in saves, (
        "a 0.0-validation-accuracy epoch must still write best_model "
        "(epoch 1 always saves under the -inf init)"
    )
    assert summary["best_val_acc"] == pytest.approx(0.0)
