"""RED tests for R-004 (shuffled_nll ablation) and R-010 (hazard_history.json).

Covers the hazard-efficacy-eval spec, instrumentation half only:

- R-004 [integration]: ``run_hazard_pretrain`` gains keyword-only
  ``ablation: str | None = None``; ``"shuffled_nll"`` runs the identical loop
  (same optimizer-step count) but permutes each question's per-prefix NLL
  vector with a seeded RNG before the loss.  ``scripts/train_t5_policy.py``
  exposes ``--hazard-ablation shuffled_nll`` (requires ``--hazard-pretrain``;
  unknown values rejected), threads it into ``run_hazard_pretrain`` and records
  it in the ``config_used.json`` hazard block.
- R-010 [integration] (producer contract ONLY — the harness-side stop-prob
  probe is out of scope here): ``run_hazard_pretrain`` writes
  ``hazard_history.json`` into the hazard checkpoint parent dir
  (``<checkpoint_dir>/hazard/``) with the PINNED schema
  ``{"steps": [{"epoch": int, "question_index": int, "loss": float}],
  "config": {"beta_terminal": float, "freeze_answer_head": bool,
  "ablation": str|null, "lr": float, "epochs": int},
  "wall_clock_seconds": float}`` (top-level hazard-phase wall clock added
  in QA fix round 1, QA-006 — spec Format-pinning amended); one record per
  optimizer step; all losses finite; the returned checkpoint path and
  ``policy_head.pt`` format are UNCHANGED (PPO consumption unaffected).

AP-031 note: ``hazard_history.json`` is a NEW producer introduced by this spec
(R-010) — no real artifact exists in the repo yet, so the real-fixture
requirement is dormant; these tests ARE the format pin per the spec's "Format
pinning" section.

# DECISION (pins the GREEN agent must honor):
# - The top-level history keys and the step/config key sets are pinned EXACTLY
#   (AP-031 format pinning: the harness reads only through this schema; extend
#   only via a spec change).
# - An empty train_questions run (existing R-008 no-op) still writes the
#   history with ``steps: []`` so the harness read path never crashes.
# - ``ablation="bogus"`` raises ValueError BEFORE any hazard artifact is
#   written (fail-loud leaves no partial checkpoint/history).
# - Permutation proof: with identical seeds/checkpoint/questions the
#   non-ablated CPU run is bitwise deterministic, so per-step losses that
#   diverge FROM THE FIRST STEP prove the shuffle was applied; GREEN must pick
#   a fixed ablation RNG whose first permutations are not the identity (the
#   spec Exit demands exactly this observable).
# - Identity discriminator (S-1): the ablation RNG must be a DEDICATED seeded
#   generator (never the global torch/numpy/random streams), and the ablation
#   must be a true permutation of the per-question NLL vector — T=1 questions
#   force the identity, so their per-step losses must EXACTLY equal the
#   ablation=None run's (kills scale/noise stand-ins and global-RNG shuffles).

Model-touching tests use t5-small on CPU and skip when transformers is absent
(idiom copied from tests/test_hazard_pretrain.py).  ``run_hazard_pretrain`` is
resolved as a module attribute at test time so unimplemented pieces surface as
runtime errors, never collection errors.

RED expectation: failures are runtime ``AssertionError`` (missing
hazard_history.json), ``TypeError`` (unexpected keyword ``ablation``), or
``SystemExit`` (argparse does not know ``--hazard-ablation``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _hazard_module():
    """Import training.hazard_pretrain fresh (attribute access at test time)."""
    return importlib.import_module("training.hazard_pretrain")


def _make_mc_question(
    qid: str,
    gold_index: int = 0,
    prefixes: list[str] | None = None,
) -> MCQuestion:
    """Build an MCQuestion with a controllable number of cumulative prefixes."""
    if prefixes is None:
        prefixes = ["Who", "Who was the", "Who was the first president"]
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


def _hazard_config(base_dir: Path, epochs: int = 2, lr: float = 1e-3) -> dict:
    """Minimal flat config for ``run_hazard_pretrain`` on CPU / t5-small."""
    return {
        "checkpoint_dir": str(base_dir),
        "device": "cpu",
        "num_choices": 4,
        "max_input_length": 64,
        "hazard_epochs": epochs,
        "hazard_lr": lr,
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_history(returned_ckpt_path: str) -> dict:
    """Read hazard_history.json from the hazard checkpoint PARENT dir."""
    hist_path = Path(returned_ckpt_path).parent / "hazard_history.json"
    assert hist_path.is_file(), (
        "R-010: run_hazard_pretrain must write hazard_history.json into the "
        f"hazard checkpoint parent dir; missing at {hist_path}"
    )
    return json.loads(hist_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def supervised_ckpt(tmp_path_factory) -> str:
    """Create and save a t5-small policy checkpoint as the warm start."""
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


_Q_T3 = [
    "This Founding Father",
    "This Founding Father commanded the continental army",
    "This Founding Father commanded the army and became president",
]
_Q_T2 = ["This statesman", "This statesman wrote many letters from Paris"]
_Q_T4 = [
    "This figure",
    "This figure surveyed land in Virginia",
    "This figure surveyed land and crossed the Delaware",
    "This figure surveyed land, crossed the Delaware, and led the nation",
]


# ---------------------------------------------------------------------------
# R-010: hazard_history.json producer contract (pinned schema)
# ---------------------------------------------------------------------------


def test_r010a_history_written_with_pinned_schema_and_step_count(
    supervised_ckpt: str, tmp_path: Path
) -> None:
    """Tests R-010 [integration]: history file, pinned schema, step count.

    2 epochs over [T=0 (skipped), T=3, T=2] questions ⇒ exactly
    ``2 epochs × 2 trainable questions = 4`` optimizer steps recorded.  The
    zero-prefix question must contribute NO step (it performs no optimizer
    step).  Schema is pinned exactly per the spec's Format pinning section.
    The returned checkpoint path/format must be unchanged and re-loadable
    (PPO consumption unaffected).
    """
    pytest.importorskip("transformers")
    hz = _hazard_module()
    config = _hazard_config(tmp_path / "checkpoints", epochs=2, lr=1e-3)
    questions = [
        _make_mc_question("q_zero", prefixes=[]),  # T == 0 -> skipped, no step
        _make_mc_question("q_t3", gold_index=0, prefixes=list(_Q_T3)),
        _make_mc_question("q_t2", gold_index=1, prefixes=list(_Q_T2)),
    ]

    out_path = hz.run_hazard_pretrain(
        config=config,
        train_questions=questions,
        pretrained_model_path=supervised_ckpt,
        beta_terminal=1.5,
        freeze_answer_head=False,
    )

    # Additive contract: returned path and checkpoint format UNCHANGED.
    out = Path(out_path)
    assert out.parts[-2:] == ("hazard", "best_model")
    assert (out / "policy_head.pt").exists()

    history = _load_history(out_path)
    assert set(history) == {"steps", "config", "wall_clock_seconds"}, (
        "hazard_history.json top-level keys are pinned to {steps, config, "
        "wall_clock_seconds} (wall_clock_seconds added in QA fix round 1, "
        "QA-006 — the hazard-PHASE wall clock, spec Format-pinning amended)"
    )
    wall_clock = history["wall_clock_seconds"]
    assert isinstance(wall_clock, float) and not isinstance(wall_clock, bool)
    assert math.isfinite(wall_clock) and wall_clock >= 0.0

    steps = history["steps"]
    assert isinstance(steps, list)
    assert len(steps) == 4, (
        "one record per optimizer step: 2 epochs x 2 T>0 questions = 4 "
        f"(T=0 questions must not add steps); got {len(steps)}"
    )
    for rec in steps:
        assert set(rec) == {"epoch", "question_index", "loss"}, (
            f"step record keys pinned to (epoch, question_index, loss); got {set(rec)}"
        )
        assert isinstance(rec["epoch"], int) and not isinstance(rec["epoch"], bool)
        qi = rec["question_index"]
        assert isinstance(qi, int) and not isinstance(qi, bool)
        assert qi >= 0
        assert isinstance(rec["loss"], float)
        assert math.isfinite(rec["loss"])

    # Exactly two distinct epoch values, two steps each; two distinct
    # question indices (the two T>0 questions), stable across epochs.
    epochs_seen = sorted({rec["epoch"] for rec in steps})
    assert len(epochs_seen) == 2
    for epoch in epochs_seen:
        assert sum(1 for rec in steps if rec["epoch"] == epoch) == 2
    assert len({rec["question_index"] for rec in steps}) == 2

    cfg_block = history["config"]
    assert set(cfg_block) == {
        "beta_terminal",
        "freeze_answer_head",
        "ablation",
        "lr",
        "epochs",
    }, "history config block keys are pinned per the spec Format pinning"
    assert isinstance(cfg_block["beta_terminal"], float)
    assert cfg_block["beta_terminal"] == 1.5
    assert cfg_block["freeze_answer_head"] is False
    assert cfg_block["ablation"] is None
    assert isinstance(cfg_block["lr"], float)
    assert cfg_block["lr"] == pytest.approx(1e-3)
    assert isinstance(cfg_block["epochs"], int) and not isinstance(
        cfg_block["epochs"], bool
    )
    assert cfg_block["epochs"] == 2

    # PPO consumption unaffected: checkpoint still loads and runs finite.
    reloaded = T5PolicyModel.load_pretrained(str(out), device="cpu")
    with torch.no_grad():
        wait_logits, answer_logits, _ = reloaded(
            ["CLUES: Who | CHOICES: (1) a (2) b (3) c (4) d"]
        )
    assert torch.isfinite(wait_logits).all()
    assert torch.isfinite(answer_logits).all()


def test_r010a_empty_run_still_writes_history(
    supervised_ckpt: str, tmp_path: Path
) -> None:
    """Tests R-010 [integration]: the R-008 empty no-op still writes history.

    ``train_questions=[]`` saves an unchanged checkpoint (existing contract);
    the history must exist with ``steps: []`` and a complete config block
    reflecting the ACTUAL call arguments, so a harness read never crashes on a
    degenerate run.
    """
    pytest.importorskip("transformers")
    hz = _hazard_module()
    config = _hazard_config(tmp_path / "checkpoints", epochs=3, lr=2e-4)

    out_path = hz.run_hazard_pretrain(
        config=config,
        train_questions=[],
        pretrained_model_path=supervised_ckpt,
        beta_terminal=2.0,
        freeze_answer_head=True,
    )

    history = _load_history(out_path)
    assert history["steps"] == []
    cfg_block = history["config"]
    assert cfg_block["beta_terminal"] == 2.0
    assert cfg_block["freeze_answer_head"] is True
    assert cfg_block["ablation"] is None
    assert cfg_block["lr"] == pytest.approx(2e-4)
    assert cfg_block["epochs"] == 3
    # QA-006: even the degenerate no-op records its hazard-phase wall clock.
    assert isinstance(history["wall_clock_seconds"], float)
    assert history["wall_clock_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# R-004: shuffled_nll — step-matched compute, provably different losses
# ---------------------------------------------------------------------------


def test_r004_shuffled_nll_step_matched_but_losses_diverge(
    supervised_ckpt: str, tmp_path: Path
) -> None:
    """Tests R-004 [integration]: compute-matched null-signal ablation.

    Two runs from the IDENTICAL checkpoint, questions ([T=4, T=3]), epochs and
    global seed on CPU — one with ``ablation=None``, one with
    ``ablation="shuffled_nll"``.  Verified via each run's hazard_history.json
    (R-010 instrumentation):

    - identical optimizer-step counts (the compute-matching contract),
    - all losses finite in both runs,
    - the loss sequences DIFFER, starting at the very first step (before any
      weight divergence the models are bitwise identical, so a first-step loss
      delta proves the NLL permutation was actually applied — a no-op
      ``ablation`` kwarg cannot pass),
    - the ablation value is recorded in each history config block,
    - the ablated checkpoint remains re-loadable (PPO consumption unaffected).
    """
    pytest.importorskip("transformers")
    hz = _hazard_module()
    questions = [
        _make_mc_question("q_t4", gold_index=0, prefixes=list(_Q_T4)),
        _make_mc_question("q_t3", gold_index=2, prefixes=list(_Q_T3)),
    ]

    _seed_all(1234)
    out_plain = hz.run_hazard_pretrain(
        config=_hazard_config(tmp_path / "arm_none", epochs=2),
        train_questions=[_make_mc_question(q.qid, q.gold_index, list(q.cumulative_prefixes)) for q in questions],
        pretrained_model_path=supervised_ckpt,
        beta_terminal=1.0,
        freeze_answer_head=False,
    )

    _seed_all(1234)
    out_ablated = hz.run_hazard_pretrain(
        config=_hazard_config(tmp_path / "arm_shuffled", epochs=2),
        train_questions=[_make_mc_question(q.qid, q.gold_index, list(q.cumulative_prefixes)) for q in questions],
        pretrained_model_path=supervised_ckpt,
        beta_terminal=1.0,
        freeze_answer_head=False,
        ablation="shuffled_nll",
    )

    hist_plain = _load_history(out_plain)
    hist_ablated = _load_history(out_ablated)

    losses_plain = [rec["loss"] for rec in hist_plain["steps"]]
    losses_ablated = [rec["loss"] for rec in hist_ablated["steps"]]

    # Step-matched compute: identical optimizer-step counts.
    assert len(losses_plain) == len(losses_ablated) == 4  # 2 epochs x 2 questions
    assert all(math.isfinite(loss) for loss in losses_plain)
    assert all(math.isfinite(loss) for loss in losses_ablated)

    # Permutation provably applied under the fixed seeds: the deterministic
    # CPU baseline makes any loss delta attributable to the shuffle alone.
    assert abs(losses_plain[0] - losses_ablated[0]) > 1e-9, (
        "first-step loss identical with and without shuffled_nll — the "
        "seeded NLL permutation was not applied (or is the identity; pick a "
        "fixed ablation RNG whose first permutation moves positions)"
    )
    assert losses_plain != losses_ablated

    # Ablation recorded in the pinned history config block.
    assert hist_plain["config"]["ablation"] is None
    assert hist_ablated["config"]["ablation"] == "shuffled_nll"

    # Checkpoint format unchanged for the ablated arm too.
    out = Path(out_ablated)
    assert out.parts[-2:] == ("hazard", "best_model")
    assert (out / "policy_head.pt").exists()
    T5PolicyModel.load_pretrained(str(out), device="cpu")


def test_r004_shuffled_nll_identity_on_singleton_prefix_questions(
    supervised_ckpt: str, tmp_path: Path
) -> None:
    """Tests R-004 [integration]: T=1 questions make the permutation the identity.

    Complement to the divergence test above: when EVERY question exposes a
    single cumulative prefix (T=1), each per-question NLL vector has length 1,
    so the only possible permutation is the identity — a REAL shuffled_nll must
    leave every step's loss EXACTLY equal to the ``ablation=None`` run
    (identical checkpoint, questions, epochs, global seeds; CPU runs are
    bitwise deterministic per the module DECISION note).  This kills
    implementations that fake the ablation by scaling or noising the loss
    (those pass the divergence test but change T=1 losses), and pins the
    ablation RNG as a DEDICATED seeded generator: drawing permutations from
    the global torch/numpy/random streams would desync the deterministic
    baseline and change the losses even though the permutation is the
    identity.
    """
    pytest.importorskip("transformers")
    hz = _hazard_module()

    def singleton_questions() -> list[MCQuestion]:
        return [
            _make_mc_question(
                "q_s1",
                gold_index=0,
                prefixes=["This statesman wrote many letters from Paris"],
            ),
            _make_mc_question(
                "q_s2",
                gold_index=1,
                prefixes=["This composer premiered nine symphonies in Vienna"],
            ),
        ]

    _seed_all(4321)
    out_plain = hz.run_hazard_pretrain(
        config=_hazard_config(tmp_path / "singleton_none", epochs=2),
        train_questions=singleton_questions(),
        pretrained_model_path=supervised_ckpt,
        beta_terminal=1.0,
        freeze_answer_head=False,
    )

    _seed_all(4321)
    out_ablated = hz.run_hazard_pretrain(
        config=_hazard_config(tmp_path / "singleton_shuffled", epochs=2),
        train_questions=singleton_questions(),
        pretrained_model_path=supervised_ckpt,
        beta_terminal=1.0,
        freeze_answer_head=False,
        ablation="shuffled_nll",
    )

    hist_plain = _load_history(out_plain)
    hist_ablated = _load_history(out_ablated)
    losses_plain = [rec["loss"] for rec in hist_plain["steps"]]
    losses_ablated = [rec["loss"] for rec in hist_ablated["steps"]]

    # Step-matched compute either way: 2 epochs x 2 T=1 questions.
    assert len(losses_plain) == len(losses_ablated) == 4
    assert all(math.isfinite(loss) for loss in losses_plain + losses_ablated)

    assert losses_ablated == losses_plain, (
        "shuffling a length-1 NLL vector is the identity permutation: the "
        "ablated per-step losses must be EXACTLY the ablation=None losses "
        f"(got {losses_ablated} vs {losses_plain}) — a scale/noise stand-in "
        "for the permutation, or a shuffle drawn from the global RNG "
        "streams, cannot satisfy this"
    )

    # Both histories still record their own ablation value.
    assert hist_plain["config"]["ablation"] is None
    assert hist_ablated["config"]["ablation"] == "shuffled_nll"


def test_r004_unknown_ablation_raises_valueerror(
    supervised_ckpt: str, tmp_path: Path
) -> None:
    """Tests R-004 [integration]: unknown ablation fails loud, leaves nothing.

    ``ablation="bogus"`` must raise ValueError (not silently train), and the
    rejected call must not leave a partial hazard checkpoint or history behind.
    """
    pytest.importorskip("transformers")
    hz = _hazard_module()
    config = _hazard_config(tmp_path / "checkpoints_bogus", epochs=1)

    with pytest.raises(ValueError):
        hz.run_hazard_pretrain(
            config=config,
            train_questions=[_make_mc_question("q_t2", prefixes=list(_Q_T2))],
            pretrained_model_path=supervised_ckpt,
            beta_terminal=1.0,
            ablation="bogus",
        )

    hazard_dir = Path(config["checkpoint_dir"]) / "hazard"
    assert not hazard_dir.exists(), (
        "a rejected ablation value must not write hazard artifacts"
    )


# ---------------------------------------------------------------------------
# R-004: CLI surface — --hazard-ablation exposure, gating, threading
# ---------------------------------------------------------------------------


def _parse(monkeypatch: pytest.MonkeyPatch, tail: list[str]) -> argparse.Namespace:
    import scripts.train_t5_policy as train_mod

    monkeypatch.setattr(sys, "argv", ["train_t5_policy.py", *tail])
    return train_mod.parse_args()


def _cli_namespace(**overrides) -> argparse.Namespace:
    """Full CLI namespace for direct validate_args calls (parse_args parity)."""
    ns = argparse.Namespace(
        config="configs/t5_policy.yaml",
        smoke=False,
        skip_supervised=False,
        model_path=None,
        mc_path=None,
        ppo_iterations=None,
        hazard_pretrain=False,
        beta_terminal=1.0,
        freeze_answer_head=False,
        hazard_ablation=None,
        seed=None,
        overrides=[],
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_r004_cli_exposes_hazard_ablation_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests R-004 [integration]: --hazard-ablation shuffled_nll parses.

    The flag defaults to None, parses to the string value alongside
    ``--hazard-pretrain``, and ``validate_args`` accepts the valid combination.
    """
    import scripts.train_t5_policy as train_mod

    args = _parse(
        monkeypatch, ["--hazard-pretrain", "--hazard-ablation", "shuffled_nll"]
    )
    assert args.hazard_ablation == "shuffled_nll"
    assert args.hazard_pretrain is True

    assert _parse(monkeypatch, ["--hazard-pretrain"]).hazard_ablation is None

    ok = _cli_namespace(hazard_pretrain=True, hazard_ablation="shuffled_nll")
    assert train_mod.validate_args(ok) is None


def test_r004_cli_rejects_ablation_without_hazard_pretrain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests R-004 [integration]: validate_args gates ablation on the bridge.

    ``--hazard-ablation`` without ``--hazard-pretrain`` must SystemExit from
    ``validate_args`` (the flag is meaningless without the hazard phase).
    The parse guard first proves the flag exists at all.
    """
    import scripts.train_t5_policy as train_mod

    guard = _parse(
        monkeypatch, ["--hazard-pretrain", "--hazard-ablation", "shuffled_nll"]
    )
    assert guard.hazard_ablation == "shuffled_nll"

    bad = _cli_namespace(hazard_pretrain=False, hazard_ablation="shuffled_nll")
    with pytest.raises(SystemExit):
        train_mod.validate_args(bad)


def test_r004_cli_rejects_unknown_ablation_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests R-004 [integration]: unknown ablation values are rejected loud.

    End-to-end through ``parse_args`` + ``validate_args`` so either an
    argparse ``choices`` rejection or a validate_args rejection satisfies the
    contract — both surface as SystemExit.  The parse guard first proves the
    valid spelling is accepted (so this test cannot pass by the flag simply
    not existing).
    """
    import scripts.train_t5_policy as train_mod

    guard = _parse(
        monkeypatch, ["--hazard-pretrain", "--hazard-ablation", "shuffled_nll"]
    )
    assert guard.hazard_ablation == "shuffled_nll"

    with pytest.raises(SystemExit):
        args = _parse(
            monkeypatch, ["--hazard-pretrain", "--hazard-ablation", "bogus"]
        )
        train_mod.validate_args(args)


def test_r004_main_threads_ablation_into_hazard_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests R-004 [integration]: main() threads ablation and records it.

    With ``--hazard-pretrain --hazard-ablation shuffled_nll``, ``main()`` must
    pass ``ablation="shuffled_nll"`` (keyword) into ``run_hazard_pretrain``,
    PPO must consume the hazard checkpoint, and the ``config_used.json``
    hazard block must record the ablation.  Patches target the lazy-import
    source modules (pattern: tests/test_hazard_pretrain.py).
    """
    import scripts.train_t5_policy as train_mod
    import training.hazard_pretrain as hz_mod
    import training.train_ppo_t5 as ppo_mod

    fake_q = type("Q", (), {"qid": "q1"})()
    fake_manifest = {
        "source": "persisted_artifacts",
        "mc_path": None,
        "train_path": None,
        "val_path": None,
        "test_path": None,
        "train_qids": ["q1"],
        "val_qids": ["q1"],
        "test_qids": ["q1"],
    }
    ckpt_dir = tmp_path / "ppo_t5"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hazard_path = str(tmp_path / "hazard" / "best_model")
    supervised_path = str(tmp_path / "supervised_pretrained")

    calls: dict = {}

    class _FakeTrainer:
        def __init__(self, checkpoint_dir: Path) -> None:
            self.checkpoint_dir = checkpoint_dir

    def fake_hazard(*_args, **kwargs):
        calls["hazard"] = dict(kwargs)
        return hazard_path

    def fake_ppo(**kwargs):
        calls["ppo"] = {"pretrained_model_path": kwargs.get("pretrained_model_path")}
        return object(), _FakeTrainer(ckpt_dir)

    monkeypatch.setattr(
        train_mod,
        "load_question_splits_with_metadata",
        lambda _args, _config: ([fake_q], [fake_q], [fake_q], fake_manifest),
    )
    monkeypatch.setattr(ppo_mod, "run_ppo_training", fake_ppo)
    monkeypatch.setattr(hz_mod, "run_hazard_pretrain", fake_hazard)

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
            "--hazard-ablation",
            "shuffled_nll",
            "--beta-terminal",
            "2.0",
        ],
    )

    train_mod.main()

    assert "hazard" in calls, "run_hazard_pretrain must run with --hazard-pretrain"
    assert calls["hazard"].get("ablation") == "shuffled_nll", (
        "main() must thread --hazard-ablation into run_hazard_pretrain as the "
        "keyword-only ablation argument"
    )
    assert calls["hazard"].get("pretrained_model_path") == supervised_path
    assert calls["ppo"]["pretrained_model_path"] == hazard_path

    cfg = json.loads((ckpt_dir / "config_used.json").read_text(encoding="utf-8"))
    hazard_block = cfg["hazard"]
    assert hazard_block["pretrain"] is True
    assert hazard_block["ablation"] == "shuffled_nll"
    assert hazard_block["beta_terminal"] == 2.0
    assert hazard_block["freeze_answer_head"] is False


# ---------------------------------------------------------------------------
# Mini-audit-verify F2: periodic progress output from the hazard loop
# ---------------------------------------------------------------------------


def test_f2_hazard_loop_prints_periodic_progress(
    supervised_ckpt: str, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Tests mini-audit-verify F2 [integration]: periodic progress lines.

    The hazard training loop prints one terse progress line every 25
    optimizer steps (the efficacy harness's MA-006 output-staleness
    watchdog needs periodic child output between phase banners) while
    ``hazard_history.json`` keeps its pinned R-010 schema exactly.

    25 epochs x 1 single-prefix question = 25 optimizer steps => exactly
    one progress line, at step 25 (epoch 24, question 0).
    """
    pytest.importorskip("transformers")
    hz = _hazard_module()
    config = _hazard_config(tmp_path / "checkpoints", epochs=25, lr=1e-3)
    questions = [_make_mc_question("q_t1", prefixes=["Who was the first"])]

    out_path = hz.run_hazard_pretrain(
        config=config,
        train_questions=questions,
        pretrained_model_path=supervised_ckpt,
    )

    printed = capsys.readouterr().out
    progress_lines = [
        line for line in printed.splitlines()
        if line.startswith("[hazard] step ")
    ]
    assert len(progress_lines) == 1, printed
    line = progress_lines[0]
    assert line.startswith("[hazard] step 25 ")
    assert "epoch 24" in line
    assert "question 0" in line
    assert "loss" in line

    # R-010: the pinned history schema is unchanged by the progress print.
    history = _load_history(out_path)
    assert set(history) == {"steps", "config", "wall_clock_seconds"}
    assert len(history["steps"]) == 25
    assert set(history["steps"][0]) == {"epoch", "question_index", "loss"}
