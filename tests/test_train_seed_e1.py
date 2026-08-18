"""RED tests for R-001 (enabler E-1): ``--seed`` on scripts/train_t5_policy.py.

Covers the hazard-efficacy-eval spec:

- R-001 [integration]: ``--seed <int>`` (default None) seeds Python ``random``,
  NumPy and torch **before each** training phase (supervised / hazard / PPO);
  different seeds produce different training-phase RNG draws, identical seeds
  reproduce them; the split manifest qids are identical across seeds; the seed
  is recorded top-level in ``config_used.json``.
- R-003 (producer contract only): ``main()`` records the ``hazard`` block
  ``{pretrain, beta_terminal, freeze_answer_head, ablation}`` plus top-level
  ``seed`` into ``config_used.json`` so every run dir is self-describing.

Test design (per the spec's Entry/Through/Exit):

- Entry: ``main()`` with the three trainers monkeypatched **on their lazy-import
  source modules** (``training.train_supervised_t5`` / ``training.hazard_pretrain``
  / ``training.train_ppo_t5``), following
  ``tests/test_train_t5_policy_script.py::test_main_writes_t5_config_and_split_manifest``.
- Through: real ``parse_args`` / ``load_config_with_overrides`` /
  ``load_question_splits_with_metadata``. The split loader is NOT mocked
  (# DECISION: the R-001 Through constraint requires the real loader, so the
  fixture writes real ``{train,val,test}_dataset.json`` files into a tmp dir and
  points ``--mc-path`` at it — this is deterministic AND real, satisfying both
  the spec and the determinism requirement).  The seeding helper is never mocked.
- Exit: per-phase captured RNG draws differ across ``--seed 1`` vs ``--seed 2``
  and match exactly across two ``--seed 1`` runs; split-manifest qids identical
  in all cases; ``config_used.json`` records the seed.

Phase-boundary discriminator: each phase fake, after capturing its draws,
scrambles all three global RNGs from OS entropy.  A later phase's captures can
only reproduce across two same-seed runs if ``main()`` re-seeds *before that
phase* — seeding once at the top of ``main()`` cannot pass these tests.

RED expectation: every test fails at runtime today via ``SystemExit(2)``
(argparse does not know ``--seed``) or ``AttributeError``/``AssertionError`` —
never at import/collection time.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.train_t5_policy as train_mod


# ---------------------------------------------------------------------------
# Fixture data: real persisted split artifacts for the REAL split loader
# ---------------------------------------------------------------------------

_TRAIN_QIDS = ["seed_train_a", "seed_train_b"]
_VAL_QIDS = ["seed_val_a"]
_TEST_QIDS = ["seed_test_a"]


def _mc_question_dict(qid: str, gold_index: int = 0) -> dict:
    """Return a complete MCQuestion JSON record (schema: qb_data/mc_builder.py).

    All fields required by ``scripts._common.mc_question_from_dict`` are
    present so the REAL persisted-split loader can deserialize it.
    """
    return {
        "qid": qid,
        "question": "Who was the first president of the United States",
        "tokens": ["Who", "was", "the", "first", "president"],
        "answer_primary": "George Washington",
        "clean_answers": ["George Washington"],
        "run_indices": [0, 2, 4],
        "human_buzz_positions": [],
        "category": "History",
        "cumulative_prefixes": [
            "Who",
            "Who was the",
            "Who was the first president",
        ],
        "options": [
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        "gold_index": gold_index,
        "option_profiles": [
            "George Washington first president commander revolutionary war",
            "Thomas Jefferson third president declaration independence",
            "John Adams second president Massachusetts diplomat",
            "Benjamin Franklin inventor diplomat Philadelphia",
        ],
        "option_answer_primary": [
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        "distractor_strategy": "test",
    }


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory) -> Path:
    """Write real persisted split artifacts so the REAL loader resolves them."""
    d = tmp_path_factory.mktemp("seed_e1_splits")
    train = [_mc_question_dict(q, gold_index=i) for i, q in enumerate(_TRAIN_QIDS)]
    val = [_mc_question_dict(q, gold_index=1) for q in _VAL_QIDS]
    test = [_mc_question_dict(q, gold_index=2) for q in _TEST_QIDS]
    (d / "mc_dataset.json").write_text(
        json.dumps(train + val + test), encoding="utf-8"
    )
    (d / "train_dataset.json").write_text(json.dumps(train), encoding="utf-8")
    (d / "val_dataset.json").write_text(json.dumps(val), encoding="utf-8")
    (d / "test_dataset.json").write_text(json.dumps(test), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Harness: run main() with phase fakes that capture RNG draws at call time
# ---------------------------------------------------------------------------


class _FakeTrainer:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir


def _draw() -> tuple[float, float, float]:
    """Draw one value from each global RNG (torch, python random, numpy)."""
    return (torch.rand(1).item(), random.random(), float(np.random.rand()))


def _scramble_rngs() -> None:
    """Scramble all three global RNGs from OS entropy (never reproducible)."""
    random.seed(os.urandom(16))
    np.random.seed(int.from_bytes(os.urandom(4), "little"))
    torch.manual_seed(int.from_bytes(os.urandom(4), "little"))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    split_dir: Path,
    run_dir: Path,
    extra_argv: list[str],
    *,
    hazard: bool = True,
    perturb: bool = True,
) -> tuple[dict, Path]:
    """Run ``main()`` with capture fakes; return (captures, ppo_checkpoint_dir).

    Patches target the lazy-import SOURCE modules (main() does
    ``from training.X import run_X`` inside the function body, so the source
    module attribute is what gets resolved at call time).
    """
    import training.hazard_pretrain as hz_mod
    import training.train_ppo_t5 as ppo_mod
    import training.train_supervised_t5 as sup_mod

    ppo_dir = run_dir / "ppo_t5"
    sup_dir = run_dir / "supervised"
    hazard_path = str(run_dir / "hazard" / "best_model")
    captures: dict[str, list] = {"supervised": [], "hazard": [], "ppo": []}

    def fake_supervised(*_args, **_kwargs):
        captures["supervised"].append(_draw())
        if perturb:
            _scramble_rngs()
        return object(), _FakeTrainer(sup_dir)

    def fake_hazard(*_args, **_kwargs):
        captures["hazard"].append(_draw())
        if perturb:
            _scramble_rngs()
        return hazard_path

    def fake_ppo(*_args, **_kwargs):
        captures["ppo"].append(_draw())
        return object(), _FakeTrainer(ppo_dir)

    monkeypatch.setattr(sup_mod, "run_supervised_training", fake_supervised)
    monkeypatch.setattr(hz_mod, "run_hazard_pretrain", fake_hazard)
    monkeypatch.setattr(ppo_mod, "run_ppo_training", fake_ppo)

    argv = [
        "train_t5_policy.py",
        "--config",
        str(train_mod.PROJECT_ROOT / "configs" / "t5_policy.yaml"),
        "--mc-path",
        str(split_dir / "mc_dataset.json"),
    ]
    if hazard:
        argv += ["--hazard-pretrain", "--beta-terminal", "2.5", "--freeze-answer-head"]
    argv += extra_argv
    monkeypatch.setattr(sys, "argv", argv)

    train_mod.main()
    return captures, ppo_dir


def _read_sidecar(ppo_dir: Path, name: str) -> dict:
    return json.loads((ppo_dir / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# R-001: per-phase seeding — divergence across seeds, reproduction within one
# ---------------------------------------------------------------------------


def test_r001_seed_reseeds_before_each_phase_and_reproduces(
    split_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests R-001 [integration]: seeds diverge across values, replay within one.

    Three ``main()`` runs (``--seed 1``, ``--seed 2``, ``--seed 1``) with all
    three phase trainers faked.  Each fake records one draw from each global
    RNG at call time, then scrambles the RNGs from OS entropy — so a later
    phase's draws can only match across the two seed-1 runs if ``main()``
    re-seeds immediately before that phase (the spec's "before each training
    phase" requirement).  The split manifest must be byte-identical across
    seeds (the training seed is separate from ``data.seed``).
    """
    runs: dict[str, tuple[dict, Path]] = {}
    for label, seed in (("a", 1), ("b", 2), ("c", 1)):
        _scramble_rngs()  # main() entry state is never reproducible by luck
        runs[label] = _run_main(
            monkeypatch,
            split_dir,
            tmp_path / f"run_{label}",
            ["--seed", str(seed)],
        )

    caps_a, dir_a = runs["a"]
    caps_b, dir_b = runs["b"]
    caps_c, dir_c = runs["c"]

    for phase in ("supervised", "hazard", "ppo"):
        assert len(caps_a[phase]) == 1, f"{phase} trainer must be called once"
        assert len(caps_b[phase]) == 1
        assert len(caps_c[phase]) == 1
        a, b, c = caps_a[phase][0], caps_b[phase][0], caps_c[phase][0]
        # Same seed ⇒ exact replay of every RNG family, per phase. This is
        # the load-bearing assertion: an unseeded (or only-once-seeded) RNG
        # cannot reproduce after the previous phase scrambled global state.
        assert a == c, (
            f"--seed 1 must reproduce {phase}-phase draws exactly "
            f"(torch, random, numpy): {a} != {c}"
        )
        # Different seed ⇒ every RNG family diverges (catches an
        # implementation that seeds only one of the three libraries).
        for idx, rng_name in enumerate(("torch", "random", "numpy")):
            assert a[idx] != b[idx], (
                f"{phase}-phase {rng_name} draw identical across --seed 1 "
                f"and --seed 2: {a[idx]}"
            )

    # Split manifest qids identical across seeds and equal to the fixture.
    manifests = [
        _read_sidecar(d, "split_manifest.json") for d in (dir_a, dir_b, dir_c)
    ]
    for manifest in manifests:
        assert manifest["train_qids"] == _TRAIN_QIDS
        assert manifest["val_qids"] == _VAL_QIDS
        assert manifest["test_qids"] == _TEST_QIDS
        assert manifest["source"] == "persisted_artifacts"

    # config_used.json records the top-level seed per run.
    assert _read_sidecar(dir_a, "config_used.json")["seed"] == 1
    assert _read_sidecar(dir_b, "config_used.json")["seed"] == 2
    assert _read_sidecar(dir_c, "config_used.json")["seed"] == 1


# ---------------------------------------------------------------------------
# R-001 + R-003 producer contract: config_used.json seed + hazard block
# ---------------------------------------------------------------------------


def test_r001_r003_config_used_records_seed_and_hazard_block(
    split_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests R-001/R-003 [integration]: config_used.json is self-describing.

    A hazard run must record top-level ``seed`` (int, not bool) and the
    ``hazard`` block ``{pretrain, beta_terminal, freeze_answer_head, ablation}``
    with the ACTUAL threaded CLI values; a control run (no hazard flags, no
    seed) must still carry the full hazard block with ``pretrain: false`` so
    the R-003 harness can diff arms on a stable key set.

    # DECISION: the hazard block is pinned to exactly the four R-003 keys and
    # is present on EVERY run (control arms included) — R-003's config-diff
    # assertion needs a stable schema, so absence-when-disabled is a bug.
    """
    _, hazard_dir = _run_main(
        monkeypatch,
        split_dir,
        tmp_path / "hazard_run",
        ["--seed", "7"],
        hazard=True,
        perturb=False,
    )
    cfg = _read_sidecar(hazard_dir, "config_used.json")

    assert cfg["seed"] == 7
    assert isinstance(cfg["seed"], int) and not isinstance(cfg["seed"], bool)

    hazard_block = cfg["hazard"]
    assert set(hazard_block) == {
        "pretrain",
        "beta_terminal",
        "freeze_answer_head",
        "ablation",
    }
    assert hazard_block["pretrain"] is True
    assert isinstance(hazard_block["beta_terminal"], float)
    assert hazard_block["beta_terminal"] == 2.5  # threaded from --beta-terminal
    assert hazard_block["freeze_answer_head"] is True  # threaded from the flag
    assert hazard_block["ablation"] is None  # no --hazard-ablation passed

    # Control arm: no hazard flags, no --seed — block still present.
    captures, control_dir = _run_main(
        monkeypatch,
        split_dir,
        tmp_path / "control_run",
        [],
        hazard=False,
        perturb=False,
    )
    assert captures["hazard"] == [], "hazard trainer must not run without the flag"
    control_cfg = _read_sidecar(control_dir, "config_used.json")
    assert control_cfg.get("seed", None) is None  # null or absent per R-001
    control_block = control_cfg["hazard"]
    assert set(control_block) == {
        "pretrain",
        "beta_terminal",
        "freeze_answer_head",
        "ablation",
    }
    assert control_block["pretrain"] is False
    assert control_block["freeze_answer_head"] is False
    assert control_block["beta_terminal"] == 1.0  # argparse default
    assert control_block["ablation"] is None


# ---------------------------------------------------------------------------
# R-001: default (no --seed) keeps today's unseeded behavior — no RNG touch
# ---------------------------------------------------------------------------


def test_r001_default_no_seed_leaves_rng_untouched(
    split_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests R-001 [integration]: ``--seed`` defaults to None and seeds nothing.

    ``parse_args`` must expose ``seed`` (default ``None``, ``type=int``).
    Without ``--seed``, ``main()`` must not call ANY seeding and must not
    consume global RNG state: with the RNGs pre-seeded to 777 by the test,
    the three phase fakes must observe exactly the draw sequence a bare
    777-seeded process would produce.  ``config_used.json`` records the seed
    as null/absent.
    """
    # Pre-import the lazy trainer modules so first-import side effects cannot
    # perturb RNG state between the expected-draw computation and main().
    import training.hazard_pretrain  # noqa: F401
    import training.train_ppo_t5  # noqa: F401
    import training.train_supervised_t5  # noqa: F401

    # Flag surface: default None, typed int.
    monkeypatch.setattr(sys, "argv", ["train_t5_policy.py", "--seed", "3"])
    args = train_mod.parse_args()
    assert args.seed == 3
    assert isinstance(args.seed, int) and not isinstance(args.seed, bool)
    monkeypatch.setattr(sys, "argv", ["train_t5_policy.py"])
    assert train_mod.parse_args().seed is None

    _seed_all(777)
    expected = [_draw(), _draw(), _draw()]  # supervised, hazard, ppo in order

    _seed_all(777)
    captures, ppo_dir = _run_main(
        monkeypatch,
        split_dir,
        tmp_path / "noseed_run",
        [],
        hazard=True,
        perturb=False,
    )
    observed = [
        captures["supervised"][0],
        captures["hazard"][0],
        captures["ppo"][0],
    ]
    assert observed == expected, (
        "without --seed, main() must not seed or consume the global RNGs; "
        f"expected {expected}, observed {observed}"
    )

    cfg = _read_sidecar(ppo_dir, "config_used.json")
    assert cfg.get("seed", None) is None
