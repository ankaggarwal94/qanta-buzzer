"""Checkpoint selection must use a successful save from the current T5 run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import train_t5_policy as script
from training import train_ppo_t5 as ppo
from training import train_supervised_t5 as supervised


class TinyModel(torch.nn.Module):
    """One parameter and an observable save/load marker; no tokenizer or download."""

    def __init__(self, config=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.device = "cpu"
        self.marker = "initial"
        self.loads = []

    def save(self, path):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "weights.txt").write_text(self.marker, encoding="utf-8")

    def load(self, path):
        self.marker = (Path(path) / "weights.txt").read_text(encoding="utf-8")
        self.loads.append((Path(path), self.marker))


def _stale_checkpoint(base, phase):
    checkpoint = base / phase / "best_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.txt").write_text("stale", encoding="utf-8")
    (checkpoint / "user_notes.txt").write_text("preserve", encoding="utf-8")
    return checkpoint


def test_all_zero_supervised_validation_saves_current_first_epoch(
    tmp_path, monkeypatch
):
    stale = _stale_checkpoint(tmp_path, "supervised")
    model = TinyModel()
    monkeypatch.setattr(supervised, "T5PolicyModel", lambda config: model)

    def train_epoch(trainer):
        trainer.model.marker = f"epoch-{trainer.current_epoch + 1}"
        return 1.0, 0.0

    monkeypatch.setattr(supervised.SupervisedTrainer, "train_epoch", train_epoch)
    monkeypatch.setattr(
        supervised.SupervisedTrainer, "validate", lambda self: (1.0, 0.0)
    )
    _, trainer = supervised.run_supervised_training(
        {"checkpoint_dir": str(tmp_path), "supervised_epochs": 2},
        [object()],
        [object()],
    )

    assert trainer.best_val_acc == 0.0
    assert trainer.best_checkpoint_path == stale
    assert (stale / "weights.txt").read_text(encoding="utf-8") == "epoch-1"
    assert not model.loads
    assert (stale / "user_notes.txt").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "module,key,phase",
    [
        (supervised, "supervised_epochs", "supervised"),
        (ppo, "ppo_iterations", "ppo_t5"),
    ],
)
@pytest.mark.parametrize("count", [0, -1])
def test_nonpositive_run_count_rejects_before_model_load_and_preserves_old_files(
    tmp_path, monkeypatch, module, key, phase, count
):
    stale = _stale_checkpoint(tmp_path, phase)

    def reject_model(*args, **kwargs):
        raise AssertionError("invalid training run must not initialize a model")

    monkeypatch.setattr(module, "T5PolicyModel", reject_model)
    run = (
        module.run_supervised_training
        if module is supervised
        else module.run_ppo_training
    )
    with pytest.raises(ValueError, match="must be positive"):
        run({"checkpoint_dir": str(tmp_path), key: count}, [object()], [object()])
    assert (stale / "weights.txt").read_text(encoding="utf-8") == "stale"
    assert (stale / "user_notes.txt").exists()


@pytest.mark.parametrize("kind", ["supervised", "ppo"])
def test_reused_trainer_clears_old_checkpoint_claim_before_zero_length_run(
    tmp_path, kind
):
    cls = supervised.SupervisedTrainer if kind == "supervised" else ppo.PPOTrainer
    trainer = cls(
        TinyModel(), [object()], [object()], {"checkpoint_dir": str(tmp_path)}
    )
    stale = _stale_checkpoint(
        tmp_path, "supervised" if kind == "supervised" else "ppo_t5"
    )
    trainer.best_checkpoint_path = stale
    if kind == "supervised":
        trainer.epochs = 0
    else:
        trainer.iterations = 0
    with pytest.raises(ValueError, match="must be positive"):
        trainer.train()
    assert trainer.best_checkpoint_path is None
    assert (stale / "weights.txt").read_text(encoding="utf-8") == "stale"


def test_short_ppo_run_validates_and_reloads_new_checkpoint_before_test(
    tmp_path, monkeypatch
):
    stale = _stale_checkpoint(tmp_path, "ppo_t5")
    model = TinyModel()
    monkeypatch.setattr(ppo, "T5PolicyModel", lambda config: model)
    monkeypatch.setattr(
        ppo.PPOTrainer, "collect_rollouts", lambda self, count: ppo.RolloutBuffer()
    )

    def update(trainer, buffer):
        trainer.model.marker = f"iteration-{trainer.current_iteration + 1}"
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    evaluations = []

    def evaluate(trainer, questions, split):
        evaluations.append((split, trainer.model.marker))
        return {"accuracy": 0.0, "average_reward": -1.0, "avg_episode_length": 1.0}

    monkeypatch.setattr(ppo.PPOTrainer, "update_policy", update)
    monkeypatch.setattr(ppo.PPOTrainer, "evaluate_questions", evaluate)
    _, trainer = ppo.run_ppo_training(
        {
            "checkpoint_dir": str(tmp_path),
            "ppo_iterations": 2,
            "eval_interval": 10,
            "save_interval": 20,
        },
        [object()],
        [object()],
        test_questions=[object()],
    )

    assert evaluations == [("val", "iteration-2"), ("test", "iteration-2")]
    assert model.loads == [(stale, "iteration-2")]
    assert trainer.best_checkpoint_path == stale
    assert trainer.history[-1]["iteration"] == 2
    assert (stale / "user_notes.txt").exists()


@pytest.mark.parametrize("module,phase", [(supervised, "supervised"), (ppo, "ppo_t5")])
def test_training_wrapper_never_reloads_an_existing_checkpoint_without_a_current_save(
    tmp_path, monkeypatch, module, phase
):
    stale = _stale_checkpoint(tmp_path, phase)
    model = TinyModel()
    monkeypatch.setattr(module, "T5PolicyModel", lambda config: model)
    cls = module.SupervisedTrainer if module is supervised else module.PPOTrainer
    monkeypatch.setattr(cls, "train", lambda self: {})
    run = (
        module.run_supervised_training
        if module is supervised
        else module.run_ppo_training
    )
    with pytest.raises(
        RuntimeError, match="did not save a best checkpoint in this run"
    ):
        run(
            {"checkpoint_dir": str(tmp_path)},
            [object()],
            [object()],
            test_questions=[object()],
        )
    assert not model.loads
    assert (stale / "weights.txt").read_text(encoding="utf-8") == "stale"


@pytest.mark.parametrize("cls", [supervised.SupervisedTrainer, ppo.PPOTrainer])
def test_failed_checkpoint_save_does_not_mark_a_checkpoint_fresh(
    tmp_path, monkeypatch, cls
):
    trainer = cls(
        TinyModel(), [object()], [object()], {"checkpoint_dir": str(tmp_path)}
    )

    checkpoint = trainer.save_checkpoint(is_best=True)
    trainer.model.marker = "partially written"

    def fail_save(*args, **kwargs):
        raise OSError("simulated optimizer-state write failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="write failure"):
        trainer.save_checkpoint(is_best=True)
    assert (checkpoint / "weights.txt").read_text(encoding="utf-8") == "initial"
    assert trainer.best_checkpoint_path == checkpoint


def test_concurrent_training_cannot_overwrite_the_checkpoint_being_evaluated(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from filelock import Timeout
    from training.checkpoints import training_lock

    with training_lock(tmp_path), ThreadPoolExecutor(max_workers=1) as pool:
        overlapping = pool.submit(
            ppo.run_ppo_training,
            {"checkpoint_dir": str(tmp_path)},
            [object()],
            [object()],
        )
        with pytest.raises(Timeout):
            overlapping.result()


def test_script_refuses_to_stamp_lineage_on_an_old_directory(tmp_path):
    stale = _stale_checkpoint(tmp_path, "supervised")
    trainer = SimpleNamespace(checkpoint_dir=stale.parent, best_checkpoint_path=None)
    with pytest.raises(RuntimeError, match="refusing to use the checkpoint"):
        script._current_best_checkpoint(trainer)
    assert not (stale / "split_manifest.json").exists()


def test_script_stamps_only_fresh_checkpoint_not_stale_parent_or_siblings(
    tmp_path, monkeypatch
):
    trainers = {}
    for phase in ("supervised", "ppo_t5"):
        best = _stale_checkpoint(tmp_path, phase)
        (best / "weights.txt").write_text("fresh weights", encoding="utf-8")
        (best.parent / "split_manifest.json").write_text(
            '{"old": true}', encoding="utf-8"
        )
        stale_sibling = best.parent / "epoch_99"
        stale_sibling.mkdir()
        periodic = best.parent / "iter_1"
        periodic.mkdir()
        trainers[phase] = SimpleNamespace(
            checkpoint_dir=best.parent,
            best_checkpoint_path=best,
            periodic_checkpoints=[periodic],
        )

    args = SimpleNamespace(
        skip_supervised=False, hazard_pretrain=False, model_path=None, overrides=[]
    )
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(
        script,
        "load_config_with_overrides",
        lambda args: {
            "model": {"device": "cpu"},
            "supervised": {"checkpoint_dir": str(tmp_path)},
            "ppo": {},
            "data": {},
        },
    )
    manifest = {"source": "current-run"}
    monkeypatch.setattr(
        script,
        "load_question_splits_with_metadata",
        lambda *args: (
            [object()],
            [object()],
            [object()],
            manifest,
        ),
    )

    def train_supervised(**kwargs):
        kwargs["write_checkpoint_sidecars"](trainers["supervised"].best_checkpoint_path)
        return object(), trainers["supervised"]

    monkeypatch.setattr(supervised, "run_supervised_training", train_supervised)
    inputs = []

    def run_ppo(**kwargs):
        inputs.append(kwargs["pretrained_model_path"])
        trainer = trainers["ppo_t5"]
        for checkpoint in [trainer.best_checkpoint_path, *trainer.periodic_checkpoints]:
            kwargs["write_checkpoint_sidecars"](checkpoint)
        return object(), trainer

    monkeypatch.setattr(ppo, "run_ppo_training", run_ppo)
    script.main()

    assert inputs == [str(trainers["supervised"].best_checkpoint_path)]
    for trainer in trainers.values():
        current = trainer.best_checkpoint_path / "split_manifest.json"
        assert json.loads(current.read_text(encoding="utf-8")) == manifest
        assert (trainer.checkpoint_dir / "split_manifest.json").read_text(
            encoding="utf-8"
        ) == '{"old": true}'
        assert not (
            trainer.checkpoint_dir / "epoch_99" / "split_manifest.json"
        ).exists()
    for checkpoint in [
        trainers["ppo_t5"].best_checkpoint_path,
        *trainers["ppo_t5"].periodic_checkpoints,
    ]:
        assert (checkpoint / "config_used.json").exists()
        assert (
            json.loads((checkpoint / "split_manifest.json").read_text(encoding="utf-8"))
            == manifest
        )


@pytest.mark.parametrize("kind", ["supervised", "ppo"])
def test_nonfinite_validation_cannot_select_an_old_checkpoint(
    tmp_path, monkeypatch, kind
):
    cls = supervised.SupervisedTrainer if kind == "supervised" else ppo.PPOTrainer
    config = {
        "checkpoint_dir": str(tmp_path),
        "supervised_epochs": 1,
        "ppo_iterations": 1,
    }
    trainer = cls(TinyModel(), [object()], [object()], config)
    stale = _stale_checkpoint(
        tmp_path, "supervised" if kind == "supervised" else "ppo_t5"
    )
    if kind == "supervised":
        monkeypatch.setattr(trainer, "train_epoch", lambda: (1.0, 0.0))
        monkeypatch.setattr(trainer, "validate", lambda: (1.0, float("nan")))
    else:
        monkeypatch.setattr(
            trainer, "collect_rollouts", lambda count: ppo.RolloutBuffer()
        )
        monkeypatch.setattr(
            trainer,
            "update_policy",
            lambda buffer: {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            },
        )
        monkeypatch.setattr(
            trainer, "validate", lambda: {"average_reward": float("nan")}
        )
    with pytest.raises(ValueError, match="must be finite"):
        trainer.train()
    assert trainer.best_checkpoint_path is None
    assert (stale / "weights.txt").read_text(encoding="utf-8") == "stale"


@pytest.mark.parametrize("phase", ["supervised", "ppo"])
def test_training_requires_validation_examples_to_select_a_best_checkpoint(
    tmp_path, phase
):
    cls = supervised.SupervisedTrainer if phase == "supervised" else ppo.PPOTrainer
    trainer = cls(TinyModel(), [object()], [], {"checkpoint_dir": str(tmp_path)})
    with pytest.raises(ValueError, match="validation questions"):
        trainer.train()
    assert trainer.best_checkpoint_path is None


def test_replacing_checkpoint_retires_obsolete_tokenizer_adapter_and_shards(tmp_path):
    trainer = ppo.PPOTrainer(
        TinyModel(), [object()], [object()], {"checkpoint_dir": str(tmp_path)}
    )
    old = _stale_checkpoint(tmp_path, "ppo_t5")
    obsolete = [
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.4.20.0.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
    ]
    for name in obsolete:
        (old / name).touch()
    templates = old / "additional_chat_templates"
    templates.mkdir()
    (templates / "old.jinja").touch()
    trainer.save_checkpoint(is_best=True)
    assert not any((old / name).exists() for name in obsolete)
    assert not templates.exists()
    assert (old / "user_notes.txt").read_text(encoding="utf-8") == "preserve"


def test_programmatic_checkpoint_does_not_inherit_legacy_parent_metadata(tmp_path):
    from scripts._common import load_checkpoint_sidecar

    trainer = ppo.PPOTrainer(
        TinyModel(), [object()], [object()], {"checkpoint_dir": str(tmp_path)}
    )
    parent = trainer.checkpoint_dir
    for name in ("split_manifest.json", "config_used.json"):
        (parent / name).write_text('{"source": "stale"}', encoding="utf-8")
    path = trainer.save_checkpoint(is_best=True)
    for name in ("split_manifest.json", "config_used.json"):
        assert load_checkpoint_sidecar(path, name) == (None, path / name, None)


def test_rejected_supervised_validation_loss_cannot_select_a_model(
    tmp_path, monkeypatch
):
    trainer = supervised.SupervisedTrainer(
        TinyModel(),
        [object()],
        [object()],
        {
            "checkpoint_dir": str(tmp_path),
            "supervised_epochs": 1,
        },
    )
    monkeypatch.setattr(trainer, "train_epoch", lambda: (1.0, 0.5))
    monkeypatch.setattr(trainer, "validate", lambda: (float("nan"), 0.5))
    with pytest.raises(ValueError, match="must be finite"):
        trainer.train()
    assert trainer.best_checkpoint_path is None


def test_failed_publication_restores_previous_weights_sidecar_and_notes(
    tmp_path, monkeypatch
):
    trainer = ppo.PPOTrainer(
        TinyModel(), [object()], [object()], {"checkpoint_dir": str(tmp_path)}
    )
    old = _stale_checkpoint(tmp_path, "ppo_t5")
    (old / "split_manifest.json").write_text('{"source": "first"}', encoding="utf-8")
    original_rename = Path.rename

    def fail_publication(path, target):
        if path.name == "new" and target == old:
            raise PermissionError("simulated publication failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publication)
    with pytest.raises(PermissionError, match="publication failure"):
        trainer.save_checkpoint(is_best=True)
    assert (old / "weights.txt").read_text(encoding="utf-8") == "stale"
    assert (old / "user_notes.txt").read_text(encoding="utf-8") == "preserve"
    assert json.loads((old / "split_manifest.json").read_text(encoding="utf-8")) == {
        "source": "first"
    }
    assert trainer.best_checkpoint_path is None
