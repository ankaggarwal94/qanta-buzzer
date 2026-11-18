"""Prefix answer warm-start uses exactly the displayed clues and valid answer slots."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from scripts.train_t5_policy import flatten_config
from tests.test_supervised_t5 import _make_question
from training.train_supervised_t5 import SupervisedTrainer


class AnswerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        words = ["<pad>", "<unk>", "CLUES:", "|", "CHOICES:", "(1)", "(2)", "(3)",
                 "opening", "middle", "ending", "secret", "north", "south", "east"]
        backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>"
        )
        self.logits = torch.nn.Parameter(torch.tensor([0., 1., 5.]))
        self.seen = []

    def predict_answer(self, input_ids, attention_mask):
        self.seen.extend(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True))
        logits = self.logits.expand(input_ids.shape[0], -1)
        return logits, logits.argmax(-1)


def question(qid, prefixes, options, gold):
    text = prefixes[-1] + " secret"
    return replace(_make_question(qid), question=text, tokens=text.split(),
                   cumulative_prefixes=prefixes, run_indices=[len(s.split()) - 1 for s in prefixes],
                   options=options, option_answer_primary=options, gold_index=gold, answer_primary=options[gold])


def test_prefix_training_and_validation_keep_visible_clues_and_active_choices(tmp_path):
    train = question("tr", ["opening", "opening", "opening middle"], ["north", "south"], 1)
    val = question("va", ["ending"], ["east", "north", "south"], 2)
    model = AnswerModel()
    config = flatten_config({"model": {"device": "cpu", "num_choices": 3},
                             "supervised": {"input_mode": "prefixes"}})
    trainer = SupervisedTrainer(model, [train], [val], {**config, "checkpoint_dir": tmp_path})
    dormant_before = model.logits.detach().clone()
    trainer.weight_decay = 0
    trainer.optimizer.param_groups[0]["weight_decay"] = 0
    loss, acc = trainer.train_epoch()
    assert sorted(model.seen) == [
        "CLUES: opening middle | CHOICES: (1) north (2) south",
        "CLUES: opening | CHOICES: (1) north (2) south",
    ]
    assert acc == 1 and loss == pytest.approx(torch.nn.functional.softplus(torch.tensor(-1.)).item())
    assert not torch.equal(model.logits[:2], dormant_before[:2])
    assert model.logits[2] == dormant_before[2]
    model.seen.clear()
    assert trainer.validate()[1] == 1
    assert model.seen == ["CLUES: ending | CHOICES: (1) east (2) north (3) south"]


def test_prefix_split_detects_source_reuse(tmp_path):
    q = question("source", ["opening"], ["north", "south"], 0)
    with pytest.raises(ValueError):
        SupervisedTrainer(AnswerModel(), [q], [replace(q, qid="another")],
                          {"supervised_input_mode": "prefixes", "checkpoint_dir": tmp_path})


def test_long_prefix_keeps_the_full_menu(tmp_path):
    train = question("tr", ["opening " * 30 + "ending"], ["north", "south"], 1)
    val = question("va", ["middle"], ["north", "south"], 0)
    model = AnswerModel()
    trainer = SupervisedTrainer(model, [train], [val], {"max_input_length": 11,
        "num_choices": 3, "supervised_input_mode": "prefixes", "checkpoint_dir": tmp_path})
    trainer.train_epoch()
    assert model.seen == ["CLUES: opening opening opening ending | CHOICES: (1) north (2) south"]


@pytest.mark.parametrize("options,gold", [(["north", "south"], 2), (["north"] * 4, 0)])
def test_prefix_rejects_invalid_answer_slots(tmp_path, options, gold):
    q = question("tr", ["opening"], options, 0)
    val = question("va", ["middle"], ["north","south"], 0)
    with pytest.raises(ValueError, match="menu|gold"):
        SupervisedTrainer(AnswerModel(), [replace(q, gold_index=gold)], [val],
                          {"supervised_input_mode": "prefixes", "num_choices": 3,
                           "checkpoint_dir": tmp_path})


def test_default_uses_complete_text_once(tmp_path):
    train = question("tr", ["opening", "opening middle"], ["north","south","east"], 2)
    val = question("va", ["ending"], ["north","south","east"], 2)
    model = AnswerModel()
    trainer = SupervisedTrainer(model, [train], [val], {"checkpoint_dir": tmp_path})
    trainer.train_epoch()
    assert model.seen == ["CLUES: opening middle secret | CHOICES: (1) north (2) south (3) east"]

def test_reveal_that_contains_a_future_clue_fails(tmp_path):
    train = question("tr", ["opening", "opening middle"], ["north", "south"], 0)
    val = question("va", ["ending"], ["east", "south"], 0)
    leaked = replace(train, cumulative_prefixes=["opening middle", "opening middle"])
    with pytest.raises(ValueError, match="prefix|reveal"):
        SupervisedTrainer(AnswerModel(), [leaked], [val],
                          {"supervised_input_mode": "prefixes", "checkpoint_dir": tmp_path})


def test_supervised_only_cli_saves_updated_t5_and_original_source_ids(tmp_path, monkeypatch):
    import json
    import sys
    from dataclasses import asdict
    from tokenizers.models import Unigram
    from transformers import T5Config, T5EncoderModel

    from scripts import train_t5_policy

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model_dir, output = tmp_path / "encoder", tmp_path / "output"
    pieces = ["<pad>", "</s>", "<unk>", *list("abcdefghijklmnopqrstuvwxyz:|()123")]
    tokenizer = Tokenizer(Unigram([(piece, -1.) for piece in pieces], unk_id=2))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    encoder = T5EncoderModel(T5Config(vocab_size=256, d_model=16,
                                      d_kv=8, d_ff=32, num_heads=2, num_layers=1, dropout_rate=0))
    encoder.save_pretrained(model_dir)
    tokenizer.save(str(model_dir / "tokenizer.json"))
    initial = encoder.get_input_embeddings().weight.detach().clone()
    for split, q in {
        "train": question("tr", ["opening", "opening middle"], ["north", "south"], 1),
        "val": question("va", ["ending"], ["east", "north", "south"], 2),
        "test": question("te", ["middle"], ["south", "east"], 1),
    }.items():
        (tmp_path / f"{split}_dataset.json").write_text(json.dumps([asdict(q)]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "train_t5_policy.py", "--supervised-only", "--mc-path", str(tmp_path / "mc_dataset.json"),
        f"model.model_name={model_dir.as_posix()}", "model.device=cpu", "model.num_choices=3",
        f"supervised.checkpoint_dir={output.as_posix()}", "supervised.input_mode=prefixes",
        "supervised.epochs=1", "supervised.batch_size=2", "supervised.grad_accum_steps=1",
    ])
    train_t5_policy.main()
    checkpoint = output / "supervised" / "best_model"
    actual = T5EncoderModel.from_pretrained(checkpoint).get_input_embeddings().weight.detach()
    assert not torch.equal(actual, initial)
    assert (checkpoint / "policy_head.pt").is_file()
    provenance = json.loads((checkpoint / "split_manifest.json").read_bytes())
    assert [provenance[f"{s}_qids"] for s in ["train", "val", "test"]] == [["tr"], ["va"], ["te"]]
    history = json.loads((checkpoint.parent / "history.json").read_bytes())
    assert (history["config"]["train_examples"], history["config"]["val_examples"]) == (2, 1)
    assert not (output / "ppo_t5").exists()
