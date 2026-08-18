"""Hazard-efficacy harness: stop-probability probe + hazard_dynamics (R-010b).

Model-touching tests use t5-small on CPU with module-scoped fixtures and
skip when transformers is absent (guard/fixture idiom copied from
``tests/test_hazard_pretrain.py``). The probe contract being pinned:

- ``select_probe_questions``: deterministic sample = first
  ``min(32, len(train))`` train questions in split order.
- ``stop_prob_probe(model, questions)``: per-question P(BUZZ) per prefix
  position, probed under ``model.eval()`` + ``torch.no_grad()`` (dropout
  OFF => two calls are bit-identical), values in [0, 1].
- ``build_hazard_dynamics(before, after, hazard_history)``: per-position
  means before/after, expected-buzz-time delta, and first/second-half
  mean hazard loss from the pinned ``hazard_history.json`` schema.
- ``probe_and_write_hazard_dynamics``: real checkpoint save/load path,
  persists the block as ``hazard_dynamics.json`` for report assembly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_hazard_efficacy as harness
from qb_data.mc_builder import MCQuestion
from tests._hazard_efficacy_fixtures import make_hazard_history, write_json

_PREFIXES_T3 = ["Who", "Who was the", "Who was the first president"]
_PREFIXES_T4 = [
    "This inventor",
    "This inventor and diplomat",
    "This inventor and diplomat flew",
    "This inventor and diplomat flew a kite",
]


def _make_mc_question(
    qid: str = "q",
    gold_index: int = 0,
    prefixes: list[str] | None = None,
) -> MCQuestion:
    """Build an MCQuestion with a controllable number of cumulative prefixes.

    Mirrors the producer schema in ``qb_data/mc_builder.py`` (same helper
    idiom as ``tests/test_hazard_pretrain.py``).
    """
    if prefixes is None:
        prefixes = list(_PREFIXES_T3)
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


@pytest.fixture(scope="module")
def probe_model():
    """A real t5-small T5PolicyModel on CPU (fresh heads, train mode).

    Deliberately left in its constructor-default ``train()`` state so the
    determinism test below discriminates a probe that forgets
    ``model.eval()`` (dropout would make two probes differ).
    """
    pytest.importorskip("transformers")
    from models.t5_policy import T5PolicyModel

    return T5PolicyModel(
        {
            "model_name": "t5-small",
            "device": "cpu",
            "max_input_length": 64,
            "num_choices": 4,
        }
    )


@pytest.fixture(scope="module")
def probe_ckpts(probe_model, tmp_path_factory) -> tuple[str, str]:
    """Two saved checkpoints acting as the supervised / hazard pair."""
    pytest.importorskip("transformers")
    sup_dir = tmp_path_factory.mktemp("probe_supervised") / "best_model"
    hz_dir = tmp_path_factory.mktemp("probe_hazard") / "best_model"
    probe_model.save(str(sup_dir))
    probe_model.save(str(hz_dir))
    return str(sup_dir), str(hz_dir)


@pytest.fixture()
def probe_questions() -> list[MCQuestion]:
    return [
        _make_mc_question("q_t3", gold_index=0, prefixes=list(_PREFIXES_T3)),
        _make_mc_question("q_t4", gold_index=3, prefixes=list(_PREFIXES_T4)),
    ]


# Tests R-010b [unit]: the probe sample is the deterministic first
# min(32, len(train)) questions in split order.
def test_r010b_select_probe_questions_first_min_32_in_order() -> None:
    forty = list(range(40))
    selected = harness.select_probe_questions(forty)
    assert selected == forty[:32]

    five = ["a", "b", "c", "d", "e"]
    assert harness.select_probe_questions(five) == five

    assert harness.select_probe_questions([]) == []


# Tests R-010b [integration]: per-question per-position arrays of the right
# shape with probabilities in [0, 1] (real t5-small forward).
def test_r010b_stop_prob_probe_shapes_and_range(
    probe_model, probe_questions
) -> None:
    pytest.importorskip("transformers")
    per_question = harness.stop_prob_probe(probe_model, probe_questions)

    assert len(per_question) == 2
    assert [len(row) for row in per_question] == [3, 4]
    for row in per_question:
        for p in row:
            assert isinstance(p, float)
            assert 0.0 <= p <= 1.0


# Tests R-010b [integration]: the probe runs in eval mode under no_grad —
# dropout OFF is exactly what determinism across two calls pins.
def test_r010b_stop_prob_probe_deterministic_across_calls(
    probe_model, probe_questions
) -> None:
    pytest.importorskip("transformers")
    first = harness.stop_prob_probe(probe_model, probe_questions)
    second = harness.stop_prob_probe(probe_model, probe_questions)
    assert first == second, (
        "probe must be deterministic (model.eval() + torch.no_grad(); "
        "dropout left on would make repeated probes differ)"
    )


# Tests R-010b [unit]: the hazard_dynamics block carries per-position means
# before/after, the expected-buzz-time delta, and the loss halves from the
# pinned hazard_history.json schema.
def test_r010b_build_hazard_dynamics_block() -> None:
    before = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    after = [[0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]
    history = make_hazard_history([4.0, 4.0, 2.0, 2.0])

    block = harness.build_hazard_dynamics(before, after, history)

    # Ragged questions: position means over the questions reaching them.
    assert block["per_position_mean_before"] == pytest.approx([0.1, 0.1, 0.1, 0.1])
    assert block["per_position_mean_after"] == pytest.approx([0.3, 0.4, 0.5, 0.6])
    assert block["first_half_mean_loss"] == pytest.approx(4.0)
    assert block["second_half_mean_loss"] == pytest.approx(2.0)

    # Internal consistency of the shift summary: delta == after - before,
    # and uniformly higher stop probabilities buzz EARLIER (delta < 0).
    assert block["expected_buzz_time_delta"] == pytest.approx(
        block["expected_buzz_time_after"] - block["expected_buzz_time_before"]
    )
    assert block["expected_buzz_time_delta"] < 0.0

    # No shift => zero delta.
    null_block = harness.build_hazard_dynamics(before, before, history)
    assert null_block["expected_buzz_time_delta"] == pytest.approx(0.0)


# Tests R-010b [unit]: defensive — an empty step history (or empty probe)
# cannot produce loss halves and fails loud.
def test_r010b_build_hazard_dynamics_empty_inputs_raise() -> None:
    before = [[0.1, 0.2]]
    empty_history = {"steps": [], "config": {"beta_terminal": 1.0,
                                              "freeze_answer_head": False,
                                              "ablation": None,
                                              "lr": 1e-3, "epochs": 1}}
    with pytest.raises(ValueError):
        harness.build_hazard_dynamics(before, before, empty_history)
    with pytest.raises(ValueError):
        harness.build_hazard_dynamics([], [], make_hazard_history())


# Tests R-010b [integration]: end-to-end probe over the REAL checkpoint
# save/load path persists hazard_dynamics.json for report assembly.
def test_r010b_probe_and_write_hazard_dynamics_real_checkpoints(
    probe_ckpts, probe_questions, tmp_path: Path
) -> None:
    pytest.importorskip("transformers")
    sup_ckpt, hz_ckpt = probe_ckpts
    history_path = write_json(
        tmp_path / "hazard" / "hazard_history.json", make_hazard_history()
    )
    out_path = tmp_path / "hazard_dynamics.json"

    block = harness.probe_and_write_hazard_dynamics(
        sup_ckpt, hz_ckpt, probe_questions, history_path, out_path
    )

    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    for payload in (block, on_disk):
        assert len(payload["per_position_mean_before"]) == 4  # max T
        assert len(payload["per_position_mean_after"]) == 4
        for key in (
            "expected_buzz_time_before",
            "expected_buzz_time_after",
            "expected_buzz_time_delta",
            "first_half_mean_loss",
            "second_half_mean_loss",
        ):
            assert isinstance(payload[key], float)
    # Identical checkpoints => identical distributions => zero shift.
    assert block["expected_buzz_time_delta"] == pytest.approx(0.0, abs=1e-6)
