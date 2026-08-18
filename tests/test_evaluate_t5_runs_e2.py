"""RED tests for R-002 (enabler E-2): ``evaluate_t5_policy`` ``return_runs``.

Covers the hazard-efficacy-eval spec:

- R-002 [integration]: ``evaluate_t5_policy`` gains keyword-only
  ``return_runs: bool = False``.  When True the returned dict additionally
  carries ``"runs"``: one record per evaluated question with at least
  ``{qid, sq, buzz_position (int|None), buzzed (bool — policy buzz),
  correct (bool), forced_correct (bool), confidence (float|None),
  episode_reward (float), n_steps (int)}``.  Default False leaves the existing
  payload unchanged.  Aggregates must be consistent with the records.

Test design (per the spec's Entry/Through/Exit):

- Entry: ``evaluate_t5_policy`` on a tiny t5-small CPU fixture checkpoint plus
  a 3-question fixture (module-scoped checkpoint fixture mirrors
  ``tests/test_hazard_pretrain.py::supervised_ckpt``).
- Through: the REAL ``TossupMCEnv`` + ``TextObservationWrapper`` episode loop;
  no metric function is mocked.
- Exit: ``runs`` present with one record per question, aggregate-consistent;
  absent when ``return_runs`` is omitted; positional passing rejected.

# DECISION (pinned semantics the GREEN agent must honor):
# - runs preserve the input question order (the eval loop iterates
#   ``test_questions`` in order).
# - ``buzz_position`` is non-None IFF ``buzzed`` is True (policy buzz).  This
#   matches the existing aggregate, which appends to ``buzz_positions`` only on
#   ``terminated`` episodes (a WAIT at the final prefix truncates with a forced
#   answer and today contributes nothing to ``avg_buzz_pos``).
# - aggregate consistency extends beyond the two spec examples to
#   ``accuracy == mean(correct)`` and ``forced_correct_rate ==
#   mean(forced_correct)`` (same "Aggregates must be consistent" clause).
# - ``correct`` and ``forced_correct`` are mutually exclusive, and an episode
#   that never policy-buzzed cannot be ``correct``.

Model-touching tests use t5-small on CPU and skip when transformers is absent
(guard idiom copied from tests/test_hazard_pretrain.py).

RED expectation: every test fails at runtime today via ``KeyError`` (signature
introspection) or ``TypeError`` (unexpected keyword ``return_runs``) — never at
import/collection time.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from qb_data.mc_builder import MCQuestion
from scripts.compare_policies import evaluate_t5_policy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mc_question(
    qid: str,
    gold_index: int,
    prefixes: list[str],
) -> MCQuestion:
    """Build an MCQuestion with real option_profiles (TF-IDF corpus source).

    ``evaluate_t5_policy`` builds a TfIdfLikelihood from the REFERENCE
    questions' ``option_profiles``, so those strings must be non-empty text.
    """
    options = [
        "George Washington",
        "Thomas Jefferson",
        "John Adams",
        "Benjamin Franklin",
    ]
    return MCQuestion(
        qid=qid,
        question=prefixes[-1],
        tokens=prefixes[-1].split(),
        answer_primary=options[gold_index],
        clean_answers=[options[gold_index]],
        run_indices=list(range(len(prefixes))),
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=list(prefixes),
        options=options,
        gold_index=gold_index,
        option_profiles=[
            "George Washington first president commander revolutionary war",
            "Thomas Jefferson third president declaration independence Virginia",
            "John Adams second president Massachusetts diplomat lawyer",
            "Benjamin Franklin inventor diplomat Philadelphia printing press",
        ],
        option_answer_primary=list(options),
        distractor_strategy="test",
    )


@pytest.fixture(scope="module")
def policy_ckpt(tmp_path_factory) -> str:
    """Save a t5-small policy checkpoint for the evaluation entry point."""
    pytest.importorskip("transformers")
    from models.t5_policy import T5PolicyModel

    model = T5PolicyModel(
        {
            "model_name": "t5-small",
            "device": "cpu",
            "max_input_length": 64,
            "num_choices": 4,
        }
    )
    ckpt_dir = tmp_path_factory.mktemp("t5_eval_ckpt")
    model.save(str(ckpt_dir))
    return str(ckpt_dir)


@pytest.fixture(scope="module")
def test_questions() -> list[MCQuestion]:
    """Three evaluation questions with >=2 cumulative prefixes each."""
    return [
        _make_mc_question(
            "eval_q1",
            gold_index=0,
            prefixes=[
                "This man",
                "This man led the continental army",
                "This man led the continental army and became first president",
            ],
        ),
        _make_mc_question(
            "eval_q2",
            gold_index=1,
            prefixes=[
                "This author",
                "This author wrote the declaration of independence",
                "This author wrote the declaration and served as third president",
            ],
        ),
        _make_mc_question(
            "eval_q3",
            gold_index=2,
            prefixes=[
                "This diplomat",
                "This diplomat from Massachusetts became second president",
            ],
        ),
    ]


@pytest.fixture(scope="module")
def reference_questions() -> list[MCQuestion]:
    """Distinct reference questions supplying the TF-IDF reward corpus."""
    return [
        _make_mc_question(
            "ref_q1",
            gold_index=3,
            prefixes=["This inventor", "This inventor flew a kite in a storm"],
        ),
        _make_mc_question(
            "ref_q2",
            gold_index=0,
            prefixes=["This general", "This general crossed the Delaware river"],
        ),
    ]


def _evaluate(ckpt: str, test_qs: list, ref_qs: list, **kwargs) -> dict:
    return evaluate_t5_policy(ckpt, test_qs, ref_qs, "fixture", {}, **kwargs)


# ---------------------------------------------------------------------------
# R-002: signature contract — keyword-only, default False
# ---------------------------------------------------------------------------


def test_r002_return_runs_is_keyword_only_defaulting_false() -> None:
    """Tests R-002 [unit-of-contract]: signature pins the enabler's API shape.

    ``return_runs`` must exist, be KEYWORD_ONLY, and default to False —
    the spec text is explicit ("gains keyword-only ``return_runs: bool =
    False``"), so the signature IS the contract.  Positional passing must
    raise TypeError (binding fails before the body runs).
    """
    params = inspect.signature(evaluate_t5_policy).parameters
    assert "return_runs" in params, "evaluate_t5_policy must accept return_runs"
    param = params["return_runs"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False

    with pytest.raises(TypeError):
        # 7th positional slot — must never bind to return_runs.
        evaluate_t5_policy(object(), [], [], "fixture", {}, None, True)


# ---------------------------------------------------------------------------
# R-002: runs records — one per question, typed, aggregate-consistent
# ---------------------------------------------------------------------------


def test_r002_return_runs_records_one_per_question_and_consistent(
    policy_ckpt: str,
    test_questions: list[MCQuestion],
    reference_questions: list[MCQuestion],
) -> None:
    """Tests R-002 [integration]: per-question records through the real env loop."""
    pytest.importorskip("transformers")
    result = _evaluate(
        policy_ckpt, test_questions, reference_questions, return_runs=True
    )

    assert "runs" in result
    runs = result["runs"]
    assert isinstance(runs, list)
    assert len(runs) == len(test_questions)
    assert result["n_questions"] == len(runs)
    assert result["n_questions_evaluated"] == len(runs)

    # Input order preserved, qids intact.
    assert [r["qid"] for r in runs] == [q.qid for q in test_questions]

    required = {
        "qid",
        "sq",
        "buzz_position",
        "buzzed",
        "correct",
        "forced_correct",
        "confidence",
        "episode_reward",
        "n_steps",
    }
    by_qid = {q.qid: q for q in test_questions}
    for rec in runs:
        missing = required - set(rec)
        assert not missing, f"run record for {rec.get('qid')} missing {missing}"

        horizon = len(by_qid[rec["qid"]].cumulative_prefixes)

        assert isinstance(rec["sq"], float) and math.isfinite(rec["sq"])
        assert isinstance(rec["buzzed"], bool)
        assert isinstance(rec["correct"], bool)
        assert isinstance(rec["forced_correct"], bool)
        assert isinstance(rec["episode_reward"], float)
        assert math.isfinite(rec["episode_reward"])
        n_steps = rec["n_steps"]
        assert isinstance(n_steps, int) and not isinstance(n_steps, bool)
        assert 1 <= n_steps <= horizon

        bp = rec["buzz_position"]
        if bp is not None:
            assert isinstance(bp, int) and not isinstance(bp, bool)
            assert 0 <= bp < horizon
        # Policy-buzz linkage: a position exists iff the policy buzzed.
        assert (bp is not None) == rec["buzzed"]

        conf = rec["confidence"]
        if conf is not None:
            assert isinstance(conf, float)
            assert 0.0 <= conf <= 1.0
        if rec["buzzed"]:
            assert conf is not None, "a policy buzz must record its confidence"

        # An episode either policy-buzzed or was forced at the horizon.
        assert not (rec["correct"] and rec["forced_correct"])
        if not rec["buzzed"]:
            assert rec["correct"] is False, (
                "correct means POLICY-buzz correct; a never-buzzed episode "
                "cannot be correct"
            )

    # Aggregate consistency (spec: "Aggregates must be consistent").
    assert result["mean_sq"] == pytest.approx(
        float(np.mean([r["sq"] for r in runs]))
    )
    assert result["accuracy"] == pytest.approx(
        float(np.mean([1.0 if r["correct"] else 0.0 for r in runs]))
    )
    assert result["forced_correct_rate"] == pytest.approx(
        float(np.mean([1.0 if r["forced_correct"] else 0.0 for r in runs]))
    )
    positions = [r["buzz_position"] for r in runs if r["buzz_position"] is not None]
    if positions:
        assert result["avg_buzz_pos"] == pytest.approx(float(np.mean(positions)))
    else:
        assert result["avg_buzz_pos"] == 0.0


# ---------------------------------------------------------------------------
# R-002: backward compatibility — default payload unchanged, no "runs" key
# ---------------------------------------------------------------------------


def test_r002_default_omits_runs_and_payload_unchanged(
    policy_ckpt: str,
    test_questions: list[MCQuestion],
    reference_questions: list[MCQuestion],
) -> None:
    """Tests R-002 [integration]: omitting/False leaves the payload unchanged.

    Three calls against the identical checkpoint and questions: omitted flag,
    ``return_runs=False``, and ``return_runs=True``.  The first two must be
    key-identical with no ``"runs"``; the third must differ ONLY by adding
    ``"runs"``.  Scalar metrics agree across all three calls (deterministic
    eval: ``model.eval()`` + ``deterministic=True`` actions; loose tolerance
    only to stay robust to device auto-selection).
    """
    pytest.importorskip("transformers")
    res_plain = _evaluate(policy_ckpt, test_questions, reference_questions)
    res_false = _evaluate(
        policy_ckpt, test_questions, reference_questions, return_runs=False
    )
    res_true = _evaluate(
        policy_ckpt, test_questions, reference_questions, return_runs=True
    )

    assert "runs" not in res_plain
    assert "runs" not in res_false
    assert "runs" in res_true

    assert set(res_false) == set(res_plain), (
        "return_runs=False must not alter the payload key set"
    )
    assert set(res_true) - {"runs"} == set(res_plain), (
        "return_runs=True must only ADD the runs key"
    )

    for metric in (
        "accuracy",
        "mean_sq",
        "ece",
        "brier",
        "avg_buzz_pos",
        "mean_reward",
        "forced_correct_rate",
    ):
        assert res_false[metric] == pytest.approx(
            res_plain[metric], rel=1e-3, abs=1e-6
        ), f"{metric} changed when passing return_runs=False"
        assert res_true[metric] == pytest.approx(
            res_plain[metric], rel=1e-3, abs=1e-6
        ), f"{metric} changed when passing return_runs=True"
