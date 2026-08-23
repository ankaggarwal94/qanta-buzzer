"""Regression tests for fair-QA coverage and producer provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _generation(**overrides: object) -> dict[str, object]:
    generation: dict[str, object] = {
        "schema_version": 1,
        "script_path": "scripts/stopdff_fair_qa_retest.py",
        "script_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "git_status_relevant_paths": "",
    }
    generation.update(overrides)
    return generation


def test_fair_qa_provenance_accepts_exact_committed_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa

    monkeypatch.setattr(
        fair_qa,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )

    generation = _generation()
    fair_qa._require_exact_producer_binding(generation)

    assert generation["commit_script_sha256"] == "a" * 64
    assert generation["commit_contains_exact_script"] is True


@pytest.mark.parametrize(
    ("overrides", "committed_sha", "message"),
    [
        ({"git_commit": None}, "a" * 64, "missing producer git commit"),
        ({"git_commit": "g" * 40}, "a" * 64, "missing producer git commit"),
        ({"git_dirty": True}, "a" * 64, "uncommitted producer"),
        ({}, None, "does not contain the producer"),
        ({}, "c" * 64, "does not match the producer"),
        ({"script_sha256": "not-a-sha"}, "not-a-sha", "invalid producer script SHA-256"),
    ],
)
def test_fair_qa_provenance_rejects_unbound_or_mismatched_producer(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    committed_sha: str | None,
    message: str,
) -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa

    monkeypatch.setattr(
        fair_qa,
        "_committed_script_sha256",
        lambda _commit, _path: committed_sha,
    )

    with pytest.raises(RuntimeError, match=message):
        fair_qa._require_exact_producer_binding(_generation(**overrides))


def test_fair_qa_provenance_rejects_redirected_producer_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa

    monkeypatch.setattr(
        fair_qa,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )

    with pytest.raises(RuntimeError, match="canonical producer script path"):
        fair_qa._require_exact_producer_binding(
            _generation(script_path="scripts/compute_stopdff_learned_value.py")
        )


def test_fair_qa_output_provenance_hashes_inputs_and_coverage_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa

    captured: dict[str, object] = {}
    generation = _generation()

    def fake_build_generation(
        script_path,
        argv,
        *,
        output_path,
        extra_paths,
    ):
        captured.update(
            {
                "script_path": script_path,
                "argv": argv,
                "output_path": output_path,
                "extra_paths": extra_paths,
            }
        )
        return generation

    monkeypatch.setattr(fair_qa, "build_generation_provenance", fake_build_generation)
    monkeypatch.setattr(
        fair_qa,
        "_committed_script_sha256",
        lambda _commit, _path: "a" * 64,
    )
    monkeypatch.setattr(
        fair_qa,
        "coverage_helper_sha256s",
        lambda: {
            "scripts/stopdff_dp/diagnostics.py": "d" * 64,
            "scripts/stopdff_dp/dp_solver.py": "e" * 64,
        },
    )
    monkeypatch.setattr(
        fair_qa,
        "coverage_helper_paths",
        lambda: [
            fair_qa._REPO / "scripts" / "stopdff_dp" / "diagnostics.py",
            fair_qa._REPO / "scripts" / "stopdff_dp" / "dp_solver.py",
        ],
    )

    out = tmp_path / "stopdff_fair_qa.json"
    calibration = tmp_path / "calibration.json"
    inputs = [tmp_path / "train.json", tmp_path / "test.json"]
    calibration.write_text('{"calibration": 1}\n', encoding="utf-8")
    for index, path in enumerate(inputs):
        path.write_text(f'{{"input": {index}}}\n', encoding="utf-8")
    actual = fair_qa._build_output_provenance(
        out=out,
        effective_argv=["--out", str(out)],
        calibration_path=calibration,
        data_inputs=inputs,
    )

    assert actual["helper_sha256s"] == {
        "scripts/stopdff_dp/diagnostics.py": "d" * 64,
        "scripts/stopdff_dp/dp_solver.py": "e" * 64,
    }
    extras = {str(path) for path in captured["extra_paths"]}
    assert str(calibration) in extras
    assert {str(path) for path in inputs}.issubset(extras)
    assert any(
        Path(path).as_posix().endswith("scripts/stopdff_dp/diagnostics.py")
        for path in extras
    )
    original_hashes = dict(actual["input_sha256s"])
    assert set(original_hashes) == {
        str(calibration),
        *(str(path) for path in inputs),
    }

    inputs[0].write_text('{"input": "mutated"}\n', encoding="utf-8")
    mutated = fair_qa._build_output_provenance(
        out=out,
        effective_argv=["--out", str(out)],
        calibration_path=calibration,
        data_inputs=inputs,
    )
    assert mutated["input_sha256s"][str(inputs[0])] != original_hashes[str(inputs[0])]
    assert mutated["input_sha256s"][str(inputs[1])] == original_hashes[str(inputs[1])]


def _question(qid: str, gold: str) -> dict[str, object]:
    return {
        "qid": qid,
        "question": f"question {qid}",
        "cumulative_prefixes": [f"prefix {qid}"],
        "options": [gold, "wrong-1", "wrong-2", "wrong-3"],
        "answer_primary": gold,
        "gold_index": 0,
        "category": "History",
    }


def test_krandom_candidate_assignment_is_question_order_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa

    pool = [f"answer-{idx}" for idx in range(30)]
    pool_embs = np.column_stack([np.arange(len(pool)), np.ones(len(pool))])
    gold_to_idx = {answer: idx for idx, answer in enumerate(pool)}
    questions = [_question(f"q{idx}", pool[idx]) for idx in range(8)]

    monkeypatch.setattr(
        fair_qa,
        "_encode",
        lambda _model, strings, batch_size=256: np.zeros((len(strings), 2)),
    )
    monkeypatch.setattr(
        fair_qa,
        "cosine_similarity",
        lambda left, right: np.tile(np.asarray(right)[:, 0], (len(left), 1)),
    )

    def score(items):
        frame = fair_qa.score_arms(
            items,
            "test",
            None,
            pool,
            pool_embs,
            gold_to_idx,
            7,
            {"krandom"},
        )["krandom"]
        return dict(zip(frame["item_id"], frame["top_answer"]))

    assert score(questions) == score(list(reversed(questions)))


class _CyclingEstimator:
    def __init__(self) -> None:
        self._tags = iter(("exact", "pooled", "missing") * 20)
        self._last_tag = "exact"

    def estimate(self, **_kwargs) -> float:
        self._last_tag = next(self._tags)
        return 0.05


def _paired_frame() -> pd.DataFrame:
    rows = []
    for item_id in ("q1", "q2"):
        for fmt, offset in (("MC", 0.0), ("QA", 0.05)):
            for prefix_idx, p in enumerate((0.2, 0.45, 0.75)):
                rows.append(
                    {
                        "item_id": item_id,
                        "format": fmt,
                        "prefix_idx": prefix_idx,
                        "prefix_fraction": (prefix_idx + 1) / 3,
                        "p_calibrated": p + offset,
                        "subject": "sbert:History",
                    }
                )
    return pd.DataFrame(rows)


def test_coverage_tags_are_mixed_reconcilable_and_numerically_inert() -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa
    from scripts.stopdff_dp.rewards import get_schedule

    schedule = get_schedule("power_mark")
    traces = []
    observed, observed_never = fair_qa.signed_per_item(
        _paired_frame(),
        _CyclingEstimator(),
        schedule,
        myopic=False,
        collect_traces=traces,
    )
    repeated, repeated_never = fair_qa.signed_per_item(
        _paired_frame(),
        _CyclingEstimator(),
        schedule,
        myopic=False,
    )
    coverage = fair_qa.summarize_coverage(traces)

    assert observed == repeated
    assert observed_never == repeated_never
    assert coverage["n_cells"] == sum(len(trace.coverage_tags) for trace in traces)
    assert coverage["fraction_exact"] > 0
    assert coverage["fraction_pooled"] > 0
    assert coverage["fraction_missing"] > 0


def test_fair_qa_rejects_missing_gold_and_insufficient_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa

    monkeypatch.setattr(
        fair_qa,
        "_encode",
        lambda _model, strings, batch_size=256: np.zeros((len(strings), 2)),
    )
    pool = ["gold", "other"]
    embeddings = np.ones((len(pool), 2))

    with pytest.raises(ValueError, match="q-missing.*gold"):
        fair_qa.score_arms(
            [_question("q-missing", "absent")],
            "test",
            None,
            pool,
            embeddings,
            {answer: idx for idx, answer in enumerate(pool)},
            1,
            {"krandom"},
        )
    with pytest.raises(ValueError, match="q-small.*candidate pool"):
        fair_qa.score_arms(
            [_question("q-small", "gold")],
            "test",
            None,
            pool,
            embeddings,
            {answer: idx for idx, answer in enumerate(pool)},
            1,
            {"krandom"},
        )


def test_fair_qa_rejects_unpaired_and_single_prefix_evaluations() -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa
    from scripts.stopdff_dp.rewards import get_schedule

    schedule = get_schedule("power_mark")
    one_row = _paired_frame().query(
        "item_id == 'q1' and format == 'MC' and prefix_idx == 0"
    )

    with pytest.raises(ValueError, match="q1.*MC.*at least two prefixes"):
        fair_qa.signed_per_item(
            one_row,
            _CyclingEstimator(),
            schedule,
            myopic=False,
        )


def test_bootstrap_count_must_be_positive() -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa

    with pytest.raises(ValueError, match="num_bootstrap must be positive"):
        fair_qa.bootstrap_ci([1.0], 0, 1)


def test_fair_qa_rejects_fit_eval_qid_overlap() -> None:
    from scripts import stopdff_fair_qa_retest as fair_qa

    with pytest.raises(ValueError, match="fit/evaluation QID overlap.*q2"):
        fair_qa._require_disjoint_split_qids(
            fit_ids=["q1", "q2"],
            eval_ids=["q2", "q3"],
            fit_split="val",
            eval_split="test",
        )
