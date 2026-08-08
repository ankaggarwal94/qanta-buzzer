"""Durable Modal-controller recovery regressions introduced in round 6."""
from __future__ import annotations

import pytest

from tests.test_pr30_control_repairs import _fake_control_api, _load_modal_runner


def _plan(ids: dict[str, str]) -> dict:
    return {
        "source_id": ids["source"],
        "raw_id": ids["raw"],
        "adapter_subdirs": ["build_a", "build_b"],
        "gate_overrides": {},
        "resource_summary": {"backend": "modal"},
    }


def test_lost_adapter_response_rebinds_to_attempt_scoped_destinations(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api()
    adapter_call = api["adapter_determinism_receipt"]
    attempts: list[tuple] = []

    def lose_first_response(*args):
        attempts.append(args)
        result = adapter_call(*args)
        if len(attempts) == 1:
            raise RuntimeError("lost adapter response")
        return result

    api["adapter_determinism_receipt"] = lose_first_response
    state_path = tmp_path / "control.json"
    with pytest.raises(RuntimeError, match="lost adapter response"):
        runner.run_control_plane(
            _plan(ids),
            state_path,
            resume=False,
            stage_api=api,
        )

    state = runner.run_control_plane(
        _plan(ids),
        state_path,
        resume=True,
        stage_api=api,
    )

    assert state["status"] == "completed"
    assert attempts[0][:2] == ("build_a", "build_b")
    assert attempts[1][:2] == (
        "build_a__attempt_2",
        "build_b__attempt_2",
    )
    promotion = next(
        args
        for name, args in calls
        if name == "promote_adapter"
    )
    assert promotion[0] == "build_a__attempt_2"


def test_control_plan_reserves_attempt_scoped_adapter_namespace(
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    _api, _calls, ids = _fake_control_api()
    plan = _plan(ids)
    plan["adapter_subdirs"] = ["build_a", "build_a__attempt_2"]
    with pytest.raises(ValueError, match="reserved retry namespace"):
        runner._validate_control_plan(plan)


def test_nonterminal_resume_revalidates_checkpointed_package(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api()
    state_path = tmp_path / "control.json"
    original_record = runner._record_control_event
    interrupted = False

    def interrupt_after_validation(*args, **kwargs):
        nonlocal interrupted
        original_record(*args, **kwargs)
        if (
            not interrupted
            and kwargs.get("event") == "stage_completed"
            and kwargs.get("stage") == "validate_package"
        ):
            interrupted = True
            raise RuntimeError("crash before control completion")

    monkeypatch.setattr(runner, "_record_control_event", interrupt_after_validation)
    with pytest.raises(RuntimeError, match="crash before control completion"):
        runner.run_control_plane(
            _plan(ids),
            state_path,
            resume=False,
            stage_api=api,
        )

    validate_calls_before = sum(name == "validate" for name, _ in calls)
    state = runner.run_control_plane(
        _plan(ids),
        state_path,
        resume=True,
        stage_api=api,
    )

    assert state["status"] == "completed"
    assert sum(name == "validate" for name, _ in calls) == (
        validate_calls_before + 1
    )
    assert state["stage_attempts"]["validate_package"] == 2
