from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MODAL_RUNNER = REPO / "scripts" / "modal_stopdff_v5_runner.py"


def _load_modal_runner(monkeypatch):
    class DummyImage:
        @classmethod
        def debian_slim(cls, **_kwargs):
            return cls()

        def apt_install(self, *_args):
            return self

        def pip_install(self, *_args):
            return self

        def env(self, *_args):
            return self

        def add_local_dir(self, *_args, **_kwargs):
            return self

    class DummyVolume:
        @classmethod
        def from_name(cls, *_args, **_kwargs):
            return cls()

        def reload(self):
            return None

        def commit(self):
            return None

    class DummyApp:
        def __init__(self, *_args, **_kwargs):
            self.include_source = _kwargs.get("include_source")

        def function(self, **_kwargs):
            def decorate(function):
                function.remote = function
                sequence = {"value": 0}

                def spawn(*args, **kwargs):
                    sequence["value"] += 1
                    result = function(*args, **kwargs)
                    return types.SimpleNamespace(
                        object_id=(
                            f"fc-{function.__name__}-{sequence['value']}"
                        ),
                        get=lambda: result,
                    )

                function.spawn = spawn
                return function

            return decorate

        def local_entrypoint(self):
            return lambda function: function

    monkeypatch.setitem(
        sys.modules,
        "modal",
        types.SimpleNamespace(
            Image=DummyImage,
            Volume=DummyVolume,
            App=DummyApp,
            is_local=lambda: True,
        ),
    )
    from scripts.stopdff_v5.identity import build_manifest, sha256_file
    from scripts.stopdff_v5.manifests import source_manifest_identity

    source_bundle = Path(tempfile.mkdtemp(prefix="stopdff_v5_test_source_"))
    source = source_bundle / "source"
    source.mkdir()
    source_names = (
        "pyproject.toml",
        "scripts/stopdff_v5/checker.py",
        "scripts/stopdff_v5/sweep.py",
        "uv.lock",
    )
    for name in source_names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    files = [
        {
            "path": name,
            "mode": "100644",
            "size": (source / name).stat().st_size,
            "sha256": sha256_file(source / name),
        }
        for name in source_names
    ]
    source_manifest = build_manifest(
        source_manifest_identity(
            git_sha="a" * 40,
            files=files,
            pyproject_sha256=files[0]["sha256"],
            uv_lock_sha256=files[-1]["sha256"],
        )
    )
    (source_bundle / "source_manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    monkeypatch.setenv("STOPDFF_V5_SOURCE_DIR", str(source_bundle))
    name = f"_pr30_round2_modal_runner_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(name, MODAL_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.IMAGE_SOURCE_MANIFEST_ID = "1" * 64
    return module


def _plan() -> dict:
    return {
        "source_id": "1" * 64,
        "raw_id": "2" * 64,
        "adapter_subdirs": ["build_a", "build_b"],
        "gate_overrides": {},
        "resource_summary": {},
    }


def _reachable_completed_control_state(
    runner,
    state_path: Path,
    *,
    plan: dict | None = None,
) -> dict:
    """Write one schema-v4 terminal state through every canonical stage."""
    result = {
        "run_id": "final_modal_aaaaaaaaaaaa",
        "run_spec_id": "b" * 64,
        "adapter_id": "a" * 64,
        "receipt_ids": {
            "determinism": "c" * 64,
            "mutation": "d" * 64,
            "smoke": "e" * 64,
        },
        "validation": {
            "passed": True,
            "errors": [],
            "recomputed": {
                "release_status": "VALID",
                "adapter_bundle_id": "a" * 64,
            },
        },
    }
    state = {
        "schema_version": 4,
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    if plan is not None:
        state["plan"] = plan
        state["plan_digest"] = runner._control_plan_digest(plan)
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )
    for stage in runner._CONTROL_STAGE_ORDER:
        runner._run_control_stage(
            state_path,
            state,
            name=stage,
            invoke=lambda _attempt, stage=stage: {
                "ok": True,
                "stage": stage,
            },
            validate_result=lambda _result: None,
        )
    state["status"] = "completed"
    state["result"] = result
    runner._record_control_event(
        state_path,
        state,
        event="control_completed",
        detail={
            "run_id": result["run_id"],
            "run_spec_id": result["run_spec_id"],
        },
    )
    return state


@pytest.mark.parametrize("result", [{"ok": False}, {"passed": False}])
def test_semantic_failure_is_retryable_not_completed(
    tmp_path, monkeypatch, result
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )

    with pytest.raises(ValueError, match="returned .*false"):
        runner._run_control_stage(
            state_path,
            state,
            name="verify_source",
            invoke=lambda _attempt: result,
            validate_result=lambda _result: None,
        )

    assert "verify_source" not in state["completed"]
    assert state["status"] == "failed"
    accepted = runner._run_control_stage(
        state_path,
        state,
        name="verify_source",
        invoke=lambda _attempt: {"ok": True},
        validate_result=lambda _result: None,
    )
    assert accepted == {"ok": True}
    assert state["stage_attempts"]["verify_source"] == 2


def test_invalid_cached_result_is_discarded_and_retried(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )
    state["status"] = "running"
    state["stage_attempts"]["verify_source"] = 1
    runner._record_control_event(
        state_path,
        state,
        event="stage_started",
        stage="verify_source",
        detail={"attempt": 1},
    )
    state["completed"]["verify_source"] = {"id": "bad"}
    runner._record_control_event(
        state_path,
        state,
        event="stage_completed",
        stage="verify_source",
        detail={"attempt": 1},
    )
    calls = []

    def validate(result):
        if result.get("id") != "good":
            raise ValueError("wrong identity")

    accepted = runner._run_control_stage(
        state_path,
        state,
        name="verify_source",
        invoke=lambda attempt: calls.append(attempt) or {"id": "good"},
        validate_result=validate,
    )

    assert accepted == {"id": "good"}
    assert calls == [2]
    assert state["stage_attempts"]["verify_source"] == 2
    events = [
        json.loads(line)["event"]
        for line in state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-3:] == [
        "stage_checkpoint_invalid",
        "stage_started",
        "stage_completed",
    ]


def test_resume_closes_host_interrupted_stage_before_cache_invalidation(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {
        "schema_version": 4,
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )
    runner._run_control_stage(
        state_path,
        state,
        name="verify_source",
        invoke=lambda _attempt: {"id": "stale"},
        validate_result=lambda _result: None,
    )
    for stage in ("verify_raw", "environment_probe"):
        runner._run_control_stage(
            state_path,
            state,
            name=stage,
            invoke=lambda _attempt, stage=stage: {"stage": stage},
            validate_result=lambda _result: None,
        )
    state["stage_attempts"]["freeze_model"] = 1
    state["status"] = "running"
    runner._record_control_event(
        state_path,
        state,
        event="stage_started",
        stage="freeze_model",
        detail={"attempt": 1},
    )

    assert runner._close_interrupted_control_attempt(state_path, state)

    def require_fresh(result: dict) -> None:
        if result.get("id") != "fresh":
            raise ValueError("stale source checkpoint")

    result = runner._run_control_stage(
        state_path,
        state,
        name="verify_source",
        invoke=lambda _attempt: {"id": "fresh"},
        validate_result=require_fresh,
    )
    assert result == {"id": "fresh"}
    assert state["stage_attempts"] == {
        "verify_source": 2,
        "verify_raw": 1,
        "environment_probe": 1,
        "freeze_model": 1,
    }
    assert "last_error" not in state
    records = [
        json.loads(line)
        for line in state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["event"] for record in records][-6:] == [
        "stage_failed",
        "stage_checkpoint_invalid",
        "stage_checkpoint_invalid",
        "stage_checkpoint_invalid",
        "stage_started",
        "stage_completed",
    ]
    assert records[-6]["stage"] == "freeze_model"
    assert records[-6]["detail"]["type"] == "HostControllerInterrupted"
    assert [record["stage"] for record in records[-5:-2]] == [
        "environment_probe",
        "verify_raw",
        "verify_source",
    ]
    runner._reconcile_control_journal(
        state_path,
        runner._load_control_json(state_path),
    )


def test_stage_postcondition_rejects_fresh_result_before_checkpoint(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )

    def require_expected_identity(result):
        if result.get("id") != "expected":
            raise ValueError("wrong identity")

    with pytest.raises(ValueError, match="wrong identity"):
        runner._run_control_stage(
            state_path,
            state,
            name="verify_source",
            invoke=lambda _attempt: {"ok": True, "id": "wrong"},
            validate_result=require_expected_identity,
        )

    assert "verify_source" not in state["completed"]
    assert state["status"] == "failed"


def test_cached_validation_interrupt_does_not_discard_or_retry(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    cached = {"id": "cached"}
    state = {
        "status": "running",
        "sequence": 0,
        "stage_attempts": {"verify_source": 1},
        "completed": {"verify_source": cached},
    }
    calls = []

    def interrupted(_result):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner._run_control_stage(
            state_path,
            state,
            name="verify_source",
            invoke=lambda attempt: calls.append(attempt),
            validate_result=interrupted,
        )

    assert state["completed"]["verify_source"] is cached
    assert state["stage_attempts"]["verify_source"] == 1
    assert calls == []
    assert not state_path.exists()


def test_state_ahead_of_journal_repairs_exactly_one_event(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {"status": "initialized", "sequence": 0}
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )
    journal = state_path.with_name(state_path.name + ".jsonl")
    journal.write_text("", encoding="utf-8")

    durable = runner._load_control_json(state_path)
    runner._reconcile_control_journal(state_path, durable)

    assert json.loads(journal.read_text(encoding="utf-8")) == durable["last_event"]


def test_failed_journal_replace_leaves_recoverable_state_ahead_window(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {"status": "initialized", "sequence": 0}
    journal = state_path.with_name("control.json.jsonl")
    real_replace = runner.os.replace

    def fail_journal_replace(source, destination):
        if Path(destination) == journal:
            raise OSError("simulated journal replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_journal_replace)
    with pytest.raises(OSError, match="simulated journal replace failure"):
        runner._record_control_event(
            state_path,
            state,
            event="control_initialized",
        )

    assert not journal.exists()
    durable = runner._load_control_json(state_path)
    monkeypatch.setattr(runner.os, "replace", real_replace)
    runner._reconcile_control_journal(state_path, durable)

    assert json.loads(journal.read_text(encoding="utf-8")) == durable["last_event"]


@pytest.mark.parametrize("cut", ["first", "middle", "complete_json"])
def test_torn_final_journal_record_repairs_only_matching_prefix(
    tmp_path, monkeypatch, cut
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {"status": "initialized", "sequence": 0}
    runner._record_control_event(state_path, state, event="control_initialized")
    runner._record_control_event(
        state_path,
        state,
        event="stage_started",
        stage="verify_source",
        detail={"attempt": 1},
    )
    journal = state_path.with_name("control.json.jsonl")
    complete = journal.read_bytes()
    lines = complete.splitlines(keepends=True)
    final_without_newline = lines[-1][:-1]
    lengths = {
        "first": 1,
        "middle": max(1, len(final_without_newline) // 2),
        "complete_json": len(final_without_newline),
    }
    journal.write_bytes(
        b"".join(lines[:-1]) + final_without_newline[:lengths[cut]]
    )

    runner._reconcile_control_journal(
        state_path,
        runner._load_control_json(state_path),
    )

    assert journal.read_bytes() == complete


def test_unprovable_torn_journal_tail_is_rejected_without_rewrite(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {"status": "initialized", "sequence": 0}
    runner._record_control_event(state_path, state, event="control_initialized")
    runner._record_control_event(
        state_path,
        state,
        event="stage_started",
        stage="verify_source",
        detail={"attempt": 1},
    )
    journal = state_path.with_name("control.json.jsonl")
    lines = journal.read_bytes().splitlines(keepends=True)
    torn = b"".join(lines[:-1]) + b"X"
    journal.write_bytes(torn)

    with pytest.raises(ValueError, match="unprovable torn tail"):
        runner._reconcile_control_journal(
            state_path,
            runner._load_control_json(state_path),
        )

    assert journal.read_bytes() == torn


def test_completed_resume_revalidates_or_requires_recovery(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    plan = runner._validate_control_plan(_plan())
    state_path = tmp_path / "control.json"
    _reachable_completed_control_state(
        runner,
        state_path,
        plan=plan,
    )
    calls = []

    revalidated = runner.run_control_plane(
        plan,
        state_path,
        resume=True,
        stage_api={
            "validate": lambda *args: calls.append(args)
            or {
                "passed": True,
                "errors": [],
                "recomputed": {
                    "release_status": "VALID",
                    "adapter_bundle_id": "a" * 64,
                },
            }
        },
    )
    assert revalidated["status"] == "completed"
    assert calls == [
        ("final_modal_aaaaaaaaaaaa", "a" * 64, True, True)
    ]
    assert revalidated["last_event"]["event"] == "control_revalidated"

    recovery = runner.run_control_plane(
        plan,
        state_path,
        resume=True,
        stage_api={"validate": None},
    )
    assert recovery["status"] == "recovery_required"
    assert recovery["last_event"]["event"] == "control_recovery_required"

    restored = runner.run_control_plane(
        plan,
        state_path,
        resume=True,
        stage_api={
            "validate": lambda *_args: {
                "passed": True,
                "errors": [],
                "recomputed": {
                    "release_status": "VALID",
                    "adapter_bundle_id": "a" * 64,
                },
            }
        },
    )
    assert restored["status"] == "completed"
    assert restored["last_event"]["event"] == "control_revalidated"


def test_schema_v4_rejects_mutated_completed_stage_payload(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {
        "schema_version": 4,
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )
    runner._run_control_stage(
        state_path,
        state,
        name="verify_source",
        invoke=lambda _attempt: {"ok": True, "id": "1" * 64},
        validate_result=lambda _result: None,
    )

    durable = runner._load_control_json(state_path)
    durable["completed"]["verify_source"]["id"] = "2" * 64
    state_path.write_text(
        json.dumps(durable, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="completed payload disagrees with journal",
    ):
        runner._reconcile_control_journal(state_path, durable)


def test_schema_v4_rejects_mutated_terminal_result(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    _reachable_completed_control_state(runner, state_path)

    durable = runner._load_control_json(state_path)
    durable["result"]["adapter_id"] = "f" * 64
    state_path.write_text(
        json.dumps(durable, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="control state result disagrees with journal",
    ):
        runner._reconcile_control_journal(state_path, durable)


def test_schema_v4_rejects_out_of_order_stage_start(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {
        "schema_version": 4,
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )
    state["status"] = "running"
    state["stage_attempts"]["verify_raw"] = 1
    runner._record_control_event(
        state_path,
        state,
        event="stage_started",
        stage="verify_raw",
        detail={"attempt": 1},
    )

    with pytest.raises(ValueError, match="order|predecessor"):
        runner._reconcile_control_journal(
            state_path,
            runner._load_control_json(state_path),
        )


def test_schema_v4_rejects_incomplete_terminal_history(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {
        "schema_version": 4,
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )
    runner._run_control_stage(
        state_path,
        state,
        name="verify_source",
        invoke=lambda _attempt: {"ok": True},
        validate_result=lambda _result: None,
    )
    state["status"] = "completed"
    state["result"] = {
        "run_id": "final_modal_aaaaaaaaaaaa",
        "run_spec_id": "b" * 64,
        "adapter_id": "a" * 64,
        "receipt_ids": {},
        "validation": {"passed": True},
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_completed",
        detail={
            "run_id": "final_modal_aaaaaaaaaaaa",
            "run_spec_id": "b" * 64,
        },
    )

    with pytest.raises(ValueError, match="complete|canonical stage"):
        runner._reconcile_control_journal(
            state_path,
            runner._load_control_json(state_path),
        )


def test_schema_v4_rejects_stage_activity_after_terminal_event(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = _reachable_completed_control_state(runner, state_path)
    state["status"] = "running"
    state["stage_attempts"]["verify_source"] = 2
    runner._record_control_event(
        state_path,
        state,
        event="stage_started",
        stage="verify_source",
        detail={"attempt": 2},
    )

    with pytest.raises(ValueError, match="terminal"):
        runner._reconcile_control_journal(
            state_path,
            runner._load_control_json(state_path),
        )


def test_invalid_cached_stage_discards_every_completed_suffix(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {
        "schema_version": 4,
        "status": "initialized",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_initialized",
    )
    for stage in (
        "verify_source",
        "verify_raw",
        "environment_probe",
        "freeze_model",
    ):
        runner._run_control_stage(
            state_path,
            state,
            name=stage,
            invoke=lambda _attempt, stage=stage: {
                "stage": stage,
                "valid": stage != "verify_raw",
            },
            validate_result=lambda _result: None,
        )

    def require_fresh_raw(result: dict) -> None:
        if result.get("valid") is not True:
            raise ValueError("stale raw checkpoint")

    accepted = runner._run_control_stage(
        state_path,
        state,
        name="verify_raw",
        invoke=lambda _attempt: {"stage": "verify_raw", "valid": True},
        validate_result=require_fresh_raw,
    )

    assert accepted == {"stage": "verify_raw", "valid": True}
    assert set(state["completed"]) == {"verify_source", "verify_raw"}
    assert state["stage_attempts"] == {
        "verify_source": 1,
        "verify_raw": 2,
        "environment_probe": 1,
        "freeze_model": 1,
    }
    records = [
        json.loads(line)
        for line in state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    invalidated = [
        record["stage"]
        for record in records
        if record["event"] == "stage_checkpoint_invalid"
    ]
    assert invalidated[-3:] == [
        "freeze_model",
        "environment_probe",
        "verify_raw",
    ]
    runner._reconcile_control_journal(
        state_path,
        runner._load_control_json(state_path),
    )


def test_probe_main_prints_canonical_json(monkeypatch, capsys) -> None:
    runner = _load_modal_runner(monkeypatch)
    runner.probe.remote = lambda: {"z": "é", "a": [1, True]}

    runner.probe_main()

    assert capsys.readouterr().out == '{"a":[1,true],"z":"é"}\n'


def test_content_manifest_rejects_unlisted_bytes(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    from scripts.stopdff_v5.identity import build_manifest, sha256_file
    from scripts.stopdff_v5.manifests import source_manifest_identity

    content = tmp_path / "source"
    content.mkdir()
    declared = content / "declared.py"
    declared.write_text("declared\n", encoding="utf-8")
    (content / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (content / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    entries = [
        {
            "path": path.name,
            "mode": "100644",
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(content.iterdir())
    ]
    manifest = build_manifest(
        source_manifest_identity(
            git_sha="a" * 40,
            files=entries,
            pyproject_sha256=sha256_file(content / "pyproject.toml"),
            uv_lock_sha256=sha256_file(content / "uv.lock"),
        )
    )
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (content / "unlisted.py").write_text("unlisted\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory mismatch"):
        runner._verified_content_manifest(
            tmp_path,
            manifest_name="source_manifest.json",
            expected_id=manifest["id"],
            file_key="files",
            name_key="path",
            content_subdir="source",
            expected_kind="source_snapshot",
        )


@pytest.mark.parametrize("extra_name", ["sitecustomize.py", "numpy.py"])
def test_modal_image_preflight_rejects_unlisted_executable_source(
    tmp_path,
    monkeypatch,
    extra_name,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    bundle = tmp_path / "source_bundle"
    shutil.copytree(runner._IMAGE_SOURCE_DIR, bundle / "source")
    shutil.copy2(
        runner._IMAGE_SOURCE_DIR.parent / "source_manifest.json",
        bundle / "source_manifest.json",
    )
    (bundle / "source" / extra_name).write_text(
        "raise RuntimeError('unlisted')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inventory mismatch"):
        runner._materialize_image_source(bundle)


def test_control_plan_source_mismatch_makes_no_remote_calls(
    tmp_path,
    monkeypatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    runner.IMAGE_SOURCE_MANIFEST_ID = "2" * 64
    state_path = tmp_path / "control.json"
    with pytest.raises(ValueError, match="validated Modal image source"):
        runner.run_control_plane(
            _plan(),
            state_path,
            resume=False,
            stage_api={},
        )
    assert not state_path.exists()


def test_direct_modal_stages_reject_a_source_other_than_the_image(
    monkeypatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    assert runner.app.include_source is False
    wrong_source = "2" * 64

    with pytest.raises(ValueError, match="validated Modal image source"):
        runner.build_adapter("candidate", wrong_source, "3" * 64, "4" * 64)

    with pytest.raises(ValueError, match="validated Modal image source"):
        runner.adapter_determinism_receipt(
            "candidate_a",
            "candidate_b",
            wrong_source,
            "3" * 64,
            "4" * 64,
        )

    with pytest.raises(ValueError, match="validated Modal image source"):
        runner.mutation_gate(
            json.dumps({"source_manifest_id": wrong_source})
        )

    wrapper = {
        "run_spec_identity": {
            "identity": {"source_manifest_id": wrong_source},
        }
    }
    with pytest.raises(ValueError, match="validated Modal image source"):
        runner.run_sweep(json.dumps(wrapper), "5" * 64, "6" * 64, False)


@pytest.mark.parametrize(
    "subdirs",
    [
        ["build_a", "./build_a"],
        ["build_a", "build_a/"],
        ["build_a", "build_a"],
    ],
)
def test_adapter_subdirs_must_be_canonical_and_distinct(
    monkeypatch,
    subdirs,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    plan = _plan()
    plan["adapter_subdirs"] = subdirs
    with pytest.raises(ValueError, match="noncanonical|distinct"):
        runner._validate_control_plan(plan)


def test_public_schema_and_docs_match_round2_contracts() -> None:
    schema = json.loads(
        (REPO / "schemas" / "stopdff_run_spec.schema.json").read_text(
            encoding="utf-8"
        )
    )
    branches = schema["$defs"]["identity"]["allOf"]
    counts = {
        branch["if"]["properties"]["profile_variant"]["const"]:
        branch["then"]["properties"]["bootstrap"]["properties"]
        ["replicate_count"]["const"]
        for branch in branches
    }
    assert counts == {"smoke": 100, "final": 1000}

    contract = (REPO / "SCIENTIFIC_CONTRACT.md").read_text(encoding="utf-8")
    reproduction = (REPO / "docs" / "stopdff_v5" / "REPRODUCTION.md").read_text(
        encoding="utf-8"
    )
    runner_source = MODAL_RUNNER.read_text(encoding="utf-8")
    manifests_source = (
        REPO / "scripts" / "stopdff_v5" / "manifests.py"
    ).read_text(encoding="utf-8")
    assert "schemas/stopdff_scientific_profile.schema.json" in contract
    assert "checker_calibration.py" in reproduction
    assert "checker_package.py" in reproduction
    assert "qualitative verdict is stable" not in reproduction
    assert "Environments are pinned" not in reproduction
    assert "ENVIRONMENT_PACKAGES" in runner_source
    assert '"matplotlib",' in manifests_source
    assert "FVI_PRODUCER_FILES" in runner_source
    assert '"fvi_study.py",' in manifests_source

    # AST-anchored structural pins (not substring matches, which pass on any
    # unrelated occurrence anywhere in the file). The run_control_plane
    # driver lives in scripts/stopdff_v5_control_plane.py (the runner
    # re-exports it as a facade), so its pin parses that module; run_sweep
    # remains a runner stage function.
    runner_module = ast.parse(runner_source)
    control_plane_module = ast.parse(
        MODAL_RUNNER.with_name("stopdff_v5_control_plane.py").read_text(
            encoding="utf-8"
        )
    )

    def _module_function(module: ast.Module, name: str) -> ast.FunctionDef:
        return next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def _keyword_value(call: ast.Call, name: str) -> ast.expr:
        return next(kw.value for kw in call.keywords if kw.arg == name)

    # Both control-plane run specs (smoke and final) bind the runtime-verified
    # producer hashes into run-spec identity.
    spec_calls = [
        node
        for node in ast.walk(
            _module_function(control_plane_module, "run_control_plane")
        )
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_spec_identity"
    ]
    assert len(spec_calls) == 2
    for call in spec_calls:
        value = _keyword_value(call, "producer_hashes")
        assert isinstance(value, ast.Name) and value.id == "producer_hashes"

    # The remote sweep body forwards the control-validated gate overrides from
    # the run-spec binding into its sweep context.
    sweep_context_call = next(
        node
        for node in ast.walk(_module_function(runner_module, "run_sweep"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "SweepContext"
    )
    overrides = _keyword_value(sweep_context_call, "gate_overrides")
    assert (
        isinstance(overrides, ast.Subscript)
        and isinstance(overrides.value, ast.Name)
        and overrides.value.id == "binding"
        and isinstance(overrides.slice, ast.Constant)
        and overrides.slice.value == "gate_overrides"
    )
