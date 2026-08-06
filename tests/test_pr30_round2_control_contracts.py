from __future__ import annotations

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
            pass

        def function(self, **_kwargs):
            def decorate(function):
                function.remote = function
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

    with pytest.raises(ValueError, match="returned .*false"):
        runner._run_control_stage(
            state_path,
            state,
            name="verify",
            invoke=lambda _attempt: result,
            validate_result=lambda _result: None,
        )

    assert "verify" not in state["completed"]
    assert state["status"] == "failed"
    accepted = runner._run_control_stage(
        state_path,
        state,
        name="verify",
        invoke=lambda _attempt: {"ok": True},
        validate_result=lambda _result: None,
    )
    assert accepted == {"ok": True}
    assert state["stage_attempts"]["verify"] == 2


def test_invalid_cached_result_is_discarded_and_retried(
    tmp_path, monkeypatch
) -> None:
    runner = _load_modal_runner(monkeypatch)
    state_path = tmp_path / "control.json"
    state = {
        "status": "running",
        "sequence": 0,
        "stage_attempts": {"verify": 1},
        "completed": {"verify": {"id": "bad"}},
    }
    calls = []

    def validate(result):
        if result.get("id") != "good":
            raise ValueError("wrong identity")

    accepted = runner._run_control_stage(
        state_path,
        state,
        name="verify",
        invoke=lambda attempt: calls.append(attempt) or {"id": "good"},
        validate_result=validate,
    )

    assert accepted == {"id": "good"}
    assert calls == [2]
    assert state["stage_attempts"]["verify"] == 2
    events = [
        json.loads(line)["event"]
        for line in state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events == [
        "stage_checkpoint_invalid",
        "stage_started",
        "stage_completed",
    ]


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

    def require_expected_identity(result):
        if result.get("id") != "expected":
            raise ValueError("wrong identity")

    with pytest.raises(ValueError, match="wrong identity"):
        runner._run_control_stage(
            state_path,
            state,
            name="verify",
            invoke=lambda _attempt: {"ok": True, "id": "wrong"},
            validate_result=require_expected_identity,
        )

    assert "verify" not in state["completed"]
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
        "stage_attempts": {"verify": 1},
        "completed": {"verify": cached},
    }
    calls = []

    def interrupted(_result):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner._run_control_stage(
            state_path,
            state,
            name="verify",
            invoke=lambda attempt: calls.append(attempt),
            validate_result=interrupted,
        )

    assert state["completed"]["verify"] is cached
    assert state["stage_attempts"]["verify"] == 1
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
    runner._record_control_event(state_path, state, event="stage_started")
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
    runner._record_control_event(state_path, state, event="stage_started")
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
    state = {
        "schema_version": 1,
        "plan": plan,
        "plan_digest": runner._control_plan_digest(plan),
        "status": "completed",
        "sequence": 0,
        "stage_attempts": {},
        "completed": {},
        "result": {
            "run_id": "final_modal_aaaaaaaaaaaa",
            "adapter_id": "a" * 64,
        },
    }
    runner._record_control_event(
        state_path,
        state,
        event="control_completed",
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
    wrong_source = "2" * 64

    with pytest.raises(ValueError, match="validated Modal image source"):
        runner.build_adapter("candidate", wrong_source, "3" * 64, "4" * 64)

    wrapper = {
        "run_spec_identity": {
            "identity": {"source_manifest_id": wrong_source},
        }
    }
    with pytest.raises(ValueError, match="validated Modal image source"):
        runner.run_sweep(json.dumps(wrapper), "5" * 64, "6" * 64, False)


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
    assert "producer_hashes=producer_hashes" in runner_source
    assert 'gate_overrides=binding["gate_overrides"]' in runner_source
