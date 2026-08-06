from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

from scripts import run_stopdff_v5_local as local_runner
from scripts.stopdff_v5.bootstrap import build_bootstrap_plan
from scripts.stopdff_v5.identity import build_manifest, sha256_file
from scripts.stopdff_v5.manifests import ENVIRONMENT_PACKAGES, RAW_INPUT_ROLES


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

    fake_modal = types.SimpleNamespace(
        Image=DummyImage,
        Volume=DummyVolume,
        App=DummyApp,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
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
    name = f"_pr30_modal_runner_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(name, MODAL_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.IMAGE_SOURCE_MANIFEST_ID = "1" * 64
    return module


def _write_raw_manifest(base: Path, *, passed: bool, kind: str = "raw_input_bundle"):
    files = []
    for role in sorted(RAW_INPUT_ROLES):
        content = (json.dumps({"role": role}, sort_keys=True) + "\n").encode()
        path = base / role
        path.write_bytes(content)
        files.append(
            {
                "role": role,
                "size": len(content),
                "sha256": sha256_file(path),
            }
        )
    identity = {
        "kind": kind,
        "files": files,
        "semantic_checks": {
            "all_semantic_checks_pass": passed,
            "question_trajectory_binding_id": "c" * 64,
        },
    }
    manifest = build_manifest(identity)
    (base / "raw_input_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest


def test_raw_manifest_requires_kind_and_passing_semantics(tmp_path, monkeypatch):
    runner = _load_modal_runner(monkeypatch)
    manifest = _write_raw_manifest(tmp_path, passed=True)
    assert runner._verified_raw_input_manifest(
        tmp_path,
        expected_id=manifest["id"],
    ) == manifest

    failed = _write_raw_manifest(tmp_path, passed=False)
    with pytest.raises(ValueError, match="passing semantic checks"):
        runner._verified_raw_input_manifest(
            tmp_path,
            expected_id=failed["id"],
        )

    wrong_kind = _write_raw_manifest(tmp_path, passed=True, kind="source_snapshot")
    with pytest.raises(ValueError, match="kind"):
        runner._verified_raw_input_manifest(
            tmp_path,
            expected_id=wrong_kind["id"],
        )


def test_cached_adapter_and_fvi_require_complete_bound_evidence(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_id = "a" * 64
    adapter_manifest = {
        "id": adapter_id,
        "identity": {
            "source_manifest_id": "1" * 64,
            "raw_input_bundle_id": "2" * 64,
            "model_snapshot_id": "3" * 64,
            "fit_rows_sha256": "4" * 64,
            "eval_rows_sha256": "5" * 64,
        },
    }
    (adapter / "manifest.json").write_text(json.dumps(adapter_manifest))

    class AdapterChecker:
        @staticmethod
        def validate_adapter(_path):
            return types.SimpleNamespace(
                passed=True,
                errors=[],
                recomputed={"adapter_bundle_id": adapter_id},
            )

        @staticmethod
        def load_json(path):
            return json.loads(path.read_text())

    cached = runner._validated_cached_adapter(
        adapter,
        subdir="retry",
        source_id="1" * 64,
        raw_id="2" * 64,
        model_id="3" * 64,
        checker_module=AdapterChecker,
    )
    assert cached["cached"] is True
    with pytest.raises(FileExistsError, match="incompatible"):
        runner._validated_cached_adapter(
            adapter,
            subdir="retry",
            source_id="1" * 64,
            raw_id="9" * 64,
            model_id="3" * 64,
            checker_module=AdapterChecker,
        )

    adapter_manifest["identity"]["mc_retention_evidence"] = {
        "splits": {
            "fit": {"overridden": True},
            "eval": {"overridden": True},
        }
    }
    (adapter / "manifest.json").write_text(json.dumps(adapter_manifest))
    with pytest.raises(FileExistsError, match="low-retention gate override"):
        runner._validated_cached_adapter(
            adapter,
            subdir="retry",
            source_id="1" * 64,
            raw_id="2" * 64,
            model_id="3" * 64,
            checker_module=AdapterChecker,
        )
    assert runner._validated_cached_adapter(
        adapter,
        subdir="retry",
        source_id="1" * 64,
        raw_id="2" * 64,
        model_id="3" * 64,
        checker_module=AdapterChecker,
        allow_low_mc_retention=True,
    )["cached"] is True

    fvi = tmp_path / "fvi"
    fvi.mkdir()
    execution = {"selected_parameters": {"tolerance": "1e-6", "max_iterations": 50}}
    fvi_manifest = {"id": "6" * 64, "identity": {"kind": "fvi_study"}}
    (fvi / "fvi_study.json").write_text(json.dumps(fvi_manifest))
    (fvi / "fvi_study_execution.json").write_text(json.dumps(execution))
    result = runner._validated_cached_fvi(
        fvi,
        manifest=fvi_manifest,
        identity=fvi_manifest["identity"],
        execution=execution,
        checker_module=AdapterChecker,
    )
    assert result["cached"] is True
    (fvi / "fvi_study_execution.json").unlink()
    with pytest.raises(FileExistsError, match="incomplete"):
        runner._validated_cached_fvi(
            fvi,
            manifest=fvi_manifest,
            identity=fvi_manifest["identity"],
            execution=execution,
            checker_module=AdapterChecker,
        )


def _fake_control_api(*, fail_first_smoke: bool = False):
    calls = []
    ids = {key: value * 64 for key, value in {
        "source": "1",
        "raw": "2",
        "model": "3",
        "adapter": "4",
        "determinism": "5",
        "fvi": "6",
        "smoke_plan": "7",
        "final_plan": "8",
        "smoke_receipt": "9",
        "mutation": "a",
        "myopic": "b",
    }.items()}
    fail = {"smoke": fail_first_smoke}

    def record(name, result):
        def call(*args):
            calls.append((name, args))
            return result(*args) if callable(result) else dict(result)

        return call

    def verify(_rel, kind):
        result = {
            "ok": True,
            "id": ids[kind],
            "mismatches": [],
            "n_files": 1,
        }
        if kind == "raw":
            result["myopic_artifact_sha256"] = ids["myopic"]
        return result

    def bootstrap(_adapter, replicates):
        return {
            "bootstrap_plan_id": (
                ids["smoke_plan"] if replicates == 100 else ids["final_plan"]
            ),
            "replicates": replicates,
            "n_items": 10,
            "cached": False,
        }

    def sweep(spec_json, _adapter, _bootstrap, resume):
        wrapper = json.loads(spec_json)
        variant = wrapper["run_spec_identity"]["profile_variant"]
        if variant == "smoke" and fail["smoke"]:
            fail["smoke"] = False
            raise RuntimeError("lost smoke response")
        result = {
            "run_id": wrapper["run_id"],
            "requested": 3,
            "completed": 3,
            "skipped": 0,
            "failed": 0,
            "release_status": "VALID",
            "family": {"verdict": "PASS"},
            "resume": resume,
        }
        if variant == "smoke":
            result["prerequisite_receipt_id"] = ids["smoke_receipt"]
        return result

    api = {
        "probe": record("probe", {
            "python": "3.11.0",
            "package_versions": {
                name: "1.0" for name in ENVIRONMENT_PACKAGES
            },
        }),
        "verify_volume_artifact": record("verify", verify),
        "freeze_model": record("freeze_model", {
            "model_id": ids["model"],
            "cached": False,
        }),
        "build_adapter": record(
            "build_adapter",
            lambda subdir, *_args: {
                "adapter_id": ids["adapter"],
                "fit_rows_sha256": "c" * 64,
                "eval_rows_sha256": "d" * 64,
                "subdir": subdir,
                "cached": False,
            },
        ),
        "adapter_determinism_receipt": record(
            "adapter_determinism",
            {
                "ok": True,
                "adapter_id": ids["adapter"],
                "prerequisite_receipt_id": ids["determinism"],
            },
        ),
        "promote_adapter": record(
            "promote_adapter",
            lambda _subdir, adapter_id: {
                "canonical_subdir": f"canonical_{adapter_id}",
                "cached": False,
            },
        ),
        "fvi_study": record("fvi_study", {
            "fvi_study_id": ids["fvi"],
            "selected": {"tolerance": "1e-6", "max_iterations": 50},
            "cached": False,
        }),
        "bootstrap_plan": record("bootstrap", bootstrap),
        "run_sweep": record("run_sweep", sweep),
        "mutation_gate": record("mutation_gate", {
            "ok": True,
            "n": 40,
            "unexpected": [],
            "prerequisite_receipt_id": ids["mutation"],
        }),
        "validate": record(
            "validate",
            lambda _run_id, adapter_id, *_args: {
                "passed": True,
                "errors": [],
                "recomputed": {
                    "release_status": "VALID",
                    "adapter_bundle_id": adapter_id,
                },
            },
        ),
        "package": record(
            "package",
            lambda run_id: {"run_id": run_id, "packaged": True},
        ),
    }
    return api, calls, ids


def test_control_plane_journals_order_and_resumes_lost_sweep(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api(fail_first_smoke=True)
    plan = {
        "source_id": ids["source"],
        "raw_id": ids["raw"],
        "adapter_subdirs": ["build_a", "build_b"],
        "gate_overrides": {"allow_low_mc_retention": True},
        "resource_summary": {"backend": "modal"},
    }
    state_path = tmp_path / "control.json"
    with pytest.raises(RuntimeError, match="lost smoke response"):
        runner.run_control_plane(
            plan,
            state_path,
            resume=False,
            stage_api=api,
        )
    adapter_calls = [call for call in calls if call[0] == "build_adapter"]
    assert len(adapter_calls) == 2
    assert all(call[1][-1] is True for call in adapter_calls)
    smoke_call = next(call for call in calls if call[0] == "run_sweep")
    smoke_spec = json.loads(smoke_call[1][0])["run_spec_identity"]
    assert smoke_spec["gate"]["allow_low_mc_retention"] is True
    assert json.loads(state_path.read_text())["stage_attempts"]["smoke_sweep"] == 1

    calls.clear()
    state = runner.run_control_plane(
        plan,
        state_path,
        resume=True,
        stage_api=api,
    )
    assert state["status"] == "completed"
    assert calls[0][0] == "run_sweep"
    smoke_wrapper = json.loads(calls[0][1][0])
    assert "attempt" not in smoke_wrapper
    assert calls[0][1][-1] is True
    assert (tmp_path / "control.json.jsonl").is_file()


def test_remote_sweep_attempt_is_derived_from_durable_state(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    from scripts.stopdff_v5 import sweep

    absent = tmp_path / "absent"
    assert runner._resolve_remote_sweep_attempt(
        absent,
        recovery_requested=True,
        sweep_module=sweep,
    ) == (False, 1)
    assert not absent.exists()

    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(ValueError, match="attempt history"):
        runner._resolve_remote_sweep_attempt(
            partial,
            recovery_requested=True,
            sweep_module=sweep,
        )
    assert list(partial.iterdir()) == []

    records = [
        {
            "attempt": number,
            "state": "started",
            "mode": "fresh" if number == 1 else "resume",
            "command": ["dp_sweep"] + (["--resume"] if number > 1 else []),
        }
        for number in (1, 2)
    ]
    (partial / "attempts.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    assert runner._resolve_remote_sweep_attempt(
        partial,
        recovery_requested=True,
        sweep_module=sweep,
    ) == (True, 3)
    with pytest.raises(FileExistsError, match="already exists"):
        runner._resolve_remote_sweep_attempt(
            partial,
            recovery_requested=False,
            sweep_module=sweep,
        )


def test_local_resume_attempt_and_sweep_context(tmp_path, monkeypatch):
    run_spec_id = "1" * 64
    adapter_id = "2" * 64
    bootstrap_id = "3" * 64
    run_root = tmp_path / "run"
    run_root.mkdir()
    records = [
        {
            "attempt": 1,
            "state": "started",
            "mode": "fresh",
            "command": ["run_stopdff_v5_local"],
            "run_spec_id": run_spec_id,
            "adapter_id": adapter_id,
            "bootstrap_plan_id": bootstrap_id,
        },
        {
            "attempt": 2,
            "state": "started",
            "mode": "resume",
            "command": ["run_stopdff_v5_local", "--resume"],
            "run_spec_id": run_spec_id,
            "adapter_id": adapter_id,
            "bootstrap_plan_id": bootstrap_id,
        },
    ]
    (run_root / "attempts.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    assert local_runner._next_resume_attempt(
        run_root,
        run_spec_id=run_spec_id,
        adapter_id=adapter_id,
        bootstrap_plan_id=bootstrap_id,
    ) == 3

    plan = build_bootstrap_plan(["q1"], replicates=1, seed=1)
    captured = {}
    binding = {
        "rows": [],
        "calibration": {},
        "run_spec_identity": {},
        "run_spec_id": run_spec_id,
        "bootstrap_plan": plan,
        "fvi_tolerance": "1e-6",
        "fvi_max_iterations": 50,
        "variant": "smoke",
        "adapter_bundle_id": adapter_id,
        "fit_rows_sha256": "4" * 64,
        "eval_rows_sha256": "5" * 64,
        "bootstrap_plan_id": bootstrap_id,
    }
    monkeypatch.setattr(
        local_runner.checker,
        "resolve_run_binding",
        lambda **_kwargs: binding,
    )
    monkeypatch.setattr(
        local_runner.sweep,
        "run_sweep",
        lambda ctx: captured.setdefault("ctx", ctx) or {},
    )
    monkeypatch.setattr(
        local_runner.checker,
        "validate_run",
        lambda *_args, **_kwargs: types.SimpleNamespace(passed=True, errors=[]),
    )
    local_runner._run_bound_sweep(
        adapter_dir=tmp_path / "adapter",
        run_spec={},
        plan=plan,
        run_root=run_root,
        myopic_sha256="6" * 64,
        producer_hashes={},
        environment={},
        cells=[],
        command=["run_stopdff_v5_local", "--resume"],
        resume=True,
        attempt_number=3,
    )
    assert captured["ctx"].resume is True
    assert captured["ctx"].attempt["attempt"] == 3
    assert captured["ctx"].attempt["mode"] == "resume"
