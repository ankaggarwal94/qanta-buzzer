from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

from scripts import run_stopdff_v5_local as local_runner
from scripts.stopdff_v5.attempt_history import canonical_attempt_line
from scripts.stopdff_v5.bootstrap import build_bootstrap_plan
from scripts.stopdff_v5.identity import build_manifest, compute_id, sha256_file
from scripts.stopdff_v5.manifests import (
    ENVIRONMENT_PACKAGES,
    RAW_INPUT_ROLES,
    environment_contract_identity,
)


REPO = Path(__file__).resolve().parents[1]
MODAL_RUNNER = REPO / "scripts" / "modal_stopdff_v5_runner.py"


def _load_modal_runner(monkeypatch, *, modal_is_local: bool = True):
    image_envs: list[dict] = []
    local_dirs: list[tuple[tuple, dict]] = []

    class DummyImage:
        @classmethod
        def debian_slim(cls, **_kwargs):
            return cls()

        def apt_install(self, *_args):
            return self

        def pip_install(self, *_args):
            return self

        def env(self, values):
            image_envs.append(dict(values))
            return self

        def add_local_dir(self, *args, **kwargs):
            local_dirs.append((args, kwargs))
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

    fake_modal = types.SimpleNamespace(
        Image=DummyImage,
        Volume=DummyVolume,
        App=DummyApp,
        is_local=lambda: modal_is_local,
        image_envs=image_envs,
        local_dirs=local_dirs,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    from scripts.stopdff_v5.identity import build_manifest, sha256_file
    from scripts.stopdff_v5.manifests import source_manifest_identity

    if modal_is_local:
        source_bundle = Path(
            tempfile.mkdtemp(prefix="stopdff_v5_test_source_")
        )
        source = source_bundle / "source"
        source.mkdir()
        source_names = (
            "pyproject.toml",
            "scripts/stopdff_v5/checker.py",
            "scripts/stopdff_v5/sweep.py",
            "uv.lock",
        )
        for source_name in source_names:
            path = source / source_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{source_name}\n", encoding="utf-8")
        files = [
            {
                "path": source_name,
                "mode": "100644",
                "size": (source / source_name).stat().st_size,
                "sha256": sha256_file(source / source_name),
            }
            for source_name in source_names
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
    else:
        monkeypatch.delenv("STOPDFF_V5_SOURCE_DIR", raising=False)
        monkeypatch.setenv(
            "STOPDFF_V5_IMAGE_SOURCE_MANIFEST_ID",
            "1" * 64,
        )
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


def test_modal_remote_import_uses_baked_source_identity_without_host_bundle(
    monkeypatch,
) -> None:
    app_name = "cs321m-stopdff-v5-assurance-deadbeef"
    monkeypatch.setenv("STOPDFF_V5_APP_NAME", app_name)

    runner = _load_modal_runner(monkeypatch, modal_is_local=False)

    assert runner.SOURCE_BUNDLE_DIR == ""
    assert runner.SOURCE_DIR == runner.REMOTE_SRC
    assert runner._IMAGE_SOURCE_OWNER is None
    assert runner.IMAGE_SOURCE_MANIFEST_ID == "1" * 64
    assert runner.APP_NAME == app_name
    assert runner.modal.local_dirs == []
    assert runner.modal.image_envs[-1][
        "STOPDFF_V5_IMAGE_SOURCE_MANIFEST_ID"
    ] == "1" * 64
    assert runner.modal.image_envs[-1]["STOPDFF_V5_APP_NAME"] == app_name


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


def test_modal_executing_source_rehash_rejects_remote_runtime_byte_drift(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    manifest = runner._IMAGE_SOURCE_MANIFEST
    source_id = manifest["id"]
    staged_root = tmp_path / "volume" / "inputs" / f"source_{source_id}"
    staged_root.mkdir(parents=True)
    shutil.copytree(Path(runner.SOURCE_DIR), staged_root / "source")
    (staged_root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_root = tmp_path / "remote_src"
    shutil.copytree(Path(runner.SOURCE_DIR), runtime_root)
    drifted_entry = manifest["identity"]["files"][0]
    runtime_path = runtime_root / drifted_entry["path"]
    runtime_path.write_bytes(runtime_path.read_bytes() + b"\n# runtime drift\n")

    runner.MNT = str(tmp_path / "volume")
    runner.REMOTE_SRC = str(runtime_root)
    runner.IMAGE_SOURCE_MANIFEST_ID = source_id
    with pytest.raises(
        ValueError,
        match="executing source does not match source manifest",
    ) as exc_info:
        runner._verified_executing_source(source_id)
    assert drifted_entry["path"] in str(exc_info.value)


def test_adapter_build_is_fresh_only_and_fvi_cache_remains_bound(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    existing = tmp_path / "adapters" / "retry"
    existing.mkdir(parents=True)
    marker = existing / "preseeded.txt"
    marker.write_text("must remain untouched", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_verified_executing_source",
        lambda _source_id: {"id": "1" * 64},
    )
    monkeypatch.setattr(
        runner,
        "_verified_raw_input_manifest",
        lambda *_args, **_kwargs: {"id": "2" * 64},
    )
    monkeypatch.setattr(
        runner,
        "_verified_content_manifest",
        lambda *_args, **_kwargs: {"id": "3" * 64},
    )
    with pytest.raises(FileExistsError, match="fresh adapter build destination"):
        runner.build_adapter(
            "retry",
            "1" * 64,
            "2" * 64,
            "3" * 64,
        )
    assert marker.read_text(encoding="utf-8") == "must remain untouched"
    dangling = tmp_path / "adapters" / "dangling"
    dangling.symlink_to(tmp_path / "missing-adapter", target_is_directory=True)
    with pytest.raises(FileExistsError, match="fresh adapter build destination"):
        runner.build_adapter(
            "dangling",
            "1" * 64,
            "2" * 64,
            "3" * 64,
        )
    assert dangling.is_symlink()

    class AdapterChecker:
        @staticmethod
        def validate_adapter(_path):
            return types.SimpleNamespace(
                passed=True,
                errors=[],
                recomputed={"adapter_bundle_id": "a" * 64},
            )

        @staticmethod
        def load_json(path):
            return json.loads(path.read_text())

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


@pytest.mark.parametrize(
    "mutation",
    [
        "root_symlink",
        "plan_symlink",
        "dangling_root",
        "dangling_plan",
        "root_file",
        "plan_directory",
    ],
)
def test_bootstrap_cache_rejects_noncanonical_paths_before_decode(
    tmp_path,
    monkeypatch,
    mutation,
):
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    adapter_id = "a" * 64
    from scripts.stopdff_v5 import checker

    monkeypatch.setattr(
        checker,
        "validate_adapter",
        lambda _path: types.SimpleNamespace(
            passed=True,
            errors=[],
            recomputed={"adapter_bundle_id": adapter_id},
        ),
    )
    monkeypatch.setattr(
        checker,
        "load_adapter_rows",
        lambda _path: [
            {"item_id": "q1", "split": "test", "format": "MC"},
            {"item_id": "q1", "split": "test", "format": "QA"},
        ],
    )

    created = runner.bootstrap_plan(adapter_id, 1)
    assert created["cached"] is False
    root = tmp_path / "bootstrap" / created["bootstrap_plan_id"]
    plan_path = root / "bootstrap_plan.json"
    decoded: list[Path] = []
    original_load_json = checker.load_json

    def recording_load_json(path):
        decoded.append(Path(path))
        return original_load_json(path)

    monkeypatch.setattr(checker, "load_json", recording_load_json)
    reused = runner.bootstrap_plan(adapter_id, 1)
    assert reused["cached"] is True
    assert reused["bootstrap_plan_id"] == created["bootstrap_plan_id"]
    assert decoded == [plan_path]

    if mutation in {"root_symlink", "dangling_root"}:
        external = tmp_path / "external-bootstrap"
        root.rename(external)
        target = external if mutation == "root_symlink" else tmp_path / "missing"
        root.symlink_to(target, target_is_directory=True)
    elif mutation in {"plan_symlink", "dangling_plan"}:
        external = tmp_path / "external-bootstrap-plan.json"
        plan_path.rename(external)
        target = external if mutation == "plan_symlink" else tmp_path / "missing.json"
        plan_path.symlink_to(target)
    elif mutation == "root_file":
        shutil.rmtree(root)
        root.write_text("not a directory", encoding="utf-8")
    else:
        plan_path.unlink()
        plan_path.mkdir()

    with pytest.raises(FileExistsError, match="incomplete or noncanonical"):
        runner.bootstrap_plan(adapter_id, 1)
    assert decoded == [plan_path]


@pytest.mark.parametrize("mutation", ["root_symlink", "plan_symlink"])
def test_run_sweep_rejects_symlinked_bootstrap_cache_before_decode(
    tmp_path,
    monkeypatch,
    mutation,
):
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    bootstrap_id = "b" * 64
    root = tmp_path / "bootstrap" / bootstrap_id
    root.mkdir(parents=True)
    plan_path = root / "bootstrap_plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    if mutation == "root_symlink":
        external = tmp_path / "external-bootstrap"
        root.rename(external)
        root.symlink_to(external, target_is_directory=True)
    else:
        external = tmp_path / "external-bootstrap-plan.json"
        plan_path.rename(external)
        plan_path.symlink_to(external)

    from scripts.stopdff_v5 import checker

    monkeypatch.setattr(
        checker,
        "load_json",
        lambda _path: pytest.fail("noncanonical bootstrap plan was decoded"),
    )
    monkeypatch.setattr(
        checker,
        "resolve_run_binding",
        lambda **_kwargs: pytest.fail("noncanonical bootstrap plan was bound"),
    )
    wrapper = {
        "run_spec_identity": {
            "identity": {
                "source_manifest_id": runner.IMAGE_SOURCE_MANIFEST_ID,
            }
        }
    }
    with pytest.raises(ValueError, match="incomplete or noncanonical"):
        runner.run_sweep(
            json.dumps(wrapper),
            "a" * 64,
            bootstrap_id,
            False,
        )


def _write_model_manifest(root: Path, *, kind: str, model_id: str) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "snapshot").mkdir(exist_ok=True)
    manifest = {
        "id": model_id,
        "identity": {
            "kind": kind,
            "files": [],
        },
    }
    (root / "model_snapshot_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest


def test_freeze_model_cached_reuse_requires_model_snapshot_kind(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    model_id = "3" * 64
    root = tmp_path / "inputs" / "model"
    expected_kinds: list[object] = []

    def verifying_manifest(base, **kwargs):
        expected_kinds.append(kwargs.get("expected_kind"))
        manifest = json.loads(
            (Path(base) / kwargs["manifest_name"]).read_text(encoding="utf-8")
        )
        if kwargs.get("expected_kind") is not None and (
            manifest["identity"].get("kind") != kwargs["expected_kind"]
        ):
            raise ValueError("kind mismatch")
        return manifest

    monkeypatch.setattr(runner, "_verified_content_manifest", verifying_manifest)
    _write_model_manifest(root, kind="source_snapshot", model_id=model_id)
    with pytest.raises(ValueError, match="kind mismatch"):
        runner.freeze_model()

    _write_model_manifest(root, kind="model_snapshot", model_id=model_id)
    result = runner.freeze_model()

    assert result == {"model_id": model_id, "cached": True}
    assert expected_kinds == ["model_snapshot", "model_snapshot"]


def test_freeze_model_postfreeze_recheck_requires_model_snapshot_kind(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    model_id = "3" * 64
    from scripts.stopdff_v5 import adapter_build

    def freeze_to_wrong_kind(root):
        return _write_model_manifest(
            Path(root),
            kind="source_snapshot",
            model_id=model_id,
        )

    def verifying_manifest(base, **kwargs):
        manifest = json.loads(
            (Path(base) / kwargs["manifest_name"]).read_text(encoding="utf-8")
        )
        if kwargs.get("expected_kind") is not None and (
            manifest["identity"].get("kind") != kwargs["expected_kind"]
        ):
            raise ValueError("kind mismatch")
        return manifest

    monkeypatch.setattr(adapter_build, "freeze_model_snapshot", freeze_to_wrong_kind)
    monkeypatch.setattr(runner, "_verified_content_manifest", verifying_manifest)

    with pytest.raises(ValueError, match="kind mismatch"):
        runner.freeze_model()


def test_build_adapter_recheck_requires_model_snapshot_kind(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    source_id = "1" * 64
    raw_id = "2" * 64
    model_id = "3" * 64
    from scripts.stopdff_v5 import adapter_build

    _write_model_manifest(
        tmp_path / "inputs" / "model",
        kind="source_snapshot",
        model_id=model_id,
    )
    monkeypatch.setattr(
        runner,
        "_verified_executing_source",
        lambda _source_id: {"id": source_id},
    )
    monkeypatch.setattr(
        runner,
        "_verified_raw_input_manifest",
        lambda *_args, **_kwargs: {"id": raw_id},
    )

    def verifying_manifest(base, **kwargs):
        manifest = json.loads(
            (Path(base) / kwargs["manifest_name"]).read_text(encoding="utf-8")
        )
        if kwargs.get("expected_kind") is not None and (
            manifest["identity"].get("kind") != kwargs["expected_kind"]
        ):
            raise ValueError("kind mismatch")
        return manifest

    def forbidden_build(**_kwargs):
        raise AssertionError("adapter build must not run on poisoned model cache")

    monkeypatch.setattr(runner, "_verified_content_manifest", verifying_manifest)
    monkeypatch.setattr(adapter_build, "build_adapter_bundle", forbidden_build)

    with pytest.raises(ValueError, match="kind mismatch"):
        runner.build_adapter(
            "fresh",
            source_id,
            raw_id,
            model_id,
        )


def test_determinism_stage_owns_two_fresh_producer_calls(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    source_id = "1" * 64
    raw_id = "2" * 64
    model_id = "3" * 64
    hashes = {
        "fit_rows.jsonl.gz": "4" * 64,
        "eval_rows.jsonl.gz": "5" * 64,
        "calibration.json": "6" * 64,
        "build_metadata.json": "7" * 64,
    }
    adapter_manifest = build_manifest(
        {
            "kind": "adapter_bundle",
            "source_manifest_id": source_id,
            "raw_input_bundle_id": raw_id,
            "model_snapshot_id": model_id,
            "fit_rows_sha256": hashes["fit_rows.jsonl.gz"],
            "eval_rows_sha256": hashes["eval_rows.jsonl.gz"],
            "calibration_sha256": hashes["calibration.json"],
            "mc_retention_evidence": {
                "build_metadata_sha256": hashes["build_metadata.json"],
            },
        }
    )
    adapter_id = adapter_manifest["id"]
    monkeypatch.setattr(
        runner,
        "_verified_executing_source",
        lambda _source_id: {"id": source_id},
    )
    from scripts.stopdff_v5 import checker, identity

    monkeypatch.setattr(
        checker,
        "validate_adapter",
        lambda _path: types.SimpleNamespace(
            passed=True,
            errors=[],
            recomputed={"adapter_bundle_id": adapter_id},
        ),
    )
    monkeypatch.setattr(checker, "load_json", lambda _path: adapter_manifest)
    monkeypatch.setattr(identity, "sha256_file", lambda path: hashes[path.name])
    calls = []

    class ProducerCall:
        def __init__(self, execution_id, result):
            self.object_id = execution_id
            self._result = result

        def get(self):
            return dict(self._result)

    def spawn(subdir, source, raw, model, allow_override):
        calls.append((subdir, source, raw, model, allow_override))
        return ProducerCall(
            f"fc-{len(calls)}",
            {
                "adapter_id": adapter_id,
                "fit_rows_sha256": hashes["fit_rows.jsonl.gz"],
                "eval_rows_sha256": hashes["eval_rows.jsonl.gz"],
                "source_manifest_id": source,
                "raw_input_bundle_id": raw,
                "model_snapshot_id": model,
                "subdir": subdir,
                "cached": False,
            },
        )

    monkeypatch.setattr(runner.build_adapter, "spawn", spawn)
    result = runner.adapter_determinism_receipt(
        "fresh_a",
        "fresh_b",
        source_id,
        raw_id,
        model_id,
        True,
    )
    assert calls == [
        ("fresh_a", source_id, raw_id, model_id, True),
        ("fresh_b", source_id, raw_id, model_id, True),
    ]
    assert result["ok"] is True
    assert result["first_build_execution_id"] == "fc-1"
    assert result["second_build_execution_id"] == "fc-2"
    receipt_id = result["prerequisite_receipt_id"]
    assert (tmp_path / "receipts" / "determinism" / f"{receipt_id}.json").is_file()


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
        "adapter_determinism_receipt": record(
            "adapter_determinism",
            {
                "ok": True,
                "adapter_id": ids["adapter"],
                "source_manifest_id": ids["source"],
                "first_build_execution_id": "fc-first",
                "second_build_execution_id": "fc-second",
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
            "n": 42,
            "unexpected": [],
            "source_manifest_id": ids["source"],
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


def _probe_payload(*, torch_version: str = "1.0") -> dict:
    versions = {name: "1.0" for name in ENVIRONMENT_PACKAGES}
    versions["torch"] = torch_version
    return {
        "python": "3.11.0",
        "package_versions": versions,
    }


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
    determinism_call = next(
        call for call in calls if call[0] == "adapter_determinism"
    )
    assert determinism_call[1] == (
        "build_a",
        "build_b",
        ids["source"],
        ids["raw"],
        ids["model"],
        True,
    )
    smoke_call = next(call for call in calls if call[0] == "run_sweep")
    smoke_spec = json.loads(smoke_call[1][0])["run_spec_identity"]
    assert smoke_spec["gate"]["allow_low_mc_retention"] is True
    assert json.loads(state_path.read_text())["stage_attempts"]["smoke_sweep"] == 1
    resume_journal_offset = len(
        state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    calls.clear()
    state = runner.run_control_plane(
        plan,
        state_path,
        resume=True,
        stage_api=api,
    )
    assert state["status"] == "completed"
    assert [name for name, _args in calls[:2]] == ["probe", "run_sweep"]
    smoke_wrapper = json.loads(calls[1][1][0])
    assert "attempt" not in smoke_wrapper
    assert calls[1][1][-1] is True
    assert state["stage_attempts"]["environment_probe"] == 1
    assert state["stage_attempts"]["freeze_model"] == 1
    assert state["stage_attempts"]["adapter_determinism"] == 1
    resume_records = [
        json.loads(line)
        for line in state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[resume_journal_offset:]
    ]
    assert not any(
        record["event"]
        in {"stage_checkpoint_invalid", "stage_checkpoint_refresh_required"}
        for record in resume_records
    )
    assert (tmp_path / "control.json.jsonl").is_file()


def test_nonterminal_resume_refreshes_changed_environment_once(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api(fail_first_smoke=True)
    plan = {
        "source_id": ids["source"],
        "raw_id": ids["raw"],
        "adapter_subdirs": ["build_a", "build_b"],
        "gate_overrides": {},
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
    old_smoke_call = next(call for call in calls if call[0] == "run_sweep")
    old_environment_id = json.loads(old_smoke_call[1][0])[
        "run_spec_identity"
    ]["identity"]["environment_contract_id"]
    resume_journal_offset = len(
        state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    current_probe = _probe_payload(torch_version="2.0")
    current_environment_id = compute_id(
        environment_contract_identity(
            python_version=current_probe["python"],
            package_versions=current_probe["package_versions"],
        )
    )
    probe_calls = []

    def probe_current_environment():
        probe_calls.append(True)
        return current_probe

    original_sweep = api["run_sweep"]
    interrupt_refreshed_smoke = True

    def interrupt_first_refreshed_smoke(spec_json, *args):
        nonlocal interrupt_refreshed_smoke
        result = original_sweep(spec_json, *args)
        variant = json.loads(spec_json)["run_spec_identity"]["profile_variant"]
        if variant == "smoke" and interrupt_refreshed_smoke:
            interrupt_refreshed_smoke = False
            raise RuntimeError("lost refreshed smoke response")
        return result

    api["probe"] = probe_current_environment
    api["run_sweep"] = interrupt_first_refreshed_smoke
    calls.clear()
    with pytest.raises(RuntimeError, match="lost refreshed smoke response"):
        runner.run_control_plane(
            plan,
            state_path,
            resume=True,
            stage_api=api,
        )

    drift_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert drift_state["status"] == "failed"
    assert probe_calls == [True]
    assert old_environment_id != current_environment_id
    resume_records = [
        json.loads(line)
        for line in state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[resume_journal_offset:]
    ]
    assert [
        (record["event"], record["stage"])
        for record in resume_records[:6]
    ] == [
        ("stage_checkpoint_invalid", "smoke_bootstrap"),
        ("stage_checkpoint_invalid", "fvi_study"),
        ("stage_checkpoint_invalid", "promote_adapter"),
        ("stage_checkpoint_invalid", "adapter_determinism"),
        ("stage_checkpoint_invalid", "freeze_model"),
        ("stage_checkpoint_refresh_required", "environment_probe"),
    ]
    second_resume_offset = len(
        state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    state = runner.run_control_plane(
        plan,
        state_path,
        resume=True,
        stage_api=api,
    )

    assert state["status"] == "completed"
    assert probe_calls == [True, True]
    sweep_calls = [args for name, args in calls if name == "run_sweep"]
    assert len(sweep_calls) == 3
    assert {
        json.loads(args[0])["run_spec_identity"]["identity"][
            "environment_contract_id"
        ]
        for args in sweep_calls
    } == {current_environment_id}
    assert state["completed"]["environment_probe"] == current_probe
    assert state["stage_attempts"]["verify_source"] == 1
    assert state["stage_attempts"]["verify_raw"] == 1
    for stage in (
        "environment_probe",
        "freeze_model",
        "adapter_determinism",
        "promote_adapter",
        "fvi_study",
        "smoke_bootstrap",
    ):
        assert state["stage_attempts"][stage] == 2
    assert state["stage_attempts"]["smoke_sweep"] == 3
    second_resume_records = [
        json.loads(line)
        for line in state_path.with_name("control.json.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[second_resume_offset:]
    ]
    assert not any(
        record["event"]
        in {"stage_checkpoint_invalid", "stage_checkpoint_refresh_required"}
        for record in second_resume_records
    )
    runner._reconcile_control_journal(
        state_path,
        runner._load_control_json(state_path),
    )


@pytest.mark.parametrize("probe_mode", ["raises", "malformed"])
def test_nonterminal_resume_probe_failure_preserves_checkpoints(
    tmp_path,
    monkeypatch,
    probe_mode,
):
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api(fail_first_smoke=True)
    plan = {
        "source_id": ids["source"],
        "raw_id": ids["raw"],
        "adapter_subdirs": ["build_a", "build_b"],
        "gate_overrides": {},
        "resource_summary": {"backend": "modal"},
    }
    state_path = tmp_path / "control.json"
    journal_path = state_path.with_name("control.json.jsonl")
    with pytest.raises(RuntimeError, match="lost smoke response"):
        runner.run_control_plane(
            plan,
            state_path,
            resume=False,
            stage_api=api,
        )
    state_before = state_path.read_bytes()
    journal_before = journal_path.read_bytes()
    probe_calls = []

    def unavailable_probe():
        probe_calls.append(True)
        if probe_mode == "raises":
            raise RuntimeError("probe unavailable")
        malformed = _probe_payload()
        malformed["package_versions"].pop(ENVIRONMENT_PACKAGES[-1])
        return malformed

    api["probe"] = unavailable_probe
    calls.clear()
    expected_error = (
        "probe unavailable"
        if probe_mode == "raises"
        else "incomplete package set"
    )
    with pytest.raises((RuntimeError, ValueError), match=expected_error):
        runner.run_control_plane(
            plan,
            state_path,
            resume=True,
            stage_api=api,
        )

    assert probe_calls == [True]
    assert calls == []
    assert state_path.read_bytes() == state_before
    assert journal_path.read_bytes() == journal_before


def test_resume_without_probe_checkpoint_invokes_probe_once(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api()
    plan = {
        "source_id": ids["source"],
        "raw_id": ids["raw"],
        "adapter_subdirs": ["build_a", "build_b"],
        "gate_overrides": {},
        "resource_summary": {"backend": "modal"},
    }
    original_verify = api["verify_volume_artifact"]
    fail_raw = True

    def fail_raw_once(rel_dir, kind):
        nonlocal fail_raw
        if kind == "raw" and fail_raw:
            fail_raw = False
            raise RuntimeError("raw verification interrupted")
        return original_verify(rel_dir, kind)

    api["verify_volume_artifact"] = fail_raw_once
    state_path = tmp_path / "control.json"
    with pytest.raises(RuntimeError, match="raw verification interrupted"):
        runner.run_control_plane(
            plan,
            state_path,
            resume=False,
            stage_api=api,
        )
    assert "environment_probe" not in json.loads(
        state_path.read_text(encoding="utf-8")
    )["completed"]

    calls.clear()
    state = runner.run_control_plane(
        plan,
        state_path,
        resume=True,
        stage_api=api,
    )

    assert state["status"] == "completed"
    assert [name for name, _args in calls].count("probe") == 1
    assert state["stage_attempts"]["environment_probe"] == 1


def test_fresh_control_rejects_orphan_journal_before_remote_work(
    tmp_path,
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    api, calls, ids = _fake_control_api()
    plan = {
        "source_id": ids["source"],
        "raw_id": ids["raw"],
        "adapter_subdirs": ["build_a", "build_b"],
        "gate_overrides": {},
        "resource_summary": {"backend": "modal"},
    }
    state_path = tmp_path / "control.json"
    state_path.with_name("control.json.jsonl").write_text(
        '{"stale":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="state or journal"):
        runner.run_control_plane(
            plan,
            state_path,
            resume=False,
            stage_api=api,
        )
    assert calls == []


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
            "run_spec_id": "1" * 64,
            "adapter_id": "2" * 64,
            "bootstrap_plan_id": "3" * 64,
        }
        for number in (1, 2)
    ]
    (partial / "attempts.jsonl").write_bytes(
        b"".join(canonical_attempt_line(record) for record in records)
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


@pytest.mark.parametrize(
    ("resume", "attempt_number"),
    [(False, 1), (True, 3)],
)
def test_local_attempt_and_sweep_context(
    tmp_path,
    monkeypatch,
    resume,
    attempt_number,
):
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
    (run_root / "attempts.jsonl").write_bytes(
        b"".join(canonical_attempt_line(record) for record in records)
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
        "gate_overrides": {
            "allow_low_mc_retention": True,
            "allow_incomplete_mc_coverage": False,
        },
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
        command=[
            "run_stopdff_v5_local",
            *(["--resume"] if resume else []),
        ],
        resume=resume,
        attempt_number=attempt_number,
    )
    assert captured["ctx"].gate_overrides == binding["gate_overrides"]
    assert captured["ctx"].resume is resume
    assert captured["ctx"].attempt["attempt"] == attempt_number
    assert captured["ctx"].attempt["mode"] == (
        "resume" if resume else "fresh"
    )


def test_local_runner_makes_reviewed_checkout_authoritative_and_rejects_drift(
    tmp_path,
):
    checkout_b = tmp_path / "checkout_b"
    shutil.copytree(REPO / "scripts", checkout_b / "scripts")
    foreign_adapter = checkout_b / "scripts" / "stopdff_v5" / "adapter_build.py"
    foreign_adapter.write_text(
        foreign_adapter.read_text(encoding="utf-8")
        + "\n# runtime-byte-drift\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(checkout_b), str(REPO)))

    ordinary = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_stopdff_v5_local.py"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert ordinary.returncode == 0, ordinary.stderr

    probe = f"""
import importlib.util
import sys
from pathlib import Path

foreign = Path({str(checkout_b)!r})
reviewed = Path({str(REPO)!r})
sys.path.insert(0, str(foreign))
from scripts.stopdff_v5 import adapter_build  # preload checkout B

spec = importlib.util.spec_from_file_location(
    "_reviewed_local_runner",
    reviewed / "scripts" / "run_stopdff_v5_local.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runtime = reviewed / "scripts" / "stopdff_v5" / "adapter_build.py"
manifest = runner.build_manifest({{
    "kind": "source_snapshot",
    "files": [{{
        "path": "scripts/stopdff_v5/adapter_build.py",
        "mode": "100644",
        "sha256": runner.sha256_file(runtime),
    }}],
}})
runner._verified_local_source_execution(reviewed, manifest)
"""
    drift = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert drift.returncode != 0
    assert "does not originate from the executing repository" in drift.stderr


def test_local_source_rehash_rejects_correct_origin_runtime_byte_drift(tmp_path):
    reviewed = tmp_path / "reviewed"
    shutil.copytree(REPO / "scripts", reviewed / "scripts")
    shutil.copytree(REPO / "qb_data", reviewed / "qb_data")
    probe = f"""
import importlib.util
import sys
from pathlib import Path

reviewed = Path({str(reviewed)!r})
sys.path.insert(0, str(reviewed))
spec = importlib.util.spec_from_file_location(
    "_reviewed_local_runner_byte_drift",
    reviewed / "scripts" / "run_stopdff_v5_local.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
relative = "scripts/stopdff_v5/adapter_build.py"
runtime = reviewed / relative
manifest = runner.build_manifest({{
    "kind": "source_snapshot",
    "files": [{{
        "path": relative,
        "mode": "100644",
        "sha256": runner.sha256_file(runtime),
    }}],
}})
runtime.write_bytes(runtime.read_bytes() + b"\\n# runtime drift\\n")
runner._verified_local_source_execution(reviewed, manifest)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "executing source does not match source manifest: "
        "scripts/stopdff_v5/adapter_build.py"
    ) in result.stderr
    assert "does not originate from the executing repository" not in result.stderr


def test_local_source_rehash_rejects_runtime_executable_mode_drift(tmp_path):
    reviewed = tmp_path / "reviewed"
    shutil.copytree(REPO / "scripts", reviewed / "scripts")
    shutil.copytree(REPO / "qb_data", reviewed / "qb_data")
    probe = f"""
import importlib.util
import sys
from pathlib import Path

reviewed = Path({str(reviewed)!r})
sys.path.insert(0, str(reviewed))
spec = importlib.util.spec_from_file_location(
    "_reviewed_local_runner_mode_drift",
    reviewed / "scripts" / "run_stopdff_v5_local.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
relative = "scripts/stopdff_v5/adapter_build.py"
runtime = reviewed / relative
manifest = runner.build_manifest({{
    "kind": "source_snapshot",
    "files": [{{
        "path": relative,
        "mode": "100644",
        "sha256": runner.sha256_file(runtime),
    }}],
}})
runtime.chmod(runtime.stat().st_mode | 0o111)
runner._verified_local_source_execution(reviewed, manifest)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "executing source does not match source manifest: "
        "scripts/stopdff_v5/adapter_build.py"
    ) in result.stderr


@pytest.mark.parametrize(
    ("runtime_drift", "expected_error"),
    [
        (True, "hidden runtime drift"),
        (False, "after source preflight"),
    ],
)
def test_run_present_resume_rehashes_executing_source_before_cached_evidence(
    tmp_path,
    monkeypatch,
    runtime_drift,
    expected_error,
):
    out = tmp_path / "reproduction"
    out.mkdir()
    source = {
        "id": "1" * 64,
        "identity": {
            "kind": "source_snapshot",
            "git_sha": "a" * 40,
            "files": [],
        },
    }
    raw = {
        "id": "2" * 64,
        "identity": {
            "kind": "raw_input_bundle",
            "files": [],
            "semantic_checks": {"all_semantic_checks_pass": True},
        },
    }
    model = {
        "id": "3" * 64,
        "identity": {"kind": "model_snapshot", "files": []},
    }
    manifests = {
        "source_snapshot": source,
        "raw_inputs": raw,
        "model": model,
    }
    preflight_order = []

    def load_manifest(base, **_kwargs):
        name = Path(base).name
        preflight_order.append(f"load:{name}")
        return manifests[name]

    monkeypatch.setattr(
        local_runner,
        "_load_bound_content_manifest",
        load_manifest,
    )
    source_preflights = []

    def verify_source(repo_root, manifest):
        preflight_order.append("verify:executing_source")
        source_preflights.append((repo_root, manifest))
        if runtime_drift:
            raise ValueError("hidden runtime drift")
        return {
            "environment": "local_clean_worktree",
            "executing_source_manifest_id": manifest["id"],
            "runtime_source_manifest_id": manifest["id"],
        }

    monkeypatch.setattr(
        local_runner,
        "_verified_local_source_execution",
        verify_source,
    )

    def after_source_preflight(_path):
        raise ValueError("after source preflight")

    monkeypatch.setattr(
        local_runner.checker,
        "validate_adapter",
        after_source_preflight,
    )
    args = types.SimpleNamespace(repo_root=tmp_path)

    with pytest.raises(ValueError, match=expected_error):
        local_runner._resume_local_run(
            args=args,
            out=out,
            run_sha="a" * 40,
        )
    assert source_preflights == [(tmp_path, source)]
    assert preflight_order[:2] == [
        "load:source_snapshot",
        "verify:executing_source",
    ]


def test_local_versions_require_the_exact_declared_package_set(monkeypatch):
    missing = ENVIRONMENT_PACKAGES[-1]

    def incomplete(name):
        if name == missing:
            raise local_runner.im.PackageNotFoundError(name)
        return f"version-{name}"

    monkeypatch.setattr(local_runner.im, "version", incomplete)
    with pytest.raises(ValueError, match=f"missing: {missing}"):
        local_runner._versions()

    monkeypatch.setattr(
        local_runner.im,
        "version",
        lambda name: f"version-{name}",
    )
    versions = local_runner._versions()
    assert tuple(versions) == ENVIRONMENT_PACKAGES
    assert set(versions) == set(ENVIRONMENT_PACKAGES)


@pytest.mark.parametrize(
    ("stored_skip", "requested_skip", "compatible"),
    [
        (True, False, False),
        (False, True, False),
        (True, True, True),
        (False, False, True),
    ],
)
def test_run_present_resume_validates_current_fvi_mode_before_dispatch(
    tmp_path,
    monkeypatch,
    stored_skip,
    requested_skip,
    compatible,
):
    out = tmp_path / "reproduction"
    (out / "runs" / "smoke_local_candidate").mkdir(parents=True)
    lifecycle = {
        "schema_version": 1,
        "run_sha": "a" * 40,
        "variant": "smoke",
        "skip_fvi_study": stored_skip,
        "fvi_tolerance": "1e-6",
        "fvi_max_iterations": 100,
        "allow_low_mc_retention": False,
        "adapter_executions": {},
    }
    (out / "local_lifecycle.json").write_text(
        json.dumps(lifecycle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def git_run(command, **_kwargs):
        stdout = "" if "status" in command else "a" * 40 + "\n"
        return types.SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(local_runner, "_verify_imported_producer_origins", lambda: None)
    monkeypatch.setattr(local_runner.subprocess, "run", git_run)
    dispatched = []
    monkeypatch.setattr(
        local_runner,
        "_resume_local_run",
        lambda **kwargs: dispatched.append(kwargs) or 0,
    )
    argv = ["--out-dir", str(out), "--variant", "smoke", "--resume"]
    if requested_skip:
        argv.append("--skip-fvi-study")

    if compatible:
        assert local_runner.main(argv) == 0
        assert len(dispatched) == 1
    else:
        with pytest.raises(
            ValueError,
            match="local lifecycle checkpoint does not match this command",
        ):
            local_runner.main(argv)
        assert dispatched == []


def test_public_local_lifecycle_resumes_before_run_directory_exists(
    tmp_path,
    monkeypatch,
):
    out = tmp_path / "reproduction"
    ids = {
        "source": "1" * 64,
        "raw": "2" * 64,
        "model": "3" * 64,
        "adapter": "4" * 64,
    }
    manifests = {
        "source_snapshot": {
            "id": ids["source"],
            "identity": {
                "kind": "source_snapshot",
                "git_sha": "a" * 40,
                "files": [],
            },
        },
        "raw_inputs": {
            "id": ids["raw"],
            "identity": {
                "kind": "raw_input_bundle",
                "files": [
                    {"role": "stopdff.json", "sha256": "b" * 64},
                ],
                "semantic_checks": {"all_semantic_checks_pass": True},
            },
        },
        "model": {
            "id": ids["model"],
            "identity": {
                "kind": "model_snapshot",
                "model_revision": "c" * 40,
                "files": [],
            },
        },
    }
    adapter_manifest = {
        "id": ids["adapter"],
        "identity": {
            "source_manifest_id": ids["source"],
            "raw_input_bundle_id": ids["raw"],
            "model_snapshot_id": ids["model"],
        },
    }
    calls = {"source": 0, "raw": 0, "model": 0, "adapter": 0, "run": 0}

    def git_run(command, **_kwargs):
        stdout = "" if "status" in command else "a" * 40 + "\n"
        return types.SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(local_runner.subprocess, "run", git_run)
    monkeypatch.setattr(
        local_runner,
        "_verified_local_source_execution",
        lambda _repo, manifest: {
            "environment": "local_clean_worktree",
            "executing_source_manifest_id": manifest["id"],
            "runtime_source_manifest_id": manifest["id"],
        },
    )
    monkeypatch.setattr(
        local_runner,
        "_load_bound_content_manifest",
        lambda base, **_kwargs: manifests[Path(base).name],
    )

    def source_build(_repo, _sha, staged):
        calls["source"] += 1
        staged.mkdir(parents=True)
        return manifests["source_snapshot"]

    def raw_build(_roles, staged):
        calls["raw"] += 1
        (staged / "raw").mkdir(parents=True)
        return manifests["raw_inputs"]

    def model_build(staged):
        calls["model"] += 1
        if calls["model"] == 1:
            raise RuntimeError("interrupted before sweep creation")
        (staged / "snapshot").mkdir(parents=True)
        return manifests["model"]

    def adapter_build(**kwargs):
        calls["adapter"] += 1
        destination = Path(kwargs["out_dir"])
        destination.mkdir(parents=True)
        (destination / "calibration.json").write_text("{}", encoding="utf-8")
        return adapter_manifest

    monkeypatch.setattr(local_runner.producers, "build_source_snapshot", source_build)
    monkeypatch.setattr(local_runner.producers, "stage_raw_inputs", raw_build)
    monkeypatch.setattr(local_runner.adapter_build, "freeze_model_snapshot", model_build)
    monkeypatch.setattr(local_runner.adapter_build, "build_adapter_bundle", adapter_build)
    monkeypatch.setattr(
        local_runner,
        "_load_valid_adapter_stage",
        lambda *_args, **_kwargs: adapter_manifest,
    )
    monkeypatch.setattr(
        local_runner.selftest,
        "run_self_test",
        lambda _path: (
            True,
            [
                {
                    "mutation": mutation,
                    "expected": "PASS" if index == 0 else "REJECT",
                    "passed_check": index == 0,
                    "ok": True,
                    "errors": [],
                }
                for index, mutation in enumerate(local_runner.MUTATION_ROSTER)
            ],
        ),
    )
    monkeypatch.setattr(
        local_runner.checker,
        "load_adapter_rows",
        lambda _path: [
            {"item_id": "q1", "split": "test", "format": "MC"},
            {"item_id": "q1", "split": "test", "format": "QA"},
        ],
    )
    monkeypatch.setattr(
        local_runner,
        "_versions",
        lambda: {name: "1.0" for name in ENVIRONMENT_PACKAGES},
    )
    monkeypatch.setattr(local_runner, "sha256_file", lambda _path: "d" * 64)

    def run_sweep(**kwargs):
        calls["run"] += 1
        kwargs["run_root"].mkdir(parents=True)
        return (
            {
                "release_status": "VALID",
                "requested": 1,
                "completed": 1,
                "failed": 0,
                "family": {"verdict": "PASS"},
            },
            types.SimpleNamespace(passed=True, errors=[]),
        )

    packaged = []
    monkeypatch.setattr(local_runner, "_run_bound_sweep", run_sweep)
    monkeypatch.setattr(
        local_runner,
        "_package_and_validate_local_run",
        lambda **kwargs: packaged.append(kwargs) or 0,
    )
    argv = [
        "--out-dir",
        str(out),
        "--variant",
        "smoke",
        "--skip-fvi-study",
    ]
    with pytest.raises(RuntimeError, match="interrupted before sweep creation"):
        local_runner.main(argv)
    assert (out / "local_lifecycle.json").is_file()
    assert (out / "source_snapshot").is_dir()
    assert (out / "raw_inputs").is_dir()
    assert not (out / "runs").exists()

    assert local_runner.main([*argv, "--resume"]) == 0
    assert calls == {"source": 1, "raw": 1, "model": 2, "adapter": 1, "run": 1}
    assert len(packaged) == 1


def test_local_status_query_excludes_only_resumed_in_repo_output(tmp_path):
    repo = tmp_path / "repo"
    out = repo / "stopdff_v5_final_out"
    command = local_runner._worktree_status_command(
        repo_root=repo,
        out=out,
        resume=True,
    )
    assert command[-2:] == [
        ".",
        ":(top,exclude)stopdff_v5_final_out",
    ]
    assert local_runner._worktree_status_command(
        repo_root=repo,
        out=out,
        resume=False,
    )[-1] == "."


def test_checkpointed_mutation_results_require_complete_roster():
    with pytest.raises(ValueError, match="mutation roster"):
        local_runner._validate_checkpointed_mutation_results(
            [
                {
                    "mutation": "<baseline valid>",
                    "expected": "PASS",
                    "passed_check": True,
                    "ok": True,
                    "errors": [],
                }
            ]
        )


def test_local_resume_rejects_symlinked_runs_directory(tmp_path, monkeypatch):
    out = tmp_path / "reproduction"
    out.mkdir()
    external = tmp_path / "external-runs"
    external.mkdir()
    (out / "runs").symlink_to(external, target_is_directory=True)

    def git_run(command, **_kwargs):
        stdout = "" if "status" in command else "a" * 40 + "\n"
        return types.SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(local_runner.subprocess, "run", git_run)
    with pytest.raises(ValueError, match="runs directory.*symlink"):
        local_runner.main(
            [
                "--out-dir",
                str(out),
                "--variant",
                "smoke",
                "--skip-fvi-study",
                "--resume",
            ]
        )
