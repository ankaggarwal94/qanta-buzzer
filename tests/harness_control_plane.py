"""Shared StopDFF v5 test harness (not collected: no ``test_`` prefix).

Home of the cross-file test helpers that previously lived in
``tests/test_stopdff_v5_control_plane.py``, ``tests/test_modal_runner_recovery.py``
and ``tests/test_stopdff_v5_pipeline.py`` and were imported test-file-to-test-file
by a dozen suites. Test files are not a stable import surface (renaming or
collecting one executes another); this plain module is.

Helpers
-------
_load_modal_runner
    Load ``scripts/modal_stopdff_v5_runner.py`` under a Dummy-Modal stub with a
    staged synthetic source bundle (``modal_is_local=True``) or a baked image
    identity (``modal_is_local=False``).
_fake_control_api
    In-memory control-plane stage API with recorded calls and canonical ids.
_write_raw_manifest / _write_model_manifest
    Minimal on-disk input manifests for the volume-verification stages.
_plan
    Canonical control-plane plan dict over ``_fake_control_api`` ids.
_synth_rows / _calibration_json / _make_ctx
    Synthetic adapter rows, Platt calibration block, and a bound
    ``sweep.SweepContext`` for pipeline-level sweeps.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import bootstrap, identity, sweep  # noqa: E402
from scripts.stopdff_v5.identity import build_manifest, sha256_file  # noqa: E402
from scripts.stopdff_v5.manifests import (  # noqa: E402
    ENVIRONMENT_PACKAGES,
    RAW_INPUT_ROLES,
    environment_contract_identity,
    run_spec_identity,
)

MODAL_RUNNER = REPO / "scripts" / "modal_stopdff_v5_runner.py"

CATEGORIES = ["history", "science", "arts"]
PREFIX_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def _load_modal_runner(monkeypatch, *, modal_is_local: bool = True):
    image_envs: list[dict] = []
    local_dirs: list[tuple[tuple, dict]] = []
    apt_installs: list[tuple] = []

    class DummyImage:
        @classmethod
        def debian_slim(cls, **_kwargs):
            return cls()

        def apt_install(self, *args):
            apt_installs.append(tuple(args))
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
        apt_installs=apt_installs,
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


def _plan(ids: dict[str, str]) -> dict:
    return {
        "source_id": ids["source"],
        "raw_id": ids["raw"],
        "adapter_subdirs": ["build_a", "build_b"],
        "gate_overrides": {},
        "resource_summary": {"backend": "modal"},
    }


def _synth_rows(n_items: int = 40, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_items):
        qid = f"q{i:03d}"
        split = "val" if i < n_items // 2 else "test"
        cat = CATEGORIES[i % len(CATEGORIES)]
        item_off = rng.uniform(-0.15, 0.15)
        for t, frac in enumerate(PREFIX_FRACS):
            mc_sim = float(np.clip(0.25 + 0.55 * frac + item_off + rng.uniform(-0.05, 0.05), 0.0, 1.0))
            qa_sim = float(np.clip(0.20 + 0.60 * frac + item_off + rng.uniform(-0.05, 0.05), 0.0, 1.0))
            mc_correct = int(mc_sim + rng.uniform(-0.15, 0.15) > 0.55)
            rows.append({
                "item_id": qid, "prefix_idx": t, "prefix_fraction": frac, "format": "MC",
                "split": split, "raw_similarity": mc_sim, "correct": mc_correct, "category": cat,
            })
            rows.append({
                "item_id": qid, "prefix_idx": t, "prefix_fraction": frac, "format": "QA",
                "split": split, "raw_similarity": qa_sim, "correct": 1, "category": cat,
            })
    return rows


def _calibration_json() -> dict:
    block = {"platt_coef": 5.0, "platt_intercept": -2.5}
    return {"per_bucket": {"early": dict(block), "mid": dict(block), "late": dict(block)}}


def _test_item_ids(rows: list[dict]) -> list[str]:
    mc = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "MC"}
    qa = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "QA"}
    return sorted(mc & qa)


def _make_ctx(tmp_path: Path, rows, cells, replicates=100) -> sweep.SweepContext:
    plan = bootstrap.build_bootstrap_plan(_test_item_ids(rows), replicates=replicates, seed=1)
    adapter_id = "e" * 64
    myopic_sha256 = "c" * 64
    producer_hashes = {
        "checker.py": "d" * 64,
        "sweep.py": "f" * 64,
    }
    environment = {
        "python_version": "3.11.0",
        "package_versions": {name: "test" for name in ENVIRONMENT_PACKAGES},
    }
    environment_id = identity.compute_id(
        environment_contract_identity(**environment)
    )
    run_spec = run_spec_identity(
        source_manifest_id="1" * 64,
        raw_input_bundle_id="2" * 64,
        model_snapshot_id="3" * 64,
        adapter_bundle_id=adapter_id,
        fvi_study_id="4" * 64,
        bootstrap_plan_id=identity.compute_id(bootstrap.plan_identity(plan)),
        environment_contract_id=environment_id,
        resource_summary_id=identity.compute_id({"backend": "modal"}),
        fvi_selected={"tolerance": "1e-8", "max_iterations": 100},
        replicate_count=replicates,
        profile_variant="smoke",
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=producer_hashes,
        prerequisite_receipts={},
    )
    return sweep.SweepContext(
        rows=rows, calibration_json=_calibration_json(),
        run_spec=run_spec, run_spec_id=identity.compute_id(run_spec),
        bootstrap_plan=plan, output_dir=tmp_path / "run",
        fvi_tolerance="1e-8", fvi_max_iterations=100, backend="modal",
        profile_variant="smoke", adapter_bundle_id=adapter_id,
        adapter_fit_rows_sha256="a" * 64,
        adapter_eval_rows_sha256="b" * 64,
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=producer_hashes,
        cells=cells,
        environment=environment,
        resource_summary={"backend": "modal"},
        attempt={"attempt": 1, "mode": "fresh", "command": ["dp_sweep"]},
    )
