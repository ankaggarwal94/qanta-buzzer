#!/usr/bin/env python3
"""Modal standalone runner for the StopDFF v5 evidentiary pipeline.

Local entrypoints (run in the control venv; import only modal + stdlib):
  build_image_probe   -- force image build + report versions
  verify_inputs       -- recompute source/raw hashes on Modal vs manifests
  build_adapter       -- GPU: freeze model snapshot + build deterministic adapter bundle
  fvi_study           -- CPU: preregistered FVI candidate study + selector
  bootstrap_plans     -- CPU: build 100-replicate + 1000-replicate common plans
  smoke               -- CPU: exact two-cell smoke (100 replicates)
  mutation_gate       -- CPU: negative mutation suite on a real package
  dp_sweep            -- CPU: 96-cell final sweep (1000 replicates), per-cell commit
  validate            -- CPU: standalone checker (backend=modal)
  package             -- CPU: reports/figures/checksums/external artifacts
  durability_pilot    -- cheap detached-path durability proof

The Modal image context is ONLY the verified source snapshot dir given by
STOPDFF_V5_SOURCE_DIR. Raw inputs and the model snapshot are separate hash-verified
Volume artifacts. One writer per run dir, max_containers=1, explicit Volume commits,
reload before resume, stage-specific resources (GPU only for build_adapter).
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "cs321m-stopdff-v5"
VOLUME_NAME = "cs321m-stopdff-artifacts"
VOLUME_MOUNT = PurePosixPath("/stopdff")
REMOTE_SRC = PurePosixPath("/root/src")
MAX_TIMEOUT = 86400

# Source-only image context (git-archive snapshot). Set by the control plane.
SOURCE_DIR = os.environ.get("STOPDFF_V5_SOURCE_DIR", "")

_PIP = [
    "numpy>=1.26,<3",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "pandas>=2.1",
    "matplotlib>=3.7",
    "sentence-transformers>=2.7",
    "huggingface_hub>=0.23",
]

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_PIP)
    .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "HF_HUB_DISABLE_TELEMETRY": "1",
          "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
          "TOKENIZERS_PARALLELISM": "false"})
)
if SOURCE_DIR:
    _image = _image.add_local_dir(SOURCE_DIR, remote_path=str(REMOTE_SRC), copy=True)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=_image)


def _init_paths():
    import sys
    sys.path.insert(0, str(REMOTE_SRC))


# --- remote functions -------------------------------------------------------------


@app.function(volumes={str(VOLUME_MOUNT): vol}, timeout=1800, max_containers=1)
def probe() -> dict:
    _init_paths()
    import numpy, scipy, sklearn, pandas
    import sentence_transformers, transformers, huggingface_hub
    return {
        "python": __import__("platform").python_version(),
        "numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "pandas": pandas.__version__, "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__, "huggingface_hub": huggingface_hub.__version__,
    }


@app.function(volumes={str(VOLUME_MOUNT): vol}, timeout=3600, max_containers=1)
def verify_volume_artifact(rel_dir: str, manifest_rel: str, kind: str) -> dict:
    """Recompute file hashes under /stopdff/<rel_dir> and compare to the manifest."""
    _init_paths()
    import json
    from scripts.stopdff_v5.identity import compute_id, sha256_file, verify_manifest_id
    base = Path(str(VOLUME_MOUNT)) / rel_dir
    manifest = json.loads((base / manifest_rel).read_text())
    verify_manifest_id(manifest)
    ident = manifest["identity"]
    mismatches = []
    if kind == "raw":
        for f in ident["files"]:
            p = base / f["role"]
            if not p.is_file() or sha256_file(p) != f["sha256"]:
                mismatches.append(f["role"])
    elif kind == "source":
        for f in ident["files"]:
            p = base / "source" / f["path"]
            if not p.is_file() or sha256_file(p) != f["sha256"]:
                mismatches.append(f["path"])
    elif kind == "model":
        for f in ident["files"]:
            p = base / "snapshot" / f["path"]
            if not p.is_file() or sha256_file(p) != f["sha256"]:
                mismatches.append(f["path"])
    return {"kind": kind, "id": manifest["id"], "mismatches": mismatches[:20], "ok": not mismatches}


@app.function(volumes={str(VOLUME_MOUNT): vol}, gpu="L40S", timeout=MAX_TIMEOUT, max_containers=1)
def build_adapter_remote(raw_id: str, source_id: str, pilot_tag: str) -> dict:
    """GPU: freeze model snapshot (if absent) + build a deterministic adapter bundle."""
    _init_paths()
    import json
    from scripts.stopdff_v5 import adapter_build
    from scripts.stopdff_v5.identity import compute_id, sha256_file

    vol.reload()
    raw_dir = Path(str(VOLUME_MOUNT)) / "inputs" / f"raw_{raw_id}" / "raw"
    model_root = Path(str(VOLUME_MOUNT)) / "inputs" / "model"
    model_manifest_path = model_root / "model_snapshot_manifest.json"
    if not model_manifest_path.exists():
        model_root.mkdir(parents=True, exist_ok=True)
        adapter_build.freeze_model_snapshot(model_root)
        vol.commit()
    model_manifest = json.loads(model_manifest_path.read_text())
    model_id = model_manifest["id"]

    out_dir = Path(str(VOLUME_MOUNT)) / "adapters" / f"pilot_{pilot_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = adapter_build.build_adapter_bundle(
        mc_dataset_path=raw_dir / "mc_dataset.json",
        val_dataset_path=raw_dir / "val_dataset.json",
        test_dataset_path=raw_dir / "test_dataset.json",
        calibration_path=raw_dir / "calibration.json",
        model_snapshot_dir=model_root / "snapshot",
        out_dir=out_dir, source_manifest_id=source_id, raw_input_bundle_id=raw_id,
        model_snapshot_id=model_id,
        producer_hashes={"adapter_build.py": sha256_file(Path(str(REMOTE_SRC)) / "scripts/stopdff_v5/adapter_build.py")},
    )
    vol.commit()
    return {"adapter_id": manifest["id"], "model_id": model_id,
            "fit_rows_sha256": manifest["identity"]["fit_rows_sha256"],
            "eval_rows_sha256": manifest["identity"]["eval_rows_sha256"],
            "out_dir": str(out_dir)}


@app.function(volumes={str(VOLUME_MOUNT): vol}, timeout=MAX_TIMEOUT, max_containers=1)
def fvi_study_remote(adapter_id: str) -> dict:
    _init_paths()
    import json
    from scripts.stopdff_v5 import fvi_study
    from scripts.stopdff_v5.checker import load_adapter_rows
    from scripts.stopdff_v5.identity import build_manifest, compute_id
    from scripts.stopdff_v5.manifests import fvi_study_identity

    vol.reload()
    adir = Path(str(VOLUME_MOUNT)) / "adapters" / f"canonical_{adapter_id}"
    rows = load_adapter_rows(adir)
    calibration = json.loads((adir / "calibration.json").read_text())
    study = fvi_study.run_fvi_study(rows=rows, calibration_json=calibration)
    ident = fvi_study_identity(
        adapter_bundle_id=adapter_id, candidate_grid=study["candidate_grid"],
        representative_generator=study["representative_cell_generator"],
        candidate_results=study["candidate_convergence_results"],
        strict_reference_results=study["strict_reference"], selector_rule=study["selector_rule"],
        selected_parameters=study["selected_parameters"], all96_validation=study["all96_fit_only_validation"],
        producer_hashes={},
    )
    man = build_manifest(ident)
    out = Path(str(VOLUME_MOUNT)) / "fvi" / man["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "fvi_study.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    (out / "fvi_study_execution.json").write_text(json.dumps(study, indent=2, sort_keys=True, default=str))
    vol.commit()
    return {"fvi_study_id": man["id"], "selected": study["selected_parameters"]}


@app.function(volumes={str(VOLUME_MOUNT): vol}, timeout=MAX_TIMEOUT, max_containers=1)
def sweep_remote(spec_json: str, adapter_id: str, resume: bool) -> dict:
    """CPU: run a smoke or final sweep with per-cell Volume commits."""
    _init_paths()
    import json
    from scripts.stopdff_v5 import profile, sweep
    from scripts.stopdff_v5.bootstrap import build_bootstrap_plan
    from scripts.stopdff_v5.checker import load_adapter_rows
    from scripts.stopdff_v5.identity import compute_id

    vol.reload()
    spec = json.loads(spec_json)
    adir = Path(str(VOLUME_MOUNT)) / "adapters" / f"canonical_{adapter_id}"
    rows = load_adapter_rows(adir)
    calibration = json.loads((adir / "calibration.json").read_text())

    variant = spec["profile_variant"]
    replicates = 1000 if variant == "final" else 100
    cells = profile.full_grid() if variant == "final" else profile.smoke_cells()
    # eval item set = paired test items
    mc = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "MC"}
    qa = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "QA"}
    plan = build_bootstrap_plan(sorted(mc & qa), replicates=replicates, seed=1)

    run_root = Path(str(VOLUME_MOUNT)) / "runs" / spec["run_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    ctx = sweep.SweepContext(
        rows=rows, calibration_json=calibration, run_spec=spec["run_spec_identity"],
        run_spec_id=spec["run_spec_id"], bootstrap_plan=plan, output_dir=run_root,
        fvi_tolerance=spec["fvi_tolerance"], fvi_max_iterations=spec["fvi_max_iterations"],
        backend="modal", profile_variant=variant, adapter_fit_rows_sha256=spec["fit_rows_sha256"],
        adapter_eval_rows_sha256=spec["eval_rows_sha256"], myopic_artifact_sha256=spec["myopic_sha256"],
        producer_hashes=spec.get("producer_hashes", {}), cells=cells,
        commit_fn=lambda: vol.commit(),
        environment=spec.get("environment", {}), resource_summary=spec.get("resource_summary", {}),
        attempt={"attempt": spec.get("attempt", 1), "mode": "resume" if resume else "fresh",
                 "command": ["dp_sweep"] + (["--resume"] if resume else []),
                 "run_spec_id": spec["run_spec_id"], "adapter_id": adapter_id},
    )
    aggregate = sweep.run_sweep(ctx)
    vol.commit()
    return {"run_id": spec["run_id"], "requested": aggregate["requested"],
            "completed": aggregate["completed"], "failed": aggregate["failed"],
            "release_status": aggregate["release_status"],
            "family": aggregate.get("family")}


@app.function(volumes={str(VOLUME_MOUNT): vol}, timeout=MAX_TIMEOUT, max_containers=1)
def validate_remote(run_id: str, adapter_id: str, require_final: bool, require_package: bool) -> dict:
    _init_paths()
    from scripts.stopdff_v5 import checker
    vol.reload()
    run_root = Path(str(VOLUME_MOUNT)) / "runs" / run_id
    adir = Path(str(VOLUME_MOUNT)) / "adapters" / f"canonical_{adapter_id}"
    res = checker.validate_run(run_root, backend="modal", adapter_bundle=adir,
                              require_final_profile=require_final, require_package=require_package)
    return {"passed": res.passed, "errors": res.errors[:40], "recomputed": res.recomputed}


@app.function(volumes={str(VOLUME_MOUNT): vol}, timeout=MAX_TIMEOUT, max_containers=1)
def package_remote(run_id: str) -> dict:
    _init_paths()
    import json
    from scripts.stopdff_v5 import writers
    vol.reload()
    run_root = Path(str(VOLUME_MOUNT)) / "runs" / run_id
    aggregate = json.loads((run_root / "aggregate.json").read_text())
    writers.package_run(run_root, aggregate, resource_summary={"backend": "modal"},
                        external_artifacts=[])
    vol.commit()
    return {"run_id": run_id, "packaged": True}


@app.function(volumes={str(VOLUME_MOUNT): vol}, timeout=7200, max_containers=1)
def mutation_gate_remote() -> dict:
    _init_paths()
    import tempfile
    from scripts.stopdff_v5 import selftest
    ok, results = selftest.run_self_test(Path(tempfile.mkdtemp()))
    return {"ok": ok, "n": len(results),
            "unexpected": [r["mutation"] for r in results if not r["ok"]]}


@app.function(volumes={str(VOLUME_MOUNT): vol}, timeout=1800, max_containers=1)
def durability_heartbeat(tag: str) -> dict:
    _init_paths()
    import json, time
    d = Path(str(VOLUME_MOUNT)) / "pilots" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "heartbeat.json").write_text(json.dumps({"tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    vol.commit()
    return {"tag": tag, "committed": True}


# --- local entrypoints ------------------------------------------------------------


@app.local_entrypoint()
def probe_main():
    print(probe.remote())
