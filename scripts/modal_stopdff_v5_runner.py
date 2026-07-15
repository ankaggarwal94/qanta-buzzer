#!/usr/bin/env python3
"""Modal standalone runner for the StopDFF v5 evidentiary pipeline (remote functions).

Source-only image (STOPDFF_V5_SOURCE_DIR = git-archive snapshot). One Volume
`cs321m-stopdff-artifacts` mounted at /stopdff. One writer per run dir, max_containers=1,
explicit Volume commits (per cell in the sweep), reload before resume. GPU (L40S) is used
only by build_adapter; every other stage is CPU. Orchestration lives in the control-plane
driver; these are the individual stage functions.

Volume layout:
  /stopdff/inputs/source_<id>/{source/, source_manifest.json}
  /stopdff/inputs/raw_<id>/{<role files>, raw_input_manifest.json}
  /stopdff/inputs/model/{snapshot/, model_snapshot_manifest.json}
  /stopdff/adapters/<subdir>/{fit_rows.jsonl.gz, eval_rows.jsonl.gz, calibration.json, manifest.json}
  /stopdff/fvi/<id>/{fvi_study.json, fvi_study_execution.json}
  /stopdff/bootstrap/<id>/bootstrap_plan.json
  /stopdff/runs/<run_id>/...
"""
from __future__ import annotations

import os
from pathlib import PurePosixPath

import modal

APP_NAME = "cs321m-stopdff-v5"
VOLUME_NAME = "cs321m-stopdff-artifacts"
MNT = "/stopdff"
REMOTE_SRC = "/root/src"
DAY = 86400
SOURCE_DIR = os.environ.get("STOPDFF_V5_SOURCE_DIR", "")

_PIP = [
    "numpy>=1.26,<3", "scipy>=1.11", "scikit-learn>=1.3", "pandas>=2.1",
    "matplotlib>=3.7", "sentence-transformers>=2.7", "huggingface_hub>=0.23",
]
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_PIP)
    .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "HF_HUB_DISABLE_TELEMETRY": "1",
          "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": REMOTE_SRC})
)
if SOURCE_DIR:
    _image = _image.add_local_dir(SOURCE_DIR, remote_path=REMOTE_SRC, copy=True)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=_image)


def _p(*parts) -> str:
    return str(PurePosixPath(MNT, *parts))


@app.function(volumes={MNT: vol}, timeout=1800, max_containers=1)
def probe() -> dict:
    import platform
    import numpy, scipy, sklearn, pandas, sentence_transformers, transformers, huggingface_hub
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pandas": pandas.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__, "huggingface_hub": huggingface_hub.__version__}


@app.function(volumes={MNT: vol}, timeout=3600, max_containers=1)
def verify_volume_artifact(rel_dir: str, kind: str) -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5.identity import compute_id, sha256_file
    vol.reload()
    base = Path(_p(rel_dir))
    if kind == "raw":
        manifest = json.loads((base / "raw_input_manifest.json").read_text())
        sub, key, name = "", "files", "role"
    elif kind == "source":
        manifest = json.loads((base / "source_manifest.json").read_text())
        sub, key, name = "source", "files", "path"
    elif kind == "model":
        manifest = json.loads((base / "model_snapshot_manifest.json").read_text())
        sub, key, name = "snapshot", "files", "path"
    else:
        return {"ok": False, "error": f"unknown kind {kind}"}
    ident = manifest["identity"]
    if compute_id(ident) != manifest["id"]:
        return {"ok": False, "error": "manifest id mismatch", "id": manifest["id"]}
    mism = []
    for f in ident[key]:
        p = base / sub / f[name] if sub else base / f[name]
        if not p.is_file() or sha256_file(p) != f["sha256"]:
            mism.append(f[name])
    return {"ok": not mism, "id": manifest["id"], "mismatches": mism[:20], "n_files": len(ident[key])}


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1, memory=8192)
def freeze_model() -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5 import adapter_build
    vol.reload()
    root = Path(_p("inputs", "model"))
    mpath = root / "model_snapshot_manifest.json"
    if mpath.exists():
        return {"model_id": json.loads(mpath.read_text())["id"], "cached": True}
    root.mkdir(parents=True, exist_ok=True)
    man = adapter_build.freeze_model_snapshot(root)
    vol.commit()
    return {"model_id": man["id"], "revision": man["identity"]["model_revision"], "cached": False}


@app.function(volumes={MNT: vol}, gpu="L40S", timeout=DAY, max_containers=1, memory=32768)
def build_adapter(dest_subdir: str, source_id: str, raw_id: str, model_id: str) -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5 import adapter_build
    from scripts.stopdff_v5.identity import sha256_file
    vol.reload()
    raw = Path(_p("inputs", f"raw_{raw_id}"))
    model_dir = Path(_p("inputs", "model", "snapshot"))
    out = Path(_p("adapters", dest_subdir))
    if out.exists():
        import shutil
        shutil.rmtree(out)
    man = adapter_build.build_adapter_bundle(
        mc_dataset_path=raw / "mc_dataset.json", val_dataset_path=raw / "val_dataset.json",
        test_dataset_path=raw / "test_dataset.json", calibration_path=raw / "calibration.json",
        model_snapshot_dir=model_dir, out_dir=out, source_manifest_id=source_id,
        raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        producer_hashes={"adapter_build.py": sha256_file(Path(REMOTE_SRC) / "scripts/stopdff_v5/adapter_build.py")},
    )
    vol.commit()
    return {"adapter_id": man["id"], "fit_rows_sha256": man["identity"]["fit_rows_sha256"],
            "eval_rows_sha256": man["identity"]["eval_rows_sha256"], "subdir": dest_subdir}


@app.function(volumes={MNT: vol}, timeout=3600, max_containers=1)
def promote_adapter(from_subdir: str, adapter_id: str) -> dict:
    import shutil
    from pathlib import Path
    vol.reload()
    src = Path(_p("adapters", from_subdir))
    dst = Path(_p("adapters", f"canonical_{adapter_id}"))
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    vol.commit()
    return {"canonical_subdir": f"canonical_{adapter_id}"}


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1, memory=16384)
def fvi_study(adapter_id: str) -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5 import fvi_study as fs
    from scripts.stopdff_v5.checker import load_adapter_rows
    from scripts.stopdff_v5.identity import build_manifest
    from scripts.stopdff_v5.manifests import fvi_study_identity
    vol.reload()
    adir = Path(_p("adapters", f"canonical_{adapter_id}"))
    rows = load_adapter_rows(adir)
    calibration = json.loads((adir / "calibration.json").read_text())
    study = fs.run_fvi_study(rows=rows, calibration_json=calibration)
    ident = fvi_study_identity(
        adapter_bundle_id=adapter_id, candidate_grid=study["candidate_grid"],
        representative_generator=study["representative_cell_generator"],
        candidate_results=study["candidate_convergence_results"],
        strict_reference_results=study["strict_reference"], selector_rule=study["selector_rule"],
        selected_parameters=study["selected_parameters"], all96_validation=study["all96_fit_only_validation"],
        producer_hashes={})
    man = build_manifest(ident)
    out = Path(_p("fvi", man["id"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "fvi_study.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    (out / "fvi_study_execution.json").write_text(json.dumps(study, indent=2, sort_keys=True, default=str))
    vol.commit()
    return {"fvi_study_id": man["id"], "selected": study["selected_parameters"]}


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1)
def bootstrap_plan(adapter_id: str, replicates: int) -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5.bootstrap import build_bootstrap_plan, plan_identity
    from scripts.stopdff_v5.checker import load_adapter_rows
    from scripts.stopdff_v5.identity import compute_id
    vol.reload()
    adir = Path(_p("adapters", f"canonical_{adapter_id}"))
    rows = load_adapter_rows(adir)
    mc = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "MC"}
    qa = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "QA"}
    plan = build_bootstrap_plan(sorted(mc & qa), replicates=int(replicates), seed=1)
    ident = plan_identity(plan)
    pid = compute_id(ident)
    out = Path(_p("bootstrap", pid))
    out.mkdir(parents=True, exist_ok=True)
    (out / "bootstrap_plan.json").write_text(
        json.dumps({"id": pid, "identity": ident, "item_ids": plan.item_ids}, indent=2, sort_keys=True))
    vol.commit()
    return {"bootstrap_plan_id": pid, "replicates": int(replicates), "n_items": plan.n_items}


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1, memory=16384)
def run_sweep(spec_json: str, adapter_id: str, bootstrap_plan_id: str, resume: bool) -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5 import profile, sweep
    from scripts.stopdff_v5.bootstrap import build_bootstrap_plan
    from scripts.stopdff_v5.checker import load_adapter_rows
    vol.reload()
    spec = json.loads(spec_json)
    adir = Path(_p("adapters", f"canonical_{adapter_id}"))
    rows = load_adapter_rows(adir)
    calibration = json.loads((adir / "calibration.json").read_text())
    plan_manifest = json.loads((Path(_p("bootstrap", bootstrap_plan_id)) / "bootstrap_plan.json").read_text())
    replicates = int(plan_manifest["identity"]["replicate_count"])
    plan = build_bootstrap_plan(plan_manifest["item_ids"], replicates=replicates, seed=1)
    variant = spec["profile_variant"]
    cells = profile.full_grid() if variant == "final" else profile.smoke_cells()
    run_root = Path(_p("runs", spec["run_id"]))
    run_root.mkdir(parents=True, exist_ok=True)
    ctx = sweep.SweepContext(
        rows=rows, calibration_json=calibration, run_spec=spec["run_spec_identity"],
        run_spec_id=spec["run_spec_id"], bootstrap_plan=plan, output_dir=run_root,
        fvi_tolerance=spec["fvi_tolerance"], fvi_max_iterations=spec["fvi_max_iterations"],
        backend="modal", profile_variant=variant, adapter_fit_rows_sha256=spec["fit_rows_sha256"],
        adapter_eval_rows_sha256=spec["eval_rows_sha256"], myopic_artifact_sha256=spec["myopic_sha256"],
        producer_hashes=spec.get("producer_hashes", {}), cells=cells, commit_fn=lambda: vol.commit(),
        environment=spec.get("environment", {}), resource_summary=spec.get("resource_summary", {}),
        attempt={"attempt": spec.get("attempt", 1), "mode": "resume" if resume else "fresh",
                 "command": ["dp_sweep"] + (["--resume"] if resume else []),
                 "run_spec_id": spec["run_spec_id"], "adapter_id": adapter_id})
    agg = sweep.run_sweep(ctx)
    vol.commit()
    return {"run_id": spec["run_id"], "requested": agg["requested"], "completed": agg["completed"],
            "skipped": agg["skipped"], "failed": agg["failed"], "release_status": agg["release_status"],
            "family": agg.get("family")}


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1)
def package(run_id: str) -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5 import writers
    vol.reload()
    run_root = Path(_p("runs", run_id))
    agg = json.loads((run_root / "aggregate.json").read_text())
    writers.package_run(run_root, agg, resource_summary={"backend": "modal"}, external_artifacts=[])
    vol.commit()
    return {"run_id": run_id, "packaged": True}


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1)
def validate(run_id: str, adapter_id: str, require_final: bool, require_package: bool) -> dict:
    from pathlib import Path
    from scripts.stopdff_v5 import checker
    vol.reload()
    res = checker.validate_run(Path(_p("runs", run_id)), backend="modal",
                              adapter_bundle=Path(_p("adapters", f"canonical_{adapter_id}")),
                              require_final_profile=require_final, require_package=require_package)
    return {"passed": res.passed, "errors": res.errors[:50], "recomputed": res.recomputed}


@app.function(volumes={MNT: vol}, timeout=7200, max_containers=1)
def mutation_gate() -> dict:
    import tempfile
    from pathlib import Path
    from scripts.stopdff_v5 import selftest
    ok, results = selftest.run_self_test(Path(tempfile.mkdtemp()))
    return {"ok": ok, "n": len(results), "unexpected": [r["mutation"] for r in results if not r["ok"]]}


@app.function(volumes={MNT: vol}, timeout=1800, max_containers=1)
def durability_heartbeat(tag: str) -> dict:
    import json, time
    from pathlib import Path
    d = Path(_p("pilots", tag))
    d.mkdir(parents=True, exist_ok=True)
    hb = {"tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (d / "heartbeat.json").write_text(json.dumps(hb))
    vol.commit()
    return {"tag": tag, "committed": True}


@app.local_entrypoint()
def probe_main():
    print(probe.remote())
