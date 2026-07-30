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
if not SOURCE_DIR:
    raise RuntimeError(
        "STOPDFF_V5_SOURCE_DIR must point to the frozen git-archive source tree"
    )

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
_image = _image.add_local_dir(SOURCE_DIR, remote_path=REMOTE_SRC, copy=True)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=_image)


def _p(*parts) -> str:
    path = PurePosixPath(MNT)
    for part in parts:
        candidate = PurePosixPath(str(part))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe volume path component: {part!r}")
        path /= candidate
    return str(path)


def _receipt_rel(gate: str, receipt_id: str) -> str:
    if gate not in {"smoke", "mutation", "determinism"}:
        raise ValueError(f"unknown prerequisite receipt gate: {gate}")
    if (
        not isinstance(receipt_id, str)
        or len(receipt_id) != 64
        or any(ch not in "0123456789abcdef" for ch in receipt_id)
    ):
        raise ValueError(f"invalid {gate} prerequisite receipt id")
    return _p("receipts", gate, f"{receipt_id}.json")


def _verified_content_manifest(
    base,
    *,
    manifest_name: str,
    expected_id: str,
    file_key: str,
    name_key: str,
    content_subdir: str = "",
) -> dict:
    """Verify a content-addressed manifest and every file it declares."""
    from pathlib import Path

    from scripts.stopdff_v5.identity import (
        compute_id,
        loads_no_duplicate_keys,
        sha256_file,
    )

    base = Path(base)
    manifest_path = base / manifest_name
    manifest = loads_no_duplicate_keys(
        manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("identity"),
        dict,
    ):
        raise ValueError(f"{manifest_name} is not a manifest object")
    if compute_id(manifest["identity"]) != manifest.get("id"):
        raise ValueError(f"{manifest_name} id mismatch")
    if manifest["id"] != expected_id:
        raise ValueError(
            f"{manifest_name} id {manifest['id']} != expected {expected_id}"
        )
    entries = manifest["identity"].get(file_key)
    if not isinstance(entries, list):
        raise ValueError(f"{manifest_name} lacks {file_key}")
    seen: set[str] = set()
    content_root = base / content_subdir if content_subdir else base
    for entry in entries:
        name = entry.get(name_key) if isinstance(entry, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or ".." in PurePosixPath(name).parts
            or name in seen
        ):
            raise ValueError(f"{manifest_name} contains unsafe/duplicate path")
        seen.add(name)
        path = content_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{manifest_name} file missing: {name}")
        if int(entry.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"{manifest_name} size mismatch: {name}")
        if entry.get("sha256") != sha256_file(path):
            raise ValueError(f"{manifest_name} sha mismatch: {name}")
    return manifest


@app.function(volumes={MNT: vol}, timeout=1800, max_containers=1)
def probe() -> dict:
    import platform
    import numpy, scipy, sklearn, pandas, sentence_transformers, transformers, huggingface_hub
    package_versions = {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pandas.__version__,
        "sentence-transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
    }
    return {
        "python": platform.python_version(),
        "package_versions": package_versions,
    }


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
        cached = json.loads(mpath.read_text())
        _verified_content_manifest(
            root,
            manifest_name="model_snapshot_manifest.json",
            expected_id=cached["id"],
            file_key="files",
            name_key="path",
            content_subdir="snapshot",
        )
        return {"model_id": cached["id"], "cached": True}
    root.mkdir(parents=True, exist_ok=True)
    man = adapter_build.freeze_model_snapshot(root)
    _verified_content_manifest(
        root,
        manifest_name="model_snapshot_manifest.json",
        expected_id=man["id"],
        file_key="files",
        name_key="path",
        content_subdir="snapshot",
    )
    vol.commit()
    return {"model_id": man["id"], "revision": man["identity"]["model_revision"], "cached": False}


@app.function(volumes={MNT: vol}, gpu="L40S", timeout=DAY, max_containers=1, memory=32768)
def build_adapter(dest_subdir: str, source_id: str, raw_id: str, model_id: str) -> dict:
    from pathlib import Path
    from scripts.stopdff_v5 import adapter_build
    from scripts.stopdff_v5 import checker
    from scripts.stopdff_v5.identity import sha256_file
    vol.reload()
    source_root = Path(_p("inputs", f"source_{source_id}"))
    raw = Path(_p("inputs", f"raw_{raw_id}"))
    model_root = Path(_p("inputs", "model"))
    model_dir = model_root / "snapshot"
    source_manifest = _verified_content_manifest(
        source_root,
        manifest_name="source_manifest.json",
        expected_id=source_id,
        file_key="files",
        name_key="path",
        content_subdir="source",
    )
    _verified_content_manifest(
        raw,
        manifest_name="raw_input_manifest.json",
        expected_id=raw_id,
        file_key="files",
        name_key="role",
    )
    _verified_content_manifest(
        model_root,
        manifest_name="model_snapshot_manifest.json",
        expected_id=model_id,
        file_key="files",
        name_key="path",
        content_subdir="snapshot",
    )
    # The bytes imported by this function must be the frozen source bytes.
    runtime_source = Path(REMOTE_SRC)
    for entry in source_manifest["identity"]["files"]:
        runtime_path = runtime_source / entry["path"]
        if (
            runtime_path.is_symlink()
            or not runtime_path.is_file()
            or sha256_file(runtime_path) != entry["sha256"]
        ):
            raise ValueError(
                f"executing source does not match source manifest: {entry['path']}"
            )

    out = Path(_p("adapters", dest_subdir))
    if out.exists():
        raise FileExistsError(f"adapter destination already exists: {dest_subdir}")
    man = adapter_build.build_adapter_bundle(
        mc_dataset_path=raw / "mc_dataset.json", val_dataset_path=raw / "val_dataset.json",
        test_dataset_path=raw / "test_dataset.json", calibration_path=raw / "calibration.json",
        model_snapshot_dir=model_dir, out_dir=out, source_manifest_id=source_id,
        raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        producer_hashes={"adapter_build.py": sha256_file(Path(REMOTE_SRC) / "scripts/stopdff_v5/adapter_build.py")},
    )
    result = checker.validate_adapter(out)
    if not result.passed or result.recomputed.get("adapter_bundle_id") != man["id"]:
        raise ValueError(
            "new adapter failed validation: " + "; ".join(result.errors)
        )
    vol.commit()
    return {"adapter_id": man["id"], "fit_rows_sha256": man["identity"]["fit_rows_sha256"],
            "eval_rows_sha256": man["identity"]["eval_rows_sha256"], "subdir": dest_subdir}


@app.function(volumes={MNT: vol}, timeout=3600, max_containers=1)
def promote_adapter(from_subdir: str, adapter_id: str) -> dict:
    import shutil
    from pathlib import Path
    from scripts.stopdff_v5 import checker
    vol.reload()
    src = Path(_p("adapters", from_subdir))
    result = checker.validate_adapter(src)
    if not result.passed:
        raise ValueError(
            "adapter promotion source failed validation: "
            + "; ".join(result.errors)
        )
    if result.recomputed.get("adapter_bundle_id") != adapter_id:
        raise ValueError("adapter promotion ID does not match source manifest")
    dst = Path(_p("adapters", f"canonical_{adapter_id}"))
    if dst.exists():
        existing = checker.validate_adapter(dst)
        if (
            existing.passed
            and existing.recomputed.get("adapter_bundle_id") == adapter_id
        ):
            return {
                "canonical_subdir": f"canonical_{adapter_id}",
                "cached": True,
            }
        raise FileExistsError("canonical adapter destination exists but is invalid")
    shutil.copytree(src, dst)
    copied = checker.validate_adapter(dst)
    if not copied.passed:
        raise ValueError(
            "copied adapter failed validation: " + "; ".join(copied.errors)
        )
    vol.commit()
    return {"canonical_subdir": f"canonical_{adapter_id}", "cached": False}


@app.function(volumes={MNT: vol}, timeout=3600, max_containers=1)
def adapter_determinism_receipt(
    first_subdir: str,
    second_subdir: str,
    binding_json: str,
) -> dict:
    """Validate two independent adapter builds and persist their success receipt."""
    import json
    from pathlib import Path
    from scripts.stopdff_v5 import checker, sweep, writers
    from scripts.stopdff_v5.identity import (
        canonical_bytes,
        sha256_bytes,
        sha256_file,
    )

    vol.reload()
    bindings = json.loads(binding_json)
    first = Path(_p("adapters", first_subdir))
    second = Path(_p("adapters", second_subdir))
    first_result = checker.validate_adapter(first)
    second_result = checker.validate_adapter(second)
    adapter_id = bindings.get("adapter_bundle_id")
    if (
        not first_result.passed
        or not second_result.passed
        or first_result.recomputed.get("adapter_bundle_id") != adapter_id
        or second_result.recomputed.get("adapter_bundle_id") != adapter_id
    ):
        raise ValueError("adapter determinism inputs are invalid or unbound")
    compared = ("fit_rows.jsonl.gz", "eval_rows.jsonl.gz", "calibration.json")
    first_hashes = {name: sha256_file(first / name) for name in compared}
    second_hashes = {name: sha256_file(second / name) for name in compared}
    if first_hashes != second_hashes:
        raise ValueError("independent adapter builds are not byte-identical")
    receipt = writers.build_prerequisite_receipt(
        gate="determinism",
        bindings=bindings,
        evidence={
            "bundle_files_sha256": sha256_bytes(
                canonical_bytes(first_hashes)
            )
        },
    )
    sweep._write_bound_json(
        Path(_receipt_rel("determinism", receipt["id"])),
        receipt,
        resume=True,
    )
    vol.commit()
    return {
        "ok": True,
        "adapter_id": adapter_id,
        "prerequisite_receipt_id": receipt["id"],
    }


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1, memory=16384)
def fvi_study(adapter_id: str) -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5 import fvi_study as fs
    from scripts.stopdff_v5 import checker
    from scripts.stopdff_v5.checker import load_adapter_rows
    from scripts.stopdff_v5.identity import build_manifest
    from scripts.stopdff_v5.manifests import fvi_study_identity
    vol.reload()
    adir = Path(_p("adapters", f"canonical_{adapter_id}"))
    adapter_result = checker.validate_adapter(adir)
    if (
        not adapter_result.passed
        or adapter_result.recomputed.get("adapter_bundle_id") != adapter_id
    ):
        raise ValueError(
            "FVI adapter failed validation: "
            + "; ".join(adapter_result.errors)
        )
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
    if out.exists():
        existing = checker.load_json(out / "fvi_study.json")
        if existing.get("id") == man["id"] and existing.get("identity") == ident:
            return {
                "fvi_study_id": man["id"],
                "selected": study["selected_parameters"],
                "cached": True,
            }
        raise FileExistsError("FVI study destination exists with different content")
    out.mkdir(parents=True)
    (out / "fvi_study.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    (out / "fvi_study_execution.json").write_text(json.dumps(study, indent=2, sort_keys=True, default=str))
    vol.commit()
    return {
        "fvi_study_id": man["id"],
        "selected": study["selected_parameters"],
        "cached": False,
    }


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1)
def bootstrap_plan(adapter_id: str, replicates: int) -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5.bootstrap import build_bootstrap_plan, plan_identity
    from scripts.stopdff_v5 import checker
    from scripts.stopdff_v5.checker import load_adapter_rows
    from scripts.stopdff_v5.identity import compute_id
    vol.reload()
    adir = Path(_p("adapters", f"canonical_{adapter_id}"))
    adapter_result = checker.validate_adapter(adir)
    if (
        not adapter_result.passed
        or adapter_result.recomputed.get("adapter_bundle_id") != adapter_id
    ):
        raise ValueError(
            "bootstrap adapter failed validation: "
            + "; ".join(adapter_result.errors)
        )
    rows = load_adapter_rows(adir)
    mc = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "MC"}
    qa = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "QA"}
    plan = build_bootstrap_plan(sorted(mc & qa), replicates=int(replicates), seed=1)
    ident = plan_identity(plan)
    pid = compute_id(ident)
    out = Path(_p("bootstrap", pid))
    manifest = {"id": pid, "identity": ident, "item_ids": plan.item_ids}
    if out.exists():
        existing = checker.load_json(out / "bootstrap_plan.json")
        if existing == manifest:
            return {
                "bootstrap_plan_id": pid,
                "replicates": int(replicates),
                "n_items": plan.n_items,
                "cached": True,
            }
        raise FileExistsError("bootstrap destination exists with different content")
    out.mkdir(parents=True)
    (out / "bootstrap_plan.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    vol.commit()
    return {
        "bootstrap_plan_id": pid,
        "replicates": int(replicates),
        "n_items": plan.n_items,
        "cached": False,
    }


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1, memory=16384)
def run_sweep(spec_json: str, adapter_id: str, bootstrap_plan_id: str, resume: bool) -> dict:
    import importlib.metadata as im
    import platform
    from pathlib import Path
    from scripts.stopdff_v5 import checker, profile, sweep, writers
    from scripts.stopdff_v5.identity import (
        compute_id,
        loads_no_duplicate_keys,
        sha256_file,
    )
    from scripts.stopdff_v5.manifests import environment_contract_identity
    vol.reload()
    spec = loads_no_duplicate_keys(spec_json)
    if not isinstance(spec, dict):
        raise ValueError("sweep wrapper must be an object")
    run_spec_manifest = {
        "id": spec.get("run_spec_id"),
        "identity": spec.get("run_spec_identity"),
    }
    adir = Path(_p("adapters", f"canonical_{adapter_id}"))
    plan_path = Path(_p("bootstrap", bootstrap_plan_id)) / "bootstrap_plan.json"
    plan_manifest = checker.load_json(plan_path)
    binding = checker.resolve_run_binding(
        run_spec_manifest=run_spec_manifest,
        adapter_bundle=adir,
        bootstrap_plan_manifest=plan_manifest,
    )
    if binding["adapter_bundle_id"] != adapter_id:
        raise ValueError("adapter argument does not match verified run spec")
    if binding["bootstrap_plan_id"] != bootstrap_plan_id:
        raise ValueError("bootstrap argument does not match verified run spec")

    # Reject contradictory duplicated wrapper fields instead of trusting them.
    wrapper_bindings = {
        "profile_variant": binding["variant"],
        "fvi_tolerance": binding["fvi_tolerance"],
        "fvi_max_iterations": binding["fvi_max_iterations"],
        "fit_rows_sha256": binding["fit_rows_sha256"],
        "eval_rows_sha256": binding["eval_rows_sha256"],
    }
    for field, expected in wrapper_bindings.items():
        if field in spec and spec[field] != expected:
            raise ValueError(
                f"sweep wrapper {field} does not match verified manifests"
            )

    spec_ids = binding["spec_ids"]
    source_id = spec_ids["source_manifest_id"]
    raw_id = spec_ids["raw_input_bundle_id"]
    source_root = Path(_p("inputs", f"source_{source_id}"))
    raw_root = Path(_p("inputs", f"raw_{raw_id}"))
    source_manifest = _verified_content_manifest(
        source_root,
        manifest_name="source_manifest.json",
        expected_id=source_id,
        file_key="files",
        name_key="path",
        content_subdir="source",
    )
    raw_manifest = _verified_content_manifest(
        raw_root,
        manifest_name="raw_input_manifest.json",
        expected_id=raw_id,
        file_key="files",
        name_key="role",
    )
    for entry in source_manifest["identity"]["files"]:
        runtime_path = Path(REMOTE_SRC) / entry["path"]
        if (
            runtime_path.is_symlink()
            or not runtime_path.is_file()
            or sha256_file(runtime_path) != entry["sha256"]
        ):
            raise ValueError(
                f"executing source does not match source manifest: {entry['path']}"
            )

    raw_files = {
        entry["role"]: entry
        for entry in raw_manifest["identity"]["files"]
    }
    myopic_entry = raw_files.get("stopdff.json")
    if myopic_entry is None:
        raise ValueError("raw-input manifest lacks stopdff.json")
    myopic_sha256 = myopic_entry["sha256"]

    fvi_id = spec_ids["fvi_study_id"]
    fvi_manifest = checker.load_json(
        Path(_p("fvi", fvi_id)) / "fvi_study.json"
    )
    if (
        compute_id(fvi_manifest.get("identity", {})) != fvi_manifest.get("id")
        or fvi_manifest.get("id") != fvi_id
        or fvi_manifest.get("identity", {}).get("adapter_bundle_id")
        != adapter_id
        or fvi_manifest.get("identity", {}).get("selected_parameters")
        != {
            "tolerance": binding["fvi_tolerance"],
            "max_iterations": binding["fvi_max_iterations"],
        }
    ):
        raise ValueError("FVI study does not match the verified run spec")

    package_names = (
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "sentence-transformers",
        "transformers",
        "huggingface_hub",
    )
    package_versions = {
        name: im.version(name)
        for name in package_names
    }
    environment_identity = environment_contract_identity(
        python_version=platform.python_version(),
        package_versions=package_versions,
    )
    if compute_id(environment_identity) != spec_ids["environment_contract_id"]:
        raise ValueError("remote environment does not match run spec")

    evidence_roots = binding["run_spec_identity"].get("evidence_roots")
    if not isinstance(evidence_roots, dict):
        raise ValueError("run spec lacks evidence_roots")
    if evidence_roots.get("myopic_artifact_sha256") != myopic_sha256:
        raise ValueError("run spec myopic artifact does not match raw inputs")
    runtime_producer_hashes = {
        "sweep.py": sha256_file(
            Path(REMOTE_SRC) / "scripts/stopdff_v5/sweep.py"
        ),
        "checker.py": sha256_file(
            Path(REMOTE_SRC) / "scripts/stopdff_v5/checker.py"
        ),
    }
    if evidence_roots.get("producer_hashes") != runtime_producer_hashes:
        raise ValueError("run spec producer hashes do not match executing source")
    receipt_ids = evidence_roots.get("prerequisite_receipts")
    if not isinstance(receipt_ids, dict):
        raise ValueError("run spec prerequisite_receipts must be an object")
    receipts = {}
    for gate, receipt_id in receipt_ids.items():
        receipt_path = Path(_receipt_rel(gate, receipt_id))
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError(f"{gate} prerequisite receipt is unavailable")
        receipts[gate] = loads_no_duplicate_keys(
            receipt_path.read_text(encoding="utf-8")
        )
    writers.validate_prerequisite_receipts(
        profile_variant=binding["variant"],
        identity_bindings={
            key: spec_ids[key]
            for key in (
                "source_manifest_id",
                "raw_input_bundle_id",
                "model_snapshot_id",
                "adapter_bundle_id",
                "fvi_study_id",
                "environment_contract_id",
            )
        },
        receipt_ids=receipt_ids,
        receipts=receipts,
    )

    rows = binding["rows"]
    calibration = binding["calibration"]
    plan = binding["bootstrap_plan"]
    variant = binding["variant"]
    cells = profile.full_grid() if variant == "final" else profile.smoke_cells()
    run_root = Path(_p("runs", spec["run_id"]))
    if binding["run_spec_id"][:12] not in str(spec["run_id"]):
        raise ValueError("run_id is not bound to run_spec_id")
    if run_root.exists() and not resume:
        raise FileExistsError("fresh sweep destination already exists")
    if resume:
        if not run_root.is_dir():
            raise FileNotFoundError("resume destination does not exist")
        existing_spec = run_root / "run_spec.json"
        if existing_spec.exists():
            existing = checker.load_json(existing_spec)
            if existing != run_spec_manifest:
                raise ValueError("resume destination is bound to another run spec")
    run_root.mkdir(parents=True, exist_ok=resume)
    ctx = sweep.SweepContext(
        rows=rows, calibration_json=calibration,
        run_spec=binding["run_spec_identity"],
        run_spec_id=binding["run_spec_id"], bootstrap_plan=plan,
        output_dir=run_root, fvi_tolerance=binding["fvi_tolerance"],
        fvi_max_iterations=binding["fvi_max_iterations"],
        backend="modal", profile_variant=variant,
        adapter_bundle_id=adapter_id,
        adapter_fit_rows_sha256=binding["fit_rows_sha256"],
        adapter_eval_rows_sha256=binding["eval_rows_sha256"],
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=runtime_producer_hashes,
        cells=cells, commit_fn=lambda: vol.commit(),
        environment={
            "python_version": platform.python_version(),
            "package_versions": package_versions,
        },
        resource_summary=spec.get("resource_summary", {}),
        attempt={"attempt": spec.get("attempt", 1), "mode": "resume" if resume else "fresh",
                 "command": ["dp_sweep"] + (["--resume"] if resume else []),
                 "run_spec_id": binding["run_spec_id"], "adapter_id": adapter_id,
                 "bootstrap_plan_id": bootstrap_plan_id},
        resume=resume)
    agg = sweep.run_sweep(ctx)
    vol.commit()
    result = {"run_id": spec["run_id"], "requested": agg["requested"], "completed": agg["completed"],
              "skipped": agg["skipped"], "failed": agg["failed"], "release_status": agg["release_status"],
              "family": agg.get("family")}
    if variant == "smoke":
        validation = checker.validate_run(
            run_root,
            backend="modal",
            adapter_bundle=adir,
            require_final_profile=False,
            require_package=False,
        )
        if not validation.passed or agg["release_status"] != "VALID":
            raise ValueError("smoke run cannot issue a success receipt")
        receipt = writers.build_prerequisite_receipt(
            gate="smoke",
            bindings={
                key: spec_ids[key]
                for key in (
                    "source_manifest_id",
                    "raw_input_bundle_id",
                    "model_snapshot_id",
                    "adapter_bundle_id",
                    "fvi_study_id",
                    "environment_contract_id",
                )
            },
            evidence={
                "run_id": spec["run_id"],
                "aggregate_sha256": sha256_file(
                    run_root / "aggregate.json"
                ),
            },
        )
        receipt_path = Path(_receipt_rel("smoke", receipt["id"]))
        sweep._write_bound_json(receipt_path, receipt, resume=True)
        vol.commit()
        result["prerequisite_receipt_id"] = receipt["id"]
    return result


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1)
def package(run_id: str) -> dict:
    from pathlib import Path
    from scripts.stopdff_v5 import checker, writers
    from scripts.stopdff_v5.identity import (
        build_manifest,
        sha256_file,
    )
    from scripts.stopdff_v5.manifests import environment_contract_identity
    vol.reload()
    run_root = Path(_p("runs", run_id))
    agg = checker.load_json(run_root / "aggregate.json")
    adapter_id = agg.get("adapter_bundle_id")
    if not isinstance(adapter_id, str):
        raise ValueError("aggregate lacks adapter_bundle_id")
    validation = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=Path(_p("adapters", f"canonical_{adapter_id}")),
        require_final_profile=agg.get("profile_variant") == "final",
        require_package=False,
    )
    if not validation.passed:
        raise ValueError(
            "run failed validation before packaging: "
            + "; ".join(validation.errors)
        )
    spec_manifest = checker.load_json(run_root / "run_spec.json")
    spec_ids = spec_manifest["identity"]["identity"]
    source_path = Path(_p(
        "inputs",
        f"source_{spec_ids['source_manifest_id']}",
        "source_manifest.json",
    ))
    raw_path = Path(_p(
        "inputs",
        f"raw_{spec_ids['raw_input_bundle_id']}",
        "raw_input_manifest.json",
    ))
    model_path = Path(_p(
        "inputs",
        "model",
        "model_snapshot_manifest.json",
    ))
    fvi_path = Path(_p(
        "fvi",
        spec_ids["fvi_study_id"],
        "fvi_study.json",
    ))
    evidence_bytes = {
        "evidence/fvi_study.json": fvi_path.read_bytes(),
    }
    environment = checker.load_json(run_root / "environment.json")
    environment_manifest = build_manifest(environment_contract_identity(
        python_version=environment["python_version"],
        package_versions=environment["package_versions"],
    ))
    if environment_manifest["id"] != spec_ids["environment_contract_id"]:
        raise ValueError("packaged environment does not match run spec")
    environment_bytes = (
        __import__("json").dumps(
            environment_manifest,
            indent=2,
            sort_keys=True,
        ) + "\n"
    ).encode("utf-8")
    evidence_bytes["evidence/environment_contract.json"] = environment_bytes

    def ledger_entry(role, content_id, path, retrieval_path):
        return {
            "role": role,
            "content_id": content_id,
            "sha256": sha256_file(path),
            "byte_size": path.stat().st_size,
            "retrieval_path": retrieval_path,
        }

    external_artifacts = [
        ledger_entry(
            "source_manifest",
            spec_ids["source_manifest_id"],
            source_path,
            str(source_path),
        ),
        ledger_entry(
            "raw_input_manifest",
            spec_ids["raw_input_bundle_id"],
            raw_path,
            str(raw_path),
        ),
        ledger_entry(
            "model_snapshot_manifest",
            spec_ids["model_snapshot_id"],
            model_path,
            str(model_path),
        ),
        {
            "role": "fvi_study",
            "content_id": spec_ids["fvi_study_id"],
            "sha256": sha256_file(fvi_path),
            "byte_size": fvi_path.stat().st_size,
            "retrieval_path": "evidence/fvi_study.json",
        },
        {
            "role": "environment_contract",
            "content_id": environment_manifest["id"],
            "sha256": __import__("hashlib").sha256(
                environment_bytes
            ).hexdigest(),
            "byte_size": len(environment_bytes),
            "retrieval_path": "evidence/environment_contract.json",
        },
    ]
    writers.package_run(
        run_root,
        agg,
        resource_summary={"backend": "modal"},
        external_artifacts=external_artifacts,
        evidence_files=evidence_bytes,
    )
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
def mutation_gate(binding_json: str) -> dict:
    import json
    import tempfile
    from pathlib import Path
    from scripts.stopdff_v5 import selftest, sweep, writers
    from scripts.stopdff_v5.identity import canonical_bytes, sha256_bytes
    bindings = json.loads(binding_json)
    ok, results = selftest.run_self_test(Path(tempfile.mkdtemp()))
    unexpected = [r["mutation"] for r in results if not r["ok"]]
    if not ok:
        return {"ok": False, "n": len(results), "unexpected": unexpected}
    receipt = writers.build_prerequisite_receipt(
        gate="mutation",
        bindings=bindings,
        evidence={"result_sha256": sha256_bytes(canonical_bytes(results))},
    )
    sweep._write_bound_json(
        Path(_receipt_rel("mutation", receipt["id"])),
        receipt,
        resume=True,
    )
    vol.commit()
    return {"ok": True, "n": len(results), "unexpected": [],
            "prerequisite_receipt_id": receipt["id"]}


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
