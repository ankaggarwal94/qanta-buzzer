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

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path, PurePosixPath

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
    expected_id: str | None,
    file_key: str,
    name_key: str,
    content_subdir: str = "",
    expected_kind: str | None = None,
    require_semantic_pass: bool = False,
) -> dict:
    """Verify a content-addressed manifest and every file it declares."""
    from scripts.stopdff_v5.identity import (
        compute_id,
        loads_no_duplicate_keys,
        sha256_file,
    )

    base = Path(base)
    manifest_path = base / manifest_name
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"{manifest_name} is missing or not a regular file")
    manifest = loads_no_duplicate_keys(
        manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("identity"),
        dict,
    ):
        raise ValueError(f"{manifest_name} is not a manifest object")
    identity = manifest["identity"]
    if compute_id(identity) != manifest.get("id"):
        raise ValueError(f"{manifest_name} id mismatch")
    if expected_id is not None and manifest["id"] != expected_id:
        raise ValueError(
            f"{manifest_name} id {manifest['id']} != expected {expected_id}"
        )
    if expected_kind is not None and identity.get("kind") != expected_kind:
        raise ValueError(
            f"{manifest_name} kind {identity.get('kind')!r} != "
            f"expected {expected_kind!r}"
        )
    if require_semantic_pass:
        semantic_checks = identity.get("semantic_checks")
        if (
            not isinstance(semantic_checks, dict)
            or semantic_checks.get("all_semantic_checks_pass") is not True
        ):
            raise ValueError(
                f"{manifest_name} does not record passing semantic checks"
            )
    entries = identity.get(file_key)
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


def _verified_raw_input_manifest(base, *, expected_id: str | None) -> dict:
    """Verify raw bytes plus the fail-closed semantic decision they carry."""
    return _verified_content_manifest(
        base,
        manifest_name="raw_input_manifest.json",
        expected_id=expected_id,
        file_key="files",
        name_key="role",
        expected_kind="raw_input_bundle",
        require_semantic_pass=True,
    )


def _validated_cached_adapter(
    out: Path,
    *,
    subdir: str,
    source_id: str,
    raw_id: str,
    model_id: str,
    checker_module,
    allow_low_mc_retention: bool = False,
) -> dict:
    """Return a cached adapter only after complete validation and binding checks."""
    if out.is_symlink() or not out.is_dir():
        raise FileExistsError("adapter destination exists but is not a directory")
    result = checker_module.validate_adapter(out)
    if not result.passed:
        raise FileExistsError(
            "adapter destination exists but is invalid: "
            + "; ".join(result.errors)
        )
    manifest = checker_module.load_json(out / "manifest.json")
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    expected_bindings = {
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
    }
    if (
        not isinstance(identity, dict)
        or any(identity.get(key) != value for key, value in expected_bindings.items())
        or result.recomputed.get("adapter_bundle_id") != manifest.get("id")
    ):
        raise FileExistsError(
            "adapter destination exists with incompatible identity bindings"
        )
    retention = identity.get("mc_retention_evidence")
    splits = retention.get("splits") if isinstance(retention, dict) else None
    used_override = (
        isinstance(splits, dict)
        and any(
            isinstance(splits.get(role), dict)
            and splits[role].get("overridden") is True
            for role in ("fit", "eval")
        )
    )
    if used_override and not allow_low_mc_retention:
        raise FileExistsError(
            "adapter destination requires the low-retention gate override"
        )
    return {
        "adapter_id": manifest["id"],
        "fit_rows_sha256": identity["fit_rows_sha256"],
        "eval_rows_sha256": identity["eval_rows_sha256"],
        "subdir": subdir,
        "cached": True,
    }


def _validated_cached_fvi(
    out: Path,
    *,
    manifest: dict,
    identity: dict,
    execution: dict,
    checker_module,
) -> dict:
    """Validate both durable FVI records before accepting a cache hit."""
    manifest_path = out / "fvi_study.json"
    execution_path = out / "fvi_study_execution.json"
    if (
        out.is_symlink()
        or not out.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or execution_path.is_symlink()
        or not execution_path.is_file()
    ):
        raise FileExistsError("FVI destination is incomplete or noncanonical")
    try:
        existing_manifest = checker_module.load_json(manifest_path)
        existing_execution = checker_module.load_json(execution_path)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise FileExistsError("FVI destination contains invalid evidence") from exc
    canonical_execution = json.loads(
        json.dumps(execution, sort_keys=True, default=str)
    )
    if (
        existing_manifest != manifest
        or existing_manifest.get("identity") != identity
        or existing_execution != canonical_execution
    ):
        raise FileExistsError("FVI destination exists with different content")
    return {
        "fvi_study_id": manifest["id"],
        "selected": canonical_execution["selected_parameters"],
        "cached": True,
    }


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
    vol.reload()
    base = Path(_p(rel_dir))
    try:
        if kind == "raw":
            manifest = _verified_raw_input_manifest(base, expected_id=None)
        elif kind == "source":
            manifest = _verified_content_manifest(
                base,
                manifest_name="source_manifest.json",
                expected_id=None,
                file_key="files",
                name_key="path",
                content_subdir="source",
                expected_kind="source_snapshot",
            )
        elif kind == "model":
            manifest = _verified_content_manifest(
                base,
                manifest_name="model_snapshot_manifest.json",
                expected_id=None,
                file_key="files",
                name_key="path",
                content_subdir="snapshot",
                expected_kind="model_snapshot",
            )
        else:
            return {"ok": False, "error": f"unknown kind {kind}"}
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    identity = manifest["identity"]
    result = {
        "ok": True,
        "id": manifest["id"],
        "mismatches": [],
        "n_files": len(identity["files"]),
    }
    if kind == "raw":
        stopdff = next(
            (
                entry
                for entry in identity["files"]
                if entry.get("role") == "stopdff.json"
            ),
            None,
        )
        if stopdff is None:
            return {"ok": False, "error": "raw-input manifest lacks stopdff.json"}
        result["myopic_artifact_sha256"] = stopdff["sha256"]
    return result


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
def build_adapter(
    dest_subdir: str,
    source_id: str,
    raw_id: str,
    model_id: str,
    allow_low_mc_retention: bool = False,
) -> dict:
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
    _verified_raw_input_manifest(raw, expected_id=raw_id)
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
        return _validated_cached_adapter(
            out,
            subdir=dest_subdir,
            source_id=source_id,
            raw_id=raw_id,
            model_id=model_id,
            checker_module=checker,
            allow_low_mc_retention=allow_low_mc_retention,
        )
    man = adapter_build.build_adapter_bundle(
        mc_dataset_path=raw / "mc_dataset.json", val_dataset_path=raw / "val_dataset.json",
        test_dataset_path=raw / "test_dataset.json", calibration_path=raw / "calibration.json",
        model_snapshot_dir=model_dir, out_dir=out, source_manifest_id=source_id,
        raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        producer_hashes={"adapter_build.py": sha256_file(Path(REMOTE_SRC) / "scripts/stopdff_v5/adapter_build.py")},
        allow_low_mc_retention=allow_low_mc_retention,
    )
    result = checker.validate_adapter(out)
    if not result.passed or result.recomputed.get("adapter_bundle_id") != man["id"]:
        raise ValueError(
            "new adapter failed validation: " + "; ".join(result.errors)
        )
    vol.commit()
    return {"adapter_id": man["id"], "fit_rows_sha256": man["identity"]["fit_rows_sha256"],
            "eval_rows_sha256": man["identity"]["eval_rows_sha256"],
            "subdir": dest_subdir, "cached": False}


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
        return _validated_cached_fvi(
            out,
            manifest=man,
            identity=ident,
            execution=study,
            checker_module=checker,
        )
    out.mkdir(parents=True)
    (out / "fvi_study.json").write_text(
        json.dumps(man, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "fvi_study_execution.json").write_text(
        json.dumps(study, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
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
    raw_manifest = _verified_raw_input_manifest(
        raw_root,
        expected_id=raw_id,
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
    _verified_raw_input_manifest(
        raw_path.parent,
        expected_id=spec_ids["raw_input_bundle_id"],
    )
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


def _control_plan_digest(plan: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            plan,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_control_json(path: Path) -> dict:
    from scripts.stopdff_v5.identity import loads_no_duplicate_keys

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"control JSON is missing or noncanonical: {path}")
    value = loads_no_duplicate_keys(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"control JSON must contain an object: {path}")
    return value


def _write_control_state(path: Path, state: dict) -> None:
    """Atomically replace and fsync the local control-plane checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_control_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_control_event(
    state_path: Path,
    state: dict,
    *,
    event: str,
    stage: str | None = None,
    detail: dict | None = None,
) -> None:
    state["sequence"] = int(state.get("sequence", 0)) + 1
    record = {
        "sequence": state["sequence"],
        "event": event,
        "stage": stage,
        "utc_epoch_seconds": int(time.time()),
        "detail": detail or {},
    }
    state["last_event"] = record
    _write_control_state(state_path, state)
    _append_control_event(
        state_path.with_name(state_path.name + ".jsonl"),
        record,
    )


def _validate_control_plan(plan: dict) -> dict:
    allowed = {
        "source_id",
        "raw_id",
        "adapter_subdirs",
        "gate_overrides",
        "resource_summary",
    }
    if set(plan) - allowed:
        raise ValueError(
            f"unknown control-plan fields: {sorted(set(plan) - allowed)}"
        )

    def require_sha(name: str) -> str:
        value = plan.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise ValueError(f"control plan {name} must be canonical 64-hex")
        return value

    source_id = require_sha("source_id")
    raw_id = require_sha("raw_id")
    subdirs = plan.get("adapter_subdirs")
    if (
        not isinstance(subdirs, list)
        or len(subdirs) != 2
        or not all(isinstance(value, str) and value for value in subdirs)
        or len(set(subdirs)) != 2
    ):
        raise ValueError("control plan requires two distinct adapter_subdirs")
    for subdir in subdirs:
        parsed = PurePosixPath(subdir)
        if parsed.is_absolute() or ".." in parsed.parts or len(parsed.parts) != 1:
            raise ValueError(f"unsafe adapter subdir: {subdir!r}")
    gate_overrides = plan.get("gate_overrides", {})
    if (
        not isinstance(gate_overrides, dict)
        or set(gate_overrides)
        - {"allow_low_mc_retention", "allow_incomplete_mc_coverage"}
        or not all(isinstance(value, bool) for value in gate_overrides.values())
    ):
        raise ValueError("control plan contains invalid gate_overrides")
    resource_summary = plan.get("resource_summary", {})
    if not isinstance(resource_summary, dict):
        raise ValueError("control plan resource_summary must be an object")
    return {
        "source_id": source_id,
        "raw_id": raw_id,
        "adapter_subdirs": list(subdirs),
        "gate_overrides": dict(gate_overrides),
        "resource_summary": dict(resource_summary),
    }


def _default_control_stage_api() -> dict[str, object]:
    return {
        "probe": probe.remote,
        "verify_volume_artifact": verify_volume_artifact.remote,
        "freeze_model": freeze_model.remote,
        "build_adapter": build_adapter.remote,
        "adapter_determinism_receipt": adapter_determinism_receipt.remote,
        "promote_adapter": promote_adapter.remote,
        "fvi_study": fvi_study.remote,
        "bootstrap_plan": bootstrap_plan.remote,
        "run_sweep": run_sweep.remote,
        "mutation_gate": mutation_gate.remote,
        "validate": validate.remote,
        "package": package.remote,
    }


def _run_control_stage(
    state_path: Path,
    state: dict,
    *,
    name: str,
    invoke,
) -> dict:
    completed = state.setdefault("completed", {})
    if name in completed:
        result = completed[name]
        if not isinstance(result, dict):
            raise ValueError(f"control stage {name} has a non-object checkpoint")
        return result
    attempts = state.setdefault("stage_attempts", {})
    attempt = int(attempts.get(name, 0)) + 1
    attempts[name] = attempt
    state["status"] = "running"
    _record_control_event(
        state_path,
        state,
        event="stage_started",
        stage=name,
        detail={"attempt": attempt},
    )
    try:
        result = invoke(attempt)
        if not isinstance(result, dict):
            raise TypeError(f"control stage {name} returned a non-object")
    except BaseException as exc:
        state["status"] = "failed"
        state["last_error"] = {
            "stage": name,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _record_control_event(
            state_path,
            state,
            event="stage_failed",
            stage=name,
            detail={"attempt": attempt, **state["last_error"]},
        )
        raise
    completed[name] = result
    state.pop("last_error", None)
    _record_control_event(
        state_path,
        state,
        event="stage_completed",
        stage=name,
        detail={"attempt": attempt},
    )
    return result


def _require_control_result(
    stage: str,
    result: dict,
    *,
    expected_id: tuple[str, str] | None = None,
) -> dict:
    if result.get("ok") is False:
        raise ValueError(
            f"control stage {stage} failed verification: {result.get('error')}"
        )
    if expected_id is not None:
        field, expected = expected_id
        if result.get(field) != expected:
            raise ValueError(
                f"control stage {stage} returned {field}={result.get(field)!r}, "
                f"expected {expected!r}"
            )
    return result


def run_control_plane(
    plan: dict,
    state_path: Path,
    *,
    resume: bool,
    stage_api: dict[str, object] | None = None,
) -> dict:
    """Run the canonical Modal stage order with a durable local checkpoint."""
    from scripts.stopdff_v5.identity import compute_id, sha256_file
    from scripts.stopdff_v5.manifests import (
        environment_contract_identity,
        run_spec_identity,
    )

    plan = _validate_control_plan(plan)
    state_path = Path(state_path)
    digest = _control_plan_digest(plan)
    if resume:
        state = _load_control_json(state_path)
        if state.get("plan_digest") != digest or state.get("plan") != plan:
            raise ValueError("resume control plan does not match durable state")
        if state.get("schema_version") != 1:
            raise ValueError("unsupported control-state schema")
        if state.get("status") == "completed":
            return state
    else:
        if state_path.exists() or state_path.is_symlink():
            raise FileExistsError("fresh control state already exists")
        state = {
            "schema_version": 1,
            "plan": plan,
            "plan_digest": digest,
            "status": "initialized",
            "sequence": 0,
            "stage_attempts": {},
            "completed": {},
        }
        _record_control_event(
            state_path,
            state,
            event="control_initialized",
        )

    api = stage_api or _default_control_stage_api()
    required_api = {
        "probe",
        "verify_volume_artifact",
        "freeze_model",
        "build_adapter",
        "adapter_determinism_receipt",
        "promote_adapter",
        "fvi_study",
        "bootstrap_plan",
        "run_sweep",
        "mutation_gate",
        "validate",
        "package",
    }
    if set(api) != required_api or not all(callable(api[name]) for name in api):
        raise ValueError("control stage API does not match the canonical stage set")

    source_id = plan["source_id"]
    raw_id = plan["raw_id"]
    source_check = _run_control_stage(
        state_path,
        state,
        name="verify_source",
        invoke=lambda _: api["verify_volume_artifact"](
            f"inputs/source_{source_id}",
            "source",
        ),
    )
    _require_control_result(
        "verify_source",
        source_check,
        expected_id=("id", source_id),
    )
    raw_check = _run_control_stage(
        state_path,
        state,
        name="verify_raw",
        invoke=lambda _: api["verify_volume_artifact"](
            f"inputs/raw_{raw_id}",
            "raw",
        ),
    )
    _require_control_result(
        "verify_raw",
        raw_check,
        expected_id=("id", raw_id),
    )
    myopic_sha256 = raw_check.get("myopic_artifact_sha256")
    if (
        not isinstance(myopic_sha256, str)
        or len(myopic_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in myopic_sha256)
    ):
        raise ValueError("verified raw bundle lacks a canonical myopic artifact hash")

    probe_result = _run_control_stage(
        state_path,
        state,
        name="environment_probe",
        invoke=lambda _: api["probe"](),
    )
    if (
        not isinstance(probe_result.get("python"), str)
        or not isinstance(probe_result.get("package_versions"), dict)
    ):
        raise ValueError("environment probe returned an invalid contract")
    environment_identity = environment_contract_identity(
        python_version=probe_result["python"],
        package_versions=probe_result["package_versions"],
    )
    environment_contract_id = compute_id(environment_identity)
    producer_hashes = {
        name: sha256_file(Path(SOURCE_DIR) / "scripts" / "stopdff_v5" / name)
        for name in ("checker.py", "sweep.py")
    }

    model_result = _run_control_stage(
        state_path,
        state,
        name="freeze_model",
        invoke=lambda _: api["freeze_model"](),
    )
    model_id = model_result.get("model_id")
    if not isinstance(model_id, str):
        raise ValueError("model stage returned no model_id")

    first_subdir, second_subdir = plan["adapter_subdirs"]
    first_adapter = _run_control_stage(
        state_path,
        state,
        name="build_adapter_first",
        invoke=lambda _: api["build_adapter"](
            first_subdir,
            source_id,
            raw_id,
            model_id,
            bool(plan["gate_overrides"].get("allow_low_mc_retention", False)),
        ),
    )
    second_adapter = _run_control_stage(
        state_path,
        state,
        name="build_adapter_second",
        invoke=lambda _: api["build_adapter"](
            second_subdir,
            source_id,
            raw_id,
            model_id,
            bool(plan["gate_overrides"].get("allow_low_mc_retention", False)),
        ),
    )
    adapter_id = first_adapter.get("adapter_id")
    if not isinstance(adapter_id, str) or second_adapter.get("adapter_id") != adapter_id:
        raise ValueError("two adapter builds did not return one bound identity")
    determinism_bindings = {
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
        "adapter_bundle_id": adapter_id,
    }
    determinism = _run_control_stage(
        state_path,
        state,
        name="adapter_determinism",
        invoke=lambda _: api["adapter_determinism_receipt"](
            first_subdir,
            second_subdir,
            json.dumps(determinism_bindings, sort_keys=True),
        ),
    )
    determinism_receipt_id = determinism.get("prerequisite_receipt_id")
    if not isinstance(determinism_receipt_id, str):
        raise ValueError("adapter determinism stage returned no receipt")

    _run_control_stage(
        state_path,
        state,
        name="promote_adapter",
        invoke=lambda _: api["promote_adapter"](first_subdir, adapter_id),
    )
    fvi_result = _run_control_stage(
        state_path,
        state,
        name="fvi_study",
        invoke=lambda _: api["fvi_study"](adapter_id),
    )
    fvi_id = fvi_result.get("fvi_study_id")
    selected = fvi_result.get("selected")
    if not isinstance(fvi_id, str) or not isinstance(selected, dict):
        raise ValueError("FVI stage returned an incomplete selection")

    common_bindings = {
        **determinism_bindings,
        "fvi_study_id": fvi_id,
        "environment_contract_id": environment_contract_id,
    }
    smoke_bootstrap = _run_control_stage(
        state_path,
        state,
        name="smoke_bootstrap",
        invoke=lambda _: api["bootstrap_plan"](adapter_id, 100),
    )
    smoke_bootstrap_id = smoke_bootstrap.get("bootstrap_plan_id")
    if not isinstance(smoke_bootstrap_id, str):
        raise ValueError("smoke bootstrap stage returned no identity")
    smoke_spec = run_spec_identity(
        source_manifest_id=source_id,
        raw_input_bundle_id=raw_id,
        model_snapshot_id=model_id,
        adapter_bundle_id=adapter_id,
        fvi_study_id=fvi_id,
        bootstrap_plan_id=smoke_bootstrap_id,
        environment_contract_id=environment_contract_id,
        fvi_selected=selected,
        replicate_count=100,
        profile_variant="smoke",
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=producer_hashes,
        prerequisite_receipts={},
        gate_overrides=plan["gate_overrides"],
    )
    smoke_spec_id = compute_id(smoke_spec)
    smoke_run_id = f"smoke_modal_{smoke_spec_id[:12]}"

    def invoke_smoke(attempt: int) -> dict:
        wrapper = {
            "run_id": smoke_run_id,
            "run_spec_id": smoke_spec_id,
            "run_spec_identity": smoke_spec,
            "attempt": attempt,
            "resource_summary": plan["resource_summary"],
        }
        return api["run_sweep"](
            json.dumps(wrapper, sort_keys=True),
            adapter_id,
            smoke_bootstrap_id,
            attempt > 1,
        )

    smoke_result = _run_control_stage(
        state_path,
        state,
        name="smoke_sweep",
        invoke=invoke_smoke,
    )
    smoke_receipt_id = smoke_result.get("prerequisite_receipt_id")
    if not isinstance(smoke_receipt_id, str):
        raise ValueError("smoke stage returned no prerequisite receipt")

    mutation = _run_control_stage(
        state_path,
        state,
        name="mutation_gate",
        invoke=lambda _: api["mutation_gate"](
            json.dumps(common_bindings, sort_keys=True)
        ),
    )
    if mutation.get("ok") is not True:
        raise ValueError("mutation gate did not pass")
    mutation_receipt_id = mutation.get("prerequisite_receipt_id")
    if not isinstance(mutation_receipt_id, str):
        raise ValueError("mutation gate returned no prerequisite receipt")

    final_bootstrap = _run_control_stage(
        state_path,
        state,
        name="final_bootstrap",
        invoke=lambda _: api["bootstrap_plan"](adapter_id, 1000),
    )
    final_bootstrap_id = final_bootstrap.get("bootstrap_plan_id")
    if not isinstance(final_bootstrap_id, str):
        raise ValueError("final bootstrap stage returned no identity")
    receipt_ids = {
        "determinism": determinism_receipt_id,
        "mutation": mutation_receipt_id,
        "smoke": smoke_receipt_id,
    }
    final_spec = run_spec_identity(
        source_manifest_id=source_id,
        raw_input_bundle_id=raw_id,
        model_snapshot_id=model_id,
        adapter_bundle_id=adapter_id,
        fvi_study_id=fvi_id,
        bootstrap_plan_id=final_bootstrap_id,
        environment_contract_id=environment_contract_id,
        fvi_selected=selected,
        replicate_count=1000,
        profile_variant="final",
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=producer_hashes,
        prerequisite_receipts=receipt_ids,
        gate_overrides=plan["gate_overrides"],
    )
    final_spec_id = compute_id(final_spec)
    final_run_id = f"final_modal_{final_spec_id[:12]}"

    def invoke_final(attempt: int) -> dict:
        wrapper = {
            "run_id": final_run_id,
            "run_spec_id": final_spec_id,
            "run_spec_identity": final_spec,
            "attempt": attempt,
            "resource_summary": plan["resource_summary"],
        }
        return api["run_sweep"](
            json.dumps(wrapper, sort_keys=True),
            adapter_id,
            final_bootstrap_id,
            attempt > 1,
        )

    _run_control_stage(
        state_path,
        state,
        name="final_sweep",
        invoke=invoke_final,
    )
    prepackage = _run_control_stage(
        state_path,
        state,
        name="validate_unpacked",
        invoke=lambda _: api["validate"](
            final_run_id,
            adapter_id,
            True,
            False,
        ),
    )
    if prepackage.get("passed") is not True:
        raise ValueError(
            "final run failed prepackage validation: "
            + "; ".join(prepackage.get("errors", [])[:10])
        )
    _run_control_stage(
        state_path,
        state,
        name="package",
        invoke=lambda _: api["package"](final_run_id),
    )
    final_validation = _run_control_stage(
        state_path,
        state,
        name="validate_package",
        invoke=lambda _: api["validate"](
            final_run_id,
            adapter_id,
            True,
            True,
        ),
    )
    if final_validation.get("passed") is not True:
        raise ValueError(
            "final package validation failed: "
            + "; ".join(final_validation.get("errors", [])[:10])
        )
    state["status"] = "completed"
    state["result"] = {
        "run_id": final_run_id,
        "run_spec_id": final_spec_id,
        "adapter_id": adapter_id,
        "receipt_ids": receipt_ids,
        "validation": final_validation,
    }
    _record_control_event(
        state_path,
        state,
        event="control_completed",
        detail={"run_id": final_run_id, "run_spec_id": final_spec_id},
    )
    return state


@app.local_entrypoint()
def control_main(
    plan_path: str,
    state_path: str,
    resume: bool = False,
):
    """Execute or resume the durable Modal control plane from a JSON plan."""
    plan = _load_control_json(Path(plan_path))
    state = run_control_plane(
        plan,
        Path(state_path),
        resume=resume,
    )
    print(json.dumps(state["result"], indent=2, sort_keys=True))


@app.local_entrypoint()
def probe_main():
    print(probe.remote())
