#!/usr/bin/env python3
"""Modal standalone runner for the StopDFF v5 evidentiary pipeline (remote functions).

Source-only image (STOPDFF_V5_SOURCE_DIR = validated source-snapshot bundle). One Volume
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
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "cs321m-stopdff-v5"
VOLUME_NAME = "cs321m-stopdff-artifacts"
MNT = "/stopdff"
REMOTE_SRC = "/root/src"
DAY = 86400
SOURCE_BUNDLE_DIR = os.environ.get("STOPDFF_V5_SOURCE_DIR", "")
if not SOURCE_BUNDLE_DIR:
    raise RuntimeError(
        "STOPDFF_V5_SOURCE_DIR must point to the frozen source-snapshot bundle"
    )


def _materialize_image_source(source_bundle: Path) -> tuple[Path, dict]:
    """Copy and revalidate the exact source tree used to build the image.

    The first validation rejects unlisted executable bytes.  The second binds
    the private copy actually handed to Modal, avoiding a validate-then-upload
    race against an operator-controlled directory.
    """
    from scripts.stopdff_v5.content_manifest import (
        validate_bound_content_manifest,
    )

    source_bundle = Path(source_bundle)
    manifest = validate_bound_content_manifest(
        source_bundle,
        manifest_name="source_manifest.json",
        expected_id=None,
        file_key="files",
        name_key="path",
        content_subdir="source",
        expected_kind="source_snapshot",
    )
    staged_bundle = Path(tempfile.mkdtemp(prefix="stopdff_v5_image_source_"))
    shutil.copy2(
        source_bundle / "source_manifest.json",
        staged_bundle / "source_manifest.json",
    )
    shutil.copytree(source_bundle / "source", staged_bundle / "source")
    staged_manifest = validate_bound_content_manifest(
        staged_bundle,
        manifest_name="source_manifest.json",
        expected_id=manifest["id"],
        file_key="files",
        name_key="path",
        content_subdir="source",
        expected_kind="source_snapshot",
    )
    return staged_bundle / "source", staged_manifest


_IMAGE_SOURCE_DIR, _IMAGE_SOURCE_MANIFEST = _materialize_image_source(
    Path(SOURCE_BUNDLE_DIR)
)
SOURCE_DIR = str(_IMAGE_SOURCE_DIR)
IMAGE_SOURCE_MANIFEST_ID = _IMAGE_SOURCE_MANIFEST["id"]


def _require_image_source_id(source_id: object) -> None:
    """Reject a stage claim that is not bound to the source in its image."""
    if source_id != IMAGE_SOURCE_MANIFEST_ID:
        raise ValueError(
            "stage source_id does not match the validated Modal image source"
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
          "PYTHONDONTWRITEBYTECODE": "1",
          "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": REMOTE_SRC})
)
_image = _image.add_local_dir(SOURCE_DIR, remote_path=REMOTE_SRC, copy=True)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=_image, include_source=False)


def _p(*parts) -> str:
    path = PurePosixPath(MNT)
    for part in parts:
        candidate = PurePosixPath(str(part))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe volume path component: {part!r}")
        path /= candidate
    return str(path)


def _canonical_adapter_subdir(value: object) -> str:
    """Return one canonical adapter path component or fail closed."""
    if not isinstance(value, str) or not value:
        raise ValueError("adapter subdir must be a nonempty string")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or len(parsed.parts) != 1
        or str(parsed) != value
    ):
        raise ValueError(f"unsafe or noncanonical adapter subdir: {value!r}")
    return value


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


def _receipt_evidence_rel(gate: str, receipt_id: str) -> str:
    """Return the evidence sidecar paired with one content-addressed receipt."""
    receipt = Path(_receipt_rel(gate, receipt_id))
    return str(receipt.with_suffix(".evidence.json"))


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
    """Verify the canonical identity and every staged byte it exhaustively lists."""
    from scripts.stopdff_v5.content_manifest import (
        validate_bound_content_manifest,
    )

    return validate_bound_content_manifest(
        Path(base),
        manifest_name=manifest_name,
        expected_id=expected_id,
        file_key=file_key,
        name_key=name_key,
        content_subdir=content_subdir,
        expected_kind=expected_kind,
        require_semantic_pass=require_semantic_pass,
    )


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


def _verified_executing_source(source_id: object) -> dict:
    """Bind a stage to both its staged manifest and executing source bytes."""
    _require_image_source_id(source_id)
    from scripts.stopdff_v5.identity import sha256_file

    source_root = Path(_p("inputs", f"source_{source_id}"))
    source_manifest = _verified_content_manifest(
        source_root,
        manifest_name="source_manifest.json",
        expected_id=source_id,
        file_key="files",
        name_key="path",
        content_subdir="source",
        expected_kind="source_snapshot",
    )
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
    return source_manifest


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
    from importlib import metadata as im
    import platform
    from scripts.stopdff_v5.manifests import ENVIRONMENT_PACKAGES

    package_versions = {
        name: im.version(name)
        for name in ENVIRONMENT_PACKAGES
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
    dest_subdir = _canonical_adapter_subdir(dest_subdir)
    _require_image_source_id(source_id)
    from pathlib import Path
    from scripts.stopdff_v5 import adapter_build
    from scripts.stopdff_v5 import checker
    from scripts.stopdff_v5.identity import sha256_file
    vol.reload()
    raw = Path(_p("inputs", f"raw_{raw_id}"))
    model_root = Path(_p("inputs", "model"))
    model_dir = model_root / "snapshot"
    _verified_executing_source(source_id)
    _verified_raw_input_manifest(raw, expected_id=raw_id)
    _verified_content_manifest(
        model_root,
        manifest_name="model_snapshot_manifest.json",
        expected_id=model_id,
        file_key="files",
        name_key="path",
        content_subdir="snapshot",
    )
    out = Path(_p("adapters", dest_subdir))
    if out.exists() or out.is_symlink():
        raise FileExistsError(
            "fresh adapter build destination already exists; choose a new subdir"
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
    return {
        "adapter_id": man["id"],
        "fit_rows_sha256": man["identity"]["fit_rows_sha256"],
        "eval_rows_sha256": man["identity"]["eval_rows_sha256"],
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
        "subdir": dest_subdir,
        "cached": False,
    }


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


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1)
def adapter_determinism_receipt(
    first_subdir: str,
    second_subdir: str,
    source_id: str,
    raw_id: str,
    model_id: str,
    allow_low_mc_retention: bool = False,
) -> dict:
    """Own two fresh adapter executions and persist their determinism receipt."""
    first_subdir = _canonical_adapter_subdir(first_subdir)
    second_subdir = _canonical_adapter_subdir(second_subdir)
    if first_subdir == second_subdir:
        raise ValueError("adapter determinism requires distinct canonical subdirs")
    _require_image_source_id(source_id)
    vol.reload()
    source_manifest = _verified_executing_source(source_id)

    from pathlib import Path
    from scripts.stopdff_v5 import checker, sweep, writers
    from scripts.stopdff_v5.identity import sha256_file

    first_call = build_adapter.spawn(
        first_subdir,
        source_id,
        raw_id,
        model_id,
        allow_low_mc_retention,
    )
    first_result = first_call.get()
    _validate_adapter_result(
        "adapter_determinism_first_build",
        first_result,
        expected_subdir=first_subdir,
        expected_source_id=source_id,
        expected_raw_id=raw_id,
        expected_model_id=model_id,
        require_fresh=True,
    )
    second_call = build_adapter.spawn(
        second_subdir,
        source_id,
        raw_id,
        model_id,
        allow_low_mc_retention,
    )
    second_result = second_call.get()
    adapter_id = first_result["adapter_id"]
    _validate_adapter_result(
        "adapter_determinism_second_build",
        second_result,
        expected_subdir=second_subdir,
        expected_id=adapter_id,
        expected_source_id=source_id,
        expected_raw_id=raw_id,
        expected_model_id=model_id,
        require_fresh=True,
    )
    first_execution_id = getattr(first_call, "object_id", None)
    second_execution_id = getattr(second_call, "object_id", None)
    if (
        not isinstance(first_execution_id, str)
        or not first_execution_id
        or not isinstance(second_execution_id, str)
        or not second_execution_id
        or first_execution_id == second_execution_id
    ):
        raise ValueError("adapter builds lack distinct Modal function-call IDs")

    vol.reload()
    first = Path(_p("adapters", first_subdir))
    second = Path(_p("adapters", second_subdir))
    first_result = checker.validate_adapter(first)
    second_result = checker.validate_adapter(second)
    if (
        not first_result.passed
        or not second_result.passed
        or first_result.recomputed.get("adapter_bundle_id") != adapter_id
        or second_result.recomputed.get("adapter_bundle_id") != adapter_id
    ):
        raise ValueError("adapter determinism inputs are invalid or unbound")
    compared = (
        "fit_rows.jsonl.gz",
        "eval_rows.jsonl.gz",
        "calibration.json",
        "build_metadata.json",
    )
    first_hashes = {name: sha256_file(first / name) for name in compared}
    second_hashes = {name: sha256_file(second / name) for name in compared}
    if first_hashes != second_hashes:
        raise ValueError("independent adapter builds are not byte-identical")
    first_manifest = checker.load_json(first / "manifest.json")
    second_manifest = checker.load_json(second / "manifest.json")
    bindings = {
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
        "adapter_bundle_id": adapter_id,
    }
    source_execution = {
        "environment": "modal_image",
        "executing_source_manifest_id": IMAGE_SOURCE_MANIFEST_ID,
        "runtime_source_manifest_id": source_manifest["id"],
    }

    def build_execution(*, execution_id: str, subdir: str, hashes: dict) -> dict:
        return {
            "environment": "modal_function_call",
            "execution_id": execution_id,
            "adapter_subdir": subdir,
            "source_manifest_id": source_id,
            "raw_input_bundle_id": raw_id,
            "model_snapshot_id": model_id,
            "adapter_bundle_id": adapter_id,
            "cached": False,
            "output_sha256": hashes,
        }

    evidence = writers.build_prerequisite_evidence(
        gate="determinism",
        bindings=bindings,
        details={
            "source_execution": source_execution,
            "first_build_execution": build_execution(
                execution_id=first_execution_id,
                subdir=first_subdir,
                hashes=first_hashes,
            ),
            "second_build_execution": build_execution(
                execution_id=second_execution_id,
                subdir=second_subdir,
                hashes=second_hashes,
            ),
            "first_adapter_manifest": first_manifest,
            "second_adapter_manifest": second_manifest,
            "first_file_sha256": first_hashes,
            "second_file_sha256": second_hashes,
        },
    )
    receipt = writers.build_evidenced_prerequisite_receipt(
        gate="determinism",
        bindings=bindings,
        evidence=evidence,
    )
    sweep._write_bound_json(
        Path(_receipt_evidence_rel("determinism", receipt["id"])),
        evidence,
        resume=True,
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
        "source_manifest_id": source_id,
        "first_build_execution_id": first_execution_id,
        "second_build_execution_id": second_execution_id,
        "prerequisite_receipt_id": receipt["id"],
    }


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1, memory=16384)
def fvi_study(adapter_id: str) -> dict:
    import json
    from pathlib import Path
    from scripts.stopdff_v5 import fvi_study as fs
    from scripts.stopdff_v5 import checker
    from scripts.stopdff_v5.checker import load_adapter_rows
    from scripts.stopdff_v5.identity import build_manifest, sha256_file
    from scripts.stopdff_v5.manifests import (
        FVI_PRODUCER_FILES,
        fvi_study_identity,
    )
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
    producer_hashes = {
        name: sha256_file(Path(REMOTE_SRC) / "scripts" / "stopdff_v5" / name)
        for name in FVI_PRODUCER_FILES
    }
    ident = fvi_study_identity(
        adapter_bundle_id=adapter_id, candidate_grid=study["candidate_grid"],
        representative_generator=study["representative_cell_generator"],
        candidate_results=study["candidate_convergence_results"],
        strict_reference_results=study["strict_reference"], selector_rule=study["selector_rule"],
        selected_parameters=study["selected_parameters"], all96_validation=study["all96_fit_only_validation"],
        producer_hashes=producer_hashes)
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


def _resolve_remote_sweep_attempt(
    run_root: Path,
    *,
    recovery_requested: bool,
    sweep_module,
) -> tuple[bool, int]:
    """Derive evidence mode/number from authoritative durable run state."""
    run_root = Path(run_root)
    if run_root.is_symlink():
        raise ValueError("sweep destination is a symlink")
    if not run_root.exists():
        return False, 1
    if not run_root.is_dir():
        raise ValueError("sweep destination is not a directory")
    if not recovery_requested:
        raise FileExistsError("fresh sweep destination already exists")
    _, history = sweep_module._load_attempt_history(
        run_root / "attempts.jsonl"
    )
    if not history:
        raise ValueError("recovery destination has no durable attempt history")
    return True, len(history) + 1


@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1, memory=16384)
def run_sweep(
    spec_json: str,
    adapter_id: str,
    bootstrap_plan_id: str,
    recovery_requested: bool,
) -> dict:
    # This guard deliberately uses only the stdlib already loaded by the image.
    # It runs before importing reviewed project modules, so a directly invoked
    # stage cannot claim a strict-subset source manifest while executing the
    # larger source tree baked into this Modal image.  The canonical parser and
    # complete run-spec validation still run below.
    try:
        early_spec = json.loads(spec_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("sweep wrapper is not JSON") from exc
    early_identity = (
        early_spec.get("run_spec_identity")
        if isinstance(early_spec, dict)
        else None
    )
    early_ids = (
        early_identity.get("identity")
        if isinstance(early_identity, dict)
        else None
    )
    _require_image_source_id(
        early_ids.get("source_manifest_id")
        if isinstance(early_ids, dict)
        else None
    )
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

    from scripts.stopdff_v5.manifests import ENVIRONMENT_PACKAGES

    package_versions = {
        name: im.version(name)
        for name in ENVIRONMENT_PACKAGES
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
    if binding["variant"] == "final":
        for gate, receipt in receipts.items():
            evidence_path = Path(_receipt_evidence_rel(gate, receipt["id"]))
            if evidence_path.is_symlink() or not evidence_path.is_file():
                raise ValueError(f"{gate} prerequisite evidence is unavailable")
            writers.verify_prerequisite_evidence_bytes(
                gate=gate,
                bindings=receipt["identity"]["bindings"],
                receipt_evidence=receipt["identity"]["evidence"],
                data=evidence_path.read_bytes(),
            )

    rows = binding["rows"]
    calibration = binding["calibration"]
    plan = binding["bootstrap_plan"]
    variant = binding["variant"]
    cells = profile.full_grid() if variant == "final" else profile.smoke_cells()
    run_root = Path(_p("runs", spec["run_id"]))
    if binding["run_spec_id"][:12] not in str(spec["run_id"]):
        raise ValueError("run_id is not bound to run_spec_id")
    actual_resume, evidence_attempt = _resolve_remote_sweep_attempt(
        run_root,
        recovery_requested=bool(recovery_requested),
        sweep_module=sweep,
    )
    if actual_resume:
        existing_spec = run_root / "run_spec.json"
        if existing_spec.exists():
            existing = checker.load_json(existing_spec)
            if existing != run_spec_manifest:
                raise ValueError("resume destination is bound to another run spec")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
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
        gate_overrides=binding["gate_overrides"],
        cells=cells, commit_fn=lambda: vol.commit(),
        environment={
            "python_version": platform.python_version(),
            "package_versions": package_versions,
        },
        resource_summary=spec.get("resource_summary", {}),
        attempt={"attempt": evidence_attempt,
                 "mode": "resume" if actual_resume else "fresh",
                 "command": ["dp_sweep"] + (["--resume"] if actual_resume else []),
                 "run_spec_id": binding["run_spec_id"], "adapter_id": adapter_id,
                 "bootstrap_plan_id": bootstrap_plan_id},
        resume=actual_resume)
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
        receipt_bindings = {
            key: spec_ids[key]
            for key in (
                "source_manifest_id",
                "raw_input_bundle_id",
                "model_snapshot_id",
                "adapter_bundle_id",
                "fvi_study_id",
                "environment_contract_id",
            )
        }
        evidence = writers.build_prerequisite_evidence(
            gate="smoke",
            bindings=receipt_bindings,
            details={
                "run_spec": {
                    "id": binding["run_spec_id"],
                    "identity": binding["run_spec_identity"],
                },
                "aggregate": agg,
            },
        )
        receipt = writers.build_evidenced_prerequisite_receipt(
            gate="smoke",
            bindings=receipt_bindings,
            evidence=evidence,
        )
        receipt_path = Path(_receipt_rel("smoke", receipt["id"]))
        sweep._write_bound_json(
            Path(_receipt_evidence_rel("smoke", receipt["id"])),
            evidence,
            resume=True,
        )
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

    bindings = json.loads(binding_json)
    if not isinstance(bindings, dict):
        raise ValueError("mutation gate bindings must be an object")
    _require_image_source_id(bindings.get("source_manifest_id"))
    vol.reload()
    source_manifest = _verified_executing_source(
        bindings["source_manifest_id"]
    )

    from scripts.stopdff_v5 import selftest, sweep, writers
    from scripts.stopdff_v5.receipt_evidence import (
        validate_prerequisite_bindings,
    )

    bindings = validate_prerequisite_bindings(
        gate="mutation",
        bindings=bindings,
    )
    ok, results = selftest.run_self_test(Path(tempfile.mkdtemp()))
    unexpected = [r["mutation"] for r in results if not r["ok"]]
    if not ok:
        return {"ok": False, "n": len(results), "unexpected": unexpected}
    evidence = writers.build_prerequisite_evidence(
        gate="mutation",
        bindings=bindings,
        details={
            "source_execution": {
                "environment": "modal_image",
                "executing_source_manifest_id": IMAGE_SOURCE_MANIFEST_ID,
                "runtime_source_manifest_id": source_manifest["id"],
            },
            "results": results,
        },
    )
    receipt = writers.build_evidenced_prerequisite_receipt(
        gate="mutation",
        bindings=bindings,
        evidence=evidence,
    )
    sweep._write_bound_json(
        Path(_receipt_evidence_rel("mutation", receipt["id"])),
        evidence,
        resume=True,
    )
    sweep._write_bound_json(
        Path(_receipt_rel("mutation", receipt["id"])),
        receipt,
        resume=True,
    )
    vol.commit()
    return {
        "ok": True,
        "n": len(results),
        "unexpected": [],
        "source_manifest_id": IMAGE_SOURCE_MANIFEST_ID,
        "prerequisite_receipt_id": receipt["id"],
    }


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
    data = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_replace_control_bytes(path, data)


def _atomic_replace_control_bytes(path: Path, data: bytes) -> None:
    """Replace one local control artifact without exposing partial bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.exists() or path.is_symlink()) and (
        path.is_symlink() or not path.is_file()
    ):
        raise ValueError(f"control artifact is noncanonical: {path}")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
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
    """Atomically append one canonical event to the local journal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = b""
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"control journal is noncanonical: {path}")
        existing = path.read_bytes()
        if existing and not existing.endswith(b"\n"):
            raise ValueError("control journal has an unterminated tail")
    line = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    _atomic_replace_control_bytes(path, existing + line)


def _reconcile_control_journal(state_path: Path, state: dict) -> None:
    """Repair one provable final-record gap or reject journal drift."""
    journal_path = state_path.with_name(state_path.name + ".jsonl")
    journal_bytes = b""
    if journal_path.exists() or journal_path.is_symlink():
        if journal_path.is_symlink() or not journal_path.is_file():
            raise ValueError("control journal is noncanonical")
        journal_bytes = journal_path.read_bytes()

    complete_lines: list[bytes]
    torn_tail: bytes | None
    if journal_bytes and not journal_bytes.endswith(b"\n"):
        parts = journal_bytes.split(b"\n")
        complete_lines = parts[:-1]
        torn_tail = parts[-1]
    else:
        complete_lines = (
            journal_bytes[:-1].split(b"\n") if journal_bytes else []
        )
        torn_tail = None

    records: list[dict] = []
    for line_number, line in enumerate(complete_lines, start=1):
        if not line:
            raise ValueError(
                f"control journal line {line_number} is empty"
            )
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"control journal line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"control journal line {line_number} is not an object"
            )
        if record.get("sequence") != line_number:
            raise ValueError("control journal sequence is not contiguous")
        records.append(record)

    sequence = state.get("sequence", 0)
    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 0
    ):
        raise ValueError("control state sequence is invalid")
    last_event = state.get("last_event")
    if torn_tail is not None:
        canonical_last = (
            json.dumps(last_event, sort_keys=True) + "\n"
        ).encode("utf-8") if isinstance(last_event, dict) else b""
        if (
            not torn_tail
            or len(records) != sequence - 1
            or not isinstance(last_event, dict)
            or last_event.get("sequence") != sequence
            or not canonical_last[:-1].startswith(torn_tail)
        ):
            raise ValueError("control journal has an unprovable torn tail")
        complete_prefix = journal_bytes[:-len(torn_tail)]
        _atomic_replace_control_bytes(
            journal_path,
            complete_prefix + canonical_last,
        )
        records.append(last_event)

    if len(records) == sequence:
        if sequence and records[-1] != last_event:
            raise ValueError("control state and journal last event disagree")
        return
    if (
        len(records) == sequence - 1
        and isinstance(last_event, dict)
        and last_event.get("sequence") == sequence
    ):
        _append_control_event(journal_path, last_event)
        return
    raise ValueError("control state and journal sequence disagree")


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
    ):
        raise ValueError("control plan requires two distinct adapter_subdirs")
    canonical_subdirs = [_canonical_adapter_subdir(value) for value in subdirs]
    if len(set(canonical_subdirs)) != 2:
        raise ValueError("control plan requires two distinct adapter_subdirs")
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
        "adapter_subdirs": canonical_subdirs,
        "gate_overrides": dict(gate_overrides),
        "resource_summary": dict(resource_summary),
    }


def _default_control_stage_api() -> dict[str, object]:
    return {
        "probe": probe.remote,
        "verify_volume_artifact": verify_volume_artifact.remote,
        "freeze_model": freeze_model.remote,
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
    validate_result,
) -> dict:
    if not callable(validate_result):
        raise TypeError(f"control stage {name} requires a result validator")
    completed = state.setdefault("completed", {})
    attempts = state.setdefault("stage_attempts", {})
    if name in completed:
        result = completed[name]
        try:
            _validate_control_stage_result(name, result, validate_result)
        except Exception as exc:
            completed.pop(name, None)
            prior_attempt = attempts.get(name, 0)
            if (
                not isinstance(prior_attempt, int)
                or isinstance(prior_attempt, bool)
                or prior_attempt < 1
            ):
                prior_attempt = 1
                attempts[name] = prior_attempt
            state["status"] = "running"
            state["last_error"] = {
                "stage": name,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _record_control_event(
                state_path,
                state,
                event="stage_checkpoint_invalid",
                stage=name,
                detail={"attempt": prior_attempt, **state["last_error"]},
            )
        else:
            return result
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
        _validate_control_stage_result(name, result, validate_result)
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


def _validate_control_stage_result(
    stage: str,
    result,
    validate_result,
) -> None:
    """Apply generic and stage-specific success checks before checkpointing."""
    if not isinstance(result, dict):
        raise TypeError(f"control stage {stage} returned a non-object")
    if result.get("ok") is False:
        raise ValueError(
            f"control stage {stage} returned ok=false: "
            f"{result.get('error') or result.get('errors')}"
        )
    if result.get("passed") is False:
        raise ValueError(
            f"control stage {stage} returned passed=false: "
            f"{result.get('error') or result.get('errors')}"
        )
    validate_result(result)


def _require_control_sha(
    stage: str,
    result: dict,
    *,
    field: str,
    expected: str | None = None,
) -> str:
    value = result.get(field)
    if not _is_control_sha(value):
        raise ValueError(
            f"control stage {stage} returned a noncanonical {field}"
        )
    if expected is not None and value != expected:
        raise ValueError(
            f"control stage {stage} returned {field}={value!r}, "
            f"expected {expected!r}"
        )
    return value


def _is_control_sha(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_final_control_run_id(value) -> bool:
    prefix = "final_modal_"
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and len(value) == len(prefix) + 12
        and all(ch in "0123456789abcdef" for ch in value[len(prefix):])
    )


def _require_control_bool(stage: str, result: dict, field: str) -> bool:
    value = result.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"control stage {stage} returned an invalid {field}")
    return value


def _require_control_count(
    stage: str,
    result: dict,
    field: str,
    *,
    positive: bool = False,
) -> int:
    value = result.get(field)
    lower_bound = 1 if positive else 0
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < lower_bound
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(
            f"control stage {stage} returned a non-{qualifier} {field}"
        )
    return value


def _validate_verified_artifact_result(
    stage: str,
    result: dict,
    *,
    expected_id: str,
    require_myopic: bool,
) -> None:
    if result.get("ok") is not True or result.get("mismatches") != []:
        raise ValueError(f"control stage {stage} did not verify cleanly")
    _require_control_sha(stage, result, field="id", expected=expected_id)
    _require_control_count(stage, result, "n_files", positive=True)
    if require_myopic:
        _require_control_sha(stage, result, field="myopic_artifact_sha256")


def _validate_probe_result(result: dict, package_names: tuple[str, ...]) -> None:
    stage = "environment_probe"
    python_version = result.get("python")
    versions = result.get("package_versions")
    if not isinstance(python_version, str) or not python_version:
        raise ValueError("environment probe returned an invalid Python version")
    if not isinstance(versions, dict) or set(versions) != set(package_names):
        raise ValueError("environment probe returned an incomplete package set")
    if not all(isinstance(value, str) and value for value in versions.values()):
        raise ValueError(f"control stage {stage} returned an invalid package version")


def _validate_model_result(result: dict) -> None:
    _require_control_sha("freeze_model", result, field="model_id")
    _require_control_bool("freeze_model", result, "cached")


def _validate_adapter_result(
    stage: str,
    result: dict,
    *,
    expected_subdir: str,
    expected_id: str | None = None,
    expected_source_id: str | None = None,
    expected_raw_id: str | None = None,
    expected_model_id: str | None = None,
    require_fresh: bool = False,
) -> None:
    _require_control_sha(
        stage,
        result,
        field="adapter_id",
        expected=expected_id,
    )
    _require_control_sha(stage, result, field="fit_rows_sha256")
    _require_control_sha(stage, result, field="eval_rows_sha256")
    if result.get("subdir") != expected_subdir:
        raise ValueError(f"control stage {stage} returned the wrong subdir")
    _require_control_bool(stage, result, "cached")
    expected_bindings = {
        "source_manifest_id": expected_source_id,
        "raw_input_bundle_id": expected_raw_id,
        "model_snapshot_id": expected_model_id,
    }
    for field, expected in expected_bindings.items():
        if expected is not None and result.get(field) != expected:
            raise ValueError(f"control stage {stage} returned the wrong {field}")
    if require_fresh and result.get("cached") is not False:
        raise ValueError(f"control stage {stage} did not execute a fresh build")


def _validate_determinism_result(result: dict, source_id: str) -> None:
    stage = "adapter_determinism"
    if result.get("ok") is not True:
        raise ValueError("adapter determinism did not pass")
    _require_control_sha(stage, result, field="adapter_id")
    _require_control_sha(
        stage, result, field="source_manifest_id", expected=source_id
    )
    first_execution = result.get("first_build_execution_id")
    second_execution = result.get("second_build_execution_id")
    if (
        not isinstance(first_execution, str)
        or not first_execution
        or not isinstance(second_execution, str)
        or not second_execution
        or first_execution == second_execution
    ):
        raise ValueError("adapter determinism returned invalid build executions")
    _require_control_sha(stage, result, field="prerequisite_receipt_id")


def _validate_promotion_result(result: dict, adapter_id: str) -> None:
    stage = "promote_adapter"
    if result.get("canonical_subdir") != f"canonical_{adapter_id}":
        raise ValueError("adapter promotion returned the wrong destination")
    _require_control_bool(stage, result, "cached")


def _validate_fvi_result(result: dict) -> None:
    stage = "fvi_study"
    _require_control_sha(stage, result, field="fvi_study_id")
    selected = result.get("selected")
    if not isinstance(selected, dict) or set(selected) != {
        "tolerance",
        "max_iterations",
    }:
        raise ValueError("FVI stage returned an incomplete selection")
    if not isinstance(selected["tolerance"], str) or not selected["tolerance"]:
        raise ValueError("FVI stage returned an invalid tolerance")
    if (
        not isinstance(selected["max_iterations"], int)
        or isinstance(selected["max_iterations"], bool)
        or selected["max_iterations"] < 1
    ):
        raise ValueError("FVI stage returned invalid max_iterations")
    _require_control_bool(stage, result, "cached")


def _validate_bootstrap_result(
    stage: str,
    result: dict,
    replicates: int,
) -> None:
    _require_control_sha(stage, result, field="bootstrap_plan_id")
    if result.get("replicates") != replicates:
        raise ValueError(f"control stage {stage} returned the wrong replicate count")
    _require_control_count(stage, result, "n_items", positive=True)
    _require_control_bool(stage, result, "cached")


def _validate_sweep_result(
    stage: str,
    result: dict,
    *,
    run_id: str,
    require_receipt: bool,
) -> None:
    if result.get("run_id") != run_id:
        raise ValueError(f"control stage {stage} returned the wrong run_id")
    requested = _require_control_count(stage, result, "requested", positive=True)
    completed = _require_control_count(stage, result, "completed")
    skipped = _require_control_count(stage, result, "skipped")
    failed = _require_control_count(stage, result, "failed")
    if completed != requested or skipped != 0 or failed != 0:
        raise ValueError(f"control stage {stage} did not complete every cell")
    if result.get("release_status") != "VALID":
        raise ValueError(f"control stage {stage} did not produce a valid release")
    if not isinstance(result.get("family"), dict):
        raise ValueError(f"control stage {stage} returned no family result")
    if require_receipt:
        _require_control_sha(stage, result, field="prerequisite_receipt_id")


def _validate_mutation_result(result: dict, source_id: str) -> None:
    stage = "mutation_gate"
    if result.get("ok") is not True or result.get("unexpected") != []:
        raise ValueError("mutation gate did not pass cleanly")
    _require_control_count(stage, result, "n", positive=True)
    _require_control_sha(
        stage, result, field="source_manifest_id", expected=source_id
    )
    _require_control_sha(stage, result, field="prerequisite_receipt_id")


def _validate_checker_result(
    stage: str,
    result: dict,
    *,
    expected_adapter_id: str | None = None,
) -> None:
    if result.get("passed") is not True or result.get("errors") != []:
        raise ValueError(f"control stage {stage} did not pass cleanly")
    recomputed = result.get("recomputed")
    if (
        not isinstance(recomputed, dict)
        or recomputed.get("release_status") != "VALID"
    ):
        raise ValueError(
            f"control stage {stage} did not recompute a valid release"
        )
    if (
        expected_adapter_id is not None
        and recomputed.get("adapter_bundle_id") != expected_adapter_id
    ):
        raise ValueError(
            f"control stage {stage} recomputed the wrong adapter identity"
        )


def _validate_package_result(result: dict, run_id: str) -> None:
    if result.get("run_id") != run_id or result.get("packaged") is not True:
        raise ValueError("package stage did not package the expected run")


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
        ENVIRONMENT_PACKAGES,
        environment_contract_identity,
        run_spec_identity,
    )

    plan = _validate_control_plan(plan)
    if plan["source_id"] != IMAGE_SOURCE_MANIFEST_ID:
        raise ValueError(
            "control plan source_id does not match the validated Modal image source"
        )
    state_path = Path(state_path)
    digest = _control_plan_digest(plan)
    api = stage_api or _default_control_stage_api()
    if resume:
        state = _load_control_json(state_path)
        _reconcile_control_journal(state_path, state)
        if state.get("plan_digest") != digest or state.get("plan") != plan:
            raise ValueError("resume control plan does not match durable state")
        if state.get("schema_version") != 2:
            raise ValueError("unsupported control-state schema")
        if state.get("status") in {"completed", "recovery_required"}:
            stored_result = state.get("result")
            validator = api.get("validate") if isinstance(api, dict) else None
            if (
                not isinstance(stored_result, dict)
                or not _is_final_control_run_id(stored_result.get("run_id"))
                or not _is_control_sha(stored_result.get("adapter_id"))
                or not callable(validator)
            ):
                state["status"] = "recovery_required"
                state["last_error"] = {
                    "stage": "completed_resume_revalidation",
                    "type": "RecoveryRequired",
                    "message": "completed state cannot re-prove the final package",
                }
                _record_control_event(
                    state_path,
                    state,
                    event="control_recovery_required",
                    stage="validate_package",
                    detail=state["last_error"],
                )
                return state
            try:
                current_validation = validator(
                    stored_result["run_id"],
                    stored_result["adapter_id"],
                    True,
                    True,
                )
                _validate_control_stage_result(
                    "completed_resume_revalidation",
                    current_validation,
                    lambda result: _validate_checker_result(
                        "completed_resume_revalidation",
                        result,
                        expected_adapter_id=stored_result["adapter_id"],
                    ),
                )
            except BaseException as exc:
                state["status"] = "recovery_required"
                state["last_error"] = {
                    "stage": "completed_resume_revalidation",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                _record_control_event(
                    state_path,
                    state,
                    event="control_recovery_required",
                    stage="validate_package",
                    detail=state["last_error"],
                )
                return state
            stored_result["validation"] = current_validation
            state["status"] = "completed"
            state.pop("last_error", None)
            _record_control_event(
                state_path,
                state,
                event="control_revalidated",
                stage="validate_package",
                detail={"run_id": stored_result["run_id"]},
            )
            return state
    else:
        if state_path.exists() or state_path.is_symlink():
            raise FileExistsError("fresh control state already exists")
        state = {
            "schema_version": 2,
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

    required_api = {
        "probe",
        "verify_volume_artifact",
        "freeze_model",
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
    _run_control_stage(
        state_path,
        state,
        name="verify_source",
        invoke=lambda _: api["verify_volume_artifact"](
            f"inputs/source_{source_id}",
            "source",
        ),
        validate_result=lambda result: _validate_verified_artifact_result(
            "verify_source",
            result,
            expected_id=source_id,
            require_myopic=False,
        ),
    )
    raw_check = _run_control_stage(
        state_path,
        state,
        name="verify_raw",
        invoke=lambda _: api["verify_volume_artifact"](
            f"inputs/raw_{raw_id}",
            "raw",
        ),
        validate_result=lambda result: _validate_verified_artifact_result(
            "verify_raw",
            result,
            expected_id=raw_id,
            require_myopic=True,
        ),
    )
    myopic_sha256 = raw_check["myopic_artifact_sha256"]

    probe_result = _run_control_stage(
        state_path,
        state,
        name="environment_probe",
        invoke=lambda _: api["probe"](),
        validate_result=lambda result: _validate_probe_result(
            result,
            ENVIRONMENT_PACKAGES,
        ),
    )
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
        validate_result=_validate_model_result,
    )
    model_id = model_result["model_id"]

    first_subdir, second_subdir = plan["adapter_subdirs"]
    determinism = _run_control_stage(
        state_path,
        state,
        name="adapter_determinism",
        invoke=lambda _: api["adapter_determinism_receipt"](
            first_subdir,
            second_subdir,
            source_id,
            raw_id,
            model_id,
            bool(plan["gate_overrides"].get("allow_low_mc_retention", False)),
        ),
        validate_result=lambda result: _validate_determinism_result(
            result,
            source_id,
        ),
    )
    adapter_id = determinism["adapter_id"]
    determinism_receipt_id = determinism["prerequisite_receipt_id"]
    determinism_bindings = {
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
        "adapter_bundle_id": adapter_id,
    }

    _run_control_stage(
        state_path,
        state,
        name="promote_adapter",
        invoke=lambda _: api["promote_adapter"](first_subdir, adapter_id),
        validate_result=lambda result: _validate_promotion_result(
            result,
            adapter_id,
        ),
    )
    fvi_result = _run_control_stage(
        state_path,
        state,
        name="fvi_study",
        invoke=lambda _: api["fvi_study"](adapter_id),
        validate_result=_validate_fvi_result,
    )
    fvi_id = fvi_result["fvi_study_id"]
    selected = fvi_result["selected"]

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
        validate_result=lambda result: _validate_bootstrap_result(
            "smoke_bootstrap",
            result,
            100,
        ),
    )
    smoke_bootstrap_id = smoke_bootstrap["bootstrap_plan_id"]
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
        validate_result=lambda result: _validate_sweep_result(
            "smoke_sweep",
            result,
            run_id=smoke_run_id,
            require_receipt=True,
        ),
    )
    smoke_receipt_id = smoke_result["prerequisite_receipt_id"]

    mutation = _run_control_stage(
        state_path,
        state,
        name="mutation_gate",
        invoke=lambda _: api["mutation_gate"](
            json.dumps(common_bindings, sort_keys=True)
        ),
        validate_result=lambda result: _validate_mutation_result(
            result,
            source_id,
        ),
    )
    mutation_receipt_id = mutation["prerequisite_receipt_id"]

    final_bootstrap = _run_control_stage(
        state_path,
        state,
        name="final_bootstrap",
        invoke=lambda _: api["bootstrap_plan"](adapter_id, 1000),
        validate_result=lambda result: _validate_bootstrap_result(
            "final_bootstrap",
            result,
            1000,
        ),
    )
    final_bootstrap_id = final_bootstrap["bootstrap_plan_id"]
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
        validate_result=lambda result: _validate_sweep_result(
            "final_sweep",
            result,
            run_id=final_run_id,
            require_receipt=False,
        ),
    )
    _run_control_stage(
        state_path,
        state,
        name="validate_unpacked",
        invoke=lambda _: api["validate"](
            final_run_id,
            adapter_id,
            True,
            False,
        ),
        validate_result=lambda result: _validate_checker_result(
            "validate_unpacked",
            result,
            expected_adapter_id=adapter_id,
        ),
    )
    _run_control_stage(
        state_path,
        state,
        name="package",
        invoke=lambda _: api["package"](final_run_id),
        validate_result=lambda result: _validate_package_result(
            result,
            final_run_id,
        ),
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
        validate_result=lambda result: _validate_checker_result(
            "validate_package",
            result,
            expected_adapter_id=adapter_id,
        ),
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
    if state.get("status") != "completed":
        raise RuntimeError(
            "control plane requires recovery: "
            + str(state.get("last_error", {}).get("message", "unknown error"))
        )
    print(json.dumps(state["result"], indent=2, sort_keys=True))


@app.local_entrypoint()
def probe_main():
    from scripts.stopdff_v5.identity import canonical_bytes

    print(canonical_bytes(probe.remote()).decode("utf-8"))
