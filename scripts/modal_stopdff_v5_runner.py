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
import re
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath

import modal

APP_NAME = os.environ.get("STOPDFF_V5_APP_NAME", "cs321m-stopdff-v5")
if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", APP_NAME) is None:
    raise RuntimeError("STOPDFF_V5_APP_NAME is not a canonical Modal app name")
VOLUME_NAME = "cs321m-stopdff-artifacts"
MNT = "/stopdff"
REMOTE_SRC = "/root/src"
DAY = 86400
_ADAPTER_COMPONENT_MAX_BYTES = 255
_SOURCE_ID_ENV = "STOPDFF_V5_IMAGE_SOURCE_MANIFEST_ID"
_MODAL_IS_LOCAL = modal.is_local()
SOURCE_BUNDLE_DIR = (
    os.environ.get("STOPDFF_V5_SOURCE_DIR", "")
    if _MODAL_IS_LOCAL
    else ""
)


def _materialize_image_source(
    source_bundle: Path,
) -> tuple[tempfile.TemporaryDirectory, Path, dict]:
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
    owner = tempfile.TemporaryDirectory(prefix="stopdff_v5_image_source_")
    staged_bundle = Path(owner.name)
    try:
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
    except BaseException:
        owner.cleanup()
        raise
    return owner, staged_bundle / "source", staged_manifest


if _MODAL_IS_LOCAL:
    if not SOURCE_BUNDLE_DIR:
        raise RuntimeError(
            "STOPDFF_V5_SOURCE_DIR must point to the frozen source-snapshot bundle"
        )
    (
        _IMAGE_SOURCE_OWNER,
        _IMAGE_SOURCE_DIR,
        _IMAGE_SOURCE_MANIFEST,
    ) = _materialize_image_source(Path(SOURCE_BUNDLE_DIR))
    SOURCE_DIR = str(_IMAGE_SOURCE_DIR)
    IMAGE_SOURCE_MANIFEST_ID = _IMAGE_SOURCE_MANIFEST["id"]
else:
    _IMAGE_SOURCE_OWNER = None
    _IMAGE_SOURCE_DIR = Path(REMOTE_SRC)
    _IMAGE_SOURCE_MANIFEST = None
    SOURCE_DIR = REMOTE_SRC
    IMAGE_SOURCE_MANIFEST_ID = os.environ.get(_SOURCE_ID_ENV, "")

if (
    not isinstance(IMAGE_SOURCE_MANIFEST_ID, str)
    or re.fullmatch(r"[0-9a-f]{64}", IMAGE_SOURCE_MANIFEST_ID) is None
):
    raise RuntimeError(
        f"{_SOURCE_ID_ENV} must be the validated source manifest ID"
    )


def _require_image_source_id(source_id: object) -> None:
    """Reject a stage claim that is not bound to the source in its image."""
    if source_id != IMAGE_SOURCE_MANIFEST_ID:
        raise ValueError(
            "stage source_id does not match the validated Modal image source"
        )


_PIP = [
    "numpy>=1.26,<3", "scipy>=1.11", "scikit-learn>=1.3", "pandas>=2.1",
    "matplotlib>=3.7", "sentence-transformers>=2.7", "torch>=2.0",
    "huggingface_hub>=0.23",
]
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_PIP)
    .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "HF_HUB_DISABLE_TELEMETRY": "1",
          "PYTHONDONTWRITEBYTECODE": "1",
          "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": REMOTE_SRC,
          _SOURCE_ID_ENV: IMAGE_SOURCE_MANIFEST_ID,
          "STOPDFF_V5_APP_NAME": APP_NAME})
)
if _MODAL_IS_LOCAL:
    _image = _image.add_local_dir(
        SOURCE_DIR,
        remote_path=REMOTE_SRC,
        copy=True,
    )

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
    if "\0" in value:
        raise ValueError("adapter subdir must not contain NUL")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or len(parsed.parts) != 1
        or str(parsed) != value
    ):
        raise ValueError(f"unsafe or noncanonical adapter subdir: {value!r}")
    if len(value.encode("utf-8")) > _ADAPTER_COMPONENT_MAX_BYTES:
        raise ValueError("adapter subdir must be at most 255 UTF-8 bytes")
    return value


def _retry_adapter_subdir(base: str, attempt: int) -> str:
    """Derive a stable retry component within the internal byte contract."""
    attempt_text = str(attempt)
    readable_suffix = f"__attempt_{attempt_text}"
    candidate = f"{base}{readable_suffix}"
    if len(candidate.encode("utf-8")) <= _ADAPTER_COMPONENT_MAX_BYTES:
        return candidate

    digest = hashlib.sha256(
        f"{base}\0{attempt_text}".encode("utf-8")
    ).hexdigest()[:16]
    if len(readable_suffix.encode("utf-8")) <= 48:
        suffix = f"{readable_suffix}_{digest}"
    else:
        suffix = f"__attempt_{digest}"
    prefix_budget = _ADAPTER_COMPONENT_MAX_BYTES - len(suffix.encode("utf-8"))
    prefix = base.encode("utf-8")[:prefix_budget].decode(
        "utf-8",
        errors="ignore",
    )
    return f"{prefix}{suffix}"


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
    from scripts.stopdff_v5.content_manifest import git_mode_for_path
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
            or git_mode_for_path(runtime_path) != entry["mode"]
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


def _canonical_bootstrap_plan_path(root: Path) -> Path:
    """Return the durable plan path only for a canonical cache entry."""
    plan_path = root / "bootstrap_plan.json"
    if (
        root.is_symlink()
        or not root.is_dir()
        or plan_path.is_symlink()
        or not plan_path.is_file()
    ):
        raise ValueError("bootstrap cache is incomplete or noncanonical")
    return plan_path


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


def _validated_local_input_bundle(bundle: Path, kind: str) -> tuple[dict, str]:
    """Validate a complete local source/raw bundle before any host upload."""
    from scripts.stopdff_v5.content_manifest import (
        validate_bound_content_manifest,
    )

    bundle = Path(bundle)
    if kind == "source":
        manifest_name = "source_manifest.json"
        manifest = validate_bound_content_manifest(
            bundle,
            manifest_name=manifest_name,
            expected_id=None,
            file_key="files",
            name_key="path",
            content_subdir="source",
            expected_kind="source_snapshot",
        )
    elif kind == "raw":
        manifest_name = "raw_input_manifest.json"
        manifest = validate_bound_content_manifest(
            bundle,
            manifest_name=manifest_name,
            expected_id=None,
            file_key="files",
            name_key="role",
            content_subdir="raw",
            expected_kind="raw_input_bundle",
            require_semantic_pass=True,
        )
    else:
        raise ValueError(f"unsupported input-bundle kind {kind!r}")
    return manifest, manifest_name


def _materialize_raw_upload_bundle(
    bundle: Path,
    manifest: dict,
) -> tuple[tempfile.TemporaryDirectory, Path]:
    """Return a private, flat, revalidated copy for the Modal Volume."""
    from scripts.stopdff_v5.content_manifest import (
        validate_bound_content_manifest,
    )

    owner = tempfile.TemporaryDirectory(prefix="stopdff_v5_raw_upload_")
    staged_bundle = Path(owner.name) / "raw"
    try:
        shutil.copytree(bundle / "raw", staged_bundle)
        shutil.copy2(
            bundle / "raw_input_manifest.json",
            staged_bundle / "raw_input_manifest.json",
        )
        staged_manifest = validate_bound_content_manifest(
            staged_bundle,
            manifest_name="raw_input_manifest.json",
            expected_id=manifest["id"],
            file_key="files",
            name_key="role",
            expected_kind="raw_input_bundle",
            require_semantic_pass=True,
        )
        if staged_manifest != manifest:
            raise ValueError("raw-input manifest changed during materialization")
    except BaseException:
        owner.cleanup()
        raise
    return owner, staged_bundle


def _is_volume_not_found(exc: BaseException) -> bool:
    """Recognize only the SDK/local missing-path boundary."""
    return isinstance(exc, FileNotFoundError) or type(exc).__name__ == "NotFoundError"


def _stage_one_input_bundle(
    bundle: Path,
    kind: str,
    *,
    volume=None,
    verifier=None,
) -> dict:
    """Create-once stage, remotely verify, and host-read one input bundle."""
    from scripts.stopdff_v5.identity import sha256_bytes

    bundle = Path(bundle)
    manifest, manifest_name = _validated_local_input_bundle(bundle, kind)
    manifest_id = manifest["id"]
    remote_dir = f"inputs/{kind}_{manifest_id}"
    target_volume = volume or vol
    remote_verifier = verifier or verify_volume_artifact.remote
    try:
        existing_entries = list(
            target_volume.listdir(remote_dir, recursive=True)
        )
    except BaseException as exc:
        if not _is_volume_not_found(exc):
            raise
        existing_entries = []

    if existing_entries:
        status = "cached"
    else:
        upload_owner = None
        upload_bundle = bundle
        try:
            if kind == "raw":
                upload_owner, upload_bundle = _materialize_raw_upload_bundle(
                    bundle,
                    manifest,
                )
            with target_volume.batch_upload(force=False) as batch:
                batch.put_directory(str(upload_bundle), remote_dir)
        finally:
            if upload_owner is not None:
                upload_owner.cleanup()
        status = "created"

    verified = remote_verifier(remote_dir, kind)
    if (
        not isinstance(verified, dict)
        or verified.get("ok") is not True
        or verified.get("mismatches") != []
        or verified.get("id") != manifest_id
        or not isinstance(verified.get("n_files"), int)
        or isinstance(verified.get("n_files"), bool)
        or verified.get("n_files") < 1
    ):
        raise ValueError(f"staged {kind} bundle failed remote verification")

    local_manifest_bytes = (bundle / manifest_name).read_bytes()
    remote_manifest_bytes = b"".join(
        target_volume.read_file(f"{remote_dir}/{manifest_name}")
    )
    if remote_manifest_bytes != local_manifest_bytes:
        raise ValueError(f"staged {kind} manifest readback mismatch")
    return {
        "status": status,
        "kind": kind,
        "id": manifest_id,
        "remote_dir": remote_dir,
        "n_files": verified["n_files"],
        "manifest_sha256": sha256_bytes(remote_manifest_bytes),
    }


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
            expected_kind="model_snapshot",
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
        expected_kind="model_snapshot",
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
        expected_kind="model_snapshot",
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
    if out.exists() or out.is_symlink():
        try:
            plan_path = _canonical_bootstrap_plan_path(out)
        except ValueError as exc:
            raise FileExistsError(
                "bootstrap destination is incomplete or noncanonical"
            ) from exc
        existing = checker.load_json(plan_path)
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
    from scripts.stopdff_v5.content_manifest import git_mode_for_path
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
    plan_path = _canonical_bootstrap_plan_path(
        Path(_p("bootstrap", bootstrap_plan_id))
    )
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
            or git_mode_for_path(runtime_path) != entry["mode"]
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
    resource_summary = checker.load_json(run_root / "resource_summary.json")
    if not isinstance(resource_summary, dict):
        raise ValueError("run resource summary must contain an object")
    writers.package_run(
        run_root,
        agg,
        resource_summary=resource_summary,
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
    with tempfile.TemporaryDirectory(
        prefix="stopdff_v5_mutation_selftest_"
    ) as selftest_dir:
        ok, results = selftest.run_self_test(Path(selftest_dir))
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


def _canonical_assurance_tag(tag: object) -> str:
    if (
        not isinstance(tag, str)
        or not re.fullmatch(r"[0-9a-f][0-9a-f-]{7,63}", tag)
        or ".." in tag
    ):
        raise ValueError("assurance tag must be 8-64 lowercase hex/hyphen characters")
    return tag


def _modal_runtime_identity() -> dict:
    """Return stable call/input identifiers plus this container hostname."""
    import socket

    def required_call(name: str) -> str:
        function = getattr(modal, name, None)
        value = function() if callable(function) else None
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"Modal runtime did not expose a nonempty {name}")
        return value

    return {
        "container_hostname": socket.gethostname(),
        "function_call_id": required_call("current_function_call_id"),
        "input_id": required_call("current_input_id"),
    }


def _assurance_sweep_context(
    tag: str,
    *,
    attempt: int,
    resume: bool,
    commit_fn,
):
    """Build a zero-cell context that still uses the production attempt protocol."""
    from scripts.stopdff_v5.bootstrap import build_bootstrap_plan, plan_identity
    from scripts.stopdff_v5.identity import compute_id
    from scripts.stopdff_v5.sweep import SweepContext

    def oracle_id(label: str) -> str:
        return compute_id({"kind": "modal_assurance", "tag": tag, "label": label})

    plan = build_bootstrap_plan(["oracle-item"], replicates=1, seed=1)
    bootstrap_plan_id = compute_id(plan_identity(plan))
    adapter_id = oracle_id("adapter")
    myopic_id = oracle_id("myopic")
    producer_hashes = {
        "checker.py": oracle_id("checker"),
        "sweep.py": oracle_id("sweep"),
    }
    run_spec = {
        "profile_variant": "smoke",
        "identity": {
            "source_manifest_id": IMAGE_SOURCE_MANIFEST_ID,
            "raw_input_bundle_id": oracle_id("raw"),
            "model_snapshot_id": oracle_id("model"),
            "adapter_bundle_id": adapter_id,
            "fvi_study_id": oracle_id("fvi"),
            "bootstrap_plan_id": bootstrap_plan_id,
            "environment_contract_id": oracle_id("environment"),
            "resource_summary_id": oracle_id("resources"),
        },
        "evidence_roots": {
            "myopic_artifact_sha256": myopic_id,
            "producer_hashes": producer_hashes,
        },
        "fvi_selected": {"tolerance": "1e-6", "max_iterations": 1},
        "gate": {
            "allow_low_mc_retention": False,
            "allow_incomplete_mc_coverage": False,
        },
    }
    run_spec_id = compute_id(run_spec)
    return SweepContext(
        rows=[],
        calibration_json=None,
        run_spec=run_spec,
        run_spec_id=run_spec_id,
        bootstrap_plan=plan,
        output_dir=Path(_p("pilots", tag, "run")),
        fvi_tolerance="1e-6",
        fvi_max_iterations=1,
        backend="modal",
        profile_variant="smoke",
        adapter_bundle_id=adapter_id,
        adapter_fit_rows_sha256=oracle_id("fit-rows"),
        adapter_eval_rows_sha256=oracle_id("eval-rows"),
        myopic_artifact_sha256=myopic_id,
        producer_hashes=producer_hashes,
        cells=[],
        commit_fn=commit_fn,
        resource_summary={"backend": "modal", "assurance_tag": tag},
        attempt={
            "attempt": attempt,
            "mode": "resume" if resume else "fresh",
            "command": ["modal_assurance"] + (["--resume"] if resume else []),
        },
        resume=resume,
    )


def _assurance_observation(tag: str) -> dict:
    """Read and validate the durable attempt records for one assurance tag."""
    from scripts.stopdff_v5.attempt_history import load_attempt_history
    from scripts.stopdff_v5.identity import (
        loads_no_duplicate_keys,
        sha256_bytes,
        sha256_file,
    )

    root = Path(_p("pilots", tag))
    run_root = root / "run"
    _, attempts = load_attempt_history(run_root / "attempts.jsonl")
    results: dict[str, dict] = {}
    result_sha256: dict[str, str] = {}
    results_dir = run_root / "attempt_results"
    if results_dir.is_dir():
        for path in sorted(results_dir.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise ValueError("assurance attempt result path is noncanonical")
            value = loads_no_duplicate_keys(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("assurance attempt result is not an object")
            results[path.name] = value
            result_sha256[path.name] = sha256_file(path)
    arm_path = root / "crash_arm.json"
    if arm_path.is_symlink() or not arm_path.is_file():
        raise ValueError("assurance crash arm is missing")
    arm = loads_no_duplicate_keys(arm_path.read_text(encoding="utf-8"))
    if not isinstance(arm, dict):
        raise ValueError("assurance crash arm is invalid")

    def identity_file(name: str, fields: set[str]) -> tuple[dict, str]:
        path = run_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"assurance {name} is missing or noncanonical")
        data = path.read_bytes()
        try:
            value = loads_no_duplicate_keys(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"assurance {name} is invalid JSON") from exc
        expected_bytes = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or data != expected_bytes
        ):
            raise ValueError(f"assurance {name} is noncanonical")
        return value, sha256_bytes(data)

    run_spec, run_spec_sha256 = identity_file(
        "run_spec.json",
        {"id", "identity"},
    )
    bootstrap_plan, bootstrap_plan_sha256 = identity_file(
        "bootstrap_plan.json",
        {"id", "identity", "item_ids"},
    )
    return {
        "tag": tag,
        "attempts": attempts,
        "results": results,
        "result_sha256": result_sha256,
        "crash_arm": arm,
        "attempts_sha256": sha256_file(run_root / "attempts.jsonl"),
        "crash_arm_sha256": sha256_file(arm_path),
        "run_spec": run_spec,
        "run_spec_sha256": run_spec_sha256,
        "bootstrap_plan": bootstrap_plan,
        "bootstrap_plan_sha256": bootstrap_plan_sha256,
    }


def _assurance_expected_evidence(tag: str) -> dict:
    """Derive the exact canary histories and immutable run identities."""
    from scripts.stopdff_v5.bootstrap import plan_identity
    from scripts.stopdff_v5.identity import compute_id

    first_context = _assurance_sweep_context(
        tag,
        attempt=1,
        resume=False,
        commit_fn=lambda: None,
    )
    second_context = _assurance_sweep_context(
        tag,
        attempt=2,
        resume=True,
        commit_fn=lambda: None,
    )

    def attempt_record(context) -> dict:
        return {
            **context.attempt,
            "run_spec_id": context.run_spec_id,
            "adapter_id": context.adapter_bundle_id,
            "bootstrap_plan_id": compute_id(
                plan_identity(context.bootstrap_plan)
            ),
            "state": "started",
        }

    first = attempt_record(first_context)
    second = attempt_record(second_context)
    return {
        "first_context": first_context,
        "second_context": second_context,
        "first_attempt": first,
        "second_attempt": second,
        "interrupted": {
            "attempt": 1,
            "state": "interrupted",
            "run_spec_id": first_context.run_spec_id,
            "reason": "terminal_result_missing_at_resume",
        },
        "completed": {
            "attempt": 2,
            "state": "completed",
            "run_spec_id": first_context.run_spec_id,
            "completed": 0,
            "failed": 0,
        },
        "run_spec": {
            "id": first_context.run_spec_id,
            "identity": first_context.run_spec,
        },
        "bootstrap_plan": {
            "id": first["bootstrap_plan_id"],
            "identity": plan_identity(first_context.bootstrap_plan),
            "item_ids": first_context.bootstrap_plan.item_ids,
        },
    }


def _assurance_phase_state(tag: str, observation: dict) -> tuple[str, dict]:
    """Classify only exact initial/classified/finished durable states."""
    expected = _assurance_expected_evidence(tag)
    arm = observation.get("crash_arm")
    if not isinstance(arm, dict) or set(arm) != {
        "tag",
        "source_manifest_id",
        "runtime",
        "exit_code",
        "armed_after_attempt_start_commit",
    }:
        raise ValueError("assurance crash arm schema is invalid")
    runtime = arm.get("runtime")
    if (
        arm.get("tag") != tag
        or arm.get("source_manifest_id") != IMAGE_SOURCE_MANIFEST_ID
        or arm.get("exit_code") != 91
        or arm.get("armed_after_attempt_start_commit") is not True
        or not isinstance(runtime, dict)
        or set(runtime)
        != {"container_hostname", "function_call_id", "input_id"}
        or not all(
            isinstance(runtime[field], str) and runtime[field]
            for field in runtime
        )
    ):
        raise ValueError("assurance crash arm bindings are invalid")
    if observation.get("run_spec") != expected["run_spec"]:
        raise ValueError("assurance run spec does not match the canary context")
    if observation.get("bootstrap_plan") != expected["bootstrap_plan"]:
        raise ValueError(
            "assurance bootstrap plan does not match the canary context"
        )

    attempts = observation.get("attempts")
    results = observation.get("results")
    if attempts == [expected["first_attempt"]] and results == {}:
        return "initial", expected
    if attempts == [expected["first_attempt"]] and results == {
        "1.json": expected["interrupted"]
    }:
        return "classified", expected
    if attempts == [
        expected["first_attempt"],
        expected["second_attempt"],
    ] and results == {
        "1.json": expected["interrupted"],
        "2.json": expected["completed"],
    }:
        return "finished", expected
    raise ValueError("assurance durable phase state is noncanonical")


def _assurance_expected_aggregate(context, sweep_module) -> dict:
    identity = context.run_spec["identity"]
    return {
        "profile_name": sweep_module.PROFILE_NAME,
        "profile_variant": "smoke",
        "backend": "modal",
        "run_spec_id": context.run_spec_id,
        "adapter_bundle_id": context.adapter_bundle_id,
        "bootstrap_plan_id": identity["bootstrap_plan_id"],
        "fvi_study_id": identity["fvi_study_id"],
        "adapter_fit_rows_sha256": context.adapter_fit_rows_sha256,
        "adapter_eval_rows_sha256": context.adapter_eval_rows_sha256,
        "myopic_artifact_sha256": context.myopic_artifact_sha256,
        "requested": 0,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "expected_cell_keys": [],
        "fvi_selected": {"tolerance": "1e-6", "max_iterations": 1},
        "cells": {},
        "family": None,
        "gate_overrides": {
            "allow_low_mc_retention": False,
            "allow_incomplete_mc_coverage": False,
        },
        "release_status": "INVALID",
        "release_reasons": [
            "bootstrap evidence invalid",
            "family-max evidence invalid",
        ],
    }


def _load_assurance_aggregate(tag: str, context, sweep_module) -> dict:
    from scripts.stopdff_v5.identity import loads_no_duplicate_keys

    path = Path(_p("pilots", tag, "run", "aggregate.json"))
    if path.is_symlink() or not path.is_file():
        raise ValueError("assurance aggregate is missing or noncanonical")
    data = path.read_bytes()
    try:
        aggregate = loads_no_duplicate_keys(data.decode("utf-8"))
        expected_bytes = (
            json.dumps(
                aggregate,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("assurance aggregate is invalid JSON") from exc
    if (
        data != expected_bytes
        or aggregate != _assurance_expected_aggregate(context, sweep_module)
    ):
        raise ValueError("assurance aggregate is noncanonical")
    return aggregate


@app.function(volumes={MNT: vol}, timeout=300, max_containers=1)
def recovery_assurance(tag: str, phase: str) -> dict:
    """One-shot hard-exit, classification, finish, and readback canary."""
    from scripts.stopdff_v5 import sweep

    tag = _canonical_assurance_tag(tag)
    if phase not in {"crash", "classify", "finish", "verify"}:
        raise ValueError("unknown recovery-assurance phase")
    vol.reload()
    root = Path(_p("pilots", tag))
    arm_path = root / "crash_arm.json"
    runtime = _modal_runtime_identity()

    if phase == "crash" and arm_path.is_file():
        observation = _assurance_observation(tag)
        state, _ = _assurance_phase_state(tag, observation)
        if state != "initial":
            raise ValueError("rescheduled crash call found a non-initial state")
        first_runtime = observation["crash_arm"]["runtime"]
        if (
            first_runtime.get("container_hostname")
            == runtime.get("container_hostname")
        ):
            raise ValueError("hard-exit call did not cross a container boundary")
        if first_runtime.get("input_id") != runtime.get("input_id"):
            raise ValueError("rescheduled hard-exit call changed its input identity")
        if first_runtime.get("function_call_id") != runtime.get(
            "function_call_id"
        ):
            raise ValueError(
                "rescheduled hard-exit call changed its function-call identity"
            )
        return {
            "phase": "crash_rescheduled",
            "runtime": runtime,
            "observation": observation,
        }

    if phase == "crash":
        if root.exists() or root.is_symlink():
            raise FileExistsError("fresh assurance namespace already exists")
        root.mkdir(parents=True)
        commit_count = 0

        def commit_then_crash_once() -> None:
            nonlocal commit_count
            commit_count += 1
            vol.commit()
            if commit_count != 1:
                return
            sweep.atomic_write_json(
                arm_path,
                {
                    "tag": tag,
                    "source_manifest_id": IMAGE_SOURCE_MANIFEST_ID,
                    "runtime": runtime,
                    "exit_code": 91,
                    "armed_after_attempt_start_commit": True,
                },
            )
            vol.commit()
            os._exit(91)

        context = _assurance_sweep_context(
            tag,
            attempt=1,
            resume=False,
            commit_fn=commit_then_crash_once,
        )
        sweep.run_sweep(context)
        raise AssertionError("hard-exit assurance returned without exiting")

    if not arm_path.is_file():
        raise ValueError("assurance crash phase has not committed its arm")
    before = _assurance_observation(tag)
    durable_state, expected = _assurance_phase_state(tag, before)
    if phase == "verify":
        if durable_state != "finished":
            raise ValueError("verify phase requires an exact finished state")
        _load_assurance_aggregate(
            tag,
            expected["second_context"],
            sweep,
        )
        return {
            "phase": "verify",
            "runtime": runtime,
            "observation": before,
        }

    if phase == "classify":
        if durable_state == "classified":
            return {
                "phase": "classified",
                "runtime": runtime,
                "observation": before,
            }
        if durable_state != "initial":
            raise ValueError("classify phase requires an exact initial state")
        commit_count = 0

        class ClassificationCommitted(BaseException):
            pass

        def commit_then_stop() -> None:
            nonlocal commit_count
            commit_count += 1
            vol.commit()
            if commit_count == 1:
                raise ClassificationCommitted()

        context = _assurance_sweep_context(
            tag,
            attempt=2,
            resume=True,
            commit_fn=commit_then_stop,
        )
        try:
            sweep.run_sweep(context)
        except ClassificationCommitted:
            observation = _assurance_observation(tag)
            classified_state, _ = _assurance_phase_state(tag, observation)
            if classified_state != "classified":
                raise ValueError("classification did not stop canonically")
            return {
                "phase": "classified",
                "runtime": runtime,
                "observation": observation,
            }
        raise AssertionError("classification phase did not stop at its commit")

    if durable_state == "finished":
        return {
            "phase": "finished",
            "runtime": runtime,
            "aggregate": _load_assurance_aggregate(
                tag,
                expected["second_context"],
                sweep,
            ),
            "observation": before,
        }
    if durable_state != "classified":
        raise ValueError("finish phase requires an exact classified state")
    interrupted_before = before["results"].get("1.json")
    interrupted_sha256_before = before["result_sha256"].get("1.json")
    context = _assurance_sweep_context(
        tag,
        attempt=2,
        resume=True,
        commit_fn=vol.commit,
    )
    aggregate = sweep.run_sweep(context)
    observation = _assurance_observation(tag)
    finished_state, _ = _assurance_phase_state(tag, observation)
    if finished_state != "finished":
        raise ValueError("finish phase did not produce an exact finished state")
    if observation["results"].get("1.json") != interrupted_before:
        raise ValueError("finish phase rewrote interruption evidence")
    if (
        observation["result_sha256"].get("1.json")
        != interrupted_sha256_before
    ):
        raise ValueError("finish phase changed interruption-result bytes")
    if aggregate != _assurance_expected_aggregate(context, sweep):
        raise ValueError("finish phase returned a noncanonical aggregate")
    if aggregate != _load_assurance_aggregate(tag, context, sweep):
        raise ValueError("finish phase aggregate readback mismatch")
    return {
        "phase": "finished",
        "runtime": runtime,
        "aggregate": aggregate,
        "observation": observation,
    }


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


def _atomic_create_control_bytes(path: Path, data: bytes) -> None:
    """Publish one fsynced control artifact without replacing any path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"create-once control artifact already exists: {path}"
            ) from exc
        os.unlink(temporary)
        temporary = ""
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary and os.path.exists(temporary):
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


_CONTROL_EVENT_NAMES = {
    "control_completed",
    "control_initialized",
    "control_recovery_required",
    "control_revalidated",
    "stage_checkpoint_invalid",
    "stage_checkpoint_refresh_required",
    "stage_completed",
    "stage_failed",
    "stage_started",
}
_CONTROL_STAGE_ORDER = (
    "verify_source",
    "verify_raw",
    "environment_probe",
    "freeze_model",
    "adapter_determinism",
    "promote_adapter",
    "fvi_study",
    "smoke_bootstrap",
    "smoke_sweep",
    "mutation_gate",
    "final_bootstrap",
    "final_sweep",
    "package",
    "validate_package",
)
_CONTROL_STAGE_NAMES = set(_CONTROL_STAGE_ORDER)
_CONTROL_RESULT_FIELDS = {
    "run_id",
    "run_spec_id",
    "adapter_id",
    "receipt_ids",
    "validation",
}


def _control_event_sha256(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _control_payload_sha256(payload: object) -> str:
    """Hash finite canonical JSON used in a durable controller checkpoint."""
    try:
        data = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("control payload is not finite canonical JSON") from exc
    return hashlib.sha256(data).hexdigest()


def _validate_control_event_record(
    record: object,
    *,
    expected_sequence: int,
    previous_record: dict | None,
) -> dict:
    """Validate one canonical, hash-linked control-journal record."""
    required = {
        "sequence",
        "event",
        "stage",
        "utc_epoch_seconds",
        "detail",
        "previous_event_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("control journal record schema is invalid")
    if record.get("sequence") != expected_sequence:
        raise ValueError("control journal sequence is not contiguous")
    event = record.get("event")
    if not isinstance(event, str) or event not in _CONTROL_EVENT_NAMES:
        raise ValueError("control journal event is unknown")
    timestamp = record.get("utc_epoch_seconds")
    if (
        not isinstance(timestamp, int)
        or isinstance(timestamp, bool)
        or timestamp < 0
    ):
        raise ValueError("control journal timestamp is invalid")
    detail = record.get("detail")
    if not isinstance(detail, dict):
        raise ValueError("control journal detail must be an object")

    expected_previous = (
        _control_event_sha256(previous_record)
        if previous_record is not None
        else None
    )
    if record.get("previous_event_sha256") != expected_previous:
        raise ValueError("control journal hash chain is invalid")
    if (
        previous_record is not None
        and timestamp < previous_record["utc_epoch_seconds"]
    ):
        raise ValueError("control journal timestamps are not monotonic")

    stage = record.get("stage")
    stage_event = event.startswith("stage_") or event in {
        "control_recovery_required",
        "control_revalidated",
    }
    if stage_event:
        if not isinstance(stage, str) or stage not in _CONTROL_STAGE_NAMES:
            raise ValueError("control journal stage is invalid")
    elif stage is not None:
        raise ValueError("control journal event must not name a stage")

    if expected_sequence == 1 and event != "control_initialized":
        raise ValueError("control journal must begin with initialization")
    if event == "control_initialized" and (
        expected_sequence != 1 or detail
    ):
        raise ValueError("control initialization event is invalid")
    detail_fields = {
        "control_completed": {"run_id", "run_spec_id", "result_sha256"},
        "control_initialized": set(),
        "control_recovery_required": {"stage", "type", "message"},
        "control_revalidated": {"run_id", "result_sha256"},
        "stage_checkpoint_invalid": {"attempt", "stage", "type", "message"},
        "stage_checkpoint_refresh_required": {"reason"},
        "stage_completed": {"attempt", "result_sha256"},
        "stage_failed": {"attempt", "stage", "type", "message"},
        "stage_started": {"attempt"},
    }
    if set(detail) != detail_fields[event]:
        raise ValueError("control journal event detail schema is invalid")
    if event in {
        "stage_checkpoint_invalid",
        "stage_completed",
        "stage_failed",
        "stage_started",
    }:
        attempt = detail.get("attempt")
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
        ):
            raise ValueError("control journal stage attempt is invalid")
    if event in {"stage_checkpoint_invalid", "stage_failed"}:
        if (
            detail.get("stage") != stage
            or not isinstance(detail.get("type"), str)
            or not detail.get("type")
            or not isinstance(detail.get("message"), str)
        ):
            raise ValueError("control journal stage failure detail is invalid")
    if event == "stage_checkpoint_refresh_required" and not isinstance(
        detail.get("reason"), str
    ):
        raise ValueError("control journal refresh detail is invalid")
    if event == "control_recovery_required" and (
        not isinstance(detail.get("stage"), str)
        or not isinstance(detail.get("type"), str)
        or not isinstance(detail.get("message"), str)
    ):
        raise ValueError("control journal recovery detail is invalid")
    if event == "control_revalidated" and not _is_final_control_run_id(
        detail.get("run_id")
    ):
        raise ValueError("control journal revalidation detail is invalid")
    if event == "control_completed" and (
        not _is_final_control_run_id(detail.get("run_id"))
        or not _is_control_sha(detail.get("run_spec_id"))
    ):
        raise ValueError("control journal completion detail is invalid")
    if event in {
        "stage_completed",
        "control_completed",
        "control_revalidated",
    } and not _is_control_sha(detail.get("result_sha256")):
        raise ValueError("control journal result digest is invalid")
    return record


def _validate_control_journal_projection(
    records: list[dict],
    state: dict,
) -> None:
    """Replay the journal's stage projection and bind it to the checkpoint."""
    attempts: dict[str, int] = {}
    active: tuple[str, int] | None = None
    completed: set[str] = set()
    completed_digests: dict[str, str] = {}
    terminal_result_digest: str | None = None
    terminal_run_id: str | None = None
    terminal_run_spec_id: str | None = None
    terminal_seen = False

    def require_completed_prefix() -> None:
        expected = set(_CONTROL_STAGE_ORDER[: len(completed)])
        if completed != expected:
            raise ValueError(
                "control journal completed stages are not a canonical prefix"
            )

    for record in records:
        event = record["event"]
        stage = record["stage"]
        detail = record["detail"]
        if terminal_seen and event.startswith("stage_"):
            raise ValueError(
                "control journal has stage activity after a terminal event"
            )
        if event == "stage_started":
            attempt = detail["attempt"]
            if attempt != attempts.get(stage, 0) + 1:
                raise ValueError("control journal stage attempts are inconsistent")
            if active is not None:
                raise ValueError("control journal has overlapping active stages")
            stage_index = _CONTROL_STAGE_ORDER.index(stage)
            if completed != set(_CONTROL_STAGE_ORDER[:stage_index]):
                raise ValueError(
                    "control journal stage start lacks its completed predecessors"
                )
            attempts[stage] = attempt
            active = (stage, attempt)
        elif event in {"stage_completed", "stage_failed"}:
            if active != (stage, detail["attempt"]):
                raise ValueError(
                    "control journal stage terminal event lacks its start"
                )
            active = None
            if event == "stage_completed":
                completed.add(stage)
                completed_digests[stage] = detail["result_sha256"]
                require_completed_prefix()
            else:
                completed.discard(stage)
                completed_digests.pop(stage, None)
        elif event == "stage_checkpoint_invalid":
            if active is not None:
                raise ValueError(
                    "control journal invalidated a checkpoint with an active stage"
                )
            if (
                stage not in completed
                or detail["attempt"] != attempts.get(stage)
            ):
                raise ValueError(
                    "control journal invalidation lacks a completed checkpoint"
                )
            completed.remove(stage)
            completed_digests.pop(stage, None)
            require_completed_prefix()
        elif event == "stage_checkpoint_refresh_required":
            if active is not None:
                raise ValueError(
                    "control journal refreshed a checkpoint with an active stage"
                )
            if stage not in completed:
                raise ValueError(
                    "control journal refresh lacks a completed checkpoint"
                )
            completed.remove(stage)
            completed_digests.pop(stage, None)
            require_completed_prefix()
        elif event in {
            "control_completed",
            "control_recovery_required",
            "control_revalidated",
        }:
            if active is not None:
                raise ValueError(
                    "control journal terminal event has an active stage"
                )
            if completed != _CONTROL_STAGE_NAMES:
                raise ValueError(
                    "control journal terminal event lacks completed stages"
                )
            if event == "control_completed" and terminal_seen:
                raise ValueError(
                    "control journal has a duplicate completion event"
                )
            if event in {
                "control_recovery_required",
                "control_revalidated",
            } and not terminal_seen:
                raise ValueError(
                    "control journal recovery event lacks prior completion"
                )
            terminal_seen = True
            if event in {"control_completed", "control_revalidated"}:
                terminal_result_digest = detail["result_sha256"]
                terminal_run_id = detail["run_id"]
                terminal_run_spec_id = (
                    detail["run_spec_id"]
                    if event == "control_completed"
                    else None
                )

    strict_state = state.get("schema_version") == 4
    state_attempts = state.get("stage_attempts")
    if strict_state and not isinstance(state_attempts, dict):
        raise ValueError("control state stage attempts must be an object")
    if isinstance(state_attempts, dict) and state_attempts != attempts:
        raise ValueError("control state stage attempts disagree with journal")
    state_completed = state.get("completed")
    if strict_state and not isinstance(state_completed, dict):
        raise ValueError("control state completed stages must be an object")
    if isinstance(state_completed, dict) and set(state_completed) != completed:
        raise ValueError("control state completed stages disagree with journal")
    if isinstance(state_completed, dict):
        for stage in sorted(completed):
            if (
                _control_payload_sha256(state_completed[stage])
                != completed_digests.get(stage)
            ):
                raise ValueError(
                    "control state completed payload disagrees with journal"
                )

    if strict_state:
        if not records:
            raise ValueError("schema-v4 control journal cannot be empty")
        last = records[-1]
        expected_status = {
            "control_completed": "completed",
            "control_initialized": "initialized",
            "control_recovery_required": "recovery_required",
            "control_revalidated": "completed",
            "stage_checkpoint_invalid": "running",
            "stage_checkpoint_refresh_required": "running",
            "stage_completed": "running",
            "stage_failed": "failed",
            "stage_started": "running",
        }[last["event"]]
        if state.get("status") != expected_status:
            raise ValueError("control state status disagrees with journal")
        if terminal_result_digest is None:
            if "result" in state:
                raise ValueError("control state has an unbound terminal result")
        else:
            result = state.get("result")
            if (
                not isinstance(result, dict)
                or set(result) != _CONTROL_RESULT_FIELDS
                or result.get("run_id") != terminal_run_id
                or _control_payload_sha256(result)
                != terminal_result_digest
                or (
                    terminal_run_spec_id is not None
                    and result.get("run_spec_id")
                    != terminal_run_spec_id
                )
            ):
                raise ValueError("control state result disagrees with journal")
        if last["event"] in {"control_recovery_required", "stage_failed"}:
            last_error = state.get("last_error")
            expected_error = {
                key: last["detail"][key]
                for key in ("stage", "type", "message")
            }
            if last_error != expected_error:
                raise ValueError("control state error disagrees with journal")


def _reconcile_control_journal(state_path: Path, state: dict) -> None:
    """Repair one provable final-record gap or reject journal drift."""
    from scripts.stopdff_v5.identity import loads_no_duplicate_keys

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
            record = loads_no_duplicate_keys(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"control journal line {line_number} is invalid JSON"
            ) from exc
        canonical_line = json.dumps(record, sort_keys=True).encode("utf-8")
        if line != canonical_line:
            raise ValueError(
                f"control journal line {line_number} is not canonical JSON"
            )
        records.append(_validate_control_event_record(
            record,
            expected_sequence=line_number,
            previous_record=records[-1] if records else None,
        ))

    sequence = state.get("sequence", 0)
    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 0
    ):
        raise ValueError("control state sequence is invalid")
    last_event = state.get("last_event")
    if sequence == 0 and last_event is not None:
        raise ValueError("empty control state must not contain a last event")
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
        validated_last = _validate_control_event_record(
            last_event,
            expected_sequence=sequence,
            previous_record=records[-1] if records else None,
        )
        _validate_control_journal_projection(records + [validated_last], state)
        complete_prefix = journal_bytes[:-len(torn_tail)]
        _atomic_replace_control_bytes(
            journal_path,
            complete_prefix + canonical_last,
        )
        records.append(validated_last)

    if len(records) == sequence:
        if sequence and records[-1] != last_event:
            raise ValueError("control state and journal last event disagree")
        _validate_control_journal_projection(records, state)
        return
    if (
        len(records) == sequence - 1
        and isinstance(last_event, dict)
        and last_event.get("sequence") == sequence
    ):
        validated_last = _validate_control_event_record(
            last_event,
            expected_sequence=sequence,
            previous_record=records[-1] if records else None,
        )
        _validate_control_journal_projection(records + [validated_last], state)
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
    event_detail = dict(detail or {})
    if event == "stage_completed":
        completed = state.get("completed")
        if not isinstance(completed, dict) or stage not in completed:
            raise ValueError("completed stage event lacks checkpoint payload")
        event_detail["result_sha256"] = _control_payload_sha256(
            completed[stage]
        )
    elif event in {"control_completed", "control_revalidated"}:
        event_detail["result_sha256"] = _control_payload_sha256(
            state.get("result")
        )
    state["sequence"] = int(state.get("sequence", 0)) + 1
    previous_event = state.get("last_event")
    record = {
        "sequence": state["sequence"],
        "event": event,
        "stage": stage,
        "utc_epoch_seconds": int(time.time()),
        "detail": event_detail,
        "previous_event_sha256": (
            _control_event_sha256(previous_event)
            if isinstance(previous_event, dict)
            else None
        ),
    }
    _validate_control_event_record(
        record,
        expected_sequence=state["sequence"],
        previous_record=(
            previous_event if isinstance(previous_event, dict) else None
        ),
    )
    state["last_event"] = record
    _write_control_state(state_path, state)
    _append_control_event(
        state_path.with_name(state_path.name + ".jsonl"),
        record,
    )


def _close_interrupted_control_attempt(
    state_path: Path,
    state: dict,
) -> bool:
    """Close a host-abandoned stage before a resumed controller does work."""
    last_event = state.get("last_event")
    if (
        not isinstance(last_event, dict)
        or last_event.get("event") != "stage_started"
    ):
        return False
    stage = last_event.get("stage")
    detail = last_event.get("detail")
    attempt = detail.get("attempt") if isinstance(detail, dict) else None
    if stage not in _CONTROL_STAGE_NAMES or not isinstance(attempt, int):
        raise ValueError("active control attempt is noncanonical")
    error = {
        "stage": stage,
        "type": "HostControllerInterrupted",
        "message": (
            "controller resumed after a stage start without a terminal event"
        ),
    }
    state["status"] = "failed"
    state["last_error"] = error
    _record_control_event(
        state_path,
        state,
        event="stage_failed",
        stage=stage,
        detail={"attempt": attempt, **error},
    )
    return True


def _validate_control_plan(plan: dict) -> dict:
    from scripts.stopdff_v5.identity import compute_id

    allowed = {
        "source_id",
        "raw_id",
        "adapter_subdirs",
        "gate_overrides",
        "resource_summary",
        "resource_summary_id",
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
    if any("__attempt_" in value for value in canonical_subdirs):
        raise ValueError(
            "control plan adapter_subdirs use a reserved retry namespace"
        )
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
    try:
        resource_summary_id = compute_id(resource_summary)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "control plan resource_summary is not canonically identity-safe"
        ) from exc
    supplied_resource_id = plan.get("resource_summary_id")
    if (
        supplied_resource_id is not None
        and supplied_resource_id != resource_summary_id
    ):
        raise ValueError("control plan resource_summary_id mismatch")
    return {
        "source_id": source_id,
        "raw_id": raw_id,
        "adapter_subdirs": canonical_subdirs,
        "gate_overrides": dict(gate_overrides),
        "resource_summary": dict(resource_summary),
        "resource_summary_id": resource_summary_id,
    }


def _adapter_attempt_subdirs(
    base_subdirs: list[str],
    attempt: int,
) -> tuple[str, str]:
    """Derive fresh adapter destinations for one controller attempt."""
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
    ):
        raise ValueError("adapter attempt must be a positive integer")
    if len(base_subdirs) != 2:
        raise ValueError("adapter attempt requires two base subdirs")
    canonical_bases = tuple(
        _canonical_adapter_subdir(base) for base in base_subdirs
    )
    if attempt == 1:
        candidates = canonical_bases
    else:
        candidates = tuple(
            _retry_adapter_subdir(base, attempt) for base in canonical_bases
        )
    first = _canonical_adapter_subdir(candidates[0])
    second = _canonical_adapter_subdir(candidates[1])
    if first == second:
        raise ValueError("adapter attempt requires distinct subdirs")
    return first, second


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


def _invalidate_control_dependents(
    state_path: Path,
    state: dict,
    *,
    upstream: str,
    reason: str,
) -> None:
    """Remove and journal every completed suffix dependent of ``upstream``."""
    if upstream not in _CONTROL_STAGE_NAMES:
        raise ValueError("cannot invalidate dependents of an unknown stage")
    completed = state.setdefault("completed", {})
    attempts = state.setdefault("stage_attempts", {})
    upstream_index = _CONTROL_STAGE_ORDER.index(upstream)
    for dependent in reversed(_CONTROL_STAGE_ORDER[upstream_index + 1 :]):
        if dependent not in completed:
            continue
        attempt = attempts.get(dependent)
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
        ):
            raise ValueError(
                f"completed dependent {dependent} lacks a canonical attempt"
            )
        completed.pop(dependent)
        state["status"] = "running"
        _record_control_event(
            state_path,
            state,
            event="stage_checkpoint_invalid",
            stage=dependent,
            detail={
                "attempt": attempt,
                "stage": dependent,
                "type": "DependencyInvalidated",
                "message": (
                    f"upstream stage {upstream} requires refresh: {reason}"
                ),
            },
        )


def _refresh_control_stage(
    state_path: Path,
    state: dict,
    *,
    stage: str,
    reason: str,
) -> None:
    """Explicitly refresh a completed stage and all transitive dependents."""
    completed = state.setdefault("completed", {})
    if stage not in completed:
        return
    _invalidate_control_dependents(
        state_path,
        state,
        upstream=stage,
        reason=reason,
    )
    completed.pop(stage)
    state["status"] = "running"
    _record_control_event(
        state_path,
        state,
        event="stage_checkpoint_refresh_required",
        stage=stage,
        detail={"reason": reason},
    )


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
            _invalidate_control_dependents(
                state_path,
                state,
                upstream=name,
                reason=f"{type(exc).__name__}: {exc}",
            )
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
    state.pop("last_error", None)
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
        if state.get("schema_version") != 4:
            raise ValueError("unsupported control-state schema")
        _reconcile_control_journal(state_path, state)
        if state.get("plan_digest") != digest or state.get("plan") != plan:
            raise ValueError("resume control plan does not match durable state")
        _close_interrupted_control_attempt(state_path, state)
        if (
            state.get("status") not in {"completed", "recovery_required"}
            and "validate_package" in state.get("completed", {})
        ):
            _refresh_control_stage(
                state_path,
                state,
                stage="validate_package",
                reason="nonterminal resume must re-read packaged bytes",
            )
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
        journal_path = state_path.with_name(state_path.name + ".jsonl")
        if (
            state_path.exists()
            or state_path.is_symlink()
            or journal_path.exists()
            or journal_path.is_symlink()
        ):
            raise FileExistsError("fresh control state or journal already exists")
        state = {
            "schema_version": 4,
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

    resumed_probe_result = None
    if resume and "environment_probe" in state["completed"]:
        current_probe = api["probe"]()
        _validate_control_stage_result(
            "environment_probe",
            current_probe,
            lambda result: _validate_probe_result(
                result,
                ENVIRONMENT_PACKAGES,
            ),
        )
        current_environment_id = compute_id(
            environment_contract_identity(
                python_version=current_probe["python"],
                package_versions=current_probe["package_versions"],
            )
        )
        cached_probe = state["completed"]["environment_probe"]
        refresh_reason = None
        try:
            _validate_control_stage_result(
                "environment_probe",
                cached_probe,
                lambda result: _validate_probe_result(
                    result,
                    ENVIRONMENT_PACKAGES,
                ),
            )
            cached_environment_id = compute_id(
                environment_contract_identity(
                    python_version=cached_probe["python"],
                    package_versions=cached_probe["package_versions"],
                )
            )
        except Exception as exc:
            refresh_reason = (
                "cached environment probe is invalid: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            if cached_environment_id != current_environment_id:
                refresh_reason = (
                    "nonterminal resume observed a different Modal "
                    f"environment contract: {cached_environment_id} -> "
                    f"{current_environment_id}"
                )
        if refresh_reason is not None:
            _refresh_control_stage(
                state_path,
                state,
                stage="environment_probe",
                reason=refresh_reason,
            )
            # The live probe is a read-only resume preflight. Reuse its already
            # validated payload when checkpointing the refreshed stage so one
            # resume performs exactly one remote environment probe.
            resumed_probe_result = current_probe

    probe_result = _run_control_stage(
        state_path,
        state,
        name="environment_probe",
        invoke=lambda _: (
            resumed_probe_result
            if resumed_probe_result is not None
            else api["probe"]()
        ),
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

    def invoke_adapter_determinism(attempt: int) -> dict:
        attempt_first, attempt_second = _adapter_attempt_subdirs(
            plan["adapter_subdirs"],
            attempt,
        )
        return api["adapter_determinism_receipt"](
            attempt_first,
            attempt_second,
            source_id,
            raw_id,
            model_id,
            bool(
                plan["gate_overrides"].get(
                    "allow_low_mc_retention",
                    False,
                )
            ),
        )

    determinism = _run_control_stage(
        state_path,
        state,
        name="adapter_determinism",
        invoke=invoke_adapter_determinism,
        validate_result=lambda result: _validate_determinism_result(
            result,
            source_id,
        ),
    )
    first_subdir, second_subdir = _adapter_attempt_subdirs(
        plan["adapter_subdirs"],
        state["stage_attempts"]["adapter_determinism"],
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
        resource_summary_id=plan["resource_summary_id"],
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
        resource_summary_id=plan["resource_summary_id"],
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
    # package() performs its own fail-closed unpacked validation.  A separate
    # controller validation here would repeat the complete 96-cell computation
    # without adding a trust boundary; keep the independent packaged validation
    # below, after publication has changed the evidence surface.
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
def stage_inputs_main(
    source_bundle: str,
    raw_bundle: str,
    receipt_path: str,
):
    """Create-once stage and independently read back both canonical inputs."""
    receipt = Path(receipt_path)
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(f"staging receipt already exists: {receipt}")
    source_result = _stage_one_input_bundle(
        Path(source_bundle),
        "source",
    )
    raw_result = _stage_one_input_bundle(
        Path(raw_bundle),
        "raw",
    )
    result = {
        "schema_version": 1,
        "volume": VOLUME_NAME,
        "source": source_result,
        "raw": raw_result,
    }
    data = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_create_control_bytes(receipt, data)
    print(data.decode("utf-8"), end="")


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
