#!/usr/bin/env python3
"""Modal stage functions for the StopDFF v5 evidentiary pipeline (remote functions).

Source-only image (STOPDFF_V5_SOURCE_DIR = validated source-snapshot bundle). One Volume
`cs321m-stopdff-artifacts` mounted at /stopdff. One writer per run dir, max_containers=1,
explicit Volume commits (per cell in the sweep), reload before resume. GPU (L40S) is used
only by build_adapter; every other stage is CPU.

This module owns the Modal app object, the frozen image, and every registered
stage function and local entrypoint. The other two subsystems live in sibling
modules and are re-exported here so existing imports and
``modal run scripts/modal_stopdff_v5_runner.py::control_main`` keep working:

- host-side orchestration (durable checkpoint/journal, stage validators, and
  the ``run_control_plane`` driver): ``scripts/stopdff_v5_control_plane.py``;
- recovery-assurance canary derivations: ``scripts/stopdff_v5_assurance_stages.py``
  (this module keeps only their Modal-facing bindings and the registered
  ``recovery_assurance`` stage).

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

import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

import modal

from scripts import stopdff_v5_assurance_stages as _assurance_stages
from scripts import stopdff_v5_control_plane as _control_plane
from scripts.stopdff_v5.identity import is_sha256_hex

# Deliberate strangler-fig intermediate: this facade re-exports the moved
# control-plane names so existing imports keep working; the re-export list
# below is frozen (no additions) pending full migration of callers to
# ``scripts.stopdff_v5_control_plane``.
from scripts.stopdff_v5_control_plane import (  # noqa: F401  (facade re-exports)
    _ADAPTER_COMPONENT_MAX_BYTES,
    _adapter_attempt_subdirs,
    _append_control_event,
    _atomic_create_control_bytes,
    _atomic_replace_control_bytes,
    _canonical_adapter_subdir,
    _close_interrupted_control_attempt,
    _CONTROL_EVENT_NAMES,
    _control_event_sha256,
    _control_payload_sha256,
    _control_plan_digest,
    _CONTROL_RESULT_FIELDS,
    _CONTROL_STAGE_NAMES,
    _CONTROL_STAGE_ORDER,
    _invalidate_control_dependents,
    _is_control_sha,
    _is_final_control_run_id,
    _load_control_json,
    _reconcile_control_journal,
    _record_control_event,
    _refresh_control_stage,
    _require_control_bool,
    _require_control_count,
    _require_control_sha,
    _retry_adapter_subdir,
    _run_control_stage,
    _validate_adapter_result,
    _validate_bootstrap_result,
    _validate_checker_result,
    _validate_control_event_record,
    _validate_control_journal_projection,
    _validate_control_plan,
    _validate_control_stage_result,
    _validate_determinism_result,
    _validate_fvi_result,
    _validate_model_result,
    _validate_mutation_result,
    _validate_package_result,
    _validate_probe_result,
    _validate_promotion_result,
    _validate_sweep_result,
    _validate_verified_artifact_result,
    _write_control_state,
)

_MODAL_IS_LOCAL = modal.is_local()
VOLUME_NAME = "cs321m-stopdff-artifacts"
_DEFAULT_APP_NAME = "cs321m-stopdff-v5"
APP_NAME = os.environ.get("STOPDFF_V5_APP_NAME", _DEFAULT_APP_NAME)
if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", APP_NAME) is None:
    raise RuntimeError("STOPDFF_V5_APP_NAME is not a canonical Modal app name")
if _MODAL_IS_LOCAL and APP_NAME != _DEFAULT_APP_NAME:
    # max_containers=1 serializes writers per *app*; a second app name mounts
    # the same hardcoded Volume, so overriding it weakens the single-writer
    # invariant. The override remains a documented workflow (uniquely named
    # recovery-assurance campaigns, REPRODUCTION.md), but only behind an
    # explicit opt-in, and never silently.
    if os.environ.get("STOPDFF_V5_ALLOW_APP_OVERRIDE") != "1":
        raise RuntimeError(
            "STOPDFF_V5_APP_NAME overrides the app name that scopes the "
            f"max_containers=1 single-writer serialization on the shared "
            f"'{VOLUME_NAME}' Volume; set STOPDFF_V5_ALLOW_APP_OVERRIDE=1 to "
            "confirm no other deployment writes the same slots"
        )
    print(
        f"WARNING: STOPDFF_V5_APP_NAME={APP_NAME!r} deploys a second app "
        f"against the shared '{VOLUME_NAME}' Volume; max_containers=1 no "
        "longer serializes writers across apps. Ensure no concurrent "
        "deployment writes the same slots.",
        file=sys.stderr,
    )
MNT = "/stopdff"
REMOTE_SRC = "/root/src"
DAY = 86400
_STAGING_PREFIX = ".staging_"
# Reclaim only staging directories that provably cannot belong to a live
# container: the longest-running staging user (freeze_model/fvi_study/
# bootstrap_plan) runs under timeout=DAY, so anything older than twice that
# bound is crash garbage even across Volume background-commit clock skew.
# Younger leftovers are inert (uuid-named, never published, never audited)
# and are reclaimed by a later fresh attempt once they age out.
_STAGING_REAP_AGE_S = 2 * DAY
_SOURCE_ID_ENV = "STOPDFF_V5_IMAGE_SOURCE_MANIFEST_ID"
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

if not is_sha256_hex(IMAGE_SOURCE_MANIFEST_ID):
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
    "matplotlib>=3.7",
    # The library floor is sentence-transformers 2.3.0 (trust_remote_code;
    # see pyproject.toml); >=2.7 is the stricter pin this image runs against.
    "sentence-transformers>=2.7",
    # torch.load-based checkpoint loading; >=2.6 required (CVE-2025-32434,
    # fixed in 2.6.0) — parity with pyproject.toml.
    "torch>=2.6",
    "huggingface_hub>=0.23",
]
# No apt packages: model snapshots arrive via huggingface_hub and provenance
# is host-injected (no in-container git; see the repo's provenance learning).
_image = (
    modal.Image.debian_slim(python_version="3.11")
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


_MODAL_RUN_ID_RE = re.compile(r"(smoke|final)_modal_([0-9a-f]{12})")


def _canonical_modal_run_id(
    value: object,
    *,
    variant: str | None = None,
    run_spec_id: str | None = None,
) -> str:
    """Return the exact one-component run slot bound to a run-spec ID."""
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("run_id must be a nonempty canonical string")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or len(parsed.parts) != 1
        or str(parsed) != value
        or _MODAL_RUN_ID_RE.fullmatch(value) is None
    ):
        raise ValueError(f"unsafe or noncanonical run_id: {value!r}")
    if (variant is None) != (run_spec_id is None):
        raise ValueError("run_id binding requires both variant and run_spec_id")
    if variant is not None:
        if variant not in {"smoke", "final"} or not is_sha256_hex(run_spec_id):
            raise ValueError("run_id binding inputs are noncanonical")
        expected = f"{variant}_modal_{run_spec_id[:12]}"
        if value != expected:
            raise ValueError("run_id is not bound to run_spec_id")
    return value


def _receipt_rel(gate: str, receipt_id: str) -> str:
    if gate not in {"smoke", "mutation", "determinism"}:
        raise ValueError(f"unknown prerequisite receipt gate: {gate}")
    if not is_sha256_hex(receipt_id):
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


def _canonical_fvi_study_path(root: Path) -> Path:
    """Return the FVI manifest only from its canonical cache slot."""
    root = Path(root)
    manifest_path = root / "fvi_study.json"
    if (
        root.is_symlink()
        or not root.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise ValueError("FVI cache is incomplete or noncanonical")
    return manifest_path


def _validated_cached_fvi(
    out: Path,
    *,
    manifest: dict,
    identity: dict,
    execution: dict,
    checker_module,
) -> dict:
    """Validate both durable FVI records before accepting a cache hit."""
    try:
        manifest_path = _canonical_fvi_study_path(out)
    except ValueError as exc:
        raise FileExistsError(
            "FVI destination is incomplete or noncanonical"
        ) from exc
    execution_path = out / "fvi_study_execution.json"
    if execution_path.is_symlink() or not execution_path.is_file():
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


def _model_cache_state(root: Path) -> tuple[Path, bool]:
    """Classify the immutable model cache without following unsafe paths."""
    root = Path(root)
    manifest_path = root / "model_snapshot_manifest.json"
    if root.is_symlink():
        raise ValueError("model cache root must be a non-symlink directory")
    if root.exists() and not root.is_dir():
        raise ValueError("model cache root must be a non-symlink directory")
    if manifest_path.is_symlink():
        raise ValueError(
            "model cache manifest must be a non-symlink regular file"
        )
    if manifest_path.exists():
        if not manifest_path.is_file():
            raise ValueError(
                "model cache manifest must be a non-symlink regular file"
            )
        return manifest_path, True
    if root.exists():
        try:
            nonempty = next(root.iterdir(), None) is not None
        except OSError as exc:
            raise ValueError("model cache root cannot be inspected") from exc
        if nonempty:
            raise FileExistsError(
                "model cache destination is incomplete or noncanonical"
            )
    return manifest_path, False


def _reclaim_staging_dirs(parent: Path) -> int:
    """Remove crash-abandoned ``.staging_*`` directories under one cache parent.

    Staging directories are private to a single in-flight materialization and
    never referenced by any manifest, so any that persist (e.g. via a Volume
    background commit racing a crash) are reclaimable garbage — not evidence.
    Reclaiming them keeps a crashed download from bricking future runs.

    Only entries older than ``_STAGING_REAP_AGE_S`` are removed: a second
    controller sharing the Volume (dual ``modal run`` invocations, or the
    ``STOPDFF_V5_ALLOW_APP_OVERRIDE`` escape hatch) may surface a live peer's
    in-flight staging directory via a background commit, and deleting it would
    fail that peer's publish after it loses the whole materialization. The age
    gate exceeds every staging stage's function timeout, so a directory past
    it cannot belong to a live container.
    """
    parent = Path(parent)
    if parent.is_symlink() or not parent.is_dir():
        return 0
    reclaimed = 0
    now = time.time()
    for entry in parent.iterdir():
        if (
            entry.name.startswith(_STAGING_PREFIX)
            and not entry.is_symlink()
            and entry.is_dir()
            and (now - entry.stat().st_mtime) > _STAGING_REAP_AGE_S
        ):
            shutil.rmtree(entry)
            reclaimed += 1
    return reclaimed


def _new_staging_dir(parent: Path) -> Path:
    """Create one private staging directory beside its publish destination."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f"{_STAGING_PREFIX}{uuid.uuid4().hex}"
    staging.mkdir()
    return staging


def _publish_staged_dir(staging: Path, dest: Path) -> None:
    """Atomically publish a fully-materialized staging directory.

    ``os.rename`` either installs the complete directory under its live name
    or fails; the live slot never holds partial content. A destination that
    materialized concurrently (or was left non-empty by damage predating the
    staging discipline) fails closed instead of being overwritten.
    """
    staging = Path(staging)
    dest = Path(dest)
    if dest.is_symlink():
        raise FileExistsError(f"publish destination is a symlink: {dest}")
    try:
        os.rename(staging, dest)
    except OSError as exc:
        raise FileExistsError(
            f"publish destination appeared concurrently or is not replaceable: {dest}"
        ) from exc


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
    mpath, cached_entry = _model_cache_state(root)
    if cached_entry:
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
    # Materialize into a sibling staging directory and publish by one atomic
    # rename: the singleton live slot never holds a partial download, so a
    # crash mid-snapshot cannot brick every future run (crash-persisted
    # staging leftovers are reclaimed on the next fresh attempt).
    _reclaim_staging_dirs(root.parent)
    staging = _new_staging_dir(root.parent)
    try:
        man = adapter_build.freeze_model_snapshot(staging)
        _verified_content_manifest(
            staging,
            manifest_name="model_snapshot_manifest.json",
            expected_id=man["id"],
            file_key="files",
            name_key="path",
            content_subdir="snapshot",
            expected_kind="model_snapshot",
        )
        _, appeared_during_freeze = _model_cache_state(root)
        if appeared_during_freeze:
            raise FileExistsError("model cache appeared during fresh creation")
        _publish_staged_dir(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
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
    # Copy into a sibling staging directory and publish by one atomic rename
    # (same discipline as freeze_model/fvi_study/bootstrap_plan): the
    # create-once canonical slot never holds a partial copy, so a crash
    # mid-copytree cannot brick this adapter_id permanently — crash-persisted
    # leftovers land in the reclaimable ``.staging_*`` namespace instead.
    _reclaim_staging_dirs(dst.parent)
    staging = _new_staging_dir(dst.parent)
    try:
        copy = staging / "bundle"
        shutil.copytree(src, copy)
        copied = checker.validate_adapter(copy)
        if not copied.passed:
            raise ValueError(
                "copied adapter failed validation: " + "; ".join(copied.errors)
            )
        _publish_staged_dir(copy, dst)
    finally:
        # On success only the emptied staging holder remains; on any failure
        # this also removes the partial copy without masking the exception.
        shutil.rmtree(staging, ignore_errors=True)
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
    sweep.write_bound_json(
        Path(_receipt_evidence_rel("determinism", receipt["id"])),
        evidence,
        resume=True,
    )
    sweep.write_bound_json(
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
    # Stage both durable records, then publish the slot by one atomic rename
    # so a crash between the two writes cannot persist a half-written cache.
    _reclaim_staging_dirs(out.parent)
    staging = _new_staging_dir(out.parent)
    try:
        (staging / "fvi_study.json").write_text(
            json.dumps(man, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "fvi_study_execution.json").write_text(
            json.dumps(study, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        _publish_staged_dir(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
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
    # Stage the plan and publish the slot by one atomic rename; the live slot
    # is either absent or complete, never a manifest-less directory.
    _reclaim_staging_dirs(out.parent)
    staging = _new_staging_dir(out.parent)
    try:
        (staging / "bootstrap_plan.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True))
        _publish_staged_dir(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
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
    run_id = _canonical_modal_run_id(
        spec.get("run_id"),
        variant=binding["variant"],
        run_spec_id=binding["run_spec_id"],
    )

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
        _canonical_fvi_study_path(Path(_p("fvi", fvi_id)))
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
    run_root = Path(_p("runs", run_id))
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
    result = {"run_id": run_id, "requested": agg["requested"], "completed": agg["completed"],
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
        sweep.write_bound_json(
            Path(_receipt_evidence_rel("smoke", receipt["id"])),
            evidence,
            resume=True,
        )
        sweep.write_bound_json(receipt_path, receipt, resume=True)
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
    run_id = _canonical_modal_run_id(run_id)
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
    _canonical_modal_run_id(
        run_id,
        variant=spec_manifest["identity"]["profile_variant"],
        run_spec_id=spec_manifest["id"],
    )
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
    fvi_path = _canonical_fvi_study_path(Path(_p(
        "fvi",
        spec_ids["fvi_study_id"],
    )))
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
    run_id = _canonical_modal_run_id(run_id)
    run_root = Path(_p("runs", run_id))
    spec_path = run_root / "run_spec.json"
    if not spec_path.is_symlink() and spec_path.is_file():
        try:
            spec_manifest = checker.load_json(spec_path)
        except (OSError, UnicodeError, TypeError, ValueError):
            pass
        else:
            identity = (
                spec_manifest.get("identity")
                if isinstance(spec_manifest, dict)
                else None
            )
            if (
                isinstance(identity, dict)
                and identity.get("profile_variant") in {"smoke", "final"}
                and is_sha256_hex(spec_manifest.get("id"))
            ):
                _canonical_modal_run_id(
                    run_id,
                    variant=identity["profile_variant"],
                    run_spec_id=spec_manifest["id"],
                )
    res = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=Path(_p("adapters", f"canonical_{adapter_id}")),
        require_final_profile=require_final,
        require_package=require_package,
    )
    return {
        "passed": res.passed,
        "errors": res.errors[:50],
        "recomputed": res.recomputed,
    }


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
    sweep.write_bound_json(
        Path(_receipt_evidence_rel("mutation", receipt["id"])),
        evidence,
        resume=True,
    )
    sweep.write_bound_json(
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


# Recovery-assurance canary: derivations live in
# scripts/stopdff_v5_assurance_stages.py; the wrappers below bind this
# deployment's image source ID and volume paths at call time.


def _canonical_assurance_tag(tag: object) -> str:
    return _assurance_stages.canonical_assurance_tag(tag)


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
    return _assurance_stages.assurance_sweep_context(
        tag,
        attempt=attempt,
        resume=resume,
        commit_fn=commit_fn,
        image_source_id=IMAGE_SOURCE_MANIFEST_ID,
        run_root=Path(_p("pilots", tag, "run")),
    )


def _assurance_observation(tag: str) -> dict:
    """Read and validate the durable attempt records for one assurance tag."""
    return _assurance_stages.assurance_observation(
        tag,
        root=Path(_p("pilots", tag)),
    )


def _assurance_expected_evidence(tag: str) -> dict:
    """Derive the exact canary histories and immutable run identities."""
    return _assurance_stages.assurance_expected_evidence(
        tag,
        image_source_id=IMAGE_SOURCE_MANIFEST_ID,
        run_root=Path(_p("pilots", tag, "run")),
    )


def _assurance_phase_state(tag: str, observation: dict) -> tuple[str, dict]:
    """Classify only exact initial/classified/finished durable states."""
    return _assurance_stages.assurance_phase_state(
        tag,
        observation,
        image_source_id=IMAGE_SOURCE_MANIFEST_ID,
        run_root=Path(_p("pilots", tag, "run")),
    )


def _assurance_expected_aggregate(context, sweep_module) -> dict:
    return _assurance_stages.assurance_expected_aggregate(context, sweep_module)


def _load_assurance_aggregate(tag: str, context, sweep_module) -> dict:
    return _assurance_stages.load_assurance_aggregate(
        context,
        sweep_module,
        path=Path(_p("pilots", tag, "run", "aggregate.json")),
    )


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


def run_control_plane(
    plan: dict,
    state_path: Path,
    *,
    resume: bool,
    stage_api: dict[str, object] | None = None,
) -> dict:
    """Run the canonical Modal stage order with a durable local checkpoint.

    The driver itself lives in ``scripts/stopdff_v5_control_plane.py``; this
    facade binds the deployment-specific values that only this module knows —
    the validated image source-manifest ID, the staged source tree, and the
    default stage API over the registered Modal functions.
    """
    return _control_plane.run_control_plane(
        plan,
        state_path,
        resume=resume,
        stage_api=stage_api or _default_control_stage_api(),
        image_source_id=IMAGE_SOURCE_MANIFEST_ID,
        source_dir=SOURCE_DIR,
    )


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
