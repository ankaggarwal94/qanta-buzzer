"""Control-plane producers: source snapshot, raw-input staging, environment contract.

These run on Device 1 (no real-data scientific computation): they build verified,
content-addressed inputs from a frozen git SHA and the staged Dropbox raw files.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from .identity import build_manifest, compute_id, sha256_file
from .manifests import (
    environment_contract_identity,
    raw_input_identity,
    source_manifest_identity,
)

RAW_INPUT_ROLES = (
    "mc_dataset.json",
    "val_dataset.json",
    "test_dataset.json",
    "build_metadata.json",
    "split_metadata.json",
    "calibration.json",
    "stopdff.json",
    "threshold_manifest.json",
    "threshold_manifest.json.sha256",
)


# --- source snapshot --------------------------------------------------------------


def build_source_snapshot(repo_dir: Path, run_sha: str, out_dir: Path) -> dict[str, Any]:
    """Create source/ from `git archive RUN_SHA` and a self-excluding source manifest."""
    repo_dir = Path(repo_dir)
    out_dir = Path(out_dir)
    src_dir = out_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)

    # git archive to a tar, then extract (rejecting unsafe members).
    archive = out_dir / "source.tar"
    with open(archive, "wb") as fh:
        subprocess.run(["git", "-C", str(repo_dir), "archive", "--format=tar", run_sha],
                       check=True, stdout=fh)
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            name = member.name
            if member.issym() or member.islnk():
                raise ValueError(f"source snapshot rejects link member: {name}")
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe source member: {name}")
        tar.extractall(src_dir)  # noqa: S202 (members validated above)
    archive.unlink()

    # File modes/sizes from git; sha256 from extracted files.
    ls = subprocess.run(["git", "-C", str(repo_dir), "ls-tree", "-r", "-l", run_sha],
                        check=True, capture_output=True, text=True).stdout
    files: list[dict[str, Any]] = []
    for line in ls.splitlines():
        # <mode> <type> <sha> <size>\t<path>
        meta, path = line.split("\t", 1)
        mode, _type, _obj, size = meta.split()
        fpath = src_dir / path
        if not fpath.is_file():
            continue
        files.append({"path": path, "mode": mode, "size": int(size), "sha256": sha256_file(fpath)})

    def _sha_of(rel: str) -> str:
        p = src_dir / rel
        return sha256_file(p) if p.is_file() else ""

    identity = source_manifest_identity(
        git_sha=run_sha, files=files,
        pyproject_sha256=_sha_of("pyproject.toml"), uv_lock_sha256=_sha_of("uv.lock"),
    )
    manifest = build_manifest(identity, file_count=len(files))
    (out_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "source_build_record.json").write_text(
        json.dumps({"git_sha": run_sha, "repo_dir": str(repo_dir), "file_count": len(files)},
                   indent=2, sort_keys=True), encoding="utf-8")
    return manifest


# --- raw-input staging ------------------------------------------------------------


def _extract_qids(dataset_path: Path) -> set[str]:
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    records = data["questions"] if isinstance(data, dict) and "questions" in data else data
    qids: set[str] = set()
    for rec in records:
        for key in ("qid", "question_id", "id"):
            if key in rec:
                qids.add(str(rec[key]))
                break
    return qids


def stage_raw_inputs(source_paths: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    """Stage the nine raw inputs, run semantic checks, and emit the raw-input identity.

    source_paths maps each role in RAW_INPUT_ROLES to an absolute source path (Dropbox).
    """
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing = [r for r in RAW_INPUT_ROLES if r not in source_paths]
    if missing:
        raise ValueError(f"missing raw-input roles: {missing}")

    files: list[dict[str, Any]] = []
    staged: dict[str, Path] = {}
    for role in RAW_INPUT_ROLES:
        src = Path(source_paths[role])
        if not src.is_file():
            raise FileNotFoundError(f"raw input {role} not found at {src}")
        dst = raw_dir / role
        dst.write_bytes(src.read_bytes())
        staged[role] = dst
        files.append({"role": role, "size": dst.stat().st_size, "sha256": sha256_file(dst)})

    checks: dict[str, Any] = {}

    # threshold sidecar verification
    manifest_sha = sha256_file(staged["threshold_manifest.json"])
    sidecar_text = staged["threshold_manifest.json.sha256"].read_text(encoding="utf-8").strip()
    sidecar_sha = sidecar_text.split()[0] if sidecar_text else ""
    checks["threshold_sidecar_ok"] = (manifest_sha == sidecar_sha)

    # calibration fit split
    cal = json.loads(staged["calibration.json"].read_text(encoding="utf-8"))
    fit_split = cal.get("metadata", {}).get("fit_split")
    checks["calibration_fit_split"] = fit_split
    checks["calibration_fit_split_is_val"] = (fit_split == "val")

    # val/test QID disjointness + counts
    val_qids = _extract_qids(staged["val_dataset.json"])
    test_qids = _extract_qids(staged["test_dataset.json"])
    overlap = val_qids & test_qids
    checks["val_qid_count"] = len(val_qids)
    checks["test_qid_count"] = len(test_qids)
    checks["val_test_disjoint"] = (len(overlap) == 0)
    if overlap:
        checks["val_test_overlap_examples"] = sorted(overlap)[:10]

    # build-metadata retention consistency
    bm = json.loads(staged["build_metadata.json"].read_text(encoding="utf-8"))
    bm_ok = True
    for split, block in (bm.get("splits", {}) or {}).items():
        if all(k in block for k in ("raw_count", "retained_count", "dropped_count")):
            if int(block["retained_count"]) + int(block["dropped_count"]) != int(block["raw_count"]):
                bm_ok = False
    checks["build_metadata_retention_consistent"] = bm_ok

    # myopic producer/provenance
    myopic = json.loads(staged["stopdff.json"].read_text(encoding="utf-8"))
    checks["myopic_metric_type"] = myopic.get("metric_type") or myopic.get("metadata", {}).get("metric_type")
    checks["myopic_is_diagnostic"] = "myopic" in json.dumps(myopic).lower()

    # no test row used to fit a calibrator/continuation is structurally guaranteed by
    # fit_split=val + val/test disjointness (recorded above).
    all_ok = (checks["threshold_sidecar_ok"] and checks["calibration_fit_split_is_val"]
              and checks["val_test_disjoint"] and checks["build_metadata_retention_consistent"])
    checks["all_semantic_checks_pass"] = all_ok

    identity = raw_input_identity(files=files, semantic_checks=checks)
    manifest = build_manifest(identity)
    (out_dir / "raw_input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "raw_input_stage_record.json").write_text(
        json.dumps({"source_paths": {k: str(v) for k, v in source_paths.items()},
                    "staged_dir": str(raw_dir)}, indent=2, sort_keys=True), encoding="utf-8")
    if not all_ok:
        raise ValueError(f"raw-input semantic checks failed: {checks}")
    return manifest


# --- environment contract ---------------------------------------------------------


def environment_contract(package_versions: dict[str, str]) -> dict[str, Any]:
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    identity = environment_contract_identity(python_version=py, package_versions=package_versions)
    return build_manifest(identity)
