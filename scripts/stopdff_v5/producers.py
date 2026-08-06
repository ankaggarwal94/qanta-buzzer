"""Control-plane producers: source snapshot, raw-input staging, environment contract.

These run on Device 1 (no real-data scientific computation): they build verified,
content-addressed inputs from a frozen git SHA and the staged Dropbox raw files.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tarfile
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any

from qb_data.dataset_splits import normalize_question_text, normalize_split_answer

from .identity import build_manifest, sha256_file
from .manifests import (
    RAW_INPUT_ROLES,
    environment_contract_identity,
    raw_input_identity,
    source_manifest_identity,
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


def _load_json_unique(path: Path) -> Any:
    """Decode JSON while rejecting duplicate object keys and non-finite constants."""
    path = Path(path)

    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def _reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _dataset_records(dataset_path: Path) -> list[dict[str, Any]]:
    data = _load_json_unique(dataset_path)
    records = data["questions"] if isinstance(data, dict) and "questions" in data else data
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"dataset {dataset_path} must contain a list of question objects")
    return records


def _record_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _split_semantics(
    records_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], bool]:
    """Recompute QID/text disjointness and answer consistency from staged bytes."""
    checks: dict[str, Any] = {}
    qids_by_split: dict[str, set[str]] = {}
    texts_by_split: dict[str, set[str]] = {}
    answers_by_text: dict[str, set[str]] = defaultdict(set)
    complete = True

    for split, records in records_by_split.items():
        qids: list[str] = []
        texts: list[str] = []
        for record in records:
            raw_qid = _record_value(record, ("qid", "question_id", "id"))
            raw_text = _record_value(record, ("question", "text"))
            raw_answer = _record_value(
                record,
                ("answer_primary", "answer", "answer_text"),
            )
            qid = (
                str(raw_qid)
                if isinstance(raw_qid, (str, int)) and not isinstance(raw_qid, bool)
                else ""
            )
            try:
                text = normalize_question_text(raw_text)
                answer = normalize_split_answer(raw_answer)
            except TypeError:
                text = ""
                answer = ""
            if not qid or not text or not answer:
                complete = False
            if qid:
                qids.append(qid)
            if text:
                texts.append(text)
                answers_by_text[text].add(answer)

        qid_set = set(qids)
        text_set = set(texts)
        qids_by_split[split] = qid_set
        texts_by_split[split] = text_set
        checks[f"{split}_record_count"] = len(records)
        checks[f"{split}_qid_count"] = len(qid_set)
        checks[f"{split}_normalized_text_count"] = len(text_set)
        checks[f"{split}_qids_unique"] = len(qids) == len(qid_set)
        complete = complete and checks[f"{split}_qids_unique"]

    checks["records_have_qid_text_answer"] = complete
    pairwise_ok = True
    split_names = ("train", "val", "test")
    for left_idx, left in enumerate(split_names):
        for right in split_names[left_idx + 1 :]:
            qid_overlap = qids_by_split[left] & qids_by_split[right]
            text_overlap = texts_by_split[left] & texts_by_split[right]
            prefix = f"{left}_{right}"
            checks[f"{prefix}_qid_disjoint"] = not qid_overlap
            checks[f"{prefix}_normalized_text_disjoint"] = not text_overlap
            if qid_overlap:
                checks[f"{prefix}_qid_overlap_examples"] = sorted(qid_overlap)[:10]
            if text_overlap:
                checks[f"{prefix}_normalized_text_overlap_examples"] = sorted(text_overlap)[:10]
            pairwise_ok = pairwise_ok and not qid_overlap and not text_overlap

    conflicts = {
        text: sorted(answers)
        for text, answers in answers_by_text.items()
        if len(answers) > 1
    }
    checks["normalized_text_answers_consistent"] = not conflicts
    if conflicts:
        checks["conflicting_answer_examples"] = [
            {"normalized_text": text, "normalized_answers": conflicts[text]}
            for text in sorted(conflicts)[:10]
        ]

    return checks, complete and pairwise_ok and not conflicts


def stage_raw_inputs(source_paths: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    """Stage the ten raw inputs, run semantic checks, and emit the raw-input identity.

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

    # Parse every declared JSON role up front. Some inputs are only identity
    # material here, but ambiguous duplicate-key JSON must never enter the bundle.
    decoded = {
        role: _load_json_unique(path)
        for role, path in staged.items()
        if role.endswith(".json")
    }

    checks: dict[str, Any] = {}

    # threshold sidecar verification
    manifest_sha = sha256_file(staged["threshold_manifest.json"])
    sidecar_text = staged["threshold_manifest.json.sha256"].read_text(encoding="utf-8").strip()
    sidecar_sha = sidecar_text.split()[0] if sidecar_text else ""
    checks["threshold_sidecar_ok"] = (manifest_sha == sidecar_sha)

    # calibration fit split
    cal = decoded["calibration.json"]
    cal_metadata = cal.get("metadata") if isinstance(cal, dict) else None
    fit_split = (
        cal_metadata.get("fit_split")
        if isinstance(cal_metadata, dict)
        else None
    )
    checks["calibration_fit_split"] = fit_split
    checks["calibration_fit_split_is_val"] = (fit_split == "val")

    # Recompute three-way split integrity from the staged bytes.
    records_by_split = {
        split: _dataset_records(staged[f"{split}_dataset.json"])
        for split in ("train", "val", "test")
    }
    split_checks, split_ok = _split_semantics(records_by_split)
    checks.update(split_checks)
    # Keep the legacy summary key, but derive it from both required dimensions.
    checks["val_test_disjoint"] = bool(
        checks["val_test_qid_disjoint"]
        and checks["val_test_normalized_text_disjoint"]
    )

    # build-metadata retention consistency
    bm = decoded["build_metadata.json"]
    splits = bm.get("splits") if isinstance(bm, dict) else None
    bm_ok = isinstance(splits, dict)
    for split in ("train", "val", "test"):
        block = splits.get(split) if isinstance(splits, dict) else None
        required = ("raw_count", "retained_count", "dropped_count")
        if not isinstance(block, dict) or not all(key in block for key in required):
            bm_ok = False
            continue
        values = [block[key] for key in required]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        ):
            bm_ok = False
            continue
        raw_count = block["raw_count"]
        retained_count = block["retained_count"]
        dropped_count = block["dropped_count"]
        if retained_count + dropped_count != raw_count:
            bm_ok = False
        if retained_count != len(records_by_split[split]):
            bm_ok = False
    checks["build_metadata_retention_consistent"] = bm_ok

    # myopic producer/provenance
    myopic = decoded["stopdff.json"]
    myopic_metadata = myopic.get("metadata") if isinstance(myopic, dict) else None
    myopic_value = (
        myopic.get("median_abs_prefix_shift")
        if isinstance(myopic, dict)
        else None
    )
    metric_valid = (
        isinstance(myopic_value, Real)
        and not isinstance(myopic_value, bool)
        and math.isfinite(float(myopic_value))
        and float(myopic_value) >= 0.0
    )
    metric_type = (
        myopic_metadata.get("metric_type")
        if isinstance(myopic_metadata, dict)
        else None
    )
    stopping_policy = (
        myopic_metadata.get("stopping_policy")
        if isinstance(myopic_metadata, dict)
        else None
    )
    myopic_semantics_valid = bool(
        metric_valid
        and metric_type == "diagnostic_only"
        and stopping_policy == "myopic_threshold"
    )
    checks["myopic_metric_name"] = "median_abs_prefix_shift"
    checks["myopic_metric_value"] = (
        format(float(myopic_value), ".17g")
        if isinstance(myopic_value, Real)
        and not isinstance(myopic_value, bool)
        and math.isfinite(float(myopic_value))
        else None
    )
    checks["myopic_metric_valid"] = metric_valid
    checks["myopic_metric_type"] = metric_type
    checks["myopic_stopping_policy"] = stopping_policy
    checks["myopic_semantics_valid"] = myopic_semantics_valid

    # No evaluation row may be used to fit a calibrator/continuation. This now
    # follows from fit_split=val plus byte-derived three-way disjointness.
    all_ok = (checks["threshold_sidecar_ok"] and checks["calibration_fit_split_is_val"]
              and split_ok and checks["build_metadata_retention_consistent"]
              and checks["myopic_semantics_valid"])
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
