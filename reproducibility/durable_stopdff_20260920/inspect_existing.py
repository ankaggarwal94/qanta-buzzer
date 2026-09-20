"""Inspect one exactly bound existing full Sandbox; never submit or modify it."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace

CONTROL_DIRECTORY = Path("reproducibility/durable_stopdff_20260920")
VOLUME_NAME = "cs321m-stopdff-rerun-20260920-v2"
VOLUME_ID = "vo-kxgd4TezueO4b7TvuwGyaU"
VOLUME_VERSION = 2
SUPERVISOR_SHA256 = "a3218776a94a6b897f90179541355e06d9f7a37a911249dd451a3a0c8f255260"
LAUNCHER_SHA256 = "0d3345f82eb3fe2628f3221558099c5e15f3761b20ab01ed7d21ee6ff86efd0d"
FROZEN = {
    "INPUT_SHA256": "6afc62e1cb91d0d2ac958c251ba68aec1c83f2da2b9d0354dc4635234dbc1cf2",
    "INPUT_SIZE": 1234697660,
    "FINAL_COMMIT": "2ed304f6598b94c0c5dc15f77fb8b391ae2f03a3",
    "CONTROLLER_SHA256": "48d4bb32186ddb201f53b4a7f4dafdc10d5f9f382f494612388d678bb381ec50",
    "VOLUME_NAME": VOLUME_NAME,
    "ROOT_PREFIX": "durable-rerun-20260920",
    "SDK_VERSION": "1.5.5",
}
MAX_METADATA_BYTES = 65536


def validate_target(args):
    patterns = {"run_id": r"[a-z0-9][a-z0-9._-]{0,47}",
                "sandbox_id": r"sb-[A-Za-z0-9]+", "image_id": r"im-[A-Za-z0-9]+",
                "submission_code_commit": r"[0-9a-f]{40}"}
    for key, pattern in patterns.items():
        if not re.fullmatch(pattern, getattr(args, key, "")):
            raise ValueError("Invalid frozen monitor target")
    if args.submission_code_commit == "0" * 40:
        raise ValueError("Submission commit is not populated")
    if (args.volume_id != VOLUME_ID or args.volume_version != VOLUME_VERSION
            or args.supervisor_sha256 != SUPERVISOR_SHA256):
        raise ValueError("Monitor target differs from the fixed Volume or supervisor")


def load_submission_launcher(args):
    """Import only the second checkout, pinned to the exact submission commit."""
    root = Path(args.submission_root).resolve(strict=True)
    git = root / ".git"
    head = git / "HEAD"
    if git.is_symlink() or not git.is_dir() or head.is_symlink() or not head.is_file():
        raise ValueError("A detached submission checkout is required")
    with head.open("rb") as stream:
        if stream.read(128).strip() != args.submission_code_commit.encode("ascii"):
            raise ValueError("Submission checkout commit mismatch")
    controls = root / CONTROL_DIRECTORY
    path = controls / "launch_modal.py"
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("Pinned launcher must be a regular file")
    supervisor = controls / "supervisor.py"
    for item, expected in ((path, LAUNCHER_SHA256), (supervisor, SUPERVISOR_SHA256)):
        if item.is_symlink() or not item.is_file():
            raise ValueError("Pinned implementation must be regular files")
        with item.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Pinned implementation hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_submission_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_launch(record, args, launcher):
    if not isinstance(record, dict):
        raise ValueError("Launch receipt must be an object")
    expected = {
        "schema_version": 1, "status": "SUBMITTED", "mode": "full",
        "run_id": args.run_id, "sandbox_id": args.sandbox_id, "image_id": args.image_id,
        "durable_prefix": f"{FROZEN['ROOT_PREFIX']}/submissions/{args.run_id}",
        "volume_name": VOLUME_NAME, "volume_id": VOLUME_ID, "volume_version": VOLUME_VERSION,
        "supervisor_sha256": SUPERVISOR_SHA256, "commit_mode": "sync-v2",
        "input_sha256": FROZEN["INPUT_SHA256"], "input_size": FROZEN["INPUT_SIZE"],
        "final_commit": FROZEN["FINAL_COMMIT"], "controller_sha256": FROZEN["CONTROLLER_SHA256"],
        "modal_sdk_version": FROZEN["SDK_VERSION"], "scientific_acceptance": False,
    }
    if any(type(record.get(key)) is not type(value) or record[key] != value
           for key, value in expected.items()):
        raise ValueError("Launch receipt frozen bindings mismatch")
    launcher.require_bindings(record, {key: expected[key] for key in launcher.BINDING_KEYS})


class BoundedOutput(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > MAX_METADATA_BYTES:
            raise ValueError("Inspector metadata output exceeded its limit")
        return super().write(text)


def safe_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        return None
    result = {}
    for key in ("available", "current_liveness_claimed"):
        if type(snapshot.get(key)) is bool:
            result[key] = snapshot[key]
    if isinstance(snapshot.get("generation"), str) and re.fullmatch(r"[0-9]{6}", snapshot["generation"]):
        result["generation"] = snapshot["generation"]
    for label in ("supervisor", "controller_heartbeat"):
        record = snapshot.get(label)
        if not isinstance(record, dict):
            continue
        safe = {}
        for key in ("status", "stage", "error_type"):
            if isinstance(record.get(key), str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", record[key]):
                safe[key] = record[key]
        for key in ("updated_utc", "started_utc", "ended_utc"):
            if isinstance(record.get(key), str) and re.fullmatch(r"[0-9T:+.Z-]{10,40}", record[key]):
                safe[key] = record[key]
        for key in ("controller_returncode", "pid", "child_pid"):
            value = record.get(key)
            if value is None or (type(value) is int and abs(value) < 2**31):
                safe[key] = value
        result[label] = safe
    return result


def inspect_existing(args, launcher):
    validate_target(args)
    if any(getattr(launcher, key, None) != value for key, value in FROZEN.items()):
        raise ValueError("Pinned launcher frozen constants mismatch")
    volume = launcher.bound_volume(launcher.modal_sdk(), VOLUME_ID)
    if launcher.volume_version(volume) != VOLUME_VERSION:
        raise ValueError("Provider Volume version mismatch")
    prefix = f"{FROZEN['ROOT_PREFIX']}/submissions/{args.run_id}"
    record = launcher.load_json_bytes(launcher.remote_bytes(volume, f"{prefix}/launch.json", limit=MAX_METADATA_BYTES))
    validate_launch(record, args, launcher)
    launcher.write_local_once(args.recovered_receipt, record)
    output, errors = BoundedOutput(), BoundedOutput()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        exitcode = launcher.cmd_inspect(SimpleNamespace(
            launch_receipt=Path(args.recovered_receipt), verify=True,
            verify_canary=False, receipt=None,
        ))
    result = launcher.load_json_bytes(output.getvalue())
    if (not isinstance(result, dict) or result.get("sandbox_id") != args.sandbox_id
            or result.get("run_id") != args.run_id or result.get("volume_version") != VOLUME_VERSION):
        raise ValueError("Inspection result identity mismatch")
    status, returncode = result.get("status"), result.get("returncode")
    accepted = result.get("scientific_acceptance")
    valid = ((status == "RUNNING" and returncode is None and accepted is False and exitcode == 0)
             or (status == "SANDBOX_FAILED" and type(returncode) is int and returncode != 0
                 and accepted is False and exitcode == 1)
             or (status == "DURABLE_RESULT_VERIFIED" and type(returncode) is int and returncode == 0
                 and accepted is True and exitcode == 0 and result.get("controller_status") == "PASSED"
                 and type(result.get("files_verified")) is int and result["files_verified"] > 0))
    if not valid:
        raise ValueError("Inspection result does not establish its reported state")
    safe = {"status": status, "sandbox_id": args.sandbox_id, "run_id": args.run_id,
            "image_id": args.image_id, "submission_code_commit": args.submission_code_commit,
            "volume_id": VOLUME_ID, "volume_version": VOLUME_VERSION,
            "returncode": returncode, "scientific_acceptance": accepted}
    snapshot = safe_snapshot(result.get("last_durable_snapshot"))
    if snapshot is not None:
        safe["last_durable_snapshot"] = snapshot
    if status == "DURABLE_RESULT_VERIFIED":
        safe.update(controller_status="PASSED", files_verified=result["files_verified"])
    return exitcode, safe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "sandbox-id", "image-id", "submission-code-commit", "volume-id", "supervisor-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--volume-version", required=True, type=int)
    parser.add_argument("--submission-root", required=True, type=Path)
    parser.add_argument("--recovered-receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        validate_target(args)
        launcher = load_submission_launcher(args)
        exitcode, result = inspect_existing(args, launcher)
    except Exception:
        print(json.dumps({"status": "INSPECTION_FAILED", "scientific_acceptance": False}))
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
