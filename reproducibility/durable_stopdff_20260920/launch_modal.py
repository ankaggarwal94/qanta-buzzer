#!/usr/bin/env python3
"""Build, submit, and independently inspect the frozen StopDFF CPU Sandbox.

Importing this module performs no authentication or remote operations. The CLI
requires Modal 1.5.5 only for explicit remote subcommands. Credentials are never
written into an image, receipt, command line, or log by this launcher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tarfile
import tempfile

SDK_VERSION = "1.5.5"
APP_NAME = "stopdff-durable-rerun-20260920"
VOLUME_NAME = "cs321m-stopdff-artifacts"
ROOT_PREFIX = "durable-rerun-20260920"
FINAL_COMMIT = "2ed304f6598b94c0c5dc15f77fb8b391ae2f03a3"
INPUT_SHA256 = "6afc62e1cb91d0d2ac958c251ba68aec1c83f2da2b9d0354dc4635234dbc1cf2"
INPUT_SIZE = 1234697660
INPUT_PATH = f"{ROOT_PREFIX}/inputs/{INPUT_SHA256}.tar.gz"
CONTROLLER_SHA256 = "48d4bb32186ddb201f53b4a7f4dafdc10d5f9f382f494612388d678bb381ec50"
CONTROL_HASHES = {
    "README.md": "83bee4ece96adf5b8ca6a9fc4e30425218b9d086642cabb01638a4f2499026d5",
    "requirements.txt": "4de420b711df349ffe5e227dd851c4ef69f1f6a797b0c1f70d33dc5f1811f308",
    "run_durable_rerun.py": CONTROLLER_SHA256,
    "run_persistent.sh": "7a4142f007928d5c88f77d3c716b8849c2c219b5c311e2abff2db37011dc1f82",
}
SCIENTIFIC_PACKAGES = {
    "huggingface_hub": "1.29.0", "matplotlib": "3.11.1", "numpy": "2.4.6",
    "pandas": "3.0.5", "scikit-learn": "1.9.0", "scipy": "1.17.1",
    "sentence-transformers": "6.0.0", "torch": "2.13.0", "transformers": "5.16.1",
    "jsonschema": "4.26.0",
}
BINDING_KEYS = (
    "input_sha256", "input_size", "final_commit", "controller_sha256",
    "supervisor_sha256", "image_id", "commit_mode",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def load_json_bytes(data):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def write_local_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def update_local(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".launch-receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_run_id(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,47}", value):
        raise ValueError("run-id must be 1–48 lowercase letters, digits, periods, underscores, or dashes")
    return value


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Unsafe artifact path")
    return value


def modal_sdk():
    import modal
    if modal.__version__ != SDK_VERSION:
        raise RuntimeError(f"Modal SDK must be exactly {SDK_VERSION}")
    return modal


def expected_bindings(image_receipt, commit_mode):
    if image_receipt.get("status") != "IMAGE_BUILT" or image_receipt.get("final_commit") != FINAL_COMMIT:
        raise ValueError("An exact, successful image build receipt is required")
    if image_receipt.get("control_sha256") != CONTROL_HASHES:
        raise ValueError("Image receipt does not bind the frozen controls")
    if not re.fullmatch(r"im-[A-Za-z0-9]+", image_receipt.get("image_id", "")):
        raise ValueError("Invalid image identity")
    supervisor_sha = image_receipt.get("supervisor_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", supervisor_sha):
        raise ValueError("Missing supervisor identity")
    if commit_mode not in ("sdk", "sync-v2"):
        raise ValueError("Unsupported commit mode")
    return {
        "input_sha256": INPUT_SHA256, "input_size": INPUT_SIZE,
        "final_commit": FINAL_COMMIT, "controller_sha256": CONTROLLER_SHA256,
        "supervisor_sha256": supervisor_sha, "image_id": image_receipt["image_id"],
        "commit_mode": commit_mode,
    }


def require_bindings(record, bindings):
    if any(record.get(key) != bindings[key] for key in BINDING_KEYS):
        raise ValueError("Artifact identity bindings differ from the admitted job")


def build_image(modal, directory):
    """Construct an image recipe without contacting the provider itself."""
    directory = Path(directory)
    for name, expected in CONTROL_HASHES.items():
        if sha256_file(directory / "controls" / name) != expected:
            raise ValueError(f"Frozen control hash mismatch: {name}")
    supervisor = directory / "supervisor.py"
    if not supervisor.is_file() or supervisor.is_symlink():
        raise ValueError("A regular supervisor.py is required")
    storage_canary = directory / "storage_canary.py"
    if not storage_canary.is_file() or storage_canary.is_symlink():
        raise ValueError("A regular storage_canary.py is required")
    assertion = (
        "import importlib.metadata as m,platform; "
        "assert platform.python_version() == '3.11.12'; "
        f"expected={SCIENTIFIC_PACKAGES!r}; "
        "assert {k:m.version(k) for k in expected} == expected"
    )
    image = modal.Image.from_registry("python:3.11.12-slim-bookworm").apt_install("git", "ca-certificates")
    image = image.pip_install(f"modal=={SDK_VERSION}")
    for name in CONTROL_HASHES:
        image = image.add_local_file(directory / "controls" / name, f"/opt/controls/{name}", copy=True)
    image = image.run_commands(
        "python -c \"import platform; assert platform.python_version() == '3.11.12'\"",
        "python -m venv /opt/stopdff-env",
        "/opt/stopdff-env/bin/python -m pip install --disable-pip-version-check -r /opt/controls/requirements.txt",
        "/opt/stopdff-env/bin/python -c " + shlex.quote(assertion),
        "/opt/stopdff-env/bin/python -m pip freeze --all > /opt/scientific-pip-freeze.txt",
        "git clone https://github.com/ankaggarwal94/qanta-buzzer.git /opt/stopdff-code",
        f"git -C /opt/stopdff-code checkout --detach {FINAL_COMMIT}",
        "test -z \"$(git -C /opt/stopdff-code status --porcelain --untracked-files=normal)\"",
    )
    image = image.add_local_file(supervisor, "/opt/supervisor.py", copy=True)
    image = image.add_local_file(storage_canary, "/opt/storage_canary.py", copy=True)
    image = image.env({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
    return image, sha256_file(supervisor)


def cmd_build(args):
    modal = modal_sdk()
    receipt = {
        "schema_version": 1, "status": "BUILDING", "started_utc": utc_now(),
        "final_commit": FINAL_COMMIT, "control_sha256": CONTROL_HASHES,
        "modal_sdk_version": SDK_VERSION, "python_version": "3.11.12",
        "scientific_packages": SCIENTIFIC_PACKAGES,
    }
    write_local_once(args.receipt, receipt)
    try:
        image, supervisor_sha = build_image(modal, args.directory)
        receipt["supervisor_sha256"] = supervisor_sha
        receipt["storage_canary_sha256"] = sha256_file(args.directory / "storage_canary.py")
        update_local(args.receipt, receipt)
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        with modal.enable_output():
            image.build(app)
        receipt.update(status="IMAGE_BUILT", image_id=image.object_id, app_id=app.app_id, ended_utc=utc_now())
    except BaseException as exc:
        receipt.update(status="BUILD_FAILED", error_type=type(exc).__name__, ended_utc=utc_now())
        raise
    finally:
        update_local(args.receipt, receipt)
    print(json.dumps({"status": receipt["status"], "image_id": receipt["image_id"]}))


def remote_bytes(volume, path, limit=16 * 1024 * 1024):
    result = bytearray()
    for chunk in volume.read_file(path):
        result.extend(chunk)
        if len(result) > limit:
            raise ValueError("Receipt exceeds bounded size")
    return bytes(result)


def upload_json_once(volume, path, value):
    body = json_bytes(value)
    with volume.batch_upload(force=False) as batch:
        batch.put_file(io.BytesIO(body), path)
    if remote_bytes(volume, path) != body:
        raise ValueError("Committed record readback mismatch")
    return hashlib.sha256(body).hexdigest()


def bound_volume(modal, expected_id=None):
    """Resolve the existing volume before using its provider identity."""
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    volume.hydrate()
    if not re.fullmatch(r"vo-[A-Za-z0-9]+", volume.object_id):
        raise ValueError("Invalid provider Volume identity")
    if expected_id is not None and volume.object_id != expected_id:
        raise ValueError("Selected Volume identity differs from the admitted job")
    return volume


def supervisor_command(run_id, bindings, volume_id, preflight, canary_path=None, work_seconds=82800):
    prefix = f"/persist/{ROOT_PREFIX}/submissions/{run_id}"
    command = [
        "/usr/local/bin/python", "/opt/supervisor.py", "--run-id", run_id,
        "--input-tar", f"/persist/{INPUT_PATH}", "--durable-dir", prefix,
        "--admission-marker", f"{prefix}/admission.json", "--job-root", "/job",
        "--volume-name", VOLUME_NAME, "--volume-id", volume_id, "--image-id", bindings["image_id"],
        "--commit-mode", bindings["commit_mode"], "--work-seconds", str(work_seconds),
    ]
    if preflight:
        command.append("--preflight-only")
    else:
        if not canary_path:
            raise ValueError("Full execution requires a verified canary receipt")
        command.extend(["--canary-receipt", f"/persist/{safe_relative(canary_path)}"])
    return command


def require_canary(proof, bindings):
    require_bindings(proof, bindings)
    if (proof.get("schema_version") != 1 or proof.get("status") != "CANARY_VERIFIED"
            or proof.get("fresh_reader_verified") is not True
            or not proof.get("canary_sandbox_id") or not proof.get("canary_run_id")
            or not re.fullmatch(r"[0-9a-f]{64}", proof.get("canary_receipt_sha256", ""))):
        raise ValueError("Full run requires a fresh-reader-verified preflight canary")


def sandbox_secret(modal, commit_mode):
    if commit_mode != "sdk":
        return []
    # Called only during explicit launch. Never serialize this dictionary.
    keys = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    if not all(os.environ.get(key) for key in keys):
        raise ValueError("SDK commit mode requires the authorized credential pair in the launcher environment")
    return [modal.Secret.from_dict({key: os.environ[key] for key in keys})]


def cmd_storage_canary(args):
    """Submit only a synthetic storage/environment probe, with no input archive."""
    run_id = validate_run_id(args.run_id)
    image_receipt = load_json_bytes(Path(args.image_receipt).read_bytes())
    expected_bindings(image_receipt, args.commit_mode)
    helper_sha = image_receipt.get("storage_canary_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", helper_sha):
        raise ValueError("Image receipt must bind the synthetic storage canary")
    modal = modal_sdk()
    secret = sandbox_secret(modal, args.commit_mode)
    volume = bound_volume(modal)
    prefix = f"{ROOT_PREFIX}/storage-canaries/{run_id}"
    command = ["/usr/local/bin/python", "/opt/storage_canary.py", "--run-id", run_id,
               "--image-id", image_receipt["image_id"], "--volume-name", VOLUME_NAME,
               "--volume-id", volume.object_id,
               "--commit-mode", args.commit_mode]
    receipt = {
        "schema_version": 1, "status": "ADMITTING", "mode": "storage-canary",
        "run_id": run_id, "image_id": image_receipt["image_id"],
        "supervisor_sha256": image_receipt["supervisor_sha256"],
        "storage_canary_sha256": helper_sha, "commit_mode": args.commit_mode,
        "started_utc": utc_now(), "volume_name": VOLUME_NAME, "volume_id": volume.object_id,
        "durable_prefix": prefix,
        "scientific_acceptance": False, "scope": "synthetic-only",
        "command": command, "cpu": 1.0, "memory_mib": 2048, "timeout_seconds": 600,
    }
    write_local_once(args.receipt, receipt)
    sandbox = None
    try:
        upload_json_once(volume, f"{prefix}/admission.json", receipt)
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        sandbox = modal.Sandbox.create(
            *command, app=app, name=f"stopdff-{run_id}", image=modal.Image.from_id(image_receipt["image_id"]),
            volumes={"/persist": volume}, secrets=secret, timeout=600, cpu=1.0, memory=2048,
            env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
            tags={"run_id": run_id, "mode": "storage-canary", "scope": "synthetic-only"},
        )
        receipt.update(status="SUBMITTED", sandbox_id=sandbox.object_id, app_id=app.app_id,
                       submitted_utc=utc_now())
        update_local(args.receipt, receipt)
        print(json.dumps({"status": "SUBMITTED", "sandbox_id": sandbox.object_id, "run_id": run_id,
                          "scope": "synthetic-only", "scientific_acceptance": False}))
        upload_json_once(volume, f"{prefix}/launch.json", receipt)
    except BaseException as exc:
        receipt.update(status="SUBMISSION_UNCERTAIN" if receipt.get("sandbox_id") else "SUBMISSION_FAILED",
                       error_type=type(exc).__name__, updated_utc=utc_now())
        update_local(args.receipt, receipt)
        raise
    finally:
        if sandbox is not None:
            sandbox.detach()


def verify_storage_canary(volume, launch, completion):
    if launch.get("mode") != "storage-canary" or launch.get("scope") != "synthetic-only":
        raise ValueError("Expected a synthetic-only launch")
    if not re.fullmatch(r"vo-[A-Za-z0-9]+", launch.get("volume_id", "")):
        raise ValueError("Synthetic launch is missing the bound Volume identity")
    for key in ("run_id", "image_id", "volume_id", "supervisor_sha256", "storage_canary_sha256", "commit_mode"):
        if completion.get(key) != launch.get(key):
            raise ValueError("Synthetic canary identity mismatch")
    if (completion.get("schema_version") != 1 or completion.get("status") != "STORAGE_CANARY_PASSED"
            or completion.get("scientific_acceptance") is not False
            or completion.get("research_data_accessed") is not False
            or completion.get("scientific_execution_performed") is not False
            or completion.get("explicit_payload_commit_succeeded") is not True):
        raise ValueError("Synthetic storage gate not passed")
    posix = {name: True for name in ("flock", "hardlinks", "rename", "file_fsync", "directory_fsync", "executable_modes")}
    if completion.get("posix") != posix:
        raise ValueError("Local POSIX probe incomplete")
    if completion.get("environment") != {"python": "3.11.12", "packages": SCIENTIFIC_PACKAGES}:
        raise ValueError("Scientific environment differs from frozen version contract")
    prefix = launch["durable_prefix"]
    payload_receipt = bound_remote_json(volume, prefix, completion["payload_receipt"])
    expected_payload = {key: value for key, value in completion.items()
                        if key not in ("completed_utc", "payload_receipt", "explicit_payload_commit_succeeded")}
    expected_payload["status"] = "SYNTHETIC_DATA_WRITTEN"
    if payload_receipt != expected_payload:
        raise ValueError("Synthetic payload receipt mismatch")
    payload = completion["payload"]
    if payload.get("path") != "synthetic.bin" or payload.get("size") != 4096 or payload.get("mode") != 0o751:
        raise ValueError("Unexpected synthetic payload scope")
    body = remote_bytes(volume, f"{prefix}/synthetic.bin", limit=4096)
    if len(body) != 4096 or hashlib.sha256(body).hexdigest() != payload.get("sha256"):
        raise ValueError("Synthetic fresh-reader payload mismatch")
    return {
        "schema_version": 1, "status": "STORAGE_CANARY_VERIFIED", "scope": "synthetic-only",
        "run_id": launch["run_id"], "image_id": launch["image_id"], "sandbox_id": launch["sandbox_id"],
        "volume_id": launch["volume_id"],
        "fresh_reader_verified": True, "synthetic_payload_bytes": 4096,
        "posix": posix, "environment": completion["environment"],
        "scientific_acceptance": False, "research_data_accessed": False,
        "scientific_execution_performed": False, "verified_utc": utc_now(),
    }


def cmd_inspect_storage_canary(args):
    launch = load_json_bytes(Path(args.launch_receipt).read_bytes())
    if launch.get("mode") != "storage-canary" or launch.get("scope") != "synthetic-only":
        raise ValueError("Expected a synthetic-only launch receipt")
    modal = modal_sdk()
    sandbox = modal.Sandbox.from_id(launch["sandbox_id"])
    diagnostics = None
    try:
        returncode = sandbox.poll()
        if returncode is not None and returncode != 0:
            # This entrypoint emits only bounded JSON metadata; still parse and
            # allowlist fields instead of relaying arbitrary subprocess output.
            output = sandbox.stdout.read()
            if len(output) <= 65536:
                for line in output.splitlines():
                    try:
                        value = load_json_bytes(line)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(value, dict) and value.get("run_id") == launch["run_id"] and value.get("status") == "STORAGE_CANARY_FAILED":
                        diagnostics = {key: value[key] for key in ("stage", "error_type")
                                       if isinstance(value.get(key), str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", value[key])}
    finally:
        sandbox.detach()
    result = {"schema_version": 1, "run_id": launch["run_id"], "sandbox_id": launch["sandbox_id"],
              "returncode": returncode, "scope": "synthetic-only", "scientific_acceptance": False}
    if returncode is None:
        result["status"] = "RUNNING"
        print(json.dumps(result, sort_keys=True))
        return 0
    if returncode != 0:
        result.update(status="STORAGE_CANARY_FAILED", diagnostics=diagnostics)
        if args.receipt:
            write_local_once(args.receipt, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    volume = bound_volume(modal, launch["volume_id"])
    completion = load_json_bytes(remote_bytes(volume, f"{launch['durable_prefix']}/completion.json"))
    result.update(verify_storage_canary(volume, launch, completion))
    if args.receipt:
        write_local_once(args.receipt, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def cmd_launch(args):
    run_id = validate_run_id(args.run_id)
    image_receipt = load_json_bytes(Path(args.image_receipt).read_bytes())
    bindings = expected_bindings(image_receipt, args.commit_mode)
    modal = modal_sdk()
    volume = bound_volume(modal)
    proof = None
    if args.mode == "full":
        if not args.canary_receipt:
            raise ValueError("--canary-receipt is required for full mode")
        proof = load_json_bytes(remote_bytes(volume, safe_relative(args.canary_receipt)))
        require_canary(proof, bindings)
        if run_id == proof["canary_run_id"]:
            raise ValueError("Full execution must use a new submission namespace")
    secret = sandbox_secret(modal, args.commit_mode)
    command = supervisor_command(run_id, bindings, volume.object_id, args.mode == "preflight", args.canary_receipt,
                                 min(82800, args.timeout - 600))
    prefix = f"{ROOT_PREFIX}/submissions/{run_id}"
    receipt = {
        "schema_version": 1, "status": "ADMITTING", "started_utc": utc_now(),
        "run_id": run_id, "mode": args.mode, **bindings,
        "volume_name": VOLUME_NAME, "volume_id": volume.object_id, "durable_prefix": prefix, "command": command,
        "cpu": args.cpu, "memory_mib": args.memory, "timeout_seconds": args.timeout,
        "modal_sdk_version": SDK_VERSION, "canary_receipt": args.canary_receipt,
        "scientific_acceptance": False,
    }
    write_local_once(args.receipt, receipt)
    sandbox = None
    try:
        admission = {"schema_version": 1, "run_id": run_id, "mode": args.mode,
                     "volume_id": volume.object_id, **bindings}
        receipt["admission_sha256"] = upload_json_once(volume, f"{prefix}/admission.json", admission)
        receipt["status"] = "ADMITTED"
        update_local(args.receipt, receipt)
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        sandbox = modal.Sandbox.create(
            *command, app=app, name=f"stopdff-{run_id}",
            image=modal.Image.from_id(bindings["image_id"]), volumes={"/persist": volume},
            secrets=secret, timeout=args.timeout, cpu=args.cpu, memory=args.memory,
            env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
            tags={"run_id": run_id, "mode": args.mode, "final_commit": FINAL_COMMIT},
        )
        receipt.update(status="SUBMITTED", sandbox_id=sandbox.object_id, app_id=app.app_id,
                       submitted_utc=utc_now())
        update_local(args.receipt, receipt)
        print(json.dumps({"status": "SUBMITTED", "sandbox_id": sandbox.object_id, "run_id": run_id}))
        upload_json_once(volume, f"{prefix}/launch.json", receipt)
    except BaseException as exc:
        # A submission error is not proof that no Sandbox exists; retain IDs and
        # admission evidence, never automatically resubmit or terminate here.
        receipt.update(status="SUBMISSION_UNCERTAIN" if receipt.get("sandbox_id") else "SUBMISSION_FAILED",
                       error_type=type(exc).__name__, updated_utc=utc_now())
        update_local(args.receipt, receipt)
        raise
    finally:
        if sandbox is not None:
            sandbox.detach()


def bound_remote_json(volume, prefix, reference):
    path = safe_relative(reference["path"])
    body = remote_bytes(volume, f"{prefix}/{path}")
    if hashlib.sha256(body).hexdigest() != reference["sha256"] or len(body) != reference["size"]:
        raise ValueError("Committed JSON hash or size mismatch")
    return load_json_bytes(body)


def verify_archive(volume, prefix, reference, inventory):
    """Stream the archive from durable storage, then verify all regular members."""
    expected = {}
    for entry in inventory:
        name = safe_relative(entry["path"])
        if name in expected:
            raise ValueError("Duplicate archive inventory entry")
        expected[name] = entry
    digest = hashlib.sha256()
    size = 0
    controller_results = []
    with tempfile.TemporaryFile() as holder:
        for chunk in volume.read_file(f"{prefix}/{safe_relative(reference['path'])}"):
            holder.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        if digest.hexdigest() != reference["sha256"] or size != reference["size"]:
            raise ValueError("Durable archive hash or size mismatch")
        holder.seek(0)
        seen = set()
        with tarfile.open(fileobj=holder, mode="r:gz") as archive:
            for member in archive:
                name = safe_relative(member.name.rstrip("/") if member.isdir() else member.name)
                if member.isdir():
                    continue
                if not member.isfile() or name in seen or name not in expected:
                    raise ValueError("Unsafe or unexpected archive member")
                seen.add(name)
                row = expected[name]
                if member.size != row["size"] or (member.mode & 0o7777) != row["mode"]:
                    raise ValueError("Archive member metadata mismatch")
                file_hash = hashlib.sha256()
                capture = bytearray() if name.startswith("receipts/") and name.endswith("/execution.json") else None
                stream = archive.extractfile(member)
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_hash.update(chunk)
                    if capture is not None:
                        capture.extend(chunk)
                        if len(capture) > 16 * 1024 * 1024:
                            raise ValueError("Controller receipt exceeds bound")
                if file_hash.hexdigest() != row["sha256"]:
                    raise ValueError("Archive member content mismatch")
                if capture is not None:
                    controller_results.append((name, load_json_bytes(capture)))
        if seen != set(expected):
            raise ValueError("Archive inventory is incomplete")
    return controller_results


def verify_completion(volume, launch, completion):
    prefix = launch["durable_prefix"]
    bindings = {key: launch[key] for key in BINDING_KEYS}
    require_bindings(completion, bindings)
    expected_status = "PREFLIGHT_PASSED" if launch["mode"] == "preflight" else "PASSED"
    if (completion.get("schema_version") != 1 or completion.get("run_id") != launch["run_id"]
            or completion.get("status") != expected_status
            or completion.get("scientific_acceptance") is not (launch["mode"] == "full")):
        raise ValueError("Completion does not establish the expected terminal acceptance")
    manifest = bound_remote_json(volume, prefix, completion["final_manifest"])
    require_bindings(manifest, bindings)
    if (manifest.get("run_id") != launch["run_id"] or manifest.get("status") != expected_status
            or manifest.get("scientific_acceptance") != completion["scientific_acceptance"]
            or manifest.get("archive") != completion["archive"]):
        raise ValueError("Final export manifest disagrees with completion")
    results = verify_archive(volume, prefix, completion["archive"], manifest["files"])
    expected_receipts = ["preflight"] if launch["mode"] == "preflight" else ["preflight", "full"]
    if len(results) != len(expected_receipts):
        raise ValueError("Archive lacks the exact controller execution receipt inventory")
    by_stage = {}
    for path, record in results:
        parts = PurePosixPath(path).parts
        if len(parts) != 4 or parts[1] not in expected_receipts or parts[1] in by_stage:
            raise ValueError("Unexpected controller receipt path")
        by_stage[parts[1]] = record
    if set(by_stage) != set(expected_receipts):
        raise ValueError("Missing controller receipt stage")
    for stage, record in by_stage.items():
        full = stage == "full"
        required_stages = ["preflight"] if not full else [
            "preflight", "smoke_package_validation", "model_import", "final_runner",
            "final_integrated_acceptance", "final_numerical_reducer",
        ]
        stages = record.get("stages", [])
        if (record.get("status") != ("PASSED" if full else "PREFLIGHT_PASSED")
                or record.get("stage") != "complete" or not record.get("ended_utc")
                or record.get("scientific_acceptance") is not full
                or record.get("orchestrator_sha256") != CONTROLLER_SHA256
                or [s.get("name") for s in stages] != required_stages
                or any(s.get("status") != "PASSED" or s.get("returncode", 0) != 0 for s in stages)
                or (full and record.get("final_code_commit") != FINAL_COMMIT)):
            raise ValueError("Controller acceptance evidence is incomplete or inconsistent")
    terminal = completion.get("terminal_generation")
    if not terminal:
        raise ValueError("Missing terminal receipt generation")
    generation = bound_remote_json(volume, prefix, terminal)
    require_bindings(generation, bindings)
    if generation.get("schema_version") != 1 or generation.get("run_id") != launch["run_id"]:
        raise ValueError("Terminal generation identity mismatch")
    # Each generation payload path is relative to its generation directory.
    generation_prefix = f"{prefix}/{PurePosixPath(terminal['path']).parent}"
    generation_paths = set()
    for entry in generation["files"]:
        if entry["path"] in generation_paths:
            raise ValueError("Duplicate terminal generation entry")
        generation_paths.add(entry["path"])
        digest = hashlib.sha256()
        size = 0
        for chunk in volume.read_file(f"{generation_prefix}/{safe_relative(entry['path'])}"):
            digest.update(chunk)
            size += len(chunk)
            if size > entry["size"]:
                raise ValueError("Terminal generation payload mismatch")
        if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("Terminal generation payload mismatch")
    return {"status": "DURABLE_RESULT_VERIFIED", "controller_status": expected_status,
            "scientific_acceptance": launch["mode"] == "full", "files_verified": len(manifest["files"])}


def last_durable_status(volume, launch):
    """Return only allowlisted receipt fields, never arbitrary logs or SDK errors."""
    prefix = launch["durable_prefix"]
    bindings = {key: launch[key] for key in BINDING_KEYS}
    result = {"current_liveness_claimed": False,
              "note": "Committed snapshot only; timestamps may be stale. Live state comes from Sandbox.poll."}
    try:
        names = sorted({PurePosixPath(row.path).name for row in volume.listdir(f"{prefix}/generations", recursive=False)
                        if re.fullmatch(r"[0-9]{6}", PurePosixPath(row.path).name)}, reverse=True)
        generation = None
        generation_name = None
        # The newest generation may still be being written; inspect bounded predecessors.
        for name in names[:3]:
            try:
                candidate = load_json_bytes(remote_bytes(volume, f"{prefix}/generations/{name}/manifest.json"))
                require_bindings(candidate, bindings)
                if candidate.get("schema_version") != 1 or candidate.get("run_id") != launch["run_id"]:
                    raise ValueError("Generation run identity mismatch")
                generation, generation_name = candidate, name
                break
            except Exception:
                continue
        if generation is None:
            result["available"] = False
            return result
        result.update(available=True, generation=generation_name)
        entries = {entry["path"]: entry for entry in generation["files"]}
        selected = ["supervisor.json"]
        heartbeats = sorted(path for path in entries if path.startswith("receipts/") and path.endswith("/heartbeat.json"))
        full_heartbeats = [path for path in heartbeats if path.startswith("receipts/full/")]
        if full_heartbeats or heartbeats:
            selected.append((full_heartbeats or heartbeats)[-1])
        for path in selected:
            entry = entries[path]
            body = remote_bytes(volume, f"{prefix}/generations/{generation_name}/{safe_relative(path)}")
            if len(body) != entry["size"] or hashlib.sha256(body).hexdigest() != entry["sha256"]:
                raise ValueError("Snapshot file hash mismatch")
            record = load_json_bytes(body)
            safe = {}
            for key in ("status", "stage", "error_type"):
                value = record.get(key)
                if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", value):
                    safe[key] = value
            for key in ("updated_utc", "started_utc", "ended_utc"):
                value = record.get(key)
                if isinstance(value, str) and re.fullmatch(r"[0-9T:+.Z-]{10,40}", value):
                    safe[key] = value
            for key in ("controller_returncode", "pid", "child_pid"):
                value = record.get(key)
                if value is None or type(value) is int:
                    safe[key] = value
            result["supervisor" if path == "supervisor.json" else "controller_heartbeat"] = safe
    except Exception as exc:
        result.update(available=False, diagnostic_error_type=type(exc).__name__)
    return result


def cmd_inspect(args):
    launch = load_json_bytes(Path(args.launch_receipt).read_bytes())
    modal = modal_sdk()
    sandbox = modal.Sandbox.from_id(launch["sandbox_id"])
    try:
        returncode = sandbox.poll()
    finally:
        sandbox.detach()
    result = {"sandbox_id": launch["sandbox_id"], "run_id": launch["run_id"], "returncode": returncode,
              "scientific_acceptance": False}
    volume = bound_volume(modal, launch["volume_id"])
    if returncode is None:
        result["status"] = "RUNNING"
        result["last_durable_snapshot"] = last_durable_status(volume, launch)
        print(json.dumps(result, sort_keys=True))
        return 0
    if returncode != 0:
        result["status"] = "SANDBOX_FAILED"
        result["last_durable_snapshot"] = last_durable_status(volume, launch)
        print(json.dumps(result, sort_keys=True))
        return 1
    completion_body = remote_bytes(volume, f"{launch['durable_prefix']}/completion.json")
    completion = load_json_bytes(completion_body)
    if not args.verify and not args.verify_canary:
        result.update(status="EXITED_UNVERIFIED", reported_status=completion.get("status"))
        print(json.dumps(result, sort_keys=True))
        return 0
    result.update(verify_completion(volume, launch, completion))
    if args.verify_canary:
        if launch["mode"] != "preflight":
            raise ValueError("Canary verification requires a preflight submission")
        proof = {
            "schema_version": 1, "status": "CANARY_VERIFIED", "fresh_reader_verified": True,
            **{key: launch[key] for key in BINDING_KEYS}, "canary_run_id": launch["run_id"],
            "canary_sandbox_id": launch["sandbox_id"],
            "canary_receipt_sha256": hashlib.sha256(completion_body).hexdigest(),
            "verified_utc": utc_now(),
        }
        proof_path = f"{launch['durable_prefix']}/verified-canary.json"
        upload_json_once(volume, proof_path, proof)
        result["verified_canary_path"] = proof_path
    if args.receipt:
        write_local_once(args.receipt, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build the exact image; no scientific execution")
    build.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    build.add_argument("--receipt", type=Path, required=True)
    build.set_defaults(handler=cmd_build)
    storage = commands.add_parser("storage-canary", help="Launch a synthetic-only 4096-byte persistence/environment probe")
    storage.add_argument("--run-id", required=True)
    storage.add_argument("--image-receipt", type=Path, required=True)
    storage.add_argument("--receipt", type=Path, required=True)
    storage.add_argument("--commit-mode", choices=("sdk", "sync-v2"), default="sdk")
    storage.set_defaults(handler=cmd_storage_canary)
    storage_inspect = commands.add_parser("inspect-storage-canary", help="Poll and independently verify the synthetic-only probe")
    storage_inspect.add_argument("--launch-receipt", type=Path, required=True)
    storage_inspect.add_argument("--receipt", type=Path)
    storage_inspect.set_defaults(handler=cmd_inspect_storage_canary)
    launch = commands.add_parser("launch", help="Submit one detached CPU Sandbox")
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--mode", choices=("preflight", "full"), required=True)
    launch.add_argument("--image-receipt", type=Path, required=True)
    launch.add_argument("--receipt", type=Path, required=True)
    launch.add_argument("--canary-receipt", help="Volume-relative verified canary JSON; required for full mode")
    launch.add_argument("--commit-mode", choices=("sdk", "sync-v2"), default="sdk")
    launch.add_argument("--cpu", type=float, default=8.0)
    launch.add_argument("--memory", type=int, default=32768)
    launch.add_argument("--timeout", type=int, default=86400)
    launch.set_defaults(handler=cmd_launch)
    inspect = commands.add_parser("inspect", help="Poll a prior Sandbox; optionally verify durable exports")
    inspect.add_argument("--launch-receipt", type=Path, required=True)
    inspect.add_argument("--verify", action="store_true")
    inspect.add_argument("--verify-canary", action="store_true")
    inspect.add_argument("--receipt", type=Path)
    inspect.set_defaults(handler=cmd_inspect)
    args = parser.parse_args(argv)
    if args.command == "launch" and (not 1200 <= args.timeout <= 86400 or args.cpu <= 0 or args.memory < 20480):
        parser.error("Use timeout 1200–86400 seconds (including export reserve), positive CPU, and at least 20480 MiB memory")
    return args.handler(args) or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # SDK exceptions can include provider details; do not print their text.
        print(f"Launcher stopped ({type(exc).__name__}); inspect the safe local receipt.", file=sys.stderr)
        raise SystemExit(1)
