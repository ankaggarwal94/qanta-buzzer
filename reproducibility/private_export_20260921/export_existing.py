"""Read and verify one existing StopDFF result; emit only encrypted evidence.

No build, launch, exec, write, commit, or terminate operation is made to Modal.
The unchanged frozen verifier receives a facade exposing only read_file.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

RUN_ID = "full-35536866002-1"
SUBMISSION = "ede184cf77369022ca7cd5fe51b9318573d85630"
PREFIX = "durable-rerun-20260920/submissions/" + RUN_ID
MAX_ARCHIVE = 8 * 1024**3
MAX_METADATA = 16 * 1024**2


def relative_name(value: str) -> str:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError("Unsafe path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(x in (".", "..") for x in value.split("/")):
        raise ValueError("Unsafe path")
    if str(path) != value:
        raise ValueError("Noncanonical path")
    return value


class ReadOnlyEvidence:
    """Cache bounded exact-prefix reads, providing no remote mutation interface."""

    def __init__(self, volume, root: Path):
        self._volume = volume
        self.root = root
        self.archive_limit = MAX_ARCHIVE
        self.total = 0

    def read_file(self, name):
        if not name.startswith(PREFIX + "/"):
            raise ValueError("Read outside fixed result prefix")
        relative = relative_name(name[len(PREFIX) + 1:])
        dest = self.root / relative
        limit = self.archive_limit if relative == "final/evidence.tar.gz" else MAX_METADATA
        if dest.is_file():
            with dest.open("rb") as stream:
                yield from iter(lambda: stream.read(1024**2), b"")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_name(dest.name + ".partial")
        size = 0
        try:
            with temporary.open("xb") as stream:
                for chunk in self._volume.read_file(name):
                    size += len(chunk)
                    self.total += len(chunk)
                    if size > limit or self.total > MAX_ARCHIVE + 512 * 1024**2:
                        raise ValueError("Evidence exceeds export bound")
                    stream.write(chunk)
                    yield chunk
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)


def encrypt_directory(root: Path, certificate: Path, output: Path):
    """OpenSSL CMS: AES-256-GCM with RSA-OAEP/SHA256 recipient key wrapping."""
    if output.exists():
        raise ValueError("Ciphertext destination already exists")
    temporary = output.with_name(output.name + ".partial")
    command = ["openssl", "cms", "-encrypt", "-binary", "-stream", "-aes-256-gcm",
               "-outform", "DER", "-out", str(temporary), "-recip", str(certificate),
               "-keyopt", "rsa_padding_mode:oaep", "-keyopt", "rsa_oaep_md:sha256"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Evidence cache contains a symlink")
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        process.stdin.close()
        # Errors are deliberately not echoed: only a bounded failure category is public.
        if process.wait(timeout=300):
            raise RuntimeError("Evidence encryption failed")
        temporary.replace(output)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        with contextlib.suppress(OSError, ValueError):
            process.stdin.close()
        temporary.unlink(missing_ok=True)


def export(args):
    os.umask(0o077)
    args.private_dir.mkdir(parents=True, exist_ok=False)
    args.encrypted_dir.mkdir(parents=True, exist_ok=False)
    helper_path = args.monitor_root / "reproducibility/durable_stopdff_20260920/inspect_existing.py"
    spec = importlib.util.spec_from_file_location("frozen_inspector", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    target = SimpleNamespace(
        submission_root=args.submission_root, submission_code_commit=SUBMISSION,
        run_id=RUN_ID, sandbox_id="sb-tNlY3NhD7UH0QbiryD3llN",
        image_id="im-tQ21QaPPgHkTu2KZXqHEOD", volume_id="vo-kxgd4TezueO4b7TvuwGyaU",
        volume_version=2,
        supervisor_sha256="a3218776a94a6b897f90179541355e06d9f7a37a911249dd451a3a0c8f255260")
    helper.validate_target(target)
    launcher = helper.load_submission_launcher(target)
    if any(getattr(launcher, key, None) != value for key, value in helper.FROZEN.items()):
        raise ValueError("Frozen implementation differs")
    volume = launcher.bound_volume(launcher.modal_sdk(), target.volume_id)
    if launcher.volume_version(volume) != 2:
        raise ValueError("Volume version mismatch")
    reader = ReadOnlyEvidence(volume, args.private_dir)
    launch = launcher.load_json_bytes(launcher.remote_bytes(reader, PREFIX + "/launch.json"))
    helper.validate_launch(launch, target, launcher)
    completion = launcher.load_json_bytes(launcher.remote_bytes(reader, PREFIX + "/completion.json"))
    reference = completion.get("archive", {})
    size = reference.get("size")
    if (reference.get("path") != "final/evidence.tar.gz" or type(size) is not int
            or not 0 < size <= MAX_ARCHIVE):
        raise ValueError("Archive reference or size is invalid")
    if completion.get("final_manifest", {}).get("path") != "final/manifest.json":
        raise ValueError("Unexpected final manifest path")
    if shutil.disk_usage(args.private_dir).free < size * 3 + 1024**3:
        raise ValueError("Insufficient export disk capacity")
    reader.archive_limit = size
    # This function validates archive hash/size, every member/hash/mode,
    # controller gate receipts, and the terminal-generation payloads unchanged.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = launcher.verify_completion(reader, launch, completion)
    if (result.get("status") != "DURABLE_RESULT_VERIFIED"
            or result.get("controller_status") != "PASSED"
            or result.get("scientific_acceptance") is not True
            or result.get("files_verified") != 1754):
        raise ValueError("Frozen acceptance was not reproduced")
    certificate_hash = hashlib.sha256(args.certificate.read_bytes()).hexdigest()
    receipt = {"run_id": RUN_ID, "sandbox_id": target.sandbox_id,
               "submission_code_commit": SUBMISSION, "scientific_code_commit": helper.FROZEN["FINAL_COMMIT"],
               "archive": reference, "verification": result,
               "recipient_certificate_sha256": certificate_hash,
               "remote_operations": "read existing volume only; no Sandbox operations",
               "source_inspection_url": "https://github.com/ankaggarwal94/qanta-buzzer/actions/runs/35538509457/job/106176946395"}
    (args.private_dir / "export-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    output = args.encrypted_dir / (RUN_ID + ".cms")
    encrypt_directory(args.private_dir, args.certificate, output)
    with output.open("rb") as stream:
        encrypted_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({"status": "VERIFIED_ENCRYPTED_EXPORT_READY", "run_id": RUN_ID,
                      "files_verified": 1754, "encrypted_bytes": output.stat().st_size,
                      "encrypted_sha256": encrypted_hash,
                      "recipient_certificate_sha256": certificate_hash}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("monitor-root", "submission-root", "private-dir", "encrypted-dir", "certificate"):
        parser.add_argument("--" + name, required=True, type=Path)
    try:
        export(parser.parse_args())
        return 0
    except Exception as error:
        print(json.dumps({"status": "EXPORT_FAILED", "error_type": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
