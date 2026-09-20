#!/usr/bin/env python3
"""Durable Sandbox transport and supervision; scientific gates stay in frozen controls.

The mounted Volume is transport only. All scientific filesystem operations use
the Sandbox's local POSIX filesystem. A heartbeat is historical evidence only;
only completion.json, its committed archive, and the frozen controller receipts
can establish a terminal result. This module does not submit or retry jobs.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

INPUT_SHA256 = "6afc62e1cb91d0d2ac958c251ba68aec1c83f2da2b9d0354dc4635234dbc1cf2"
INPUT_SIZE = 1234697660
FINAL_COMMIT = "2ed304f6598b94c0c5dc15f77fb8b391ae2f03a3"
CONTROLLER_SHA256 = "48d4bb32186ddb201f53b4a7f4dafdc10d5f9f382f494612388d678bb381ec50"
PYTHON = Path("/opt/stopdff-env/bin/python")
CONTROLLER = Path("/opt/controls/run_durable_rerun.py")
CODE = Path("/opt/stopdff-code")
EXPORT_SECONDS = 60
MAX_WORK_SECONDS = 23 * 60 * 60
MAX_TOTAL_SECONDS = 24 * 60 * 60 - 100
VOLUME_BACKING_ROOT = Path("/__modal/volumes")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def regular_path(path, directory=False):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"Expected absolute path without symlinks: {path}")
    mode = path.lstat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError(f"Expected regular {'directory' if directory else 'file'}: {path}")
    return path


def canonical_volume_root(mount, volume_id):
    """Allow only Modal's exact alias for the launcher-bound Volume ID.

    SDK 1.5.5 uses one VolumeMount construction for Functions and Sandboxes,
    with backing paths /__modal/volumes/<volume_id>. No other symlink receives
    an exception: the mount parent and the backing directory remain canonical.
    """
    mount = Path(mount)
    if not re.fullmatch(r"vo-[A-Za-z0-9]+", volume_id):
        raise ValueError("Invalid bound Volume ID")
    if not mount.is_absolute() or mount.parent.resolve(strict=True) != mount.parent:
        raise ValueError("Volume mount parent must be an absolute canonical directory")
    if mount.is_symlink():
        expected = VOLUME_BACKING_ROOT / volume_id
        if os.readlink(mount) != str(expected):
            raise ValueError("Volume mount alias does not match the bound Volume ID")
        return regular_path(expected, directory=True)
    return regular_path(mount, directory=True)


def canonical_transport_path(path, mount, root):
    """Translate the verified mount alias once, rejecting descendant symlinks."""
    path, mount, root = Path(path), Path(mount), Path(root)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(mount):
        raise ValueError("Transport path must be contained beneath the mounted Volume")
    target = root / path.relative_to(mount)
    if target.resolve() != target:
        raise ValueError("Transport path traverses a symlink below the mounted Volume")
    return target


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, value, *, exclusive=False):
    """Write local receipts durably, or create immutable transport metadata."""
    path = Path(path)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(json_bytes(value))
        return
    fd, name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    regular_path(path)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def files_under(root, *, skip_receipt_temps=False):
    if not root.exists():
        return []
    regular_path(root, directory=True)
    result = []
    for path in sorted(root.rglob("*")):
        if skip_receipt_temps and path.name.startswith(".receipt-"):
            continue
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            result.append(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"Unsupported link or special file: {path}")
    return result


def copy_exact(source, target, *, check=lambda: None):
    """Copy one fixed-size file snapshot, returning the copied bytes' digest."""
    regular_path(source)
    h = hashlib.sha256()
    size = 0
    with source.open("rb") as inp, target.open("xb") as out:
        # Receipts are atomically replaced while the controller runs. The open
        # descriptor, not another path lookup, defines this snapshot's inode.
        info = os.fstat(inp.fileno())
        remaining = info.st_size
        while remaining:
            check()
            chunk = inp.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"File shrank during copy: {source}")
            out.write(chunk)
            h.update(chunk)
            size += len(chunk)
            remaining -= len(chunk)
        out.flush()
    os.chmod(target, stat.S_IMODE(info.st_mode))
    return {"sha256": h.hexdigest(), "size": size, "mode": stat.S_IMODE(info.st_mode)}


def copy_input(source, target, *, check=lambda: None):
    if regular_path(source).stat().st_size != INPUT_SIZE:
        raise ValueError("Input tar size differs from frozen approval")
    result = copy_exact(source, target, check=check)
    if result["sha256"] != INPUT_SHA256 or result["size"] != INPUT_SIZE:
        raise ValueError("Input tar digest differs from frozen approval")
    with target.open("rb") as stream:
        os.fsync(stream.fileno())
    return result


def safe_extract(archive, destination, *, check=lambda: None):
    """Extract only ordinary files/directories, preserving permission bits.

    No extract/extractall call is used. Traversal, links, devices, duplicate
    entries, privileged modes, and sparse extensions fail closed.
    """
    regular_path(destination, directory=True)
    if any(destination.iterdir()):
        raise ValueError("Extraction destination must be empty")
    seen = set()
    directories = []
    count = 0
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            check()
            raw = member.name.rstrip("/")
            parts = raw.split("/")
            if raw.startswith("/") or any(p in ("", ".", "..") for p in parts):
                raise ValueError("Unsafe tar member path")
            name = PurePosixPath(raw)
            if name in seen:
                raise ValueError("Duplicate tar member")
            seen.add(name)
            if not (member.isdir() or member.isreg()) or member.issparse():
                raise ValueError("Tar contains a link, sparse, or special entry")
            if member.mode & ~0o777:
                raise ValueError("Tar contains privileged file modes")
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                target.mkdir(exist_ok=True)
                directories.append((target, member.mode))
                continue
            if member.size < 0:
                raise ValueError("Negative tar member size")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError("Tar member has no file payload")
            with extracted, target.open("xb") as stream:
                remaining = member.size
                while remaining:
                    check()
                    chunk = extracted.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Truncated tar member")
                    stream.write(chunk)
                    remaining -= len(chunk)
                stream.flush()
                os.fchmod(stream.fileno(), member.mode)
                os.fsync(stream.fileno())
            count += 1
    for path, mode in sorted(directories, key=lambda pair: len(pair[0].parts), reverse=True):
        os.chmod(path, mode)
        fsync_dir(path)
    fsync_dir(destination)
    return count


def check_posix(root):
    """Exercise the local filesystem primitives required by frozen code."""
    with tempfile.TemporaryDirectory(prefix=".posix-", dir=root) as name:
        base = Path(name)
        a, b, c = (base / name for name in ("file", "hardlink", "renamed"))
        with a.open("xb") as stream:
            stream.write(b"posix-canary\n")
            stream.flush()
            os.fchmod(stream.fileno(), 0o751)
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with a.open("rb") as contender:
                try:
                    fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise ValueError("Independent flock did not exclude a contender")
        os.link(a, b)
        if a.stat().st_ino != b.stat().st_ino or a.stat().st_nlink != 2:
            raise ValueError("Hardlink identity mismatch")
        os.rename(b, c)
        fsync_dir(base)
        if b.exists() or c.read_bytes() != b"posix-canary\n" or stat.S_IMODE(c.stat().st_mode) != 0o751:
            raise ValueError("Rename, data, or executable-mode check failed")
    fsync_dir(root)
    return {"flock": True, "hardlinks": True, "rename": True, "file_fsync": True,
            "directory_fsync": True, "executable_modes": True}


def scientific_environment(job):
    """An allowlist prevents supervisor credentials reaching scientific children."""
    return {"PATH": "/opt/stopdff-env/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(job / "home"), "TMPDIR": str(job / "tmp"),
            "STOPDFF_SUPERVISOR_OWNER": str(job),
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def enable_subreaper():
    """Adopt controller grandchildren even if their immediate parent crashes."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot establish scientific child ownership")


def descendants(pid):
    """Track detached scientific process groups as well as the controller."""
    parents = {}
    births = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            data = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(entry.name)] = int(data[1])
            births[int(entry.name)] = data[19]
        except (OSError, ValueError, IndexError):
            continue
    result = set()
    frontier = {pid}
    while frontier:
        frontier = {child for child, parent in parents.items() if parent in frontier} - result
        result.update(frontier)
    return {child: births[child] for child in result}


def same_process(pid, birth):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19] == birth
    except (OSError, IndexError):
        return False


def owned_descendants(owner):
    result = {}
    if owner is None:
        return result
    marker = ("STOPDFF_SUPERVISOR_OWNER=" + str(owner)).encode()
    for pid, birth in descendants(os.getpid()).items():
        try:
            environment = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
            if marker in environment:
                result[pid] = birth
        except OSError:
            pass
    return result


def stop_controller(child, tracked=None, owner=None):
    """Forward TERM and finish descendant cleanup within 25 seconds."""
    deadline = time.monotonic() + 25
    tracked = dict(tracked or {})
    tracked.update(descendants(child.pid))
    tracked.update(owned_descendants(owner))
    if child.poll() is None:
        child.send_signal(signal.SIGTERM)
    while child.poll() is None and time.monotonic() < deadline - 1:
        tracked.update(descendants(child.pid))
        tracked.update(owned_descendants(owner))
        time.sleep(0.1)
    tracked.update(descendants(child.pid))
    tracked.update(owned_descendants(owner))
    for pid, birth in tracked.items():
        if not same_process(pid, birth):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if child.poll() is None:
        child.kill()
    child.wait(timeout=max(0.01, deadline - time.monotonic()))
    # Reap adopted descendants as well, so PID existence cannot masquerade as
    # a surviving scientific process after cleanup.
    for pid in tracked:
        while same_process(pid, tracked[pid]) and time.monotonic() < deadline:
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
                if reaped:
                    break
            except ChildProcessError:
                # A killed grandchild may not have been reparented to the
                # subreaper yet; allow the kernel to complete that transition.
                pass
            time.sleep(0.01)
        if same_process(pid, tracked[pid]):
            raise RuntimeError("Scientific descendant cleanup is unconfirmed")


def require_gate(receipt_root, *, preflight):
    candidates = list(receipt_root.glob("*/execution.json"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one frozen-controller receipt")
    receipt = read_json(candidates[0])
    expected = "PREFLIGHT_PASSED" if preflight else "PASSED"
    if (receipt.get("status") != expected or receipt.get("stage") != "complete"
            or receipt.get("scientific_acceptance") is not (not preflight)
            or receipt.get("orchestrator_sha256") != CONTROLLER_SHA256
            or not receipt.get("ended_utc")):
        raise ValueError("Frozen controller did not record the required terminal gate")
    stages = receipt.get("stages", [])
    required = ["preflight"] if preflight else ["preflight", "smoke_package_validation",
        "model_import", "final_runner", "final_integrated_acceptance", "final_numerical_reducer"]
    if [s.get("name") for s in stages] != required or any(s.get("status") != "PASSED" for s in stages):
        raise ValueError("Frozen controller stage gates are incomplete")
    if not preflight and receipt.get("final_code_commit") != FINAL_COMMIT:
        raise ValueError("Terminal controller scientific commit mismatch")
    return {"path": str(candidates[0]), "sha256": digest(candidates[0]), "status": expected}


class Exporter:
    def __init__(self, job, durable, commit, bindings, run_id):
        self.job, self.durable, self.commit = job, durable, commit
        self.bindings, self.run_id = bindings, run_id
        self.number = 0
        self.mutex = threading.Lock()

    def generation(self):
        """Export fixed file snapshots; commit manifest and payload as one generation."""
        with self.mutex:
            self.number += 1
            relative = f"generations/{self.number:06d}"
            target = self.durable / relative
            target.mkdir(parents=True, exist_ok=False)
            files = []
            for folder in ("receipts", "logs"):
                for source in files_under(self.job / folder, skip_receipt_temps=folder == "receipts"):
                    rel = source.relative_to(self.job)
                    dest = target / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    files.append({"path": str(rel), **copy_exact(source, dest)})
            source = self.job / "supervisor.json"
            files.append({"path": "supervisor.json", **copy_exact(source, target / "supervisor.json")})
            manifest = {"schema_version": 1, "run_id": self.run_id, "generation": self.number,
                        "created_utc": now(), "files": files, **self.bindings,
                        "note": "Historical snapshot only; this does not establish current liveness or acceptance."}
            write_json(target / "manifest.json", manifest, exclusive=True)
            self.commit()
            return {"path": f"{relative}/manifest.json", "sha256": digest(target / "manifest.json"),
                    "size": (target / "manifest.json").stat().st_size}

    def finalize(self, state, *, check=lambda: None):
        """Never create completion.json unless the archive commit succeeded."""
        terminal_generation = self.generation()
        with self.mutex:
            final = self.durable / "final"
            final.mkdir(exist_ok=False)
            archive = self.job / "evidence.tar.gz"
            files = []
            with archive.open("xb") as raw:
                with tarfile.open(fileobj=raw, mode="w|gz", dereference=True) as tar:
                    sources = [self.job / "supervisor.json"]
                    for folder in ("receipts", "logs", "output", "work"):
                        sources.extend(files_under(self.job / folder))
                    for source in sources:
                        check()
                        rel = str(source.relative_to(self.job))
                        regular_path(source)
                        files.append({"path": rel, "sha256": digest(source), "size": source.stat().st_size,
                                      "mode": stat.S_IMODE(source.stat().st_mode)})
                        tar.add(source, arcname=rel, recursive=False)
                raw.flush()
                os.fsync(raw.fileno())
            archive_info = {"path": "final/evidence.tar.gz",
                            **copy_exact(archive, final / "evidence.tar.gz", check=check)}
            manifest = {"schema_version": 1, "run_id": self.run_id, "created_utc": now(),
                        "status": state["status"], "scientific_acceptance": state["scientific_acceptance"],
                        **self.bindings, "archive": archive_info, "files": files}
            write_json(final / "manifest.json", manifest, exclusive=True)
            self.commit()  # Evidence and its hash must be durable before completion exists.
            completion = {"schema_version": 1, "run_id": self.run_id, "completed_utc": now(),
                "status": state["status"], "scientific_acceptance": state["scientific_acceptance"],
                **self.bindings, "archive": archive_info,
                "final_manifest": {"path": "final/manifest.json", "sha256": digest(final / "manifest.json"),
                                   "size": (final / "manifest.json").stat().st_size},
                "terminal_generation": terminal_generation}
            marker = self.durable / "completion.json"
            write_json(marker, completion, exclusive=True)
            try:
                self.commit()
            except BaseException:
                # A background commit might have persisted the marker already; it
                # still references a previously successful evidence commit, never
                # an uncommitted archive. Remove the local marker on reported failure.
                marker.unlink(missing_ok=True)
                raise
            return completion


def modal_commit(mode, volume_name, mount):
    """Use a bounded subprocess so SDK retries cannot consume the export reserve."""
    if mode == "sync-v2":
        command = ["sync", str(mount)]
    else:
        command = [sys.executable, "-c",
                   "import modal,sys; modal.Volume.from_name(sys.argv[1], create_if_missing=False).commit()",
                   volume_name]
    # Never relay SDK stderr or the environment; authentication details stay private.
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Explicit Volume commit timed out") from exc
    if result.returncode:
        raise RuntimeError(f"Explicit Volume commit failed with code {result.returncode}")


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.export_stop = threading.Event()
        self.stop_signal = None
        self.export_error = None
        self.state_lock = threading.RLock()
        self.bindings = {"input_sha256": INPUT_SHA256, "input_size": INPUT_SIZE,
            "final_commit": FINAL_COMMIT, "controller_sha256": CONTROLLER_SHA256,
            "supervisor_sha256": digest(Path(__file__)), "image_id": args.image_id,
            "commit_mode": args.commit_mode}
        self.state = {"schema_version": 1, "run_id": args.run_id, **self.bindings,
            "transport_volume_id": args.volume_id,
            "started_utc": now(), "status": "INITIALIZING", "stage": "admission",
            "scientific_acceptance": False, "controller_pid": None,
            "note": "A historical RUNNING receipt never establishes current liveness or success."}
        self.exporter = None
        self.thread = None

    def save(self, **values):
        with self.state_lock:
            self.state.update(values, updated_utc=now())
            write_json(self.args.job_root / "supervisor.json", self.state)

    def signal(self, number, _frame):
        self.stop_signal = number
        self.stop.set()

    def check(self):
        if self.export_error:
            raise RuntimeError("Periodic evidence export failed")
        if self.stop.is_set():
            raise InterruptedError("Supervisor received termination request")
        if time.monotonic() - self.started >= self.args.work_seconds:
            raise TimeoutError("Scientific deadline reached; export reserve retained")

    def check_total(self):
        if time.monotonic() - self.started >= MAX_TOTAL_SECONDS:
            raise TimeoutError("Terminal evidence export deadline reached")

    def validate_admission(self):
        args = self.args
        admission = read_json(args.admission_marker)
        expected = {"schema_version": 1, "run_id": args.run_id,
                    "mode": "preflight" if args.preflight_only else "full",
                    "volume_id": args.volume_id, **self.bindings}
        if any(admission.get(k) != v for k, v in expected.items()):
            raise ValueError("Permanent admission marker does not match this invocation")
        if digest(CONTROLLER) != CONTROLLER_SHA256:
            raise ValueError("Frozen controller bytes differ from approval")
        if not args.preflight_only:
            if args.canary_receipt is None:
                raise ValueError("Full mode requires a fresh-reader-verified canary")
            canary = read_json(args.canary_receipt)
            if (canary.get("schema_version") != 1 or canary.get("status") != "CANARY_VERIFIED"
                    or canary.get("fresh_reader_verified") is not True
                    or any(canary.get(k) != v for k, v in self.bindings.items())
                    or not all(canary.get(k) for k in ("canary_run_id", "canary_sandbox_id", "canary_receipt_sha256"))):
                raise ValueError("Canary readback binding is absent or inconsistent")
            self.state["verified_canary_sha256"] = digest(args.canary_receipt)
        self.state["admission_sha256"] = digest(args.admission_marker)

    def export_loop(self):
        while not self.export_stop.wait(EXPORT_SECONDS):
            try:
                self.save()
                self.exporter.generation()
            except BaseException as exc:
                self.export_error = type(exc).__name__
                self.stop.set()
                return

    def controller(self, *, preflight):
        self.check()
        args = self.args
        stage = "preflight" if preflight else "full"
        receipts = args.job_root / "receipts" / stage
        receipts.mkdir()
        command = [str(PYTHON), "-u", str(CONTROLLER), "--input-dir", str(args.job_root / "input"),
            "--code-dir", str(CODE), "--work-dir", str(args.job_root / "work"),
            "--output-dir", str(args.job_root / "output"), "--receipt-dir", str(receipts)]
        if preflight:
            command.append("--preflight-only")
        self.save(stage=stage, status="RUNNING")
        with (args.job_root / "logs" / f"{stage}.stdout.log").open("xb") as out, \
                (args.job_root / "logs" / f"{stage}.stderr.log").open("xb") as err:
            child = subprocess.Popen(command, cwd=CODE, env=scientific_environment(args.job_root),
                                     stdout=out, stderr=err, start_new_session=True)
            tracked = {}
            try:
                self.save(controller_pid=child.pid)
                while child.poll() is None:
                    tracked.update(descendants(child.pid))
                    self.check()
                    self.stop.wait(0.5)
            finally:
                stop_controller(child, tracked, args.job_root)
                self.save(controller_pid=None, controller_returncode=child.returncode)
        self.check()
        if child.returncode != 0:
            raise RuntimeError(f"Frozen controller {stage} exited {child.returncode}")
        gate = require_gate(receipts, preflight=preflight)
        self.save(**{f"{stage}_gate": gate})

    def run(self):
        args = self.args
        self.validate_admission()
        args.job_root.mkdir(exist_ok=False)
        for name in ("receipts", "logs", "work", "home", "tmp", "unpacked"):
            (args.job_root / name).mkdir()
        self.save()
        commit = lambda: modal_commit(args.commit_mode, args.volume_name, args.volume_mount)
        args.durable_dir.mkdir(parents=True, exist_ok=True)
        self.exporter = Exporter(args.job_root, args.durable_dir, commit, self.bindings, args.run_id)
        # Never reuse a started submission. This is a permanent admission record,
        # not a distributed lock; the launcher is responsible for one submission.
        write_json(args.durable_dir / "started.json", {**self.state, "mode": "preflight" if args.preflight_only else "full"}, exclusive=True)
        result = 1
        try:
            commit()
            self.exporter.generation()
            self.thread = threading.Thread(target=self.export_loop, daemon=True)
            self.thread.start()
            self.save(stage="local_posix", posix=check_posix(args.job_root))
            enable_subreaper()
            self.save(stage="copy_input")
            copied = args.job_root / "input.tar.gz"
            copy_input(args.input_tar, copied, check=self.check)
            self.save(stage="extract_input")
            unpacked = args.job_root / "unpacked"
            count = safe_extract(copied, unpacked, check=self.check)
            if {p.name for p in unpacked.iterdir()} != {"input"}:
                raise ValueError("Expected only input/ at tar root")
            if {p.name for p in (unpacked / "input").iterdir()} != {"smoke_package", "adapter_bundle", "provenance.json"}:
                raise ValueError("Unexpected frozen input layout")
            os.rename(unpacked / "input", args.job_root / "input")
            fsync_dir(args.job_root)
            self.save(input_files=count)
            self.controller(preflight=True)
            self.exporter.generation()
            if not args.preflight_only:
                self.controller(preflight=False)
            self.save(status="PREFLIGHT_PASSED" if args.preflight_only else "PASSED",
                      stage="terminal_export", scientific_acceptance=not args.preflight_only, ended_utc=now())
            result = 0
        except BaseException as exc:
            interrupted = isinstance(exc, (InterruptedError, KeyboardInterrupt)) and not self.export_error
            self.save(status="INTERRUPTED" if interrupted else "FAILED", stage="terminal_export",
                      scientific_acceptance=False, error_type=type(exc).__name__,
                      error=str(exc), ended_utc=now())
            result = 128 + (self.stop_signal or signal.SIGTERM) if interrupted else 1
        finally:
            self.export_stop.set()
            if self.thread:
                self.thread.join(timeout=130)
            if self.thread and self.thread.is_alive():
                self.save(status="EXPORT_FAILED", scientific_acceptance=False,
                          error="Export worker did not stop; no completion claimed")
                return 2
        try:
            self.exporter.finalize(self.state, check=self.check_total)
        except BaseException as exc:
            self.save(status="EXPORT_FAILED", scientific_acceptance=False,
                      error_type=type(exc).__name__, error="Terminal evidence export failed; no completion claimed")
            print("Terminal evidence export failed; local evidence retained; no completion claimed", flush=True)
            return 2
        print(json.dumps({"run_id": args.run_id, "status": self.state["status"],
                          "scientific_acceptance": self.state["scientific_acceptance"],
                          "completion": str(args.durable_dir / "completion.json")}), flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-tar", type=Path, required=True)
    parser.add_argument("--durable-dir", type=Path, required=True)
    parser.add_argument("--admission-marker", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--job-root", type=Path, default=Path("/job"))
    parser.add_argument("--volume-name", default="cs321m-stopdff-artifacts")
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--volume-mount", type=Path, default=Path("/persist"))
    parser.add_argument("--commit-mode", choices=("sdk", "sync-v2"), default="sdk")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--work-seconds", type=int, default=MAX_WORK_SECONDS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", args.run_id):
        parser.error("Invalid run ID")
    if not 1 <= args.work_seconds <= MAX_WORK_SECONDS:
        parser.error("Work deadline must leave the terminal export reserve")
    try:
        mount = args.volume_mount
        root = canonical_volume_root(mount, args.volume_id)
        for name in ("input_tar", "durable_dir", "admission_marker", "canary_receipt"):
            path = getattr(args, name)
            if path is not None:
                setattr(args, name, canonical_transport_path(path, mount, root))
        args.volume_mount = root
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if (not args.job_root.is_absolute() or args.job_root.resolve() != args.job_root
            or args.job_root.is_relative_to(args.volume_mount) or args.volume_mount.is_relative_to(args.job_root)):
        parser.error("Job root must be a separate regular local filesystem path")
    supervisor = Supervisor(args)
    signal.signal(signal.SIGTERM, supervisor.signal)
    signal.signal(signal.SIGINT, supervisor.signal)
    try:
        return supervisor.run()
    except BaseException as exc:
        print(f"Supervisor initialization failed ({type(exc).__name__}); no completion claimed", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
