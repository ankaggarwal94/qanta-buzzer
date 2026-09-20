"""One-hour, authenticated transfer of the single approved StopDFF archive.

Deploy with Modal 1.5.5 after setting STOPDFF_TRANSFER_TOKEN and
STOPDFF_TRANSFER_EXPIRES. Only the derived transfer token enters this app;
the deployer must never inject its Modal credentials. No endpoint reads data.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import hmac
import json
import math
import os
import re
import stat
import time
from pathlib import Path

ARCHIVE_SIZE = 1_234_697_660
ARCHIVE_SHA256 = "6afc62e1cb91d0d2ac958c251ba68aec1c83f2da2b9d0354dc4635234dbc1cf2"
CHUNK_SIZE = 32 * 1024 * 1024
VOLUME_NAME = "cs321m-stopdff-rerun-20260920-v2"
MOUNT = "/transfer-volume"
VOLUME_BACKING_DIRECTORY = Path("/__modal/volumes")
INPUT_DIRECTORY = "durable-rerun-20260920/inputs"
REQUEST_SECONDS = 130
MAX_REQUESTS = 512
HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")


class TransferError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


def publish_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish on Linux without replacing any existing object.

    Unsupported filesystems fail closed; plain rename is deliberately not a
    fallback, because it would overwrite historical evidence in a race.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise TransferError(503, "atomic no-replace publication unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == 17:  # EEXIST
            raise TransferError(409, "destination already exists")
        raise TransferError(503, "atomic no-replace publication failed")


class TransferASGI:
    """Small ASGI HTTP adapter with injected async volume operations for tests."""

    def __init__(self, root: Path, token: str, expires: int, reload_volume, commit_volume,
                 *, size: int = ARCHIVE_SIZE, sha256: str = ARCHIVE_SHA256,
                 chunk_size: int = CHUNK_SIZE, request_seconds: float = REQUEST_SECONDS,
                 volume_id: str | None = None):
        if not token or len(token) < 32 or not HEX_SHA.fullmatch(sha256):
            raise ValueError("invalid transfer configuration")
        if size <= 0 or chunk_size <= 0 or expires > time.time() + 3600 + 5:
            raise ValueError("invalid transfer limits")
        if volume_id is not None and not re.fullmatch(r"vo-[A-Za-z0-9_-]+", volume_id):
            raise ValueError("invalid bound volume identity")
        self.public_root = Path(root)
        self.expected_root = VOLUME_BACKING_DIRECTORY / volume_id if volume_id is not None else None
        self.token = token.encode("utf-8")
        self.expires = expires
        self.reload = reload_volume
        self.commit = commit_volume
        self.size = size
        self.sha256 = sha256
        self.chunk_size = chunk_size
        self.chunk_count = math.ceil(size / chunk_size)
        self.request_seconds = request_seconds
        self.requests = 0
        self.lock = asyncio.Lock()
        self.set_root(self.public_root)

    def set_root(self, root):
        self.root = root
        self.inputs = self.root / INPUT_DIRECTORY
        self.staging = self.inputs / (".transfer-" + self.sha256)
        self.final = self.inputs / (self.sha256 + ".tar.gz")

    def check_time(self, deadline):
        if time.time() >= self.expires:
            raise TransferError(410, "transfer expired")
        if time.monotonic() >= deadline:
            raise TransferError(408, "request deadline exceeded")

    def prepare(self):
        """Validate the bound mount, then reject every descendant symlink."""
        root = self.public_root
        try:
            root.mkdir()
        except FileExistsError:
            pass
        if stat.S_ISLNK(root.lstat().st_mode) and self.expected_root is not None:
            # Modal may expose its mounted Volume via a provider-created link.
            # The only permitted target comes from the hydrated bound Volume,
            # never a request value or an arbitrary filesystem resolution.
            try:
                if Path(os.readlink(root)) != self.expected_root:
                    raise TransferError(409, "storage mount backing mismatch")
                if (not stat.S_ISDIR(self.expected_root.lstat().st_mode)
                        or self.expected_root.resolve(strict=True) != self.expected_root):
                    raise TransferError(409, "storage mount backing mismatch")
            except OSError:
                raise TransferError(409, "storage mount backing unavailable") from None
            root = self.expected_root
        self.set_root(root)
        paths = (("mount", root),
                 ("namespace", root / "durable-rerun-20260920"),
                 ("inputs", self.inputs), ("staging", self.staging))
        for label, path in paths:
            try:
                path.mkdir()
            except FileExistsError:
                pass
            mode = path.lstat().st_mode
            if not stat.S_ISDIR(mode):
                # Diagnostic labels are fixed constants. Never follow a
                # symlink or reveal its target, a full path, or file contents.
                file_type = {stat.S_IFREG: "regular", stat.S_IFLNK: "symlink",
                             stat.S_IFIFO: "fifo", stat.S_IFSOCK: "socket",
                             stat.S_IFBLK: "block_device", stat.S_IFCHR: "character_device"
                             }.get(stat.S_IFMT(mode), "unknown")
                raise TransferError(409, f"storage path conflict: {label} ({file_type})")

    @staticmethod
    def exists(path):
        return os.path.lexists(path)

    @staticmethod
    def open_regular(path, *, write=False):
        flags = os.O_NOFOLLOW | (os.O_WRONLY | os.O_CREAT | os.O_TRUNC if write else os.O_RDONLY)
        try:
            fd = os.open(path, flags, 0o600)
        except OSError:
            raise TransferError(409, "storage file conflict") from None
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise TransferError(409, "storage file conflict")
        return os.fdopen(fd, "wb" if write else "rb")

    def digest_file(self, path, deadline):
        digest = hashlib.sha256()
        count = 0
        with self.open_regular(path) as handle:
            while block := handle.read(1024 * 1024):
                self.check_time(deadline)
                digest.update(block)
                count += len(block)
                if count > self.size:
                    raise TransferError(409, "existing file size mismatch")
        return count, digest.hexdigest()

    def chunk_path(self, index):
        return self.staging / f"chunk-{index:04d}"

    def expected_chunk_size(self, index):
        return min(self.chunk_size, self.size - index * self.chunk_size)

    async def receive_body(self, receive, expected, deadline, sink=None):
        digest = hashlib.sha256()
        count = 0
        while True:
            self.check_time(deadline)
            event = await asyncio.wait_for(receive(), max(0.001, deadline - time.monotonic()))
            if event["type"] == "http.disconnect":
                raise TransferError(400, "request disconnected")
            if event["type"] != "http.request":
                raise TransferError(400, "invalid request")
            body = event.get("body", b"")
            count += len(body)
            if count > expected:
                raise TransferError(413, "request body too large")
            digest.update(body)
            if sink is not None:
                sink.write(body)
            if not event.get("more_body", False):
                break
        if count != expected:
            raise TransferError(400, "request body length mismatch")
        return digest.hexdigest()

    async def chunk(self, index, headers, receive, deadline):
        expected = self.expected_chunk_size(index)
        if headers.get(b"content-length") != str(expected).encode():
            raise TransferError(400, "exact Content-Length required")
        supplied = headers.get(b"x-chunk-sha256", b"").decode("ascii", errors="replace")
        if not HEX_SHA.fullmatch(supplied):
            raise TransferError(400, "valid X-Chunk-SHA256 required")
        path = self.chunk_path(index)
        temp = self.staging / "incoming.tmp"
        try:
            with self.open_regular(temp, write=True) as handle:
                actual = await self.receive_body(receive, expected, deadline, handle)
                handle.flush()
                os.fsync(handle.fileno())
            if not hmac.compare_digest(actual, supplied):
                raise TransferError(422, "chunk digest mismatch")
            if self.exists(path):
                if self.digest_file(path, deadline) != (expected, actual):
                    raise TransferError(409, "immutable chunk conflict")
                status = "already_present"
            else:
                self.check_time(deadline)
                publish_no_replace(temp, path)
                status = "stored"
            if self.exists(temp):
                temp.unlink()
            # Also retry commit for idempotent requests after an earlier failure.
            await self.commit()
            return {"status": status, "index": index, "size": expected, "sha256": actual}
        finally:
            if self.exists(temp):
                temp.unlink()

    async def verify_final(self, deadline):
        if self.digest_file(self.final, deadline) != (self.size, self.sha256):
            raise TransferError(409, "existing archive mismatch")
        await self.commit()  # Never report complete after a failed prior commit.
        return {"status": "complete", "size": self.size, "sha256": self.sha256}

    async def finalize(self, deadline):
        if self.exists(self.final):
            return await self.verify_final(deadline)
        paths = [self.chunk_path(index) for index in range(self.chunk_count)]
        if any(not self.exists(path) for path in paths):
            raise TransferError(409, "chunks missing")
        temp = self.staging / "assembled.tmp"
        digest = hashlib.sha256()
        total = 0
        try:
            with self.open_regular(temp, write=True) as output:
                for index, path in enumerate(paths):
                    count = 0
                    with self.open_regular(path) as source:
                        while block := source.read(1024 * 1024):
                            self.check_time(deadline)
                            count += len(block)
                            if count > self.expected_chunk_size(index):
                                raise TransferError(409, "stored chunk size mismatch")
                            digest.update(block)
                            output.write(block)
                    if count != self.expected_chunk_size(index):
                        raise TransferError(409, "stored chunk size mismatch")
                    total += count
                output.flush()
                os.fsync(output.fileno())
            if total != self.size or not hmac.compare_digest(digest.hexdigest(), self.sha256):
                raise TransferError(422, "archive digest mismatch")
            self.check_time(deadline)
            publish_no_replace(temp, self.final)
            await self.commit()
            return {"status": "complete", "size": total, "sha256": self.sha256}
        finally:
            if self.exists(temp):
                temp.unlink()

    async def dispatch(self, scope, receive, deadline):
        pairs = scope.get("headers", [])
        headers = {key.lower(): value for key, value in pairs}
        if any(sum(key.lower() == name for key, _ in pairs) > 1 for name in
               (b"x-transfer-token", b"content-length", b"x-chunk-sha256")):
            raise TransferError(400, "duplicate request header")
        if not hmac.compare_digest(headers.get(b"x-transfer-token", b""), self.token):
            raise TransferError(401, "unauthorized")
        self.check_time(deadline)
        self.requests += 1
        if self.requests > MAX_REQUESTS:
            raise TransferError(429, "transfer request limit exceeded")
        method, path = scope["method"], scope["path"]
        match = re.fullmatch(r"/chunk/(0|[1-9][0-9]{0,3})", path)
        if not ((method == "PUT" and match) or (method, path) in
                (("GET", "/status"), ("POST", "/finalize"))):
            raise TransferError(404, "not found")
        if scope.get("query_string") or b"content-encoding" in headers:
            raise TransferError(400, "unsupported request options")
        if match and int(match[1]) >= self.chunk_count:
            raise TransferError(400, "chunk index out of range")
        await self.reload()
        self.prepare()
        if match:
            return await self.chunk(int(match[1]), headers, receive, deadline)
        await self.receive_body(receive, 0, deadline)
        if path == "/finalize":
            return await self.finalize(deadline)
        complete = self.exists(self.final)
        if complete:
            await self.verify_final(deadline)
        return {"status": "complete" if complete else "uploading", "size": self.size,
                "sha256": self.sha256, "chunk_count": self.chunk_count,
                "received_chunks": [i for i in range(self.chunk_count) if self.exists(self.chunk_path(i))]}

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        try:
            async with asyncio.timeout(self.request_seconds):
                async with self.lock:
                    result = await self.dispatch(scope, receive, time.monotonic() + self.request_seconds)
            status = 200
        except TransferError as exc:
            status, result = exc.status, {"status": "error", "detail": exc.message}
        except TimeoutError:
            status, result = 408, {"status": "error", "detail": "request deadline exceeded"}
        except Exception:
            # No exception text, request headers, credentials, or data in replies.
            status, result = 503, {"status": "error", "detail": "storage operation failed; retry safely"}
        body = json.dumps(result, separators=(",", ":")).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})


# Tests import only the stdlib adapter. Deployment explicitly opts in so local
# adapter tests never read any environment-held token or construct a Secret.
if os.environ.get("STOPDFF_TRANSFER_DEPLOY") == "1":
    import modal

    app = modal.App("stopdff-input-transfer-20260920")
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    secret = modal.Secret.from_dict({
        "STOPDFF_TRANSFER_TOKEN": os.environ.get("STOPDFF_TRANSFER_TOKEN", ""),
        "STOPDFF_TRANSFER_EXPIRES": os.environ.get("STOPDFF_TRANSFER_EXPIRES", "0"),
    })

    @app.function(image=modal.Image.debian_slim(python_version="3.11"),
                  env={"STOPDFF_TRANSFER_DEPLOY": "1"},
                  volumes={MOUNT: volume}, secrets=[secret], cpu=2, memory=1024,
                  timeout=140, max_containers=1, min_containers=0,
                  scaledown_window=60, enable_memory_snapshot=False)
    @modal.concurrent(max_inputs=1)
    @modal.asgi_app()
    def transfer_api():
        return TransferASGI(Path(MOUNT), os.environ["STOPDFF_TRANSFER_TOKEN"],
                            int(os.environ["STOPDFF_TRANSFER_EXPIRES"]),
                            volume.reload.aio, volume.commit.aio, volume_id=volume.object_id)
