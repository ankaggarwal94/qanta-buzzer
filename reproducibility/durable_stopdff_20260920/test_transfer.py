"""Offline REST/ASGI tests using only dummy credentials and a temporary folder."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from transfer_modal import TransferASGI, publish_no_replace, TransferError


class TransferTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = b"approved archive contents"
        self.token = "local-test-token-never-a-real-secret-1234"
        self.commits = 0
        self.reloads = 0
        self.fail_commit = False

        async def reload_volume():
            self.reloads += 1

        async def commit_volume():
            self.commits += 1
            if self.fail_commit:
                raise RuntimeError("simulated commit failure")

        self.app = TransferASGI(self.root, self.token, int(time.time()) + 300,
                                reload_volume, commit_volume, size=len(self.data),
                                sha256=hashlib.sha256(self.data).hexdigest(), chunk_size=8)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def request(self, method, path, body=b"", *, token=None, headers=None,
                      events=None, query=b""):
        pairs = [(b"x-transfer-token", (self.token if token is None else token).encode())]
        pairs.extend((key.encode(), str(value).encode()) for key, value in (headers or {}).items())
        sent = []
        incoming = list(events or [{"type": "http.request", "body": body}])

        async def receive():
            return incoming.pop(0)

        async def send(message):
            sent.append(message)

        await self.app({"type": "http", "method": method, "path": path,
                        "headers": pairs, "query_string": query}, receive, send)
        return sent[0]["status"], json.loads(sent[1]["body"])

    async def put(self, index, data=None, **kwargs):
        if data is None:
            data = self.data[index * 8:(index + 1) * 8]
        return await self.request("PUT", f"/chunk/{index}", data,
                                  headers={"content-length": len(data),
                                           "x-chunk-sha256": hashlib.sha256(data).hexdigest()}, **kwargs)

    async def upload(self):
        for index in range(self.app.chunk_count):
            self.assertEqual((await self.put(index))[0], 200)

    async def test_round_trip_resume_finalize_and_only_metadata_reads(self):
        await self.upload()
        code, duplicate = await self.put(0)
        self.assertEqual((code, duplicate["status"]), (200, "already_present"))
        code, status = await self.request("GET", "/status")
        self.assertEqual(code, 200)
        self.assertEqual(status["received_chunks"], list(range(self.app.chunk_count)))
        self.assertEqual((await self.request("POST", "/finalize"))[0], 200)
        self.assertEqual(self.app.final.read_bytes(), self.data)
        self.assertEqual((await self.request("POST", "/finalize"))[0], 200)
        self.assertEqual((await self.request("GET", "/status"))[1]["status"], "complete")
        for path in ("/", "/chunk/0", "/docs", "/openapi.json", "/../secret", "/" + self.app.final.name):
            self.assertEqual((await self.request("GET", path))[0], 404)

    async def test_unauthorized_and_expired_do_not_touch_storage(self):
        self.assertEqual((await self.request("GET", "/status", token="wrong"))[0], 401)
        self.app.expires = 0
        self.assertEqual((await self.request("GET", "/status"))[0], 410)
        self.assertEqual((self.reloads, self.commits), (0, 0))
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_lengths_digest_indices_and_body_stream_are_bounded(self):
        for path in ("/chunk/9999", "/chunk/-1", "/chunk/00", "/chunk/0/../1"):
            self.assertIn((await self.request("PUT", path))[0], (400, 404))
        self.assertEqual((await self.request("PUT", "/chunk/0", b"a"))[0], 400)
        headers = {"content-length": 8, "x-chunk-sha256": "0" * 64}
        self.assertEqual((await self.request("PUT", "/chunk/0", b"12345678", headers=headers))[0], 422)
        self.assertEqual((await self.request("PUT", "/chunk/0", b"123456789", headers=headers))[0], 413)
        self.assertEqual((await self.request("PUT", "/chunk/0", b"123", headers=headers))[0], 400)
        self.assertEqual((await self.request("POST", "/finalize", b"unexpected"))[0], 413)
        self.assertFalse(self.app.chunk_path(0).exists())
        self.assertFalse((self.app.staging / "incoming.tmp").exists())

    async def test_immutable_chunk_and_existing_archive_mismatch(self):
        self.assertEqual((await self.put(0))[0], 200)
        self.assertEqual((await self.put(0, b"different"[:8]))[0], 409)
        self.assertEqual(self.app.chunk_path(0).read_bytes(), self.data[:8])
        self.app.final.write_bytes(b"historical evidence")
        self.assertEqual((await self.request("POST", "/finalize"))[0], 409)
        self.assertEqual(self.app.final.read_bytes(), b"historical evidence")

    async def test_final_hash_failure_never_publishes(self):
        await self.upload()
        self.app.chunk_path(0).write_bytes(b"tampered")
        self.assertEqual((await self.request("POST", "/finalize"))[0], 422)
        self.assertFalse(self.app.final.exists())
        self.assertFalse((self.app.staging / "assembled.tmp").exists())

    async def test_commit_failure_is_retried_before_success(self):
        self.fail_commit = True
        self.assertEqual((await self.put(0))[0], 503)
        self.fail_commit = False
        self.assertEqual((await self.put(0))[0], 200)
        await self.upload()
        self.fail_commit = True
        self.assertEqual((await self.request("POST", "/finalize"))[0], 503)
        self.assertEqual((await self.request("GET", "/status"))[0], 503)
        self.fail_commit = False
        self.assertEqual((await self.request("POST", "/finalize"))[0], 200)
        self.assertEqual(self.app.final.read_bytes(), self.data)

    async def test_symlink_and_atomic_no_replace_fail_closed(self):
        self.app.prepare()
        self.app.final.symlink_to(self.root / "missing")
        self.assertEqual((await self.request("POST", "/finalize"))[0], 409)
        self.assertTrue(self.app.final.is_symlink())
        source, destination = self.root / "source", self.root / "destination"
        source.write_bytes(b"new")
        destination.write_bytes(b"old")
        with self.assertRaises(TransferError):
            publish_no_replace(source, destination)
        self.assertEqual(destination.read_bytes(), b"old")

    async def test_deadline_cancels_slow_request_without_publishing(self):
        self.app.request_seconds = 0.01

        async def slow():
            await asyncio.sleep(1)
            return {"type": "http.request", "body": b""}

        sent = []

        async def send(event):
            sent.append(event)

        await self.app({"type": "http", "method": "PUT", "path": "/chunk/0",
                        "headers": [(b"x-transfer-token", self.token.encode()),
                                    (b"content-length", b"8"), (b"x-chunk-sha256", b"0" * 64)]}, slow, send)
        self.assertEqual(sent[0]["status"], 408)
        self.assertFalse(self.app.chunk_path(0).exists())
        self.assertFalse((self.app.staging / "incoming.tmp").exists())

    async def test_request_cap_and_unsupported_options(self):
        self.assertEqual((await self.request("GET", "/status", query=b"path=secret"))[0], 400)
        self.app.requests = 512
        self.assertEqual((await self.request("GET", "/status"))[0], 429)

    def test_modal_deployment_constructs_without_hydration_or_credentials(self):
        # The child receives a fresh environment containing only dummy transfer
        # values. Constructing decorators must not make a network request.
        result = subprocess.run(
            [sys.executable, "-c", "import transfer_modal; assert transfer_modal.app is not None; "
             "assert transfer_modal.transfer_api is not None"],
            cwd=Path(__file__).parent,
            env={"STOPDFF_TRANSFER_DEPLOY": "1", "STOPDFF_TRANSFER_EXPIRES": "0",
                 "STOPDFF_TRANSFER_TOKEN": "local-test-token-never-a-real-secret-1234"},
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
