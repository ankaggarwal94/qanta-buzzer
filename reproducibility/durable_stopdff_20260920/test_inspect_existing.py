"""Offline monitoring checks with a provider exposing only existing objects."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import inspect_existing as monitor
import launch_modal as launcher


class FakeProvider:
    def __init__(self, launch, returncode=None):
        self.calls, self.mutations, self.sandbox_calls = [], [], []
        self.object_id = monitor.VOLUME_ID
        self.version = 2
        self.returncode = returncode
        self.files = {launch["durable_prefix"] + "/launch.json": launcher.json_bytes(launch)}
        self.Volume = SimpleNamespace(from_name=self.from_name)
        self.Sandbox = SimpleNamespace(from_id=self.from_id, create=self.forbidden)
        self.App = SimpleNamespace(lookup=self.forbidden)

    def forbidden(self, *args, **kwargs):
        self.mutations.append((args, kwargs))
        raise AssertionError("Remote mutation attempted")

    commit = reload = batch_upload = exec = build = create = forbidden

    def from_name(self, name, *, create_if_missing):
        if name != monitor.VOLUME_NAME or create_if_missing is not False:
            return self.forbidden(name, create_if_missing=create_if_missing)
        self.calls.append("Volume.from_name(existing-only)")
        return self

    def hydrate(self):
        self.calls.append("hydrate")

    def _get_metadata(self):
        return SimpleNamespace(version=self.version)

    def read_file(self, path):
        self.calls.append(("read_file", path))
        yield self.files[path]

    def listdir(self, path, recursive=False):
        self.calls.append(("listdir", path))
        return []

    def from_id(self, sandbox_id):
        self.sandbox_calls.append(sandbox_id)
        return SimpleNamespace(poll=lambda: self.returncode, detach=lambda: None, exec=self.forbidden)


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.args = SimpleNamespace(
            run_id="full-observed-run", sandbox_id="sb-Observed123", image_id="im-Observed123",
            submission_code_commit="1" * 40, volume_id=monitor.VOLUME_ID, volume_version=2,
            supervisor_sha256=monitor.SUPERVISOR_SHA256,
            recovered_receipt=Path(self.temp.name) / "launch.json",
        )
        self.launch = {
            "schema_version": 1, "status": "SUBMITTED", "mode": "full",
            "run_id": self.args.run_id, "sandbox_id": self.args.sandbox_id, "image_id": self.args.image_id,
            "durable_prefix": f"{launcher.ROOT_PREFIX}/submissions/{self.args.run_id}",
            "volume_name": monitor.VOLUME_NAME, "volume_id": monitor.VOLUME_ID, "volume_version": 2,
            "supervisor_sha256": monitor.SUPERVISOR_SHA256, "commit_mode": "sync-v2",
            "input_sha256": launcher.INPUT_SHA256, "input_size": launcher.INPUT_SIZE,
            "final_commit": launcher.FINAL_COMMIT, "controller_sha256": launcher.CONTROLLER_SHA256,
            "modal_sdk_version": launcher.SDK_VERSION, "scientific_acceptance": False,
        }

    def tearDown(self):
        self.temp.cleanup()

    def inspect(self, provider):
        with patch.object(launcher, "modal_sdk", return_value=provider):
            return monitor.inspect_existing(self.args, launcher)

    def test_running_polls_exact_existing_sandbox_and_never_mutates(self):
        provider = FakeProvider(self.launch)
        code, result = self.inspect(provider)
        self.assertEqual((code, result["status"], result["scientific_acceptance"]), (0, "RUNNING", False))
        self.assertEqual(provider.sandbox_calls, [self.args.sandbox_id])
        self.assertEqual(provider.mutations, [])
        self.assertEqual(launcher.load_json_bytes(self.args.recovered_receipt.read_bytes()), self.launch)
        reads = [call[1] for call in provider.calls if isinstance(call, tuple) and call[0] == "read_file"]
        self.assertEqual(reads, [self.launch["durable_prefix"] + "/launch.json"])

    def test_every_frozen_receipt_binding_rejects_before_sandbox_lookup(self):
        for key, value in self.launch.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(self.launch)
                changed[key] = "MISMATCH" if not isinstance(value, bool) else True
                provider = FakeProvider(self.launch)
                provider.files[self.launch["durable_prefix"] + "/launch.json"] = launcher.json_bytes(changed)
                with self.assertRaises(ValueError):
                    self.inspect(provider)
                self.assertEqual(provider.sandbox_calls, [])
                self.assertEqual(provider.mutations, [])
                self.assertFalse(self.args.recovered_receipt.exists())

    def test_missing_oversized_duplicate_and_wrong_provider_fail_closed(self):
        for case in ("missing", "oversized", "duplicate", "wrong_id", "wrong_version"):
            with self.subTest(case=case):
                provider = FakeProvider(self.launch)
                path = self.launch["durable_prefix"] + "/launch.json"
                if case == "missing":
                    provider.files.clear()
                elif case == "oversized":
                    provider.files[path] = b" " * (monitor.MAX_METADATA_BYTES + 1)
                elif case == "duplicate":
                    provider.files[path] = b'{"schema_version":1,"schema_version":1}'
                elif case == "wrong_id":
                    provider.object_id = "vo-wrong"
                else:
                    provider.version = 1
                with self.assertRaises((ValueError, KeyError)):
                    self.inspect(provider)
                self.assertEqual(provider.sandbox_calls, [])
                self.assertEqual(provider.mutations, [])

    def test_exclusive_receipt_never_overwrites_or_polls_after_collision(self):
        self.args.recovered_receipt.write_bytes(b"preserve")
        provider = FakeProvider(self.launch)
        with self.assertRaises(FileExistsError):
            self.inspect(provider)
        self.assertEqual(self.args.recovered_receipt.read_bytes(), b"preserve")
        self.assertEqual(provider.sandbox_calls, [])
        self.assertEqual(provider.mutations, [])

    def test_nonzero_exit_is_failure_and_never_scientific_acceptance(self):
        provider = FakeProvider(self.launch, returncode=9)
        code, result = self.inspect(provider)
        self.assertEqual((code, result["status"], result["returncode"]), (1, "SANDBOX_FAILED", 9))
        self.assertIs(result["scientific_acceptance"], False)
        self.assertEqual(provider.mutations, [])

    def test_verify_true_canary_false_and_only_safe_terminal_metadata(self):
        provider = FakeProvider(self.launch)

        def inspected(args):
            self.assertIs(args.verify, True)
            self.assertIs(args.verify_canary, False)
            self.assertIsNone(args.receipt)
            print(json.dumps({"status": "DURABLE_RESULT_VERIFIED", "sandbox_id": self.args.sandbox_id,
                              "run_id": self.args.run_id, "volume_version": 2, "returncode": 0,
                              "scientific_acceptance": True, "controller_status": "PASSED",
                              "files_verified": 7, "logs": "private-never-print", "archive": "private-never-print"}))
            return 0

        with patch.object(launcher, "cmd_inspect", side_effect=inspected):
            code, result = self.inspect(provider)
        self.assertEqual((code, result["scientific_acceptance"]), (0, True))
        self.assertNotIn("private-never-print", json.dumps(result))
        self.assertEqual(provider.mutations, [])

    def test_placeholders_fail_before_any_provider_access(self):
        self.args.run_id = "PENDING_FULL_RUN"
        provider = FakeProvider(self.launch)
        with self.assertRaises(ValueError):
            self.inspect(provider)
        self.assertEqual(provider.calls, [])
        self.assertEqual(provider.sandbox_calls, [])


if __name__ == "__main__":
    unittest.main()
