"""Offline tests for synthetic-only persistence evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

import launch_modal as launcher
import storage_canary
from test_launch_modal import MemoryVolume, reference


def synthetic_fixture():
    run_id = "storage-test"
    prefix = f"{launcher.ROOT_PREFIX}/storage-canaries/{run_id}"
    posix = {name: True for name in ("flock", "hardlinks", "rename", "file_fsync", "directory_fsync", "executable_modes")}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        record = storage_canary.synthetic_files(
            root, run_id, "im-synthetic", posix,
            {"python": "3.11.12", "packages": storage_canary.PACKAGES}, "sdk",
        )
        volume = MemoryVolume()
        for name in ("synthetic.bin", "payload.json"):
            volume.files[f"{prefix}/{name}"] = (root / name).read_bytes()
    launch = {"mode": "storage-canary", "scope": "synthetic-only", "durable_prefix": prefix,
        "run_id": run_id, "image_id": "im-synthetic", "sandbox_id": "sb-synthetic",
        "commit_mode": "sdk", "storage_canary_sha256": record["storage_canary_sha256"],
        "supervisor_sha256": record["supervisor_sha256"]}
    completion = {**record, "status": "STORAGE_CANARY_PASSED", "explicit_payload_commit_succeeded": True,
        "completed_utc": "2026-09-20T20:00:00+00:00",
        "payload_receipt": reference("payload.json", volume.files[f"{prefix}/payload.json"])}
    return volume, launch, completion


class SyntheticCanaryTests(unittest.TestCase):
    def test_small_synthetic_payload_round_trip_preserves_no_science_scope(self):
        volume, launch, completion = synthetic_fixture()
        result = launcher.verify_storage_canary(volume, launch, completion)
        self.assertEqual(result["status"], "STORAGE_CANARY_VERIFIED")
        self.assertEqual(result["synthetic_payload_bytes"], 4096)
        self.assertFalse(result["scientific_acceptance"])
        self.assertFalse(result["research_data_accessed"])
        self.assertFalse(result["scientific_execution_performed"])
        self.assertEqual(len(volume.files), 2)

    def test_durable_payload_tampering_is_rejected(self):
        volume, launch, completion = synthetic_fixture()
        volume.files[f"{launch['durable_prefix']}/synthetic.bin"] = b"x" * 4096
        with self.assertRaisesRegex(ValueError, "payload mismatch"):
            launcher.verify_storage_canary(volume, launch, completion)

    def test_wrong_runtime_missing_commit_or_science_claim_is_rejected(self):
        volume, launch, completion = synthetic_fixture()
        for change in (
            {"environment": {"python": "3.11.16", "packages": storage_canary.PACKAGES}},
            {"explicit_payload_commit_succeeded": False},
            {"scientific_acceptance": True},
        ):
            changed = copy.deepcopy(completion)
            changed.update(change)
            with self.assertRaises(ValueError):
                launcher.verify_storage_canary(volume, launch, changed)

    def test_synthetic_canary_cannot_admit_scientific_run(self):
        volume, launch, completion = synthetic_fixture()
        result = launcher.verify_storage_canary(volume, launch, completion)
        scientific_bindings = {"input_sha256": launcher.INPUT_SHA256, "input_size": launcher.INPUT_SIZE,
            "final_commit": launcher.FINAL_COMMIT, "controller_sha256": launcher.CONTROLLER_SHA256,
            "supervisor_sha256": launch["supervisor_sha256"], "image_id": launch["image_id"], "commit_mode": "sdk"}
        with self.assertRaises(ValueError):
            launcher.require_canary(result, scientific_bindings)

    def test_no_overwrite_of_synthetic_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "synthetic.bin").write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                storage_canary.synthetic_files(root, "run", "im-test", {}, {}, "sdk")
            self.assertEqual((root / "synthetic.bin").read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
