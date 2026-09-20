"""Offline integrity tests: no Modal authentication, image builds, or jobs."""
import copy
import hashlib
import io
import tarfile
import unittest

import launch_modal as launcher


class MemoryVolume:
    def __init__(self):
        self.files = {}

    def read_file(self, path):
        data = self.files[path]
        for start in range(0, len(data), 31):
            yield data[start:start + 31]


def reference(path, data):
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def controller(full=False):
    names = ["preflight"] if not full else ["preflight", "smoke_package_validation",
        "model_import", "final_runner", "final_integrated_acceptance", "final_numerical_reducer"]
    return {
        "status": "PASSED" if full else "PREFLIGHT_PASSED", "stage": "complete",
        "scientific_acceptance": full, "ended_utc": "2026-09-20T18:00:00+00:00",
        "orchestrator_sha256": launcher.CONTROLLER_SHA256,
        "stages": [{"name": name, "status": "PASSED", "returncode": 0} for name in names],
        **({"final_code_commit": launcher.FINAL_COMMIT} if full else {}),
    }


def fixture(full=False, alter_controller=None, extra_member=None):
    bindings = {
        "input_sha256": launcher.INPUT_SHA256, "input_size": launcher.INPUT_SIZE,
        "final_commit": launcher.FINAL_COMMIT, "controller_sha256": launcher.CONTROLLER_SHA256,
        "supervisor_sha256": "a" * 64, "image_id": "im-example123", "commit_mode": "sdk",
    }
    prefix = f"{launcher.ROOT_PREFIX}/submissions/test-run"
    launch = {"run_id": "test-run", "durable_prefix": prefix, "mode": "full" if full else "preflight", **bindings}
    records = {"receipts/preflight/attempt-a/execution.json": controller()}
    if full:
        records["receipts/full/attempt-b/execution.json"] = controller(True)
    if alter_controller:
        alter_controller(records)
    payload = {path: launcher.json_bytes(value) for path, value in records.items()}
    payload["supervisor.json"] = launcher.json_bytes({"status": "PASSED" if full else "PREFLIGHT_PASSED"})
    payload["logs/preflight.stdout.log"] = b"Recorded preflight output\n"
    stream = io.BytesIO()
    inventory = []
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path, body in payload.items():
            item = tarfile.TarInfo(path)
            item.size, item.mode = len(body), 0o600
            archive.addfile(item, io.BytesIO(body))
            inventory.append({**reference(path, body), "mode": 0o600})
        if extra_member:
            item = tarfile.TarInfo(extra_member)
            item.type = tarfile.SYMTYPE
            item.linkname = "/etc/passwd"
            archive.addfile(item)
    archive_bytes = stream.getvalue()
    archive_ref = reference("final/evidence.tar.gz", archive_bytes)
    final_manifest = {"schema_version": 1, "run_id": "test-run", **bindings,
        "status": "PASSED" if full else "PREFLIGHT_PASSED", "scientific_acceptance": full,
        "archive": archive_ref, "files": inventory}
    final_body = launcher.json_bytes(final_manifest)
    generation = {"schema_version": 1, "run_id": "test-run", **bindings,
        "files": [reference(path, body) for path, body in payload.items()]}
    generation_body = launcher.json_bytes(generation)
    volume = MemoryVolume()
    volume.files[f"{prefix}/final/evidence.tar.gz"] = archive_bytes
    volume.files[f"{prefix}/final/manifest.json"] = final_body
    generation_dir = f"{prefix}/generations/000001"
    volume.files[f"{generation_dir}/manifest.json"] = generation_body
    for path, body in payload.items():
        volume.files[f"{generation_dir}/{path}"] = body
    completion = {"schema_version": 1, "run_id": "test-run", **bindings,
        "status": "PASSED" if full else "PREFLIGHT_PASSED", "scientific_acceptance": full,
        "archive": archive_ref,
        "final_manifest": reference("final/manifest.json", final_body),
        "terminal_generation": reference("generations/000001/manifest.json", generation_body)}
    return volume, launch, completion


class DurableVerifierTests(unittest.TestCase):
    def test_preflight_export_passes_without_scientific_claim(self):
        volume, launch, completion = fixture()
        result = launcher.verify_completion(volume, launch, completion)
        self.assertEqual(result["controller_status"], "PREFLIGHT_PASSED")
        self.assertIs(result["scientific_acceptance"], False)

    def test_realistic_full_export_has_two_controller_receipts(self):
        volume, launch, completion = fixture(full=True)
        result = launcher.verify_completion(volume, launch, completion)
        self.assertEqual(result["controller_status"], "PASSED")
        self.assertIs(result["scientific_acceptance"], True)

    def test_archive_damage_rejected(self):
        volume, launch, completion = fixture()
        path = f"{launch['durable_prefix']}/final/evidence.tar.gz"
        volume.files[path] += b"damage"
        with self.assertRaisesRegex(ValueError, "archive hash"):
            launcher.verify_completion(volume, launch, completion)

    def test_rehashed_false_controller_acceptance_rejected(self):
        def change(records):
            records["receipts/full/attempt-b/execution.json"]["stages"][-1]["returncode"] = 1
        volume, launch, completion = fixture(full=True, alter_controller=change)
        with self.assertRaisesRegex(ValueError, "acceptance evidence"):
            launcher.verify_completion(volume, launch, completion)

    def test_rehashed_missing_preflight_rejected(self):
        volume, launch, completion = fixture(full=True, alter_controller=lambda d: d.pop("receipts/preflight/attempt-a/execution.json"))
        with self.assertRaisesRegex(ValueError, "inventory"):
            launcher.verify_completion(volume, launch, completion)

    def test_rehashed_wrong_controller_identity_rejected(self):
        def change(records):
            records["receipts/preflight/attempt-a/execution.json"]["orchestrator_sha256"] = "0" * 64
        volume, launch, completion = fixture(alter_controller=change)
        with self.assertRaisesRegex(ValueError, "acceptance evidence"):
            launcher.verify_completion(volume, launch, completion)

    def test_unlisted_symlink_rejected_even_with_correct_archive_hash(self):
        volume, launch, completion = fixture(extra_member="output/unsafe-link")
        with self.assertRaisesRegex(ValueError, "Unsafe or unexpected"):
            launcher.verify_completion(volume, launch, completion)

    def test_terminal_payload_tampering_rejected(self):
        volume, launch, completion = fixture()
        path = f"{launch['durable_prefix']}/generations/000001/logs/preflight.stdout.log"
        volume.files[path] = b"different"
        with self.assertRaisesRegex(ValueError, "payload mismatch"):
            launcher.verify_completion(volume, launch, completion)

    def test_binding_substitution_rejected(self):
        volume, launch, completion = fixture()
        completion["image_id"] = "im-different"
        with self.assertRaisesRegex(ValueError, "identity bindings"):
            launcher.verify_completion(volume, launch, completion)

    def test_false_canary_or_different_image_rejected(self):
        _, launch, _ = fixture()
        bindings = {key: launch[key] for key in launcher.BINDING_KEYS}
        proof = {"schema_version": 1, "status": "CANARY_VERIFIED", "fresh_reader_verified": True,
            **bindings, "canary_run_id": "canary", "canary_sandbox_id": "sb-canary",
            "canary_receipt_sha256": "b" * 64}
        launcher.require_canary(proof, bindings)
        for key, value in [("fresh_reader_verified", False), ("image_id", "im-changed")]:
            changed = copy.deepcopy(proof)
            changed[key] = value
            with self.assertRaises(ValueError):
                launcher.require_canary(changed, bindings)

    def test_path_traversal_rejected(self):
        for path in ["../outside", "/absolute", "safe/../outside", "a//b", "a\\b"]:
            with self.assertRaises(ValueError):
                launcher.safe_relative(path)


if __name__ == "__main__":
    unittest.main()
