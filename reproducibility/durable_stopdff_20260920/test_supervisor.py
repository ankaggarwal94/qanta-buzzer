"""Focused safety and acceptance tests; never import Modal or submit work."""
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import supervisor as s


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, entries):
        archive = self.root / "input.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, kind, mode in entries:
                item = tarfile.TarInfo(name)
                item.type, item.mode = kind, mode
                if kind == tarfile.REGTYPE:
                    item.size = len(b"payload")
                    tar.addfile(item, io.BytesIO(b"payload"))
                else:
                    item.linkname = "/outside"
                    tar.addfile(item)
        return archive

    def test_extract_preserves_executable_modes(self):
        archive = self.archive([("input", tarfile.DIRTYPE, 0o750),
                                ("input/run.sh", tarfile.REGTYPE, 0o751)])
        dest = self.root / "extracted"
        dest.mkdir()
        self.assertEqual(s.safe_extract(archive, dest), 1)
        self.assertEqual((dest / "input/run.sh").read_bytes(), b"payload")
        self.assertEqual(stat.S_IMODE((dest / "input/run.sh").stat().st_mode), 0o751)
        self.assertEqual(stat.S_IMODE((dest / "input").stat().st_mode), 0o750)

    def test_extract_rejects_traversal_links_devices_and_privileged_modes(self):
        cases = [("../outside", tarfile.REGTYPE, 0o644), ("/outside", tarfile.REGTYPE, 0o644),
                 ("input/../outside", tarfile.REGTYPE, 0o644), ("input/a", tarfile.SYMTYPE, 0o777),
                 ("input/a", tarfile.LNKTYPE, 0o644), ("input/a", tarfile.FIFOTYPE, 0o644),
                 ("input/a", tarfile.CHRTYPE, 0o644), ("input/a", tarfile.REGTYPE, 0o4755)]
        for number, case in enumerate(cases):
            with self.subTest(case=case):
                archive = self.archive([case])
                dest = self.root / str(number)
                dest.mkdir()
                with self.assertRaises(ValueError):
                    s.safe_extract(archive, dest)
                self.assertFalse((self.root / "outside").exists())

    def test_duplicate_archive_members_fail_closed(self):
        archive = self.archive([("input/a", tarfile.REGTYPE, 0o644)] * 2)
        dest = self.root / "extracted"
        dest.mkdir()
        with self.assertRaises(ValueError):
            s.safe_extract(archive, dest)

    def exporter(self, commit):
        job = self.root / "job"
        job.mkdir()
        for name in ("receipts", "logs", "work", "output"):
            (job / name).mkdir()
        s.write_json(job / "supervisor.json", {"status": "PASSED"})
        (job / "output/result.json").write_text('{"result":1}\n')
        (job / "logs/controller.log").write_text("evidence\n")
        durable = self.root / "durable"
        durable.mkdir()
        return s.Exporter(job, durable, commit, {"input_sha256": s.INPUT_SHA256}, "test-run")

    def test_failed_generation_or_archive_commit_prevents_completion(self):
        for fail_at in (1, 2, 3):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as temp:
                original = self.root
                self.root = Path(temp)
                calls = []
                def commit():
                    calls.append(None)
                    if len(calls) == fail_at:
                        raise RuntimeError("simulated commit failure")
                exporter = self.exporter(commit)
                with self.assertRaises(RuntimeError):
                    exporter.finalize({"status": "PASSED", "scientific_acceptance": True})
                self.assertFalse((exporter.durable / "completion.json").exists())
                self.assertTrue((exporter.job / "output/result.json").exists())
                self.root = original

    def test_completion_only_follows_committed_archive_and_all_hashes_match(self):
        events = []
        def commit():
            marker = exporter.durable / "completion.json"
            archive = exporter.durable / "final/evidence.tar.gz"
            events.append((marker.exists(), archive.exists()))
        exporter = self.exporter(commit)
        result = exporter.finalize({"status": "PASSED", "scientific_acceptance": True})
        self.assertEqual(events, [(False, False), (False, True), (True, True)])
        self.assertEqual(s.digest(exporter.durable / result["archive"]["path"]), result["archive"]["sha256"])
        manifest = s.read_json(exporter.durable / result["final_manifest"]["path"])
        expected = {item["path"]: item for item in manifest["files"]}
        with tarfile.open(exporter.durable / result["archive"]["path"]) as archive:
            self.assertEqual(set(archive.getnames()), set(expected))
            for member in archive:
                self.assertTrue(member.isreg())
                self.assertEqual(member.mode, expected[member.name]["mode"])
                self.assertEqual(s.hashlib.sha256(archive.extractfile(member).read()).hexdigest(),
                                 expected[member.name]["sha256"])

    def test_receipt_temp_files_are_not_exported(self):
        exporter = self.exporter(lambda: None)
        (exporter.job / "receipts/.receipt-unfinished").write_text('{"incomplete":')
        manifest = exporter.generation()
        payload = s.read_json(exporter.durable / manifest["path"])
        self.assertFalse(any(".receipt-" in item["path"] for item in payload["files"]))

    def receipt(self, preflight):
        root = self.root / "receipts"
        root.mkdir(exist_ok=True)
        invocation = root / "invocation"
        invocation.mkdir(exist_ok=True)
        names = ["preflight"] if preflight else ["preflight", "smoke_package_validation", "model_import",
                    "final_runner", "final_integrated_acceptance", "final_numerical_reducer"]
        data = {"status": "PREFLIGHT_PASSED" if preflight else "PASSED", "stage": "complete",
                "scientific_acceptance": not preflight, "orchestrator_sha256": s.CONTROLLER_SHA256,
                "ended_utc": "2026-09-20T00:00:00+00:00", "final_code_commit": s.FINAL_COMMIT,
                "stages": [{"name": name, "status": "PASSED"} for name in names]}
        path = invocation / "execution.json"
        s.write_json(path, data)
        return root, path, data

    def test_preflight_is_never_scientific_acceptance(self):
        root, path, data = self.receipt(True)
        self.assertEqual(s.require_gate(root, preflight=True)["status"], "PREFLIGHT_PASSED")
        with self.assertRaises(ValueError):
            s.require_gate(root, preflight=False)
        data["scientific_acceptance"] = True
        s.write_json(path, data)
        with self.assertRaises(ValueError):
            s.require_gate(root, preflight=True)

    def test_full_requires_all_scientific_stages_and_frozen_controller(self):
        root, path, good = self.receipt(False)
        self.assertEqual(s.require_gate(root, preflight=False)["status"], "PASSED")
        for change in ({"status": "RUNNING"}, {"scientific_acceptance": False},
                       {"orchestrator_sha256": "wrong"}, {"final_code_commit": "wrong"},
                       {"stages": good["stages"][:-1]}):
            with self.subTest(change=change):
                s.write_json(path, {**good, **change})
                with self.assertRaises(ValueError):
                    s.require_gate(root, preflight=False)

    def test_scientific_children_receive_no_auth_environment(self):
        with patch.dict(os.environ, {"MODAL_TOKEN_ID": "do-not-propagate", "MODAL_TOKEN_SECRET": "private",
                                    "OPENAI_API_KEY": "private", "AWS_SECRET_ACCESS_KEY": "private"}):
            env = s.scientific_environment(self.root)
        self.assertFalse(any(key in env for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY")))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_full_admission_requires_canary_readback_bound_to_current_image(self):
        args = SimpleNamespace(run_id="full-test", image_id="im-test", commit_mode="sdk",
                               preflight_only=False, canary_receipt=self.root / "canary.json",
                               admission_marker=self.root / "admission.json")
        supervisor = s.Supervisor(args)
        s.write_json(args.admission_marker, {"schema_version": 1, "run_id": args.run_id,
                     "mode": "full", **supervisor.bindings})
        canary = {"schema_version": 1, "status": "CANARY_VERIFIED", "fresh_reader_verified": True,
                  "canary_run_id": "canary-test", "canary_sandbox_id": "sb-test",
                  "canary_receipt_sha256": "a" * 64, **supervisor.bindings}
        base = Path(__file__).resolve().parent
        candidates = (base / "run_durable_rerun.py", base / "controls/run_durable_rerun.py")
        controls = next((path for path in candidates if path.is_file()), None)
        self.assertIsNotNone(controls, "Frozen controller fixture is missing")
        self.assertEqual(s.digest(controls), s.CONTROLLER_SHA256)
        with patch.object(s, "CONTROLLER", controls):
            s.write_json(args.canary_receipt, canary)
            supervisor.validate_admission()
            for change in ({"fresh_reader_verified": False}, {"image_id": "im-stale"},
                           {"input_sha256": "b" * 64}, {"status": "PREFLIGHT_PASSED"}):
                with self.subTest(change=change):
                    s.write_json(args.canary_receipt, {**canary, **change})
                    with self.assertRaises(ValueError):
                        supervisor.validate_admission()

    def test_local_posix_primitives(self):
        self.assertTrue(all(s.check_posix(self.root).values()))

    def test_detached_child_is_reaped_after_immediate_controller_crash(self):
        s.enable_subreaper()
        pid_file = self.root / "detached.pid"
        script = ("import os,subprocess,sys; "
                  "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
                  "open(sys.argv[1],'w').write(str(child.pid)); os._exit(17)")
        child = subprocess.Popen([sys.executable, "-c", script, str(pid_file)],
                                 env=s.scientific_environment(self.root), start_new_session=True)
        child.wait(timeout=5)
        detached = int(pid_file.read_text())
        try:
            s.stop_controller(child, owner=self.root)
            with self.assertRaises(ProcessLookupError):
                os.kill(detached, 0)
        finally:
            try:
                os.kill(detached, 9)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    unittest.main()
