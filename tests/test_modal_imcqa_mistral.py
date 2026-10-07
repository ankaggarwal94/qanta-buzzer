"""Offline falsification tests for Mistral approval, cache and allocation bounds."""
from __future__ import annotations

from contextlib import nullcontext
import ast
import copy
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import modal_imcqa_mistral as runner
from scripts import imcqa_mistral_budget as budget

ROOT = Path(__file__).resolve().parents[1]


class FakeVolume:
    def __init__(self): self.files = {}
    def batch_upload(self, *, force):
        if force: raise AssertionError("no replacement uploads")
        volume = self
        class Upload:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def put_file(self, stream, name):
                name = name.lstrip("/")
                if name in volume.files: raise FileExistsError(name)
                volume.files[name] = stream.read()
        return Upload()
    def read_file(self, name): return iter([self.files[name.lstrip("/")]])
    def iterdir(self, prefix, recursive=True):
        return [SimpleNamespace(path=n) for n in self.files if n.startswith(prefix.lstrip("/") + "/")]


class FakeProvider:
    def __init__(self):
        self.volumes, self.calls, self.resources, self.mounts = {}, [], [], []
        self.app_count = 0
        def create(name, *, version, allow_existing):
            if version != 2 or allow_existing: raise AssertionError("non-atomic claim")
            if name in self.volumes: raise FileExistsError(name)
            self.volumes[name] = FakeVolume()
        def from_name(name, *, create_if_missing):
            if create_if_missing: raise AssertionError("implicit volume creation")
            return self.volumes[name]
        self.Volume = SimpleNamespace(objects=SimpleNamespace(create=create, delete=self.volumes.pop), from_name=from_name)
        provider = self
        class Image:
            @classmethod
            def from_id(cls, value):
                if value != runner.BASE_IMAGE_ID: raise AssertionError("unapproved base image")
                return cls()
            def add_local_file(self, local, *, remote_path, copy):
                if copy: raise AssertionError("unbounded image build")
                provider.mounts.append((local, remote_path))
                return self
        self.Image = Image
        class App:
            def __init__(self, name, **kwargs): provider.app_count += 1
            def function(self, **resources):
                provider.resources.append(resources)
                return lambda fn: FakeScorer(provider.calls)
            def run(self): return nullcontext()
        self.App = App
    def enable_output(self): return nullcontext()


class FakeCall:
    object_id = "fc-fake"
    def __init__(self, error=None): self.error, self.cancelled, self.timeout = error, [], None
    def get(self, *, timeout):
        self.timeout = timeout
        if self.error: raise self.error
        return {"status": "complete"}
    def cancel(self, *, terminate_containers): self.cancelled.append(terminate_containers)


class FakeScorer:
    def __init__(self, calls=None, error=None): self.calls, self.error = calls if calls is not None else [], error
    def spawn(self, control):
        call = FakeCall(self.error)
        self.calls.append((control, call))
        return call


class ModalMistralTests(unittest.TestCase):
    def control(self, run="imcqa-mistral-test"):
        return runner.cache_control(ROOT, run, "a" * 40, datetime.now(timezone.utc).isoformat())

    def test_exact_cumulative_reservation_and_no_builds(self):
        result = self.control()["budget"]
        self.assertEqual(result["reserved_estimate_usd"], "7.63713288")
        self.assertEqual(result["unallocated_headroom_usd"], "0.36286712")
        self.assertEqual(result["image_builds_allowed"], 0)
        self.assertEqual(result["automatic_function_retries"], 0)
        self.assertEqual(result["gpu_resource_maximums"]["cpu"], [2, 2])
        self.assertEqual(result["gpu_resource_maximums"]["memory_mib"], [32768, 32768])

    def test_changed_budget_or_stale_rate_fails(self):
        result = copy.deepcopy(self.control()["budget"])
        result["ceiling_usd"] = "80"
        with self.assertRaises(ValueError): budget.validate_budget(result)
        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        with self.assertRaises(ValueError): budget.budget_plan(old)

    def test_source_closure_contains_scorer_and_no_evaluators(self):
        from scripts import imcqa_mistral_scoring as scoring
        self.assertTrue(set(scoring.SOURCE_FILES).issubset(runner.SOURCES))
        self.assertIn("configs/imcqa_mistral_replication.json", runner.SOURCES)
        self.assertFalse(any("evaluator" in name or "dataset" in name for name in runner.SOURCES))
        self.assertEqual(set(scoring.MODEL_FILES), set(runner.MODEL_FILES))

    def test_all_declared_local_imports_have_frozen_source_hashes(self):
        closure = set(runner.SOURCES) | set(runner.CONTROL_SOURCES)
        for filename in sorted(closure):
            if not filename.endswith(".py"):
                continue
            tree = ast.parse((ROOT / filename).read_text())
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module == "scripts":
                    modules = ["scripts." + alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scripts."):
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names if alias.name.startswith("scripts.")]
                for module in modules:
                    self.assertIn(module.replace(".", "/") + ".py", closure, (filename, node.lineno))

    def test_preflight_has_no_provider_contact(self):
        control = self.control()
        with patch.object(runner, "connect", side_effect=AssertionError("contacted provider")):
            result = runner.execute(ROOT, control, Path("unused"), dry_run=True)
        self.assertFalse(result["provider_contacted"])
        self.assertFalse(result["gpu_executed"])

    def test_wrong_cache_and_extra_weights_fail(self):
        control = self.control()
        changed = copy.deepcopy(control)
        changed["cache_volume"] = "old-qwen-cache"
        with self.assertRaises(ValueError): runner.validate_control(changed)
        sidecar = {"model": runner.MODEL, "revision": runner.REVISION, "model_files_sha256": dict(runner.MODEL_HASHES)}
        receipt = {**sidecar, "status": "complete", "model_receipts": {runner.MODEL_TAG: sidecar},
            "schema_version": "imcqa-mistral-cache-v1", "chat_template_sha256": runner.CHAT_TEMPLATE_SHA256,
            "base_image_id": runner.BASE_IMAGE_ID, "dependency_versions": runner.ADDED_VERSIONS,
            "dependency_files_sha256": {"file": "a" * 64}}
        runner.verify_prepare(runner.canonical(receipt))
        receipt["model_files_sha256"]["consolidated.safetensors"] = "b" * 64
        with self.assertRaises(ValueError): runner.verify_prepare(runner.canonical(receipt))

    def test_call_cancelled_on_success_and_timeout(self):
        for error in (None, TimeoutError("provider timeout")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                scorer = FakeScorer(error=error)
                if error:
                    with self.assertRaises(TimeoutError): runner.await_one(scorer, self.control(), Path(tmp), 1860)
                else:
                    runner.await_one(scorer, self.control(), Path(tmp), 1860)
                sent, call = scorer.calls[0]
                self.assertEqual(call.cancelled, [True])
                self.assertGreater(call.timeout, 0)
                self.assertLessEqual(call.timeout, 1950)
                self.assertIn("absolute_deadline_unix", sent)

    def test_write_failure_after_spawn_still_cancels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "calls.json").write_text("existing")
            scorer = FakeScorer()
            with self.assertRaises(FileExistsError): runner.await_one(scorer, self.control(), out, 1860)
            self.assertEqual(scorer.calls[0][1].cancelled, [True])
            self.assertEqual(len(scorer.calls), 1)

    def test_duplicate_approval_and_alternate_run_cannot_spawn(self):
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "connect", return_value=(provider, {})), \
                patch.object(runner, "verify_sources"):
            runner.execute(ROOT, self.control(), Path(tmp) / "first")
            for index, control in enumerate((self.control(), self.control("imcqa-mistral-alternate"))):
                with self.assertRaises(FileExistsError): runner.execute(ROOT, control, Path(tmp) / str(index))
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.resources[0]["memory"], (8192, 8192))
        self.assertNotIn("gpu", provider.resources[0])

    def test_collection_error_never_relaunches(self):
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "connect", return_value=(provider, {})), \
                patch.object(runner, "verify_sources"), patch.object(runner, "collect", side_effect=IOError("download failed")):
            with self.assertRaises(IOError): runner.execute(ROOT, self.control(), Path(tmp) / "run")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0][1].cancelled, [True])

    def test_duplicate_gpu_stage_is_blocked_before_another_spawn(self):
        provider, cache = FakeProvider(), self.control()
        preparation_raw = b"synthetic prepare fixture"
        provider.volumes[runner.APPROVAL_ID] = FakeVolume()
        provider.volumes[runner.APPROVAL_ID].files["control.json"] = runner.canonical(cache)
        provider.volumes[cache["cache_volume"]] = FakeVolume()
        provider.volumes[cache["cache_volume"]].files["output/prepare_receipt.json"] = preparation_raw
        control = {**cache, "mode": "score-stage", "stage": "development",
            "volume": cache["run_id"] + "-development", "expected_contexts": 5600,
            "prepare_receipt_sha256": runner.sha(preparation_raw),
            "public_input_sha256": "b" * 64, "stage_manifest_sha256": "c" * 64}
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "connect", return_value=(provider, {})), \
                patch.object(runner, "verify_sources"):
            runner.execute(ROOT, control, Path(tmp) / "first", public_raw=b"{}", preparation_raw=preparation_raw)
            with self.assertRaises(FileExistsError):
                runner.execute(ROOT, control, Path(tmp) / "second", public_raw=b"{}", preparation_raw=preparation_raw)
        self.assertEqual(len(provider.calls), 1)
        settings = provider.resources[0]
        self.assertEqual(settings["gpu"], "L40S")
        self.assertEqual(settings["cpu"], (2, 2))
        self.assertEqual(settings["memory"], (32768, 32768))
        self.assertEqual(settings["max_containers"], 1)
        self.assertEqual(settings["timeout"], 1860)
        self.assertEqual(settings["retries"], 0)

    def test_status_never_constructs_app_or_claim(self):
        provider, control = FakeProvider(), self.control()
        volume = FakeVolume()
        volume.files = {"control.json": runner.canonical(control), "output/state.json": b"{}"}
        provider.volumes[control["volume"]] = volume
        with patch.object(runner, "connect", return_value=(provider, {})):
            result = runner.read_only(control)
        self.assertFalse(result["gpu_executed"])
        self.assertEqual(provider.app_count, 0)
        self.assertEqual(list(provider.volumes), [control["volume"]])

    def test_collection_rejects_traversal(self):
        volume = FakeVolume()
        volume.files["output/../escape.json"] = b"{}"
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            runner.collect(volume, Path(tmp), self.control())

    def test_only_owned_new_cache_is_deleted(self):
        provider, control = FakeProvider(), self.control()
        for name in (runner.APPROVAL_ID, control["cache_volume"]):
            provider.volumes[name] = FakeVolume()
            provider.volumes[name].files["control.json"] = runner.canonical(control)
        provider.volumes["old-qwen-cache"] = FakeVolume()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "connect", return_value=(provider, {})):
            runner.cleanup_cache(control, Path(tmp) / "cleanup.json")
        self.assertIn("old-qwen-cache", provider.volumes)
        self.assertIn(runner.APPROVAL_ID, provider.volumes)
        self.assertNotIn(control["cache_volume"], provider.volumes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
