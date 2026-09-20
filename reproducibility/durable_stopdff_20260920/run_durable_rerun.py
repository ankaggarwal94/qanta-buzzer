#!/usr/bin/env python3
"""Provider-independent, evidence-recording execution of the remaining rerun."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid

sys.dont_write_bytecode = True
FINAL_COMMIT = "2ed304f6598b94c0c5dc15f77fb8b391ae2f03a3"
FINAL_SOURCE = "bf426eea340ea90d3bc62f8170f2da8155da6e2bca57e27798b3a9f3102d150e"
SMOKE_COMMIT = "abca5c6e81463bc7fbf22b2afe2a199d566dbd7b"
SMOKE_SOURCE = "4f8c133241fb01e16bccc1c9572bc72a9e2dece115638a4eb7ab798b91ba3f90"
SMOKE_ADAPTER = "b56bc3b475367e522e35448d904525b84577503317a1e2a6bf8498c402c0f930"
SMOKE_FVI = "42ee78e10d61005c4b885b131f654b0a696c5fb7447b28d588f3ebfe92438818"
SMOKE_RUN = "3055aa40e7dcf7ad5d91a7ca40cbf5380dfa9ecbdc594f9a0c1d73dbae69c8d0"
MODEL = "33b48dc6daf60b6e0a2190964bf3faafc249732269fed1f717f197940cd6f893"
RAW = "18cbc0671b82610e248776b11833dc62901330cdc53f35ea7b95d69e12db221a"
ENVIRONMENT = "551449e48416c6f927e0034b26226da424616c5378b10f1398be310830ba445e"
PACKAGES = {
    "huggingface_hub": "1.29.0", "matplotlib": "3.11.1", "numpy": "2.4.6",
    "pandas": "3.0.5", "scikit-learn": "1.9.0", "scipy": "1.17.1",
    "sentence-transformers": "6.0.0", "torch": "2.13.0", "transformers": "5.16.1",
}
ENV_OVERRIDES = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
}
STOP = threading.Event()
STOP_SIGNAL = None
TERM_TIMEOUT_SECONDS = 20.0
KILL_TIMEOUT_SECONDS = 5.0


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    """Replace only this invocation's external receipt, durably."""
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def canonical_path(value):
    path = Path(os.path.abspath(value))
    if path.resolve(strict=False) != path:
        raise ValueError(f"Path must not traverse symlinks: {path}")
    return path


def regular_tree(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Expected a regular directory: {root}")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"Link or special file is unsupported: {path}")


def cancelled():
    if STOP.is_set():
        raise InterruptedError(f"Received signal {STOP_SIGNAL}; no completion is claimed")


def on_signal(number, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = number
    STOP.set()


def stop_child(child):
    """Reap a detached child, escalating to KILL with bounded waits."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=TERM_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=KILL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Child group {child.pid} did not exit after TERM/KILL; cleanup is unconfirmed") from exc


class Journal:
    def __init__(self, root, paths, interval):
        invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
        self.directory = root / invocation
        self.directory.mkdir(parents=True, exist_ok=False)
        self.mutex = threading.RLock()
        self.closed = threading.Event()
        self.state = {
            "schema_version": 1, "invocation": invocation, "status": "RUNNING",
            "stage": "initializing", "started_utc": now(), "pid": os.getpid(),
            "hostname": socket.gethostname(), "paths": {k: str(v) for k,v in paths.items()},
            "command": sys.argv, "orchestrator_sha256": sha(Path(__file__)), "stages": [],
            "scope": "Preflight is not scientific acceptance; terminal PASSED requires all recorded scientific gates.",
        }
        self.interval = interval
        self.save()
        self.thread = threading.Thread(target=self._heartbeats, daemon=True)
        self.thread.start()

    def save(self):
        with self.mutex:
            self.state["updated_utc"] = now()
            atomic_json(self.directory / "execution.json", self.state)

    def update(self, **values):
        with self.mutex:
            self.state.update(values)
            self.save()

    def heartbeat(self):
        with self.mutex:
            atomic_json(self.directory / "heartbeat.json", {
                "schema_version": 1, "updated_utc": now(), "pid": os.getpid(),
                "hostname": socket.gethostname(), "status": self.state["status"],
                "stage": self.state["stage"], "child_pid": self.state.get("child_pid"),
                "note": "A stale heartbeat or RUNNING receipt never establishes success.",
            })

    def _heartbeats(self):
        while not self.closed.is_set():
            try:
                self.heartbeat()
            except Exception as exc:
                print(f"Heartbeat write failed: {exc}", file=sys.stderr, flush=True)
            self.closed.wait(self.interval)

    @contextmanager
    def stage(self, name):
        cancelled()
        record = {"name": name, "started_utc": now(), "status": "RUNNING"}
        with self.mutex:
            self.state["stages"].append(record)
            self.state["stage"] = name
            self.save()
        print(f"{now()} {name}", flush=True)
        started = time.monotonic()
        try:
            yield record
            cancelled()
        except BaseException as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            record["status"] = "PASSED"
        finally:
            record.update(ended_utc=now(), elapsed_seconds=time.monotonic()-started)
            self.save()

    def command(self, name, command, cwd):
        with self.stage(name) as record:
            stdout = self.directory / f"{name}.stdout.log"
            stderr = self.directory / f"{name}.stderr.log"
            record.update(command=[str(x) for x in command], cwd=str(cwd),
                          stdout=str(stdout), stderr=str(stderr), environment_overrides=ENV_OVERRIDES)
            self.save()
            with stdout.open("xb") as out, stderr.open("xb") as err:
                child = subprocess.Popen(record["command"], cwd=cwd,
                    env={**os.environ, **ENV_OVERRIDES}, stdout=out, stderr=err, start_new_session=True)
                try:
                    # Own the child before any fallible receipt write. A disk
                    # error here must still terminate and reap its process group.
                    self.update(child_pid=child.pid)
                    while child.poll() is None:
                        if STOP.wait(1):
                            break
                finally:
                    try:
                        stop_child(child)
                    finally:
                        record["returncode"] = child.returncode
                        self.update(child_pid=child.pid if child.poll() is None else None)
            record.update(stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
            if child.returncode:
                raise RuntimeError(f"{name} exited {child.returncode}; inspect {stderr} and {stdout}")
            return stdout

    def close(self):
        self.closed.set()
        self.thread.join(timeout=5)
        self.heartbeat()


def git(code, *arguments):
    return subprocess.check_output(["git", "-C", str(code), *arguments], text=True).strip()


def check_code(code):
    if git(code, "rev-parse", "HEAD") != FINAL_COMMIT:
        raise ValueError(f"--code-dir must be at exact commit {FINAL_COMMIT}")
    if git(code, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Final code checkout must be clean, including untracked files")


def preflight(paths):
    code, inputs = paths["code"], paths["input"]
    check_code(code)
    actual = {name: importlib.metadata.version(name) for name in PACKAGES}
    if platform.python_version() != "3.11.12" or actual != PACKAGES:
        raise ValueError(f"Runtime mismatch: Python={platform.python_version()}, packages={actual}")
    # Extra runtime dependency used by the validator, outside the nine-package
    # scientific identity. Import here so failure precedes any expensive work.
    import jsonschema
    sys.path.insert(0, str(code))
    from scripts.inspect_stopdff_source_package import inspect_package
    from scripts.stopdff_v5 import checker
    from scripts.stopdff_v5.content_manifest import validate_bound_content_manifest
    from scripts.stopdff_v5.identity import compute_id, loads_no_duplicate_keys
    from scripts.stopdff_v5.manifests import environment_contract_identity
    env_id = compute_id(environment_contract_identity(python_version="3.11.12", package_versions=actual))
    if env_id != ENVIRONMENT:
        raise ValueError("Actual environment identity differs from the frozen contract")
    package, adapter = inputs / "smoke_package", inputs / "adapter_bundle"
    regular_tree(package)
    regular_tree(adapter)
    inspection = inspect_package(package, expected_source_id=SMOKE_SOURCE)
    if inspection["mode_mismatches"] or inspection["git_sha"] != SMOKE_COMMIT or inspection["source_files_verified"] != 640:
        raise ValueError(f"Smoke source inventory or executable permissions mismatch: {inspection}")
    evidence = package / "evidence"
    for name, kind, expected, subdir, name_key in (
        ("raw_input_manifest.json", "raw_input_bundle", RAW, "raw_inputs/raw", "role"),
        ("model_snapshot_manifest.json", "model_snapshot", MODEL, "model_snapshot/snapshot", "path"),
    ):
        validate_bound_content_manifest(evidence, manifest_name=name, expected_id=expected,
            expected_kind=kind, file_key="files", name_key=name_key, content_subdir=subdir,
            require_semantic_pass=kind == "raw_input_bundle")
    adapter_result = checker.validate_adapter(adapter)
    if not adapter_result.passed or adapter_result.recomputed.get("adapter_bundle_id") != SMOKE_ADAPTER:
        raise ValueError(f"Smoke adapter is invalid or wrong: {adapter_result.errors}")
    spec = loads_no_duplicate_keys((package / "run_spec.json").read_text())
    fvi = loads_no_duplicate_keys((evidence / "fvi_study.json").read_text())
    if compute_id(spec["identity"]) != SMOKE_RUN or spec["id"] != SMOKE_RUN:
        raise ValueError("Smoke run-spec identity mismatch")
    expected = {"source_manifest_id": SMOKE_SOURCE, "raw_input_bundle_id": RAW,
                "model_snapshot_id": MODEL, "adapter_bundle_id": SMOKE_ADAPTER,
                "fvi_study_id": SMOKE_FVI, "environment_contract_id": ENVIRONMENT}
    if any(spec["identity"]["identity"].get(key) != value for key,value in expected.items()):
        raise ValueError("Smoke run-spec bindings mismatch")
    if fvi["id"] != SMOKE_FVI or compute_id(fvi["identity"]) != SMOKE_FVI:
        raise ValueError("Smoke FVI identity mismatch")
    if paths["output"].exists():
        raise FileExistsError("Final output already exists; preserve it. This orchestrator never fabricates or overwrites a resume.")
    return {"python": platform.python_version(), "packages": actual,
        "jsonschema_version": importlib.metadata.version("jsonschema"), "environment_id": env_id,
        "final_commit": FINAL_COMMIT, "expected_final_source_id": FINAL_SOURCE,
        "smoke_inspection": inspection, "smoke_adapter_id": SMOKE_ADAPTER, "smoke_run_id": SMOKE_RUN,
        "model_id": MODEL, "raw_id": RAW, "fvi_id": SMOKE_FVI,
        "scientific_recomputation_performed": False,
        "checksum_manifest_sha256": sha(package / "SHA256SUMS")}


def model_stage(paths):
    from scripts.run_stopdff_v5_local import _load_imported_model_manifest
    destination = paths["work"] / "model"
    if destination.exists():
        _load_imported_model_manifest(destination, out=paths["output"], expected_model_snapshot_id=MODEL)
        return destination
    evidence = paths["input"] / "smoke_package/evidence"
    with tempfile.TemporaryDirectory(prefix=".model-", dir=paths["work"]) as holder:
        staged = Path(holder) / "model"
        staged.mkdir()
        shutil.copy2(evidence / "model_snapshot_manifest.json", staged / "model_snapshot_manifest.json")
        shutil.copytree(evidence / "model_snapshot/snapshot", staged / "snapshot", symlinks=True)
        _load_imported_model_manifest(staged, out=paths["output"], expected_model_snapshot_id=MODEL)
        from scripts.stopdff_v5.fileio import fsync_tree, publish_dir_create_once
        fsync_tree(staged)
        publish_dir_create_once(staged, destination)
    return destination


def integrated_final(paths, stdout):
    from scripts.stopdff_v5.identity import compute_id, loads_no_duplicate_keys
    candidates = list((paths["output"] / "runs").glob("final_local_*"))
    if len(candidates) != 1 or candidates[0].is_symlink():
        raise ValueError("Expected exactly one regular final run")
    run = candidates[0]
    log = stdout.read_text()
    decoder = json.JSONDecoder()
    results = []
    for offset, character in enumerate(log):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(log[offset:])
        except ValueError:
            continue
        if isinstance(value, dict) and "checker_passed" in value:
            results.append(value)
    if len(results) != 1 or f"LOCAL REPRODUCTION OK -> {run}" not in log:
        raise ValueError("Final runner lacks unambiguous integrated acceptance output")
    result = results[0]
    if (result.get("checker_passed") is not True or result.get("checker_errors") != []
            or result.get("release_status") != "VALID" or result.get("requested") != 96
            or result.get("completed") != 96 or result.get("failed") != 0):
        raise ValueError(f"Final integrated package gate did not pass: {result}")
    spec = loads_no_duplicate_keys((run / "run_spec.json").read_text())
    if compute_id(spec["identity"]) != spec["id"]:
        raise ValueError("Final run-spec identity mismatch")
    bindings = spec["identity"]["identity"]
    for key, expected in (("source_manifest_id", FINAL_SOURCE), ("model_snapshot_id", MODEL),
                          ("raw_input_bundle_id", RAW), ("environment_contract_id", ENVIRONMENT)):
        if bindings.get(key) != expected:
            raise ValueError(f"Final {key} mismatch")
    check_code(paths["code"])
    return run, {"run_root": str(run), "run_spec_id": spec["id"], "bindings": bindings,
        "checker_result": result, "backend": "local", "require_package": True,
        "require_final_profile": True, "standalone_validator_repeated": False,
        "log_sha256": sha(stdout), "checksum_manifest_sha256": sha(run / "SHA256SUMS")}


def require_smoke_acceptance(result):
    """Require a VALID two-cell smoke result, not just internally consistent data."""
    recomputed = result.get("recomputed") if isinstance(result, dict) else None
    if (not isinstance(result, dict) or result.get("passed") is not True
            or result.get("errors") != [] or not isinstance(recomputed, dict)
            or recomputed.get("release_status") != "VALID"
            or recomputed.get("completed") != 2 or recomputed.get("failed") != 0
            or recomputed.get("run_spec_id") != SMOKE_RUN):
        raise ValueError(f"Smoke package lacks VALID two-cell acceptance: {result}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent / "input")
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.heartbeat_seconds <= 300:
        parser.error("--heartbeat-seconds must be between 1 and 300")
    paths = {key: canonical_path(getattr(args, key + "_dir"))
             for key in ("input", "code", "work", "output", "receipt")}
    for key, path in paths.items():
        for other, other_path in paths.items():
            if key < other and (path.is_relative_to(other_path) or other_path.is_relative_to(path)):
                raise ValueError(f"{key} and {other} directories must be disjoint")
    paths["work"].mkdir(parents=True, exist_ok=True)
    paths["receipt"].mkdir(parents=True, exist_ok=True)
    journal = Journal(paths["receipt"], paths, args.heartbeat_seconds)
    print(f"Receipts: {journal.directory}", flush=True)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        with (paths["work"] / ".durable-rerun.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with journal.stage("preflight") as stage:
                stage["result"] = preflight(paths)
            if args.preflight_only:
                journal.update(status="PREFLIGHT_PASSED", stage="complete", ended_utc=now(),
                               scientific_acceptance=False)
                return 0
            package = paths["input"] / "smoke_package"
            old_validator = package / "evidence/source_snapshot/source/scripts/validate_stopdff_bucketed_sweep.py"
            log = journal.command("smoke_package_validation", [sys.executable, "-u", old_validator,
                "validate", package, "--backend", "local", "--adapter-bundle",
                paths["input"] / "adapter_bundle", "--require-package", "--json"], paths["code"])
            smoke = json.loads(log.read_text())
            require_smoke_acceptance(smoke)
            journal.update(smoke_acceptance={"mode": "standalone_full_package_validation", "result": smoke,
                "validator_commit": SMOKE_COMMIT, "log_sha256": sha(log), "runner_resume_claimed": False})
            with journal.stage("model_import") as stage:
                model = model_stage(paths)
                stage["result"] = {"model_snapshot_id": MODEL, "stage": str(model)}
            check_code(paths["code"])
            raw = package / "evidence/raw_inputs/raw"
            final_log = journal.command("final_runner", [sys.executable, "-u",
                paths["code"] / "scripts/run_stopdff_v5_local.py", "--repo-root", paths["code"],
                "--data-dir", raw, "--paper-exports", raw, "--model-snapshot-dir", model,
                "--expected-model-snapshot-id", MODEL, "--out-dir", paths["output"], "--variant", "final"], paths["code"])
            with journal.stage("final_integrated_acceptance") as stage:
                run, result = integrated_final(paths, final_log)
                stage["result"] = result
            journal.command("final_numerical_reducer", [sys.executable,
                paths["code"] / "reproducibility/stopdff_final_modal_5d5328102912/verify_expected_results.py", run], paths["code"])
            journal.update(status="PASSED", stage="complete", ended_utc=now(), scientific_acceptance=True,
                           final_run_root=str(run), final_code_commit=FINAL_COMMIT)
            return 0
    except BaseException as exc:
        interrupted = STOP.is_set() or isinstance(exc, KeyboardInterrupt)
        journal.update(status="INTERRUPTED" if interrupted else "FAILED", ended_utc=now(),
                       scientific_acceptance=False, error=f"{type(exc).__name__}: {exc}")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 128 + (STOP_SIGNAL or signal.SIGINT) if interrupted else 1
    finally:
        journal.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Pre-initialization failure: {exc}", file=sys.stderr)
        raise SystemExit(2)
