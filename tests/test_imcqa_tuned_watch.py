"""Read-only status observer boundaries with synthetic remote metadata."""
import json
from pathlib import Path

import pytest
import yaml

from scripts import imcqa_tuned_watch as watch


class Volume:
    def __init__(self, files):
        self.files, self.reads = files, []

    def read_file(self, path):
        self.reads.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        yield json.dumps(self.files[path]).encode()


@pytest.fixture
def remote(monkeypatch):
    # Frozen-source verification is tested separately; this isolates observer IO.
    monkeypatch.setattr(watch, "verify_sources", lambda root, control: None)
    return Volume({"control.json": {"run_id": watch.RUN_ID, "input_sha256": watch.INPUT_SHA256,
        "source_commit": watch.SOURCE_COMMIT, "production_contexts_per_model": 34000,
        "budget": {"n_questions": 850, "ceiling_usd": "5.60", "allocation_rate_usd_per_second": "0.00063924"}}})


def test_absent_optional_files_are_a_valid_nonterminal_observation(remote):
    report = watch.inspect(remote)
    assert remote.reads == ["control.json", *watch.FILES]
    assert report["terminal"] is False and report["summary"]["worker_entered"] is False
    assert all(value == {"present": False} for value in report["files"].values())


def test_worker_claim_distinguishes_entry_without_claiming_scoring(remote):
    remote.files[watch.FILES[0]] = {"source_commit": watch.SOURCE_COMMIT, "model_tag": "qwen7b", "started_unix": 123.}
    report = watch.inspect(remote)
    assert report["summary"]["worker_entered"] is True
    assert report["summary"]["phase"] is None and report["terminal"] is False


@pytest.mark.parametrize("status", sorted(watch.TERMINAL))
def test_each_terminal_receipt_stops_observation(remote, status):
    remote.files[watch.FILES[3]] = {"source_commit": watch.SOURCE_COMMIT, "public_input_sha256": watch.INPUT_SHA256,
        "expected_rows": 34000, "status": status}
    report = watch.inspect(remote)
    assert report["terminal"] is True and report["summary"]["worker_status"] == status


def test_control_mismatch_prevents_any_status_reads(remote):
    remote.files["control.json"]["source_commit"] = "0"*40
    with pytest.raises(ValueError, match="identity"):
        watch.inspect(remote)
    assert remote.reads == ["control.json"]


def test_wrong_terminal_receipt_cannot_stop_the_watcher(remote):
    remote.files[watch.FILES[3]] = {"source_commit": "0"*40, "public_input_sha256": watch.INPUT_SHA256,
        "expected_rows": 34000, "status": "complete"}
    with pytest.raises(ValueError, match="terminal"):
        watch.inspect(remote)


def test_workflow_has_one_bounded_read_sequence_with_each_snapshot_persisted():
    root = Path(__file__).resolve().parents[1]
    wf = yaml.load((root/".github/workflows/imcqa-tuned-recovery-watch.yml").read_text(), Loader=yaml.BaseLoader)
    assert wf["on"]["push"] == {"branches": ["feat/imcqa-tuned-policy-comparison-20261005"],
        "paths": [".github/workflows/imcqa-tuned-recovery-watch.yml"]}
    job = wf["jobs"]["watch"]
    assert int(job["timeout-minutes"]) == 155
    assert "github.run_attempt == 1" in job["if"]
    assert "github.actor == 'ankaggarwal94'" in job["if"]
    assert "ops: watch frozen tuned IMCQA recovery without compute" in job["if"]
    steps = job["steps"]
    reads = [s for s in steps if s.get("id", "").startswith("snapshot_")]
    waits = [s for s in steps if s.get("run") == "sleep 600"]
    artifacts = [s for s in steps if s.get("uses") == "actions/upload-artifact@v4"]
    assert len(reads) == len(artifacts) == 15 and len(waits) == 14
    for index, step in enumerate(reads):
        assert step["run"] == f"python -m scripts.imcqa_tuned_watch --index {index}"
        assert "env.IMCQA_WATCH_COMPLETE != 'true'" in step["if"]
        assert "MODAL_TOKEN_SECRET" in step["env"]
        assert artifacts[index]["with"]["path"].endswith(f"snapshot-{index:02d}.json")
    assert all("env.IMCQA_WATCH_COMPLETE != 'true'" in step["if"] for step in waits)
