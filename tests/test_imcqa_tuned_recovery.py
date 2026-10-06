"""Cumulative budget and immutable-data gates for one manual recovery."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pytest
import yaml

from scripts import imcqa_tuned_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 6, 2, 0, tzinfo=timezone.utc)


def evidence():
    folder = ROOT/"imcqa_tuned_recovery"
    manifest = json.loads((folder/"recovery_manifest.json").read_bytes())
    prior = [json.loads((folder/"prior"/name).read_bytes()) for name in recovery.PRIOR_FILES]
    return manifest, prior


def test_actual_stopped_attempt_plus_full_recovery_fits_total_ceiling():
    manifest, prior = evidence()
    budget, ledger = recovery.cumulative_budget(manifest, *prior, now=NOW)
    assert ledger["prior_interval_cost_bound_usd"] == "0.399442880"
    assert ledger["prior_attempt_reserved_usd"] == "0.40"
    assert ledger["combined_reserved_estimate_usd"] == "5.93254008"
    assert budget["ceiling_usd"] == "5.60"
    assert (budget["worker_timeout_seconds"], budget["internal_deadline_seconds"]) == (8250, 8130)
    assert ledger["invoice_verified"] is False and ledger["reused_prior_scores"] == 0


@pytest.mark.parametrize("mutation", ["prior_debit", "total_cap", "allocation_cap", "unfinished",
    "more_time", "wrong_commit", "complete_scores", "reused_rows", "retry", "nan_elapsed", "wrong_rate"])
def test_cumulative_budget_rejects_changed_cost_or_attempt(mutation):
    manifest, prior = evidence()
    control, worker, allocation, job = prior
    if mutation == "prior_debit": manifest["prior_attempt_reserved_usd"] = "0.39"
    elif mutation == "total_cap": manifest["total_user_ceiling_usd"] = "6.01"
    elif mutation == "allocation_cap": manifest["pricing"]["max_cost_usd"] = "6.00"
    elif mutation == "unfinished": job["status"] = "in_progress"
    elif mutation == "more_time": job["completed_at"] = "2026-10-06T01:53:35Z"
    elif mutation == "wrong_commit": control["source_commit"] = "0"*40
    elif mutation == "complete_scores": worker["status"] = "complete"
    elif mutation == "reused_rows": worker["reused_rows"] = 128
    elif mutation == "retry": worker["automatic_retries"] = 1
    elif mutation == "nan_elapsed": allocation["elapsed_seconds"] = float("nan")
    elif mutation == "wrong_rate": allocation["allocation_rate_usd_per_second"] = "0.0005"
    with pytest.raises(ValueError):
        recovery.cumulative_budget(manifest, *prior, now=NOW)


def test_cumulative_sum_is_enforced_independently_of_manifest(monkeypatch):
    manifest, prior = evidence()
    actual = recovery.budget_plan
    def too_expensive(*args, **kwargs):
        plan = actual(*args, **kwargs)
        plan["reserved_estimate_usd"] = "5.60000001"
        return plan
    monkeypatch.setattr(recovery, "budget_plan", too_expensive)
    with pytest.raises(ValueError, match="cumulative"):
        recovery.cumulative_budget(manifest, *prior, now=NOW)


@pytest.fixture
def copied_inputs(tmp_path):
    for name in ("imcqa_tuned_recovery", "imcqa_tuned_public"):
        shutil.copytree(ROOT/name, tmp_path/name)
    for name in (*recovery.SOURCES, "configs/imcqa_tuned_fresh.json"):
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    return tmp_path


def test_real_transport_restores_identical_full_input_and_prior_ledger(copied_inputs):
    root = copied_inputs
    ledger = recovery.prepare(root, root/"out", now=NOW)
    assert ledger["combined_reserved_estimate_usd"] == "5.93254008"
    assert recovery.sha((root/"out/public.json").read_bytes()) == recovery.PUBLIC_SHA256
    for name in recovery.PRIOR_FILES:
        assert (root/"out/prior"/name).read_bytes() == (root/"imcqa_tuned_recovery/prior"/name).read_bytes()
    assert not (root/"out/run").exists()
    with pytest.raises(FileExistsError):
        recovery.prepare(root, root/"out", now=NOW)


@pytest.mark.parametrize("mutation", ["prior_bytes", "prior_rebound", "transport_bytes", "gzip_bytes",
    "public_hash", "source_bytes", "scientific_source_rebound", "policy_config", "source_manifest"])
def test_recovery_input_corruption_creates_no_handoff(copied_inputs, mutation):
    root = copied_inputs
    path = root/"imcqa_tuned_recovery/recovery_manifest.json"
    manifest = json.loads(path.read_bytes())
    if mutation in ("prior_bytes", "prior_rebound"):
        prior = root/"imcqa_tuned_recovery/prior/worker_receipt.json"
        receipt = json.loads(prior.read_bytes())
        receipt["status"] = "complete"
        prior.write_text(json.dumps(receipt))
        if mutation == "prior_rebound": manifest["prior_files_sha256"]["worker_receipt.json"] = recovery.sha(prior.read_bytes())
    elif mutation == "transport_bytes":
        (root/"imcqa_tuned_public/transport_manifest.json").write_text("{}")
    elif mutation == "gzip_bytes":
        (root/"imcqa_tuned_public/public.json.gz").write_bytes(b"different")
    elif mutation == "public_hash":
        prior = root/"imcqa_tuned_recovery/prior/control.json"
        control = json.loads(prior.read_bytes())
        control["input_sha256"] = "0"*64
        prior.write_text(json.dumps(control))
        manifest["prior_files_sha256"]["control.json"] = recovery.sha(prior.read_bytes())
    elif mutation == "source_bytes":
        (root/"scripts/imcqa_tuned_scoring.py").write_text("changed")
    elif mutation == "scientific_source_rebound":
        name = "scripts/imcqa_protocol_scoring.py"
        (root/name).write_text("changed")
        manifest["source_files_sha256"][name] = recovery.sha((root/name).read_bytes())
    elif mutation == "policy_config":
        (root/"configs/imcqa_tuned_fresh.json").write_text("{}")
    elif mutation == "source_manifest":
        manifest["source_files_sha256"]["scripts/imcqa_tuned_scoring.py"] = "0"*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        recovery.prepare(root, root/"out", now=NOW)
    assert not (root/"out").exists()


def test_single_manual_recovery_is_scoped_and_gated_before_credentials():
    data = yaml.load((ROOT/".github/workflows/imcqa-tuned-recovery.yml").read_text(), Loader=yaml.BaseLoader)
    assert data["on"] == {"push": {"branches": ["feat/imcqa-tuned-policy-comparison-20261005"],
                                  "paths": [".github/workflows/imcqa-tuned-recovery.yml"]}}
    assert data["permissions"] == {"contents": "read"}
    job = data["jobs"]["score"]
    for guard in ("github.actor == 'ankaggarwal94'", "github.run_attempt == 1",
                  "github.event_name == 'push'", "ops: recover frozen tuned IMCQA comparison within total six dollars"):
        assert guard in job["if"]
    steps = job["steps"]
    credential = [i for i, step in enumerate(steps) if "MODAL_TOKEN_SECRET" in step.get("env", {})]
    assert len(credential) == 1
    i = credential[0]
    assert any("scripts.imcqa_tuned_recovery" in step.get("run", "") for step in steps[:i-1])
    assert steps[i-1]["run"].endswith("--dry-run")
    assert steps[i-1]["run"].removesuffix(" --dry-run") == steps[i]["run"]
    assert "--max-cost-usd 5.60" in steps[i]["run"]
    assert "--run-id "+recovery.RECOVERY_RUN in steps[i]["run"]
    assert int(steps[i]["timeout-minutes"]) == 160
    assert steps[-1]["if"] == "always()" and steps[-1]["with"]["path"] == "imcqa_tuned_run_artifacts/"
    assert "evaluator" not in steps[i]["run"] and "frozen_policies" not in steps[i]["run"]
