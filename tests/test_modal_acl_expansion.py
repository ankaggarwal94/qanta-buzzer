"""Focused CPU guards for expansion execution; never load weights or contact Modal."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import modal_acl_expansion as runner


def package(n=16):
    jobs = []
    for index in range(n):
        prompt = f"Synthetic test prefix {index}"
        jobs.append({"job_id": f"job-{index:04d}", "qid": f"question-{index}",
                     "group_id": f"group-{index}", "split": "calibration", "format": "oe",
                     "condition": "oe", "menu_id": None, "prefix_id": "p1", "fraction": 0.1,
                     "prompt": prompt, "prompt_sha256": runner.sha(prompt.encode())})
    return {"schema_version": "jane-public-jobs-v1", "evidence_scope": "scientific", "jobs": jobs}


def sources():
    return {name: runner.sha(name.encode()) for name in runner.SOURCE_FILES}


def trace_for(shard):
    jobs = shard["package"]["jobs"]
    model = dict(runner.MODELS)[shard["model_tag"]]
    rows = []
    for i, job in enumerate(jobs):
        value = {"answer": "France", "confidence": 0.75, "status": "answer"}
        rows.append({"job_id": job["job_id"], "prompt_sha256": job["prompt_sha256"], **value,
                     "raw_response": json.dumps(value, separators=(",", ":")), "input_token_ids": [1, 2],
                     "generated_token_ids": [3, 4], "input_tokens": 2, "output_tokens": 2,
                     "constraint_failure_kind": None, "batch_index": i // 8,
                     "batch_job_ids": [j["job_id"] for j in jobs[i // 8 * 8:i // 8 * 8 + 8]]})
    metadata = {"model": model, "revision": runner.PINNED_MODELS[model],
                "execution": "actual_cuda_model_generation", "context_policy": "fresh_per_prefix",
                "evidence_scope": "scientific", "batch_size": 8, "seed": 1,
                "max_input_tokens": 2048, "max_new_tokens": 160, "greedy": True,
                "dtype": "bfloat16", "n_jobs": len(jobs), "confidence_method": runner.CONFIDENCE_METHOD,
                "backend_sha256": sources()["scripts/jane_gpu_backend.py"],
                "parser_sha256": sources()["scripts/jane_qwen_backend.py"],
                "output_constraints": {"source_sha256": sources()["scripts/jane_output_constraints.py"]},
                "public_package_canonical_sha256": runner.sha(runner.canonical(shard["package"])[:-1]),
                "total_seconds": 2, "model_load_seconds": 1, "n_length_capped_invalid_predictions": 0}
    return {"schema_version": "jane-traces-v1", "metadata": metadata, "predictions": rows}


def plan_stub(seconds=57600):
    return {"run_identity": "a" * 64, "run_id": "acl5000-test", "source_files_sha256": sources(),
            "shard_size": 8, "public_files": {"test": "identity"},
            "budget": runner.budget_plan("80", seconds)}


def save_completed(directory, shard, trace):
    runner.write_once(directory / "started.json", {"shard_id": shard["shard_id"],
                      "source_files_sha256": sources()})
    runner.write_once(directory / "trace.json", trace)
    runner.write_once(directory / "completion.json", {"shard_id": shard["shard_id"],
                      "trace_sha256": runner.file_hash(directory / "trace.json")})


def test_full_initial_budget_is_bounded_and_independent_of_old_pilot():
    budget = runner.budget_plan("80", 57600)
    assert Decimal(budget["maximum_reserved_estimate_usd"]) == Decimal("79.640448")
    assert budget["previous_reserved_seconds"] == 0
    assert budget["prior_reservations"] == []
    assert runner.MAX_DOWNLOAD_BYTES >= 3 * 1024**3


@pytest.mark.parametrize("ceiling,seconds", [("NaN", 57600), ("Infinity", 57600),
    ("81", 57600), ("6", 57600), ("70", 57600), ("80", 57601), ("80", True)])
def test_invalid_budget_rejected_without_provider(ceiling, seconds):
    with pytest.raises(ValueError):
        runner.budget_plan(ceiling, seconds)


def test_resume_reserves_full_cost_of_prior_failed_attempts():
    plan = plan_stub()
    previous = runner.initial_allocations(plan)
    runner.budget_plan("80", 300, previous, additional_models=1)
    with pytest.raises(ValueError, match="cumulative"):
        runner.budget_plan("80", 600, previous, additional_models=1)
    mutated = deepcopy(previous); mutated[1]["reservation_index"] = 0
    with pytest.raises(ValueError):
        runner.validate_allocations(plan, mutated)


def test_shards_preserve_every_original_batch_and_stable_identity():
    p = package(40)
    shards = runner.shards_for(p, "qwen3b", "main", 16)
    assert [j for s in shards for j in s["package"]["jobs"]] == p["jobs"]
    assert [(s["start"], s["stop"]) for s in shards] == [(0, 16), (16, 32), (32, 40)]
    assert shards == runner.shards_for(p, "qwen3b", "main", 16)
    assert {s["shard_id"] for s in shards}.isdisjoint({s["shard_id"] for s in runner.shards_for(p, "qwen7b", "main", 16)})
    with pytest.raises(ValueError):
        runner.shards_for(p, "qwen3b", "main", 15)


@pytest.mark.parametrize("field", ["answerline", "gold_option_id", "future_question"])
def test_public_boundary_rejects_hidden_evaluator_data(field):
    p = package(8); p["jobs"][0][field] = "hidden"
    with pytest.raises(ValueError, match="unexpected"):
        runner.validate_package(p, max_jobs=8)


def test_exact_frozen_file_hash_required_before_json_or_provider(tmp_path):
    (tmp_path / "main_jobs.json").write_text(json.dumps(package(8)))
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.load_public_inputs(tmp_path)


def test_verified_completion_skips_but_started_incomplete_blocks(tmp_path):
    shard = runner.shards_for(package(8), "qwen3b", "main", 8)[0]
    absent = tmp_path / "absent"
    assert runner.completed_shard(absent, shard, sources()) is None
    runner.write_once(absent / "started.json", {"shard_id": shard["shard_id"], "source_files_sha256": sources()})
    with pytest.raises(RuntimeError, match="AMBIGUOUS_INCOMPLETE_SHARD"):
        runner.completed_shard(absent, shard, sources())
    complete = tmp_path / "complete"; trace = trace_for(shard)
    save_completed(complete, shard, trace)
    assert runner.completed_shard(complete, shard, sources()) == trace
    (complete / "trace.json").write_text('{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.completed_shard(complete, shard, sources())


@pytest.mark.parametrize("mutation", ["raw", "tokens", "backend", "package", "batch", "duplicate"])
def test_trace_tampering_rejected(mutation):
    shard = runner.shards_for(package(8), "qwen3b", "main", 8)[0]
    trace = trace_for(shard)
    runner.verify_trace(shard, trace, sources())
    if mutation == "raw": trace["predictions"][0]["answer"] = "Germany"
    elif mutation == "tokens": trace["predictions"][0]["input_token_ids"] = []
    elif mutation == "backend": trace["metadata"]["backend_sha256"] = "0" * 64
    elif mutation == "package": trace["metadata"]["public_package_canonical_sha256"] = "0" * 64
    elif mutation == "batch": trace["predictions"][0]["batch_job_ids"].reverse()
    else: trace["predictions"][1] = deepcopy(trace["predictions"][0])
    with pytest.raises(ValueError): runner.verify_trace(shard, trace, sources())


def test_backend_exception_before_progress_still_closes_and_commits_checkpoint(tmp_path, monkeypatch):
    plan = plan_stub(300)
    allocations = runner.initial_allocations(plan)
    runner.write_once(tmp_path / "allocations" / "initial.json", allocations)
    monkeypatch.setattr(runner, "validate_plan", lambda _plan: None)
    monkeypatch.setattr(runner, "source_identity", lambda _repo: sources())
    monkeypatch.setattr(runner, "load_public_inputs", lambda _path: (
        {"main_jobs.json": package(8), "main_choices_only.json": package(0)}, plan["public_files"]))
    # A generous test-only remaining budget permits the first tiny shard.
    plan["budget"] = runner.budget_plan("80", 1000)
    allocations = runner.initial_allocations(plan)
    (tmp_path / "allocations" / "initial.json").write_bytes(runner.canonical(allocations))
    writes, checkpoints = [], []
    def commit(): writes.append(True)
    def fail(_package, _config, checkpoint, progress):
        checkpoint.write('{"partial":"retained"}\n'); checkpoints.append(checkpoint)
        raise ValueError("simulated backend constraint failure before progress")
    result = runner.execute_model(plan, allocations[0], tmp_path, tmp_path, commit, generate=fail)
    assert result["status"] == "FAILED_REQUIRES_REVIEW"
    assert checkpoints[0].stream is None
    assert checkpoints[0].path.read_text() == '{"partial":"retained"}\n'
    assert len(writes) >= 4


def test_launch_without_source_commit_rejected_before_provider(tmp_path, monkeypatch):
    plan = plan_stub(); plan["source_commit"] = None
    monkeypatch.setattr(runner, "validate_plan", lambda _plan: None)
    monkeypatch.setattr(runner, "connect", lambda: pytest.fail("provider accessed"))
    with pytest.raises(ValueError, match="source-commit"):
        runner.launch(plan, tmp_path, tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_atomic_create_once_prevents_duplicate_local_claim(tmp_path):
    path = tmp_path / "claim.json"
    runner.write_once(path, {"reservation": 0})
    with pytest.raises(FileExistsError): runner.write_once(path, {"reservation": 1})
    assert runner.load(path.read_bytes()) == {"reservation": 0}
