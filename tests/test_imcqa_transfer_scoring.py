"""CPU evidence gates: real physical batches, exact replay, and bounded accounting."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from scripts import imcqa_transfer_scoring as scoring


def job(index=0, **changes):
    return {"score_id": str(index), "allowed_actions": "ABCDE", "option_source_ids": dict(zip("ABCD", "ABCD")), **changes}


def test_diagnostic_feature_batch_is_not_the_physical_forward_batch(monkeypatch):
    observed = []
    def fake_forward(torch, model, tokenizer, contexts, *, cached, batch_size):
        observed.append(([c["i"] for c in contexts], cached, batch_size))
        return [{"i": c["i"]} for c in contexts]
    monkeypatch.setattr(scoring, "forward", fake_forward)
    contexts = [{"i": i} for i in range(30)]
    result = scoring.forward_batches(None, None, None, contexts, cached=True, permute=True)
    assert result == contexts
    assert [len(batch[0]) for batch in observed] == [8, 8, 8, 6]
    assert all(batch[1:] == (True, 8) for batch in observed)
    assert scoring.original.BATCH_SIZE == 32


@pytest.mark.parametrize("contexts", [[], [{}], [{}]*3])
def test_incomplete_cache_pairs_rejected(contexts):
    with pytest.raises(ValueError, match="complete context pairs"):
        scoring.forward_batches(None, None, None, contexts, cached=True)


def test_diagnostics_include_true_individual_minimum_not_only_shortest_pair(monkeypatch):
    contexts = [{"scored_input_token_ids": [0]*length} for length in (1, 100, 10, 10, 200, 200)]
    jobs = [job(i) for i in range(6)]
    monkeypatch.setattr(scoring.original, "diagnostic_indices", lambda jobs, contexts: [2, 3, 4, 5])
    assert scoring.diagnostic_indices(jobs, contexts) == list(range(6))


def test_production_checks_cover_actual_first_middle_last_batches():
    assert scoring.production_offsets(8000) == [0, 4000, 7992]
    with pytest.raises(ValueError):
        scoring.production_offsets(7999)


def test_failure_preserves_raw_numeric_sides_before_rejection(tmp_path):
    path = tmp_path/"failure_raw.json"
    left, right = [{"logits": [0, 1, 2, 3, 4]}], [{"logits": [0, 1, 2, 3, 4.01]}]
    with pytest.raises(ValueError, match="logits exceed"):
        scoring.record_numeric_gate(path, left, right, [job()])
    saved = json.loads(path.read_text())
    assert saved["production"] == left and saved["reference"] == right
    assert saved["score_ids"] == ["0"]


def test_unchanged_numeric_gate_catches_argmax_even_within_raw_tolerance():
    with pytest.raises(ValueError, match="argmax"):
        scoring.numeric_agreement([{"logits": [1.0001, 1, 0, 0, 0]}],
            [{"logits": [1, 1.0001, 0, 0, 0]}], [job()])
    gate = scoring.numeric_agreement([{"logits": [1, 0, 0, 0, 0]}],
            [{"logits": [1, 0, 0, 0, 0]}], [job()])
    assert (gate["atol"], gate["rtol"], gate["probability_atol"]) == (.001, .00001, .001)


def active_evidence():
    rows = []
    for qid in ("q0", "q1", "q2", "q3"):
        for condition in scoring.CONDITIONS:
            for rotation in (0, 2):
                rows.append({"score_id": f"{qid}|{condition}|{rotation}", "qid": qid,
                    "condition": condition, "rotation": rotation, "round": 1,
                    "chosen_action": "A", "agreement": {"passed": True}})
    return {"qids": [f"q{i}" for i in range(4)], "episodes": 16, "rows": rows}


def test_complete_nonempty_active_replay_accepted():
    scoring.validate_active_evidence(active_evidence())


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "short_wait", "round_gap", "missing_episode"])
def test_false_active_completion_rejected(mutation):
    evidence = active_evidence()
    if mutation == "empty":
        evidence["rows"] = []
    elif mutation == "duplicate":
        evidence["rows"][1] = evidence["rows"][0]
    elif mutation == "short_wait":
        evidence["rows"][0]["chosen_action"] = "E"
    elif mutation == "round_gap":
        evidence["rows"][0]["round"] = 2
    else:
        evidence["rows"] = evidence["rows"][1:]
    with pytest.raises(ValueError):
        scoring.validate_active_evidence(evidence)


def test_projection_reserves_active_and_actual_production_validation():
    projected = scoring.projection_with_validation(128, 8000, 12.8, 2000, [0.8]*16, [0.1, 0.5])
    assert projected["forecast_method"] == "max(mean_times_1_2,nearest_rank_p95_batch_projection)"
    assert projected["total_validation_reserve_seconds"] == pytest.approx((80+24+48)*.5*1.2+45)
    assert projected["proceed"] is True
    stopped = scoring.projection_with_validation(128, 8000, 12.8, 900, [0.8]*16, [.5])
    assert stopped["proceed"] is False
    with pytest.raises(ValueError):
        scoring.projection_with_validation(128, 8000, 12.8, 2000, [.8]*16, [math.nan])


def test_only_new_7b_allocation_accepted_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="invalid pinned model"):
        scoring.run_scoring("qwen3b", tmp_path/"missing", "0"*64, tmp_path, tmp_path/"out",
                            source_commit="1"*40, max_seconds=2880)
    with pytest.raises(ValueError, match="invalid pinned model"):
        scoring.run_scoring("qwen7b", tmp_path/"missing", "0"*64, tmp_path, tmp_path/"out",
                            source_commit="1"*40, max_seconds=2881)


def test_mocked_complete_worker_exercises_runtime_and_live_replay(tmp_path, monkeypatch):
    """Execute the complete orchestration with a fake model, including receipt construction."""
    tag = "qwen7b"
    revision = scoring.base.PINNED_MODELS[scoring.base.MODELS[tag]]
    snapshot = tmp_path/revision
    snapshot.mkdir()
    weight = snapshot/"model.safetensors"
    weight.write_bytes(b"fake pinned weight")
    (tmp_path/f"{tag}_expected_model_hashes.json").write_text(json.dumps({
        "model": scoring.base.MODELS[tag], "revision": revision,
        "model_files_sha256": {weight.name: scoring.base.file_hash(weight)}}))
    tokenizer = SimpleNamespace(padding_side="left", pad_token_id=0, chat_template="test")
    model = SimpleNamespace()
    model.to = lambda *args: model
    model.eval = lambda: model
    model.float = lambda: model
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)))
    nothing = lambda *args, **kwargs: None
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True,
        manual_seed_all=nothing, synchronize=nothing, empty_cache=nothing, max_memory_allocated=lambda: 123)
    torch = SimpleNamespace(cuda=cuda, set_num_threads=nothing, manual_seed=nothing, use_deterministic_algorithms=nothing,
        set_float32_matmul_precision=nothing, bfloat16="bfloat16",
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "torch", torch)
    versions = {"torch": "2.6.0", "transformers": "4.51.3", "tokenizers": "0.21.1", "safetensors": "0.5.3", "huggingface-hub": "0.30.2"}
    monkeypatch.setattr(scoring.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(scoring.base, "snapshot_tensor_state", lambda model: {})
    monkeypatch.setattr(scoring.paired, "validate_promotion", lambda model, state: {"all_checks_passed": True})
    monkeypatch.setattr(scoring, "SOURCE_FILES", ())
    jobs = []
    for arm in ("plain", "wait"):
        for qid in ("q0", "q1", "q2", "q3"):
            for condition in scoring.CONDITIONS:
                for rotation in (0, 2):
                    jobs.append(job(len(jobs), arm=arm, qid=qid, condition=condition, rotation=rotation,
                        block="factorial", split="selection", execution="new", wait_label="E", round=1))
    monkeypatch.setattr(scoring, "EXPECTED_ROWS", len(jobs))
    monkeypatch.setattr(scoring.design, "validate_public_package", lambda package: jobs)
    monkeypatch.setattr(scoring, "prepare_context", lambda tokenizer, job: {
        "option_token_ids": dict(zip("ABCDE", range(5))), "scored_context_sha256": "a"*64,
        "scored_input_token_ids": [1, 2, int(job["score_id"])+3]})
    monkeypatch.setattr(scoring, "paired_order", lambda contexts: list(range(len(contexts))))
    monkeypatch.setattr(scoring.original, "diagnostic_indices", lambda jobs, contexts: list(range(8)))
    def fake_forward(torch, model, tokenizer, contexts, **kwargs):
        return [{"logits": [2., 0., 0., 0., -2.], "vocabulary_logsumexp": 3.,
                 "unconstrained_top_token_id": 0, "unconstrained_top_logit": 2.} for _ in contexts]
    monkeypatch.setattr(scoring, "forward", fake_forward)
    monkeypatch.setattr(scoring, "validate_rows", lambda *args, **kwargs: None)
    public = tmp_path/"public.json"
    public.write_text("{}")
    receipt = scoring.run_scoring(tag, public, scoring.base.file_hash(public), tmp_path, tmp_path/"out",
                                  source_commit="1"*40, max_seconds=2880)
    assert receipt["status"] == "complete", receipt
    assert receipt["completed_rows"] == 32
    assert receipt["production_single_gate"]["rows"] == 24
    active = json.loads((tmp_path/"out/attempts/000_active_only.json").read_text())
    assert active["episodes"] == 16 and len(active["rows"]) == 16
    assert (tmp_path/"out/attempts/000_active_000_raw.json").is_file()
    assert receipt["scores_sha256"] == scoring.base.file_hash(tmp_path/"out/scores.jsonl")
