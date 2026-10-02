#!/usr/bin/env python3
"""Validate the 5,000-question expansion on CPU without loading model weights.

This opt-in validator does not change the historical 200-question Modal runner.
It validates evaluator/public agreement, exact public prompt identities, and
every chat-rendered input against the pinned tokenizer and 2,048-token cap.
Install tokenizers==0.21.1 to run the token-length check. No network is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qb_data.jane_paired import _canonical, _prompt, validate_dataset
from scripts.jane_gpu_backend import PINNED_MODELS, validate_package
from scripts.jane_qwen_backend import _reject_constant, _unique_object


def read_json(path: Path) -> Any:
    """Read finite JSON while rejecting duplicate object keys."""
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def file_hash(path: Path) -> str:
    """Hash exact file bytes with bounded working memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def render_single_user(prompt: str) -> str:
    """Render the hash-pinned Qwen template for one user and no tool messages.

    This exact specialization includes its default system message and the
    assistant generation prefix. Tokenization uses add_special_tokens=False,
    matching scripts/jane_gpu_backend.py. Other chat-template inputs are not
    supported by this preflight.
    """
    return ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
            "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            + prompt + "<|im_end|>\n<|im_start|>assistant\n")


def tokenizer_from_files(directory: Path, config: dict) -> tuple[Any, dict]:
    """Verify both pinned model identities before opening their shared tokenizer."""
    import tokenizers

    expected = config["tokenizer"]
    if tokenizers.__version__ != expected["tokenizers_version"]:
        raise ValueError("tokenizers version does not match the pinned preflight")
    tokenizer_path = directory / "qwen3b" / "tokenizer.json"
    if file_hash(tokenizer_path) != expected["tokenizer_json_sha256"]:
        raise ValueError("tokenizer.json SHA256 differs from pinned asset")
    raw = tokenizer_path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != expected["tokenizer_git_blob_id"]:
        raise ValueError("tokenizer.json Git object does not match pinned models")
    provenance = []
    for tag, model in (("qwen3b", "Qwen/Qwen2.5-3B-Instruct"),
                       ("qwen7b", "Qwen/Qwen2.5-7B-Instruct")):
        template_path = directory / tag / "tokenizer_config.json"
        if file_hash(template_path) != expected["tokenizer_config_sha256"]:
            raise ValueError(f"{tag} tokenizer_config differs from pinned template")
        metadata_path = directory / tag / "hub_model_metadata.json"
        metadata = read_json(metadata_path)
        if metadata.get("sha") != PINNED_MODELS[model]:
            raise ValueError(f"{tag} model metadata does not identify pinned revision")
        sibling = [entry for entry in metadata["siblings"]
                   if entry["rfilename"] == "tokenizer.json"]
        if len(sibling) != 1 or sibling[0].get("blobId") != blob:
            raise ValueError(f"{tag} pinned metadata does not bind shared tokenizer")
        provenance.append({"model": model, "revision": metadata["sha"],
                           "metadata_sha256": file_hash(metadata_path),
                           "tokenizer_config_sha256": file_hash(template_path),
                           "shared_tokenizer_git_blob_id": blob})
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer, {"version": tokenizers.__version__, "models": provenance,
                       "tokenizer_json_sha256": file_hash(tokenizer_path),
                       "rendering": "exact pinned single-user template specialization"}


def validate_config(config: dict) -> None:
    """Reject drift from the agreed dataset design and generation limits."""
    expected = {
        "schema_version": "acl-expansion-preflight-v1",
        "execution_authorized_by_this_file": False,
        "prompt_template": "concise_json_v4", "questions": 5000,
        "split_counts": {"calibration": 1000, "selection": 1000, "test": 3000},
        "checkpoints_per_question": 10,
        "arms": ["oe", "independent_pool", "same_category_pool"],
        "main_jobs_per_model": 150000, "choices_only_jobs_per_model": 10000,
        "models": PINNED_MODELS, "historical_runner_compatible": False,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"unexpected expansion configuration: {name}")
    generation = config["generation"]
    for name, value in {"max_input_tokens": 2048, "max_new_tokens": 160,
                        "do_sample": False, "num_beams": 1, "seed": 1,
                        "batch_size": 8, "precision": "bfloat16",
                        "truncate_inputs": False,
                        "confidence_method": "self_reported_correctness_probability_uncalibrated",
                        "constraint_failure_policy": "retain_invalid_at_token_cap_v1"}.items():
        if type(generation.get(name)) is not type(value) or generation[name] != value:
            raise ValueError(f"unexpected generation setting: {name}")


def validate_inputs(dataset_path: Path, public_dir: Path, tokenizer_dir: Path,
                    config_path: Path) -> dict:
    """Validate frozen inputs and return a provenance-bound CPU-only receipt.

    Parameters
    ----------
    dataset_path : Path
        Evaluator dataset, used only by this local validation process.
    public_dir : Path
        Folder containing main_jobs.json and main_choices_only.json.
    tokenizer_dir : Path
        Local hash-pinned tokenizer files and per-model metadata.
    config_path : Path
        Explicit opt-in expansion profile, never an execution authorization.

    Returns
    -------
    dict
        Counts, exact input hashes, token lengths, and unexecuted-run status.
    """
    started = time.monotonic()
    config = read_json(config_path)
    validate_config(config)
    dataset = read_json(dataset_path)
    validate_dataset(dataset)
    if dataset.get("prompt_template") != config["prompt_template"]:
        raise ValueError("dataset must use the unchanged concise_json_v4 template")
    questions = {q["qid"]: q for q in dataset["questions"]}
    if len(questions) != config["questions"]:
        raise ValueError("expansion requires exactly 5,000 distinct question IDs")
    splits = Counter(q["split"] for q in questions.values())
    if dict(splits) != config["split_counts"]:
        raise ValueError("expansion split cardinalities differ from 1,000/1,000/3,000")
    for question in questions.values():
        if not re.fullmatch(r"(?:[A-Za-z0-9_-]+:)?[0-9a-f]{16,64}", question["group_id"]):
            raise ValueError("group_id must be an opaque hex identity without answer text")
        if len(question["prefixes"]) != config["checkpoints_per_question"]:
            raise ValueError("every question must have ten checkpoints")
        total = len(question["question"].split())
        for index, prefix in enumerate(question["prefixes"], start=1):
            if len(prefix["text"].split()) != math.floor(total * index / 10):
                raise ValueError("checkpoints must end at floor(N * k / 10) words")
        if sorted(menu["condition"] for menu in question["menus"]) != sorted(config["arms"][1:]):
            raise ValueError("question does not have exactly the two expected MC menus")
    tokenizer, tokenizer_evidence = tokenizer_from_files(tokenizer_dir, config)
    files = []
    all_job_ids: set[str] = set()
    maximum = {"input_tokens": -1}
    oversized = []
    observed_counts: Counter = Counter()
    tokens_by_phase: dict[str, dict] = {}
    for filename, controls, expected_count in (
            ("main_jobs.json", False, config["main_jobs_per_model"]),
            ("main_choices_only.json", True, config["choices_only_jobs_per_model"])):
        path = public_dir / filename
        package = read_json(path)
        jobs = validate_package(package, max_jobs=expected_count)
        if len(jobs) != expected_count:
            raise ValueError(f"{filename} has wrong job count")
        required_schema = "jane-choice-controls-v1" if controls else "jane-public-jobs-v1"
        if package["schema_version"] != required_schema:
            raise ValueError(f"{filename} has incorrect schema")
        if package["evidence_scope"] != dataset["evidence_scope"]:
            raise ValueError("public and evaluator evidence scopes differ")
        phase_tokens = []
        for start in range(0, len(jobs), 256):
            batch = jobs[start:start + 256]
            for job in batch:
                if job["job_id"] in all_job_ids:
                    raise ValueError("duplicate job ID across public packages")
                all_job_ids.add(job["job_id"])
                question = questions.get(job["qid"])
                if question is None or any(job[key] != question[key] for key in ("split", "group_id")):
                    raise ValueError("public identity does not match evaluator identity")
                identity = {key: value for key, value in job.items()
                            if key not in {"job_id", "prompt_sha256"}}
                if hashlib.sha256(_canonical(identity).encode()).hexdigest() != job["job_id"]:
                    raise ValueError("public job ID is not the canonical content hash")
                menu = None
                if job["format"] == "mc":
                    candidates = [m for m in question["menus"]
                                  if (m["condition"], m["menu_id"]) == (job["condition"], job["menu_id"])]
                    if len(candidates) != 1:
                        raise ValueError("public MC menu does not match evaluator")
                    menu = candidates[0]
                if controls:
                    if job["options"] != menu["options"]:
                        raise ValueError("choices-only options differ from evaluator menu")
                    key = (job["qid"], "control", job["condition"])
                else:
                    prefixes = [p for p in question["prefixes"] if p["prefix_id"] == job["prefix_id"]]
                    if len(prefixes) != 1 or prefixes[0]["fraction"] != job["fraction"]:
                        raise ValueError("public prefix does not match evaluator")
                    expected_prompt = _prompt(prefixes[0]["text"], menu, config["prompt_template"])
                    if job["prompt"] != expected_prompt:
                        raise ValueError("public prompt is not exact canonical v4 prompt")
                    key = (job["qid"], job["prefix_id"], job["condition"])
                observed_counts[key] += 1
            encoded = tokenizer.encode_batch([render_single_user(j["prompt"]) for j in batch],
                                             add_special_tokens=False)
            for job, row in zip(batch, encoded, strict=True):
                length = len(row.ids)
                phase_tokens.append(length)
                if length > maximum["input_tokens"]:
                    maximum = {"input_tokens": length, "job_id": job["job_id"],
                               "qid": job["qid"], "condition": job["condition"],
                               "package": filename, "fraction": job.get("fraction")}
                if length > config["generation"]["max_input_tokens"]:
                    oversized.append({"job_id": job["job_id"], "input_tokens": length,
                                      "qid": job["qid"], "package": filename})
        tokens_by_phase[filename] = {"jobs": len(jobs), "min": min(phase_tokens),
                                     "max": max(phase_tokens), "total": sum(phase_tokens),
                                     "mean": sum(phase_tokens) / len(phase_tokens)}
        files.append({"path": filename, "sha256": file_hash(path),
                      "bytes": path.stat().st_size, "jobs": len(jobs)})
        del package, jobs
    if any(count != 1 for count in observed_counts.values()):
        raise ValueError("repeated question/prefix/arm or choices-only menu")
    expected_keys = set()
    for question in questions.values():
        for prefix in question["prefixes"]:
            for arm in config["arms"]:
                expected_keys.add((question["qid"], prefix["prefix_id"], arm))
        for arm in config["arms"][1:]:
            expected_keys.add((question["qid"], "control", arm))
    if set(observed_counts) != expected_keys:
        raise ValueError("public requests do not cover the exact paired design")
    if oversized:
        raise ValueError(json.dumps({"error": "input cap exceeded; no truncation permitted",
                                     "oversized_jobs": len(oversized), "examples": oversized[:20]}))
    return {
        "schema_version": "acl-expansion-preflight-result-v1", "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": "not_run", "new_inference_run": False,
        "gpu_used": False, "network_used": False, "model_weights_loaded": False,
        "dataset_sha256": file_hash(dataset_path), "config_sha256": file_hash(config_path),
        "validator_sha256": file_hash(Path(__file__)), "public_files": files,
        "implementation_files": {name: file_hash(ROOT / name) for name in (
            "qb_data/__init__.py", "qb_data/jane_paired.py", "qb_data/data_loader.py",
            "qb_data/text_utils.py", "scripts/__init__.py", "scripts/jane_gpu_backend.py",
            "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py")},
        "questions": len(questions), "split_counts": dict(splits),
        "distinct_public_jobs_per_model": len(all_job_ids),
        "planned_main_responses_two_models": config["main_jobs_per_model"] * len(config["models"]),
        "planned_control_responses_two_models": config["choices_only_jobs_per_model"] * len(config["models"]),
        "planned_total_responses_two_models": len(all_job_ids) * len(config["models"]),
        "tokenizer": tokenizer_evidence, "token_counts": tokens_by_phase,
        "maximum_prompt": maximum, "oversized_inputs": 0,
        "max_input_tokens": 2048, "max_new_tokens": 160, "inputs_truncated": False,
        "historical_runner_compatible": False,
        "historical_runner_reason": "modal_jane_pilot.py requires exactly 200 main questions, at most 10,000 jobs per package, and a separate historical development package; all historical gates remain unchanged",
        "remaining_launch_requirements": config["launch_requirements"],
        "scope_limits": ["This checks input validity and token lengths, not model output quality",
                         "This does not independently certify semantic answer equivalence or distractor validity",
                         "No generated output length has been observed; 160 is the configured generation cap"],
        "wall_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "acl_expansion_5000.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new receipt path")
    report = validate_inputs(args.dataset, args.public_dir, args.tokenizer_dir, args.config)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "questions": report["questions"],
                      "max_input_tokens_observed": report["maximum_prompt"]["input_tokens"],
                      "execution_status": report["execution_status"]}))


if __name__ == "__main__":
    main()
