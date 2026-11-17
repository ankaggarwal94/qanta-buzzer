"""Small pinned CPU inference backend consuming only exported paired public jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from qb_data.acl_protocol import digest, publish_json
from scripts.acl_paired_inference import IMCQA_REVISION, JANE_REVISION
from scripts.jane_output_constraints import (
    OutputConstraints, constraint_provenance, validate_constrained_completion,
)
from scripts.stopdff_v5.identity import load_json_strict, sha256_file


MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
BOUNDARY = '{"action":"'


def validate_public(package):
    if set(package) != {"schema", "jobs"} or package["schema"] != "acl-paired-public-v1":
        raise ValueError("expected paired public job package")
    ids = set()
    for job in package["jobs"]:
        identity_keys = ("qid", "group_id", "split", "prefix_index", "arm", "template_revision",
                         "prompt_sha256")
        if set(job) != {*identity_keys, "job_id", "prompt", "round"}:
            raise ValueError("unexpected public job fields")
        if (digest({k: job[k] for k in identity_keys}) != job["job_id"] or
                hashlib.sha256(job["prompt"].encode()).hexdigest() != job["prompt_sha256"] or
                job["job_id"] in ids):
            raise ValueError("public job identity mismatch")
        ids.add(job["job_id"])
        arm = job["arm"]
        expected = IMCQA_REVISION if arm in ("forced", "wait") else JANE_REVISION
        if arm not in ("mc", "oe", "forced", "wait") or job["template_revision"] != expected:
            raise ValueError("unsupported public template")
        if arm == "mc":
            payload = json.loads(job["prompt"].rsplit("\n\n", 1)[1])
            if [opt["id"] for opt in payload["options"]] != list("ABCD"):
                raise ValueError("the pinned paired CPU generation grammar requires K=4")
    if not ids:
        raise ValueError("empty public package")


def infer(package, *, local_only=False, max_new_tokens=128, threads=2):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validate_public(package)
    if min(threads, max_new_tokens) < 1:
        raise ValueError("positive CPU threads and generation cap required")
    started = perf_counter()
    snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=local_only,
                                 allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    torch.set_num_threads(threads)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                                                  torch_dtype=torch.float32, attn_implementation="eager").eval()
    eos = tokenizer.eos_token_id
    constraints = OutputConstraints(tokenizer, {eos})
    responses = []
    ready = perf_counter()
    with torch.inference_mode():
        for job in package["jobs"]:
            arm = job["arm"]
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": job["prompt"]}],
                                                     tokenize=False, add_generation_prompt=True)
            action_job = arm in ("forced", "wait")
            context = rendered + BOUNDARY if action_job else rendered
            inputs = tokenizer(context, add_special_tokens=False, return_tensors="pt")
            length = inputs.input_ids.shape[1]
            required = length + (1 if action_job else max_new_tokens)
            if required > model.config.max_position_embeddings:
                raise ValueError("prompt exceeds the pinned model context")
            if action_job:
                actions = "ABCDE" if arm == "wait" else "ABCD"
                token_ids = []
                for label in actions:
                    tokenized = tokenizer.encode(context + label, add_special_tokens=False)
                    if tokenized[:-1] != inputs.input_ids[0].tolist():
                        raise ValueError("candidate label must add exactly one boundary token")
                    token_ids.append(tokenized[-1])
                logits = model(**inputs, logits_to_keep=1).logits[0, -1, token_ids]
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite action logits")
                response = {"action": actions[int(logits.argmax())]}
                diagnostics = {"legal_logits": dict(zip(actions, logits.tolist()))}
            else:
                output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                        pad_token_id=eos, eos_token_id=eos,
                                        prefix_allowed_tokens_fn=constraints.for_batch([arm]))
                tokens = output[0, length:].tolist()
                finished = bool(tokens) and tokens[-1] == eos
                raw = tokenizer.decode(tokens[:-1] if finished else tokens,
                                       skip_special_tokens=False, clean_up_tokenization_spaces=False)
                diagnostics = {"raw_response": raw, "completion_token_ids": tokens}
                try:
                    validate_constrained_completion(raw, arm, finished_eos=finished)
                except ValueError as error:
                    response = None
                    diagnostics["invalid_reason"] = str(error)
                else:
                    response = json.loads(raw)
            responses.append({"job_id": job["job_id"], "prompt_sha256": job["prompt_sha256"],
                              "response": response, **diagnostics})
    finished_at = perf_counter()
    model_files = {p.relative_to(snapshot).as_posix(): sha256_file(p)
                   for p in sorted(Path(snapshot).rglob("*")) if p.is_file()}
    return {"schema": "acl-paired-responses-v1", "model": MODEL, "revision": REVISION,
            "model_files": model_files, "backend_sha256": sha256_file(__file__),
            "timing_seconds": {"load_and_constraints": ready-started,
                               "inference": finished_at-ready},
            "public_sha256": digest(package), "threads": threads, "dtype": "cpu-fp32",
            "context_policy": "fresh_per_prefix", "max_new_tokens": max_new_tokens,
            "grammar": constraint_provenance(), "imcqa_boundary": BOUNDARY,
            "packages": {p: version(p) for p in ("transformers", "torch", "tokenizers")},
            "responses": responses}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    result = infer(load_json_strict(args.public), local_only=args.local_only,
                   threads=args.threads, max_new_tokens=args.max_new_tokens)
    publish_json(args.out, result)


if __name__ == "__main__":
    main()
