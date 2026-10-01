#!/usr/bin/env python3
"""Optional pinned CPU Qwen backend for public Jane evaluation jobs on stdin.

This generates real model responses, but is not a reproduction of Jane's pilot.
Only public prompts reach the model. Confidence is the model's JSON self-report,
not calibrated correctness or normalized MC-option probability. Install the
optional pinned requirements separately. Unit tests do not load model weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
PUBLIC_KEYS = {'job_id', 'qid', 'group_id', 'split', 'format', 'condition',
               'menu_id', 'prefix_id', 'fraction', 'prompt', 'prompt_sha256'}
SCOPES = {'synthetic_fixture', 'legacy_engineering_control', 'engineering_smoke', 'scientific'}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f'nonfinite JSON number: {value}')


def parse_response(raw: str) -> dict[str, Any]:
    """Parse exact JSON; preserve failures rather than repairing an answer.

    Parameters
    ----------
    raw : str
        Entire decoded generated completion without special end tokens.

    Returns
    -------
    dict
        Answer, status, confidence, and parse_error. Invalid completions always
        have null answer/confidence. No answerline or gold data is accepted.
    """
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
        if not isinstance(value, dict) or set(value) != {'answer', 'confidence', 'status'}:
            raise ValueError('expected exactly answer, confidence, and status')
        if value['status'] == 'abstain':
            if value['answer'] is not None or value['confidence'] is not None:
                raise ValueError('abstention requires null answer and confidence')
        elif value['status'] == 'answer':
            if not isinstance(value['answer'], str) or not value['answer'].strip():
                raise ValueError('answer must be a nonempty string')
            confidence = value['confidence']
            if (isinstance(confidence, bool) or not isinstance(confidence, (float, int))
                    or not 0 <= confidence <= 1 or not math.isfinite(confidence)):
                raise ValueError('answer confidence must be finite and within [0,1]')
        else:
            raise ValueError('unsupported response status')
        return {**value, 'parse_error': None}
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return {'answer': None, 'confidence': None, 'status': 'invalid',
                'parse_error': str(error)}


def validate_jobs(jobs: Any, *, max_jobs: int) -> list[dict[str, Any]]:
    """Validate public-only requests and their prompt hashes before loading ML.

    Parameters
    ----------
    jobs : object
        Expected to be a nonempty list of contract public job dictionaries.
    max_jobs : int
        Maximum number of requests this invocation may execute.

    Returns
    -------
    list of dict
        Validated jobs. Unexpected keys, including gold metadata, fail closed.
    """
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('expected a nonempty public jobs array')
    if len(jobs) > max_jobs:
        raise ValueError(f'job budget exceeded: {len(jobs)} > {max_jobs}')
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError('every job must be an object')
        if set(job) != PUBLIC_KEYS:
            raise ValueError('missing or unexpected public job keys')
        for field in PUBLIC_KEYS - {'fraction', 'menu_id'}:
            if not isinstance(job[field], str) or not job[field].strip():
                raise ValueError(f'{field} must be a nonempty string')
        if job['job_id'] in seen:
            raise ValueError('duplicate job_id')
        seen.add(job['job_id'])
        if job['split'] not in {'calibration', 'selection', 'test'}:
            raise ValueError('unsupported split')
        if job['format'] not in {'mc', 'oe'}:
            raise ValueError('unsupported format')
        if job['format'] == 'oe':
            if job['condition'] != 'oe' or job['menu_id'] is not None:
                raise ValueError('OE requires condition oe and null menu_id')
        elif not isinstance(job['menu_id'], str) or not job['menu_id'].strip():
            raise ValueError('MC requires a nonempty menu_id')
        fraction = job['fraction']
        if (isinstance(fraction, bool) or not isinstance(fraction, (float, int))
                or not 0 < fraction <= 1 or not math.isfinite(fraction)):
            raise ValueError('fraction must be finite and within (0,1]')
        if _sha256(job['prompt'].encode()) != job['prompt_sha256']:
            raise ValueError('prompt hash mismatch')
    return jobs


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def run(jobs: list[dict[str, Any]], args: argparse.Namespace, checkpoint=None) -> dict[str, Any]:
    """Generate one fresh-context completion per validated public job."""
    # Lazy imports keep dependency-free parser and boundary tests usable.
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    backend_hash_start = _file_hash(Path(__file__))
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.revision != MODEL_REVISION:
        raise ValueError('backend supports only the pinned model revision')
    if args.model_path is not None:
        if args.model_receipt is None:
            raise ValueError('--model-path requires --model-receipt for identity verification')
        snapshot = args.model_path.resolve()
        receipt = json.loads(args.model_receipt.read_bytes(), object_pairs_hook=_unique_object)
        if receipt.get('model') != MODEL_ID or receipt.get('revision') != MODEL_REVISION:
            raise ValueError('model receipt identity does not match pinned backend')
        expected = {name: row['sha256'] for name, row in receipt['files'].items()}
        file_hashes = {str(path.relative_to(snapshot)): _file_hash(path)
                       for path in sorted(snapshot.rglob('*'))
                       if path.is_file() and '.cache' not in path.relative_to(snapshot).parts}
        if file_hashes != expected:
            raise ValueError('local model files do not match downloaded-model receipt')
    else:
        snapshot = Path(snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, cache_dir=args.cache_dir,
            local_files_only=not args.allow_download,
            allow_patterns=['*.json', '*.safetensors', '*.txt'],
        ))
        if snapshot.name != MODEL_REVISION:
            raise ValueError('resolved model snapshot does not match pinned revision')
        file_hashes = {str(path.relative_to(snapshot)): _file_hash(path)
                       for path in sorted(snapshot.rglob('*'))
                       if path.is_file() and '.cache' not in path.relative_to(snapshot).parts}
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager',
    ).to('cpu').eval()
    load_seconds = time.perf_counter() - start
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    predictions = []
    for index, job in enumerate(jobs):
        # There is no conversation history or cache reuse between requests.
        messages = [{'role': 'user', 'content': job['prompt']}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        encoded = tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
        input_ids = encoded['input_ids'][0].tolist()
        if len(input_ids) > args.max_input_tokens:
            raise ValueError(f'job {job["job_id"]} exceeds input-token cap; no truncation performed')
        item_start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, generation_config=generation)
        elapsed = time.perf_counter() - item_start
        generated = output[0, len(input_ids):].tolist()
        raw = tokenizer.decode(generated, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        eos_ids = generation.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        stopped_on_eos = bool(generated) and generated[-1] in eos_ids
        predictions.append({
            'job_id': job['job_id'], 'prompt_sha256': job['prompt_sha256'],
            **parse_response(raw), 'raw_response': raw,
            'rendered_prompt_sha256': _sha256(rendered.encode()),
            'input_token_ids': input_ids, 'generated_token_ids': generated,
            'input_tokens': len(input_ids), 'output_tokens': len(generated),
            'generation_seconds': elapsed,
            'finish_reason': 'eos' if stopped_on_eos else 'length',
        })
        if checkpoint is not None:
            checkpoint.write(json.dumps(predictions[-1], ensure_ascii=False, allow_nan=False) + '\n')
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
        print(f'Qwen generated {index + 1}/{len(jobs)} jobs; '
              f'{predictions[-1]["status"]}; {elapsed:.3f}s', file=sys.stderr, flush=True)
    if _file_hash(Path(__file__)) != backend_hash_start:
        raise ValueError('backend source changed during generation; no completed trace')
    return {
        'schema_version': 'jane-traces-v1',
        'metadata': {
            'model': MODEL_ID, 'revision': MODEL_REVISION,
            'confidence_method': 'self_reported_correctness_probability_uncalibrated',
            'context_policy': 'fresh_per_prefix', 'evidence_scope': args.evidence_scope,
            'execution': 'actual_local_cpu_model_generation',
            'device': 'cpu', 'dtype': 'float32', 'attention_implementation': 'eager',
            'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'threads': args.threads,
            'seed': args.seed, 'greedy': True,
            'max_new_tokens': args.max_new_tokens, 'max_input_tokens': args.max_input_tokens,
            'max_jobs': args.max_jobs, 'n_jobs': len(jobs),
            'started_at': started, 'model_load_seconds': load_seconds,
            'total_seconds': time.perf_counter() - start,
            'model_files_sha256': file_hashes,
            'chat_template_sha256': _sha256(tokenizer.chat_template.encode()),
            'backend_sha256': backend_hash_start,
            'backend_sha256_end': _file_hash(Path(__file__)),
            'model_receipt_sha256': (_file_hash(args.model_receipt) if args.model_receipt else None),
            'confidence_scope': 'JSON self-report; not normalized option probability or calibration',
            'parser_policy': 'exact JSON object; malformed responses invalid; no repair',
        },
        'predictions': predictions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-scope', choices=sorted(SCOPES),
                        help='Optional assertion; normally read from the public jobs envelope')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--model-receipt', type=Path)
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--checkpoint', type=Path,
                        help='Create-once JSONL of completed predictions; partial until final stdout succeeds')
    parser.add_argument('--allow-download', action='store_true',
                        help='Permit fetching the pinned public snapshot; default is cached-only')
    parser.add_argument('--max-jobs', type=int, default=72)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--max-input-tokens', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args(argv)
    for field in ('max_jobs', 'max_new_tokens', 'max_input_tokens', 'threads'):
        if getattr(args, field) < 1:
            parser.error(f'--{field.replace("_", "-")} must be positive')
    try:
        raw = sys.stdin.buffer.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError('public jobs JSON exceeds 16 MiB limit')
        package = json.loads(raw, object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
        if (not isinstance(package, dict) or set(package) != {'schema_version', 'evidence_scope', 'jobs'}
                or package['schema_version'] != 'jane-public-jobs-v1'
                or package['evidence_scope'] not in SCOPES):
            raise ValueError('expected a jane-public-jobs-v1 envelope with valid evidence scope')
        if args.evidence_scope is not None and args.evidence_scope != package['evidence_scope']:
            raise ValueError('evidence scope differs from public jobs envelope')
        args.evidence_scope = package['evidence_scope']
        jobs = validate_jobs(package['jobs'], max_jobs=args.max_jobs)
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with args.checkpoint.open('x', encoding='utf-8') as checkpoint:
                result = run(jobs, args, checkpoint=checkpoint)
        else:
            result = run(jobs, args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f'Qwen backend failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
