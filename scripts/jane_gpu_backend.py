#!/usr/bin/env python3
"""Pinned CUDA-only backend for Jane's new exploratory paired pilot.

The model receives public prompts only. This adapter does not reproduce the
historical pilot. Confidence is a JSON self-report, not option probability.
Model-stack imports are lazy so boundary and parser tests need no GPU stack.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, TextIO

try:
    from scripts.jane_qwen_backend import (
        SCOPES, _reject_constant, _unique_object, parse_response, validate_jobs,
    )
    from scripts.jane_output_constraints import (
        OutputConstraints, constraint_provenance, validate_constrained_completion,
        verify_dependencies,
    )
except ModuleNotFoundError:
    # Direct execution places scripts/, rather than the repo root, on sys.path.
    from jane_qwen_backend import (
        SCOPES, _reject_constant, _unique_object, parse_response, validate_jobs,
    )
    from jane_output_constraints import (
        OutputConstraints, constraint_provenance, validate_constrained_completion,
        verify_dependencies,
    )


# Resolved from the official public model APIs on 2026-10-01. These are new
# prospective identities, not recovered identities for Jane's original pilot.
PINNED_MODELS = {
    'Qwen/Qwen2.5-3B-Instruct': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
    'Qwen/Qwen2.5-7B-Instruct': 'a09a35458c702b33eeacc393d103063234e8bc28',
}
CONTROL_KEYS = {
    'job_id', 'qid', 'group_id', 'split', 'format', 'condition', 'menu_id',
    'options', 'prompt', 'prompt_sha256',
}
CONFIDENCE_METHOD = 'self_reported_correctness_probability_uncalibrated'
PLACEHOLDERS = {'...', '…', '[answer]', '<answer>', 'your answer', 'answer here'}


@dataclass(frozen=True)
class GPUConfig:
    """Immutable execution limits and one pinned model identity."""

    model: str
    revision: str
    batch_size: int = 8
    max_jobs: int = 4000
    max_input_tokens: int = 2048
    max_new_tokens: int = 160
    max_elapsed_seconds: float = 1800.0
    seed: int = 1
    threads: int = 4
    cache_dir: Path | None = None
    model_path: Path | None = None
    model_receipt: Path | None = None
    allow_download: bool = False


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _driver_info() -> dict[str, str | None]:
    """Query the actual driver without failing an otherwise usable CUDA stack."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5, check=False,
        )
        lines = result.stdout.strip().splitlines()
        if result.returncode == 0 and len(lines) == 1 and lines[0].strip():
            return {'driver_version': lines[0].strip(), 'driver_query_error': None}
        return {'driver_version': None,
                'driver_query_error': f'nvidia-smi returned status {result.returncode}, {len(lines)} rows'}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'driver_version': None, 'driver_query_error': f'{type(error).__name__}: {error}'}


def _options(options: Any) -> list[dict[str, str]]:
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError('choices-only controls require four options')
    texts = set()
    for expected, option in zip('ABCD', options):
        if not isinstance(option, dict) or set(option) != {'id', 'text'}:
            raise ValueError('options require exactly id and text, without gold metadata')
        text = option['text']
        if (option['id'] != expected or not isinstance(text, str) or not text.strip()
                or text != text.strip() or any(c in text for c in '\r\n\t')):
            raise ValueError('options must be ordered A/B/C/D with nonempty single-line text')
        if text.casefold() in texts:
            raise ValueError('duplicate option text')
        texts.add(text.casefold())
    return options


def build_choice_control_prompt(options: list[dict[str, str]]) -> str:
    """Build a deterministic menu-only prompt with no answer example.

    Parameters
    ----------
    options : list of dict
        Public A/B/C/D option IDs and texts; no gold designation.

    Returns
    -------
    str
        Exact choices-only control prompt accepted by the boundary validator.
    """
    _options(options)
    menu = '\n'.join(f'{option["id"]}. {option["text"]}' for option in options)
    return (
        'Choose the option most likely to be the answer without seeing the question.\n\n'
        f'Options:\n{menu}\n\n'
        'Return exactly one JSON object with the keys answer, confidence, and status. '
        'Use an option ID as answer and "answer" as status. Confidence is your '
        'probability that this answer is correct, as a number from 0 to 1. '
        'If you decline to answer, use null for answer and confidence, and '
        '"abstain" for status. Do not add markdown or explanation.'
    )


def validate_control_jobs(jobs: Any, *, max_jobs: int) -> list[dict[str, Any]]:
    """Validate a separate menu-only control schema; reject hidden gold fields."""
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('expected a nonempty choices-only jobs array')
    if len(jobs) > max_jobs:
        raise ValueError('job budget exceeded')
    seen = set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) != CONTROL_KEYS:
            raise ValueError('missing or unexpected choices-only public keys')
        for field in CONTROL_KEYS - {'options'}:
            if not isinstance(job[field], str) or not job[field].strip():
                raise ValueError(f'{field} must be a nonempty string')
        if job['job_id'] in seen:
            raise ValueError('duplicate job_id')
        seen.add(job['job_id'])
        if job['split'] not in {'calibration', 'selection', 'test'} or job['format'] != 'mc':
            raise ValueError('unsupported choices-only split or format')
        expected_prompt = build_choice_control_prompt(job['options'])
        if job['prompt'] != expected_prompt:
            raise ValueError('choices-only prompt must contain only the canonical menu template')
        if _sha256(job['prompt'].encode()) != job['prompt_sha256']:
            raise ValueError('prompt hash mismatch')
    return jobs


def validate_package(package: Any, *, max_jobs: int) -> list[dict[str, Any]]:
    """Validate exactly one public main or choices-only envelope."""
    if (not isinstance(package, dict)
            or set(package) != {'schema_version', 'evidence_scope', 'jobs'}
            or not isinstance(package['evidence_scope'], str)
            or package['evidence_scope'] not in SCOPES):
        raise ValueError('expected public envelope with schema_version, evidence_scope, jobs')
    schema = package['schema_version']
    if schema == 'jane-public-jobs-v1':
        return validate_jobs(package['jobs'], max_jobs=max_jobs)
    if schema == 'jane-choice-controls-v1':
        return validate_control_jobs(package['jobs'], max_jobs=max_jobs)
    raise ValueError('unsupported public jobs schema')


def validate_config(config: GPUConfig) -> None:
    """Reject unsupported identities and unbounded/invalid execution settings."""
    if (not isinstance(config.model, str) or not isinstance(config.revision, str)
            or PINNED_MODELS.get(config.model) != config.revision):
        raise ValueError('model and revision must match a pinned prospective identity')
    for field in ('batch_size', 'max_jobs', 'max_input_tokens', 'max_new_tokens', 'threads'):
        value = getattr(config, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{field} must be a positive integer')
    if (isinstance(config.seed, bool) or not isinstance(config.seed, int)
            or not 0 <= config.seed < 2 ** 32):
        raise ValueError('seed must be an integer in [0, 2**32)')
    seconds = config.max_elapsed_seconds
    try:
        valid_seconds = (not isinstance(seconds, bool) and isinstance(seconds, (float, int))
                         and math.isfinite(seconds) and seconds > 0)
    except OverflowError:
        valid_seconds = False
    if not valid_seconds:
        raise ValueError('max_elapsed_seconds must be finite and positive')
    if config.model_path is not None and config.model_receipt is None:
        raise ValueError('model_path requires an identity-bound model_receipt')
    if config.model_receipt is not None and config.model_path is None:
        raise ValueError('model_receipt requires model_path')


def check_deadline(started: float, seconds: float, *, now: float | None = None) -> None:
    """Fail after the elapsed budget; an external timeout bounds blocking calls."""
    if (time.monotonic() if now is None else now) - started >= seconds:
        raise TimeoutError('GPU execution elapsed-time budget exhausted')


def slice_generated_tokens(output_ids: list[list[int]], padded_input_width: int) -> list[list[int]]:
    """Cut generation at the shared padded width, preserving per-row token IDs."""
    if (isinstance(padded_input_width, bool) or not isinstance(padded_input_width, int)
            or padded_input_width < 1):
        raise ValueError('padded input width must be positive')
    if any(len(row) < padded_input_width for row in output_ids):
        raise ValueError('model output is shorter than its padded input')
    return [row[padded_input_width:] for row in output_ids]


def validate_prediction_coverage(jobs: list[dict[str, Any]], predictions: Any) -> None:
    """Require one hash-bound prediction per input; incomplete traces cannot finish."""
    if not isinstance(predictions, list) or len(predictions) != len(jobs):
        raise ValueError('prediction coverage differs from public jobs')
    expected = {job['job_id']: job['prompt_sha256'] for job in jobs}
    seen = set()
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError('prediction must be an object')
        job_id = prediction.get('job_id')
        if job_id not in expected or job_id in seen:
            raise ValueError('extra or duplicate prediction job_id')
        seen.add(job_id)
        if prediction.get('prompt_sha256') != expected[job_id]:
            raise ValueError('prediction prompt hash mismatch')


def interface_diagnostics(jobs: list[dict[str, Any]], trace: dict[str, Any]) -> dict[str, Any]:
    """Count interface failures separately from correctness, without reading gold."""
    predictions = trace['predictions']
    validate_prediction_coverage(jobs, predictions)
    by_id = {row['job_id']: row for row in predictions}
    report: dict[str, Any] = {}
    for fmt in ('mc', 'oe'):
        counts = {key: 0 for key in ('total', 'schema_valid', 'answered', 'abstained',
                                     'parse_invalid', 'illegal_option_id', 'placeholder')}
        for job in jobs:
            if job['format'] != fmt:
                continue
            row = by_id[job['job_id']]
            parsed = parse_response(row['raw_response'])
            # Check recorded parsed fields instead of trusting a claimed status.
            if any(row.get(key) != parsed[key] for key in ('answer', 'confidence', 'status')):
                raise ValueError('recorded parsed response differs from exact raw response')
            counts['total'] += 1
            if parsed['status'] == 'invalid':
                counts['parse_invalid'] += 1
                continue
            counts['schema_valid'] += 1
            if parsed['status'] == 'abstain':
                counts['abstained'] += 1
                continue
            counts['answered'] += 1
            answer = parsed['answer']
            if fmt == 'mc':
                allowed = {option['id'] for option in job.get('options', [])} or set('ABCD')
                counts['illegal_option_id'] += answer not in allowed
            if answer.strip().casefold() in PLACEHOLDERS:
                counts['placeholder'] += 1
        counts['schema_valid_fraction'] = (counts['schema_valid'] / counts['total']
                                            if counts['total'] else None)
        report[fmt] = counts
    return report


def run(package: dict[str, Any], config: GPUConfig, checkpoint: TextIO | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Generate bounded, batched CUDA completions from public prompts only.

    Parameters
    ----------
    package : dict
        Validated public main jobs or choices-only controls envelope.
    config : GPUConfig
        Pinned model identity and explicit execution limits.
    checkpoint : text stream, optional
        New caller-owned JSONL stream. Each completed batch is flushed/fsynced.
    progress : callable, optional
        Receives batch completion counts and elapsed seconds, without secrets.

    Returns
    -------
    dict
        Complete trace. Any exception leaves only partial checkpoint evidence.
    """
    validate_config(config)
    jobs = validate_package(package, max_jobs=config.max_jobs)
    verify_dependencies()
    constraint_metadata = constraint_provenance()
    started_clock = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    source_hash = _file_hash(Path(__file__))
    parser_path = Path(sys.modules[parse_response.__module__].__file__)
    parser_hash = _file_hash(parser_path)
    input_hash = _sha256(json.dumps(package, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode())
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in {':4096:8', ':16:8'}:
        raise ValueError('unsupported CUBLAS deterministic workspace configuration')

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one CUDA GPU is required; no CPU fallback')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU must support BF16; no dtype fallback')
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check_deadline(started_clock, config.max_elapsed_seconds)
    if config.model_path is not None:
        snapshot = Path(config.model_path).resolve()
        receipt = json.loads(Path(config.model_receipt).read_bytes(),
                             object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if receipt.get('model') != config.model or receipt.get('revision') != config.revision:
            raise ValueError('model receipt identity mismatch')
        expected_hashes = {name: row['sha256'] for name, row in receipt['files'].items()}
    else:
        snapshot = Path(snapshot_download(
            repo_id=config.model, revision=config.revision, cache_dir=config.cache_dir,
            local_files_only=not config.allow_download,
            allow_patterns=['*.json', '*.safetensors', '*.txt'],
        ))
        if snapshot.name != config.revision:
            raise ValueError('downloaded model snapshot does not match pinned revision')
        expected_hashes = None
    check_deadline(started_clock, config.max_elapsed_seconds)
    file_hashes = {}
    for path in sorted(snapshot.rglob('*')):
        if path.is_file() and '.cache' not in path.relative_to(snapshot).parts:
            check_deadline(started_clock, config.max_elapsed_seconds)
            file_hashes[str(path.relative_to(snapshot))] = _file_hash(path)
    if expected_hashes is not None and file_hashes != expected_hashes:
        raise ValueError('local model files do not match model receipt')
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                              trust_remote_code=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('tokenizer has no usable padding token')
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation='eager',
    ).to('cuda:0').eval()
    check_deadline(started_clock, config.max_elapsed_seconds)
    load_seconds = time.monotonic() - started_clock
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=config.max_new_tokens,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=True,
    )
    eos_ids = generation.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    constraints = OutputConstraints(tokenizer, eos_ids)
    constraint_metadata['tokenizer_eos_token_id'] = tokenizer.eos_token_id
    predictions = []
    for offset in range(0, len(jobs), config.batch_size):
        check_deadline(started_clock, config.max_elapsed_seconds)
        batch = jobs[offset:offset + config.batch_size]
        rendered = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': job['prompt']}], tokenize=False,
            add_generation_prompt=True,
        ) for job in batch]
        input_rows = [tokenizer(text, add_special_tokens=False)['input_ids'] for text in rendered]
        if any(len(ids) > config.max_input_tokens for ids in input_rows):
            raise ValueError('input-token cap exceeded; no truncation performed')
        encoded = tokenizer(rendered, return_tensors='pt', add_special_tokens=False,
                            padding=True, truncation=False)
        padded_width = encoded['input_ids'].shape[1]
        encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
        torch.cuda.synchronize()
        batch_start = time.monotonic()
        prefix_allowed_tokens_fn = constraints.for_batch([job['format'] for job in batch])
        with torch.inference_mode():
            output = model.generate(**encoded, generation_config=generation,
                                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        torch.cuda.synchronize()
        batch_seconds = time.monotonic() - batch_start
        generated_rows = slice_generated_tokens(output.tolist(), padded_width)
        if len(generated_rows) != len(batch):
            raise ValueError('model generated the wrong batch cardinality')
        constraint_errors = []
        for job, text, input_ids, generated in zip(batch, rendered, input_rows, generated_rows):
            # generate pads completed rows to batch width. Preserve raw IDs but
            # mark the first EOS rather than mistaking trailing padding for EOS.
            first_eos = next((index for index, token in enumerate(generated)
                              if token in eos_ids), None)
            raw = tokenizer.decode(generated, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
            prediction = {
                'job_id': job['job_id'], 'prompt_sha256': job['prompt_sha256'],
                **parse_response(raw), 'raw_response': raw,
                'rendered_prompt_sha256': _sha256(text.encode()),
                'input_token_ids': input_ids, 'generated_token_ids': generated,
                'input_tokens': len(input_ids), 'output_tokens': len(generated),
                'effective_output_tokens': first_eos + 1 if first_eos is not None else len(generated),
                'finish_reason': 'eos' if first_eos is not None else 'length',
                'constraint_schema_version': constraint_metadata['schema_version'],
                'constraint_format': job['format'],
                'constraint_grammar_sha256': constraint_metadata['grammars'][job['format']]['sha256'],
                'batch_index': offset // config.batch_size,
                'batch_job_ids': [item['job_id'] for item in batch],
                'batch_padded_input_width': padded_width, 'batch_generation_seconds': batch_seconds,
            }
            predictions.append(prediction)
            if checkpoint is not None:
                checkpoint.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + '\n')
            try:
                validate_constrained_completion(
                    raw, job['format'],
                    finished_eos=first_eos is not None and generated[first_eos] == tokenizer.eos_token_id,
                )
                if prediction['status'] == 'invalid':
                    raise ValueError('constrained completion rejected by strict response parser')
            except ValueError as error:
                constraint_errors.append(f'{job["job_id"]}: {error}')
        if checkpoint is not None:
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
        if constraint_errors:
            # Preserve exact failed generations in the checkpoint, then stop.
            # No resampling, rewriting, permissive parsing, or test-driven repair.
            raise ValueError('; '.join(constraint_errors))
        update = {'completed_jobs': len(predictions), 'total_jobs': len(jobs),
                  'batch_seconds': batch_seconds, 'elapsed_seconds': time.monotonic() - started_clock}
        if progress is not None:
            progress(update)
        else:
            print(json.dumps(update, sort_keys=True), file=sys.stderr, flush=True)
        check_deadline(started_clock, config.max_elapsed_seconds)
    validate_prediction_coverage(jobs, predictions)
    if _file_hash(Path(__file__)) != source_hash or _file_hash(parser_path) != parser_hash:
        raise ValueError('backend/parser source changed during generation')
    if constraint_provenance()['source_sha256'] != constraint_metadata['source_sha256']:
        raise ValueError('output constraint source changed during generation')
    final_input_hash = _sha256(json.dumps(package, sort_keys=True, ensure_ascii=False,
                                          allow_nan=False, separators=(',', ':')).encode())
    if final_input_hash != input_hash:
        raise ValueError('public jobs mutated during generation')
    trace = {
        'schema_version': ('jane-traces-v1' if package['schema_version'] == 'jane-public-jobs-v1'
                           else 'jane-choice-control-traces-v1'),
        'metadata': {
            'model': config.model, 'revision': config.revision,
            'confidence_method': CONFIDENCE_METHOD,
            'confidence_scope': 'JSON self-report; not normalized option probability or calibration',
            'context_policy': 'fresh_per_prefix', 'evidence_scope': package['evidence_scope'],
            'execution': 'actual_cuda_model_generation', 'device': 'cuda:0',
            'gpu_name': torch.cuda.get_device_name(0), 'cuda_runtime': torch.version.cuda,
            'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'gpu_compute_capability': list(torch.cuda.get_device_capability(0)),
            **_driver_info(),
            'dtype': 'bfloat16', 'attention_implementation': 'eager',
            'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'threads': config.threads,
            'seed': config.seed, 'greedy': True, 'batch_size': config.batch_size,
            'padding_side': 'left', 'deterministic_algorithms': True,
            'allow_tf32': False, 'cublas_workspace_config': workspace,
            'max_new_tokens': config.max_new_tokens, 'max_input_tokens': config.max_input_tokens,
            'max_jobs': config.max_jobs, 'n_jobs': len(jobs),
            'max_elapsed_seconds': config.max_elapsed_seconds, 'started_at': started_at,
            'model_load_seconds': load_seconds, 'total_seconds': time.monotonic() - started_clock,
            'model_files_sha256': file_hashes, 'chat_template_sha256': _sha256(tokenizer.chat_template.encode()),
            'backend_sha256': source_hash, 'parser_sha256': parser_hash,
            'output_constraints': constraint_metadata,
            'public_package_canonical_sha256': input_hash,
            'model_receipt_sha256': _file_hash(Path(config.model_receipt)) if config.model_receipt else None,
            'parser_policy': 'exact JSON object; malformed responses invalid; no repair',
            'retry_policy': 'none; failure leaves partial checkpoint only',
        },
        'predictions': predictions,
    }
    # No GPU tensors/model instances escape in the JSON trace. Release model
    # weights and final batch tensors before another phase loads its model.
    del output, encoded, model, generation
    torch.cuda.empty_cache()
    return trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(PINNED_MODELS), required=True)
    parser.add_argument('--revision', required=True)
    for name, default in [('batch_size', 8), ('max_jobs', 4000), ('max_input_tokens', 2048),
                          ('max_new_tokens', 160), ('threads', 4), ('seed', 1)]:
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    parser.add_argument('--max-elapsed-seconds', type=float, default=1800.0)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--model-receipt', type=Path)
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='New JSONL path. Partial evidence until full trace succeeds.')
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.buffer.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError('public JSON envelope exceeds 32 MiB input cap')
        package = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        config = GPUConfig(**{key: value for key, value in vars(args).items() if key != 'checkpoint'})
        validate_config(config)
        validate_package(package, max_jobs=config.max_jobs)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with args.checkpoint.open('x', encoding='utf-8') as checkpoint:
            trace = run(package, config, checkpoint)
        print(json.dumps(trace, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, ImportError, RuntimeError, TimeoutError) as error:
        print(f'GPU backend failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
