"""Bounded, create-once menu-only scoring using the existing ACL public inputs.

CPU staging precedes two GPU calls. No retries or speculative allocations are
permitted. Model caches contain only exact original model bytes and public jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import time

RUN_ID = 'acl5000-option-scores-20261003'
ORIGINAL_RUN = 'acl5000-20261002'
INPUT_HASH = '9db13301d928cd31dc54c97f0c5cfd88b9bc25774ceb56027c032c52e4afb043'
MODELS = {'qwen3b': 'Qwen/Qwen2.5-3B-Instruct', 'qwen7b': 'Qwen/Qwen2.5-7B-Instruct'}
SOURCES = ('scripts/__init__.py', 'scripts/modal_acl_option_scores.py',
           'scripts/acl_option_scoring.py', 'scripts/modal_acl_expansion.py',
           'scripts/jane_gpu_backend.py', 'scripts/jane_qwen_backend.py',
           'scripts/jane_output_constraints.py')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def budget_plan() -> dict:
    gpu_rate = Decimal('0.00063924')
    cpu_rate = Decimal('0.00004396')  # Two cores and 8 GiB, no GPU.
    reserved = gpu_rate * 2 * (1800 + 90 + 2) + cpu_rate * (1200 + 90 + 2) + Decimal('0.40')
    return {'schema': 'acl-option-score-budget-v1', 'ceiling_usd': '3',
            'gpu_rate_usd_per_second': str(gpu_rate), 'cpu_rate_usd_per_second': str(cpu_rate),
            'gpu_function_timeout_seconds': 1800, 'scorer_deadline_seconds': 1650,
            'cpu_function_timeout_seconds': 1200, 'startup_timeout_seconds': 90,
            'scaledown_seconds': 2, 'gpu_calls': 2, 'cpu_calls': 1,
            'contingency_usd': '0.40', 'reserved_estimate_usd': str(reserved),
            'automatic_retries': 0, 'benchmark_jobs_per_model': 256,
            'invoice_verified': False}


def validate_plan(plan: dict) -> None:
    if plan != budget_plan() or Decimal(plan['reserved_estimate_usd']) > Decimal('3'):
        raise ValueError('plan must match the authorized three-dollar allocation')


def safe_output_path(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not name.startswith('output/'):
        raise ValueError('only safe output-relative paths may be collected')
    return name


def verify_sources(root: Path, control: dict) -> None:
    actual = {name: digest(root / name) for name in SOURCES}
    if actual != control['source_files_sha256']:
        raise ValueError('source hashes differ from frozen scoring control')
    validate_plan(control['budget'])


def remote_prepare(control: dict) -> dict:
    """Download and hash-check original pinned weights using CPU allocation."""
    import modal
    from huggingface_hub import snapshot_download
    from scripts.jane_gpu_backend import PINNED_MODELS, validate_package
    from scripts.acl_option_scoring import prepare_context
    from transformers import AutoTokenizer
    verify_sources(Path('/opt/scoring'), control)
    old_volume = modal.Volume.from_name(ORIGINAL_RUN, create_if_missing=False)
    volume = modal.Volume.from_name(RUN_ID, create_if_missing=False)
    old_volume.reload(); volume.reload()
    root = Path('/scores')
    started = time.monotonic()
    write_json(root / 'output' / 'prepare_started.json', {'source_commit': control['source_commit']})
    volume.commit()
    source = Path('/original/public/main_choices_only.json')
    if digest(source) != INPUT_HASH:
        raise ValueError('original choices-only inputs failed frozen hash')
    jobs = json.loads(source.read_bytes())
    validate_package(jobs, max_jobs=10000)
    if len(jobs['jobs']) != 10000:
        raise ValueError('expected all 10000 menus')
    (root / 'public').mkdir(exist_ok=False)
    (root / 'public/main_choices_only.json').write_bytes(source.read_bytes())
    cache = root / 'models'
    cache.mkdir(exist_ok=False)
    receipts = {}
    for tag, model in MODELS.items():
        trace_path = next(iter(sorted(Path('/original/output', tag, 'choices_only').glob('*/trace.json'))))
        completion = json.loads((trace_path.parent / 'completion.json').read_text())
        if digest(trace_path) != completion['trace_sha256']:
            raise ValueError('original model receipt trace failed hash')
        metadata = json.loads(trace_path.read_bytes())['metadata']
        if metadata['model'] != model or metadata['revision'] != PINNED_MODELS[model]:
            raise ValueError('original model identity mismatch')
        expected = metadata['model_files_sha256']
        snapshot = Path(snapshot_download(repo_id=model, revision=PINNED_MODELS[model],
            cache_dir=str(cache), allow_patterns=['*.json', '*.safetensors', '*.txt'], max_workers=4))
        actual = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*'))
                  if p.is_file() and '.cache' not in p.relative_to(snapshot).parts}
        if actual != expected:
            raise ValueError('downloaded weights differ from original run')
        write_json(cache / f'{tag}_expected_model_hashes.json',
                   {'model': model, 'revision': PINNED_MODELS[model], 'model_files_sha256': expected})
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        # Reject a broken scoring boundary before renting either GPU.
        for job in jobs['jobs']:
            prepare_context(tokenizer, job)
        receipts[tag] = {'model': model, 'revision': PINNED_MODELS[model],
                         'model_files_sha256': actual, 'original_trace_sha256': completion['trace_sha256'],
                         'tokenization_preflight_jobs': len(jobs['jobs'])}
    result = {'status': 'completed', 'elapsed_seconds': time.monotonic() - started,
              'input_sha256': INPUT_HASH, 'model_receipts': receipts}
    write_json(root / 'output/prepare_receipt.json', result)
    volume.commit()
    return result


def remote_score(tag: str, control: dict) -> dict:
    import modal
    verify_sources(Path('/opt/scoring'), control)
    if tag not in MODELS:
        raise ValueError('unexpected model')
    volume = modal.Volume.from_name(RUN_ID, create_if_missing=False)
    volume.reload()
    root = Path('/scores')
    if time.time() >= control['gpu_absolute_deadline_unix']:
        raise TimeoutError('shared allocation wall-clock deadline expired')
    # Provider infrastructure replay can occur independently of retries=0.
    # Claim durably before importing torch, hashing weights, or loading a GPU.
    write_json(root / 'output' / f'{tag}_claim.json',
               {'source_commit': control['source_commit'], 'started_unix': time.time()})
    volume.commit()
    from scripts.acl_option_scoring import run_scoring
    if not (root / 'output/prepare_receipt.json').is_file():
        raise ValueError('CPU staging is incomplete')
    def progress(update):
        volume.commit()
        print(json.dumps({'model_tag': tag, **update}), flush=True)
    started = time.monotonic()
    try:
        receipt = run_scoring(tag, root / 'public/main_choices_only.json', root / 'models',
            root / 'output' / tag, max_seconds=control['budget']['scorer_deadline_seconds'],
            batch_size=32, progress=progress)
        return receipt
    finally:
        write_json(root / 'output' / f'{tag}_allocation_receipt.json',
                   {'model_tag': tag, 'elapsed_seconds': time.monotonic() - started,
                    'allocation_rate_usd_per_second': control['budget']['gpu_rate_usd_per_second'],
                    'invoice_verified': False})
        volume.commit()


def collect(volume, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    total = 0
    for entry in volume.iterdir('/output', recursive=True):
        name = str(entry.path).lstrip('/')
        if not name.endswith(('.json', '.jsonl', '.txt')):
            continue
        name = safe_output_path(name)
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as stream:
            for chunk in volume.read_file(name):
                total += len(chunk)
                if total > 100 * 1024**2:
                    raise ValueError('output collection exceeds 100 MiB')
                stream.write(chunk)
        files.append({'path': name, 'bytes': target.stat().st_size, 'sha256': digest(target)})
    report = {'files': files, 'total_bytes': total}
    write_json(out / 'download_manifest.json', report)
    return report


def launch(repo: Path, out: Path, source_commit: str) -> dict:
    from scripts.modal_acl_expansion import connect, verify_source_commit
    modal, workspace = connect()
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('exact source commit required')
    control = {'run_id': RUN_ID, 'original_run_id': ORIGINAL_RUN,
               'source_commit': source_commit, 'input_sha256': INPUT_HASH,
               'source_files_sha256': {name: digest(repo / name) for name in SOURCES},
               'budget': budget_plan(), 'created_utc': datetime.now(timezone.utc).isoformat(),
               'estimand': 'A-D token preference conditional on fixed assistant answer prefix'}
    verify_sources(repo, control)
    verify_source_commit(repo, source_commit, control['source_files_sha256'])
    volume = modal.Volume.from_name(RUN_ID, create_if_missing=True)
    if list(volume.iterdir('/')):
        raise ValueError('scoring volume already initialized; never relaunch automatically')
    with volume.batch_upload(force=False) as upload:
        upload.put_file(io.BytesIO(json.dumps(control).encode()), '/control.json')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'control.json', control)
    write_json(out / 'workspace.json', workspace)
    image = modal.Image.debian_slim(python_version='3.11').pip_install(
        'torch==2.6.0', 'transformers==4.51.3', 'tokenizers==0.21.1', 'safetensors==0.5.3',
        'huggingface-hub==0.30.2', 'accelerate==1.6.0', 'modal==1.6.0',
        'numpy==2.2.4', 'jinja2==3.1.6', 'lm-format-enforcer==0.11.3', 'interegular==0.3.3')
    for name in SOURCES:
        image = image.add_local_file(str(repo / name), remote_path='/opt/scoring/' + name, copy=True)
    image = image.env({'PYTHONPATH': '/opt/scoring', 'PYTHONUNBUFFERED': '1',
                       'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    app = modal.App(RUN_ID, image=image, include_source=False)
    original = modal.Volume.from_name(ORIGINAL_RUN, create_if_missing=False)
    prepare = app.function(cpu=(2, 2), memory=(8192, 8192), timeout=1200, startup_timeout=90,
        max_containers=1, scaledown_window=2, retries=0, include_source=False,
        volumes={'/original': original, '/scores': volume})(remote_prepare)
    scorer = app.function(gpu='L40S', cpu=(2, 2), memory=(32768, 32768), timeout=1800,
        startup_timeout=90, max_containers=2, scaledown_window=2, retries=0,
        include_source=False, volumes={'/scores': volume})(remote_score)
    result = {'run_id': RUN_ID, 'models': {}}
    try:
        with modal.enable_output(), app.run():
            prepare_call = prepare.spawn(control)
            try:
                result['prepare'] = prepare_call.get(timeout=1290)
            except Exception:
                prepare_call.cancel(terminate_containers=True)
                raise
            worker_control = {**control, 'gpu_absolute_deadline_unix': time.time() + 1890}
            write_json(out / 'gpu_allocation_window.json', worker_control)
            calls = {tag: scorer.spawn(tag, worker_control) for tag in MODELS}
            write_json(out / 'calls.json', {tag: call.object_id for tag, call in calls.items()})
            for tag, call in calls.items():
                try:
                    result['models'][tag] = call.get(
                        timeout=max(1, worker_control['gpu_absolute_deadline_unix'] - time.time()))
                except Exception as error:
                    call.cancel(terminate_containers=True)
                    result['models'][tag] = {'status': 'provider_error', 'error_type': type(error).__name__,
                                             'error': str(error)}
    finally:
        collect(volume, out)
        write_json(out / 'launch_result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = launch(Path(__file__).resolve().parents[1], args.out, args.source_commit)
    print(json.dumps(result, sort_keys=True))
    if len(result['models']) != 2 or any(row.get('status') != 'complete' for row in result['models'].values()):
        raise SystemExit('scoring incomplete; inspect preserved evidence before any further allocation')


if __name__ == '__main__':
    main()
