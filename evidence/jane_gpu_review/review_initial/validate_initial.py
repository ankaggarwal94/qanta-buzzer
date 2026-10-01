#!/usr/bin/env python3
"""Read-only validation of retained initial Jane Modal development evidence.

Requires tokenizers 0.21.1 and Jinja2 3.1.6; no torch, weights, auth, or cloud job.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
from decimal import Decimal

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent
RUN = ROOT.parent / 'initial_run'
REPO = ROOT.parent / 'qanta-buzzer'


def read(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode()


CHECKS = []


def check(name, condition, detail=None):
    CHECKS.append({'check': name, 'passed': bool(condition), 'detail': detail})


def independent_parse(raw):
    """Strict independent schema replay; diagnostics checked with original parser."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(value, dict) or set(value) != {'answer', 'status', 'confidence'}:
            raise ValueError('keys')
        if value['status'] == 'abstain':
            if value['answer'] is not None or value['confidence'] is not None:
                raise ValueError('abstention fields')
        elif value['status'] == 'answer':
            if not isinstance(value['answer'], str) or not value['answer'].strip():
                raise ValueError('answer')
            c = value['confidence']
            if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1:
                raise ValueError('confidence')
        else:
            raise ValueError('status')
        return value
    except (ValueError, TypeError):
        return {'answer': None, 'confidence': None, 'status': 'invalid'}


def main():
    manifest = read(REPO / 'modal_pilot_data/public/manifest.json')
    control = read(RUN / 'submission_control.json')
    frozen = read(RUN / 'frozen_control.json')
    check('frozen_control_identical_except_deadline', frozen == {k: v for k, v in control.items() if k != 'absolute_deadline_unix'})
    files = []
    for record in control['files']:
        raw = (REPO / 'modal_pilot_data/public' / record['path']).read_bytes()
        data = json.loads(raw)
        check('public_file_' + record['path'], sha(raw) == record['sha256'] and len(raw) == record['bytes'] and len(data['jobs']) == record['job_count'])
        files.append(record)
    public_identity = {'schema_version': 'jane-modal-public-inputs-v1', 'files': files}
    check('public_input_identity', sha(canonical(public_identity) + b'\n') == control['public_input_id'])
    remote = read(ROOT / 'remote_sources.json')
    working_copy_source_status = {}
    for path, expected in control['source_files_sha256'].items():
        working_copy_source_status[path] = {'expected_initial_sha256': expected,
                                          'current_local_sha256': sha((REPO / path).read_bytes()),
                                          'matches_initial': sha((REPO / path).read_bytes()) == expected}
        check('remote_commit_source_' + path, sha(remote[path].encode()) == expected)
    parser_source = remote['scripts/jane_qwen_backend.py']
    if sha(parser_source.encode()) != control['source_files_sha256']['scripts/jane_qwen_backend.py']:
        raise ValueError('immutable parser source hash mismatch')
    parser_namespace = {'__name__': 'independent_initial_pinned_parser',
                        '__file__': 'original_launch/scripts/jane_qwen_backend.py'}
    exec(compile(parser_source, parser_namespace['__file__'], 'exec'), parser_namespace)
    parse_response = parser_namespace['parse_response']
    evidence = RUN / 'evidence'
    receipt = read(RUN / 'download_receipt.json')
    listed = {r['path'] for r in receipt['files']}
    actual = {str(p.relative_to(evidence)) for p in evidence.rglob('*') if p.is_file()}
    check('download_exact_inventory', listed == actual)
    check('download_total_bytes', sum(r['bytes'] for r in receipt['files']) == receipt['bytes'])
    for record in receipt['files']:
        raw = (evidence / record['path']).read_bytes()
        check('download_' + record['path'], len(raw) == record['bytes'] and sha(raw) == record['sha256'])
    check('no_main_or_choice_control_outputs', not any('/main/' in n or '/choices_only/' in n for n in actual))
    check('no_throughput_or_remote_failure_receipt', 'throughput_gate.json' not in actual and 'failure.json' not in actual)
    check('authorized_workspace', read(RUN / 'workspace_receipt.json')['workspace_name'] == 'ankaggarwal94')
    alloc = read(RUN / 'allocation_claim.json')
    started = read(evidence / 'gpu_execution_started.json')
    execution = read(evidence / 'execution_receipt.json')
    for name, record in [('allocation', alloc), ('started', started), ('execution', execution)]:
        check(name + '_source_and_input', record['source_commit'] == control['source_commit'] and record['public_input_id'] == control['public_input_id'])
    jobs_package = read(REPO / 'modal_pilot_data/public/dev_jobs.json')
    jobs = jobs_package['jobs']
    tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer/qwen3b/tokenizer.json'))
    models = {}
    for tag, model_name, revision in control['model_pins']:
        trace = read(evidence / tag / 'development/trace.json')
        metadata = trace['metadata']
        predictions = trace['predictions']
        checkpoint = [json.loads(row) for row in (evidence / tag / 'development/predictions.checkpoint.jsonl').read_text().splitlines()]
        completion = read(evidence / tag / 'development/completion.json')
        stored_gate = read(evidence / tag / 'development_gate.json')
        config = read(ROOT / 'tokenizer' / tag / 'tokenizer_config.json')
        generation = read(ROOT / 'tokenizer' / tag / 'generation_config.json')
        hub = read(ROOT / 'tokenizer' / tag / 'hub_model_metadata.json')
        check(tag + '_model_pin', metadata['model'] == model_name and metadata['revision'] == revision and hub['sha'] == revision)
        check(tag + '_gpu_runtime', metadata['execution'] == 'actual_cuda_model_generation' and metadata['device'] == 'cuda:0' and metadata['gpu_name'] == 'NVIDIA L40S' and metadata['dtype'] == 'bfloat16')
        check(tag + '_runtime_versions', metadata['python'] == '3.11.12' and metadata['torch'] == '2.6.0+cu124' and metadata['transformers'] == '4.51.3')
        check(tag + '_source_hashes', metadata['backend_sha256'] == control['source_files_sha256']['scripts/jane_gpu_backend.py'] and metadata['parser_sha256'] == control['source_files_sha256']['scripts/jane_qwen_backend.py'])
        check(tag + '_public_canonical_hash', metadata['public_package_canonical_sha256'] == sha(canonical(jobs_package)))
        check(tag + '_trace_complete_count', len(predictions) == len(jobs) == 177 == metadata['n_jobs'] == completion['jobs'])
        check(tag + '_ordered_coverage', [p['job_id'] for p in predictions] == [j['job_id'] for j in jobs])
        check(tag + '_checkpoint_identical', checkpoint == predictions)
        check(tag + '_completion_hash', completion['trace_sha256'] == sha((evidence / tag / 'development/trace.json').read_bytes()))
        for filename in ['config.json', 'generation_config.json', 'tokenizer_config.json']:
            check(tag + '_official_' + filename, sha((ROOT / 'tokenizer' / tag / filename).read_bytes()) == metadata['model_files_sha256'][filename])
        check(tag + '_official_shared_tokenizer', sha((ROOT / 'tokenizer/qwen3b/tokenizer.json').read_bytes()) == metadata['model_files_sha256']['tokenizer.json'])
        tokenizer_raw = (ROOT / 'tokenizer/qwen3b/tokenizer.json').read_bytes()
        tokenizer_blob = hashlib.sha1(b'blob ' + str(len(tokenizer_raw)).encode() + b'\0' + tokenizer_raw).hexdigest()
        official_tokenizer = next(row for row in hub['siblings'] if row['rfilename'] == 'tokenizer.json')
        check(tag + '_official_tokenizer_git_blob', official_tokenizer['blobId'] == tokenizer_blob)
        weights = {row['rfilename']: row['lfs']['sha256'] for row in hub['siblings'] if row['rfilename'].endswith('.safetensors')}
        check(tag + '_recorded_weight_hashes_vs_hub_lfs', all(metadata['model_files_sha256'].get(name) == value for name, value in weights.items()) and len(weights) == (2 if tag == 'qwen3b' else 4))
        template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(config['chat_template'])
        check(tag + '_chat_template_hash', sha(config['chat_template'].encode()) == metadata['chat_template_sha256'])
        eos_ids = set(generation['eos_token_id'])
        counts = {fmt: collections.Counter() for fmt in ['mc', 'oe']}
        invalid = []
        errors = []
        for index, (job, prediction) in enumerate(zip(jobs, predictions)):
            batch = jobs[index // 8 * 8:index // 8 * 8 + 8]
            rendered = template.render(messages=[{'role': 'user', 'content': job['prompt']}], add_generation_prompt=True)
            input_ids = tokenizer.encode(rendered, add_special_tokens=False).ids
            generated = prediction['generated_token_ids']
            first_eos = next((n for n, token in enumerate(generated) if token in eos_ids), None)
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            parsed = independent_parse(raw)
            replay = parse_response(raw)
            checks = {
                'prompt_sha': sha(job['prompt'].encode()) == prediction['prompt_sha256'] == job['prompt_sha256'],
                'rendered_sha': sha(rendered.encode()) == prediction['rendered_prompt_sha256'],
                'input_token_ids': input_ids == prediction['input_token_ids'],
                'input_token_count': len(input_ids) == prediction['input_tokens'] <= metadata['max_input_tokens'],
                'generated_decoding': raw == prediction['raw_response'],
                'independent_parse': all(parsed[k] == prediction[k] for k in parsed),
                'original_parse_with_error': all(replay[k] == prediction[k] for k in replay),
                'output_token_count': len(generated) == prediction['output_tokens'] <= metadata['max_new_tokens'],
                'effective_output_count': prediction['effective_output_tokens'] == (first_eos + 1 if first_eos is not None else len(generated)),
                'finish_reason': prediction['finish_reason'] == ('eos' if first_eos is not None else 'length'),
                'post_eos_only_padding': first_eos is None or all(t == generation['pad_token_id'] for t in generated[first_eos + 1:]),
                'batch_index': prediction['batch_index'] == index // 8,
                'batch_job_ids': prediction['batch_job_ids'] == [j['job_id'] for j in batch],
            }
            # Independently rebuild every padded batch's common prefix width.
            widths = [len(tokenizer.encode(template.render(messages=[{'role': 'user', 'content': j['prompt']}], add_generation_prompt=True), add_special_tokens=False).ids) for j in batch]
            checks['padded_input_width'] = prediction['batch_padded_input_width'] == max(widths)
            for name, passed in checks.items():
                if not passed:
                    errors.append({'job_id': job['job_id'], 'check': name})
            c = counts[job['format']]
            c['jobs'] += 1
            c[parsed['status']] += 1
            if parsed['status'] != 'invalid':
                c['schema_valid'] += 1
            if parsed['status'] == 'answer' and job['format'] == 'mc' and parsed['answer'] not in set('ABCD'):
                c['illegal_mc_ids'] += 1
            if parsed['status'] == 'answer' and job['format'] == 'oe' and parsed['answer'].strip().lower() in {'...', '…', '[answer]', '<answer>', 'answer', 'your answer', 'answer here', 'your answer here'}:
                c['placeholders'] += 1
            if parsed['status'] == 'invalid':
                invalid.append({'job_id': job['job_id'], 'qid': job['qid'], 'format': job['format'], 'finish_reason': prediction['finish_reason'], 'effective_output_tokens': prediction['effective_output_tokens'], 'raw_response': raw, 'parse_error': replay['parse_error']})
        check(tag + '_all_token_and_parser_rows', not errors, {'rows': len(predictions), 'checks_per_row': 14, 'errors': errors})
        gate_counts = {fmt: {key: counts[fmt][key] for key in ['jobs', 'schema_valid', 'illegal_mc_ids', 'placeholders']} for fmt in ['mc', 'oe']}
        gate_passed = all(c['schema_valid'] / c['jobs'] >= .95 for c in gate_counts.values()) and not gate_counts['mc']['illegal_mc_ids'] and not gate_counts['oe']['placeholders']
        check(tag + '_independent_gate_replay', stored_gate['per_format'] == gate_counts and stored_gate['passed'] == gate_passed)
        check(tag + '_receipt_gate_consistency', execution['models'][tag]['interface_gate'] == stored_gate and execution['models'][tag]['development'] == completion)
        check(tag + '_phase_timing_arithmetic', math.isclose(completion['inference_seconds'] + completion['loading_download_hash_seconds'], completion['backend_total_seconds'], abs_tol=1e-7) and completion['backend_total_seconds'] <= completion['phase_wall_seconds'])
        models[tag] = {'model': model_name, 'revision': revision, 'counts': {fmt: dict(counts[fmt]) for fmt in counts}, 'gate_passed': gate_passed, 'invalid_outputs': invalid, 'token_and_parser_checks': len(predictions) * 14, 'max_input_tokens_observed': max(p['input_tokens'] for p in predictions), 'max_generated_tokens_observed': max(p['output_tokens'] for p in predictions), 'unique_batches': len({p['batch_index'] for p in predictions})}
    host = read(RUN / 'host_receipt.json')
    rate = Decimal(control['budget']['allocation_rate_usd_per_second'])
    elapsed = Decimal(str(execution['elapsed_seconds']))
    session = Decimal(str(host['session_elapsed_seconds']))
    reserve = Decimal(control['budget']['reserve_usd'])
    check('allocation_estimate_arithmetic', rate * elapsed == Decimal(execution['allocation_estimate_usd']))
    check('resource_max_with_reserve', rate * Decimal(control['budget']['max_session_seconds']) + reserve == Decimal(control['budget']['max_allocation_plus_reserve_usd']) <= Decimal(control['budget']['ceiling_usd']) == 10)
    check('status_gate_stop_consistency', execution['status'] == host['status'] == 'DEVELOPMENT_INTERFACE_GATE_FAILED' and host['exception_type'] is None and host['result']['status'] == execution['status'])
    check('attached_no_auto_retry', host['attached'] is True and host['automatic_function_retries'] == 0)
    report = {
        'schema_version': 'jane-independent-initial-validation-v1',
        'source_commit': control['source_commit'],
        'public_input_id': control['public_input_id'],
        'summary': '354 development completions independently replayed; 7B OE interface gate failed; no main or choices-only execution outputs.',
        'all_checks_passed': all(c['passed'] for c in CHECKS),
        'checks': CHECKS,
        'mutable_working_copy_source_status': working_copy_source_status,
        'development_questions': len({j['qid'] for j in jobs}),
        'development_responses': 2 * len(jobs),
        'models': models,
        'budget': {
            'ceiling_usd': control['budget']['ceiling_usd'],
            'remote_elapsed_seconds': str(elapsed),
            'allocation_estimate_usd': str(rate * elapsed),
            'host_session_elapsed_seconds': str(session),
            'conservative_host_elapsed_resource_estimate_usd': str(rate * session),
            'conservative_host_elapsed_plus_full_reserve_usd': str(rate * session + reserve),
            'maximum_prospective_resource_plus_reserve_usd': control['budget']['max_allocation_plus_reserve_usd'],
            'invoice_verified': host['actual_invoice_verified'],
        },
        'limitations': [
            'Model weight hashes are checked against recorded remote hashes and official pinned Hub LFS metadata; full weight files were not downloaded or independently rehashed locally.',
            'GPU and runtime identity are retained executing-code receipts, corroborated by the trusted exact-source run; this audit did not independently access the remote GPU.',
            'Costs are arithmetic estimates from configured resources and runtime, not provider billing receipts or an account-wide hard spend cap.',
            'Development is only 12 questions and lacks science examples; no main results, stopping-policy fit, or headline reproduction can be claimed.',
            '3B OE formatting passes through 59 valid abstentions; schema success is not answer coverage or accuracy.',
            'The current working copy is being prepared for a later candidate; immutable original source was fetched and hash-verified at the original launch commit.',
        ],
        'upheld_flaws': [],
    }
    (ROOT / 'initial_validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ['all_checks_passed', 'development_questions', 'development_responses', 'budget', 'upheld_flaws']}, indent=2))
    print('checks:', len(CHECKS), 'failed:', [c['check'] for c in CHECKS if not c['passed']])
    if not report['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
