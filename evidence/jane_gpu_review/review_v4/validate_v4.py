#!/usr/bin/env python3
"""Independently validate retained v4 GPU evidence, without loading weights.

Usage: review_initial/venv/bin/python review_v4/validate_v4.py --run PATH
Needs tokenizers==0.21.1, Jinja2==3.1.6 and the verified review_initial assets.
The validator never reads evaluator labels, invokes Modal, or performs inference.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import pathlib
from decimal import Decimal

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent
LAUNCH_SHA = 'a2013af95f7fa49f055423a9b398358aaf252c64'
PUBLIC_ID = 'd7d2168092ca6de6f89381f62581426a7405881bae9678a6348cc7334a4171bd'
MODEL_PINS = [('qwen3b', 'Qwen/Qwen2.5-3B-Instruct', 'aa8e72537993ba99e69dfaafa59ed015b17504d1'),
              ('qwen7b', 'Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28')]
PHASE_FILES = {'development': 'dev_jobs.json', 'main': 'main_jobs.json',
               'choices_only': 'main_choices_only.json'}
PHASE_ORDER = [('qwen3b', 'development'), ('qwen7b', 'development'),
               ('qwen3b', 'main'), ('qwen3b', 'choices_only'),
               ('qwen7b', 'main'), ('qwen7b', 'choices_only')]
INPUT_ORDER = ['dev_jobs.json', 'main_jobs.json', 'dev_choices_only.json', 'main_choices_only.json']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value, newline=False):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(',', ':')).encode()
    return raw + b'\n' if newline else raw


def read(path):
    return json.loads(path.read_bytes())


def independent_parse(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate key')
            value[key] = item
        return value
    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(value, dict) or set(value) != {'answer', 'status', 'confidence'}:
            raise ValueError('keys')
        if value['status'] == 'abstain':
            if value['answer'] is not None or value['confidence'] is not None:
                raise ValueError('abstention')
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


def stamp(value):
    return dt.datetime.fromisoformat(value)


class Validator:
    def __init__(self, run, data, assets, prior):
        self.run, self.data, self.assets, self.prior = run, data, assets, prior
        self.checks, self.notes, self.phases = [], [], {}
        self.control = read(run / 'submission_control.json')
        self.host = read(run / 'host_receipt.json')
        self.status = self.host['status']
        self.evidence = run / 'evidence'
        self.execution = read(self.evidence / 'execution_receipt.json') if (self.evidence / 'execution_receipt.json').exists() else None
        self.packages = {name: read(data / 'public' / name) for name in INPUT_ORDER}
        self.remote_sources = read(ROOT / 'remote_sources.json')
        parser_source = self.remote_sources['scripts/jane_qwen_backend.py']
        if sha(parser_source.encode()) != self.control['source_files_sha256']['scripts/jane_qwen_backend.py']:
            raise ValueError('immutable parser hash mismatch; refusing parser execution')
        namespace = {'__name__': 'independent_v4_pinned_parser', '__file__': 'immutable_source/jane_qwen_backend.py'}
        exec(compile(parser_source, namespace['__file__'], 'exec'), namespace)
        self.original_parse = namespace['parse_response']
        self.tokenizer = Tokenizer.from_file(str(assets / 'qwen3b/tokenizer.json'))
        self.tokenizer_config = read(assets / 'qwen3b/tokenizer_config.json')
        self.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(self.tokenizer_config['chat_template'])
        self.input_cache = {}
        for name, package in self.packages.items():
            jobs, cached = package['jobs'], []
            for job in jobs:
                rendered = self.template.render(messages=[{'role': 'user', 'content': job['prompt']}], add_generation_prompt=True)
                ids = self.tokenizer.encode(rendered, add_special_tokens=False).ids
                cached.append({'ids': ids, 'rendered_sha': sha(rendered.encode()), 'prompt_sha': sha(job['prompt'].encode())})
            for start in range(0, len(jobs), 8):
                batch = jobs[start:start + 8]
                width = max(len(c['ids']) for c in cached[start:start + 8])
                for index in range(start, min(start + 8, len(jobs))):
                    cached[index].update({'batch_ids': [j['job_id'] for j in batch], 'batch_width': width, 'batch_index': start // 8})
            self.input_cache[name] = cached

    def check(self, name, condition, detail=None):
        self.checks.append({'check': name, 'passed': bool(condition), 'detail': detail})

    def identities(self):
        c = self.control
        self.check('launch_commit_and_input', c['source_commit'] == LAUNCH_SHA and c['public_input_id'] == PUBLIC_ID)
        self.check('frozen_control_exact', read(self.run / 'frozen_control.json') == {k: v for k, v in c.items() if k != 'absolute_deadline_unix'})
        manifest, files = read(self.data / 'public/manifest.json'), []
        for name in INPUT_ORDER:
            raw = (self.data / 'public' / name).read_bytes()
            record = {'path': name, 'sha256': sha(raw), 'bytes': len(raw), 'job_count': len(self.packages[name]['jobs'])}
            files.append(record)
            self.check('public_manifest_' + name, manifest['files'][name] == {'sha256': record['sha256'], 'byte_count': record['bytes'], 'job_count': record['job_count']})
            self.check('public_schema_' + name, self.packages[name]['schema_version'] == ('jane-choice-controls-v1' if 'choices_only' in name else 'jane-public-jobs-v1'))
        self.check('public_byte_receipt', c['files'] == files)
        self.check('public_input_canonical_identity', sha(canonical({'schema_version': 'jane-modal-public-inputs-v1', 'files': files}, newline=True)) == PUBLIC_ID)
        self.check('model_pins', c['model_pins'] == [list(row) for row in MODEL_PINS])
        for name, expected in c['source_files_sha256'].items():
            self.check('immutable_source_' + name, sha(self.remote_sources[name].encode()) == expected)
        workspace = read(self.run / 'workspace_receipt.json')
        self.check('workspace', workspace['workspace_name'] == 'ankaggarwal94' and bool(workspace['workspace_id']))
        alloc = read(self.run / 'allocation_claim.json')
        provider = read(self.run / 'provider_launch.json')
        self.check('create_once_candidate_volume', alloc['volume_name'] == provider['volume_name'] == 'jane-mcq-pilot-20261001-v4')
        self.check('allocation_identity', alloc['source_commit'] == LAUNCH_SHA and alloc['public_input_id'] == PUBLIC_ID)
        self.check('provider_allocation_ids_present', bool(alloc['volume_id']) and bool(provider['app_id']) and bool(provider['function_call_id']))
        if (self.evidence / 'gpu_execution_started.json').exists():
            started = read(self.evidence / 'gpu_execution_started.json')
            self.check('gpu_start_identity', started['source_commit'] == LAUNCH_SHA and started['public_input_id'] == PUBLIC_ID)
            freeze = read(self.data / 'prompt_candidate_freeze.json')
            self.check('prospective_candidate_freeze', freeze['candidate_public_input_id'] == PUBLIC_ID and stamp(freeze['frozen_utc']) <= stamp(started['started_utc']))
        elif self.status == 'COMPLETED':
            self.check('gpu_start_required_for_completion', False)
        if self.execution:
            self.check('execution_identity', self.execution['source_commit'] == LAUNCH_SHA and self.execution['public_input_id'] == PUBLIC_ID and self.execution['model_pins'] == c['model_pins'])
            self.check('execution_host_status', self.execution['status'] == self.status)
            if self.host['result'] is not None:
                self.check('execution_models_vs_host_result', self.execution['models'] == self.host['result']['models'])
        elif self.status != 'FAILED':
            self.check('execution_receipt_required_for_nonfailure', False)
        else:
            self.notes.append('Remote receipt absent under FAILED status; phase completion and remote cost cannot be fully established.')
        self.check('attached_no_automatic_retry', self.host['attached'] is True and self.host['automatic_function_retries'] == 0)
        self.check('host_budget', self.host['budget'] == c['budget'])
        if self.execution:
            self.check('execution_budget', self.execution['budget'] == c['budget'])

    def inventory(self):
        actual = {str(p.relative_to(self.evidence)) for p in self.evidence.rglob('*') if p.is_file()}
        allowed = {'gpu_execution_started.json', 'execution_receipt.json', 'failure.json', 'throughput_gate.json'}
        for tag, _, _ in MODEL_PINS:
            allowed.add(tag + '/development_gate.json')
            for phase in PHASE_FILES:
                for name in ['trace.json', 'completion.json', 'predictions.checkpoint.jsonl']:
                    allowed.add(tag + '/' + phase + '/' + name)
        self.check('evidence_path_allowlist', actual <= allowed, sorted(actual - allowed))
        if self.status == 'COMPLETED':
            self.check('completion_no_download_failure', not (self.run / 'download_failure.json').exists())
        self.check('no_reserved_dev_controls_executed', not any('dev_choices_only' in name for name in actual))
        if (self.run / 'download_receipt.json').exists():
            receipt = read(self.run / 'download_receipt.json')
            self.check('download_exact_inventory', {r['path'] for r in receipt['files']} == actual)
            self.check('download_total_bytes', receipt['bytes'] == sum(r['bytes'] for r in receipt['files']))
            for record in receipt['files']:
                raw = (self.evidence / record['path']).read_bytes()
                self.check('download_' + record['path'], sha(raw) == record['sha256'] and len(raw) == record['bytes'])
        elif self.status != 'FAILED':
            self.check('download_receipt_required', False)
        else:
            self.notes.append('Download receipt absent under FAILED status; hash inventory cannot be fully established.')
        self.inventory_paths = actual

    def row_replay(self, tag, phase, rows, complete):
        name = PHASE_FILES[phase]
        jobs, cache = self.packages[name]['jobs'], self.input_cache[name]
        generation = read(self.assets / tag / 'generation_config.json')
        eos_ids = set(generation['eos_token_id'])
        self.check(tag + '_' + phase + '_coverage', len(rows) <= len(jobs) and [p['job_id'] for p in rows] == [j['job_id'] for j in jobs[:len(rows)]] and (not complete or len(rows) == len(jobs)))
        counts = {fmt: collections.Counter() for fmt in ['mc', 'oe']}
        errors, invalid, outputs = [], [], collections.Counter()
        for index, (job, cached, row) in enumerate(zip(jobs, cache, rows)):
            generated = row['generated_token_ids']
            raw = self.tokenizer.decode(generated, skip_special_tokens=True)
            parsed, original = independent_parse(raw), self.original_parse(raw)
            first_eos = next((n for n, token in enumerate(generated) if token in eos_ids), None)
            tests = {
                'prompt_sha': row['prompt_sha256'] == job['prompt_sha256'] == cached['prompt_sha'],
                'rendered_sha': row['rendered_prompt_sha256'] == cached['rendered_sha'],
                'input_ids': row['input_token_ids'] == cached['ids'],
                'input_count': row['input_tokens'] == len(cached['ids']) <= 2048,
                'raw_decoding': row['raw_response'] == raw,
                'independent_schema': all(row[k] == parsed[k] for k in parsed),
                'original_parser_error': all(row[k] == original[k] for k in original),
                'generated_count': row['output_tokens'] == len(generated) <= 96,
                'effective_tokens': row['effective_output_tokens'] == (first_eos + 1 if first_eos is not None else len(generated)),
                'finish': row['finish_reason'] == ('eos' if first_eos is not None else 'length'),
                'post_eos_padding': first_eos is None or all(t == generation['pad_token_id'] for t in generated[first_eos + 1:]),
                'batch_index': row['batch_index'] == cached['batch_index'],
                'batch_jobs': row['batch_job_ids'] == cached['batch_ids'],
                'padded_width': row['batch_padded_input_width'] == cached['batch_width'],
                'batch_seconds': isinstance(row['batch_generation_seconds'], (float, int)) and math.isfinite(row['batch_generation_seconds']) and row['batch_generation_seconds'] >= 0,
            }
            errors += [{'job_id': job['job_id'], 'check': key} for key, ok in tests.items() if not ok]
            c = counts[job['format']]
            c['jobs'] += 1
            c[parsed['status']] += 1
            c['schema_valid'] += parsed['status'] != 'invalid'
            if parsed['status'] == 'answer' and job['format'] == 'mc' and parsed['answer'] not in set('ABCD'):
                c['illegal_mc_ids'] += 1
            if parsed['status'] == 'answer' and job['format'] == 'oe' and parsed['answer'].strip().lower() in {'...', '…', '[answer]', '<answer>', 'answer', 'your answer', 'answer here', 'your answer here'}:
                c['placeholders'] += 1
            outputs[row['finish_reason']] += 1
            if parsed['status'] == 'invalid':
                invalid.append({'job_id': job['job_id'], 'format': job['format'], 'finish_reason': row['finish_reason'], 'raw_response': raw, 'parse_error': original['parse_error']})
        self.check(tag + '_' + phase + '_all_token_parser_checks', not errors, {'rows': len(rows), 'checks_per_row': 15, 'errors': errors})
        return {'complete': complete, 'rows': len(rows), 'expected_rows': len(jobs), 'token_parser_checks': 15 * len(rows),
                'counts': {fmt: dict(c) for fmt, c in counts.items()}, 'finish_reasons': dict(outputs), 'invalid_outputs': invalid}

    def metadata(self, tag, model, revision, phase, trace, completion):
        m = trace['metadata']
        prefix = tag + '_' + phase
        expected_schema = 'jane-choice-control-traces-v1' if phase == 'choices_only' else 'jane-traces-v1'
        self.check(prefix + '_separate_trace_schema', trace['schema_version'] == expected_schema)
        self.check(prefix + '_model_pin', m['model'] == model and m['revision'] == revision)
        self.check(prefix + '_gpu_runtime', m['execution'] == 'actual_cuda_model_generation' and m['device'] == 'cuda:0' and m['gpu_name'] == 'NVIDIA L40S' and m['dtype'] == 'bfloat16')
        self.check(prefix + '_versions', m['python'] == '3.11.12' and m['torch'] == '2.6.0+cu124' and m['transformers'] == '4.51.3')
        self.check(prefix + '_source_hashes', m['backend_sha256'] == self.control['source_files_sha256']['scripts/jane_gpu_backend.py'] and m['parser_sha256'] == self.control['source_files_sha256']['scripts/jane_qwen_backend.py'])
        self.check(prefix + '_fixed_generation', m['seed'] == 1 and m['greedy'] is True and m['batch_size'] == 8 and m['padding_side'] == 'left' and m['max_input_tokens'] == 2048 and m['max_new_tokens'] == 96 and m['allow_tf32'] is False and m['deterministic_algorithms'] is True)
        self.check(prefix + '_confidence_scope', m['confidence_method'] == 'self_reported_correctness_probability_uncalibrated' and m['context_policy'] == 'fresh_per_prefix')
        package = self.packages[PHASE_FILES[phase]]
        self.check(prefix + '_public_hash', m['public_package_canonical_sha256'] == sha(canonical(package)))
        self.check(prefix + '_job_count', m['n_jobs'] == len(package['jobs']))
        self.check(prefix + '_chat_template', m['chat_template_sha256'] == sha(self.tokenizer_config['chat_template'].encode()))
        hub = read(self.assets / tag / 'hub_model_metadata.json')
        self.check(prefix + '_hub_pin', hub['sha'] == revision)
        for name in ['config.json', 'generation_config.json', 'tokenizer_config.json']:
            self.check(prefix + '_official_' + name, m['model_files_sha256'][name] == sha((self.assets / tag / name).read_bytes()))
        tok = (self.assets / 'qwen3b/tokenizer.json').read_bytes()
        official = next(r for r in hub['siblings'] if r['rfilename'] == 'tokenizer.json')
        self.check(prefix + '_official_tokenizer', m['model_files_sha256']['tokenizer.json'] == sha(tok) and official['blobId'] == hashlib.sha1(b'blob ' + str(len(tok)).encode() + b'\0' + tok).hexdigest())
        weights = {r['rfilename']: r['lfs']['sha256'] for r in hub['siblings'] if r['rfilename'].endswith('.safetensors')}
        self.check(prefix + '_weight_hashes_vs_official_lfs', all(m['model_files_sha256'].get(n) == h for n, h in weights.items()) and len(weights) == (2 if tag == 'qwen3b' else 4))
        if completion:
            self.check(prefix + '_completion_count', completion['jobs'] == len(package['jobs']))
            self.check(prefix + '_phase_timing', math.isclose(completion['inference_seconds'] + completion['loading_download_hash_seconds'], completion['backend_total_seconds'], abs_tol=1e-7) and completion['backend_total_seconds'] == m['total_seconds'] and completion['loading_download_hash_seconds'] == m['model_load_seconds'] and completion['backend_total_seconds'] <= completion['phase_wall_seconds'])
            if self.execution and phase in self.execution['models'].get(tag, {}):
                self.check(prefix + '_execution_phase_receipt', completion == self.execution['models'][tag][phase])

    def phase_evidence(self):
        self.traces, self.completions = {}, {}
        for tag, model, revision in MODEL_PINS:
            for phase in PHASE_FILES:
                path = self.evidence / tag / phase
                if not path.exists():
                    continue
                checkpoint_path, trace_path = path / 'predictions.checkpoint.jsonl', path / 'trace.json'
                rows = [json.loads(line) for line in checkpoint_path.read_text().splitlines()] if checkpoint_path.exists() else None
                trace = read(trace_path) if trace_path.exists() else None
                completion = read(path / 'completion.json') if (path / 'completion.json').exists() else None
                if trace is not None:
                    self.check(tag + '_' + phase + '_checkpoint_identical', rows == trace['predictions'])
                    self.check(tag + '_' + phase + '_completion_hash', completion is not None and completion['trace_sha256'] == sha(trace_path.read_bytes()))
                    rows = trace['predictions']
                    self.metadata(tag, model, revision, phase, trace, completion)
                    self.traces[(tag, phase)] = trace
                elif completion:
                    self.check(tag + '_' + phase + '_no_completion_without_trace', False)
                if rows is not None:
                    self.phases[(tag, phase)] = self.row_replay(tag, phase, rows, trace is not None)
                else:
                    self.notes.append(tag + '/' + phase + ' directory retained without replayable rows.')
                if completion:
                    self.completions[(tag, phase)] = completion
        present = [p for p in PHASE_ORDER if p in self.phases]
        # A sequential attached execution can only produce a prefix of phases.
        self.check('sequential_phase_inventory', present == PHASE_ORDER[:len(present)])
        for index, key in enumerate(present):
            if index + 1 < len(present):
                self.check('_'.join(key) + '_complete_before_next_phase', self.phases[key]['complete'])
        dated = [(key, self.traces[key]) for key in PHASE_ORDER if key in self.traces]
        for (old_key, old), (new_key, new) in zip(dated, dated[1:]):
            elapsed = (stamp(new['metadata']['started_at']) - stamp(old['metadata']['started_at'])).total_seconds()
            self.check('_'.join(new_key) + '_starts_after_prior_phase', elapsed + 1 >= old['metadata']['total_seconds'])

    def gates_and_status(self):
        self.gates = {}
        for tag, _, _ in MODEL_PINS:
            path = self.evidence / tag / 'development_gate.json'
            if not path.exists():
                continue
            gate = read(path)
            phase = self.phases.get((tag, 'development'))
            self.check(tag + '_gate_requires_complete_dev', phase is not None and phase['complete'])
            if not phase:
                continue
            counts = {fmt: {k: phase['counts'][fmt].get(k, 0) for k in ['jobs', 'schema_valid', 'illegal_mc_ids', 'placeholders']} for fmt in ['mc', 'oe']}
            passed = all(c['jobs'] and c['schema_valid'] / c['jobs'] >= .95 for c in counts.values()) and not counts['mc']['illegal_mc_ids'] and not counts['oe']['placeholders']
            self.check(tag + '_independent_interface_gate', gate['passed'] == passed and gate['per_format'] == counts and gate['uses_gold_accuracy'] is False)
            if self.execution and tag in self.execution['models']:
                self.check(tag + '_gate_receipt', gate == self.execution['models'][tag]['interface_gate'])
            self.gates[tag] = gate
        both_pass = len(self.gates) == 2 and all(g['passed'] for g in self.gates.values())
        main_present = any(phase != 'development' for tag, phase in self.phases)
        if ('qwen7b', 'development') in self.phases or 'qwen7b' in self.gates:
            self.check('7b_development_only_after_3b_gate', self.gates.get('qwen3b', {}).get('passed') is True)
        throughput_path = self.evidence / 'throughput_gate.json'
        throughput = read(throughput_path) if throughput_path.exists() else None
        if throughput:
            self.check('throughput_only_after_both_dev_gates', both_pass)
            expected, components = 1800.0, []
            for tag, _, _ in MODEL_PINS:
                measured = self.completions[(tag, 'development')]
                seconds_per_job = measured['inference_seconds'] / measured['jobs']
                jobs = len(self.packages['main_jobs.json']['jobs']) + len(self.packages['main_choices_only.json']['jobs'])
                generation = 3 * seconds_per_job * jobs
                expected += generation
                components.append({'model': tag, 'main_and_control_jobs': jobs, 'development_seconds_per_job': seconds_per_job, 'buffered_generation_seconds': generation})
            self.check('throughput_projection_replay', math.isclose(expected, throughput['predicted_seconds_with_load_reserve'], abs_tol=1e-7) and throughput['components'] == components and throughput['generation_multiplier'] == 3 and throughput['per_model_load_reserve_seconds'] == 900)
            self.check('throughput_pass_replay', throughput['passed'] == (expected <= throughput['remaining_seconds'] - 120))
        if main_present:
            self.check('main_eligible_after_both_gates', both_pass and throughput is not None and throughput['passed'] is True)
        if self.status == 'COMPLETED':
            self.check('completion_exact_six_phases', len(self.phases) == 6 and all(p['complete'] for p in self.phases.values()) and sum(p['rows'] for p in self.phases.values()) == 6866)
            self.check('completion_all_gates', both_pass and throughput is not None and throughput['passed'] is True)
            self.check('completion_no_remote_failure', 'failure.json' not in self.inventory_paths and self.host['exception_type'] is None)
        elif self.status == 'DEVELOPMENT_INTERFACE_GATE_FAILED':
            failed = [tag for tag, gate in self.gates.items() if not gate['passed']]
            self.check('interface_failure_boundary', len(failed) == 1 and not main_present and throughput is None and not any(not p['complete'] for p in self.phases.values()))
            if failed == ['qwen3b']:
                self.check('3b_failure_stops_before_7b', ('qwen7b', 'development') not in self.phases)
        elif self.status == 'DEVELOPMENT_THROUGHPUT_GATE_FAILED':
            self.check('throughput_failure_boundary', both_pass and throughput is not None and throughput['passed'] is False and not main_present)
        elif self.status == 'FAILED':
            self.notes.append('Runtime failure: only retained complete phases and valid checkpoint rows are covered by this audit.')
        else:
            self.check('known_terminal_status', False, self.status)

    def budget(self):
        b = self.control['budget']
        rate = Decimal(b['allocation_rate_usd_per_second'])
        max_seconds, reserve = Decimal(b['max_session_seconds']), Decimal(b['reserve_usd'])
        prior = Decimal(b['prior_reserved_usd'])
        self.check('budget_fixed_candidate_ceiling', Decimal(b['ceiling_usd']) == Decimal('7.74') and prior == Decimal('2.26') and rate == Decimal('0.00063924') and reserve == 2 and max_seconds == 8979)
        self.check('budget_cumulative_initial_ceiling', Decimal(b['cumulative_initial_ceiling_usd']) == 10 and Decimal(b['ceiling_usd']) + prior == 10)
        maximum = rate * max_seconds + reserve
        self.check('budget_maximum_arithmetic', maximum == Decimal(b['max_allocation_plus_reserve_usd']) == Decimal('7.73973596') and maximum <= Decimal(b['ceiling_usd']))
        self.check('budget_cumulative_maximum', maximum + prior == Decimal(b['cumulative_max_estimate_plus_reserves_usd']) == Decimal('9.99973596') <= 10)
        prior_raw = (self.prior / 'host_receipt.json').read_bytes()
        prior_host = json.loads(prior_raw)
        self.check('prior_ledger_identity', sha(prior_raw) == b['prior_host_receipt_sha256'] == 'b48aabe8b521ba9f0d8d7fd9ae1e89706146109c1c7be54983bcc8feaa79d423' and b['prior_run_id'] == 36918829142 and b['prior_source_commit'] == 'd4105ca1f3a705611c1e65076723838ee67cc0df')
        prior_estimate = rate * Decimal(str(prior_host['session_elapsed_seconds']))
        self.check('prior_reservation_covers_duration_and_reserve', Decimal(b['prior_host_session_seconds']) == Decimal(str(prior_host['session_elapsed_seconds'])) and prior_estimate + reserve <= prior)
        host_estimate = rate * Decimal(str(self.host['session_elapsed_seconds']))
        remote_estimate = None
        if self.execution:
            elapsed = Decimal(str(self.execution['elapsed_seconds']))
            remote_estimate = rate * elapsed
            self.check('actual_remote_estimate_arithmetic', remote_estimate == Decimal(self.execution['allocation_estimate_usd']))
            self.check('remote_duration_within_session', 0 <= elapsed <= max_seconds)
        self.check('invoice_caveat_retained', self.host['actual_invoice_verified'] is False)
        prior_remote = read(self.prior / 'evidence/execution_receipt.json') if (self.prior / 'evidence/execution_receipt.json').exists() else None
        combined_remote_estimate = (Decimal(prior_remote['allocation_estimate_usd']) + remote_estimate) if prior_remote and remote_estimate is not None else None
        self.budget_report = {'initial_ceiling_usd': '10', 'prior_reserved_usd': str(prior), 'candidate_ceiling_usd': b['ceiling_usd'],
                              'maximum_cumulative_resource_plus_reserves_usd': str(maximum + prior),
                              'candidate_remote_resource_estimate_usd': str(remote_estimate) if remote_estimate is not None else None,
                              'candidate_entire_host_resource_estimate_usd': str(host_estimate),
                              'combined_remote_resource_estimate_usd': str(combined_remote_estimate) if combined_remote_estimate is not None else None,
                              'combined_entire_host_resource_estimate_usd': str(prior_estimate + host_estimate),
                              'conservative_observed_cumulative_host_plus_both_full_reserves_usd': str(prior_estimate + host_estimate + 2 * reserve),
                              'invoice_verified': False}

    def run_validation(self):
        self.identities()
        self.inventory()
        self.phase_evidence()
        self.gates_and_status()
        self.budget()
        failures = [c for c in self.checks if not c['passed']]
        phases = {tag + '/' + phase: result for (tag, phase), result in self.phases.items()}
        return {'schema_version': 'jane-independent-v4-validation-v1', 'source_commit': LAUNCH_SHA, 'public_input_id': PUBLIC_ID,
                'status': self.status, 'all_available_evidence_checks_passed': not failures,
                'checks': self.checks, 'upheld_flaws': failures, 'notes': self.notes, 'phases': phases,
                'retained_responses': sum(p['rows'] for p in self.phases.values()),
                'token_parser_row_checks': sum(p['token_parser_checks'] for p in self.phases.values()),
                'development_gates': self.gates, 'budget': self.budget_report,
                'limitations': ['No accuracy, grading, calibration, or stopping-policy conclusions are evaluated by this validator.',
                                'Recorded weight hashes are compared with official pinned Hub LFS metadata; full weights were not downloaded locally.',
                                'GPU identity is executing-code metadata corroborated by immutable trusted source, not an independent remote hardware inspection.',
                                'Spend estimates are not provider invoices or an account-wide hard spend cap.',
                                'Incomplete runtime output establishes only the retained boundary; missing phases are not inferred to have completed.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=pathlib.Path, required=True)
    parser.add_argument('--data', type=pathlib.Path, default=ROOT.parent / 'qanta-buzzer/modal_pilot_v4_data')
    parser.add_argument('--assets', type=pathlib.Path, default=ROOT.parent / 'review_initial/tokenizer')
    parser.add_argument('--prior', type=pathlib.Path, default=ROOT.parent / 'initial_run')
    parser.add_argument('--out', type=pathlib.Path, default=ROOT / 'v4_validation.json')
    args = parser.parse_args()
    report = Validator(args.run, args.data, args.assets, args.prior).run_validation()
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'all_available_evidence_checks_passed', 'retained_responses', 'token_parser_row_checks', 'budget', 'upheld_flaws']}, indent=2))
    print('Aggregate checks:', len(report['checks']))
    if report['upheld_flaws']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
