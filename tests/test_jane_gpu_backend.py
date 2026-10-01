"""Public-boundary and batched CUDA-contract tests without model downloads."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import jane_gpu_backend as gpu


def config(**updates):
    model = 'Qwen/Qwen2.5-3B-Instruct'
    return replace(gpu.GPUConfig(model=model, revision=gpu.PINNED_MODELS[model]), **updates)


def main_job(**updates):
    prompt = 'Please answer the question.'
    job = {'job_id': 'job1', 'qid': 'q1', 'group_id': 'g1', 'split': 'test',
           'format': 'oe', 'condition': 'oe', 'menu_id': None, 'prefix_id': 'p1',
           'fraction': 0.5, 'prompt': prompt,
           'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    job.update(updates)
    if 'prompt' in updates and 'prompt_sha256' not in updates:
        job['prompt_sha256'] = hashlib.sha256(job['prompt'].encode()).hexdigest()
    return job


def choices():
    return [{'id': letter, 'text': text} for letter, text in
            zip('ABCD', ('France', 'Spain', 'Italy', 'Germany'))]


def control_job(**updates):
    options = choices()
    prompt = gpu.build_choice_control_prompt(options)
    job = {'job_id': 'c1', 'qid': 'q1', 'group_id': 'g1', 'split': 'test',
           'format': 'mc', 'condition': 'independent_pool', 'menu_id': 'm1',
           'options': options, 'prompt': prompt,
           'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    job.update(updates)
    return job


def package(jobs=None, *, schema='jane-public-jobs-v1'):
    return {'schema_version': schema, 'evidence_scope': 'engineering_smoke',
            'jobs': jobs if jobs is not None else [main_job()]}


def prediction(job, raw='{"answer":"Paris","confidence":0.7,"status":"answer"}'):
    return {'job_id': job['job_id'], 'prompt_sha256': job['prompt_sha256'],
            'raw_response': raw, **gpu.parse_response(raw)}


def test_import_is_lazy_and_identity_is_exact():
    assert gpu.PINNED_MODELS == {
        'Qwen/Qwen2.5-3B-Instruct': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
        'Qwen/Qwen2.5-7B-Instruct': 'a09a35458c702b33eeacc393d103063234e8bc28',
    }
    gpu.validate_config(config())
    for model, revision in gpu.PINNED_MODELS.items():
        gpu.validate_config(config(model=model, revision=revision))


@pytest.mark.parametrize('updates', [
    {'model': 'Qwen/Unknown'}, {'revision': 'main'}, {'model': None},
    {'batch_size': 0}, {'batch_size': True}, {'max_jobs': -1}, {'threads': 1.5},
    {'max_input_tokens': 0}, {'max_new_tokens': 0}, {'seed': True}, {'seed': -1},
    {'seed': 2 ** 32}, {'max_elapsed_seconds': float('nan')},
    {'max_elapsed_seconds': float('inf')}, {'max_elapsed_seconds': True},
    {'max_elapsed_seconds': 10 ** 400}, {'max_elapsed_seconds': 0},
    {'model_path': Path('/missing')}, {'model_receipt': Path('/missing')},
])
def test_invalid_config_fails_before_stack_import(updates):
    with pytest.raises(ValueError):
        gpu.validate_config(config(**updates))


def test_menu_only_prompt_is_deterministic_and_contains_no_placeholder():
    prompt = gpu.build_choice_control_prompt(choices())
    assert prompt == gpu.build_choice_control_prompt(choices())
    assert 'A. France\nB. Spain\nC. Italy\nD. Germany' in prompt
    assert '...' not in prompt and 'gold' not in prompt
    assert gpu.validate_control_jobs([control_job()], max_jobs=1)


@pytest.mark.parametrize('options', [
    [], choices()[:3], choices() + [{'id': 'E', 'text': 'Japan'}],
    [{'id': 'Z', 'text': 'France'}] + choices()[1:],
    [{'id': 'A', 'text': 'France', 'gold': True}] + choices()[1:],
    [{'id': 'A', 'text': 'France\nQuestion clue'}] + choices()[1:],
    [{'id': 'A', 'text': ' France'}] + choices()[1:],
    [{'id': 'A', 'text': ' '} ] + choices()[1:],
    [choices()[0], {'id': 'B', 'text': 'france'}] + choices()[2:],
])
def test_bad_options_fail(options):
    with pytest.raises(ValueError):
        gpu.build_choice_control_prompt(options)


@pytest.mark.parametrize('updates', [
    {'gold_option_id': 'A'}, {'question': 'A clue'}, {'fraction': 0},
    {'prefix_id': 'p0'}, {'split': 'train'}, {'format': 'oe'},
    {'job_id': ''}, {'prompt_sha256': 'wrong'},
    {'prompt': 'Question clue\n' + gpu.build_choice_control_prompt(choices())},
])
def test_control_boundary_rejects_hidden_metadata_or_prompt_changes(updates):
    with pytest.raises(ValueError):
        gpu.validate_control_jobs([control_job(**updates)], max_jobs=1)


def test_both_envelopes_and_job_limits():
    assert gpu.validate_package(package(), max_jobs=1)
    assert gpu.validate_package(package([control_job()], schema='jane-choice-controls-v1'), max_jobs=1)
    with pytest.raises(ValueError, match='duplicate'):
        gpu.validate_control_jobs([control_job(), control_job()], max_jobs=2)
    with pytest.raises(ValueError, match='budget'):
        gpu.validate_control_jobs([control_job()], max_jobs=0)
    with pytest.raises(ValueError, match='unexpected'):
        gpu.validate_package(package([main_job(gold='Paris')]), max_jobs=1)
    for bad in ({}, package(schema='unknown'), {**package(), 'dataset': 'secret'},
                {**package(), 'evidence_scope': []}):
        with pytest.raises(ValueError):
            gpu.validate_package(bad, max_jobs=1)


def test_elapsed_budget_fails_at_boundary():
    gpu.check_deadline(10, 5, now=14.99)
    for now in (15, 99):
        with pytest.raises(TimeoutError):
            gpu.check_deadline(10, 5, now=now)


def test_batch_slice_uses_common_padded_width():
    # First row is left padded and shorter before padding: cutting at its
    # unpadded length (2) would leak two input IDs into the completion.
    rows = [[0, 0, 11, 12, 31, 99, 0], [21, 22, 23, 24, 41, 42, 99]]
    assert gpu.slice_generated_tokens(rows, 4) == [[31, 99, 0], [41, 42, 99]]
    with pytest.raises(ValueError):
        gpu.slice_generated_tokens(rows, 20)


@pytest.mark.parametrize('width', [0, -1, True, 2.5])
def test_invalid_padded_width(width):
    with pytest.raises(ValueError):
        gpu.slice_generated_tokens([[1, 2, 3]], width)


def test_complete_prediction_coverage_is_hash_bound():
    jobs = [main_job(), main_job(job_id='job2')]
    valid = [prediction(job) for job in jobs]
    gpu.validate_prediction_coverage(jobs, valid)
    for bad in (valid[:1], [valid[0], valid[0]],
                [valid[0], {**valid[1], 'job_id': 'extra'}],
                [valid[0], {**valid[1], 'prompt_sha256': 'wrong'}]):
        with pytest.raises(ValueError):
            gpu.validate_prediction_coverage(jobs, bad)


def test_interface_diagnostics_do_not_require_gold_or_repair():
    jobs = [main_job(), main_job(job_id='j2'),
            main_job(job_id='j3', format='mc', condition='pool', menu_id='m1'),
            main_job(job_id='j4', format='mc', condition='pool', menu_id='m1')]
    raw = [
        '{"answer":"...","confidence":0.7,"status":"answer"}',
        'not JSON', '{"answer":"Z","confidence":0.9,"status":"answer"}',
        '{"answer":null,"confidence":null,"status":"abstain"}',
    ]
    trace = {'predictions': [prediction(job, value) for job, value in zip(jobs, raw)]}
    diagnostics = gpu.interface_diagnostics(jobs, trace)
    assert diagnostics['oe']['placeholder'] == 1
    assert diagnostics['oe']['parse_invalid'] == 1
    assert diagnostics['oe']['schema_valid_fraction'] == 0.5
    assert diagnostics['mc']['illegal_option_id'] == 1
    assert diagnostics['mc']['abstained'] == 1
    trace['predictions'][0]['answer'] = 'silently repaired'
    with pytest.raises(ValueError, match='differs'):
        gpu.interface_diagnostics(jobs, trace)


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def to(self, device):
        assert device == 'cuda:0'
        return self

    def tolist(self):
        return self.rows


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 99
    chat_template = 'fake chat template'

    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        return 'chat:' + messages[0]['content']

    def __call__(self, texts, **kwargs):
        def ids(text):
            return list(range(1, len(text) % 6 + 2))
        if isinstance(texts, str):
            return {'input_ids': ids(texts)}
        assert self.padding_side == 'left' and kwargs['truncation'] is False
        rows = [ids(text) for text in texts]
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row for row in rows]
        mask = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {'input_ids': Tensor(padded), 'attention_mask': Tensor(mask)}

    def decode(self, ids, **kwargs):
        assert ids in ([100, 99, 0], [101, 99, 0]), 'input-token or padding slicing bug'
        answer = 'Paris' if ids[0] == 100 else 'A'
        return '{"answer":"' + answer + '","confidence":0.7,"status":"answer"}'


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def install_fake_stack(monkeypatch, *, cuda=True, count=1, bf16=True, generate_error=False):
    calls = []

    class Model:
        generation_config = SimpleNamespace(eos_token_id=99)

        def to(self, device):
            assert device == 'cuda:0'
            return self

        def eval(self):
            return self

        def generate(self, input_ids, attention_mask, generation_config, prefix_allowed_tokens_fn):
            calls.append((input_ids.rows, attention_mask.rows, generation_config))
            if generate_error:
                raise RuntimeError('mock CUDA out of memory')
            return Tensor([row + [prefix_allowed_tokens_fn(index, row)[0], 99, 0]
                           for index, row in enumerate(input_ids.rows)])

    fake_torch = SimpleNamespace(
        __version__='fake-torch', bfloat16='bf16', version=SimpleNamespace(cuda='fake-cuda'),
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: count,
                             is_bf16_supported=lambda: bf16, manual_seed_all=lambda seed: None,
                             synchronize=lambda: None, get_device_name=lambda index: 'fake GPU',
                             get_device_properties=lambda index: SimpleNamespace(total_memory=48 * 2**30),
                             get_device_capability=lambda index: (8, 9), empty_cache=lambda: None),
        set_num_threads=lambda threads: None, manual_seed=lambda seed: None,
        use_deterministic_algorithms=lambda flag: None, inference_mode=Context,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()),
                                 cudnn=SimpleNamespace()),
    )

    def load_model(path, **kwargs):
        assert kwargs['torch_dtype'] == 'bf16'
        assert kwargs['trust_remote_code'] is False
        assert kwargs['local_files_only'] is True
        assert kwargs['attn_implementation'] == 'eager'
        return Model()

    fake_transformers = SimpleNamespace(
        __version__='fake-transformers', GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
    )
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda **kwargs: pytest.fail('test must not download weights')))
    monkeypatch.setattr(gpu, '_driver_info', lambda: {'driver_version': 'fake-driver', 'driver_query_error': None})
    class FakeConstraints:
        def __init__(self, tokenizer, eos_ids):
            assert eos_ids == {99}

        def for_batch(self, formats):
            return lambda index, row: [100 if formats[index] == 'oe' else 101]

    monkeypatch.setattr(gpu, 'verify_dependencies', lambda: None)
    monkeypatch.setattr(gpu, 'OutputConstraints', FakeConstraints)
    return calls


def local_model(tmp_path):
    snapshot = tmp_path / 'model'
    snapshot.mkdir()
    (snapshot / 'config.json').write_text('{}')
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({
        'model': config().model, 'revision': config().revision,
        'files': {'config.json': {'sha256': hashlib.sha256(b'{}').hexdigest()}},
    }))
    return config(model_path=snapshot, model_receipt=receipt)


def test_mocked_gpu_run_retains_tokens_checkpoint_and_complete_metadata(monkeypatch, tmp_path):
    calls = install_fake_stack(monkeypatch)
    cfg = replace(local_model(tmp_path), batch_size=2)
    jobs = [main_job(), main_job(job_id='job2', prompt='A longer question for a different input width.')]
    updates = []
    checkpoint_path = tmp_path / 'new.jsonl'
    with checkpoint_path.open('x') as checkpoint:
        trace = gpu.run(package(jobs), cfg, checkpoint, updates.append)
    assert len(calls) == 1 and len(updates) == 1
    assert updates[0]['completed_jobs'] == 2
    assert trace['schema_version'] == 'jane-traces-v1'
    assert trace['metadata']['dtype'] == 'bfloat16'
    assert trace['metadata']['context_policy'] == 'fresh_per_prefix'
    assert trace['metadata']['model'] == cfg.model
    assert trace['metadata']['gpu_total_memory_bytes'] == 48 * 2**30
    assert trace['metadata']['gpu_compute_capability'] == [8, 9]
    assert trace['metadata']['driver_version'] == 'fake-driver'
    assert trace['metadata']['backend_sha256'] == gpu._file_hash(Path(gpu.__file__))
    assert trace['metadata']['output_constraints']['schema_version'] == 'jane-constrained-json-v1'
    assert trace['metadata']['output_constraints']['posthoc_repair'] is False
    assert trace['metadata']['output_constraints']['tokenizer_eos_token_id'] == 99
    rows = trace['predictions']
    assert rows[0]['generated_token_ids'] == [100, 99, 0]
    assert rows[0]['effective_output_tokens'] == 2
    assert rows[0]['finish_reason'] == 'eos'
    assert rows[0]['constraint_format'] == 'oe'
    assert rows[0]['constraint_grammar_sha256'] == trace['metadata']['output_constraints']['grammars']['oe']['sha256']
    assert rows[0]['input_tokens'] != rows[1]['input_tokens']
    assert rows[0]['batch_padded_input_width'] == max(row['input_tokens'] for row in rows)
    assert [json.loads(line) for line in checkpoint_path.read_text().splitlines()] == rows


@pytest.mark.parametrize('settings', [{'cuda': False}, {'count': 2}, {'bf16': False}])
def test_cuda_or_bf16_failure_never_falls_back(monkeypatch, settings):
    calls = install_fake_stack(monkeypatch, **settings)
    with pytest.raises(RuntimeError):
        gpu.run(package(), config())
    assert not calls


def test_token_cap_fails_without_generation_or_truncation(monkeypatch, tmp_path):
    calls = install_fake_stack(monkeypatch)
    with pytest.raises(ValueError, match='no truncation'):
        gpu.run(package(), replace(local_model(tmp_path), max_input_tokens=1))
    assert not calls


def test_cuda_generation_error_is_not_retried(monkeypatch, tmp_path):
    calls = install_fake_stack(monkeypatch, generate_error=True)
    with pytest.raises(RuntimeError, match='out of memory'):
        gpu.run(package(), local_model(tmp_path))
    assert len(calls) == 1


def test_wrong_local_receipt_fails_before_generation(monkeypatch, tmp_path):
    calls = install_fake_stack(monkeypatch)
    cfg = local_model(tmp_path)
    (cfg.model_path / 'config.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='do not match'):
        gpu.run(package(), cfg)
    assert not calls


def test_menu_only_control_trace_stays_separate(monkeypatch, tmp_path):
    install_fake_stack(monkeypatch)
    trace = gpu.run(package([control_job()], schema='jane-choice-controls-v1'), local_model(tmp_path))
    assert trace['schema_version'] == 'jane-choice-control-traces-v1'


def test_deadline_after_batch_leaves_checkpoint_but_no_completed_trace(monkeypatch, tmp_path):
    install_fake_stack(monkeypatch)
    real_check = gpu.check_deadline
    checks = []

    def timeout_after_progress(started, seconds, **kwargs):
        checks.append(None)
        if len(checks) == 6:
            raise TimeoutError('test budget exhausted after completed batch')
        real_check(started, seconds, **kwargs)

    monkeypatch.setattr(gpu, 'check_deadline', timeout_after_progress)
    path = tmp_path / 'partial.jsonl'
    with path.open('x') as checkpoint:
        with pytest.raises(TimeoutError):
            gpu.run(package(), local_model(tmp_path), checkpoint)
    assert len(path.read_text().splitlines()) == 1


def test_driver_info_records_actual_version(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='570.86.15\n'))
    assert gpu._driver_info() == {'driver_version': '570.86.15', 'driver_query_error': None}


def test_driver_query_failure_is_explicit(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi unavailable')
    monkeypatch.setattr(gpu.subprocess, 'run', missing)
    result = gpu._driver_info()
    assert result['driver_version'] is None
    assert 'FileNotFoundError' in result['driver_query_error']


@pytest.mark.parametrize('raw', [
    '{"answer":"Z","confidence":0.8,"status":"answer"}',
    '{"answer":"A","cofidence":0.8,"status":"answer"}',
    '{"answer":"A","confidence":1.1,"status":"answer"}',
])
def test_mask_failure_checkpoints_exact_raw_before_failing(monkeypatch, tmp_path, raw):
    install_fake_stack(monkeypatch)
    monkeypatch.setattr(Tokenizer, 'decode', lambda self, ids, **kw: raw)
    path = tmp_path / 'failed-mask.jsonl'
    with path.open('x') as checkpoint:
        with pytest.raises(ValueError, match='grammar'):
            gpu.run(package([control_job()], schema='jane-choice-controls-v1'),
                    local_model(tmp_path), checkpoint)
    recorded = json.loads(path.read_text())
    assert recorded['raw_response'] == raw
    assert recorded['generated_token_ids'] == [101, 99, 0]
    assert all(recorded[field] == gpu.parse_response(raw)[field]
               for field in ['answer', 'confidence', 'status', 'parse_error'])


def test_length_failure_retains_checkpoint_without_completed_trace(monkeypatch, tmp_path):
    install_fake_stack(monkeypatch)
    real_slice = gpu.slice_generated_tokens
    monkeypatch.setattr(gpu, 'slice_generated_tokens',
                        lambda rows, width: [[row[0], 98, 0] for row in real_slice(rows, width)])
    raw = '{"answer":"Paris","confidence":0.7,"status":"answer"}'
    monkeypatch.setattr(Tokenizer, 'decode', lambda self, ids, **kw: raw)
    path = tmp_path / 'length.jsonl'
    with path.open('x') as checkpoint:
        with pytest.raises(ValueError, match='without EOS'):
            gpu.run(package(), local_model(tmp_path), checkpoint)
    recorded = json.loads(path.read_text())
    assert recorded['finish_reason'] == 'length'
    assert recorded['raw_response'] == raw
    assert recorded['status'] == 'answer'


def test_mixed_batch_selects_mc_and_oe_constraints(monkeypatch, tmp_path):
    install_fake_stack(monkeypatch)
    jobs = [main_job(), main_job(job_id='job2', format='mc', condition='pool', menu_id='m1')]
    trace = gpu.run(package(jobs), replace(local_model(tmp_path), batch_size=2))
    assert [row['answer'] for row in trace['predictions']] == ['Paris', 'A']
    assert [row['constraint_format'] for row in trace['predictions']] == ['oe', 'mc']
