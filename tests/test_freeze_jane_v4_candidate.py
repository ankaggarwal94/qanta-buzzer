"""Historical v4 regeneration stays separate from current paid execution."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from qb_data.jane_paired import build_jobs
from scripts import modal_jane_pilot as current
from scripts.prepare_jane_gpu_pilot import build_choice_controls


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def _dataset(phase, count):
    questions = []
    for index in range(count):
        qid = f'{phase}-{index}'
        text = f'Synthetic question {qid} with a unique final clue.'
        answer = f'Synthetic answer {qid}'
        questions.append({
            'qid': qid, 'group_id': qid,
            'split': ('calibration', 'selection', 'test')[index % 3],
            'question': text,
            'prefixes': [{'prefix_id': 'full', 'text': text, 'fraction': 1.0}],
            'answer': {'raw': answer, 'accepted': [answer], 'rejected': [], 'prompt': []},
            'menus': [{'condition': 'synthetic', 'menu_id': 'fixed',
                       'options': [{'id': letter, 'text': answer if letter == 'A'
                                    else f'Synthetic distractor {qid} {letter}'} for letter in 'ABCD'],
                       'gold_option_id': 'A', 'provenance': {'construction': 'synthetic'}}],
        })
    return {'schema_version': 'jane-paired-v1', 'evidence_scope': 'engineering_smoke',
            'prompt_template': 'concise_json_v3',
            'source': {'origin': 'unit test', 'provenance': 'synthetic fixture only'},
            'questions': questions}


def _original(tmp_path):
    root = tmp_path / 'original'
    for phase, count in (('dev', 3), ('main', 200)):
        dataset = _dataset(phase, count)
        _write_json(root / 'evaluator' / f'{phase}_dataset.json', dataset)
        _write_json(root / 'public' / f'{phase}_jobs.json', {
            'schema_version': 'jane-public-jobs-v1', 'evidence_scope': 'engineering_smoke',
            'jobs': build_jobs(dataset)})
        controls = build_choice_controls(dataset)
        _write_json(root / 'public' / f'{phase}_choices_only.json', controls)
        _write_json(root / 'evaluator' / f'{phase}_choices_only_gold.json',
                    {job['job_id']: 'A' for job in controls['jobs']})
    manifest = {'schema_version': 'jane-public-input-manifest-v1', 'files': {}}
    for name in current.INPUT_FILES:
        raw = (root / 'public' / name).read_bytes()
        manifest['files'][name] = {'sha256': hashlib.sha256(raw).hexdigest(),
                                   'byte_count': len(raw), 'job_count': len(json.loads(raw)['jobs'])}
    _write_json(root / 'public' / 'manifest.json', manifest)
    return root


def test_module_import_and_cli_help_are_offline():
    helper = importlib.import_module('scripts.freeze_jane_v4_candidate')
    assert callable(helper.freeze)
    result = subprocess.run([sys.executable, '-m', 'scripts.freeze_jane_v4_candidate', '--help'],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'historical' in result.stdout.lower()


def test_archived_budget_is_exact_and_cannot_authorize_current_compute():
    helper = importlib.import_module('scripts.freeze_jane_v4_candidate')
    budget = helper.historical_v4_budget()
    assert budget['schema_version'] == 'jane-modal-budget-v1'
    assert budget['ceiling_usd'] == '7.74'
    assert budget['max_session_seconds'] == 8979
    assert budget['prior_host_receipt_sha256'] == helper.PRIOR_HOST_RECEIPT_SHA256
    assert hashlib.sha256(json.dumps(budget, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == (
        '540e13580e6317ab44d3040271b080d6c916912c7814d7121956f6daee88b6dd')
    with pytest.raises(ValueError):
        current.validate_budget_control({'budget': budget})
    budget['ceiling_usd'] = '100'
    assert helper.historical_v4_budget()['ceiling_usd'] == '7.74'


def test_invalid_prior_receipt_fails_before_writing(tmp_path):
    helper = importlib.import_module('scripts.freeze_jane_v4_candidate')
    receipt = tmp_path / 'wrong.json'
    receipt.write_text('{}')
    out = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='prior host receipt'):
        helper.freeze(tmp_path / 'missing-original', out, receipt)
    assert not out.exists()


def test_full_synthetic_regeneration_preserves_public_identity_and_gold(tmp_path, monkeypatch):
    helper = importlib.import_module('scripts.freeze_jane_v4_candidate')
    original = _original(tmp_path)
    receipt = tmp_path / 'synthetic_prior_receipt.json'
    _write_json(receipt, {'status': 'DEVELOPMENT_INTERFACE_GATE_FAILED'})
    monkeypatch.setattr(helper, 'PRIOR_HOST_RECEIPT_SHA256',
                        hashlib.sha256(receipt.read_bytes()).hexdigest())
    out = tmp_path / 'reconstructed'
    record = helper.freeze(original, out, receipt)
    assert record['historical_reconstruction'] is True
    assert record['current_execution_authorized'] is False
    assert record['budget'] == helper.historical_v4_budget()
    assert record['historical_source_commit'] == 'a2013af95f7fa49f055423a9b398358aaf252c64'
    assert record['historical_freeze_sha256'] == 'c84e978e0ba918552414204f8eddc2fada6f54240519e497d4155fad73deb2b0'
    _, identity = current.load_public_inputs(out / 'public')
    assert record['candidate_public_input_id'] == identity['public_input_id']
    assert record['candidate_public_input_id'] != record['original_public_input_id']
    for phase in ('dev', 'main'):
        expected = json.loads((original / 'evaluator' / f'{phase}_dataset.json').read_text())
        expected['prompt_template'] = 'concise_json_v4'
        actual = json.loads((out / 'evaluator' / f'{phase}_dataset.json').read_text())
        assert actual == expected
        public = json.loads((out / 'public' / f'{phase}_jobs.json').read_text())
        assert public['jobs'] == build_jobs(expected)
        for subdir, name in (('public', f'{phase}_choices_only.json'),
                             ('evaluator', f'{phase}_choices_only_gold.json')):
            assert (out / subdir / name).read_bytes() == (original / subdir / name).read_bytes()
    with ZipFile(out / 'public_inputs.zip') as archive:
        assert set(archive.namelist()) == set(current.INPUT_FILES) | {'manifest.json'}
        for name in archive.namelist():
            assert archive.read(name) == (out / 'public' / name).read_bytes()
    with pytest.raises(FileExistsError):
        helper.freeze(original, out, receipt)
