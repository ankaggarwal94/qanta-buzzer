"""Non-executing checks for the explicit one-shot development workflow."""
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow():
    return (ROOT / '.github/workflows/imcqa-mistral-replication.yml').read_text()


def test_narrow_trigger_and_no_manual_retry():
    raw = workflow()
    parsed = yaml.safe_load(raw)
    trigger = parsed.get('on', parsed.get(True))
    assert set(trigger) == {'push'}
    assert trigger['push']['branches'] == ['feat/imcqa-mistral-replication-20261007']
    assert trigger['push']['paths'] == ['.github/workflows/imcqa-mistral-replication.yml']
    gate = parsed['jobs']['development']['if']
    assert 'github.run_attempt == 1' in gate
    assert "github.actor == 'ankaggarwal94'" in gate
    assert "github.event.head_commit.message == 'ops: run approved eight-dollar Mistral development once'" in gate
    assert parsed['concurrency']['cancel-in-progress'] is False


def test_paid_steps_and_safe_cleanup_are_explicit():
    steps = yaml.safe_load(workflow())['jobs']['development']['steps']
    paid = [s for s in steps if 'MODAL_TOKEN_SECRET' in s.get('env', {})]
    assert len(paid) == 3
    assert '--dry-run' not in paid[0]['run']
    assert 'prepare-cache' in paid[0]['run']
    assert 'score-stage --stage development' in paid[1]['run']
    assert 'cleanup-cache' in paid[2]['run']
    assert "steps.score.outcome != 'success'" in paid[2]['if']
    assert all(s['timeout-minutes'] <= 36 for s in paid)
    assert steps[-1]['if'] == 'always()'
    assert not any('evaluator' in s.get('run', '') for s in steps)
    for step in steps:
        if 'uses' in step:
            assert re.fullmatch(r'actions/[a-z-]+@[0-9a-f]{40}', step['uses'])
