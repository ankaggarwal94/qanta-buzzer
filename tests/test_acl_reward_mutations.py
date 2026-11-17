"""A red control never earns a mutation kill."""
import subprocess

import pytest

from scripts import acl_reward_mutations as gate


def test_failed_control_aborts_before_mutation(monkeypatch):
    calls = []
    def fail(args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(gate.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        gate.run_cases(gate.MUTATIONS, "fixture-python")
    assert len(calls) == 1 and "--control" in calls[0]
