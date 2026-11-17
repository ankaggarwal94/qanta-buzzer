"""Run targeted source mutations of ACL rewards in isolated Python processes."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


# name, method, original expression, mutant expression, witness
MUTATIONS = (
    ("wrong-exponent", "RewardSpec.correct_reward", "remaining ** self.shape",
     "remaining ** 1.", "test_hand_computed_interior_values"),
    ("step-off-by-one", "RewardSpec.correct_reward", "p < self.threshold",
     "p <= self.threshold", "test_hand_computed_interior_values"),
    ("singular-at-zero", "RewardSpec.correct_reward", "if k == 0.:",
     "if False:", "test_hand_computed_interior_values"),
    ("duplicate-payout", "TossupMCEnv._step_acl", "settle_reward(wait_reward, outcome_reward,",
     "settle_reward(wait_reward, 2 * outcome_reward,", "test_all_reachable_action_paths"),
    ("drop-discount", "TossupMCEnv._step_acl", "outcome_reward, spec.gamma, phi,",
     "outcome_reward, 1., phi,", "test_all_reachable_action_paths"),
    ("cutoff-bootstrap", "TossupMCEnv._step_acl", "0. if self.terminated else",
     "0. if True else", "test_external_time_limit_keeps_bootstrap"),
    ("skip-last-prefix", "TossupMCEnv._step_acl", "elif visible < horizon:",
     "elif visible + 1 < horizon:", "test_all_reachable_action_paths"),
    ("future-length", "TossupMCEnv._obs", "1 if self.acl_reward is not None else self.total_steps",
     "self.total_steps", "test_suffix_length_and_gold_are_invisible"),
)


def run_child(case, *, baseline):
    """Require a passing control, or an actual test-call failure exercising the mutant."""
    import pytest
    from qb_env.acl_rewards import RewardSpec
    from qb_env.tossup_env import TossupMCEnv

    name, target, old, new, witness = case
    cls, method_name = target.split(".")
    owner = {"RewardSpec": RewardSpec, "TossupMCEnv": TossupMCEnv}[cls]
    method = getattr(owner, method_name)
    source = dedent(inspect.getsource(method))
    if source.count(old) != 1:
        raise ValueError(f"Mutation {name} needs review against current source")
    outcomes = []
    reached = False

    class Reports:
        def pytest_runtest_setup(self, item):
            nonlocal reached
            reached = False

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                outcomes.append((report.outcome, reached))

    if not baseline:
        scope = method.__globals__.copy()
        exec(compile(source.replace(old, new), f"<{name}>", "exec"), scope)
        mutated = scope[method_name]

        def instrumented(*args, **kwargs):
            nonlocal reached
            reached = True
            return mutated(*args, **kwargs)

        setattr(owner, method_name, instrumented)
    result = pytest.main(["-q", "--tb=no", "-x", f"tests/test_acl_rewards.py::{witness}"],
                         plugins=[Reports()])
    if baseline:
        return 0 if result == pytest.ExitCode.OK and outcomes and all(status == "passed" for status, _ in outcomes) else 1
    return 0 if result == pytest.ExitCode.TESTS_FAILED and ("failed", True) in outcomes else 1


def run_cases(cases, python):
    """Run pristine witnesses before granting any mutant kill."""
    checked = set()
    for case in cases:
        if case[-1] not in checked:
            subprocess.run([str(python), "-m", "scripts.acl_reward_mutations", "--control", case[0]],
                           check=True, cwd=ROOT)
            checked.add(case[-1])
    for case in cases:
        subprocess.run([str(python), "-m", "scripts.acl_reward_mutations", "--mutant", case[0]],
                       check=True, cwd=ROOT)
        print(f"Killed {case[0]}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("--control", "--mutant"):
        case = next(case for case in MUTATIONS if case[0] == sys.argv[2])
        sys.exit(run_child(case, baseline=sys.argv[1] == "--control"))
    cases = MUTATIONS if len(sys.argv) == 1 else [next(case for case in MUTATIONS if case[0] == sys.argv[1])]
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run_cases(cases, python)
