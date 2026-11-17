"""Run targeted source mutations of ACL rewards in isolated Python processes."""

from __future__ import annotations

import inspect
import subprocess
import sys
from textwrap import dedent

import pytest

from qb_env.acl_rewards import RewardSpec
from qb_env.tossup_env import TossupMCEnv


# name, method, original expression, mutant expression, witness
MUTATIONS = (
    ("wrong-exponent", RewardSpec.correct_reward, "remaining ** self.shape",
     "remaining ** 1.", "test_hand_computed_interior_values"),
    ("step-off-by-one", RewardSpec.correct_reward, "p < self.threshold",
     "p <= self.threshold", "test_hand_computed_interior_values"),
    ("singular-at-zero", RewardSpec.correct_reward, "if k == 0.:",
     "if False:", "test_hand_computed_interior_values"),
    ("duplicate-payout", TossupMCEnv._step_acl, "settle_reward(wait_reward, outcome_reward,",
     "settle_reward(wait_reward, 2 * outcome_reward,", "test_all_reachable_action_paths"),
    ("drop-discount", TossupMCEnv._step_acl, "outcome_reward, spec.gamma, phi,",
     "outcome_reward, 1., phi,", "test_all_reachable_action_paths"),
    ("cutoff-bootstrap", TossupMCEnv._step_acl, "0. if self.terminated else",
     "0. if True else", "test_external_time_limit_keeps_bootstrap"),
    ("skip-last-prefix", TossupMCEnv._step_acl, "elif visible < horizon:",
     "elif visible + 1 < horizon:", "test_all_reachable_action_paths"),
    ("future-length", TossupMCEnv._obs, "1 if self.acl_reward is not None else self.total_steps",
     "self.total_steps", "test_suffix_length_and_gold_are_invisible"),
)


def run_child(case):
    """Use the source of the installed checkout and mutate only the in-memory method."""
    name, method, old, new, witness = case
    source = dedent(inspect.getsource(method))
    if source.count(old) != 1:
        raise ValueError(f"Mutation {name} needs review against current source")
    scope = method.__globals__.copy()
    exec(compile(source.replace(old, new), f"<{name}>", "exec"), scope)
    owner = scope[method.__qualname__.split(".")[0]]
    setattr(owner, method.__name__, scope[method.__name__])
    result = pytest.main(["-q", "--tb=no", "-x", f"tests/test_acl_rewards.py::{witness}"])
    return 0 if result == pytest.ExitCode.TESTS_FAILED else 1


if __name__ == "__main__":
    if len(sys.argv) == 2:
        sys.exit(run_child(next(case for case in MUTATIONS if case[0] == sys.argv[1])))
    for case in MUTATIONS:
        subprocess.run([sys.executable, "-m", "scripts.acl_reward_mutations", case[0]], check=True)
        print(f"Killed {case[0]}", flush=True)
