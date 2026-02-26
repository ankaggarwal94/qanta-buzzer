"""
Quiz Bowl Environment Package

Gymnasium-compliant POMDP environment for quiz bowl question answering.
The agent observes belief features (K+6 dimensional) and chooses to WAIT
for more clues or BUZZ with a specific answer option.
"""

from qb_env.tossup_env import TossupMCEnv, make_env_from_config

__all__ = ["TossupMCEnv", "make_env_from_config"]
