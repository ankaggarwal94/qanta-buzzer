from agents.threshold_buzzer import (
    ThresholdBuzzer,
    AlwaysBuzzFinalBuzzer,
    EpisodeResult,
    sweep_thresholds,
    result_to_dict,
)
from agents.bayesian_buzzer import (
    SoftmaxProfileBuzzer,
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
)

# Lazy import: PPOBuzzer requires stable_baselines3 which may not be installed
# in all environments (e.g., baseline-only runs). Import on demand.


def __getattr__(name: str):
    if name in ("PPOBuzzer", "PPOEpisodeTrace"):
        from agents.ppo_buzzer import PPOBuzzer, PPOEpisodeTrace
        return {"PPOBuzzer": PPOBuzzer, "PPOEpisodeTrace": PPOEpisodeTrace}[name]
    raise AttributeError(f"module 'agents' has no attribute {name!r}")


__all__ = [
    "ThresholdBuzzer",
    "AlwaysBuzzFinalBuzzer",
    "SoftmaxProfileBuzzer",
    "SequentialBayesBuzzer",
    "PPOBuzzer",
    "EpisodeResult",
    "SoftmaxEpisodeResult",
    "PPOEpisodeTrace",
    "sweep_thresholds",
    "result_to_dict",
]
