import pytest

from scripts.stopdff_v5.identity import load_json_strict


def test_interrupted_training_resumes_completed_block(tmp_path, monkeypatch):
    from scripts import acl_expansion as acl
    from scripts.acl_smoke_data import inputs
    from qb_data.acl_protocol import ArmSpec, prepare
    from pathlib import Path
    p = prepare(ArmSpec.from_dict({"version": 1, "menu": "uniform_global", "reward": {"arm": "linear"},
                                  "seed": 1, "k": 4}), inputs(),
                load_json_strict(Path("configs/acl_expansion/smoke_oracle.json")))
    settings = acl.Training(updates=2, rollout_size=4, epochs=1, batch_size=4, checkpoint_updates=1)
    from agents.ppo_buzzer import PPOBuzzer
    real_train = PPOBuzzer.train
    seen = []
    def interrupt(self, total_timesteps):
        seen.append(total_timesteps)
        if len(seen) == 2:
            raise InterruptedError("simulated between blocks")
        real_train(self, total_timesteps)
    monkeypatch.setattr(PPOBuzzer, "train", interrupt)
    with pytest.raises(InterruptedError):
        acl.run(p, tmp_path, "ppo", settings)
    folder = acl.run(p, tmp_path, "ppo", settings)
    assert len(seen) == 3 and seen == [4, 4, 4]
    acl.run(p, tmp_path, "ppo", settings, replay=True)
    assert len(seen) == 3
