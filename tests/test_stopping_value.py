from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training import stopping_value as sv


def episode(qid, observations, buzz, wait, group_id=None):
    return SimpleNamespace(
        qid=qid, group_id=group_id, observations=np.asarray(observations, dtype=np.float32),
        buzz_rewards=np.asarray(buzz, dtype=np.float32),
        wait_rewards=np.asarray(wait, dtype=np.float32),
    )


def test_bellman_targets_use_online_selection_and_delayed_evaluation_only_after_wait():
    rewards = torch.tensor([[-0.1, 0.4], [0.3, -1.0]])
    live = torch.tensor([True, False])
    online = torch.tensor([[5.0, 2.0], [1.0, 9.0]])
    delayed = torch.tensor([[0.8, 4.5], [100.0, 200.0]], requires_grad=True)
    targets = sv._bellman_targets(rewards, live, online, delayed)
    torch.testing.assert_close(targets, torch.tensor([[0.7, 0.4], [0.3, -1.0]]))
    assert not targets.requires_grad


def test_fitted_value_estimates_expected_payoff_before_maximizing():
    # Two identical visible trajectories disagree about the final buzz reward.
    # A hindsight maximum per item would inflate WAIT to 0.55, beating BUZZ.
    items = [episode(i, [[1, 0], [0, 1]], [0.35, (-1) ** i], [-0.05, 0.2])
             for i in range(12)]
    model = sv.fit_value(items, "fitted_value", seed=1, epochs=4)
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with torch.no_grad():
        torch.testing.assert_close(model(states), torch.tensor([[0.15, 0.35], [0.2, 0.0]]),
                                   atol=0.01, rtol=0)


def test_double_dqn_replays_with_a_lagged_target_and_resynchronizes(monkeypatch):
    items = [episode(1, [[1, 0], [0, 1]], [-0.5, 1], [-0.05, -1])]
    agreements = []
    original = sv._bellman_targets

    def observe(rewards, live, online, delayed):
        agreements.append(torch.equal(online, delayed))
        return original(rewards, live, online, delayed)

    monkeypatch.setattr(sv, "_bellman_targets", observe)
    sv.fit_value(items, "double_dqn", seed=2, epochs=3)
    assert agreements[0]
    assert not any(agreements[1:20])
    assert agreements[20]


@pytest.mark.parametrize("method", ["fitted_value", "double_dqn", "stop_imitation"])
def test_learners_fit_wait_then_voluntary_buzz_and_reload(method):
    items = [episode(i, [[1, 0], [0, 1]], [-0.5, 1], [-0.05, -1]) for i in range(8)]
    model = sv.fit_value(items, method, seed=2, epochs=12)
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    restored = sv.make_value_model(2)
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert model(states).argmax(dim=1).tolist() == [0, 1]
        torch.testing.assert_close(restored(states), model(states))


def test_imitation_teacher_keeps_duplicate_question_groups_in_one_fold(monkeypatch):
    items = [episode(i, [[g]], [1], [0], group_id=str(g))
             for i, g in enumerate([1, 1, 2, 3, 3, 4])]
    checked = []

    def fake_fit(train, method, *, seed, epochs):
        assert method == "fitted_value"
        seen = {int(ep.observations[0, 0]) for ep in train}

        def predict(obs):
            for group in obs[:, 0].tolist():
                assert group not in seen
                checked.append(group)
            return torch.tensor([[0.0, 1.0]]).expand(len(obs), -1)

        return predict

    monkeypatch.setattr(sv, "fit_value", fake_fit)
    assert sv._teacher_actions(items, seed=3, epochs=2).tolist() == [1] * 6
    assert sorted(checked) == [1, 1, 2, 3, 3, 4]


def test_one_training_group_cannot_provide_out_of_fold_teacher_labels():
    items = [episode(i, [[0]], [1], [0], group_id="same clue") for i in range(2)]
    with pytest.raises(ValueError, match="two.*groups"):
        sv.fit_value(items, "stop_imitation", seed=1, epochs=2)


def test_value_training_rejects_misaligned_rewards():
    with pytest.raises(ValueError, match="reward"):
        sv.fit_value([episode(1, [[0], [1]], [1], [0, 1])], "double_dqn", seed=0, epochs=1)
