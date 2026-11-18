"""Check tabulated WAIT/BUZZ trajectories against each legal live stopping path."""

import numpy as np
import pytest
from dataclasses import asdict, replace

from qb_data.mc_builder import validate_gold_option, validate_reveal_grid
from scripts._common import mc_question_from_dict
from qb_env.stop_only_env import StopOnlyEnv
from training.stopping_data import collect_episodes, evaluate_actions, ReplayEnv


@pytest.mark.parametrize("end_mode", ["force_commit", "no_buzz"])
def test_every_stopping_path_matches_live_env(sample_tfidf_env, end_mode):
    sample_tfidf_env.end_mode = end_mode
    sample_tfidf_env.no_buzz_reward = -0.25
    live = StopOnlyEnv(sample_tfidf_env)
    episodes = collect_episodes(sample_tfidf_env, seed=7)
    replay = ReplayEnv(episodes)
    for i, episode in enumerate(episodes):
        horizon = len(episode.observations)
        for stop in range(horizon + 1):
            obs, _ = live.reset(seed=7 + i, options={"question_idx": i})
            replay_obs, _ = replay.reset(options={"question_idx": i})
            total = 0
            for t in range(horizon):
                np.testing.assert_array_equal(obs, replay_obs)
                action = int(t == stop)
                obs, reward, terminated, truncated, info = live.step(action)
                replay_obs, other, done, timeout, _ = replay.step(action)
                assert other == pytest.approx(reward)
                assert (terminated or truncated) == done and not timeout
                total += reward
                if done:
                    break
            actions = np.arange(horizon) == stop
            entry = evaluate_actions(episode, actions)
            assert entry["return"] == pytest.approx(total)
            assert entry["outcome"] == ("voluntary" if stop < horizon else end_mode)
            assert entry["correct"] == info.get("correct", info.get("forced_correct", False))
            if stop < horizon:
                assert entry["visible_prefixes"] == stop


def test_action_policy_must_match_horizon(sample_tfidf_env):
    [episode, *_] = collect_episodes(sample_tfidf_env, seed=7)
    with pytest.raises(ValueError):
        evaluate_actions(episode, [1])
    with pytest.raises(ValueError):
        evaluate_actions(episode, np.full(len(episode.observations), 2))


@pytest.mark.parametrize("indices,prefixes", [([0], ["Who was"]), ([1, 0], ["Who was", "Who"]),
                                              ([True], ["Who was"]), ([10], ["Who"]), ([], [])])
def test_saved_grid_cannot_extend_or_reorder_visible_clues(sample_mc_question, indices, prefixes):
    sample_mc_question.question = " ".join(sample_mc_question.tokens)
    sample_mc_question.run_indices = indices
    sample_mc_question.cumulative_prefixes = prefixes
    with pytest.raises(ValueError, match="reveal"):
        validate_reveal_grid(sample_mc_question)


def test_reveal_boundaries_are_bound_to_source_words(sample_mc_question):
    sample_mc_question.question = "opening secret"
    sample_mc_question.tokens = ["opening secret"]
    sample_mc_question.run_indices = [0]
    sample_mc_question.cumulative_prefixes = ["opening secret"]
    with pytest.raises(ValueError, match="source question"):
        validate_reveal_grid(sample_mc_question)


@pytest.mark.parametrize("bad", [True, 0.5])
def test_deserialization_rejects_lossy_gold_coercion(sample_mc_question, bad):
    with pytest.raises(ValueError, match="gold_index"):
        mc_question_from_dict({**asdict(sample_mc_question), "gold_index": bad})


def test_shifted_menu_key_cannot_train_or_score(sample_mc_question):
    with pytest.raises(ValueError, match="gold_index"):
        validate_gold_option(replace(sample_mc_question, gold_index=1))
