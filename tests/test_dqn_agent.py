"""Tests for the DQN agent (fake game, no Chrome)"""

from dataclasses import replace

import numpy as np
import pytest

from src.config.dqn_config import DQNConfig
from src.models.dqn_agent import DQNAgent
from src.training.dqn_smoke import RandomPolicy, evaluate
from src.training.envs import make_dino_env
from tests.test_environment import FakeGame

SMALL = replace(DQNConfig(), buffer_size=200, batch_size=8)


@pytest.fixture
def env():
    env = make_dino_env(game=FakeGame(crash_after=10))
    yield env
    env.close()


def make_agent(env, **overrides):
    return DQNAgent(env, replace(SMALL, **overrides), device="cpu")


def test_env_stacks_four_frames(env):
    assert env.observation_space.shape == (84, 84, 4)
    assert env.reset().shape == (1, 84, 84, 4)


def test_agent_uses_config_values(env):
    model = make_agent(env, learning_rate=3e-4, gamma=0.9).model

    assert model.learning_rate == 3e-4
    assert model.gamma == 0.9
    assert model.buffer_size == 200
    assert model.optimize_memory_usage


def test_train_fills_buffer_and_predicts_valid_actions(env):
    agent = make_agent(env)
    agent.train(64)

    assert agent.model.replay_buffer.size() > 0
    action, _ = agent.predict(env.reset())
    assert action.shape == (1,)
    assert action[0] in (0, 1, 2)


def test_save_load_round_trip_keeps_actions(env, tmp_path):
    agent = make_agent(env)
    agent.train(64)
    obs = np.random.default_rng(0).integers(0, 256, (1, 84, 84, 4), dtype=np.uint8)
    before, _ = agent.predict(obs)

    agent.save(tmp_path / "dqn")
    other = make_agent(env)
    other.load(tmp_path / "dqn")

    after, _ = other.predict(obs)
    assert (before == after).all()


def test_evaluate_random_policy_runs_requested_episodes(env):
    rewards, lengths = evaluate(RandomPolicy(env), env, episodes=3)

    assert len(rewards) == len(lengths) == 3
    assert lengths == [10, 10, 10]
