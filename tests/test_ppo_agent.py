"""Tests for the PPO agent (fake game, no Chrome)"""

from dataclasses import replace

import numpy as np
import pytest

from src.config.ppo_config import PPOConfig
from src.models.ppo_agent import PPOAgent
from src.training.envs import make_dino_env
from tests.test_environment import FakeGame

SMALL = replace(PPOConfig(), n_steps=32, batch_size=16)


@pytest.fixture
def game():
    return FakeGame(crash_after=10)


@pytest.fixture
def env(game):
    env = make_dino_env(game=game)
    yield env
    env.close()


def make_agent(env, **overrides):
    return PPOAgent(env, replace(SMALL, **overrides), device="cpu")


def test_agent_uses_config_values(env):
    model = make_agent(env, learning_rate=3e-4, gae_lambda=0.9, ent_coef=0.02).model

    assert model.learning_rate == 3e-4
    assert model.gae_lambda == 0.9
    assert model.ent_coef == 0.02


def test_train_and_predict_valid_actions(env):
    agent = make_agent(env)
    agent.train(64)

    assert agent.model.num_timesteps == 64
    action, _ = agent.predict(env.reset())
    assert action[0] in (0, 1, 2)


def test_game_is_paused_only_between_rollouts(env, game):
    agent = make_agent(env)
    pauses = []
    original_train = agent.model.train
    agent.model.train = lambda: (pauses.append(game.paused), original_train())[1]

    agent.train(64)

    assert pauses == [True, True]  # paused during both updates
    assert not game.paused  # resumed when training ends


def test_save_load_round_trip_keeps_actions(env, tmp_path):
    agent = make_agent(env)
    agent.train(32)
    obs = np.random.default_rng(0).integers(0, 256, (1, 84, 84, 4), dtype=np.uint8)
    before, _ = agent.predict(obs)

    agent.save(tmp_path / "ppo")
    other = make_agent(env)
    other.load(tmp_path / "ppo")

    assert (other.predict(obs)[0] == before).all()
