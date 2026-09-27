"""Tests for the DQN agent (fake game, no Chrome)"""

from dataclasses import replace

import numpy as np
import pytest
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn

from src.config.dqn_config import DQNConfig
from src.evaluation.evaluate import RandomPolicy, evaluate
from src.models.dqn_agent import DQNAgent
from src.models_mgmt.checkpoint_manager import CheckpointManager
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


class StopAt(BaseCallback):
    """Ends training at `step`, like a crash or power cut"""

    def __init__(self, step):
        super().__init__()
        self.step = step

    def _on_step(self) -> bool:
        return self.num_timesteps < self.step


class EpsilonAt(BaseCallback):
    """Records the exploration rate at `step`"""

    def __init__(self, step):
        super().__init__()
        self.step = step
        self.value = None

    def _on_step(self) -> bool:
        if self.num_timesteps == self.step:
            self.value = self.model.exploration_rate
        return True


def decaying_over_whole_run(agent):
    """Epsilon decays over the whole run (SB3's default is the first 10%), so
    a schedule that restarted on resume would show"""
    model = agent.model
    model.exploration_fraction = 1.0  # saved, and rebuilt by load()
    model.exploration_schedule = get_linear_fn(
        model.exploration_initial_eps, model.exploration_final_eps, 1.0
    )
    return agent


def test_resume_continues_steps_epsilon_and_replay_buffer(env, tmp_path):
    interrupted = decaying_over_whole_run(make_agent(env))
    interrupted.model.learn(128, callback=StopAt(64))
    latest = CheckpointManager(tmp_path).save_latest(
        interrupted.model, "dqn", step=64, episodes=6
    )

    resumed = make_agent(env)
    resumed.resume(latest)
    assert resumed.model.num_timesteps == 64
    saved_size = interrupted.model.replay_buffer.size()
    assert saved_size > 0 and resumed.model.replay_buffer.size() == saved_size
    epsilon = EpsilonAt(100)
    resumed.train(128 - 64, callback=epsilon)

    uninterrupted = decaying_over_whole_run(make_agent(env))
    reference = EpsilonAt(100)
    uninterrupted.train(128, callback=reference)
    assert resumed.model.num_timesteps == 128
    assert epsilon.value == pytest.approx(reference.value)


def test_evaluate_random_policy_runs_requested_episodes(env):
    rewards, lengths = evaluate(RandomPolicy(env), env, episodes=3)

    assert len(rewards) == len(lengths) == 3
    assert lengths == [10, 10, 10]
