"""Tests for the configuration system"""

from dataclasses import asdict, replace

import pytest
from stable_baselines3 import DQN, PPO

from src.config import local_config
from src.config.dqn_config import DQNConfig
from src.config.local_config import LocalConfig, detect_device
from src.config.ppo_config import PPOConfig


def test_dqn_config_matches_plan():
    assert asdict(DQNConfig()) == {
        "learning_rate": 1e-4,
        "buffer_size": 50_000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.1,
        "target_update_interval": 1000,
    }


def test_ppo_config_matches_plan():
    assert asdict(PPOConfig()) == {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
    }


def test_configs_are_valid_stable_baselines3_arguments():
    # small buffer so the test does not allocate the full replay buffer
    dqn = DQN(
        "MlpPolicy", "CartPole-v1", **asdict(replace(DQNConfig(), buffer_size=100))
    )
    ppo = PPO("MlpPolicy", "CartPole-v1", **asdict(PPOConfig()))
    assert dqn.target_update_interval == 1000
    assert ppo.ent_coef == 0.01


@pytest.mark.parametrize(
    "cuda, mps, expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_detect_device_prefers_cuda_then_mps(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(local_config.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(local_config.torch.backends.mps, "is_available", lambda: mps)
    assert detect_device() == expected


def test_local_config_is_consistent():
    config = LocalConfig()
    assert config.USE_GPU == (config.DEVICE != "cpu")
    assert config.BATCH_SIZE == (64 if config.USE_GPU else 32)
    assert 0 <= config.NUM_WORKERS <= 4
    assert config.AVAILABLE_RAM_GB > 0
