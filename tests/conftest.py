"""Test configuration and fixtures"""

from dataclasses import replace

import pytest

from src.config.dqn_config import DQNConfig
from src.models.dqn_agent import DQNAgent


@pytest.fixture
def dummy_fixture():
    """Dummy fixture for testing"""
    return "test_value"


@pytest.fixture(autouse=True)
def small_replay_buffer(monkeypatch):
    """DQN agents built with the default config get a 200-step replay buffer: the
    real 50k one is ~1.4GB, and every latest checkpoint writes it to disk"""
    monkeypatch.setattr(
        DQNAgent,
        "default_config",
        staticmethod(lambda: replace(DQNConfig(), buffer_size=200)),
    )
