"""Tests for the smoke run (fake game, no Chrome)"""

import pytest

from src.models.dqn_agent import DQNAgent
from src.training import smoke
from src.training.envs import make_dino_env
from tests.test_dqn_agent import SMALL
from tests.test_environment import FakeGame


@pytest.fixture(autouse=True)
def fake_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        smoke, "make_dino_env", lambda: make_dino_env(game=FakeGame(10))
    )
    monkeypatch.setattr(
        smoke,
        "AGENTS",
        {"dqn": lambda env: DQNAgent(env, SMALL, device="cpu")},
    )


def test_smoke_compares_random_and_trained_agent(tmp_path, capsys):
    smoke.main(agent="dqn", steps=50, episodes=2)

    random_line, agent_line = capsys.readouterr().out.splitlines()[-2:]
    assert random_line.startswith("random: reward")
    assert agent_line.startswith("dqn: reward")
    assert (tmp_path / "models" / "dqn_smoke.zip").exists()
