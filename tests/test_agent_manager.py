"""Tests for the agent manager (fake agents and infos, no Chrome)"""

import numpy as np
import pytest
from typer.testing import CliRunner

from src.cli import main as cli
from src.config.local_config import LocalConfig
from src.models.dqn_agent import DQNAgent
from src.models.ppo_agent import PPOAgent
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from src.routing.agent_manager import AgentManager
from src.routing.laya_router import LayaRouter
from src.training.envs import make_dino_env
from tests.test_dqn_agent import SMALL as SMALL_DQN
from tests.test_environment import FakeGame
from tests.test_ppo_agent import SMALL as SMALL_PPO

HARD_SPEED = 12.0


class FakeAgent:
    def __init__(self, action):
        self.action = action

    def predict(self, obs):
        return np.array([self.action]), None


@pytest.fixture
def db():
    with MetricsDB(":memory:") as db:
        yield db


def make_manager(db=None, min_tries=1):
    agents = {"dqn": FakeAgent(1), "ppo": FakeAgent(2)}
    return AgentManager(agents, LayaRouter(), db, min_tries=min_tries)


def play_episode(manager, score, speed=6.0):
    """One step at `speed`, then a crash at `score`"""
    manager.get_action(None)
    manager.observe({"score": score / 2, "speed": speed}, done=False)
    manager.observe({"score": score, "speed": speed}, done=True)


def test_untried_agents_play_first():
    manager = make_manager(min_tries=2)

    played = []
    for _ in range(4):
        played.append(manager.get_action(None)[1])
        manager.observe({"score": 100}, done=True)

    assert sorted(played) == ["dqn", "dqn", "ppo", "ppo"]


def test_router_picks_the_agent_that_scores_more(db):
    manager = make_manager(db)
    play_episode(manager, score=100)  # dqn explores
    play_episode(manager, score=300)  # ppo explores

    action, agent = manager.get_action(None)

    assert (agent, action[0]) == ("ppo", 2)
    assert manager.score("ppo", "EASY") == 300
    last = db.history("routing")[-1]
    assert (last["agent"], last["difficulty"], last["source"]) == (
        "ppo",
        "EASY",
        "heuristic",
    )
    assert last["confidence"] > 0.9


def test_difficulty_change_closes_the_stretch_and_decides_again():
    manager = make_manager()
    manager.get_action(None)  # dqn, EASY
    manager.observe({"score": 400, "speed": HARD_SPEED}, done=False)

    assert manager.score("dqn", "EASY") == 400
    assert manager.difficulty == "HARD"
    assert manager.current == "dqn"  # untried at HARD: first agent explores

    manager.observe({"score": 450, "speed": HARD_SPEED}, done=True)
    assert manager.score("dqn", "HARD") == 50  # gained since the stretch began
    assert manager.difficulty == "EASY"  # next episode starts slow


def test_switches_are_counted(db):
    manager = make_manager(db)
    play_episode(manager, score=100)  # dqn
    play_episode(manager, score=300)  # switch to ppo (explore)
    manager.get_action(None)  # ppo again (heuristic)

    assert manager.switches == 1
    assert [r["source"] for r in db.history("routing")] == [
        "explore",
        "explore",
        "heuristic",
    ]


@pytest.fixture
def trained(tmp_path, monkeypatch):
    """Both agents have a checkpoint (tiny, untrained) in tmp_path"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(LocalConfig, "DEVICE", "cpu")
    monkeypatch.setattr(DQNAgent, "default_config", staticmethod(lambda: SMALL_DQN))
    monkeypatch.setattr(PPOAgent, "default_config", staticmethod(lambda: SMALL_PPO))
    env = make_dino_env(game=FakeGame(10))
    checkpoints = CheckpointManager()
    checkpoints.save(DQNAgent(env), "dqn", step=1, reward=0.0)
    checkpoints.save(PPOAgent(env), "ppo", step=1, reward=0.0)
    env.close()
    monkeypatch.setattr(cli, "make_dino_env", lambda: make_dino_env(game=FakeGame(10)))


def test_from_checkpoints_loads_every_agent(trained):
    env = make_dino_env(game=FakeGame(10))

    manager = AgentManager.from_checkpoints(env, LayaRouter())

    assert sorted(manager.agents) == ["dqn", "ppo"]
    env.close()


def test_cli_play_routes_between_agents(trained):
    result = CliRunner().invoke(cli.app, ["play", "--episodes", "4"])

    assert result.exit_code == 0, result.output
    assert "episode 4: score 0" in result.output
    # min_tries=3: agents alternate while exploring (dqn, ppo, dqn, ppo, dqn)
    assert "over 4 episodes, 4 switches" in result.output
    with MetricsDB() as db:
        assert len(db.history("routing")) == 5  # a decision per episode start


def test_cli_play_without_checkpoints_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(LocalConfig, "DEVICE", "cpu")
    monkeypatch.setattr(cli, "make_dino_env", lambda: make_dino_env(game=FakeGame(10)))

    result = CliRunner().invoke(cli.app, ["play"])

    assert result.exit_code == 1
    assert "run `train --all` first" in result.output
