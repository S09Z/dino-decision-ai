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


def test_score_averages_only_the_last_window_stretches():
    manager = AgentManager({"dqn": FakeAgent(1)}, LayaRouter(), window=2)

    assert manager.score("dqn", "EASY") == 0  # nothing played yet
    for score in (100, 200, 400):
        play_episode(manager, score)

    assert manager.score("dqn", "EASY") == 300  # 200 and 400; 100 dropped


def test_scores_are_kept_per_difficulty():
    """PPO is better at EASY, DQN at HARD: each plays where it is better"""
    manager = make_manager()
    for agent, easy, hard in (("dqn", 100, 900), ("ppo", 300, 50)):
        assert manager.get_action(None)[1] == agent  # exploring EASY
        manager.observe({"score": easy, "speed": HARD_SPEED}, done=False)
        assert manager.current == agent  # also untried at HARD: explores
        manager.observe({"score": easy + hard, "speed": HARD_SPEED}, done=True)

    assert manager.score("ppo", "EASY") == 300
    assert manager.score("dqn", "HARD") == 900
    assert manager.get_action(None)[1] == "ppo"  # EASY
    manager.observe({"score": 10, "speed": HARD_SPEED}, done=False)
    assert manager.current == "dqn"  # HARD


def test_router_switches_back_when_the_leader_gets_worse(db):
    manager = AgentManager(
        {"dqn": FakeAgent(1), "ppo": FakeAgent(2)}, LayaRouter(), db, window=1
    )
    play_episode(manager, score=100)  # dqn explores
    play_episode(manager, score=300)  # ppo explores
    play_episode(manager, score=50)  # ppo leads, plays, and scores less

    assert manager.get_action(None)[1] == "dqn"
    assert manager.switches == 2  # dqn -> ppo, ppo -> dqn


def test_keeping_the_same_agent_is_not_a_switch():
    manager = make_manager()
    play_episode(manager, score=300)  # dqn explores
    play_episode(manager, score=100)  # ppo explores (1 switch)
    for _ in range(3):
        play_episode(manager, score=300)  # dqn leads and keeps playing

    assert manager.current == "dqn"
    assert manager.switches == 2


def test_every_decision_is_logged_once_with_its_difficulty(db):
    manager = make_manager(db)
    manager.get_action(None)  # decision 1: EASY
    manager.observe({"score": 50, "speed": 9.0}, done=False)  # 2: MEDIUM
    manager.observe({"score": 90, "speed": HARD_SPEED}, done=False)  # 3: HARD
    manager.observe({"score": 95, "speed": HARD_SPEED}, done=False)  # same
    manager.observe({"score": 120, "speed": HARD_SPEED}, done=True)  # 4: EASY

    rows = db.history("routing")
    assert [r["difficulty"] for r in rows] == ["EASY", "MEDIUM", "HARD", "EASY"]
    assert all(r["agent"] in ("dqn", "ppo") for r in rows)
    assert all(0.5 <= r["confidence"] <= 1.0 for r in rows)


def test_failing_classifier_mid_play_falls_back_and_is_logged(db):
    def broken(scores, difficulty):
        raise RuntimeError("out of memory")

    manager = AgentManager(
        {"dqn": FakeAgent(1), "ppo": FakeAgent(2)},
        LayaRouter(classifier=broken),
        db,
        min_tries=1,
    )
    play_episode(manager, score=100)
    play_episode(manager, score=300)

    assert manager.get_action(None)[1] == "ppo"
    assert db.history("routing")[-1]["source"] == "heuristic"


def test_eval_compare_adds_routed_play(trained):
    result = CliRunner().invoke(cli.app, ["eval", "--compare", "--episodes", "4"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()[-4:]
    assert [line.split(":")[0] for line in lines] == ["random", "dqn", "ppo", "routed"]
    assert lines[3].startswith("routed: reward")
    assert lines[3].endswith("length 10 ± 0 steps, 4 switches")  # FakeGame(10)
