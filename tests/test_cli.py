"""Tests for the CLI: train/eval end to end on the fake game"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli import main as cli
from src.config.local_config import LocalConfig
from src.models.ppo_agent import PPOAgent
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from src.training.envs import make_dino_env
from tests.test_environment import FakeGame

runner = CliRunner()


@pytest.fixture(autouse=True)
def fake_setup(tmp_path, monkeypatch):
    """Run in tmp_path (models/ goes there) on the fake game and CPU"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "make_dino_env", lambda: make_dino_env(game=FakeGame(10)))
    monkeypatch.setattr(LocalConfig, "DEVICE", "cpu")


def test_train_records_metrics_and_saves_checkpoints():
    result = runner.invoke(
        cli.app, ["train", "--steps", "300", "--checkpoint-every", "100"]
    )

    assert result.exit_code == 0, result.output
    assert "300 steps, 30 episodes" in result.output
    with MetricsDB() as db:
        assert len(db.query_recent_episodes(agent="dqn", limit=100)) == 30
        training = db.conn.execute("SELECT step FROM training").fetchall()
        assert [row[0] for row in training] == [100, 200, 300]
        assert db.conn.execute("SELECT COUNT(*) FROM performance").fetchone()[0] == 3
    best = CheckpointManager().best("dqn")
    assert best is not None and best.step in (100, 200, 300)


def test_eval_loads_best_checkpoint():
    runner.invoke(cli.app, ["train", "--steps", "100", "--checkpoint-every", "100"])

    result = runner.invoke(cli.app, ["eval", "--episodes", "2"])

    assert result.exit_code == 0, result.output
    assert "Loaded models/checkpoints/dqn_step100.zip" in result.output
    assert "dqn: reward" in result.output


def test_eval_without_checkpoint_fails_clearly():
    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 1
    assert "run `train` first" in result.output


@pytest.fixture
def small_ppo(monkeypatch):
    from tests.test_ppo_agent import SMALL

    monkeypatch.setattr(PPOAgent, "default_config", staticmethod(lambda: SMALL))


def test_train_ppo(small_ppo):
    result = runner.invoke(
        cli.app,
        ["train", "--agent", "ppo", "--steps", "64", "--checkpoint-every", "32"],
    )

    assert result.exit_code == 0, result.output
    assert CheckpointManager().best("ppo") is not None


def test_unknown_agent_is_rejected():
    result = runner.invoke(cli.app, ["train", "--agent", "a2c"])

    assert result.exit_code == 1
    assert "available: dqn, ppo" in result.output


def test_train_all_trains_every_agent(small_ppo):
    result = runner.invoke(
        cli.app, ["train", "--all", "--steps", "64", "--checkpoint-every", "32"]
    )

    assert result.exit_code == 0, result.output
    assert "Trained dqn for 64 steps" in result.output
    assert "Trained ppo for 64 steps" in result.output


def test_train_profile_prints_step_timings():
    result = runner.invoke(
        cli.app, ["train", "--steps", "50", "--checkpoint-every", "50", "--profile"]
    )

    assert result.exit_code == 0, result.output
    assert "| `ChromeDinoEnv.step` |" in result.output  # DQN rounds up to train_freq
    assert "`FakeGame.frame`" in result.output


def test_eval_compare_lists_random_and_every_agent():
    runner.invoke(cli.app, ["train", "--steps", "50", "--checkpoint-every", "50"])

    result = runner.invoke(cli.app, ["eval", "--compare", "--episodes", "2"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()[-3:]
    assert lines[0].startswith("random: reward")
    assert lines[1].startswith("dqn: reward")
    assert lines[2] == "ppo: no checkpoint (run `train --agent ppo`)"


def test_inspect_shows_checkpoints_and_episodes():
    runner.invoke(cli.app, ["train", "--steps", "50", "--checkpoint-every", "50"])

    result = runner.invoke(cli.app, ["inspect"])

    assert result.exit_code == 0, result.output
    assert "dqn: 5 episodes recorded" in result.output
    assert "step      50" in result.output
    assert "models/checkpoints/dqn_step50.zip" in result.output
    assert "ppo: 0 episodes recorded\n  no checkpoints" in result.output


def test_clean_keeps_best_n():
    runner.invoke(cli.app, ["train", "--steps", "40", "--checkpoint-every", "10"])
    assert len(CheckpointManager().checkpoints("dqn")) == 4

    result = runner.invoke(cli.app, ["clean", "--keep", "1"])

    assert result.exit_code == 0, result.output
    assert "dqn: removed 3, kept 1" in result.output
    assert len(list(Path("models/checkpoints").glob("*.zip"))) == 1
