"""Tests for the CLI: train/eval end to end on the fake game"""

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


def test_train_ppo(monkeypatch):
    from tests.test_ppo_agent import SMALL

    monkeypatch.setattr(PPOAgent, "default_config", staticmethod(lambda: SMALL))
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
