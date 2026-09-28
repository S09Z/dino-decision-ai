"""Training on several games at once (fake games, no Chrome)"""

from dataclasses import replace

import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from typer.testing import CliRunner

from src.cli import main as cli
from src.config.dqn_config import DQNConfig
from src.config.local_config import LocalConfig
from src.config.ppo_config import PPOConfig
from src.environment.dino_env import ChromeDinoEnv
from src.models.dqn_agent import DQNAgent
from src.models.ppo_agent import PPOAgent
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from src.training.callbacks import TrainingMonitor
from src.training.envs import make_dino_env
from src.training.parallel import ram_warning, train_command
from tests.test_environment import FakeGame

DQN_SMALL = replace(DQNConfig(), buffer_size=200, batch_size=8)
PPO_SMALL = replace(PPOConfig(), n_steps=64, batch_size=16)


def games(n, crash_after=10):
    """`n` fake games side by side, in this process"""
    env = DummyVecEnv(
        [lambda: ChromeDinoEnv(step_seconds=0, game=FakeGame(crash_after))] * n
    )
    return VecFrameStack(env, 4, channels_order="last")


def test_several_games_run_in_their_own_processes():
    env = make_dino_env(game=FakeGame(crash_after=5), n_envs=2)
    try:
        assert isinstance(env.venv, SubprocVecEnv)
        assert env.reset().shape == (2, 84, 84, 4)
        env.step([1, 0])
        env.env_method("pause")  # reaches every game, as pauses must
        assert env.get_attr("_game")[0].paused
    finally:
        env.close()


def test_one_game_stays_in_process():
    env = make_dino_env(game=FakeGame())

    assert isinstance(env.venv, DummyVecEnv)
    env.close()


def test_dqn_trains_on_every_transition_it_collects():
    assert DQNAgent(games(1), DQN_SMALL, "cpu").model.gradient_steps == 1
    assert DQNAgent(games(4), DQN_SMALL, "cpu").model.gradient_steps == 4


def test_ppo_keeps_its_samples_per_update():
    one = PPOAgent(games(1), PPO_SMALL, "cpu").model
    two = PPOAgent(games(2), PPO_SMALL, "cpu").model

    assert one.n_steps * one.n_envs == two.n_steps * two.n_envs == 64


def test_loading_keeps_the_settings_for_this_many_games(tmp_path):
    DQNAgent(games(1), DQN_SMALL, "cpu").save(tmp_path / "dqn.zip")
    PPOAgent(games(1), PPO_SMALL, "cpu").save(tmp_path / "ppo.zip")
    dqn = DQNAgent(games(3), DQN_SMALL, "cpu")
    ppo = PPOAgent(games(2), PPO_SMALL, "cpu")

    dqn.load(tmp_path / "dqn.zip")
    ppo.load(tmp_path / "ppo.zip")

    assert dqn.model.gradient_steps == 3  # not the checkpoint's 1
    assert ppo.model.n_steps == 32
    assert ppo.model.rollout_buffer.buffer_size == 32


def test_resume_on_more_games_starts_a_new_replay_buffer(tmp_path):
    checkpoints = CheckpointManager(tmp_path)
    one = DQNAgent(games(1), DQN_SMALL, "cpu")
    one.train(40)
    checkpoints.save_latest(one.model, "dqn", one.model.num_timesteps, episodes=4)

    two = DQNAgent(games(2), DQN_SMALL, "cpu")
    two.resume(checkpoints.latest("dqn"))

    assert two.model.num_timesteps == 40
    assert two.model.replay_buffer.n_envs == 2
    two.train(20)  # SB3 rounds up to whole rollouts (4 steps x 2 games)
    assert two.model.num_timesteps >= 60
    assert two.model.replay_buffer.size() > 0  # took the two games' transitions


def test_checkpoints_every_n_steps_when_steps_come_in_threes(tmp_path):
    checkpoints = CheckpointManager(tmp_path)
    with MetricsDB(":memory:") as db:
        monitor = TrainingMonitor("dqn", db, checkpoints, checkpoint_every=10)
        DQNAgent(games(3), DQN_SMALL, "cpu").train(36, callback=monitor)
        steps = [row["step"] for row in db.history("training", "dqn")]

    # at least 10 steps apart; exact multiples of 10 would come only every 30
    assert steps == [12, 24, 36]
    assert checkpoints.latest("dqn").step == 36


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """The CLI on fake games; returns the n_envs each env was built with"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(LocalConfig, "DEVICE", "cpu")
    built = []

    def fake_env(n_envs=1, **options):
        built.append(n_envs)
        return games(n_envs)

    monkeypatch.setattr(cli, "make_dino_env", fake_env)
    return built


def test_cli_trains_on_n_games(cli_env):
    result = CliRunner().invoke(
        cli.app,
        ["train", "--steps", "40", "--n-envs", "2", "--no-progress-bar"],
    )

    assert result.exit_code == 0, result.output
    assert cli_env == [2]


def test_cli_profile_needs_one_game(cli_env):
    result = CliRunner().invoke(cli.app, ["train", "--profile", "--n-envs", "2"])

    assert result.exit_code == 1
    assert cli_env == []


def test_parallel_children_get_the_game_count():
    assert train_command("dqn", 100, 50, n_envs=4)[-2:] == ["--n-envs", "4"]


def test_ram_warning_counts_the_extra_games():
    assert ram_warning(["dqn", "ppo"], available_gb=5.0) is None
    assert "needs ~6.0GB" in ram_warning(["dqn", "ppo"], 5.0, n_envs=3)
