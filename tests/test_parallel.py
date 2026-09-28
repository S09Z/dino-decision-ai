"""Tests for parallel training (fake processes, no Chrome)"""

import subprocess
import sys

import pytest
from typer.testing import CliRunner

from src.cli import main as cli
from src.monitoring.local_db import MetricsDB
from src.training import parallel
from src.training.parallel import (
    ram_warning,
    status_line,
    train_command,
    train_parallel,
)


class FakeProcess:
    """Popen stand-in that finishes after `runs_for` waits"""

    def __init__(self, command, runs_for=1, returncode=0):
        self.command = command
        self.runs_for = runs_for
        self.final_code = returncode
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None and self.runs_for > 0 and timeout is not None:
            self.runs_for -= 1
            raise subprocess.TimeoutExpired(self.command, timeout)
        if self.returncode is None:
            self.returncode = self.final_code
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MetricsDB goes to tmp_path/models/logs


def test_train_command_runs_one_agent_without_progress_bar():
    command = train_command("ppo", steps=100, checkpoint_every=50)

    assert command[:3] == [sys.executable, "-m", "src.cli.main"]
    assert command[3:] == [
        "train",
        "--agent",
        "ppo",
        "--steps",
        "100",
        "--checkpoint-every",
        "50",
        "--no-progress-bar",
        "--n-envs",
        "1",
    ]


def test_train_command_passes_resume():
    command = train_command("dqn", steps=100, checkpoint_every=50, resume=True)

    assert command[-1] == "--resume"


def test_train_parallel_launches_every_agent_and_returns_exit_codes():
    launched = []

    def launch(command):
        agent = command[command.index("--agent") + 1]
        launched.append(FakeProcess(command, returncode=1 if agent == "ppo" else 0))
        return launched[-1]

    codes = train_parallel(["dqn", "ppo"], 100, 50, launch=launch, poll_seconds=0)

    assert codes == {"dqn": 0, "ppo": 1}
    assert len(launched) == 2


def test_train_parallel_logs_status_while_running(monkeypatch):
    logged = []
    monkeypatch.setattr(parallel.logger, "info", lambda *args: logged.append(args))
    with MetricsDB() as db:
        db.add_episode("dqn", 1, reward=-90.0, length=70)

    train_parallel(
        ["dqn"], 100, 50, launch=lambda c: FakeProcess(c, runs_for=2), poll_seconds=0
    )

    assert len(logged) == 2
    assert "dqn: 1 episodes, last-10 reward -90.0" in logged[0][1]


def test_stopping_the_parent_terminates_children(monkeypatch):
    launched = []

    def launch(command):
        launched.append(FakeProcess(command, runs_for=10))
        return launched[-1]

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(parallel, "status_line", interrupt)

    with pytest.raises(KeyboardInterrupt):
        train_parallel(["dqn", "ppo"], 100, 50, launch=launch, poll_seconds=0)
    assert all(p.terminated for p in launched)


def test_status_line_without_episodes():
    with MetricsDB(":memory:") as db:
        assert status_line(db, ["ppo"]) == "ppo: 0 episodes, last-10 reward -"


def test_ram_warning_only_when_free_ram_is_short():
    assert ram_warning(["dqn", "ppo"], available_gb=8.0) is None
    warning = ram_warning(["dqn", "ppo"], available_gb=2.4)
    assert "~4.0GB RAM but 2.4GB is free" in warning


def test_cli_train_all_parallel(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "train_parallel",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"dqn": 0, "ppo": 0},
    )

    result = CliRunner().invoke(
        cli.app, ["train", "--all", "--parallel", "--steps", "10"]
    )
    resumed = CliRunner().invoke(
        cli.app, ["train", "--all", "--parallel", "--steps", "10", "--resume"]
    )

    assert result.exit_code == resumed.exit_code == 0, result.output
    assert calls == [
        ((["dqn", "ppo"], 10, 5_000), {"resume": False, "n_envs": 1}),
        ((["dqn", "ppo"], 10, 5_000), {"resume": True, "n_envs": 1}),
    ]
    assert "dqn: done" in result.output


def test_cli_parallel_reports_failures_and_rejects_profile(monkeypatch):
    monkeypatch.setattr(
        cli, "train_parallel", lambda *args, **kwargs: {"dqn": 0, "ppo": 2}
    )
    runner = CliRunner()

    failed = runner.invoke(cli.app, ["train", "--all", "--parallel"])
    assert failed.exit_code == 1
    assert "ppo: failed (exit 2)" in failed.output

    rejected = runner.invoke(cli.app, ["train", "--all", "--parallel", "--profile"])
    assert rejected.exit_code == 1
    assert "--profile does not work with --parallel" in rejected.output
