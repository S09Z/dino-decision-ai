"""Tests for training experiments (fake game, no Chrome)"""

import json

import pytest
from typer.testing import CliRunner

from src.config.local_config import LocalConfig
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.training import experiment
from src.training.envs import make_dino_env
from tests.test_environment import FakeGame


@pytest.fixture(autouse=True)
def fake_setup(tmp_path, monkeypatch):
    """Run in tmp_path (models/ goes there) on the fake game and CPU"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(LocalConfig, "DEVICE", "cpu")
    envs = []

    def fake_env(**options):
        envs.append(options)
        return make_dino_env(game=FakeGame(10), **options)

    monkeypatch.setattr(experiment, "make_dino_env", fake_env)
    return envs


def results():
    lines = experiment.RESULTS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_variant_trains_under_its_own_name_and_is_evaluated(fake_setup):
    result = experiment.run("dqn-a2", steps=200, checkpoint_every=100, episodes=2)

    assert fake_setup == [{"n_actions": 2}]
    assert CheckpointManager().latest("dqn-a2").step == 200
    assert CheckpointManager().best("dqn") is None  # baseline untouched
    assert set(result["evaluated"]) == {"best", "latest"}
    assert result["evaluated"]["latest"]["mean_length"] == 10  # FakeGame(10)
    assert results() == [result]


def test_variant_config_overrides_reach_the_agent(monkeypatch):
    built = []

    class Spy(experiment.AGENTS["dqn"]):  # type: ignore[misc,valid-type]
        def __init__(self, env, config=None, device=None):
            built.append(config)
            super().__init__(env, config, device)

    monkeypatch.setitem(experiment.AGENTS, "dqn", Spy)

    experiment.run("dqn-a2-eps", steps=100, checkpoint_every=100, episodes=1)

    assert all(config.exploration_final_eps == 0.01 for config in built)


def test_resume_continues_and_eval_only_skips_training():
    experiment.run("dqn-a2", steps=100, checkpoint_every=100, episodes=1)

    experiment.run("dqn-a2", steps=200, checkpoint_every=100, episodes=1, resume=True)
    assert CheckpointManager().latest("dqn-a2").step == 200

    evaluated = experiment.run("dqn-a2", steps=999, episodes=1, eval_only=True)
    assert CheckpointManager().latest("dqn-a2").step == 200
    assert evaluated["evaluated"]["latest"]["step"] == 200
    assert len(results()) == 3


def test_cli_prints_results_and_rejects_unknown_variants():
    runner = CliRunner()
    app = experiment.typer.Typer()
    app.command()(experiment.main)

    ok = runner.invoke(
        app,
        ["dqn-a2", "--steps", "100", "--checkpoint-every", "100", "--episodes", "1"],
    )
    assert ok.exit_code == 0, ok.output
    assert "dqn-a2 latest (step 100): length 10" in ok.output

    unknown = runner.invoke(app, ["nope"])
    assert unknown.exit_code == 1
    assert "Unknown variant 'nope'" in unknown.output
