"""Tests for training experiments (fake game, no Chrome)"""

import json

import pytest
from typer.testing import CliRunner

from src.config.local_config import LocalConfig
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
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

    # a training env, then a one-game env for the evaluation
    assert fake_setup == [{"n_envs": 1, "n_actions": 2}, {"n_actions": 2}]
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


def test_watch_plays_the_best_checkpoint_in_a_visible_chrome(fake_setup):
    experiment.run("dqn-a2", steps=200, checkpoint_every=100, episodes=1)
    fake_setup.clear()

    lengths = experiment.watch("dqn-a2", episodes=2, echo=lambda _: None)

    assert lengths == [10, 10]  # FakeGame(10)
    assert fake_setup == [{"render_mode": "human", "n_actions": 2}]
    with MetricsDB() as db:
        assert db.episode_count("dqn-a2-watch") == 2  # shown on the dashboard


def test_watch_without_a_checkpoint_says_to_train_first():
    app = experiment.typer.Typer()
    app.command()(experiment.main)

    result = CliRunner().invoke(app, ["dqn-a2", "--watch"])

    assert result.exit_code == 1
    assert "No best checkpoint for dqn-a2; train it first" in result.output


def test_watch_records_what_the_agent_thinks_of_each_action(fake_setup):
    experiment.run("dqn-a2", steps=200, checkpoint_every=100, episodes=1)

    experiment.watch("dqn-a2", episodes=1, echo=lambda _: None)

    with MetricsDB() as db:
        rows = db.history("decisions", "dqn-a2-watch")
    assert len(rows) == 9  # every step of the 10-step game but the first
    scores = json.loads(rows[0]["scores"])
    assert set(scores) == {"hold", "jump"}  # dqn-a2 has two actions
    assert rows[0]["kind"] == "q_value"
    assert rows[0]["answer"] == max(scores, key=scores.get)  # greedy
    assert rows[0]["obstacle"] == "cactus"


def test_ppo_scores_are_action_probabilities():
    from src.config.ppo_config import PPOConfig
    from src.models.ppo_agent import PPOAgent

    env = make_dino_env(game=FakeGame(10), n_actions=2)
    model = PPOAgent(env, PPOConfig(n_steps=64, batch_size=16), "cpu").model

    scores, kind = experiment.action_scores(model, env.reset())

    assert kind == "probability"
    assert abs(sum(scores.values()) - 1) < 1e-5


def test_lockstep_variants_copy_each_variant_with_frames_per_step():
    lockstep = experiment.VARIANTS["dqn-a2-eps-crop-ls4"]
    real_time = experiment.VARIANTS["dqn-a2-eps-crop"]

    assert lockstep.env == {**real_time.env, "frames_per_step": 4}
    assert lockstep.config == real_time.config
    assert "frames_per_step" not in real_time.env


def test_watching_a_lockstep_variant_is_paced_to_real_time(fake_setup, monkeypatch):
    experiment.run("dqn-a2-ls4", steps=200, checkpoint_every=100, episodes=1)
    naps = []
    monkeypatch.setattr(experiment.time, "sleep", naps.append)

    experiment.watch("dqn-a2-ls4", episodes=1, echo=lambda _: None)

    assert len(naps) == 10  # one per step of the 10-step game
    assert all(0 < nap <= 4 / 60 for nap in naps)
