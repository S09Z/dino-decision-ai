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


def test_training_only_options_stay_out_of_the_evaluation(fake_setup):
    name = "dqn-a2-eps-crop-ls4-speed"
    variant = experiment.VARIANTS[name]
    assert variant.train == {"start_speed": (6, 13)}

    experiment.run(name, steps=100, checkpoint_every=100, episodes=1)

    training, evaluation = fake_setup
    assert training == {"n_envs": 1, **variant.env, "start_speed": (6, 13)}
    assert evaluation == variant.env  # the normal game, like every variant
    assert results()[0]["train"] == {"start_speed": [6, 13]}


def test_evaluation_records_what_ended_each_game():
    result = experiment.run("dqn-a2", steps=100, checkpoint_every=100, episodes=2)

    latest = result["evaluated"]["latest"]
    assert latest["median_length"] == 10 and latest["over_1000"] == 0
    assert latest["hits"] == {"cactus": 2}  # FakeGame's only obstacle
    assert latest["died_in_air"] == 0
    assert latest["death_speed"] == 6.0
    assert [game["length"] for game in latest["games"]] == [10, 10]


def test_evaluate_games_stops_a_game_at_max_length():
    from src.evaluation.evaluate import evaluate_games

    env = make_dino_env(game=FakeGame(crash_after=1000))
    model = experiment.AGENTS["dqn"](env)

    games = evaluate_games(model.model, env, 2, max_length=5)

    assert [g["length"] for g in games] == [5, 5]
    assert [g["hit"] for g in games] == ["nothing (stopped)"] * 2
    assert games[0]["reward"] == pytest.approx(0.5)  # 5 steps alive, no crash


def test_eval_every_evaluates_while_training_and_keeps_that_pick(fake_setup):
    name = "dqn-a2-eps-crop-ls4-mix"

    result = experiment.run(
        name, steps=200, checkpoint_every=100, episodes=1, eval_every=100
    )

    # training games, the evaluation game while training, the final one
    assert len(fake_setup) == 3 and fake_setup[1] == fake_setup[2]
    assert [step for step, _, _ in result["eval_history"]] == [100, 200]
    assert CheckpointManager().best(f"{name}-eval").reward == 10  # FakeGame(10)
    # the evaluation's pick replaces the best by training reward
    assert set(result["evaluated"]) == {"latest", "eval_best"}


def test_duck_variant_is_mix_with_three_actions():
    mix = experiment.VARIANTS["dqn-a2-eps-crop-ls4-mix"]
    duck = experiment.VARIANTS["dqn-a3-eps-crop-ls4-mix"]

    assert duck.env == {**mix.env, "n_actions": 3}
    assert (duck.config, duck.train) == (mix.config, mix.train)


def test_cost_variants_add_a_press_cost_to_training_only():
    mix = experiment.VARIANTS["dqn-a2-eps-crop-ls4-mix"]
    for name, actions in (
        ("dqn-a2-eps-crop-ls4-mix-cost", 2),
        ("dqn-a3-eps-crop-ls4-mix-cost", 3),
    ):
        variant = experiment.VARIANTS[name]
        assert variant.train == {**mix.train, "press_cost": 0.05}
        assert variant.env == {**mix.env, "n_actions": actions}


def test_projected_score_extends_play_at_top_speed_by_its_death_rate():
    from src.evaluation.evaluate import STOPPED, projected_score

    games = [
        # reached top speed at 2,000 points; stopped at 12,000 (10,000 more)
        {"score": 12_000, "top_score": 2_000, "hit": STOPPED},
        # reached it at 2,000, died at 7,000 (5,000 more)
        {"score": 7_000, "top_score": 2_000, "hit": "cactus"},
        # died before top speed: counts as it is
        {"score": 500, "top_score": None, "hit": "cactus"},
    ]

    # 15,000 points at top speed per 1 death there
    assert projected_score(games) == pytest.approx((500 + 2 * 17_000) / 3)
    assert projected_score(games[:1]) is None  # no death at top speed yet
    assert projected_score(games[2:]) == 500


def test_evaluate_games_notes_the_score_at_top_speed():
    from src.evaluation.evaluate import evaluate_games

    env = make_dino_env(game=FakeGame(10))
    games = evaluate_games(experiment.AGENTS["dqn"](env).model, env, 1)

    assert games[0]["top_score"] is None  # FakeGame runs at speed 6


def test_a_tagged_name_repeats_a_variant_with_its_own_checkpoints(fake_setup):
    result = experiment.run("dqn-a2@2", steps=100, checkpoint_every=100, episodes=1)

    assert fake_setup[0] == {"n_envs": 1, "n_actions": 2}  # dqn-a2's settings
    assert CheckpointManager().latest("dqn-a2@2").step == 100
    assert CheckpointManager().latest("dqn-a2") is None  # the original untouched
    assert result["variant"] == "dqn-a2@2"


def test_eval_every_needs_a_lockstep_variant():
    with pytest.raises(ValueError, match="lockstep"):
        experiment.run("dqn-a2", steps=100, eval_every=50)


def test_evaluate_games_names_birds_by_height():
    from src.evaluation.evaluate import evaluate_games

    class Crash:
        def reset(self):
            return None

        def step(self, action):
            info = {
                "episode": {"r": -100.0, "l": 1},
                "score": 5,
                "speed": 9.0,
                "jumping": True,
                "obstacles": [birds.pop(0)] if birds else [],
            }
            return None, [-100.0], [True], [info]

    class Policy:
        def predict(self, obs, deterministic=False):
            return [0], None

    birds = [
        {"type": "bird", "d": 0, "w": 46, "y": 50, "h": 40},  # over a runner
        {"type": "bird", "d": 0, "w": 46, "y": 75, "h": 40},
    ]

    games = evaluate_games(Policy(), Crash(), 3)

    assert [g["hit"] for g in games] == [
        "high bird",
        "bird at head height",
        "unknown",
    ]
    assert all(g["in_air"] for g in games)


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
    assert "died on: cactus 1; 0 in the air" in ok.output

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


def test_closing_the_watch_window_stops_and_keeps_the_games_played(monkeypatch):
    from playwright.sync_api import Error as PlaywrightError

    experiment.run("dqn-a2", steps=100, checkpoint_every=100, episodes=1)
    presses = []

    def act(self, action):
        presses.append(action)
        if len(presses) > 15:  # in game 2: FakeGame(10) ends a game in 10
            raise PlaywrightError("Target page, context or browser has been closed")
        self.actions.append(action)

    monkeypatch.setattr(FakeGame, "act", act)
    lines: list[str] = []

    assert experiment.watch("dqn-a2", episodes=5, echo=lines.append) == [10]
    assert lines[-1] == "Chrome was closed: stopped"


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
