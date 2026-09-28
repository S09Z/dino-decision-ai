"""Tests for Laya as the player (a fake Laya and a scripted game, no Chrome)"""

import io
import json

import pytest
from typer.testing import CliRunner

from src.cli import main as cli
from src.routing import laya_player_benchmark
from src.routing.laya_player import (
    QUESTIONS,
    LayaPlayer,
    RulePlayer,
    answer,
    describe,
    kind,
    play,
    record,
    threat,
    truth,
    window,
)


def cactus(d, w=17):
    return {"type": "cactus", "d": d, "w": w, "y": 105, "h": 35}


def bird(d, y):
    return {"type": "bird", "d": d, "w": 46, "y": y, "h": 40}


def state(*obstacles, speed=8.0, jumping=False, crashed=False, score=100):
    return {
        "crashed": crashed,
        "score": score,
        "speed": speed,
        "jumping": jumping,
        "ducking": False,
        "obstacles": list(obstacles),
    }


def test_birds_are_jumped_ducked_or_ignored_by_height():
    assert kind(cactus(50)) == "cactus"
    assert kind(bird(50, y=100)) == "low bird"  # bottom 140: hits a ducking dino
    assert kind(bird(50, y=75)) == "bird at head height"  # bottom 115
    assert kind(bird(50, y=50)) is None  # bottom 90: flies over


def test_threat_skips_birds_that_fly_over():
    near = threat(state(bird(20, y=50), cactus(90)))

    assert near["kind"] == "cactus" and near["d"] == 90


def test_window_grows_with_speed_and_shrinks_for_wide_obstacles():
    assert window(6, lead=14) == 84
    assert window(13, lead=14) == 182
    assert window(8, width=75, lead=14) == 112 - 38  # jump later over a group


def test_sentence_states_how_distance_compares_with_the_window():
    near = describe(state(cactus(88)), lead=14)  # window 112 - 8 = 104
    far = describe(state(cactus(150)), lead=14)

    assert near == (
        "The dinosaur is on the ground, running at speed 8.0. Nearest obstacle:"
        " a cactus. Distance to the cactus: 88 pixels, which is less than the"
        " 104-pixel window."
    )
    assert "150 pixels, which is more than the 104-pixel window" in far


def test_sentence_when_nothing_is_ahead_or_in_the_air():
    text = describe(state(bird(40, y=50), jumping=True))

    assert text == (
        "The dinosaur is in the air, running at speed 8.0. Nothing to jump over"
        " or duck under is ahead."
    )


def test_truth_answers_jump_or_duck_inside_the_window_only():
    assert truth(state(cactus(50)), lead=14) == (True, False)
    assert truth(state(bird(50, y=100)), lead=14) == (True, False)
    assert truth(state(bird(50, y=75)), lead=14) == (False, True)
    assert truth(state(cactus(200)), lead=14) == (False, False)
    assert truth(state()) == (False, False)


def test_answer_needs_a_detector_at_the_cut():
    assert answer((0.69, 0.2)) == "hold"
    assert answer((0.7, 0.2)) == "jump"
    assert answer((0.75, 0.9)) == "duck"
    assert answer((0.5, 0.2), cut=0.5) == "jump"


def test_rule_player_presses_the_answer():
    player = RulePlayer(lead=14)

    jump = player.decide(state(cactus(50)))
    duck = player.decide(state(bird(50, y=75)))
    hold = player.decide(state(cactus(300)))

    assert (jump.laya, jump.exec, jump.action) == ("jump", "jump", 1)
    assert (duck.exec, duck.action) == ("duck", 2)
    assert (hold.exec, hold.action) == ("hold", 0)


def test_no_key_in_the_air_but_the_answer_is_kept():
    decision = RulePlayer(lead=14).decide(state(cactus(50), jumping=True))

    assert decision.laya == "jump"
    assert (decision.exec, decision.action) == ("skip:air", 0)


class FakeLaya:
    """Answers like laya's Agent.system_one for two noul questions"""

    def __init__(self, jump=0.9, duck=0.1):
        self.jump, self.duck = jump, duck
        self.states: list = []

    def system_one(self, state, questions):
        assert questions == QUESTIONS
        self.states.append(state)
        return {
            "answers": {
                "jump": {"type": "noul", "noul": self.jump},
                "duck": {"type": "noul", "noul": self.duck},
            }
        }


def test_laya_player_asks_about_the_sentence():
    fake = FakeLaya(jump=0.94, duck=0.11)
    player = LayaPlayer(lead=14, agent=fake)

    decision = player.decide(state(cactus(88)))

    assert fake.states == [describe(state(cactus(88)), lead=14)]
    assert decision.det == (0.94, 0.11)
    assert (decision.laya, decision.action) == ("jump", 1)


def test_laya_player_loads_the_model_once(monkeypatch):
    import laya

    loads = []
    monkeypatch.setattr(
        laya, "load", lambda model, device: loads.append(1) or FakeLaya()
    )
    player = LayaPlayer()

    player.decide(state())
    player.decide(state())

    assert len(loads) == 1


def test_record_has_the_log_fields():
    s = state(cactus(-5), jumping=True)
    line = record(s, RulePlayer().decide(s), seconds=9.84)

    assert line["t"] == 9.84
    assert (line["obs"], line["d"], line["ow"], line["oy"]) == ("cactus", -5, 17, 105)
    assert (line["air"], line["laya"], line["exec"]) == (1, "jump", "skip:air")
    assert line["det"] == [1.0, 0.0]
    assert line["ahead"] == [cactus(-5)]
    assert record(state(), RulePlayer().decide(state()), 0)["d"] == 999


class ScriptedGame:
    """Plays back one list of states per episode"""

    def __init__(self, episodes):
        self.episodes = episodes
        self.states: list = []
        self.actions: list = []
        self.closed = False

    def restart(self):
        self.states = list(self.episodes.pop(0))

    def state(self):
        return self.states.pop(0)

    def act(self, action):
        self.actions.append(action)

    def close(self):
        self.closed = True


def test_play_logs_every_decision_and_scores_each_game():
    game = ScriptedGame(
        [
            [state(cactus(300)), state(cactus(50)), state(crashed=True, score=42)],
            [state(), state(crashed=True, score=7)],
        ]
    )
    log = io.StringIO()

    results = play(game, RulePlayer(lead=14), 2, log, echo=lambda _: None)

    assert [r["score"] for r in results] == [42, 7]
    assert [r["decisions"] for r in results] == [2, 1]
    assert game.actions == [0, 1, 0]
    lines = [json.loads(line) for line in log.getvalue().splitlines()]
    assert [line["exec"] for line in lines] == ["hold", "jump", "hold"]


@pytest.fixture
def played(tmp_path, monkeypatch):
    """The CLI with a scripted game instead of Chrome; returns what it made"""
    made: dict = {}

    def fake_chrome(headless, game):
        made.update(headless=headless, game=game)
        made["chrome"] = ScriptedGame([[state(), state(crashed=True, score=10)]])
        return made["chrome"]

    monkeypatch.setattr(cli, "ChromeGame", fake_chrome)
    monkeypatch.setattr(cli, "LOG_PATH", tmp_path / "logs" / "play.jsonl")
    return made


def test_cli_rule_player_plays_and_logs(played, tmp_path):
    result = CliRunner().invoke(
        cli.app, ["play", "--player", "rule", "--episodes", "1", "--game", "chrome"]
    )

    assert result.exit_code == 0, result.output
    assert "rule: mean score 10.0 over 1 episodes" in result.output
    assert played == {"headless": True, "game": "chrome", "chrome": played["chrome"]}
    assert played["chrome"].closed
    assert (tmp_path / "logs" / "play.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 1


def test_cli_laya_player_loads_laya_up_front(played, monkeypatch):
    fake = FakeLaya(jump=0.2, duck=0.1)
    monkeypatch.setattr(cli, "LayaPlayer", lambda lead: LayaPlayer(lead, agent=fake))

    result = CliRunner().invoke(
        cli.app, ["play", "--player", "laya", "--episodes", "1", "--show"]
    )

    assert result.exit_code == 0, result.output
    assert len(fake.states) == 1
    assert played["headless"] is False


def test_cli_rejects_an_unknown_player(played):
    result = CliRunner().invoke(cli.app, ["play", "--player", "dqn"])

    assert result.exit_code == 1
    assert "chrome" not in played  # nothing launched


class ChoiceOrNoulLaya(FakeLaya):
    """Also answers the benchmark's choice question: always `choice`"""

    def __init__(self, choice="hold", **kwargs):
        super().__init__(**kwargs)
        self.choice = choice

    def system_one(self, state, questions):
        if questions == laya_player_benchmark.CHOICE:
            return {"answers": {"action": {"type": "choice", "choice": self.choice}}}
        return super().system_one(state, questions)


def test_scenario_names_match_their_answers():
    for name, s in laya_player_benchmark.SCENARIOS:
        want = laya_player_benchmark.expected(s)
        if "inside" in name or "in the air" in name:
            assert want != "hold", name
        else:
            assert want == "hold", name


def test_benchmark_counts_answers_that_match_the_rule():
    scenarios = laya_player_benchmark.SCENARIOS
    holds = sum(laya_player_benchmark.expected(s) == "hold" for _, s in scenarios)
    fake = ChoiceOrNoulLaya(choice="hold", jump=0.1, duck=0.1)  # always hold

    results = laya_player_benchmark.benchmark(fake, repeats=1)

    assert list(results) == list(laya_player_benchmark.STYLES)
    assert all(r["correct"] == holds for r in results.values())
    assert isinstance(fake.states[0], dict)  # raw numbers first
    assert isinstance(fake.states[-1], str)  # the sentence last


def test_benchmark_report_marks_wrong_answers():
    fake = ChoiceOrNoulLaya(choice="jump", jump=0.1, duck=0.1)
    results = laya_player_benchmark.benchmark(fake, repeats=1)

    text = laya_player_benchmark.report(results, device=None, repeats=1)

    assert text.startswith("# Laya player benchmark")
    assert "| nothing ahead | hold | jump (wrong) | hold | hold |" in text
    assert "| cactus inside | jump | jump | hold (wrong) | hold (wrong) |" in text
