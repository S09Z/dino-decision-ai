"""Tests for Laya as the router's classifier (a fake Laya, no download)"""

import laya
import pytest
from typer.testing import CliRunner

from src.cli import main as cli
from src.monitoring.local_db import MetricsDB
from src.routing import laya_benchmark
from src.routing.laya_classifier import QUESTIONS, LayaClassifier
from src.routing.laya_router import LayaRouter
from src.training.envs import make_dino_env
from tests.test_agent_manager import trained  # noqa: F401 (fixture)
from tests.test_environment import FakeGame


class FakeLaya:
    """Answers like laya's Agent.system_one: always `choice`"""

    def __init__(self, choice="ppo", confidence=0.8, game=None):
        self.choice, self.confidence, self.game = choice, confidence, game
        self.states: list[dict] = []
        self.paused: list[bool] = []

    def system_one(self, state, questions):
        assert questions == QUESTIONS
        self.states.append(state)
        if self.game is not None:
            self.paused.append(self.game.paused)
        return {
            "answers": {
                "agent": {
                    "type": "choice",
                    "choice": self.choice,
                    "confidence": 0.1,  # uncalibrated; answer_confidence is used
                    "answer_confidence": self.confidence,
                }
            }
        }


def test_classifier_asks_laya_about_scores_and_difficulty():
    fake = FakeLaya("ppo", 0.8)

    decision = LayaClassifier(agent=fake)({"dqn": 1850.0, "ppo": 2340.0}, "HARD")

    assert decision == ("ppo", 0.8)
    assert fake.states == [
        {
            "game": "Chrome Dino",
            "difficulty": "HARD",
            "dqn_score": 1850.0,
            "ppo_score": 2340.0,
        }
    ]


def test_model_loads_once_on_first_use(monkeypatch):
    loads = []
    monkeypatch.setattr(
        laya, "load", lambda model, device: loads.append((model, device)) or FakeLaya()
    )
    classifier = LayaClassifier(device="cpu")

    classifier({"dqn": 1.0, "ppo": 2.0}, "EASY")
    classifier({"dqn": 1.0, "ppo": 2.0}, "EASY")

    assert loads == [("convaiinnovations/laya", "cpu")]


def test_router_uses_laya_and_falls_back_when_it_fails():
    router = LayaRouter(LayaClassifier(agent=FakeLaya("dqn", 0.7)))
    assert router.decide_agent(100, 500, "EASY") == ("dqn", 0.7)
    assert router.last_source == "laya"

    broken = LayaClassifier(agent=object())  # no system_one
    router = LayaRouter(broken)
    assert router.decide_agent(100, 500, "EASY")[0] == "ppo"
    assert router.last_source == "heuristic"


def test_benchmark_reports_agreement_with_the_heuristic(tmp_path, monkeypatch):
    fake = FakeLaya("ppo")
    monkeypatch.setattr(
        laya_benchmark, "LayaClassifier", lambda device: LayaClassifier(agent=fake)
    )
    out = tmp_path / "LAYA.md"

    laya_benchmark.main(out=out, repeats=2)

    report = out.read_text(encoding="utf-8")
    # ppo wins 3 of 8 score pairs per difficulty under the heuristic
    assert "**Agrees with the score heuristic:** 38% of 24 cases" in report
    assert "| HARD | 1850 | 2340 | ppo | 0.80 | ppo | 0.89 |" in report
    assert len(fake.states) == 24 * 2


def test_cli_play_with_laya_pauses_the_game_while_it_decides(trained, monkeypatch):
    game = FakeGame(10)
    fake = FakeLaya("ppo", 0.9, game=game)
    monkeypatch.setattr(cli, "make_dino_env", lambda: make_dino_env(game=game))
    monkeypatch.setattr(cli, "LayaClassifier", lambda: LayaClassifier(agent=fake))

    result = CliRunner().invoke(cli.app, ["play", "--episodes", "7", "--laya"])

    assert result.exit_code == 0, result.output
    assert fake.paused and all(fake.paused)  # paused during every Laya call
    assert not game.paused  # and resumed after
    with MetricsDB() as db:
        sources = [row["source"] for row in db.history("routing")]
    # 3 explore tries each, then Laya decides
    assert sources[:6] == ["explore"] * 6
    assert set(sources[6:]) == {"laya"}


@pytest.mark.parametrize("flag, loads", [([], 0), (["--laya"], 1)])
def test_cli_play_loads_laya_only_when_asked(trained, monkeypatch, flag, loads):
    created = []

    def classifier():
        created.append(LayaClassifier(agent=FakeLaya()))
        return created[-1]

    monkeypatch.setattr(cli, "LayaClassifier", classifier)

    result = CliRunner().invoke(cli.app, ["play", "--episodes", "1", *flag])

    assert result.exit_code == 0, result.output
    assert len(created) == loads
