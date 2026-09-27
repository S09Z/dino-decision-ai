"""Tests for the agent router (no Laya model needed)"""

import pytest

from src.routing.laya_router import (
    LayaRouter,
    difficulty_from_speed,
    heuristic,
)


@pytest.mark.parametrize(
    "speed, expected",
    [(6.0, "EASY"), (8.0, "EASY"), (8.5, "MEDIUM"), (10.8, "HARD"), (13.0, "HARD")],
)
def test_difficulty_from_speed(speed, expected):
    assert difficulty_from_speed(speed) == expected


def test_difficulty_clamps_out_of_range_speeds():
    assert difficulty_from_speed(0.0) == "EASY"
    assert difficulty_from_speed(20.0) == "HARD"


def test_plan_example_picks_ppo_confidently():
    agent, confidence = LayaRouter().decide_agent(1850, 2340, "HARD")

    assert agent == "ppo"
    assert confidence == pytest.approx(0.9, abs=0.02)


def test_confidence_grows_with_the_lead():
    close = heuristic({"dqn": 1000, "ppo": 1020})[1]
    far = heuristic({"dqn": 1000, "ppo": 2000})[1]

    assert 0.5 < close < far < 1.0


def test_tie_goes_to_dqn_with_half_confidence():
    assert heuristic({"dqn": 0.0, "ppo": 0.0}) == ("dqn", 0.5)
    assert heuristic({"dqn": 500.0, "ppo": 500.0}) == ("dqn", 0.5)


def test_unknown_difficulty_is_rejected():
    with pytest.raises(ValueError):
        LayaRouter().decide_agent(1, 2, "hard")


def test_classifier_decides_when_it_works():
    router = LayaRouter(classifier=lambda scores, difficulty: ("dqn", 0.7))

    assert router.decide_agent(1850, 2340, "HARD") == ("dqn", 0.7)
    assert router.last_source == "laya"


@pytest.mark.parametrize(
    "classifier",
    [
        lambda scores, difficulty: ("a2c", 0.9),  # unknown agent
        lambda scores, difficulty: ("dqn", 1.5),  # confidence out of range
        lambda scores, difficulty: (_ for _ in ()).throw(RuntimeError("no model")),
    ],
)
def test_falls_back_to_heuristic_when_classifier_fails(classifier):
    router = LayaRouter(classifier=classifier)

    agent, _ = router.decide_agent(1850, 2340, "HARD")

    assert agent == "ppo"
    assert router.last_source == "heuristic"
