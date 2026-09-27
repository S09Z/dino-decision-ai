"""Pick which agent should play: Laya when it is plugged in, otherwise a
score heuristic.

Usage:
    router = LayaRouter()
    agent, confidence = router.decide_agent(
        dqn_score=1850, ppo_score=2340, difficulty="HARD"
    )  # -> ("ppo", 0.9)

Scores are recent game scores (the distance shown in the game) of each agent
at this difficulty; tracking them per difficulty is the AgentManager's job.
"""

import math
from typing import Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

DIFFICULTIES = ("EASY", "MEDIUM", "HARD")
MIN_SPEED, MAX_SPEED = 6.0, 13.0  # the game's SPEED and MAX_SPEED (game/index.js)
SENSITIVITY = 5.0  # a 20% score lead gives ~0.9 confidence

# (scores by agent, difficulty) -> (agent, confidence); Laya plugs in here (5.3)
Classifier = Callable[[dict[str, float], str], tuple[str, float]]


def difficulty_from_speed(speed: float) -> str:
    """EASY, MEDIUM or HARD: thirds of the game's speed range"""
    fraction = (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
    return DIFFICULTIES[min(max(int(fraction * 3), 0), 2)]


def heuristic(scores: dict[str, float]) -> tuple[str, float]:
    """The higher-scoring agent; confidence grows from 0.5 (tie) towards 1
    with its relative lead. Ties go to the first agent."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    scale = max(abs(top), abs(second))
    lead = (top - second) / scale if scale else 0.0
    return best, 0.5 + 0.5 * math.tanh(SENSITIVITY * lead)


class LayaRouter:
    """Decides between DQN and PPO. With a `classifier` (Laya) it asks that
    first and falls back to the heuristic if it fails or answers nonsense;
    `last_source` says which one decided."""

    def __init__(self, classifier: Optional[Classifier] = None):
        self.classifier = classifier
        self.last_source = "heuristic"

    def decide_agent(
        self, dqn_score: float, ppo_score: float, difficulty: str
    ) -> tuple[str, float]:
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        scores = {"dqn": float(dqn_score), "ppo": float(ppo_score)}
        if self.classifier is not None:
            try:
                agent, confidence = self.classifier(scores, difficulty)
                if agent in scores and 0.0 <= confidence <= 1.0:
                    self.last_source = "laya"
                    return agent, confidence
                logger.warning(
                    "Laya answered %r; using the score heuristic", (agent, confidence)
                )
            except Exception as error:  # e.g. model not downloaded, out of memory
                logger.warning("Laya failed (%s); using the score heuristic", error)
        self.last_source = "heuristic"
        return heuristic(scores)
