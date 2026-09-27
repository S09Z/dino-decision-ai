"""Laya as the router's classifier: it reads each agent's recent score at a
difficulty and answers one choice question, which agent should play.

Laya (`laya` 0.3.x) is a text decision model: `laya.load()` downloads
convaiinnovations/laya (~843MB, ModernBERT-large encoder) into the
HuggingFace cache on first use, and `agent.system_one(state, questions)`
returns a choice with calibrated confidence.

Usage:
    router = LayaRouter(classifier=LayaClassifier())

Measure it before relying on it: python -m src.routing.laya_benchmark
"""

import time
from typing import Any, Optional

MODEL = "convaiinnovations/laya"

QUESTIONS = {
    "agent": {
        "type": "choice",
        "instructions": (
            "Two trained agents play the Chrome Dino game. Pick the one that"
            " should play now: the one expected to score higher at the"
            " current difficulty."
        ),
        "criteria": {
            "dqn": "The DQN agent (its recent score is dqn_score)",
            "ppo": "The PPO agent (its recent score is ppo_score)",
        },
    }
}


def state(scores: dict[str, float], difficulty: str) -> dict[str, Any]:
    """What Laya reads: the difficulty and each agent's recent score"""
    return {
        "game": "Chrome Dino",
        "difficulty": difficulty,
        "dqn_score": round(scores["dqn"], 1),
        "ppo_score": round(scores["ppo"], 1),
    }


class LayaClassifier:
    """Callable for LayaRouter(classifier=...). The model loads on first use,
    or up front with load(); pass `agent` to use an already loaded one."""

    def __init__(
        self, model: str = MODEL, device: Optional[str] = None, agent: Any = None
    ):
        self.model = model
        self.device = device  # None: Laya picks CUDA, MPS, then CPU
        self.agent = agent
        self.load_seconds = 0.0

    def load(self) -> Any:
        if self.agent is None:
            import laya  # type: ignore[import-untyped]

            start = time.perf_counter()
            self.agent = laya.load(self.model, device=self.device)
            self.load_seconds = time.perf_counter() - start
        return self.agent

    def __call__(self, scores: dict[str, float], difficulty: str) -> tuple[str, float]:
        result = self.load().system_one(state(scores, difficulty), QUESTIONS)
        answer = result["answers"]["agent"]
        return answer["choice"], float(answer["answer_confidence"])
