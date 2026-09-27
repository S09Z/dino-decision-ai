"""Let the router choose which trained agent plays.

The router decides at the start of each episode and whenever the difficulty
(game speed) changes, not every step, so one agent never undoes another's
move halfway through. Each stretch an agent plays earns it the score gained
in that stretch, kept per difficulty; those averages feed the router.

Usage (one env from make_dino_env):
    manager = AgentManager.from_checkpoints(env, LayaRouter(), db)
    obs = env.reset()
    while playing:
        action, agent = manager.get_action(obs)
        obs, _, dones, infos = env.step(action)
        manager.observe(infos[0], bool(dones[0]))
"""

from collections import defaultdict, deque
from typing import Any, Optional

import numpy as np

from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from src.routing.laya_router import MIN_SPEED, LayaRouter, difficulty_from_speed
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AgentManager:
    """Plays with whichever agent the router picks. Until every agent has
    played `min_tries` stretches at a difficulty, the least-tried one plays
    there ("explore"), so no agent is locked out by an early tie."""

    def __init__(
        self,
        agents: dict[str, Any],
        router: LayaRouter,
        db: Optional[MetricsDB] = None,
        window: int = 10,
        min_tries: int = 3,
    ):
        self.agents = agents  # name -> anything with predict(obs)
        self.router = router
        self.db = db
        self.min_tries = min_tries
        self.stretches: defaultdict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self.current: Optional[str] = None
        self.difficulty = "EASY"
        self.switches = 0
        self._stretch_start = 0.0

    @classmethod
    def from_checkpoints(
        cls,
        env,
        router: LayaRouter,
        db: Optional[MetricsDB] = None,
        checkpoints: Optional[CheckpointManager] = None,
    ) -> "AgentManager":
        """Every agent loaded from its best checkpoint (FileNotFoundError if
        one has none)"""
        from src.models import AGENTS

        checkpoints = checkpoints or CheckpointManager()
        agents = {}
        for name, agent_class in AGENTS.items():
            agents[name] = agent_class(env)
            checkpoints.load_best(agents[name], name)
        return cls(agents, router, db)

    def score(self, agent: str, difficulty: str) -> float:
        """Mean score gained per stretch by `agent` at `difficulty` (0 if none)"""
        stretches = self.stretches[(agent, difficulty)]
        return float(np.mean(stretches)) if stretches else 0.0

    def get_action(self, obs) -> tuple[np.ndarray, str]:
        if self.current is None:
            self._decide()
        assert self.current is not None
        action, _ = self.agents[self.current].predict(obs)
        return action, self.current

    def observe(self, info: dict, done: bool) -> None:
        """Call after each env step with that step's info and done flag"""
        score = float(info.get("score", 0))
        if done:
            self._close_stretch(score)
            self.difficulty, self._stretch_start = "EASY", 0.0
            self._decide()
            return
        difficulty = difficulty_from_speed(float(info.get("speed", MIN_SPEED)))
        if difficulty != self.difficulty:
            self._close_stretch(score)
            self.difficulty, self._stretch_start = difficulty, score
            self._decide()

    def _close_stretch(self, score: float) -> None:
        if self.current is not None:
            gained = score - self._stretch_start
            self.stretches[(self.current, self.difficulty)].append(gained)

    def _decide(self) -> None:
        tries = {
            name: len(self.stretches[(name, self.difficulty)]) for name in self.agents
        }
        untried = [name for name in self.agents if tries[name] < self.min_tries]
        if untried:
            agent, confidence, source = (
                min(untried, key=tries.__getitem__),
                0.5,
                "explore",
            )
        else:
            agent, confidence = self.router.decide_agent(
                self.score("dqn", self.difficulty),
                self.score("ppo", self.difficulty),
                self.difficulty,
            )
            source = self.router.last_source
        if self.current is not None and agent != self.current:
            self.switches += 1
            logger.info(
                "Switch %s -> %s at %s (%s, confidence %.2f)",
                self.current,
                agent,
                self.difficulty,
                source,
                confidence,
            )
        self.current = agent
        if self.db is not None:
            self.db.add_routing(agent, confidence, self.difficulty, source)
