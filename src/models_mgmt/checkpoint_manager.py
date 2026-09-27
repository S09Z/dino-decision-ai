"""Keep only the best N training checkpoints per agent.

Usage:
    manager = CheckpointManager(keep_best_n=5)
    manager.save(agent, name="dqn", step=10_000, reward=-80.0)
    manager.load_best(agent, name="dqn")
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

CHECKPOINT_DIR = Path("models") / "checkpoints"  # git-ignored


@dataclass
class Checkpoint:
    name: str  # agent name, e.g. "dqn"
    step: int
    reward: float  # higher is better
    path: str
    timestamp: str


class CheckpointManager:
    """Saves agents (anything with save(path)/load(path)) and deletes all but
    the `keep_best_n` highest-reward checkpoints of each agent. Each agent has
    its own index file (<name>.json), so agents training in parallel processes
    never overwrite each other's index."""

    def __init__(self, directory: Path = CHECKPOINT_DIR, keep_best_n: int = 5):
        self.directory = Path(directory)
        self.keep_best_n = keep_best_n
        self.directory.mkdir(parents=True, exist_ok=True)

    def _load_index(self, name: str) -> list[Checkpoint]:
        index = self.directory / f"{name}.json"
        if not index.exists():
            return []
        return [Checkpoint(**c) for c in json.loads(index.read_text(encoding="utf-8"))]

    def _write_index(self, name: str, checkpoints: list[Checkpoint]) -> None:
        (self.directory / f"{name}.json").write_text(
            json.dumps([asdict(c) for c in checkpoints], indent=2), encoding="utf-8"
        )

    def checkpoints(self, name: str) -> list[Checkpoint]:
        """This agent's kept checkpoints, best first"""
        return sorted(self._load_index(name), key=lambda c: c.reward, reverse=True)

    def save(self, agent, name: str, step: int, reward: float) -> Optional[Checkpoint]:
        """Save if the reward makes the top N; returns None when skipped"""
        kept = self.checkpoints(name)
        if len(kept) >= self.keep_best_n and reward <= kept[-1].reward:
            return None
        path = self.directory / f"{name}_step{step}.zip"
        agent.save(path)
        checkpoint = Checkpoint(
            name,
            step,
            reward,
            path.as_posix(),  # forward slashes work on Windows too
            datetime.now().isoformat(timespec="seconds"),
        )
        self._write_index(name, self._load_index(name) + [checkpoint])
        self.prune(name, self.keep_best_n)
        return checkpoint

    def prune(self, name: str, keep: int) -> list[Checkpoint]:
        """Delete all but the `keep` best checkpoints of `name`; returns the
        removed ones"""
        ranked = self.checkpoints(name)
        for checkpoint in ranked[keep:]:
            Path(checkpoint.path).unlink(missing_ok=True)
        self._write_index(name, ranked[:keep])
        return ranked[keep:]

    def best(self, name: str) -> Optional[Checkpoint]:
        kept = self.checkpoints(name)
        return kept[0] if kept else None

    def load_best(self, agent, name: str) -> Checkpoint:
        """Load the best checkpoint into `agent`"""
        best = self.best(name)
        if best is None:
            raise FileNotFoundError(f"No checkpoints for {name!r} in {self.directory}")
        agent.load(best.path)
        return best
