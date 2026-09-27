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
INDEX_FILE = "checkpoints.json"


@dataclass
class Checkpoint:
    name: str  # agent name, e.g. "dqn"
    step: int
    reward: float  # higher is better
    path: str
    timestamp: str


class CheckpointManager:
    """Saves agents (anything with save(path)/load(path)) and deletes all but
    the `keep_best_n` highest-reward checkpoints of each agent"""

    def __init__(self, directory: Path = CHECKPOINT_DIR, keep_best_n: int = 5):
        self.directory = Path(directory)
        self.keep_best_n = keep_best_n
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index = self.directory / INDEX_FILE

    def _load_index(self) -> list[Checkpoint]:
        if not self._index.exists():
            return []
        return [
            Checkpoint(**c) for c in json.loads(self._index.read_text(encoding="utf-8"))
        ]

    def _write_index(self, checkpoints: list[Checkpoint]) -> None:
        self._index.write_text(
            json.dumps([asdict(c) for c in checkpoints], indent=2), encoding="utf-8"
        )

    def checkpoints(self, name: str) -> list[Checkpoint]:
        """This agent's kept checkpoints, best first"""
        mine = [c for c in self._load_index() if c.name == name]
        return sorted(mine, key=lambda c: c.reward, reverse=True)

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
        self._write_index(self._load_index() + [checkpoint])
        self.prune(name, self.keep_best_n)
        return checkpoint

    def prune(self, name: str, keep: int) -> list[Checkpoint]:
        """Delete all but the `keep` best checkpoints of `name`; returns the
        removed ones"""
        removed = self.checkpoints(name)[keep:]
        for checkpoint in removed:
            Path(checkpoint.path).unlink(missing_ok=True)
        self._write_index([c for c in self._load_index() if c not in removed])
        return removed

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
