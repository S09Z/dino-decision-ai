"""Keep only the best N training checkpoints per agent, plus the latest one
to resume an interrupted run from.

Usage:
    manager = CheckpointManager(keep_best_n=5)
    manager.save(agent, name="dqn", step=10_000, reward=-80.0)
    manager.load_best(agent, name="dqn")
    manager.save_latest(model, name="dqn", step=10_000, episodes=150)
    manager.latest("dqn")  # -> Latest(step=10_000, ...) or None
"""

import json
import os
import time
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


@dataclass
class Latest:
    """Where an interrupted run stopped: enough to resume it"""

    name: str
    step: int
    episodes: int  # episodes recorded so far, so numbering continues
    path: str
    buffer: Optional[str]  # replay buffer (off-policy agents such as DQN)
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

    def save_latest(self, model, name: str, step: int, episodes: int) -> Latest:
        """Overwrite `name`'s latest checkpoint with an SB3 model and, if it has
        one, its replay buffer. Each file is written under a temporary name and
        then renamed, so a crash or power cut mid-save leaves the previous
        latest checkpoint intact."""
        path = self.directory / f"{name}_latest.zip"
        _save_atomic(model.save, path)
        buffer = None
        if getattr(model, "replay_buffer", None) is not None:
            buffer = self.directory / f"{name}_latest_buffer.pkl"
            _save_atomic(model.save_replay_buffer, buffer)
        latest = Latest(
            name,
            step,
            episodes,
            path.as_posix(),
            buffer.as_posix() if buffer else None,
            datetime.now().isoformat(timespec="seconds"),
        )
        _save_atomic(
            lambda tmp: tmp.write_text(json.dumps(asdict(latest)), encoding="utf-8"),
            self.directory / f"{name}_latest.json",
        )
        return latest

    def latest(self, name: str) -> Optional[Latest]:
        """`name`'s latest checkpoint, or None if it has none"""
        index = self.directory / f"{name}_latest.json"
        if not index.exists():
            return None
        return Latest(**json.loads(index.read_text(encoding="utf-8")))


def _save_atomic(save, path: Path, tries: int = 20, wait: float = 0.05) -> None:
    """save(tmp) then rename tmp to `path` (os.replace is atomic). Windows
    refuses to replace a file another process has open (the dashboard reads
    the latest index every second, for milliseconds), so retry for up to
    `tries` x `wait` seconds."""
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")  # keeps SB3's suffix
    save(tmp)
    for attempt in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(wait)
