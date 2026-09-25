"""SQLite store for episode, training, performance and routing metrics.

Usage:
    with MetricsDB() as db:
        db.add_episode(agent="dqn", episode=100, reward=850.0, length=420)
        rows = db.query_recent_episodes(agent="dqn", limit=10)
"""

import sqlite3
from pathlib import Path
from typing import Any, Optional, Union

DB_PATH = Path("models") / "logs" / "metrics.db"  # git-ignored

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    agent TEXT NOT NULL,
    episode INTEGER NOT NULL,
    reward REAL NOT NULL,
    length INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS training (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    agent TEXT NOT NULL,
    step INTEGER NOT NULL,
    loss REAL,
    learning_rate REAL
);
CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    steps_per_s REAL NOT NULL,
    memory_mb REAL NOT NULL,
    cpu_percent REAL NOT NULL,
    gpu_memory_mb REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS routing (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    agent TEXT NOT NULL,
    confidence REAL NOT NULL
);
"""


class MetricsDB:
    """Metrics database; creates the file and tables on first use"""

    def __init__(self, path: Union[str, Path] = DB_PATH):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def _insert(self, table: str, **values: Any) -> None:
        columns = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        with self.conn:  # commits
            self.conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({marks})",
                tuple(values.values()),
            )

    def add_episode(self, agent: str, episode: int, reward: float, length: int):
        self._insert(
            "episodes", agent=agent, episode=episode, reward=reward, length=length
        )

    def add_training(
        self,
        agent: str,
        step: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        self._insert(
            "training", agent=agent, step=step, loss=loss, learning_rate=learning_rate
        )

    def add_performance(
        self,
        steps_per_s: float,
        memory_mb: float,
        cpu_percent: float,
        gpu_memory_mb: float,
    ):
        """Arguments match profiling.resource_usage() plus the env speed"""
        self._insert(
            "performance",
            steps_per_s=steps_per_s,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory_mb,
        )

    def add_routing(self, agent: str, confidence: float):
        self._insert("routing", agent=agent, confidence=confidence)

    def query_recent_episodes(
        self, agent: Optional[str] = None, limit: int = 10
    ) -> list[dict]:
        """Newest episodes first, optionally for one agent"""
        where, params = ("WHERE agent = ?", (agent,)) if agent else ("", ())
        rows = self.conn.execute(
            f"SELECT * FROM episodes {where} ORDER BY id DESC LIMIT ?",
            (*params, limit),
        )
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MetricsDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
