"""Read-only queries behind the dashboard, shared by the REST API and the
WebSocket stream. Timestamps are SQLite's, in UTC."""

import json
from typing import Any, Optional

import numpy as np

from src.models import AGENTS
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB

TABLES = ("episodes", "training", "performance", "routing", "decisions")
# An agent counts as training while its last episode is this recent
ACTIVE_SECONDS = 120
# Mean reward and length over this many recent episodes
WINDOW = 10


def agent_names(db: MetricsDB) -> list[str]:
    """The CLI agents, then any other agent with episodes (experiments)"""
    recorded = [r[0] for r in db.conn.execute("SELECT DISTINCT agent FROM episodes")]
    return list(AGENTS) + sorted(set(recorded) - set(AGENTS))


def _dict(table: str, row: Any) -> dict:
    out = dict(row)
    if table == "decisions":
        out["scores"] = json.loads(out["scores"])  # stored as JSON text
    return out


def last_row(db: MetricsDB, table: str, agent: Optional[str]) -> Optional[dict]:
    where, params = ("WHERE agent = ?", (agent,)) if agent else ("", ())
    row = db.conn.execute(
        f"SELECT * FROM {table} {where} ORDER BY id DESC LIMIT 1", params
    ).fetchone()
    return dict(row) if row else None


def last_id(db: MetricsDB, table: str) -> int:
    return db.conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()[0]


def tail(
    db: MetricsDB,
    table: str,
    limit: int,
    agent: Optional[str] = None,
    upto_id: Optional[int] = None,
) -> list[dict]:
    """The last `limit` rows of `table` (up to id `upto_id`), oldest first"""
    conditions: list[str] = []
    params: list[Any] = []
    if agent:
        conditions.append("agent = ?")
        params.append(agent)
    if upto_id is not None:
        conditions.append("id <= ?")
        params.append(upto_id)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = db.conn.execute(
        f"SELECT * FROM {table} {where} ORDER BY id DESC LIMIT ?", (*params, limit)
    )
    return [_dict(table, row) for row in reversed(rows.fetchall())]


def rows_after(db: MetricsDB, table: str, after_id: int) -> list[dict]:
    """Rows of `table` added after id `after_id`, oldest first"""
    rows = db.conn.execute(
        f"SELECT * FROM {table} WHERE id > ? ORDER BY id", (after_id,)
    )
    return [_dict(table, row) for row in rows]


def latest(db: MetricsDB) -> dict[str, Any]:
    """Per agent: last episode, recent means, last performance and training
    rows; plus the last routing decision"""
    agents = {}
    for name in agent_names(db):
        recent = db.query_recent_episodes(name, limit=WINDOW)
        agents[name] = {
            "episodes": db.episode_count(name),
            "last_episode": recent[0] if recent else None,
            "mean_reward": (
                float(np.mean([r["reward"] for r in recent])) if recent else None
            ),
            "mean_length": (
                float(np.mean([r["length"] for r in recent])) if recent else None
            ),
            "performance": last_row(db, "performance", name),
            "training": last_row(db, "training", name),
        }
    return {"agents": agents, "routing": last_row(db, "routing", None)}


def status(db: MetricsDB, checkpoints: CheckpointManager) -> dict[str, Any]:
    """Per agent: training or idle, episodes, and its checkpoints"""
    agents = {}
    for name in agent_names(db):
        last = last_row(db, "episodes", name)
        idle_for = None
        if last:
            idle_for = db.conn.execute(
                "SELECT (julianday('now') - julianday(?)) * 86400",
                (last["timestamp"],),
            ).fetchone()[0]
        best = checkpoints.best(name)
        resume = checkpoints.latest(name)
        agents[name] = {
            "state": (
                "training"
                if idle_for is not None and idle_for < ACTIVE_SECONDS
                else "idle"
            ),
            "episodes": db.episode_count(name),
            "last_episode_at": last["timestamp"] if last else None,
            "best_checkpoint": (
                {"step": best.step, "reward": best.reward} if best else None
            ),
            "latest_step": resume.step if resume else None,
        }
    return {"agents": agents}
