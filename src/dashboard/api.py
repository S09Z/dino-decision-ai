"""Dashboard API: read-only views of MetricsDB and the checkpoints.

Safe to run while training writes the same DB: every request opens its own
connection and only reads. Timestamps are SQLite's, in UTC.

Usage:
    uvicorn src.dashboard.api:app       (make dashboard / dino-ai dashboard)
    GET /health, /agents/status, /metrics/latest, /metrics/history?table=episodes
    WS  /ws                              (a /metrics/latest snapshot on each change)
"""

import asyncio
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from src.models import AGENTS
from src.models_mgmt.checkpoint_manager import CHECKPOINT_DIR, CheckpointManager
from src.monitoring.local_db import DB_PATH, MetricsDB

TABLES = ("episodes", "training", "performance", "routing")
# An agent counts as training while its last episode is this recent
ACTIVE_SECONDS = 120
# Mean reward and length over this many recent episodes
WINDOW = 10


def create_app(
    db_path: Path = DB_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    ws_interval: float = 1.0,
) -> FastAPI:
    app = FastAPI(title="Chrome Dino AI dashboard")

    def agent_names(db: MetricsDB) -> list[str]:
        """The CLI agents, then any other agent with episodes (experiments)"""
        recorded = [
            r[0] for r in db.conn.execute("SELECT DISTINCT agent FROM episodes")
        ]
        return list(AGENTS) + sorted(set(recorded) - set(AGENTS))

    def last_row(db: MetricsDB, table: str, agent: Optional[str]) -> Optional[dict]:
        where, params = ("WHERE agent = ?", (agent,)) if agent else ("", ())
        row = db.conn.execute(
            f"SELECT * FROM {table} {where} ORDER BY id DESC LIMIT 1", params
        ).fetchone()
        return dict(row) if row else None

    def seconds_since(db: MetricsDB, timestamp: str) -> float:
        row = db.conn.execute(
            "SELECT (julianday('now') - julianday(?)) * 86400", (timestamp,)
        ).fetchone()
        return float(row[0])

    def latest(db: MetricsDB) -> dict[str, Any]:
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

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/agents/status")
    def agents_status() -> dict:
        """Per agent: training or idle, episodes, and its checkpoints"""
        checkpoints = CheckpointManager(checkpoint_dir)
        status = {}
        with MetricsDB(db_path) as db:
            for name in agent_names(db):
                last = last_row(db, "episodes", name)
                idle_for = seconds_since(db, last["timestamp"]) if last else None
                best = checkpoints.best(name)
                resume = checkpoints.latest(name)
                status[name] = {
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
        return {"agents": status}

    @app.get("/metrics/latest")
    def metrics_latest() -> dict:
        """Per agent: last episode, recent means, last performance and training
        rows; plus the last routing decision"""
        with MetricsDB(db_path) as db:
            return latest(db)

    @app.get("/metrics/history")
    def metrics_history(
        table: str = "episodes", agent: Optional[str] = None, limit: int = 500
    ) -> dict:
        """The last `limit` rows of `table`, oldest first"""
        if table not in TABLES:
            raise HTTPException(400, f"table must be one of {', '.join(TABLES)}")
        with MetricsDB(db_path) as db:
            rows = db.history(table, agent)
        return {"table": table, "agent": agent, "rows": rows[-limit:]}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        """Sends a /metrics/latest snapshot on connect and whenever it changes
        (checked every `ws_interval` seconds)"""
        await websocket.accept()
        sent = None
        try:
            while True:
                with MetricsDB(db_path) as db:
                    snapshot = latest(db)
                if snapshot != sent:
                    await websocket.send_json(snapshot)
                    sent = snapshot
                try:  # wait, but notice a client that disconnects meanwhile
                    await asyncio.wait_for(websocket.receive_text(), ws_interval)
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
