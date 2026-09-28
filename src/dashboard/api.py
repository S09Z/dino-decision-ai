"""Dashboard API: read-only views of MetricsDB and the checkpoints, and the
web UI in frontend/.

Safe to run while training writes the same DB: it only reads. Timestamps are
SQLite's, in UTC.

Usage:
    uvicorn src.dashboard.api:app       (make dashboard / dino-ai dashboard)
    GET /                                (the dashboard page)
    GET /health, /agents/status, /metrics/latest, /metrics/history?table=episodes
    WS  /ws                              (see ws_handler.py)
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles

from src.dashboard import queries
from src.dashboard.ws_handler import stream
from src.models_mgmt.checkpoint_manager import CHECKPOINT_DIR, CheckpointManager
from src.monitoring.local_db import DB_PATH, MetricsDB

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


def create_app(
    db_path: Path = DB_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    ws_interval: float = 0.1,
    ws_status_every: float = 1.0,
) -> FastAPI:
    app = FastAPI(title="Chrome Dino AI dashboard")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/agents/status")
    def agents_status() -> dict:
        """Per agent: training or idle, episodes, and its checkpoints"""
        with MetricsDB(db_path) as db:
            return queries.status(db, CheckpointManager(checkpoint_dir))

    @app.get("/metrics/latest")
    def metrics_latest() -> dict:
        """Per agent: last episode, recent means, last performance and training
        rows; plus the last routing decision"""
        with MetricsDB(db_path) as db:
            return queries.latest(db)

    @app.get("/metrics/history")
    def metrics_history(
        table: str = "episodes", agent: Optional[str] = None, limit: int = 500
    ) -> dict:
        """The last `limit` rows of `table`, oldest first"""
        if table not in queries.TABLES:
            raise HTTPException(
                400, f"table must be one of {', '.join(queries.TABLES)}"
            )
        with MetricsDB(db_path) as db:
            rows = queries.tail(db, table, limit, agent=agent)
        return {"table": table, "agent": agent, "rows": rows}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await stream(websocket, db_path, checkpoint_dir, ws_interval, ws_status_every)

    # last, so the API routes above take precedence
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
