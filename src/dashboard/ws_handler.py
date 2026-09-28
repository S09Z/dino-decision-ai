"""WebSocket stream for the dashboard.

On connect the client gets a snapshot; after that, every `interval` seconds
(100ms by default) the rows added to each table since the last check, and
every `status_every` seconds the agents' status and latest metrics. Routing
rows are the router's log; decisions rows are a watched player's every step
(what it read, how it scored each action, what it pressed). A client that reconnects simply gets a new
snapshot, so it never has to replay what it missed.

Messages (JSON):
    {"type": "snapshot", "status": {...}, "latest": {...},
     "history": {"episodes": [...], "performance": [...], "routing": [...], ...}}
    {"type": "rows", "table": "episodes", "rows": [...]}
    {"type": "status", "status": {...}, "latest": {...}}
"""

import asyncio
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from src.dashboard import queries
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB

# Rows of each table in the snapshot (enough for the charts and the log)
HISTORY = {
    "episodes": 3000,
    "training": 200,
    "performance": 500,
    "routing": 100,
    "decisions": 200,
}


async def stream(
    websocket: WebSocket,
    db_path: Path,
    checkpoint_dir: Path,
    interval: float = 0.1,
    status_every: float = 1.0,
) -> None:
    await websocket.accept()
    checkpoints = CheckpointManager(checkpoint_dir)
    with MetricsDB(db_path) as db:
        # history stops at these ids, so no row is sent twice
        ids = {table: queries.last_id(db, table) for table in queries.TABLES}
        await websocket.send_json(
            {
                "type": "snapshot",
                "status": queries.status(db, checkpoints),
                "latest": queries.latest(db),
                "history": {
                    table: queries.tail(db, table, HISTORY[table], upto_id=ids[table])
                    for table in queries.TABLES
                },
            }
        )
        ticks_per_status = max(1, round(status_every / interval))
        tick = 0
        try:
            while True:
                try:  # wait, but notice a client that disconnects meanwhile
                    await asyncio.wait_for(websocket.receive_text(), interval)
                except asyncio.TimeoutError:
                    pass
                for table in queries.TABLES:
                    rows = queries.rows_after(db, table, ids[table])
                    if rows:
                        ids[table] = rows[-1]["id"]
                        await websocket.send_json(
                            {"type": "rows", "table": table, "rows": rows}
                        )
                tick += 1
                if tick % ticks_per_status == 0:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "status": queries.status(db, checkpoints),
                            "latest": queries.latest(db),
                        }
                    )
        except WebSocketDisconnect:
            pass
