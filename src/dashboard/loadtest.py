"""Measure the dashboard's latency under load.

Serves the API from a temporary DB (the real metrics DB is never touched),
connects `clients` WebSocket clients, and has a writer add episode rows at
`rate` per second for `seconds`. For every row and client it measures the
delay from the row's commit to its arrival; alongside, it times
GET /metrics/latest. Everything runs in one process, so one clock times both
ends.

Usage: python -m src.dashboard.loadtest --clients 20 --out docs/DASHBOARD_LATENCY.md
"""

import asyncio
import json
import socket
import tempfile
import threading
import time
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import typer
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import create_app
from src.monitoring.local_db import MetricsDB

AGENT = "loadtest"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ms(values: list[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q) * 1000) if values else None


def run(
    clients: int = 20, rate: float = 20.0, seconds: float = 20.0, interval: float = 0.1
) -> dict:
    """Returns delivery counts and latency percentiles (ms); `interval` is how
    often the server checks for new rows"""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "metrics.db"
        MetricsDB(db_path).close()
        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(db_path, Path(tmp) / "checkpoints", ws_interval=interval),
                host="127.0.0.1",
                port=port,
                log_level="warning",
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.01)

        committed: dict[int, float] = {}  # episode number -> commit time
        stop = threading.Event()

        def write() -> None:
            with MetricsDB(db_path) as db:
                for i in range(1, int(rate * seconds) + 1):
                    db.add_episode(AGENT, i, -90.0, 70)
                    committed[i] = time.perf_counter()
                    time.sleep(1 / rate)

        rest: list[float] = []

        def poll_rest() -> None:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
                while not stop.is_set():
                    start = time.perf_counter()
                    client.get("/metrics/latest").raise_for_status()
                    rest.append(time.perf_counter() - start)
                    time.sleep(0.2)

        delays: list[float] = []
        connected = [0]  # clients are asyncio tasks on one thread

        async def client() -> None:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=None) as ws:
                connected[0] += 1
                while not stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 0.2)
                    except asyncio.TimeoutError:
                        continue
                    arrived = time.perf_counter()
                    message = json.loads(raw)
                    if message["type"] == "rows" and message["table"] == "episodes":
                        for row in message["rows"]:
                            delays.append(arrived - committed[row["episode"]])

        async def main() -> None:
            tasks = [asyncio.create_task(client()) for _ in range(clients)]
            while connected[0] < clients:
                await asyncio.sleep(0.05)
            writer = threading.Thread(target=write)
            poller = threading.Thread(target=poll_rest)
            writer.start()
            poller.start()
            while writer.is_alive():
                await asyncio.sleep(0.1)
            await asyncio.sleep(1.0)  # let the last rows arrive
            stop.set()
            poller.join()
            await asyncio.gather(*tasks)

        try:
            asyncio.run(main())
        finally:
            server.should_exit = True
            thread.join()
    rows = len(committed)
    return {
        "clients": clients,
        "interval_ms": interval * 1000,
        "rate": rate,
        "seconds": seconds,
        "rows": rows,
        "expected": rows * clients,
        "delivered": len(delays),
        "p50_ms": _ms(delays, 50),
        "p95_ms": _ms(delays, 95),
        "max_ms": _ms(delays, 100),
        "rest_p50_ms": _ms(rest, 50),
        "rest_p95_ms": _ms(rest, 95),
    }


def report(results: list[dict]) -> str:
    lines = [
        "# Dashboard latency",
        "",
        f"Measured {date.today()} with `python -m src.dashboard.loadtest` on a"
        " temporary DB: a writer adds episode rows at a fixed rate while WebSocket"
        " clients listen; delay is commit → arrival, per row and client; the"
        " server checks for new rows every *check* ms.",
        "",
        "| Check | Clients | Rows/s | Delivered | p50 | p95 | max | REST p50 | REST p95 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['interval_ms']:g} ms | {r['clients']} | {r['rate']:g} | {r['delivered']}/{r['expected']}"
            f" | {r['p50_ms']:.0f} ms | {r['p95_ms']:.0f} ms | {r['max_ms']:.0f} ms"
            f" | {r['rest_p50_ms']:.0f} ms | {r['rest_p95_ms']:.0f} ms |"
        )
    return "\n".join(lines) + "\n"


def main(
    clients: list[int] = typer.Option([1, 5, 20], help="Client counts to try"),
    interval: list[float] = typer.Option([0.1], help="Server check intervals (s)"),
    rate: float = 20.0,
    seconds: float = 10.0,
    out: Optional[Path] = None,
) -> None:
    text = report([run(n, rate, seconds, i) for i in interval for n in clients])
    typer.echo(text)
    if out is not None:
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    typer.run(main)
