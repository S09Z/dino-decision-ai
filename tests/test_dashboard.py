"""Tests for the dashboard API (temporary DB and checkpoints, no Chrome)"""

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from src.cli import main as cli
from src.dashboard.api import create_app
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from tests.test_models_mgmt import FakeAgent


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "metrics.db", tmp_path / "checkpoints"


@pytest.fixture
def client(paths):
    db_path, checkpoint_dir = paths
    return TestClient(create_app(db_path, checkpoint_dir, ws_interval=0.01))


def add_episodes(db_path, agent, rewards):
    with MetricsDB(db_path) as db:
        for i, reward in enumerate(rewards, 1):
            db.add_episode(agent, i, reward, length=int(100 + reward))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_agents_status_without_data_lists_idle_agents(client):
    agents = client.get("/agents/status").json()["agents"]

    assert set(agents) == {"dqn", "ppo"}
    assert agents["dqn"] == {
        "state": "idle",
        "episodes": 0,
        "last_episode_at": None,
        "best_checkpoint": None,
        "latest_step": None,
    }


def test_agents_status_shows_training_agents_and_checkpoints(client, paths):
    db_path, checkpoint_dir = paths
    add_episodes(db_path, "dqn", [-90.0, -80.0])
    add_episodes(db_path, "dqn-a2", [-70.0])  # experiments show up too
    checkpoints = CheckpointManager(checkpoint_dir)
    checkpoints.save(FakeAgent(), "dqn", step=5000, reward=-85.0)
    checkpoints.save_latest(FakeAgent(), "dqn", step=6000, episodes=2)

    agents = client.get("/agents/status").json()["agents"]

    assert list(agents) == ["dqn", "ppo", "dqn-a2"]
    assert agents["dqn"]["state"] == "training"  # episode just now
    assert agents["dqn"]["episodes"] == 2
    assert agents["dqn"]["best_checkpoint"] == {"step": 5000, "reward": -85.0}
    assert agents["dqn"]["latest_step"] == 6000
    assert agents["ppo"]["state"] == "idle"


def test_agent_is_idle_once_its_last_episode_is_old(client, paths):
    db_path, _ = paths
    add_episodes(db_path, "dqn", [-90.0])
    with MetricsDB(db_path) as db, db.conn:
        db.conn.execute("UPDATE episodes SET timestamp = datetime('now', '-5 minutes')")

    assert client.get("/agents/status").json()["agents"]["dqn"]["state"] == "idle"


def test_metrics_latest_has_recent_means_and_last_rows(client, paths):
    db_path, _ = paths
    add_episodes(db_path, "ppo", [-100.0] * 5 + [-80.0] * 10)
    with MetricsDB(db_path) as db:
        db.add_performance(13.0, 2000.0, 30.0, 90.0, agent="ppo", step=5000)
        db.add_training("ppo", 5000, loss=0.5, learning_rate=1e-4)
        db.add_routing("ppo", 0.9, difficulty="EASY", source="heuristic")

    latest = client.get("/metrics/latest").json()
    ppo = latest["agents"]["ppo"]

    assert ppo["episodes"] == 15
    assert ppo["mean_reward"] == -80.0  # last 10 only
    assert ppo["mean_length"] == 20.0
    assert ppo["last_episode"]["episode"] == 15
    assert ppo["performance"]["steps_per_s"] == 13.0
    assert ppo["training"]["loss"] == 0.5
    assert latest["agents"]["dqn"]["mean_reward"] is None
    assert latest["routing"]["agent"] == "ppo"


def test_metrics_history_returns_the_last_rows_oldest_first(client, paths):
    db_path, _ = paths
    add_episodes(db_path, "dqn", [-90.0, -80.0, -70.0])
    add_episodes(db_path, "ppo", [-60.0])

    history = client.get("/metrics/history?agent=dqn&limit=2").json()

    assert [row["reward"] for row in history["rows"]] == [-80.0, -70.0]
    assert client.get("/metrics/history?table=routing").json()["rows"] == []


def test_metrics_history_rejects_unknown_tables(client):
    response = client.get("/metrics/history?table=users")

    assert response.status_code == 400


def test_websocket_sends_a_snapshot_then_each_change(client, paths):
    db_path, _ = paths
    add_episodes(db_path, "dqn", [-90.0])

    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        add_episodes(db_path, "ppo", [-70.0])
        second = ws.receive_json()

    assert first["agents"]["dqn"]["episodes"] == 1
    assert first["agents"]["ppo"]["episodes"] == 0
    assert second["agents"]["ppo"]["episodes"] == 1


def test_cli_dashboard_starts_the_api(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

    result = CliRunner().invoke(cli.app, ["dashboard", "--port", "8123"])

    assert result.exit_code == 0, result.output
    assert calls == [(("src.dashboard.api:app",), {"host": "127.0.0.1", "port": 8123})]
