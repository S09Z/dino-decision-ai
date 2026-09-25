"""Tests for the SQLite metrics database"""

import sqlite3

import pytest

from src.monitoring.local_db import MetricsDB


@pytest.fixture
def db():
    with MetricsDB(":memory:") as db:
        yield db


def rows(db, table):
    return [dict(r) for r in db.conn.execute(f"SELECT * FROM {table}")]


def test_creates_all_tables(db):
    tables = {
        r[0]
        for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"episodes", "training", "performance", "routing"} <= tables


def test_add_episode_is_queryable_with_timestamp(db):
    db.add_episode(agent="dqn", episode=1, reward=-92.5, length=74)

    (row,) = db.query_recent_episodes()
    assert row["agent"] == "dqn"
    assert row["episode"] == 1
    assert row["reward"] == -92.5
    assert row["length"] == 74
    assert row["timestamp"]


def test_query_recent_episodes_newest_first_filtered_and_limited(db):
    for episode in range(5):
        db.add_episode(agent="dqn", episode=episode, reward=0.0, length=10)
        db.add_episode(agent="ppo", episode=episode, reward=0.0, length=10)

    recent = db.query_recent_episodes(agent="dqn", limit=3)

    assert [r["episode"] for r in recent] == [4, 3, 2]
    assert {r["agent"] for r in recent} == {"dqn"}
    assert len(db.query_recent_episodes(limit=100)) == 10


def test_add_training_performance_and_routing(db):
    db.add_training(agent="dqn", step=1000, loss=0.5, learning_rate=1e-4)
    db.add_training(agent="ppo", step=2000)
    db.add_performance(
        steps_per_s=13.4, memory_mb=1336, cpu_percent=20, gpu_memory_mb=13
    )
    db.add_routing(agent="ppo", confidence=0.8)

    assert [(r["step"], r["loss"]) for r in rows(db, "training")] == [
        (1000, 0.5),
        (2000, None),
    ]
    assert rows(db, "performance")[0]["steps_per_s"] == 13.4
    assert rows(db, "routing")[0]["agent"] == "ppo"


def test_data_persists_in_file_and_parent_dir_is_created(tmp_path):
    path = tmp_path / "logs" / "metrics.db"
    with MetricsDB(path) as db:
        db.add_episode(agent="dqn", episode=1, reward=1.0, length=5)

    with MetricsDB(path) as db:
        assert len(db.query_recent_episodes()) == 1


def test_close_releases_connection(tmp_path):
    db = MetricsDB(tmp_path / "metrics.db")
    db.close()

    with pytest.raises(sqlite3.ProgrammingError):
        db.query_recent_episodes()
