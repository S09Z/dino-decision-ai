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
    assert db.episode_count() == 10
    assert db.episode_count("ppo") == 5


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


def test_performance_rows_record_agent_and_step(db):
    db.add_performance(1.0, 2.0, 3.0, 4.0, agent="ppo", step=5000)
    db.add_performance(1.0, 2.0, 3.0, 4.0, agent="dqn", step=5000)

    (row,) = db.history("performance", agent="ppo")
    assert (row["agent"], row["step"], row["memory_mb"]) == ("ppo", 5000, 2.0)


def test_history_is_oldest_first_and_limited_to_known_tables(db):
    for episode in range(3):
        db.add_episode(agent="dqn", episode=episode, reward=0.0, length=1)

    assert [r["episode"] for r in db.history("episodes")] == [0, 1, 2]
    with pytest.raises(ValueError):
        db.history("routing; DROP TABLE episodes")


def test_old_db_without_agent_and_step_columns_is_upgraded(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE performance (id INTEGER PRIMARY KEY, timestamp TEXT,"
        " steps_per_s REAL, memory_mb REAL, cpu_percent REAL, gpu_memory_mb REAL)"
    )
    old.execute("INSERT INTO performance VALUES (1, 't', 12.0, 1300.0, 20.0, 13.0)")
    old.commit()
    old.close()

    with MetricsDB(path) as db:
        db.add_performance(13.0, 1400.0, 21.0, 13.0, agent="dqn", step=100)
        rows = db.history("performance")

    assert [(r["agent"], r["step"]) for r in rows] == [(None, None), ("dqn", 100)]


def test_old_routing_table_gains_difficulty_and_source(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE routing (id INTEGER PRIMARY KEY, timestamp TEXT,"
        " agent TEXT NOT NULL, confidence REAL NOT NULL)"
    )
    old.commit()
    old.close()

    with MetricsDB(path) as db:
        db.add_routing("ppo", 0.9, difficulty="HARD", source="heuristic")
        (row,) = db.history("routing")

    assert (row["difficulty"], row["source"]) == ("HARD", "heuristic")
