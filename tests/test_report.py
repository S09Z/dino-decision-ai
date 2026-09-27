"""Tests for the HTML training report"""

import pytest
from typer.testing import CliRunner

from src.cli import main as cli
from src.monitoring.local_db import MetricsDB
from src.profiling.report import (
    LEAK_GROWTH_PERCENT,
    build_report,
    memory_growth,
    summary,
)


def perf(step, memory_mb, steps_per_s=12.0):
    return {"step": step, "memory_mb": memory_mb, "steps_per_s": steps_per_s}


@pytest.fixture
def db():
    with MetricsDB(":memory:") as db:
        yield db


def test_memory_growth_compares_last_third_with_middle_third():
    warm_up = [perf(0, 100), perf(1, 900)]
    middle = [perf(2, 1000), perf(3, 1000)]
    last = [perf(4, 1100), perf(5, 1100)]

    assert memory_growth(warm_up + middle + last) == pytest.approx(10.0)


def test_warm_up_then_saw_tooth_memory_is_not_a_leak():
    """Measured: 3,000 random steps in real Chrome, total RSS every 250 steps.
    A straight-line fit reads this as +1584MB per 10k steps."""
    rss = [2247, 1312, 2038, 1767, 1950, 2347, 1920, 2320, 1937, 2367, 2054, 2321]
    rows = [perf(250 * (i + 1), mb) for i, mb in enumerate(rss)]

    assert memory_growth(rows) == pytest.approx(1.7, abs=0.1)
    assert memory_growth(rows) < LEAK_GROWTH_PERCENT


def test_memory_growth_needs_six_stepped_samples():
    assert memory_growth([perf(s, 1) for s in range(5)]) is None
    assert memory_growth([perf(None, 1)] * 6) is None


def test_summary_flags_fast_memory_growth():
    episodes = [{"reward": -90.0, "length": 70}] * 3
    growing = [perf(s * 10_000, 1000 + s * 300) for s in range(6)]
    steady = [perf(s * 10_000, 1000 + (s % 2) * 10) for s in range(6)]

    assert summary(episodes, growing)["Memory trend"] == (
        "possible leak: +34% after warm-up"
    )
    assert summary(episodes, steady)["Memory trend"].startswith("ok (")
    assert summary(episodes, steady)["Mean reward, last 100"] == "-90.0"


def test_report_compares_agents_with_data(db):
    for episode in range(25):
        db.add_episode("dqn", episode, reward=-90.0 + episode, length=70)
        db.add_episode("ppo", episode, reward=-95.0, length=60)
    for step in (5000, 10_000, 15_000):
        db.add_performance(12.0, 1300.0, 20.0, 13.0, agent="dqn", step=step)

    page = build_report(db, ["dqn", "ppo", "a2c"])

    assert "<th>dqn</th><th>ppo</th></tr>" in page  # a2c has no data
    assert "<tr><th>Episodes</th><td>25</td><td>25</td></tr>" in page
    assert page.count("data:image/png;base64,") == 3


def test_report_without_data(db):
    assert "run <code>train</code> first" in build_report(db, ["dqn"])


def test_cli_report_writes_html(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with MetricsDB() as db:
        db.add_episode("dqn", 1, reward=-90.0, length=70)

    result = CliRunner().invoke(cli.app, ["report", "--out", "out/report.html"])

    assert result.exit_code == 0, result.output
    assert (
        (tmp_path / "out" / "report.html")
        .read_text(encoding="utf-8")
        .startswith("<!doctype html>")
    )
