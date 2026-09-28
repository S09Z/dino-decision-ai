"""Tests for the dashboard load test (a real local server on a temporary DB)"""

from src.dashboard.loadtest import report, run


def test_every_row_reaches_every_client_quickly():
    result = run(clients=2, rate=20, seconds=0.5)

    assert result["rows"] == 10
    assert result["delivered"] == result["expected"] == 20
    assert result["p95_ms"] < 1000  # generous: CI machines vary
    assert result["rest_p50_ms"] is not None


def test_report_is_a_markdown_table():
    row = {
        "clients": 5,
        "interval_ms": 100.0,
        "rate": 20.0,
        "delivered": 1000,
        "expected": 1000,
        "p50_ms": 51.2,
        "p95_ms": 98.4,
        "max_ms": 130.0,
        "rest_p50_ms": 5.0,
        "rest_p95_ms": 16.0,
    }

    text = report([row])

    assert text.startswith("# Dashboard latency")
    assert (
        "| 100 ms | 5 | 20 | 1000/1000 | 51 ms | 98 ms | 130 ms | 5 ms | 16 ms |"
        in text
    )
