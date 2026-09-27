"""HTML report comparing agents' training runs from MetricsDB: learning
curves, speed and memory over training, and a memory-leak check.

Usage: python -m src.cli.main report --out models/logs/report.html
"""

import base64
import html
import io
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # render to files, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.monitoring.local_db import MetricsDB  # noqa: E402

# Memory growing more than this (last third of training vs middle third) is
# flagged as a possible leak
LEAK_GROWTH_PERCENT = 20.0
ROLLING = 20  # episodes in the rolling mean of the reward curve


def memory_growth(performance: list[dict]) -> Optional[float]:
    """Percent change in mean memory from the middle third of the samples to
    the last third. The first third is skipped: memory climbs while Chrome and
    Python warm up, then saw-tooths with garbage collection, so a straight-line
    fit over short runs misreads both as leaks. None below 6 samples."""
    memory = [r["memory_mb"] for r in performance if r["step"] is not None]
    third = len(memory) // 3
    if third < 2:
        return None
    middle = np.mean(memory[third : 2 * third])
    last = np.mean(memory[-third:])
    return float((last - middle) / middle * 100)


def summary(episodes: list[dict], performance: list[dict]) -> dict[str, str]:
    rewards = [r["reward"] for r in episodes]
    lengths = [r["length"] for r in episodes]
    speeds = [r["steps_per_s"] for r in performance]
    memory = [r["memory_mb"] for r in performance]
    growth = memory_growth(performance)
    if growth is None:
        leak = "not enough data"
    elif growth > LEAK_GROWTH_PERCENT:
        leak = f"possible leak: {growth:+.0f}% after warm-up"
    else:
        leak = f"ok ({growth:+.0f}% after warm-up)"
    last = slice(-100, None)
    return {
        "Episodes": str(len(episodes)),
        "Mean reward, last 100": f"{np.mean(rewards[last]):.1f}" if rewards else "-",
        "Mean length, last 100": f"{np.mean(lengths[last]):.0f}" if lengths else "-",
        "Mean steps/s": f"{np.mean(speeds):.1f}" if speeds else "-",
        "Peak memory": f"{max(memory):.0f}MB" if memory else "-",
        "Memory trend": leak,
    }


def _png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _charts(runs: dict[str, tuple[list[dict], list[dict]]]) -> list[tuple[str, str]]:
    """(title, base64 PNG) for reward, speed and memory, one line per agent"""
    reward_fig, reward_ax = plt.subplots(figsize=(8, 3.5))
    speed_fig, speed_ax = plt.subplots(figsize=(8, 3.5))
    memory_fig, memory_ax = plt.subplots(figsize=(8, 3.5))
    for name, (episodes, performance) in runs.items():
        rewards = np.array([r["reward"] for r in episodes], dtype=float)
        if len(rewards) >= ROLLING:
            rolling = np.convolve(rewards, np.ones(ROLLING) / ROLLING, mode="valid")
            reward_ax.plot(np.arange(ROLLING, len(rewards) + 1), rolling, label=name)
        elif len(rewards):
            reward_ax.plot(np.arange(1, len(rewards) + 1), rewards, label=name)
        stepped = [r for r in performance if r["step"] is not None]
        steps = [r["step"] for r in stepped]
        speed_ax.plot(
            steps, [r["steps_per_s"] for r in stepped], marker="o", label=name
        )
        memory_ax.plot(steps, [r["memory_mb"] for r in stepped], marker="o", label=name)
    reward_ax.set(
        xlabel="episode", ylabel=f"reward (rolling mean of {ROLLING} once available)"
    )
    speed_ax.set(xlabel="training step", ylabel="steps/s")
    memory_ax.set(xlabel="training step", ylabel="memory, Python + Chrome (MB)")
    charts = []
    for title, fig, ax in (
        ("Learning curve", reward_fig, reward_ax),
        ("Speed", speed_fig, speed_ax),
        ("Memory (leak check)", memory_fig, memory_ax),
    ):
        ax.legend()
        ax.grid(alpha=0.3)
        charts.append((title, _png(fig)))
    return charts


def build_report(db: MetricsDB, agents: list[str]) -> str:
    """Self-contained HTML page (charts embedded) for the agents with data"""
    runs = {
        name: (db.history("episodes", name), db.history("performance", name))
        for name in agents
    }
    runs = {name: run for name, run in runs.items() if run[0] or run[1]}
    if not runs:
        return "<p>No training data yet: run <code>train</code> first.</p>"
    summaries = {name: summary(*run) for name, run in runs.items()}
    rows = [
        "<tr><th></th>" + "".join(f"<th>{html.escape(n)}</th>" for n in runs) + "</tr>"
    ]
    for label in next(iter(summaries.values())):
        cells = "".join(f"<td>{html.escape(s[label])}</td>" for s in summaries.values())
        rows.append(f"<tr><th>{label}</th>{cells}</tr>")
    charts = "".join(
        f'<h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{png}">'
        for title, png in _charts(runs)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Training report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 860px;
       padding: 0 1rem; color: #222; background: #fff; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: .35rem .7rem; text-align: left; }}
img {{ max-width: 100%; }}
</style></head><body>
<h1>Training report</h1>
<p>From <code>models/logs/metrics.db</code>. Memory is Python plus Chrome;
a rise above {LEAK_GROWTH_PERCENT:.0f}% from the middle to the last third of
training is flagged.</p>
<table>{''.join(rows)}</table>
{charts}
</body></html>
"""
