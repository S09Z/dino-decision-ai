"""Measure Laya as the router: load time, memory, time per decision, and how
often it agrees with the score heuristic. Downloads the model (~843MB) on
first run.

Usage: python -m src.routing.laya_benchmark --out docs/LAYA_BENCHMARK.md
"""

import platform
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import typer

from src.profiling.profiler import resource_usage
from src.routing.laya_classifier import MODEL, LayaClassifier
from src.routing.laya_router import DIFFICULTIES, heuristic

MB = 1024**2

# (dqn, ppo) recent scores: ties, small and large leads either way
SCORE_PAIRS = [
    (0, 0),
    (100, 100),
    (100, 120),
    (120, 100),
    (50, 500),
    (500, 50),
    (1850, 2340),
    (2340, 1850),
]


def benchmark(classifier: LayaClassifier, repeats: int = 5) -> dict:
    """Load the model, then decide every (scores, difficulty) case `repeats`
    times; returns timings, memory and one row per case"""
    before = psutil.Process().memory_info().rss
    classifier.load()
    loaded = psutil.Process().memory_info().rss
    rows, seconds = [], []
    for difficulty in DIFFICULTIES:
        for dqn, ppo in SCORE_PAIRS:
            scores = {"dqn": float(dqn), "ppo": float(ppo)}
            for _ in range(repeats):
                start = time.perf_counter()
                agent, confidence = classifier(scores, difficulty)
                seconds.append(time.perf_counter() - start)
            expected, expected_confidence = heuristic(scores)
            rows.append(
                (difficulty, dqn, ppo, agent, confidence, expected, expected_confidence)
            )
    ms = np.array(seconds) * 1000
    return {
        "load_seconds": classifier.load_seconds,
        "model_mb": (loaded - before) / MB,
        "gpu_memory_mb": resource_usage()["gpu_memory_mb"],
        "mean_ms": float(ms.mean()),
        "p95_ms": float(np.percentile(ms, 95)),
        "first_ms": float(ms[0]),
        "agreement": sum(r[3] == r[5] for r in rows) / len(rows),
        "rows": rows,
    }


def main(
    out: Optional[Path] = None, device: Optional[str] = None, repeats: int = 5
) -> None:
    result = benchmark(LayaClassifier(device=device), repeats)
    table = [
        "| Difficulty | DQN | PPO | Laya | Confidence | Heuristic | Confidence |",
        "|---|---|---|---|---|---|---|",
    ] + [
        f"| {d} | {dqn} | {ppo} | {agent} | {conf:.2f} | {exp} | {exp_conf:.2f} |"
        for d, dqn, ppo, agent, conf, exp, exp_conf in result["rows"]
    ]
    report = "\n".join(
        [
            "# Laya router benchmark",
            "",
            f"Measured {date.today()} with `python -m src.routing.laya_benchmark"
            f" --repeats {repeats}` ({MODEL}, device {device or 'auto'}).",
            f"Machine: {platform.platform()}.",
            "",
            f"- **Load time:** {result['load_seconds']:.1f}s"
            " (after the download; the first run downloads ~843MB)",
            f"- **Memory added by the model (process RSS):**"
            f" {result['model_mb']:.0f}MB",
            f"- **GPU memory:** {result['gpu_memory_mb']:.0f}MB",
            f"- **Per decision:** mean {result['mean_ms']:.0f}ms,"
            f" p95 {result['p95_ms']:.0f}ms, first {result['first_ms']:.0f}ms"
            " (the game runs ~50ms per step, and play pauses it while Laya decides)",
            f"- **Agrees with the score heuristic:** {result['agreement']:.0%}"
            f" of {len(result['rows'])} cases",
            "",
            "## Decisions",
            "",
            *table,
            "",
        ]
    )
    print(report)
    if out is not None:
        out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    typer.run(main)
