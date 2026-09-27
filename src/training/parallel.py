"""Train several agents at once: each in its own process with its own Chrome.

The game is real time, so two agents cannot share one game; separate
processes also keep each agent's torch and Chrome apart. The parent process
only launches, monitors (episodes from MetricsDB, RAM of all children) and
stops them.
"""

import subprocess
import sys
from typing import Callable, Optional, Sequence

import numpy as np

from src.monitoring.local_db import MetricsDB
from src.profiling.profiler import resource_usage
from src.utils.logger import get_logger

logger = get_logger(__name__)

# RAM per agent, measured on the dev machine: Python + torch + Chrome ~1.3GB
# (docs/PROFILING_BASELINE.md), plus up to ~1.4GB for DQN's 50k replay buffer
RAM_NEEDED_GB = {"dqn": 2.7, "ppo": 1.3}


def ram_warning(names: Sequence[str], available_gb: float) -> Optional[str]:
    """A warning when free RAM looks too small to train `names` together"""
    needed = sum(RAM_NEEDED_GB.get(name, 1.3) for name in names)
    if available_gb >= needed:
        return None
    return (
        f"Training {', '.join(names)} in parallel needs ~{needed:.1f}GB RAM but "
        f"{available_gb:.1f}GB is free, so it may swap and slow the game down. "
        "Close other apps or train one at a time (without --parallel)."
    )


def train_command(name: str, steps: int, checkpoint_every: int) -> list[str]:
    """`train` for one agent, run in a child process (no progress bar: several
    bars would garble the shared terminal)"""
    return [
        sys.executable,
        "-m",
        "src.cli.main",
        "train",
        "--agent",
        name,
        "--steps",
        str(steps),
        "--checkpoint-every",
        str(checkpoint_every),
        "--no-progress-bar",
    ]


def status_line(db: MetricsDB, names: Sequence[str]) -> str:
    parts = []
    for name in names:
        recent = [row["reward"] for row in db.query_recent_episodes(name, limit=10)]
        mean = f"{np.mean(recent):.1f}" if recent else "-"
        parts.append(
            f"{name}: {db.episode_count(name)} episodes, last-10 reward {mean}"
        )
    return " | ".join(parts)


def train_parallel(
    names: Sequence[str],
    steps: int,
    checkpoint_every: int,
    launch: Callable[[list[str]], subprocess.Popen] = subprocess.Popen,
    poll_seconds: float = 60.0,
) -> dict[str, int]:
    """Train `names` at the same time; logs progress every `poll_seconds` and
    returns each agent's exit code. Stopping the parent (Ctrl+C) stops all."""
    processes = {
        name: launch(train_command(name, steps, checkpoint_every)) for name in names
    }
    try:
        with MetricsDB() as db:
            while running := [p for p in processes.values() if p.poll() is None]:
                try:
                    running[0].wait(timeout=poll_seconds)
                except subprocess.TimeoutExpired:
                    memory_gb = resource_usage()["memory_mb"] / 1024
                    logger.info("%s | RAM %.1fGB", status_line(db, names), memory_gb)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            process.wait()
    return {name: process.returncode for name, process in processes.items()}
