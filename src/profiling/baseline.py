"""Baseline performance report for the Chrome Dino env and DQN inference.

Usage: python -m src.profiling.baseline --steps 500 --out docs/PROFILING_BASELINE.md
"""

import platform
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from src.config.local_config import LocalConfig
from src.environment.dino_env import ChromeDinoEnv
from src.profiling.profiler import Profiler, resource_usage


def profile_env(env: ChromeDinoEnv, profiler: Profiler, steps: int) -> float:
    """Run `steps` random actions, timing env and game calls; returns steps/s"""
    env.reset()
    game = env._game
    for method in ("act", "state", "frame"):
        setattr(game, method, profiler.profile(getattr(game, method)))
    step = profiler.profile(env.step)
    start = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, _, _ = step(env.action_space.sample())
        if terminated:
            env.reset()
    return steps / (time.perf_counter() - start)


def profile_inference(profiler: Profiler, calls: int, device: str) -> float:
    """Time DQN action selection on 4 stacked frames (untrained weights);
    returns the GPU memory (MB) held while the model is loaded"""
    from src.models.dqn_agent import DQNAgent
    from src.training.envs import make_dino_env

    agent = DQNAgent(make_dino_env(game=_NoGame()), device=device)
    predict = profiler.profile(agent.predict)
    obs = np.zeros((1, 84, 84, 4), dtype=np.uint8)
    for _ in range(calls):
        predict(obs)
    return resource_usage()["gpu_memory_mb"]


class _NoGame:
    """Placeholder game so the DQN can be built without launching Chrome"""

    def restart(self):
        pass

    def close(self):
        pass


def main(steps: int = 500, out: Optional[Path] = None):
    profiler = Profiler()
    env = ChromeDinoEnv()
    try:
        steps_per_s = profile_env(env, profiler, steps)
        usage = resource_usage()  # while Chrome is still running
    finally:
        env.close()
    gpu_memory_mb = profile_inference(profiler, calls=200, device=LocalConfig.DEVICE)

    report = "\n".join(
        [
            "# Profiling baseline",
            "",
            f"Measured {date.today()} with `python -m src.profiling.baseline"
            f" --steps {steps}`: random actions in headless Chrome"
            f" (`step_seconds={env.step_seconds}`), then 200 DQN predictions"
            f" on {LocalConfig.DEVICE}.",
            f"Machine: {platform.platform()}, {LocalConfig.CPU_CORES} cores,"
            f" {LocalConfig.AVAILABLE_RAM_GB:.0f}GB RAM.",
            "",
            f"- **Env speed:** {steps_per_s:.1f} steps/s",
            f"- **Memory (Python + Chrome):** {usage['memory_mb']:.0f}MB",
            f"- **CPU (Python + Chrome):** {usage['cpu_percent']:.0f}%",
            f"- **System RAM in use:** {usage['system_ram_percent']:.0f}%",
            f"- **GPU memory (DQN loaded):** {gpu_memory_mb:.0f}MB",
            "",
            "## Function timing",
            "",
            profiler.report(),
            "",
        ]
    )
    print(report)
    if out is not None:
        out.write_text(report)


if __name__ == "__main__":
    typer.run(main)
