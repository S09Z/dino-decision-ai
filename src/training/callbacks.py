"""SB3 callback that records training to MetricsDB and keeps checkpoints"""

import time
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from src.profiling.profiler import resource_usage


class TrainingMonitor(BaseCallback):
    """Writes every finished episode to the DB. Every `checkpoint_every` steps
    and at the end of training it also records loss, learning rate and
    performance, and saves a checkpoint scored by the mean reward of the last
    `window` episodes (CheckpointManager keeps only the best)."""

    def __init__(
        self,
        name: str,
        db: MetricsDB,
        checkpoints: CheckpointManager,
        checkpoint_every: int = 5_000,
        window: int = 10,
    ):
        super().__init__()
        self.name = name
        self.db = db
        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.episodes = 0
        self.recent_rewards: deque[float] = deque(maxlen=window)
        self._last_step = 0
        self._last_time = time.perf_counter()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            episode = info.get("episode")  # added by Monitor when an episode ends
            if episode:
                self.episodes += 1
                self.recent_rewards.append(float(episode["r"]))
                self.db.add_episode(
                    self.name, self.episodes, float(episode["r"]), int(episode["l"])
                )
        if self.num_timesteps % self.checkpoint_every == 0:
            self._checkpoint()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps != self._last_step:
            self._checkpoint()

    def _checkpoint(self) -> None:
        now = time.perf_counter()
        steps_per_s = (self.num_timesteps - self._last_step) / (now - self._last_time)
        usage = resource_usage()
        self.db.add_performance(
            steps_per_s=steps_per_s,
            memory_mb=usage["memory_mb"],
            cpu_percent=usage["cpu_percent"],
            gpu_memory_mb=usage["gpu_memory_mb"],
        )
        logged = self.model.logger.name_to_value
        self.db.add_training(
            self.name,
            self.num_timesteps,
            loss=logged.get("train/loss"),
            learning_rate=logged.get("train/learning_rate"),
        )
        if self.recent_rewards:
            self.checkpoints.save(
                self.model,
                self.name,
                step=self.num_timesteps,
                reward=float(np.mean(self.recent_rewards)),
            )
        self._last_step = self.num_timesteps
        self._last_time = time.perf_counter()
