"""SB3 callback that records training to MetricsDB and keeps checkpoints"""

import time
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.evaluation.evaluate import evaluate_games
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from src.profiling.profiler import resource_usage


class PauseDuringUpdates(BaseCallback):
    """Pauses the real-time game while the algorithm updates its networks
    (between rollouts) and resumes it for the next rollout and at the end"""

    def _on_rollout_end(self) -> None:
        self.training_env.env_method("pause")

    def _on_rollout_start(self) -> None:
        self.training_env.env_method("resume")

    def _on_training_end(self) -> None:
        self.training_env.env_method("resume")

    def _on_step(self) -> bool:
        return True


class EvalMonitor(BaseCallback):
    """Every `eval_every` steps, plays `episodes` greedy games on `eval_env`
    (one normal game, stopped at `max_length` steps) and saves the model as
    checkpoint `<name>-eval`, scored by their mean length: training rewards
    can mislead (games that start fast are shorter), this cannot. Lockstep
    only: real-time training games would run on unattended meanwhile.
    `history` holds (step, mean, median) per evaluation."""

    def __init__(
        self,
        name: str,
        eval_env,
        checkpoints: CheckpointManager,
        eval_every: int = 50_000,
        episodes: int = 10,
        max_length: int = 5_000,  # 3,000 was reached by most games of -mix@2
    ):
        super().__init__()
        self.name = f"{name}-eval"
        self.eval_env = eval_env
        self.checkpoints = checkpoints
        self.eval_every = eval_every
        self.episodes = episodes
        self.max_length = max_length
        self.history: list[tuple[int, float, float]] = []
        self._last_step = 0

    def _on_training_start(self) -> None:
        self._last_step = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_step >= self.eval_every:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps != self._last_step:
            self._evaluate()

    def _evaluate(self) -> None:
        games = evaluate_games(
            self.model, self.eval_env, self.episodes, self.max_length
        )
        lengths = [game["length"] for game in games]
        mean = float(np.mean(lengths))
        self.history.append((self.num_timesteps, mean, float(np.median(lengths))))
        self.checkpoints.save(self.model, self.name, self.num_timesteps, mean)
        self._last_step = self.num_timesteps


class TrainingMonitor(BaseCallback):
    """Writes every finished episode to the DB. Every `checkpoint_every` steps
    (counted over all games) and at the end of training it also records loss, learning rate and
    performance, saves a checkpoint scored by the mean reward of the last
    `window` episodes (CheckpointManager keeps only the best) and overwrites
    the latest checkpoint, which `train --resume` continues from. `episodes`
    is the count so far when resuming, so numbering carries on."""

    def __init__(
        self,
        name: str,
        db: MetricsDB,
        checkpoints: CheckpointManager,
        checkpoint_every: int = 5_000,
        window: int = 10,
        episodes: int = 0,
    ):
        super().__init__()
        self.name = name
        self.db = db
        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.episodes = episodes
        self.recent_rewards: deque[float] = deque(maxlen=window)
        self._last_step = 0
        self._last_time = time.perf_counter()

    def _on_training_start(self) -> None:
        # a resumed model starts at its saved step, not 0: measure speed from here
        self._last_step = self.num_timesteps
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
        # a distance, not a multiple: with several games the count moves in
        # steps of n_envs and would land on a multiple only every
        # lcm(n_envs, checkpoint_every) steps
        if self.num_timesteps - self._last_step >= self.checkpoint_every:
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
            agent=self.name,
            step=self.num_timesteps,
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
        # DQN's replay buffer takes seconds to write: pause the real-time game
        # meanwhile, or the dino crashes with nobody playing
        self.training_env.env_method("pause")
        try:
            self.checkpoints.save_latest(
                self.model, self.name, self.num_timesteps, self.episodes
            )
        finally:
            self.training_env.env_method("resume")
        self._last_step = self.num_timesteps
        self._last_time = time.perf_counter()
