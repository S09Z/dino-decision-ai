"""Shared wrapper for Stable-Baselines3 agents"""

import warnings
from dataclasses import asdict
from typing import Any, ClassVar

from src.config.local_config import LocalConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
# SB3's progress bar uses tqdm.rich, which warns that it is experimental
warnings.filterwarnings("ignore", message="rich is experimental")


class SB3Agent:
    """Builds `algorithm("CnnPolicy", env, **asdict(config))` and exposes
    train/predict/save/load. Subclasses set `algorithm` and `default_config`."""

    algorithm: ClassVar[Any]
    default_config: ClassVar[Any]

    def __init__(self, env, config=None, device=None):
        """Build the model on `env` (see src.training.envs.make_dino_env)"""
        self.env = env
        self.config = config if config is not None else self.default_config()
        self.model = self.algorithm(
            "CnnPolicy",
            env,
            device=device or LocalConfig.DEVICE,
            **asdict(self.config),
        )

    @property
    def name(self) -> str:
        return self.algorithm.__name__

    def train(self, total_steps, callback=None, progress_bar=False):
        """Train the agent for `total_steps` more env steps; `callback` is an
        SB3 callback (e.g. src.training.callbacks.TrainingMonitor);
        `progress_bar` shows steps done and time left. The step count carries
        on from a resumed model, and so do schedules such as DQN's epsilon."""
        logger.info(
            "%s training for %d steps on %s", self.name, total_steps, self.model.device
        )
        self.model.learn(
            total_steps,
            callback=callback,
            progress_bar=progress_bar,
            reset_num_timesteps=False,
        )
        logger.info("%s training done", self.name)

    def predict(self, obs, deterministic=True):
        """Predict (action, state) for an observation"""
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path):
        """Save model (SB3 zip)"""
        self.model.save(path)

    def load(self, path):
        """Load model weights and settings saved with `save`"""
        self.model = self.algorithm.load(path, env=self.env, device=self.model.device)

    def resume(self, latest):
        """Continue an interrupted run from a CheckpointManager `Latest`: its
        weights, step count and, for DQN, its replay buffer"""
        self.load(latest.path)
        if latest.buffer:
            self.model.load_replay_buffer(latest.buffer)
