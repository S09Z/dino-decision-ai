"""Proximal Policy Optimization (PPO) Agent"""

from dataclasses import replace

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from src.config.ppo_config import PPOConfig
from src.models.sb3_agent import SB3Agent
from src.training.callbacks import PauseDuringUpdates


class PPOAgent(SB3Agent):
    """PPO agent for Chrome Dinosaur Game.

    Stable-Baselines3 PPO: CnnPolicy with shared-CNN actor and critic heads,
    GAE advantages and the clipped surrogate loss, with PPOConfig values.
    """

    algorithm = PPO
    default_config = PPOConfig

    @staticmethod
    def for_envs(config, n_envs: int):
        """Keep n_steps samples per update, split across the games"""
        return replace(config, n_steps=config.n_steps // n_envs)

    def train(self, total_steps, callback=None, progress_bar=False):
        """Like SB3Agent.train, but the game is paused during each update
        (~9s on MPS), otherwise the dino crashes with nobody playing"""
        callbacks = [PauseDuringUpdates()] + ([callback] if callback else [])
        super().train(total_steps, CallbackList(callbacks), progress_bar)
