"""Deep Q-Network (DQN) Agent"""

from dataclasses import replace

from stable_baselines3 import DQN

from src.config.dqn_config import DQNConfig
from src.models.sb3_agent import SB3Agent


class DQNAgent(SB3Agent):
    """DQN agent for Chrome Dinosaur Game.

    Stable-Baselines3 DQN: CnnPolicy (Nature CNN), replay buffer, target
    network and epsilon-greedy exploration, with DQNConfig values.
    """

    algorithm = DQN
    default_config = DQNConfig

    @staticmethod
    def for_envs(config, n_envs: int):
        """Each update step brings `n_envs` transitions: train on all of them"""
        return replace(config, gradient_steps=config.gradient_steps * n_envs)
