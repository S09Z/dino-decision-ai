"""Deep Q-Network (DQN) Agent"""

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
