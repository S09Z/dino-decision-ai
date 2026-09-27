"""DQN hyperparameters"""

from dataclasses import dataclass


@dataclass
class DQNConfig:
    """Field names match stable_baselines3.DQN arguments:
    DQN(policy, env, **asdict(DQNConfig()))
    """

    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    batch_size: int = 32
    gamma: float = 0.99
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.1
    target_update_interval: int = 1000
