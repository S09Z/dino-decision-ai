"""PPO hyperparameters"""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Field names match stable_baselines3.PPO arguments:
    PPO(policy, env, **asdict(PPOConfig()))
    """

    learning_rate: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
