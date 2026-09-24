"""Base configuration"""


class BaseConfig:
    """Base configuration for all environments"""

    # Game settings
    GAME_SPEED = 1.0

    # Algorithm hyperparameters live in dqn_config.py and ppo_config.py

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
