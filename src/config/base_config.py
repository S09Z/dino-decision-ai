"""Base configuration"""


class BaseConfig:
    """Base configuration for all environments"""

    # Game settings
    GAME_SPEED = 1.0

    # Algorithm hyperparameters live in dqn_config.py and ppo_config.py

    def __repr__(self):
        # settings are UPPER_CASE class attributes, so self.__dict__ is empty
        settings = {name: getattr(self, name) for name in dir(self) if name.isupper()}
        return f"{self.__class__.__name__}({settings})"
