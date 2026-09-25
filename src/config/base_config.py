"""Base configuration"""


class BaseConfig:
    """Base configuration for all environments"""

    # Game settings
    GAME_SPEED = 1.0

    # Training settings
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    GAMMA = 0.99

    # DQN settings
    BUFFER_SIZE = 50000
    EPSILON_START = 1.0
    EPSILON_END = 0.1

    # PPO settings
    ENTROPY_COEFF = 0.01
    GAE_LAMBDA = 0.95

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
