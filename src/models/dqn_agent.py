"""Deep Q-Network (DQN) Agent"""

from dataclasses import asdict

from stable_baselines3 import DQN

from src.config.dqn_config import DQNConfig
from src.config.local_config import LocalConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DQNAgent:
    """DQN agent for Chrome Dinosaur Game.

    Wraps Stable-Baselines3 DQN: CnnPolicy (Nature CNN), replay buffer,
    target network and epsilon-greedy exploration, with DQNConfig values.
    """

    def __init__(self, env, config: DQNConfig = DQNConfig(), device=None):
        """Build the model on `env` (see src.training.envs.make_dino_env)"""
        self.env = env
        self.config = config
        self.model = DQN(
            "CnnPolicy", env, device=device or LocalConfig.DEVICE, **asdict(config)
        )

    def train(self, total_steps, callback=None):
        """Train the agent for `total_steps` env steps; `callback` is an SB3
        callback (e.g. src.training.callbacks.TrainingMonitor)"""
        logger.info("DQN training for %d steps on %s", total_steps, self.model.device)
        self.model.learn(total_steps, callback=callback)
        logger.info("DQN training done")

    def predict(self, obs, deterministic=True):
        """Predict (action, state) for an observation"""
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path):
        """Save model (SB3 zip)"""
        self.model.save(path)

    def load(self, path):
        """Load model weights and settings saved with `save`"""
        self.model = DQN.load(path, env=self.env, device=self.model.device)
