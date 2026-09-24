"""Deep Q-Network (DQN) Agent"""

import torch
import torch.nn as nn
from stable_baselines3 import DQN


class DQNAgent:
    """DQN agent for Chrome Dinosaur Game"""

    def __init__(self, env, config=None):
        """Initialize DQN agent"""
        self.env = env
        self.config = config or {}
        self.model = None

    def train(self, total_steps):
        """Train the agent"""
        # TODO: Implement training loop
        pass

    def predict(self, obs):
        """Predict action for observation"""
        # TODO: Implement prediction
        return 0, None

    def save(self, path):
        """Save model"""
        # TODO: Implement saving
        pass

    def load(self, path):
        """Load model"""
        # TODO: Implement loading
        pass
