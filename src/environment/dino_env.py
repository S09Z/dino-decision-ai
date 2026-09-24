"""Chrome Dinosaur Game Gymnasium Environment"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ChromeDinoEnv(gym.Env):
    """Gymnasium environment for Chrome Dinosaur Game"""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        """Initialize environment"""
        super().__init__()
        
        # Action space: 0=nothing, 1=jump, 2=duck
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 84x84 grayscale image (4-frame stack)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 1),
            dtype=np.uint8
        )
        
        self.episode_return = 0
        self.episode_length = 0
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        # TODO: Implement reset logic
        obs = np.zeros((84, 84, 1), dtype=np.uint8)
        return obs, {}
    
    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        # TODO: Implement step logic
        obs = np.zeros((84, 84, 1), dtype=np.uint8)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render environment"""
        pass
