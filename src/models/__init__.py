"""RL agents; AGENTS maps CLI names to agent classes"""

from src.models.dqn_agent import DQNAgent
from src.models.ppo_agent import PPOAgent

AGENTS = {"dqn": DQNAgent, "ppo": PPOAgent}
