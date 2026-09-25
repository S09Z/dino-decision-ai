"""Evaluate a policy over whole episodes and summarise the result"""

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy


class RandomPolicy:
    """Uniform random actions, with the predict() signature evaluate_policy uses"""

    def __init__(self, env):
        self.env = env

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        return np.array([self.env.action_space.sample() for _ in obs]), state


def evaluate(policy, env, episodes: int) -> tuple[list[float], list[int]]:
    """Per-episode rewards and lengths over `episodes` episodes"""
    rewards, lengths = evaluate_policy(
        policy, env, n_eval_episodes=episodes, return_episode_rewards=True
    )
    return list(rewards), list(lengths)  # type: ignore[arg-type]


def report(name: str, rewards: list[float], lengths: list[int]) -> str:
    return (
        f"{name}: reward {np.mean(rewards):.1f} ± {np.std(rewards):.1f}, "
        f"length {np.mean(lengths):.0f} ± {np.std(lengths):.0f} steps"
    )
