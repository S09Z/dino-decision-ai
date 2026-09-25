"""Short DQN training run compared against a random policy.

Usage: python -m src.training.dqn_smoke --steps 20000 --episodes 20
(real time: ~12 steps/s, so 20k steps is ~30 minutes)
"""

from pathlib import Path

import numpy as np
import typer
from stable_baselines3.common.evaluation import evaluate_policy

from src.models.dqn_agent import DQNAgent
from src.training.envs import make_dino_env
from src.utils.logger import get_logger

logger = get_logger(__name__)
MODEL_PATH = Path("models") / "dqn_smoke"  # git-ignored


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


def main(steps: int = 20_000, episodes: int = 20):
    env = make_dino_env()
    try:
        random_result = report("random", *evaluate(RandomPolicy(env), env, episodes))
        logger.info(random_result)
        agent = DQNAgent(env)
        agent.train(steps)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        agent.save(MODEL_PATH)
        dqn_result = report("dqn", *evaluate(agent.model, env, episodes))
        logger.info(dqn_result)
        print(f"{random_result}\n{dqn_result}")
    finally:
        env.close()


if __name__ == "__main__":
    typer.run(main)
