"""Short DQN training run compared against a random policy.

Usage: python -m src.training.dqn_smoke --steps 20000 --episodes 20
(real time: ~12 steps/s, so 20k steps is ~30 minutes)
"""

from pathlib import Path

import typer

from src.evaluation.evaluate import RandomPolicy, evaluate, report
from src.models.dqn_agent import DQNAgent
from src.training.envs import make_dino_env
from src.utils.logger import get_logger

logger = get_logger(__name__)
MODEL_PATH = Path("models") / "dqn_smoke"  # git-ignored


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
