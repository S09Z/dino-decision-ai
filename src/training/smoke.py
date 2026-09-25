"""Short training run compared against a random policy.

Usage: python -m src.training.smoke --agent ppo --steps 20000 --episodes 20
(real time: ~12 steps/s, so 20k steps is ~30 minutes)
"""

from pathlib import Path

import typer

from src.evaluation.evaluate import RandomPolicy, evaluate, report
from src.models import AGENTS
from src.training.envs import make_dino_env
from src.utils.logger import get_logger

logger = get_logger(__name__)
MODELS_DIR = Path("models")  # git-ignored


def main(agent: str = "dqn", steps: int = 20_000, episodes: int = 20):
    env = make_dino_env()
    try:
        random_result = report("random", *evaluate(RandomPolicy(env), env, episodes))
        logger.info(random_result)
        model = AGENTS[agent](env)
        model.train(steps, progress_bar=True)
        MODELS_DIR.mkdir(exist_ok=True)
        model.save(MODELS_DIR / f"{agent}_smoke")
        agent_result = report(agent, *evaluate(model.model, env, episodes))
        logger.info(agent_result)
        print(f"{random_result}\n{agent_result}")
    finally:
        env.close()


if __name__ == "__main__":
    typer.run(main)
