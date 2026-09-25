"""Build the vectorised, frame-stacked Dino env that SB3 agents train on"""

from typing import Optional

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from src.environment.dino_env import ChromeDinoEnv


def make_dino_env(
    n_stack: int = 4, render_mode: Optional[str] = None, game=None
) -> VecEnv:
    """One ChromeDinoEnv (Monitor-wrapped for episode stats) stacking the last
    `n_stack` frames into (84, 84, n_stack) observations; `game` replaces
    Chrome (used by tests)"""
    kwargs = {} if game is None else {"step_seconds": 0, "game": game}
    env = make_vec_env(
        lambda: ChromeDinoEnv(render_mode=render_mode, **kwargs), n_envs=1
    )
    return VecFrameStack(env, n_stack, channels_order="last")
