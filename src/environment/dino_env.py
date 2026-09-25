"""Chrome Dinosaur Game Gymnasium Environment"""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .chrome_game import ChromeGame

ALIVE_REWARD = 0.1
CRASH_REWARD = -100.0


class ChromeDinoEnv(gym.Env):
    """Gymnasium environment for Chrome Dinosaur Game.

    The game runs in real Chrome (see chrome_game.py) in real time, so each
    step waits `step_seconds` of game time. render_mode="human" shows the
    Chrome window instead of running headless.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, step_seconds=0.05, game=None):
        """Initialize environment; `game` replaces Chrome (used by tests)"""
        super().__init__()
        self.render_mode = render_mode
        self.step_seconds = step_seconds
        self._game = game

        # Action space: 0=nothing, 1=jump, 2=duck
        self.action_space = spaces.Discrete(3)

        # Observation space: one 84x84 grayscale frame; stack frames with a
        # wrapper (e.g. SB3 VecFrameStack) at training time
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        """Start a new run; launches Chrome on first use"""
        super().reset(seed=seed)
        if self._game is None:
            self._game = ChromeGame(headless=self.render_mode != "human")
        self._game.restart()
        return self._game.frame(), self._game.state()

    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        self._game.act(int(action))
        time.sleep(self.step_seconds)
        state = self._game.state()
        terminated = bool(state["crashed"])
        reward = CRASH_REWARD if terminated else ALIVE_REWARD
        return self._game.frame(), reward, terminated, False, state

    def pause(self):
        """Freeze the real-time game between steps (no-op before reset)"""
        if self._game is not None:
            self._game.pause()

    def resume(self):
        if self._game is not None:
            self._game.resume()

    def render(self):
        """Nothing to do: with render_mode="human" Chrome is visible"""

    def close(self):
        if self._game is not None:
            self._game.close()
            self._game = None
