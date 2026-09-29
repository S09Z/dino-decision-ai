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
    Chrome window instead of running headless. `n_actions=2` drops duck
    (0 = nothing, 1 = jump); `crop` is passed on to ChromeGame.

    `frames_per_step=N` plays in lockstep instead: each step runs exactly N
    game frames (1/60s each) as fast as Chrome computes them, and the game
    waits between steps.

    `start_speed=(low, high)` starts each run at a random speed in that range
    instead of 6, so training reaches fast games (birds from 8.5) without
    first surviving the slow part. `start_speed_share` is the share of runs
    that do; the rest start at 6 as usual (so the slow part is practised too).

    `press_cost` is taken off the reward of every step that presses a key:
    otherwise jumping with nothing ahead costs nothing, the agent's values
    for "jump" and "nothing" differ only by noise there, and it hops at random.

    `step_seconds` is 0.065 so a real-time step still lasts ~72ms, as it did
    when frames took 15ms to read (now 1.6ms): agents trained then collapse
    at shorter steps (round 2's best: 112 steps at 0.05 vs 1782 at 0.065).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode=None,
        step_seconds=0.065,
        game=None,
        n_actions=3,
        crop=None,
        frames_per_step=None,
        start_speed=None,
        start_speed_share=1.0,
        press_cost=0.0,
    ):
        """Initialize environment; `game` replaces Chrome (used by tests)"""
        super().__init__()
        self.render_mode = render_mode
        self.step_seconds = step_seconds
        self.crop = crop
        self.frames_per_step = frames_per_step
        self.start_speed = start_speed
        self.start_speed_share = start_speed_share
        self.press_cost = press_cost
        self._game = game

        # Action space: 0=nothing, 1=jump, 2=duck (n_actions=2: no duck)
        self.action_space = spaces.Discrete(n_actions)

        # Observation space: one 84x84 grayscale frame; stack frames with a
        # wrapper (e.g. SB3 VecFrameStack) at training time
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        """Start a new run; launches Chrome on first use"""
        super().reset(seed=seed)
        if self._game is None:
            self._game = ChromeGame(
                headless=self.render_mode != "human",
                crop=self.crop,
                lockstep=self.frames_per_step is not None,
            )
        self._game.restart()
        if (
            self.start_speed is not None
            and self.np_random.random() < self.start_speed_share
        ):
            self._game.set_speed(float(self.np_random.uniform(*self.start_speed)))
        return self._game.frame(), self._game.state()

    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        self._game.act(int(action))
        if self.frames_per_step is not None:
            self._game.advance(self.frames_per_step)
        else:
            time.sleep(self.step_seconds)
        state = self._game.state()
        terminated = bool(state["crashed"])
        reward = CRASH_REWARD if terminated else ALIVE_REWARD
        if int(action) != 0:
            reward -= self.press_cost
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
