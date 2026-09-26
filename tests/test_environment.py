"""Tests for the Chrome Dino Gymnasium environment"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.environment.dino_env import ALIVE_REWARD, CRASH_REWARD, ChromeDinoEnv


class FakeGame:
    """Stands in for ChromeGame: crashes after `crash_after` actions"""

    def __init__(self, crash_after=3):
        self.crash_after = crash_after
        self.actions = []
        self.closed = False

    def restart(self):
        self.actions = []

    def act(self, action):
        self.actions.append(action)

    def state(self):
        return {"crashed": len(self.actions) >= self.crash_after, "score": 0}

    def frame(self):
        return np.zeros((84, 84, 1), dtype=np.uint8)

    def close(self):
        self.closed = True


def make_env(crash_after=3):
    return ChromeDinoEnv(step_seconds=0, game=FakeGame(crash_after))


def test_env_passes_gymnasium_checker():
    check_env(make_env(crash_after=1000), skip_render_check=True)


def test_observation_is_single_grayscale_frame():
    obs, _ = make_env().reset(seed=0)
    assert obs.shape == (84, 84, 1)


def test_alive_steps_reward_until_crash_terminates():
    env = make_env(crash_after=3)
    env.reset()
    results = [env.step(1)[1:3] for _ in range(3)]
    assert results == [
        (ALIVE_REWARD, False),
        (ALIVE_REWARD, False),
        (CRASH_REWARD, True),
    ]


def test_actions_reach_the_game_and_close_releases_it():
    game = FakeGame()
    env = ChromeDinoEnv(step_seconds=0, game=game)
    env.reset()
    for action in (0, 1, 2):
        env.step(action)
    assert game.actions == [0, 1, 2]
    env.close()
    assert game.closed


def test_random_agent_in_real_chrome():
    env = ChromeDinoEnv()
    try:
        obs, info = env.reset(seed=0)
    except Exception as e:  # Chrome or Playwright unavailable on this machine
        pytest.skip(f"real Chrome unavailable: {e}")
    try:
        assert obs.shape == (84, 84, 1) and obs.max() > obs.min()
        for _ in range(400):
            obs, reward, terminated, _, info = env.step(env.action_space.sample())
            if terminated:
                break
        assert terminated and reward == CRASH_REWARD
        obs, info = env.reset()
        assert not info["crashed"]
    finally:
        env.close()
