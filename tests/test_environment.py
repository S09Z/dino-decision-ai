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
        self.paused = False
        self.frames = 0  # lockstep frames run
        self.speeds = []  # set_speed() calls

    def restart(self):
        self.actions = []
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def act(self, action):
        self.actions.append(action)

    def advance(self, frames):
        self.frames += frames

    def set_speed(self, speed):
        self.speeds.append(speed)

    def state(self):
        return {
            "crashed": len(self.actions) >= self.crash_after,
            "score": 0,
            "speed": 6.0,
            "jumping": False,
            "ducking": False,
            "obstacles": [{"type": "cactus", "d": 50, "w": 17, "y": 105, "h": 35}],
        }

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
        # the obstacle it crashed into, as the Laya player reads it
        assert info["obstacles"][0]["type"] in ("cactus", "bird")
        assert info["obstacles"][0]["d"] < 20
        obs, info = env.reset()
        assert not info["crashed"]
        assert info["speed"] >= 6  # the game's starting SPEED
    finally:
        env.close()


def test_pause_and_resume_reach_the_game():
    game = FakeGame()
    env = ChromeDinoEnv(step_seconds=0, game=game)

    env.pause()
    assert game.paused
    env.resume()
    assert not game.paused


def test_pause_before_reset_is_a_no_op():
    env = ChromeDinoEnv(step_seconds=0)

    env.pause()
    env.resume()  # no Chrome launched


def test_two_actions_drop_duck():
    env = ChromeDinoEnv(step_seconds=0, game=FakeGame(), n_actions=2)

    assert env.action_space.n == 2
    check_env(env, skip_render_check=True)


def test_make_dino_env_passes_env_options():
    from src.training.envs import make_dino_env

    env = make_dino_env(game=FakeGame(), n_actions=2, crop=(0, 0, 300, 150))

    assert env.action_space.n == 2
    assert env.get_attr("crop") == [(0, 0, 300, 150)]
    env.close()


def test_lockstep_steps_advance_the_game_instead_of_waiting():
    game = FakeGame(crash_after=10)
    env = ChromeDinoEnv(game=game, frames_per_step=4)  # step_seconds unused
    env.reset()

    env.step(0)
    env.step(1)

    assert game.frames == 8


def test_start_speed_starts_each_run_at_a_random_speed_in_range():
    game = FakeGame()
    env = ChromeDinoEnv(step_seconds=0, game=game, start_speed=(6, 13))

    env.reset(seed=0)
    for _ in range(20):
        env.reset()

    assert len(game.speeds) == 21
    assert all(6 <= speed <= 13 for speed in game.speeds)
    assert len(set(game.speeds)) == 21  # a new speed each run


def test_start_speed_share_leaves_the_other_runs_at_the_normal_start():
    game = FakeGame()
    env = ChromeDinoEnv(
        step_seconds=0, game=game, start_speed=(6, 13), start_speed_share=0.5
    )

    env.reset(seed=0)
    for _ in range(99):
        env.reset()

    assert 30 < len(game.speeds) < 70  # about half of 100 runs


def test_press_cost_comes_off_steps_that_press_a_key():
    env = ChromeDinoEnv(step_seconds=0, game=FakeGame(crash_after=10), press_cost=0.05)
    env.reset()

    rewards = [env.step(action)[1] for action in (0, 1, 2)]

    pressed = ALIVE_REWARD - 0.05  # jump and duck both press a key
    assert rewards == pytest.approx([ALIVE_REWARD, pressed, pressed])


def test_runs_start_at_the_games_own_speed_by_default():
    game = FakeGame()
    ChromeDinoEnv(step_seconds=0, game=game).reset()

    assert game.speeds == []
