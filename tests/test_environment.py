"""Tests for the Chrome Dino Gymnasium environment"""

from gymnasium.utils.env_checker import check_env

from src.environment.dino_env import ChromeDinoEnv


def test_env_passes_gymnasium_checker():
    check_env(ChromeDinoEnv(), skip_render_check=True)


def test_observation_is_single_grayscale_frame():
    obs, _ = ChromeDinoEnv().reset(seed=0)
    assert obs.shape == (84, 84, 1)
