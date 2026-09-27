"""ChromeGame logic against a fake Playwright page (no Chrome)"""

import numpy as np
import pytest

from src.environment.chrome_game import ChromeGame


class FakeKeyboard:
    def __init__(self):
        self.events = []

    def press(self, key):
        self.events.append(("press", key))

    def down(self, key):
        self.events.append(("down", key))

    def up(self, key):
        self.events.append(("up", key))


class FakePage:
    """Records keyboard events and evaluated JS; `frame` is returned by
    the frame script"""

    def __init__(self, size):
        self.keyboard = FakeKeyboard()
        self.scripts = []
        self.frame = list(range(size * size))

    def evaluate(self, script, *args):
        self.scripts.append(script)
        return self.frame if args else {"crashed": False}

    def wait_for_function(self, script):
        pass


@pytest.fixture
def game():
    """ChromeGame with a fake page instead of launching Chrome"""
    game = ChromeGame.__new__(ChromeGame)
    game.frame_size = 4
    game._ducking = False
    game._started = False
    game._paused = False
    game._page = FakePage(size=4)
    return game


def keys(game):
    return game._page.keyboard.events


def test_jump_presses_space(game):
    game.act(1)

    assert keys(game) == [("press", "Space")]


def test_duck_is_held_until_another_action(game):
    game.act(2)
    game.act(2)  # still ducking: no second key-down
    game.act(0)

    assert keys(game) == [("down", "ArrowDown"), ("up", "ArrowDown")]


def test_jump_while_ducking_releases_duck_first(game):
    game.act(2)
    game.act(1)

    assert keys(game) == [
        ("down", "ArrowDown"),
        ("up", "ArrowDown"),
        ("press", "Space"),
    ]


def test_first_restart_starts_with_space_then_uses_runner_restart(game):
    game.restart()
    assert keys(game) == [("press", "Space")]

    game.restart()
    assert "Runner.instance_.restart()" in game._page.scripts[-1]
    assert keys(game) == [("press", "Space")]  # no second key press


def test_pause_and_resume_run_once_each(game):
    game.pause()
    game.pause()
    game.resume()
    game.resume()  # play() starts a new game loop, so it must not repeat

    assert game._page.scripts == [
        "() => Runner.instance_.stop()",
        "() => Runner.instance_.play()",
    ]


def test_restart_clears_paused_state(game):
    game.pause()
    game.restart()
    game.resume()  # nothing to resume: restart runs the game loop

    assert not any("play()" in s for s in game._page.scripts)


def test_frame_is_grayscale_uint8_of_frame_size(game):
    frame = game.frame()

    assert frame.shape == (4, 4, 1)
    assert frame.dtype == np.uint8
    assert frame[0, 1, 0] == 1  # row-major, as the page returns pixels
