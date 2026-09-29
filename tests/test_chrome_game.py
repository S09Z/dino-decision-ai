"""ChromeGame logic against a fake Playwright page (no Chrome)"""

import numpy as np
import pytest

from src.environment.chrome_game import RUNNER, ChromeGame


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
        self.args = []
        # the frame script returns one char per pixel value
        self.frame = "".join(chr(i) for i in range(size * size))
        self.waited = []

    def evaluate(self, script, *args):
        self.scripts.append(script)
        self.args.append(args)
        return self.frame if args else {"crashed": False}

    def wait_for_function(self, script):
        self.waited.append(script)


@pytest.fixture
def game():
    """ChromeGame with a fake page instead of launching Chrome"""
    game = ChromeGame.__new__(ChromeGame)
    game.frame_size = 4
    game.crop = None
    game._ducking = False
    game._started = False
    game._paused = False
    game.lockstep = False
    game._clock_started = False
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
    assert "r.restart()" in game._page.scripts[-1]
    assert keys(game) == [("press", "Space")]  # no second key press


def test_pause_and_resume_run_once_each(game):
    game.pause()
    game.pause()
    game.resume()
    game.resume()  # play() starts a new game loop, so it must not repeat

    assert game._page.scripts == [f"() => {RUNNER}.stop()", f"() => {RUNNER}.play()"]


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


def test_frame_crop_is_passed_to_the_page(game):
    game.frame()
    game.crop = [0, 0, 300, 150]
    game.frame()

    assert game._page.args == [([4, None],), ([4, [0, 0, 300, 150]],)]


def test_unknown_game_is_rejected_before_launching_chrome():
    with pytest.raises(ValueError, match="vendored"):
        ChromeGame(game="dino.example")


def test_set_speed_calls_the_games_own_set_speed(game):
    game.set_speed(9.5)

    assert game._page.scripts == [f"(s) => {RUNNER}.setSpeed(s)"]
    assert game._page.args == [(9.5,)]


def test_lockstep_needs_the_vendored_game():
    with pytest.raises(ValueError, match="vendored"):
        ChromeGame(game="chrome", lockstep=True)


def test_lockstep_clock_starts_after_the_intro_then_advances_on_request(game):
    game.lockstep = True

    game.restart()
    game.restart()  # started once only
    game.advance(4)

    # after the whole intro, with the T-Rex put where the intro walks it to
    assert game._page.waited.count("!Runner.instance_.playingIntro") == 1
    starts = [s for s in game._page.scripts if "__clock.start()" in s]
    assert len(starts) == 1 and "t.xPos = t.config.START_X_POS" in starts[0]
    assert game._page.scripts[-1] == "(n) => __clock.advance(n)"
    assert game._page.args[-1] == (4,)
