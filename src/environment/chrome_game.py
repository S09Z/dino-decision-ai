"""Drive the Chrome Dinosaur game in real Chrome through Playwright"""

from pathlib import Path

import numpy as np
from playwright.sync_api import Route, sync_playwright

GAME_DIR = Path(__file__).parent / "game"
# Served through request routing rather than file://, which would make the
# sprites cross-origin and block reading canvas pixels. Nothing hits the network.
GAME_ORIGIN = "http://dino.local"

# Downscale the game canvas to a size x size grayscale frame inside the page,
# so only size*size values cross the Playwright bridge per frame.
_FRAME_JS = """(size) => {
  if (!window.__frame) {
    window.__frame = document.createElement('canvas');
    window.__frame.width = size;
    window.__frame.height = size;
  }
  const ctx = window.__frame.getContext('2d', {willReadFrequently: true});
  ctx.fillStyle = '#fff';  // game canvas is transparent; page background is white
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(Runner.instance_.canvas, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const gray = new Array(size * size);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = (rgba[4 * i] * 299 + rgba[4 * i + 1] * 587 + rgba[4 * i + 2] * 114) / 1000 | 0;
  }
  return gray;
}"""

_STATE_JS = """() => {
  const r = Runner.instance_;
  return {
    crashed: r.crashed,
    playing: r.playing,
    distance: r.distanceRan,
    score: r.distanceMeter.getActualDistance(r.distanceRan),
  };
}"""


def _serve_game_file(route: Route) -> None:
    url = route.request.url
    if not url.startswith(GAME_ORIGIN + "/"):
        route.abort()  # e.g. the Google Fonts link in index.html
        return
    path = (GAME_DIR / url[len(GAME_ORIGIN) + 1 :].split("?")[0]).resolve()
    if GAME_DIR.resolve() in path.parents and path.is_file():
        route.fulfill(path=path)
    else:
        route.fulfill(status=404)


class ChromeGame:
    """One Chrome instance running the dino game.

    Actions: 0 = nothing, 1 = jump, 2 = duck (held until another action).
    """

    def __init__(self, headless: bool = True, frame_size: int = 84):
        self.frame_size = frame_size
        self._ducking = False
        self._playwright = sync_playwright().start()
        # channel="chrome" uses the installed Google Chrome; no browser download
        self._browser = self._playwright.chromium.launch(
            channel="chrome", headless=headless
        )
        self._page = self._browser.new_page(viewport={"width": 800, "height": 400})
        self._page.route("**/*", _serve_game_file)
        self._page.goto(f"{GAME_ORIGIN}/index.html")
        self._page.wait_for_function("!!(window.Runner && Runner.instance_)")
        self._started = False
        self._paused = False

    def restart(self) -> None:
        """Start a new run and wait until the game is playing."""
        self._paused = False  # restart() runs the game loop again
        self.act(0)
        if not self._started:
            self._page.keyboard.press("Space")  # first run starts on a key press
            self._started = True
        else:
            self._page.evaluate(
                "() => { Runner.instance_.stop(); Runner.instance_.restart(); }"
            )
        self._page.wait_for_function(
            "Runner.instance_.playing && !Runner.instance_.crashed"
        )

    def act(self, action: int) -> None:
        if action == 2:
            if not self._ducking:
                self._page.keyboard.down("ArrowDown")
                self._ducking = True
            return
        if self._ducking:
            self._page.keyboard.up("ArrowDown")
            self._ducking = False
        if action == 1:
            self._page.keyboard.press("Space")

    def pause(self) -> None:
        """Freeze the game (e.g. while PPO updates its networks)."""
        if not self._paused:
            self._page.evaluate("() => Runner.instance_.stop()")
            self._paused = True

    def resume(self) -> None:
        """Continue a paused game from where it stopped."""
        if self._paused:
            # play() starts a new game loop, so only call it when paused
            self._page.evaluate("() => Runner.instance_.play()")
            self._paused = False

    def state(self) -> dict:
        return self._page.evaluate(_STATE_JS)

    def frame(self) -> np.ndarray:
        """Current screen as a (frame_size, frame_size, 1) uint8 grayscale image."""
        gray = self._page.evaluate(_FRAME_JS, self.frame_size)
        return np.array(gray, dtype=np.uint8).reshape(
            self.frame_size, self.frame_size, 1
        )

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()
