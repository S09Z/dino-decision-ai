"""Drive the Chrome Dinosaur game in real Chrome through Playwright"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Route, sync_playwright

GAME_DIR = Path(__file__).parent / "game"
# Served through request routing rather than file://, which would make the
# sprites cross-origin and block reading canvas pixels. Nothing hits the network.
GAME_ORIGIN = "http://dino.local"
# game="chrome" plays Chrome's own offline game instead (Chrome 153 has
# Runner.getInstance(); the vendored copy has Runner.instance_)
CHROME_DINO = "chrome://dino"
RUNNER = "(Runner.getInstance ? Runner.getInstance() : Runner.instance_)"

# Downscale the game canvas (or its `crop` rectangle [x, y, width, height]) to
# a size x size grayscale frame inside the page, so only size*size values
# cross the Playwright bridge per frame. They cross as one string, a char per
# pixel (0-255): 1.6ms per frame, where an array of numbers took 15ms.
_FRAME_JS = """([size, crop]) => {
  if (!window.__frame) {
    window.__frame = document.createElement('canvas');
    window.__frame.width = size;
    window.__frame.height = size;
  }
  const ctx = window.__frame.getContext('2d', {willReadFrequently: true});
  ctx.fillStyle = '#fff';  // game canvas is transparent; page background is white
  ctx.fillRect(0, 0, size, size);
  const canvas = RUNNER.canvas;
  const [x, y, w, h] = crop || [0, 0, canvas.width, canvas.height];
  ctx.drawImage(canvas, x, y, w, h, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const gray = new Array(size * size);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = String.fromCharCode(
      (rgba[4 * i] * 299 + rgba[4 * i + 1] * 587 + rgba[4 * i + 2] * 114) / 1000 | 0);
  }
  return gray.join('');
}""".replace("RUNNER", RUNNER)

# Obstacles not yet passed, nearest first: `d` is the gap from the T-Rex's
# front to the obstacle (negative while passing over it), `y` its top (the
# ground is at 150), `type` "cactus" or "bird". Collisions use tRex.xPos in
# both games, so distances do too.
_STATE_JS = """() => {
  const r = RUNNER;
  const t = r.tRex;
  const front = t.xPos + (t.config.WIDTH || t.config.width || 44);
  const obstacles = r.horizon.obstacles
    .filter(o => o.xPos + o.width > t.xPos)
    .map(o => ({
      type: /ptero/i.test(o.typeConfig.type) ? 'bird' : 'cactus',
      d: Math.round(o.xPos - front),
      w: o.width,
      y: o.yPos,
      h: o.typeConfig.height,
    }));
  return {
    crashed: r.crashed,
    playing: r.playing,
    distance: r.distanceRan,
    score: r.distanceMeter.getActualDistance(r.distanceRan),
    speed: r.currentSpeed,  // 6 at the start, up to MAX_SPEED 13
    jumping: t.jumping,
    ducking: t.ducking,
    obstacles: obstacles,
  };
}""".replace("RUNNER", RUNNER)


# Lockstep: the page's clock stands still and advance(n) runs n frames of
# 1000/60 ms each, as a 60fps browser would, then returns. The game moves only
# by performance.now() and requestAnimationFrame (speed grows once per frame),
# so this plays exactly like real time but as fast as Chrome can compute it,
# and the game waits for the agent (no pausing during updates). Installed
# before the page loads; start() switches over once the intro (a real-time CSS
# animation) is done.
_CLOCK_JS = """(() => {
  const realNow = performance.now.bind(performance);
  const realRequest = window.requestAnimationFrame.bind(window);
  const realCancel = window.cancelAnimationFrame.bind(window);
  let now = null;  // virtual time once started
  let due = new Map();  // callbacks for the next virtual frame
  const waiting = new Map();  // real-time requests not yet run
  let nextId = 1;
  performance.now = () => (now === null ? realNow() : now);
  window.requestAnimationFrame = (callback) => {
    if (now === null) {
      const id = realRequest((t) => { waiting.delete(id); callback(t); });
      waiting.set(id, callback);
      return id;
    }
    const id = -(nextId++);  // negative: ours
    due.set(id, callback);
    return id;
  };
  window.cancelAnimationFrame = (id) => {
    due.delete(id);  // also a real-time request moved over by start()
    if (waiting.delete(id)) realCancel(id);
  };
  window.__clock = {
    start() {
      now = realNow();
      // a frame requested in real time would otherwise run once, at any moment
      waiting.forEach((callback, id) => { realCancel(id); due.set(id, callback); });
      waiting.clear();
    },
    advance(frames) {
      for (let i = 0; i < frames; i++) {
        now += 1000 / 60;
        const callbacks = due;
        due = new Map();
        callbacks.forEach((callback) => callback(now));
      }
    },
  };
})();"""


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
    `crop` ([x, y, width, height] of the 600x150 canvas) keeps only that part
    of the screen in frames. `game="chrome"` plays Chrome's own chrome://dino
    instead of the vendored copy (for the Laya player, which reads state()
    and not frames). `lockstep=True` (vendored game only) stops the game's
    clock: advance(frames) moves it on, so the game runs only between calls.
    """

    def __init__(
        self,
        headless: bool = True,
        frame_size: int = 84,
        crop: Optional[Sequence[int]] = None,
        game: str = "vendored",
        lockstep: bool = False,
    ):
        if game not in ("vendored", "chrome"):
            raise ValueError(f"game must be 'vendored' or 'chrome', not {game!r}")
        if lockstep and game != "vendored":
            raise ValueError("lockstep needs the vendored game")
        self.lockstep = lockstep
        self._clock_started = False
        self.frame_size = frame_size
        self.crop = list(crop) if crop else None
        self._ducking = False
        self._playwright = sync_playwright().start()
        # channel="chrome" uses the installed Google Chrome; no browser download
        self._browser = self._playwright.chromium.launch(
            channel="chrome", headless=headless
        )
        self._page = self._browser.new_page(viewport={"width": 800, "height": 400})
        if game == "chrome":
            try:
                self._page.goto(CHROME_DINO)
            except PlaywrightError:
                pass  # Chrome reports the offline page as a failed load
            # its Runner instance appears on the first key press (restart())
            self._page.wait_for_function("typeof Runner !== 'undefined'")
        else:
            self._page.route("**/*", _serve_game_file)
            if lockstep:
                self._page.add_init_script(_CLOCK_JS)
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
                f"() => {{ const r = {RUNNER}; r.stop(); r.restart(); }}"
            )
        self._page.wait_for_function(
            f"(() => {{ const r = {RUNNER}; return !!r && r.playing && !r.crashed; }})()"
        )
        if self.lockstep and not self._clock_started:
            # the first run's intro is a real-time CSS animation
            self._page.wait_for_function("Runner.instance_.activated")
            self._page.evaluate("() => __clock.start()")
            self._clock_started = True

    def advance(self, frames: int) -> None:
        """Lockstep: run the game `frames` frames (1/60s each) and stop."""
        self._page.evaluate("(n) => __clock.advance(n)", frames)

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
            self._page.evaluate(f"() => {RUNNER}.stop()")
            self._paused = True

    def resume(self) -> None:
        """Continue a paused game from where it stopped."""
        if self._paused:
            # play() starts a new game loop, so only call it when paused
            self._page.evaluate(f"() => {RUNNER}.play()")
            self._paused = False

    def state(self) -> dict:
        return self._page.evaluate(_STATE_JS)

    def frame(self) -> np.ndarray:
        """Current screen as a (frame_size, frame_size, 1) uint8 grayscale image."""
        gray = self._page.evaluate(_FRAME_JS, [self.frame_size, self.crop])
        return np.frombuffer(gray.encode("latin-1"), dtype=np.uint8).reshape(
            self.frame_size, self.frame_size, 1
        )

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()
