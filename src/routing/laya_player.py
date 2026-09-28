"""Laya as the player: at every decision it reads the game state as a sentence
and answers two yes/no (noul) questions, jump and duck; the code presses the
key. No training, no frames: the state comes from the game's JavaScript.

From "Laya plays Dino": two yes/no questions beat one three-way choice, and a
sentence that states the relation ("88 pixels, which is less than the
150-pixel window") beats raw numbers. The code guards Laya's answer (no second
jump while in the air) and logs both: `laya` is what Laya answered, `exec`
what was done (e.g. `skip:air`).

RulePlayer answers the same two questions from the numbers directly. It is
the ground truth for the scenario benchmark and checks the jump window
without the model.

Usage: dino-ai play --player laya|rule [--game chrome] [--show]
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Optional

from src.routing.laya_classifier import MODEL

ACTIONS = {"hold": 0, "jump": 1, "duck": 2}
# A detector at or above the cut answers yes. 0.7, not 0.5: in 5 live games
# Laya's jump detector reached 0.5-0.69 on 8% of "more than the window"
# sentences (early jumps onto cacti) but 0.7+ on 96% of "less than" ones
# (docs/LAYA_PLAYER.md)
CUT = 0.7
# Frames of warning to an obstacle's middle: the window is speed (pixels per
# frame) x LEAD, less half the obstacle's width, so wide cactus groups are
# jumped later
LEAD = 14
# Tops of the T-Rex on the 150px canvas, running and ducking (bottom 140)
TREX_TOP, TREX_DUCK_TOP = 93, 115
LOG_PATH = Path("logs/laya_play.jsonl")

QUESTIONS = {
    "jump": {
        "type": "noul",
        "instructions": (
            "The dinosaur must jump when a cactus or a low bird is inside the"
            " window, that is, when its distance is less than the window. Is a"
            " cactus or a low bird inside the window?"
        ),
        "criteria": {
            "true": "A cactus or a low bird is closer than the window: jump now",
            "false": "Nothing to jump over is inside the window",
        },
    },
    "duck": {
        "type": "noul",
        "instructions": (
            "The dinosaur must duck when a bird at head height is inside the"
            " window, that is, when its distance is less than the window. Is a"
            " bird at head height inside the window?"
        ),
        "criteria": {
            "true": "A bird at head height is closer than the window: duck now",
            "false": "No bird at head height is inside the window",
        },
    },
}


def kind(obstacle: dict) -> Optional[str]:
    """What the dinosaur must do about an obstacle, by its name in sentences:
    None for a bird that flies over a running dinosaur"""
    if obstacle["type"] == "cactus":
        return "cactus"
    bottom = obstacle["y"] + obstacle["h"]
    if bottom <= TREX_TOP:
        return None
    return "bird at head height" if bottom <= TREX_DUCK_TOP else "low bird"


def threat(state: dict) -> Optional[dict]:
    """The nearest obstacle the dinosaur must jump or duck, with its `kind`"""
    for obstacle in state["obstacles"]:
        name = kind(obstacle)
        if name:
            return {**obstacle, "kind": name}
    return None


def window(speed: float, width: float = 0, lead: float = LEAD) -> int:
    return round(speed * lead - width / 2)


def describe(state: dict, lead: float = LEAD) -> str:
    """The state as Laya reads it: the numbers and how they compare"""
    where = "in the air" if state["jumping"] else "on the ground"
    text = f"The dinosaur is {where}, running at speed {state['speed']:.1f}."
    near = threat(state)
    if near is None:
        return f"{text} Nothing to jump over or duck under is ahead."
    size = window(state["speed"], near["w"], lead)
    relation = "less" if near["d"] < size else "more"
    return (
        f"{text} Nearest obstacle: a {near['kind']}. Distance to the"
        f" {near['kind']}: {near['d']} pixels, which is {relation} than the"
        f" {size}-pixel window."
    )


def truth(state: dict, lead: float = LEAD) -> tuple[bool, bool]:
    """The right answers to the two questions: (jump, duck)"""
    near = threat(state)
    if near is None or near["d"] >= window(state["speed"], near["w"], lead):
        return False, False
    duck = near["kind"] == "bird at head height"
    return not duck, duck


@dataclass
class Decision:
    text: str
    det: tuple[float, float]  # P(yes) for jump and duck
    laya: str  # hold, jump or duck: the answer
    exec: str  # what was done: the answer, or skip:air
    action: int  # ChromeGame action for `exec`
    ms: float  # time to decide


def answer(det: tuple[float, float], cut: float = CUT) -> str:
    jump, duck = det
    if max(jump, duck) < cut:
        return "hold"
    return "jump" if jump >= duck else "duck"


class RulePlayer:
    """Answers the questions from the numbers (`truth`). `delay_ms` makes it
    take as long as Laya does, so a window tuned with it holds for Laya: the
    game keeps moving while the player decides."""

    name = "rule"

    def __init__(self, lead: float = LEAD, cut: float = CUT, delay_ms: float = 0):
        self.lead, self.cut, self.delay_ms = lead, cut, delay_ms

    def detect(self, state: dict, text: str) -> tuple[float, float]:
        time.sleep(self.delay_ms / 1000)
        jump, duck = truth(state, self.lead)
        return float(jump), float(duck)

    def decide(self, state: dict) -> Decision:
        start = time.perf_counter()
        text = describe(state, self.lead)
        det = self.detect(state, text)
        said = answer(det, self.cut)
        # a second press in the air does nothing, and down in the air cuts the
        # jump short (speed drop)
        done = "skip:air" if said != "hold" and state["jumping"] else said
        ms = (time.perf_counter() - start) * 1000
        return Decision(text, det, said, done, ACTIONS.get(done, 0), ms)


class LayaPlayer(RulePlayer):
    """Laya answers the questions from the sentence. The model loads on first
    use, or up front with load(); pass `agent` to use an already loaded one."""

    name = "laya"

    def __init__(
        self,
        lead: float = LEAD,
        cut: float = CUT,
        model: str = MODEL,
        device: Optional[str] = None,
        agent: Any = None,
    ):
        super().__init__(lead, cut)
        self.model, self.device, self.agent = model, device, agent

    def load(self) -> Any:
        if self.agent is None:
            import laya  # type: ignore[import-untyped]

            self.agent = laya.load(self.model, device=self.device)
        return self.agent

    def detect(self, state: dict, text: str) -> tuple[float, float]:
        answers = self.load().system_one(text, QUESTIONS)["answers"]
        return float(answers["jump"]["noul"]), float(answers["duck"]["noul"])


def record(state: dict, decision: Decision, seconds: float) -> dict:
    """One JSONL line per decision (the fields of the video's log)"""
    near = threat(state)
    return {
        "t": round(seconds, 2),
        "spd": round(state["speed"], 2),
        "obs": near["kind"] if near else None,
        "d": near["d"] if near else 999,
        "ow": near["w"] if near else 0,
        "oy": near["y"] if near else 0,
        "air": int(state["jumping"]),
        "laya": decision.laya,
        "exec": decision.exec,
        "det": [round(p, 3) for p in decision.det],
        "ms": round(decision.ms, 1),
        "score": state["score"],
        "text": decision.text,
        "ahead": state["obstacles"],
    }


def play(
    game: Any,
    player: RulePlayer,
    episodes: int,
    log: IO[str],
    echo: Callable[[str], Any] = print,
    db: Any = None,
) -> list[dict]:
    """Play `episodes` games without pausing (the game runs while the player
    decides); writes each decision to `log`, and to `db` (a MetricsDB) for the
    dashboard; returns one summary per game"""
    results: list[dict] = []
    for _ in range(episodes):
        game.restart()
        start = time.perf_counter()
        times: list[float] = []
        while True:
            state = game.state()
            if state["crashed"]:
                break
            decision = player.decide(state)
            game.act(decision.action)
            times.append(decision.ms)
            line = record(state, decision, time.perf_counter() - start)
            log.write(json.dumps(line) + "\n")
            if db is not None:
                db.add_decision(
                    player.name,
                    len(results) + 1,
                    line["t"],
                    decision.laya,
                    decision.exec,
                    {"jump": decision.det[0], "duck": decision.det[1]},
                    "probability",
                    cut=player.cut,
                    input=decision.text,
                    speed=state["speed"],
                    obstacle=line["obs"],
                    distance=line["d"],
                    ms=decision.ms,
                )
        results.append(
            {
                "score": state["score"],
                "decisions": len(times),
                "mean_ms": sum(times) / len(times) if times else 0.0,
            }
        )
        echo(
            f"episode {len(results)}: score {state['score']:.0f},"
            f" {len(times)} decisions, {results[-1]['mean_ms']:.0f}ms each"
        )
    return results
