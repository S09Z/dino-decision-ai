"""Measure Laya as the player on game states with known answers, for three
ways of asking: one three-way choice on raw numbers, two yes/no (noul)
questions on raw numbers, and two yes/no questions on a sentence that states
how the numbers compare (what LayaPlayer asks). Downloads the model (~843MB)
on first run.

Usage: python -m src.routing.laya_player_benchmark --out docs/LAYA_PLAYER.md
"""

import platform
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import typer

from src.routing.laya_classifier import MODEL
from src.routing.laya_player import (
    LEAD,
    QUESTIONS,
    LayaPlayer,
    answer,
    describe,
    threat,
    truth,
    window,
)


def _cactus(d: int, w: int = 17) -> dict:
    return {"type": "cactus", "d": d, "w": w, "y": 105, "h": 35}


def _bird(d: int, y: int) -> dict:
    return {"type": "bird", "d": d, "w": 46, "y": y, "h": 40}


def _state(*obstacles: dict, speed: float = 8.0, jumping: bool = False) -> dict:
    return {
        "speed": speed,
        "jumping": jumping,
        "ducking": False,
        "obstacles": list(obstacles),
    }


# (name, state): the answer comes from `truth`, at the default LEAD
SCENARIOS = [
    ("nothing ahead", _state()),
    ("cactus far", _state(_cactus(300))),
    ("cactus inside", _state(_cactus(40))),
    ("cactus just inside", _state(_cactus(95))),
    ("cactus just outside", _state(_cactus(115))),
    ("cactus inside, fast", _state(_cactus(150), speed=12.0)),
    ("cactus outside, slow", _state(_cactus(100), speed=6.0)),
    ("cactus group inside", _state(_cactus(60, w=75))),
    ("cactus group outside", _state(_cactus(90, w=75))),
    ("low bird inside", _state(_bird(60, y=100), speed=10.0)),
    ("low bird far", _state(_bird(250, y=100), speed=10.0)),
    ("head-height bird inside", _state(_bird(60, y=75), speed=10.0)),
    ("head-height bird far", _state(_bird(250, y=75), speed=10.0)),
    ("high bird, cactus far", _state(_bird(40, y=50), _cactus(260), speed=10.0)),
    ("in the air, cactus inside", _state(_cactus(-20), jumping=True)),
    ("passing over, in the air", _state(_cactus(-60, w=50), jumping=True)),
]

CHOICE = {
    "action": {
        "type": "choice",
        "instructions": (
            "Pick the dinosaur's action now. Jump when a cactus or a low bird is"
            " closer than the window, duck when a bird at head height is closer"
            " than the window, otherwise keep running."
        ),
        "criteria": {
            "jump": "Jump over a cactus or low bird closer than the window",
            "duck": "Duck under a bird at head height closer than the window",
            "hold": "Keep running: nothing is closer than the window",
        },
    }
}


def raw(state: dict, lead: float = LEAD) -> dict:
    """The same facts as `describe`, as raw numbers"""
    near = threat(state)
    return {
        "dinosaur": "in the air" if state["jumping"] else "on the ground",
        "speed": round(state["speed"], 1),
        "obstacle": near["kind"] if near else "none",
        "distance_px": near["d"] if near else None,
        "window_px": window(state["speed"], near["w"], lead) if near else None,
    }


def _noul(agent: Any, text: Any) -> str:
    answers = agent.system_one(text, QUESTIONS)["answers"]
    return answer((answers["jump"]["noul"], answers["duck"]["noul"]))


def _choice(agent: Any, text: Any) -> str:
    return agent.system_one(text, CHOICE)["answers"]["action"]["choice"]


# name -> (how the state is written, how Laya is asked)
STYLES: dict[str, tuple[Callable[[dict], Any], Callable[[Any, Any], str]]] = {
    "choice, raw numbers": (raw, _choice),
    "yes/no, raw numbers": (raw, _noul),
    "yes/no, sentence": (describe, _noul),
}


def expected(state: dict) -> str:
    jump, duck = truth(state)
    return answer((float(jump), float(duck)))


def benchmark(agent: Any, repeats: int = 3) -> dict:
    """Every style on every scenario; returns per style the correct count and
    ms per decision, and the answers per scenario"""
    results = {}
    for style, (write, ask) in STYLES.items():
        answers, seconds = [], []
        for _, state in SCENARIOS:
            text = write(state)
            for _ in range(repeats):
                start = time.perf_counter()
                said = ask(agent, text)
                seconds.append(time.perf_counter() - start)
            answers.append(said)
        ms = np.array(seconds) * 1000
        results[style] = {
            "answers": answers,
            "correct": sum(a == expected(s) for a, (_, s) in zip(answers, SCENARIOS)),
            "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)),
        }
    return results


def report(results: dict, device: Optional[str], repeats: int) -> str:
    styles = list(results)
    lines = [
        "# Laya player benchmark",
        "",
        f"Measured {date.today()} with `python -m src.routing.laya_player_benchmark"
        f" --repeats {repeats}` ({MODEL}, device {device or 'auto'}).",
        f"Machine: {platform.platform()}. Answers are checked against"
        f" `truth` (the rule player) at lead {LEAD}.",
        "",
        "| Style | Correct | Median | p95 |",
        "|---|---|---|---|",
        *(
            f"| {s} | {r['correct']}/{len(SCENARIOS)} | {r['median_ms']:.0f}ms"
            f" | {r['p95_ms']:.0f}ms |"
            for s, r in results.items()
        ),
        "",
        "## Answers",
        "",
        "| Scenario | Expected | " + " | ".join(styles) + " |",
        "|---|---|" + "---|" * len(styles),
    ]
    for i, (name, state) in enumerate(SCENARIOS):
        want = expected(state)
        cells = [
            results[s]["answers"][i]
            + ("" if results[s]["answers"][i] == want else " (wrong)")
            for s in styles
        ]
        lines.append(f"| {name} | {want} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Sentence for the first cactus inside:",
        "",
        f"> {describe(SCENARIOS[2][1])}",
        "",
    ]
    return "\n".join(lines)


def main(
    out: Optional[Path] = None, device: Optional[str] = None, repeats: int = 3
) -> None:
    player = LayaPlayer(device=device)
    text = report(benchmark(player.load(), repeats), device, repeats)
    if out is not None:
        out.write_text(text, encoding="utf-8")
    typer.echo(text)


if __name__ == "__main__":
    typer.run(main)
