"""Training experiments: agent variants that may make the game easier to learn.

Each variant trains like `dino-ai train` but under its own name, so its
episodes and checkpoints stay apart and `dino-ai report` compares it with the
baseline agents. Afterwards its best and latest checkpoints are evaluated
greedily and the result is appended to models/logs/experiments.jsonl.

Usage:
    python -m src.training.experiment dqn-a2 --steps 100000 [--resume] [--n-envs 4]
    python -m src.training.experiment dqn-a2 --eval-only
    python -m src.training.experiment dqn-a2 --watch [--checkpoint latest|eval]
    python -m src.training.experiment dqn-a2-ls4 --eval-every 50000  # lockstep
    python -m src.training.experiment dqn-a2@2 --steps 100000  # a repeat run
"""

import json
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import typer
from playwright.sync_api import Error as PlaywrightError
from stable_baselines3.common.callbacks import CallbackList

from src.evaluation.evaluate import evaluate_games, projected_score
from src.models import AGENTS
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from src.routing.laya_player import ACTIONS, describe, threat
from src.training.callbacks import EvalMonitor, TrainingMonitor
from src.training.envs import make_dino_env

RESULTS = Path("models") / "logs" / "experiments.jsonl"  # git-ignored

# Left half of the 600x150 canvas: the dino and ~250px ahead of it, without
# the score; obstacles come out about twice as wide in the 84x84 frame
CROP_AHEAD = (0, 0, 300, 150)


@dataclass
class Variant:
    agent: str  # key of AGENTS
    env: dict[str, Any] = field(default_factory=dict)  # ChromeDinoEnv options
    config: dict[str, Any] = field(default_factory=dict)  # config overrides
    # ChromeDinoEnv options for training only: evaluation and --watch play
    # the normal game, so variants compare on the same one
    train: dict[str, Any] = field(default_factory=dict)


VARIANTS = {
    # no duck: pressing down mid-jump cuts the jump short, and birds only
    # appear from speed 8.5, which the agents do not reach yet
    "dqn-a2": Variant("dqn", env={"n_actions": 2}),
    # ...and 1% instead of 10% random actions once exploration is over
    "dqn-a2-eps": Variant(
        "dqn", env={"n_actions": 2}, config={"exploration_final_eps": 0.01}
    ),
    # ...and frames of the ground ahead only
    "dqn-a2-eps-crop": Variant(
        "dqn",
        env={"n_actions": 2, "crop": CROP_AHEAD},
        config={"exploration_final_eps": 0.01},
    ),
    "ppo-a2-crop": Variant("ppo", env={"n_actions": 2, "crop": CROP_AHEAD}),
}
# Lockstep copies (ChromeDinoEnv frames_per_step): the game waits for the
# agent, so a step costs compute time, not 50ms of real time (~13x faster on
# one game). Timing differs from real time, so checkpoints do not carry over
# between the two: these train, evaluate and watch in lockstep only.
LOCKSTEP_FRAMES = 4
VARIANTS.update(
    {
        f"{name}-ls{LOCKSTEP_FRAMES}": replace(
            variant, env={**variant.env, "frames_per_step": LOCKSTEP_FRAMES}
        )
        for name, variant in list(VARIANTS.items())
    }
)
# ...and every training game starts at a random speed (6-13) instead of 6:
# agents died around speed 9, just after birds appear (8.5), so they had
# hardly practised on birds (half their deaths: jumping into a high bird)
VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}-speed"] = replace(
    VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}"], train={"start_speed": (6, 13)}
)
# ...but only half of them: round 3's -speed agent lasted 1,700-24,000 steps
# in 9 of 30 games yet died at speed 6-7 in the rest, having practised the
# slow start about 7 times less than a normal agent
VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}-mix"] = replace(
    VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}"],
    train={"start_speed": (6, 13), "start_speed_share": 0.5},
)
# ...and every key press costs 0.05 (half a step's reward): -mix@2 jumped in
# 26% of steps with nothing ahead, its values for jump and nothing 0.2 apart
VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}-mix-cost"] = replace(
    VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}-mix"],
    train={
        **VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}-mix"].train,
        "press_cost": 0.05,
    },
)
# ...with duck back: pressing down in the air drops the dino fast, so at
# speed ~12 it can land before a high bird (-mix's main cause of death)
VARIANTS[f"dqn-a3-eps-crop-ls{LOCKSTEP_FRAMES}-mix"] = replace(
    VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}-mix"],
    env={**VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}-mix"].env, "n_actions": 3},
)
# ...and a key press cost too: without it, the three values were ~0.1 apart
# far from obstacles and the greedy agent jumped at almost every step
VARIANTS[f"dqn-a3-eps-crop-ls{LOCKSTEP_FRAMES}-mix-cost"] = replace(
    VARIANTS[f"dqn-a3-eps-crop-ls{LOCKSTEP_FRAMES}-mix"],
    train=VARIANTS[f"dqn-a2-eps-crop-ls{LOCKSTEP_FRAMES}-mix-cost"].train,
)


def variant_of(name: str) -> Variant:
    """A variant by name; "<variant>@<tag>" (e.g. dqn-a2@2) repeats it under
    that name, with its own checkpoints and results"""
    return VARIANTS[name.split("@")[0]]


def run(
    name: str,
    steps: int,
    checkpoint_every: int = 5_000,
    resume: bool = False,
    episodes: int = 20,
    eval_only: bool = False,
    n_envs: int = 1,
    eval_every: int = 0,
    max_length: Optional[int] = None,
) -> dict:
    """Train variant `name` up to `steps` (unless eval_only) on `n_envs` games
    at once, then evaluate its best and latest checkpoints on one game
    (games stopped at `max_length` steps; projected_score estimates the
    rest); returns and records the result. `eval_every` > 0 (lockstep
    variants) also evaluates every that many steps while training
    (EvalMonitor); the checkpoint those picked ("eval_best") is then
    evaluated instead of the best by training reward, which misled in
    every round that had both (e.g. 526 steps vs 10,095)."""
    variant = variant_of(name)
    if eval_every and "frames_per_step" not in variant.env:
        raise ValueError("eval_every needs a lockstep variant")
    agent_class = AGENTS[variant.agent]
    config = replace(agent_class.default_config(), **variant.config)
    checkpoints = CheckpointManager()
    history: list = []
    if not eval_only:
        env = make_dino_env(n_envs=n_envs, **variant.env, **variant.train)
        eval_env = make_dino_env(**variant.env) if eval_every else None
        try:
            model = agent_class(env, config)
            latest = checkpoints.latest(name) if resume else None
            if latest:
                model.resume(latest)
            remaining = steps - model.model.num_timesteps
            if remaining > 0:
                with MetricsDB() as db:
                    callbacks: list = [
                        TrainingMonitor(
                            name,
                            db,
                            checkpoints,
                            checkpoint_every,
                            episodes=latest.episodes if latest else 0,
                        )
                    ]
                    if eval_env is not None:
                        callbacks.append(
                            EvalMonitor(name, eval_env, checkpoints, eval_every)
                        )
                    model.train(remaining, callback=CallbackList(callbacks))
                    if eval_env is not None:
                        history = callbacks[1].history
        finally:
            env.close()
            if eval_env is not None:
                eval_env.close()
    evaluated = {}
    eval_best = checkpoints.best(f"{name}-eval")
    env = make_dino_env(**variant.env)  # evaluation: one game
    try:
        for kind, checkpoint in (
            ("best", None if eval_best else checkpoints.best(name)),
            ("latest", checkpoints.latest(name)),
            ("eval_best", eval_best),
        ):
            if checkpoint is None:
                continue
            model = agent_class(env, config)
            model.load(checkpoint.path)
            games = evaluate_games(model.model, env, episodes, max_length)
            lengths = [game["length"] for game in games]
            evaluated[kind] = {
                "step": checkpoint.step,
                "mean_length": float(np.mean(lengths)),
                "std_length": float(np.std(lengths)),
                # the mean hides games that split into short and very long
                "median_length": float(np.median(lengths)),
                "over_1000": sum(length > 1000 for length in lengths),
                "max_length": int(np.max(lengths)),
                "mean_reward": float(np.mean([game["reward"] for game in games])),
                "mean_score": float(np.mean([game["score"] for game in games])),
                # without the max_length cap (None: no death at top speed yet)
                "projected_score": projected_score(games),
                # what ended each game: {"cactus": 12, "bird at head height": 8}
                "hits": dict(Counter(game["hit"] for game in games)),
                "died_in_air": sum(game["in_air"] for game in games),
                "death_speed": float(np.mean([game["speed"] for game in games])),
                "games": games,
            }
    finally:
        env.close()
    result = {
        "variant": name,
        "agent": variant.agent,
        "env": variant.env,
        "config": variant.config,
        "train": variant.train,
        "episodes": episodes,
        "evaluated": evaluated,
        "eval_history": history,  # [(step, mean, median)] while training
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result) + "\n")
    return result


def action_scores(model: Any, obs: Any) -> tuple[dict[str, float], str]:
    """What the policy thinks of each action for `obs`: DQN's Q-values, or
    PPO's action probabilities"""
    policy = model.policy
    tensor, _ = policy.obs_to_tensor(obs)
    with torch.no_grad():
        if hasattr(policy, "q_net"):
            values, kind = policy.q_net(tensor)[0], "q_value"
        else:
            values = policy.get_distribution(tensor).distribution.probs[0]
            kind = "probability"
    return {name: float(v) for name, v in zip(ACTIONS, values)}, kind


def watch(
    name: str,
    checkpoint: str = "best",
    episodes: int = 5,
    echo: Callable[[str], Any] = print,
) -> list[int]:
    """Play `episodes` games greedily with variant `name`'s best or latest
    checkpoint in a visible Chrome. Each game goes to the metrics DB as
    `<name>-watch`, and so does every decision (the value of each action and
    the one taken), so the dashboard shows it live. Returns their lengths."""
    variant = variant_of(name)
    agent_class = AGENTS[variant.agent]
    config = replace(agent_class.default_config(), **variant.config)
    checkpoints = CheckpointManager()
    saved: Any
    if checkpoint == "latest":
        saved = checkpoints.latest(name)
    elif checkpoint == "eval":
        saved = checkpoints.best(f"{name}-eval")  # picked by EvalMonitor
    else:
        saved = checkpoints.best(name)
    if saved is None:
        raise FileNotFoundError(f"No {checkpoint} checkpoint for {name}")
    env = make_dino_env(render_mode="human", **variant.env)
    lengths: list[int] = []
    try:
        model = agent_class(env, config)
        model.load(saved.path)
        echo(f"{name} {checkpoint} (step {saved.step}) is playing")
        frames = variant.env.get("frames_per_step")
        # lockstep runs as fast as Chrome can: slow it to real time to watch
        step_seconds = frames / 60 if frames else 0.0
        with MetricsDB() as db:
            obs = env.reset()
            state, start = None, time.perf_counter()
            while len(lengths) < episodes:
                decided = time.perf_counter()
                action, _ = model.predict(obs)
                scores, kind = action_scores(model.model, obs)
                ms = (time.perf_counter() - decided) * 1000
                if state is not None:  # the game state these frames show
                    answer = list(ACTIONS)[int(action[0])]
                    near = threat(state)
                    db.add_decision(
                        f"{name}-watch",
                        len(lengths) + 1,
                        round(time.perf_counter() - start, 2),
                        answer,
                        answer,
                        scores,
                        kind,
                        input=describe(state),
                        speed=state["speed"],
                        obstacle=near["kind"] if near else None,
                        distance=near["d"] if near else 999,
                        ms=ms,
                    )
                time.sleep(max(0.0, step_seconds - (time.perf_counter() - decided)))
                try:
                    obs, _, dones, infos = env.step(action)
                except PlaywrightError:  # the watcher closed the window
                    echo("Chrome was closed: stopped")
                    break
                state = infos[0]
                if dones[0]:
                    state, start = None, time.perf_counter()
                    episode = infos[0]["episode"]  # added by Monitor
                    lengths.append(int(episode["l"]))
                    db.add_episode(
                        f"{name}-watch",
                        len(lengths),
                        float(episode["r"]),
                        int(episode["l"]),
                    )
                    echo(
                        f"game {len(lengths)}: {lengths[-1]} steps,"
                        f" score {infos[0]['score']:.0f}"
                    )
    finally:
        env.close()
    return lengths


def main(
    name: str,
    steps: int = 100_000,
    checkpoint_every: int = 5_000,
    resume: bool = False,
    episodes: int = 20,
    eval_only: bool = False,
    n_envs: int = typer.Option(1, help="Games at once while training"),
    watch_play: bool = typer.Option(
        False, "--watch", help="Play --episodes games in a visible Chrome"
    ),
    checkpoint: str = typer.Option("best", help="--watch: best, latest or eval"),
    eval_every: int = typer.Option(
        0, help="Lockstep: also evaluate every N steps while training"
    ),
    max_length: int = typer.Option(
        0, help="Stop evaluation games after N steps (0: never)"
    ),
) -> None:
    if name.split("@")[0] not in VARIANTS:
        typer.echo(f"Unknown variant {name!r}; available: {', '.join(VARIANTS)}")
        raise typer.Exit(1)
    if watch_play:
        try:
            lengths = watch(name, checkpoint, episodes, echo=typer.echo)
        except FileNotFoundError as error:
            typer.echo(f"{error}; train it first", err=True)
            raise typer.Exit(1)
        if lengths:  # none if the window was closed during the first game
            typer.echo(f"mean {np.mean(lengths):.0f} steps over {len(lengths)} games")
        return
    result = run(
        name,
        steps,
        checkpoint_every,
        resume,
        episodes,
        eval_only,
        n_envs,
        eval_every,
        max_length or None,
    )
    for step, mean, median in result["eval_history"]:
        typer.echo(f"while training, step {step}: mean {mean:.0f}, median {median:.0f}")
    for kind, row in result["evaluated"].items():
        typer.echo(
            f"{name} {kind} (step {row['step']}): length"
            f" {row['mean_length']:.0f} ± {row['std_length']:.0f} steps"
            f" (median {row['median_length']:.0f}, max {row['max_length']},"
            f" {row['over_1000']} over 1000), reward {row['mean_reward']:.1f},"
            f" score {row['mean_score']:.0f}"
        )
        projected = row["projected_score"]
        typer.echo(
            "  projected score without a length cap: "
            + (f"{projected:.0f}" if projected else "no death at top speed yet")
        )
        hits = ", ".join(f"{name} {n}" for name, n in row["hits"].items())
        typer.echo(
            f"  died on: {hits}; {row['died_in_air']} in the air;"
            f" mean speed {row['death_speed']:.1f}"
        )


if __name__ == "__main__":
    typer.run(main)
