"""Training experiments: agent variants that may make the game easier to learn.

Each variant trains like `dino-ai train` but under its own name, so its
episodes and checkpoints stay apart and `dino-ai report` compares it with the
baseline agents. Afterwards its best and latest checkpoints are evaluated
greedily and the result is appended to models/logs/experiments.jsonl.

Usage:
    python -m src.training.experiment dqn-a2 --steps 100000 [--resume] [--n-envs 4]
    python -m src.training.experiment dqn-a2 --eval-only
    python -m src.training.experiment dqn-a2 --watch [--checkpoint latest]
"""

import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import typer

from src.evaluation.evaluate import evaluate
from src.models import AGENTS
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from src.routing.laya_player import ACTIONS, describe, threat
from src.training.callbacks import TrainingMonitor
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


def run(
    name: str,
    steps: int,
    checkpoint_every: int = 5_000,
    resume: bool = False,
    episodes: int = 20,
    eval_only: bool = False,
    n_envs: int = 1,
) -> dict:
    """Train variant `name` up to `steps` (unless eval_only) on `n_envs` games
    at once, then evaluate its best and latest checkpoints on one game;
    returns and records the result"""
    variant = VARIANTS[name]
    agent_class = AGENTS[variant.agent]
    config = replace(agent_class.default_config(), **variant.config)
    checkpoints = CheckpointManager()
    if not eval_only:
        env = make_dino_env(n_envs=n_envs, **variant.env)
        try:
            model = agent_class(env, config)
            latest = checkpoints.latest(name) if resume else None
            if latest:
                model.resume(latest)
            remaining = steps - model.model.num_timesteps
            if remaining > 0:
                with MetricsDB() as db:
                    monitor = TrainingMonitor(
                        name,
                        db,
                        checkpoints,
                        checkpoint_every,
                        episodes=latest.episodes if latest else 0,
                    )
                    model.train(remaining, callback=monitor)
        finally:
            env.close()
    evaluated = {}
    env = make_dino_env(**variant.env)  # evaluation: one game
    try:
        for kind, checkpoint in (
            ("best", checkpoints.best(name)),
            ("latest", checkpoints.latest(name)),
        ):
            if checkpoint is None:
                continue
            model = agent_class(env, config)
            model.load(checkpoint.path)
            rewards, lengths = evaluate(model.model, env, episodes)
            evaluated[kind] = {
                "step": checkpoint.step,
                "mean_length": float(np.mean(lengths)),
                "std_length": float(np.std(lengths)),
                "max_length": int(np.max(lengths)),
                "mean_reward": float(np.mean(rewards)),
            }
    finally:
        env.close()
    result = {
        "variant": name,
        "agent": variant.agent,
        "env": variant.env,
        "config": variant.config,
        "episodes": episodes,
        "evaluated": evaluated,
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
    variant = VARIANTS[name]
    agent_class = AGENTS[variant.agent]
    config = replace(agent_class.default_config(), **variant.config)
    checkpoints = CheckpointManager()
    saved = (
        checkpoints.latest(name) if checkpoint == "latest" else checkpoints.best(name)
    )
    if saved is None:
        raise FileNotFoundError(f"No {checkpoint} checkpoint for {name}")
    env = make_dino_env(render_mode="human", **variant.env)
    lengths: list[int] = []
    try:
        model = agent_class(env, config)
        model.load(saved.path)
        echo(f"{name} {checkpoint} (step {saved.step}) is playing")
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
                obs, _, dones, infos = env.step(action)
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
    checkpoint: str = typer.Option("best", help="--watch: best or latest"),
) -> None:
    if name not in VARIANTS:
        typer.echo(f"Unknown variant {name!r}; available: {', '.join(VARIANTS)}")
        raise typer.Exit(1)
    if watch_play:
        try:
            lengths = watch(name, checkpoint, episodes, echo=typer.echo)
        except FileNotFoundError as error:
            typer.echo(f"{error}; train it first", err=True)
            raise typer.Exit(1)
        typer.echo(f"mean {np.mean(lengths):.0f} steps over {len(lengths)} games")
        return
    result = run(name, steps, checkpoint_every, resume, episodes, eval_only, n_envs)
    for kind, row in result["evaluated"].items():
        typer.echo(
            f"{name} {kind} (step {row['step']}): length"
            f" {row['mean_length']:.0f} ± {row['std_length']:.0f} steps"
            f" (max {row['max_length']}), reward {row['mean_reward']:.1f}"
        )


if __name__ == "__main__":
    typer.run(main)
