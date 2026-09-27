"""CLI Entry Point"""

from pathlib import Path
from typing import Optional

import psutil
import typer

from src.evaluation.evaluate import RandomPolicy, evaluate, report
from src.models import AGENTS
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.models_mgmt.model_registry import ModelRegistry
from src.monitoring.local_db import MetricsDB
from src.profiling.baseline import instrument
from src.profiling.profiler import Profiler
from src.training.callbacks import TrainingMonitor
from src.training.envs import make_dino_env
from src.training.parallel import ram_warning, train_parallel

app = typer.Typer()
MB = 1024**2
GB = 1024**3


def _check_agent(agent: str) -> str:
    if agent not in AGENTS:
        typer.echo(f"Unknown agent {agent!r}; available: {', '.join(AGENTS)}", err=True)
        raise typer.Exit(1)
    return agent


@app.command()
def train(
    agent: str = "dqn",
    all_agents: bool = typer.Option(False, "--all", help="Train every agent in turn"),
    steps: int = 20_000,
    checkpoint_every: int = 5_000,
    profile: bool = typer.Option(False, help="Print per-call timings at the end"),
    parallel: bool = typer.Option(
        False, help="With --all: train agents at once, each with its own Chrome"
    ),
    progress_bar: bool = typer.Option(True, "--progress-bar/--no-progress-bar"),
):
    """Train agents, recording episodes to the metrics DB and keeping the
    best checkpoints (real time: ~12 steps/s per agent)"""
    names = list(AGENTS) if all_agents else [_check_agent(agent)]
    if parallel and len(names) > 1:
        if profile:
            typer.echo("--profile does not work with --parallel", err=True)
            raise typer.Exit(1)
        warning = ram_warning(names, psutil.virtual_memory().available / GB)
        if warning:
            typer.echo(warning, err=True)
        codes = train_parallel(names, steps, checkpoint_every)
        for name, code in codes.items():
            typer.echo(f"{name}: " + ("done" if code == 0 else f"failed (exit {code})"))
        if any(codes.values()):
            raise typer.Exit(1)
        return
    env = make_dino_env()
    profiler = Profiler()
    try:
        if profile:
            instrument(env.get_attr("unwrapped")[0], profiler)
        with MetricsDB() as db:
            for name in names:
                monitor = TrainingMonitor(
                    name, db, CheckpointManager(), checkpoint_every
                )
                model = AGENTS[name](env)
                model.train(steps, callback=monitor, progress_bar=progress_bar)
                # actual steps: SB3 rounds up (DQN to train_freq, PPO to n_steps)
                typer.echo(
                    f"Trained {name} for {model.model.num_timesteps} steps,"
                    f" {monitor.episodes} episodes"
                )
    finally:
        env.close()
    if profile:
        typer.echo(profiler.report())


@app.command()
def eval(
    agent: str = "dqn",
    episodes: int = 20,
    compare: bool = typer.Option(
        False, help="Evaluate random and every agent's best checkpoint"
    ),
):
    """Evaluate the agent's best checkpoint"""
    names = list(AGENTS) if compare else [_check_agent(agent)]
    env = make_dino_env()
    results = []
    try:
        if compare:
            results.append(
                report("random", *evaluate(RandomPolicy(env), env, episodes))
            )
        for name in names:
            model = AGENTS[name](env)
            try:
                best = CheckpointManager().load_best(model, name)
            except FileNotFoundError as error:
                if not compare:
                    typer.echo(f"{error}; run `train` first", err=True)
                    raise typer.Exit(1)
                results.append(f"{name}: no checkpoint (run `train --agent {name}`)")
                continue
            typer.echo(
                f"Loaded {best.path} (step {best.step}, reward {best.reward:.1f})"
            )
            results.append(report(name, *evaluate(model.model, env, episodes)))
    finally:
        env.close()
    typer.echo("\n".join(results))


@app.command()
def inspect(agent: Optional[str] = None):
    """Show kept checkpoints, released versions and recorded episodes"""
    names = [_check_agent(agent)] if agent else list(AGENTS)
    checkpoints, registry = CheckpointManager(), ModelRegistry()
    with MetricsDB() as db:
        for name in names:
            typer.echo(f"{name}: {db.episode_count(name)} episodes recorded")
            kept = checkpoints.checkpoints(name)
            if not kept:
                typer.echo("  no checkpoints")
            for c in kept:
                size = Path(c.path).stat().st_size / MB
                typer.echo(
                    f"  step {c.step:>7}  reward {c.reward:7.1f}"
                    f"  {size:5.1f}MB  {c.timestamp}  {c.path}"
                )
            for v in registry.versions(name):
                typer.echo(f"  released v{v.version} ({v.timestamp}): {v.notes}")


@app.command()
def clean(keep: int = typer.Option(3, min=0, help="Checkpoints to keep per agent")):
    """Delete all but the best `keep` checkpoints of each agent"""
    manager = CheckpointManager()
    for name in AGENTS:
        removed = manager.prune(name, keep)
        typer.echo(
            f"{name}: removed {len(removed)}, kept {len(manager.checkpoints(name))}"
        )


@app.command()
def profile(steps: int = 500, out: Optional[Path] = None):
    """Profile env speed and DQN inference (see docs/PROFILING_BASELINE.md)"""
    from src.profiling import baseline

    baseline.main(steps=steps, out=out)


@app.command()
def dashboard(port: int = 8000):
    """Start dashboard"""
    typer.echo(f"Starting dashboard on port {port}...")
    # TODO: Start dashboard


if __name__ == "__main__":
    app()
