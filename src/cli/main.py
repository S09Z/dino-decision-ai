"""CLI Entry Point"""

from pathlib import Path
from typing import Optional

import typer

from src.evaluation.evaluate import evaluate, report
from src.models import AGENTS
from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.monitoring.local_db import MetricsDB
from src.training.callbacks import TrainingMonitor
from src.training.envs import make_dino_env

app = typer.Typer()


def _agent_class(agent: str):
    if agent not in AGENTS:
        typer.echo(f"Unknown agent {agent!r}; available: {', '.join(AGENTS)}", err=True)
        raise typer.Exit(1)
    return AGENTS[agent]


@app.command()
def train(agent: str = "dqn", steps: int = 20_000, checkpoint_every: int = 5_000):
    """Train an agent, recording episodes to the metrics DB and keeping the
    best checkpoints (real time: ~12 steps/s)"""
    agent_class = _agent_class(agent)
    env = make_dino_env()
    try:
        with MetricsDB() as db:
            monitor = TrainingMonitor(agent, db, CheckpointManager(), checkpoint_every)
            agent_class(env).train(steps, callback=monitor, progress_bar=True)
            typer.echo(
                f"Trained {agent} for {steps} steps, {monitor.episodes} episodes"
            )
    finally:
        env.close()


@app.command()
def eval(agent: str = "dqn", episodes: int = 20):
    """Evaluate the agent's best checkpoint"""
    agent_class = _agent_class(agent)
    env = make_dino_env()
    try:
        model = agent_class(env)
        try:
            best = CheckpointManager().load_best(model, agent)
        except FileNotFoundError as error:
            typer.echo(f"{error}; run `train` first", err=True)
            raise typer.Exit(1)
        typer.echo(f"Loaded {best.path} (step {best.step}, reward {best.reward:.1f})")
        typer.echo(report(agent, *evaluate(model.model, env, episodes)))
    finally:
        env.close()


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
