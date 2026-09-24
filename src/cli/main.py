"""CLI Entry Point"""

import typer

app = typer.Typer()


@app.command()
def train(agent: str = "both", profile: bool = False):
    """Train the agent(s)"""
    typer.echo(f"Training {agent} agent...")
    if profile:
        typer.echo("Profiling enabled")
    # TODO: Implement training


@app.command()
def eval(agent: str = "dqn", episodes: int = 100):
    """Evaluate the agent(s)"""
    typer.echo(f"Evaluating {agent} for {episodes} episodes...")
    # TODO: Implement evaluation


@app.command()
def profile(duration: int = 60):
    """Profile performance"""
    typer.echo(f"Profiling for {duration} seconds...")
    # TODO: Implement profiling


@app.command()
def dashboard(port: int = 8000):
    """Start dashboard"""
    typer.echo(f"Starting dashboard on port {port}...")
    # TODO: Start dashboard


if __name__ == "__main__":
    app()
