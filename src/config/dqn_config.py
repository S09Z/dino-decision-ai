"""DQN hyperparameters"""

from dataclasses import dataclass, field


@dataclass
class DQNConfig:
    """Field names match stable_baselines3.DQN arguments:
    DQN(policy, env, **asdict(DQNConfig()))
    """

    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    batch_size: int = 32
    gamma: float = 0.99
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.1
    target_update_interval: int = 1000
    # Store each frame once in the replay buffer (~1.4GB instead of ~2.8GB at
    # 50k stacked frames); SB3 requires timeout handling off with it, which is
    # safe because ChromeDinoEnv never truncates
    optimize_memory_usage: bool = True
    replay_buffer_kwargs: dict = field(
        default_factory=lambda: {"handle_timeout_termination": False}
    )
