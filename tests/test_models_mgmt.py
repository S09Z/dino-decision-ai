"""Tests for checkpoint management and the model registry"""

from pathlib import Path

import pytest

from src.models_mgmt.checkpoint_manager import CheckpointManager
from src.models_mgmt.model_registry import ModelRegistry, next_version


class FakeAgent:
    """save/load like DQNAgent, storing a marker instead of weights"""

    def __init__(self, marker="weights"):
        self.marker = marker

    def save(self, path):
        Path(path).write_text(self.marker)

    def load(self, path):
        self.marker = Path(path).read_text()


@pytest.fixture
def manager(tmp_path):
    return CheckpointManager(tmp_path / "checkpoints", keep_best_n=2)


def zips(directory):
    return sorted(p.name for p in directory.glob("*.zip"))


def test_keeps_only_best_n_and_deletes_the_rest(manager):
    for step, reward in [(1, -90.0), (2, -50.0), (3, -70.0)]:
        manager.save(FakeAgent(f"s{step}"), "dqn", step=step, reward=reward)

    assert [c.step for c in manager.checkpoints("dqn")] == [2, 3]
    assert zips(manager.directory) == ["dqn_step2.zip", "dqn_step3.zip"]


def test_skips_saving_when_not_in_top_n(manager):
    manager.save(FakeAgent(), "dqn", step=1, reward=-10.0)
    manager.save(FakeAgent(), "dqn", step=2, reward=-20.0)

    assert manager.save(FakeAgent(), "dqn", step=3, reward=-30.0) is None
    assert not (manager.directory / "dqn_step3.zip").exists()


def test_agents_are_ranked_separately(manager):
    manager.save(FakeAgent(), "dqn", step=1, reward=-10.0)
    manager.save(FakeAgent(), "ppo", step=1, reward=-99.0)

    assert manager.best("dqn").reward == -10.0
    assert manager.best("ppo").reward == -99.0


def test_load_best_restores_the_best_weights(manager):
    manager.save(FakeAgent("good"), "dqn", step=1, reward=-10.0)
    manager.save(FakeAgent("bad"), "dqn", step=2, reward=-90.0)

    agent = FakeAgent("fresh")
    best = manager.load_best(agent, "dqn")

    assert best.step == 1
    assert agent.marker == "good"


def test_index_survives_a_new_manager(manager):
    manager.save(FakeAgent(), "dqn", step=7, reward=1.0)

    again = CheckpointManager(manager.directory, keep_best_n=2)
    assert again.best("dqn").step == 7


def test_prune_deletes_all_but_the_best(manager):
    manager.save(FakeAgent(), "dqn", step=1, reward=-10.0)
    manager.save(FakeAgent(), "dqn", step=2, reward=-20.0)
    manager.save(FakeAgent(), "ppo", step=1, reward=-5.0)

    removed = manager.prune("dqn", keep=1)

    assert [c.step for c in removed] == [2]
    assert zips(manager.directory) == ["dqn_step1.zip", "ppo_step1.zip"]
    assert manager.best("ppo").step == 1  # other agents untouched


def test_two_processes_saving_different_agents_keep_both(manager, monkeypatch):
    """Parallel training: dqn and ppo run in separate processes, each with its
    own CheckpointManager on the same directory. dqn saves between ppo reading
    the index and ppo writing it back; neither save may be lost."""
    dqn = CheckpointManager(manager.directory, keep_best_n=2)
    ppo = CheckpointManager(manager.directory, keep_best_n=2)
    original = ppo._load_index
    reads = []

    def read_then_dqn_saves(*args):
        index = original(*args)
        reads.append(1)
        if len(reads) == 2:  # the read that save() writes back
            dqn.save(FakeAgent(), "dqn", step=1, reward=-10.0)
        return index

    monkeypatch.setattr(ppo, "_load_index", read_then_dqn_saves)
    ppo.save(FakeAgent(), "ppo", step=1, reward=-20.0)

    fresh = CheckpointManager(manager.directory)
    assert fresh.best("dqn") is not None
    assert fresh.best("ppo") is not None


def test_load_best_without_checkpoints_raises(manager):
    with pytest.raises(FileNotFoundError):
        manager.load_best(FakeAgent(), "dqn")


class FakeOffPolicyModel(FakeAgent):
    """Like an SB3 DQN model: has a replay buffer to save"""

    replay_buffer = ["transitions"]

    def save_replay_buffer(self, path):
        Path(path).write_text("buffer")


def test_save_latest_overwrites_the_previous_one(manager):
    manager.save_latest(FakeAgent("first"), "ppo", step=100, episodes=3)
    latest = manager.save_latest(FakeAgent("second"), "ppo", step=200, episodes=7)

    assert manager.latest("ppo") == latest
    assert (latest.step, latest.episodes, latest.buffer) == (200, 7, None)
    assert Path(latest.path).read_text() == "second"
    files = sorted(p.name for p in manager.directory.iterdir())
    assert files == ["ppo_latest.json", "ppo_latest.zip"]  # no temporary files


def test_save_latest_keeps_the_replay_buffer(manager):
    latest = manager.save_latest(FakeOffPolicyModel(), "dqn", step=100, episodes=3)

    assert latest.buffer is not None
    assert Path(latest.buffer).read_text() == "buffer"


def test_latest_is_none_before_any_run(manager):
    assert manager.latest("dqn") is None


def test_power_cut_while_saving_keeps_the_previous_latest(manager):
    manager.save_latest(FakeAgent("good"), "dqn", step=100, episodes=3)

    class PowerCut(FakeAgent):
        def save(self, path):
            Path(path).write_text("half-written")
            raise OSError("power cut")

    with pytest.raises(OSError):
        manager.save_latest(PowerCut(), "dqn", step=200, episodes=5)

    latest = manager.latest("dqn")
    assert latest is not None and latest.step == 100
    assert Path(latest.path).read_text() == "good"


def test_pruning_best_checkpoints_keeps_the_latest(manager):
    manager.save(FakeAgent(), "dqn", step=100, reward=-50.0)
    latest = manager.save_latest(FakeAgent(), "dqn", step=100, episodes=3)

    manager.prune("dqn", keep=0)

    assert Path(latest.path).exists()


@pytest.mark.parametrize(
    "current, bump, expected",
    [
        (None, "minor", "1.0.0"),
        ("1.0.0", "minor", "1.1.0"),
        ("1.1.0", "patch", "1.1.1"),
        ("1.9.3", "major", "2.0.0"),
    ],
)
def test_next_version(current, bump, expected):
    assert next_version(current, bump) == expected


def test_next_version_rejects_unknown_bump():
    with pytest.raises(ValueError):
        next_version("1.0.0", "huge")


def test_register_copies_checkpoint_and_numbers_versions(tmp_path):
    checkpoint = tmp_path / "dqn_step1.zip"
    checkpoint.write_text("weights")
    registry = ModelRegistry(tmp_path / "models")

    first = registry.register(checkpoint, "dqn", {"mean_reward": -90.0}, "smoke run")
    second = registry.register(checkpoint, "ppo", {"mean_reward": -80.0}, bump="patch")

    assert (first.version, second.version) == ("1.0.0", "1.0.1")
    assert Path(first.path) == tmp_path / "models" / "v1.0.0" / "dqn.zip"
    assert Path(first.path).read_text() == "weights"
    assert registry.latest().version == "1.0.1"
    assert registry.latest("dqn").version == "1.0.0"
    assert ModelRegistry(tmp_path / "models").get("1.0.0").notes == "smoke run"


def test_compare_and_changelog(tmp_path):
    checkpoint = tmp_path / "c.zip"
    checkpoint.write_text("w")
    registry = ModelRegistry(tmp_path / "models")
    registry.register(checkpoint, "dqn", {"mean_reward": -90.0, "length": 64}, "a")
    registry.register(checkpoint, "dqn", {"mean_reward": -70.0}, "b")

    assert registry.compare("1.0.0", "1.1.0") == {"mean_reward": (-90.0, -70.0, 20.0)}
    log = registry.changelog().splitlines()
    assert log[0].startswith("- **v1.1.0** (dqn")
    assert "mean_reward -70" in log[0]
    with pytest.raises(KeyError):
        registry.get("9.9.9")


def test_works_with_a_real_dqn_agent(manager):
    from src.models.dqn_agent import DQNAgent
    from src.training.envs import make_dino_env
    from tests.test_dqn_agent import SMALL
    from tests.test_environment import FakeGame

    env = make_dino_env(game=FakeGame())
    agent = DQNAgent(env, SMALL, device="cpu")

    saved = manager.save(agent, "dqn", step=1, reward=0.0)
    manager.load_best(DQNAgent(env, SMALL, device="cpu"), "dqn")

    assert Path(saved.path).exists()
    env.close()
