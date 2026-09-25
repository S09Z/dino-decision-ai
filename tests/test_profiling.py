"""Tests for the profiler and baseline (fake game, no Chrome)"""

import pytest

from src.environment.dino_env import ChromeDinoEnv
from src.profiling import profiler as profiler_module
from src.profiling.baseline import profile_env, profile_inference
from src.profiling.profiler import Profiler, resource_usage
from tests.test_environment import FakeGame


def test_profile_records_calls_and_keeps_function_behaviour():
    profiler = Profiler()

    @profiler.profile
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    add(1, 1)

    timing = profiler.timings[add.__qualname__]
    assert add.__name__ == "add"
    assert timing.calls == 2
    assert 0 <= timing.mean <= timing.max <= timing.total


def test_profile_records_calls_that_raise():
    profiler = Profiler()

    @profiler.profile
    def fail():
        raise ValueError

    with pytest.raises(ValueError):
        fail()
    assert profiler.timings[fail.__qualname__].calls == 1


def test_report_lists_slowest_first_and_reset_clears():
    profiler = Profiler()
    profiler.timings["fast"].total = 0.1
    profiler.timings["slow"].total = 2.0
    profiler.timings["fast"].calls = profiler.timings["slow"].calls = 1

    rows = profiler.report().splitlines()[2:]
    assert rows[0].startswith("| `slow`")
    assert rows[1].startswith("| `fast`")

    profiler.reset()
    assert profiler.report().count("\n") == 1  # header only


def test_resource_usage_reports_positive_memory():
    usage = resource_usage()

    assert usage["memory_mb"] > 0
    assert usage["cpu_percent"] >= 0
    assert usage["gpu_memory_mb"] >= 0
    assert 0 < usage["system_ram_percent"] <= 100


def test_shared_profiler_instance_exists():
    assert isinstance(profiler_module.profiler, Profiler)


def test_profile_env_times_env_and_game_calls():
    profiler = Profiler()
    env = ChromeDinoEnv(step_seconds=0, game=FakeGame(crash_after=5))

    steps_per_s = profile_env(env, profiler, steps=12)

    assert steps_per_s > 0
    names = {name.split(".")[-1]: t.calls for name, t in profiler.timings.items()}
    assert names["step"] == 12
    assert names["act"] == 12


def test_profile_inference_times_dqn_predict():
    profiler = Profiler()

    gpu_memory_mb = profile_inference(profiler, calls=3, device="cpu")

    assert gpu_memory_mb >= 0
    assert profiler.timings["DQNAgent.predict"].calls == 3
