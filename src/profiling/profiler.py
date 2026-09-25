"""Function timing and resource sampling.

Usage:
    from src.profiling.profiler import profiler

    @profiler.profile
    def step(...): ...

    print(profiler.report())
"""

import functools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import psutil
import torch

F = TypeVar("F", bound=Callable[..., Any])
MB = 1024**2


@dataclass
class Timing:
    calls: int = 0
    total: float = 0.0  # seconds
    max: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.calls if self.calls else 0.0


class Profiler:
    """Collects wall-clock timings of decorated functions"""

    def __init__(self) -> None:
        self.timings: defaultdict[str, Timing] = defaultdict(Timing)

    def profile(self, func: F) -> F:
        """Decorator: record each call's duration under the function's name"""
        name = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                timing = self.timings[name]
                timing.calls += 1
                timing.total += elapsed
                timing.max = max(timing.max, elapsed)

        return wrapper  # type: ignore[return-value]

    def reset(self) -> None:
        self.timings.clear()

    def report(self) -> str:
        """Markdown table, slowest total time first"""
        lines = [
            "| Function | Calls | Mean (ms) | Max (ms) | Total (s) |",
            "|---|---|---|---|---|",
        ]
        for name, t in sorted(self.timings.items(), key=lambda kv: -kv[1].total):
            lines.append(
                f"| `{name}` | {t.calls} | {t.mean * 1000:.2f}"
                f" | {t.max * 1000:.2f} | {t.total:.2f} |"
            )
        return "\n".join(lines)


def resource_usage() -> dict[str, float]:
    """Memory and CPU of this process plus its children (Chrome runs as a
    child via Playwright), and GPU memory held by torch"""
    procs = [psutil.Process()]
    procs += procs[0].children(recursive=True)
    rss, cpu = 0.0, 0.0
    for proc in procs:
        try:
            rss += proc.memory_info().rss
            cpu += proc.cpu_percent(interval=0.1)
        except psutil.NoSuchProcess:
            pass
    if torch.cuda.is_available():
        gpu = torch.cuda.memory_allocated()
    elif torch.backends.mps.is_available():
        gpu = torch.mps.current_allocated_memory()
    else:
        gpu = 0
    return {
        "memory_mb": rss / MB,
        "cpu_percent": cpu,
        "gpu_memory_mb": gpu / MB,
        "system_ram_percent": psutil.virtual_memory().percent,
    }


profiler = Profiler()
