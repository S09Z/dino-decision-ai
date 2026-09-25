"""Released model versions: models/vX.Y.Z/<agent>.zip plus a registry file
with metrics and notes (the changelog).

Usage:
    registry = ModelRegistry()
    v = registry.register(checkpoint.path, agent="dqn",
                          metrics={"mean_reward": -80.0}, notes="20k-step smoke run")
    registry.compare("1.0.0", v.version)
"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

MODELS_DIR = Path("models")  # git-ignored
REGISTRY_FILE = "registry.json"
FIRST_VERSION = "1.0.0"


def next_version(current: Optional[str], bump: str = "minor") -> str:
    """Semantic version after `current`; FIRST_VERSION when there is none"""
    if current is None:
        return FIRST_VERSION
    major, minor, patch = (int(part) for part in current.split("."))
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"bump must be major, minor or patch, not {bump!r}")


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


@dataclass
class ModelVersion:
    version: str
    agent: str
    path: str
    metrics: dict[str, float]
    notes: str
    timestamp: str


class ModelRegistry:
    """File-based registry of released models"""

    def __init__(self, root: Path = MODELS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._file = self.root / REGISTRY_FILE

    def versions(self, agent: Optional[str] = None) -> list[ModelVersion]:
        """Registered versions, oldest first, optionally for one agent"""
        if not self._file.exists():
            return []
        entries = [ModelVersion(**v) for v in json.loads(self._file.read_text())]
        entries.sort(key=lambda v: _key(v.version))
        return [v for v in entries if agent is None or v.agent == agent]

    def latest(self, agent: Optional[str] = None) -> Optional[ModelVersion]:
        entries = self.versions(agent)
        return entries[-1] if entries else None

    def get(self, version: str) -> ModelVersion:
        for entry in self.versions():
            if entry.version == version:
                return entry
        raise KeyError(f"Unknown model version {version}")

    def register(
        self,
        checkpoint: Union[str, Path],
        agent: str,
        metrics: dict[str, float],
        notes: str = "",
        bump: str = "minor",
    ) -> ModelVersion:
        """Copy a checkpoint to models/vX.Y.Z/<agent>.zip and record it"""
        latest = self.latest()
        version = next_version(latest.version if latest else None, bump)
        destination = self.root / f"v{version}" / f"{agent}.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, destination)
        entry = ModelVersion(
            version,
            agent,
            str(destination),
            metrics,
            notes,
            datetime.now().isoformat(timespec="seconds"),
        )
        entries = self.versions() + [entry]
        self._file.write_text(json.dumps([asdict(v) for v in entries], indent=2))
        return entry

    def compare(self, old: str, new: str) -> dict[str, tuple[float, float, float]]:
        """{metric: (old value, new value, change)} for metrics both versions have"""
        a, b = self.get(old).metrics, self.get(new).metrics
        return {m: (a[m], b[m], b[m] - a[m]) for m in a.keys() & b.keys()}

    def changelog(self) -> str:
        """Markdown list, newest first"""
        lines = []
        for v in reversed(self.versions()):
            metrics = ", ".join(f"{k} {val:g}" for k, val in sorted(v.metrics.items()))
            lines.append(f"- **v{v.version}** ({v.agent}, {v.timestamp}): {v.notes}")
            if metrics:
                lines[-1] += f" [{metrics}]"
        return "\n".join(lines)
