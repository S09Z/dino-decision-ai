"""Detect GPU/CPU resources and summarise them with recommended settings"""

from dataclasses import dataclass
from typing import Optional

import psutil
import torch

from src.config.local_config import LocalConfig, detect_device

GB = 1024**3
LOW_RAM_GB = 4.0  # below this, a 50k-frame replay buffer risks swapping
LOW_GPU_MEMORY_GB = 2.0


@dataclass
class Hardware:
    device: str  # "cuda", "mps" or "cpu"
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]  # MPS: what torch recommends of unified memory
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float


def detect_hardware() -> Hardware:
    device = detect_device()
    gpu_name, gpu_memory_gb = None, None
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu_name, gpu_memory_gb = props.name, props.total_memory / GB
    elif device == "mps":
        gpu_name = "Apple GPU"
        gpu_memory_gb = torch.mps.recommended_max_memory() / GB
    memory = psutil.virtual_memory()
    return Hardware(
        device=device,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        cpu_cores=psutil.cpu_count() or 1,
        ram_total_gb=memory.total / GB,
        ram_available_gb=memory.available / GB,
    )


def recommendations(hw: Hardware) -> list[str]:
    """Warnings about resources that will slow or break training"""
    tips = []
    if hw.device == "cpu":
        tips.append("No GPU found: training runs on CPU and will be slow")
    elif hw.gpu_memory_gb is not None and hw.gpu_memory_gb < LOW_GPU_MEMORY_GB:
        tips.append(f"Only {hw.gpu_memory_gb:.1f}GB GPU memory: lower the batch size")
    if hw.ram_available_gb < LOW_RAM_GB:
        tips.append(
            f"Only {hw.ram_available_gb:.1f}GB RAM free: close other apps"
            " or lower the replay buffer size"
        )
    return tips


def summary(hw: Hardware, config: type[LocalConfig] = LocalConfig) -> str:
    """Human-readable hardware report with the settings LocalConfig picked"""
    if hw.gpu_name is None:
        gpu = "✗ GPU: none (CPU only)"
    else:
        gpu = f"✓ GPU: {hw.device.upper()} ({hw.gpu_name}, {hw.gpu_memory_gb:.0f}GB)"
    lines = [
        gpu,
        f"✓ CPU: {hw.cpu_cores} cores",
        f"✓ RAM: {hw.ram_available_gb:.1f}GB available of {hw.ram_total_gb:.0f}GB",
        "✓ Recommended settings:",
        f"  - Batch size: {config.BATCH_SIZE}",
        f"  - Num workers: {config.NUM_WORKERS}",
        f"  - Frame cache: {'Enabled' if config.FRAME_CACHE_ENABLED else 'Disabled'}",
    ]
    lines += [f"! {tip}" for tip in recommendations(hw)]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(detect_hardware()))
