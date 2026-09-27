"""Local machine optimization configuration"""

import psutil
import torch

from .base_config import BaseConfig


def detect_device() -> str:
    """Best available torch device: CUDA, then Apple MPS, then CPU"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalConfig(BaseConfig):
    """Configuration optimized for local machine"""

    # Auto-detect GPU (CUDA or Apple MPS)
    DEVICE = detect_device()
    USE_GPU = DEVICE != "cpu"

    # Auto-detect resources
    AVAILABLE_RAM_GB = psutil.virtual_memory().total / (1024**3)
    CPU_CORES = psutil.cpu_count() or 1

    # Auto-size batch size
    BATCH_SIZE = 64 if USE_GPU else 32

    # Auto-size workers
    NUM_WORKERS = min(CPU_CORES // 2, 4)

    # Performance settings
    FRAME_CACHE_ENABLED = True
    GPU_PREPROCESSING = USE_GPU

    @staticmethod
    def auto_optimize():
        """Auto-optimize configuration for local machine"""
        config = LocalConfig()
        print(f"✓ Device: {config.DEVICE}")
        print(f"✓ CPU Cores: {config.CPU_CORES}")
        print(f"✓ RAM: {config.AVAILABLE_RAM_GB:.1f}GB")
        print(f"✓ Batch Size: {config.BATCH_SIZE}")
        print(f"✓ Workers: {config.NUM_WORKERS}")
        return config
