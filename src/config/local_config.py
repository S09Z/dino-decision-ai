"""Local machine optimization configuration"""

import torch
import psutil
from .base_config import BaseConfig


class LocalConfig(BaseConfig):
    """Configuration optimized for local machine"""
    
    # Auto-detect GPU
    USE_GPU = torch.cuda.is_available()
    
    # Auto-detect resources
    AVAILABLE_RAM_GB = psutil.virtual_memory().total / (1024**3)
    CPU_CORES = psutil.cpu_count()
    
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
        print(f"✓ GPU Available: {config.USE_GPU}")
        print(f"✓ CPU Cores: {config.CPU_CORES}")
        print(f"✓ RAM: {config.AVAILABLE_RAM_GB:.1f}GB")
        print(f"✓ Batch Size: {config.BATCH_SIZE}")
        print(f"✓ Workers: {config.NUM_WORKERS}")
        return config
