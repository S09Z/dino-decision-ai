"""Tests for GPU/CPU detection (torch and psutil are faked)"""

from types import SimpleNamespace

import pytest

from src.config.local_config import LocalConfig
from src.performance import gpu_detector
from src.performance.gpu_detector import (
    GB,
    Hardware,
    detect_hardware,
    recommendations,
    summary,
)


def hardware(**overrides):
    values = dict(
        device="cuda",
        gpu_name="RTX 4090",
        gpu_memory_gb=24.0,
        cpu_cores=12,
        ram_total_gb=32.0,
        ram_available_gb=20.0,
    )
    values.update(overrides)
    return Hardware(**values)


@pytest.fixture
def fake_machine(monkeypatch):
    """Fake psutil (12 cores, 32GB, 20GB free); returns a device setter"""
    monkeypatch.setattr(gpu_detector.psutil, "cpu_count", lambda: 12)
    monkeypatch.setattr(
        gpu_detector.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=32 * GB, available=20 * GB),
    )

    def use(device):
        monkeypatch.setattr(gpu_detector, "detect_device", lambda: device)

    return use


def test_detects_cuda_gpu(monkeypatch, fake_machine):
    fake_machine("cuda")
    props = SimpleNamespace(name="RTX 4090", total_memory=24 * GB)
    monkeypatch.setattr(
        gpu_detector.torch.cuda, "get_device_properties", lambda i: props
    )
    assert detect_hardware() == hardware()


def test_detects_apple_gpu(monkeypatch, fake_machine):
    fake_machine("mps")
    monkeypatch.setattr(
        gpu_detector.torch.mps, "recommended_max_memory", lambda: 12 * GB
    )
    assert detect_hardware() == hardware(
        device="mps", gpu_name="Apple GPU", gpu_memory_gb=12.0
    )


def test_detects_cpu_only(fake_machine):
    fake_machine("cpu")
    assert detect_hardware() == hardware(
        device="cpu", gpu_name=None, gpu_memory_gb=None
    )


def test_no_recommendations_for_a_capable_machine():
    assert recommendations(hardware()) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(device="cpu", gpu_name=None, gpu_memory_gb=None), "No GPU"),
        (dict(gpu_memory_gb=1.5), "1.5GB GPU memory"),
        (dict(ram_available_gb=2.4), "2.4GB RAM free"),
    ],
)
def test_warns_about_limited_resources(overrides, expected):
    (tip,) = recommendations(hardware(**overrides))
    assert expected in tip


def test_summary_reports_hardware_and_local_config_settings():
    text = summary(hardware(ram_available_gb=2.4))
    assert "✓ GPU: CUDA (RTX 4090, 24GB)" in text
    assert "✓ CPU: 12 cores" in text
    assert "✓ RAM: 2.4GB available of 32GB" in text
    assert f"  - Batch size: {LocalConfig.BATCH_SIZE}" in text
    assert f"  - Num workers: {LocalConfig.NUM_WORKERS}" in text
    assert text.endswith("close other apps or lower the replay buffer size")


def test_summary_without_gpu():
    text = summary(hardware(device="cpu", gpu_name=None, gpu_memory_gb=None))
    assert "✗ GPU: none (CPU only)" in text
