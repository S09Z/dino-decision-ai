<#
Windows PowerShell version of the Makefile (see docs/WINDOWS.md).

Usage:
    .\make.ps1 <target> [extra arguments]

Examples:
    .\make.ps1 setup
    .\make.ps1 cuda
    .\make.ps1 test
    .\make.ps1 train --all --steps 100000
    .\make.ps1 eval --compare

Extra arguments are passed on to the command (train, eval, profile, test).
Keep this file ASCII-only: Windows PowerShell 5.1 reads BOM-less files as ANSI.
#>
param([string]$Target = "help")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# CUDA build for the RTX 5070 (RTX 50-series needs CUDA 12.8+); keep the
# versions in step with torch/torchvision in poetry.lock
$TorchVersion = "2.14.0"
$TorchvisionVersion = "0.29.0"
$CudaIndex = "https://download.pytorch.org/whl/cu130"

function Run {
    # Native commands do not stop on failure by themselves; stop on a non-zero exit code
    & $args[0] @($args | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    "help" {
        Write-Host "Chrome Dino AI - Development Commands (PowerShell)"
        Write-Host "setup      : Install dependencies, create .env.local"
        Write-Host "install    : Install dependencies (plus requirements.txt via uv, if installed)"
        Write-Host "cuda       : Swap in the CUDA torch build for the NVIDIA GPU (rerun after install)"
        Write-Host "train      : Train agents     e.g. .\make.ps1 train --all --steps 100000"
        Write-Host "eval       : Evaluate agents  e.g. .\make.ps1 eval --compare"
        Write-Host "profile    : Profile performance"
        Write-Host "dashboard  : Start dashboard"
        Write-Host "test       : Run tests"
        Write-Host "lint       : Code quality checks"
        Write-Host "format     : Auto-format code"
        Write-Host "clean      : Clean up files"
    }
    "setup" {
        Run poetry install
        if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
    }
    "install" {
        Run poetry install
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Run uv pip install -r requirements.txt
        } else {
            Write-Host "uv not found: skipped requirements.txt (poetry install covers it)"
        }
    }
    "cuda" {
        Run poetry run pip install --force-reinstall --no-deps "torch==$TorchVersion" "torchvision==$TorchvisionVersion" --index-url $CudaIndex
        Run poetry run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
        Run poetry run python -m src.performance.gpu_detector
    }
    "train" { Run poetry run python -m src.cli.main train --agent dqn @args }
    "eval" { Run poetry run python -m src.cli.main eval @args }
    "profile" { Run poetry run python -m src.cli.main profile --steps 500 @args }
    "dashboard" { Run poetry run uvicorn src.dashboard.api:app --reload --host 0.0.0.0 }
    "test" { Run poetry run pytest tests/ -v --cov=src --cov-fail-under=80 @args }
    "lint" {
        Run poetry run black --check src tests
        Run poetry run mypy src
    }
    "format" {
        Run poetry run black src tests
        Run poetry run isort src tests
    }
    "clean" {
        Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .pytest_cache, .coverage, htmlcov
        Remove-Item -Force -ErrorAction SilentlyContinue *.db, *.log
    }
    default {
        Write-Host "Unknown target '$Target'. Run .\make.ps1 help"
        exit 1
    }
}
