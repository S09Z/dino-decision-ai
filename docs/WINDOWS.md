# Running on Windows 11

Everything runs on Windows; three things differ from macOS/Linux: there is no `make`, PyPI's
torch for Windows is CPU-only, and the machine must not sleep during real-time training.
Commands below are PowerShell, run from the repo root.

## 1. Install the tools

- **Python 3.12** from python.org (tick "Add python.exe to PATH"). 3.10–3.13 work; 3.12 matches the dev machine.
- **Git** and **Google Chrome** (the env drives the installed Chrome through Playwright; no separate browser download).
- **Poetry**:

  ```powershell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
  ```

  Then add `%APPDATA%\Python\Scripts` to PATH if the installer asks.

## 2. Set up the project

```powershell
git clone https://github.com/S09Z/dino-decision-ai.git
cd dino-decision-ai
poetry install
Copy-Item .env.example .env.local
```

## 3. Use the NVIDIA GPU (CUDA)

`poetry install` puts the CPU-only torch on Windows. With an NVIDIA GPU and a current driver,
swap in the CUDA build of the same versions (`torch` 2.14.0, `torchvision` 0.29.0 in `poetry.lock`).
Pick the CUDA tag (`cu126`, `cu128`, ...) that pytorch.org's "Get Started" page lists for your
driver:

```powershell
poetry run pip install --force-reinstall torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu128
```

Running `poetry install` or `poetry sync` again puts the CPU build back; repeat this step after it.

Check the result. It should print `✓ GPU: CUDA (<your GPU>, <memory>GB)`:

```powershell
poetry run python -m src.performance.gpu_detector
```

## 4. Check everything works

```powershell
poetry run pytest tests/ -v --cov=src --cov-fail-under=80
poetry run black --check src tests
poetry run mypy src
```

One test plays the game in real Chrome for 1,000 steps (~80s); it is skipped if Chrome is missing.

## 5. Commands without `make`

| `make` target | PowerShell |
|---|---|
| `make train` | `poetry run dino-ai train --agent dqn` |
| `make eval` | `poetry run dino-ai eval` |
| `make profile` | `poetry run dino-ai profile --steps 500` |
| `make test` | `poetry run pytest tests/ -v --cov=src --cov-fail-under=80` |
| `make lint` | `poetry run black --check src tests` then `poetry run mypy src` |
| `make format` | `poetry run black src tests` then `poetry run isort src tests` |

## 6. Long training runs

```powershell
poetry run dino-ai train --all --steps 100000
poetry run dino-ai eval --compare
```

- The game runs in real time (~12 steps/s), so 100k steps per agent is ~2.3 hours each. The progress bar shows time left.
- Stop Windows from sleeping for the whole run: Settings → System → Power → Screen and sleep → "Never" while plugged in. Sleep or a heavy background job (video export, games, updates) slows the game loop and the dino dies more, which makes results look worse than they are.
- Results stay on that machine: checkpoints in `models/checkpoints/` (their index uses forward-slash paths, so the folder can be copied between Windows and macOS), metrics in `models/logs/metrics.db`, logs in `logs/`. All are git-ignored.
- `poetry run dino-ai inspect` shows what was kept; `poetry run dino-ai clean --keep 3` frees disk.
