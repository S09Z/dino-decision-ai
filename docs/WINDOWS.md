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

`poetry install` puts the CPU-only torch on Windows. Swap in the CUDA build of the same versions
(`torch` 2.14.0, `torchvision` 0.29.0 in `poetry.lock`).

For the training machine (**RTX 5070, 12GB**) use **`cu130`**. RTX 50-series (Blackwell, `sm_120`)
needs a CUDA 12.8+ build; for torch 2.14.0 on Windows, pytorch.org has `cu126` (too old for
RTX 50), `cu130` and `cu132` (checked 2026-09-27). Update the NVIDIA driver first
(CUDA 13 needs driver 580 or newer; GeForce Game Ready/Studio from nvidia.com or the NVIDIA app).

```powershell
poetry run pip install --force-reinstall --no-deps torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
```

Running `poetry install` or `poetry sync` again puts the CPU build back; repeat this step after it.

Check the result. The first command should print the GPU and a list that includes `sm_120`;
the second should print `✓ GPU: CUDA (NVIDIA GeForce RTX 5070, 12GB)`:

```powershell
poetry run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
poetry run python -m src.performance.gpu_detector
```

If training fails with "no kernel image is available for execution on the device", the installed
build lacks `sm_120`: reinstall with `cu130` as above.

## 4. Check everything works

```powershell
poetry run pytest tests/ -v --cov=src --cov-fail-under=80
poetry run black --check src tests
poetry run mypy src
```

One test plays the game in real Chrome with random actions until the dino crashes (a few seconds); it is skipped if Chrome is missing.

## 5. Commands without `make`: `make.ps1`

`make.ps1` in the repo root has the same targets as the Makefile, plus `cuda` (step 3 in one go).
Extra arguments go through to the command:

```powershell
.\make.ps1 help
.\make.ps1 setup
.\make.ps1 cuda
.\make.ps1 test
.\make.ps1 train --all --steps 100000
.\make.ps1 eval --compare
```

Windows blocks scripts until you allow them. Either allow scripts you created or cloned
(once, for your user), or bypass the policy for a single run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
powershell -ExecutionPolicy Bypass -File .\make.ps1 test
```

## 6. Long training runs

```powershell
.\make.ps1 train --all --parallel --steps 100000
.\make.ps1 eval --compare
poetry run dino-ai play --episodes 20
poetry run dino-ai report
```

- The game runs in real time (~12 steps/s), so 100k steps per agent is ~2.3 hours. `--parallel` trains DQN and PPO at the same time (one Chrome each, ~4GB RAM), so both finish in ~2.3 hours instead of ~4.6; without it they train one after the other with a progress bar. In parallel mode a status line every minute shows episodes, recent reward and RAM.
- `--n-envs 4` trains each agent on 4 games at once (4 Chromes, each in its own process): ~46 steps/s instead of ~12, measured on the RTX 5070 machine while three other runs trained, so 100k steps take ~40 minutes. Each extra game needs ~0.5GB RAM. Resuming a DQN run with a different `--n-envs` keeps its weights and step count but starts a new replay buffer.
- Stop Windows from sleeping for the whole run: Settings → System → Power → Screen and sleep → "Never" while plugged in. Sleep or a heavy background job (video export, games, updates) slows the game loop and the dino dies more, which makes results look worse than they are.
- If the machine shuts down or restarts mid-run, run the same command with `--resume`, e.g. `.\make.ps1 train --all --parallel --steps 100000 --resume`: each agent continues from its latest checkpoint (saved every `--checkpoint-every` steps, 5,000 by default, so at most ~7 minutes are lost) with the same step count, epsilon schedule and DQN replay buffer. Pause Windows Update for the week of a long run (Settings → Windows Update → Pause updates): its automatic restarts are the usual cause.
- Results stay on that machine: checkpoints in `models/checkpoints/` (their index uses forward-slash paths, so the folder can be copied between Windows and macOS), metrics in `models/logs/metrics.db`, logs in `logs/`. All are git-ignored.
- `eval --compare` evaluates random, each agent's best checkpoint and routed play (the router choosing between them) the same way; the routed line should match or beat the better single agent (Phase 5's exit criterion).
- Laya (the router's decision model, Phase 5.3) is measured here, where the GPU is: `poetry run python -m src.routing.laya_benchmark --out docs/LAYA_BENCHMARK.md` downloads convaiinnovations/laya (~843MB, into the HuggingFace cache) on first run, then reports load time, memory, time per decision and how often it agrees with the score heuristic. After that, `poetry run dino-ai play --episodes 20 --laya` lets Laya choose the agent (the game pauses while it decides); compare its mean score with `play` without `--laya`.
- `poetry run dino-ai report` writes `models/logs/report.html`: learning curves, speed and memory per agent, and a leak check. `poetry run dino-ai inspect` shows what was kept; `poetry run dino-ai clean --keep 3` frees disk.
