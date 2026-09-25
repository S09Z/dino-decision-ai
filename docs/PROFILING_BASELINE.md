# Profiling baseline

Measured 2026-09-25 with `python -m src.profiling.baseline --steps 500`: random actions in headless Chrome (`step_seconds=0.05`), then 200 DQN predictions on mps.
Machine: macOS-26.6.2-arm64-arm-64bit, 8 cores, 16GB RAM.

- **Env speed:** 13.4 steps/s
- **Memory (Python + Chrome):** 1336MB
- **CPU (Python + Chrome):** 20%
- **System RAM in use:** 82%
- **GPU memory (DQN loaded):** 13MB

## Function timing

| Function | Calls | Mean (ms) | Max (ms) | Total (s) |
|---|---|---|---|---|
| `ChromeDinoEnv.step` | 500 | 74.18 | 366.48 | 37.09 |
| `ChromeGame.frame` | 507 | 18.69 | 305.69 | 9.48 |
| `ChromeGame.state` | 507 | 2.05 | 90.86 | 1.04 |
| `ChromeGame.act` | 507 | 1.06 | 12.28 | 0.54 |
| `DQNAgent.predict` | 200 | 2.66 | 325.88 | 0.53 |

## Findings

- A step takes ~74ms: 50ms is the deliberate `step_seconds` wait, and reading the frame from the canvas (`ChromeGame.frame`, ~19ms) is the largest remaining cost. Shortening `step_seconds` or a faster frame grab are the levers for speed; the game is real time, so steps/s cannot exceed ~1/`step_seconds`.
- DQN inference is cheap (~2.7ms per action on MPS); the model holds ~13MB of GPU memory. Max times (~300ms) are first-call warm-up and Chrome start-up, not steady state.
- Python + Chrome use ~1.3GB and the system was at 82% RAM. The 50k-frame replay buffer adds up to ~1.4GB, so long runs are memory-bound on this machine: close other apps or lower `buffer_size`.
