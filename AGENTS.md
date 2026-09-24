# Chrome Dino AI

Multi-agent RL system (DQN + PPO) that plays the Chrome Dinosaur game, with a Laya-based router that picks the best agent in real time and a FastAPI + static-web dashboard. Python 3.10+, Poetry (with uv for `requirements.txt`).

## Commands

Run from the repo root. Prefer `make` targets over ad-hoc invocations.

```bash
make setup      # poetry install + create .env.local from .env.example
make install    # poetry + uv pip install -r requirements.txt
make train      # python -m src.cli.main train --agent both
make eval       # python -m src.cli.main eval
make profile    # python -m src.cli.main profile --duration 60
make dashboard  # uvicorn src.dashboard.api:app --reload (http://localhost:8000)
make test       # pytest tests/ -v --cov=src
make lint       # black --check + mypy (src, tests)
make format     # black + isort (src, tests)
```

Run a single test: `poetry run pytest tests/test_example.py::test_name -v`.

## Layout

- `src/environment/` Gymnasium env (`dino_env.py`), Chrome driver (`chrome_game.py`), vendored game (`game/`, do not edit; see `game/SOURCE.md`)
- `src/models/` DQN and PPO agent code (source code, not weights)
- `src/training/` training pipelines, parallel trainer
- `src/routing/` Laya router and agent manager
- `src/dashboard/` FastAPI backend; `frontend/` is the plain-JS web UI
- `src/config/`, `src/performance/`, `src/monitoring/`, `src/profiling/`, `src/cli/`
- `/models/` (top level) holds trained weights and is git-ignored; do not confuse it with `src/models/`
- `docs/` architecture and RL notes; `PLAN.md` is the original roadmap

## Conventions

- Format with black and isort before finishing; `make lint` must pass (mypy on `src`).
- Add or update tests in `tests/` for behaviour changes; run `make test`.
- Never commit `.env*` (except `.env.example`), model weights, logs, or datasets.
- `poetry.lock` is committed. Change dependencies in `pyproject.toml` and keep `requirements.txt` in sync.

## Behavioral guidelines

Adapted from [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills) (MIT). They bias toward caution over speed; for trivial tasks, use judgment.

1. **Think before coding.** State assumptions; if uncertain, ask. If several interpretations exist, present them instead of picking silently. Say so when a simpler approach exists. If something is unclear, stop and name it.
2. **Simplicity first.** Minimum code that solves the problem: no unrequested features, no abstractions for single-use code, no speculative configurability, no error handling for impossible cases. If 200 lines could be 50, rewrite.
3. **Surgical changes.** Touch only what the request needs; match existing style. Do not refactor or reformat adjacent code. Mention unrelated dead code, do not delete it. Remove only the imports/variables your own change orphaned. Every changed line should trace to the request.
4. **Goal-driven execution.** Turn tasks into verifiable goals (bug fix → reproducing test first; refactor → tests pass before and after). For multi-step work, state a short plan with a verify step for each item, and loop until verified.

## Current state (verified 2026-09-24)

The project is an early scaffold (~250 lines of Python); nothing trains or plays yet. Track progress in `TODO.md`, not `PLAN.md`.

- **Real code:** `src/config/`: `LocalConfig` (machine resources; `DEVICE` is CUDA → Apple MPS → CPU via `detect_device()`), `DQNConfig` and `PPOConfig` (dataclasses whose fields are Stable-Baselines3 constructor arguments: `DQN(policy, env, **asdict(DQNConfig()))`; put algorithm hyperparameters here, not in `BaseConfig`). `src/performance/gpu_detector.py`: hardware report and warnings (`python -m src.performance.gpu_detector`); its recommended settings come from `LocalConfig`. `src/utils/logger.py`: `get_logger(__name__)` (rich console + rotating `logs/dino_ai.log`). The Typer CLI shell in `src/cli/main.py` (`python -m src.cli.main --help` works).
- **Stubs (`# TODO`/no-op):** `DQNAgent` and `PPOAgent` (`train/predict/save/load` do nothing; they wrap Stable-Baselines3 but never build a model), every CLI command body.
- **Empty packages (only `__init__.py`):** `training`, `routing`, `dashboard`, `cache`, `evaluation`, `monitoring`, `profiling`, `models_mgmt`. The UI is top-level `frontend/` (33-line `app.js`, 20-line `index.html`).
- **Environment:** `ChromeDinoEnv` runs the dino game in real Google Chrome (must be installed) via Playwright, headless by default (`render_mode="human"` shows the window). It is real time: ~12 steps/s with the default `step_seconds=0.05`, so training is wall-clock bound. Game randomness is not seedable.
- **Baseline checks:** `make lint` and `make test` pass (28 tests; ~16s because one test drives real Chrome and is skipped if Chrome is unavailable). Makefile targets use `poetry run`, so no venv activation is needed. Keep both green.
- **Broken targets:** `make dashboard` and `docker compose up` run `uvicorn src.dashboard.api:app`, but `src/dashboard/api.py` does not exist. The Dockerfile `CMD` runs `main.py` (the CLI), which conflicts with the compose command.
- **Runtime:** local `.venv` is Python 3.12 (`pyproject` allows `^3.10`). All runtime and dev dependencies, including `typer`, are declared.
- **Git:** repo initialised on `main` with `origin` → github.com/S09Z/dino-decision-ai ; `push-draft-pr` needs a feature branch off `main`.

## Gotchas

- The `laya` package on PyPI tops out at 0.3.x; do not pin it to `>=1.0.0` (PLAN.md is outdated on this). Nothing imports it yet, and PLAN.md's `Laya().classify(...)` example is unverified against the real API.
- `ChromeDinoEnv` emits one `(84, 84, 1)` frame; stack frames with `VecFrameStack` at training time, not inside the env.
- `chrome://dino` cannot be automated (sandboxed error page), hence the vendored game. It is served through Playwright request routing, not `file://`, because `file://` taints the canvas and blocks pixel reads; all other network requests are blocked.
- `mss`, `pydirectinput` (Windows-only) and `pytesseract` are still declared but unused since the Playwright approach; do not build on them.
- Unit tests use a fake game (`FakeGame` in `tests/test_environment.py`); only one test launches Chrome. Keep it that way.
- PLAN.md is the original roadmap (goals marked ⬜ are targets, estimate 139–182h); `TODO.md` is the status.
- `models/` (top level) is git-ignored and mounted into Docker; `src/models/` is source. Do not confuse them.
