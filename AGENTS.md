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

- `src/environment/` Gymnasium env (`dino_env.py`)
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

## Gotchas

- The `laya` package on PyPI tops out at 0.3.x; do not pin it to `>=1.0.0` (PLAN.md is outdated on this).
- Screen capture (`mss`) and `pydirectinput` are OS/display dependent; keep them out of unit tests and mock them.
- Many `src/` modules are still scaffolds. Check that a module has real logic before assuming behaviour.
