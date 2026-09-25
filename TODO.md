# TODO — Chrome Dino AI

Phase-by-phase task list derived from [PLAN.md](PLAN.md). Check items off as they land.
Each phase ends with an **Exit criteria** line: do not start the next phase until it holds.

Legend: `[x]` done · `[ ]` open · ⭐ priority (Priority 1 = do first)
Hours: plan with PLAN.md's 139–182h estimate.

## Progress (updated 2026-09-25)

| Phase | Done | Total | Status |
|---|---|---|---|
| 0 Decisions & plan hygiene | 9 | 11 | Done except Docker (blocked on 6.1) and `.env` loader (deferred) |
| 1 Foundation | 21 | 22 | Done except docker_config (deferred to 8.2) |
| 2 Core RL + monitoring | 12 | 13 | Exit criteria met; only 2.1's "better than random" check is open (needs a longer run) |
| 3 Caching & testing | 1 | 5 | 3.1 dropped (frames never repeat, see below); 3.2 PPO built, "better than random" not conclusive yet |
| 4–8 | 0 | 19 | Not started |

**Next up:** Phase 3.3 (testing framework); a longer training run for DQN and PPO closes both "better than random" checks.
**Branches:** Phase 0 is PR [#1](https://github.com/S09Z/dino-decision-ai/pull/1), merged into `main`; Phase 1.5 is draft PR [#2](https://github.com/S09Z/dino-decision-ai/pull/2), based on `main`; Phase 1.2 is draft PR [#3](https://github.com/S09Z/dino-decision-ai/pull/3), stacked on #2; Phase 1.3 is draft PR [#4](https://github.com/S09Z/dino-decision-ai/pull/4), stacked on #3; Phase 1.4 is draft PR [#5](https://github.com/S09Z/dino-decision-ai/pull/5), stacked on #4; Phase 2.1 is draft PR [#6](https://github.com/S09Z/dino-decision-ai/pull/6), stacked on #5; Phase 2.2 is draft PR [#7](https://github.com/S09Z/dino-decision-ai/pull/7), stacked on #6; Phase 2.3 is draft PR [#8](https://github.com/S09Z/dino-decision-ai/pull/8), stacked on #7; Phase 2.4 is draft PR [#9](https://github.com/S09Z/dino-decision-ai/pull/9), stacked on #8; Phase 2.5 is draft PR [#10](https://github.com/S09Z/dino-decision-ai/pull/10), stacked on #9; Phase 3.2 is `claude/ppo-agent`, stacked on #10.

---

## Phase 0 — Decisions & plan hygiene (new, do first)

- [x] **Decide the game environment approach** → real Chrome, driven by Playwright:
  - ~~Option A: simulated Dino game~~ (not chosen)
  - [x] Option B: real Chrome. `chrome://dino` cannot be automated, so Chrome runs a vendored copy of Chromium's dino code (`src/environment/game/`) via Playwright, headless by default
- [x] Fix PLAN.md: change "✅" on unbuilt features and Success Metrics to targets (⬜)
- [x] Fix PLAN.md: reconcile 66–88h vs 139–182h estimates (now 139–182h throughout)
- [x] Fix PLAN.md: `laya>=1.0.0` → `>=0.3.13`; verify the Laya link/homepage (real package: huggingface.co/convaiinnovations/laya)
- [x] Declare `typer` in `pyproject.toml` + `requirements.txt`
- [x] Get `make lint` green: black/isort (black profile), `types-psutil`, `cpu_count() or 1`; Makefile now uses `poetry run`
- [ ] Fix Docker: compose runs `src.dashboard.api` (missing) while Dockerfile `CMD` runs `main.py`; align once the dashboard exists — blocked on Phase 6.1
- [x] Remove stray `src/frontend/` (real UI is top-level `frontend/`)
- [ ] Add a `.env` loader dependency if config will read env vars (e.g. `python-dotenv`) — deferred: nothing reads env vars yet
- [x] Fix observation space: kept one `(84, 84, 1)` frame (stack via `VecFrameStack` at training); fixed `reset(seed, options)` so `check_env` passes; added `tests/test_environment.py`

**Exit criteria:** environment approach chosen; PLAN.md no longer claims unbuilt work is done; `make lint` and `make test` both pass.

---

## Phase 1 — Foundation & local optimization (20–26h)

### 1.1 Project setup
- [x] Poetry project, dependencies, virtualenv
- [x] `.env.example` → `.env.local` via `make setup`
- [x] Makefile
- [x] `.gitignore`, `AGENTS.md`, `CLAUDE.md`, `push-draft-pr` skill
- [x] `git init`, `origin` remote (github.com/S09Z/dino-decision-ai), initial commit

### 1.2 Configuration system ⭐ P1 (4–6h)
- [x] `base_config.py`, `local_config.py` exist
- [x] Auto-sizing in `LocalConfig`: `DEVICE` (CUDA → MPS → CPU), RAM/CPU detection, `BATCH_SIZE`, `NUM_WORKERS`, frame-cache and GPU-preprocessing flags (GPU memory moves to 1.3)
- [x] `dqn_config.py` (lr 1e-4, buffer 50k, batch 32, gamma 0.99, eps 1.0→0.1, target update 1000); fields are SB3 `DQN` arguments
- [x] `ppo_config.py` (lr 1e-4, batch 32, gamma 0.99, GAE λ 0.95, ent coef 0.01); fields are SB3 `PPO` arguments
- [ ] `docker_config.py` — deferred to Phase 8.2: the Docker image has no Chrome, so the env cannot run there yet
- [x] Unit tests for config (`tests/test_config.py`, incl. SB3 accepting the configs)

### 1.3 GPU/CPU detection ⭐ P1 (3–4h)
- [x] `src/performance/gpu_detector.py`: GPU name/memory (CUDA, Apple MPS), CPU, RAM; warnings for CPU-only, low GPU memory, low free RAM; reuses `detect_device()`
- [x] Human-readable summary output (`python -m src.performance.gpu_detector`); recommended settings come from `LocalConfig`
- [x] Unit tests (`tests/test_gpu_detector.py`, torch/psutil faked)

### 1.4 Logging ⭐ P1 (2–3h)
- [x] `src/utils/logger.py` with `get_logger(__name__)`: rich console, rotating file `logs/dino_ai.log` (10MB × 5), `time | level | name | message` format; stdlib `logging` + rich (loguru stays unused)
- [x] Unit tests (`tests/test_logger.py`)

### 1.5 Environment (8–10h)
- [x] `dino_env.py`: real `reset` / `step`; Discrete(3) actions; reward +0.1/step, −100 on collision
- [x] Game state and game-over detection: read from the game's JS in `chrome_game.py` (no separate `game_state.py`)
- [x] Capture + preprocessing to 84×84 grayscale: done in-page by `ChromeGame.frame()` (~45 frames/s)
- [x] Input control: Playwright keyboard (jump = Space, duck = hold ArrowDown)
- [x] Pass `gymnasium.utils.env_checker.check_env`
- [x] `tests/test_environment.py` (fake-game unit tests + one real-Chrome test)

**Exit criteria:** a random-action agent runs 1,000 steps in `ChromeDinoEnv`; `make lint` and `make test` pass. ✅ Met for 1.5: 1,000 steps in 81s (12.3 steps/s), 16 episodes, random agent scores ~42.

---

## Phase 2 — Core RL + performance monitoring (22–28h)

### 2.1 DQN agent (6–8h)
- [x] CNN architecture, replay buffer, target network, epsilon-greedy: SB3 `DQN` with `CnnPolicy` in `DQNAgent`, on `make_dino_env()` (4 stacked frames); replay buffer stores each frame once to halve RAM
- [x] Training loop with the plan's hyperparameters (`DQNConfig`); `python -m src.training.smoke --agent dqn`
- [ ] Learns measurably better than random on a short run (smoke test). Not conclusive: after 20k steps (28 min) DQN survives 74 ± 20 steps vs random 64 ± 7 (20 episodes each; reward −92.7 vs −93.7). Too short for pixel DQN; re-check with a longer run

### 2.2 Profiling & benchmarking ⭐ P2 (4–5h)
- [x] `src/profiling/`: steps/s, memory and CPU (Python + Chrome), GPU memory, function timing, `@profiler.profile`. GPU utilisation % is not available from torch on MPS
- [x] Baseline report: [docs/PROFILING_BASELINE.md](docs/PROFILING_BASELINE.md) (13.4 steps/s; frame capture ~19ms is the main cost after the 50ms step wait)

### 2.3 Metrics database ⭐ P2 (3–4h)
- [x] `src/monitoring/local_db.py` (stdlib `sqlite3`, no SQLAlchemy; tables: episodes, training, performance, routing; default file `models/logs/metrics.db`, git-ignored)
- [x] `MetricsDB.add_episode`, `add_training`, `add_performance`, `add_routing`, `query_recent_episodes`. Nothing writes to it yet: training hooks it up in 2.5
- [x] Tests (`tests/test_local_db.py`)

### 2.4 Model versioning ⭐ P2 (3–4h)
- [x] `checkpoint_manager.py` (save best, keep N, cleanup): `CheckpointManager(keep_best_n=5)`, ranked by reward per agent, `models/checkpoints/` + `checkpoints.json`
- [x] `model_registry.py`: releases to `models/vX.Y.Z/<agent>.zip`, `registry.json` with metrics and notes, `compare`, `changelog`; version numbering (`next_version`) lives here, so no separate `version_tracker.py`
- [x] Tests (`tests/test_models_mgmt.py`, incl. a real `DQNAgent` save/load)

### 2.5 CLI phase 1 ⭐ P2 (3–4h)
- [x] Wire `train`, `eval`, `profile` in `src/cli/main.py` to real code: `train` records to `MetricsDB` and keeps the best checkpoints via `TrainingMonitor` (`src/training/callbacks.py`); `eval` loads the best checkpoint; `profile` runs the baseline. `--agent` accepts `dqn` only until PPO (3.2); `dashboard` is still a stub
- [x] Register `dino-ai` script entry point in `pyproject.toml` (`poetry run dino-ai --help`)

**Exit criteria:** `make train` trains DQN end-to-end, logs to the DB, saves a checkpoint; `make eval` loads it. ✅ Met (2026-09-25): a 400-step real-Chrome run logged 7 episodes, 2 training/performance rows and 2 checkpoints; `eval` loaded the best one.

---

## Phase 3 — Caching & testing (22–28h)

### 3.1 Frame caching ⭐ P2 (5–6h) — dropped (2026-09-25)
Measured first: over 1,000 real-Chrome steps, 0% of frames and 0% of 4-frame stacks repeated (the ground and clouds scroll every frame), so a frame or prediction cache would never hit. The memory saving is already done in 2.1 (replay buffer stores each frame once, ~50% less RAM). A model cache can come with the router (Phase 5) if switching agents turns out to be slow.
- ~~`src/cache/`: `frame_cache.py`, `model_cache.py`, `prediction_cache.py`, `cache_manager.py`~~
- ~~LRU eviction, hash lookup, hit/miss stats~~
- ~~Measure the memory reduction (target ~30%, unverified)~~

### 3.2 PPO agent (6–8h)
- [x] Actor/critic networks, GAE, clipped loss, training loop: SB3 `PPO` with `CnnPolicy` in `PPOAgent` (shares `SB3Agent` with DQN); `PPOConfig` gains `n_steps` (2048). The game is paused during each update (~9s on MPS), otherwise the dino crashes with nobody playing (`PauseDuringUpdates`, `ChromeGame.pause/resume`)
- [ ] Smoke test vs random baseline. Not conclusive: after 20k steps (29 min + 10 paused updates) PPO survives 80 ± 20 steps vs random 68 ± 11 (20 episodes each; reward −92.1 vs −93.3), the same picture as DQN in 2.1. Re-check with a longer run

### 3.3 Testing framework ⭐ P3 (6–8h)
- [ ] `test_environment.py`, `test_agents.py`, `test_screen_capture.py` (mocked), `test_routing.py`, `test_dashboard.py`
- [ ] Add `pytest-xdist`; coverage target >80% on real (non-stub) modules

### 3.4 CLI expansion (2–3h)
- [ ] `train --all --profile`, `eval --compare`, `inspect --model`, `clean --keep N`

**Exit criteria:** both agents train via CLI; coverage ≥80%; `make lint` clean.

---

## Phase 4 — Optimization & comparison (20–30h)

- [ ] 4.1 `ParallelTrainer` (DQN + PPO in parallel, shared env, resource monitor, auto batch size) — 4–5h
- [ ] 4.2 Deep profiling: line profiler, leak detection, HTML comparison reports — 3–4h
- [ ] 4.3 DQN vs PPO: 250K steps each, compare metrics, write up findings — 4–6h
- [ ] 4.4 Hyperparameter tuning (lr, entropy coef, GAE λ), ablations, document — 8–10h

**Exit criteria:** comparison report committed; best hyperparameters recorded in configs.

---

## Phase 5 — Multi-agent routing (8–12h)

- [ ] 5.1 `LayaRouter.decide_agent(dqn_score, ppo_score, difficulty)` → (agent, confidence); heuristic fallback when Laya is unavailable — 2–3h
- [ ] 5.2 `AgentManager`: init both agents, track scores, switch, log switches — 2–3h
- [ ] 5.3 Laya integration against the real API (`laya` 0.3.x; verify the plan's `classify` example against the docs) — 1–2h
- [ ] 5.4 Routing tests: decisions, switching, score tracking, decision logging — 2–3h

**Exit criteria:** routed agent scores ≥ the better single agent on the eval set, or the gap is documented.

---

## Phase 6 — Dashboard UI (12–16h)

- [ ] 6.1 `src/dashboard/api.py`: `/health`, `/agents/status`, `/metrics/latest`, `/metrics/history`, `WS /ws` (`make dashboard` currently fails: file missing) — 3–4h
- [ ] 6.2 `ws_handler.py`: 100ms streaming, decision log, reconnect handling — 2–3h
- [ ] 6.3 Frontend (vanilla JS): live agent switching, metrics, decision log, charts — 4–6h
- [ ] 6.4 Integration: API ↔ training, WS ↔ metrics, latency <100ms, load test — 2–3h

**Exit criteria:** dashboard shows live metrics from a running training job.

---

## Phase 7 — Training & evaluation (27–32h)

- [ ] 7.1 Long runs: DQN and PPO to 500K+ steps, monitor, checkpoint — 20h+
- [ ] 7.2 Evaluate both (100+ episodes), report, compare with Laya routing — 4–5h
- [ ] 7.3 Learning curves, reward distributions, action frequencies, comparison report — 3–4h

**Exit criteria:** final report and best checkpoints saved (checkpoints stay out of git).

---

## Phase 8 — Documentation & deployment (7–10h)

- [ ] 8.1 Docstrings, type hints (`mypy src` clean), README, architecture, setup guide — 4–5h
- [ ] 8.2 Docker: `Dockerfile`, `docker-compose.yml`, `.env.example`, deployment guide — 2–3h
- [ ] 8.3 GitHub release: push, README, release notes — 1–2h

**Exit criteria:** fresh clone → `make setup` → `make test` passes; release tagged.

---

## Backlog (post v1.0)

Curriculum training · transfer learning · multi-game support · cloud deployment · mobile dashboard · W&B tracking · Ray Tune · model ensemble · distributed training · quantization · edge deployment
