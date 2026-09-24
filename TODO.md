# TODO — Chrome Dino AI

Phase-by-phase task list derived from [PLAN.md](PLAN.md). Check items off as they land.
Each phase ends with an **Exit criteria** line: do not start the next phase until it holds.

Legend: `[x]` done · `[ ]` open · ⭐ priority (Priority 1 = do first)
Hours: plan with PLAN.md's 139–182h estimate.

## Progress (updated 2026-09-24)

| Phase | Done | Total | Status |
|---|---|---|---|
| 0 Decisions & plan hygiene | 7 | 12 | Only the environment decision, Docker (blocked) and `.env` loader (deferred) remain |
| 1 Foundation | 6 | 22 | 1.1 setup done; 1.2 partial; 1.3–1.5 not started |
| 2–8 | 0 | 40 | Not started |

**Next up:** decide the environment approach (blocks 1.5), then Phase 1.2–1.4.
**Branch:** Phase 0 work is on `claude/phase-0-hygiene` (not pushed yet).

---

## Phase 0 — Decisions & plan hygiene (new, do first)

- [ ] **Decide the game environment approach** (blocks Phase 1.5 and everything after):
  - [ ] Option A (recommended): headless simulated Dino game, fast, deterministic, testable
  - [ ] Option B: drive real Chrome via `mss` screen capture + input control (`pydirectinput` is Windows-only; pick a macOS input library if chosen)
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
- [x] `base_config.py`, `local_config.py` exist (partial: no MPS detection; `dqn_config`/`ppo_config` still missing)
- [ ] Auto-sizing in `LocalConfig`: GPU flag, GPU/RAM/CPU detection, `BATCH_SIZE`, `NUM_WORKERS`, frame-cache and GPU-preprocessing flags
- [ ] `dqn_config.py` (lr 1e-4, buffer 50k–100k, batch 32, gamma 0.99, eps 1.0→0.1, target update 1000)
- [ ] `ppo_config.py`
- [ ] `docker_config.py`
- [ ] Unit tests for config

### 1.3 GPU/CPU detection ⭐ P1 (3–4h)
- [ ] `src/performance/gpu_detector.py`: GPU (CUDA **and Apple MPS**), GPU memory, CPU cores, RAM, recommendations
- [ ] Human-readable summary output
- [ ] Unit tests (mock torch/psutil)

### 1.4 Logging ⭐ P1 (2–3h)
- [ ] `src/utils/logger.py` with `get_logger(__name__)`: rich console, file logging, rotation, structured format
- [ ] Unit tests

### 1.5 Environment (8–10h)
- [ ] `dino_env.py`: real `reset` / `step` (replace stubs); Discrete(3) actions; reward +0.1/frame, −100 on collision
- [ ] `game_state.py`: state management, game-over detection
- [ ] `screen_utils.py`: capture + preprocessing to 84×84 grayscale (only if Option B or for a render path)
- [ ] Input control (per Phase 0 decision)
- [ ] Pass `gymnasium.utils.env_checker.check_env`
- [ ] `tests/test_environment.py`

**Exit criteria:** a random-action agent runs 1,000 steps in `ChromeDinoEnv`; `make lint` and `make test` pass.

---

## Phase 2 — Core RL + performance monitoring (22–28h)

### 2.1 DQN agent (6–8h)
- [ ] CNN architecture, replay buffer, target network, epsilon-greedy
- [ ] Training loop with the plan's hyperparameters
- [ ] Learns measurably better than random on a short run (smoke test)

### 2.2 Profiling & benchmarking ⭐ P2 (4–5h)
- [ ] `src/profiling/`: FPS, memory, CPU/GPU utilisation, function timing, `@profiler.profile`
- [ ] Baseline report

### 2.3 Metrics database ⭐ P2 (3–4h)
- [ ] `src/monitoring/local_db.py` (SQLite; tables: episodes, training, performance, routing)
- [ ] `MetricsDB.add_episode`, `query_recent_episodes`, etc.
- [ ] Tests

### 2.4 Model versioning ⭐ P2 (3–4h)
- [ ] `checkpoint_manager.py` (save best, keep N, cleanup)
- [ ] `model_registry.py`, `version_tracker.py`
- [ ] Tests

### 2.5 CLI phase 1 ⭐ P2 (3–4h)
- [ ] Wire `train`, `eval`, `profile` in `src/cli/main.py` to real code (currently `# TODO` stubs)
- [ ] Register `dino-ai` script entry point in `pyproject.toml`

**Exit criteria:** `make train` trains DQN end-to-end, logs to the DB, saves a checkpoint; `make eval` loads it.

---

## Phase 3 — Caching & testing (22–28h)

### 3.1 Frame caching ⭐ P2 (5–6h)
- [ ] `src/cache/`: `frame_cache.py`, `model_cache.py`, `prediction_cache.py`, `cache_manager.py`
- [ ] LRU eviction, hash lookup, hit/miss stats
- [ ] Measure the memory reduction (target ~30%, unverified)

### 3.2 PPO agent (6–8h)
- [ ] Actor/critic networks, GAE, clipped loss, training loop
- [ ] Smoke test vs random baseline

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
