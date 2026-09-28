# TODO — Chrome Dino AI

Phase-by-phase task list derived from [PLAN.md](PLAN.md). Check items off as they land.
Each phase ends with an **Exit criteria** line: do not start the next phase until it holds.

Legend: `[x]` done · `[ ]` open · ⭐ priority (Priority 1 = do first)
Hours: plan with PLAN.md's 139–182h estimate.

## Progress (updated 2026-09-27)

| Phase | Done | Total | Status |
|---|---|---|---|
| 0 Decisions & plan hygiene | 9 | 11 | Done except Docker (blocked on 6.1) and `.env` loader (deferred) |
| 1 Foundation | 21 | 22 | Done except docker_config (deferred to 8.2) |
| 2 Core RL + monitoring | 12 | 13 | Exit criteria met; only 2.1's "better than random" check is open (needs a longer run) |
| 3 Caching & testing | 4 | 5 | Exit criteria met; 3.1 dropped (frames never repeat); only 3.2's "better than random" check is open |
| 4 Optimization & comparison | 4 | 6 | 4.1, 4.2, 4.5 (resume) and 4.6 (`--n-envs`, 3.7x faster) done; 4.3/4.4 need long runs on the Windows machine |
| 5 Multi-agent routing | 3 | 5 | 5.1, 5.2 and 5.4 done; 5.3 Laya code done, measuring it on Windows; 5.5 Laya as a player working (mean score 3872), head-height birds open |
| 6–8 | 4 | 11 | Phase 6 done (exit criteria met); Phases 7–8 not started |

**Next up:** a long `dino-ai train --all --parallel` run on the Windows machine, then `dino-ai eval --compare` and `dino-ai report`, to close the DQN and PPO "better than random" checks and start 4.3. Also on Windows: `python -m src.routing.laya_benchmark` and `play --laya` to close 5.3. Here: Phase 6 done; round 1 of the training experiments is running.
**Branches:** Phase 0 is PR [#1](https://github.com/S09Z/dino-decision-ai/pull/1), merged into `main`; Phase 1.5 is draft PR [#2](https://github.com/S09Z/dino-decision-ai/pull/2), based on `main`; Phase 1.2 is draft PR [#3](https://github.com/S09Z/dino-decision-ai/pull/3), stacked on #2; Phase 1.3 is draft PR [#4](https://github.com/S09Z/dino-decision-ai/pull/4), stacked on #3; Phase 1.4 is draft PR [#5](https://github.com/S09Z/dino-decision-ai/pull/5), stacked on #4; Phase 2.1 is draft PR [#6](https://github.com/S09Z/dino-decision-ai/pull/6), stacked on #5; Phase 2.2 is draft PR [#7](https://github.com/S09Z/dino-decision-ai/pull/7), stacked on #6; Phase 2.3 is draft PR [#8](https://github.com/S09Z/dino-decision-ai/pull/8), stacked on #7; Phase 2.4 is draft PR [#9](https://github.com/S09Z/dino-decision-ai/pull/9), stacked on #8; Phase 2.5 is draft PR [#10](https://github.com/S09Z/dino-decision-ai/pull/10), stacked on #9; Phase 3.2 is draft PR [#11](https://github.com/S09Z/dino-decision-ai/pull/11), stacked on #10; Phase 3.3 is draft PR [#13](https://github.com/S09Z/dino-decision-ai/pull/13), stacked on #11 (#12 was the user's merge of 1.5 + 1.2 into `main`); Phase 3.4 is draft PR [#14](https://github.com/S09Z/dino-decision-ai/pull/14), stacked on #13; Windows 11 support ([docs/WINDOWS.md](docs/WINDOWS.md)) is draft PR [#15](https://github.com/S09Z/dino-decision-ai/pull/15), stacked on #14; Phase 4.1 is draft PR [#16](https://github.com/S09Z/dino-decision-ai/pull/16), stacked on #15; Phase 4.2 is draft PR [#17](https://github.com/S09Z/dino-decision-ai/pull/17), stacked on #16; Phase 5.1 is draft PR [#18](https://github.com/S09Z/dino-decision-ai/pull/18), stacked on #17; Phase 5.2 is draft PR [#19](https://github.com/S09Z/dino-decision-ai/pull/19), stacked on #18; Phase 5.3 is draft PR [#20](https://github.com/S09Z/dino-decision-ai/pull/20), stacked on #19; Phase 5.4 is `claude/routing-5-4`, stacked on #20.

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
- [ ] Fix Docker: compose runs `src.dashboard.api` (missing) while Dockerfile `CMD` runs `main.py`; align once the dashboard exists — unblocked by 6.1 (`uvicorn src.dashboard.api:app` now works); the image still has no Chrome (8.2)
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
- [ ] Learns measurably better than random on a short run (smoke test). Not conclusive: after 20k steps (28 min) DQN survives 74 ± 20 steps vs random 64 ± 7 (20 episodes each; reward −92.7 vs −93.7). Too short for pixel DQN; re-check with a longer run 100k-step run on the Windows machine (RTX 5070, 2026-09-27/28, paused at 90k and resumed): greedy `eval --compare --episodes 20` → random 69 ± 5 steps, DQN (best, step 90k) 70 ± 9, PPO (best, step 85k) 67 ± 1, routed 82 ± 24. Still random level. In training, PPO rose to ~123 steps per episode around 65–85k, then fell back to ~65 (instability); DQN crept to 74–86 at the end. Next: round 1 of `src/training/experiment.py` (two actions, epsilon 0.01, cropped frames)

### 2.2 Profiling & benchmarking ⭐ P2 (4–5h)
- [x] `src/profiling/`: steps/s, memory and CPU (Python + Chrome), GPU memory, function timing, `@profiler.profile`. GPU utilisation % is not available from torch on MPS
- [x] Baseline report: [docs/PROFILING_BASELINE.md](docs/PROFILING_BASELINE.md) (13.4 steps/s; frame capture ~19ms is the main cost after the 50ms step wait)

### 2.3 Metrics database ⭐ P2 (3–4h)
- [x] `src/monitoring/local_db.py` (stdlib `sqlite3`, no SQLAlchemy; tables: episodes, training, performance, routing; default file `models/logs/metrics.db`, git-ignored)
- [x] `MetricsDB.add_episode`, `add_training`, `add_performance`, `add_routing`, `query_recent_episodes`. Nothing writes to it yet: training hooks it up in 2.5
- [x] Tests (`tests/test_local_db.py`)

### 2.4 Model versioning ⭐ P2 (3–4h)
- [x] `checkpoint_manager.py` (save best, keep N, cleanup): `CheckpointManager(keep_best_n=5)`, ranked by reward per agent, `models/checkpoints/` + one index per agent (`dqn.json`, `ppo.json`)
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
- [ ] Smoke test vs random baseline. Not conclusive: after 20k steps (29 min + 10 paused updates) PPO survives 80 ± 20 steps vs random 68 ± 11 (20 episodes each; reward −92.1 vs −93.3), the same picture as DQN in 2.1. Re-check with a longer run 100k-step run on the Windows machine (RTX 5070, 2026-09-27/28, paused at 90k and resumed): greedy `eval --compare --episodes 20` → random 69 ± 5 steps, DQN (best, step 90k) 70 ± 9, PPO (best, step 85k) 67 ± 1, routed 82 ± 24. Still random level. In training, PPO rose to ~123 steps per episode around 65–85k, then fell back to ~65 (instability); DQN crept to 74–86 at the end. Next: round 1 of `src/training/experiment.py` (two actions, epsilon 0.01, cropped frames)

### 3.3 Testing framework ⭐ P3 (6–8h)
- [x] `test_environment.py`, agents (`test_dqn_agent.py`, `test_ppo_agent.py`), screen capture and input with a fake Playwright page (`test_chrome_game.py`), `test_smoke.py`. `test_routing.py` and `test_dashboard.py` come with their code (5.4, 6.x)
- [x] Coverage >80%: 94% total, enforced by `--cov-fail-under=80` in `make test`. `pytest-xdist` measured and not added: parallel runs were slower (4 workers 43s, `-n auto` 54s vs 30s serial) because each worker re-imports torch/SB3 and one real-Chrome test dominates; revisit when the suite grows. It did expose an order-dependent logger test, now fixed

### 3.4 CLI expansion (2–3h)
- [x] `train --all --profile`, `eval --compare`, `inspect [--agent]`, `clean --keep N`. `eval --compare` evaluates random plus every agent's best checkpoint (PLAN's `--compare v1.0.0` needs registered versions, and nothing registers them yet); `inspect` shows checkpoints, releases and recorded episodes

**Exit criteria:** both agents train via CLI; coverage ≥80%; `make lint` clean. ✅ Met (2026-09-27): `dino-ai train --agent ppo|--all` (PPO trained for real in the 3.2 smoke run), 94% coverage with an 80% gate, lint clean.

---

## Phase 4 — Optimization & comparison (20–30h)

- [x] 4.1 Parallel training (`dino-ai train --all --parallel`, `src/training/parallel.py`): each agent in its own process with its own Chrome, since two agents cannot share one real-time game; a free-RAM check before starting and a status line every minute (episodes, last-10 reward, RAM of all processes). Checkpoint indexes are now per agent, fixing a lost-update race between processes (reproduced in a test). "Auto batch size" dropped: the model holds ~13MB of GPU memory, so batch size is not what limits resources; RAM (Chrome + replay buffer) is, and the RAM check covers it. Real run on the Mac: DQN + PPO together used 2.0–2.5GB — 4–5h
- [x] 4.2 Deep profiling: `dino-ai report` writes an HTML page comparing agents from `MetricsDB` (summary table, learning curve, speed and memory charts) with a leak check: mean memory in the last third of training vs the middle third, flagged above +20% (the first third is warm-up). Performance rows now record agent and step, so parallel runs stay apart (older DBs are upgraded). Measured 3,000 real-Chrome steps: Python heap saw-tooths 240–640MB with GC (trend −49MB per 10k steps), total RSS climbs for ~1,500 steps then stays at 1.9–2.4GB: no leak; a straight-line fit misreads that as +1,584MB per 10k steps, hence the thirds comparison. Line profiler not added: the main cost after the step wait is the frame grab inside Chrome's JS (2.2 baseline), which a Python line profiler cannot see — 3–4h
- [x] 4.5 Resume after a crash or power cut: every checkpoint also overwrites `models/checkpoints/<agent>_latest.zip` (plus DQN's replay buffer, `<agent>_latest_buffer.pkl`, ~1.4GB) and `<agent>_latest.json` (step, episodes); files are written under a temporary name and renamed, so a power cut mid-save keeps the previous one. The game is paused while saving. `train --resume` (also with `--all --parallel`) continues from it: same step count and epsilon schedule, episode numbers carry on; `--steps` is the total. `clean` keeps the latest checkpoint
- [ ] 4.3 DQN vs PPO: 250K steps each, compare metrics, write up findings — 4–6h
- [ ] 4.4 Hyperparameter tuning (lr, entropy coef, GAE λ), ablations, document — 8–10h
- [x] 4.6 Several games per agent (`--n-envs N` on `train` and `python -m src.training.experiment`): each game is a Chrome in its own process (`SubprocVecEnv`), so the 50ms real-time steps overlap. DQN takes N gradient steps per update and PPO splits `n_steps` across the games, so learning per sample stays the same; those settings survive loading a checkpoint. Checkpoints come every `checkpoint_every` steps as a distance (the count moves in steps of N). A DQN replay buffer resumes only into the same N (otherwise a new one starts, with a warning). Measured: DQN on 4 games 46 steps/s vs 12.4 on one (3.7x), with round 2 training alongside, so 250k steps take ~1.5h instead of ~5.6h

**Exit criteria:** comparison report committed; best hyperparameters recorded in configs.

---

## Phase 5 — Multi-agent routing (8–12h)

- [x] 5.1 `LayaRouter.decide_agent(dqn_score, ppo_score, difficulty)` → (agent, confidence) in `src/routing/laya_router.py`; heuristic fallback when Laya is unavailable, fails or answers out of range (`last_source` says which decided). Scores are game scores (rewards are always ~−93, so they cannot rank agents); difficulty is EASY/MEDIUM/HARD from the game speed (6 → 13, now in the env's state). Heuristic: the higher score wins, confidence 0.5 + 0.5·tanh(5 × relative lead), so the plan's DQN 1850 vs PPO 2340 gives PPO at 0.89. Laya plugs in as a `classifier` in 5.3, after measuring its speed and RAM — 2–3h
- [x] 5.2 `AgentManager` (`src/routing/agent_manager.py`) and `dino-ai play`: loads both agents' best checkpoints; the router decides at each episode start and difficulty change (not every step, so one agent never undoes the other's move); each stretch earns its agent the score gained, averaged per difficulty over the last 10; an agent tried fewer than 3 times at a difficulty plays first ("explore"), so an early tie cannot lock PPO out; every decision goes to the `routing` table, which gains `difficulty` and `source` columns — 2–3h
- [ ] 5.3 Laya integration against the real API (`laya` 0.3.x; verify the plan's `classify` example against the docs) — 1–2h
  - [x] Real API checked: there is no `Laya().classify`; it is `laya.load("convaiinnovations/laya")` (~843MB, ModernBERT-large encoder, Apache-2.0) and `agent.system_one(state, questions)`, whose `choice` answer has a calibrated `answer_confidence`. `LayaClassifier` (`src/routing/laya_classifier.py`) asks one choice question (dqn or ppo) about both agents' recent scores and the difficulty, and plugs into `LayaRouter(classifier=...)`; `play --laya` loads it up front and pauses the game while it decides. Tested with a fake Laya (no download in CI)
  - [ ] Measure on the Windows machine: `python -m src.routing.laya_benchmark --out docs/LAYA_BENCHMARK.md` (load time, RAM, ms per decision, agreement with the heuristic), then `play --episodes 20 --laya` vs `play --episodes 20`. Keep the heuristic as the default unless Laya scores better
- [x] 5.4 Routing tests: decisions, switching, score tracking, decision logging — 2–3h. Added: scores average only the last `window` stretches and are kept per difficulty (an agent plays where it is better), the router switches back when the leader's scores drop, keeping the same agent is not a switch, every decision (including mid-episode difficulty changes) is logged once with its difficulty, and a classifier failing mid-play falls back to the heuristic. For the exit criterion, `eval --compare` now adds a `routed` row (AgentManager choosing, same reward/length measure as the single agents; includes its explore tries)
- [ ] 5.5 Laya as a player (from "Laya plays Dino"): Laya decides every step from the game state, not frames, and the game keeps running while it decides
  - [x] `ChromeGame.state()` adds `jumping`, `ducking` and the obstacles ahead (`cactus`/`bird`, gap `d` from the T-Rex's front, width, top, height), the same in the vendored game and Chrome 153; `ChromeGame(game="chrome")` plays `chrome://dino` (`Runner.getInstance()` exists only after the first key press)
  - [x] `src/routing/laya_player.py`: a sentence stating the relation ("88 pixels, which is less than the 104-pixel window"; window = speed x 14 frames, less half the obstacle's width, so cactus groups are jumped later), two `noul` questions (jump, duck) cut at 0.7, `skip:air` guard, one JSONL line per decision (`logs/laya_play.jsonl`, the video's fields: `laya` answer vs `exec` key). `RulePlayer` answers from the numbers (ground truth; `--delay-ms` mimics Laya). CLI `play --player laya|rule [--game chrome] [--show]`
  - [x] Scenario benchmark (`python -m src.routing.laya_player_benchmark` → [docs/LAYA_PLAYER.md](docs/LAYA_PLAYER.md)), 16 labelled states: choice on raw numbers 7/16, yes/no on raw numbers 1/16, yes/no on the sentence 14/16 at cut 0.5 (13/16 at 0.7), ~36ms per decision on the RTX 5070; the article's pattern holds
  - [x] Live (5 games each): Laya mean game score 289 at cut 0.5 (early jumps onto cacti) → **3872** at 0.7 (max 7341, up to 386s alive; the best RL agent lasts ~15s); rule player with 45ms delay 9274 at lead 12
  - [ ] Every Laya game at 0.7 ends on a bird at head height: Laya answers jump, not duck (its detectors barely tell obstacle kinds apart). Next: ask only the question that fits the obstacle; try lead 12; fast drop after close landings
  - [x] Feed the HUD panels 01–05 (6.3): every decision goes to `MetricsDB.decisions` and streams to the dashboard's LIVE PLAYER section (input, scores per action with the cut, trace of the last 100, output with the key pressed, decision stream). RL agents too: `python -m src.training.experiment <variant> --watch` plays a checkpoint in a visible Chrome and records DQN's Q-values (PPO: action probabilities) per step. Checked live with `dqn-a2-eps-crop` (225k): 5 games, mean 569 steps, best 1056

**Exit criteria:** routed agent scores ≥ the better single agent on the eval set, or the gap is documented. Measure with `dino-ai eval --compare --episodes 20` on the Windows machine after the long training run. Measured after the 100k run: routed 82 ± 24 vs DQN 70 ± 9 (20 episodes each), so routed ≥ better single agent, but all three are at random level (69), so this is not yet meaningful.

---

## Phase 6 — Dashboard UI (12–16h)

- [x] 6.1 `src/dashboard/api.py` (FastAPI, read-only over `MetricsDB` and the checkpoints, so it runs beside training): `/health`, `/agents/status` (training/idle from the last episode's age, episodes, best checkpoint, resume step; experiment variants listed after dqn/ppo), `/metrics/latest` (last episode, last-10 mean reward/length, last performance and training rows, last routing decision), `/metrics/history?table=&agent=&limit=` (last rows, oldest first), `WS /ws` (a `/metrics/latest` snapshot on connect and on each change, checked every second; 100ms streaming is 6.2). `dino-ai dashboard --port` / `make dashboard` start it. Checked against the live DB during the resumed 100k run: both agents `training`, history in 0.2s. Timestamps are UTC — 3–4h
- [x] 6.2 `src/dashboard/ws_handler.py`: on connect a snapshot (status, latest, recent history of every table), then every 100ms only the rows added since (episodes, training, performance, routing = the decision log) and the status every second; a client that reconnects gets a fresh snapshot, so nothing is replayed. Query helpers shared with the REST API in `src/dashboard/queries.py` — 2–3h
- [x] 6.3 Frontend (vanilla JS, no libraries, served by the API at `/`): agent cards (training/idle, last-10 steps survived vs random, reward, best and resume checkpoints, speed), learning curve (moving average of 20, random baseline), speed and memory charts, last router decision, decision stream; reconnects with backoff (0.5s → 8s); UTC shown as local time. Checked in a browser against the live DB: it exposed a resume bug (first speed after `--resume` counted steps since 0: 240 steps/s logged at 95k), now fixed with a test — 4–6h
  - Look and layout from the **Laya Player** design (claude.ai artifact https://claude.ai/artifact/SREtSABPzuMBszYXqMgbsc, private to the owner): dark phosphor-terminal HUD, JetBrains Mono + Chakra Petch (Noto Sans Thai for Thai), lime accent for cactus/jump, cyan for bird/duck, red decision cut; header stats, then numbered panels 00 game view, 01 input (state sentence), 02 yes/no detectors with the 0.50 cut, 03 trace (last 100), 04 output action, 05 decision stream (Laya's answer and the key actually pressed as separate columns, e.g. `skip:air`)
  - Panels 00–05 need the Laya player (5.5) for data; until then the same look shows training (agents, curves, speed) and router decisions from the 6.1 API
- [x] 6.4 Integration: `python -m src.dashboard.loadtest` (temporary DB and server; a writer adds rows while WebSocket clients listen; commit → arrival per row and client, plus REST timing) → [docs/DASHBOARD_LATENCY.md](docs/DASHBOARD_LATENCY.md): at the default 100ms check, p50 ≈ 51ms, p95 ≈ 100ms, all rows delivered up to 20 clients; 50ms checks halve latency but starve REST at 20 clients (p95 3.8s), so 100ms stays. Live against round 1: all variants `training`, 30 episodes and 41 status updates streamed in 45s — 2–3h

**Exit criteria:** dashboard shows live metrics from a running training job. ✅ Met (2026-09-28) during round 1 of the experiments.

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
