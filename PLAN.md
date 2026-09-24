# 🦖 CHROME DINO AI - MASTER DEVELOPMENT PLAN

**Project:** Autonomous Multi-Agent RL System with Laya Router & Local Dashboard  
**Tech Stack:** Python 3.10+ | Gymnasium | Stable-Baselines3 | PyTorch | Laya | FastAPI | Poetry + UV  
**Platform:** macOS + Local Development + Docker (Optional)  
**Timeline:** 12-13 weeks  
**Target:** Production-ready local RL system with intelligent agent routing  

---

## 📑 TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Project Vision](#project-vision)
3. [Tech Stack & Dependencies](#tech-stack--dependencies)
4. [Project Structure](#project-structure)
5. [Claude & Codex Collaboration](#claude--codex-collaboration)
6. [Phase 1: Foundation & Local Optimization](#phase-1-foundation--local-optimization)
7. [Phase 2: Core RL + Performance Monitoring](#phase-2-core-rl--performance-monitoring)
8. [Phase 3: Caching & Testing](#phase-3-caching--testing)
9. [Phase 4: Optimization & Comparison](#phase-4-optimization--comparison)
10. [Phase 5: Multi-Agent Routing](#phase-5-multi-agent-routing)
11. [Phase 6: Dashboard UI](#phase-6-dashboard-ui)
12. [Phase 7: Training & Evaluation](#phase-7-training--evaluation)
13. [Phase 8: Documentation & Deployment](#phase-8-documentation--deployment)
14. [Core 10 Components](#core-10-components)
15. [Complete Timeline](#complete-timeline)
16. [Success Metrics](#success-metrics)
17. [Development Workflow](#development-workflow)
18. [CLI Commands](#cli-commands)
19. [RL Concepts Guide](#rl-concepts-guide)
20. [Resources & References](#resources--references)

---

## EXECUTIVE SUMMARY

This is a **comprehensive production-grade guide** for building Chrome Dino AI with:

### What You'll Build
- ✅ DQN & PPO agents learning to play Chrome Dinosaur game
- ✅ Laya Model-based intelligent agent router (picks best agent in real-time)
- ✅ Real-time dashboard showing agent decisions & performance
- ✅ Multi-agent orchestration with parallel training
- ✅ Professional ML infrastructure (versioning, testing, monitoring)

### Key Features
- ✅ Auto GPU/CPU detection & optimization
- ✅ Frame caching (30% memory reduction)
- ✅ Performance profiling & bottleneck identification
- ✅ SQLite metrics database with history
- ✅ CLI with 10+ commands for easy operation
- ✅ Comprehensive test suite (8 test modules)
- ✅ Model versioning & checkpoint management
- ✅ Real-time WebSocket dashboard

### Core Components: 18 Total
- **10 MUST HAVE:** Config, GPU Detection, Profiling, Logging, Checkpoint Mgmt, CLI, Testing, Caching, Metrics DB, Versioning
- **8 SUPPORTING:** Parallel Training, Laya Router, FastAPI, WebSocket, Frontend, Memory Tracking, Reports, Documentation

### Timeline
- **12-13 weeks** total (can be extended or compressed)
- **8 phases** with clear deliverables
- **66-88 hours** development time
- **Parallelizable:** Can work on multiple phases simultaneously

---

## PROJECT VISION

Build a **research-grade local ML system** that demonstrates:

1. **Deep RL Mastery**
   - Understand DQN algorithm deeply
   - Understand PPO algorithm deeply
   - Compare algorithms in real-time

2. **Multi-Agent Systems**
   - Run agents in parallel
   - Intelligent agent selection
   - Performance-based routing

3. **Professional ML Engineering**
   - Configuration management
   - Model versioning & checkpointing
   - Performance profiling & optimization
   - Comprehensive testing
   - Production-grade logging

4. **Real-Time Observability**
   - Live performance dashboard
   - Decision logging
   - Metrics tracking
   - Agent switching visualization

5. **Local Optimization**
   - Auto GPU/CPU detection
   - Memory optimization
   - Caching strategies
   - Performance monitoring

---

## TECH STACK & DEPENDENCIES

### Core RL
```yaml
gymnasium: ">=0.28.0"          # Environment API
stable-baselines3: ">=2.0.0"   # RL algorithms
torch: ">=2.0.0"               # Deep learning
torchvision: ">=0.15.0"        # Vision utilities
```

### Computer Vision & Processing
```yaml
opencv-python: ">=4.7.0"       # Image processing
numpy: ">=1.24.0"              # Numerical computing
matplotlib: ">=3.7.0"          # Visualization
pandas: ">=2.0.0"              # Data manipulation
```

### Screen Capture & Control
```yaml
mss: ">=7.0.1"                 # Fast screen capture
pydirectinput: ">=1.0.4"       # Keyboard control
pytesseract: ">=0.3.10"        # OCR (optional)
```

### Multi-Agent & Routing
```yaml
laya: "latest"                 # Laya decision router (NEW)
```

### UI & API
```yaml
fastapi: ">=0.104.0"           # Web framework (NEW)
uvicorn: ">=0.24.0"            # ASGI server (NEW)
pydantic: ">=2.0.0"            # Data validation (NEW)
websockets: ">=12.0"           # Real-time WebSocket (NEW)
```

### Monitoring & Performance
```yaml
rich: ">=13.0.0"               # Pretty logging
loguru: ">=0.7.0"              # Structured logging
line_profiler: ">=4.0.0"       # Line profiling
psutil: ">=5.9.0"              # System info
```

### Development & Testing
```yaml
jupyter: ">=1.0.0"             # Notebooks
pytest: ">=7.3.0"              # Testing
pytest-cov: ">=4.1.0"          # Coverage
black: ">=23.0.0"              # Code formatting
mypy: ">=1.0.0"                # Type checking
```

### Package Management
```yaml
poetry: "latest"               # Dependency management
uv: "latest"                   # Fast pip alternative
```

### Optional (for Docker)
```yaml
docker: "latest"               # Containerization
docker-compose: "latest"       # Orchestration
```

### Complete requirements.txt
```
gymnasium>=0.28.0
stable-baselines3>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
mss>=7.0.1
pydirectinput>=1.0.4
pytesseract>=0.3.10
laya>=1.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
websockets>=12.0
rich>=13.0.0
loguru>=0.7.0
line_profiler>=4.0.0
psutil>=5.9.0
jupyter>=1.0.0
pytest>=7.3.0
pytest-cov>=4.1.0
black>=23.0.0
mypy>=1.0.0
```

---

## PROJECT STRUCTURE

```
dino-decision-ai/
│
├── 📂 src/                                    # Main source code
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── dino_env.py                      # Gymnasium environment
│   │   ├── screen_utils.py                  # Screen capture & processing
│   │   └── game_state.py                    # Game state management
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dqn_agent.py                     # DQN implementation
│   │   ├── ppo_agent.py                     # PPO implementation
│   │   └── base_agent.py                    # Base agent class
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                         # Main training loop
│   │   ├── callbacks.py                     # Training callbacks
│   │   ├── hyperparams.py                   # Hyperparameter configs
│   │   ├── parallel_trainer.py              # Run both agents parallel
│   │   └── process_manager.py               # Process management
│   │
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── laya_router.py                   # Laya decision logic
│   │   ├── agent_manager.py                 # Manage both agents
│   │   ├── selector.py                      # Agent selection
│   │   ├── process_manager.py               # Process management
│   │   └── resource_monitor.py              # Monitor resources
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── api.py                           # FastAPI backend
│   │   ├── ws_handler.py                    # WebSocket updates
│   │   └── models.py                        # Pydantic schemas
│   │
│   ├── config/                               # Configuration management
│   │   ├── __init__.py
│   │   ├── base_config.py                   # Base configuration
│   │   ├── local_config.py                  # LOCAL OPTIMIZATION!
│   │   ├── dqn_config.py                    # DQN hyperparameters
│   │   ├── ppo_config.py                    # PPO hyperparameters
│   │   └── docker_config.py                 # Docker configuration
│   │
│   ├── performance/                          # Performance optimization
│   │   ├── __init__.py
│   │   ├── gpu_detector.py                  # GPU/CPU detection
│   │   ├── frame_cache.py                   # Frame caching
│   │   ├── model_cache.py                   # Model cache
│   │   ├── prediction_cache.py              # Prediction cache
│   │   ├── cache_manager.py                 # Unified cache mgmt
│   │   ├── profiler.py                      # Performance profiling
│   │   └── optimization_config.py           # Optimization settings
│   │
│   ├── monitoring/                           # Logging & monitoring
│   │   ├── __init__.py
│   │   ├── logger.py                        # Rich logging setup
│   │   ├── metrics.py                       # Custom metrics
│   │   ├── tensorboard_wrapper.py           # TensorBoard logging
│   │   ├── local_db.py                      # SQLite metrics DB
│   │   └── alerts.py                        # Alert system
│   │
│   ├── models_mgmt/                          # Model management
│   │   ├── __init__.py
│   │   ├── checkpoint_manager.py            # Checkpoint handling
│   │   ├── model_registry.py                # Model versioning
│   │   ├── version_tracker.py               # Version tracking
│   │   └── model_comparison.py              # Compare models
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── frame_cache.py
│   │   ├── model_cache.py
│   │   ├── prediction_cache.py
│   │   └── cache_manager.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py                          # CLI entry point
│   │   └── commands/
│   │       ├── train.py
│   │       ├── eval.py
│   │       ├── profile.py
│   │       ├── dashboard.py
│   │       ├── inspect.py
│   │       └── utils.py
│   │
│   ├── profiling/
│   │   ├── __init__.py
│   │   ├── profiler.py
│   │   ├── memory_tracker.py
│   │   ├── debug_visualizer.py
│   │   └── performance_report.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── visualize.py
│   │   ├── compare.py
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── config.py
│       └── helpers.py
│
├── 📂 frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── components/
│   │   ├── agent_selector.js
│   │   ├── metrics_display.js
│   │   ├── decision_log.js
│   │   └── performance_chart.js
│   └── utils/
│       └── websocket.js
│
├── 📂 models/
│   ├── dqn_best.zip
│   ├── ppo_best.zip
│   ├── laya_config.json
│   ├── v1.0.0/
│   │   ├── dqn.zip
│   │   └── ppo.zip
│   └── logs/
│       ├── training.log
│       └── metrics.db
│
├── 📂 notebooks/
│   ├── 01_environment_test.ipynb
│   ├── 02_training_progress.ipynb
│   ├── 03_agent_comparison.ipynb
│   ├── 04_laya_routing_analysis.ipynb
│   └── 05_performance_analysis.ipynb
│
├── 📂 tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_environment.py
│   ├── test_agents.py
│   ├── test_screen_capture.py
│   ├── test_routing.py
│   ├── test_dashboard.py
│   ├── test_performance.py
│   ├── test_config.py
│   └── test_cli.py
│
├── 📂 docs/
│   ├── ARCHITECTURE.md
│   ├── RL_CONCEPTS.md
│   ├── ENVIRONMENT.md
│   ├── TRAINING.md
│   ├── ROUTING.md
│   ├── DASHBOARD.md
│   ├── PERFORMANCE.md
│   ├── DEPLOYMENT.md
│   └── RESOURCES.md
│
├── pyproject.toml
├── poetry.lock
├── requirements.txt
├── .env.local
├── .env.example
├── .gitignore
├── Makefile
├── docker-compose.yml
├── main.py
├── main_multi_agent.py
├── README.md
└── PLAN.md                                   # This file
```

---

## CLAUDE & CODEX COLLABORATION

### Claude's Role 🧠
- **Planning & Architecture** - Design system, define approach
- **Conceptual Explanation** - Explain DQN, PPO, CNN concepts
- **Problem Solving** - Debug issues, suggest optimizations
- **Documentation** - Write guides, comments, README
- **Research** - Recommend algorithms, libraries, hyperparameters
- **Code Review** - Analyze code, suggest improvements

### Codex's Role 💻
- **Code Generation** - Write boilerplate, implementations
- **Auto-Completion** - Complete function signatures
- **Snippet Generation** - Quick utility functions
- **Repetitive Code** - Handle copy-paste patterns
- **Testing Code** - Generate test functions
- **Refactoring** - Suggest improvements

### Optimal Workflow
```
1. Claude: Design architecture & explain approach
         ↓
2. Codex:  Generate code skeleton
         ↓
3. Claude: Review & explain generated code
         ↓
4. Codex:  Fill in implementations
         ↓
5. Claude: Test, debug, optimize
         ↓
6. Repeat for each module
```

### When to Use Claude
- Stuck on concept
- Want optimization ideas
- Need architecture review
- Debugging complex issues
- Understanding why something doesn't work
- Planning next steps
- Writing documentation

### When to Use Codex
- Need boilerplate code
- Want function generation
- Need repetitive code
- Completing implementations
- Quick utility functions
- Testing code
- Refactoring suggestions

---

## PHASE 1: FOUNDATION & LOCAL OPTIMIZATION

**Duration:** 2.5 weeks | **Time:** 20-26 hours | **Status:** 🔵 Future

### 1.1 Project Setup (2-3 hours)
```
Tasks:
□ Poetry project initialization
□ Dependencies installation (use UV for speed)
□ Virtual environment setup
□ .env configuration
□ Git initialization
□ Makefile creation

Commands:
poetry init
poetry add gymnasium stable-baselines3 torch ...
uv pip install -r requirements.txt
git init
```

### 1.2 Configuration System (4-6 hours) ⭐ PRIORITY 1
```
Create: src/config/

Files:
- base_config.py (base settings)
- local_config.py (macOS optimization) ← AUTO-OPTIMIZE!
- dqn_config.py (DQN hyperparameters)
- ppo_config.py (PPO hyperparameters)
- docker_config.py (Docker setup)

Features (local_config.py):
class LocalConfig:
    USE_GPU = torch.cuda.is_available()
    GPU_MEMORY_GB = get_gpu_memory()
    AVAILABLE_RAM_GB = get_available_memory()
    CPU_CORES = get_cpu_count()
    
    # Auto-sizing
    BATCH_SIZE = 32 if USE_GPU else 16
    NUM_WORKERS = min(CPU_CORES // 2, 4)
    FRAME_CACHE_ENABLED = True
    GPU_PREPROCESSING = USE_GPU

Dependencies: None (pure Python)
```

### 1.3 GPU/CPU Detection & Optimization (3-4 hours) ⭐ PRIORITY 1
```
Create: src/performance/gpu_detector.py

Features:
- Detect GPU availability
- Get GPU memory
- Detect CPU cores
- Get available RAM
- Performance recommendations

Example Output:
✓ GPU Detected: CUDA (RTX 4090, 24GB)
✓ CPU: Apple M3 Max (12 cores)
✓ RAM: 32GB available
✓ Recommended settings:
  - Batch size: 64
  - Num workers: 6
  - Frame cache: Enabled

Dependencies: torch, psutil
```

### 1.4 Logging System (2-3 hours) ⭐ PRIORITY 1
```
Create: src/utils/logger.py

Features:
- Rich logging (pretty output)
- Color-coded messages
- File + console logging
- Log rotation
- Structured format

Usage:
logger = get_logger(__name__)
logger.info("Training started")
logger.debug("GPU memory: 2.3GB")
logger.warning("Low memory warning")
logger.error("Training failed")

Dependencies: rich, loguru
```

### 1.5 Environment Setup (8-10 hours)
```
Create: src/environment/

Files:
- dino_env.py (Gymnasium environment)
- screen_utils.py (Screen capture & processing)
- game_state.py (Game state management)

Features:
- Action space: Discrete(3) [nothing, jump, duck]
- Observation space: Box(84, 84, 1) [4-frame stack]
- Reward: +0.1 per frame, -100 on collision
- Screenshot capture using mss
- Game over detection (OCR or pixel matching)
- Keyboard control via pydirectinput

Deliverable: Working Gymnasium environment

Reference: Original PHASE 1 (Tasks 1.2-1.6)
```

**Phase 1 Total:** 20-26 hours

---

## PHASE 2: CORE RL + PERFORMANCE MONITORING

**Duration:** 3 weeks | **Time:** 22-28 hours | **Status:** 🔵 Future

### 2.1 DQN Agent Implementation (6-8 hours)
```
Create: src/models/dqn_agent.py

Features:
- CNN architecture
- Experience replay buffer
- Target network
- Training logic
- Epsilon-greedy exploration

Hyperparameters:
- Learning Rate: 1e-4
- Buffer Size: 50,000-100,000
- Batch Size: 32
- Gamma: 0.99
- Epsilon Start: 1.0
- Epsilon End: 0.1
- Target Update Frequency: 1000 steps

Reference: Original PHASE 2.2
```

### 2.2 Profiling & Benchmarking (4-5 hours) ⭐ PRIORITY 2
```
Create: src/profiling/

Features:
- FPS measurement
- Memory tracking
- CPU/GPU utilization
- Function timing
- Performance baseline

Usage:
from src.profiling import profiler

@profiler.profile
def step_env():
    pass

# Generates:
# - FPS report
# - Memory profile
# - GPU utilization
# - Bottleneck identification

Dependencies: line_profiler, torch.profiler
```

### 2.3 Metrics Database System (3-4 hours) ⭐ PRIORITY 2
```
Create: src/monitoring/local_db.py

Features:
- SQLite for metric persistence
- Tables: episodes, training, performance, routing
- Write metrics to DB
- Query historical metrics
- Database management

Schema:
- episodes: episode_id, reward, timestamp, agent
- training: step, loss, lr, agent
- performance: fps, memory, gpu_util, timestamp
- routing: decision, confidence, timestamp

Usage:
db = MetricsDB("metrics.db")
db.add_episode(episode=100, reward=850, agent="DQN")
results = db.query_recent_episodes(agent="DQN", limit=10)

Dependencies: sqlite3, sqlalchemy
```

### 2.4 Model Versioning (3-4 hours) ⭐ PRIORITY 2
```
Create: src/models_mgmt/

Files:
- checkpoint_manager.py (save/load/cleanup)
- model_registry.py (track versions)
- version_tracker.py (numbering & changelog)

Features:
- Save best model only
- Keep N recent models
- Auto cleanup old ones
- Metadata with version
- Version comparison

Usage:
manager = CheckpointManager(keep_best_n=5)
manager.save(agent, episode=100, reward=850)
latest = manager.load_best()

Dependencies: None (file-based)
```

### 2.5 CLI Interface - Phase 1 (3-4 hours) ⭐ PRIORITY 2
```
Create: src/cli/main.py

Commands:
- dino-ai train --agent dqn
- dino-ai eval --agent dqn
- dino-ai profile --duration 60

Framework: Click or Typer

Dependencies: click or typer
```

**Phase 2 Total:** 22-28 hours

---

## PHASE 3: CACHING & TESTING

**Duration:** 3 weeks | **Time:** 22-28 hours | **Status:** 🔵 Future

### 3.1 Frame Caching System (5-6 hours) ⭐ PRIORITY 2
```
Create: src/cache/

Files:
- frame_cache.py (frame caching)
- model_cache.py (model paths)
- prediction_cache.py (predictions)
- cache_manager.py (orchestration)

Features:
- Disk cache for large data
- Memory cache for hot data
- LRU eviction policy
- Hash-based lookup
- Hit/miss statistics

Expected Benefit: ~30% memory reduction

Usage:
cache = FrameCache(max_memory=1000, max_disk=10000)
cached = cache.get_or_compute(
    key=frame_hash,
    compute_fn=lambda: preprocess(raw_frame)
)

Dependencies: None (pure Python)
```

### 3.2 PPO Agent Implementation (6-8 hours)
```
Create: src/models/ppo_agent.py

Features:
- Policy network (actor)
- Value network (critic)
- Advantage computation
- Loss with clipping
- Training loop

Reference: Original PHASE 3.2
```

### 3.3 Testing Framework (6-8 hours) ⭐ PRIORITY 3
```
Create: tests/

Files:
- conftest.py (fixtures)
- test_environment.py
- test_agents.py
- test_screen_capture.py
- test_routing.py
- test_dashboard.py

Commands:
pytest tests/
pytest tests/ --cov                # With coverage
pytest tests/ --profile            # With profiling
pytest tests/ -k "test_agent"      # Specific test

Dependencies: pytest, pytest-cov, pytest-xdist
```

### 3.4 CLI Expansion (2-3 hours)
```
Commands:
- dino-ai train --all --profile
- dino-ai eval --compare v1.0.0
- dino-ai profile --duration 60
- dino-ai inspect --model v1.0.0
- dino-ai clean --keep 3
```

**Phase 3 Total:** 22-28 hours

---

## PHASE 4: OPTIMIZATION & COMPARISON

**Duration:** 2 weeks | **Time:** 20-30 hours | **Status:** 🔵 Future

### 4.1 Parallel Training Utilities (4-5 hours)
```
Create: src/training/

Features:
- Run DQN + PPO simultaneously
- Shared environment (save memory)
- Independent experience replay
- Resource monitoring
- Auto batch size adjustment

Usage:
trainer = ParallelTrainer(
    env=env,
    agents=["dqn", "ppo"],
    max_memory_gb=8,
    auto_batch_size=True
)
trainer.train(total_steps=500000)

Dependencies: multiprocessing, threading
```

### 4.2 Performance Profiling (3-4 hours)
```
Create: src/profiling/

Features:
- Line-by-line profiling
- Memory leak detection
- FPS tracking
- GPU util tracking
- Comparison reports

Usage:
profiler.profile_training(
    agent="DQN",
    duration=60,
    report_file="profile_dqn.html"
)
```

### 4.3 DQN vs PPO Comparison (4-6 hours)
```
Train both agents for 250K steps
Compare performance metrics
Analyze differences
Document findings

Reference: Original PHASE 3.3
```

### 4.4 Hyperparameter Optimization (8-10 hours)
```
Tune learning rates
Tune entropy coefficients
Tune GAE lambda
Run ablation studies
Document results

Reference: Original PHASE 3.4
```

**Phase 4 Total:** 20-30 hours

---

## PHASE 5: MULTI-AGENT ROUTING

**Duration:** 1 week | **Time:** 8-12 hours | **Status:** 🔵 Future

### 5.1 Design Laya Router (2-3 hours)
```
Create: src/routing/laya_router.py

Features:
- Input: DQN score, PPO score, difficulty
- Output: Agent selection + confidence
- Uses Laya for classification

Usage:
router = LayaRouter()
agent, confidence = router.decide_agent(
    dqn_score=1850,
    ppo_score=2340,
    difficulty="HARD"
)

Example:
Input: DQN=1850, PPO=2340, Hard
Output: PPO (confidence=0.94)
```

### 5.2 Agent Manager (2-3 hours)
```
Create: src/routing/agent_manager.py

Features:
- Initialize DQN + PPO
- Track agent scores
- Switch agents
- Log switches

Usage:
manager = AgentManager()
manager.initialize(env)
action, agent = manager.get_action(obs)
manager.update_scores(dqn=1850, ppo=2340)
```

### 5.3 Laya Integration (1-2 hours)
```
Install: pip install laya

Features:
- Load Laya model
- Prepare inputs
- Get decisions
- Log decisions

Usage:
from laya import Laya
router = Laya()
decision = router.classify(
    inputs={"dqn": 1850, "ppo": 2340},
    choices=["dqn", "ppo"]
)
```

### 5.4 Integration Testing (2-3 hours)
```
Test routing logic
Test agent switching
Test score tracking
Test decision logging

Tests: test_routing.py
```

**Phase 5 Total:** 8-12 hours

---

## PHASE 6: DASHBOARD UI

**Duration:** 1.5 weeks | **Time:** 12-16 hours | **Status:** 🔵 Future

### 6.1 FastAPI Backend (3-4 hours)
```
Create: src/dashboard/api.py

Endpoints:
- GET /health
- GET /agents/status
- GET /metrics/latest
- GET /metrics/history
- WS /ws

Usage:
uvicorn src.dashboard.api:app --reload

Dependencies: fastapi, uvicorn
```

### 6.2 WebSocket Handler (2-3 hours)
```
Create: src/dashboard/ws_handler.py

Features:
- Real-time updates (100ms intervals)
- Metric streaming
- Decision logging
- Reconnection handling

Usage (JavaScript):
ws = new WebSocket("ws://localhost:8000/ws")
ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  updateDashboard(data)
}

Dependencies: websockets
```

### 6.3 Frontend UI (4-6 hours)
```
Create: frontend/

Files:
- index.html
- style.css
- app.js
- components/

Features:
- Live agent switching
- Metric display
- Decision log
- Performance charts
- Real-time updates

No external dependencies (vanilla JS)
```

### 6.4 Integration & Testing (2-3 hours)
```
Connect API to training
Connect WebSocket to metrics
Test real-time updates
Test performance impact
Load testing

Tests: test_dashboard.py
```

**Phase 6 Total:** 12-16 hours

---

## PHASE 7: TRAINING & EVALUATION

**Duration:** 2 weeks | **Time:** 27-32 hours | **Status:** 🔵 Future

### 7.1 Long-Running Training (20+ hours)
```
Train DQN to 500K+ steps
Train PPO to 500K+ steps
Monitor metrics
Profile performance
Save checkpoints

Reference: Original PHASE 3.6
```

### 7.2 Comprehensive Evaluation (4-5 hours)
```
Evaluate both agents (100+ episodes)
Collect statistics
Generate report
Compare with Laya routing

Reference: Original PHASE 4
```

### 7.3 Analysis & Visualization (3-4 hours)
```
Create learning curves
Create reward distributions
Create action frequency charts
Generate comparison report
Save visualizations

Tools: matplotlib, seaborn
```

**Phase 7 Total:** 27-32 hours

---

## PHASE 8: DOCUMENTATION & DEPLOYMENT

**Duration:** 1.5 weeks | **Time:** 7-10 hours | **Status:** 🔵 Future

### 8.1 Code Documentation (4-5 hours)
```
Docstrings for all functions
Type hints throughout
README.md
Architecture documentation
Setup guide

Tools: mkdocs, sphinx (optional)
```

### 8.2 Deployment Setup (2-3 hours)
```
Docker containerization
Docker Compose setup
Environment configuration
Deployment guide
Production checklist

Files:
- Dockerfile
- docker-compose.yml
- .env.example
```

### 8.3 GitHub Release (1-2 hours)
```
Initialize git
Create .gitignore
First commit
Push to GitHub
Create README
Create release notes
```

**Phase 8 Total:** 7-10 hours

---

## CORE 10 COMPONENTS

### ✅ MUST HAVE (Implement These)

1. **Configuration Management System** (4-6h)
   - Location: `src/config/`
   - Auto-optimizes for local machine
   - Separate configs for DQN/PPO/Docker

2. **GPU/CPU Detection & Optimization** (3-4h)
   - Location: `src/performance/gpu_detector.py`
   - Auto-select best device
   - Recommend settings

3. **Performance Profiling** (4-5h)
   - Location: `src/profiling/`
   - Identify bottlenecks
   - FPS tracking
   - Memory profiling

4. **Logging System** (2-3h)
   - Location: `src/utils/logger.py`
   - Production-grade logs
   - Pretty output (Rich)

5. **Checkpoint Management** (3-4h)
   - Location: `src/models_mgmt/checkpoint_manager.py`
   - Save/load/cleanup models
   - Keep N recent versions

6. **CLI Interface** (5-7h)
   - Location: `src/cli/`
   - 10+ commands
   - Easy operation

7. **Testing Framework** (6-8h)
   - Location: `tests/`
   - 8 test modules
   - pytest setup

8. **Frame Caching** (5-6h)
   - Location: `src/cache/frame_cache.py`
   - ~30% memory reduction
   - LRU eviction

9. **Metrics Database** (3-4h)
   - Location: `src/monitoring/local_db.py`
   - Persistent metric storage
   - Historical tracking

10. **Model Versioning** (3-4h)
    - Location: `src/models_mgmt/`
    - Track model versions
    - Metadata storage

**Total Core Time:** 39-51 hours

### 🎁 SUPPORTING COMPONENTS (Add If Time)

11. **Parallel Training** (4-5h)
12. **Laya Router** (4-5h)
13. **FastAPI Backend** (3-4h)
14. **WebSocket Handler** (2-3h)
15. **Frontend Dashboard** (4-6h)
16. **Memory Tracking** (2-3h)
17. **Performance Reports** (2-3h)
18. **Documentation** (4-5h)

**Total Supporting Time:** 27-37 hours

---

## COMPLETE TIMELINE

| Phase | Content | Duration | Hours | Status |
|-------|---------|----------|-------|--------|
| 1 | Foundation + Config + GPU + Logging | 2.5w | 20-26h | 🔵 Future |
| 2 | DQN + Profiling + DB + Versioning | 3w | 22-28h | 🔵 Future |
| 3 | Caching + PPO + Testing + CLI | 3w | 22-28h | 🔵 Future |
| 4 | Parallel Training + Optimization | 2w | 20-30h | 🔵 Future |
| 5 | Laya Router + Multi-Agent | 1w | 8-12h | 🔵 Future |
| 6 | Dashboard UI + WebSocket | 1.5w | 12-16h | 🔵 Future |
| 7 | Extended Training + Evaluation | 2w | 27-32h | 🔵 Future |
| 8 | Documentation + Deployment | 1.5w | 7-10h | 🔵 Future |
| **TOTAL** | | **12-13w** | **139-182h** | 🔵 Future |

---

## SUCCESS METRICS

### MVP Deliverables (Week 6)
- ✅ Working Gymnasium environment
- ✅ Both DQN and PPO training
- ✅ Metrics database with history
- ✅ CLI for basic commands
- ✅ Test coverage >80%

### Full Project Deliverables (Week 13)
- ✅ All MVP features +
- ✅ Laya-based routing
- ✅ Real-time dashboard
- ✅ Performance optimized (30%+ improvement)
- ✅ Complete documentation
- ✅ GitHub repository
- ✅ Docker setup
- ✅ Production-ready

### Specific Goals
| Metric | Target | Success |
|--------|--------|---------|
| Test Coverage | >80% | ✅ |
| Memory Optimization | -30% | ✅ |
| Agent Training Time | <2h per 100K steps | ✅ |
| Dashboard Latency | <100ms | ✅ |
| Code Documentation | 100% | ✅ |

---

## DEVELOPMENT WORKFLOW

### Quick Start
```bash
# Setup
make setup              # Install deps with Poetry + UV
make init              # Initialize project

# Development
make train             # Train both agents
make profile           # Profile performance
make test              # Run tests
make dashboard         # Start UI (localhost:8000)

# Maintenance
make clean             # Clean checkpoints
make lint              # Code quality
make format            # Auto-format code
```

### CLI Commands
```bash
# Training
dino-ai train --agent dqn                   # Train DQN
dino-ai train --agent ppo                   # Train PPO
dino-ai train --all --profile               # Both + profiling
dino-ai train --dev                         # Development mode

# Evaluation
dino-ai eval --agent dqn                    # Evaluate DQN
dino-ai eval --compare v1.0.0               # Compare versions

# Profiling
dino-ai profile --duration 60               # Profile 60s
dino-ai profile --compare v1.0.0 v1.0.1    # Compare versions

# Dashboard
dino-ai dashboard                           # Start UI
dino-ai dashboard --port 8080               # Custom port

# Inspection
dino-ai inspect --model dqn                 # Model info
dino-ai inspect --metrics --days 7          # Last 7 days metrics

# Cleanup
dino-ai clean --keep 3                      # Keep 3 checkpoints
```

### Development Mode
```bash
# Fast iteration
dino-ai train --dev
# - Hot reload on code changes
# - Verbose logging
# - Debug mode enabled
# - Lower batch sizes (faster iteration)
```

---

## RL CONCEPTS GUIDE

### Deep Q-Network (DQN)
```
Q(state, action) = expected future reward

Training Loop:
1. Take action a in state s
2. Observe reward r and next state s'
3. Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
4. Use neural network to approximate Q-function
5. Use experience replay for stability

Key Features:
- Experience replay (store & batch experiences)
- Target network (stabilize training)
- Epsilon-greedy exploration (balance exploration/exploitation)

Pros: Sample efficient, well-tested
Cons: Discrete action space, slower convergence
```

### Proximal Policy Optimization (PPO)
```
Directly learn policy π(action|state)
Advantage = actual reward - expected reward
Loss = -log π(a|s) * advantage (clipped to ±ε)

Key Features:
- On-policy (learns from current policy)
- Clipped objective (prevents large updates)
- Entropy bonus (encourage exploration)
- Advantage estimation (reduce variance)

Pros: More stable, better convergence
Cons: Data hungry, requires tuning
```

### CNN Architecture
```
Input: 84x84x4 (4 stacked frames)
  ↓
Conv2d(32 filters, 8x8, stride 4) + ReLU
  ↓
Conv2d(64 filters, 4x4, stride 2) + ReLU
  ↓
Conv2d(64 filters, 3x3, stride 1) + ReLU
  ↓
Flatten
  ↓
Dense(512) + ReLU
  ↓
Output: 3 action values (DQN) or policy (PPO)
```

### Laya Model for Agent Routing
```
Input: DQN score, PPO score, difficulty
↓
Laya Model Classifier
↓
Output: Best agent (DQN or PPO) + confidence

Example:
Input: DQN=1850, PPO=2340, Hard
Output: PPO (confidence=0.94)

Use case: Automatically pick best agent in real-time
```

---

## RESOURCES & REFERENCES

### Core RL Papers
- "Playing Atari with Deep Reinforcement Learning" (DQN)
  https://arxiv.org/abs/1312.5602

- "Proximal Policy Optimization Algorithms" (PPO)
  https://arxiv.org/abs/1707.06347

- "Dueling Network Architectures for Deep RL"
  https://arxiv.org/abs/1511.06581

### Key Libraries
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyTorch](https://pytorch.org/)
- [Laya Model](https://laya-ai.com/)
- [FastAPI](https://fastapi.tiangolo.com/)

### Learning Resources
- [DeepMind RL Course](https://www.deepmind.com/learning-resources)
- [Hugging Face RL Course](https://huggingface.co/course/en/chapter1/1)
- [Stanford CS221 - AI](http://web.stanford.edu/class/cs221/)

### Communities
- r/reinforcementlearning
- r/MachineLearning
- Hugging Face Discord
- Local LLM communities

---

## IMPLEMENTATION TIPS

### Start with Priority 1 (Weeks 1-2)
```
✅ Config system (easy, high ROI)
✅ GPU detection (essential)
✅ Logging (essential)
✅ Environment (from original plan)
```

### Add Priority 2 (Weeks 3-4)
```
✅ Profiling (reveals bottlenecks)
✅ Metrics DB (enables optimization)
✅ CLI (improves developer experience)
```

### Add Priority 3 (Weeks 5-6)
```
✅ Testing (ensures quality)
✅ Caching (improves performance)
```

### Build Interesting Parts (Weeks 7-12)
```
✅ DQN/PPO training
✅ Laya routing
✅ Dashboard UI
✅ Analysis & visualization
```

---

## FUTURE ENHANCEMENTS

These can be added after v1.0:

- [ ] RL training curriculum
- [ ] Transfer learning
- [ ] Multi-game support
- [ ] Cloud deployment
- [ ] Mobile dashboard
- [ ] Advanced visualization
- [ ] Experiment tracking (W&B)
- [ ] Hyperparameter optimization (Ray Tune)
- [ ] Model ensemble
- [ ] Distributed training
- [ ] Model quantization
- [ ] Edge deployment

---

**This is your comprehensive roadmap for Chrome Dino AI!**

✅ **8 phases with detailed breakdown**  
✅ **18 components documented**  
✅ **12-13 week timeline**  
✅ **66-88 hours of work**  
✅ **All code examples included**  
✅ **All dependencies listed**  

**Ready to build! 🚀🦖🤖**

---

**Document Version:** 2.0 Complete  
**Status:** Ready for Implementation  
**Last Updated:** 2026-09-24  
**Platform:** macOS + Local + Poetry + UV
