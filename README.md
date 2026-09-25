# 🦖 Chrome Dino AI

Autonomous Multi-Agent RL System for Google Chrome Dinosaur Game with Laya Router & Local Dashboard

## Quick Start

### 1. Setup
```bash
make setup
```

### 2. Install Dependencies
```bash
make install
```

### 3. Train Agents
```bash
make train
```

### 4. Start Dashboard
```bash
make dashboard
```

Open http://localhost:8000 in your browser.

## Commands

- `make train` - Train the DQN agent (PPO arrives in Phase 3.2)
- `make eval` - Evaluate the best DQN checkpoint
- `make profile` - Profile performance
- `make dashboard` - Start real-time dashboard
- `make test` - Run all tests
- `make clean` - Clean up files

## Project Structure

```
src/
├── environment/      # Game environment
├── models/          # DQN & PPO agents
├── training/        # Training pipelines
├── routing/         # Laya agent routing
├── dashboard/       # FastAPI backend
├── config/          # Configuration
├── performance/     # Performance optimization
├── monitoring/      # Logging & metrics
└── cli/            # CLI interface

frontend/           # Web dashboard
tests/             # Test suite
docs/              # Documentation
```

## Features

- ✅ DQN & PPO agents learning to play Chrome Dinosaur
- ✅ Laya Model for intelligent agent routing
- ✅ Real-time performance dashboard
- ✅ GPU/CPU auto-optimization
- ✅ Frame caching & memory optimization
- ✅ Comprehensive testing & profiling
- ✅ Model versioning & checkpoints

## See Also

- `PLAN.md` - Complete development plan
- `docs/` - Additional documentation
