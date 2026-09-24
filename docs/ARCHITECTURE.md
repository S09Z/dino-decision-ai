# System Architecture

## Overview

This project implements a multi-agent reinforcement learning system for the Chrome Dinosaur game.

## Key Components

### Environment
- `src/environment/dino_env.py` - Gymnasium environment

### Agents
- `src/models/dqn_agent.py` - Deep Q-Network agent
- `src/models/ppo_agent.py` - Proximal Policy Optimization agent

### Training
- `src/training/train.py` - Training pipelines
- `src/training/parallel_trainer.py` - Run both agents in parallel

### Routing
- `src/routing/laya_router.py` - Laya-based agent selection
- `src/routing/agent_manager.py` - Manage both agents

### Dashboard
- `src/dashboard/api.py` - FastAPI backend
- `frontend/` - Web UI

### Supporting Systems
- `src/config/` - Configuration management
- `src/performance/` - GPU detection & optimization
- `src/monitoring/` - Logging & metrics
- `src/cache/` - Caching system
- `src/cli/` - Command-line interface

## Data Flow

```
Game Screen
    ↓
Screen Capture & Preprocessing
    ↓
DQN Agent → Predictions
PPO Agent → Predictions
    ↓
Laya Router (Agent Selection)
    ↓
Execute Action
    ↓
Collect Reward & Next State
    ↓
Dashboard (Real-time Visualization)
```
