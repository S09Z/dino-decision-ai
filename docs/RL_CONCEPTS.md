# RL Concepts Guide

## Deep Q-Network (DQN)

Q(state, action) = expected future reward

### Training Loop
1. Take action a in state s
2. Observe reward r and next state s'
3. Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

### Key Features
- Experience replay (stability)
- Target network (stabilization)
- Epsilon-greedy exploration

## Proximal Policy Optimization (PPO)

Directly learn policy π(action|state)

### Key Features
- Clipped objective (prevents large updates)
- Entropy bonus (exploration)
- Advantage estimation (variance reduction)

## Laya Model

System 1 decision model for intelligent agent routing.

Input: Agent scores, difficulty
Output: Best agent (DQN or PPO) + confidence
