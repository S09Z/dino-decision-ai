"""Evaluate a policy over whole episodes and summarise the result"""

from typing import Optional

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from src.routing.laya_player import kind

TOP_SPEED = 13  # the game's MAX_SPEED: from there on it gets no harder
STOPPED = "nothing (stopped)"  # hit of a game stopped at max_length


class RandomPolicy:
    """Uniform random actions, with the predict() signature evaluate_policy uses"""

    def __init__(self, env):
        self.env = env

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        return np.array([self.env.action_space.sample() for _ in obs]), state


def evaluate(policy, env, episodes: int) -> tuple[list[float], list[int]]:
    """Per-episode rewards and lengths over `episodes` episodes"""
    rewards, lengths = evaluate_policy(
        policy, env, n_eval_episodes=episodes, return_episode_rewards=True
    )
    return list(rewards), list(lengths)  # type: ignore[arg-type]


def evaluate_games(
    policy, env, episodes: int, max_length: Optional[int] = None
) -> list[dict]:
    """Like evaluate() on a one-game env, but also how each game ended: its
    score, the speed, what the dinosaur hit and whether it was in the air.
    A game still going after `max_length` steps is stopped there, with hit
    "nothing (stopped)", so one very long game cannot take all the time.
    `top_score` is the score when the game reached top speed (None if it
    did not), for projected_score()."""
    games: list[dict] = []
    obs = env.reset()
    length, reward, top_score = 0, 0.0, None
    while len(games) < episodes:
        action, _ = policy.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        length, reward = length + 1, reward + float(rewards[0])
        info = infos[0]  # the game state after this step (at a crash: then)
        if top_score is None and info["speed"] >= TOP_SPEED:
            top_score = float(info["score"])
        stopped = not dones[0] and length == max_length
        if dones[0] or stopped:
            hit = info["obstacles"][0] if info["obstacles"] else None
            games.append(
                {
                    "reward": reward,
                    "length": length,
                    "score": float(info["score"]),
                    "top_score": top_score,
                    "speed": float(info["speed"]),
                    # kind() is None for a bird above a running dinosaur
                    "hit": (
                        STOPPED
                        if stopped
                        else (kind(hit) or "high bird") if hit else "unknown"
                    ),
                    "in_air": bool(info["jumping"]),
                }
            )
            if stopped:
                obs = env.reset()
            length, reward, top_score = 0, 0.0, None
    return games


def projected_score(games: list[dict]) -> Optional[float]:
    """The mean score these games would have reached without a length cap.
    Past top speed the game no longer gets harder, so a death is assumed
    equally likely at every point there: points per death at top speed =
    points played there / deaths there (stopped games count their points,
    not a death). None if nothing died at top speed (no estimate yet)."""
    top = [g for g in games if g["top_score"] is not None]
    deaths = sum(g["hit"] != STOPPED for g in top)
    if top and not deaths:
        return None
    projected = [g["score"] for g in games if g["top_score"] is None]
    if top:
        per_death = sum(g["score"] - g["top_score"] for g in top) / deaths
        projected += [g["top_score"] + per_death for g in top]
    return float(np.mean(projected))


def evaluate_routed(manager, env, episodes: int) -> tuple[list[float], list[int]]:
    """Like evaluate(), with an AgentManager choosing who plays each stretch"""
    rewards: list[float] = []
    lengths: list[int] = []
    obs = env.reset()
    while len(rewards) < episodes:
        action, _ = manager.get_action(obs)
        obs, _, dones, infos = env.step(action)
        manager.observe(infos[0], bool(dones[0]))
        if dones[0]:
            rewards.append(float(infos[0]["episode"]["r"]))  # from Monitor
            lengths.append(int(infos[0]["episode"]["l"]))
    return rewards, lengths


def report(name: str, rewards: list[float], lengths: list[int]) -> str:
    return (
        f"{name}: reward {np.mean(rewards):.1f} ± {np.std(rewards):.1f}, "
        f"length {np.mean(lengths):.0f} ± {np.std(lengths):.0f} steps"
    )
