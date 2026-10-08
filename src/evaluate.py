"""Deterministic evaluation, isolated from training."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from src.networks import Actor


@dataclass
class EvalResult:
    returns: list[float]  # undiscounted return of each evaluation episode
    mean_return: float


def evaluate(
    actor: Actor,
    env: gym.Env,
    num_episodes: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> EvalResult:
    """Run `num_episodes` with the deterministic (squashed-mean) action.

    Episode i is reset with seed `seed + i`, so repeated evaluations of the same
    policy see the same start states. Touches no optimizer, no replay buffer and
    no random number stream of the actor (no sampling); `env` is owned (and
    closed) by the caller.
    """
    device = torch.device(device)
    returns: list[float] = []
    with torch.no_grad():
        for i in range(num_episodes):
            obs, _ = env.reset(seed=seed + i)
            episode_return, done = 0.0, False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor.get_deterministic_action(obs_t)[0].cpu().numpy()
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                done = terminated or truncated
            returns.append(episode_return)
    return EvalResult(returns=returns, mean_return=float(np.mean(returns)))
