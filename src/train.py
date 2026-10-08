"""Baseline SAC training loop (skeleton: no logging, checkpointing or CLI yet).

Control flow follows CleanRL's `sac_continuous_action.py` (single env, 0-indexed
global_step), see the comments next to each step.
"""

from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from src.envs import EVAL_SEED_OFFSET, make_env, set_global_seed
from src.evaluate import EvalResult, evaluate
from src.replay_buffer import ReplayBuffer
from src.sac_update import SACAgent


@dataclass
class TrainConfig:
    """Defaults are CleanRL's SAC continuous-action defaults, except where noted."""

    env_id: str = "HalfCheetah-v5"
    seed: int = 0  # project seeds are 0, 1, 2 (CleanRL's own default is 1)
    device: str = "cpu"  # e.g. "cuda"; the GPU index comes from CUDA_VISIBLE_DEVICES
    total_steps: int = 1_000_000  # CleanRL total_timesteps
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000
    policy_frequency: int = 2
    target_network_frequency: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    eval_interval: int = 25_000  # project protocol; 0 disables evaluation
    eval_episodes: int = 5  # project protocol (final evaluation uses 10)


@dataclass
class TrainResult:
    steps_completed: int = 0
    num_random_actions: int = 0
    num_policy_actions: int = 0
    num_critic_updates: int = 0
    num_actor_updates: int = 0
    num_target_updates: int = 0
    episode_returns: list[float] = field(default_factory=list)
    episode_end_steps: list[int] = field(default_factory=list)  # global_step of each episode end
    eval_results: list[tuple[int, EvalResult]] = field(default_factory=list)  # (steps completed, result)
    agent: SACAgent | None = None
    replay_buffer: ReplayBuffer | None = None


def train(config: TrainConfig, env_fn: Callable[[int], gym.Env] | None = None) -> TrainResult:
    """Run SAC for `config.total_steps` environment steps.

    `env_fn(seed)` must return a fresh env with its action space seeded; the default
    builds `config.env_id`. It is called twice: a training env (seed) and a separate
    evaluation env (seed + EVAL_SEED_OFFSET).
    """
    if env_fn is None:
        env_fn = lambda seed: make_env(config.env_id, seed)

    set_global_seed(config.seed)
    eval_seed = config.seed + EVAL_SEED_OFFSET
    train_env = env_fn(config.seed)
    eval_env = env_fn(eval_seed)
    try:
        return _train(config, train_env, eval_env, eval_seed)
    finally:
        train_env.close()
        eval_env.close()


def _train(config: TrainConfig, train_env: gym.Env, eval_env: gym.Env, eval_seed: int) -> TrainResult:
    device = torch.device(config.device)
    obs_space, act_space = train_env.observation_space, train_env.action_space
    agent = SACAgent(
        obs_space, act_space, device, config.gamma, config.tau, config.policy_lr, config.q_lr
    )
    rb = ReplayBuffer(
        config.buffer_size,
        int(np.prod(obs_space.shape)),
        int(np.prod(act_space.shape)),
        device,
        seed=config.seed,
    )
    result = TrainResult(agent=agent, replay_buffer=rb)

    obs, _ = train_env.reset(seed=config.seed)  # explicit seed on the first reset only
    episode_return = 0.0

    for global_step in range(config.total_steps):
        # 1. Action: random during warm-up, policy afterwards (CleanRL: step < learning_starts).
        if global_step < config.learning_starts:
            action = act_space.sample()
            result.num_random_actions += 1
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = agent.actor.get_action(obs_t)[0][0].cpu().numpy()
            result.num_policy_actions += 1

        # 2. Step. Non-vectorized Gymnasium returns the TRUE next observation even on
        #    termination/truncation (no autoreset), so next_obs is stored as-is.
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        rb.add(obs, next_obs, action, reward, terminated, truncated)
        episode_return += float(reward)

        # 3. Episode end on termination OR truncation -> reset (unseeded, continues the RNG stream).
        #    Only `terminated` stops bootstrapping, and that happens in SACAgent.critic_loss.
        if terminated or truncated:
            result.episode_returns.append(episode_return)
            result.episode_end_steps.append(global_step)
            episode_return = 0.0
            obs, _ = train_env.reset()
        else:
            obs = next_obs

        # 4. Updates (CleanRL: global_step > learning_starts, strictly greater).
        if global_step > config.learning_starts:
            batch = rb.sample(config.batch_size)
            agent.update_critic(batch)
            result.num_critic_updates += 1

            # Delayed actor update with compensation, exactly as CleanRL: every
            # `policy_frequency` steps do `policy_frequency` actor(+alpha) updates on
            # the same batch, so the long-run ratio stays 1 actor update per step.
            if global_step % config.policy_frequency == 0:
                for _ in range(config.policy_frequency):
                    agent.update_actor_and_alpha(batch)
                    result.num_actor_updates += 1

            if global_step % config.target_network_frequency == 0:
                agent.update_targets()
                result.num_target_updates += 1

        result.steps_completed = global_step + 1

        # 5. Periodic deterministic evaluation on the separate eval env.
        if config.eval_interval > 0 and result.steps_completed % config.eval_interval == 0:
            eval_result = evaluate(agent.actor, eval_env, config.eval_episodes, eval_seed, device)
            result.eval_results.append((result.steps_completed, eval_result))

    return result
