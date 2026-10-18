"""Baseline SAC training loop and launcher.

Control flow follows CleanRL's `sac_continuous_action.py` (single env, 0-indexed
global_step), see the comments next to each step.

CLI:  python -m src.train --config configs/baseline.json [--seed N] [--device cuda]
      [--total-steps N] [--output-dir DIR] [--run-id ID]
"""

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from src.checkpoint import save_checkpoint
from src.envs import EVAL_SEED_OFFSET, make_env, set_global_seed
from src.evaluate import EvalResult, evaluate
from src.networks import HIDDEN_DEPTH, HIDDEN_SIZE, _check_residual_connections
from src.replay_buffer import ReplayBuffer
from src.run_logger import RunLogger
from src.run_metadata import count_parameters
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
    hidden_width: int = HIDDEN_SIZE  # width of every hidden layer of actor and critics
    hidden_depth: int = HIDDEN_DEPTH  # number of hidden layers of actor and critics
    eval_interval: int = 25_000  # project protocol; 0 disables periodic evaluation
    eval_episodes: int = 5  # project protocol
    final_eval_episodes: int = 10  # project protocol; 0 disables the final evaluation
    # --- run recording (do not affect learning) ---
    output_dir: str = "results"
    run_id: str | None = None  # None -> generated
    log_interval: int = 1_000  # train-metric row every log_interval env steps; 0 disables
    checkpoint_interval: int = 0  # periodic checkpoint every N steps; 0 = final only
    tensorboard: bool = True
    input_norm: bool = False  # appended: preserve historical positional config arguments
    residual_connections: bool = False  # appended after all historical fields

    def __post_init__(self) -> None:
        _check_residual_connections(self.residual_connections)


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
    final_eval: EvalResult | None = None
    agent: SACAgent | None = None
    replay_buffer: ReplayBuffer | None = None


def train(
    config: TrainConfig,
    env_fn: Callable[[int], gym.Env] | None = None,
    logger: RunLogger | None = None,
) -> TrainResult:
    """Run SAC for `config.total_steps` environment steps.

    `env_fn(seed)` must return a fresh env with its action space seeded; the default
    builds `config.env_id`. It is called twice: a training env (seed) and a separate
    evaluation env (seed + EVAL_SEED_OFFSET). `logger` is optional; without it nothing
    is written to disk.
    """
    if env_fn is None:
        env_fn = lambda seed: make_env(config.env_id, seed)

    set_global_seed(config.seed)
    eval_seed = config.seed + EVAL_SEED_OFFSET
    train_env = env_fn(config.seed)
    eval_env = env_fn(eval_seed)
    try:
        return _train(config, train_env, eval_env, eval_seed, logger)
    finally:
        train_env.close()
        eval_env.close()


def _train(
    config: TrainConfig,
    train_env: gym.Env,
    eval_env: gym.Env,
    eval_seed: int,
    logger: RunLogger | None,
) -> TrainResult:
    device = torch.device(config.device)
    obs_space, act_space = train_env.observation_space, train_env.action_space
    agent = SACAgent(
        obs_space, act_space, device, config.gamma, config.tau, config.policy_lr, config.q_lr,
        config.hidden_width, config.hidden_depth,
        input_norm=config.input_norm,
        residual_connections=config.residual_connections,
    )
    rb = ReplayBuffer(
        config.buffer_size,
        int(np.prod(obs_space.shape)),
        int(np.prod(act_space.shape)),
        device,
        seed=config.seed,
    )
    result = TrainResult(agent=agent, replay_buffer=rb)
    normalizer = agent.observation_normalizer
    if logger is not None:
        logger.update_metadata(
            parameter_counts=count_parameters(agent),
            observation_shape=list(obs_space.shape),
            action_shape=list(act_space.shape),
        )
        if normalizer is not None:
            logger.update_metadata(input_normalization=normalizer.summary())

    obs, _ = train_env.reset(seed=config.seed)  # explicit seed on the first reset only
    episode_return, episode_length = 0.0, 0

    for global_step in range(config.total_steps):
        steps_completed = global_step + 1  # every logged `step` means "env steps completed"
        if normalizer is not None:
            normalizer.update(obs)  # exactly once per decision state, including random warm-up

        # 1. Action: random during warm-up, policy afterwards (CleanRL: step < learning_starts).
        if global_step < config.learning_starts:
            action = act_space.sample()
            result.num_random_actions += 1
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = agent.actor.get_action(agent.normalize_observation(obs_t))[0][0].cpu().numpy()
            result.num_policy_actions += 1

        # 2. Step. Non-vectorized Gymnasium returns the TRUE next observation even on
        #    termination/truncation (no autoreset), so next_obs is stored as-is.
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        rb.add(obs, next_obs, action, reward, terminated, truncated)
        episode_return += float(reward)
        episode_length += 1

        # 3. Episode end on termination OR truncation -> reset (unseeded, continues the RNG stream).
        #    Only `terminated` stops bootstrapping, and that happens in SACAgent.critic_loss.
        if terminated or truncated:
            result.episode_returns.append(episode_return)
            result.episode_end_steps.append(global_step)
            if logger is not None:
                logger.log_episode(steps_completed, episode_return, episode_length)
            episode_return, episode_length = 0.0, 0
            obs, _ = train_env.reset()
        else:
            obs = next_obs

        # 4. Updates (CleanRL: global_step > learning_starts, strictly greater).
        if global_step > config.learning_starts:
            # Diagnostics (grad norms, |Q|) are computed only on logging steps.
            log_now = logger is not None and config.log_interval > 0 and global_step % config.log_interval == 0
            batch = rb.sample(config.batch_size)
            critic_loss = agent.update_critic(batch, diagnostics=log_now)
            result.num_critic_updates += 1

            # Delayed actor update with compensation, exactly as CleanRL: every
            # `policy_frequency` steps do `policy_frequency` actor(+alpha) updates on
            # the same batch, so the long-run ratio stays 1 actor update per step.
            actor_loss = alpha_loss = None
            if global_step % config.policy_frequency == 0:
                for _ in range(config.policy_frequency):
                    actor_loss, alpha_loss = agent.update_actor_and_alpha(batch, diagnostics=log_now)
                    result.num_actor_updates += 1

            if global_step % config.target_network_frequency == 0:
                agent.update_targets()
                result.num_target_updates += 1

            if log_now:
                d = agent.diagnostics
                logger.log_train(
                    steps_completed,
                    {
                        "critic_loss": critic_loss,
                        "actor_loss": actor_loss,  # blank when no actor update on this step
                        "alpha_loss": alpha_loss,
                        "alpha": agent.alpha.item(),
                        "q_mean_abs": d.get("q_mean_abs"),
                        "actor_grad_norm": d.get("actor_grad_norm"),
                        "critic_grad_norm": d.get("critic_grad_norm"),
                    },
                )
                d.clear()

        result.steps_completed = steps_completed
        if logger is not None:
            logger.note_step(steps_completed)

        # 5. Periodic deterministic evaluation on the separate eval env.
        if config.eval_interval > 0 and steps_completed % config.eval_interval == 0:
            eval_kwargs = {} if normalizer is None else {"observation_transform": normalizer.snapshot()}
            eval_result = evaluate(agent.actor, eval_env, config.eval_episodes, eval_seed, device, **eval_kwargs)
            result.eval_results.append((steps_completed, eval_result))
            if logger is not None:
                logger.log_eval(steps_completed, eval_result, "periodic")

        # 6. Periodic checkpoint (latest one overwrites the previous).
        if logger is not None and config.checkpoint_interval > 0 and steps_completed % config.checkpoint_interval == 0:
            save_checkpoint(logger.checkpoint_path, agent, steps_completed, config, config.seed)

    # Final checkpoint first (so it exists even if the final evaluation fails), then final evaluation.
    if logger is not None:
        save_checkpoint(logger.checkpoint_path, agent, result.steps_completed, config, config.seed)
    if config.final_eval_episodes > 0:
        eval_kwargs = {} if normalizer is None else {"observation_transform": normalizer.snapshot()}
        result.final_eval = evaluate(agent.actor, eval_env, config.final_eval_episodes, eval_seed, device, **eval_kwargs)
        if logger is not None:
            logger.log_eval(result.steps_completed, result.final_eval, "final")

    if normalizer is not None and logger is not None:
        logger.update_metadata(input_normalization=normalizer.summary())

    return result


def run_experiment(
    config: TrainConfig,
    env_fn: Callable[[int], gym.Env] | None = None,
    verbose: bool = False,
) -> TrainResult:
    """Create the run directory, train, and record status. On any exception the run is
    marked failed (type + message), already-written metrics are kept, and the exception
    is re-raised."""
    logger = RunLogger(config, verbose=verbose)
    if verbose:
        print(f"Run directory: {logger.run_dir}", flush=True)
    try:
        result = train(config, env_fn, logger)
    except BaseException as exc:
        logger.finish("failed", exc)
        raise
    else:
        logger.finish("completed")
        if verbose:
            print(f"Run completed: {logger.run_dir}", flush=True)
        return result
    finally:
        logger.close()


def load_config(path: str) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return TrainConfig(**data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline SAC from a JSON config.")
    p.add_argument("--config", required=True, help="path to a JSON config, e.g. configs/baseline.json")
    p.add_argument("--seed", type=int)
    p.add_argument("--device", help='e.g. "cpu" or "cuda" (GPU chosen via CUDA_VISIBLE_DEVICES)')
    p.add_argument("--total-steps", type=int)
    p.add_argument("--output-dir")
    p.add_argument("--run-id")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    overrides = {
        "seed": args.seed,
        "device": args.device,
        "total_steps": args.total_steps,
        "output_dir": args.output_dir,
        "run_id": args.run_id,
    }
    config = dataclasses.replace(config, **{k: v for k, v in overrides.items() if v is not None})
    run_experiment(config, verbose=True)


if __name__ == "__main__":
    main()
