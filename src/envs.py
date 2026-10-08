"""Seeding and environment-construction helpers."""

import random

import gymnasium as gym
import numpy as np
import torch

# Evaluation uses its own seed stream: eval_seed = seed + EVAL_SEED_OFFSET.
EVAL_SEED_OFFSET = 10_000


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU and (if present) all CUDA devices.

    We do NOT call torch.use_deterministic_algorithms(True): it needs extra CUDA
    configuration (CUBLAS_WORKSPACE_CONFIG), can raise on some ops and can slow
    GPU runs. Seeds therefore give statistical, not bit-for-bit, reproducibility
    on GPU. Environment resets, the action space and the replay buffer are seeded
    separately by the code that owns them (see train.py / evaluate.py).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int) -> gym.Env:
    """Create one non-vectorized env (no rendering) with its action space seeded.

    The caller must still reset it with an explicit seed: env.reset(seed=...).
    """
    env = gym.make(env_id)
    env.action_space.seed(seed)
    return env
