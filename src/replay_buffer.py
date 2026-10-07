"""Fixed-capacity circular replay buffer for off-policy SAC (single environment)."""

from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    """A sampled minibatch. All tensors are float32 on the buffer's device.

    Shapes: obs/next_obs (B, obs_dim), actions (B, act_dim), the rest (B, 1).
    There is deliberately no combined `done` field; see SACAgent.critic_loss.
    """

    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor  # env reached a true terminal state
    truncations: torch.Tensor  # episode cut by a time limit (not a real terminal)


class ReplayBuffer:
    """Stores transitions in a ring: once full, the oldest ones are overwritten."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.terminations = np.zeros((capacity, 1), dtype=np.float32)
        self.truncations = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0  # next slot to write
        self.size = 0  # number of valid transitions

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        termination: bool,
        truncation: bool,
    ) -> None:
        """Add one transition. `next_obs` must be the true observation after the
        step (not the post-reset observation of the next episode)."""
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminations[self.pos] = float(termination)
        self.truncations[self.pos] = float(truncation)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Sample uniformly (with replacement) from the stored transitions."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        idx = self.rng.integers(0, self.size, size=batch_size)

        def to_t(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(a[idx], device=self.device)

        return Batch(
            to_t(self.obs),
            to_t(self.next_obs),
            to_t(self.actions),
            to_t(self.rewards),
            to_t(self.terminations),
            to_t(self.truncations),
        )
