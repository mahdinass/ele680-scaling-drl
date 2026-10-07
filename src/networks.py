"""Neural-network primitives for the SAC baseline (no losses, optimizers or env code).

Adapted from the ideas in CleanRL's `sac_continuous_action.py`: ReLU MLPs,
tanh-bounded log_std, tanh-squashed Gaussian actor with action scale/bias,
and the tanh change-of-variables correction to the log-probability.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box

HIDDEN_SIZE = 256
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class QNetwork(nn.Module):
    """One SAC Q-function Q(s, a) -> scalar. Q1 and Q2 are separate instances."""

    def __init__(self, observation_space: Box, action_space: Box) -> None:
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim + act_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim), action: (B, act_dim) -> Q-values (B, 1)."""
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    """Squashed-Gaussian SAC policy for bounded continuous (Box) actions."""

    def __init__(self, observation_space: Box, action_space: Box) -> None:
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_mean = nn.Linear(HIDDEN_SIZE, act_dim)
        self.fc_logstd = nn.Linear(HIDDEN_SIZE, act_dim)

        # Map tanh output in (-1, 1) onto [low, high]: a = tanh(u) * scale + bias.
        # Buffers follow .to(device) but are not trained.
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs (B, obs_dim) -> Gaussian mean and log_std, each (B, act_dim)."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        # Bound log_std smoothly: tanh -> (-1, 1) -> rescale to (LOG_STD_MIN, LOG_STD_MAX).
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a (reparameterized) action and its log-probability.

        Returns action (B, act_dim) within [low, high] and log_prob (B, 1).
        """
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()  # u = mean + std * eps, differentiable
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # Change of variables for tanh and the affine rescale:
        # log pi(a) = log N(u) - log(scale * (1 - tanh(u)^2) + eps)
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluation action: squashed, rescaled Gaussian mean (no sampling)."""
        mean, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias
