"""Neural-network primitives for the SAC baseline (no losses, optimizers or env code).

Adapted from the ideas in CleanRL's `sac_continuous_action.py`: ReLU MLPs,
tanh-bounded log_std, tanh-squashed Gaussian actor with action scale/bias,
and the tanh change-of-variables correction to the log-probability.

Optional single-layer identity shortcuts use the ELE680 residual-only protocol:
ReLU(Wx + b + x), after the unchanged first projection. Conceptual attribution:
He et al., Deep Residual Learning for Image Recognition, arXiv:1512.03385.
No external residual implementation is copied; this is not the full ResNet/SimBa.

Optional LayerNorm-only uses independent learned normalization after every hidden
Linear, before ReLU (Ba, Kiros and Hinton, arXiv:1607.06450). See LAYERNORM_ABLATION.md.
"""

import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box

HIDDEN_SIZE = 256  # default hidden width (the frozen 2x256 baseline)
HIDDEN_DEPTH = 2  # default number of hidden layers (the frozen 2x256 baseline)
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _check_residual_connections(value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError("residual_connections must be a bool")
    return value


def _check_layer_norm(value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError("layer_norm must be a bool")
    return value


def _check_width(hidden_width: int) -> int:
    if hidden_width < 1:
        raise ValueError(f"hidden_width must be a positive integer, got {hidden_width}")
    return hidden_width


def _check_depth(hidden_depth: int) -> int:
    """Depth = number of fully connected hidden layers before the output head.
    fc1 and fc2 always exist (keeps the baseline layer names), so the minimum is 2."""
    if isinstance(hidden_depth, bool) or not isinstance(hidden_depth, numbers.Integral):
        raise ValueError(f"hidden_depth must be an integer >= 2, got {hidden_depth!r}")
    if hidden_depth < 2:
        raise ValueError(f"hidden_depth must be an integer >= 2, got {hidden_depth}")
    return int(hidden_depth)


def _extra_hidden_layers(hidden_width: int, hidden_depth: int) -> nn.ModuleList:
    """Width-preserving hidden layers beyond fc1/fc2 (empty for depth 2, so the
    baseline state_dict keys and RNG consumption are unchanged)."""
    return nn.ModuleList([nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth - 2)])


class QNetwork(nn.Module):
    """One SAC Q-function Q(s, a) -> scalar. Q1 and Q2 are separate instances."""

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_width: int = HIDDEN_SIZE,
        hidden_depth: int = HIDDEN_DEPTH,
        residual_connections: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.residual_connections = _check_residual_connections(residual_connections)
        self.layer_norm = _check_layer_norm(layer_norm)
        if self.layer_norm and self.residual_connections:
            raise ValueError("LayerNorm-only cannot enable residual_connections")
        hidden_width = _check_width(hidden_width)
        hidden_depth = _check_depth(hidden_depth)
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.hidden_extra = _extra_hidden_layers(hidden_width, hidden_depth)  # hidden layers 3..depth
        self.fc3 = nn.Linear(hidden_width, 1)
        # Construct after all historical Linears: their ordering/init stays fixed.
        # No additional modules or state_dict entries exist on the disabled path.
        if self.layer_norm:
            self.hidden_norms = nn.ModuleList([
                nn.LayerNorm(hidden_width, eps=1e-5, elementwise_affine=True)
                for _ in range(hidden_depth)
            ])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim), action: (B, act_dim) -> Q-values (B, 1)."""
        x = torch.cat([obs, action], dim=1)
        if self.layer_norm:
            x = F.relu(self.hidden_norms[0](self.fc1(x)))
            x = F.relu(self.hidden_norms[1](self.fc2(x)))
            for i, layer in enumerate(self.hidden_extra, start=2):
                x = F.relu(self.hidden_norms[i](layer(x)))
            return self.fc3(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x) if self.residual_connections else F.relu(self.fc2(x))
        for layer in self.hidden_extra:
            x = F.relu(layer(x) + x) if self.residual_connections else F.relu(layer(x))
        return self.fc3(x)


class Actor(nn.Module):
    """Squashed-Gaussian SAC policy for bounded continuous (Box) actions."""

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_width: int = HIDDEN_SIZE,
        hidden_depth: int = HIDDEN_DEPTH,
        residual_connections: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.residual_connections = _check_residual_connections(residual_connections)
        self.layer_norm = _check_layer_norm(layer_norm)
        if self.layer_norm and self.residual_connections:
            raise ValueError("LayerNorm-only cannot enable residual_connections")
        hidden_width = _check_width(hidden_width)
        hidden_depth = _check_depth(hidden_depth)
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.hidden_extra = _extra_hidden_layers(hidden_width, hidden_depth)  # hidden layers 3..depth
        self.fc_mean = nn.Linear(hidden_width, act_dim)
        self.fc_logstd = nn.Linear(hidden_width, act_dim)
        if self.layer_norm:
            self.hidden_norms = nn.ModuleList([
                nn.LayerNorm(hidden_width, eps=1e-5, elementwise_affine=True)
                for _ in range(hidden_depth)
            ])

        # Map tanh output in (-1, 1) onto [low, high]: a = tanh(u) * scale + bias.
        # Buffers follow .to(device) but are not trained.
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs (B, obs_dim) -> Gaussian mean and log_std, each (B, act_dim)."""
        if self.layer_norm:
            x = F.relu(self.hidden_norms[0](self.fc1(obs)))
            x = F.relu(self.hidden_norms[1](self.fc2(x)))
            for i, layer in enumerate(self.hidden_extra, start=2):
                x = F.relu(self.hidden_norms[i](layer(x)))
        else:
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x) + x) if self.residual_connections else F.relu(self.fc2(x))
            for layer in self.hidden_extra:
                x = F.relu(layer(x) + x) if self.residual_connections else F.relu(layer(x))
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
