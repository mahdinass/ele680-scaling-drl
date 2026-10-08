"""SAC update primitives (no environment interaction, no training loop).

Formulas follow CleanRL's `sac_continuous_action.py`:
  target  y = r + gamma * (1 - termination) * (min(Q1', Q2')(s', a') - alpha * log pi(a'|s')),  a' ~ pi(.|s')
  critic  L_Q = MSE(Q1(s, a), y) + MSE(Q2(s, a), y)
  actor   L_pi = mean(alpha * log pi(a_pi|s) - min(Q1, Q2)(s, a_pi)),  a_pi ~ pi(.|s) (reparameterized)
  alpha   L_alpha = mean(-alpha * (log pi(a_pi|s) + target_entropy)),  alpha = exp(log_alpha)
  target  theta' <- tau * theta + (1 - tau) * theta'
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box
from torch.optim import Adam

from src.networks import Actor, QNetwork
from src.replay_buffer import Batch


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target <- tau * source + (1 - tau) * target."""
    for t, s in zip(target.parameters(), source.parameters()):
        t.copy_(tau * s + (1.0 - tau) * t)


@torch.no_grad()
def grad_norm(parameters) -> float:
    """Global L2 norm of the gradients of `parameters` (no clipping)."""
    norms = [p.grad.norm() for p in parameters if p.grad is not None]
    return torch.stack(norms).norm().item() if norms else 0.0


class SACAgent:
    """Holds the SAC networks, optimizers and entropy temperature.

    A class (instead of loose functions) because every update needs the same
    pieces of state: actor, two critics, two target critics, three optimizers,
    log_alpha and the target entropy. The training loop decides when to call
    each update method (e.g. delayed actor updates).
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        device: torch.device | str = "cpu",
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(observation_space, action_space).to(self.device)
        self.qf1 = QNetwork(observation_space, action_space).to(self.device)
        self.qf2 = QNetwork(observation_space, action_space).to(self.device)
        # Targets start as independent copies and are never trained by an optimizer.
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        for p in [*self.qf1_target.parameters(), *self.qf2_target.parameters()]:
            p.requires_grad_(False)

        # Automatic entropy tuning: optimize log_alpha so alpha = exp(log_alpha) > 0.
        # Default target entropy for continuous actions is -act_dim.
        self.target_entropy = -float(torch.tensor(action_space.shape).prod().item())
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = Adam([*self.qf1.parameters(), *self.qf2.parameters()], lr=q_lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=q_lr)

        # Filled by update methods when called with diagnostics=True.
        self.diagnostics: dict[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy coefficient (no gradient)."""
        return self.log_alpha.exp().detach()

    def critic_loss(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(batch.next_obs)
            min_q_next = torch.min(
                self.qf1_target(batch.next_obs, next_action),
                self.qf2_target(batch.next_obs, next_action),
            )
            # The ONLY place episode endings enter the target. Only true terminations
            # cut the bootstrap; time-limit truncations still bootstrap from next_obs.
            mask = 1.0 - batch.terminations
            target_q = batch.rewards + self.gamma * mask * (min_q_next - self.alpha * next_log_prob)
        return F.mse_loss(self.qf1(batch.obs, batch.actions), target_q) + F.mse_loss(
            self.qf2(batch.obs, batch.actions), target_q
        )

    def actor_loss(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (actor loss, log_pi). Only actor_optimizer steps on this loss;
        gradients that also reach the critics are ignored (never stepped)."""
        action, log_pi = self.actor.get_action(batch.obs)
        min_q = torch.min(self.qf1(batch.obs, action), self.qf2(batch.obs, action))
        return (self.alpha * log_pi - min_q).mean(), log_pi

    def alpha_loss(self, log_pi: torch.Tensor) -> torch.Tensor:
        return (-self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)).mean()

    def update_critic(self, batch: Batch, diagnostics: bool = False) -> float:
        """One critic step; returns the loss. With `diagnostics=True` also fills
        `self.diagnostics` with critic_grad_norm and q_mean_abs (mean |Q1|, |Q2|).
        Diagnostics only read values; they do not change the update."""
        loss = self.critic_loss(batch)
        self.critic_optimizer.zero_grad()
        loss.backward()
        if diagnostics:
            self.diagnostics["critic_grad_norm"] = grad_norm([*self.qf1.parameters(), *self.qf2.parameters()])
            with torch.no_grad():
                q1 = self.qf1(batch.obs, batch.actions).abs().mean()
                q2 = self.qf2(batch.obs, batch.actions).abs().mean()
            self.diagnostics["q_mean_abs"] = (0.5 * (q1 + q2)).item()
        self.critic_optimizer.step()
        return loss.item()

    def update_actor_and_alpha(self, batch: Batch, diagnostics: bool = False) -> tuple[float, float]:
        """One actor step and one alpha step. Returns (actor_loss, alpha_loss).
        With `diagnostics=True` also fills `self.diagnostics["actor_grad_norm"]`."""
        actor_loss, _ = self.actor_loss(batch)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if diagnostics:
            self.diagnostics["actor_grad_norm"] = grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        # Alpha uses a fresh log_pi from the just-updated policy (no gradient).
        with torch.no_grad():
            _, log_pi = self.actor.get_action(batch.obs)
        alpha_loss = self.alpha_loss(log_pi)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return actor_loss.item(), alpha_loss.item()

    def update_targets(self) -> None:
        soft_update(self.qf1_target, self.qf1, self.tau)
        soft_update(self.qf2_target, self.qf2, self.tau)
