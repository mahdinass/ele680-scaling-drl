"""Checkpoint save/load.

A checkpoint holds model + optimizer state, log_alpha, the step, config and seed.
It does NOT contain the replay buffer (up to ~170 MB at 1M transitions) or RNG
states, so a checkpoint is for inspection/evaluation; exact resumption of training
would additionally need those.
"""

import os
from dataclasses import asdict
from pathlib import Path

import torch

from src.sac_update import SACAgent


def save_checkpoint(path: str | Path, agent: SACAgent, global_step: int, config, seed: int) -> None:
    """Write atomically (temp file + rename) so a crash never leaves a half-written file."""
    path = Path(path)
    state = {
        "actor": agent.actor.state_dict(),
        "qf1": agent.qf1.state_dict(),
        "qf2": agent.qf2.state_dict(),
        "qf1_target": agent.qf1_target.state_dict(),
        "qf2_target": agent.qf2_target.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "alpha_optimizer": agent.alpha_optimizer.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "global_step": global_step,
        "config": asdict(config),
        "seed": seed,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, agent: SACAgent) -> dict:
    """Restore `agent` in place; returns {"global_step", "config", "seed"}.

    Uses weights_only=True, so only tensors and plain Python containers are loaded.
    """
    state = torch.load(path, map_location=agent.device, weights_only=True)
    agent.actor.load_state_dict(state["actor"])
    agent.qf1.load_state_dict(state["qf1"])
    agent.qf2.load_state_dict(state["qf2"])
    agent.qf1_target.load_state_dict(state["qf1_target"])
    agent.qf2_target.load_state_dict(state["qf2_target"])
    agent.actor_optimizer.load_state_dict(state["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(state["critic_optimizer"])
    agent.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
    with torch.no_grad():
        agent.log_alpha.copy_(state["log_alpha"])
    return {"global_step": state["global_step"], "config": state["config"], "seed": state["seed"]}
