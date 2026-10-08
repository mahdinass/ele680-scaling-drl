"""Run provenance: git commit, library versions, hardware and parameter counts.

Every lookup degrades to None instead of raising, so metadata collection can
never break a run.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip()
    except Exception:  # git missing, not a repo, timeout, ...
        return None


def get_git_commit() -> str | None:
    """Full HEAD commit hash, or None if git/the repo is unavailable."""
    return _git("rev-parse", "HEAD") or None


def get_git_dirty() -> bool | None:
    """True if the working tree is not clean: tracked modifications OR untracked files
    (generated outputs such as results/ are git-ignored, so they do not count)."""
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def collect_environment_info(device: str) -> dict:
    """Versions, CUDA/GPU info and git state for metadata.json."""
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    if torch.device(device).type == "cuda" and cuda_available:
        gpu_name = torch.cuda.get_device_name(torch.device(device).index or 0)
    return {
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": gpu_name,
        "numpy_version": _version("numpy"),
        "gymnasium_version": _version("gymnasium"),
        "mujoco_version": _version("mujoco"),
        "tensorboard_version": _version("tensorboard"),
        "hostname": platform.node(),
    }


def count_parameters(agent) -> dict[str, int]:
    """Trainable parameter counts of the actor and both critics."""

    def n(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    actor, q1, q2 = n(agent.actor), n(agent.qf1), n(agent.qf2)
    return {"actor": actor, "q1": q1, "q2": q2, "total": actor + q1 + q2}
