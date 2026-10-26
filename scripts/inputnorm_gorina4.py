"""Manual Gorina4 gates/audits. No training is launched by this module.

Receipts live under ignored results/. They apply only to an exact clean commit.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.checkpoint import load_checkpoint
from src.observation_normalization import ObservationNormalizer
from src.sac_update import SACAgent
from src.train import load_config

CONFIG = ROOT / "configs/ablation_inputnorm_8x256.json"
CANONICAL_PYTHON = "/bhome/u273406/conda-envs/ele680/bin/python"
EXTERNAL_ROOT = Path("/bhome/u273406")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def verify_source(sha):
    require(len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), "Require full lowercase SHA")
    require(command("git", "rev-parse", "HEAD") == sha, "Wrong HEAD")
    require(not command("git", "status", "--porcelain"), "Dirty worktree/index")
    control = json.loads((ROOT / "configs/depth_8x256.json").read_text())
    proposed = json.loads(CONFIG.read_text())
    require(proposed.pop("input_norm", None) is True and proposed == control, "Unexpected config diff")


def receipt_dir(sha):
    return ROOT / "results/inputnorm_gates" / sha


def write_receipt(sha, name, payload):
    directory = receipt_dir(sha)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload, indent=2) + "\n")


def verify_environment():
    import gymnasium
    import mujoco
    import tensorboard
    require(Path(sys.executable).absolute() == Path(CANONICAL_PYTHON), "Wrong Python executable")
    require(sys.version_info[:2] == (3, 12), "Expected Python 3.12")
    for name, actual, expected in (("torch", torch.__version__, "2.5.1+cu121"),
                                   ("numpy", np.__version__, "1.26.4"),
                                   ("gymnasium", gymnasium.__version__, "1.1.1"),
                                   ("mujoco", mujoco.__version__, "3.15.0"),
                                   ("tensorboard", tensorboard.__version__, "2.18.0")):
        require(actual == expected, f"{name}: expected {expected}, got {actual}")


def verify_gpu():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Require CUDA_VISIBLE_DEVICES=0")
    name = command("nvidia-smi", "-i", "0", "--query-gpu=name", "--format=csv,noheader").strip()
    require("P100" in name, f"Wrong physical GPU 0: {name}")
    uuid = command("nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader").strip()
    apps = command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")
    require(not any(row.split(",")[0].strip() == uuid for row in apps.splitlines()), "GPU 0 has active compute processes")
    utilization = int(command("nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"))
    require(utilization == 0, "GPU 0 is busy; repeat idle checks before launch")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "CUDA device mapping unavailable")
    require("P100" in torch.cuda.get_device_name(0), "Logical CUDA device is not P100")


def verify_test_receipts(sha):
    for mode in ("cpu", "cuda"):
        data = json.loads((receipt_dir(sha) / f"tests_{mode}.json").read_text())
        require(data["sha"] == sha and data["successful"] and data["tests"] > 0, "Invalid test receipt")
        if mode == "cuda":
            require(data["skipped"] == 0, "CUDA tests were skipped")


def run_tests(sha, mode):
    verify_source(sha)
    # A failed rerun must invalidate an earlier success for the same SHA.
    receipt = receipt_dir(sha) / f"tests_{mode}.json"
    if receipt.exists():
        receipt.unlink()
    verify_environment()
    if mode == "cuda":
        verify_gpu()
    else:
        require(not torch.cuda.is_available(), "Hide CUDA for CPU suite")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "Tests failed")
    require(mode != "cuda" or not result.skipped, "CUDA gate cannot pass with skipped tests")
    verify_source(sha)
    write_receipt(sha, f"tests_{mode}.json", {"sha": sha, "successful": True,
                                           "tests": result.testsRun, "skipped": len(result.skipped)})


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def audit(sha, run_id, steps, seed):
    verify_source(sha)
    require(steps in (6200, 26000, 1_000_000) and seed in (0, 1, 2), "Unsupported audit protocol")
    require(Path(run_id).name == run_id and run_id.startswith("INPUTNORM_8x256_"), "Invalid run ID")
    run = ROOT / "results" / run_id
    require((EXTERNAL_ROOT / f"{run_id}.exitcode").read_text().strip() == "0", "Nonzero/missing Python exit")
    telemetry = EXTERNAL_ROOT / f"{run_id}.gpu.csv"
    require(telemetry.is_file() and len(telemetry.read_text().splitlines()) >= 2, "Missing GPU telemetry")
    cfg = json.loads((run / "config.json").read_text())
    base = vars(load_config(str(CONFIG))).copy()
    base.update(seed=seed, total_steps=steps, device="cuda", run_id=run_id)
    require(cfg == base, "Run configuration drift")
    meta = json.loads((run / "metadata.json").read_text())
    status = json.loads((run / "status.json").read_text())
    require(meta["config"] == cfg and meta["git_commit"] == sha and meta["git_dirty"] is False, "Wrong provenance")
    for data in (meta, status):
        require(data["status"] == "completed" and data["steps_completed"] == steps, "Incomplete run")
    require(meta["cuda_visible_devices"] == "0" and "P100" in meta["gpu_name"] and meta["device"] == "cuda", "Wrong recorded GPU")
    require(meta["parameter_counts"] == {"actor": 468236, "q1": 466945, "q2": 466945, "total": 1402126}, "Wrong architecture")
    state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    require(state["global_step"] == steps and state["seed"] == seed and state["config"] == cfg, "Wrong checkpoint identity")
    require(finite_tree(state), "Nonfinite checkpoint")
    normalizer = ObservationNormalizer(17)
    normalizer.load_state_dict(state["observation_normalization"])
    require(normalizer.count == steps, "Wrong normalization count")
    require(meta["input_normalization"] == normalizer.summary(), "Metadata/checkpoint statistics differ")
    import gymnasium as gym
    env = gym.make("HalfCheetah-v5")
    try:
        agent = SACAgent(env.observation_space, env.action_space, hidden_depth=8, input_norm=True)
        load_checkpoint(run / "checkpoint.pt", agent)
        obs, _ = env.reset(seed=seed + 10000)
        x = agent.normalize_observation(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        with torch.no_grad():
            action = agent.actor.get_deterministic_action(x)
            require(bool(torch.isfinite(agent.qf1(x, action)).all()), "Reload inference failed")
    finally:
        env.close()
    rows = read_rows(run / "eval_metrics.csv")
    expected = [(s, "periodic", 5) for s in range(25000, steps + 1, 25000)] + [(steps, "final", 10)]
    require([(int(r["step"]), r["kind"], int(r["num_episodes"])) for r in rows] == expected, "Wrong evaluation schedule")
    for row in rows:
        returns = np.asarray(json.loads(row["returns"]), dtype=float)
        require(len(returns) == int(row["num_episodes"]) and np.isfinite(returns).all(), "Invalid episode returns")
        require(np.isclose(returns.mean(), float(row["mean_return"])) and np.isclose(returns.std(), float(row["std_return"])), "Incorrect eval aggregate")
    final = meta["final_eval"]
    require(final["step"] == steps and final["num_episodes"] == 10 and final["returns"] == json.loads(rows[-1]["returns"]), "Wrong final metadata")
    training = read_rows(run / "train_metrics.csv")
    require([int(r["step"]) for r in training] == list(range(6001, steps + 1, 1000)), "Wrong training metric schedule")
    for row in training:
        require(all(math.isfinite(float(v)) for v in row.values() if v != ""), "Nonfinite training metric")
    episodes = read_rows(run / "episode_metrics.csv")
    require([int(r["step"]) for r in episodes] == list(range(1000, steps + 1, 1000)), "Wrong training episodes")
    require(all(int(r["episode_length"]) == 1000 and math.isfinite(float(r["episode_return"])) for r in episodes), "Invalid training episode")
    write_receipt(sha, f"audit_{steps}_seed{seed}.json", {"sha": sha, "run_id": run_id, "steps": steps, "seed": seed})
    print(f"AUDIT PASSED: {run_id}; count={normalizer.count}; final={rows[-1]['mean_return']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tests", "gate", "audit"))
    parser.add_argument("--sha", required=True)
    parser.add_argument("--mode", choices=("cpu", "cuda"))
    parser.add_argument("--run-id")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.action == "tests":
        require(args.mode is not None, "Require test mode")
        run_tests(args.sha, args.mode)
    elif args.action == "gate":
        verify_source(args.sha)
        verify_environment()
        verify_test_receipts(args.sha)
        verify_gpu()
        print("GATE PASSED: exact clean source, config, environment, tests, idle physical GPU 0")
    else:
        require(args.run_id is not None and args.steps is not None, "Require run ID and steps")
        verify_environment()
        audit(args.sha, args.run_id, args.steps, args.seed)


if __name__ == "__main__":
    main()
