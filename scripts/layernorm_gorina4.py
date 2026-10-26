"""Manual Gorina4 gates/audits. No training is launched by this module.

Receipts live under ignored results/. They apply only to an exact clean commit.
"""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.checkpoint import load_checkpoint
from src.run_metadata import count_parameters
from src.run_logger import TRAIN_FIELDS, EPISODE_FIELDS, EVAL_FIELDS
from src.sac_update import SACAgent
from src.train import load_config

CONFIG = ROOT / "configs/ablation_layernorm_8x256.json"
CANONICAL_PYTHON = "/bhome/u273406/conda-envs/ele680/bin/python"
EXTERNAL_ROOT = Path("/bhome/u273406")
CONTROL = "6c674babc344e62566692e1a4fcf3611dc53a6cc"


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
    require(control == json.loads(command("git", "show", f"{CONTROL}:configs/depth_8x256.json")),
            "Frozen control config drift")
    proposed = json.loads(CONFIG.read_text())
    require(proposed.pop("layer_norm", None) is True, "LayerNorm flag must be true")
    require(proposed.pop("input_norm", None) is False and
            proposed.pop("residual_connections", None) is False and proposed == control, "Unexpected config diff")
    require(control.get("hidden_depth") == 8 and control.get("hidden_width", 256) == 256, "Wrong control architecture")


def receipt_dir(sha):
    require(len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), "Require full lowercase SHA")
    return ROOT / "results/layernorm_gates" / sha


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
        require(data.get("experiment") == "layernorm_only_8x256_v1", "Wrong experiment receipt")
        require(data["sha"] == sha and data["successful"] is True
                and type(data["tests"]) is int and data["tests"] > 0
                and type(data["skipped"]) is int and 0 <= data["skipped"] <= data["tests"], "Invalid test receipt")
        if mode == "cuda":
            require(data["skipped"] == 0, "CUDA tests were skipped")


def run_tests(sha, mode):
    require(mode in ("cpu", "cuda"), "Invalid test mode")
    # A failed rerun must invalidate an earlier success for the same SHA.
    receipt = receipt_dir(sha) / f"tests_{mode}.json"
    if receipt.exists():
        receipt.unlink()
    verify_source(sha)
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
    write_receipt(sha, f"tests_{mode}.json", {"sha": sha, "experiment": "layernorm_only_8x256_v1", "successful": True,
                                           "tests": result.testsRun, "skipped": len(result.skipped)})


def read_rows(path, fields=None):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if fields is not None:
            require(reader.fieldnames == fields, f"Wrong CSV schema: {path.name}")
        rows = list(reader)
        require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
                f"Malformed CSV: {path.name}")
        return rows


def verify_telemetry(path, gpu_name):
    rows = read_rows(path, ["timestamp", "uuid", "name", "utilization.gpu [%]",
                            "memory.used [MiB]", "memory.total [MiB]"])
    require(bool(rows), "Missing GPU telemetry samples")
    uuid = rows[0]["uuid"].strip()
    require(uuid.startswith("GPU-"), "Missing physical GPU UUID")
    for row in rows:
        require(row["timestamp"].strip() and row["uuid"].strip() == uuid
                and row["name"].strip() == gpu_name, "Wrong telemetry identity")
        utilization, used, total = [float(row[key].split()[0]) for key in
                                    ("utilization.gpu [%]", "memory.used [MiB]", "memory.total [MiB]")]
        require(all(math.isfinite(v) for v in (utilization, used, total)) and
                0 <= utilization <= 100 and 0 <= used <= total and total > 0,
                "Invalid GPU telemetry values")


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def verify_architecture(agent):
    require(agent.layer_norm is True and agent.residual_connections is False
            and agent.observation_normalizer is None, "Require LayerNorm-only agent")
    require(count_parameters(agent) == {"actor": 472332, "q1": 471041, "q2": 471041, "total": 1414414},
            "Wrong actor/online critic parameter counts")
    norm_parameters = []
    for name in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
        network = getattr(agent, name)
        require(network.layer_norm is True and network.residual_connections is False
                and len(network.hidden_extra) == 6, f"Wrong LayerNorm topology: {name}")
        hidden = [network.fc1, network.fc2, *network.hidden_extra]
        require(len(hidden) == 8 and all(type(layer) is torch.nn.Linear and layer.out_features == 256
                                       for layer in hidden), f"Wrong hidden transformations: {name}")
        require(hidden[0].in_features == (17 if name == "actor" else 23)
                and all(layer.in_features == 256 for layer in hidden[1:]), "Wrong hidden inputs")
        require(len(network.hidden_norms) == 8 and all(
            type(norm) is torch.nn.LayerNorm and norm.normalized_shape == (256,)
            and norm.eps == 1e-5 and norm.elementwise_affine
            and norm.weight is not None and norm.bias is not None
            for norm in network.hidden_norms), f"Wrong hidden LayerNorms: {name}")
        norm_parameters.extend(p for norm in network.hidden_norms for p in norm.parameters())
        require(all(type(module) in (type(network), torch.nn.ModuleList, torch.nn.Linear, torch.nn.LayerNorm)
                    for module in network.modules()), "Unexpected network module")
        require(sum(type(module) is torch.nn.LayerNorm for module in network.modules()) == 8,
                "Unexpected normalization outside hidden layers")
        heads = [network.fc_mean, network.fc_logstd] if name == "actor" else [network.fc3]
        require(all(type(head) is torch.nn.Linear and head.in_features == 256
                    and head.out_features == (6 if name == "actor" else 1) for head in heads), "Wrong heads")
        expected_modules = {"", "fc1", "fc2", "hidden_extra", "hidden_norms",
                            *[f"hidden_extra.{i}" for i in range(6)],
                            *[f"hidden_norms.{i}" for i in range(8)]}
        expected_modules.update(("fc_mean", "fc_logstd") if name == "actor" else ("fc3",))
        require(set(dict(network.named_modules())) == expected_modules, "Unexpected or shared modules")
        require(all(p.requires_grad is (name not in ("qf1_target", "qf2_target"))
                    for p in network.parameters()), "Wrong parameter gradient mode")
    require(len({p.data_ptr() for p in norm_parameters}) == 80, "Shared LayerNorm affine parameters")
    for online, target in ((agent.qf1, agent.qf1_target), (agent.qf2, agent.qf2_target)):
        require(set(online.state_dict()) == set(target.state_dict()) and
                all(p.data_ptr() != t.data_ptr() and not t.requires_grad
                    for p, t in zip(online.parameters(), target.parameters())), "Targets not independent/frozen")


def verify_mode_flags(config):
    require(config.get("layer_norm") is True and config.get("input_norm") is False
            and config.get("residual_connections") is False, "Wrong LayerNorm-only semantic flags")


def audit(sha, run_id, steps, seed):
    # Failed re-audits invalidate earlier receipts; raw artifacts are always reread.
    receipt = receipt_dir(sha) / f"audit_{steps}_seed{seed}.json"
    if receipt.exists():
        receipt.unlink()
    verify_source(sha)
    require(type(steps) is int and steps in (6200, 26000, 1_000_000)
            and type(seed) is int and seed in (0, 1, 2), "Unsupported audit protocol")
    kind = {6200: "preflight6200", 26000: "preflight26000", 1_000_000: "scientific"}[steps]
    require(steps == 1_000_000 or seed == 0, "Preflights require seed 0")
    require(re.fullmatch(rf"LAYERNORM_8x256_{kind}_seed{seed}_[0-9]{{8}}T[0-9]{{6}}Z", run_id)
            is not None, "Invalid run ID/kind/seed")
    run = ROOT / "results" / run_id
    require((EXTERNAL_ROOT / f"{run_id}.exitcode").read_text().strip() == "0", "Nonzero/missing Python exit")
    log = (EXTERNAL_ROOT / f"{run_id}.log").read_text()
    require(f"Run completed: results/{run_id}" in log.splitlines(), "Missing completion log")
    cfg = json.loads((run / "config.json").read_text())
    verify_mode_flags(cfg)
    base = vars(load_config(str(CONFIG))).copy()
    base.update(seed=seed, total_steps=steps, device="cuda", run_id=run_id)
    require(cfg == base, "Run configuration drift")
    meta = json.loads((run / "metadata.json").read_text())
    status = json.loads((run / "status.json").read_text())
    require(finite_tree(cfg) and finite_tree(meta) and finite_tree(status), "Nonfinite JSON artifact")
    verify_mode_flags(meta["config"])
    require(meta["config"] == cfg and meta["git_commit"] == sha and meta["git_dirty"] is False, "Wrong provenance")
    for data in (meta, status):
        require(data["status"] == "completed" and data["steps_completed"] == steps, "Incomplete run")
    require(meta["cuda_visible_devices"] == "0" and "P100" in meta["gpu_name"] and meta["device"] == "cuda", "Wrong recorded GPU")
    require(meta["parameter_counts"] == {"actor": 472332, "q1": 471041, "q2": 471041, "total": 1414414}, "Wrong architecture")
    for key, expected_version in (("torch_version", "2.5.1+cu121"), ("torch_cuda_runtime", "12.1"),
                                  ("numpy_version", "1.26.4"), ("gymnasium_version", "1.1.1"),
                                  ("mujoco_version", "3.15.0"), ("tensorboard_version", "2.18.0")):
        require(meta[key] == expected_version, f"Wrong recorded version: {key}")
    require(meta["python_executable"] == CANONICAL_PYTHON and meta["python_version"].startswith("3.12."),
            "Wrong recorded Python")
    verify_telemetry(EXTERNAL_ROOT / f"{run_id}.gpu.csv", meta["gpu_name"])
    state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    verify_mode_flags(state["config"])
    require(state["global_step"] == steps and state["seed"] == seed and state["config"] == cfg, "Wrong checkpoint identity")
    require(finite_tree(state), "Nonfinite checkpoint")
    require("observation_normalization" not in state and "input_normalization" not in meta, "LayerNorm-only run contains observation normalization")
    import gymnasium as gym
    env = gym.make("HalfCheetah-v5")
    try:
        agent = SACAgent(env.observation_space, env.action_space, hidden_depth=8, input_norm=False, residual_connections=False, layer_norm=True)
        verify_architecture(agent)
        load_checkpoint(run / "checkpoint.pt", agent)
        obs, _ = env.reset(seed=seed + 10000)
        x = agent.normalize_observation(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        with torch.no_grad():
            action = agent.actor.get_deterministic_action(x)
            require(action.shape == (1, 6) and bool(torch.isfinite(action).all())
                    and bool((action >= agent.actor.action_bias - agent.actor.action_scale).all())
                    and bool((action <= agent.actor.action_bias + agent.actor.action_scale).all()), "Reload actor failed")
            require(all(bool(torch.isfinite(q(x, action)).all()) for q in
                        (agent.qf1, agent.qf2, agent.qf1_target, agent.qf2_target)), "Reload critics failed")
    finally:
        env.close()
    rows = read_rows(run / "eval_metrics.csv", EVAL_FIELDS)
    expected = [(s, "periodic", 5) for s in range(25000, steps + 1, 25000)] + [(steps, "final", 10)]
    require([(int(r["step"]), r["kind"], int(r["num_episodes"])) for r in rows] == expected, "Wrong evaluation schedule")
    for row in rows:
        returns = np.asarray(json.loads(row["returns"]), dtype=float)
        require(len(returns) == int(row["num_episodes"]) and np.isfinite(returns).all(), "Invalid episode returns")
        require(np.isclose(returns.mean(), float(row["mean_return"])) and np.isclose(returns.std(), float(row["std_return"])), "Incorrect eval aggregate")
    final = meta["final_eval"]
    require(final["step"] == steps and final["num_episodes"] == 10 and final["returns"] == json.loads(rows[-1]["returns"]), "Wrong final metadata")
    require(np.isclose(final["mean_return"], float(rows[-1]["mean_return"])) and
            np.isclose(final["std_return"], float(rows[-1]["std_return"])), "Wrong final metadata aggregate")
    training = read_rows(run / "train_metrics.csv", TRAIN_FIELDS)
    require([int(r["step"]) for r in training] == list(range(6001, steps + 1, 1000)), "Wrong training metric schedule")
    for row in training:
        require(all(v != "" and math.isfinite(float(v)) for v in row.values()), "Missing/nonfinite training metric")
    episodes = read_rows(run / "episode_metrics.csv", EPISODE_FIELDS)
    require([int(r["step"]) for r in episodes] == list(range(1000, steps + 1, 1000)), "Wrong training episodes")
    require(all(int(r["episode_length"]) == 1000 and math.isfinite(float(r["episode_return"])) for r in episodes), "Invalid training episode")
    write_receipt(sha, f"audit_{steps}_seed{seed}.json", {"sha": sha, "run_id": run_id, "steps": steps, "seed": seed, "experiment": "layernorm_only_8x256_v1"})
    print(f"AUDIT PASSED: {run_id}; layer_norm=True; final={rows[-1]['mean_return']}")


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
