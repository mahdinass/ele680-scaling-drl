"""Synthetic artifact audit tests. These fixtures are NOT training results."""

import csv
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import torch

from scripts import inputnorm_gorina4 as gates
from src.checkpoint import save_checkpoint
from src.sac_update import SACAgent
from src.train import load_config


class AuditTests(unittest.TestCase):
    def test_valid_synthetic_record_and_corruptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "INPUTNORM_8x256_preflight6200_seed0_SYNTHETIC"
            run = root / "results" / run_id
            run.mkdir(parents=True)
            config = dataclasses.replace(load_config(str(gates.CONFIG)), total_steps=6200, run_id=run_id)
            cfg = dataclasses.asdict(config)
            env = gym.make("HalfCheetah-v5")
            try:
                agent = SACAgent(env.observation_space, env.action_space, hidden_depth=8, input_norm=True)
            finally:
                env.close()
            for _ in range(6200):
                agent.observation_normalizer.update(np.zeros(17))
            save_checkpoint(run / "checkpoint.pt", agent, 6200, config, 0)
            meta = {"config": cfg, "git_commit": "a" * 40, "git_dirty": False,
                    "status": "completed", "steps_completed": 6200, "device": "cuda",
                    "cuda_visible_devices": "0", "gpu_name": "Tesla P100-PCIE-12GB",
                    "parameter_counts": {"actor": 468236, "q1": 466945, "q2": 466945, "total": 1402126},
                    "input_normalization": agent.observation_normalizer.summary(),
                    "final_eval": {"step": 6200, "num_episodes": 10, "returns": [1.] * 10}}
            for name, data in (("config.json", cfg), ("metadata.json", meta),
                               ("status.json", {"status": "completed", "steps_completed": 6200})):
                (run / name).write_text(json.dumps(data))
            def csv_file(name, fields, rows):
                with (run / name).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    writer.writerows(rows)
            csv_file("eval_metrics.csv", ["step", "kind", "num_episodes", "returns", "mean_return", "std_return"],
                     [[6200, "final", 10, json.dumps([1.] * 10), 1., 0.]])
            csv_file("train_metrics.csv", ["step", "critic_loss"], [[6001, 1.]])
            csv_file("episode_metrics.csv", ["step", "episode_length", "episode_return"],
                     [[s, 1000, 1.] for s in range(1000, 6201, 1000)])
            exit_file = root / f"{run_id}.exitcode"
            exit_file.write_text("0\n")
            telemetry = root / f"{run_id}.gpu.csv"
            telemetry.write_text("timestamp,name\nsynthetic,Tesla P100\n")
            with mock.patch.object(gates, "ROOT", root), mock.patch.object(gates, "EXTERNAL_ROOT", root), \
                    mock.patch.object(gates, "verify_source"):
                gates.audit("a" * 40, run_id, 6200, 0)
                receipt = gates.receipt_dir("a" * 40) / "audit_6200_seed0.json"
                self.assertTrue(receipt.is_file())
                # Alter exactly one artifact each time, then restore its original bytes.
                mutations = [(exit_file, "1\n"), (telemetry, ""),
                             (run / "config.json", json.dumps({**cfg, "hidden_depth": 4})),
                             (run / "metadata.json", json.dumps({**meta, "git_dirty": True})),
                             (run / "metadata.json", json.dumps({**meta, "input_normalization": {}})),
                             (run / "eval_metrics.csv", "step,kind,num_episodes,returns,mean_return,std_return\n"),
                             (run / "train_metrics.csv", "step,critic_loss\n6001,nan\n")]
                for path, changed in mutations:
                    with self.subTest(path=path.name, changed=changed[:40]):
                        original = path.read_bytes()
                        path.write_text(changed)
                        try:
                            with self.assertRaises((RuntimeError, ValueError)):
                                gates.audit("a" * 40, run_id, 6200, 0)
                        finally:
                            path.write_bytes(original)
                state = torch.load(run / "checkpoint.pt", weights_only=True)
                state["observation_normalization"]["count"] = 6199
                torch.save(state, run / "checkpoint.pt")
                with self.assertRaises(RuntimeError):
                    gates.audit("a" * 40, run_id, 6200, 0)
