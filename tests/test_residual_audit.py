"""Synthetic artifact fixtures only: no preflight or scientific training is run."""

import csv
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts import residual_gorina4 as gates
from src.checkpoint import save_checkpoint
from src.sac_update import SACAgent
from src.train import load_config
from tests.test_sac_update import spaces


class AuditTests(unittest.TestCase):
    def fixture(self, root, steps):
        run_id = f"RESIDUAL_8x256_synthetic_seed0_{steps}"
        run = root / "results" / run_id
        run.mkdir(parents=True)
        config = dataclasses.replace(load_config(str(gates.CONFIG)), total_steps=steps, run_id=run_id)
        cfg = dataclasses.asdict(config)
        agent = SACAgent(*spaces(), hidden_depth=8, residual_connections=True)
        save_checkpoint(run / "checkpoint.pt", agent, steps, config, 0)
        meta = {"config": cfg, "git_commit": "a" * 40, "git_dirty": False,
                "status": "completed", "steps_completed": steps, "device": "cuda",
                "cuda_visible_devices": "0", "gpu_name": "Tesla P100-PCIE-12GB",
                "parameter_counts": {"actor": 468236, "q1": 466945, "q2": 466945, "total": 1402126},
                "python_executable": gates.CANONICAL_PYTHON, "python_version": "3.12.14",
                "torch_version": "2.5.1+cu121", "torch_cuda_runtime": "12.1", "numpy_version": "1.26.4",
                "gymnasium_version": "1.1.1", "mujoco_version": "3.15.0", "tensorboard_version": "2.18.0",
                "final_eval": {"step": steps, "num_episodes": 10, "returns": [1.] * 10,
                               "mean_return": 1., "std_return": 0.}}
        for name, data in (("config.json", cfg), ("metadata.json", meta),
                           ("status.json", {"status": "completed", "steps_completed": steps})):
            (run / name).write_text(json.dumps(data))

        def csv_file(name, fields, rows):
            with (run / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        evals = [dict(step=s, kind="periodic", num_episodes=5, returns=json.dumps([1.] * 5),
                      mean_return=1., std_return=0.) for s in range(25000, steps + 1, 25000)]
        evals.append(dict(step=steps, kind="final", num_episodes=10, returns=json.dumps([1.] * 10), mean_return=1., std_return=0.))
        csv_file("eval_metrics.csv", gates.EVAL_FIELDS, evals)
        csv_file("train_metrics.csv", gates.TRAIN_FIELDS,
                 [{**{key: 1. for key in gates.TRAIN_FIELDS}, "step": s} for s in range(6001, steps + 1, 1000)])
        csv_file("episode_metrics.csv", gates.EPISODE_FIELDS,
                 [dict(step=s, elapsed_sec=1., episode_return=1., episode_length=1000)
                  for s in range(1000, steps + 1, 1000)])
        (root / f"{run_id}.exitcode").write_text("0\n")
        (root / f"{run_id}.log").write_text(f"Run completed: results/{run_id}\n")
        (root / f"{run_id}.gpu.csv").write_text(
            "timestamp, uuid, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB]\n"
            "2026/10/18 12:00:00.000, GPU-SYNTHETIC, Tesla P100-PCIE-12GB, 25 %, 361 MiB, 12288 MiB\n")
        return run_id, run, cfg, meta

    def test_accepts_separate_6200_26000_and_million_step_synthetic_artifacts(self):
        for steps in (6200, 26000, 1000000):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_id, run, cfg, meta = self.fixture(root, steps)
                with mock.patch.object(gates, "ROOT", root), mock.patch.object(gates, "EXTERNAL_ROOT", root), \
                        mock.patch.object(gates, "verify_source"):
                    gates.audit("a" * 40, run_id, steps, 0)
                    receipt = gates.receipt_dir("a" * 40) / f"audit_{steps}_seed0.json"
                    self.assertEqual(json.loads(receipt.read_text())["experiment"], "residual_only_8x256_v1")

    def test_corruptions_rejected_and_failed_reaudit_invalidates_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, run, cfg, meta = self.fixture(root, 26000)
            with mock.patch.object(gates, "ROOT", root), mock.patch.object(gates, "EXTERNAL_ROOT", root), \
                    mock.patch.object(gates, "verify_source"):
                gates.audit("a" * 40, run_id, 26000, 0)
                receipt = gates.receipt_dir("a" * 40) / "audit_26000_seed0.json"
                original_train = (run / "train_metrics.csv").read_text()
                original_eval = (run / "eval_metrics.csv").read_text()
                mutations = [
                    (root / f"{run_id}.exitcode", "1\n"),
                    (root / f"{run_id}.log", "Run failed\n"),
                    (root / f"{run_id}.gpu.csv", ""),
                    (run / "config.json", json.dumps({**cfg, "input_norm": True})),
                    (run / "config.json", json.dumps({**cfg, "residual_connections": False})),
                    (run / "metadata.json", json.dumps({**meta, "git_dirty": True})),
                    (run / "metadata.json", json.dumps({**meta, "input_normalization": {}})),
                    (run / "metadata.json", json.dumps({**meta, "torch_version": "different"})),
                    (run / "metadata.json", json.dumps({**meta, "parameter_counts": {"total": 1}})),
                    (run / "metadata.json", json.dumps({**meta, "final_eval": {**meta["final_eval"], "mean_return": 999.}})),
                    (run / "status.json", json.dumps({"status": "completed", "steps_completed": 25999})),
                    (run / "train_metrics.csv", original_train.replace("1.0", "nan", 1)),
                    (run / "train_metrics.csv", original_train.replace("critic_loss", "missing_metric", 1)),
                    (run / "eval_metrics.csv", "\n".join(original_eval.splitlines()[0:1] + original_eval.splitlines()[2:]) + "\n"),
                ]
                for path, changed in mutations:
                    with self.subTest(path=path.name, changed=changed[:60]):
                        original = path.read_bytes()
                        path.write_text(changed)
                        try:
                            with self.assertRaises((RuntimeError, ValueError, KeyError)):
                                gates.audit("a" * 40, run_id, 26000, 0)
                            self.assertFalse(receipt.exists())
                        finally:
                            path.write_bytes(original)
                checkpoint = run / "checkpoint.pt"
                original = checkpoint.read_bytes()
                for mutation in ("flag", "normalization", "nonfinite"):
                    state = torch.load(checkpoint, weights_only=True)
                    if mutation == "flag":
                        state["config"].pop("residual_connections")
                    elif mutation == "normalization":
                        state["observation_normalization"] = {}
                    else:
                        state["actor"]["fc1.weight"][0, 0] = float("nan")
                    torch.save(state, checkpoint)
                    try:
                        with self.assertRaises((RuntimeError, ValueError)):
                            gates.audit("a" * 40, run_id, 26000, 0)
                    finally:
                        checkpoint.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
