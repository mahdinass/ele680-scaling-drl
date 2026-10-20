"""Fail-closed validation of manual server gate helpers, without server access."""

import json
import tempfile
import unittest
import copy
from pathlib import Path
from unittest import mock

from scripts import layernorm_gorina4 as gates


class GateTests(unittest.TestCase):
    def test_wrong_sha_and_dirty_source_rejected(self):
        sha = "a" * 40
        with self.assertRaises(RuntimeError):
            gates.verify_source("main")
        with mock.patch.object(gates, "command", return_value="b" * 40):
            with self.assertRaises(RuntimeError):
                gates.verify_source(sha)
        with mock.patch.object(gates, "command", side_effect=[sha, " M src/train.py"]):
            with self.assertRaises(RuntimeError):
                gates.verify_source(sha)
        for invalid in ("main", "../../outside", "A" * 40):
            with self.assertRaises(RuntimeError):
                gates.receipt_dir(invalid)

    def test_config_drift_rejected(self):
        control = json.loads((gates.ROOT / "configs/depth_8x256.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "bad.json"
            valid = {**control, "layer_norm": True, "residual_connections": False, "input_norm": False}
            for changed in (valid, {**valid, "input_norm": True}, {**valid, "hidden_depth": 4},
                            {**valid, "layer_norm": 1}, {**valid, "residual_connections": True}, {**valid, "layer_norm": False}, {**valid, "policy_lr": .5}):
                config.write_text(json.dumps(changed))
                with mock.patch.object(gates, "command", side_effect=["a" * 40, "", json.dumps(control)]), \
                        mock.patch.object(gates, "CONFIG", config):
                    if changed == valid and changed["layer_norm"] is True:
                        gates.verify_source("a" * 40)
                    else:
                        with self.assertRaises(RuntimeError):
                            gates.verify_source("a" * 40)

    def test_cuda_wrong_device_and_occupied_gpu_rejected(self):
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
            with self.assertRaises(RuntimeError):
                gates.verify_gpu()
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            for outputs in (["RTX"], ["Tesla P100", "GPU-123", "GPU-123, 77"],
                            ["Tesla P100", "GPU-123", "", "90"]):
                with mock.patch.object(gates, "command", side_effect=outputs):
                    with self.assertRaises(RuntimeError):
                        gates.verify_gpu()

    def test_missing_failed_or_skipped_cuda_receipts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gates, "receipt_dir", return_value=Path(tmp)):
            with self.assertRaises(FileNotFoundError):
                gates.verify_test_receipts("a" * 40)
            data = {"sha": "a" * 40, "experiment": "layernorm_only_8x256_v1", "successful": True, "tests": 100, "skipped": 0}
            for mode in ("cpu", "cuda"):
                (Path(tmp) / f"tests_{mode}.json").write_text(json.dumps(data))
            gates.verify_test_receipts("a" * 40)
            data["skipped"] = 1
            (Path(tmp) / "tests_cuda.json").write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                gates.verify_test_receipts("a" * 40)
            for key, value in (("successful", False), ("successful", 1), ("sha", "b" * 40),
                               ("tests", 0), ("experiment", "residual_only_8x256_v1")):
                changed = {**data, "skipped": 0, key: value}
                (Path(tmp) / "tests_cuda.json").write_text(json.dumps(changed))
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    gates.verify_test_receipts("a" * 40)

    def test_wrong_python_environment_rejected(self):
        with mock.patch.object(gates.sys, "executable", "wrong/python"):
            with self.assertRaises(RuntimeError):
                gates.verify_environment()

    def test_failed_rerun_invalidates_old_test_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gates, "receipt_dir", return_value=Path(tmp)):
            receipt = Path(tmp) / "tests_cuda.json"
            receipt.write_text('{"successful": true}')
            with mock.patch.object(gates, "verify_source"), mock.patch.object(gates, "verify_environment", side_effect=RuntimeError("bad env")):
                with self.assertRaises(RuntimeError):
                    gates.run_tests("a" * 40, "cuda")
            self.assertFalse(receipt.exists())

    def test_nonfinite_checkpoint_detection(self):
        import torch
        self.assertFalse(gates.finite_tree({"optimizer": [torch.tensor(float("nan"))]}))
        self.assertTrue(gates.finite_tree({"weights": torch.ones(3), "seed": 0}))

    def test_inputnorm_receipts_and_runner_cannot_be_reused(self):
        from scripts import inputnorm_gorina4, residual_gorina4
        self.assertNotEqual(gates.receipt_dir("a" * 40), inputnorm_gorina4.receipt_dir("a" * 40))
        self.assertNotEqual(gates.receipt_dir("a" * 40), residual_gorina4.receipt_dir("a" * 40))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gates, "receipt_dir", return_value=Path(tmp)):
            for mode in ("cpu", "cuda"):
                (Path(tmp) / f"tests_{mode}.json").write_text(json.dumps(
                    {"sha": "a" * 40, "successful": True, "tests": 154, "skipped": 0}))
            with self.assertRaisesRegex(RuntimeError, "experiment"):
                gates.verify_test_receipts("a" * 40)
        runner = (gates.ROOT / "scripts/run_layernorm_gorina4.sh").read_text()
        self.assertNotIn("inputnorm", runner.lower())
        self.assertNotIn("residual", runner.lower())
        self.assertIn('exec 9>results/layernorm_gpu0.lock', runner)
        self.assertIn('PIPE_CODES=("${PIPESTATUS[@]}")', runner)
        self.assertIn('kill -0 "$MONITOR_PID"', runner)
        self.assertIn('prior_audit 6200 0', runner)
        self.assertIn('prior_audit 26000 0', runner)
        self.assertIn('prior_audit 1000000 "$((SEED - 1))"', runner)

    def test_actual_architecture_rejects_naive_other_interventions_and_wrong_modules(self):
        import torch
        from src.sac_update import SACAgent
        from tests.test_sac_update import spaces
        valid = SACAgent(*spaces(), hidden_depth=8, layer_norm=True)
        gates.verify_architecture(valid)
        for agent in (SACAgent(*spaces(), hidden_depth=8),
                      SACAgent(*spaces(), hidden_depth=8, input_norm=True),
                      SACAgent(*spaces(), hidden_depth=8, residual_connections=True),
                      SACAgent(*spaces(), hidden_depth=4, layer_norm=True)):
            with self.assertRaises(RuntimeError):
                gates.verify_architecture(agent)
        for name in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
            for corruption in ("flag", "eps", "norm", "head", "extra", "shared", "target"):
                altered = copy.deepcopy(valid)
                net = getattr(altered, name)
                if corruption == "flag":
                    net.layer_norm = False
                elif corruption == "eps":
                    net.hidden_norms[0].eps = 1e-3
                elif corruption == "norm":
                    net.hidden_norms[0] = torch.nn.Identity()
                elif corruption == "head":
                    net.add_module("output_norm", torch.nn.LayerNorm(1))
                elif corruption == "extra":
                    net.add_module("extra", torch.nn.Linear(256, 256))
                elif corruption == "shared":
                    net.hidden_norms[1] = net.hidden_norms[0]
                else:
                    altered.qf1_target = altered.qf1
                with self.subTest(name=name, corruption=corruption), self.assertRaises(RuntimeError):
                    gates.verify_architecture(altered)

    def test_malformed_telemetry_rejected(self):
        header = "timestamp,uuid,name,utilization.gpu [%],memory.used [MiB],memory.total [MiB]\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            for row in ("", "now,GPU-synthetic,RTX,25 %,300 MiB,12000 MiB\n",
                        "now,GPU-synthetic,Tesla P100,NaN %,300 MiB,12000 MiB\n",
                        "now,GPU-synthetic,Tesla P100,25 %,13000 MiB,12000 MiB\n"):
                path.write_text(header + row)
                with self.assertRaises((RuntimeError, ValueError)):
                    gates.verify_telemetry(path, "Tesla P100")
