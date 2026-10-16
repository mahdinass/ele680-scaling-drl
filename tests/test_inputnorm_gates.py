"""Fail-closed validation of manual server gate helpers, without server access."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import inputnorm_gorina4 as gates


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

    def test_config_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "bad.json"
            config.write_text('{"input_norm": true, "hidden_depth": 4}')
            with mock.patch.object(gates, "command", side_effect=["a" * 40, ""]), mock.patch.object(gates, "CONFIG", config):
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
            data = {"sha": "a" * 40, "successful": True, "tests": 100, "skipped": 0}
            for mode in ("cpu", "cuda"):
                (Path(tmp) / f"tests_{mode}.json").write_text(json.dumps(data))
            gates.verify_test_receipts("a" * 40)
            data["skipped"] = 1
            (Path(tmp) / "tests_cuda.json").write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
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
