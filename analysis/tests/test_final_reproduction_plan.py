"""Validate historical reproduction commands without running any training."""
import contextlib
import importlib.util
import io
import json
import shlex
import subprocess
import unittest
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("final_reproduce", ROOT / "analysis/final_24run_reproduce.py")
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


@lru_cache(maxsize=None)
def historical(sha, path):
    return subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=ROOT).decode("utf-8-sig")


class HistoricalReproductionPlan(unittest.TestCase):
    def test_all_24_commands_match_recorded_sources_and_configs(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            reproduce.plan()
        lines = output.getvalue().splitlines()
        checkouts = [line for line in lines if line.startswith("git switch")]
        commands = [line for line in lines if line.startswith("CUDA_VISIBLE_DEVICES=")]
        rows = reproduce.read_csv(reproduce.DATA / "run_inventory.csv")
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(checkouts), 24)
        self.assertEqual(len(commands), 24)
        self.assertEqual({(r["architecture"], int(r["seed"])) for r in rows},
                         {(a, s) for a in reproduce.ORDER for s in reproduce.SEEDS})
        for row, checkout, command in zip(rows, checkouts, commands):
            with self.subTest(run_id=row["run_id"]):
                sha = row["git_commit"]
                self.assertEqual(checkout, f"git switch --detach {sha}")
                parts = shlex.split(command)
                self.assertEqual(parts[:4], ["CUDA_VISIBLE_DEVICES=0", "python", "-m", "src.train"])
                opts = dict(zip(parts[4::2], parts[5::2]))
                config_path = "configs/" + reproduce.CONFIGS[row["architecture"]]
                self.assertEqual(opts["--config"], config_path)
                self.assertEqual(opts["--seed"], row["seed"])
                self.assertEqual(opts["--device"], "cuda")
                self.assertNotEqual(opts["--run-id"], row["run_id"])
                self.assertEqual(opts["--output-dir"], "results/reproduced")
                config = json.loads(historical(sha, config_path))
                self.assertEqual(config, json.loads((ROOT / config_path).read_text()))
                self.assertEqual(config["total_steps"], 1_000_000)
                self.assertEqual(config["env_id"], "HalfCheetah-v5")
                for option in opts:
                    self.assertIn(option, historical(sha, "src/train.py"))

    def test_new_output_root_is_ignored_by_every_historical_source(self):
        rows = reproduce.read_csv(reproduce.DATA / "run_inventory.csv")
        for sha in sorted({row["git_commit"] for row in rows}):
            with self.subTest(commit=sha):
                self.assertIn("results/", historical(sha, ".gitignore").splitlines())


if __name__ == "__main__":
    unittest.main()
