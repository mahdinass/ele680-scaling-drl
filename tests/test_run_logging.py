import contextlib
import csv
import dataclasses
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.evaluate import EvalResult
from src.run_logger import EPISODE_FIELDS, EVAL_FIELDS, TRAIN_FIELDS, RunLogger
from src.run_metadata import count_parameters, get_git_commit, get_git_dirty
from src.sac_update import SACAgent, grad_norm
from src.train import TrainConfig, load_config, main, run_experiment, train
from tests.test_evaluate import FakeEnv
from tests.test_sac_update import fake_batch, spaces

REQUIRED_METADATA = [
    "run_id", "status", "start_time", "end_time", "runtime_seconds", "failure_type",
    "failure_message", "git_commit", "python_version", "torch_version", "torch_cuda_runtime",
    "cuda_available", "gpu_name", "gymnasium_version", "mujoco_version", "env_id", "seed",
    "device", "config", "parameter_counts", "checkpoint_path",
]  # fmt: skip


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class RunTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def cfg(self, **overrides) -> TrainConfig:
        base = dict(
            seed=0, total_steps=40, learning_starts=10, batch_size=8, buffer_size=100,
            eval_interval=0, eval_episodes=1, final_eval_episodes=0, output_dir=self.tmp,
            run_id="run1", log_interval=5, checkpoint_interval=0, tensorboard=False,
        )  # fmt: skip
        base.update(overrides)
        return TrainConfig(**base)

    @property
    def run_dir(self) -> Path:
        return Path(self.tmp) / "run1"


fake_env_fn = lambda seed: FakeEnv(seed)


class RunDirectoryTests(RunTestCase):
    def test_run_directory_files_and_no_overwrite(self):
        run_experiment(self.cfg(), fake_env_fn)
        for name in [
            "config.json", "metadata.json", "status.json", "train_metrics.csv",
            "episode_metrics.csv", "eval_metrics.csv", "checkpoint.pt",
        ]:  # fmt: skip
            self.assertTrue((self.run_dir / name).is_file(), name)
        with self.assertRaises(FileExistsError):
            RunLogger(self.cfg())  # same run_id must never overwrite a previous run

    def test_generated_run_id_is_unique_and_descriptive(self):
        a = RunLogger(self.cfg(run_id=None))
        b = RunLogger(self.cfg(run_id=None))
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        self.assertNotEqual(a.run_id, b.run_id)
        self.assertTrue(a.run_id.startswith("HalfCheetah-v5_seed0_"))

    def test_config_serialization_roundtrip(self):
        cfg = self.cfg(seed=7, run_id="run1")
        run_experiment(cfg, fake_env_fn)
        saved = read_json(self.run_dir / "config.json")
        self.assertEqual(saved, dataclasses.asdict(cfg))
        path = Path(self.tmp) / "cfg.json"
        path.write_text(json.dumps(saved))
        self.assertEqual(load_config(str(path)), cfg)

    def test_load_config_rejects_unknown_keys(self):
        path = Path(self.tmp) / "bad.json"
        path.write_text(json.dumps({"total_steps": 5, "learning_rate": 1}))
        with self.assertRaises(ValueError):
            load_config(str(path))

    def test_baseline_config_file_loads(self):
        cfg = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "baseline.json"))
        self.assertEqual(cfg.env_id, "HalfCheetah-v5")
        self.assertEqual((cfg.total_steps, cfg.buffer_size, cfg.batch_size), (1_000_000, 1_000_000, 256))
        self.assertEqual((cfg.learning_starts, cfg.policy_frequency, cfg.target_network_frequency), (5000, 2, 1))
        self.assertEqual((cfg.eval_interval, cfg.eval_episodes, cfg.final_eval_episodes), (25000, 5, 10))


class MetadataTests(RunTestCase):
    def test_metadata_keys_status_and_parameter_counts(self):
        run_experiment(self.cfg(), fake_env_fn)
        meta = read_json(self.run_dir / "metadata.json")
        for key in REQUIRED_METADATA:
            self.assertIn(key, meta)
        self.assertEqual(meta["status"], "completed")
        self.assertIsNone(meta["failure_type"])
        self.assertIsNotNone(meta["end_time"])
        self.assertGreaterEqual(meta["runtime_seconds"], 0)
        self.assertEqual(meta["steps_completed"], 40)
        self.assertEqual(meta["env_id"], "HalfCheetah-v5")
        self.assertEqual(meta["torch_version"], torch.__version__)
        # FakeEnv: obs 3, action 2 -> hand-computed 2x256 parameter counts.
        pc = meta["parameter_counts"]
        self.assertEqual(pc["actor"], 67844)
        self.assertEqual(pc["q1"], 67585)
        self.assertEqual(pc["q2"], 67585)
        self.assertEqual(pc["total"], 67844 + 2 * 67585)
        status = read_json(self.run_dir / "status.json")
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["checkpoint_path"], meta["checkpoint_path"])

    def test_count_parameters_nonzero(self):
        counts = count_parameters(SACAgent(*spaces()))
        self.assertTrue(all(v > 0 for v in counts.values()))
        self.assertEqual(counts["total"], counts["actor"] + counts["q1"] + counts["q2"])

    def test_git_lookup_degrades_gracefully(self):
        with mock.patch("src.run_metadata.subprocess.run", side_effect=FileNotFoundError("git")):
            self.assertIsNone(get_git_commit())
            self.assertIsNone(get_git_dirty())
        commit = get_git_commit()  # real git: a 40-char hash, or None outside a repo
        self.assertTrue(commit is None or len(commit) == 40)

    def test_git_dirty_counts_tracked_changes_and_untracked_files(self):
        for output, expected in [("", False), ("?? new_file.py", True), (" M src/train.py", True)]:
            with self.subTest(output=output), mock.patch("src.run_metadata._git", return_value=output) as git:
                self.assertIs(get_git_dirty(), expected)
                git.assert_called_once_with("status", "--porcelain")  # no --untracked-files=no

    def test_run_still_works_without_git(self):
        with mock.patch("src.run_metadata.subprocess.run", side_effect=FileNotFoundError("git")):
            run_experiment(self.cfg(), fake_env_fn)
        self.assertIsNone(read_json(self.run_dir / "metadata.json")["git_commit"])


class MetricsTests(RunTestCase):
    def test_train_csv_schema_and_rows(self):
        # Update steps are global_step 11..39; rows at global_step % 5 == 0 -> 15,20,25,30,35.
        # Rows are labeled by env steps completed (global_step + 1).
        run_experiment(self.cfg(), fake_env_fn)
        path = self.run_dir / "train_metrics.csv"
        self.assertEqual(path.read_text().splitlines()[0].split(","), TRAIN_FIELDS)
        rows = read_csv(path)
        self.assertEqual([int(r["step"]) for r in rows], [16, 21, 26, 31, 36])
        for r in rows:
            for key in ("elapsed_sec", "critic_loss", "alpha", "q_mean_abs", "critic_grad_norm"):
                self.assertTrue(np.isfinite(float(r[key])), key)
            self.assertGreater(float(r["alpha"]), 0)
        # Actor update happens only when global_step % 2 == 0 (20, 30) -> others are blank.
        by_step = {int(r["step"]): r for r in rows}
        for step in (21, 31):
            for key in ("actor_loss", "alpha_loss", "actor_grad_norm"):
                self.assertTrue(np.isfinite(float(by_step[step][key])), key)
        for step in (16, 26, 36):
            for key in ("actor_loss", "alpha_loss", "actor_grad_norm"):
                self.assertEqual(by_step[step][key], "")

    def test_episode_csv(self):
        run_experiment(self.cfg(), fake_env_fn)  # FakeEnv episodes last 5 steps
        path = self.run_dir / "episode_metrics.csv"
        self.assertEqual(path.read_text().splitlines()[0].split(","), EPISODE_FIELDS)
        rows = read_csv(path)
        self.assertEqual([int(r["step"]) for r in rows], [5, 10, 15, 20, 25, 30, 35, 40])
        self.assertTrue(all(float(r["episode_return"]) == 5.0 and int(r["episode_length"]) == 5 for r in rows))

    def test_eval_csv_periodic_and_final(self):
        cfg = self.cfg(eval_interval=20, eval_episodes=2, final_eval_episodes=3)
        run_experiment(cfg, fake_env_fn)
        path = self.run_dir / "eval_metrics.csv"
        self.assertEqual(path.read_text().splitlines()[0].split(","), EVAL_FIELDS)
        rows = read_csv(path)
        self.assertEqual([(int(r["step"]), r["kind"], int(r["num_episodes"])) for r in rows],
                         [(20, "periodic", 2), (40, "periodic", 2), (40, "final", 3)])
        final = rows[-1]
        self.assertEqual(json.loads(final["returns"]), [5.0, 5.0, 5.0])
        self.assertEqual(float(final["mean_return"]), 5.0)
        self.assertEqual(float(final["std_return"]), 0.0)
        meta = read_json(self.run_dir / "metadata.json")
        self.assertEqual(meta["final_eval"]["num_episodes"], 3)
        self.assertEqual(meta["final_eval"]["returns"], [5.0, 5.0, 5.0])

    def test_eval_row_mean_std_and_per_episode_returns(self):
        logger = RunLogger(self.cfg())
        logger.log_eval(100, EvalResult([1.0, 2.0, 3.0], 2.0), "periodic")
        logger.close()
        (row,) = read_csv(self.run_dir / "eval_metrics.csv")
        self.assertEqual(json.loads(row["returns"]), [1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(row["mean_return"]), 2.0)
        self.assertAlmostEqual(float(row["std_return"]), float(np.std([1.0, 2.0, 3.0])))

    def test_diagnostics_do_not_change_training(self):
        # Same seed with full diagnostics logging vs. no logger at all -> identical learning.
        logged = run_experiment(self.cfg(log_interval=1), fake_env_fn)
        plain = train(self.cfg(run_id=None), fake_env_fn)
        for p, q in zip(logged.agent.actor.parameters(), plain.agent.actor.parameters()):
            self.assertTrue(torch.equal(p, q))
        for p, q in zip(logged.agent.qf1.parameters(), plain.agent.qf1.parameters()):
            self.assertTrue(torch.equal(p, q))
        self.assertTrue(torch.equal(logged.agent.log_alpha, plain.agent.log_alpha))


class AgentDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_only_when_requested(self):
        torch.manual_seed(0)
        agent = SACAgent(*spaces())
        batch = fake_batch()
        agent.update_critic(batch)
        agent.update_actor_and_alpha(batch)
        self.assertEqual(agent.diagnostics, {})
        agent.update_critic(batch, diagnostics=True)
        agent.update_actor_and_alpha(batch, diagnostics=True)
        self.assertEqual(set(agent.diagnostics), {"critic_grad_norm", "q_mean_abs", "actor_grad_norm"})
        self.assertTrue(all(np.isfinite(v) and v >= 0 for v in agent.diagnostics.values()))

    def test_grad_norm(self):
        w = torch.nn.Parameter(torch.zeros(2))
        b = torch.nn.Parameter(torch.zeros(1))
        w.grad, b.grad = torch.tensor([3.0, 0.0]), torch.tensor([4.0])
        self.assertAlmostEqual(grad_norm([w, b]), 5.0)
        self.assertEqual(grad_norm([torch.nn.Parameter(torch.zeros(1))]), 0.0)  # no grad


class CheckpointTests(RunTestCase):
    def test_final_checkpoint_contents_and_roundtrip(self):
        result = run_experiment(self.cfg(seed=3), fake_env_fn)  # checkpoint_interval=0 -> final only
        path = self.run_dir / "checkpoint.pt"
        state = torch.load(path, weights_only=True)
        self.assertEqual(
            set(state),
            {"actor", "qf1", "qf2", "qf1_target", "qf2_target", "actor_optimizer", "critic_optimizer",
             "alpha_optimizer", "log_alpha", "global_step", "config", "seed"},
        )  # fmt: skip
        self.assertEqual(state["global_step"], 40)
        self.assertEqual(state["seed"], 3)
        self.assertEqual(state["config"]["run_id"], "run1")
        self.assertGreater(len(state["actor_optimizer"]["state"]), 0)

        env = FakeEnv()
        fresh = SACAgent(env.observation_space, env.action_space)
        info = load_checkpoint(path, fresh)
        self.assertEqual(info["global_step"], 40)
        for p, q in zip(result.agent.actor.parameters(), fresh.actor.parameters()):
            self.assertTrue(torch.equal(p, q))
        for p, q in zip(result.agent.qf2_target.parameters(), fresh.qf2_target.parameters()):
            self.assertTrue(torch.equal(p, q))
        self.assertTrue(torch.equal(result.agent.log_alpha, fresh.log_alpha))
        self.assertFalse(list(Path(self.tmp).rglob("*.tmp")))  # atomic write left no temp files

    def test_periodic_checkpoints(self):
        with mock.patch("src.train.save_checkpoint", wraps=save_checkpoint) as save:
            run_experiment(self.cfg(checkpoint_interval=20), fake_env_fn)
        self.assertEqual([c.args[2] for c in save.call_args_list], [20, 40, 40])  # periodic x2 + final

    def test_no_periodic_checkpoint_when_disabled(self):
        with mock.patch("src.train.save_checkpoint", wraps=save_checkpoint) as save:
            run_experiment(self.cfg(checkpoint_interval=0), fake_env_fn)
        self.assertEqual(save.call_count, 1)  # final only


class BoomEnv(FakeEnv):
    """Raises on its 13th step."""

    n = 0

    def step(self, action):
        self.n += 1
        if self.n > 12:
            raise RuntimeError("boom")
        return super().step(action)


class FailureTests(RunTestCase):
    def test_failed_run_is_recorded_and_reraised(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            # Sparse logging: no train row is ever written, yet progress must be exact.
            run_experiment(self.cfg(log_interval=1000), lambda seed: BoomEnv(seed))
        status = read_json(self.run_dir / "status.json")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["failure_type"], "RuntimeError")
        self.assertEqual(status["failure_message"], "boom")
        self.assertEqual(status["steps_completed"], 12)  # 12 completed env steps, 13th raised
        meta = read_json(self.run_dir / "metadata.json")
        self.assertEqual(meta["status"], "failed")
        self.assertIsNotNone(meta["end_time"])
        # Metrics written before the failure are preserved.
        self.assertEqual(read_json(self.run_dir / "metadata.json")["steps_completed"], 12)
        self.assertEqual(len(read_csv(self.run_dir / "train_metrics.csv")), 0)  # log_interval=1000
        self.assertEqual(len(read_csv(self.run_dir / "episode_metrics.csv")), 2)  # steps 5 and 10


class LauncherTests(RunTestCase):
    def test_cli_tiny_real_halfcheetah_run(self):
        cfg = dataclasses.replace(
            self.cfg(), total_steps=1_000_000, run_id=None, output_dir="unused", device="cuda"
        )  # file values that the CLI flags below must override
        path = Path(self.tmp) / "cfg.json"
        path.write_text(json.dumps(dataclasses.asdict(cfg)))
        with contextlib.redirect_stdout(io.StringIO()):
            main([
                "--config", str(path), "--seed", "3", "--device", "cpu", "--total-steps", "30",
                "--output-dir", self.tmp, "--run-id", "cli",
            ])  # fmt: skip
        run_dir = Path(self.tmp) / "cli"
        saved = read_json(run_dir / "config.json")
        self.assertEqual((saved["seed"], saved["device"], saved["total_steps"]), (3, "cpu", 30))
        self.assertEqual(read_json(run_dir / "status.json")["status"], "completed")
        meta = read_json(run_dir / "metadata.json")
        self.assertEqual(meta["parameter_counts"]["actor"], 17 * 256 + 256 + 65792 + 2 * (256 * 6 + 6))
        self.assertEqual(meta["observation_shape"], [17])
        self.assertEqual(meta["action_shape"], [6])
        self.assertTrue((run_dir / "checkpoint.pt").is_file())


@unittest.skipUnless(importlib.util.find_spec("tensorboard"), "tensorboard not installed")
class TensorBoardTests(RunTestCase):
    def test_writer_initializes_writes_and_closes(self):
        run_experiment(self.cfg(tensorboard=True, log_interval=5), fake_env_fn)
        events = list((self.run_dir / "tensorboard").rglob("events.out.tfevents.*"))
        self.assertTrue(events)
        self.assertGreater(events[0].stat().st_size, 0)
        self.assertTrue(read_json(self.run_dir / "metadata.json")["tensorboard_enabled"])


if __name__ == "__main__":
    unittest.main()
