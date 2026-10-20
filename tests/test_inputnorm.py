"""InputNorm correctness, frozen-control equivalence and real local smoke gates."""

import copy
import dataclasses
import json
import random
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.evaluate import evaluate
from src.observation_normalization import ObservationNormalizer
from src.run_metadata import count_parameters
from src.sac_update import SACAgent
from src.train import TrainConfig, load_config, run_experiment, train
from tests.test_evaluate import FakeEnv
from tests.test_sac_update import fake_batch, spaces

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "6c674babc344e62566692e1a4fcf3611dc53a6cc"
NETWORKS = ("actor", "qf1", "qf2", "qf1_target", "qf2_target")


def tiny(**kwargs):
    values = dict(total_steps=24, learning_starts=5, batch_size=4, buffer_size=50,
                  eval_interval=8, eval_episodes=2, final_eval_episodes=3,
                  hidden_depth=8, input_norm=True, tensorboard=False)
    values.update(kwargs)
    return TrainConfig(**values)


def frozen_modules():
    """Load actual control source, including its train loop, without writing files."""
    saved = {name: sys.modules.get(name) for name in ("src.networks", "src.sac_update", "src.train")}
    loaded = {}
    try:
        for name in ("src.networks", "src.sac_update", "src.train"):
            source = subprocess.run(
                ["git", "show", f"{REFERENCE}:{name.replace('.', '/')}.py"],
                cwd=ROOT, capture_output=True, check=True, timeout=20,
            ).stdout.decode("utf-8")
            module = types.ModuleType(name)
            sys.modules[name] = module
            exec(compile(source, f"{REFERENCE}/{name}", "exec"), module.__dict__)
            loaded[name] = module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return loaded["src.sac_update"], loaded["src.train"]


class EqualityMixin:
    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for k in a:
                self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def snapshot(self, n):
        return {"count": n.count, "mean": n.mean.copy(), "m2": n.m2.copy(),
                "cached_mean": n.cached_mean.clone(), "cached_inv_std": n.cached_inv_std.clone()}


class StatisticsTests(EqualityMixin, unittest.TestCase):
    def test_mean_population_variance_and_float32_collection_precision(self):
        x = np.random.default_rng(4).normal(size=(200, 3)).astype(np.float32)
        n = ObservationNormalizer(3)
        for row in x:
            n.update(row.astype(np.float64) + 1e-12)
        self.assertEqual(n.count, 200)
        np.testing.assert_allclose(n.mean, x.astype(np.float64).mean(0), atol=1e-14)
        np.testing.assert_allclose(n.m2 / n.count, x.astype(np.float64).var(0), atol=1e-14)

    def test_identity_low_count_zero_variance_and_shapes(self):
        n = ObservationNormalizer(3)
        x = torch.tensor([1., 2., 3.])
        self.assertTrue(torch.equal(n.transform(x), x))
        n.update(x.numpy())
        self.assertTrue(torch.equal(n.transform(x), torch.zeros(3)))
        n.update(x.numpy())
        for value in (x, x.repeat(4, 1)):
            out = n.transform(value)
            self.assertEqual(out.shape, value.shape)
            self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(n.transform(x + 1), torch.full((3,), 10000.))

    def test_large_offset_small_variance(self):
        x = (10000 + np.arange(100, dtype=np.float32)[:, None] / 100).repeat(3, 1)
        n = ObservationNormalizer(3)
        for row in x:
            n.update(row)
        np.testing.assert_allclose(n.m2 / n.count, x.astype(np.float64).var(0), rtol=1e-10)

    def test_invalid_inputs_do_not_mutate(self):
        n = ObservationNormalizer(3)
        before = self.snapshot(n)
        for value in (np.zeros((2, 3)), np.ones(4), [np.nan, 0, 1], [np.inf, 0, 1]):
            with self.assertRaises(ValueError):
                n.update(value)
            self.assert_nested_equal(before, self.snapshot(n))
        for value in (torch.zeros(4), torch.zeros(1, 1, 3), torch.ones(3, dtype=torch.float64),
                      torch.tensor([float('nan'), 0, 0])):
            with self.assertRaises(ValueError):
                n.transform(value)

    def test_snapshot_has_independent_storage_and_no_update_method(self):
        n = ObservationNormalizer(3)
        n.update(np.ones(3))
        frozen = n.snapshot()
        self.assertFalse(hasattr(frozen, "update"))
        x = torch.ones(3)
        expected = frozen.transform(x).clone()
        n.update(np.zeros(3))
        self.assertTrue(torch.equal(frozen.transform(x), expected))
        frozen.mean.add_(100)
        self.assertFalse(torch.equal(frozen.mean, n.cached_mean))

    def test_state_roundtrip_and_corrupt_state_rejected(self):
        n = ObservationNormalizer(3)
        n.update(np.ones(3)); n.update(np.arange(3))
        other = ObservationNormalizer(3)
        other.load_state_dict(n.state_dict())
        self.assert_nested_equal(self.snapshot(n), self.snapshot(other))
        for key, value in (("count", -1), ("count", True), ("epsilon", 1e-4),
                           ("m2", torch.full((3,), -1., dtype=torch.float64)),
                           ("mean", torch.zeros(4, dtype=torch.float64))):
            bad = copy.deepcopy(n.state_dict()); bad[key] = value
            with self.assertRaises(ValueError):
                other.load_state_dict(bad)
            self.assert_nested_equal(self.snapshot(n), self.snapshot(other))

    def test_one_million_decision_updates_count(self):
        # Estimator stress test only: no policy optimization/environment experiment.
        n = ObservationNormalizer(1)
        x = np.zeros(1, dtype=np.float32)
        for _ in range(1_000_000):
            n.update(x)
        self.assertEqual(n.count, 1_000_000)
        np.testing.assert_array_equal(n.mean, x)
        np.testing.assert_array_equal(n.m2, x)


class AgentTests(EqualityMixin, unittest.TestCase):
    def agent(self, device="cpu"):
        agent = SACAgent(*spaces(), device=device, hidden_depth=8, input_norm=True)
        for x in (np.arange(17), np.arange(17) + 4):
            agent.observation_normalizer.update(x)
        return agent

    def test_all_forward_paths_use_same_statistics_actions_unchanged(self):
        agent = self.agent()
        batch = fake_batch()
        raw = [x.clone() for x in batch]
        n = agent.observation_normalizer
        before = self.snapshot(n)
        obs, nxt = n.transform(batch.obs), n.transform(batch.next_obs)
        seen = {name: [] for name in NETWORKS}
        handles = []
        for name in NETWORKS:
            def hook(module, args, name=name):
                seen[name].append(tuple(x.detach().clone() for x in args))
            handles.append(getattr(agent, name).register_forward_pre_hook(hook))
        try:
            agent.update_critic(batch, diagnostics=True)
            agent.update_actor_and_alpha(batch, diagnostics=True)
            agent.update_targets()
        finally:
            for h in handles:
                h.remove()
        self.assertEqual(len(seen["actor"]), 3)  # target, actor, fresh alpha
        for args, expected in zip(seen["actor"], (nxt, obs, obs)):
            self.assertTrue(torch.equal(args[0], expected))
        for name in ("qf1_target", "qf2_target"):
            self.assertEqual(len(seen[name]), 1)
            self.assertTrue(torch.equal(seen[name][0][0], nxt))
        self.assertTrue(torch.equal(seen["qf1_target"][0][1], seen["qf2_target"][0][1]))
        for name in ("qf1", "qf2"):
            self.assertEqual(len(seen[name]), 3)  # loss, diagnostic, actor
            for args in seen[name]:
                self.assertTrue(torch.equal(args[0], obs))
            for args in seen[name][:2]:
                self.assertTrue(torch.equal(args[1], batch.actions))
        self.assert_nested_equal(before, self.snapshot(n))
        self.assert_nested_equal(raw, list(batch))

    def test_architecture_and_initialization_unchanged(self):
        for depth, total in ((2, 217870), (8, 1402126)):
            torch.manual_seed(5)
            a = SACAgent(*spaces(), hidden_depth=depth)
            rng = torch.get_rng_state()
            torch.manual_seed(5)
            b = SACAgent(*spaces(), hidden_depth=depth, input_norm=True)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(count_parameters(a), count_parameters(b))
            self.assertEqual(count_parameters(b)["total"], total)
            for name in NETWORKS:
                self.assertEqual(2 + len(getattr(b, name).hidden_extra), depth)
                self.assert_nested_equal(getattr(a, name).state_dict(), getattr(b, name).state_dict())

    def test_periodic_and_final_evaluation_preserve_every_training_statistic(self):
        agent = self.agent()
        n = agent.observation_normalizer
        before = self.snapshot(n)
        weights = copy.deepcopy(agent.actor.state_dict())
        rng = torch.get_rng_state().clone()
        # Explicitly use 5 periodic and 10 final episodes, not just no_grad.
        for episodes in (5, 10):
            env = FakeEnv()
            # Agent's spaces are 17/6; override FakeEnv shape without changing its reset semantics.
            env._obs = lambda: np.full(17, env.t + env.episode, dtype=np.float32)
            with mock.patch.object(n, "update", side_effect=AssertionError("evaluation updated statistics")):
                evaluate(agent.actor, env, episodes, 10000, observation_transform=n.snapshot())
            self.assert_nested_equal(before, self.snapshot(n))
        self.assert_nested_equal(weights, agent.actor.state_dict())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_config_diff_and_strict_boolean(self):
        control = json.loads((ROOT / "configs/depth_8x256.json").read_text())
        proposed = json.loads((ROOT / "configs/ablation_inputnorm_8x256.json").read_text())
        self.assertEqual(proposed.pop("input_norm"), True)
        self.assertEqual(proposed, control)
        self.assertTrue(load_config(str(ROOT / "configs/ablation_inputnorm_8x256.json")).input_norm)
        for invalid in (1, "true", None):
            with self.assertRaises(ValueError):
                SACAgent(*spaces(), input_norm=invalid)

    @unittest.skipUnless(torch.cuda.is_available(), "BLOCKED: CUDA unavailable locally")
    def test_cuda_normalization_updates_evaluation_and_checkpoint(self):
        agent = self.agent("cuda")
        agent.update_critic(fake_batch("cuda"), diagnostics=True)
        agent.update_actor_and_alpha(fake_batch("cuda"), diagnostics=True)
        agent.update_targets()
        n = agent.observation_normalizer
        self.assertEqual(n.cached_mean.device.type, "cuda")
        before = self.snapshot(n)
        env = FakeEnv(); env._obs = lambda: np.ones(17, dtype=np.float32)
        evaluate(agent.actor, env, 1, 10000, "cuda", observation_transform=n.snapshot())
        self.assert_nested_equal(before, self.snapshot(n))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, agent, 2, TrainConfig(input_norm=True, hidden_depth=8), 0)
            restored = self.agent("cuda")
            load_checkpoint(path, restored)
            self.assert_nested_equal(before, self.snapshot(restored.observation_normalizer))


class IntegrationTests(EqualityMixin, unittest.TestCase):
    def test_training_timing_raw_replay_and_evaluation_isolation(self):
        traces = []
        original = ObservationNormalizer.update
        def update(n, obs):
            traces.append(np.asarray(obs).copy())
            original(n, obs)
        with mock.patch.object(ObservationNormalizer, "update", update):
            result = train(tiny(), FakeEnv)
        self.assertEqual(len(traces), 24)
        self.assertEqual(result.agent.observation_normalizer.count, 24)
        self.assertEqual((result.num_random_actions, result.num_policy_actions), (5, 19))
        self.assertEqual((result.num_critic_updates, result.num_actor_updates, result.num_target_updates), (18, 18, 18))
        np.testing.assert_array_equal(np.array(traces), result.replay_buffer.obs[:24])
        # End of episode uses true next_obs; following decision uses reset obs.
        self.assertFalse(np.array_equal(result.replay_buffer.next_obs[4], traces[5]))
        self.assertEqual([s for s, _ in result.eval_results], [8, 16, 24])

    def test_first_policy_action_has_count_5001(self):
        from src.networks import Actor
        counts = []
        active = []
        original_update = ObservationNormalizer.update
        original_action = Actor.get_action
        def update(n, obs):
            active[:] = [n]; original_update(n, obs)
        def action(actor, obs):
            counts.append(active[0].count)
            return original_action(actor, obs)
        with mock.patch.object(ObservationNormalizer, "update", update), mock.patch.object(Actor, "get_action", action):
            result = train(tiny(total_steps=5001, learning_starts=5000, buffer_size=5001,
                                eval_interval=0, final_eval_episodes=0), FakeEnv)
        self.assertEqual(counts, [5001])
        self.assertEqual(result.num_critic_updates, 0)

    def test_enabling_evaluation_does_not_change_training(self):
        a = train(tiny(), FakeEnv)
        b = train(tiny(eval_interval=0, final_eval_episodes=0), FakeEnv)
        self.assert_nested_equal(self.snapshot(a.agent.observation_normalizer), self.snapshot(b.agent.observation_normalizer))
        for name in NETWORKS:
            self.assert_nested_equal(getattr(a.agent, name).state_dict(), getattr(b.agent, name).state_dict())
        self.assert_nested_equal(a.replay_buffer.obs, b.replay_buffer.obs)

    def test_training_evaluator_receives_independent_snapshot_for_both_kinds(self):
        import src.train as training
        original_evaluate = training.evaluate
        observations = []
        active = []
        original_update = ObservationNormalizer.update
        def update(n, obs):
            active[:] = [n]; original_update(n, obs)
        def checked(*args, **kwargs):
            n = active[0]; before = self.snapshot(n)
            frozen = kwargs["observation_transform"]
            self.assertNotEqual(frozen.mean.data_ptr(), n.cached_mean.data_ptr())
            with mock.patch.object(n, "update", side_effect=AssertionError("statistics leaked")):
                result = original_evaluate(*args, **kwargs)
            self.assert_nested_equal(before, self.snapshot(n))
            observations.append(args[2])
            return result
        with mock.patch.object(ObservationNormalizer, "update", update), mock.patch.object(training, "evaluate", checked):
            result = train(tiny(total_steps=8, eval_interval=8, eval_episodes=5, final_eval_episodes=10), FakeEnv)
        self.assertEqual(observations, [5, 10])
        self.assertEqual(result.agent.observation_normalizer.count, 8)

    def test_real_halfcheetah_cpu_smoke(self):
        result = train(tiny(total_steps=8, learning_starts=2, eval_interval=4,
                            eval_episodes=1, final_eval_episodes=1))
        self.assertEqual(result.agent.observation_normalizer.count, 8)
        self.assertEqual(result.steps_completed, 8)
        self.assertGreater(result.num_critic_updates, 0)
        self.assertTrue(np.isfinite(result.final_eval.mean_return))


class CheckpointTests(EqualityMixin, unittest.TestCase):
    def test_enabled_roundtrip_inference_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny(output_dir=tmp, run_id="inputnorm", checkpoint_interval=8)
            trained = run_experiment(cfg, FakeEnv).agent
            fresh = SACAgent(FakeEnv.observation_space, FakeEnv.action_space, hidden_depth=8, input_norm=True)
            path = Path(tmp) / "inputnorm/checkpoint.pt"
            info = load_checkpoint(path, fresh)
            self.assertEqual(info["global_step"], cfg.total_steps)
            self.assert_nested_equal(self.snapshot(trained.observation_normalizer), self.snapshot(fresh.observation_normalizer))
            obs = torch.ones(2, 3); actions = torch.zeros(2, 2)
            a, b = trained.normalize_observation(obs), fresh.normalize_observation(obs)
            self.assert_nested_equal(trained.actor.get_deterministic_action(a), fresh.actor.get_deterministic_action(b))
            for name in ("qf1", "qf2", "qf1_target", "qf2_target"):
                self.assert_nested_equal(getattr(trained, name)(a, actions), getattr(fresh, name)(b, actions))
            meta = json.loads((path.parent / "metadata.json").read_text())
            self.assertEqual(meta["input_normalization"]["count"], 24)

    def test_mismatches_fail_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            for enabled in (False, True):
                source = SACAgent(*spaces(), hidden_depth=8, input_norm=enabled)
                save_checkpoint(path, source, 0, TrainConfig(hidden_depth=8, input_norm=enabled), 0)
                target = SACAgent(*spaces(), hidden_depth=8, input_norm=not enabled)
                before = copy.deepcopy(target.actor.state_dict())
                with self.assertRaises(ValueError):
                    load_checkpoint(path, target)
                self.assert_nested_equal(before, target.actor.state_dict())
            state = torch.load(path, weights_only=True)
            del state["observation_normalization"]
            torch.save(state, path)
            target = SACAgent(*spaces(), hidden_depth=8, input_norm=True)
            before = copy.deepcopy(target.actor.state_dict())
            with self.assertRaises(ValueError):
                load_checkpoint(path, target)
            self.assert_nested_equal(before, target.actor.state_dict())


class FrozenCompatibilityTests(EqualityMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Missing reference is a failure, never silently skipped.
        cls.old_sac, cls.old_train = frozen_modules()

    def test_disabled_initialization_rng_and_full_update_trajectory(self):
        for depth in (2, 8):
            torch.manual_seed(22)
            old = self.old_sac.SACAgent(*spaces(), hidden_depth=depth)
            old_rng = torch.get_rng_state()
            torch.manual_seed(22)
            new = SACAgent(*spaces(), hidden_depth=depth, input_norm=False)
            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            x = torch.ones(2, 17)
            self.assertIs(new.normalize_observation(x), x)
            batch = fake_batch()
            for step in range(3):
                torch.manual_seed(100 + step)
                old_action = old.actor.get_action(x)
                old_losses = [old.update_critic(batch, True), *old.update_actor_and_alpha(batch, True)]
                old.update_targets(); old_rng = torch.get_rng_state()
                torch.manual_seed(100 + step)
                new_action = new.actor.get_action(x)
                new_losses = [new.update_critic(batch, True), *new.update_actor_and_alpha(batch, True)]
                new.update_targets()
                self.assert_nested_equal(old_action, new_action)
                self.assertEqual(old_losses, new_losses)
                self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            for name in NETWORKS:
                self.assert_nested_equal(getattr(old, name).state_dict(), getattr(new, name).state_dict())
            for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
                self.assert_nested_equal(getattr(old, name).state_dict(), getattr(new, name).state_dict())
            self.assert_nested_equal(old.log_alpha, new.log_alpha)

    def test_disabled_training_matches_actual_frozen_loop_both_depths(self):
        for depth in (2, 8):
            cfg = tiny(hidden_depth=depth, input_norm=False)
            old_cfg = self.old_train.TrainConfig(**{k: v for k, v in dataclasses.asdict(cfg).items()
                                                  if k not in ("input_norm", "residual_connections", "layer_norm")})
            a = self.old_train.train(old_cfg, FakeEnv)
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            b = train(cfg, FakeEnv)
            self.assert_nested_equal(rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
            for name in NETWORKS:
                self.assert_nested_equal(getattr(a.agent, name).state_dict(), getattr(b.agent, name).state_dict())
            for name in ("obs", "next_obs", "actions", "rewards", "terminations", "truncations"):
                self.assert_nested_equal(getattr(a.replay_buffer, name), getattr(b.replay_buffer, name))
            for name in ("num_random_actions", "num_policy_actions", "num_critic_updates", "num_actor_updates", "num_target_updates", "eval_results", "final_eval"):
                self.assertEqual(getattr(a, name), getattr(b, name))

    def test_old_checkpoint_without_flag_loads_and_disabled_weights_load_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            old = self.old_sac.SACAgent(*spaces(), hidden_depth=8)
            cfg = TrainConfig(hidden_depth=8)
            fresh = SACAgent(*spaces(), hidden_depth=8)
            save_checkpoint(path, fresh, 1, cfg, 0)
            state = torch.load(path, weights_only=True)
            state["config"].pop("input_norm")
            for name in NETWORKS:
                state[name] = getattr(old, name).state_dict()
            torch.save(state, path)
            load_checkpoint(path, fresh)
            for name in NETWORKS:
                getattr(old, name).load_state_dict(getattr(fresh, name).state_dict(), strict=True)
                self.assert_nested_equal(getattr(old, name).state_dict(), getattr(fresh, name).state_dict())


if __name__ == "__main__":
    unittest.main()
