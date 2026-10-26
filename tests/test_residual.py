"""Residual-only behavioral, frozen-control, checkpoint and training tests."""

import copy
import dataclasses
import inspect
import json
import operator
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from src.checkpoint import load_checkpoint, save_checkpoint
from src.evaluate import evaluate
from src.networks import Actor, QNetwork
from src.run_metadata import count_parameters
from src.sac_update import SACAgent
from src.train import TrainConfig, load_config, run_experiment, train
from tests.test_evaluate import FakeEnv
from tests.test_inputnorm import EqualityMixin, frozen_modules
from tests.test_sac_update import fake_batch, spaces

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ("actor", "qf1", "qf2", "qf1_target", "qf2_target")


def tiny(**kwargs):
    values = dict(total_steps=24, learning_starts=5, batch_size=4, buffer_size=50,
                  eval_interval=8, eval_episodes=5, final_eval_episodes=10,
                  hidden_depth=8, residual_connections=True, tensorboard=False)
    values.update(kwargs)
    return TrainConfig(**values)


def agent_snapshot(agent):
    return copy.deepcopy({
        **{name: getattr(agent, name).state_dict() for name in NETWORKS},
        **{name: getattr(agent, name).state_dict() for name in
           ("actor_optimizer", "critic_optimizer", "alpha_optimizer")},
        "log_alpha": agent.log_alpha,
        "normalizer": None if agent.observation_normalizer is None else agent.observation_normalizer.state_dict(),
    })


class ArchitectureTests(EqualityMixin, unittest.TestCase):
    def test_exact_seven_shortcuts_eight_linears_and_eight_relus(self):
        for kind in (Actor, QNetwork):
            net = kind(*spaces(), hidden_depth=8, residual_connections=True)
            graph = torch.fx.symbolic_trace(net).graph
            adds = [n for n in graph.nodes if n.op == "call_function" and n.target is operator.add
                    and all(isinstance(a, torch.fx.Node) for a in n.args)]
            self.assertEqual(len(adds), 7)
            self.assertEqual(sum(n.op == "call_function" and n.target is F.relu for n in graph.nodes), 8)
            self.assertEqual([n.args[0].target for n in adds], ["fc2", *[f"hidden_extra.{i}" for i in range(6)]])
            hidden = [net.fc1, net.fc2, *net.hidden_extra]
            self.assertEqual(len(hidden), 8)
            self.assertTrue(all(type(m) is torch.nn.Linear and m.out_features == 256 for m in hidden))
            self.assertTrue(all(m.in_features == 256 for m in hidden[1:]))

    def test_every_shortcut_matches_post_addition_relu(self):
        for kind in (Actor, QNetwork):
            net = kind(*spaces(), hidden_depth=8, residual_connections=True)
            records = []
            handles = [layer.register_forward_hook(
                lambda module, args, out: records.append((args[0].detach().clone(), out.detach().clone())))
                for layer in [net.fc1, net.fc2, *net.hidden_extra]]
            head_inputs = []
            head = net.fc_mean if kind is Actor else net.fc3
            handles.append(head.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0].detach())))
            try:
                x = torch.randn(3, 17)
                net(x) if kind is Actor else net(x, torch.randn(3, 6))
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(len(records), 8)
            self.assert_nested_equal(records[1][0], F.relu(records[0][1]))
            for i in range(1, 8):
                following = records[i + 1][0] if i < 7 else head_inputs[0]
                self.assert_nested_equal(following, F.relu(records[i][1] + records[i][0]))

    def test_identity_gradient_survives_seven_zero_transformations(self):
        for kind in (Actor, QNetwork):
            net = kind(*spaces(), hidden_depth=8, residual_connections=True)
            with torch.no_grad():
                net.fc1.weight.fill_(0.01)
                net.fc1.bias.fill_(1)
                for layer in [net.fc2, *net.hidden_extra]:
                    layer.weight.zero_()
                    layer.bias.zero_()
                head = net.fc_mean if kind is Actor else net.fc3
                head.weight.fill_(1)
                head.bias.zero_()
            obs = torch.ones(2, 17, requires_grad=True)
            output = net(obs)[0] if kind is Actor else net(obs, torch.ones(2, 6))
            output.sum().backward()
            multiplier = 6 if kind is Actor else 1
            self.assertTrue(torch.allclose(obs.grad, torch.full_like(obs, 256 * 0.01 * multiplier)))
            for layer in [net.fc1, net.fc2, *net.hidden_extra]:
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_counts_keys_initialization_rng_targets_and_no_normalization(self):
        for enabled in (False, True):
            torch.manual_seed(42)
            agent = SACAgent(*spaces(), hidden_depth=8, residual_connections=enabled)
            rng = torch.get_rng_state()
            self.assertEqual(count_parameters(agent), {"actor": 468236, "q1": 466945, "q2": 466945, "total": 1402126})
            self.assertIsNone(agent.observation_normalizer)
            x = torch.ones(2, 17)
            self.assertIs(agent.normalize_observation(x), x)
            for name in NETWORKS:
                net = getattr(agent, name)
                self.assertIs(net.residual_connections, enabled)
                self.assertTrue(all(type(m) in (type(net), torch.nn.Linear, torch.nn.ModuleList) for m in net.modules()))
            if not enabled:
                plain, plain_rng = agent_snapshot(agent), rng
            else:
                self.assert_nested_equal(plain, agent_snapshot(agent))
                self.assert_nested_equal(plain_rng, rng)
                for online, target in ((agent.qf1, agent.qf1_target), (agent.qf2, agent.qf2_target)):
                    self.assert_nested_equal(online.state_dict(), target.state_dict())
                    self.assertTrue(all(p.data_ptr() != t.data_ptr() and not t.requires_grad
                                        for p, t in zip(online.parameters(), target.parameters())))

    def test_strict_flag_validation_at_every_public_entry(self):
        for value in (0, 1, "true", None, np.bool_(True), [], 0.0):
            for constructor, args in ((Actor, spaces()), (QNetwork, spaces()), (SACAgent, spaces()), (TrainConfig, ())):
                with self.subTest(value=value, constructor=constructor), self.assertRaises(ValueError):
                    constructor(*args, residual_connections=value)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                if not isinstance(value, np.bool_):
                    path.write_text(json.dumps({"residual_connections": value}))
                    with self.assertRaises(ValueError):
                        load_config(str(path))

    def test_historical_positional_arguments_preserved(self):
        old_sac, old_train = frozen_modules()
        old_fields = [f.name for f in dataclasses.fields(old_train.TrainConfig)]
        self.assertEqual([f.name for f in dataclasses.fields(TrainConfig)], old_fields + ["input_norm", "residual_connections"])
        names = list(inspect.signature(old_sac.SACAgent).parameters)
        self.assertEqual(list(inspect.signature(SACAgent).parameters), names + ["input_norm", "residual_connections"])
        agent = SACAgent(*spaces(), "cpu", .99, .005, 3e-4, 1e-3, 256, 8, False, True)
        self.assertTrue(agent.actor.residual_connections)
        self.assertTrue(Actor(*spaces(), 256, 8, True).residual_connections)
        self.assertTrue(QNetwork(*spaces(), 256, 8, True).residual_connections)

    def _finite_updates(self, device):
        agent = SACAgent(*spaces(), device=device, hidden_depth=8, residual_connections=True)
        batch = fake_batch(device)
        action, log_pi = agent.actor.get_action(batch.obs)
        self.assertEqual(action.shape, (32, 6))
        self.assertEqual(log_pi.shape, (32, 1))
        self.assertTrue(torch.isfinite(action).all() and torch.isfinite(log_pi).all())
        self.assertTrue((action.abs() <= 1).all())
        self.assertEqual(agent.qf1(batch.obs, batch.actions).shape, (32, 1))
        before = agent_snapshot(agent)
        losses = [agent.update_critic(batch, True), *agent.update_actor_and_alpha(batch, True)]
        self.assertTrue(np.isfinite(losses).all())
        for name in ("actor", "qf1", "qf2"):
            for layer in [getattr(agent, name).fc1, getattr(agent, name).fc2, *getattr(agent, name).hidden_extra]:
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        for name in ("qf1_target", "qf2_target"):
            self.assert_nested_equal(before[name], getattr(agent, name).state_dict())
            self.assertTrue(all(p.grad is None for p in getattr(agent, name).parameters()))
        agent.update_targets()
        self.assertFalse(torch.equal(before["qf1_target"]["hidden_extra.5.weight"], agent.qf1_target.hidden_extra[5].weight))
        return agent

    def test_finite_cpu_forward_backward_and_target_update(self):
        self._finite_updates("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward_checkpoint_and_evaluation(self):
        agent = self._finite_updates("cuda")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, agent, 24, tiny(device="cuda"), 0)
            fresh = SACAgent(*spaces(), device="cuda", hidden_depth=8, residual_connections=True)
            load_checkpoint(path, fresh)
            cuda_device = fresh.log_alpha.device  # resolve "cuda" to the actual CUDA index
            self.assertEqual(cuda_device.type, "cuda")
            optimizer_names = ("actor_optimizer", "critic_optimizer", "alpha_optimizer")
            relocated_steps = dict.fromkeys(optimizer_names, 0)

            def assert_restored_equal(original, restored, path=()):
                if isinstance(original, torch.Tensor):
                    self.assertEqual(original.dtype, restored.dtype, path)
                    self.assertEqual(original.shape, restored.shape, path)
                    # PyTorch 2.5.1 Adam initializes these counters on CPU, but
                    # load_state_dict preserves torch.load(map_location=cuda)'s
                    # relocation for non-capturable/non-fused step tensors.
                    if (len(path) == 4 and path[0] in optimizer_names
                            and path[1] == "state" and path[3] == "step"):
                        self.assertEqual(original.shape, torch.Size([]), path)
                        self.assertEqual(original.device, torch.device("cpu"), path)
                        self.assertEqual(restored.device, cuda_device, path)
                        self.assertTrue(torch.equal(original.cpu(), restored.cpu()), path)
                        relocated_steps[path[0]] += 1
                    else:
                        self.assertEqual(original.device, cuda_device, path)
                        self.assertEqual(restored.device, original.device, path)
                        self.assertTrue(torch.equal(original, restored), path)
                elif isinstance(original, dict):
                    self.assertEqual(set(original), set(restored), path)
                    for key in original:
                        assert_restored_equal(original[key], restored[key], (*path, key))
                elif isinstance(original, (list, tuple)):
                    self.assertEqual(type(original), type(restored), path)
                    self.assertEqual(len(original), len(restored), path)
                    for index, (a, b) in enumerate(zip(original, restored)):
                        assert_restored_equal(a, b, (*path, index))
                else:
                    self.assertEqual(original, restored, path)

            assert_restored_equal(agent_snapshot(agent), agent_snapshot(fresh))
            self.assertEqual(relocated_steps, {"actor_optimizer": 20, "critic_optimizer": 36, "alpha_optimizer": 1})

            # Exercise the loaded CUDA optimizer state, not only a fresh agent.
            steps_before = {name: [state["step"].item() for state in getattr(fresh, name).state.values()]
                            for name in optimizer_names}
            batch = fake_batch(fresh.device)
            losses = [fresh.update_critic(batch), *fresh.update_actor_and_alpha(batch)]
            self.assertTrue(np.isfinite(losses).all())
            for name in optimizer_names:
                states = list(getattr(fresh, name).state.values())
                self.assertEqual([state["step"].item() for state in states],
                                 [step + 1 for step in steps_before[name]])
                self.assertTrue(all(torch.isfinite(value).all() for state in states for value in state.values()))
                self.assertTrue(all(state[key].device == cuda_device for state in states
                                    for key in ("exp_avg", "exp_avg_sq")))
            for network in NETWORKS:
                self.assertTrue(all(torch.isfinite(p).all() for p in getattr(fresh, network).parameters()))
            self.assertTrue(torch.isfinite(fresh.log_alpha).all())
        result = train(tiny(device="cuda"), FakeEnv)
        self.assertEqual(result.steps_completed, 24)


class FrozenControlTests(EqualityMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Missing frozen git source is a failure, never a skip. Includes actual networks.
        cls.old_sac, cls.old_train = frozen_modules()

    def test_disabled_initialization_outputs_state_rng_and_updates(self):
        for depth in (2, 8):
            torch.manual_seed(123)
            old = self.old_sac.SACAgent(*spaces(), hidden_depth=depth)
            old_rng = torch.get_rng_state()
            torch.manual_seed(123)
            new = SACAgent(*spaces(), hidden_depth=depth, residual_connections=False)
            self.assert_nested_equal(old_rng, torch.get_rng_state())
            for name in NETWORKS:
                self.assert_nested_equal(getattr(old, name).state_dict(), getattr(new, name).state_dict())
            x = torch.randn(4, 17)
            self.assert_nested_equal(old.actor(x), new.actor(x))
            batch = fake_batch()
            for step in range(3):
                torch.manual_seed(200 + step)
                a = old.actor.get_action(x)
                losses = [old.update_critic(batch, True), *old.update_actor_and_alpha(batch, True)]
                old.update_targets()
                rng = torch.get_rng_state()
                torch.manual_seed(200 + step)
                self.assert_nested_equal(a, new.actor.get_action(x))
                self.assertEqual(losses, [new.update_critic(batch, True), *new.update_actor_and_alpha(batch, True)])
                new.update_targets()
                self.assert_nested_equal(rng, torch.get_rng_state())
            for name in NETWORKS:
                self.assert_nested_equal(getattr(old, name).state_dict(), getattr(new, name).state_dict())
            for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
                self.assert_nested_equal(getattr(old, name).state_dict(), getattr(new, name).state_dict())
            self.assert_nested_equal(old.log_alpha, new.log_alpha)

    def test_disabled_training_trajectory_raw_replay_counters_and_rng(self):
        for depth in (2, 8):
            config = tiny(hidden_depth=depth, residual_connections=False)
            old_config = self.old_train.TrainConfig(**{k: v for k, v in dataclasses.asdict(config).items()
                                                       if k not in ("input_norm", "residual_connections")})
            old = self.old_train.train(old_config, FakeEnv)
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            new = train(config, FakeEnv)
            self.assert_nested_equal(rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
            for name in NETWORKS:
                self.assert_nested_equal(getattr(old.agent, name).state_dict(), getattr(new.agent, name).state_dict())
            for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
                self.assert_nested_equal(getattr(old.agent, name).state_dict(), getattr(new.agent, name).state_dict())
            for name in ("obs", "next_obs", "actions", "rewards", "terminations", "truncations"):
                self.assert_nested_equal(getattr(old.replay_buffer, name), getattr(new.replay_buffer, name))
            for name in ("num_random_actions", "num_policy_actions", "num_critic_updates", "num_actor_updates",
                         "num_target_updates", "eval_results", "final_eval", "episode_returns", "episode_end_steps"):
                self.assertEqual(getattr(old, name), getattr(new, name))


class CheckpointTests(EqualityMixin, unittest.TestCase):
    def test_enabled_roundtrip_including_optimizer_and_deterministic_inference(self):
        agent = SACAgent(*spaces(), hidden_depth=8, residual_connections=True)
        batch = fake_batch()
        agent.update_critic(batch)
        agent.update_actor_and_alpha(batch)
        agent.update_targets()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, agent, 24, tiny(), 0)
            state = torch.load(path, weights_only=True)
            self.assertIs(state["config"]["residual_connections"], True)
            self.assertIs(state["config"]["input_norm"], False)
            self.assertNotIn("observation_normalization", state)
            fresh = SACAgent(*spaces(), hidden_depth=8, residual_connections=True)
            info = load_checkpoint(path, fresh)
            self.assertEqual(info["global_step"], 24)
            self.assert_nested_equal(agent_snapshot(agent), agent_snapshot(fresh))
            self.assert_nested_equal(agent.actor.get_deterministic_action(batch.obs), fresh.actor.get_deterministic_action(batch.obs))

    def test_semantic_mismatches_and_malformed_flags_reject_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            for enabled in (False, True):
                source = SACAgent(*spaces(), hidden_depth=8, residual_connections=enabled)
                save_checkpoint(path, source, 24, tiny(residual_connections=enabled), 0)
                target = SACAgent(*spaces(), hidden_depth=8, residual_connections=not enabled)
                target.update_critic(fake_batch())
                target.update_actor_and_alpha(fake_batch())
                before = agent_snapshot(target)
                with self.assertRaisesRegex(ValueError, "residual_connections"):
                    load_checkpoint(path, target)
                self.assert_nested_equal(before, agent_snapshot(target))
            original = torch.load(path, weights_only=True)
            for bad in (1, 0, "false", None, []):
                state = copy.deepcopy(original)
                state["config"]["residual_connections"] = bad
                torch.save(state, path)
                with self.assertRaisesRegex(ValueError, "residual_connections"):
                    load_checkpoint(path, target)
                self.assert_nested_equal(before, agent_snapshot(target))

    def test_missing_historical_flag_is_disabled_and_legacy_weights_load(self):
        old_sac, old_train = frozen_modules()
        old = old_sac.SACAgent(*spaces(), hidden_depth=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            new = SACAgent(*spaces(), hidden_depth=8)
            save_checkpoint(path, new, 0, tiny(residual_connections=False), 0)
            state = torch.load(path, weights_only=True)
            state["config"].pop("residual_connections")
            state["config"].pop("input_norm")
            for name in NETWORKS:
                state[name] = getattr(old, name).state_dict()
            torch.save(state, path)
            load_checkpoint(path, new)
            for name in NETWORKS:
                self.assert_nested_equal(getattr(old, name).state_dict(), getattr(new, name).state_dict())
                getattr(old, name).load_state_dict(getattr(new, name).state_dict(), strict=True)
            enabled = SACAgent(*spaces(), hidden_depth=8, residual_connections=True)
            before = agent_snapshot(enabled)
            with self.assertRaisesRegex(ValueError, "residual_connections"):
                load_checkpoint(path, enabled)
            self.assert_nested_equal(before, agent_snapshot(enabled))

    def test_save_mismatch_and_inconsistent_network_flags_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            agent = SACAgent(*spaces(), hidden_depth=8, residual_connections=True)
            with self.assertRaises(ValueError):
                save_checkpoint(path, agent, 0, tiny(residual_connections=False), 0)
            self.assertFalse(path.exists())
            save_checkpoint(path, agent, 0, tiny(), 0)
            for name in NETWORKS:
                getattr(agent, name).residual_connections = False
                before = agent_snapshot(agent)
                with self.assertRaises(ValueError):
                    load_checkpoint(path, agent)
                with self.assertRaises(ValueError):
                    save_checkpoint(path, agent, 0, tiny(), 0)
                self.assert_nested_equal(before, agent_snapshot(agent))
                getattr(agent, name).residual_connections = True


class IntegrationTests(EqualityMixin, unittest.TestCase):
    def test_config_diff_is_only_residual_enabled_inputnorm_explicitly_disabled(self):
        control = json.loads((ROOT / "configs/depth_8x256.json").read_text())
        residual = json.loads((ROOT / "configs/ablation_residual_8x256.json").read_text())
        self.assertIs(residual.pop("residual_connections"), True)
        self.assertIs(residual.pop("input_norm"), False)
        self.assertEqual(control, residual)
        config = load_config(str(ROOT / "configs/ablation_residual_8x256.json"))
        self.assertTrue(config.residual_connections)
        self.assertFalse(config.input_norm)

    def test_schedule_raw_replay_rewards_actions_and_eval_isolation(self):
        def run(**kwargs):
            return train(tiny(**kwargs), FakeEnv)
        plain = run(residual_connections=False)
        with mock.patch("src.train.evaluate", wraps=evaluate) as evaluator:
            residual = run()
        self.assertEqual(evaluator.call_count, 4)
        self.assertEqual([c.args[2] for c in evaluator.call_args_list], [5, 5, 5, 10])
        self.assertTrue(all("observation_transform" not in c.kwargs for c in evaluator.call_args_list))
        for name in ("num_random_actions", "num_policy_actions", "num_critic_updates", "num_actor_updates", "num_target_updates"):
            self.assertEqual(getattr(plain, name), getattr(residual, name))
        self.assertEqual((residual.num_random_actions, residual.num_policy_actions, residual.num_critic_updates,
                          residual.num_actor_updates, residual.num_target_updates), (5, 19, 18, 18, 18))
        for name in ("obs", "next_obs", "rewards", "terminations", "truncations"):
            self.assert_nested_equal(getattr(plain.replay_buffer, name), getattr(residual.replay_buffer, name))
        self.assert_nested_equal(plain.replay_buffer.actions[:5], residual.replay_buffer.actions[:5])
        self.assertTrue(np.isfinite(residual.replay_buffer.actions).all())
        self.assertTrue((np.abs(residual.replay_buffer.actions) <= 1).all())
        rng = torch.get_rng_state()
        without_eval = run(eval_interval=0, final_eval_episodes=0)
        self.assert_nested_equal(rng, torch.get_rng_state())
        self.assert_nested_equal(agent_snapshot(residual.agent), agent_snapshot(without_eval.agent))
        self.assertIsNone(residual.agent.observation_normalizer)

    def test_real_halfcheetah_cpu_smoke_logging_checkpoint_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(tiny(total_steps=12, learning_starts=5, eval_interval=8,
                                         eval_episodes=1, final_eval_episodes=1, output_dir=tmp,
                                         run_id="RESIDUAL_CPU_SMOKE", checkpoint_interval=6))
            run = Path(tmp) / "RESIDUAL_CPU_SMOKE"
            self.assertEqual(result.steps_completed, 12)
            meta = json.loads((run / "metadata.json").read_text())
            self.assertEqual(meta["status"], "completed")
            self.assertIs(meta["config"]["residual_connections"], True)
            self.assertNotIn("input_normalization", meta)
            self.assertEqual(meta["parameter_counts"]["total"], 1402126)


if __name__ == "__main__":
    unittest.main()
