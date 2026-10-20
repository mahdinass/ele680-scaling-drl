"""LayerNorm-only architecture, frozen-source compatibility and local integration.

Short fake/real environment runs are implementation tests, never scientific data.
"""

import copy
import dataclasses
import inspect
import json
import operator
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
import torch.nn.functional as F

from src.checkpoint import load_checkpoint, save_checkpoint
from src.evaluate import evaluate
from src.networks import Actor, QNetwork, LOG_STD_MIN, LOG_STD_MAX
from src.run_metadata import count_parameters
from src.sac_update import SACAgent
from src.train import TrainConfig, load_config, run_experiment, train
from tests.test_evaluate import FakeEnv
from tests.test_inputnorm import EqualityMixin
from tests.test_residual import NETWORKS
from tests.test_sac_update import fake_batch, spaces

ROOT = Path(__file__).resolve().parents[1]
NAIVE = "6c674babc344e62566692e1a4fcf3611dc53a6cc"
INPUTNORM = "689dbee26af0ab44957bd17dafc7435d8f4a6109"
RESIDUAL = "41ffb94d03472a5ebc81b4d39fdf9c59bfbe6609"
COUNTS = {"actor": 472332, "q1": 471041, "q2": 471041, "total": 1414414}


def agent_snapshot(agent):
    # The actual naive reference predates the observation_normalizer attribute.
    normalizer = getattr(agent, "observation_normalizer", None)
    return copy.deepcopy({
        **{name: getattr(agent, name).state_dict() for name in NETWORKS},
        **{name: getattr(agent, name).state_dict() for name in
           ("actor_optimizer", "critic_optimizer", "alpha_optimizer")},
        "log_alpha": agent.log_alpha,
        "normalizer": None if normalizer is None else normalizer.state_dict(),
    })


def tiny(**kwargs):
    values = dict(total_steps=24, learning_starts=5, batch_size=4, buffer_size=50,
                  eval_interval=8, eval_episodes=5, final_eval_episodes=10,
                  hidden_depth=8, layer_norm=True, tensorboard=False)
    values.update(kwargs)
    return TrainConfig(**values)


def frozen_source(reference):
    """Load actual Git networks, SAC, checkpoints and loop; missing history fails."""
    names = ("src.networks", "src.sac_update", "src.checkpoint", "src.train")
    saved = {name: sys.modules.get(name) for name in names}
    loaded = {}
    try:
        for name in names:
            source = subprocess.run(
                ["git", "show", f"{reference}:{name.replace('.', '/')}.py"],
                cwd=ROOT, capture_output=True, check=True, timeout=20,
            ).stdout.decode("utf-8")
            module = types.ModuleType(name)
            sys.modules[name] = module
            exec(compile(source, f"{reference}/{name}", "exec"), module.__dict__)
            loaded[name] = module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return loaded


class ArchitectureTests(EqualityMixin, unittest.TestCase):
    def test_exact_linear_layernorm_relu_graph_and_unnormalized_heads(self):
        for kind in (Actor, QNetwork):
            net = kind(*spaces(), hidden_depth=8, layer_norm=True)
            graph = list(torch.fx.symbolic_trace(net).graph.nodes)
            hidden = ["fc1", "fc2", *[f"hidden_extra.{i}" for i in range(6)]]
            modules = {n.target: n for n in graph if n.op == "call_module"}
            relus = [n for n in graph if n.op == "call_function" and n.target is F.relu]
            self.assertEqual(len(relus), 8)
            self.assertEqual(len([m for m in net.modules() if type(m) is torch.nn.Linear]),
                             10 if kind is Actor else 9)
            self.assertEqual([n.args[0].target for n in relus], [f"hidden_norms.{i}" for i in range(8)])
            self.assertEqual(len([m for m in net.modules() if type(m) is torch.nn.LayerNorm]), 8)
            self.assertFalse(any(n.op == "call_function" and n.target is operator.add
                                 and all(isinstance(a, torch.fx.Node) for a in n.args) for n in graph))
            for i, name in enumerate(hidden):
                self.assertIs(modules[f"hidden_norms.{i}"].args[0], modules[name])
                if i:
                    self.assertIs(modules[name].args[0], relus[i - 1])
            for name in (("fc_mean", "fc_logstd") if kind is Actor else ("fc3",)):
                self.assertIs(modules[name].args[0], relus[-1])

    def test_numeric_placement_and_raw_inputs(self):
        # Distinct affine values make sharing, skipped norms and wrong order observable.
        for kind in (Actor, QNetwork):
            net = kind(*spaces(), hidden_depth=8, layer_norm=True)
            records, head_inputs = [], []
            handles = []
            for i, (linear, norm) in enumerate(zip([net.fc1, net.fc2, *net.hidden_extra], net.hidden_norms)):
                with torch.no_grad():
                    norm.weight.fill_(1 + .1 * i)
                    norm.bias.fill_(.03 * i)
                handles.append(linear.register_forward_hook(
                    lambda module, args, out: records.append((args[0].detach().clone(), out.detach().clone()))))
            head = net.fc_mean if kind is Actor else net.fc3
            handles.append(head.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0].detach().clone())))
            obs, actions = torch.randn(3, 17), torch.randn(3, 6)
            try:
                net(obs) if kind is Actor else net(obs, actions)
            finally:
                for handle in handles:
                    handle.remove()
            self.assert_nested_equal(records[0][0], obs if kind is Actor else torch.cat([obs, actions], dim=1))
            for i in range(8):
                following = records[i + 1][0] if i < 7 else head_inputs[0]
                self.assert_nested_equal(following, F.relu(net.hidden_norms[i](records[i][1])))

    def test_counts_affine_initialization_independence_linear_init_and_rng(self):
        torch.manual_seed(42)
        plain = SACAgent(*spaces(), hidden_depth=8)
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        agent = SACAgent(*spaces(), hidden_depth=8, layer_norm=True)
        self.assert_nested_equal(rng, torch.get_rng_state())
        self.assertEqual(count_parameters(agent), COUNTS)
        self.assertEqual(COUNTS["total"] - count_parameters(plain)["total"], 12288)
        pointers = []
        for name in NETWORKS:
            net = getattr(agent, name)
            self.assertTrue(net.layer_norm)
            self.assertFalse(net.residual_connections)
            old_state = getattr(plain, name).state_dict()
            for key, value in old_state.items():
                self.assert_nested_equal(value, net.state_dict()[key])
            self.assertEqual(set(net.state_dict()) - set(old_state),
                             {f"hidden_norms.{i}.{part}" for i in range(8) for part in ("weight", "bias")})
            for norm in net.hidden_norms:
                self.assertEqual(norm.normalized_shape, (256,))
                self.assertEqual(norm.eps, 1e-5)
                self.assertTrue(norm.elementwise_affine)
                self.assertTrue(torch.equal(norm.weight, torch.ones(256)))
                self.assertTrue(torch.equal(norm.bias, torch.zeros(256)))
                pointers.extend(p.data_ptr() for p in norm.parameters())
        self.assertEqual(len(set(pointers)), 80)
        self.assertIsNone(agent.observation_normalizer)
        for online, target in ((agent.qf1, agent.qf1_target), (agent.qf2, agent.qf2_target)):
            self.assert_nested_equal(online.state_dict(), target.state_dict())
            self.assertTrue(all(p.data_ptr() != t.data_ptr() and not t.requires_grad
                                for p, t in zip(online.parameters(), target.parameters())))

    def test_strict_flag_and_mixed_interventions_rejected(self):
        for bad in (0, 1, "true", None, np.bool_(True), [], 0.0):
            for constructor, args in ((Actor, spaces()), (QNetwork, spaces()), (SACAgent, spaces()), (TrainConfig, ())):
                with self.subTest(bad=bad, constructor=constructor), self.assertRaises(ValueError):
                    constructor(*args, layer_norm=bad)
            if not isinstance(bad, np.bool_):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "config.json"
                    path.write_text(json.dumps({"layer_norm": bad}))
                    with self.assertRaises(ValueError):
                        load_config(str(path))
        for constructor, args in ((Actor, spaces()), (QNetwork, spaces()), (SACAgent, spaces()), (TrainConfig, ())):
            with self.assertRaises(ValueError):
                constructor(*args, layer_norm=True, residual_connections=True)
        for constructor, args in ((SACAgent, spaces()), (TrainConfig, ())):
            with self.assertRaises(ValueError):
                constructor(*args, layer_norm=True, input_norm=True)

    def test_signatures_append_after_exact_residual_public_arguments(self):
        old = frozen_source(RESIDUAL)
        for new, historical in ((Actor, old["src.networks"].Actor), (QNetwork, old["src.networks"].QNetwork),
                                (SACAgent, old["src.sac_update"].SACAgent), (TrainConfig, old["src.train"].TrainConfig)):
            self.assertEqual(list(inspect.signature(new).parameters), list(inspect.signature(historical).parameters) + ["layer_norm"])
        self.assertTrue(Actor(*spaces(), 256, 8, False, True).layer_norm)
        self.assertTrue(QNetwork(*spaces(), 256, 8, False, True).layer_norm)
        self.assertTrue(SACAgent(*spaces(), "cpu", .99, .005, 3e-4, 1e-3, 256, 8, False, False, True).layer_norm)

    def _finite_updates(self, device):
        torch.manual_seed(123)
        agent = SACAgent(*spaces(), device=device, hidden_depth=8, layer_norm=True)
        batch = fake_batch(device)
        for size in (1, 32):
            obs = batch.obs[:size]
            action, log_pi = agent.actor.get_action(obs)
            self.assertEqual(action.shape, (size, 6))
            self.assertEqual(log_pi.shape, (size, 1))
            self.assertTrue(torch.isfinite(action).all() and torch.isfinite(log_pi).all())
            self.assertTrue((action.abs() <= 1).all())
            mean, log_std = agent.actor(obs * 1000)
            self.assertTrue(torch.isfinite(mean).all())
            self.assertTrue(((log_std >= LOG_STD_MIN) & (log_std <= LOG_STD_MAX)).all())
            self.assertEqual(agent.qf1(obs, action).shape, (size, 1))
        before = agent_snapshot(agent)
        losses = [agent.update_critic(batch, True), *agent.update_actor_and_alpha(batch, True)]
        self.assertTrue(np.isfinite(losses).all())
        for name in ("actor", "qf1", "qf2"):
            net = getattr(agent, name)
            for layer in [net.fc1, net.fc2, *net.hidden_extra, *net.hidden_norms]:
                for p in layer.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(p.grad.abs().sum().item(), 0)
            for i in range(8):
                self.assertFalse(torch.equal(before[name][f"hidden_norms.{i}.weight"], net.hidden_norms[i].weight))
        for name in ("qf1_target", "qf2_target"):
            self.assert_nested_equal(before[name], getattr(agent, name).state_dict())
            self.assertTrue(all(p.grad is None for p in getattr(agent, name).parameters()))
        online_before = copy.deepcopy(agent.qf1.state_dict())
        agent.update_targets()
        for key, value in agent.qf1_target.state_dict().items():
            expected = agent.tau * online_before[key] + (1 - agent.tau) * before["qf1_target"][key]
            self.assertTrue(torch.equal(value, expected), key)
        return agent

    def test_finite_cpu_gradients_optimizer_and_polyak_affine_updates(self):
        self._finite_updates("cpu")

    def test_per_sample_layernorm_has_no_running_statistics_or_batch_dependence(self):
        net = Actor(*spaces(), hidden_depth=8, layer_norm=True)
        obs = torch.randn(4, 17)
        before = copy.deepcopy(net.state_dict())
        single = net(obs[:1])
        batch = net(torch.cat([obs[:1], obs[1:] * 100], dim=0))
        for a, b in zip(single, batch):
            self.assertTrue(torch.allclose(a, b[:1], atol=1e-6, rtol=1e-5))
        net.eval()
        self.assert_nested_equal(single, net(obs[:1]))
        self.assert_nested_equal(before, net.state_dict())
        self.assertFalse(any("running" in key for key in net.state_dict()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable locally")
    def test_cuda_forward_backward_checkpoint_optimizer_evaluation_and_training(self):
        agent = self._finite_updates("cuda")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, agent, 24, tiny(device="cuda"), 0)
            fresh = SACAgent(*spaces(), device="cuda", hidden_depth=8, layer_norm=True)
            load_checkpoint(path, fresh)
            cuda_device = fresh.log_alpha.device
            optimizer_names = ("actor_optimizer", "critic_optimizer", "alpha_optimizer")
            relocated_steps = dict.fromkeys(optimizer_names, 0)

            def assert_restored_equal(original, restored, path=()):
                if isinstance(original, torch.Tensor):
                    self.assertEqual(original.dtype, restored.dtype, path)
                    self.assertEqual(original.shape, restored.shape, path)
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
            self.assertEqual(relocated_steps, {"actor_optimizer": 36, "critic_optimizer": 68, "alpha_optimizer": 1})
            steps_before = {name: [s["step"].item() for s in getattr(fresh, name).state.values()] for name in optimizer_names}
            batch = fake_batch(fresh.device)
            losses = [fresh.update_critic(batch), *fresh.update_actor_and_alpha(batch)]
            self.assertTrue(np.isfinite(losses).all())
            for name in optimizer_names:
                states = list(getattr(fresh, name).state.values())
                self.assertEqual([s["step"].item() for s in states], [step + 1 for step in steps_before[name]])
                self.assertTrue(all(torch.isfinite(v).all() for s in states for v in s.values()))
                self.assertTrue(all(s[key].device == cuda_device for s in states for key in ("exp_avg", "exp_avg_sq")))
            self.assertTrue(all(torch.isfinite(p).all() for name in NETWORKS for p in getattr(fresh, name).parameters()))
            self.assertTrue(torch.isfinite(fresh.log_alpha).all())
            before = agent_snapshot(fresh)
            rng = torch.cuda.get_rng_state()
            for episodes in (5, 10):
                result = evaluate(fresh.actor, FakeEnv(0), episodes, 10000, "cuda")
                self.assertTrue(np.isfinite(result.returns).all())
            self.assert_nested_equal(before, agent_snapshot(fresh))
            self.assert_nested_equal(rng, torch.cuda.get_rng_state())
        result = train(tiny(device="cuda"), FakeEnv)
        self.assertEqual(result.steps_completed, 24)


class FrozenCompatibilityTests(EqualityMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.references = {ref: frozen_source(ref) for ref in (NAIVE, INPUTNORM, RESIDUAL)}

    def test_disabled_exact_initialization_actions_rng_losses_and_optimizer_trajectory(self):
        for ref, flags in ((NAIVE, {}), (INPUTNORM, {"input_norm": True}), (RESIDUAL, {"residual_connections": True})):
            old_sac = self.references[ref]["src.sac_update"]
            for depth in (2, 8):
                with self.subTest(ref=ref, depth=depth):
                    torch.manual_seed(123)
                    old = old_sac.SACAgent(*spaces(), hidden_depth=depth, **flags)
                    rng = torch.get_rng_state()
                    torch.manual_seed(123)
                    new = SACAgent(*spaces(), hidden_depth=depth, **flags)
                    self.assert_nested_equal(rng, torch.get_rng_state())
                    self.assert_nested_equal(agent_snapshot(old), agent_snapshot(new))
                    batch = fake_batch()
                    if flags.get("input_norm"):
                        for agent in (old, new):
                            agent.observation_normalizer.update(np.ones(17))
                            agent.observation_normalizer.update(np.zeros(17))
                    for step in range(3):
                        torch.manual_seed(200 + step)
                        old_action = old.actor.get_action(batch.obs)
                        losses = [old.update_critic(batch, True), *old.update_actor_and_alpha(batch, True)]
                        old.update_targets()
                        rng = torch.get_rng_state()
                        torch.manual_seed(200 + step)
                        self.assert_nested_equal(old_action, new.actor.get_action(batch.obs))
                        self.assertEqual(losses, [new.update_critic(batch, True), *new.update_actor_and_alpha(batch, True)])
                        new.update_targets()
                        self.assert_nested_equal(rng, torch.get_rng_state())
                    self.assert_nested_equal(agent_snapshot(old), agent_snapshot(new))

    def test_disabled_exact_training_replay_evaluation_rng_and_schedule_all_historical_modes(self):
        for ref, flags in ((NAIVE, {}), (INPUTNORM, {"input_norm": True}), (RESIDUAL, {"residual_connections": True})):
            historical = self.references[ref]["src.train"]
            fields = {f.name for f in dataclasses.fields(historical.TrainConfig)}
            for depth in (2, 8):
                with self.subTest(ref=ref, depth=depth):
                    config = tiny(layer_norm=False, hidden_depth=depth, **flags)
                    old_config = historical.TrainConfig(**{k: v for k, v in dataclasses.asdict(config).items() if k in fields})
                    old = historical.train(old_config, FakeEnv)
                    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
                    new = train(config, FakeEnv)
                    self.assert_nested_equal(rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
                    self.assert_nested_equal(agent_snapshot(old.agent), agent_snapshot(new.agent))
                    for name in ("obs", "next_obs", "actions", "rewards", "terminations", "truncations"):
                        self.assert_nested_equal(getattr(old.replay_buffer, name), getattr(new.replay_buffer, name))
                    for name in ("num_random_actions", "num_policy_actions", "num_critic_updates", "num_actor_updates",
                                 "num_target_updates", "eval_results", "final_eval", "episode_returns", "episode_end_steps"):
                        self.assertEqual(getattr(old, name), getattr(new, name))

    def test_actual_historical_checkpoints_and_new_disabled_checkpoints_load_both_directions(self):
        for ref, flags in ((NAIVE, {}), (INPUTNORM, {"input_norm": True}), (RESIDUAL, {"residual_connections": True})):
            modules = self.references[ref]
            for depth in (2, 8):
                with self.subTest(ref=ref, depth=depth), tempfile.TemporaryDirectory() as tmp:
                    old = modules["src.sac_update"].SACAgent(*spaces(), hidden_depth=depth, **flags)
                    old_config = modules["src.train"].TrainConfig(hidden_depth=depth, **flags)
                    new = SACAgent(*spaces(), hidden_depth=depth, **flags)
                    path = Path(tmp) / "checkpoint.pt"
                    modules["src.checkpoint"].save_checkpoint(path, old, 0, old_config, 0)
                    self.assertNotIn("layer_norm", torch.load(path, weights_only=True)["config"])
                    load_checkpoint(path, new)
                    self.assert_nested_equal(agent_snapshot(old), agent_snapshot(new))
                    save_checkpoint(path, new, 0, tiny(layer_norm=False, hidden_depth=depth, **flags), 0)
                    modules["src.checkpoint"].load_checkpoint(path, old)
                    self.assert_nested_equal(agent_snapshot(old), agent_snapshot(new))


class CheckpointTests(EqualityMixin, unittest.TestCase):
    def test_enabled_roundtrip_optimizer_metadata_affine_and_inference(self):
        agent = SACAgent(*spaces(), hidden_depth=8, layer_norm=True)
        batch = fake_batch()
        agent.update_critic(batch)
        agent.update_actor_and_alpha(batch)
        agent.update_targets()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, agent, 24, tiny(), 0)
            state = torch.load(path, weights_only=True)
            self.assertIs(state["config"]["layer_norm"], True)
            self.assertIs(state["config"]["input_norm"], False)
            self.assertIs(state["config"]["residual_connections"], False)
            self.assertNotIn("observation_normalization", state)
            fresh = SACAgent(*spaces(), hidden_depth=8, layer_norm=True)
            self.assertEqual(load_checkpoint(path, fresh)["global_step"], 24)
            self.assert_nested_equal(agent_snapshot(agent), agent_snapshot(fresh))
            self.assert_nested_equal(agent.actor.get_deterministic_action(batch.obs), fresh.actor.get_deterministic_action(batch.obs))

    def test_flag_mismatches_malformed_and_missing_flag_reject_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            for enabled in (False, True):
                source = SACAgent(*spaces(), hidden_depth=8, layer_norm=enabled)
                save_checkpoint(path, source, 24, tiny(layer_norm=enabled), 0)
                target = SACAgent(*spaces(), hidden_depth=8, layer_norm=not enabled)
                target.update_critic(fake_batch())
                target.update_actor_and_alpha(fake_batch())
                before = agent_snapshot(target)
                with self.assertRaisesRegex(ValueError, "layer_norm"):
                    load_checkpoint(path, target)
                self.assert_nested_equal(before, agent_snapshot(target))
            source = SACAgent(*spaces(), hidden_depth=8, layer_norm=True)
            save_checkpoint(path, source, 24, tiny(), 0)
            original = torch.load(path, weights_only=True)
            for bad in (1, 0, "false", None, []):
                changed = copy.deepcopy(original)
                changed["config"]["layer_norm"] = bad
                torch.save(changed, path)
                target = SACAgent(*spaces(), hidden_depth=8, layer_norm=True)
                before = agent_snapshot(target)
                with self.assertRaisesRegex(ValueError, "layer_norm"):
                    load_checkpoint(path, target)
                self.assert_nested_equal(before, agent_snapshot(target))
            original["config"].pop("layer_norm")
            torch.save(original, path)
            with self.assertRaisesRegex(ValueError, "layer_norm"):
                load_checkpoint(path, target)
            self.assert_nested_equal(before, agent_snapshot(target))

    def test_save_and_load_validate_agent_and_every_network_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            agent = SACAgent(*spaces(), hidden_depth=8, layer_norm=True)
            save_checkpoint(path, agent, 24, tiny(), 0)
            original = path.read_bytes()
            for obj in (agent, *[getattr(agent, name) for name in NETWORKS]):
                obj.layer_norm = False
                before = agent_snapshot(agent)
                with self.assertRaisesRegex(ValueError, "layer_norm"):
                    load_checkpoint(path, agent)
                with self.assertRaisesRegex(ValueError, "layer_norm"):
                    save_checkpoint(path, agent, 24, tiny(), 0)
                self.assertEqual(path.read_bytes(), original)
                self.assert_nested_equal(before, agent_snapshot(agent))
                obj.layer_norm = True
            with self.assertRaisesRegex(ValueError, "layer_norm"):
                save_checkpoint(path, agent, 24, tiny(layer_norm=False), 0)

    def test_affine_key_shape_and_other_mode_corruption_reject_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            agent = SACAgent(*spaces(), hidden_depth=8, layer_norm=True)
            save_checkpoint(path, agent, 24, tiny(), 0)
            original = torch.load(path, weights_only=True)
            for name in NETWORKS:
                for corruption in ("missing", "shape"):
                    state = copy.deepcopy(original)
                    if corruption == "missing":
                        state[name].pop("hidden_norms.7.bias")
                    else:
                        state[name]["hidden_norms.7.bias"] = torch.zeros(255)
                    torch.save(state, path)
                    before = agent_snapshot(agent)
                    with self.assertRaises(RuntimeError):
                        load_checkpoint(path, agent)
                    self.assert_nested_equal(before, agent_snapshot(agent))
            for flag in ("input_norm", "residual_connections"):
                state = copy.deepcopy(original)
                state["config"][flag] = True
                torch.save(state, path)
                before = agent_snapshot(agent)
                with self.assertRaises(ValueError):
                    load_checkpoint(path, agent)
                self.assert_nested_equal(before, agent_snapshot(agent))


class IntegrationTests(EqualityMixin, unittest.TestCase):
    def test_config_only_adds_three_explicit_intervention_flags(self):
        frozen = json.loads(subprocess.run(["git", "show", f"{NAIVE}:configs/depth_8x256.json"],
                                         cwd=ROOT, capture_output=True, check=True).stdout)
        cfg = json.loads((ROOT / "configs/ablation_layernorm_8x256.json").read_text())
        self.assertIs(cfg.pop("layer_norm"), True)
        self.assertIs(cfg.pop("input_norm"), False)
        self.assertIs(cfg.pop("residual_connections"), False)
        self.assertEqual(cfg, frozen)
        self.assertTrue(load_config(str(ROOT / "configs/ablation_layernorm_8x256.json")).layer_norm)

    def test_training_schedule_raw_replay_and_evaluation_isolation(self):
        plain = train(tiny(layer_norm=False), FakeEnv)
        with mock.patch("src.train.evaluate", wraps=evaluate) as evaluator:
            result = train(tiny(), FakeEnv)
        self.assertEqual([c.args[2] for c in evaluator.call_args_list], [5, 5, 5, 10])
        self.assertTrue(all("observation_transform" not in c.kwargs for c in evaluator.call_args_list))
        for name in ("num_random_actions", "num_policy_actions", "num_critic_updates", "num_actor_updates", "num_target_updates"):
            self.assertEqual(getattr(plain, name), getattr(result, name))
        self.assertEqual((result.num_random_actions, result.num_policy_actions, result.num_critic_updates,
                          result.num_actor_updates, result.num_target_updates), (5, 19, 18, 18, 18))
        for name in ("obs", "next_obs", "rewards", "terminations", "truncations"):
            self.assert_nested_equal(getattr(plain.replay_buffer, name), getattr(result.replay_buffer, name))
        self.assert_nested_equal(plain.replay_buffer.actions[:5], result.replay_buffer.actions[:5])
        rng = torch.get_rng_state()
        no_eval = train(tiny(eval_interval=0, final_eval_episodes=0), FakeEnv)
        self.assert_nested_equal(rng, torch.get_rng_state())
        self.assert_nested_equal(agent_snapshot(result.agent), agent_snapshot(no_eval.agent))
        for episodes in (5, 10):
            before = agent_snapshot(result.agent)
            evaluate(result.agent.actor, FakeEnv(0), episodes, 10000)
            self.assert_nested_equal(before, agent_snapshot(result.agent))
            self.assert_nested_equal(rng, torch.get_rng_state())

    def test_real_halfcheetah_cpu_smoke_logging_config_checkpoint_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(tiny(total_steps=12, learning_starts=5, eval_interval=8,
                                         eval_episodes=1, final_eval_episodes=1, output_dir=tmp,
                                         run_id="LAYERNORM_CPU_SMOKE", checkpoint_interval=6))
            run = Path(tmp) / "LAYERNORM_CPU_SMOKE"
            self.assertEqual(result.steps_completed, 12)
            meta = json.loads((run / "metadata.json").read_text())
            self.assertEqual(meta["status"], "completed")
            self.assertIs(meta["config"]["layer_norm"], True)
            self.assertNotIn("input_normalization", meta)
            self.assertEqual(meta["parameter_counts"], COUNTS)
            state = torch.load(run / "checkpoint.pt", weights_only=True)
            self.assertEqual(state["global_step"], 12)
            self.assertIs(state["config"]["layer_norm"], True)


if __name__ == "__main__":
    unittest.main()
