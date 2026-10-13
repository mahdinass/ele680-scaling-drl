import dataclasses
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box

from src.checkpoint import load_checkpoint, save_checkpoint
from src.networks import Actor, QNetwork
from src.run_metadata import count_parameters
from src.sac_update import SACAgent
from src.train import TrainConfig, load_config, run_experiment, train
from tests.test_evaluate import FakeEnv
from tests.test_sac_update import fake_batch, spaces

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
OBS, ACT, WIDTH = 17, 6, 256
FROZEN_REVISION = "2bb33da5af8106e204beab03c1f2d294b94e25fe"  # last commit before configurable depth

# Independently derived from first principles (HalfCheetah: 17 obs, 6 actions, width 256):
#   critic = (obs+act)*W + W + (d-1)*(W*W + W) + W + 1
#   actor  = obs*W + W + (d-1)*(W*W + W) + 2*(W*act + act)
EXPECTED = {
    2: {"actor": 73_484, "q1": 72_193, "q2": 72_193, "total": 217_870},
    4: {"actor": 205_068, "q1": 203_777, "q2": 203_777, "total": 612_622},
    8: {"actor": 468_236, "q1": 466_945, "q2": 466_945, "total": 1_402_126},
}


def expected_counts(depth, width=WIDTH, obs=OBS, act=ACT):
    q = (obs + act) * width + width + (depth - 1) * (width * width + width) + width + 1
    actor = obs * width + width + (depth - 1) * (width * width + width) + 2 * (width * act + act)
    return {"actor": actor, "q1": q, "q2": q, "total": actor + 2 * q}


def hidden_linears(module):
    """Hidden Linear layers in forward order (everything except the output head(s))."""
    layers = [module.fc1, module.fc2, *module.hidden_extra]
    return layers


def make_agent(depth, seed=0):
    torch.manual_seed(seed)
    return SACAgent(*spaces(), hidden_depth=depth)


def git_show(path):
    try:
        out = subprocess.run(
            ["git", "show", f"{FROZEN_REVISION}:{path}"], cwd=ROOT, capture_output=True, check=True, timeout=30
        )
        return out.stdout.decode("utf-8")
    except Exception:
        return None


def load_frozen_modules():
    """Execute the pre-depth networks.py / sac_update.py from the frozen Git revision
    as standalone modules (frozen sac_update is wired to the frozen networks)."""
    nets_src, sac_src = git_show("src/networks.py"), git_show("src/sac_update.py")
    if nets_src is None or sac_src is None:
        return None
    nets = types.ModuleType("frozen_networks")
    exec(compile(nets_src, "frozen_networks.py", "exec"), nets.__dict__)
    saved = sys.modules.get("src.networks")
    sys.modules["src.networks"] = nets  # the frozen sac_update does `from src.networks import ...`
    try:
        sac = types.ModuleType("frozen_sac_update")
        exec(compile(sac_src, "frozen_sac_update.py", "exec"), sac.__dict__)
    finally:
        if saved is None:
            sys.modules.pop("src.networks", None)
        else:
            sys.modules["src.networks"] = saved
    return nets, sac


class ArchitectureTests(unittest.TestCase):
    def test_default_depth_is_two(self):
        self.assertEqual(TrainConfig().hidden_depth, 2)
        agent = SACAgent(*spaces())
        self.assertEqual(count_parameters(agent), EXPECTED[2])
        self.assertEqual(len(agent.actor.hidden_extra), 0)

    def test_depth_2_state_dict_keys_and_shapes_unchanged(self):
        obs_space, act_space = spaces()
        actor, q = Actor(obs_space, act_space, hidden_depth=2), QNetwork(obs_space, act_space, hidden_depth=2)
        self.assertEqual(
            {k: tuple(v.shape) for k, v in actor.state_dict().items()},
            {
                "fc1.weight": (WIDTH, OBS), "fc1.bias": (WIDTH,),
                "fc2.weight": (WIDTH, WIDTH), "fc2.bias": (WIDTH,),
                "fc_mean.weight": (ACT, WIDTH), "fc_mean.bias": (ACT,),
                "fc_logstd.weight": (ACT, WIDTH), "fc_logstd.bias": (ACT,),
                "action_scale": (ACT,), "action_bias": (ACT,),
            },
        )
        self.assertEqual(
            {k: tuple(v.shape) for k, v in q.state_dict().items()},
            {
                "fc1.weight": (WIDTH, OBS + ACT), "fc1.bias": (WIDTH,),
                "fc2.weight": (WIDTH, WIDTH), "fc2.bias": (WIDTH,),
                "fc3.weight": (1, WIDTH), "fc3.bias": (1,),
            },
        )

    def test_deeper_networks_only_add_hidden_extra_keys(self):
        base = make_agent(2)
        for depth in (4, 8):
            agent = make_agent(depth)
            for name in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
                with self.subTest(depth=depth, net=name):
                    ref = getattr(base, name).state_dict()
                    sd = getattr(agent, name).state_dict()
                    self.assertEqual(set(sd) - set(ref), {
                        f"hidden_extra.{i}.{p}" for i in range(depth - 2) for p in ("weight", "bias")
                    })
                    self.assertEqual(set(ref) - set(sd), set())
                    for k in ref:
                        self.assertEqual(ref[k].shape, sd[k].shape, k)

    def test_hidden_layer_counts_and_widths(self):
        for depth in (2, 4, 8):
            agent = make_agent(depth)
            for name in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
                with self.subTest(depth=depth, net=name):
                    layers = hidden_linears(getattr(agent, name))
                    self.assertEqual(len(layers), depth)
                    for layer in layers:
                        self.assertIsInstance(layer, torch.nn.Linear)
                        self.assertEqual(layer.out_features, WIDTH)
                    for layer in layers[1:]:  # every hidden layer after the first is 256 -> 256
                        self.assertEqual(layer.in_features, WIDTH)
            self.assertEqual(agent.actor.fc_mean.in_features, WIDTH)
            self.assertEqual(agent.actor.fc_logstd.in_features, WIDTH)
            self.assertEqual(agent.qf1.fc3.in_features, WIDTH)
            self.assertEqual(agent.qf1.fc3.out_features, 1)

    def test_parameter_counts(self):
        for depth in (2, 4, 8):
            with self.subTest(depth=depth):
                agent = make_agent(depth)
                self.assertEqual(count_parameters(agent), EXPECTED[depth])
                self.assertEqual(expected_counts(depth), EXPECTED[depth])  # formula matches the stated table
                trainable = sum(
                    p.numel() for m in (agent.actor, agent.qf1, agent.qf2) for p in m.parameters() if p.requires_grad
                )
                self.assertEqual(trainable, EXPECTED[depth]["total"])
                self.assertEqual(
                    sum(p.numel() for m in (agent.qf1_target, agent.qf2_target)
                        for p in m.parameters() if p.requires_grad), 0,
                )

    def test_first_two_layers_share_init_across_depths(self):
        """Extra layers are created after fc1/fc2, so fc1/fc2 start identically for a given seed."""
        torch.manual_seed(0)
        shallow = Actor(*spaces(), hidden_depth=2)
        torch.manual_seed(0)
        deep = Actor(*spaces(), hidden_depth=8)
        for name in ("fc1", "fc2"):
            self.assertTrue(torch.equal(getattr(shallow, name).weight, getattr(deep, name).weight))

    def test_invalid_depths_rejected(self):
        obs_space, act_space = spaces()
        for bad in (0, 1, -1, -4, True, False, 2.0, 4.5, "4", None):
            for cls in (Actor, QNetwork):
                with self.subTest(bad=bad, cls=cls.__name__):
                    with self.assertRaises(ValueError):
                        cls(obs_space, act_space, hidden_depth=bad)
        with self.assertRaises(ValueError):
            SACAgent(obs_space, act_space, hidden_depth=0)

    def test_positional_constructor_compatibility(self):
        obs_space, act_space = spaces()
        a = Actor(obs_space, act_space, 128)  # old call style: width positional
        self.assertEqual(a.fc1.out_features, 128)
        self.assertEqual(len(a.hidden_extra), 0)
        agent = SACAgent(obs_space, act_space, "cpu", 0.99, 0.005, 3e-4, 1e-3, 128)  # old positional order
        self.assertEqual(agent.qf1.fc2.out_features, 128)
        self.assertEqual(len(agent.qf1.hidden_extra), 0)

    def test_width_and_depth_compose(self):
        agent = SACAgent(*spaces(), hidden_width=128, hidden_depth=4)
        self.assertEqual(count_parameters(agent), expected_counts(4, width=128))


class FunctionalTests(unittest.TestCase):
    def test_forward_shapes_and_finiteness(self):
        for depth in (4, 8):
            with self.subTest(depth=depth):
                agent = make_agent(depth)
                obs = torch.randn(8, OBS)
                mean, log_std = agent.actor(obs)
                self.assertEqual(mean.shape, (8, ACT))
                self.assertEqual(log_std.shape, (8, ACT))
                action, log_prob = agent.actor.get_action(obs)
                self.assertEqual(action.shape, (8, ACT))
                self.assertEqual(log_prob.shape, (8, 1))
                self.assertTrue(all(torch.isfinite(t).all() for t in (mean, log_std, action, log_prob)))
                for q in (agent.qf1, agent.qf2, agent.qf1_target, agent.qf2_target):
                    out = q(obs, action)
                    self.assertEqual(out.shape, (8, 1))
                    self.assertTrue(torch.isfinite(out).all())

    def test_actions_within_bounds_and_log_prob_finite_for_extreme_inputs(self):
        low = np.array([-2, -1, 0, -3, -0.5, -1], dtype=np.float32)
        high = np.array([2, 1, 4, 1, 0.5, 3], dtype=np.float32)
        obs_space, _ = spaces()
        for depth in (4, 8):
            torch.manual_seed(0)
            actor = Actor(obs_space, Box(low, high, dtype=np.float32), hidden_depth=depth)
            for scale in (1.0, 1e3):
                obs = torch.randn(64, OBS) * scale
                sampled, log_prob = actor.get_action(obs)
                det = actor.get_deterministic_action(obs)
                with self.subTest(depth=depth, scale=scale):
                    self.assertTrue(torch.isfinite(log_prob).all())
                    for a in (sampled, det):
                        self.assertTrue(torch.isfinite(a).all())
                        self.assertTrue(((a >= torch.as_tensor(low)) & (a <= torch.as_tensor(high))).all())

    def test_gradients_reach_every_hidden_layer(self):
        for depth in (4, 8):
            with self.subTest(depth=depth):
                agent = make_agent(depth)
                batch = fake_batch()
                agent.critic_loss(batch).backward()
                for qname in ("qf1", "qf2"):
                    for name, p in getattr(agent, qname).named_parameters():
                        self.assertIsNotNone(p.grad, f"{qname}.{name}")
                        self.assertTrue(torch.isfinite(p.grad).all(), f"{qname}.{name}")
                        self.assertGreater(p.grad.abs().sum().item(), 0.0, f"{qname}.{name} has zero grad")
                actor_loss, _ = agent.actor_loss(batch)
                actor_loss.backward()
                for name, p in agent.actor.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(p.grad.abs().sum().item(), 0.0, f"actor.{name} has zero grad")

    def test_deep_layers_are_used_in_forward_pass(self):
        for depth in (4, 8):
            agent = make_agent(depth)
            obs, act = torch.randn(8, OBS), torch.rand(8, ACT) * 2 - 1
            for net, args in ((agent.actor, (obs,)), (agent.qf1, (obs, act))):
                before = net(*args)
                before = torch.cat(before, dim=1) if isinstance(before, tuple) else before
                for i, layer in enumerate(net.hidden_extra):
                    with self.subTest(depth=depth, net=type(net).__name__, layer=i):
                        saved = layer.weight.detach().clone()
                        with torch.no_grad():
                            layer.weight.add_(0.5)
                        after = net(*args)
                        after = torch.cat(after, dim=1) if isinstance(after, tuple) else after
                        with torch.no_grad():
                            layer.weight.copy_(saved)
                        self.assertFalse(torch.allclose(before, after))

    def test_relu_after_every_hidden_layer_and_none_on_output(self):
        """Replicate the forward pass by hand and compare; a missing ReLU would change the result."""
        agent = make_agent(8)
        obs, act = torch.randn(16, OBS), torch.rand(16, ACT) * 2 - 1
        x = torch.relu(agent.qf1.fc1(torch.cat([obs, act], dim=1)))
        x = torch.relu(agent.qf1.fc2(x))
        for layer in agent.qf1.hidden_extra:
            x = torch.relu(layer(x))
        self.assertTrue(torch.equal(agent.qf1.fc3(x), agent.qf1(obs, act)))
        h = torch.relu(agent.actor.fc1(obs))
        h = torch.relu(agent.actor.fc2(h))
        for layer in agent.actor.hidden_extra:
            h = torch.relu(layer(h))
        mean, _ = agent.actor(obs)
        self.assertTrue(torch.equal(agent.actor.fc_mean(h), mean))

    def test_optimization_updates_parameters_but_not_targets(self):
        for depth in (4, 8):
            with self.subTest(depth=depth):
                agent, batch = make_agent(depth), fake_batch()
                snap = lambda m: [p.detach().clone() for p in m.parameters()]
                q1, q2, actor, t1, t2 = (snap(m) for m in (agent.qf1, agent.qf2, agent.actor,
                                                           agent.qf1_target, agent.qf2_target))
                loss = agent.update_critic(batch)
                a_loss, al_loss = agent.update_actor_and_alpha(batch)
                self.assertTrue(np.isfinite([loss, a_loss, al_loss]).all())
                for module, before in ((agent.qf1, q1), (agent.qf2, q2), (agent.actor, actor)):
                    for (name, p), b in zip(module.named_parameters(), before):
                        self.assertFalse(torch.equal(p, b), f"{name} did not update")
                for module, before in ((agent.qf1_target, t1), (agent.qf2_target, t2)):
                    for p, b in zip(module.parameters(), before):
                        self.assertTrue(torch.equal(p, b))  # optimizers never touch targets

    def test_targets_independent_and_initially_equal(self):
        agent = make_agent(8)
        for online, target in ((agent.qf1, agent.qf1_target), (agent.qf2, agent.qf2_target)):
            self.assertEqual(len(list(online.parameters())), len(list(target.parameters())))
            for po, pt in zip(online.parameters(), target.parameters()):
                self.assertTrue(torch.equal(po, pt))
                self.assertNotEqual(po.data_ptr(), pt.data_ptr())
                self.assertFalse(pt.requires_grad)
        self.assertFalse(torch.equal(agent.qf1.hidden_extra[0].weight, agent.qf2.hidden_extra[0].weight))

    def test_polyak_update_covers_all_additional_layers(self):
        for depth in (4, 8):
            with self.subTest(depth=depth):
                agent, batch = make_agent(depth), fake_batch()
                agent.update_critic(batch)  # make online != target
                before = {n: p.detach().clone() for n, p in agent.qf1_target.named_parameters()}
                online = {n: p.detach().clone() for n, p in agent.qf1.named_parameters()}
                agent.update_targets()
                for name, p in agent.qf1_target.named_parameters():
                    expected = agent.tau * online[name] + (1 - agent.tau) * before[name]
                    self.assertTrue(torch.allclose(p, expected, atol=1e-7), name)
                    if name.startswith("hidden_extra"):
                        self.assertFalse(torch.equal(p, before[name]), f"{name} not updated")


class BaselineCompatibilityTests(unittest.TestCase):
    """hidden_depth=2 must reproduce the pre-change implementation (frozen Git revision) exactly."""

    @classmethod
    def setUpClass(cls):
        frozen = load_frozen_modules()
        if frozen is None:
            raise unittest.SkipTest(f"git revision {FROZEN_REVISION} not available")
        cls.nets, cls.sac = frozen

    def test_frozen_revision_has_no_depth_argument(self):
        self.assertNotIn("hidden_depth", self.nets.Actor.__init__.__code__.co_varnames)

    def test_initial_weights_identical_to_frozen_implementation(self):
        for name in ("Actor", "QNetwork"):
            torch.manual_seed(123)
            old = getattr(self.nets, name)(*spaces())
            torch.manual_seed(123)
            new = {"Actor": Actor, "QNetwork": QNetwork}[name](*spaces(), hidden_depth=2)
            self.assertEqual(list(old.state_dict()), list(new.state_dict()))
            for k, v in old.state_dict().items():
                self.assertTrue(torch.equal(v, new.state_dict()[k]), f"{name}.{k}")

    def test_old_checkpoint_weights_load_into_new_depth_2(self):
        torch.manual_seed(5)
        old_agent = self.sac.SACAgent(*spaces())
        new_agent = SACAgent(*spaces())  # different random init
        for n in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
            getattr(new_agent, n).load_state_dict(getattr(old_agent, n).state_dict())  # strict
        obs = torch.randn(8, OBS)
        self.assertTrue(torch.equal(old_agent.actor(obs)[0], new_agent.actor(obs)[0]))

    def test_new_depth_2_checkpoint_loads_into_frozen_implementation(self):
        torch.manual_seed(6)
        new_agent = SACAgent(*spaces())
        old_agent = self.sac.SACAgent(*spaces())
        old_agent.actor.load_state_dict(new_agent.actor.state_dict())  # strict: same keys

    def test_forward_update_trajectory_bitwise_identical(self):
        """Same seed, same batch, several SAC updates: every parameter and loss matches the frozen code."""
        torch.manual_seed(7)
        old = self.sac.SACAgent(*spaces())
        torch.manual_seed(7)
        new = SACAgent(*spaces())
        batch = fake_batch()
        obs = torch.randn(8, OBS)
        torch.manual_seed(99)
        a_old, lp_old = old.actor.get_action(obs)
        torch.manual_seed(99)
        a_new, lp_new = new.actor.get_action(obs)
        self.assertTrue(torch.equal(a_old, a_new) and torch.equal(lp_old, lp_new))
        for step in range(5):
            torch.manual_seed(1000 + step)
            lo = [old.update_critic(batch), *old.update_actor_and_alpha(batch)]
            old.update_targets()
            torch.manual_seed(1000 + step)
            ln = [new.update_critic(batch), *new.update_actor_and_alpha(batch)]
            new.update_targets()
            self.assertEqual(lo, ln, f"losses differ at step {step}")
        for n in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
            a, b = getattr(old, n).state_dict(), getattr(new, n).state_dict()
            self.assertEqual(list(a), list(b))
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), f"{n}.{k}")
        self.assertTrue(torch.equal(old.log_alpha, new.log_alpha))

    def test_short_training_run_matches_frozen_train_loop_schedule(self):
        """Depth 2 vs explicit default: the training loop itself passes depth through without changing results."""
        def run(**kw):
            cfg = TrainConfig(seed=3, total_steps=40, learning_starts=10, batch_size=8, buffer_size=100,
                              eval_interval=0, final_eval_episodes=0, **kw)
            return train(cfg, FakeEnv)
        a, b = run(), run(hidden_depth=2)
        self.assertEqual(a.episode_returns, b.episode_returns)
        for k, v in a.agent.actor.state_dict().items():
            self.assertTrue(torch.equal(v, b.agent.actor.state_dict()[k]), k)


class ConfigTests(unittest.TestCase):
    @staticmethod
    def raw(name):
        return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))

    def test_baseline_and_width_configs_untouched(self):
        self.assertNotIn("hidden_depth", self.raw("baseline.json"))
        for name in ("width_2x512.json", "width_2x1024.json"):
            self.assertNotIn("hidden_depth", self.raw(name))

    def test_depth_configs_differ_from_baseline_only_in_hidden_depth(self):
        for depth in (4, 8):
            with self.subTest(depth=depth):
                cfg = self.raw(f"depth_{depth}x256.json")
                self.assertEqual(cfg.pop("hidden_depth"), depth)
                self.assertNotIn("hidden_width", cfg)  # width stays the implicit default 256
                self.assertEqual(cfg, self.raw("baseline.json"))

    def test_depth_configs_protocol_values(self):
        for depth in (4, 8):
            cfg = self.raw(f"depth_{depth}x256.json")
            self.assertEqual(
                {k: cfg[k] for k in ("env_id", "total_steps", "buffer_size", "batch_size", "learning_starts",
                                     "policy_frequency", "target_network_frequency", "gamma", "tau", "policy_lr",
                                     "q_lr", "device", "eval_interval", "eval_episodes", "final_eval_episodes",
                                     "log_interval", "checkpoint_interval", "tensorboard")},
                {"env_id": "HalfCheetah-v5", "total_steps": 1_000_000, "buffer_size": 1_000_000, "batch_size": 256,
                 "learning_starts": 5000, "policy_frequency": 2, "target_network_frequency": 1, "gamma": 0.99,
                 "tau": 0.005, "policy_lr": 0.0003, "q_lr": 0.001, "device": "cuda", "eval_interval": 25000,
                 "eval_episodes": 5, "final_eval_episodes": 10, "log_interval": 1000,
                 "checkpoint_interval": 100000, "tensorboard": True},
            )
            self.assertIsNone(cfg["run_id"])

    def test_depth_configs_load_into_train_config(self):
        base = load_config(str(CONFIG_DIR / "baseline.json"))
        self.assertEqual(base.hidden_depth, 2)
        for depth in (4, 8):
            cfg = load_config(str(CONFIG_DIR / f"depth_{depth}x256.json"))
            self.assertEqual((cfg.hidden_depth, cfg.hidden_width), (depth, 256))
            self.assertEqual(dataclasses.replace(cfg, hidden_depth=2), base)
            self.assertEqual(dataclasses.asdict(cfg)["hidden_depth"], depth)  # recorded in config.json/checkpoints


class IntegrationTests(unittest.TestCase):
    def tiny(self, depth, **kw):
        return TrainConfig(seed=0, total_steps=30, learning_starts=10, batch_size=8, buffer_size=100,
                           eval_interval=0, eval_episodes=1, final_eval_episodes=1, hidden_depth=depth, **kw)

    def test_fake_env_training_reaches_agent_with_correct_depth(self):
        for depth in (4, 8):
            with self.subTest(depth=depth):
                result = train(self.tiny(depth), FakeEnv)
                self.assertEqual(result.steps_completed, 30)
                self.assertGreater(result.num_critic_updates, 0)
                self.assertGreater(result.num_actor_updates, 0)
                self.assertGreater(result.num_target_updates, 0)
                agent = result.agent
                for name in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
                    self.assertEqual(len(getattr(agent, name).hidden_extra), depth - 2)
                self.assertEqual(count_parameters(agent), expected_counts(depth, obs=3, act=2))
                self.assertTrue(np.isfinite(result.final_eval.mean_return))

    def test_checkpoint_roundtrip_and_metadata(self):
        for depth in (4, 8):
            with self.subTest(depth=depth), tempfile.TemporaryDirectory() as tmp:
                cfg = self.tiny(depth, output_dir=tmp, run_id=f"t{depth}", tensorboard=False,
                                checkpoint_interval=0, log_interval=5)
                result = run_experiment(cfg, FakeEnv)
                run_dir = Path(tmp) / f"t{depth}"
                self.assertEqual(json.loads((run_dir / "config.json").read_text())["hidden_depth"], depth)
                meta = json.loads((run_dir / "metadata.json").read_text())
                self.assertEqual(meta["config"]["hidden_depth"], depth)
                self.assertEqual(meta["parameter_counts"], expected_counts(depth, obs=3, act=2))
                ckpt = run_dir / "checkpoint.pt"
                self.assertTrue(ckpt.exists(), "no checkpoint written")
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                self.assertEqual(state["config"]["hidden_depth"], depth)
                fresh = SACAgent(FakeEnv.observation_space, FakeEnv.action_space, hidden_depth=depth)
                info = load_checkpoint(ckpt, fresh)
                self.assertEqual(info["config"]["hidden_depth"], depth)
                for name in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
                    a, b = getattr(result.agent, name).state_dict(), getattr(fresh, name).state_dict()
                    for k in a:
                        self.assertTrue(torch.equal(a[k], b[k]), f"{name}.{k}")

    def test_checkpoint_depth_mismatch_fails_loudly(self):
        agent = make_agent(4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            save_checkpoint(path, agent, 1, TrainConfig(hidden_depth=4), 0)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path, make_agent(2))

    def test_historical_config_without_hidden_depth_means_depth_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text((CONFIG_DIR / "baseline.json").read_text(encoding="utf-8"), encoding="utf-8")
            self.assertEqual(load_config(str(path)).hidden_depth, 2)


if __name__ == "__main__":
    unittest.main()
