import dataclasses
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box

from src.networks import Actor, QNetwork
from src.run_metadata import count_parameters
from src.sac_update import SACAgent
from src.train import TrainConfig, load_config, train
from tests.test_evaluate import FakeEnv
from tests.test_sac_update import fake_batch, spaces

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
OBS, ACT = 17, 6  # HalfCheetah


def linear_widths(module):
    """(in_features, out_features) of every Linear, keyed by state-dict layer name."""
    return {n: (m.in_features, m.out_features) for n, m in module.named_children() if isinstance(m, torch.nn.Linear)}


def storage_ptrs(module):
    return {p.data_ptr() for p in module.parameters()}


class BaselineCompatibilityTests(unittest.TestCase):
    """hidden_width defaults to 256 and reproduces the frozen baseline exactly."""

    def test_default_layer_names_and_shapes_unchanged(self):
        obs_space, act_space = spaces()
        actor, q = Actor(obs_space, act_space), QNetwork(obs_space, act_space)
        self.assertEqual(
            {k: tuple(v.shape) for k, v in actor.state_dict().items()},
            {
                "fc1.weight": (256, OBS), "fc1.bias": (256,),
                "fc2.weight": (256, 256), "fc2.bias": (256,),
                "fc_mean.weight": (ACT, 256), "fc_mean.bias": (ACT,),
                "fc_logstd.weight": (ACT, 256), "fc_logstd.bias": (ACT,),
                "action_scale": (ACT,), "action_bias": (ACT,),
            },
        )
        self.assertEqual(
            {k: tuple(v.shape) for k, v in q.state_dict().items()},
            {
                "fc1.weight": (256, OBS + ACT), "fc1.bias": (256,),
                "fc2.weight": (256, 256), "fc2.bias": (256,),
                "fc3.weight": (1, 256), "fc3.bias": (1,),
            },
        )

    def test_default_equals_explicit_256_and_has_baseline_parameter_counts(self):
        torch.manual_seed(0)
        default = SACAgent(*spaces())
        torch.manual_seed(0)
        explicit = SACAgent(*spaces(), hidden_width=256)
        self.assertEqual(
            count_parameters(default), {"actor": 73_484, "q1": 72_193, "q2": 72_193, "total": 217_870}
        )
        self.assertEqual(count_parameters(explicit), count_parameters(default))
        # Same RNG consumption -> identical initial weights (seed semantics preserved).
        for name in ("actor", "qf1", "qf2"):
            a, b = getattr(default, name).state_dict(), getattr(explicit, name).state_dict()
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), f"{name}.{k}")

    def test_train_config_default_is_256(self):
        self.assertEqual(TrainConfig().hidden_width, 256)

    def test_baseline_json_loads_with_width_256_and_is_unmodified_schema(self):
        raw = json.loads((CONFIG_DIR / "baseline.json").read_text(encoding="utf-8"))
        self.assertNotIn("hidden_width", raw)  # baseline file is frozen; width comes from the default
        self.assertEqual(load_config(str(CONFIG_DIR / "baseline.json")).hidden_width, 256)


class Width512Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.obs_space, self.act_space = spaces()
        self.agent = SACAgent(self.obs_space, self.act_space, hidden_width=512)

    def test_actor_dimensions(self):
        self.assertEqual(
            linear_widths(self.agent.actor),
            {"fc1": (OBS, 512), "fc2": (512, 512), "fc_mean": (512, ACT), "fc_logstd": (512, ACT)},
        )

    def test_critic_and_target_critic_dimensions(self):
        expected = {"fc1": (OBS + ACT, 512), "fc2": (512, 512), "fc3": (512, 1)}
        for name in ("qf1", "qf2", "qf1_target", "qf2_target"):
            with self.subTest(name):
                self.assertEqual(linear_widths(getattr(self.agent, name)), expected)

    def test_layer_names_match_baseline(self):
        base = SACAgent(self.obs_space, self.act_space)
        for name in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
            self.assertEqual(
                list(getattr(self.agent, name).state_dict()), list(getattr(base, name).state_dict()), name
            )

    def test_parameter_counts(self):
        # Q: (23*512+512) + (512*512+512) + (512+1); actor: (17*512+512) + (512*512+512) + 2*(512*6+6)
        self.assertEqual(
            count_parameters(self.agent), {"actor": 278_028, "q1": 275_457, "q2": 275_457, "total": 828_942}
        )

    def test_forward_shapes_finite(self):
        obs = torch.randn(8, OBS)
        action, log_prob = self.agent.actor.get_action(obs)
        for q in (self.agent.qf1, self.agent.qf2, self.agent.qf1_target, self.agent.qf2_target):
            out = q(obs, action)
            self.assertEqual(out.shape, (8, 1))
            self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(log_prob.shape, (8, 1))

    def test_targets_start_equal_to_online_but_do_not_share_storage(self):
        for online, target in ((self.agent.qf1, self.agent.qf1_target), (self.agent.qf2, self.agent.qf2_target)):
            for po, pt in zip(online.parameters(), target.parameters()):
                self.assertTrue(torch.equal(po, pt))
                self.assertNotEqual(po.data_ptr(), pt.data_ptr())
                self.assertFalse(pt.requires_grad)

    def test_networks_have_independent_parameters(self):
        modules = [self.agent.actor, self.agent.qf1, self.agent.qf2, self.agent.qf1_target, self.agent.qf2_target]
        ptrs = [storage_ptrs(m) for m in modules]
        for i in range(len(ptrs)):
            for j in range(i + 1, len(ptrs)):
                self.assertTrue(ptrs[i].isdisjoint(ptrs[j]), (i, j))
        # Q1 and Q2 are different random initialisations, not copies.
        self.assertFalse(torch.equal(self.agent.qf1.fc1.weight, self.agent.qf2.fc1.weight))

    def test_updates_work_and_keep_target_consistency(self):
        agent, batch = self.agent, fake_batch()
        t_before = [p.detach().clone() for p in agent.qf1_target.parameters()]
        q_before = [p.detach().clone() for p in agent.qf1.parameters()]
        loss = agent.update_critic(batch)
        a_loss, al_loss = agent.update_actor_and_alpha(batch)
        self.assertTrue(np.isfinite([loss, a_loss, al_loss]).all())
        # critic moved, target did not (yet)
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(q_before, agent.qf1.parameters())))
        self.assertTrue(all(torch.equal(b, p) for b, p in zip(t_before, agent.qf1_target.parameters())))
        agent.update_targets()
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(t_before, agent.qf1_target.parameters())))
        for p in (*agent.qf1.parameters(), *agent.actor.parameters()):
            self.assertTrue(torch.isfinite(p).all())

    def test_invalid_width_rejected(self):
        for bad in (0, -1):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    Actor(self.obs_space, self.act_space, hidden_width=bad)
                with self.assertRaises(ValueError):
                    QNetwork(self.obs_space, self.act_space, hidden_width=bad)


class WidthConfigTests(unittest.TestCase):
    def test_width_2x512_differs_from_baseline_only_in_hidden_width(self):
        base = json.loads((CONFIG_DIR / "baseline.json").read_text(encoding="utf-8"))
        wide = json.loads((CONFIG_DIR / "width_2x512.json").read_text(encoding="utf-8"))
        self.assertEqual(wide.pop("hidden_width"), 512)
        self.assertEqual(wide, base)

    def test_width_2x512_loads_into_train_config(self):
        cfg = load_config(str(CONFIG_DIR / "width_2x512.json"))
        self.assertEqual(cfg.hidden_width, 512)
        self.assertEqual(
            dataclasses.replace(cfg, hidden_width=256), load_config(str(CONFIG_DIR / "baseline.json"))
        )
        self.assertEqual(dataclasses.asdict(cfg)["hidden_width"], 512)  # recorded in config.json / checkpoints


class WidthSmokeTests(unittest.TestCase):
    def test_tiny_training_run_with_width_512(self):
        cfg = TrainConfig(
            seed=0, total_steps=30, learning_starts=10, batch_size=8, buffer_size=100,
            eval_interval=0, eval_episodes=1, final_eval_episodes=1, hidden_width=512,
        )
        result = train(cfg, FakeEnv)
        self.assertEqual(result.steps_completed, 30)
        self.assertGreater(result.num_critic_updates, 0)
        self.assertGreater(result.num_actor_updates, 0)
        # FakeEnv: obs 3, action 2, width 512.
        q = (3 + 2) * 512 + 512 + 512 * 512 + 512 + 512 + 1
        actor = 3 * 512 + 512 + 512 * 512 + 512 + 2 * (512 * 2 + 2)
        self.assertEqual(count_parameters(result.agent), {"actor": actor, "q1": q, "q2": q, "total": actor + 2 * q})
        self.assertEqual(linear_widths(result.agent.qf2_target)["fc2"], (512, 512))
        self.assertTrue(np.isfinite(result.final_eval.mean_return))


def expected_counts(width, obs=OBS, act=ACT):
    """Parameter counts from first principles: 2 hidden layers of `width`, Gaussian actor with mean+logstd heads."""
    q = (obs + act) * width + width + width * width + width + width + 1
    actor = obs * width + width + width * width + width + 2 * (width * act + act)
    return {"actor": actor, "q1": q, "q2": q, "total": actor + 2 * q}


class Width1024Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.obs_space, self.act_space = spaces()
        self.agent = SACAgent(self.obs_space, self.act_space, hidden_width=1024)

    def test_actor_has_two_hidden_layers_of_width_1024(self):
        self.assertEqual(
            linear_widths(self.agent.actor),
            {"fc1": (OBS, 1024), "fc2": (1024, 1024), "fc_mean": (1024, ACT), "fc_logstd": (1024, ACT)},
        )

    def test_critics_and_targets_have_two_hidden_layers_of_width_1024(self):
        expected = {"fc1": (OBS + ACT, 1024), "fc2": (1024, 1024), "fc3": (1024, 1)}
        for name in ("qf1", "qf2", "qf1_target", "qf2_target"):
            with self.subTest(name):
                self.assertEqual(linear_widths(getattr(self.agent, name)), expected)

    def test_layer_names_match_baseline_and_512(self):
        for other_width in (256, 512):
            other = SACAgent(self.obs_space, self.act_space, hidden_width=other_width)
            for name in ("actor", "qf1", "qf2", "qf1_target", "qf2_target"):
                self.assertEqual(
                    list(getattr(self.agent, name).state_dict()), list(getattr(other, name).state_dict()),
                    f"{name} vs width {other_width}",
                )

    def test_parameter_counts_for_halfcheetah(self):
        self.assertEqual(
            count_parameters(self.agent),
            {"actor": 1_080_332, "q1": 1_075_201, "q2": 1_075_201, "total": 3_230_734},
        )

    def test_parameter_counts_match_first_principles_formula(self):
        self.assertEqual(count_parameters(self.agent), expected_counts(1024))
        # and for the other widths, so the formula itself is anchored to the verified 256/512 counts
        self.assertEqual(expected_counts(256)["total"], 217_870)
        self.assertEqual(expected_counts(512)["total"], 828_942)

    def test_trainable_parameter_tensors_sum_to_reported_counts(self):
        trainable = sum(p.numel() for m in (self.agent.actor, self.agent.qf1, self.agent.qf2)
                        for p in m.parameters() if p.requires_grad)
        self.assertEqual(trainable, 3_230_734)
        # target critics are not trainable
        self.assertEqual(sum(p.numel() for m in (self.agent.qf1_target, self.agent.qf2_target)
                             for p in m.parameters() if p.requires_grad), 0)

    def test_forward_shapes_and_finiteness(self):
        obs = torch.randn(8, OBS)
        action, log_prob = self.agent.actor.get_action(obs)
        self.assertEqual(action.shape, (8, ACT))
        self.assertEqual(log_prob.shape, (8, 1))
        self.assertTrue(torch.isfinite(action).all() and torch.isfinite(log_prob).all())
        for q in (self.agent.qf1, self.agent.qf2, self.agent.qf1_target, self.agent.qf2_target):
            out = q(obs, action)
            self.assertEqual(out.shape, (8, 1))
            self.assertTrue(torch.isfinite(out).all())

    def test_actions_respect_bounds_even_for_extreme_observations(self):
        obs = torch.randn(64, OBS) * 1e3
        sampled, _ = self.agent.actor.get_action(obs)
        deterministic = self.agent.actor.get_deterministic_action(obs)
        for a in (sampled, deterministic):
            self.assertEqual(a.shape, (64, ACT))
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue(((a >= -1.0) & (a <= 1.0)).all())

    def test_action_rescaling_follows_box_bounds(self):
        low = np.array([-2, -1, 0, -3, -0.5, -1], dtype=np.float32)
        high = np.array([2, 1, 4, 1, 0.5, 3], dtype=np.float32)
        actor = Actor(self.obs_space, Box(low, high, dtype=np.float32), hidden_width=1024)
        a = actor.get_action(torch.randn(64, OBS) * 10)[0]
        self.assertTrue(((a >= torch.as_tensor(low)) & (a <= torch.as_tensor(high))).all())

    def test_backward_passes_give_finite_gradients_for_every_parameter(self):
        batch = fake_batch()
        critic_loss = self.agent.critic_loss(batch)
        critic_loss.backward()
        for m in (self.agent.qf1, self.agent.qf2):
            for name, p in m.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        actor_loss, _ = self.agent.actor_loss(batch)
        actor_loss.backward()
        for name, p in self.agent.actor.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_targets_start_equal_to_online_but_are_independent_copies(self):
        for online, target in ((self.agent.qf1, self.agent.qf1_target), (self.agent.qf2, self.agent.qf2_target)):
            for po, pt in zip(online.parameters(), target.parameters()):
                self.assertTrue(torch.equal(po, pt))
                self.assertNotEqual(po.data_ptr(), pt.data_ptr())
                self.assertFalse(pt.requires_grad)

    def test_networks_have_independent_parameters(self):
        modules = [self.agent.actor, self.agent.qf1, self.agent.qf2, self.agent.qf1_target, self.agent.qf2_target]
        ptrs = [storage_ptrs(m) for m in modules]
        for i in range(len(ptrs)):
            for j in range(i + 1, len(ptrs)):
                self.assertTrue(ptrs[i].isdisjoint(ptrs[j]), (i, j))
        self.assertFalse(torch.equal(self.agent.qf1.fc1.weight, self.agent.qf2.fc1.weight))

    def test_updates_work_and_target_stays_put_until_polyak_update(self):
        agent, batch = self.agent, fake_batch()
        t_before = [p.detach().clone() for p in agent.qf1_target.parameters()]
        q_before = [p.detach().clone() for p in agent.qf1.parameters()]
        loss = agent.update_critic(batch)
        a_loss, al_loss = agent.update_actor_and_alpha(batch)
        self.assertTrue(np.isfinite([loss, a_loss, al_loss]).all())
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(q_before, agent.qf1.parameters())))
        self.assertTrue(all(torch.equal(b, p) for b, p in zip(t_before, agent.qf1_target.parameters())))
        agent.update_targets()
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(t_before, agent.qf1_target.parameters())))
        for m in (agent.actor, agent.qf1, agent.qf2, agent.qf1_target, agent.qf2_target):
            for p in m.parameters():
                self.assertTrue(torch.isfinite(p).all())


class Width1024ConfigTests(unittest.TestCase):
    FROZEN_BASELINE = {
        "env_id": "HalfCheetah-v5", "seed": 0, "device": "cuda", "total_steps": 1_000_000,
        "buffer_size": 1_000_000, "batch_size": 256, "learning_starts": 5000, "policy_frequency": 2,
        "target_network_frequency": 1, "gamma": 0.99, "tau": 0.005, "policy_lr": 0.0003, "q_lr": 0.001,
        "eval_interval": 25000, "eval_episodes": 5, "final_eval_episodes": 10, "output_dir": "results",
        "run_id": None, "log_interval": 1000, "checkpoint_interval": 100000, "tensorboard": True,
    }

    @staticmethod
    def raw(name):
        return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))

    def test_frozen_baseline_and_512_configs_are_unchanged(self):
        self.assertEqual(self.raw("baseline.json"), self.FROZEN_BASELINE)
        self.assertEqual(self.raw("width_2x512.json"), {**self.FROZEN_BASELINE, "hidden_width": 512})

    def test_width_2x1024_differs_from_2x512_only_in_hidden_width(self):
        w512, w1024 = self.raw("width_2x512.json"), self.raw("width_2x1024.json")
        self.assertEqual(w512.pop("hidden_width"), 512)
        self.assertEqual(w1024.pop("hidden_width"), 1024)
        self.assertEqual(w1024, w512)

    def test_width_2x1024_differs_from_baseline_only_in_hidden_width(self):
        w1024 = self.raw("width_2x1024.json")
        self.assertEqual(w1024.pop("hidden_width"), 1024)
        self.assertEqual(w1024, self.FROZEN_BASELINE)

    def test_width_2x1024_keeps_key_order_and_run_id_placeholder(self):
        w512, w1024 = self.raw("width_2x512.json"), self.raw("width_2x1024.json")
        self.assertEqual(list(w1024), list(w512))
        self.assertIsNone(w1024["run_id"])  # overridden with --run-id at launch
        self.assertEqual(w1024["total_steps"], 1_000_000)
        self.assertEqual(w1024["device"], "cuda")

    def test_width_2x1024_loads_into_train_config(self):
        cfg = load_config(str(CONFIG_DIR / "width_2x1024.json"))
        self.assertEqual(cfg.hidden_width, 1024)
        self.assertEqual(
            dataclasses.replace(cfg, hidden_width=512), load_config(str(CONFIG_DIR / "width_2x512.json"))
        )
        self.assertEqual(
            dataclasses.replace(cfg, hidden_width=256), load_config(str(CONFIG_DIR / "baseline.json"))
        )
        self.assertEqual(dataclasses.asdict(cfg)["hidden_width"], 1024)  # recorded in config.json / checkpoints


class Width1024SmokeTests(unittest.TestCase):
    def test_tiny_training_run_with_width_1024(self):
        cfg = TrainConfig(
            seed=0, total_steps=30, learning_starts=10, batch_size=8, buffer_size=100,
            eval_interval=0, eval_episodes=1, final_eval_episodes=1, hidden_width=1024,
        )
        result = train(cfg, FakeEnv)
        self.assertEqual(result.steps_completed, 30)
        self.assertGreater(result.num_critic_updates, 0)
        self.assertGreater(result.num_actor_updates, 0)
        self.assertEqual(count_parameters(result.agent), expected_counts(1024, obs=3, act=2))  # FakeEnv: obs 3, act 2
        for name in ("qf1", "qf2", "qf1_target", "qf2_target"):
            self.assertEqual(linear_widths(getattr(result.agent, name))["fc2"], (1024, 1024))
        self.assertTrue(np.isfinite(result.final_eval.mean_return))


if __name__ == "__main__":
    unittest.main()
