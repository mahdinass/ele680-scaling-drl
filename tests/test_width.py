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


if __name__ == "__main__":
    unittest.main()
