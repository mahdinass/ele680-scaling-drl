import unittest

import numpy as np
import torch
from gymnasium.spaces import Box

from src.networks import LOG_STD_MAX, LOG_STD_MIN, Actor, QNetwork

BATCH = 64


def make_spaces(low, high, obs_dim=5):
    obs_space = Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)
    act_space = Box(np.array(low, dtype=np.float32), np.array(high, dtype=np.float32))
    return obs_space, act_space


class NetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Symmetric (HalfCheetah-like, 6-D) and asymmetric artificial bounds.
        self.cases = {
            "symmetric": make_spaces([-1.0] * 6, [1.0] * 6, obs_dim=17),
            "asymmetric": make_spaces([-2.0, 0.0], [1.0, 4.0]),
        }

    def test_shapes_finite_and_bounds(self):
        for name, (obs_space, act_space) in self.cases.items():
            with self.subTest(name):
                actor = Actor(obs_space, act_space)
                q = QNetwork(obs_space, act_space)
                obs = torch.randn(BATCH, obs_space.shape[0]) * 10  # large inputs
                act_dim = act_space.shape[0]
                low = torch.as_tensor(act_space.low)
                high = torch.as_tensor(act_space.high)

                action, log_prob = actor.get_action(obs)
                det = actor.get_deterministic_action(obs)
                q_val = q(obs, action)

                self.assertEqual(action.shape, (BATCH, act_dim))
                self.assertEqual(det.shape, (BATCH, act_dim))
                self.assertEqual(log_prob.shape, (BATCH, 1))
                self.assertEqual(q_val.shape, (BATCH, 1))
                for t in (action, det, log_prob, q_val):
                    self.assertTrue(torch.isfinite(t).all())
                for a in (action, det):
                    self.assertTrue((a >= low).all() and (a <= high).all())

    def test_action_scale_and_bias(self):
        obs_space, act_space = self.cases["asymmetric"]
        actor = Actor(obs_space, act_space)
        np.testing.assert_allclose(actor.action_scale.numpy(), [1.5, 2.0])
        np.testing.assert_allclose(actor.action_bias.numpy(), [-0.5, 2.0])

    def test_gradients_flow_through_sampled_action(self):
        obs_space, act_space = self.cases["symmetric"]
        actor = Actor(obs_space, act_space)
        q = QNetwork(obs_space, act_space)
        obs = torch.randn(BATCH, obs_space.shape[0])
        action, log_prob = actor.get_action(obs)
        (q(obs, action) - log_prob).mean().backward()
        for name, p in actor.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(actor.fc_mean.weight.grad.abs().sum().item(), 0.0)

    def test_q1_q2_do_not_share_parameters(self):
        obs_space, act_space = self.cases["symmetric"]
        q1, q2 = QNetwork(obs_space, act_space), QNetwork(obs_space, act_space)
        ptrs1 = {p.data_ptr() for p in q1.parameters()}
        ptrs2 = {p.data_ptr() for p in q2.parameters()}
        self.assertTrue(ptrs1.isdisjoint(ptrs2))

    def test_log_std_within_bounds(self):
        for name, (obs_space, act_space) in self.cases.items():
            with self.subTest(name):
                actor = Actor(obs_space, act_space)
                # Mix of small, normal and very large observations.
                scales = torch.tensor([0.01, 1.0, 10.0, 1000.0]).repeat_interleave(BATCH)
                obs = torch.randn(len(scales), obs_space.shape[0]) * scales[:, None]
                _, log_std = actor(obs)
                self.assertEqual(log_std.shape, (len(scales), act_space.shape[0]))
                self.assertTrue(torch.isfinite(log_std).all())
                self.assertTrue((log_std >= LOG_STD_MIN).all())
                self.assertTrue((log_std <= LOG_STD_MAX).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device_placement(self):
        # Device index is controlled externally via CUDA_VISIBLE_DEVICES.
        device = torch.device("cuda")
        obs_space, act_space = self.cases["asymmetric"]
        actor = Actor(obs_space, act_space).to(device)
        q = QNetwork(obs_space, act_space).to(device)
        obs = torch.randn(BATCH, obs_space.shape[0], device=device)

        action, log_prob = actor.get_action(obs)
        det = actor.get_deterministic_action(obs)
        q_val = q(obs, action)

        self.assertEqual(actor.action_scale.device.type, "cuda")
        self.assertEqual(actor.action_bias.device.type, "cuda")
        for t in (action, log_prob, det, q_val):
            self.assertEqual(t.device.type, "cuda")
            self.assertTrue(torch.isfinite(t).all())

    def test_baseline_architecture_2x256(self):
        obs_space, act_space = self.cases["symmetric"]
        obs_dim, act_dim = 17, 6
        q = QNetwork(obs_space, act_space)
        self.assertEqual(
            [tuple(m.weight.shape) for m in (q.fc1, q.fc2, q.fc3)],
            [(256, obs_dim + act_dim), (256, 256), (1, 256)],
        )
        actor = Actor(obs_space, act_space)
        self.assertEqual(
            [tuple(m.weight.shape) for m in (actor.fc1, actor.fc2, actor.fc_mean, actor.fc_logstd)],
            [(256, obs_dim), (256, 256), (act_dim, 256), (act_dim, 256)],
        )


if __name__ == "__main__":
    unittest.main()
