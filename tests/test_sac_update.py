import unittest

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from src.replay_buffer import Batch, ReplayBuffer
from src.sac_update import SACAgent, soft_update

OBS, ACT, B = 17, 6, 32


def spaces():
    return (
        Box(-np.inf, np.inf, shape=(OBS,), dtype=np.float64),
        Box(-1.0, 1.0, shape=(ACT,), dtype=np.float32),
    )


def fake_batch(device="cpu", terminations=None) -> Batch:
    g = torch.Generator().manual_seed(0)
    randn = lambda *s: torch.randn(*s, generator=g).to(device)
    term = torch.zeros(B, 1) if terminations is None else terminations
    return Batch(
        randn(B, OBS),
        randn(B, OBS),
        (torch.rand(B, ACT, generator=g) * 2 - 1).to(device),
        randn(B, 1),
        term.to(device),
        torch.zeros(B, 1, device=device),
    )


def params(m):
    return [p.detach().clone() for p in m.parameters()]


def changed(before, module):
    return any(not torch.equal(b, p) for b, p in zip(before, module.parameters()))


class SoftUpdateTests(unittest.TestCase):
    def test_numeric(self):
        src, tgt = nn.Linear(2, 2), nn.Linear(2, 2)
        with torch.no_grad():
            src.weight.fill_(1.0)
            src.bias.fill_(1.0)
            tgt.weight.fill_(0.0)
            tgt.bias.fill_(0.0)
        soft_update(tgt, src, 0.1)
        self.assertTrue(torch.allclose(tgt.weight, torch.full((2, 2), 0.1)))
        self.assertTrue(torch.allclose(tgt.bias, torch.full((2,), 0.1)))
        self.assertTrue(torch.equal(src.weight, torch.ones(2, 2)))  # source untouched


class SACAgentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.agent = SACAgent(*spaces())

    def test_targets_are_independent_copies(self):
        a = self.agent
        ptrs = {p.data_ptr() for p in [*a.qf1.parameters(), *a.qf2.parameters()]}
        tptrs = {p.data_ptr() for p in [*a.qf1_target.parameters(), *a.qf2_target.parameters()]}
        self.assertTrue(ptrs.isdisjoint(tptrs))
        for p, tp in zip(a.qf1.parameters(), a.qf1_target.parameters()):
            self.assertTrue(torch.equal(p, tp))  # same values at init
        self.assertFalse(any(p.requires_grad for p in a.qf1_target.parameters()))

    def test_target_entropy_and_alpha_init(self):
        self.assertEqual(self.agent.target_entropy, -float(ACT))
        self.assertAlmostEqual(self.agent.alpha.item(), 1.0)

    def test_losses_finite(self):
        batch = fake_batch()
        a = self.agent
        actor_loss, log_pi = a.actor_loss(batch)
        self.assertEqual(log_pi.shape, (B, 1))
        for loss in (a.critic_loss(batch), actor_loss, a.alpha_loss(log_pi)):
            self.assertEqual(loss.shape, ())
            self.assertTrue(torch.isfinite(loss))

    def test_one_update_step_changes_params(self):
        a, batch = self.agent, fake_batch()
        q1, q2, pi = params(a.qf1), params(a.qf2), params(a.actor)
        log_alpha = a.log_alpha.detach().clone()
        t1 = params(a.qf1_target)

        cl = a.update_critic(batch)
        self.assertTrue(np.isfinite(cl))
        self.assertTrue(changed(q1, a.qf1) and changed(q2, a.qf2))
        self.assertFalse(changed(pi, a.actor))  # critic step must not touch actor
        self.assertFalse(changed(t1, a.qf1_target))  # nor targets

        al, alpl = a.update_actor_and_alpha(batch)
        self.assertTrue(np.isfinite(al) and np.isfinite(alpl))
        self.assertTrue(changed(pi, a.actor))
        self.assertFalse(torch.equal(log_alpha, a.log_alpha.detach()))
        self.assertGreater(a.alpha.item(), 0.0)

        a.update_targets()
        self.assertTrue(changed(t1, a.qf1_target))

    def test_actor_step_does_not_change_critics(self):
        a, batch = self.agent, fake_batch()
        q1 = params(a.qf1)
        a.update_actor_and_alpha(batch)
        self.assertFalse(changed(q1, a.qf1))

    def test_termination_masks_bootstrap(self):
        # With terminations=1 the target is just r, so changing next_obs must not matter.
        a = self.agent
        b1 = fake_batch(terminations=torch.ones(B, 1))
        b2 = b1._replace(next_obs=b1.next_obs + 5.0)
        torch.manual_seed(1)
        l1 = a.critic_loss(b1)
        torch.manual_seed(1)
        l2 = a.critic_loss(b2)
        self.assertAlmostEqual(l1.item(), l2.item(), places=5)

    def test_truncation_still_bootstraps(self):
        # Truncation flag alone must not change the target (same loss as untruncated).
        a = self.agent
        b1 = fake_batch()  # terminations = 0
        b2 = b1._replace(truncations=torch.ones(B, 1))
        torch.manual_seed(1)
        l1 = a.critic_loss(b1)
        torch.manual_seed(1)
        l2 = a.critic_loss(b2)
        self.assertAlmostEqual(l1.item(), l2.item(), places=6)

    def test_with_replay_buffer(self):
        rb = ReplayBuffer(100, OBS, ACT, seed=0)
        for _ in range(40):
            rb.add(
                np.random.randn(OBS), np.random.randn(OBS), np.random.uniform(-1, 1, ACT), 1.0, False, False
            )
        self.assertTrue(np.isfinite(self.agent.update_critic(rb.sample(16))))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_update_on_cuda(self):
        device = torch.device("cuda")
        a = SACAgent(*spaces(), device=device)
        batch = fake_batch(device)
        self.assertTrue(np.isfinite(a.update_critic(batch)))
        al, alpl = a.update_actor_and_alpha(batch)
        self.assertTrue(np.isfinite(al) and np.isfinite(alpl))
        a.update_targets()
        self.assertEqual(a.log_alpha.device.type, "cuda")
        self.assertEqual(next(a.qf1_target.parameters()).device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
