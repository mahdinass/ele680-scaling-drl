import unittest
from unittest import mock

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from src.networks import Actor


class FakeEnv(gym.Env):
    """Tiny env: obs (3,), action (2,) in [-1, 1], reward 1 per step.

    Episode k lasts `ep_len` steps and ends by termination if k is even, else by
    truncation. Counts resets so tests can check reset behaviour.
    """

    observation_space = Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, seed: int = 0, ep_len: int = 5) -> None:
        super().__init__()
        self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space.seed(seed)
        self.ep_len = ep_len
        self.episode = -1
        self.t = 0
        self.reset_calls = 0
        self.reset_seeds = []
        self.closed = False

    def _obs(self):
        return np.full(3, 0.1 * self.t + self.episode, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_calls += 1
        self.reset_seeds.append(seed)
        self.episode += 1
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        ended = self.t >= self.ep_len
        terminated = ended and self.episode % 2 == 0
        truncated = ended and not terminated
        return self._obs(), 1.0, terminated, truncated, {}

    def close(self):
        self.closed = True


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.env = FakeEnv()
        self.actor = Actor(self.env.observation_space, self.env.action_space)

    def test_returns_count_and_mean(self):
        from src.evaluate import evaluate

        res = evaluate(self.actor, self.env, num_episodes=4, seed=100)
        self.assertEqual(len(res.returns), 4)
        self.assertEqual(res.returns, [5.0] * 4)  # 5 steps x reward 1
        self.assertAlmostEqual(res.mean_return, 5.0)
        self.assertEqual(self.env.reset_seeds, [100, 101, 102, 103])

    def test_uses_deterministic_action_only(self):
        from src.evaluate import evaluate

        with mock.patch.object(
            Actor, "get_action", side_effect=AssertionError("sampling used in evaluation")
        ), mock.patch.object(
            Actor, "get_deterministic_action", autospec=True, side_effect=Actor.get_deterministic_action
        ) as det:
            evaluate(self.actor, self.env, num_episodes=2, seed=0)
        self.assertEqual(det.call_count, 10)  # 2 episodes x 5 steps

    def test_repeatable_and_no_state_mutation(self):
        from src.evaluate import evaluate

        before = [p.detach().clone() for p in self.actor.parameters()]
        rng_state = torch.get_rng_state()
        a = evaluate(self.actor, self.env, num_episodes=2, seed=0)
        b = evaluate(self.actor, self.env, num_episodes=2, seed=0)
        self.assertEqual(a.returns, b.returns)
        for p0, p in zip(before, self.actor.parameters()):
            self.assertTrue(torch.equal(p0, p))
            self.assertIsNone(p.grad)
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))  # no sampling

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        from src.evaluate import evaluate

        actor = Actor(self.env.observation_space, self.env.action_space).to("cuda")
        res = evaluate(actor, self.env, num_episodes=1, seed=0, device="cuda")
        self.assertEqual(len(res.returns), 1)


if __name__ == "__main__":
    unittest.main()
