import unittest

import numpy as np
import torch

from src.replay_buffer import ReplayBuffer

OBS, ACT = 5, 3


def fill(rb: ReplayBuffer, n: int, start: int = 0) -> None:
    for i in range(start, start + n):
        rb.add(
            np.full(OBS, i), np.full(OBS, i + 0.5), np.full(ACT, i), float(i), i % 2 == 0, i % 3 == 0
        )


class ReplayBufferTests(unittest.TestCase):
    def test_sample_shapes_types_finite(self):
        rb = ReplayBuffer(100, OBS, ACT, seed=0)
        fill(rb, 20)
        self.assertEqual(len(rb), 20)
        b = rb.sample(8)
        self.assertEqual(b.obs.shape, (8, OBS))
        self.assertEqual(b.next_obs.shape, (8, OBS))
        self.assertEqual(b.actions.shape, (8, ACT))
        for t in (b.rewards, b.terminations, b.truncations):
            self.assertEqual(t.shape, (8, 1))
        for t in b:
            self.assertEqual(t.dtype, torch.float32)
            self.assertTrue(torch.isfinite(t).all())

    def test_circular_overwrite(self):
        rb = ReplayBuffer(10, OBS, ACT, seed=0)
        fill(rb, 13)  # transitions 0..12; 0,1,2 are overwritten by 10,11,12
        self.assertEqual(len(rb), 10)
        self.assertEqual(rb.pos, 3)
        self.assertEqual(sorted(rb.rewards[:, 0].tolist()), [float(i) for i in range(3, 13)])
        self.assertEqual(rb.obs[0, 0], 10)  # slot 0 now holds transition 10

    def test_termination_and_truncation_stored_separately(self):
        rb = ReplayBuffer(10, OBS, ACT)
        rb.add(np.zeros(OBS), np.zeros(OBS), np.zeros(ACT), 0.0, False, True)
        rb.add(np.zeros(OBS), np.zeros(OBS), np.zeros(ACT), 0.0, True, False)
        self.assertEqual(rb.terminations[:2, 0].tolist(), [0.0, 1.0])
        self.assertEqual(rb.truncations[:2, 0].tolist(), [1.0, 0.0])

    def test_sample_only_valid_entries_and_empty_error(self):
        rb = ReplayBuffer(50, OBS, ACT, seed=0)
        with self.assertRaises(ValueError):
            rb.sample(4)
        fill(rb, 5)
        b = rb.sample(200)
        self.assertTrue((b.rewards >= 0).all() and (b.rewards <= 4).all())


if __name__ == "__main__":
    unittest.main()
