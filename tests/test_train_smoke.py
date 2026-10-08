import random
import unittest

import numpy as np
import torch

from src.envs import EVAL_SEED_OFFSET, make_env, set_global_seed
from src.train import TrainConfig, train
from tests.test_evaluate import FakeEnv


def tiny_config(**overrides) -> TrainConfig:
    cfg = dict(
        seed=0,
        total_steps=40,
        learning_starts=10,
        batch_size=8,
        buffer_size=100,
        policy_frequency=2,
        target_network_frequency=1,
        eval_interval=0,
        eval_episodes=1,
    )
    cfg.update(overrides)
    return TrainConfig(**cfg)


def run_fake(config: TrainConfig):
    """Train on FakeEnv; also return the list of envs that train() created."""
    envs = []

    def env_fn(seed):
        env = FakeEnv(seed)
        envs.append(env)
        return env

    return train(config, env_fn), envs


class SeedTests(unittest.TestCase):
    def test_repeated_draws_reproducible(self):
        def draw():
            set_global_seed(0)
            return random.random(), np.random.rand(), torch.rand(1).item()

        self.assertEqual(draw(), draw())
        first = draw()
        set_global_seed(1)
        self.assertNotEqual(first, (random.random(), np.random.rand(), torch.rand(1).item()))

    def test_make_env_seeds_action_space(self):
        a, b = make_env("HalfCheetah-v5", 3), make_env("HalfCheetah-v5", 3)
        self.assertTrue(np.array_equal(a.action_space.sample(), b.action_space.sample()))
        a.close()
        b.close()


class TrainScheduleTests(unittest.TestCase):
    def test_train_and_eval_envs_are_separate(self):
        result, envs = run_fake(tiny_config())
        self.assertEqual(len(envs), 2)
        self.assertIsNot(envs[0], envs[1])
        self.assertTrue(all(e.closed for e in envs))
        self.assertEqual(envs[0].reset_seeds[0], 0)  # explicit seeded first reset
        self.assertTrue(all(s is None for s in envs[0].reset_seeds[1:]))

    def test_schedule_counts_policy_frequency_2(self):
        # Update steps: 11..39 (global_step > learning_starts) -> 29 critic updates.
        # Actor: steps with step % 2 == 0 among those (12,...,38) = 14, x2 updates each.
        result, _ = run_fake(tiny_config())
        self.assertEqual(result.steps_completed, 40)
        self.assertEqual(result.num_random_actions, 10)  # steps 0..9
        self.assertEqual(result.num_policy_actions, 30)  # steps 10..39
        self.assertEqual(len(result.replay_buffer), 40)
        self.assertEqual(result.num_critic_updates, 29)
        self.assertEqual(result.num_actor_updates, 28)
        self.assertEqual(result.num_target_updates, 29)

    def test_schedule_counts_other_frequencies(self):
        # policy_frequency=3: steps 12,15,...,39 = 10 -> 30 actor updates.
        # target_network_frequency=2: steps 12,14,...,38 = 14 target updates.
        result, _ = run_fake(tiny_config(policy_frequency=3, target_network_frequency=2))
        self.assertEqual(result.num_critic_updates, 29)
        self.assertEqual(result.num_actor_updates, 30)
        self.assertEqual(result.num_target_updates, 14)

    def test_no_updates_during_warmup(self):
        result, _ = run_fake(tiny_config(total_steps=10))  # steps 0..9, all warm-up
        self.assertEqual(result.num_critic_updates, 0)
        self.assertEqual(result.num_policy_actions, 0)
        # step == learning_starts is a policy step but not yet an update step
        result, _ = run_fake(tiny_config(total_steps=11))
        self.assertEqual(result.num_policy_actions, 1)
        self.assertEqual(result.num_critic_updates, 0)

    def test_termination_or_truncation_resets(self):
        # ep_len 5: episodes end at steps 4,9,14,19 (0-indexed); even episodes
        # terminate, odd ones truncate.
        result, envs = run_fake(tiny_config(total_steps=23))
        rb = result.replay_buffer
        self.assertEqual(len(result.episode_returns), 4)
        self.assertEqual(result.episode_returns, [5.0] * 4)
        self.assertEqual(result.episode_end_steps, [4, 9, 14, 19])
        self.assertEqual(envs[0].reset_calls, 5)  # initial + one per episode end
        self.assertEqual(rb.terminations[:23].sum(), 2)
        self.assertEqual(rb.truncations[:23].sum(), 2)

    def test_eval_called_on_interval_and_does_not_affect_training(self):
        with_eval, envs = run_fake(tiny_config(total_steps=20, learning_starts=5, eval_interval=10))
        without_eval, _ = run_fake(tiny_config(total_steps=20, learning_starts=5, eval_interval=0))
        self.assertEqual([s for s, _ in with_eval.eval_results], [10, 20])
        self.assertEqual(len(with_eval.eval_results[0][1].returns), 1)
        self.assertEqual(envs[1].reset_seeds[0], EVAL_SEED_OFFSET)  # eval stream seed
        # Evaluation must not change replay contents or learned parameters.
        a, b = with_eval.replay_buffer, without_eval.replay_buffer
        for name in ("obs", "next_obs", "actions", "rewards"):
            self.assertTrue(np.array_equal(getattr(a, name), getattr(b, name)), name)
        for p, q in zip(with_eval.agent.actor.parameters(), without_eval.agent.actor.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_real_halfcheetah_smoke(self):
        result = train(tiny_config(total_steps=30, learning_starts=10, eval_interval=0))
        self.assertEqual(result.steps_completed, 30)
        self.assertEqual(result.num_critic_updates, 19)
        self.assertEqual(len(result.replay_buffer), 30)
        env = make_env("HalfCheetah-v5", 0)
        self.assertEqual(env.observation_space.shape, (17,))
        env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_smoke(self):
        result, _ = run_fake(tiny_config(device="cuda", total_steps=25, eval_interval=25))
        self.assertEqual(result.num_critic_updates, 14)
        self.assertEqual(len(result.eval_results), 1)
        self.assertEqual(result.agent.log_alpha.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
