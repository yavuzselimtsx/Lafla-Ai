import unittest

from lafla_ai_core.training.checkpoint_policy import CheckpointPolicy, retention_victims, should_save_checkpoint


class CheckpointPolicyTest(unittest.TestCase):
    def test_save_every_and_final(self):
        policy = CheckpointPolicy(save_every=100, keep_last=3)
        self.assertFalse(should_save_checkpoint(0, 1000, policy))
        self.assertTrue(should_save_checkpoint(100, 1000, policy))
        self.assertTrue(should_save_checkpoint(1000, 1000, policy))

    def test_retention_keeps_last_steps(self):
        self.assertEqual(retention_victims([250, 500, 750, 1000], keep_last=3), (250,))


if __name__ == "__main__":
    unittest.main()
