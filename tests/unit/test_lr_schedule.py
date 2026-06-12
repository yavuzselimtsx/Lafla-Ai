import unittest

from lafla_ai_core.training.lr_schedule import cosine_with_warmup_lr


class LearningRateScheduleTest(unittest.TestCase):
    def test_warmup_and_decay_bounds(self):
        self.assertAlmostEqual(cosine_with_warmup_lr(0, 10, 2, 1.0, 0.1), 0.5)
        self.assertAlmostEqual(cosine_with_warmup_lr(1, 10, 2, 1.0, 0.1), 1.0)
        self.assertGreater(cosine_with_warmup_lr(5, 10, 2, 1.0, 0.1), 0.1)
        self.assertAlmostEqual(cosine_with_warmup_lr(10, 10, 2, 1.0, 0.1), 0.1)


if __name__ == "__main__":
    unittest.main()
