import unittest

from lafla_ai_core.training.stability import StabilityMonitor


class StabilityMonitorTest(unittest.TestCase):
    def test_detects_non_finite_loss(self):
        monitor = StabilityMonitor(window_size=3, threshold_std=2.0)
        with self.assertRaises(ValueError):
            monitor.observe(step=1, loss=float("nan"), grad_norm=1.0)

    def test_flags_large_spike_after_history_is_warm(self):
        monitor = StabilityMonitor(window_size=3, threshold_std=1.0)
        monitor.observe(step=1, loss=1.0, grad_norm=1.0)
        monitor.observe(step=2, loss=1.1, grad_norm=1.0)
        monitor.observe(step=3, loss=0.9, grad_norm=1.0)
        result = monitor.observe(step=4, loss=10.0, grad_norm=1.0)
        self.assertTrue(result.loss_spike)


if __name__ == "__main__":
    unittest.main()
